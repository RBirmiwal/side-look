#!/usr/bin/env python3
"""Rule A (lean / SRTM) vs rule B (in place) on one strip.

Does not modify dataset v2 or sar/output/image_space/. Writes only to
sar/output/rule_check/. Does not build tiles.
"""

from __future__ import annotations

import csv
import gc
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from acquire_geometry import ecef_to_geodetic, enu_basis  # noqa: E402
from image_space import (  # noqa: E402
    LAYOVER,
    NOMINAL,
    SHADOW,
    UNKNOWN,
    labels_from_coverage,
    own_range_grid,
    project_grid,
    registration_check,
)
from image_space_run import (  # noqa: E402
    PAD_M,
    PAD_PX,
    STRIP,
    load_hgt,
    prepare_ahn,
    sample_srtm,
    upsample,
)
from near_field import (  # noqa: E402
    add_undulation,
    enu_from_map,
    fit_flight_line,
    look_azimuth_from_track,
    perpendicular_geometry,
)
from shift_check import gpkg_wkb, local_ground_5th  # noqa: E402

MAG = ROOT / "data" / f"SN6_AOI_11_Rotterdam_SAR-MAG-POL_{STRIP}.tif"
SLC_NPY = ROOT / "output" / "shift_check" / "slc_on_mag.npy"
V2_MASK = ROOT / "output" / "masks_v2" / f"{STRIP}.tif"
WATER_V3 = ROOT / "output" / "image_space" / "egm96" / f"{STRIP}_water.tif"
AHN = ROOT / "data" / "ahn_dsm_1m_aoi.tif"
HGT = ROOT / "data" / "srtm" / "N51E004.hgt.gz"
BAG = ROOT / "data" / "3dbag"
OUT = ROOT / "output" / "rule_check"
GSD = 0.25
CHUNK_COLS = 1000
NAN_MARGIN = 400
HOLE_MAX_M2 = 100

# v3 crop windows: row 1346, these columns, 512 px
V3_CROPS = [0, 7168, 22016, 33280, 39424, 43520]


def log(msg: str) -> None:
    print(msg, flush=True)


def flight_geo():
    desc = json.loads(rasterio.open(MAG).tags()["TIFFTAG_IMAGEDESCRIPTION"])
    collect = desc["collect"]
    target = np.array(collect["image"]["center_pixel"]["target_position"], dtype=np.float64)
    lat, lon, height = ecef_to_geodetic(target)
    basis = enu_basis(lat, lon)
    pos = np.array([sv["position"] for sv in collect["state"]["state_vectors"]], dtype=np.float64)
    enu = (basis @ (pos - target).T).T
    fit = fit_flight_line(enu)
    look = look_azimuth_from_track(fit["direction_enu"], collect["radar"]["pointing"])
    flight = {
        "origin_enu": np.asarray(fit["origin_enu"], np.float64),
        "direction_enu": np.asarray(fit["direction_enu"], np.float64),
        "height_above_origin_m": float(fit["height_above_origin_m"]),
        "look_deg": float(look),
    }
    geo = {"lat": float(lat), "lon": float(lon), "height": float(height)}
    return flight, geo, desc


def mag_grid():
    with rasterio.open(MAG) as src:
        t = src.transform
        h, w = src.height, src.width
    west, north = float(t.c), float(t.f)
    south = north + float(t.e) * h
    east = west + float(t.a) * w
    return dict(west=west, north=north, south=south, east=east, height=h, width=w, transform=t)


def fill_small_holes(z: np.ndarray, water: np.ndarray) -> tuple[np.ndarray, dict]:
    """Median-of-ring fill for non-water nodata components smaller than 100 m²."""
    from scipy.ndimage import binary_dilation, find_objects, label

    nodata = ~np.isfinite(z) & ~water
    comp, ncomp = label(nodata)
    counts = np.bincount(comp.ravel(), minlength=ncomp + 1)
    objects = find_objects(comp)
    filled_n = 0
    filled_m2 = 0
    left_m2 = 0
    out = z.copy()
    for i, sl in enumerate(objects, start=1):
        area = int(counts[i])
        if area == 0:
            continue
        if area >= HOLE_MAX_M2:
            left_m2 += area
            continue
        pad = tuple(slice(max(0, s.start - 2), min(dim, s.stop + 2)) for s, dim in zip(sl, z.shape))
        sub = comp[pad] == i
        ring = binary_dilation(sub, iterations=2) & ~sub
        vals = out[pad][ring]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            left_m2 += area
            continue
        out[pad][sub] = np.median(vals)
        filled_n += 1
        filled_m2 += area
    info = {"holes_filled": filled_n, "filled_m2": filled_m2, "large_hole_m2": left_m2}
    log(f"  holes <{HOLE_MAX_M2} m² filled: {filled_n} ({filled_m2} m²); larger holes left: {left_m2} m²")
    return out, info


def load_footprints(shape, transform):
    from pyproj import Transformer
    from rasterio.features import rasterize
    from shapely import wkb
    from shapely.ops import transform as shp_transform

    to_utm = Transformer.from_crs("EPSG:28992", "EPSG:32631", always_xy=True).transform
    to_rd = Transformer.from_crs("EPSG:32631", "EPSG:28992", always_xy=True)
    h, w = shape
    west, north = float(transform.c), float(transform.f)
    south = north - h
    east = west + w
    rd = [to_rd.transform(x, y) for x, y in ((west, south), (east, south), (east, north), (west, north))]
    xmin, xmax = min(p[0] for p in rd) - 40, max(p[0] for p in rd) + 40
    ymin, ymax = min(p[1] for p in rd) - 40, max(p[1] for p in rd) + 40
    shapes = []
    for gpkg in sorted(BAG.glob("*.gpkg")):
        con = sqlite3.connect(gpkg)
        q = """SELECT p.geom FROM pand p, rtree_pand_geom r
               WHERE p.fid = r.id AND r.maxx >= ? AND r.minx <= ? AND r.maxy >= ? AND r.miny <= ?"""
        for (blob,) in con.execute(q, (xmin, xmax, ymin, ymax)):
            if blob is None:
                continue
            try:
                geom = shp_transform(to_utm, wkb.loads(gpkg_wkb(blob)))
            except Exception:
                continue
            if geom.is_empty:
                continue
            shapes.append((geom, 1))
        con.close()
    log(f"  rasterizing {len(shapes)} footprints on {h} x {w}")
    fp = rasterize(shapes, out_shape=shape, transform=transform, fill=0, dtype=np.uint8, all_touched=True)
    return fp


def stage_prep():
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / "prep.npz"
    if dest.exists():
        log(f"prep exists {dest}")
        return
    g = mag_grid()
    log("AHN 1 m, padded 200 m in range, water fill")
    z, water, t1, level = prepare_ahn(g["west"], g["south"] - PAD_M, g["east"], g["north"] + PAD_M)
    log(f"  1 m grid {z.shape} water level {level:.3f} m NAP")
    z, holes = fill_small_holes(z, water)
    log("local ground, 5th percentile within 50 m")
    ground = local_ground_5th(z, 50)
    # water and any remaining nan ground: fall back to the DSM where it is finite
    miss = ~np.isfinite(ground) & np.isfinite(z)
    ground = ground.copy()
    ground[miss] = z[miss]
    log(f"  ground finite {np.isfinite(ground).mean():.3%}  "
        f"percentiles {np.nanpercentile(ground, [5, 50, 95]).round(2).tolist()}")
    fp = load_footprints(z.shape, t1)
    from scipy.ndimage import binary_dilation

    fp_dil = binary_dilation(fp.astype(bool), iterations=1)
    z_bldg = ground.copy()
    use = fp_dil & np.isfinite(z)
    z_bldg[use] = z[use]
    z_bldg[water & ~fp_dil] = level
    ndsm = (z - ground).astype(np.float32)
    clutter_core = (ndsm > 3.0) & ~fp.astype(bool) & np.isfinite(ndsm)
    # 30 m dilation via distance, not 30 binary passes
    from scipy.ndimage import distance_transform_edt

    clutter = distance_transform_edt(~clutter_core) <= 30.0
    log(f"  footprints {fp.mean():.3%}  dilated {fp_dil.mean():.3%}  "
        f"clutter core {clutter_core.mean():.3%}  clutter+30 m {clutter.mean():.3%}")
    log("NLGEO2018 on both surfaces")
    z_ell = add_undulation(np.where(np.isfinite(z), z, 0.0).astype(np.float64), t1, "EPSG:32631")
    z_ell[~np.isfinite(z)] = np.nan
    zb_ell = add_undulation(np.where(np.isfinite(z_bldg), z_bldg, 0.0).astype(np.float64), t1, "EPSG:32631")
    zb_ell[~np.isfinite(z_bldg)] = np.nan
    np.savez_compressed(
        dest,
        z_ell=z_ell.astype(np.float32),
        zb_ell=zb_ell.astype(np.float32),
        water=water.astype(np.uint8),
        fp=fp.astype(np.uint8),
        fp_dil=fp_dil.astype(np.uint8),
        clutter=clutter.astype(np.uint8),
        ground=ground.astype(np.float32),
        z_nap=z.astype(np.float32),
        level=np.float32(level),
        west=np.float64(g["west"]),
        north=np.float64(float(t1.f)),
    )
    (OUT / "prep_meta.json").write_text(json.dumps({
        "water_level_nap_m": level,
        "holes": holes,
        "shape_1m": list(z.shape),
        "grid_north": float(t1.f),
        "grid_west": float(g["west"]),
    }, indent=2))
    log(f"wrote {dest}")


def _open_pair(path_lab, path_cov, transform, height, width, resume: bool):
    profile = dict(
        driver="GTiff", height=height, width=width, count=1, crs="EPSG:32631",
        transform=transform, compress="deflate", tiled=True, blockxsize=512, blockysize=512,
    )
    mode = "r+" if resume else "w"
    lab = rasterio.open(path_lab, mode, dtype="uint8", nodata=255, predictor=2, **profile)
    cov = rasterio.open(path_cov, mode, dtype="float32", nodata=np.nan, predictor=3, **profile)
    return lab, cov


def stage_project():
    prep_path = OUT / "prep.npz"
    if not prep_path.exists():
        stage_prep()
    log("loading prep")
    prep = np.load(prep_path)
    z_ell = prep["z_ell"]
    zb_ell = prep["zb_ell"]
    water = prep["water"]
    g = mag_grid()
    flight, geo, _ = flight_geo()
    log(f"look {flight['look_deg']:.3f} deg  platform {flight['height_above_origin_m']:.1f} m")
    full_h = g["height"] + 2 * PAD_PX
    names = ("A_full", "A_bldg", "B_full", "B_bldg")
    progress_path = OUT / "chunks_done.json"
    done = set(json.loads(progress_path.read_text())) if progress_path.exists() else set()
    resume = bool(done)
    datasets = {}
    for name in names:
        lab_p, cov_p = OUT / f"{name}_lab.tif", OUT / f"{name}_cov.tif"
        datasets[name] = _open_pair(lab_p, cov_p, g["transform"], g["height"], g["width"], resume and lab_p.exists())
    unk_counts = {n: 0 for n in names}
    if (OUT / "unk_counts.json").exists() and resume:
        unk_counts.update(json.loads((OUT / "unk_counts.json").read_text()))

    # alpha direction once
    mid = g["width"] // 2
    z_col = upsample(z_ell, 0, mid, full_h, 1, order=1)[:, 0]
    rows = np.arange(full_h)
    xs = np.full((full_h, 1), g["west"] + (mid + 0.5) * GSD)
    ys = (float(prep["north"]) - (rows + 0.5) * GSD)[:, None]
    east, north_enu, up = enu_from_map(xs, ys, z_col[:, None], "EPSG:32631", geo["lat"], geo["lon"], geo["height"])
    _, alpha0 = perpendicular_geometry(east, north_enu, up, flight["origin_enu"], flight["direction_enu"])
    dalpha = float(np.nanmean(alpha0[-50:]) - np.nanmean(alpha0[:50]))
    log(f"  alpha(far)-alpha(near) {dalpha:+.4f} rad")
    if dalpha < 0:
        raise RuntimeError("sensor is not at row 0; refusing to project")

    from pyproj import Transformer

    to_ll = Transformer.from_crs("EPSG:32631", "EPSG:4326", always_xy=True)
    hgt = load_hgt(HGT)
    sl = slice(PAD_PX, PAD_PX + g["height"])
    t0 = time.time()
    for col0 in range(0, g["width"], CHUNK_COLS):
        if col0 in done:
            continue
        cw = min(CHUNK_COLS, g["width"] - col0)
        t1 = time.time()
        log(f"chunk {col0}:{col0+cw}")
        z_full = upsample(z_ell, 0, col0, full_h, cw, order=1)
        z_bldg = upsample(zb_ell, 0, col0, full_h, cw, order=1)
        cols = np.arange(col0, col0 + cw)
        xs = g["west"] + (cols[None, :] + 0.5) * GSD
        ys = float(prep["north"]) - (np.arange(full_h)[:, None] + 0.5) * GSD
        xs = np.broadcast_to(xs, (full_h, cw)).copy()
        ys = np.broadcast_to(ys, (full_h, cw)).copy()
        lon, lat = to_ll.transform(xs, ys)
        zref = sample_srtm(lon, lat, hgt, orthometric=False)
        del lon, lat
        east, north_enu, up = enu_from_map(xs, ys, zref, "EPSG:32631", geo["lat"], geo["lon"], geo["height"])
        r_ref, _ = perpendicular_geometry(east, north_enu, up, flight["origin_enu"], flight["direction_enu"])
        del east, north_enu, up, zref
        if not np.isfinite(r_ref).all():
            raise RuntimeError("r_ref has NaN")

        def slant_of(zsurf):
            e, n, u = enu_from_map(xs, ys, zsurf, "EPSG:32631", geo["lat"], geo["lon"], geo["height"])
            return perpendicular_geometry(e, n, u, flight["origin_enu"], flight["direction_enu"])

        r_f, a_f = slant_of(z_full)
        del z_full
        r_b, a_b = slant_of(z_bldg)
        del z_bldg, xs, ys
        gc.collect()
        products = {
            "A_full": project_grid(r_f, a_f, r_ref, axis=0, nan_margin=NAN_MARGIN),
            "B_full": own_range_grid(r_f, a_f, axis=0, nan_margin=NAN_MARGIN),
            "A_bldg": project_grid(r_b, a_b, r_ref, axis=0, nan_margin=NAN_MARGIN),
            "B_bldg": own_range_grid(r_b, a_b, axis=0, nan_margin=NAN_MARGIN),
        }
        del r_f, a_f, r_b, a_b, r_ref
        win = Window(col0, 0, cw, g["height"])
        for name, (cov, unk, _vis) in products.items():
            lab = labels_from_coverage(cov, unk)
            datasets[name][0].write(lab[sl].astype(np.uint8), 1, window=win)
            datasets[name][1].write(cov[sl].astype(np.float32), 1, window=win)
            unk_counts[name] += int(unk[sl].sum())
        del products
        gc.collect()
        done.add(col0)
        progress_path.write_text(json.dumps(sorted(done)))
        (OUT / "unk_counts.json").write_text(json.dumps(unk_counts))
        log(f"  chunk {time.time()-t1:.1f}s  total {time.time()-t0:.0f}s")
    for lab, cov in datasets.values():
        lab.close()
        cov.close()
    n_pix = g["height"] * g["width"]
    frac = {k: v / n_pix for k, v in unk_counts.items()}
    log("unknown fraction " + "  ".join(f"{k} {v:.3%}" for k, v in frac.items()))
    (OUT / "unknown.json").write_text(json.dumps({"counts": unk_counts, "fraction": frac, "n": n_pix}, indent=2))


def _hh_block(col0, cols, row0, rows):
    with rasterio.open(MAG) as src:
        return src.read(1, window=Window(col0, row0, cols, rows)).astype(np.float32)


def stage_basics():
    """Water floor, AHN version, far-range SLC vs MAG. No labels required."""
    OUT.mkdir(parents=True, exist_ok=True)
    g = mag_grid()
    log("basics: open water and land HH")
    with rasterio.open(WATER_V3) as src:
        water = src.read(1)
    # 4x4 max-pool to 1 m so the distance transform stays small
    h, w = water.shape
    water_1m = water.reshape(h // 4, 4, w // 4, 4).max(axis=(1, 3)).astype(bool)
    del water
    from scipy.ndimage import distance_transform_edt

    dist_m = distance_transform_edt(water_1m)
    open_w = water_1m & (dist_m > 30.0)
    del dist_m
    land = ~water_1m
    g = mag_grid()
    edges = [0, h // 3, 2 * h // 3, h]
    names = ["near", "middle", "far"]
    water_stats = {}
    for name, lo, hi in zip(names, edges[:-1], edges[1:]):
        hh = _hh_block(0, g["width"], lo, hi - lo)
        hh_1m = hh.reshape(hh.shape[0] // 4, 4, hh.shape[1] // 4, 4).mean(axis=(1, 3))
        r0, r1 = lo // 4, hi // 4
        def med(mask):
            v = hh_1m[mask]
            v = v[np.isfinite(v)]
            return float(np.median(v)) if v.size else float("nan")
        mw, ml = med(open_w[r0:r1]), med(land[r0:r1])
        water_stats[name] = {
            "open_water_hh_db": mw,
            "land_hh_db": ml,
            "water_minus_land_db": mw - ml,
            "n_open_water_1m": int(open_w[r0:r1].sum()),
            "n_land_1m": int(land[r0:r1].sum()),
            "posting": "1 m mean of the 0.25 m pixels",
        }
        log(f"  {name}: water {mw:.2f}  land {ml:.2f}  diff {mw-ml:.2f}  n_water {open_w[r0:r1].sum()}")
        del hh, hh_1m
    # AHN: the WCS this file was pulled from identifies itself as AHN4.
    # 3DBAG point-cloud source under the strip is recorded separately.
    ahn = {
        "raster": "PDOK WCS https://service.pdok.nl/rws/ahn/wcs/v1_0 coverage dsm_05m",
        "version": "AHN4",
        "service_text": "Het huidige AHN is versie 4. Deze versie is ingewonnen over de jaren 2020, 2021 en 2022.",
        "file": str(AHN),
        "note": (
            "3DBAG v20250903 on buildings inside this strip picks a point cloud per building: "
            "most are tagged AHN5 / 2022, a minority AHN4 / 2020 and AHN3 / 2016. "
            "That is the building reconstruction source, not the DSM raster. "
            "The raster service abstract is AHN4."
        ),
    }
    log("basics: far window SLC vs MAG")
    far = _far_window()
    (OUT / "basics.json").write_text(json.dumps({"water": water_stats, "ahn": ahn, "far": far}, indent=2))
    log(f"wrote {OUT / 'basics.json'}")


def _far_window():
    """One far-third window. SLC was resampled onto the MAG grid in shift_check."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = mag_grid()
    # far third starts at row 2136. SLC covers the south (far) edge fully.
    row0, col0, n = 2300, 28000, 1024
    if row0 + n > g["height"]:
        row0 = g["height"] - n
    hh = _hh_block(col0, n, row0, n)
    slc = np.load(SLC_NPY, mmap_mode="r")
    log(f"  slc_on_mag shape {slc.shape} dtype {slc.dtype}")
    sw = np.array(slc[row0:row0 + n, col0:col0 + n], dtype=np.float32)
    def stats(a, name):
        fin = np.isfinite(a)
        v = a[fin]
        return {
            "name": name,
            "finite": float(fin.mean()),
            "zeros": float((fin & (a == 0)).mean()) if fin.any() else 1.0,
            "min": float(np.min(v)) if v.size else None,
            "p1": float(np.percentile(v, 1)) if v.size else None,
            "p50": float(np.percentile(v, 50)) if v.size else None,
            "p99": float(np.percentile(v, 99)) if v.size else None,
            "max": float(np.max(v)) if v.size else None,
            "std": float(np.std(v)) if v.size else None,
        }
    both = np.isfinite(hh) & np.isfinite(sw)
    if both.sum() > 100:
        c = np.corrcoef(hh[both].ravel(), sw[both].ravel())[0, 1]
    else:
        c = float("nan")
    info = {"row0": row0, "col0": col0, "size": n, "pearson": float(c),
            "mag": stats(hh, "mag"), "slc": stats(sw, "slc")}
    log(f"  far window r{row0} c{col0}  pearson {c:.3f}")
    log(f"  MAG {info['mag']}")
    log(f"  SLC {info['slc']}")
    fig, ax = plt.subplots(1, 2, figsize=(10, 5.2))
    for a, img, title in ((ax[0], hh, "MAG-POL HH"), (ax[1], sw, "SLC |HH|² on MAG grid")):
        v = img[np.isfinite(img)]
        lo, hi = np.percentile(v, [2, 98]) if v.size else (0, 1)
        a.imshow(img, cmap="gray", vmin=lo, vmax=hi)
        a.set_title(f"{title}\n{lo:.1f} .. {hi:.1f}")
        a.axis("off")
    fig.suptitle(f"far third  row {row0} col {col0}   pearson {c:.3f}")
    fig.tight_layout()
    fig.savefig(OUT / "far_window.png", dpi=110)
    plt.close(fig)
    return info


def _eval_mask_chunked():
    """Known in all five, not water, not clutter. Written once as uint8."""
    path = OUT / "eval.tif"
    if path.exists():
        log(f"eval exists {path}")
        return path
    g = mag_grid()
    prep = np.load(OUT / "prep.npz")
    # clutter and water are on the padded 1 m grid. Image row r maps to
    # 1 m row 0.25*(r+PAD_PX)-0.375, column likewise, origin = padded north.
    clutter_1m = prep["clutter"].astype(bool)
    water_1m = prep["water"].astype(bool)
    profile = dict(
        driver="GTiff", height=g["height"], width=g["width"], count=1, crs="EPSG:32631",
        transform=g["transform"], compress="deflate", dtype="uint8", tiled=True,
        blockxsize=512, blockysize=512, predictor=2,
    )
    labs = [rasterio.open(p) for p in [
        V2_MASK,
        OUT / "A_full_lab.tif", OUT / "A_bldg_lab.tif",
        OUT / "B_full_lab.tif", OUT / "B_bldg_lab.tif",
    ]]
    dst = rasterio.open(path, "w", **profile)
    kept = 0
    for col0 in range(0, g["width"], 2000):
        cw = min(2000, g["width"] - col0)
        known = np.ones((g["height"], cw), bool)
        for src in labs:
            a = src.read(1, window=Window(col0, 0, cw, g["height"]))
            known &= a != UNKNOWN
            # v2 has no 255; still fine
        rows = np.arange(g["height"])
        cols = np.arange(col0, col0 + cw)
        r1 = np.rint(0.25 * (rows + PAD_PX) - 0.375).astype(int)
        c1 = np.rint(0.25 * cols - 0.375).astype(int)
        r1 = np.clip(r1, 0, clutter_1m.shape[0] - 1)
        c1 = np.clip(c1, 0, clutter_1m.shape[1] - 1)
        bad = clutter_1m[r1][:, c1] | water_1m[r1][:, c1]
        ev = known & ~bad
        dst.write(ev.astype(np.uint8), 1, window=Window(col0, 0, cw, g["height"]))
        kept += int(ev.sum())
    for src in labs:
        src.close()
    dst.close()
    log(f"eval pixels {kept}  ({kept / (g['height']*g['width']):.3%})")
    (OUT / "eval_count.json").write_text(json.dumps({"n": kept, "n_image": g["height"] * g["width"]}))
    return path


def stage_gates():
    g = mag_grid()
    eval_path = _eval_mask_chunked()
    sets = {
        "v2": V2_MASK,
        "A_full": OUT / "A_full_lab.tif",
        "A_bldg": OUT / "A_bldg_lab.tif",
        "B_full": OUT / "B_full_lab.tif",
        "B_bldg": OUT / "B_bldg_lab.tif",
    }
    # v2 codes differ
    h = g["height"]
    thirds = {"near": (0, h // 3), "middle": (h // 3, 2 * h // 3), "far": (2 * h // 3, h)}
    bins = np.linspace(-45, 30, 7501)  # 0.01 dB
    table = {}
    for name, path in sets.items():
        log(f"physics {name}")
        hist = {k: {c: np.zeros(bins.size - 1, np.int64) for c in ("nom", "lay", "sh")} for k in thirds}
        counts = {k: {c: 0 for c in ("nom", "lay", "sh", "eval")} for k in list(thirds) + ["all"]}
        with rasterio.open(path) as lab_ds, rasterio.open(eval_path) as ev_ds, rasterio.open(MAG) as mag:
            for col0 in range(0, g["width"], 2000):
                cw = min(2000, g["width"] - col0)
                win = Window(col0, 0, cw, h)
                lab = lab_ds.read(1, window=win)
                ev = ev_ds.read(1, window=win).astype(bool)
                hh = mag.read(1, window=win)
                if name == "v2":
                    mapped = np.full(lab.shape, NOMINAL, np.uint8)
                    mapped[(lab == 1) | (lab == 2)] = LAYOVER
                    mapped[lab == 3] = SHADOW
                    lab = mapped
                for tname, (lo, hi) in thirds.items():
                    sl = (slice(lo, hi), slice(None))
                    e = ev[sl] & np.isfinite(hh[sl])
                    counts[tname]["eval"] += int(e.sum())
                    counts["all"]["eval"] += int(e.sum())
                    hb = hh[sl]
                    for code, key in ((NOMINAL, "nom"), (LAYOVER, "lay"), (SHADOW, "sh")):
                        m = e & (lab[sl] == code)
                        counts[tname][key] += int(m.sum())
                        counts["all"][key] += int(m.sum())
                        if m.any():
                            hist[tname][key] += np.histogram(hb[m], bins=bins)[0]
        def med_from(hgram):
            tot = hgram.sum()
            if tot < 100:
                return float("nan")
            c = np.cumsum(hgram)
            i = int(np.searchsorted(c, tot / 2))
            return float(0.5 * (bins[i] + bins[i + 1]))
        row = {}
        for tname in thirds:
            nom = med_from(hist[tname]["nom"])
            n = counts[tname]["eval"]
            row[tname] = {
                "layover_db": med_from(hist[tname]["lay"]) - nom,
                "shadow_db": med_from(hist[tname]["sh"]) - nom,
                "nominal_db": nom,
                "share_layover": counts[tname]["lay"] / n if n else float("nan"),
                "share_shadow": counts[tname]["sh"] / n if n else float("nan"),
                "share_nominal": counts[tname]["nom"] / n if n else float("nan"),
                "n": n,
                "n_layover": counts[tname]["lay"],
                "n_shadow": counts[tname]["sh"],
            }
        table[name] = row
        for tname in thirds:
            r = row[tname]
            log(f"  {name:7} {tname:6} lay {r['layover_db']:+.2f}  sh {r['shadow_db']:+.2f}  "
                f"share lay {r['share_layover']:.3%} sh {r['share_shadow']:.3%}")

    log("registration")
    reg = {}
    # windows are small; load HH and eval fully would be 750 MB. Read per window instead
    # by wrapping arrays only for the windows registration_check wants. Build them
    # on the fly: registration_check wants full arrays. 3204*47260 float32 = 605 MB
    # plus eval 151 MB plus one label 151 MB. Do one label at a time.
    with rasterio.open(MAG) as src:
        hh = src.read(1)
    with rasterio.open(eval_path) as src:
        ev = src.read(1).astype(bool)
    for name, path in sets.items():
        with rasterio.open(path) as src:
            lab = src.read(1)
        if name == "v2":
            mapped = np.zeros(lab.shape, np.uint8)
            mapped[(lab == 1) | (lab == 2)] = LAYOVER
            mapped[lab == 3] = SHADOW
            lab = mapped
        reg[name] = {}
        for use, tag in ((("layover", "shadow"), "both"), (("shadow",), "shadow")):
            rows, med = registration_check(
                hh, lab, range_bands=3, n_windows=8, valid=ev, use=use,
            )
            by = {b: [] for b in range(3)}
            for band, centre, dy, dx, pk in rows:
                by[int(band)].append({"col": int(centre), "dy": int(dy), "dx": int(dx), "peak": float(pk)})
            bands = {}
            for b in range(3):
                arr = by[b]
                if not arr:
                    bands[b] = {"n": 0}
                    continue
                bands[b] = {
                    "n": len(arr),
                    "dy": float(np.median([a["dy"] for a in arr])),
                    "dx": float(np.median([a["dx"] for a in arr])),
                    "peak": float(np.median([a["peak"] for a in arr])),
                    "windows": arr,
                }
            reg[name][tag] = {"bands": bands, "median_all": [float(x) for x in med], "n": len(rows)}
            log(f"  {name:7} {tag:6} n={len(rows):2d}  " +
                "  ".join(
                    f"b{b}:{bands[b].get('dy', float('nan')):+.0f}/{bands[b].get('peak', float('nan')):.2f}(n={bands[b]['n']})"
                    for b in range(3)
                ))
        del lab
        gc.collect()
    del hh, ev
    (OUT / "gates.json").write_text(json.dumps({"physics": table, "registration": reg}, indent=2))
    log(f"wrote {OUT / 'gates.json'}")
    return table, reg


def _overlay(ax, hh, lab, title):
    from matplotlib.colors import ListedColormap
    lo, hi = np.nanpercentile(hh, [2, 98])
    ax.imshow(hh, cmap="gray", vmin=lo, vmax=hi)
    show = np.ma.masked_where((lab == 0) | (lab == UNKNOWN), lab)
    cmap = ListedColormap(["none", (0.85, 0.45, 0.15, 0.75), (0.20, 0.45, 0.85, 0.75)])
    ax.imshow(show, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")


def stage_crops():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    g = mag_grid()
    prep = np.load(OUT / "prep.npz")
    fp = prep["fp"].astype(bool)
    # disagreement of A-bldg vs B-bldg on the centre 512 rows
    r0 = (g["height"] - 512) // 2
    score = np.zeros(g["width"] // 512, np.int64)
    with rasterio.open(OUT / "A_bldg_lab.tif") as a_ds, rasterio.open(OUT / "B_bldg_lab.tif") as b_ds, \
            rasterio.open(OUT / "eval.tif") as e_ds:
        for i in range(score.size):
            c0 = i * 512
            win = Window(c0, r0, min(512, g["width"] - c0), 512)
            aa = a_ds.read(1, window=win)
            bb = b_ds.read(1, window=win)
            ee = e_ds.read(1, window=win).astype(bool)
            score[i] = int((ee & (aa != bb) & (aa != UNKNOWN) & (bb != UNKNOWN)).sum())
    order = np.argsort(score)[::-1]
    picks = []
    for i in order:
        if score[i] < 50:
            break
        c0 = int(i * 512)
        if any(abs(c0 - p) < 2000 for p in picks):
            continue
        picks.append(c0)
        if len(picks) == 3:
            break
    log(f"disagreement crops {picks} scores {[int(score[c // 512]) for c in picks]}")
    positions = [("v3", c) for c in V3_CROPS] + [("diff", c) for c in picks]
    (OUT / "crop_positions.json").write_text(json.dumps({"v3": V3_CROPS, "diff": picks}))

    def fp_window(row0, col0, n):
        rows = np.arange(row0, row0 + n)
        cols = np.arange(col0, col0 + n)
        r1 = np.clip(np.rint(0.25 * (rows + PAD_PX) - 0.375).astype(int), 0, fp.shape[0] - 1)
        c1 = np.clip(np.rint(0.25 * cols - 0.375).astype(int), 0, fp.shape[1] - 1)
        return fp[r1][:, c1]

    for kind, c0 in positions:
        c0 = min(c0, g["width"] - 512)
        hh = _hh_block(c0, 512, r0, 512)
        with rasterio.open(OUT / "A_bldg_lab.tif") as src:
            a = src.read(1, window=Window(c0, r0, 512, 512))
        with rasterio.open(OUT / "B_bldg_lab.tif") as src:
            b = src.read(1, window=Window(c0, r0, 512, 512))
        foot = fp_window(r0, c0, 512)
        fig, ax = plt.subplots(1, 4, figsize=(16, 4.4))
        lo, hi = np.nanpercentile(hh, [2, 98])
        ax[0].imshow(hh, cmap="gray", vmin=lo, vmax=hi)
        ax[0].set_title("HH dB")
        ax[0].axis("off")
        _overlay(ax[1], hh, a, "A buildings")
        _overlay(ax[2], hh, b, "B buildings")
        ax[3].imshow(hh, cmap="gray", vmin=lo, vmax=hi)
        ax[3].contour(foot.astype(float), levels=[0.5], colors="white", linewidths=0.6)
        ax[3].set_title("HH + 3DBAG")
        ax[3].axis("off")
        fig.suptitle(f"{kind}  row {r0} col {c0}")
        fig.tight_layout()
        fig.savefig(OUT / f"crop_{kind}_{r0}_{c0}.png", dpi=100)
        plt.close(fig)
        log(f"  crop {kind} {c0}")


def stage_layover():
    """Layover length in front of tall facades, measured against the near-field formula."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import median_filter
    from shapely import wkb
    from shapely.ops import transform as shp_transform
    from pyproj import Transformer
    from rasterio.features import rasterize

    g = mag_grid()
    prep = np.load(OUT / "prep.npz")
    z_nap = prep["z_nap"].astype(np.float32)
    ground = prep["ground"].astype(np.float32)
    z_ell = prep["z_ell"].astype(np.float32)
    water = prep["water"].astype(bool)
    ndsm = (z_nap - ground).astype(np.float32)
    n_und = (z_ell - z_nap).astype(np.float32)
    grid_north = float(prep["north"])
    west = float(prep["west"])
    t1 = from_origin(west, grid_north, 1.0, 1.0)
    # image occupies 1 m rows [PAD_M, PAD_M + height_m)
    row_off = int(PAD_M)

    to_utm = Transformer.from_crs("EPSG:28992", "EPSG:32631", always_xy=True).transform
    to_rd = Transformer.from_crs("EPSG:32631", "EPSG:28992", always_xy=True)
    h1, w1 = ndsm.shape
    south = grid_north - h1
    east = west + w1
    rd = [to_rd.transform(x, y) for x, y in ((west, south), (east, south), (east, grid_north), (west, grid_north))]
    xmin, xmax = min(p[0] for p in rd) - 20, max(p[0] for p in rd) + 20
    ymin, ymax = min(p[1] for p in rd) - 20, max(p[1] for p in rd) + 20
    geoms = []
    for gpkg in sorted(BAG.glob("*.gpkg")):
        con = sqlite3.connect(gpkg)
        q = """SELECT p.geom FROM pand p, rtree_pand_geom r
               WHERE p.fid = r.id AND r.maxx >= ? AND r.minx <= ? AND r.maxy >= ? AND r.miny <= ?"""
        for (blob,) in con.execute(q, (xmin, xmax, ymin, ymax)):
            if not blob:
                continue
            try:
                geom = shp_transform(to_utm, wkb.loads(gpkg_wkb(blob)))
            except Exception:
                continue
            if not geom.is_empty:
                geoms.append(geom)
        con.close()
    log(f"layover: {len(geoms)} footprints")
    lab = rasterize([(geom, i + 1) for i, geom in enumerate(geoms)], out_shape=ndsm.shape,
                    transform=t1, fill=0, dtype=np.int32, all_touched=True)
    from scipy.ndimage import find_objects
    objects = find_objects(lab)

    # candidates: up to 3 columns per building with h_edge >= 20
    cand = []  # dicts
    n_short = 0
    for i, sl in enumerate(objects, start=1):
        if sl is None:
            continue
        sub = lab[sl] == i
        if sub.sum() < 4:
            continue
        heights = ndsm[sl][sub]
        heights = heights[np.isfinite(heights)]
        if heights.size == 0 or np.percentile(heights, 90) < 18:
            n_short += 1
            continue
        r0, c0 = sl[0].start, sl[1].start
        cols = []
        for c in range(sub.shape[1]):
            rr = np.flatnonzero(sub[:, c])
            if rr.size == 0:
                continue
            near = int(rr.min())
            gr, gc = r0 + near, c0 + c
            band_r = slice(gr, min(h1, gr + 3))
            band = ndsm[band_r, gc]
            inside = lab[band_r, gc] == i
            vals = band[inside & np.isfinite(band)]
            if vals.size == 0:
                continue
            cols.append((float(np.percentile(vals, 90)), gr, gc))
        cols = [c for c in cols if c[0] >= 20.0]
        cols.sort(reverse=True)
        if not cols:
            n_short += 1
            continue
        for h_edge, gr, gc in cols[:3]:
            cand.append({"id": i, "h": h_edge, "r": gr, "c": gc})
    log(f"  candidate columns {len(cand)} from buildings with a facade >= 20 m")

    flight, geo, _ = flight_geo()
    # batch geometry at the ground point of each column
    rr = np.array([c["r"] for c in cand])
    cc = np.array([c["c"] for c in cand])
    xs = west + (cc + 0.5)
    ys = grid_north - (rr + 0.5)
    g_ell = ground[rr, cc].astype(np.float64) + n_und[rr, cc].astype(np.float64)
    from pyproj import Transformer as T
    to_ll = T.from_crs("EPSG:32631", "EPSG:4326", always_xy=True)
    hgt = load_hgt(HGT)
    # sample_srtm wants 2d
    lon, lat = to_ll.transform(xs, ys)
    srtm = sample_srtm(lon[None, :], lat[None, :], hgt, orthometric=False)[0]
    s_col = srtm - g_ell
    east, north_enu, up = enu_from_map(
        xs, ys, g_ell, "EPSG:32631", geo["lat"], geo["lon"], geo["height"]
    )
    slant, alpha = perpendicular_geometry(east, north_enu, up, flight["origin_enu"], flight["direction_enu"])
    Hloc = slant * np.cos(alpha)
    horiz = slant * np.sin(alpha)

    def lpred(h):
        disc = horiz ** 2 - 2 * Hloc * h + h ** 2
        out = np.full(h.shape, np.nan)
        ok = (h > 1) & (disc > 0) & (Hloc > h)
        out[ok] = horiz[ok] - np.sqrt(disc[ok])
        return out

    h_edge = np.array([c["h"] for c in cand])
    Lp = lpred(np.maximum(h_edge - s_col, 0))
    Lp_h = lpred(h_edge)  # s ignored

    log("loading HH")
    with rasterio.open(MAG) as src:
        hh = src.read(1)
    skips = {"front": 0, "water": 0, "off": 0, "geometry": 0, "flat": 0}
    # group columns by building after filtering
    from collections import defaultdict
    by_b = defaultdict(list)
    for k, c in enumerate(cand):
        if not np.isfinite(Lp[k]) or Lp[k] < 2:
            skips["geometry"] += 1
            continue
        reach = int(np.ceil(1.5 * Lp[k]))
        r, col = c["r"], c["c"]
        r_from = max(0, r - reach)
        front = lab[r_from:r, col]
        if np.any((front != 0) & (front != c["id"])):
            skips["front"] += 1
            continue
        stretch = water[r_from:r, col]
        if stretch.size and stretch.mean() > 0.5:
            skips["water"] += 1
            continue
        # wall foot in image rows. 1 m row r's near edge.
        wall_row = 4 * (r - row_off) - 0.5
        img_c = 4 * col + 1.5
        far_m = 1.5 * float(Lp[k]) + 60.0  # background ends 60 m past 1.5 L
        row_far = wall_row - far_m / GSD
        if wall_row - 2 < 0 or row_far < 2 or img_c < 4 or img_c > g["width"] - 5:
            skips["off"] += 1
            continue
        by_b[c["id"]].append(k)

    records = []
    for bid, ks in by_b.items():
        meas = []
        for k in ks:
            c = cand[k]
            wall_row = 4 * (c["r"] - row_off) - 0.5
            img_c = int(round(4 * c["c"] + 1.5))
            L = float(Lp[k])
            end_m = 1.5 * L + 20.0
            bg0, bg1 = 1.5 * L + 20.0, 1.5 * L + 60.0
            # sample a profile slightly longer than the background
            n_pix = int(np.ceil(bg1 / GSD)) + 2
            r_start = int(np.floor(wall_row - n_pix))
            r_wall = int(np.round(wall_row))
            if r_start < 0 or r_wall >= hh.shape[0]:
                skips["off"] += 1
                continue
            cols = np.arange(img_c - 2, img_c + 3)
            cols = cols[(cols >= 0) & (cols < hh.shape[1])]
            block = hh[r_start:r_wall + 1, cols]  # row r_wall is ~the wall, r_start is toward the sensor
            # offset of each row: distance from the wall toward the sensor
            rows = np.arange(r_start, r_wall + 1)
            offset = (wall_row - rows) * GSD  # positive toward the sensor
            # keep increasing offset (flip so index 0 is the wall)
            order = np.argsort(offset)
            offset = offset[order]
            block = block[order]
            # smooth each column, then average
            sm = median_filter(block.astype(np.float64), size=(9, 1), mode="nearest")
            prof = np.nanmean(sm, axis=1)
            m_bg = (offset >= bg0) & (offset <= bg1) & np.isfinite(prof)
            m_search = (offset >= 0) & (offset <= end_m) & np.isfinite(prof)
            if m_bg.sum() < 8 or m_search.sum() < 8:
                skips["off"] += 1
                continue
            bg = float(np.median(prof[m_bg]))
            bright = m_search & (prof >= bg + 2.0)
            # farthest bright sample connected to the wall with gaps <= 5 m
            idx = np.flatnonzero(m_search)
            last = None
            end_i = None
            for i in idx:
                if not bright[i]:
                    continue
                if last is None:
                    if offset[i] <= 5.0:
                        last = i
                        end_i = i
                    else:
                        break
                elif offset[i] - offset[last] - GSD <= 5.0:
                    last = i
                    end_i = i
                else:
                    break
            L_meas = float(offset[end_i]) if end_i is not None else 0.0
            meas.append({
                "L_meas": L_meas, "L_pred": L, "L_pred_h": float(Lp_h[k]),
                "h": float(h_edge[k]), "s": float(s_col[k]), "H": float(Hloc[k]),
                "g": float(horiz[k]), "theta": float(np.degrees(alpha[k])),
                "wall_row": float(wall_row), "img_c": int(img_c),
                "bg": bg, "r": int(c["r"]), "c": int(c["c"]),
            })
        if not meas:
            continue
        records.append({
            "id": int(bid),
            "n": len(meas),
            "L_meas": float(np.median([m["L_meas"] for m in meas])),
            "L_pred": float(np.median([m["L_pred"] for m in meas])),
            "L_pred_h": float(np.median([m["L_pred_h"] for m in meas])),
            "h": float(np.median([m["h"] for m in meas])),
            "s": float(np.median([m["s"] for m in meas])),
            "theta": float(np.median([m["theta"] for m in meas])),
            "wall_row": meas[0]["wall_row"],
            "img_c": meas[0]["img_c"],
            "cols": meas,
        })
    log(f"  buildings kept {len(records)}  skips {skips}")

    def fit(x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        ok = np.isfinite(x) & np.isfinite(y) & (x > 2)
        x, y = x[ok], y[ok]
        if x.size < 3:
            return {"n": int(x.size)}
        slope = float(np.sum(x * y) / np.sum(x * x))
        return {
            "n": int(x.size),
            "median_ratio": float(np.median(y / x)),
            "slope_through_origin": slope,
            "rms_m": float(np.sqrt(np.mean((y - x) ** 2))),
            "rms_to_slope_m": float(np.sqrt(np.mean((y - slope * x) ** 2))),
        }

    xs = [r["L_pred"] for r in records]
    ys = [r["L_meas"] for r in records]
    xh = [r["L_pred_h"] for r in records]
    stats = {"with_s": fit(xs, ys), "s_ignored": fit(xh, ys), "skips": skips,
             "candidate_columns": len(cand)}
    log(f"  fit h' {stats['with_s']}")
    log(f"  fit h  {stats['s_ignored']}")
    (OUT / "layover.json").write_text(json.dumps({"stats": stats, "buildings": [
        {k: rec[k] for k in ("id", "n", "L_meas", "L_pred", "L_pred_h", "h", "s", "theta", "wall_row", "img_c")}
        for rec in records
    ]}, indent=2))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.6))
    for a, x, title, st in (
        (ax[0], xs, "h' = h − s", stats["with_s"]),
        (ax[1], xh, "h, s ignored", stats["s_ignored"]),
    ):
        a.scatter(x, ys, s=12, c="0.2", alpha=0.8)
        lim = max([1] + [v for v in list(x) + ys if np.isfinite(v)] + [10])
        a.plot([0, lim], [0, lim], color="0.5", lw=1, label="1:1")
        if st.get("slope_through_origin"):
            a.plot([0, lim], [0, st["slope_through_origin"] * lim], color="C1", lw=1,
                   label=f"slope {st['slope_through_origin']:.2f}")
        a.set_xlabel("L predicted (m)")
        a.set_ylabel("L measured (m)")
        a.set_title(f"{title}\nratio {st.get('median_ratio', float('nan')):.2f}  "
                    f"rms {st.get('rms_m', float('nan')):.1f} m  n {st.get('n', 0)}")
        a.set_xlim(0, lim)
        a.set_ylim(0, lim)
        a.set_aspect("equal")
        a.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "layover_scatter.png", dpi=120)
    plt.close(fig)

    # 12 example crops
    if records:
        ratio = np.array([
            abs(r["L_meas"] / r["L_pred"] - 1) if r["L_pred"] > 2 else 9e9 for r in records
        ])
        order = np.argsort(ratio)
        n = len(records)
        mid = n // 2
        pick_i = list(order[:4]) + list(order[max(0, mid - 2):mid + 2]) + list(order[-4:])
        seen = set()
        picks = []
        for i in pick_i:
            if int(i) in seen:
                continue
            seen.add(int(i))
            picks.append(records[int(i)])
        for j, rec in enumerate(picks[:12]):
            wall = rec["wall_row"]
            col = rec["img_c"]
            span_m = max(rec["L_pred"], rec["L_meas"], 15) + 35
            r_lo = int(max(0, wall - span_m / GSD))
            r_hi = int(min(hh.shape[0], wall + 25 / GSD))
            c_lo = int(max(0, col - 80))
            c_hi = int(min(hh.shape[1], col + 80))
            crop = hh[r_lo:r_hi, c_lo:c_hi]
            fig, a = plt.subplots(figsize=(4.2, 6.2))
            v = crop[np.isfinite(crop)]
            lo, hi = (np.percentile(v, [2, 98]) if v.size else (0, 1))
            a.imshow(crop, cmap="gray", vmin=lo, vmax=hi,
                     extent=[c_lo, c_hi, r_hi, r_lo])
            a.axvline(col, color="white", lw=0.6)
            a.axhline(wall, color="white", lw=0.7)
            a.axhline(wall - rec["L_pred"] / GSD, color="cyan", lw=0.8)
            a.axhline(wall - rec["L_meas"] / GSD, color="orange", lw=0.8)
            a.set_title(f"h {rec['h']:.0f}  Lpred {rec['L_pred']:.0f}  Lmeas {rec['L_meas']:.0f}\n"
                        f"white wall  cyan pred  orange meas", fontsize=8)
            fig.tight_layout()
            fig.savefig(OUT / f"layover_{j:02d}_b{rec['id']}.png", dpi=100)
            plt.close(fig)
        log(f"  wrote {min(12, len(picks))} layover crops")
    log(f"wrote {OUT / 'layover.json'}")


if __name__ == "__main__":
    stages = sys.argv[1:] or ["basics", "prep", "project", "gates", "crops"]
    for s in stages:
        log(f"=== {s} ===")
        {"basics": stage_basics, "prep": stage_prep, "project": stage_project,
         "gates": stage_gates, "crops": stage_crops, "layover": stage_layover}[s]()

