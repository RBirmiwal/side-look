#!/usr/bin/env python3
"""Image-space labels for one strip. Does not read or write dataset v2 tiles.

Outputs go to sar/output/image_space/. v2 masks are read only, for the
registration and brightness comparison.
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin, xy
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from acquire_geometry import ecef_to_geodetic, enu_basis  # noqa: E402
from image_space import (  # noqa: E402
    LAYOVER,
    SHADOW,
    UNKNOWN,
    labels_from_coverage,
    project_grid,
    registration_check,
)
from near_field import (  # noqa: E402
    add_undulation,
    enu_from_map,
    fit_flight_line,
    look_azimuth_from_track,
    perpendicular_geometry,
)

STRIP = "20190822074237_20190822074521"
MAG = ROOT / "data" / f"SN6_AOI_11_Rotterdam_SAR-MAG-POL_{STRIP}.tif"
SLC_JSON = ROOT / "data" / "slc" / f"CAPELLA_ARL_SM_SLC_HH_{STRIP}_extended.json"
AHN = ROOT / "data" / "ahn_dsm_1m_aoi.tif"
HGT = ROOT / "data" / "srtm" / "N51E004.hgt.gz"
EGM = ROOT / "data" / "geoid" / "us_nga_egm96_15.tif"
V2_MASK = ROOT / "output" / "masks_v2" / f"{STRIP}.tif"
OUT = ROOT / "output" / "image_space"

PAD_M = 200.0
GSD = 0.25
PAD_PX = int(round(PAD_M / GSD))  # 800
CHUNK_COLS = 2000
NAN_MARGIN = 400
# Nodata components at least this large are water. Smaller holes stay NaN.
WATER_MIN_M2 = 2500


def log(msg: str) -> None:
    print(msg, flush=True)


def load_hgt(path: Path) -> np.ndarray:
    import gzip

    raw = gzip.decompress(path.read_bytes())
    side = int(round((len(raw) / 2) ** 0.5))
    return np.frombuffer(raw, dtype=">i2").reshape(side, side).astype(np.float32)


def sample_srtm(lon: np.ndarray, lat: np.ndarray, hgt: np.ndarray, orthometric: bool) -> np.ndarray:
    """Bilinear SRTM GL1. orthometric=False adds EGM96 so the result is ellipsoidal."""
    from pyproj import Transformer
    from scipy.ndimage import map_coordinates

    side = hgt.shape[0]
    d = 1.0 / (side - 1)
    row = (52.0 - lat) / d
    col = (lon - 4.0) / d
    z = map_coordinates(hgt, [row, col], order=1, mode="nearest")
    z = z.astype(np.float64)
    if orthometric:
        return z
    # N varies by centimetres across a chunk. Sample the geoid on a coarse grid.
    pipe = (
        "+proj=pipeline +step +proj=vgridshift "
        f"+grids={EGM} +multiplier=1"
    )
    tr = Transformer.from_pipeline(pipe)
    step = 40
    rr = np.unique(np.concatenate([np.arange(0, lat.shape[0], step), [lat.shape[0] - 1]]))
    cc = np.unique(np.concatenate([np.arange(0, lat.shape[1], step), [lat.shape[1] - 1]]))
    rg, cg = np.meshgrid(rr, cc, indexing="ij")
    _, _, n = tr.transform(lon[rg, cg], lat[rg, cg], np.zeros(rg.shape))
    from scipy.interpolate import RegularGridInterpolator

    interp = RegularGridInterpolator((rr.astype(float), cc.astype(float)), n, bounds_error=False, fill_value=None)
    jj, ii = np.meshgrid(np.arange(lat.shape[1]), np.arange(lat.shape[0]))
    und = interp(np.stack([ii.ravel(), jj.ravel()], axis=1)).reshape(lat.shape)
    return z + und


def upsample(arr: np.ndarray, row0: int, col0: int, height: int, width: int, order: int) -> np.ndarray:
    from scipy.ndimage import map_coordinates

    rows = np.arange(row0, row0 + height)
    cols = np.arange(col0, col0 + width)
    r1 = 0.25 * rows - 0.375
    c1 = 0.25 * cols - 0.375
    rr, cc = np.meshgrid(r1, c1, indexing="ij")
    return map_coordinates(arr, [rr, cc], order=order, mode="nearest")


def prepare_ahn(west: float, south: float, east: float, north: float):
    """1 m NAP DSM on the padded UTM grid, water filled, other nodata left NaN."""
    from scipy.ndimage import binary_dilation, label

    height = int(np.ceil((north - south) / 1.0)) + 2
    width = int(np.ceil((east - west) / 1.0)) + 2
    transform = from_origin(west, north, 1.0, 1.0)
    dest = np.full((height, width), np.nan, dtype=np.float32)
    valid = np.zeros((height, width), dtype=np.uint8)
    with rasterio.open(AHN) as ahn:
        from pyproj import Transformer

        to_ahn = Transformer.from_crs("EPSG:32631", ahn.crs, always_xy=True)
        corners = [to_ahn.transform(x, y) for x, y in (
            (west - 3, south - 3), (east + 3, south - 3), (east + 3, north + 3), (west - 3, north + 3)
        )]
        ax = [c[0] for c in corners]
        ay = [c[1] for c in corners]
        from rasterio.windows import from_bounds

        win = from_bounds(min(ax), min(ay), max(ax), max(ay), transform=ahn.transform)
        win = win.round_offsets().round_lengths().intersection(Window(0, 0, ahn.width, ahn.height))
        z = ahn.read(1, window=win).astype(np.float32)
        src_t = rasterio.windows.transform(win, ahn.transform)
        ok = np.isfinite(z) & (np.abs(z) < 1.0e5)
        src = np.where(ok, z, np.nan).astype(np.float32)
        reproject(src, dest, src_transform=src_t, src_crs=ahn.crs, src_nodata=np.nan,
                  dst_transform=transform, dst_crs="EPSG:32631", dst_nodata=np.nan,
                  resampling=Resampling.bilinear)
        reproject(ok.astype(np.uint8), valid, src_transform=src_t, src_crs=ahn.crs,
                  dst_transform=transform, dst_crs="EPSG:32631", resampling=Resampling.nearest)
    nodata = valid == 0
    comp, ncomp = label(nodata)
    water = np.zeros(nodata.shape, dtype=bool)
    for i in range(1, ncomp + 1):
        sel = comp == i
        if int(sel.sum()) >= WATER_MIN_M2:
            water[sel] = True
    edge = binary_dilation(water, iterations=2) & ~water & (valid == 1)
    border = dest[edge]
    border = border[np.isfinite(border)]
    low = border[border < 2.0]
    if low.size > 100:
        level = float(np.median(low))
        how = "median of valid pixels within 2 m of water and below 2 m NAP"
    elif border.size:
        level = float(np.percentile(border, 10))
        how = "10th percentile of the water edge (nothing below 2 m NAP)"
    else:
        level = 0.0
        how = "fallback 0 m NAP; no water edge"
    filled = dest.copy()
    filled[water] = level
    # bilinear reproject smeared nodata into neighbours; those stay NaN unless water
    filled[nodata & ~water] = np.nan
    log(f"  water level {level:.3f} m NAP ({how})")
    log(f"  water {water.mean():.3%} of the 1 m grid, other nodata {(nodata & ~water).mean():.3%}")
    if border.size:
        log(f"  water-edge NAP percentiles "
            f"{np.percentile(border, [5, 25, 50, 75, 95]).round(2).tolist()}")
    return filled, water, transform, level


def open_outputs(path_lab, path_cov, path_water, transform, height, width):
    profile = dict(driver="GTiff", height=height, width=width, count=1,
                   crs="EPSG:32631", transform=transform, compress="deflate", tiled=True,
                   blockxsize=512, blockysize=512)
    lab = rasterio.open(path_lab, "w", dtype="uint8", nodata=255, **profile)
    cov = rasterio.open(path_cov, "w", dtype="float32", nodata=np.nan, **profile)
    water = rasterio.open(path_water, "w", dtype="uint8", **profile)
    return lab, cov, water


def project_strip(mode: str, hgt, flight, geo, z_nap, water_1m, t1, out_dir: Path):
    """mode: ellipsoidal | orthometric | flat."""
    with rasterio.open(MAG) as src:
        height, width = src.height, src.width
        img_t = src.transform
    west, north = img_t.c, img_t.f
    south = north + img_t.e * height
    pad_n = north + PAD_M
    pad_s = south - PAD_M
    full_h = height + 2 * PAD_PX
    # 1 m grid was built on these bounds by the caller
    z_ell = add_undulation(z_nap.astype(np.float64), t1, "EPSG:32631").astype(np.float32)
    log(f"  DSM ellipsoidal {np.nanmin(z_ell):.2f}..{np.nanmax(z_ell):.2f} m")

    tag = {"ellipsoidal": "egm96", "orthometric": "nogeoid", "flat": "flat"}[mode]
    sub = out_dir / tag
    sub.mkdir(parents=True, exist_ok=True)
    out_t = img_t
    lab_ds, cov_ds, wat_ds = open_outputs(
        sub / f"{STRIP}_lab.tif", sub / f"{STRIP}_cov.tif", sub / f"{STRIP}_water.tif",
        out_t, height, width,
    )
    # alpha direction from the centre column, once
    mid = width // 2
    z_col = upsample(z_ell, 0, mid, full_h, 1, order=1)[:, 0]
    cols = np.array([mid])
    rows = np.arange(full_h)
    xs = west + (cols[None, :] + 0.5) * GSD
    ys = pad_n - (rows[:, None] + 0.5) * GSD
    xs = np.broadcast_to(xs, (full_h, 1)).copy()
    east, north_enu, up = enu_from_map(xs, ys, z_col[:, None], "EPSG:32631",
                                        geo["lat"], geo["lon"], geo["height"])
    _, alpha0 = perpendicular_geometry(east, north_enu, up, flight["origin_enu"], flight["direction_enu"])
    dalpha = float(np.nanmean(alpha0[-50:]) - np.nanmean(alpha0[:50]))
    flip = dalpha < 0
    log(f"  alpha(far)-alpha(near rows) {dalpha:+.4f} rad  flip_to_sensor_at_row0 {flip}")
    if flip:
        raise RuntimeError("this strip was expected south-looking with sensor at row 0")

    from pyproj import Transformer
    to_ll = Transformer.from_crs("EPSG:32631", "EPSG:4326", always_xy=True)

    for col0 in range(0, width, CHUNK_COLS):
        cw = min(CHUNK_COLS, width - col0)
        log(f"  chunk cols {col0}:{col0+cw}")
        z25 = upsample(z_ell, 0, col0, full_h, cw, order=1)
        w25 = upsample(water_1m.astype(np.float32), 0, col0, full_h, cw, order=0) > 0.5
        rows = np.arange(full_h)
        cols = np.arange(col0, col0 + cw)
        xs = west + (cols[None, :] + 0.5) * GSD
        ys = pad_n - (rows[:, None] + 0.5) * GSD
        xs = np.broadcast_to(xs, (full_h, cw)).copy()
        ys = np.broadcast_to(ys, (full_h, cw)).copy()
        if mode == "flat":
            zref = np.full((full_h, cw), geo["height"], np.float64)
        else:
            lon, lat = to_ll.transform(xs, ys)
            zref = sample_srtm(lon, lat, hgt, orthometric=(mode == "orthometric"))
            del lon, lat
        # r_ref is the reference-surface point of the same cell, not the DSM point
        east, north_enu, up = enu_from_map(
            xs, ys, zref, "EPSG:32631", geo["lat"], geo["lon"], geo["height"]
        )
        r_ref, _ = perpendicular_geometry(
            east, north_enu, up, flight["origin_enu"], flight["direction_enu"]
        )
        del east, north_enu, up, zref
        east, north_enu, up = enu_from_map(
            xs, ys, z25, "EPSG:32631", geo["lat"], geo["lon"], geo["height"]
        )
        del xs, ys, z25
        slant, alpha = perpendicular_geometry(
            east, north_enu, up, flight["origin_enu"], flight["direction_enu"]
        )
        del east, north_enu, up
        gc.collect()
        if not np.isfinite(r_ref).all():
            raise RuntimeError("r_ref has NaN; the reference surface does not cover the chunk")
        cov, unk, _vis = project_grid(slant, alpha, r_ref, axis=0, nan_margin=NAN_MARGIN)
        del slant, alpha, r_ref
        lab = labels_from_coverage(cov, unk)
        sl = slice(PAD_PX, PAD_PX + height)
        win = Window(col0, 0, cw, height)
        lab_ds.write(lab[sl].astype(np.uint8), 1, window=win)
        cov_ds.write(cov[sl].astype(np.float32), 1, window=win)
        wat_ds.write(w25[sl].astype(np.uint8), 1, window=win)
        del w25, cov, unk, lab
        gc.collect()
    lab_ds.close()
    cov_ds.close()
    wat_ds.close()
    return sub


def hh_db() -> np.ndarray:
    with rasterio.open(MAG) as src:
        return src.read(1)


def v2_as_image_space(path: Path, shape) -> np.ndarray:
    with rasterio.open(path) as src:
        m = src.read(1)
        if m.shape != shape:
            raise RuntimeError(f"v2 mask shape {m.shape} != image {shape}")
    out = np.zeros(shape, np.uint8)
    out[(m == 1) | (m == 2)] = LAYOVER
    out[m == 3] = SHADOW
    return out, m


def physics(hh, lab, v2, water) -> dict:
    dry = ~water & np.isfinite(hh) & (lab != UNKNOWN)

    def med(sel):
        if sel.sum() < 100:
            return float("nan")
        return float(np.median(hh[sel]))

    thirds = {}
    h = hh.shape[0]
    edges = [0, h // 3, 2 * h // 3, h]
    names = ["near", "middle", "far", "all"]
    rows = [hh.shape[0]]  # placeholder
    out = {}
    nom_all = med(dry & (lab == 0))
    for name, lo, hi in zip(["near", "middle", "far"], edges[:-1], edges[1:]):
        band = np.zeros(hh.shape, bool)
        band[lo:hi] = True
        base = dry & band
        nom = med(base & (lab == 0))
        out[name] = {
            "v3_layover": med(base & (lab == LAYOVER)) - nom,
            "v3_shadow": med(base & (lab == SHADOW)) - nom,
            "v2_active": med(base & (v2 == 1)) - med(base & (v2 == 0)),
            "v2_passive": med(base & (v2 == 2)) - med(base & (v2 == 0)),
            "v2_shadow": med(base & (v2 == 3)) - med(base & (v2 == 0)),
            "n_v3_shadow": int((base & (lab == SHADOW)).sum()),
            "n_v3_layover": int((base & (lab == LAYOVER)).sum()),
        }
    base = dry
    nom = nom_all
    out["all"] = {
        "v3_layover": med(base & (lab == LAYOVER)) - nom,
        "v3_shadow": med(base & (lab == SHADOW)) - nom,
        "v2_active": med(base & (v2 == 1)) - med(base & (v2 == 0)),
        "v2_passive": med(base & (v2 == 2)) - med(base & (v2 == 0)),
        "v2_shadow": med(base & (v2 == 3)) - med(base & (v2 == 0)),
        "n_v3_shadow": int((base & (lab == SHADOW)).sum()),
        "n_v3_layover": int((base & (lab == LAYOVER)).sum()),
        "frac_v3": {
            "nominal": float((base & (lab == 0)).mean()),
            "layover": float((base & (lab == LAYOVER)).mean()),
            "shadow": float((base & (lab == SHADOW)).mean()),
            "unknown": float((~water & (lab == UNKNOWN)).mean()),
        },
    }
    return out


def render_crops(hh, lab, v2, cov, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    score = np.zeros(hh.shape[1] // 512, dtype=np.int64)
    for i in range(score.size):
        c0 = i * 512
        w = lab[:, c0:c0 + 512]
        score[i] = int(((w == LAYOVER) | (w == SHADOW)).sum())
    # six windows, spread out, preferring mixed layover and shadow
    picks = []
    order = np.argsort(score)[::-1]
    for i in order:
        if score[i] < 500:
            break
        c0 = int(i * 512)
        if any(abs(c0 - p) < 2500 for p in picks):
            continue
        picks.append(c0)
        if len(picks) == 6:
            break
    picks.sort()
    v2_cmap = ListedColormap([(0, 0, 0, 0), (0.85, 0.45, 0.15, 0.7),
                              (0.55, 0.40, 0.22, 0.7), (0.2, 0.45, 0.85, 0.7)])
    v3_cmap = ListedColormap([(0, 0, 0, 0), (0.85, 0.45, 0.15, 0.75), (0.2, 0.45, 0.85, 0.75)])
    # unknown drawn separately
    for k, c0 in enumerate(picks):
        r0 = (hh.shape[0] - 512) // 2
        sl = (slice(r0, r0 + 512), slice(c0, c0 + 512))
        hhc, lc, vc, cc = hh[sl], lab[sl], v2[sl], cov[sl]
        lo, hi = np.nanpercentile(hhc, [2, 98])
        fig, ax = plt.subplots(1, 4, figsize=(16, 4.6))
        ax[0].imshow(hhc, cmap="gray", vmin=lo, vmax=hi)
        ax[0].set_title("HH dB")
        ax[1].imshow(hhc, cmap="gray", vmin=lo, vmax=hi)
        ax[1].imshow(np.ma.masked_where(vc == 0, vc), cmap=v2_cmap, vmin=0, vmax=3, interpolation="nearest")
        ax[1].set_title("v2 ground-space")
        show = lc.copy()
        show[show == UNKNOWN] = 0
        ax[2].imshow(hhc, cmap="gray", vmin=lo, vmax=hi)
        ax[2].imshow(np.ma.masked_where(show == 0, show), cmap=v3_cmap, vmin=0, vmax=2, interpolation="nearest")
        ax[2].set_title("v3 image-space")
        ax[3].imshow(cc, cmap="magma", vmin=0, vmax=3)
        ax[3].set_title("coverage")
        for a in ax:
            a.axis("off")
        fig.suptitle(f"row {r0} col {c0}")
        fig.tight_layout()
        fig.savefig(out_dir / f"crop_{k:02d}_{r0}_{c0}.png", dpi=90)
        plt.close(fig)
        log(f"  crop {k} row {r0} col {c0}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mag_desc = json.loads(rasterio.open(MAG).tags()["TIFFTAG_IMAGEDESCRIPTION"])
    slc = json.loads(SLC_JSON.read_text())
    for name, collect in (("MAG-POL", mag_desc["collect"]), ("SLC", slc["collect"])):
        ig = collect["image"]["image_geometry"]
        log(f"{name} algorithm {collect['image'].get('algorithm')}")
        log(f"{name} image_geometry.type {ig.get('type')} surface_type {ig.get('surface_type')} "
            f"z {ig.get('z')}")
    with rasterio.open(MAG) as src:
        log(f"MAG raster {src.height} x {src.width} crs {src.crs} transform {src.transform}")
    slc_tif = SLC_JSON.with_name(SLC_JSON.name.replace("_extended.json", ".tif"))
    with rasterio.open(slc_tif) as src:
        log(f"SLC raster {src.height} x {src.width} crs {src.crs} transform {src.transform}")
    ig = mag_desc["collect"]["image"]["image_geometry"]
    if ig.get("type") != "surface" or ig.get("z", {}).get("dem_name") != "SrtmGl1":
        raise SystemExit("STOP: MAG-POL is not surface / SrtmGl1")

    collect = mag_desc["collect"]
    target = np.array(collect["image"]["center_pixel"]["target_position"], dtype=np.float64)
    lat, lon, height = ecef_to_geodetic(target)
    log(f"centre target geodetic {lat:.6f} N {lon:.6f} E  h {height:.3f} m")
    hgt = load_hgt(HGT)
    # height gate at the single target point
    from pyproj import Transformer
    tr = Transformer.from_pipeline(
        f"+proj=pipeline +step +proj=vgridshift +grids={EGM} +multiplier=1"
    )
    _, _, n = tr.transform(lon, lat, 0.0)
    d = 1.0 / (hgt.shape[0] - 1)
    row = (52.0 - lat) / d
    col = (lon - 4.0) / d
    from scipy.ndimage import map_coordinates
    ortho = float(map_coordinates(hgt, [[row], [col]], order=1)[0])
    ell = ortho + float(n)
    log(f"SRTM orthometric {ortho:.3f} m  EGM96 N {float(n):.3f} m  ellipsoidal {ell:.3f} m  "
        f"minus target {ell - height:+.3f} m")
    if abs(ell - height) > 2.0:
        raise SystemExit("STOP: SRTM ellipsoidal height is more than 2 m off the centre target")

    basis = enu_basis(lat, lon)
    pos = np.array([sv["position"] for sv in collect["state"]["state_vectors"]], dtype=np.float64)
    enu = (basis @ (pos - target).T).T
    fit = fit_flight_line(enu)
    look = look_azimuth_from_track(fit["direction_enu"], collect["radar"]["pointing"])
    log(f"look {look:.3f} deg  residual rms {fit['residual_rms_m']:.2f} m  "
        f"platform {fit['height_above_origin_m']:.1f} m above target")
    flight = {"origin_enu": fit["origin_enu"], "direction_enu": fit["direction_enu"]}
    geo = {"lat": lat, "lon": lon, "height": height}

    with rasterio.open(MAG) as src:
        west, north = src.transform.c, src.transform.f
        south = north + src.transform.e * src.height
        east = west + src.transform.a * src.width
    log("AHN 1 m, water fill")
    z_nap, water_1m, t1, level = prepare_ahn(west, south - PAD_M, east, north + PAD_M)

    modes = sys.argv[1:]
    if modes:
        for mode in modes:
            sub = project_strip(mode, hgt, flight, geo, z_nap, water_1m, t1, OUT)
            log(f"wrote {sub}  water level {level:.3f} m NAP")
        return

    sub = project_strip("ellipsoidal", hgt, flight, geo, z_nap, water_1m, t1, OUT)
    log(f"wrote {sub}")

    log("registration")
    hh = hh_db()
    with rasterio.open(sub / f"{STRIP}_lab.tif") as src:
        lab = src.read(1)
    with rasterio.open(sub / f"{STRIP}_water.tif") as src:
        water = src.read(1).astype(bool)
    with rasterio.open(sub / f"{STRIP}_cov.tif") as src:
        # coverage is large; read only for crops later if registration needs it not
        cov_path = sub / f"{STRIP}_cov.tif"
    rows, med = registration_check(hh, lab)
    for centre, dy, dx, pk in rows:
        log(f"  v3 window col {centre:6d}  dy {dy:+4d}  dx {dx:+4d}  peak {pk:.3f}")
    log(f"  v3 median dy {med[0]:+.1f}  dx {med[1]:+.1f}  peak {med[2]:.3f}")
    v2_img, v2_raw = v2_as_image_space(V2_MASK, hh.shape)
    rows_v2, med_v2 = registration_check(hh, v2_img)
    for centre, dy, dx, pk in rows_v2:
        log(f"  v2 window col {centre:6d}  dy {dy:+4d}  dx {dx:+4d}  peak {pk:.3f}")
    log(f"  v2 median dy {med_v2[0]:+.1f}  dx {med_v2[1]:+.1f}  peak {med_v2[2]:.3f}")

    passed = abs(med[0]) <= 2 and abs(med[1]) <= 2 and med[2] > 0.1
    alts = {}
    if not passed:
        log("REGISTRATION FAILED — rerunning orthometric SRTM and a flat reference")
        for mode in ("orthometric", "flat"):
            sub_a = project_strip(mode, hgt, flight, geo, z_nap, water_1m, t1, OUT)
            with rasterio.open(sub_a / f"{STRIP}_lab.tif") as src:
                lab_a = src.read(1)
            rows_a, med_a = registration_check(hh, lab_a)
            log(f"  {mode} median dy {med_a[0]:+.1f} dx {med_a[1]:+.1f} peak {med_a[2]:.3f}")
            alts[mode] = {"median": [float(x) for x in med_a],
                          "windows": [[int(a), int(b), int(c), float(d)] for a, b, c, d in rows_a]}
            for centre, dy, dx, pk in rows_a:
                log(f"    {mode} col {centre:6d} dy {dy:+4d} dx {dx:+4d} peak {pk:.3f}")

    log("physics (water excluded)")
    with rasterio.open(cov_path) as src:
        cov = src.read(1)
    stats = physics(hh, lab, v2_raw, water)
    for name in ("all", "near", "middle", "far"):
        s = stats[name]
        log(f"  {name:6}  v3 lay {s['v3_layover']:+.3f}  v3 sh {s['v3_shadow']:+.3f}  "
            f"v2 act {s['v2_active']:+.3f}  v2 pas {s['v2_passive']:+.3f}  v2 sh {s['v2_shadow']:+.3f}")
    log("crops")
    render_crops(hh, lab, v2_raw, cov, sub)
    report = {
        "strip": STRIP,
        "water_level_nap_m": level,
        "water_min_m2": WATER_MIN_M2,
        "srtm_ellipsoidal_m": ell,
        "target_height_m": height,
        "srtm_minus_target_m": ell - height,
        "look_deg": look,
        "registration_v3": {"windows": [[int(a), int(b), int(c), float(d)] for a, b, c, d in rows],
                            "median": [float(x) for x in med],
                            "pass": bool(passed)},
        "registration_v2": {"windows": [[int(a), int(b), int(c), float(d)] for a, b, c, d in rows_v2],
                            "median": [float(x) for x in med_v2]},
        "alternatives": alts,
        "physics": stats,
    }
    (sub / "report.json").write_text(json.dumps(report, indent=2))
    log(f"report {sub / 'report.json'}")
    sh_ok = stats["all"]["v3_shadow"] <= -1.5 and all(
        stats[k]["v3_shadow"] < 0 for k in ("near", "middle", "far")
    )
    lay_ok = stats["all"]["v3_layover"] >= 2.0
    log(f"registration pass {passed}  shadow pass {sh_ok}  layover pass {lay_ok}")


if __name__ == "__main__":
    main()
