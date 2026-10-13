#!/usr/bin/env python3
"""Where this image puts flat ground and buildings. Measurement only.

Does not modify image_space.py, dataset v2, or sar/output/image_space/.
Writes only to sar/output/shift_check/.
"""

from __future__ import annotations

import csv
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
from image_space import _xcorr_peak  # noqa: E402
from image_space_run import load_hgt, sample_srtm  # noqa: E402
from near_field import (  # noqa: E402
    add_undulation,
    enu_from_map,
    fit_flight_line,
    look_azimuth_from_track,
    perpendicular_geometry,
)

STRIP = "20190822074237_20190822074521"
MAG = ROOT / "data" / f"SN6_AOI_11_Rotterdam_SAR-MAG-POL_{STRIP}.tif"
SLC_TIF = ROOT / "data" / "slc" / f"CAPELLA_ARL_SM_SLC_HH_{STRIP}.tif"
AHN = ROOT / "data" / "ahn_dsm_1m_aoi.tif"
HGT = ROOT / "data" / "srtm" / "N51E004.hgt.gz"
WATER = ROOT / "output" / "image_space" / "egm96" / f"{STRIP}_water.tif"
BAG = ROOT / "data" / "3dbag"
OUT = ROOT / "output" / "shift_check"

GSD = 0.25
WIN = 1024
SEARCH = 300


def log(msg: str) -> None:
    print(msg, flush=True)


def gpkg_wkb(blob: bytes) -> bytes:
    if blob[:2] != b"GP":
        return blob
    env = (blob[3] >> 1) & 0b111
    env_bytes = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[env]
    return blob[8 + env_bytes:]


def verify_xcorr_sign() -> None:
    a = np.zeros((128, 128), np.float32)
    a[40:70, 40:70] = 1
    b = np.zeros((128, 128), np.float32)
    b[48:78, 43:73] = 1
    dy, dx, pk = _xcorr_peak(a, b, 30)
    if (dy, dx) != (-8, -3) or pk < 0.9:
        raise RuntimeError(f"xcorr sign check failed: dy={dy} dx={dx} peak={pk}")
    log("xcorr sign ok: +dy moves the second image to a larger row, away from the sensor")


def flight_from_mag():
    desc = json.loads(rasterio.open(MAG).tags()["TIFFTAG_IMAGEDESCRIPTION"])
    collect = desc["collect"]
    target = np.array(collect["image"]["center_pixel"]["target_position"], dtype=np.float64)
    lat, lon, height = ecef_to_geodetic(target)
    basis = enu_basis(lat, lon)
    pos = np.array([sv["position"] for sv in collect["state"]["state_vectors"]], dtype=np.float64)
    enu = (basis @ (pos - target).T).T
    fit = fit_flight_line(enu)
    look = look_azimuth_from_track(fit["direction_enu"], collect["radar"]["pointing"])
    flight = {"origin_enu": fit["origin_enu"], "direction_enu": fit["direction_enu"]}
    geo = {"lat": float(lat), "lon": float(lon), "height": float(height)}
    log(f"look {look:.3f} deg  platform {fit['height_above_origin_m']:.1f} m above target")
    return flight, geo


def local_ground_5th(dsm: np.ndarray, radius_m: int = 50) -> np.ndarray:
    """5th percentile of finite DSM within radius_m. No AHN DTM is on disk."""
    from scipy.ndimage import uniform_filter

    size = 2 * radius_m + 1
    valid = np.isfinite(dsm)
    lo, step = -6.0, 0.5
    edges = np.arange(lo, 25.0, step)
    zi = np.zeros(dsm.shape, np.int16)
    zi[valid] = np.clip(np.floor((dsm[valid] - lo) / step), 0, edges.size - 1).astype(np.int16)
    n_valid = uniform_filter(valid.astype(np.float32), size=size, mode="nearest")
    acc = np.zeros(dsm.shape, np.float32)
    ground = np.full(dsm.shape, np.nan, np.float32)
    filled = np.zeros(dsm.shape, bool)
    target = 0.05 * n_valid
    for k, edge in enumerate(edges):
        m = uniform_filter((valid & (zi == k)).astype(np.float32), size=size, mode="nearest")
        acc += m
        take = (~filled) & (n_valid >= 0.05) & (acc >= target)
        ground[take] = edge + 0.5 * step
        filled |= take
        if k % 10 == 0:
            log(f"  ground bin {k}/{edges.size}  filled {filled.mean():.1%}")
    return ground


def build_prep(flight, geo):
    prep_path = OUT / "prep.npz"
    if prep_path.exists():
        log(f"loading {prep_path}")
        z = np.load(prep_path)
        return {k: z[k] for k in z.files}

    with rasterio.open(MAG) as src:
        west, north = float(src.transform.c), float(src.transform.f)
        height, width = src.height, src.width
    h1, w1 = height // 4, width // 4
    t1 = from_origin(west, north, 1.0, 1.0)
    log(f"reproject AHN to {h1} x {w1} at 1 m")
    dsm = np.full((h1, w1), np.nan, np.float32)
    with rasterio.open(AHN) as ahn:
        from pyproj import Transformer
        from rasterio.windows import from_bounds

        to_ahn = Transformer.from_crs("EPSG:32631", ahn.crs, always_xy=True)
        south = north - height * GSD
        east = west + width * GSD
        corners = [to_ahn.transform(x, y) for x, y in (
            (west, south), (east, south), (east, north), (west, north)
        )]
        ax = [c[0] for c in corners]
        ay = [c[1] for c in corners]
        win = from_bounds(min(ax) - 5, min(ay) - 5, max(ax) + 5, max(ay) + 5, transform=ahn.transform)
        win = win.round_offsets().round_lengths().intersection(Window(0, 0, ahn.width, ahn.height))
        src_z = ahn.read(1, window=win).astype(np.float32)
        src_t = rasterio.windows.transform(win, ahn.transform)
        ok = np.isfinite(src_z) & (np.abs(src_z) < 1.0e5)
        src_z = np.where(ok, src_z, np.nan).astype(np.float32)
        reproject(
            src_z, dsm, src_transform=src_t, src_crs=ahn.crs, src_nodata=np.nan,
            dst_transform=t1, dst_crs="EPSG:32631", dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    log(f"  DSM finite {np.isfinite(dsm).mean():.1%}  {np.nanmin(dsm):.2f}..{np.nanmax(dsm):.2f} m NAP")
    log("local ground: 5th percentile of DSM within 50 m")
    ground = local_ground_5th(dsm, 50)
    log(f"  ground {np.nanpercentile(ground[np.isfinite(ground)], [5, 50, 95])}")
    ndsm = (dsm - ground).astype(np.float32)
    log("NLGEO2018 and SRTM+EGM96")
    g_ell = add_undulation(np.where(np.isfinite(ground), ground, 0.0), t1, "EPSG:32631").astype(np.float32)
    g_ell[~np.isfinite(ground)] = np.nan
    from pyproj import Transformer

    to_ll = Transformer.from_crs("EPSG:32631", "EPSG:4326", always_xy=True)
    cols = np.arange(w1)
    rows = np.arange(h1)
    xs = np.broadcast_to(west + (cols[None, :] + 0.5), (h1, w1)).copy()
    ys = np.broadcast_to(north - (rows[:, None] + 0.5), (h1, w1)).copy()
    hgt = load_hgt(HGT)
    srtm = np.empty((h1, w1), np.float32)
    for c0 in range(0, w1, 1500):
        c1 = min(w1, c0 + 1500)
        lon, lat = to_ll.transform(xs[:, c0:c1], ys[:, c0:c1])
        srtm[:, c0:c1] = sample_srtm(lon, lat, hgt, orthometric=False).astype(np.float32)
        log(f"  srtm {c0}:{c1}")
    s = (srtm - g_ell).astype(np.float32)
    log(f"  s median {np.nanmedian(s):+.2f} m  "
        f"p10 {np.nanpercentile(s, 10):+.2f}  p90 {np.nanpercentile(s, 90):+.2f}")

    mid = w1 // 2
    zcol = g_ell[:, mid].astype(np.float64)
    good = np.isfinite(zcol)
    zcol[~good] = np.interp(np.flatnonzero(~good), np.flatnonzero(good), zcol[good])
    xs_c = np.full(h1, west + (mid + 0.5))
    ys_c = north - (np.arange(h1) + 0.5)
    east, north_enu, up = enu_from_map(
        xs_c, ys_c, zcol, "EPSG:32631", geo["lat"], geo["lon"], geo["height"]
    )
    _, alpha = perpendicular_geometry(east, north_enu, up, flight["origin_enu"], flight["direction_enu"])
    theta_row = np.repeat(np.degrees(alpha).astype(np.float32), 4)[:height]
    log(f"  theta near/mid/far {theta_row[height // 6]:.2f} / "
        f"{theta_row[height // 2]:.2f} / {theta_row[5 * height // 6]:.2f} deg")

    with rasterio.open(WATER) as src:
        water = src.read(1)
    water_1m = (water.reshape(h1, 4, w1, 4) > 0).max(axis=(1, 3))
    del water
    np.savez_compressed(
        prep_path, dsm=dsm, ground=ground, ndsm=ndsm, s=s, g_ell=g_ell, srtm=srtm,
        theta_row=theta_row, water_1m=water_1m,
        west=np.float64(west), north=np.float64(north),
    )
    log(f"wrote {prep_path}")
    return {
        "dsm": dsm, "ground": ground, "ndsm": ndsm, "s": s, "g_ell": g_ell, "srtm": srtm,
        "theta_row": theta_row, "water_1m": water_1m,
        "west": np.float64(west), "north": np.float64(north),
    }


def mag_window(row0, col0, h, w):
    with rasterio.open(MAG) as src:
        return src.read(1, window=Window(col0, row0, w, h)).astype(np.float32)


def slant_block(cols, z_ell, west, north, flight, geo):
    h, n = z_ell.shape
    rows = np.arange(h)
    xs = west + (np.asarray(cols)[None, :] + 0.5) * GSD
    ys = north - (rows[:, None] + 0.5) * GSD
    xs = np.broadcast_to(np.asarray(xs, np.float64), (h, n)).copy()
    ys = np.broadcast_to(np.asarray(ys, np.float64), (h, n)).copy()
    east, north_enu, up = enu_from_map(
        xs, ys, z_ell, "EPSG:32631", geo["lat"], geo["lon"], geo["height"]
    )
    return perpendicular_geometry(east, north_enu, up, flight["origin_enu"], flight["direction_enu"])


def upsample_cols(arr_1m, cols):
    from scipy.ndimage import map_coordinates

    h = arr_1m.shape[0] * 4
    rows = np.arange(h)
    r = 0.25 * rows - 0.375
    c = 0.25 * np.asarray(cols, np.float64) - 0.375
    rr, cc = np.meshgrid(r, c, indexing="ij")
    return map_coordinates(arr_1m, [rr, cc], order=1, mode="nearest")


def moved_water_column(water_col, r, r_ref):
    rows = np.arange(water_col.size, dtype=np.float64)
    rr = np.maximum.accumulate(np.asarray(r_ref, np.float64) + np.arange(water_col.size) * 1e-9)
    rg = np.asarray(r, np.float64)
    if not np.isfinite(rg).all():
        rg = np.where(np.isfinite(rg), rg, np.interp(rows, rows[np.isfinite(rg)], rg[np.isfinite(rg)]))
    dest = np.maximum.accumulate(np.interp(rg, rr, rows))
    src_at = np.interp(rows, dest, rows)
    idx = np.clip(np.rint(src_at), 0, water_col.size - 1).astype(np.int32)
    return water_col[idx]


def check1(prep, flight, geo):
    log("CHECK 1  flat ground, water edges")
    with rasterio.open(WATER) as src:
        water = src.read(1).astype(bool)
    H, W = water.shape
    ndsm = prep["ndsm"]
    theta_row = prep["theta_row"]
    west, north = float(prep["west"]), float(prep["north"])
    thirds = [("near", 0, H // 3), ("middle", H // 3, 2 * H // 3), ("far", 2 * H // 3, H)]
    centres = [int((i + 0.5) * W / 4) for i in range(4)]
    cand = []
    for name, r0s, r1s in thirds:
        span = r1s - r0s
        row0 = r0s + max(0, (span - WIN) // 2)
        for col_c in centres:
            col0 = int(col_c - WIN // 2)
            wsub = water[row0:row0 + WIN, col0:col0 + WIN]
            nsub = ndsm[row0 // 4:(row0 + WIN) // 4, col0 // 4:(col0 + WIN) // 4]
            cand.append({
                "third": name, "row0": int(row0), "col0": col0,
                "water_frac": float(wsub.mean()),
                "built_frac": float(np.nanmean(nsub > 3)),
                "theta": float(theta_row[row0 + WIN // 2]),
            })
    kept, relaxed = [], []
    for name, _, _ in thirds:
        group = [c for c in cand if c["third"] == name]
        ok = [c for c in group if 0.10 <= c["water_frac"] <= 0.90 and c["built_frac"] < 0.05]
        if not ok:
            ok = [c for c in group if 0.10 <= c["water_frac"] <= 0.90 and c["built_frac"] < 0.15]
            if ok:
                relaxed.append(name)
                log(f"  {name}: relaxed built limit to 15% ({len(ok)} windows)")
            else:
                log(f"  {name}: no window. water/built = "
                    + ", ".join(f"{c['water_frac']:.2f}/{c['built_frac']:.2f}" for c in group))
        else:
            log(f"  {name}: {len(ok)} windows")
        kept.extend(ok)
    if not kept:
        raise RuntimeError("check 1 found no usable windows")

    cols = np.unique(np.concatenate([np.arange(c["col0"], c["col0"] + WIN) for c in kept]))
    log(f"  remap on {cols.size} columns")
    z_dsm = upsample_cols(np.where(np.isfinite(prep["dsm"]), prep["dsm"], prep["ground"]), cols)
    z_g = upsample_cols(prep["ground"], cols)
    z_ell = upsample_cols(prep["g_ell"], cols)
    z_ref = upsample_cols(prep["srtm"], cols)
    n_und = z_ell - z_g
    above = (z_dsm - z_g) > 2.0
    z_surf = np.where(above, z_g, np.where(np.isfinite(z_dsm), z_dsm, z_g)) + n_und
    z_surf = np.where(np.isfinite(z_surf), z_surf, z_ref)
    r_ground = np.empty((H, cols.size), np.float32)
    r_ref = np.empty((H, cols.size), np.float32)
    for c0 in range(0, cols.size, 400):
        c1 = min(cols.size, c0 + 400)
        sl, _ = slant_block(cols[c0:c1], z_surf[:, c0:c1], west, north, flight, geo)
        sr, _ = slant_block(cols[c0:c1], z_ref[:, c0:c1], west, north, flight, geo)
        r_ground[:, c0:c1] = sl
        r_ref[:, c0:c1] = sr
        log(f"    slant {c0}:{c1}")
    col_index = {int(c): i for i, c in enumerate(cols)}
    mono = float(np.mean(np.diff(r_ref, axis=0) > 0))
    log(f"  r_ref increasing on {mono:.1%} of steps")

    results = []
    for c in kept:
        hh = mag_window(c["row0"], c["col0"], WIN, WIN)
        w = water[c["row0"]:c["row0"] + WIN, c["col0"]:c["col0"] + WIN].astype(np.float32)
        dy, dx, pk = _xcorr_peak(-hh, w, SEARCH)
        moved = np.empty_like(w)
        for k, col in enumerate(range(c["col0"], c["col0"] + WIN)):
            i = col_index[col]
            full = moved_water_column(water[:, col].astype(np.float32), r_ground[:, i], r_ref[:, i])
            moved[:, k] = full[c["row0"]:c["row0"] + WIN]
        dy2, dx2, pk2 = _xcorr_peak(-hh, moved, SEARCH)
        dy_m = float(dy) * GSD
        dz = dy_m * float(np.tan(np.radians(c["theta"])))
        i0, i1 = c["row0"] // 4, (c["row0"] + WIN) // 4
        j0, j1 = c["col0"] // 4, (c["col0"] + WIN) // 4
        s_med = float(np.nanmedian(prep["s"][i0:i1, j0:j1]))
        rec = {
            "third": c["third"], "row0": c["row0"], "col0": c["col0"],
            "water_frac": round(c["water_frac"], 3), "built_frac": round(c["built_frac"], 3),
            "relaxed_15": c["third"] in relaxed, "theta_deg": round(c["theta"], 2),
            "true_dy_m": round(dy_m, 2), "true_dx_m": round(float(dx) * GSD, 2),
            "true_dy_px": int(dy), "true_dx_px": int(dx), "true_peak": round(float(pk), 3),
            "srtm_dy_m": round(float(dy2) * GSD, 2), "srtm_dx_m": round(float(dx2) * GSD, 2),
            "srtm_dy_px": int(dy2), "srtm_dx_px": int(dx2), "srtm_peak": round(float(pk2), 3),
            "dz_m": round(dz, 2), "s_measured_m": round(s_med, 2),
        }
        results.append(rec)
        log(f"  {c['third']:6} col {c['col0']:5d}  true dy {dy_m:+6.1f} m dx {dx * GSD:+5.1f} pk {pk:.3f}"
            f"   srtm dy {dy2 * GSD:+6.1f} m pk {pk2:.3f}   dz {dz:+5.1f} s {s_med:+5.1f}")
    summary = {"relaxed_thirds": relaxed, "windows": results, "r_ref_increasing": mono}
    for name, _, _ in thirds:
        g = [r for r in results if r["third"] == name]
        if not g:
            continue
        def med(key, g=g):
            return float(np.median([r[key] for r in g]))
        summary[name] = {
            "n": len(g),
            "true_dy_m": round(med("true_dy_m"), 2), "true_dx_m": round(med("true_dx_m"), 2),
            "true_peak": round(med("true_peak"), 3),
            "srtm_dy_m": round(med("srtm_dy_m"), 2), "srtm_peak": round(med("srtm_peak"), 3),
            "dz_m": round(med("dz_m"), 2), "s_measured_m": round(med("s_measured_m"), 2),
            "theta_deg": round(med("theta_deg"), 2),
        }
        s = summary[name]
        log(f"  MEDIAN {name:6} true dy {s['true_dy_m']:+.2f} m  srtm dy {s['srtm_dy_m']:+.2f} m  "
            f"dz {s['dz_m']:+.2f}  s {s['s_measured_m']:+.2f}")
    (OUT / "check1.json").write_text(json.dumps(summary, indent=2))
    with (OUT / "check1_windows.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    return summary


def raster_footprint(transform, height, width):
    corners = [(0, 0), (0, width), (height, width), (height, 0)]
    return [[round(float(v), 2) for v in xy(transform, r, c, offset="ul")] for r, c in corners]


def check2(prep):
    log("CHECK 2  SLC onto the MAG-POL grid")
    from scipy.ndimage import uniform_filter

    with rasterio.open(MAG) as mag:
        H, W = mag.height, mag.width
        mag_t = mag.transform
        mag_corners = raster_footprint(mag_t, H, W)
    with rasterio.open(SLC_TIF) as slc:
        slc_t = slc.transform
        slc_h, slc_w = slc.height, slc.width
        slc_corners = raster_footprint(slc_t, slc_h, slc_w)
        x00, y00 = xy(slc_t, 0, 0, offset="center")
        x10, y10 = xy(slc_t, 1, 0, offset="center")
        x01, y01 = xy(slc_t, 0, 1, offset="center")
        along = float(np.hypot(x10 - x00, y10 - y00))
        cross = float(np.hypot(x01 - x00, y01 - y00))
        log(f"  SLC spacing row-axis {along:.4f} m  col-axis {cross:.4f} m")
        log(f"  SLC raster {slc_w} x {slc_h} (col x row)   MAG {W} x {H}")
        z = slc.read(1)
        db = (20.0 * np.log10(np.abs(z).astype(np.float32) + 1e-6)).astype(np.float32)
        del z
        slc_crs = slc.crs
    info = {
        "slc_spacing_row_m": round(along, 4),
        "slc_spacing_col_m": round(cross, 4),
        "slc_size_col_row": [slc_w, slc_h],
        "mag_size_col_row": [W, H],
        "mag_corners_e_n": mag_corners,
        "slc_corners_e_n": slc_corners,
        "north_origin_gap_m": round(mag_corners[0][1] - slc_corners[0][1], 2),
    }
    info["why_sizes_differ"] = (
        f"Pixel size is 0.25 m on both axes. The SLC affine is rotated 0.87 deg: "
        f"its row axis runs at azimuth 89.13 deg and is {(W - slc_h) * GSD:.1f} m shorter than MAG-POL, "
        f"and its column axis runs at bearing 179.13 deg for {slc_w * cross:.1f} m "
        f"against the MAG-POL range of {H * GSD:.1f} m. "
        f"The {H * GSD - slc_w * cross:.1f} m gap is a narrower swath, not a different pixel size. "
        f"The rotation lifts the east end by 180 m, so the bounding boxes differ in northing by that much. "
        f"The south-west corners agree "
        f"({min(c[1] for c in slc_corners):.2f} N vs MAG south {min(c[1] for c in mag_corners):.2f})."
    )
    log("  " + info["why_sizes_differ"])
    dest = np.full((H, W), np.nan, np.float32)
    log("  reproject")
    reproject(
        db, dest, src_transform=slc_t, src_crs=slc_crs, src_nodata=np.nan,
        dst_transform=mag_t, dst_crs="EPSG:32631", dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    del db
    covered = np.isfinite(dest)
    info["slc_coverage_of_mag"] = round(float(covered.mean()), 3)
    log(f"  SLC covers {covered.mean():.1%} of MAG pixels")
    filled = uniform_filter(np.where(covered, dest, 0.0).astype(np.float32), size=5)
    dest = np.where(covered, filled, np.nan).astype(np.float32)
    np.save(OUT / "slc_on_mag.npy", dest)

    ndsm = prep["ndsm"]
    theta_row = prep["theta_row"]
    rows = np.arange(H)
    northing = mag_t.f + (rows + 0.5) * mag_t.e
    slc_north, slc_south = slc_corners[0][1], min(c[1] for c in slc_corners)
    row_inside = (northing <= slc_north + 0.01) & (northing >= slc_south - 0.01)
    thirds = [("near", 0, H // 3), ("middle", H // 3, 2 * H // 3), ("far", 2 * H // 3, H)]
    results = []
    for qi in range(4):
        a, b = int(qi * W / 4), int((qi + 1) * W / 4)
        for name, r0s, r1s in thirds:
            span = r1s - r0s
            row0 = r0s + max(0, (span - WIN) // 2)
            overlap = float(row_inside[row0:row0 + WIN].mean())
            best = {"buildings": None, "open": None}
            for col0 in range(a, max(a + 1, b - WIN + 1), 256):
                frac = float(np.nanmean(ndsm[row0 // 4:(row0 + WIN) // 4, col0 // 4:(col0 + WIN) // 4] > 3))
                if frac >= 0.20 and (best["buildings"] is None or frac > best["buildings"][0]):
                    best["buildings"] = (frac, col0)
                if frac < 0.02 and (best["open"] is None or frac < best["open"][0]):
                    best["open"] = (frac, col0)
            for kind in ("buildings", "open"):
                if best[kind] is None:
                    log(f"  zone {qi} {name:6} {kind:9} NONE")
                    results.append({"zone": qi, "third": name, "kind": kind, "found": False,
                                    "overlap": round(overlap, 3)})
                    continue
                frac, col0 = best[kind]
                hh = uniform_filter(np.nan_to_num(mag_window(row0, col0, WIN, WIN), nan=0.0), size=5)
                sl = dest[row0:row0 + WIN, col0:col0 + WIN]
                sel = row_inside[row0:row0 + WIN]
                if float(sel.mean()) < 0.4 or int(sel.sum()) < 64:
                    dy = dx = pk = float("nan")
                else:
                    dy, dx, pk = _xcorr_peak(hh[sel], np.nan_to_num(sl[sel], nan=0.0),
                                             min(SEARCH, int(sel.sum()) // 4))
                rec = {
                    "zone": qi, "third": name, "kind": kind, "found": True,
                    "row0": int(row0), "col0": int(col0), "built_frac": round(frac, 3),
                    "overlap": round(overlap, 3),
                    "dy_m": None if dy != dy else round(float(dy) * GSD, 2),
                    "dx_m": None if dx != dx else round(float(dx) * GSD, 2),
                    "dy_px": None if dy != dy else int(dy),
                    "dx_px": None if dx != dx else int(dx),
                    "peak": None if pk != pk else round(float(pk), 3),
                    "theta_deg": round(float(theta_row[row0 + WIN // 2]), 2),
                }
                results.append(rec)
                log(f"  zone {qi} {name:6} {kind:9} col {col0:5d} built {frac:.2f} "
                    f"overlap {overlap:.2f} dy {rec['dy_m']} dx {rec['dx_m']} pk {rec['peak']}")
    info["windows"] = results
    (OUT / "check2.json").write_text(json.dumps(info, indent=2))
    return info


def load_footprints(prep):
    import sqlite3
    from pyproj import Transformer
    from rasterio.features import rasterize
    from shapely import wkb
    from shapely.ops import transform as shp_transform

    h1, w1 = prep["ndsm"].shape
    west, north = float(prep["west"]), float(prep["north"])
    t1 = from_origin(west, north, 1.0, 1.0)
    to_utm = Transformer.from_crs("EPSG:28992", "EPSG:32631", always_xy=True).transform
    # strip bounds in RD, with margin, so we don't rasterize the whole city
    from pyproj import Transformer as T
    to_rd = T.from_crs("EPSG:32631", "EPSG:28992", always_xy=True)
    south = north - h1
    east = west + w1
    rd = [to_rd.transform(x, y) for x, y in ((west, south), (east, south), (east, north), (west, north))]
    xmin, xmax = min(p[0] for p in rd) - 30, max(p[0] for p in rd) + 30
    ymin, ymax = min(p[1] for p in rd) - 30, max(p[1] for p in rd) + 30
    shapes, meta = [], []
    for gpkg in sorted(BAG.glob("*.gpkg")):
        con = sqlite3.connect(gpkg)
        q = """SELECT p.b3_h_maaiveld, p.b3_h_nok, p.b3_opp_grond, p.geom
               FROM pand p, rtree_pand_geom r
               WHERE p.fid = r.id AND r.maxx >= ? AND r.minx <= ? AND r.maxy >= ? AND r.miny <= ?"""
        for maaiveld, nok, opp, blob in con.execute(q, (xmin, xmax, ymin, ymax)):
            if blob is None or maaiveld is None or nok is None:
                continue
            try:
                geom = shp_transform(to_utm, wkb.loads(gpkg_wkb(blob)))
            except Exception:
                continue
            if geom.is_empty:
                continue
            shapes.append((geom, len(shapes) + 1))
            meta.append({"h_bag": float(nok - maaiveld), "area_m2": None if opp is None else float(opp)})
        con.close()
    log(f"  rasterizing {len(shapes)} 3DBAG footprints")
    labels = rasterize(shapes, out_shape=(h1, w1), transform=t1, fill=0, dtype=np.int32, all_touched=True)
    return labels, meta


def check3(prep):
    log("CHECK 3  buildings by height")
    from scipy.ndimage import find_objects

    labels, meta = load_footprints(prep)
    ndsm, s_grid = prep["ndsm"], prep["s"]
    theta_row = prep["theta_row"]
    H, W = int(theta_row.size), labels.shape[1] * 4
    log("  loading HH")
    with rasterio.open(MAG) as src:
        hh = src.read(1).astype(np.float32)
    with rasterio.open(WATER) as src:
        water = src.read(1).astype(bool)
    hist = np.zeros(6000, np.int64)
    lo, step = -80.0, 0.02
    land = (~water) & np.isfinite(hh)
    b = np.floor((hh[land] - lo) / step).astype(np.int32)
    b = b[(b >= 0) & (b < hist.size)]
    hist += np.bincount(b, minlength=hist.size)
    land_med = lo + (int(np.searchsorted(np.cumsum(hist), (hist.sum() + 1) / 2)) + 0.5) * step
    log(f"  median land HH {land_med:.2f} dB")
    del land, b

    buildings = []
    for i, sl in enumerate(find_objects(labels), start=1):
        if sl is None:
            continue
        own = labels[sl] == i
        if int(own.sum()) < 20:
            continue
        vals = ndsm[sl][own]
        vals = vals[np.isfinite(vals)]
        if vals.size < 20:
            continue
        h = float(np.percentile(vals, 70))
        if h < 3.0:
            continue
        cols = np.array([c for c in range(sl[1].start, sl[1].stop) if np.any(labels[:, c] == i)])
        if cols.size < 2:
            continue
        near, far = {}, {}
        for c in cols:
            rr = np.where(labels[:, c] == i)[0]
            near[int(c)] = int(rr.min())
            far[int(c)] = int(rr.max())
        a, b2 = cols.size // 4, cols.size - cols.size // 4
        if b2 <= a:
            a, b2 = 0, cols.size
        mid_cols = [int(c) for c in cols[a:b2]]

        def isolated(gap, mid_cols=mid_cols, near=near, far=far, i=i):
            for c in mid_cols:
                for a0, b0, skirt_end in (
                    (max(0, near[c] - gap), max(0, near[c] - 3), True),
                    (min(labels.shape[0], far[c] + 1 + 3), min(labels.shape[0], far[c] + 1 + gap), False),
                ):
                    if b0 <= a0:
                        continue
                    col_n = ndsm[a0:b0, c]
                    col_l = labels[a0:b0, c]
                    if np.any((col_n > 3) & np.isfinite(col_n) & (col_l != i)):
                        return False
            return True

        s_own = s_grid[sl][own]
        s_own = s_own[np.isfinite(s_own)]
        crow = int(np.clip(np.mean([near[c] + far[c] for c in mid_cols]) * 2, 0, H - 1))
        # mean of near and far is the 1 m centre; *4 would be MAG, but theta_row is MAG.
        # near/far are 1 m indices. Centre 1 m row * 4 is the MAG row.
        crow = int(np.clip(int(np.mean([near[c] + far[c] for c in mid_cols]) / 2) * 4, 0, H - 1))
        buildings.append({
            "h": h,
            "D": float(np.median([far[c] - near[c] for c in mid_cols])),
            "s": float(np.median(s_own)) if s_own.size else float("nan"),
            "theta": float(theta_row[crow]),
            "row": crow,
            "col": int(np.mean(mid_cols) * 4),
            "third": "near" if crow < H // 3 else ("middle" if crow < 2 * H // 3 else "far"),
            "iso60": isolated(60),
            "iso30": isolated(30),
            "near": {c: near[c] for c in mid_cols},
            "h_bag": meta[i - 1]["h_bag"],
        })
    log(f"  h>=3: {len(buildings)}  iso60 {sum(b['iso60'] for b in buildings)}  "
        f"iso30 {sum(b['iso30'] for b in buildings)}")

    bins = [(3, 6), (6, 10), (10, 15), (15, 25), (25, 40), (40, 1e9)]

    def bname(lo, hi):
        return f"{lo:g}-{hi:g}" if hi < 1e8 else f">{lo:g}"

    offsets_m = np.arange(-160.0, 160.0 + GSD * 0.5, GSD)
    n_off = offsets_m.size
    hb_lo, hb_step, hb_n = -30.0, 0.1, 600
    keys = [(t, lo, hi) for t in ("near", "middle", "far") for lo, hi in bins]
    hists = {k: np.zeros((n_off, hb_n), np.int32) for k in keys}
    counts = {k: 0 for k in keys}
    used = {k: [] for k in keys}
    relaxed_flag = {k: False for k in keys}

    def add(b, key):
        used[key].append(b)
        hist = hists[key]
        for c1, near1 in b["near"].items():
            near_px = near1 * 4
            rr = near_px + np.round(offsets_m / GSD).astype(np.int32)
            for dc in range(4):
                col = c1 * 4 + dc
                if col < 0 or col >= W:
                    continue
                good = (rr >= 0) & (rr < H)
                vals = np.full(n_off, np.nan, np.float32)
                vals[good] = hh[rr[good], col] - land_med
                wat = np.zeros(n_off, bool)
                wat[good] = water[rr[good], col]
                ok = np.isfinite(vals) & ~wat
                bi = np.floor((vals[ok] - hb_lo) / hb_step).astype(np.int32)
                sel = (bi >= 0) & (bi < hb_n)
                idx = np.flatnonzero(ok)[sel]
                if idx.size:
                    hist[idx, bi[sel]] += 1
                counts[key] += 1

    for third in ("near", "middle", "far"):
        pool = [b for b in buildings if b["third"] == third]
        for lo, hi in bins:
            group = [b for b in pool if lo <= b["h"] < hi]
            strict = [b for b in group if b["iso60"]]
            use, relaxed = strict, False
            if len(strict) < 100:
                use = [b for b in group if b["iso30"]]
                relaxed = True
                log(f"  relax {third} {bname(lo, hi)}: {len(strict)} at 60 m -> {len(use)} at 30 m")
            key = (third, lo, hi)
            relaxed_flag[key] = relaxed
            for b in use:
                add(b, key)
    log("  profiles done")

    def profile_from(hist):
        prof = np.full(n_off, np.nan)
        total = hist.sum(axis=1)
        csum = np.cumsum(hist, axis=1)
        for i in range(n_off):
            if total[i] < 30:
                continue
            k = min(int(np.searchsorted(csum[i], (total[i] + 1) / 2.0)), hb_n - 1)
            prof[i] = hb_lo + (k + 0.5) * hb_step
        return prof

    def measure(prof):
        sel = (offsets_m >= -160) & (offsets_m <= -120) & np.isfinite(prof)
        if sel.sum() < 10:
            return {
                "background_db": None, "leading_m": None, "peak_m": None, "peak_db": None,
                "dark_start_m": None, "dark_end_m": None, "dark_min_db": None,
            }
        bg = float(np.mean(prof[sel]))
        above = np.isfinite(prof) & (prof >= bg + 1.0)
        lead = float(offsets_m[int(np.argmax(above))]) if above.any() else None
        peak_i = int(np.nanargmax(prof))
        below = np.where(np.isfinite(prof) & (prof <= bg - 1.0))[0]
        dark_start = dark_end = dark_min = None
        if below.size:
            breaks = np.where(np.diff(below) > 1)[0]
            starts = np.concatenate([[below[0]], below[breaks + 1]])
            ends = np.concatenate([below[breaks], [below[-1]]])
            best = None
            for a0, b0 in zip(starts, ends):
                score = (int(b0 - a0), -float(np.min(prof[a0:b0 + 1])))
                if best is None or score > best[0]:
                    best = (score, int(a0), int(b0))
            dark_start = float(offsets_m[best[1]])
            dark_end = float(offsets_m[best[2]])
            dark_min = float(np.min(prof[best[1]:best[2] + 1]))
        return {
            "background_db": round(bg, 3),
            "leading_m": None if lead is None else round(lead, 2),
            "peak_m": round(float(offsets_m[peak_i]), 2),
            "peak_db": round(float(prof[peak_i]), 3),
            "dark_start_m": None if dark_start is None else round(dark_start, 2),
            "dark_end_m": None if dark_end is None else round(dark_end, 2),
            "dark_min_db": None if dark_min is None else round(dark_min, 3),
        }

    table, profiles = [], {}
    for key in keys:
        attrs = used[key]
        if not attrs:
            continue
        third, lo, hi = key
        prof = profile_from(hists[key])
        profiles[f"{third}_{bname(lo, hi)}"] = prof
        h_m = float(np.median([b["h"] for b in attrs]))
        D_m = float(np.median([b["D"] for b in attrs]))
        th_m = float(np.median([b["theta"] for b in attrs]))
        s_m = float(np.nanmedian([b["s"] for b in attrs]))
        pred = (h_m - s_m) / np.tan(np.radians(th_m))
        row = {
            "third": third, "bin": bname(lo, hi),
            "n_buildings": len(attrs), "n_columns": int(counts[key]),
            "relaxed_30m": bool(relaxed_flag[key]),
            "h_m": round(h_m, 2), "D_m": round(D_m, 2),
            "theta_deg": round(th_m, 2), "s_m": round(s_m, 2),
            "pred_roof_near_m": round(-pred, 2),
            "pred_roof_far_m": round(D_m - pred, 2),
            "pred_shadow_end_m": round(D_m + h_m * np.tan(np.radians(th_m)), 2),
        }
        row.update(measure(prof))
        table.append(row)
        log(f"  {third:6} {row['bin']:6} n {row['n_buildings']:4d} cols {row['n_columns']:6d} "
            f"h {h_m:.1f} D {D_m:.1f} th {th_m:.1f} s {s_m:.1f} "
            f"lead {row.get('leading_m')} peak {row.get('peak_m')} "
            f"dark {row.get('dark_start_m')}..{row.get('dark_end_m')}")

    def fit(rows):
        out = {}
        for name, ykey, subtract_d in (
            ("dark_start_minus_D", "dark_start_m", True),
            ("leading_edge", "leading_m", False),
        ):
            xs, ys = [], []
            for r in rows:
                if r.get(ykey) is None:
                    continue
                y = r[ykey] - (r["D_m"] if subtract_d else 0.0)
                x = (r["h_m"] - r["s_m"]) / np.tan(np.radians(r["theta_deg"]))
                xs.append(x)
                ys.append(y)
            if len(xs) < 3 or any(r["n_buildings"] < 20 for r in rows if r.get(ykey) is not None):
                out[name] = {"n": len(xs), "ok": False, "reason": "fewer than 3 bins, or a bin has under 20 buildings"}
                continue
            from scipy.stats import linregress
            lr = linregress(np.asarray(xs, float), np.asarray(ys, float))
            out[name] = {
                "n": len(xs), "ok": True,
                "a": float(lr.intercept), "a_stderr": float(lr.intercept_stderr or 0),
                "b": float(lr.slope), "b_stderr": float(lr.stderr or 0),
                "r": float(lr.rvalue),
            }
        return out

    fits = {
        "middle": fit([r for r in table if r["third"] == "middle"]),
        "all_thirds": fit(table),
    }
    log(f"  fit middle {json.dumps(fits['middle'])}")
    log(f"  fit all {json.dumps(fits['all_thirds'])}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for third in ("near", "middle", "far"):
        rows = [r for r in table if r["third"] == third]
        if not rows:
            continue
        fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
        for ax, r in zip(axes.ravel(), rows):
            ax.plot(offsets_m, profiles[f"{third}_{r['bin']}"], color="black", lw=1)
            ax.axvline(0, color="0.35", lw=0.8)
            ax.axvline(r["D_m"], color="0.35", lw=0.8, ls="--")
            ax.axvline(r["pred_roof_near_m"], color="c", lw=1)
            ax.axvline(r["pred_roof_far_m"], color="darkorange", lw=1)
            ax.axvline(r["pred_shadow_end_m"], color="royalblue", lw=1)
            if r.get("background_db") is not None:
                ax.axhline(r["background_db"], color="0.6", lw=0.6)
            ax.set_title(f"{r['bin']} m   n={r['n_buildings']}", fontsize=9)
            ax.set_xlim(-160, 160)
            ax.grid(True, lw=0.3)
        for ax in axes.ravel()[len(rows):]:
            ax.axis("off")
        fig.suptitle(f"{third}: grey footprint 0 and D, cyan SRTM roof-near, orange roof-far, blue shadow end")
        fig.tight_layout()
        fig.savefig(OUT / f"check3_profiles_{third}.png", dpi=110)
        plt.close(fig)
    np.savez_compressed(OUT / "check3_profiles.npz", offsets_m=offsets_m, **profiles)
    (OUT / "check3.json").write_text(json.dumps({
        "land_median_hh_db": land_med, "table": table, "fits": fits,
        "n_buildings_h_ge_3": len(buildings),
        "n_iso60": int(sum(b["iso60"] for b in buildings)),
        "n_iso30": int(sum(b["iso30"] for b in buildings)),
    }, indent=2))
    if table:
        with (OUT / "check3_table.csv").open("w", newline="") as f:
            wri = csv.DictWriter(f, fieldnames=list(table[0].keys()))
            wri.writeheader()
            wri.writerows(table)
    return table


def check4(prep):
    log("CHECK 4  tall isolated structures")
    from scipy.ndimage import distance_transform_edt, find_objects, label
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ndsm = prep["ndsm"]
    tall = np.isfinite(ndsm) & (ndsm > 10)
    lab, n = label(np.isfinite(ndsm) & (ndsm > 3))
    log(f"  components {n}")
    cands = []
    for i, sl in enumerate(find_objects(lab), start=1):
        if sl is None:
            continue
        own = lab[sl] == i
        area = int(own.sum())
        if area < 50:
            continue
        vals = ndsm[sl][own]
        vals = vals[np.isfinite(vals)]
        if vals.size < 50:
            continue
        h = float(np.percentile(vals, 70))
        if h >= 30:
            cands.append((h, i, sl, area))
    cands.sort(reverse=True)
    log(f"  h>30 and area>=50: {len(cands)}")
    picked = []
    theta_row = prep["theta_row"]
    for h, i, sl, area in cands:
        other = tall & (lab != i)
        if other.any():
            dist = distance_transform_edt(~other)
            if float(dist[lab == i].min()) < 100:
                continue
            del dist
        ss = prep["s"][lab == i]
        ss = ss[np.isfinite(ss)]
        rows, cols = np.where(lab == i)
        crow = int(np.median(rows) * 4)
        ccol = int(np.median(cols) * 4)
        picked.append({
            "h": h, "s": float(np.median(ss)) if ss.size else float("nan"),
            "theta": float(theta_row[min(crow, theta_row.size - 1)]),
            "row": crow, "col": ccol, "area_m2": area, "id": i,
        })
        # keep the pixel list only until the crop is drawn
        picked[-1]["_rows"] = rows
        picked[-1]["_cols"] = cols
        if len(picked) == 10:
            break
    log(f"  isolated {len(picked)}")
    slc = np.load(OUT / "slc_on_mag.npy", mmap_mode="r")
    with rasterio.open(MAG) as src:
        Hh, Ww = src.height, src.width
        for k, p in enumerate(picked):
            half = 600
            r0 = int(np.clip(p["row"] - half, 0, max(0, Hh - 1200)))
            c0 = int(np.clip(p["col"] - half, 0, max(0, Ww - 1200)))
            hh = src.read(1, window=Window(c0, r0, 1200, 1200)).astype(np.float32)
            sl = np.array(slc[r0:r0 + 1200, c0:c0 + 1200])
            foot = np.zeros((1200, 1200), bool)
            for r1, c1 in zip(p["_rows"], p["_cols"]):
                y, x = int(r1) * 4 - r0, int(c1) * 4 - c0
                if -3 <= y < 1200 and -3 <= x < 1200:
                    foot[max(0, y):min(1200, y + 4), max(0, x):min(1200, x + 4)] = True
            shift_m = (p["h"] - p["s"]) / np.tan(np.radians(p["theta"]))
            shadow_m = p["h"] * np.tan(np.radians(p["theta"]))
            drow = int(np.round(-shift_m / GSD))
            cyan = np.zeros_like(foot)
            if drow >= 0:
                cyan[drow:] = foot[:1200 - drow]
            else:
                cyan[:drow] = foot[-drow:]
            shadow = np.zeros_like(foot)
            shadow_px = int(np.round(shadow_m / GSD))
            for c in range(1200):
                on = np.where(foot[:, c])[0]
                if on.size == 0:
                    continue
                a0 = int(on.max()) + 1
                shadow[a0:min(1200, a0 + shadow_px), c] = True
            finite = hh[np.isfinite(hh)]
            lo, hi = (np.percentile(finite, [2, 98]) if finite.size else (-20, 0))
            fig, ax = plt.subplots(1, 2, figsize=(10, 5.2))
            for a, img, title in ((ax[0], hh, "MAG-POL HH"), (ax[1], sl, "SLC |HH|^2 on MAG grid")):
                a.imshow(img, cmap="gray", vmin=lo, vmax=hi)
                a.contour(foot.astype(float), levels=[0.5], colors="white", linewidths=0.7)
                a.contour(cyan.astype(float), levels=[0.5], colors="cyan", linewidths=0.7)
                a.contour(shadow.astype(float), levels=[0.5], colors="deepskyblue", linewidths=0.6)
                a.set_title(title, fontsize=9)
                a.axis("off")
            fig.suptitle(
                f"#{k + 1}  h {p['h']:.1f} m   s {p['s']:+.1f} m   theta {p['theta']:.1f}   "
                f"row {p['row']}   col {p['col']}   cyan is {shift_m:.0f} m toward the sensor"
            )
            fig.tight_layout()
            fig.savefig(OUT / f"check4_{k + 1:02d}_r{p['row']}_c{p['col']}.png", dpi=110)
            plt.close(fig)
            log(f"  crop {k + 1} h {p['h']:.1f} s {p['s']:+.1f} th {p['theta']:.1f} "
                f"row {p['row']} col {p['col']}")
            p["roof_shift_m"] = round(float(shift_m), 2)
            p["shadow_m"] = round(float(shadow_m), 2)
            del p["_rows"], p["_cols"]
    (OUT / "check4.json").write_text(json.dumps(picked, indent=2))
    return picked


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    verify_xcorr_sign()
    flight, geo = flight_from_mag()
    prep = build_prep(flight, geo)
    which = sys.argv[1:] or ["1", "2", "3", "4"]
    if "1" in which:
        check1(prep, flight, geo)
    if "2" in which:
        check2(prep)
    if "3" in which:
        check3(prep)
    if "4" in which:
        check4(prep)
    log("STOP")


if __name__ == "__main__":
    main()
