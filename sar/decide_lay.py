#!/usr/bin/env python3
"""Part 1b (riverbank towers) and Part 2 (layover extent) for decide.py.

Imported by decide.py. Does not modify dataset v2 or image_space.py.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.windows import Window
from shapely.geometry import Point

from decide import (
    BAG,
    GSD,
    H_EDGE_MIN,
    MAGS,
    OUT,
    ROOT,
    _erode,
    _h90_some,
    _near_back,
    _open_masks,
    _px_bbox,
    _raster_poly,
    _sample_geoid,
    _sample_mask,
    annotate,
    dump,
    flight_of,
    index_walls,
    load_buildings,
    log,
    sensor_at_row0,
    stage_prep,
    stage_project,
    toward_sensor,
)
from image_space_run import load_hgt, sample_srtm  # noqa: E402
from near_field import enu_from_map, perpendicular_geometry  # noqa: E402

NB_E, NB_N = 602359.1, 5752163.5  # north bank opposite Wilhelminapier


def _prep_arrays(name: str):
    path = (ROOT / "output" / "rule_check" / "prep.npz") if name == "P" else (OUT / name / "prep.npz")
    prep = np.load(path)
    z = np.asarray(prep["z_nap"], np.float32)
    ground = np.asarray(prep["ground"], np.float32)
    fp = np.asarray(prep["fp"]).astype(bool)
    if "clutter_core" in prep.files:
        clutter = np.asarray(prep["clutter_core"]).astype(bool)
    else:
        ndsm = z - ground
        clutter = (ndsm > 3.0) & ~fp & np.isfinite(ndsm)
    return {
        "path": path,
        "z": z,
        "ground": ground,
        "fp": fp,
        "clutter": clutter,
        "north": float(prep["north"]),
        "west": float(prep["west"]),
        "water": np.asarray(prep["water"]).astype(bool),
    }


def _rc1(x, y, north, west, shape):
    r = np.rint((north - y) - 0.5).astype(np.int32)
    c = np.rint((x - west) - 0.5).astype(np.int32)
    ok = (r >= 0) & (r < shape[0]) & (c >= 0) & (c < shape[1])
    return r, c, ok


def _xy(grid, file_row, file_col):
    x = grid["west"] + (file_col + 0.5) * GSD
    y = grid["north"] - (file_row + 0.5) * GSD
    return x, y


def layover_length(g, H, h):
    if not (np.isfinite(g) and np.isfinite(H) and np.isfinite(h)):
        return np.nan
    if h <= 1.0 or H <= h + 1.0 or g <= 1.0:
        return np.nan
    inside = g * g - 2.0 * H * h + h * h
    if inside <= 1.0:
        return np.nan
    return float(g - np.sqrt(inside))


def h_edge_column(grid, prep, file_col, near_file, sens0):
    """90th percentile of nDSM within 3 m behind the near wall, on this column."""
    step_away = -toward_sensor(sens0)
    dists = np.array([0.5, 1.5, 2.5])
    rows = near_file + step_away * np.rint(dists / GSD).astype(int)
    cols = np.full(rows.shape, file_col, np.int32)
    if np.any(rows < 0) or np.any(rows >= grid["height"]):
        return np.nan
    xs, ys = _xy(grid, rows, cols)
    r, c, ok = _rc1(xs, ys, prep["north"], prep["west"], prep["z"].shape)
    if ok.sum() < 2:
        return np.nan
    nd = prep["z"][r[ok], c[ok]] - prep["ground"][r[ok], c[ok]]
    nd = nd[np.isfinite(nd)]
    if nd.size < 2:
        return np.nan
    return float(np.percentile(nd, 90))


def _own_1m(poly, prep):
    north, west = prep["north"], prep["west"]
    shape = prep["fp"].shape
    minx, miny, maxx, maxy = poly.bounds
    c0 = max(0, int(np.floor(minx - west)) - 2)
    c1 = min(shape[1], int(np.ceil(maxx - west)) + 3)
    r0 = max(0, int(np.floor(north - maxy)) - 2)
    r1 = min(shape[0], int(np.ceil(north - miny)) + 3)
    if c1 <= c0 or r1 <= r0:
        return None
    wt = from_origin(west + c0, north - r0, 1.0, 1.0)
    m = rasterize([(poly, 1)], out_shape=(r1 - r0, c1 - c0), transform=wt, fill=0,
                  all_touched=True, dtype=np.uint8)
    return m.astype(bool), r0, c0


def _blocked(grid, prep, own, file_col, near_file, sens0, dist_m):
    """True, reason if another footprint or undiluted clutter lies within dist_m toward the sensor.

    The first 1 m in front of the wall is ignored so the wall pixel itself does not self-block.
    """
    step = toward_sensor(sens0)
    i0 = int(round(1.0 / GSD))
    i1 = int(np.ceil(dist_m / GSD))
    if i1 <= i0:
        return False, ""
    rows = near_file + step * np.arange(i0, i1 + 1)
    if np.any(rows < 0) or np.any(rows >= grid["height"]):
        return True, "off_image"
    cols = np.full(rows.shape, int(file_col))
    xs, ys = _xy(grid, rows, cols)
    r, c, ok = _rc1(xs, ys, prep["north"], prep["west"], prep["fp"].shape)
    if not np.all(ok):
        return True, "off_image"
    if own is None:
        is_own = np.zeros(r.shape, bool)
    else:
        om, r0, c0 = own
        rr = r - r0
        cc = c - c0
        inside = (rr >= 0) & (rr < om.shape[0]) & (cc >= 0) & (cc < om.shape[1])
        is_own = np.zeros(r.shape, bool)
        if inside.any():
            is_own[inside] = om[rr[inside], cc[inside]]
    other = prep["fp"][r, c] & ~is_own
    clut = prep["clutter"][r, c] & ~is_own
    if other.any():
        return True, "other_footprint"
    if clut.any():
        return True, "clutter"
    return False, ""


def _smooth(col):
    k = np.ones(9, np.float64) / 9.0
    return np.convolve(np.asarray(col, np.float64), k, mode="same")


def farthest_with_gaps(above: np.ndarray, gap_px: int) -> int:
    best = -1
    gap = 0
    for i, a in enumerate(np.asarray(above, bool)):
        if a:
            best = i
            gap = 0
        else:
            gap += 1
            if gap > gap_px:
                break
    return best


def to_ratio(power, length_m):
    power = np.asarray(power, np.float64)
    off = np.arange(power.size) * GSD
    x = off / length_m
    bg_m = (x >= 1.6) & (x <= 2.0) & np.isfinite(power)
    if int(bg_m.sum()) < 5:
        return None, np.nan
    bg = float(np.median(power[bg_m]))
    if not np.isfinite(bg) or bg <= 0:
        return None, np.nan
    ratio = np.full(100, np.nan)
    bi = np.floor(np.clip(x, 0, 1.999) / 0.02).astype(int)
    finite = np.isfinite(power)
    for b in range(100):
        m = (bi == b) & finite
        if m.any():
            ratio[b] = np.mean(power[m]) / bg
    good = np.isfinite(ratio)
    if int(good.sum()) < 40:
        return None, bg
    idx = np.arange(100)
    ratio[~good] = np.interp(idx[~good], idx[good], ratio[good])
    return ratio, bg


def half_fall(ratio):
    """x where the front profile falls to half its x in [0, 0.3] linear level."""
    x = (np.arange(100) + 0.5) * 0.02
    near = ratio[x <= 0.3]
    level = float(np.nanmean(near))
    if not np.isfinite(level) or level <= 0:
        return np.nan, level
    half = 0.5 * level
    y = ratio
    for i in range(len(x) - 1):
        if x[i + 1] < 0.3:
            continue
        if np.isfinite(y[i]) and np.isfinite(y[i + 1]) and y[i] >= half > y[i + 1]:
            w = (y[i] - half) / (y[i] - y[i + 1])
            return float(x[i] + w * (x[i + 1] - x[i])), level
    return np.nan, level


def half_recover(ratio):
    """x where the back profile, normalised by its far-range level, crosses 0.5 upward."""
    x = (np.arange(100) + 0.5) * 0.02
    y = ratio
    if np.isfinite(y[0]) and y[0] >= 0.5:
        return 0.0
    for i in range(len(x) - 1):
        if np.isfinite(y[i]) and np.isfinite(y[i + 1]) and y[i] < 0.5 <= y[i + 1]:
            w = (0.5 - y[i]) / (y[i + 1] - y[i] + 1e-12)
            return float(x[i] + w * (x[i + 1] - x[i]))
    return np.nan


def _stack(cols, key):
    if not cols:
        return None
    arr = np.stack([c[key] for c in cols if c.get(key) is not None], axis=0)
    if arr.size == 0:
        return None
    return arr


def _bootstrap(arr, ids, rng, n=400):
    """arr (n_col, 100), ids aligned. Resample buildings. Returns dB mean, lo, hi."""
    mean_lin = np.nanmean(arr, axis=0)
    db = 10.0 * np.log10(np.maximum(mean_lin, 1e-6))
    uniq = np.unique(ids)
    # group row indices
    groups = {i: np.flatnonzero(ids == i) for i in uniq}
    boots = np.empty((n, arr.shape[1]), np.float64)
    for b in range(n):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        rows = np.concatenate([groups[i] for i in pick])
        boots[b] = np.nanmean(arr[rows], axis=0)
    boot_db = 10.0 * np.log10(np.maximum(boots, 1e-6))
    lo = np.percentile(boot_db, 5, axis=0)
    hi = np.percentile(boot_db, 95, axis=0)
    return db, lo, hi, mean_lin


def _summarize(cols, key, rng):
    usable = [c for c in cols if c.get(key) is not None]
    if len(usable) < 5:
        return {"n_columns": len(usable), "n_buildings": len({c["id"] for c in usable})}
    arr = np.stack([c[key] for c in usable])
    ids = np.array([c["id"] for c in usable])
    db, lo, hi, mean_lin = _bootstrap(arr, ids, rng)
    if key == "ratio_f":
        hx, level = half_fall(mean_lin)
        extra = {"half_x": hx, "near_level_lin": level}
    else:
        extra = {"recover_x": half_recover(mean_lin)}
    return {
        "n_columns": int(arr.shape[0]),
        "n_buildings": int(len(set(ids.tolist()))),
        "profile_db": db,
        "band_lo": lo,
        "band_hi": hi,
        **extra,
    }


def _plot_profiles(path, front, back, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["all", "h_15_30", "h_30_60", "h_gt_60"]
    labels = ["all kept", "h 15–30 m", "h 30–60 m", "h > 60 m"]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.2), sharex=True)
    x = (np.arange(100) + 0.5) * 0.02
    for row, (pack, ylab) in enumerate(((front, "front, / L_B"), (back, "back, / S"))):
        for col, (k, lab) in enumerate(zip(order, labels)):
            ax = axes[row, col]
            s = pack.get(k) or {}
            if "profile_db" not in s:
                ax.set_title(f"{lab}\nn=0", fontsize=8)
                ax.axvline(1.0, color="0.4", lw=0.8)
                continue
            y = np.asarray(s["profile_db"], float)
            lo = np.asarray(s["band_lo"], float)
            hi = np.asarray(s["band_hi"], float)
            ax.fill_between(x, lo, hi, color="C0", alpha=0.25, lw=0)
            ax.plot(x, y, color="C0", lw=1.2)
            ax.axvline(1.0, color="0.25", lw=0.8)
            ax.axhline(0.0, color="0.7", lw=0.5)
            n = s.get("n_columns", 0)
            mark = s.get("half_x", s.get("recover_x", None))
            ax.set_title(f"{lab}\n{n} cols  x½={mark if mark is None else round(mark, 2)}", fontsize=8)
            if row == 1:
                ax.set_xlabel("x")
            if col == 0:
                ax.set_ylabel(ylab)
            ax.set_xlim(0, 2)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_scatter(path, rows, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    if rows:
        x = np.array([r["L_B"] for r in rows], float)
        y = np.array([r["L_meas"] for r in rows], float)
        ax.scatter(x, y, s=12, c="0.2", alpha=0.75, lw=0)
        m = max(float(np.nanmax(x)), float(np.nanmax(y)), 1.0)
        ax.plot([0, m], [0, m], color="C3", lw=1.0, label="1:1")
        k = float(np.sum(x * y) / np.sum(x * x)) if np.sum(x * x) > 0 else np.nan
        ax.plot([0, m], [0, k * m], color="C0", lw=1.0, label=f"slope {k:.2f}")
        ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel("L_B (m)")
    ax.set_ylabel("L_meas (m)")
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _select_columns(name, mag):
    prep = _prep_arrays(name)
    flight, geo, grid = flight_of(mag)
    sens0 = sensor_at_row0(flight, geo, grid)
    buildings = [b for b in load_buildings(grid) if b["h_bag"] >= 15.0]
    index_walls(buildings, grid, sens0)
    annotate(buildings, prep["path"], grid, flight, geo, h90=False)
    counts = {
        "buildings_h_bag_ge_15": len(buildings),
        "columns_considered": 0,
        "dropped_low_h_edge": 0,
        "dropped_bad_geometry": 0,
        "dropped_short_L": 0,
        "dropped_other_footprint": 0,
        "dropped_clutter": 0,
        "dropped_off_image": 0,
        "kept_columns": 0,
    }
    step = toward_sensor(sens0)
    # candidates before the (expensive) blockage test
    raw = []
    for b in buildings:
        if b.get("near") is None or not np.isfinite(b.get("s", np.nan)):
            continue
        cols = np.flatnonzero(b["near"] >= 0)
        if cols.size == 0:
            continue
        if cols.size > 5:
            idx = np.unique(np.rint(np.linspace(0, cols.size - 1, 5)).astype(int))
            cols = cols[idx]
        for lc in cols:
            counts["columns_considered"] += 1
            file_col = int(b["col0"] + lc)
            near = int(b["near"][lc])
            back = int(b["back"][lc])
            h = h_edge_column(grid, prep, file_col, near, sens0)
            if not np.isfinite(h) or h < H_EDGE_MIN:
                counts["dropped_low_h_edge"] += 1
                continue
            raw.append({
                "id": b["id"], "h_bag": b["h_bag"], "s": b["s"], "poly": b["poly"],
                "file_col": file_col, "near": near, "back": back, "h_edge": h,
            })
    if not raw:
        return grid, sens0, prep, [], counts, flight, geo

    def _batch(rows, key_row):
        xs, ys = _xy(grid, np.array([r[key_row] for r in rows]), np.array([r["file_col"] for r in rows]))
        r1, c1, ok = _rc1(xs, ys, prep["north"], prep["west"], prep["ground"].shape)
        gnap = np.full(len(rows), np.nan)
        gnap[ok] = prep["ground"][r1[ok], c1[ok]]
        from pyproj import Transformer
        to_ll = Transformer.from_crs("EPSG:32631", "EPSG:4326", always_xy=True)
        lon, lat = to_ll.transform(xs, ys)
        lon = np.asarray(lon, np.float64)
        lat = np.asarray(lat, np.float64)
        geo_n = _sample_geoid(lon, lat)
        gell = gnap + geo_n
        good = np.isfinite(gell)
        Hgt = np.full(len(rows), np.nan)
        grd = np.full(len(rows), np.nan)
        th = np.full(len(rows), np.nan)
        if good.any():
            e, n, u = enu_from_map(xs[good], ys[good], gell[good], "EPSG:32631",
                                   geo["lat"], geo["lon"], geo["height"])
            slant, alpha = perpendicular_geometry(e, n, u, flight["origin_enu"], flight["direction_enu"])
            Hgt[good] = slant * np.cos(alpha)
            grd[good] = slant * np.sin(alpha)
            th[good] = np.degrees(alpha)
        return Hgt, grd, th

    log(f"  geometry on {len(raw)} columns")
    Hf, gf, thf = _batch(raw, "near")
    Hb, gb, _ = _batch(raw, "back")
    kept = []
    # own-masks cached per building
    own_cache = {}
    for i, rec in enumerate(raw):
        LB = layover_length(gf[i], Hf[i], rec["h_edge"])
        hp = rec["h_edge"] - rec["s"]
        LA = layover_length(gf[i], Hf[i], hp) if np.isfinite(hp) else np.nan
        S = np.nan
        if np.isfinite(gb[i]) and np.isfinite(Hb[i]) and Hb[i] > rec["h_edge"] + 1:
            S = float(gb[i] * rec["h_edge"] / (Hb[i] - rec["h_edge"]))
        if not np.isfinite(LB):
            counts["dropped_bad_geometry"] += 1
            continue
        if LB < 8.0:
            counts["dropped_short_L"] += 1
            continue
        # profile must fit, including the background window at 2 L_B
        last = rec["near"] + step * int(np.floor(2.0 * LB / GSD))
        if last < 0 or last >= grid["height"]:
            counts["dropped_off_image"] += 1
            continue
        bid = rec["id"]
        if bid not in own_cache:
            own_cache[bid] = _own_1m(rec["poly"], prep)
        blocked, why = _blocked(grid, prep, own_cache[bid], rec["file_col"], rec["near"], sens0, 1.3 * LB)
        if blocked:
            counts["dropped_" + why] = counts.get("dropped_" + why, 0) + 1
            if why == "other_footprint":
                counts["dropped_other_footprint"] += 1
            elif why == "clutter":
                counts["dropped_clutter"] += 1
            elif why == "off_image":
                counts["dropped_off_image"] += 1
            continue
        # _blocked already increments via the generic line AND the specific ones.
        # Undo the double count: the generic line used dropped_{why} and then we add again.
        # Fix: only use the specific counters. Remove the generic increment.
        kept.append({
            **{k: rec[k] for k in ("id", "h_bag", "s", "file_col", "near", "back", "h_edge")},
            "L_B": float(LB), "L_A": float(LA) if np.isfinite(LA) else None,
            "S": float(S) if np.isfinite(S) else None,
            "theta": float(thf[i]), "H": float(Hf[i]), "g": float(gf[i]),
        })
    # The loop above double-counts drops. Recompute from scratch cleanly.
    return grid, sens0, prep, kept, counts, flight, geo


def _select_columns_clean(name, mag):
    """Same as the selector, but drop counts are not double-incremented."""
    prep = _prep_arrays(name)
    flight, geo, grid = flight_of(mag)
    sens0 = sensor_at_row0(flight, geo, grid)
    buildings = [b for b in load_buildings(grid) if b["h_bag"] >= 15.0]
    index_walls(buildings, grid, sens0)
    annotate(buildings, prep["path"], grid, flight, geo, h90=False)
    counts = {
        "buildings_h_bag_ge_15": len(buildings),
        "columns_considered": 0,
        "dropped_low_h_edge": 0,
        "dropped_bad_geometry": 0,
        "dropped_short_L": 0,
        "dropped_other_footprint": 0,
        "dropped_clutter": 0,
        "dropped_off_image": 0,
        "kept_columns": 0,
        "h_edge_min_m": H_EDGE_MIN,
        "note": "A column is kept only if h_edge>=8 m, L_B>=8 m, the 2·L_B profile fits in the image, and no other footprint and no undiluted clutter (nDSM>3 m outside footprints) lies between 1 m and 1.3·L_B in front of the near wall.",
    }
    step = toward_sensor(sens0)
    raw = []
    for b in buildings:
        if b.get("near") is None or not np.isfinite(b.get("s", np.nan)):
            continue
        cols = np.flatnonzero(b["near"] >= 0)
        if cols.size == 0:
            continue
        if cols.size > 5:
            idx = np.unique(np.rint(np.linspace(0, cols.size - 1, 5)).astype(int))
            cols = cols[idx]
        for lc in cols:
            counts["columns_considered"] += 1
            file_col = int(b["col0"] + int(lc))
            near = int(b["near"][int(lc)])
            back = int(b["back"][int(lc)])
            h = h_edge_column(grid, prep, file_col, near, sens0)
            if not np.isfinite(h) or h < H_EDGE_MIN:
                counts["dropped_low_h_edge"] += 1
                continue
            raw.append((b, file_col, near, back, h))
    log(f"  {name}: {len(raw)} columns passed the height cut, of {counts['columns_considered']} considered")
    if not raw:
        counts["kept_columns"] = 0
        return grid, sens0, [], counts

    from pyproj import Transformer
    to_ll = Transformer.from_crs("EPSG:32631", "EPSG:4326", always_xy=True)

    def batch(pts, which):
        rows = np.array([p[2] if which == "near" else p[3] for p in pts], np.int32)
        cols = np.array([p[1] for p in pts], np.int32)
        xs, ys = _xy(grid, rows, cols)
        r1, c1, ok = _rc1(xs, ys, prep["north"], prep["west"], prep["ground"].shape)
        gnap = np.full(len(pts), np.nan)
        gnap[ok] = prep["ground"][r1[ok], c1[ok]]
        lon, lat = to_ll.transform(np.asarray(xs, float), np.asarray(ys, float))
        lon = np.asarray(lon, np.float64)
        lat = np.asarray(lat, np.float64)
        gell = gnap + _sample_geoid(lon, lat)
        Hgt = np.full(len(pts), np.nan)
        grd = np.full(len(pts), np.nan)
        th = np.full(len(pts), np.nan)
        good = np.isfinite(gell)
        if good.any():
            e, n, u = enu_from_map(np.asarray(xs)[good], np.asarray(ys)[good], gell[good],
                                   "EPSG:32631", geo["lat"], geo["lon"], geo["height"])
            slant, alpha = perpendicular_geometry(e, n, u, flight["origin_enu"], flight["direction_enu"])
            Hgt[good] = slant * np.cos(alpha)
            grd[good] = slant * np.sin(alpha)
            th[good] = np.degrees(alpha)
        return Hgt, grd, th

    Hf, gf, thf = batch(raw, "near")
    Hb, gb, _ = batch(raw, "back")
    own_cache = {}
    kept = []
    for i, (b, file_col, near, back, h) in enumerate(raw):
        LB = layover_length(gf[i], Hf[i], h)
        hp = h - b["s"]
        LA = layover_length(gf[i], Hf[i], hp) if np.isfinite(hp) else np.nan
        S = np.nan
        if np.isfinite(gb[i]) and np.isfinite(Hb[i]) and Hb[i] > h + 1.0:
            S = float(gb[i] * h / (Hb[i] - h))
        if not np.isfinite(LB):
            counts["dropped_bad_geometry"] += 1
            continue
        if LB < 8.0:
            counts["dropped_short_L"] += 1
            continue
        last = near + step * int(np.floor(2.0 * LB / GSD))
        if last < 0 or last >= grid["height"]:
            counts["dropped_off_image"] += 1
            continue
        if b["id"] not in own_cache:
            own_cache[b["id"]] = _own_1m(b["poly"], prep)
        blocked, why = _blocked(grid, prep, own_cache[b["id"]], file_col, near, sens0, 1.3 * LB)
        if blocked:
            if why == "other_footprint":
                counts["dropped_other_footprint"] += 1
            elif why == "clutter":
                counts["dropped_clutter"] += 1
            else:
                counts["dropped_off_image"] += 1
            continue
        kept.append({
            "id": b["id"], "h_bag": float(b["h_bag"]), "s": float(b["s"]),
            "file_col": int(file_col), "near": int(near), "back": int(back),
            "h_edge": float(h), "L_B": float(LB),
            "L_A": float(LA) if np.isfinite(LA) else None,
            "S": float(S) if np.isfinite(S) else None,
            "theta": float(thf[i]), "poly": b["poly"],
        })
    counts["kept_columns"] = len(kept)
    counts["kept_buildings"] = len({c["id"] for c in kept})
    log(f"  kept {counts['kept_columns']} columns on {counts['kept_buildings']} buildings")
    log("  drops " + " ".join(f"{k}={v}" for k, v in counts.items() if k.startswith("dropped")))
    return grid, sens0, kept, counts


def _measure(name, mag, grid, sens0, kept):
    """Fill ratio_f, ratio_b, L_meas from HH. One pass over column chunks."""
    if not kept:
        return
    by_chunk: dict[int, list] = {}
    for rec in kept:
        by_chunk.setdefault(rec["file_col"] // 1024, []).append(rec)
    step = toward_sensor(sens0)
    gap_px = int(round(8.0 / GSD))
    H = grid["height"]
    with rasterio.open(mag) as src:
        for key in sorted(by_chunk):
            recs = by_chunk[key]
            cols = [r["file_col"] for r in recs]
            c0 = max(0, min(cols) - 14)
            c1 = min(grid["width"], max(cols) + 15)
            hh = src.read(1, window=Window(c0, 0, c1 - c0, H)).astype(np.float32)
            for rec in recs:
                LB = rec["L_B"]
                n_front = int(np.floor(2.0 * LB / GSD)) + 1
                rows = rec["near"] + step * np.arange(n_front)
                loc = rec["file_col"] - c0
                slab_c0 = max(0, loc - 12)
                slab_c1 = min(hh.shape[1], loc + 13)
                if np.any(rows < 0) or np.any(rows >= H):
                    continue
                slab = hh[rows][:, slab_c0:slab_c1]
                center_i = loc - slab_c0
                if center_i < 0 or center_i >= slab.shape[1]:
                    continue
                if not np.isfinite(slab[:, center_i]).any():
                    continue
                lin = np.power(10.0, np.asarray(slab, np.float64) / 10.0)
                lin[~np.isfinite(slab)] = np.nan
                ratio, bg = to_ratio(lin[:, center_i], LB)
                rec["ratio_f"] = ratio
                # L_meas: 9 px mean, then max over 3 columns around the brightest within ±12 px
                sm = np.stack([_smooth(np.nan_to_num(lin[:, j], nan=np.nanmedian(lin[:, j]) if np.isfinite(lin[:, j]).any() else 0.0))
                               for j in range(lin.shape[1])], axis=1)
                n_lay = max(1, int(round(LB / GSD)))
                score = np.nanmax(sm[:min(n_lay, sm.shape[0])], axis=0)
                bi = int(np.nanargmax(score))
                lo, hi = max(0, bi - 1), min(sm.shape[1], bi + 2)
                profile = np.nanmax(sm[:, lo:hi], axis=1)
                # background from the chosen column, unsmoothed, x in [1.6, 2]
                off = np.arange(n_front) * GSD
                bg_m = (off >= 1.6 * LB) & (off <= 2.0 * LB)
                bg = float(np.nanmedian(lin[bg_m, center_i])) if bg_m.any() else np.nan
                if not np.isfinite(bg) or bg <= 0:
                    rec["L_meas"] = None
                else:
                    above = np.isfinite(profile) & (profile >= 2.0 * bg)
                    far = farthest_with_gaps(above, gap_px)
                    rec["L_meas"] = float(far * GSD) if far >= 0 else 0.0
                # back side
                S = rec.get("S")
                if S is not None and np.isfinite(S) and S >= 4.0:
                    n_back = int(np.floor(2.0 * S / GSD)) + 1
                    step_away = -step
                    brows = rec["back"] + step_away * np.arange(n_back)
                    if np.all((brows >= 0) & (brows < H)):
                        bpow = np.power(10.0, hh[brows, loc] / 10.0)
                        br, _ = to_ratio(bpow, S)
                        rec["ratio_b"] = br
                    else:
                        rec["ratio_b"] = None
                else:
                    rec["ratio_b"] = None
            log(f"  HH chunk {c0}:{c1}  {len(recs)} columns")


def _building_rows(kept):
    from collections import defaultdict
    g = defaultdict(list)
    for c in kept:
        if c.get("L_meas") is None or c.get("ratio_f") is None:
            continue
        g[c["id"]].append(c)
    rows = []
    for bid, cols in g.items():
        LB = np.median([c["L_B"] for c in cols])
        Lm = np.median([c["L_meas"] for c in cols])
        LAs = [c["L_A"] for c in cols if c.get("L_A")]
        rows.append({
            "id": bid,
            "n_columns": len(cols),
            "h_edge": float(np.median([c["h_edge"] for c in cols])),
            "h_max": float(np.max([c["h_edge"] for c in cols])),
            "h_bag": cols[0]["h_bag"],
            "theta": float(np.median([c["theta"] for c in cols])),
            "L_B": float(LB),
            "L_meas": float(Lm),
            "L_A": float(np.median(LAs)) if LAs else None,
            "ratio_LB": float(Lm / LB) if LB > 0 else None,
            "ratio_LA": float(Lm / np.median(LAs)) if LAs and np.median(LAs) > 0 else None,
        })
    return rows


def _scatter_stats(rows):
    if not rows:
        return {"n": 0}
    x = np.array([r["L_B"] for r in rows], float)
    y = np.array([r["L_meas"] for r in rows], float)
    slope = float(np.sum(x * y) / np.sum(x * x))
    rms = float(np.sqrt(np.mean((y - x) ** 2)))
    ratio = y / x
    la = np.array([r["ratio_LA"] if r.get("ratio_LA") is not None else np.nan for r in rows], float)
    return {
        "n": int(len(rows)),
        "median_ratio_LB": float(np.median(ratio)),
        "slope_through_origin": slope,
        "rms_m_vs_1to1": rms,
        "median_L_B": float(np.median(x)),
        "median_L_meas": float(np.median(y)),
        "fraction_L_meas_zero": float(np.mean(y <= 0)),
        "n_with_LA": int(np.isfinite(la).sum()),
        "median_ratio_LA": float(np.nanmedian(la)) if np.isfinite(la).any() else None,
    }


def _crops_part2(name, mag, grid, sens0, kept, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        return []
    by = {}
    for c in kept:
        by.setdefault(c["id"], []).append(c)
    # polys from kept
    polys = {}
    for c in kept:
        polys[c["id"]] = c.get("poly")
    tall = sorted(rows, key=lambda r: -r["h_max"])[:4]
    rest = [r for r in rows if r["id"] not in {t["id"] for t in tall}]
    rest = sorted(rest, key=lambda r: r["L_B"])
    extra = []
    if rest:
        idx = np.unique(np.rint(np.linspace(0, len(rest) - 1, min(8, len(rest)))).astype(int))
        extra = [rest[i] for i in idx]
    chosen = tall + extra
    step = toward_sensor(sens0)
    out = OUT / name
    crops = []
    with rasterio.open(mag) as src:
        for r in chosen:
            cols = by.get(r["id"], [])
            if not cols or polys.get(r["id"]) is None:
                continue
            poly = polys[r["id"]]
            reach = max(r["L_B"], r["L_meas"] or 0) + 30
            # range pad toward the sensor and a little the other way
            nears = [c["near"] for c in cols]
            backs = [c["back"] for c in cols]
            if sens0:
                r_lo = max(0, min(nears) - int(reach / GSD) - 20)
                r_hi = min(grid["height"], max(backs) + 40)
            else:
                r_lo = max(0, min(backs) - 40)
                r_hi = min(grid["height"], max(nears) + int(reach / GSD) + 20)
            cs = [c["file_col"] for c in cols]
            c0 = max(0, min(cs) - int(40 / GSD))
            c1 = min(grid["width"], max(cs) + int(40 / GSD))
            if r_hi - r_lo < 16 or c1 - c0 < 16:
                continue
            hh = src.read(1, window=Window(c0, r_lo, c1 - c0, r_hi - r_lo)).astype(np.float32)
            show = hh if sens0 else hh[::-1]

            def disp_row(file_row):
                local = file_row - r_lo
                if not sens0:
                    local = (r_hi - r_lo - 1) - local
                return local

            fig, ax = plt.subplots(figsize=(5.4, 5.6))
            v = show[np.isfinite(show)]
            lo, hi = (np.percentile(v, [2, 98]) if v.size else (0, 1))
            ax.imshow(show, cmap="gray", vmin=float(lo), vmax=float(hi), origin="upper")
            wt_rows = []
            # footprint
            wt = from_origin(grid["west"] + c0 * GSD, grid["north"] - r_lo * GSD, GSD, GSD)
            fp = rasterize([(poly, 1)], out_shape=hh.shape, transform=wt, fill=0, dtype=np.uint8)
            fp_s = fp if sens0 else fp[::-1]
            if fp_s.any():
                ax.contour(fp_s.astype(float), levels=[0.5], colors="white", linewidths=0.7)
            for c in cols:
                dc = c["file_col"] - c0
                y0 = disp_row(c["near"])
                yb = disp_row(c["near"] + step * int(round(c["L_B"] / GSD)))
                ax.plot([dc], [y0], marker="o", color="white", ms=2.5, lw=0)
                ax.plot([dc, dc], [y0, yb], color="cyan", lw=0.8)
                if c.get("L_meas") is not None:
                    ym = disp_row(c["near"] + step * int(round(c["L_meas"] / GSD)))
                    ax.plot([dc], [ym], marker="o", color="orange", ms=3.0, lw=0)
            ax.text(0.02, 0.98, "sensor", transform=ax.transAxes, color="white", fontsize=7, va="top")
            ax.set_title(
                f"{r['id']}\nh {r['h_edge']:.0f} m  θ {r['theta']:.0f}°  "
                f"L_B {r['L_B']:.0f}  L_meas {r['L_meas']:.0f}",
                fontsize=8,
            )
            ax.axis("off")
            fig.tight_layout()
            path = out / f"lay_{r['id']}.png"
            fig.savefig(path, dpi=110)
            plt.close(fig)
            crops.append({"id": r["id"], "file": path.name, "h_edge": r["h_edge"],
                          "L_B": r["L_B"], "L_meas": r["L_meas"]})
            log(f"    lay crop {path.name}")
    return crops


def part2(name: str):
    t0 = time.time()
    mag = MAGS[name]
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    grid, sens0, kept, counts = _select_columns_clean(name, mag)
    _measure(name, mag, grid, sens0, kept)
    # height bins on the column h_edge
    groups = {
        "all": [c for c in kept if c.get("ratio_f") is not None],
        "h_15_30": [c for c in kept if c.get("ratio_f") is not None and 15 <= c["h_edge"] < 30],
        "h_30_60": [c for c in kept if c.get("ratio_f") is not None and 30 <= c["h_edge"] < 60],
        "h_gt_60": [c for c in kept if c.get("ratio_f") is not None and c["h_edge"] >= 60],
    }
    rng = np.random.default_rng(0)
    front = {k: _summarize(v, "ratio_f", rng) for k, v in groups.items()}
    back_groups = {
        "all": [c for c in groups["all"] if c.get("ratio_b") is not None],
        "h_15_30": [c for c in groups["h_15_30"] if c.get("ratio_b") is not None],
        "h_30_60": [c for c in groups["h_30_60"] if c.get("ratio_b") is not None],
        "h_gt_60": [c for c in groups["h_gt_60"] if c.get("ratio_b") is not None],
    }
    back = {k: _summarize(v, "ratio_b", rng) for k, v in back_groups.items()}
    rows = _building_rows(kept)
    stats = _scatter_stats(rows)
    _plot_profiles(out / "part2_profiles.png", front, back, f"{name}  stacked layover / shadow, sensor toward x=0")
    _plot_scatter(out / "part2_scatter.png", rows, f"{name}  L_meas vs L_B   n={stats.get('n')}  "
                                                    f"ratio {stats.get('median_ratio_LB')}")
    crops = _crops_part2(name, mag, grid, sens0, kept, rows)
    # drop bulky arrays and polygons before json
    for pack in (front, back):
        for s in pack.values():
            for k in ("profile_db", "band_lo", "band_hi"):
                if k in s:
                    s[k] = [round(float(v), 3) for v in np.asarray(s[k], float)]
    result = {
        "strip": name,
        "sensor_at_row0": sens0,
        "counts": counts,
        "front": front,
        "back": back,
        "scatter": stats,
        "buildings": sorted(rows, key=lambda r: -r["h_max"]),
        "crops": crops,
        "definitions": {
            "L_B": "g - sqrt(g^2 - 2 H h + h^2), h = h_edge at the near wall",
            "L_A": "same with h' = h_edge - s",
            "S": "g_back * h / (H_back - h)",
            "L_meas": "farthest offset of the 9 px mean of linear power, max of 3 columns around the brightest within 3 m, >= 2x background on [1.6, 2] L_B, gaps <= 8 m. 0 if the threshold is never met.",
            "half_front": "x where the mean linear ratio falls to half its mean over x in [0, 0.3]",
            "half_back": "x where the mean linear ratio, divided by its [1.6, 2] level, crosses 0.5 upward",
            "h_edge_min_m": H_EDGE_MIN,
        },
    }
    dump(result, out / "part2.json")
    log(f"  part2 {name} done {time.time()-t0:.0f}s  scatter {stats}")
    return result


# ---------------------------------------------------------------------------
# Part 1b
# ---------------------------------------------------------------------------

def _v2_containment():
    cat = json.loads((ROOT / "output" / "strip_catalog.json").read_text())
    done = (ROOT / "output" / "dataset_v2" / "done_strips.txt").read_text().split()
    hits = []
    for s in cat:
        if s["strip_id"] not in done:
            continue
        west, south, east, north = s["bounds"]
        inside = west <= NB_E <= east and south <= NB_N <= north
        if inside:
            hits.append(s["strip_id"])
    return {"north_bank_utm": [NB_E, NB_N], "v2_strips_containing_it": hits, "n_v2": len(done)}


def _water_hit(b, sens0, grid, open_1m, n1, w1):
    """Minimum metres from the near wall toward the sensor to open water, or None."""
    if b.get("near") is None:
        return None
    step = toward_sensor(sens0)
    cols = np.flatnonzero(b["near"] >= 0)
    if cols.size == 0:
        return None
    cols = cols[:: max(1, int(round(2.0 / GSD)))]  # ~2 m along track
    npx = int(round(150.0 / GSD))
    best = None
    best_col = None
    near = b["near"]
    for lc in cols:
        file_col = int(b["col0"] + int(lc))
        nr = int(near[int(lc)])
        dist_px = np.arange(1, npx + 1)
        rows = nr + step * dist_px
        good = (rows >= 0) & (rows < grid["height"])
        if not good.any():
            continue
        rows = rows[good]
        dist_px = dist_px[good]
        xs, ys = _xy(grid, rows, np.full(rows.shape, file_col))
        r, c, ok = _rc1(xs, ys, n1, w1, open_1m.shape)
        if not ok.any():
            continue
        hit = np.zeros(rows.shape, bool)
        hit[ok] = open_1m[r[ok], c[ok]]
        if hit.any():
            d = float(dist_px[np.argmax(hit)] * GSD)
            if best is None or d < best:
                best = d
                best_col = file_col
                b["_hit_near"] = nr
                b["_hit_col"] = file_col
    return best


def _fade_tip(hh_col, water_lin, gap_px):
    """Farthest offset where the 9 px mean of linear power stays >= water_lin * 10**(1.5/10)."""
    lin = np.power(10.0, np.asarray(hh_col, np.float64) / 10.0)
    lin[~np.isfinite(hh_col)] = np.nan
    sm = _smooth(np.nan_to_num(lin, nan=0.0))
    thr = water_lin * (10 ** (1.5 / 10.0))
    above = np.isfinite(lin) & (sm >= thr)
    far = farthest_with_gaps(above, gap_px)
    return float(far * GSD) if far >= 0 else 0.0


def part1b():
    """Towers over 40 m with open water between the sensor-facing wall and the sensor."""
    t0 = time.time()
    out = OUT / "part1b"
    out.mkdir(parents=True, exist_ok=True)
    cover = _v2_containment()
    log(f"  v2 strips containing the north-bank point: {cover['v2_strips_containing_it'] or 'none'}")
    # preps
    for name in ("K", "N"):
        if not (OUT / name / "prep.npz").exists():
            stage_prep(name, MAGS[name])
    strips = []
    for name, prep_path in (
        ("P", ROOT / "output" / "rule_check" / "prep.npz"),
        ("K", OUT / "K" / "prep.npz"),
        ("N", OUT / "N" / "prep.npz"),
    ):
        if not prep_path.exists():
            log(f"  skip {name}, no prep")
            continue
        strips.append((name, prep_path))

    found = []
    contexts = {}
    for name, prep_path in strips:
        mag = MAGS[name]
        flight, geo, grid = flight_of(mag)
        sens0 = sensor_at_row0(flight, geo, grid)
        shore, open_w, n1, w1 = _open_masks(prep_path)
        buildings = [b for b in load_buildings(grid) if b["h_bag"] > 40.0]
        index_walls(buildings, grid, sens0)
        annotate(buildings, prep_path, grid, flight, geo, h90=False)
        prep = _prep_arrays(name)
        n_hit = 0
        for b in buildings:
            dist = _water_hit(b, sens0, grid, open_w, n1, w1)
            if dist is None or dist > 150:
                continue
            n_hit += 1
            h = h_edge_column(grid, prep, b["_hit_col"], b["_hit_near"], sens0)
            # geometry at that wall
            x, y = _xy(grid, b["_hit_near"], b["_hit_col"])
            r, c, ok = _rc1(np.array([x]), np.array([y]), prep["north"], prep["west"], prep["ground"].shape)
            gnap = float(prep["ground"][r[0], c[0]]) if ok[0] else np.nan
            from pyproj import Transformer
            to_ll = Transformer.from_crs("EPSG:32631", "EPSG:4326", always_xy=True)
            lon, lat = to_ll.transform(x, y)
            gell = gnap + float(_sample_geoid([lon], [lat])[0]) if np.isfinite(gnap) else np.nan
            H = g = th = np.nan
            if np.isfinite(gell):
                e, n, u = enu_from_map(np.array([x]), np.array([y]), np.array([gell]),
                                       "EPSG:32631", geo["lat"], geo["lon"], geo["height"])
                slant, alpha = perpendicular_geometry(e, n, u, flight["origin_enu"], flight["direction_enu"])
                H = float(slant[0] * np.cos(alpha[0]))
                g = float(slant[0] * np.sin(alpha[0]))
                th = float(np.degrees(alpha[0]))
            LB = layover_length(g, H, h) if np.isfinite(h) else np.nan
            rec = {
                "strip": name, "id": b["id"], "h_bag": b["h_bag"], "h_edge": h,
                "theta": th, "L_B": LB, "water_m": dist,
                "file_col": int(b["_hit_col"]), "near": int(b["_hit_near"]),
                "s": b["s"], "col0": b["col0"], "col1": b["col1"],
            }
            # keep geometry objects for the crop, not for json
            found.append(rec)
            contexts[(name, b["id"])] = (b, grid, sens0, flight, geo)
        log(f"  {name}: {len(buildings)} buildings >40 m, {n_hit} with open water within 150 m toward the sensor")
        del shore, open_w

    # T-L from part 1 where it exists; project N only around its candidates
    for name in ("P", "K"):
        idx_path = OUT / name / "t_index.json"
        L = None
        p1 = OUT / name / "part1.json"
        if p1.exists():
            L = json.loads(p1.read_text())["table"]["L"]["all"]["median_db"]
        index = {}
        if idx_path.exists():
            index = {r["id"]: r for r in json.loads(idx_path.read_text())["buildings"]}
        for rec in found:
            if rec["strip"] != name:
                continue
            hit = index.get(rec["id"])
            rec["T_minus_L"] = hit.get("T_minus_L") if hit else None
            rec["T_n"] = hit.get("n") if hit else 0
            rec["L_source"] = "part1 strip L" if L else None

    n_recs = [r for r in found if r["strip"] == "N"]
    if n_recs:
        spans = []
        for r in n_recs:
            spans.append((max(0, r["col0"] - int(80 / GSD)), r["col1"] + int(80 / GSD)))
        spans.sort()
        merged = [list(spans[0])]
        for a, b in spans[1:]:
            if a <= merged[-1][1] + 40:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        log(f"  projecting N columns {merged}")
        stage_project("N", MAGS["N"], OUT / "N" / "prep.npz",
                      products=("A_full", "B_full"), col_ranges=merged)
        _t_on_N(n_recs, contexts)

    # crops and measured tips need HH. Do them for every found building.
    _finish_tips_and_crops(found, contexts)
    # drop non-jsonable
    dump({"cover": cover, "n": len(found), "buildings": found}, out / "part1b.json")
    log(f"  part1b {len(found)} buildings  {time.time()-t0:.0f}s")
    return found


def _t_on_N(recs, contexts):
    """T and a local L on the columns that were actually projected."""
    sens0 = contexts[(recs[0]["strip"], recs[0]["id"])][2]
    grid = contexts[(recs[0]["strip"], recs[0]["id"])][1]
    prep = _prep_arrays("N")
    shore, open_w, n1, w1 = _open_masks(OUT / "N" / "prep.npz")
    H = grid["height"]
    # local land: both nominal, outside 1 m fp, not shore, on projected columns
    # and T for each building
    a_path = OUT / "N" / "A_full_cov.tif"
    b_path = OUT / "N" / "B_full_cov.tif"
    al_path = OUT / "N" / "A_full_lab.tif"
    bl_path = OUT / "N" / "B_full_lab.tif"
    cols = []
    for r in recs:
        cols.extend(range(max(0, r["col0"] - 20), min(grid["width"], r["col1"] + 20)))
    cols = sorted(set(cols))
    if not cols:
        return
    c0, c1 = cols[0], cols[-1] + 1
    c1 = min(c1, grid["width"])
    with rasterio.open(MAGS["N"]) as mag, rasterio.open(a_path) as ac, rasterio.open(b_path) as bc, \
            rasterio.open(al_path) as al, rasterio.open(bl_path) as bl:
        hh = mag.read(1, window=Window(c0, 0, c1 - c0, H)).astype(np.float32)
        A = ac.read(1, window=Window(c0, 0, c1 - c0, H))
        B = bc.read(1, window=Window(c0, 0, c1 - c0, H))
        Al = al.read(1, window=Window(c0, 0, c1 - c0, H))
        Bl = bl.read(1, window=Window(c0, 0, c1 - c0, H))
    # The read can be narrower than requested if it hits the image edge.
    c1 = c0 + hh.shape[1]
    rows = np.arange(H)
    cidx = np.arange(c0, c1)
    shore_i = _sample_mask(shore, rows, cidx, n1, w1, grid["north"], grid["west"])
    yy = grid["north"] - (rows + 0.5) * GSD
    xx = grid["west"] + (cidx + 0.5) * GSD
    r1 = np.clip(np.rint((prep["north"] - yy) - 0.5).astype(np.int32), 0, prep["fp"].shape[0] - 1)
    c1m = np.clip(np.rint((xx - prep["west"]) - 0.5).astype(np.int32), 0, prep["fp"].shape[1] - 1)
    fp = prep["fp"][r1][:, c1m]
    known = (Al != 255) & (Bl != 255) & np.isfinite(A) & np.isfinite(B) & np.isfinite(hh)
    land = (A >= 0.5) & (A < 1.5) & (B >= 0.5) & (B < 1.5) & known & ~fp & ~shore_i
    if land.sum() >= 30:
        L = float(np.median(hh[land]))
    else:
        L = float("nan")
    log(f"  N local L {L:.2f} dB from {int(land.sum())} px" if np.isfinite(L) else "  N local L nan")
    for rec in recs:
        b, grid, sens0, *_ = contexts[(rec["strip"], rec["id"])]
        if b.get("ero") is None or b.get("near") is None:
            rec["T_minus_L"] = None
            rec["T_n"] = 0
            rec["L_source"] = "local nominal land on projected columns"
            continue
        # T mask inside the window
        lc0 = max(c0, b["col0"])
        lc1 = min(c1, b["col1"])
        t_vals = []
        if lc1 > lc0 and b.get("ero") is not None:
            got = _raster_poly(b["ero"], *_px_bbox(b["ero"], grid)[:2], lc0, lc1, grid)
            if got is not None:
                emo, er0, ec0 = got
                if ec0 == lc0:
                    near_abs = b["near"][lc0 - b["col0"]: lc1 - b["col0"]]
                    file_rows = np.arange(er0, er0 + emo.shape[0])
                    row_c = file_rows if sens0 else (H - 1) - file_rows
                    near_c = np.where(near_abs >= 0, near_abs if sens0 else (H - 1) - near_abs, -10_000)
                    clear = (6.0 + max(0.0, b["s"]) / np.tan(np.radians(b["theta"]))) / GSD if b["theta"] > 3 else 6.0 / GSD
                    row_ok = row_c[:, None] >= (near_c[None, :] + clear)
                    sel = emo.astype(bool) & row_ok & (near_abs[None, :] >= 0)
                    subA = A[er0:er0 + emo.shape[0], lc0 - c0:lc0 - c0 + emo.shape[1]]
                    subB = B[er0:er0 + emo.shape[0], lc0 - c0:lc0 - c0 + emo.shape[1]]
                    subK = known[er0:er0 + emo.shape[0], lc0 - c0:lc0 - c0 + emo.shape[1]]
                    subS = shore_i[er0:er0 + emo.shape[0], lc0 - c0:lc0 - c0 + emo.shape[1]]
                    subH = hh[er0:er0 + emo.shape[0], lc0 - c0:lc0 - c0 + emo.shape[1]]
                    if subA.shape == sel.shape:
                        take = sel & (subA < 0.5) & (subB >= 0.5) & subK & ~subS
                        vals = subH[take]
                        vals = vals[np.isfinite(vals)]
                        t_vals = vals
        n = int(np.size(t_vals))
        rec["T_n"] = n
        rec["L_source"] = "local nominal land on projected N columns"
        if n >= 30 and np.isfinite(L):
            rec["T_minus_L"] = float(np.median(t_vals) - L)
        else:
            rec["T_minus_L"] = None


def _finish_tips_and_crops(found, contexts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = OUT / "part1b"
    # water linear level per strip, from open water in a thin sample if needed
    water_db = {}
    for name in {r["strip"] for r in found}:
        prep_path = (ROOT / "output" / "rule_check" / "prep.npz") if name == "P" else (OUT / name / "prep.npz")
        # use part1 W if we have it
        p1 = OUT / name / "part1.json"
        if p1.exists():
            water_db[name] = json.loads(p1.read_text())["table"]["W"]["all"]["median_db"]
        else:
            water_db[name] = None
    gap_px = int(round(8.0 / GSD))
    for rec in found:
        b, grid, sens0, flight, geo = contexts[(rec["strip"], rec["id"])]
        mag = MAGS[rec["strip"]]
        step = toward_sensor(sens0)
        # search window for the tip
        reach = 400.0
        if rec.get("L_B") and np.isfinite(rec["L_B"]):
            reach = max(reach, 2.5 * rec["L_B"])
        n = int(reach / GSD)
        rows = rec["near"] + step * np.arange(0, n)
        rows = rows[(rows >= 0) & (rows < grid["height"])]
        with rasterio.open(mag) as src:
            col = rec["file_col"]
            hh = src.read(1, window=Window(col, int(rows.min()) if False else 0, 1, grid["height"])).ravel()
        prof = hh[rows] if step == 1 or True else hh[rows]
        # rows may not be monotonic increasing; fancy index
        prof = hh[rows]
        # water level: if we don't have strip W, median of the far half of this profile
        wdb = water_db.get(rec["strip"])
        if wdb is None or not np.isfinite(wdb):
            far = prof[int(prof.size * 0.6):]
            far = far[np.isfinite(far)]
            wdb = float(np.median(far)) if far.size else -15.0
            water_db[rec["strip"]] = wdb
        wlin = 10 ** (wdb / 10.0)
        rec["L_meas"] = _fade_tip(prof, wlin, gap_px)
        rec["water_db"] = wdb
        # footprint brightness: eroded, >6 m behind the near wall
        _foot_bright(rec, b, grid, sens0, hh)
        _crop_tower(out, rec, b, grid, sens0, mag)
    # classify
    for rec in found:
        h = rec.get("h_edge") or 0
        LB = rec.get("L_B") or np.nan
        Lm = rec.get("L_meas") or 0
        ratio = (Lm / LB) if LB and np.isfinite(LB) and LB > 0 else np.nan
        rec["ratio_L"] = ratio
        if np.isfinite(h) and h >= 25 and np.isfinite(ratio) and ratio >= 0.35:
            rec["streak"] = "real_layover_of_a_standing_building"
        elif (not np.isfinite(h) or h < 12) and (not np.isfinite(ratio) or ratio < 0.25):
            rec["streak"] = "no_2019_mass_and_no_streak"
        else:
            rec["streak"] = "inspect"


def _foot_bright(rec, b, grid, sens0, hh_fullcol_unused):
    """Median HH on the eroded footprint, more than 6 m behind the near wall.

    Reads HH itself so the caller can pass a dummy.
    """
    if b.get("ero") is None or b.get("near") is None:
        rec["footprint"] = "unknown"
        return
    H = grid["height"]
    r0, r1, c0, c1 = _px_bbox(b["ero"], grid)
    got = _raster_poly(b["ero"], r0, r1, c0, c1, grid)
    if got is None:
        rec["footprint"] = "unknown"
        return
    emo, er0, ec0 = got
    with rasterio.open(MAGS[rec["strip"]]) as src:
        hh = src.read(1, window=Window(ec0, er0, emo.shape[1], emo.shape[0])).astype(np.float32)
    if hh.shape != emo.shape:
        rec["footprint"] = "unknown"
        return
    near = b["near"]
    # columns of emo
    file_cols = np.arange(ec0, ec0 + emo.shape[1])
    # near array indexed from b["col0"]
    sel_cols = (file_cols >= b["col0"]) & (file_cols < b["col1"])
    near_abs = np.full(emo.shape[1], -1, np.int32)
    if sel_cols.any():
        src_i = file_cols[sel_cols] - b["col0"]
        near_abs[sel_cols] = near[src_i]
    file_rows = np.arange(er0, er0 + emo.shape[0])
    row_c = file_rows if sens0 else (H - 1) - file_rows
    near_c = np.where(near_abs >= 0, near_abs if sens0 else (H - 1) - near_abs, -10_000)
    behind = row_c[:, None] >= (near_c[None, :] + 6.0 / GSD)
    take = emo.astype(bool) & behind & (near_abs[None, :] >= 0) & np.isfinite(hh)
    vals = hh[take]
    if vals.size < 30:
        rec["footprint"] = "too_small"
        rec["foot_median_db"] = None
        return
    med = float(np.median(vals))
    rec["foot_median_db"] = med
    # land reference: part1 L if present, else water+6 dB as a weak fallback is NOT used.
    p1 = OUT / rec["strip"] / "part1.json"
    L = None
    if p1.exists():
        L = json.loads(p1.read_text())["table"]["L"]["all"]["median_db"]
    rec["foot_minus_L"] = (med - L) if L is not None and np.isfinite(L) else None
    if rec["foot_minus_L"] is None:
        rec["footprint"] = "uncompared"
    elif rec["foot_minus_L"] >= -1.0:
        rec["footprint"] = "bright"
    else:
        rec["footprint"] = "dark"


def _crop_tower(out, rec, b, grid, sens0, mag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pad = 300.0
    minx, miny, maxx, maxy = b["poly"].bounds
    c0 = max(0, int((minx - pad - grid["west"]) / GSD))
    c1 = min(grid["width"], int((maxx + pad - grid["west"]) / GSD) + 1)
    # range: the whole swath is only ~760 m, so ±300 m is nearly all of it
    r0 = 0
    r1 = grid["height"]
    hh = None
    with rasterio.open(mag) as src:
        hh = src.read(1, window=Window(c0, r0, c1 - c0, r1 - r0)).astype(np.float32)
    show = hh if sens0 else hh[::-1]
    step = toward_sensor(sens0)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    v = show[np.isfinite(show)]
    lo, hi = (np.percentile(v, [2, 98]) if v.size else (0, 1))
    ax.imshow(show, cmap="gray", vmin=float(lo), vmax=float(hi), origin="upper")
    wt = from_origin(grid["west"] + c0 * GSD, grid["north"] - r0 * GSD, GSD, GSD)
    fp = rasterize([(b["poly"], 1)], out_shape=hh.shape, transform=wt, fill=0, dtype=np.uint8)
    fp_s = fp if sens0 else fp[::-1]
    if fp_s.any():
        ax.contour(fp_s.astype(float), levels=[0.5], colors="white", linewidths=0.8)

    def disp(file_row, file_col):
        local_r = file_row - r0
        local_c = file_col - c0
        if not sens0:
            local_r = (r1 - r0 - 1) - local_r
        return local_c, local_r

    col = rec["file_col"]
    y_wall = rec["near"]
    ax.plot(*disp(y_wall, col), marker="o", color="white", ms=4)
    if rec.get("L_B") and np.isfinite(rec["L_B"]):
        yb = y_wall + step * int(round(rec["L_B"] / GSD))
        ax.plot([disp(y_wall, col)[0], disp(yb, col)[0]],
                [disp(y_wall, col)[1], disp(yb, col)[1]], color="cyan", lw=1.0)
        ax.plot(*disp(yb, col), marker="v", color="cyan", ms=5)
    if rec.get("L_meas") is not None:
        ym = y_wall + step * int(round(float(rec["L_meas"]) / GSD))
        ax.plot(*disp(ym, col), marker="v", color="orange", ms=5)
    ax.text(0.02, 0.98, "sensor", transform=ax.transAxes, color="white", fontsize=8, va="top")
    he = rec["h_edge"] if rec.get("h_edge") and np.isfinite(rec["h_edge"]) else float("nan")
    ax.set_title(
        f"{rec['strip']} {rec['id']}\n"
        f"h_edge {he:.0f} m  θ {rec['theta'] if rec.get('theta') else float('nan'):.0f}°  "
        f"L_B {rec['L_B'] if rec.get('L_B') else float('nan'):.0f}  "
        f"L_meas {rec['L_meas'] if rec.get('L_meas') is not None else float('nan'):.0f}  "
        f"{rec.get('footprint', '')}",
        fontsize=8,
    )
    ax.axis("off")
    fig.tight_layout()
    path = out / f"{rec['strip']}_{rec['id']}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    rec["crop"] = path.name
    log(f"    tower crop {path.name} water {rec['water_m']:.0f} m  L_B {rec.get('L_B')} L_meas {rec.get('L_meas')}")


def dispatch(stage: str, limit_chunks: int = 0):
    if stage == "part2-k":
        part2("K")
    elif stage == "part2-p":
        part2("P")
    elif stage == "part1b":
        part1b()
    else:
        raise SystemExit(f"unknown stage {stage}")
