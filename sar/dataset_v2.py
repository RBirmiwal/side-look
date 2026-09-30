#!/usr/bin/env python3
"""Dataset v2: geometric look + near-field masks.

Writes a new directory. Does not modify output/dataset (v1).

    python dataset_v2.py
    python dataset_v2.py --strips ID ID --out-dir output/dataset_v2_pilot
    python dataset_v2.py --resume
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling
from rasterio.windows import Window

from acquire_geometry import ecef_to_geodetic, enu_basis
from catalog import open_strip
from dataset import (
    BLOCK_M,
    EVAL_STRIDE,
    OUTPUT,
    ROOT,
    TILE_SIZE,
    TRAIN_STRIDE,
    _covered_blocks,
    affine_from_list,
    class_fractions,
    default_dsm,
    drop_reason,
    ensure_strip,
    load_catalog,
    log,
    overlay_tile,
    prepare_dsm_for_strip,
    reset_tile_dir,
    sar_rgb,
    select_strips,
    union_bounds,
    warp_to_strip,
    window_bounds,
    write_mask_geotiff,
)
from dsm_geometry import CLASS_ACTIVE, CLASS_NOMINAL, CLASS_PASSIVE, CLASS_SHADOW
from near_field import (
    add_undulation,
    canonical_flip,
    cardinal_look,
    fit_flight_line,
    look_azimuth_from_track,
    mask_and_incidence_from_flight,
    south_looking,
)
from splits import BlockGrid, assign_blocks, assert_no_train_test_overlap, bboxes_intersect

LABEL_VERSION = "v2-nearfield"
# Both geometric looks. Every id below was fit from state vectors and is
# within 0.5° of 0 or 180, and reaches the eastern side of the AOI except
# 20190804111224 (kept because it is already local and is in v1).
DEFAULT_STRIPS = [
    "20190822074237_20190822074521",  # south, local, also in v1
    "20190804111224_20190804111453",  # north, local, also in v1
    "20190822071025_20190822071321",  # north
    "20190823134629_20190823134924",  # north
    "20190823101748_20190823102106",  # south
    "20190822084226_20190822084526",  # south
    "20190822133333_20190822133635",  # north
    "20190823072627_20190823072925",  # north
    "20190823083559_20190823083940",  # north, longest; sets the west edge of the grid
    "20190804143846_20190804144102",  # south
    # Extra passes inside the same block grid, to fill the eastern test blocks.
    "20190823083119_20190823083425",
    "20190822162246_20190822162529",
    "20190822085151_20190822085445",
    "20190823085011_20190823085317",
    "20190822090458_20190822090745",
    "20190823065021_20190823065333",
    "20190822134328_20190822134645",
    "20190823124221_20190823124528",
    "20190823125228_20190823125530",
    "20190822083256_20190822083600",
]
PRESERVE_SAR = {
    "20190804111224_20190804111453",
    "20190822074237_20190822074521",
}
# Failed the +2 dB gate. Look is not reversed (opposite side is darker); left out.
EXCLUDED = [
    {
        "strip_id": "20190822071501_20190822071737",
        "look": 180.222,
        "active_minus_nominal_db": 1.480,
        "shadow_minus_nominal_db": -1.190,
        "opposite_active_minus_nominal_db": -0.389,
        "why": (
            "Flight-line look 180.222° matches the velocity right-look (179.9°) and the "
            "mean position look (180.4°). Active pixels are on the bright side; the other "
            "side of the tile scores -0.39 dB. It misses +2.0, so it is not in the set."
        ),
    },
    {
        "strip_id": "20190823070415_20190823070716",
        "look": 180.184,
        "active_minus_nominal_db": 1.572,
        "shadow_minus_nominal_db": -0.536,
        "opposite_active_minus_nominal_db": 0.092,
        "why": (
            "Flight-line look 180.184°. Shadow is darker than nominal (-0.54 dB) and the "
            "other side of the tile is only +0.09 dB, so the look is not reversed. "
            "Active misses +2.0, so the strip is not in the set."
        ),
    },
    {
        "strip_id": "20190823083559_20190823083940",
        "look": 0.480,
        "active_minus_nominal_db": 1.409,
        "shadow_minus_nominal_db": None,
        "opposite_active_minus_nominal_db": 0.283,
        "why": "Look 0.480°. The other side scores +0.28 dB, so the look is not reversed. Active misses +2.0.",
    },
    {
        "strip_id": "20190823101748_20190823102106",
        "look": 180.405,
        "active_minus_nominal_db": 0.218,
        "shadow_minus_nominal_db": None,
        "opposite_active_minus_nominal_db": -0.685,
        "why": "Look 180.405°. The other side scores -0.69 dB, so the look is not reversed. Active misses +2.0.",
    },
]
BANDS = (1, 2, 3, 4)
V1_DIR = OUTPUT / "dataset"
MANIFEST_FIELDS = [
    "tile_id",
    "split",
    "source_strip",
    "bbox_west",
    "bbox_south",
    "bbox_east",
    "bbox_north",
    "orientation_flag",
    "look_azimuth_deg",
    "incidence_deg",
    "flipped",
    "frac_nominal",
    "frac_active",
    "frac_passive",
    "frac_shadow",
    "frac_layover",
    "path",
    "minx",
    "miny",
    "maxx",
    "maxy",
    "incidence_min",
    "incidence_max",
    "platform_height_m",
    "label_version",
]


def _collect(strip_id: str) -> dict:
    with open_strip(strip_id) as src:
        return json.loads(src.tags()["TIFFTAG_IMAGEDESCRIPTION"])["collect"]


def fit_strip(collect: dict) -> dict:
    target = np.array(collect["image"]["center_pixel"]["target_position"], dtype=np.float64)
    lat, lon, height = ecef_to_geodetic(target)
    basis = enu_basis(lat, lon)
    pos = np.array([sv["position"] for sv in collect["state"]["state_vectors"]], dtype=np.float64)
    enu = (basis @ (pos - target).T).T
    fit = fit_flight_line(enu)
    pointing = collect["radar"]["pointing"]
    look = look_azimuth_from_track(fit["direction_enu"], pointing)
    t_foot = -float(np.dot(fit["origin_enu"], fit["direction_enu"]))
    foot = fit["origin_enu"] + t_foot * fit["direction_enu"]
    return {
        "lat": lat,
        "lon": lon,
        "height": height,
        "target": target,
        "fit": fit,
        "pointing": pointing,
        "look": look,
        "snapped": cardinal_look(look),
        "foot_height_m": float(foot[2]),
        "platform_height_m": float(fit["height_above_origin_m"]),
        "residual_rms_m": float(fit["residual_rms_m"]),
        "residual_max_m": float(fit["residual_max_m"]),
        "n_states": int(len(enu)),
        "range_resolution_m": float(collect["image"]["range_resolution"]),
        "incidence_field_deg": float(collect["image"]["center_pixel"]["incidence_angle"]),
    }


def _band_median(inc: np.ndarray, r0: int, r1: int) -> float:
    sl = inc[r0:r1]
    sl = sl[np.isfinite(sl)]
    if sl.size == 0:
        return float("nan")
    return float(np.median(sl))


def range_incidence(inc: np.ndarray, look: float) -> dict:
    """Near / centre / far off-nadir, degrees. Near is the small-incidence edge."""
    h = inc.shape[0]
    band = max(1, h // 20)
    north = _band_median(inc, 0, band)
    centre = _band_median(inc, h // 2 - band // 2, h // 2 + band // 2 + 1)
    south = _band_median(inc, h - band, h)
    if south_looking(look):
        near, far = north, south
    else:
        near, far = south, north
    return {"near_deg": near, "centre_deg": centre, "far_deg": far}


def _release(strip_id: str) -> None:
    if strip_id in PRESERVE_SAR:
        return
    from download import magpol_local_path

    path = magpol_local_path(strip_id)
    if path.exists():
        log(f"  release {path.name}")
        path.unlink()


def _write_tile(path: Path, image, mask, incidence, flipped: bool, meta: dict) -> None:
    look = np.full(mask.shape, 1 if flipped else 0, dtype=np.uint8)
    np.savez_compressed(
        path,
        image=np.ascontiguousarray(image, dtype=np.float16),
        mask=np.ascontiguousarray(mask, dtype=np.uint8),
        incidence=np.ascontiguousarray(incidence, dtype=np.float16),
        look=look,
        meta=np.array(json.dumps(meta)),
    )


def tile_strip(
    row: dict,
    geom: dict,
    mask: np.ndarray,
    incidence: np.ndarray,
    valid: np.ndarray,
    grid: BlockGrid,
    block_split: dict,
    set_dir: Path,
) -> tuple[list[dict], dict[str, int]]:
    flipped_flag = canonical_flip(geom["look"])
    dropped = {"sar_nodata": 0, "dsm_nodata": 0, "block_boundary": 0, "zero_layover": 0, "no_split": 0}
    kept: list[dict] = []
    transform = affine_from_list(row["transform"])
    strip_id = row["strip_id"]
    with open_strip(strip_id) as src:
        height, width = src.height, src.width
        row_starts = list(range(0, height - TILE_SIZE + 1, TRAIN_STRIDE))
        last_r = height - TILE_SIZE
        if last_r >= 0 and last_r not in row_starts:
            row_starts.append(last_r)
        col_starts = list(range(0, width - TILE_SIZE + 1, TRAIN_STRIDE))
        last_c = width - TILE_SIZE
        if last_c >= 0 and last_c not in col_starts:
            col_starts.append(last_c)
        for gr in row_starts:
            slab = src.read(indexes=BANDS, window=Window(0, gr, width, TILE_SIZE))
            mask_slab = mask[gr : gr + TILE_SIZE]
            valid_slab = valid[gr : gr + TILE_SIZE]
            inc_slab = incidence[gr : gr + TILE_SIZE]
            for c in col_starts:
                bbox = window_bounds(transform, gr, c, TILE_SIZE)
                if grid.straddles(*bbox):
                    dropped["block_boundary"] += 1
                    continue
                cx = 0.5 * (bbox[0] + bbox[2])
                cy = 0.5 * (bbox[1] + bbox[3])
                split = block_split.get(grid.block_id(cx, cy))
                if split is None:
                    dropped["no_split"] += 1
                    continue
                stride = TRAIN_STRIDE if split == "train" else EVAL_STRIDE
                if gr % stride and gr != last_r:
                    continue
                if c % stride and c != last_c:
                    continue
                img = slab[:, :, c : c + TILE_SIZE]
                m = mask_slab[:, c : c + TILE_SIZE]
                v = valid_slab[:, c : c + TILE_SIZE]
                inc = inc_slab[:, c : c + TILE_SIZE]
                if img.dtype != np.float32:
                    img = img.astype(np.float32)
                finite_img = np.isfinite(img)
                if finite_img.any() and float(img[finite_img].min()) >= 0:
                    img = np.log10(np.clip(img, 0, None) + 1e-6).astype(np.float32)
                if flipped_flag:
                    img = np.flip(img, axis=-2).copy()
                    m = np.flip(m, axis=0).copy()
                    v = np.flip(v, axis=0).copy()
                    inc = np.flip(inc, axis=0).copy()
                reason = drop_reason(img, m, v, cap_zero=1.0)
                if reason:
                    dropped[reason] += 1
                    continue
                finite_inc = inc[np.isfinite(inc)]
                inc_min = float(finite_inc.min()) if finite_inc.size else float("nan")
                inc_max = float(finite_inc.max()) if finite_inc.size else float("nan")
                frac = class_fractions(m)
                tile_id = f"{strip_id}_{gr:05d}_{c:05d}"
                meta = {
                    "strip_id": strip_id,
                    "bbox": list(bbox),
                    "crs": row["crs"],
                    "look_azimuth_deg": geom["look"],
                    "flipped": flipped_flag,
                    "platform_height_m": geom["platform_height_m"],
                    "label_version": LABEL_VERSION,
                }
                out = set_dir / split / f"{tile_id}.npz"
                _write_tile(out, img, m, inc, flipped_flag, meta)
                west, south, east, north = bbox
                rec = {
                    "tile_id": tile_id,
                    "split": split,
                    "source_strip": strip_id,
                    "bbox_west": west,
                    "bbox_south": south,
                    "bbox_east": east,
                    "bbox_north": north,
                    "orientation_flag": row["orientation_flag"],
                    "look_azimuth_deg": geom["look"],
                    "incidence_deg": geom["incidence_centre_deg"],
                    "flipped": flipped_flag,
                    **frac,
                    "path": str(out.relative_to(set_dir)),
                    "minx": west,
                    "miny": south,
                    "maxx": east,
                    "maxy": north,
                    "incidence_min": inc_min,
                    "incidence_max": inc_max,
                    "platform_height_m": geom["platform_height_m"],
                    "label_version": LABEL_VERSION,
                }
                kept.append(rec)
    return kept, dropped


def build_mask(row: dict, geom: dict, ahn_path: Path, mask_dir: Path):
    z_nap, valid, transform = prepare_dsm_for_strip(ahn_path, row)
    z_ell = add_undulation(z_nap, transform, row["crs"])
    del z_nap
    mask_1m, inc_1m = mask_and_incidence_from_flight(
        z_ell,
        transform,
        row["crs"],
        geom["snapped"],
        geom["fit"]["origin_enu"],
        geom["fit"]["direction_enu"],
        geom["lat"],
        geom["lon"],
        geom["height"],
        geom["range_resolution_m"],
    )
    del z_ell
    mask = warp_to_strip(mask_1m.astype(np.uint8), transform, row["crs"], row, Resampling.nearest, np.uint8)
    incidence = warp_to_strip(inc_1m.astype(np.float32), transform, row["crs"], row, Resampling.bilinear, np.float32)
    valid_img = warp_to_strip(
        valid.astype(np.uint8), transform, row["crs"], row, Resampling.nearest, np.uint8
    ).astype(bool)
    del mask_1m, inc_1m, valid
    mask_dir.mkdir(parents=True, exist_ok=True)
    tagged = dict(row)
    tagged["look_azimuth_deg"] = geom["look"]
    tagged["theta_mode"] = "near-field"
    tagged["incidence_deg"] = geom["incidence_field_deg"]
    write_mask_geotiff(mask_dir / f"{row['strip_id']}.tif", mask, tagged)
    return mask, incidence, valid_img


def opposite_active(paths: list[Path]) -> float:
    """Active minus nominal if the mask is flipped onto the other side of the image."""
    sums = {CLASS_NOMINAL: 0.0, CLASS_ACTIVE: 0.0}
    counts = {CLASS_NOMINAL: 0, CLASS_ACTIVE: 0}
    for path in paths:
        z = np.load(path)
        hh = z["image"][0].astype(np.float32)
        mask = np.flip(z["mask"], axis=0)
        finite = np.isfinite(hh)
        for cls in sums:
            sel = finite & (mask == cls)
            sums[cls] += float(hh[sel].sum())
            counts[cls] += int(sel.sum())
    nom = sums[CLASS_NOMINAL] / counts[CLASS_NOMINAL]
    act = sums[CLASS_ACTIVE] / counts[CLASS_ACTIVE]
    return act - nom


def contrast(paths: list[Path]) -> dict:
    sums = {CLASS_NOMINAL: 0.0, CLASS_ACTIVE: 0.0, CLASS_SHADOW: 0.0}
    counts = {CLASS_NOMINAL: 0, CLASS_ACTIVE: 0, CLASS_SHADOW: 0}
    class_n = np.zeros(4, dtype=np.int64)
    row_sum = 0.0
    row_n = 0
    for path in paths:
        z = np.load(path)
        hh = z["image"][0].astype(np.float32)
        mask = z["mask"]
        finite = np.isfinite(hh)
        for cls in sums:
            sel = finite & (mask == cls)
            sums[cls] += float(hh[sel].sum())
            counts[cls] += int(sel.sum())
        class_n += np.bincount(mask.ravel(), minlength=4)
        active = mask == CLASS_ACTIVE
        if np.any(active):
            rr = np.arange(mask.shape[0], dtype=np.float64)[:, None]
            row_sum += float((rr * active).sum())
            row_n += int(active.sum())
    def mean(cls):
        return sums[cls] / counts[cls] if counts[cls] else float("nan")
    total = int(class_n.sum()) or 1
    nom, act, sh = mean(CLASS_NOMINAL), mean(CLASS_ACTIVE), mean(CLASS_SHADOW)
    return {
        "n_tiles": len(paths),
        "shadow_minus_nominal_db": sh - nom,
        "active_minus_nominal_db": act - nom,
        "hh_nominal_db": nom,
        "hh_active_db": act,
        "hh_shadow_db": sh,
        "frac": (class_n / total).tolist(),
        "active_mean_row": (row_sum / row_n) if row_n else float("nan"),
    }


def norm_from_train(paths: list[Path]) -> dict:
    sums = np.zeros(4, dtype=np.float64)
    sq = np.zeros(4, dtype=np.float64)
    n = 0
    for path in paths:
        image = np.load(path)["image"].astype(np.float32)
        x = image.reshape(4, -1)
        finite = np.isfinite(x)
        sums += np.where(finite, x, 0).sum(axis=1)
        sq += np.where(finite, x * x, 0).sum(axis=1)
        n += int(finite[0].sum())
    mean = (sums / max(n, 1)).tolist()
    var = sq / max(n, 1) - np.array(mean) ** 2
    std = np.sqrt(np.clip(var, 1e-12, None)).tolist()
    return {
        "bands": ["HH", "HV", "VH", "VV"],
        "mean": mean,
        "std": std,
        "n_pixels": n,
        "split": "train",
        "label_version": LABEL_VERSION,
        "note": "v2 train split only, image bands HH HV VH VV. Incidence is not included.",
    }


def choose_overlays(records: list[dict], n: int = 20) -> list[dict]:
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for rec in records:
        side = "south" if south_looking(rec["look_azimuth_deg"]) else "north"
        buckets[(rec["split"], side)].append(rec)
    for key in buckets:
        buckets[key].sort(key=lambda r: -float(r["frac_active"]))
    order = [(s, side) for s in ("train", "val", "test") for side in ("north", "south")]
    picked: list[dict] = []
    seen = set()
    for take in (2, 8):
        for key in order:
            for rec in buckets.get(key, [])[:take]:
                if rec["tile_id"] in seen:
                    continue
                picked.append(rec)
                seen.add(rec["tile_id"])
                if len(picked) >= n:
                    return picked
    return picked[:n]


def render_overlays(records: list[dict], set_dir: Path) -> list[dict]:
    ov = set_dir / "overlays"
    if ov.exists():
        shutil.rmtree(ov)
    ov.mkdir(parents=True, exist_ok=True)
    meta = []
    for i, rec in enumerate(choose_overlays(records)):
        data = np.load(set_dir / rec["path"])
        image = data["image"].astype(np.float32)
        mask = data["mask"]
        side = "S" if south_looking(rec["look_azimuth_deg"]) else "N"
        name = f"{i:02d}_{rec['split']}_{side}_{rec['tile_id'][-22:]}.png"
        overlay_tile(
            image,
            mask,
            ov / name,
            f"{rec['split']} {side} flip={int(rec['flipped'])} {rec['source_strip'][-6:]}",
        )
        sar_rgb(image).save(ov / name.replace(".png", "-sar.png"), "PNG", optimize=True)
        meta.append({"file": name, "split": rec["split"], "side": side, "strip": rec["source_strip"]})
    return meta


def sha256_tree(set_dir: Path) -> None:
    lines = []
    for path in sorted(set_dir.rglob("*.npz")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(set_dir)}")
    (set_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n")


def _bbox(rec: dict) -> tuple[float, float, float, float]:
    return (float(rec["minx"]), float(rec["miny"]), float(rec["maxx"]), float(rec["maxy"]))


def boxes_overlap(a: list[dict], b: list[dict]) -> int:
    n = 0
    bb = [_bbox(r) for r in b]
    for rec in a:
        box = _bbox(rec)
        for other in bb:
            if bboxes_intersect(box, other):
                n += 1
                break
    return n


def v1_contrast(strip_id: str) -> dict | None:
    paths = sorted(V1_DIR.rglob(f"{strip_id}_*.npz"))
    if not paths:
        return None
    return contrast(paths)


def write_report(path: Path, payload: dict) -> None:
    lines = []
    a = lines.append
    a("# Dataset v2")
    a("")
    a("Near-field layover/shadow labels and a geometric look. The v1 409-tile set was not modified.")
    a("")
    a("## What changed")
    a("")
    a("- Look azimuth is perpendicular to the fitted flight line, on the `collect.radar.pointing` side. The SpaceNet orientation flag is not used. On these collects the flag is backwards: flag 0 looks south, flag 1 looks north.")
    a("- Masks use the near-field fold test in NEAR_FIELD.md. No single theta. Slant-range bin is `collect.image.range_resolution`. DSM heights are NAP plus NLGEO2018.")
    a("- Canonical orientation: sensor at the top of the tile (row 0), layover toward row 0. A north-up product already has that for a south-looking sensor. North-looking strips are flipped. Image, mask, and incidence flip together.")
    a("- Tile size, strides, the west-to-east 1 km block split, drop rules, and class codes are unchanged from v1.")
    a("")
    a("A fitted look more than 5° from 0 or 180 is skipped rather than rotated.")
    if payload["skipped"]:
        a("")
        a("Skipped:")
        for row in payload["skipped"]:
            a(f"- `{row['strip_id']}` look {row['look']:.2f}°")
    else:
        a("No selected strip was more than 5° off north or south.")
    if payload.get("excluded"):
        a("")
        a("Strips that were within 5° of north or south but missed the +2.0 dB gate are not in the tiles. On each of them the other side of the mask is darker, so this is not a reversed look.")
        a("")
        for row in payload["excluded"]:
            shadow = row["shadow_minus_nominal_db"]
            shadow_txt = "n/a" if shadow is None else f"{shadow:+.3f} dB"
            a(
                f"- `{row['strip_id']}` look {row['look']:.3f}°. "
                f"Active − nominal {row['active_minus_nominal_db']:+.3f} dB, "
                f"shadow − nominal {shadow_txt}. "
                f"Opposite side of the same tiles: active − nominal {row['opposite_active_minus_nominal_db']:+.3f} dB. "
                f"{row['why']}"
            )
    a("")
    n_south = sum(1 for g in payload["geometry"] if g["snapped"] == 180)
    n_north = sum(1 for g in payload["geometry"] if g["snapped"] == 0)
    a("## Strips")
    a("")
    a(
        f"{len(payload['geometry'])} collects ({n_north} looking north, {n_south} looking south), "
        "spread across the Rotterdam AOI so the eastern blocks are imaged by more than one swath. "
        "The two v1 strips are included; both were already on disk."
    )
    a("")
    a("| strip | flag | look ° | snap | height m | rms m | near ° | centre ° | far ° | flip |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for g in payload["geometry"]:
        a(
            f"| `{g['strip_id']}` | {g['orientation_flag']} | {g['look']:.3f} | {g['snapped']:.0f} | "
            f"{g['platform_height_m']:.1f} | {g['residual_rms_m']:.2f} | {g['near_deg']:.2f} | "
            f"{g['centre_deg']:.2f} | {g['far_deg']:.2f} | {g['flipped']} |"
        )
    a("")
    a("## Tiles")
    a("")
    a("| split | tiles |")
    a("|---|---:|")
    for split, n in payload["counts"].items():
        a(f"| {split} | {n} |")
    a("")
    a(f"Total {payload['n_tiles']}. Image is float16 dB. `incidence` is its own float16 key, degrees, not a fifth band. `look` is 1 if the tile was flipped. `label_version` is `{LABEL_VERSION}`.")
    a("")
    a("## Class fractions")
    a("")
    a("| split | nominal | active | passive | shadow |")
    a("|---|---:|---:|---:|---:|")
    for split, fr in payload["fractions"].items():
        a(f"| {split} | {fr[0]:.4f} | {fr[1]:.4f} | {fr[2]:.4f} | {fr[3]:.4f} |")
    a("")
    a("## Acceptance")
    a("")
    a("Active layover minus nominal HH must be greater than +2.0 dB on every strip. Shadow is reported and does not block.")
    a("")
    a("| strip | tiles | active − nominal | shadow − nominal | active mean row |")
    a("|---|---:|---:|---:|---:|")
    for row in payload["per_strip"]:
        a(
            f"| `{row['strip_id']}` | {row['n_tiles']} | {row['active_minus_nominal_db']:+.3f} | "
            f"{row['shadow_minus_nominal_db']:+.3f} | {row['active_mean_row']:.1f} |"
        )
    a("")
    overall = payload["overall"]
    a(
        f"Overall: active − nominal {overall['active_minus_nominal_db']:+.3f} dB, "
        f"shadow − nominal {overall['shadow_minus_nominal_db']:+.3f} dB, "
        f"active mean row {overall['active_mean_row']:.1f} (tile centre is 256; layover should sit toward row 0)."
    )
    a("")
    a(f"Train/test bounding boxes intersecting: **{payload['train_test_hits']}**. Train/val: {payload['train_val_hits']}. Val/test: {payload['val_test_hits']}.")
    a("")
    a(f"Overlays: {payload['n_overlays']} PNGs in `overlays/`, covering {', '.join(payload['overlay_splits'])} and look sides {', '.join(payload['overlay_sides'])}.")
    a("")
    a("## Normalisation")
    a("")
    a("Train split only. Image bands. Incidence is not in these stats.")
    a("")
    a("| band | mean | std |")
    a("|---|---:|---:|")
    norm = payload["norm"]
    for band, mean, std in zip(norm["bands"], norm["mean"], norm["std"]):
        a(f"| {band} | {mean:.4f} | {std:.4f} |")
    a("")
    a(f"Pixels: {norm['n_pixels']}.")
    a("")
    a("## Against v1, shared strips")
    a("")
    a("v1 used the orientation flag as the look, which is reversed, and a single theta. v2 uses the flight line. The flip itself matches v1 on these two strips: flag 1 (actually looking north) was flipped then, and is flipped now because the sensor is on the south edge.")
    a("")
    a("| strip | set | active − nominal | shadow − nominal |")
    a("|---|---|---:|---:|")
    for row in payload["v1_compare"]:
        a(
            f"| `{row['strip_id']}` | v2 | {row['v2_active']:+.3f} | {row['v2_shadow']:+.3f} |"
        )
        if row["v1_active"] is None:
            a(f"| `{row['strip_id']}` | v1 |  |  |")
        else:
            a(
                f"| `{row['strip_id']}` | v1 | {row['v1_active']:+.3f} | {row['v1_shadow']:+.3f} |"
            )
    a("")
    a("## Download")
    a("")
    a("Same layout as v1: `train/`, `val/`, `test/` of `.npz`, plus `manifest.csv`, `norm_stats.json`, `SHA256SUMS`.")
    a("")
    if payload.get("release_url"):
        a(payload["release_url"])
    else:
        a("Release URL is filled in when the tarball is uploaded.")
    a("")
    path.write_text("\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> None:
    catalog = load_catalog()
    ids = args.strips or DEFAULT_STRIPS
    rows = select_strips(catalog, False, None, ids)
    by_id = {r["strip_id"]: r for r in rows}
    set_dir = Path(args.out_dir)
    if not set_dir.is_absolute():
        set_dir = (ROOT / set_dir).resolve()
    if set_dir.resolve() == V1_DIR.resolve():
        raise SystemExit("refusing to write v2 onto output/dataset")
    mask_dir = set_dir.parent / "masks_v2"
    if args.mask_dir:
        mask_dir = Path(args.mask_dir)
        if not mask_dir.is_absolute():
            mask_dir = (ROOT / mask_dir).resolve()

    log("=== look preflight (headers only) ===")
    kept_rows = []
    geoms = []
    skipped = []
    for row in rows:
        geom = fit_strip(_collect(row["strip_id"]))
        off = min(
            min(geom["look"] % 360.0, 360.0 - (geom["look"] % 360.0)),
            abs((geom["look"] % 360.0) - 180.0),
        )
        status = "keep" if geom["snapped"] is not None else "SKIP"
        log(
            f"  {status} {row['strip_id']}  flag {row['orientation_flag']}  "
            f"look {geom['look']:.3f}°  off {off:.2f}°  "
            f"h {geom['platform_height_m']:.1f} m  rms {geom['residual_rms_m']:.2f} m"
        )
        if geom["snapped"] is None:
            skipped.append({"strip_id": row["strip_id"], "look": geom["look"]})
            continue
        kept_rows.append(row)
        geoms.append(geom)
    if len(kept_rows) < 5 and not args.allow_few:
        raise SystemExit(f"only {len(kept_rows)} strips within 5° of 0/180")
    sides = Counter("south" if south_looking(g["look"]) else "north" for g in geoms)
    log(f"  kept {len(kept_rows)}  looks {dict(sides)}  skipped {len(skipped)}")
    if sides["north"] < 1 or sides["south"] < 1:
        raise SystemExit(f"need both look directions, got {dict(sides)}")

    west, south, east, north = union_bounds(kept_rows)
    grid = BlockGrid(
        origin_x=math.floor(west / BLOCK_M) * BLOCK_M,
        origin_y=math.floor(south / BLOCK_M) * BLOCK_M,
        size=BLOCK_M,
    )
    block_split = assign_blocks(sorted(_covered_blocks(grid, kept_rows)), (0.70, 0.15, 0.15))
    log(
        "  blocks "
        + ", ".join(f"{s}={sum(v == s for v in block_split.values())}" for s in ("train", "val", "test"))
    )
    set_dir.mkdir(parents=True, exist_ok=True)
    records_path = set_dir / "records.jsonl"
    done_path = set_dir / "done_strips.txt"
    if not args.resume:
        reset_tile_dir(set_dir)
        if records_path.exists():
            records_path.unlink()
        if done_path.exists():
            done_path.unlink()
    done = set(done_path.read_text().split()) if done_path.exists() else set()
    records: list[dict] = []
    if records_path.exists():
        records = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
    excluded_rows = list(EXCLUDED)

    ahn_path = default_dsm()
    log(f"  dsm {ahn_path}")
    log(f"  out {set_dir}")
    geom_rows = []
    for row, geom in zip(kept_rows, geoms):
        if row["strip_id"] in done:
            log(f"=== resume skip {row['strip_id']} ===")
            continue
        log(f"=== {row['strip_id']} ===")
        for split_name in ("train", "val", "test"):
            folder = set_dir / split_name
            if folder.exists():
                for stale in folder.glob(f"{row['strip_id']}_*.npz"):
                    stale.unlink()
        records = [rec for rec in records if rec["source_strip"] != row["strip_id"]]
        if records_path.exists():
            records_path.write_text("".join(json.dumps(rec) + "\n" for rec in records))
        ensure_strip(row["strip_id"], fetch=True)
        mask, incidence, valid = build_mask(row, geom, ahn_path, mask_dir)
        stats = range_incidence(incidence, geom["look"])
        geom["incidence_centre_deg"] = stats["centre_deg"]
        geom["near_deg"] = stats["near_deg"]
        geom["far_deg"] = stats["far_deg"]
        log(
            f"  look {geom['look']:.3f}° snap {geom['snapped']:.0f}  "
            f"h {geom['platform_height_m']:.1f} m  "
            f"incidence near {stats['near_deg']:.2f}  centre {stats['centre_deg']:.2f}  far {stats['far_deg']:.2f}"
        )
        tiles, dropped = tile_strip(row, geom, mask, incidence, valid, grid, block_split, set_dir)
        del mask, incidence, valid
        gc.collect()
        log(f"  kept {len(tiles)}  dropped {dropped}")
        with records_path.open("a") as fh:
            for rec in tiles:
                fh.write(json.dumps(rec) + "\n")
        records.extend(tiles)
        paths = [set_dir / rec["path"] for rec in tiles]
        physics = contrast(paths) if paths else None
        if physics is None or not (physics["active_minus_nominal_db"] > 2.0):
            got = None if physics is None else physics["active_minus_nominal_db"]
            opp = opposite_active(paths) if paths else float("nan")
            if paths and opp > (got if got is not None else -1e9):
                raise SystemExit(
                    f"STOP {row['strip_id']}: active-nominal {got} but the other side is {opp:+.3f}. Look error."
                )
            log(
                f"  EXCLUDE {row['strip_id']}: active-nominal {got} "
                f"(other side {opp:+.3f}). Look is not reversed; below +2.0, left out."
            )
            for path in paths:
                path.unlink(missing_ok=True)
            records = [rec for rec in records if rec["source_strip"] != row["strip_id"]]
            records_path.write_text("".join(json.dumps(rec) + "\n" for rec in records))
            excluded_rows.append(
                {
                    "strip_id": row["strip_id"],
                    "look": geom["look"],
                    "active_minus_nominal_db": got,
                    "shadow_minus_nominal_db": None if physics is None else physics["shadow_minus_nominal_db"],
                    "opposite_active_minus_nominal_db": opp,
                    "why": "Active layover is on the brighter side, but the contrast misses +2.0 dB.",
                }
            )
            _release(row["strip_id"])
            with done_path.open("a") as fh:
                fh.write(row["strip_id"] + "\n")
            done.add(row["strip_id"])
            continue
        log(
            f"  HH active-nominal {physics['active_minus_nominal_db']:+.3f}  "
            f"shadow-nominal {physics['shadow_minus_nominal_db']:+.3f}  "
            f"active row {physics['active_mean_row']:.1f}"
        )
        geom_rows.append(
            {
                "strip_id": row["strip_id"],
                "orientation_flag": row["orientation_flag"],
                "look": geom["look"],
                "snapped": geom["snapped"],
                "flipped": canonical_flip(geom["look"]),
                "platform_height_m": geom["platform_height_m"],
                "foot_height_m": geom["foot_height_m"],
                "residual_rms_m": geom["residual_rms_m"],
                "residual_max_m": geom["residual_max_m"],
                "near_deg": stats["near_deg"],
                "centre_deg": stats["centre_deg"],
                "far_deg": stats["far_deg"],
                "range_resolution_m": geom["range_resolution_m"],
                "pointing": geom["pointing"],
                "n_tiles": len(tiles),
                "active_minus_nominal_db": physics["active_minus_nominal_db"],
                "shadow_minus_nominal_db": physics["shadow_minus_nominal_db"],
            }
        )
        (set_dir / "geometry.json").write_text(json.dumps(geom_rows, indent=2))
        with done_path.open("a") as fh:
            fh.write(row["strip_id"] + "\n")
        done.add(row["strip_id"])
        _release(row["strip_id"])

    if not records:
        raise SystemExit("FAIL: no tiles")
    counts = Counter(r["split"] for r in records)
    log(f"=== tiles {dict(counts)} ===")
    if not args.allow_few and (counts["val"] < 50 or counts["test"] < 50):
        raise SystemExit(f"val/test below 50: {dict(counts)}")

    man_path = set_dir / "manifest.csv"
    with man_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec[k] for k in MANIFEST_FIELDS})

    by_split = defaultdict(list)
    for rec in records:
        by_split[rec["split"]].append(set_dir / rec["path"])
    norm = norm_from_train(by_split["train"])
    (set_dir / "norm_stats.json").write_text(json.dumps(norm, indent=2))
    log(
        "  norm "
        + " ".join(f"{b} {m:.3f}±{s:.3f}" for b, m, s in zip(norm["bands"], norm["mean"], norm["std"]))
    )

    per_strip = []
    for row, geom in zip(kept_rows, geoms):
        paths = [set_dir / r["path"] for r in records if r["source_strip"] == row["strip_id"]]
        if not paths:
            continue
        physics = contrast(paths)
        per_strip.append({"strip_id": row["strip_id"], **physics})
        if not (physics["active_minus_nominal_db"] > 2.0):
            raise SystemExit(f"STOP {row['strip_id']}: active-nominal {physics['active_minus_nominal_db']}")
    overall = contrast([set_dir / r["path"] for r in records])
    fractions = {}
    for split in ("train", "val", "test"):
        physics = contrast(by_split[split])
        fractions[split] = physics["frac"]

    train_boxes = [r for r in records if r["split"] == "train"]
    val_boxes = [r for r in records if r["split"] == "val"]
    test_boxes = [r for r in records if r["split"] == "test"]
    assert_no_train_test_overlap([_bbox(r) for r in train_boxes], [_bbox(r) for r in test_boxes])
    train_test_hits = boxes_overlap(train_boxes, test_boxes)
    train_val_hits = boxes_overlap(train_boxes, val_boxes)
    val_test_hits = boxes_overlap(val_boxes, test_boxes)
    log(f"  bbox hits train/test {train_test_hits} train/val {train_val_hits} val/test {val_test_hits}")

    overlays = render_overlays(records, set_dir)
    log("  sha256")
    sha256_tree(set_dir)

    # geometry.json may lack strips that were resumed. Rebuild the table from a fresh header fit
    # plus incidence numbers already stored when the strip was tiled. Fall back to geometry.json.
    stored = {}
    geo_path = set_dir / "geometry.json"
    if geo_path.exists():
        stored = {g["strip_id"]: g for g in json.loads(geo_path.read_text())}
    geometry_table = []
    tiled_ids = {r["source_strip"] for r in records}
    for row, geom in zip(kept_rows, geoms):
        if row["strip_id"] not in tiled_ids:
            continue
        extra = stored.get(row["strip_id"], {})
        geometry_table.append(
            {
                "strip_id": row["strip_id"],
                "orientation_flag": row["orientation_flag"],
                "look": geom["look"],
                "snapped": geom["snapped"],
                "flipped": canonical_flip(geom["look"]),
                "platform_height_m": geom["platform_height_m"],
                "residual_rms_m": geom["residual_rms_m"],
                "near_deg": extra.get("near_deg", float("nan")),
                "centre_deg": extra.get("centre_deg", float("nan")),
                "far_deg": extra.get("far_deg", float("nan")),
            }
        )
    per_by = {s["strip_id"]: s for s in per_strip}
    v1_rows = []
    for row in kept_rows:
        mine = per_by.get(row["strip_id"])
        if mine is None:
            continue
        old = v1_contrast(row["strip_id"])
        if old is None:
            continue
        v1_rows.append(
            {
                "strip_id": row["strip_id"],
                "v2_active": mine["active_minus_nominal_db"],
                "v2_shadow": mine["shadow_minus_nominal_db"],
                "v1_active": old["active_minus_nominal_db"],
                "v1_shadow": old["shadow_minus_nominal_db"],
            }
        )
    payload = {
        "skipped": skipped,
        "geometry": geometry_table,
        "counts": {s: counts[s] for s in ("train", "val", "test")},
        "n_tiles": len(records),
        "fractions": fractions,
        "per_strip": per_strip,
        "overall": overall,
        "train_test_hits": train_test_hits,
        "train_val_hits": train_val_hits,
        "val_test_hits": val_test_hits,
        "n_overlays": len(overlays),
        "overlay_splits": sorted({o["split"] for o in overlays}),
        "overlay_sides": sorted({o["side"] for o in overlays}),
        "norm": norm,
        "v1_compare": v1_rows,
        "excluded": excluded_rows,
    }
    report = ROOT.parent / "REPORT_v2.md"
    write_report(report, payload)
    shutil.copy(report, set_dir / "REPORT_v2.md")
    log(f"=== wrote {report} ===")
    log(f"  val {counts['val']}  test {counts['test']}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--strips", nargs="*", default=None)
    p.add_argument("--out-dir", default=str(OUTPUT / "dataset_v2"))
    p.add_argument("--mask-dir", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--allow-few", action="store_true", help="Pilot switch: do not require 5 strips or 50 val/test tiles.")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
