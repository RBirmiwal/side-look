#!/usr/bin/env python3
"""DSM layover / shadow mask pipeline (primary).

Walks an AHN surface model along ground range. The 3DBAG box-model overlays
are kept as a comparison layer and are not an input to this scan.

Usage:
    python dsm_pipeline.py
    python dsm_pipeline.py --incidence 33.6 --flip-look
    python dsm_pipeline.py --look-azimuth 0
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

from dsm_geometry import (
    CLASS_ACTIVE,
    CLASS_PASSIVE,
    CLASS_SHADOW,
    fill_nodata,
    mask_from_dsm,
    mask_is_trivial,
)
from geometry import look_azimuth_from_orientation
from pipeline import (
    DEFAULT_INCIDENCE,
    OUTPUT,
    OVERLAY_DIR,
    PUBLIC_DATA,
    PUBLIC_OVERLAYS,
    STRIP_KEY,
    STRIP_NAME,
    blend,
    extract_metadata_incidence,
    parse_orientations,
    percentile_stretch,
    print_sar_report,
    window_from_bounds_clamp,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
AHN_PATH = DATA / "ahn_dsm_1m.tif"
DSM_POSTING_M = 1.0
DSM_BUFFER_M = 120.0

# Overlay colours (sRGB). Distinct on greyscale SAR.
ACTIVE_RGB = np.array([212, 120, 58], dtype=np.float32)   # copper — roof that folded
PASSIVE_RGB = np.array([140, 84, 64], dtype=np.float32)   # umber — ground it landed on
SHADOW_RGB = np.array([91, 143, 191], dtype=np.float32)   # steel-blue
ACTIVE_ALPHA = 0.55
PASSIVE_ALPHA = 0.45
SHADOW_ALPHA = 0.42


def log(msg: str) -> None:
    print(msg, flush=True)


def print_raster_report(src: rasterio.DatasetReader, path: Path, title: str) -> None:
    log(f"=== {title} ===")
    log(f"  path      {path}")
    log(f"  CRS       {src.crs}")
    log(f"  transform {src.transform}")
    log(f"  shape     {src.height} rows x {src.width} cols")
    log(f"  dtype     {src.dtypes[0]}")
    log(f"  bounds    {tuple(round(v, 3) for v in src.bounds)}")
    log(f"  nodata    {src.nodata}")


def print_height_stats(z: np.ndarray, stage: str) -> None:
    finite = z[np.isfinite(z)]
    log(f"=== DSM heights ({stage}) ===")
    log(f"  min       {float(finite.min()):.2f} m")
    log(f"  max       {float(finite.max()):.2f} m")
    log(f"  median    {float(np.median(finite)):.2f} m")
    if float(np.median(finite)) > 500:
        log("  WARNING   median > 500 m — values may be centimetres, not metres")


def prepare_dsm_1m(ahn_path: Path, sar: rasterio.DatasetReader):
    """Reproject AHN DSM to the SAR CRS at 1 m, fill nodata."""
    buf = DSM_BUFFER_M
    west = sar.bounds.left - buf
    south = sar.bounds.bottom - buf
    east = sar.bounds.right + buf
    north = sar.bounds.top + buf
    width = int(round((east - west) / DSM_POSTING_M))
    height = int(round((north - south) / DSM_POSTING_M))
    transform = from_origin(west, north, DSM_POSTING_M, DSM_POSTING_M)
    dest = np.full((height, width), np.nan, dtype=np.float32)

    with rasterio.open(ahn_path) as ahn:
        print_raster_report(ahn, ahn_path, "AHN DSM (native)")
        z = ahn.read(1)
        nodata = ahn.nodata
        valid = np.isfinite(z) & (np.abs(z) < 1.0e6)
        if nodata is not None and np.isfinite(nodata):
            valid &= z != nodata
        print_height_stats(z[valid], "native, valid cells")
        src = np.where(valid, z, np.nan).astype(np.float32)
        reproject(
            source=src,
            destination=dest,
            src_transform=ahn.transform,
            src_crs=ahn.crs,
            src_nodata=np.nan,
            dst_transform=transform,
            dst_crs=sar.crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

    filled = fill_nodata(dest, nodata=None)
    print_height_stats(filled, f"1 m {sar.crs}, nodata filled")
    out_path = OUTPUT / "dsm_1m.tif"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "height": filled.shape[0],
        "width": filled.shape[1],
        "crs": sar.crs,
        "transform": transform,
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "nodata": None,
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(filled.astype(np.float32), 1)
    log(f"  wrote {out_path}")
    return filled, transform, str(sar.crs)


def warp_mask_to_sar(mask_1m, t_1m, crs_1m, sar: rasterio.DatasetReader) -> np.ndarray:
    out = np.zeros((sar.height, sar.width), dtype=np.uint8)
    reproject(
        source=mask_1m,
        destination=out,
        src_transform=t_1m,
        src_crs=crs_1m,
        dst_transform=sar.transform,
        dst_crs=sar.crs,
        resampling=Resampling.nearest,
    )
    return out


def write_mask(path: Path, mask: np.ndarray, sar: rasterio.DatasetReader) -> None:
    profile = sar.profile.copy()
    profile.update(
        driver="GTiff",
        dtype="uint8",
        count=1,
        nodata=0,
        compress="lzw",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(mask, 1)
        dst.set_band_description(
            1, "0=nominal, 1=active layover, 2=passive layover, 3=shadow"
        )
    n1 = int((mask == CLASS_ACTIVE).sum())
    n2 = int((mask == CLASS_PASSIVE).sum())
    n3 = int((mask == CLASS_SHADOW).sum())
    log(f"  wrote {path}  shape={mask.shape}  active={n1}  passive={n2}  shadow={n3}")
    if mask_is_trivial(mask):
        raise SystemExit("DSM mask is trivial (all one class) — nodata or load failure")


def overlay_png_dsm(
    sar_u8: np.ndarray,
    mask: np.ndarray,
    out_path: Path,
    look_azimuth_deg: float,
    incidence_deg: float,
    title: str,
) -> None:
    h, w = sar_u8.shape
    rgb = np.stack([sar_u8, sar_u8, sar_u8], axis=-1).astype(np.float32)
    blend(rgb, PASSIVE_RGB, mask == CLASS_PASSIVE, PASSIVE_ALPHA)
    blend(rgb, ACTIVE_RGB, mask == CLASS_ACTIVE, ACTIVE_ALPHA)
    blend(rgb, SHADOW_RGB, mask == CLASS_SHADOW, SHADOW_ALPHA)
    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(img)

    cx, cy = 28, 28
    ux, uy = math.sin(math.radians(look_azimuth_deg)), math.cos(math.radians(look_azimuth_deg))
    sx, sy = ux, -uy
    length = 22
    x2, y2 = cx + sx * length, cy + sy * length
    paper = (232, 230, 225)
    draw.line([(cx, cy), (x2, y2)], fill=paper, width=2)
    left = (x2 - 6 * sx + 4 * sy, y2 - 6 * sy - 4 * sx)
    right = (x2 - 6 * sx - 4 * sy, y2 - 6 * sy + 4 * sx)
    draw.polygon([(x2, y2), left, right], fill=paper)
    draw.text(
        (8, h - 18),
        f"look {look_azimuth_deg:.0f}  θ {incidence_deg:.1f}°  {title}",
        fill=paper,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "PNG", optimize=True)


def load_existing_crops() -> list[dict]:
    inspect_path = OUTPUT / "inspect.json"
    if not inspect_path.exists():
        raise SystemExit("sar/output/inspect.json missing — run pipeline.py once for crop windows")
    prev = json.loads(inspect_path.read_text())
    return prev["crops"], prev


def copy_to_public() -> None:
    PUBLIC_OVERLAYS.mkdir(parents=True, exist_ok=True)
    PUBLIC_DATA.mkdir(parents=True, exist_ok=True)
    for png in OVERLAY_DIR.glob("*.png"):
        shutil.copy2(png, PUBLIC_OVERLAYS / png.name)
    inspect_path = OUTPUT / "inspect.json"
    if inspect_path.exists():
        shutil.copy2(inspect_path, PUBLIC_DATA / "inspect.json")
        shutil.copy2(inspect_path, PUBLIC_OVERLAYS / "inspect.json")
        bundled = ROOT.parent / "src" / "lib" / "inspect-report.json"
        bundled.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(inspect_path, bundled)


def run(args: argparse.Namespace) -> None:
    sar_path = Path(args.sar)
    ori_path = Path(args.orientations)
    ahn_path = Path(args.dsm)
    if not sar_path.exists():
        raise SystemExit(f"SAR strip not found: {sar_path}")
    if not ahn_path.exists():
        raise SystemExit(f"AHN DSM not found: {ahn_path}  (python download.py)")

    orientations = parse_orientations(ori_path)
    ori_flag = orientations[STRIP_KEY]
    recorded_azimuth = look_azimuth_from_orientation(ori_flag)
    look_azimuth = args.look_azimuth if args.look_azimuth is not None else recorded_azimuth
    if args.flip_look:
        look_azimuth = (look_azimuth + 180.0) % 360.0

    crops_prev, prev_inspect = load_existing_crops()

    with rasterio.open(sar_path) as src:
        print_sar_report(src, sar_path)
        meta_inc = extract_metadata_incidence(src)
        log(f"  orientation flag {ori_flag} ({'south' if ori_flag == 1 else 'north'}-facing)")
        log(f"  recorded look azimuth {recorded_azimuth:.1f} deg")
        log(f"  using look azimuth    {look_azimuth:.1f} deg")
        log(f"  incidence (CLI)       {args.incidence:.2f} deg")
        if meta_inc is not None:
            log(f"  incidence (metadata)  {meta_inc:.2f} deg  [Capella center_pixel]")
        log("  far-field: one incidence for the whole array")

        z, t_1m, crs_1m = prepare_dsm_1m(ahn_path, src)

        log("=== scanning DSM (recorded look) ===")
        mask_1m = mask_from_dsm(z, args.incidence, look_azimuth, DSM_POSTING_M)
        log(
            f"  1 m classes  active={(mask_1m == CLASS_ACTIVE).sum()}  "
            f"passive={(mask_1m == CLASS_PASSIVE).sum()}  "
            f"shadow={(mask_1m == CLASS_SHADOW).sum()}"
        )
        mask_sar = warp_mask_to_sar(mask_1m, t_1m, crs_1m, src)
        mask_path = OUTPUT / "layover_shadow_mask_dsm.tif"
        write_mask(mask_path, mask_sar, src)

        flipped_az = (look_azimuth + 180.0) % 360.0
        inc_meta = meta_inc if meta_inc is not None else args.incidence

        log("=== scanning DSM (flipped look) ===")
        mask_flip_1m = mask_from_dsm(z, args.incidence, flipped_az, DSM_POSTING_M)
        mask_flip = warp_mask_to_sar(mask_flip_1m, t_1m, crs_1m, src)

        mask_meta = None
        if meta_inc is not None and abs(meta_inc - args.incidence) > 0.5:
            log("=== scanning DSM (metadata incidence) ===")
            mask_meta_1m = mask_from_dsm(z, inc_meta, look_azimuth, DSM_POSTING_M)
            mask_meta = warp_mask_to_sar(mask_meta_1m, t_1m, crs_1m, src)

        OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
        crop_records = []
        for crop in crops_prev:
            log(f"=== overlay {crop['id']} (DSM) ===")
            bounds = crop["bounds"]
            win = window_from_bounds_clamp(bounds, src)
            band = src.read(1, window=win)
            sar_u8 = percentile_stretch(band)
            sar_only = OVERLAY_DIR / f"{crop['id']}-sar.png"
            if not sar_only.exists():
                Image.fromarray(sar_u8, mode="L").convert("RGB").save(sar_only, "PNG", optimize=True)

            r0, c0 = int(win.row_off), int(win.col_off)
            h, w = int(win.height), int(win.width)
            overlay_png_dsm(
                sar_u8,
                mask_sar[r0 : r0 + h, c0 : c0 + w],
                OVERLAY_DIR / f"{crop['id']}-dsm.png",
                look_azimuth,
                args.incidence,
                "DSM recorded",
            )
            overlay_png_dsm(
                sar_u8,
                mask_flip[r0 : r0 + h, c0 : c0 + w],
                OVERLAY_DIR / f"{crop['id']}-dsm-flipped.png",
                flipped_az,
                args.incidence,
                "DSM flipped +180",
            )
            files = {
                "overlay": f"/overlays/{crop['id']}-dsm.png",
                "flipped": f"/overlays/{crop['id']}-dsm-flipped.png",
                "sar": f"/overlays/{crop['id']}-sar.png",
                "box": f"/overlays/{crop['id']}.png",
                "box_flipped": f"/overlays/{crop['id']}-flipped.png",
            }
            if mask_meta is not None:
                overlay_png_dsm(
                    sar_u8,
                    mask_meta[r0 : r0 + h, c0 : c0 + w],
                    OVERLAY_DIR / f"{crop['id']}-dsm-inc-meta.png",
                    look_azimuth,
                    inc_meta,
                    "DSM metadata θ",
                )
                files["inc_meta"] = f"/overlays/{crop['id']}-dsm-inc-meta.png"
            if crop.get("files", {}).get("inc_meta"):
                files["box_inc_meta"] = crop["files"]["inc_meta"]

            crop_win_mask = mask_sar[r0 : r0 + h, c0 : c0 + w]
            rec = {
                "id": crop["id"],
                "title": crop["title"],
                "bounds": crop["bounds"],
                "building_count": crop.get("building_count"),
                "max_height_m": crop.get("max_height_m"),
                "median_height_m": crop.get("median_height_m"),
                "files": files,
                "stats": {
                    "window": [c0, r0, w, h],
                    "active_px": int((crop_win_mask == CLASS_ACTIVE).sum()),
                    "passive_px": int((crop_win_mask == CLASS_PASSIVE).sum()),
                    "shadow_px": int((crop_win_mask == CLASS_SHADOW).sum()),
                    "layover_px": int(
                        ((crop_win_mask == CLASS_ACTIVE) | (crop_win_mask == CLASS_PASSIVE)).sum()
                    ),
                    "buildings": crop.get("building_count"),
                },
            }
            crop_records.append(rec)

        finite = z[np.isfinite(z)]
        inspect = {
            "strip": {
                "name": STRIP_NAME,
                "key": STRIP_KEY,
                "crs": str(src.crs),
                "transform": list(src.transform)[:6],
                "shape": [src.height, src.width],
                "dtype": str(src.dtypes[0]),
                "bands": src.count,
                "bounds": list(src.bounds),
                "gsd_m": abs(src.transform.a),
                "orientation_flag": ori_flag,
                "orientation_label": "south-facing" if ori_flag == 1 else "north-facing",
                "look_azimuth_deg": look_azimuth,
                "recorded_look_azimuth_deg": recorded_azimuth,
                "incidence_deg": args.incidence,
                "metadata_incidence_deg": meta_inc,
            },
            "dsm": {
                "source": "AHN DSM (PDOK WCS dsm_05m), resampled to 1 m",
                "crs_native": "EPSG:28992",
                "crs_matched_to_sar": str(src.crs),
                "posting_m": DSM_POSTING_M,
                "height_min_m": round(float(finite.min()), 2),
                "height_max_m": round(float(finite.max()), 2),
                "height_median_m": round(float(np.median(finite)), 2),
                "nodata_filled": True,
                "assumption": "far-field: one incidence angle for the entire array",
            },
            "footprints": prev_inspect.get("footprints"),
            "mask": {
                "path": "sar/output/layover_shadow_mask_dsm.tif",
                "codes": {
                    "0": "nominal",
                    "1": "active layover",
                    "2": "passive layover",
                    "3": "shadow",
                },
                "active_px": int((mask_sar == CLASS_ACTIVE).sum()),
                "passive_px": int((mask_sar == CLASS_PASSIVE).sum()),
                "shadow_px": int((mask_sar == CLASS_SHADOW).sum()),
                "layover_px": int(
                    ((mask_sar == CLASS_ACTIVE) | (mask_sar == CLASS_PASSIVE)).sum()
                ),
                "box_model_path": "sar/output/layover_shadow_mask.tif",
                "precedence": "shadow overwrites layover",
            },
            "how_to_flip": {
                "cli": "python dsm_pipeline.py --flip-look",
                "or": "python dsm_pipeline.py --look-azimuth 0   # this strip is south-facing (180)",
                "note": (
                    "If active layover (copper) sits on the far side of each building instead of "
                    "toward the sensor, the look azimuth is 180° off. Toggle Flip look in the "
                    "viewer or rerun with --flip-look. Toggle Box model to compare against the "
                    "3DBAG extrusion mask."
                ),
            },
            "crops": crop_records,
        }
        (OUTPUT / "inspect.json").write_text(json.dumps(inspect, indent=2))
        log(f"  wrote {OUTPUT / 'inspect.json'}")

    copy_to_public()
    log("=== done ===")
    log(f"mask     {mask_path}")
    log(f"overlays {OVERLAY_DIR}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sar", default=str(DATA / "sar_strip.tif"))
    p.add_argument("--orientations", default=str(DATA / "SAR_orientations.txt"))
    p.add_argument("--dsm", default=str(AHN_PATH))
    p.add_argument(
        "--incidence",
        type=float,
        default=DEFAULT_INCIDENCE,
        help="Incidence angle in degrees from vertical (default 53.0).",
    )
    p.add_argument(
        "--look-azimuth",
        type=float,
        default=None,
        help="Override look azimuth in degrees clockwise from grid north.",
    )
    p.add_argument(
        "--flip-look",
        action="store_true",
        help="Add 180° to the look azimuth (the usual debug for a mirrored mask).",
    )
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
