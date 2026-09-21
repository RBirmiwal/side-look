#!/usr/bin/env python3
"""End-to-end layover / shadow mask pipeline for one SpaceNet 6 MAG-POL strip.

This is a validation tool, not a production processor. The mask is pixel-aligned
with the SAR strip so a human can overlay it and judge whether look direction
and incidence are right.

Usage:
    python pipeline.py
    python pipeline.py --incidence 33.6 --flip-look
    python pipeline.py --look-azimuth 0
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from PIL import Image, ImageDraw
from rasterio.features import rasterize
from rasterio.windows import Window, from_bounds
from shapely.geometry import box

from geometry import (
    ensure_projected_crs,
    layover_region,
    look_azimuth_from_orientation,
    shadow_region,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUT = ROOT / "output"
OVERLAY_DIR = OUTPUT / "overlays"
PUBLIC_OVERLAYS = ROOT.parent / "public" / "overlays"
PUBLIC_DATA = ROOT.parent / "public" / "data"

STRIP_KEY = "20190804111224_20190804111453"
STRIP_NAME = f"SN6_AOI_11_Rotterdam_SAR-MAG-POL_{STRIP_KEY}.tif"
DEFAULT_INCIDENCE = 53.0
HEIGHT_COL = "height_m"
RD_CRS = "EPSG:28992"

# Overlay colours (sRGB). Distinct on greyscale SAR, not used as UI chrome.
LAYOVER_RGB = np.array([212, 120, 58], dtype=np.float32)
SHADOW_RGB = np.array([91, 143, 191], dtype=np.float32)
FOOTPRINT_RGB = (232, 230, 225)
LAYOVER_ALPHA = 0.48
SHADOW_ALPHA = 0.42

CROP_M = 360.0  # a few hundred metres across a built-up block


def log(msg: str) -> None:
    print(msg, flush=True)


def parse_orientations(path: Path) -> dict[str, int]:
    table: dict[str, int] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, flag = line.split()
        table[key] = int(flag)
    return table


def extract_metadata_incidence(src: rasterio.DatasetReader) -> float | None:
    raw = src.tags().get("TIFFTAG_IMAGEDESCRIPTION", "")
    if not raw:
        return None
    try:
        meta = json.loads(raw)
        return float(meta["collect"]["image"]["center_pixel"]["incidence_angle"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        match = re.search(r'"incidence_angle":\s*([0-9.]+)', raw)
        if match:
            return float(match.group(1))
    return None


def print_sar_report(src: rasterio.DatasetReader, path: Path) -> None:
    log("=== SAR strip ===")
    log(f"  path      {path}")
    log(f"  CRS       {src.crs}")
    log(f"  transform {src.transform}")
    log(f"  shape     {src.height} rows x {src.width} cols")
    log(f"  dtype     {src.dtypes[0]}")
    log(f"  bands     {src.count}")
    log(f"  bounds    {tuple(round(v, 3) for v in src.bounds)}")


def load_footprints(gpkg_dir: Path) -> gpd.GeoDataFrame:
    frames = []
    gpkgs = sorted(gpkg_dir.glob("*.gpkg"))
    if not gpkgs:
        raise FileNotFoundError(f"No GeoPackages in {gpkg_dir}")
    for path in gpkgs:
        lod = gpd.read_file(path, layer="lod12_2d")
        pand = gpd.read_file(path, layer="pand")[["identificatie", "b3_h_maaiveld"]]
        merged = lod.merge(pand, on="identificatie", how="left")
        frames.append(merged)
        log(f"  loaded {path.name}: {len(merged)} lod12 parts")
    gdf = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)
    return gdf


def add_building_height(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Height above ground = median roof (NAP) minus ground (NAP)."""
    roof = pd.to_numeric(gdf["b3_h_50p"], errors="coerce")
    ground = pd.to_numeric(gdf["b3_h_maaiveld"], errors="coerce")
    gdf = gdf.copy()
    gdf.geometry = gdf.geometry.force_2d()
    gdf[HEIGHT_COL] = roof - ground
    before = len(gdf)
    valid = gdf[HEIGHT_COL].notna() & (gdf[HEIGHT_COL] > 0) & gdf.geometry.notna()
    gdf = gdf.loc[valid].copy()
    skipped = before - len(gdf)
    log(
        f"  height column: {HEIGHT_COL} = b3_h_50p - b3_h_maaiveld "
        f"(skipped {skipped} with missing/zero/negative height)"
    )
    return gdf


def print_footprint_report(gdf: gpd.GeoDataFrame, stage: str) -> None:
    h = gdf[HEIGHT_COL]
    log(f"=== footprints ({stage}) ===")
    log(f"  CRS            {gdf.crs}")
    log(f"  feature count  {len(gdf)}")
    log(f"  height column  {HEIGHT_COL}")
    log(f"  height range   {float(h.min()):.2f} .. {float(h.max()):.2f} m")
    log(f"  height median  {float(h.median()):.2f} m")


def reproject_to_rd(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    ensure_projected_crs(gdf.crs)
    out = gdf.to_crs(RD_CRS)
    ensure_projected_crs(out.crs)
    return out


def building_regions(gdf: gpd.GeoDataFrame, incidence_deg: float, look_azimuth_deg: float):
    layovers = []
    shadows = []
    for geom, height in zip(gdf.geometry, gdf[HEIGHT_COL]):
        if geom is None or geom.is_empty:
            continue
        try:
            if not geom.is_valid:
                geom = geom.buffer(0)
            layovers.append(layover_region(geom, float(height), incidence_deg, look_azimuth_deg))
            shadows.append(shadow_region(geom, float(height), incidence_deg, look_azimuth_deg))
        except Exception:
            continue
    return layovers, shadows


def rasterise_mask(layover_geoms, shadow_geoms, transform, shape) -> np.ndarray:
    """Burn class codes onto the SAR grid. 0 nominal, 1 layover, 2 shadow.

    Shadow is written first; layover overwrites so returns dominate.
    """
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    shadow_shapes = [(g, 2) for g in shadow_geoms if g is not None and not g.is_empty]
    layover_shapes = [(g, 1) for g in layover_geoms if g is not None and not g.is_empty]
    if shadow_shapes:
        rasterize(
            shadow_shapes,
            out=mask,
            transform=transform,
            fill=0,
            dtype=np.uint8,
            all_touched=False,
        )
    if layover_shapes:
        rasterize(
            layover_shapes,
            out=mask,
            transform=transform,
            fill=0,
            dtype=np.uint8,
            all_touched=False,
        )
    return mask


def write_mask_geotiff(path: Path, mask: np.ndarray, src: rasterio.DatasetReader) -> None:
    profile = src.profile.copy()
    profile.update(
        {
            "driver": "GTiff",
            "dtype": "uint8",
            "count": 1,
            "nodata": 0,
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
        }
    )
    # Drop scanline-only keys that conflict with tiling.
    profile.pop("blockysize", None)
    profile.pop("blockxsize", None)
    profile.update(tiled=True, blockxsize=256, blockysize=256, compress="lzw", count=1, dtype="uint8", nodata=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(mask, 1)
        dst.set_band_description(1, "0=nominal, 1=layover, 2=shadow")
    log(f"  wrote {path}  shape={mask.shape}  layover={(mask == 1).sum()}  shadow={(mask == 2).sum()}")


def percentile_stretch(band: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    finite = band[np.isfinite(band)]
    if finite.size == 0:
        return np.zeros(band.shape, dtype=np.uint8)
    p_lo, p_hi = np.percentile(finite, [lo, hi])
    if p_hi <= p_lo:
        return np.zeros(band.shape, dtype=np.uint8)
    scaled = (band - p_lo) / (p_hi - p_lo)
    scaled = np.clip(scaled, 0, 1)
    scaled[~np.isfinite(band)] = 0
    return (scaled * 255).astype(np.uint8)


def blend(rgb: np.ndarray, colour: np.ndarray, where: np.ndarray, alpha: float) -> None:
    if not np.any(where):
        return
    rgb[where] = (1.0 - alpha) * rgb[where] + alpha * colour


def overlay_png(
    sar_u8: np.ndarray,
    mask: np.ndarray,
    footprints: gpd.GeoDataFrame,
    window_transform,
    out_path: Path,
    look_azimuth_deg: float,
    incidence_deg: float,
    title: str,
) -> None:
    h, w = sar_u8.shape
    rgb = np.stack([sar_u8, sar_u8, sar_u8], axis=-1).astype(np.float32)
    blend(rgb, LAYOVER_RGB, mask == 1, LAYOVER_ALPHA)
    blend(rgb, SHADOW_RGB, mask == 2, SHADOW_ALPHA)
    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(img)

    # Footprint outlines so a building can be judged against its layover/shadow.
    for geom in footprints.geometry:
        if geom is None or geom.is_empty:
            continue
        for poly in getattr(geom, "geoms", [geom]):
            exterior = getattr(poly, "exterior", None)
            if poly.is_empty or exterior is None:
                continue
            pixels = []
            for coord in exterior.coords:
                x, y = coord[0], coord[1]
                c, r = ~window_transform * (x, y)
                pixels.append((c, r))
            if len(pixels) >= 2:
                draw.line(pixels, fill=FOOTPRINT_RGB, width=1)

    # Look-direction chevron in the corner.
    cx, cy = 28, 28
    ux, uy = math.sin(math.radians(look_azimuth_deg)), math.cos(math.radians(look_azimuth_deg))
    # Image row increases south, so screen-y is opposite grid-north.
    sx, sy = ux, -uy
    length = 22
    x2, y2 = cx + sx * length, cy + sy * length
    draw.line([(cx, cy), (x2, y2)], fill=(232, 230, 225), width=2)
    # Arrow head.
    left = (
        x2 - 6 * sx + 4 * sy,
        y2 - 6 * sy - 4 * sx,
    )
    right = (
        x2 - 6 * sx - 4 * sy,
        y2 - 6 * sy + 4 * sx,
    )
    draw.polygon([(x2, y2), left, right], fill=(232, 230, 225))
    draw.text((8, h - 18), f"look {look_azimuth_deg:.0f}  θ {incidence_deg:.1f}°  {title}", fill=(232, 230, 225))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "PNG", optimize=True)


def pick_crops(buildings: gpd.GeoDataFrame, sar_bounds, n: int = 4) -> list[dict]:
    """Slide along-range windows and keep the densest built-up blocks."""
    minx, miny, maxx, maxy = sar_bounds
    # Centre the crop in the short (north-south) direction.
    mid_y = 0.5 * (miny + maxy)
    y0 = max(miny, mid_y - CROP_M / 2)
    y1 = min(maxy, y0 + CROP_M)
    x_start = minx + 40
    x_stop = maxx - CROP_M - 40
    step = CROP_M * 0.55
    scored = []
    x = x_start
    while x < x_stop:
        win = box(x, y0, x + CROP_M, y1)
        subset = buildings[buildings.intersects(win)]
        if len(subset) >= 8:
            scored.append(
                {
                    "bounds": (x, y0, x + CROP_M, y1),
                    "count": int(len(subset)),
                    "max_h": float(subset[HEIGHT_COL].max()),
                    "median_h": float(subset[HEIGHT_COL].median()),
                }
            )
        x += step
    if not scored:
        # Fallback: a single window at the strip centre.
        cx = 0.5 * (minx + maxx)
        scored = [
            {
                "bounds": (cx - CROP_M / 2, y0, cx + CROP_M / 2, y1),
                "count": int(len(buildings)),
                "max_h": float(buildings[HEIGHT_COL].max()) if len(buildings) else 0,
                "median_h": float(buildings[HEIGHT_COL].median()) if len(buildings) else 0,
            }
        ]

    # Prefer a mix of dense and tall, spread along the strip.
    scored.sort(key=lambda s: (s["count"], s["max_h"]), reverse=True)
    chosen = []
    for cand in scored:
        if any(abs(cand["bounds"][0] - c["bounds"][0]) < CROP_M * 0.8 for c in chosen):
            continue
        chosen.append(cand)
        if len(chosen) >= n:
            break
    # Keep left-to-right order for the viewer.
    chosen.sort(key=lambda s: s["bounds"][0])
    labels = ["West block", "Mid-west block", "Mid-east block", "East block"]
    for i, c in enumerate(chosen):
        c["id"] = f"crop-{chr(ord('a') + i)}"
        c["title"] = labels[i] if i < len(labels) else f"Block {i + 1}"
    return chosen


def window_from_bounds_clamp(bounds, src) -> Window:
    win = from_bounds(*bounds, transform=src.transform)
    win = win.round_offsets().round_lengths()
    row_off = max(0, int(win.row_off))
    col_off = max(0, int(win.col_off))
    height = min(int(win.height), src.height - row_off)
    width = min(int(win.width), src.width - col_off)
    return Window(col_off, row_off, width, height)


def render_crop(
    src: rasterio.DatasetReader,
    buildings: gpd.GeoDataFrame,
    crop: dict,
    incidence_deg: float,
    look_azimuth_deg: float,
    out_path: Path,
    title: str,
    write_sar_only: Path | None = None,
) -> dict:
    win = window_from_bounds_clamp(crop["bounds"], src)
    transform = src.window_transform(win)
    band = src.read(1, window=win)
    sar_u8 = percentile_stretch(band)
    if write_sar_only is not None:
        Image.fromarray(sar_u8, mode="L").convert("RGB").save(write_sar_only, "PNG", optimize=True)

    win_box = box(*rasterio.windows.bounds(win, src.transform))
    local = buildings[buildings.intersects(win_box)].copy()
    layover, shadow = building_regions(local, incidence_deg, look_azimuth_deg)
    mask = rasterise_mask(layover, shadow, transform, sar_u8.shape)
    overlay_png(
        sar_u8,
        mask,
        local,
        transform,
        out_path,
        look_azimuth_deg,
        incidence_deg,
        title,
    )
    return {
        "window": [int(win.col_off), int(win.row_off), int(win.width), int(win.height)],
        "layover_px": int((mask == 1).sum()),
        "shadow_px": int((mask == 2).sum()),
        "buildings": int(len(local)),
    }


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
    gpkg_dir = Path(args.buildings)

    if not sar_path.exists():
        raise SystemExit(f"SAR strip not found: {sar_path}")
    orientations = parse_orientations(ori_path)
    if STRIP_KEY not in orientations:
        raise SystemExit(f"No orientation flag for strip {STRIP_KEY}")
    ori_flag = orientations[STRIP_KEY]
    recorded_azimuth = look_azimuth_from_orientation(ori_flag)
    look_azimuth = args.look_azimuth if args.look_azimuth is not None else recorded_azimuth
    if args.flip_look:
        look_azimuth = (look_azimuth + 180.0) % 360.0

    with rasterio.open(sar_path) as src:
        print_sar_report(src, sar_path)
        meta_inc = extract_metadata_incidence(src)
        log(f"  orientation flag {ori_flag} ({'south' if ori_flag == 1 else 'north'}-facing)")
        log(f"  recorded look azimuth {recorded_azimuth:.1f} deg")
        log(f"  using look azimuth    {look_azimuth:.1f} deg")
        log(f"  incidence (CLI)       {args.incidence:.2f} deg")
        if meta_inc is not None:
            log(f"  incidence (metadata)  {meta_inc:.2f} deg  [Capella center_pixel]")

        log("=== loading 3DBAG ===")
        raw = load_footprints(gpkg_dir)
        log(f"  raw CRS {raw.crs}  features {len(raw)}")
        ensure_projected_crs(raw.crs)
        rd = reproject_to_rd(raw)
        rd = add_building_height(rd)
        print_footprint_report(rd, f"EPSG:28992, all loaded tiles")

        buildings = rd.to_crs(src.crs)
        ensure_projected_crs(buildings.crs)
        strip_box = box(*src.bounds)
        # Keep buildings that can throw layover/shadow into the strip.
        max_h = float(buildings[HEIGHT_COL].max()) if len(buildings) else 0.0
        buffer_m = max(
            max_h / math.tan(math.radians(min(args.incidence, meta_inc or args.incidence))),
            max_h * math.tan(math.radians(max(args.incidence, meta_inc or args.incidence))),
        ) + 20.0
        buildings = buildings[buildings.intersects(strip_box.buffer(buffer_m))].copy()
        print_footprint_report(buildings, f"clipped to SAR CRS {src.crs}")

        log("=== rasterising primary mask ===")
        layover, shadow = building_regions(buildings, args.incidence, look_azimuth)
        mask = rasterise_mask(layover, shadow, src.transform, (src.height, src.width))
        mask_path = OUTPUT / "layover_shadow_mask.tif"
        write_mask_geotiff(mask_path, mask, src)

        log("=== choosing inspection crops ===")
        crops = pick_crops(buildings[buildings.intersects(strip_box)], src.bounds, n=4)
        for c in crops:
            log(
                f"  {c['id']}  {c['title']}  buildings={c['count']}  "
                f"max_h={c['max_h']:.1f}m  bounds={tuple(round(v, 1) for v in c['bounds'])}"
            )

        OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
        flipped_az = (look_azimuth + 180.0) % 360.0
        inc_meta = meta_inc if meta_inc is not None else args.incidence

        crop_records = []
        for crop in crops:
            rec = {
                "id": crop["id"],
                "title": crop["title"],
                "bounds": [round(v, 3) for v in crop["bounds"]],
                "building_count": crop["count"],
                "max_height_m": round(crop["max_h"], 2),
                "median_height_m": round(crop["median_h"], 2),
                "files": {},
            }
            log(f"=== overlay {crop['id']} ===")
            stats = render_crop(
                src,
                buildings,
                crop,
                args.incidence,
                look_azimuth,
                OVERLAY_DIR / f"{crop['id']}.png",
                title="recorded",
                write_sar_only=OVERLAY_DIR / f"{crop['id']}-sar.png",
            )
            rec["files"]["overlay"] = f"/overlays/{crop['id']}.png"
            rec["files"]["sar"] = f"/overlays/{crop['id']}-sar.png"
            rec["stats"] = stats

            render_crop(
                src,
                buildings,
                crop,
                args.incidence,
                flipped_az,
                OVERLAY_DIR / f"{crop['id']}-flipped.png",
                title="flipped +180",
            )
            rec["files"]["flipped"] = f"/overlays/{crop['id']}-flipped.png"

            if meta_inc is not None and abs(meta_inc - args.incidence) > 0.5:
                render_crop(
                    src,
                    buildings,
                    crop,
                    inc_meta,
                    look_azimuth,
                    OVERLAY_DIR / f"{crop['id']}-inc-meta.png",
                    title="metadata θ",
                )
                rec["files"]["inc_meta"] = f"/overlays/{crop['id']}-inc-meta.png"
            crop_records.append(rec)

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
            "footprints": {
                "source": "3DBAG v20250903 lod12_2d + pand",
                "crs_loaded": str(raw.crs),
                "crs_mandatory": RD_CRS,
                "crs_matched_to_sar": str(src.crs),
                "feature_count_used": int(len(buildings)),
                "height_column": f"{HEIGHT_COL} = b3_h_50p - b3_h_maaiveld",
                "height_min_m": round(float(buildings[HEIGHT_COL].min()), 2) if len(buildings) else None,
                "height_max_m": round(float(buildings[HEIGHT_COL].max()), 2) if len(buildings) else None,
                "height_median_m": round(float(buildings[HEIGHT_COL].median()), 2) if len(buildings) else None,
            },
            "mask": {
                "path": "sar/output/layover_shadow_mask.tif",
                "codes": {"0": "nominal", "1": "layover", "2": "shadow"},
                "layover_px": int((mask == 1).sum()),
                "shadow_px": int((mask == 2).sum()),
            },
            "how_to_flip": {
                "cli": "python pipeline.py --flip-look",
                "or": "python pipeline.py --look-azimuth 0   # this strip is south-facing (180)",
                "note": (
                    "If layover sits on the far side of each building instead of toward "
                    "the sensor, the look azimuth is 180° off. Toggle Flip look in the "
                    "viewer or rerun with --flip-look."
                ),
            },
            "crops": crop_records,
        }
        OUTPUT.mkdir(parents=True, exist_ok=True)
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
    p.add_argument("--buildings", default=str(DATA / "3dbag"))
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
