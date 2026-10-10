#!/usr/bin/env python3
"""Stage a Capella SLC for SNAP SAR-Simulation.

SNAP's Capella reader treats a file whose name contains SLC as radar
geometry. SAR-Simulation then needs a real slant-range axis. SpaceNet 6
strips are often image_geometry type "surface": complex samples already on
a map grid, with no range_to_first_sample. This script refuses those.

The attached original was not in the workspace. The required fields below
are the ones CapellaProductDirectory in SNAP 14.0.0 actually reads.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

# Fields SAR-Simulation cannot invent. pixel_spacing_* is not a substitute:
# on a surface product it is map spacing, and SNAP's fallback slant range is
# the 99999 m missing-metadata sentinel.
REQUIRED = (
    "collect.image.image_geometry.range_to_first_sample",
    "collect.image.image_geometry.delta_range_sample",
    "collect.image.image_geometry.delta_line_time",
    "collect.image.image_geometry.first_line_time",
    "collect.state.direction",
    "terrain_models",
)


def _dig(obj: dict, path: str):
    cur = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _extended_json(tif: Path) -> Path:
    side = tif.with_name(tif.stem + "_extended.json")
    if side.is_file():
        return side
    raise SystemExit(f"no extended json next to {tif}")


def _missing(meta: dict) -> list[str]:
    missing = [path for path in REQUIRED if _dig(meta, path) is None]
    geom = _dig(meta, "collect.image.image_geometry.type")
    if geom == "surface":
        missing.append(
            "collect.image.image_geometry.type is 'surface' "
            "(map grid, not a slant-range axis)"
        )
    return missing


def _flat_dem(path: Path, magpol: Path, height: float, margin_m: float = 200.0) -> None:
    with rasterio.open(magpol) as src:
        left, bottom, right, top = src.bounds
        crs = src.crs
    step = 10.0
    west, south = left - margin_m, bottom - margin_m
    east, north = right + margin_m, top + margin_m
    width = int(np.ceil((east - west) / step))
    height_px = int(np.ceil((north - south) / step))
    transform = from_origin(west, north, step, step)
    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height_px,
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": -9999.0,
        "compress": "deflate",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.full((1, height_px, width), height, dtype=np.float32))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tif", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--magpol", type=Path, required=True)
    parser.add_argument(
        "--flat-height",
        type=float,
        default=None,
        help="Ellipsoidal height of flat_surface_dem.tif. "
        "Default is the centre-target height.",
    )
    parser.add_argument("--lines", type=int, default=4000)
    args = parser.parse_args()

    meta = json.loads(_extended_json(args.tif).read_text())
    geom = _dig(meta, "collect.image.image_geometry") or {}
    terrain = meta.get("terrain_models", _dig(meta, "collect.terrain_models"))
    print(f"image_geometry.type {geom.get('type', 'ABSENT')}")
    print(f"image_geometry.surface_type {geom.get('surface_type', 'ABSENT')}")
    print(f"terrain_models {json.dumps(terrain) if terrain is not None else 'ABSENT'}")
    if geom.get("z"):
        print(f"image_geometry.z {json.dumps(geom['z'])}")

    missing = _missing(meta)
    if missing:
        print("NOT READY")
        print("missing fields:")
        for item in missing:
            print(f"  - {item}")
        sys.exit(2)

    target = _dig(meta, "collect.image.center_pixel.target_position")
    # Centre-target height is not stored separately; the caller passes it.
    height = args.flat_height
    if height is None:
        raise SystemExit("READY but --flat-height is required (centre-target height)")

    with rasterio.open(args.tif) as src:
        rows, cols = src.height, src.width
    y0 = max(0, (rows - args.lines) // 2)
    lines = min(args.lines, rows - y0)
    region = f"0,{y0},{cols},{lines}"

    with rasterio.open(args.magpol) as src:
        transform = src.transform
        crs = src.crs.to_wkt()
        spacing = abs(transform.a) if abs(transform.a) >= abs(transform.b) else abs(transform.e)
        # Axis-aligned MAG-POL: a is column spacing, origin is the upper-left corner.
        origin_x = transform.c
        origin_y = transform.f

    args.out.mkdir(parents=True, exist_ok=True)
    _flat_dem(args.out / "flat_surface_dem.tif", args.magpol, height)
    print(f"region {region}")
    print(f"crs {crs}")
    print(f"pixelSpacing {spacing}")
    print(f"originX {origin_x}")
    print(f"originY {origin_y}")
    print(f"flat_height {height}")
    print(f"READY {args.out}")


if __name__ == "__main__":
    main()
