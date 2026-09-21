#!/usr/bin/env python3
"""Build the per-strip geometry catalog. No SAR pixels are read.

Opens each MAG-POL GeoTIFF for metadata only (CRS, transform, shape, Capella
JSON). Joins the SpaceNet 6 orientation flag. Fails if incidence or look
azimuth is missing — those two numbers drive every mask.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import rasterio

from geometry import look_azimuth_from_orientation
from incidence import INCIDENCE_FIELD, incidence_from_tags
from pipeline import parse_orientations

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUT = ROOT / "output"
S3_PREFIX = "spacenet-dataset/AOIs/AOI_11_Rotterdam/SAR-MAG-POL"
VSIS3 = f"/vsis3/{S3_PREFIX}"
LOCAL_STRIP = DATA / "sar_strip.tif"
LOCAL_KEY = "20190804111224_20190804111453"


def _gdal_env() -> None:
    os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")


def strip_id_from_name(name: str) -> str:
    stem = Path(name).stem
    marker = "SAR-MAG-POL_"
    if marker not in stem:
        raise ValueError(f"cannot parse strip id from {name}")
    return stem.split(marker, 1)[1]


def list_strip_keys(orientations: dict[str, int]) -> list[str]:
    return sorted(orientations.keys())


def open_strip(strip_id: str):
    local = DATA / f"SN6_AOI_11_Rotterdam_SAR-MAG-POL_{strip_id}.tif"
    if local.exists():
        return rasterio.open(local)
    if strip_id == LOCAL_KEY and LOCAL_STRIP.exists():
        return rasterio.open(LOCAL_STRIP)
    _gdal_env()
    return rasterio.open(f"{VSIS3}/SN6_AOI_11_Rotterdam_SAR-MAG-POL_{strip_id}.tif")


def catalog_one(strip_id: str, ori_flag: int) -> dict | None:
    if ori_flag not in (0, 1):
        raise SystemExit(f"FAIL {strip_id}: orientation flag missing or invalid ({ori_flag!r})")
    try:
        src = open_strip(strip_id)
    except Exception as exc:
        print(f"SKIP {strip_id}: no MAG-POL on S3 ({exc})")
        return None
    with src:
        try:
            inc = incidence_from_tags(src.tags())
        except Exception as exc:
            raise SystemExit(f"FAIL {strip_id}: incidence missing ({exc})") from exc
        look = look_azimuth_from_orientation(ori_flag)
        print(
            f"{strip_id}  ori={ori_flag} ({'south' if ori_flag == 1 else 'north'})  "
            f"look={look:.0f}°  field={inc['field']}  "
            f"raw={inc['raw_value_deg']:.4f}°  "
            f"theta_vertical={inc['theta_from_vertical_deg']:.4f}°  "
            f"90-raw={inc['ninety_minus_raw_deg']:.4f}°  "
            f"[{inc['convention']}]  "
            f"{src.width}x{src.height}"
        )
        return {
            "strip_id": strip_id,
            "orientation_flag": ori_flag,
            "orientation_label": "south-facing" if ori_flag == 1 else "north-facing",
            "look_azimuth_deg": look,
            "incidence_field": inc["field"],
            "incidence_raw_deg": inc["raw_value_deg"],
            "incidence_deg": inc["theta_from_vertical_deg"],
            "incidence_90_minus_raw_deg": inc["ninety_minus_raw_deg"],
            "incidence_convention": inc["convention"],
            "crs": str(src.crs),
            "width": src.width,
            "height": src.height,
            "transform": list(src.transform)[:6],
            "bounds": [src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top],
            "count": src.count,
            "dtype": str(src.dtypes[0]),
            "nodata": src.nodata,
        }


def write_catalog(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            for key in ("transform", "bounds"):
                if isinstance(flat.get(key), list):
                    flat[key] = json.dumps(flat[key])
            writer.writerow(flat)
    path.with_suffix(".json").write_text(json.dumps(rows, indent=2))


def main() -> None:
    ori_path = DATA / "SAR_orientations.txt"
    if not ori_path.exists():
        raise SystemExit(f"FAIL: {ori_path} missing")
    orientations = parse_orientations(ori_path)
    print(f"=== incidence field: {INCIDENCE_FIELD} ===")
    print("measured from vertical; 90-raw is printed so a grazing mix-up is obvious")
    rows = []
    skipped = []
    for strip_id, flag in orientations.items():
        row = catalog_one(strip_id, flag)
        if row is None:
            skipped.append(strip_id)
            continue
        rows.append(row)
    if not rows:
        raise SystemExit("FAIL: no strips catalogued")
    out = OUTPUT / "strip_catalog.csv"
    write_catalog(rows, out)
    north = sum(1 for r in rows if r["orientation_flag"] == 0)
    south = sum(1 for r in rows if r["orientation_flag"] == 1)
    xs = [b for r in rows for b in (r["bounds"][0], r["bounds"][2])]
    ys = [b for r in rows for b in (r["bounds"][1], r["bounds"][3])]
    print(f"=== catalog {len(rows)} strips  north={north} south={south}  skipped={len(skipped)} ===")
    if skipped:
        print("  skipped (orientation row, no MAG-POL): " + ", ".join(skipped))
    print(f"  union bounds  {min(xs):.1f},{min(ys):.1f}  {max(xs):.1f},{max(ys):.1f}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
