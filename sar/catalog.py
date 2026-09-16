#!/usr/bin/env python3
"""Build the per-strip geometry catalog. No SAR pixels are read."""
from __future__ import annotations
import csv, json, os
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

def open_strip(strip_id: str):
    local = DATA / f"SN6_AOI_11_Rotterdam_SAR-MAG-POL_{strip_id}.tif"
    if local.exists():
        return rasterio.open(local)
    if strip_id == LOCAL_KEY and LOCAL_STRIP.exists():
        return rasterio.open(LOCAL_STRIP)
    _gdal_env()
    return rasterio.open(f"{VSIS3}/SN6_AOI_11_Rotterdam_SAR-MAG-POL_{strip_id}.tif")

def catalog_one(strip_id: str, ori_flag: int):
    if ori_flag not in (0, 1):
        raise SystemExit(f"FAIL {strip_id}: orientation flag missing or invalid ({ori_flag!r})")
    try:
        src = open_strip(strip_id)
    except Exception as exc:
        print(f"SKIP {strip_id}: no MAG-POL on S3 ({exc})")
        return None
    with src:
        inc = incidence_from_tags(src.tags())
        look = look_azimuth_from_orientation(ori_flag)
        print(f"{strip_id}  ori={ori_flag} look={look:.0f} θ={inc['theta_from_vertical_deg']:.4f}°  {src.width}x{src.height}")
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

def write_catalog(rows, path: Path) -> None:
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
    print(f"=== incidence field: {INCIDENCE_FIELD} (from vertical) ===")
    rows, skipped = [], []
    for strip_id, flag in orientations.items():
        row = catalog_one(strip_id, flag)
        if row is None:
            skipped.append(strip_id)
            continue
        rows.append(row)
    if not rows:
        raise SystemExit("FAIL: no strips catalogued")
    write_catalog(rows, OUTPUT / "strip_catalog.csv")
    print(f"=== catalog {len(rows)} strips  skipped={len(skipped)} ===")

if __name__ == "__main__":
    main()
