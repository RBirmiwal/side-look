#!/usr/bin/env python3
"""Download SpaceNet 6 MAG-POL strips, 3DBAG tiles, and the AHN DSM.

Public S3, no credentials:
    aws s3 ls s3://spacenet-dataset/AOIs/AOI_11_Rotterdam/ --no-sign-request

AHN DSM is the Dutch national lidar surface model (buildings included), via
PDOK WCS, no account. DTM is intentionally not downloaded.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
BAG_DIR = DATA / "3dbag"
OUTPUT = ROOT / "output"

STRIP_KEY = "20190804111224_20190804111453"
STRIP_S3 = (
    "s3://spacenet-dataset/AOIs/AOI_11_Rotterdam/SAR-MAG-POL/"
    f"SN6_AOI_11_Rotterdam_SAR-MAG-POL_{STRIP_KEY}.tif"
)
ORI_S3 = "s3://spacenet-dataset/AOIs/AOI_11_Rotterdam/SummaryData/SAR_orientations.txt"
README_S3 = "s3://spacenet-dataset/AOIs/AOI_11_Rotterdam/SummaryData/README.txt"
MAGPOL_S3 = "s3://spacenet-dataset/AOIs/AOI_11_Rotterdam/SAR-MAG-POL"

AHN_WCS = "https://service.pdok.nl/rws/ahn/wcs/v1_0"
AHN_USER_AGENT = "sar-dsm-pipeline/1.0"
AHN_BUFFER_M = 150.0
AHN_POSTING_M = 1.0
AHN_CHUNK_M = 2200.0

# 3DBAG v20250903 tiles intersecting this strip's eastern/central built-up reach.
BAG_TILES = [
    "https://data.3dbag.nl/v20250903/tiles/8/296/504/8-296-504.gpkg.gz",
    "https://data.3dbag.nl/v20250903/tiles/8/304/504/8-304-504.gpkg.gz",
    "https://data.3dbag.nl/v20250903/tiles/9/312/504/9-312-504.gpkg.gz",
    "https://data.3dbag.nl/v20250903/tiles/9/316/504/9-316-504.gpkg.gz",
    "https://data.3dbag.nl/v20250903/tiles/9/316/508/9-316-508.gpkg.gz",
    "https://data.3dbag.nl/v20250903/tiles/9/320/504/9-320-504.gpkg.gz",
]


def s3_cp(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"exists {dest}")
        return
    cmd = ["aws", "s3", "cp", url, str(dest), "--no-sign-request"]
    print(" ".join(cmd))
    subprocess.check_call(cmd)


def magpol_s3_url(strip_id: str) -> str:
    return f"{MAGPOL_S3}/SN6_AOI_11_Rotterdam_SAR-MAG-POL_{strip_id}.tif"


def magpol_local_path(strip_id: str) -> Path:
    if strip_id == STRIP_KEY:
        return DATA / "sar_strip.tif"
    return DATA / f"SN6_AOI_11_Rotterdam_SAR-MAG-POL_{strip_id}.tif"


def download_strip(strip_id: str, dest: Path | None = None) -> Path:
    dest = dest or magpol_local_path(strip_id)
    s3_cp(magpol_s3_url(strip_id), dest)
    return dest


def fetch(url: str, dest: Path, timeout: int = 120) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"exists {dest}")
        return
    print(f"GET {url}")
    req = urllib.request.Request(url, headers={"User-Agent": AHN_USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as src, dest.open("wb") as out:
        shutil.copyfileobj(src, out)


def gunzip(src: Path, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"exists {dest}")
        return
    with gzip.open(src, "rb") as fh, dest.open("wb") as out:
        shutil.copyfileobj(fh, out)
    print(f"unzip {dest}")


def _rd_bbox_from_sar(sar_path: Path) -> tuple[float, float, float, float]:
    import rasterio
    from pyproj import Transformer

    with rasterio.open(sar_path) as src:
        b = src.bounds
        transformer = Transformer.from_crs(src.crs, "EPSG:28992", always_xy=True)
        corners = [
            transformer.transform(b.left, b.bottom),
            transformer.transform(b.right, b.bottom),
            transformer.transform(b.right, b.top),
            transformer.transform(b.left, b.top),
        ]
    xs, ys = zip(*corners)
    return (
        min(xs) - AHN_BUFFER_M,
        min(ys) - AHN_BUFFER_M,
        max(xs) + AHN_BUFFER_M,
        max(ys) + AHN_BUFFER_M,
    )


def _rd_bbox_from_utm(bounds: tuple[float, float, float, float], crs: str = "EPSG:32631"):
    from pyproj import Transformer

    west, south, east, north = bounds
    transformer = Transformer.from_crs(crs, "EPSG:28992", always_xy=True)
    corners = [
        transformer.transform(west, south),
        transformer.transform(east, south),
        transformer.transform(east, north),
        transformer.transform(west, north),
    ]
    xs, ys = zip(*corners)
    return (
        min(xs) - AHN_BUFFER_M,
        min(ys) - AHN_BUFFER_M,
        max(xs) + AHN_BUFFER_M,
        max(ys) + AHN_BUFFER_M,
    )


def _wcs_chunk(bbox: tuple[float, float, float, float], dest: Path) -> None:
    xmin, ymin, xmax, ymax = bbox
    width = int(round((xmax - xmin) / AHN_POSTING_M))
    height = int(round((ymax - ymin) / AHN_POSTING_M))
    xmax = xmin + width * AHN_POSTING_M
    ymax = ymin + height * AHN_POSTING_M
    url = (
        f"{AHN_WCS}?SERVICE=WCS&VERSION=1.0.0&REQUEST=GetCoverage"
        f"&COVERAGE=dsm_05m&CRS=EPSG:28992"
        f"&BBOX={xmin},{ymin},{xmax},{ymax}"
        f"&WIDTH={width}&HEIGHT={height}&FORMAT=image/tiff"
    )
    print(f"WCS DSM {width}x{height}  bbox=({xmin:.1f},{ymin:.1f},{xmax:.1f},{ymax:.1f})")
    req = urllib.request.Request(url, headers={"User-Agent": AHN_USER_AGENT})
    with urllib.request.urlopen(req, timeout=180) as src:
        data = src.read()
        ctype = src.headers.get("Content-Type", "")
    if b"<Exception" in data[:400] or b"<ServiceException" in data[:400] or "xml" in ctype.lower():
        raise RuntimeError(f"AHN WCS error: {data[:500]!r}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)


def download_ahn_dsm_bbox(dest: Path, xmin: float, ymin: float, xmax: float, ymax: float) -> None:
    """DSM (surface, includes buildings), 1 m posting, EPSG:28992."""
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"exists {dest}")
        return

    xmin = math.floor(xmin)
    ymin = math.floor(ymin)
    xmax = math.ceil(xmax)
    ymax = math.ceil(ymax)
    print(f"AHN DSM RD bbox {xmin},{ymin},{xmax},{ymax}")

    chunk_paths = []
    with tempfile.TemporaryDirectory(prefix="ahn_wcs_") as tmp:
        tmp_dir = Path(tmp)
        idx = 0
        x = xmin
        while x < xmax:
            x1 = min(x + AHN_CHUNK_M, xmax)
            y = ymin
            while y < ymax:
                y1 = min(y + AHN_CHUNK_M, ymax)
                chunk = tmp_dir / f"chunk_{idx:03d}.tif"
                _wcs_chunk((x, y, x1, y1), chunk)
                chunk_paths.append(chunk)
                y = y1
                idx += 1
            x = x1

        import numpy as np
        import rasterio
        from rasterio.transform import from_origin
        from rasterio.warp import Resampling, reproject

        width = int(round((xmax - xmin) / AHN_POSTING_M))
        height = int(round((ymax - ymin) / AHN_POSTING_M))
        transform = from_origin(xmin, ymax, AHN_POSTING_M, AHN_POSTING_M)
        mosaic = np.full((height, width), np.nan, dtype=np.float32)
        crs = None
        for path in chunk_paths:
            with rasterio.open(path) as src:
                crs = src.crs
                data = src.read(1).astype(np.float32)
                valid = np.isfinite(data) & (np.abs(data) < 1.0e6)
                if src.nodata is not None and np.isfinite(src.nodata):
                    valid &= data != src.nodata
                data = np.where(valid, data, np.nan)
                piece = np.full_like(mosaic, np.nan)
                reproject(
                    source=data,
                    destination=piece,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=np.nan,
                    dst_transform=transform,
                    dst_crs=src.crs,
                    dst_nodata=np.nan,
                    resampling=Resampling.nearest,
                )
                take = np.isfinite(piece)
                mosaic[take] = piece[take]

        dest.parent.mkdir(parents=True, exist_ok=True)
        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "count": 1,
            "height": height,
            "width": width,
            "crs": crs,
            "transform": transform,
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "nodata": None,
        }
        with rasterio.open(dest, "w", **profile) as dst:
            dst.write(mosaic, 1)
        finite = mosaic[np.isfinite(mosaic)]
        print(
            f"wrote {dest}  shape={mosaic.shape}  "
            f"valid={finite.size}/{mosaic.size}  "
            f"z={float(finite.min()):.2f}..{float(finite.max()):.2f} m"
        )


def download_ahn_dsm(dest: Path, sar_path: Path) -> None:
    xmin, ymin, xmax, ymax = _rd_bbox_from_sar(sar_path)
    download_ahn_dsm_bbox(dest, xmin, ymin, xmax, ymax)


def download_ahn_aoi(dest: Path | None = None) -> Path:
    """AHN covering the union of all catalogued MAG-POL strips."""
    dest = dest or (DATA / "ahn_dsm_1m_aoi.tif")
    catalog = OUTPUT / "strip_catalog.json"
    if not catalog.exists():
        raise SystemExit(f"catalog missing: {catalog}  (python catalog.py)")
    rows = json.loads(catalog.read_text())
    xs = [b for r in rows for b in (r["bounds"][0], r["bounds"][2])]
    ys = [b for r in rows for b in (r["bounds"][1], r["bounds"][3])]
    xmin, ymin, xmax, ymax = _rd_bbox_from_utm((min(xs), min(ys), max(xs), max(ys)), rows[0]["crs"])
    download_ahn_dsm_bbox(dest, xmin, ymin, xmax, ymax)
    return dest


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    s3_cp(ORI_S3, DATA / "SAR_orientations.txt")
    s3_cp(README_S3, DATA / "README_summary.txt")
    s3_cp(STRIP_S3, DATA / "sar_strip.tif")
    BAG_DIR.mkdir(parents=True, exist_ok=True)
    for url in BAG_TILES:
        gz = BAG_DIR / Path(url).name
        fetch(url, gz)
        gunzip(gz, BAG_DIR / gz.name.replace(".gz", ""))
    download_ahn_dsm(DATA / "ahn_dsm_1m.tif", DATA / "sar_strip.tif")
    print("download complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ahn-aoi", action="store_true", help="AHN DSM covering all 202 strip footprints.")
    parser.add_argument("--strip", action="append", default=[], help="Download extra MAG-POL strip id(s).")
    parser.add_argument("--all-local", action="store_true", help="Original one-strip download (default if no flags).")
    args = parser.parse_args()
    if args.ahn_aoi or args.strip:
        if args.ahn_aoi:
            download_ahn_aoi()
        for strip_id in args.strip:
            download_strip(strip_id)
    else:
        main()
