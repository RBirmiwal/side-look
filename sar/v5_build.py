#!/usr/bin/env python3
"""
Rebuild dataset v5 from public inputs.

v4 scene: AHN4 DSM/DTM (PDOK WCS dsm_05m / dtm_05m) and 3DBAG v20250903
footprints. Rule B (own_range_grid) on the buildings-only surface with the
AHN4 DTM as the ground. v5 then sets footprints dilated 2 m to 255 and stores
layover magnitude in metres.

Ground range is not one look angle. It is H * tan(incidence) from the v2
tiles, which already store the per-pixel off-nadir of the fitted flight line.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import shutil
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from scipy.ndimage import binary_dilation, distance_transform_edt, find_objects, label, map_coordinates
from shapely import STRtree, box, wkb
from shapely.ops import transform as shp_transform

sys.path.insert(0, str(Path(__file__).resolve().parent))
from image_space import (  # noqa: E402
    labels_from_coverage,
    layover_magnitude,
    own_range_displacement_grid,
    own_range_grid,
)

V2 = Path("/tmp/v2tiles")
ROOT = Path("/tmp/side-look")
OUT = ROOT / "sar" / "output" / "v5"
PACK = ROOT / "v5_masks"
TILES = ROOT / "side-look-tiles-v5"
WORK = Path("/tmp/v5_work")
BAG_DIR = Path("/tmp/bag")
AHN_WCS = "https://service.pdok.nl/rws/ahn/wcs/v1_0"
BAG_URL = "https://data.3dbag.nl/v20250903/tiles/9/{ix}/{iy}/9-{ix}-{iy}.gpkg.gz"
UA = "sar-dsm-pipeline/1.0"

PX = 0.25
RES = 0.5                 # AHN native posting, also the warp grid
RANGE_PAD_M = 250.0
NAN_MARGIN = 400          # cells at 0.25 m = 100 m, same count as v4
HOLE_M2 = 100.0
WATER_M2 = 2500.0
LABEL_VERSION = "v5-agreed-mag"
BLOCK = 512
HALO = 16                 # > 8 px dilation
K_ID = "20190823072627_20190823072925"

# 3DBAG level-9 tile names. Index step 4 covers ~1 km.
# Fitted from gpkg contents of the Rotterdam tiles already on disk:
#   easting ~= 13600 + ix * 250,  northing ~= 306900 + iy * 250.
BAG_OX, BAG_OY, BAG_CELL = 13600.0, 306900.0, 250.0

GEOM = {
    "20190822074237_20190822074521": dict(H=1614.0, look=180.180, near=24.42, centre=34.65, far=42.84, flip=False),
    "20190823072627_20190823072925": dict(H=1619.0, look=0.457, near=24.99, centre=34.64, far=42.44, flip=True),
    "20190804111224_20190804111453": dict(H=1617.2, look=359.620, near=25.21, centre=34.21, far=41.62, flip=True),
    "20190804143846_20190804144102": dict(H=1627.6, look=180.077, near=24.67, centre=34.24, far=42.02, flip=False),
    "20190822071025_20190822071321": dict(H=1621.6, look=0.401, near=25.00, centre=34.59, far=42.34, flip=True),
    "20190822084226_20190822084526": dict(H=1619.4, look=180.149, near=24.42, centre=34.59, far=42.79, flip=False),
    "20190822133333_20190822133635": dict(H=1615.4, look=0.468, near=25.00, centre=34.63, far=42.44, flip=True),
    "20190822083256_20190822083600": dict(H=1627.8, look=0.398, near=24.96, centre=34.64, far=42.58, flip=True),
    "20190822085151_20190822085445": dict(H=1620.5, look=180.197, near=24.35, centre=34.60, far=42.86, flip=False),
    "20190822134328_20190822134645": dict(H=1616.8, look=0.430, near=25.00, centre=34.60, far=42.40, flip=True),
    "20190822162246_20190822162529": dict(H=1620.7, look=180.186, near=24.48, centre=34.59, far=42.75, flip=False),
    "20190823065021_20190823065333": dict(H=1611.5, look=0.460, near=24.85, centre=34.63, far=42.54, flip=True),
    "20190823083119_20190823083425": dict(H=1790.5, look=180.158, near=24.58, centre=34.71, far=42.81, flip=False),
    "20190823085011_20190823085317": dict(H=1783.1, look=180.109, near=24.49, centre=34.67, far=42.86, flip=False),
    "20190823125228_20190823125530": dict(H=1787.2, look=0.431, near=25.13, centre=34.65, far=42.39, flip=True),
}
STRIPS = list(GEOM)

# V4.md / REPORT_v4.md, v2-tile gate (near / middle layover-nominal dB) and full-strip ignored.
V4_REF = {
    "20190822074237_20190822074521": dict(ignored=0.370, near=1.72, middle=1.34, tiles=212, kept=188, drop=24),
    "20190823072627_20190823072925": dict(ignored=0.292, near=2.90, middle=3.32, tiles=51, kept=44, drop=7),
    "20190804111224_20190804111453": dict(ignored=0.333, near=1.20, middle=2.54, tiles=197, kept=181, drop=16),
    "20190804143846_20190804144102": dict(ignored=0.376, near=1.92, middle=1.84, tiles=183, kept=156, drop=27),
    "20190822071025_20190822071321": dict(ignored=0.214, near=2.74, middle=3.88, tiles=75, kept=67, drop=8),
    "20190822084226_20190822084526": dict(ignored=0.477, near=4.82, middle=2.04, tiles=447, kept=380, drop=67),
    "20190822133333_20190822133635": dict(ignored=0.356, near=1.38, middle=2.18, tiles=264, kept=233, drop=31),
    "20190822083256_20190822083600": dict(ignored=0.199, near=1.52, middle=3.42, tiles=81, kept=79, drop=2),
    "20190822085151_20190822085445": dict(ignored=0.425, near=1.06, middle=1.62, tiles=310, kept=261, drop=49),
    "20190822134328_20190822134645": dict(ignored=0.336, near=1.48, middle=1.44, tiles=231, kept=197, drop=34),
    "20190822162246_20190822162529": dict(ignored=0.472, near=6.06, middle=3.34, tiles=443, kept=378, drop=65),
    "20190823065021_20190823065333": dict(ignored=0.376, near=1.56, middle=2.52, tiles=271, kept=239, drop=32),
    "20190823083119_20190823083425": dict(ignored=0.490, near=3.14, middle=10.16, tiles=421, kept=357, drop=64),
    "20190823085011_20190823085317": dict(ignored=0.381, near=1.94, middle=1.92, tiles=251, kept=221, drop=30),
    "20190823125228_20190823125530": dict(ignored=0.206, near=2.42, middle=3.84, tiles=66, kept=61, drop=5),
}


def log(msg: str) -> None:
    rss = 0
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS:"):
                rss = int(line.split()[1]) // 1024
    except OSError:
        pass
    print(f"[{time.strftime('%H:%M:%S')}] {msg}  (rss {rss} MB)", flush=True)


def strip_grid(strip: str) -> dict:
    import re
    cat = (ROOT / "sar" / "output" / "strip_catalog.csv").read_text().splitlines()
    for line in cat[1:]:
        if not line.startswith(strip + ","):
            continue
        m = re.search(r"\[([^\]]+)\]", line)
        nums = [float(x) for x in m.group(1).split(",")]
        wh = re.search(r"EPSG:32631,(\d+),(\d+),", line)
        width, height = int(wh.group(1)), int(wh.group(2))
        a, b, c, d, e, f = nums
        return dict(width=width, height=height, west=c, north=f, a=a, e=e,
                    south=f + e * height, east=c + a * width)
    raise KeyError(strip)


def tiles_of(strip: str) -> list[dict]:
    out = []
    for split in ("train", "val", "test"):
        for p in (V2 / split).glob(f"{strip}_*.npz"):
            gr = int(p.stem.split("_")[-2])
            gc = int(p.stem.split("_")[-1])
            out.append(dict(split=split, path=p, name=p.name, gr=gr, gc=gc))
    return out


def rd_bounds(grid: dict, pad: float):
    from pyproj import Transformer
    to_rd = Transformer.from_crs("EPSG:32631", "EPSG:28992", always_xy=True)
    west, south = grid["west"] - pad, grid["south"] - pad
    east, north = grid["east"] + pad, grid["north"] + pad
    corners = [to_rd.transform(x, y) for x, y in
               ((west, south), (east, south), (east, north), (west, north))]
    xs, ys = zip(*corners)
    return min(xs), min(ys), max(xs), max(ys)


def _get(url: str, timeout: int = 180) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    if data[:1] == b"<" or data[:5] == b"<?xml" or b"Exception" in data[:400]:
        raise RuntimeError(data[:180])
    return data


def download_ahn(coverage: str, bounds) -> tuple[np.ndarray, object]:
    """Mosaic AHN 0.5 m in RD. Returns (array row0=north, Affine)."""
    xmin = math.floor(bounds[0] * 2) / 2
    ymin = math.floor(bounds[1] * 2) / 2
    xmax = math.ceil(bounds[2] * 2) / 2
    ymax = math.ceil(bounds[3] * 2) / 2
    width = int(round((xmax - xmin) / RES))
    height = int(round((ymax - ymin) / RES))
    log(f"  AHN {coverage} {width}x{height} px  ({(xmax-xmin):.0f}x{(ymax-ymin):.0f} m)")
    tmp = WORK / f"chunks_{coverage}"
    tmp.mkdir(parents=True, exist_ok=True)
    jobs = []
    px, py = 4000, 2400
    for c in range(0, width, px):
        for r in range(0, height, py):
            cw = min(px, width - c)
            rh = min(py, height - r)
            x0 = xmin + c * RES
            y_north = ymax - r * RES
            x1 = x0 + cw * RES
            y_south = y_north - rh * RES
            raw = tmp / f"{c}_{r}.tif"
            url = (f"{AHN_WCS}?SERVICE=WCS&VERSION=1.0.0&REQUEST=GetCoverage"
                   f"&COVERAGE={coverage}&CRS=EPSG:28992"
                   f"&BBOX={x0},{y_south},{x1},{y_north}&WIDTH={cw}&HEIGHT={rh}&FORMAT=image/tiff")
            jobs.append((raw, url, r, c, rh, cw))

    def one(job):
        raw, url, r, c, rh, cw = job
        if raw.exists() and raw.stat().st_size > 500:
            return job
        last = None
        for attempt in range(5):
            try:
                raw.write_bytes(_get(url))
                return job
            except Exception as e:
                last = e
                time.sleep(1.5 + attempt)
        raise RuntimeError(f"WCS failed {coverage} {c},{r}: {last}")

    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(one, j) for j in jobs]
        done = 0
        for fut in as_completed(futs):
            fut.result()
            done += 1
            if done % 8 == 0 or done == len(jobs):
                log(f"  {coverage} chunks {done}/{len(jobs)}")

    arr = np.full((height, width), np.nan, np.float32)
    for raw, url, r, c, rh, cw in jobs:
        with rasterio.open(raw) as src:
            a = src.read(1).astype(np.float32)
            nd = src.nodata
            if nd is not None and np.isfinite(nd):
                a[np.isclose(a, float(nd))] = np.nan
            a[(~np.isfinite(a)) | (a > 1e6) | (a < -1000)] = np.nan
            if a.shape != (rh, cw):
                raise RuntimeError(f"chunk shape {a.shape} != {(rh, cw)} for {raw.name}")
            arr[r:r + rh, c:c + cw] = a
    transform = from_origin(xmin, ymax, RES, RES)
    _validate_rd(arr, transform, coverage)
    shutil.rmtree(tmp, ignore_errors=True)
    log(f"  {coverage} finite {np.isfinite(arr).mean():.1%}")
    return arr, transform


def _validate_rd(arr, transform, coverage: str) -> None:
    """A WCS chip must agree with the mosaic. Catches the row-shift bug.

    DTM is nodata on most buildings, so many 20 m chips have no overlap.
    One agreeing chip and zero disagreements is enough; a shifted mosaic
    either misses data the WCS has, or disagrees by metres where both are finite.
    """
    h, w = arr.shape
    agrees = 0
    misses = 0
    sites = [(fy, fx) for fy in (0.2, 0.4, 0.55, 0.75) for fx in (0.15, 0.4, 0.7)]
    for fy, fx in sites:
        r = int(h * fy)
        c = int(w * fx)
        r = min(r, max(0, h - 40))
        c = min(c, max(0, w - 40))
        x = transform.c + c * RES
        y_north = transform.f - r * RES
        url = (f"{AHN_WCS}?SERVICE=WCS&VERSION=1.0.0&REQUEST=GetCoverage"
               f"&COVERAGE={coverage}&CRS=EPSG:28992"
               f"&BBOX={x},{y_north-20},{x+20},{y_north}&WIDTH=40&HEIGHT=40&FORMAT=image/tiff")
        try:
            data = _get(url, timeout=90)
        except Exception as e:
            log(f"  validate chip failed {e}")
            continue
        chip_path = WORK / "_chip.tif"
        chip_path.write_bytes(data)
        with rasterio.open(chip_path) as src:
            chip = src.read(1).astype(np.float32)
            nd = src.nodata
            if nd is not None and np.isfinite(nd):
                chip[np.isclose(chip, float(nd))] = np.nan
            chip[chip > 1e6] = np.nan
        sl = arr[r:r + chip.shape[0], c:c + chip.shape[1]]
        both = np.isfinite(chip) & np.isfinite(sl[:chip.shape[0], :chip.shape[1]])
        chip_f = float(np.isfinite(chip).mean())
        mos_f = float(np.isfinite(sl).mean()) if sl.size else 0.0
        if chip_f > 0.5 and mos_f < 0.2:
            raise RuntimeError(
                f"mosaic missing data {coverage} at RD {x:.0f},{y_north:.0f}: "
                f"WCS finite {chip_f:.0%}, mosaic {mos_f:.0%}")
        if both.mean() > 0.25:
            diff = abs(float(np.median(chip[both]) - np.median(sl[:chip.shape[0], :chip.shape[1]][both])))
            log(f"  validate {coverage} RD {x:.0f},{y_north:.0f} median |d| {diff:.2f} m")
            if diff > 1.0:
                misses += 1
            else:
                agrees += 1
            if agrees >= 2 and misses == 0:
                return
    if misses:
        raise RuntimeError(f"{coverage} mosaic shifted ({misses} chips differ by > 1 m)")
    if agrees < 1:
        raise RuntimeError(f"{coverage} mosaic did not match WCS ({agrees} agreements, {misses} misses)")
    log(f"  validate {coverage}: {agrees} agreeing chip(s), rest nodata")


def warp_utm(rd: np.ndarray, rd_transform, west: float, north: float, width_m: float, height_m: float,
             dest: Path) -> np.memmap:
    W = int(math.ceil(width_m / RES - 1e-6))
    H = int(math.ceil(height_m / RES - 1e-6))
    dst_transform = from_origin(west, north, RES, RES)
    src = np.where(np.isfinite(rd), rd, -9999).astype(np.float32)
    dst = np.full((H, W), -9999, np.float32)
    reproject(
        src, dst,
        src_transform=rd_transform, src_crs="EPSG:28992", src_nodata=-9999,
        dst_transform=dst_transform, dst_crs="EPSG:32631", dst_nodata=-9999,
        resampling=Resampling.bilinear, num_threads=2,
    )
    dst[dst <= -9000] = np.nan
    dst[~np.isfinite(dst) | (np.abs(dst) > 1e5)] = np.nan
    mm = np.memmap(dest, dtype=np.float32, mode="w+", shape=(H, W))
    mm[:] = dst
    mm.flush()
    log(f"  warped {dest.name} {W}x{H} finite {np.isfinite(dst).mean():.1%}")
    del dst, src
    return mm


def water_mask(dsm: np.ndarray, row0: int, row1: int) -> tuple[np.ndarray, float]:
    """Large nodata components inside the strip (not the range pad)."""
    interior = np.zeros(dsm.shape, bool)
    interior[row0:row1] = ~np.isfinite(dsm[row0:row1])
    comp, ncomp = label(interior)
    if ncomp == 0:
        return np.zeros(dsm.shape, bool), 0.0
    counts = np.bincount(comp.ravel())
    min_cells = int(WATER_M2 / (RES * RES))
    big = np.flatnonzero(counts >= min_cells)
    big = big[big != 0]
    water = np.isin(comp, big) if big.size else np.zeros(dsm.shape, bool)
    edge = binary_dilation(water, iterations=4) & ~water & np.isfinite(dsm)
    border = dsm[edge]
    border = border[np.isfinite(border)]
    low = border[border < 2.0] if border.size else border
    if low.size > 100:
        level = float(np.median(low))
    elif border.size:
        level = float(np.percentile(border, 10))
    else:
        level = 0.0
    return water, level


def fill_small_holes(z: np.ndarray) -> int:
    """Median of a 2-pixel ring, holes under 100 m^2. In place. Returns count."""
    nodata = ~np.isfinite(z)
    if not nodata.any():
        return 0
    comp, ncomp = label(nodata)
    if ncomp == 0:
        return 0
    counts = np.bincount(comp.ravel())
    max_cells = int(HOLE_M2 / (RES * RES))
    objs = find_objects(comp)
    n = 0
    for i in range(1, ncomp + 1):
        if counts[i] == 0 or counts[i] >= max_cells:
            continue
        sl = objs[i - 1]
        if sl is None:
            continue
        sl2 = tuple(slice(max(0, s.start - 2), min(n, s.stop + 2)) for s, n in zip(sl, z.shape))
        sel = comp[sl2] == i
        ring = binary_dilation(sel, iterations=2) & ~sel
        vals = z[sl2][ring]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            sub = np.array(z[sl2], dtype=np.float32, copy=True)
            sub[sel] = np.float32(np.median(vals))
            z[sl2] = sub
            n += 1
            if n % 5000 == 0:
                log(f"    hole fill {n}")
    return n


def fill_nearest(z: np.ndarray) -> int:
    """Fill whatever nodata the water and small-hole passes left, from the nearest finite cell.

    v4's unknown mask was empty once the DTM had filled the ground. Copying the DSM
    into a DTM hole would paste the roof in as the ground and erase the building.
    """
    nan = ~np.isfinite(z)
    nnan = int(nan.sum())
    if nnan == 0 or nnan == z.size:
        return 0
    src = np.where(nan, np.uint8(1), np.uint8(0))
    inds = distance_transform_edt(src, return_distances=False, return_indices=True)
    vals = np.array(z, dtype=np.float32, copy=True)
    z[nan] = vals[inds[0][nan], inds[1][nan]]
    return nnan


def gpkg_wkb(blob: bytes) -> bytes:
    if blob[:2] != b"GP":
        return blob
    env = (blob[3] >> 1) & 0b111
    env_bytes = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[env]
    return blob[8 + env_bytes:]


def _bag_ids(bounds, buffer_m=400):
    xmin, ymin, xmax, ymax = bounds
    xmin -= buffer_m
    ymin -= buffer_m
    xmax += buffer_m
    ymax += buffer_m

    def start(v, origin):
        i = math.floor((v - origin) / BAG_CELL)
        return i - (i % 4)

    ix0, ix1 = start(xmin, BAG_OX), start(xmax, BAG_OX)
    iy0, iy1 = start(ymin, BAG_OY), start(ymax, BAG_OY)
    return [(ix, iy) for ix in range(ix0, ix1 + 1, 4) for iy in range(iy0, iy1 + 1, 4)]


def ensure_bag(bounds) -> None:
    BAG_DIR.mkdir(parents=True, exist_ok=True)
    ids = _bag_ids(bounds)
    missing = []
    for ix, iy in ids:
        gz = BAG_DIR / f"9-{ix}-{iy}.gpkg.gz"
        mark = BAG_DIR / f"9-{ix}-{iy}.missing"
        if mark.exists() or (gz.exists() and gz.stat().st_size > 400):
            continue
        missing.append((ix, iy, gz, mark))
    if not missing:
        return
    log(f"  3DBAG to fetch {len(missing)} (of {len(ids)} candidates)")

    def one(item):
        ix, iy, gz, mark = item
        url = BAG_URL.format(ix=ix, iy=iy)
        try:
            data = _get(url, timeout=180)
        except Exception:
            mark.write_text("missing\n")
            return False
        if data[:1] == b"<" or len(data) < 400:
            mark.write_text("missing\n")
            return False
        gz.write_bytes(data)
        return True

    got = 0
    with ThreadPoolExecutor(max_workers=4) as ex:
        for fut in as_completed([ex.submit(one, it) for it in missing]):
            got += int(bool(fut.result()))
    log(f"  3DBAG downloaded {got}, absent {len(missing) - got}")


def fetch_footprints(bounds):
    import sqlite3
    ensure_bag(bounds)
    xmin, ymin, xmax, ymax = bounds
    # a little extra so a wall in the pad is not dropped
    xmin, ymin, xmax, ymax = xmin - 20, ymin - 20, xmax + 20, ymax + 20
    out = []
    seen = set()
    for gz in sorted(BAG_DIR.glob("9-*.gpkg.gz")):
        gpkg = Path(str(gz)[:-3])
        if not gpkg.exists() or gpkg.stat().st_size < 1000:
            try:
                with gzip.open(gz, "rb") as i, open(gpkg, "wb") as o:
                    shutil.copyfileobj(i, o)
            except Exception:
                continue
        con = sqlite3.connect(f"file:{gpkg}?mode=ro", uri=True)
        try:
            hit = con.execute(
                "SELECT count(*) FROM rtree_pand_geom WHERE maxx>=? AND minx<=? AND maxy>=? AND miny<=?",
                (xmin, xmax, ymin, ymax)).fetchone()[0]
        except sqlite3.Error:
            con.close()
            continue
        if not hit:
            con.close()
            continue
        heights = {}
        try:
            for ident, h70, hmax in con.execute(
                "SELECT identificatie, MAX(b3_h_70p), MAX(b3_h_max) FROM lod22_2d GROUP BY identificatie"
            ):
                heights[ident] = (h70, hmax)
        except sqlite3.Error:
            heights = {}
        q = """SELECT p.identificatie, p.geom, p.b3_h_maaiveld
               FROM pand p, rtree_pand_geom r
               WHERE p.fid = r.id AND r.maxx>=? AND r.minx<=? AND r.maxy>=? AND r.miny<=?"""
        for ident, blob, mv in con.execute(q, (xmin, xmax, ymin, ymax)):
            if not blob or ident in seen:
                continue
            seen.add(ident)
            try:
                poly = wkb.loads(gpkg_wkb(blob))
            except Exception:
                continue
            if poly.is_empty:
                continue
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty or poly.area < 4.0:
                continue
            h70, hmax = heights.get(ident, (None, None))
            out.append((poly, {"id": ident, "roof70": h70, "roofmax": hmax, "ground": mv}))
        con.close()
    log(f"  3DBAG {len(out)} footprints")
    return out


def to_utm(footprints):
    from pyproj import Transformer
    fn = Transformer.from_crs("EPSG:28992", "EPSG:32631", always_xy=True).transform
    out = []
    for poly, attrs in footprints:
        g = shp_transform(fn, poly)
        if not g.is_empty:
            out.append((g, attrs))
    return out


def range_fits(tiles, H, flip, width):
    n = np.zeros(width, np.float64)
    sx = np.zeros(width)
    sy = np.zeros(width)
    sxx = np.zeros(width)
    sxy = np.zeros(width)
    for t in tiles:
        z = np.load(t["path"])
        inc = z["incidence"].astype(np.float64)
        if inc.shape != (512, 512):
            raise RuntimeError(f"incidence {inc.shape} {t['name']}")
        if flip:
            inc = np.flipud(inc)
        g = H * np.tan(np.deg2rad(inc))
        rows = (t["gr"] + np.arange(inc.shape[0], dtype=np.float64))[::8]
        gg = g[::8]
        w = np.isfinite(gg)
        x = rows[:, None]
        xf = np.where(w, x, 0.0)
        yf = np.where(w, gg, 0.0)
        sl = slice(t["gc"], t["gc"] + inc.shape[1])
        n[sl] += w.sum(0)
        sx[sl] += xf.sum(0)
        sy[sl] += yf.sum(0)
        sxx[sl] += (xf * xf).sum(0)
        sxy[sl] += (xf * yf).sum(0)
    denom = n * sxx - sx * sx
    slope = np.full(width, np.nan)
    intercept = np.full(width, np.nan)
    ok = (n >= 8) & (np.abs(denom) > 1e-3)
    slope[ok] = (n[ok] * sxy[ok] - sx[ok] * sy[ok]) / denom[ok]
    intercept[ok] = (sy[ok] - slope[ok] * sx[ok]) / n[ok]
    if ok.sum() < 10:
        raise RuntimeError("not enough incidence samples to fit ground range")
    gs = float(np.median(slope[ok]))
    gi = float(np.median(intercept[ok]))
    # source-row slope: positive when row 0 is the near edge (south-looking)
    if flip and gs >= 0:
        raise RuntimeError(f"ground-range slope {gs} should be negative on a flipped strip")
    if (not flip) and gs <= 0:
        raise RuntimeError(f"ground-range slope {gs} should be positive on an unflipped strip")
    if not (0.05 <= abs(gs) <= 0.6):
        log(f"  WARNING slope {gs:.4f} m/row outside 0.05..0.6")
    slope[~ok] = gs
    intercept[~ok] = gi
    log(f"  g fit {int(ok.sum())} columns, median slope {gs:.4f} m/source-row, intercept {gi:.1f}")
    return slope, intercept


def canon_src_rows(height: int, flip: bool, pad: int) -> np.ndarray:
    if flip:
        return np.arange(height + pad - 1, -pad - 1, -1)
    return np.arange(-pad, height + pad)


def bilinear_block(src: np.ndarray, c0: int, bw: int, bh: int) -> np.ndarray:
    """Sample the 0.5 m grid at the centres of 0.25 m pixels [0:bh, c0:c0+bw)."""
    js = (c0 + np.arange(bw) + 0.5) * (PX / RES) - 0.5
    j_lo = max(0, int(math.floor(float(js.min()))) - 1)
    j_hi = min(src.shape[1], int(math.ceil(float(js.max()))) + 2)
    window = np.array(src[:, j_lo:j_hi], dtype=np.float32)
    finite = np.isfinite(window)
    filled = np.pad(np.where(finite, window, 0).astype(np.float32), 1, mode="edge")
    weight = np.pad(finite.astype(np.float32), 1, mode="edge")
    out = np.empty((bh, bw), np.float32)
    js_local = js - j_lo + 1.0
    for r0 in range(0, bh, 700):
        r1 = min(bh, r0 + 700)
        ii = np.arange(r0, r1)
        ys = (ii + 0.5) * (PX / RES) - 0.5 + 1.0
        yy = np.broadcast_to(ys[:, None], (r1 - r0, bw))
        xx = np.broadcast_to(js_local[None, :], (r1 - r0, bw))
        coords = np.stack([np.ascontiguousarray(yy), np.ascontiguousarray(xx)])
        val = map_coordinates(filled, coords, order=1, mode="nearest").astype(np.float32)
        wgt = map_coordinates(weight, coords, order=1, mode="nearest")
        val[wgt < 0.999] = np.nan
        out[r0:r1] = val
    return out


def nearest_block(mask: np.ndarray, c0: int, bw: int, bh: int) -> np.ndarray:
    H, W = mask.shape
    ys = (np.arange(bh) + 0.5) * (PX / RES) - 0.5
    xs = (c0 + np.arange(bw) + 0.5) * (PX / RES) - 0.5
    r = np.clip(np.rint(ys).astype(np.int32), 0, H - 1)
    c = np.clip(np.rint(xs).astype(np.int32), 0, W - 1)
    return mask[r[:, None], c[None, :]]


def _assign_rows(mm, rows, cols0, cols1, values):
    """mm[rows, cols0:cols1] = values, rows a 1d integer array."""
    # fancy index mixed with a slice assigns as a rectangular gather
    mm[rows, cols0:cols1] = values


def _nbytes(path: Path, n: int) -> bool:
    return path.exists() and path.stat().st_size == n


def _save_progress(path: Path, c: int, h: int, w: int, acc: dict) -> None:
    payload = dict(
        c=int(c), h=int(h), w=int(w),
        pixels=int(acc["pixels"]), ign=int(acc["ign"]), unk_d=int(acc["unk_d"]),
        water=int(acc["water"]), a=int(acc["a"]), b=int(acc["b"]), cign=int(acc["c"]),
        cls=[int(x) for x in acc["cls"]],
    )
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)


def _load_progress(path: Path, h: int, w: int):
    try:
        p = json.loads(path.read_text())
    except Exception:
        return None
    c = int(p.get("c", -1))
    if int(p.get("h", -1)) != h or int(p.get("w", -1)) != w or not (0 <= c <= w):
        return None
    acc = dict(
        pixels=int(p["pixels"]), ign=int(p["ign"]), unk_d=int(p["unk_d"]),
        water=int(p["water"]), a=int(p["a"]), b=int(p["b"]), c=int(p["cign"]),
        cls=np.array(p["cls"], np.int64),
    )
    return c, acc


def process_strip(strip: str) -> dict:
    geom = GEOM[strip]
    grid = strip_grid(strip)
    tiles = tiles_of(strip)
    if not tiles:
        raise RuntimeError(f"no v2 tiles for {strip}")
    h, w = grid["height"], grid["width"]
    log(f"{strip}  tiles {len(tiles)}  {w}x{h}  flip {geom['flip']}  H {geom['H']}")
    WORK.mkdir(parents=True, exist_ok=True)
    slope, intercept = range_fits(tiles, geom["H"], geom["flip"], w)

    bounds = rd_bounds(grid, RANGE_PAD_M)
    t0 = time.time()
    pad_px = int(round(RANGE_PAD_M / PX))
    bh = h + 2 * pad_px
    north0 = grid["north"] + pad_px * PX
    west0 = grid["west"]
    height_m = bh * PX
    width_m = w * PX
    dsm_path = WORK / f"{strip}_dsm.f32"
    dtm_path = WORK / f"{strip}_dtm.f32"
    water_path = WORK / f"{strip}_water.u8"
    surf_path = WORK / f"{strip}_surfaces.json"
    prog_path = WORK / f"{strip}_progress.json"
    H05 = int(math.ceil(height_m / RES - 1e-6))
    W05 = int(math.ceil(width_m / RES - 1e-6))
    n05 = H05 * W05

    def surfaces_ready() -> bool:
        if not surf_path.exists():
            return False
        try:
            meta = json.loads(surf_path.read_text())
        except Exception:
            return False
        if meta.get("H") != H05 or meta.get("W") != W05 or not meta.get("ready"):
            return False
        return (_nbytes(dsm_path, n05 * 4) and _nbytes(dtm_path, n05 * 4)
                and _nbytes(water_path, n05))

    if surfaces_ready():
        dsm = np.memmap(dsm_path, dtype=np.float32, mode="r", shape=(H05, W05))
        dtm = np.memmap(dtm_path, dtype=np.float32, mode="r", shape=(H05, W05))
        water = np.array(np.memmap(water_path, dtype=np.uint8, mode="r", shape=(H05, W05)), dtype=bool)
        log(f"  resume filled DSM/DTM/water {W05}x{H05}")
    else:
        dsm_rd, rd_tf = download_ahn("dsm_05m", bounds)
        dsm = warp_utm(dsm_rd, rd_tf, west0, north0, width_m, height_m, dsm_path)
        del dsm_rd
        dtm_rd, rd_tf = download_ahn("dtm_05m", bounds)
        dtm = warp_utm(dtm_rd, rd_tf, west0, north0, width_m, height_m, dtm_path)
        del dtm_rd
        if dsm.shape != (H05, W05) or dtm.shape != (H05, W05):
            raise RuntimeError(f"warp shape {dsm.shape} != {(H05, W05)}")
        # interior rows of the 0.5 m grid: the unpadded strip
        row0 = int(round(pad_px * PX / RES))
        row1 = int(round((pad_px + h) * PX / RES))
        row1 = min(row1, H05)
        water, level = water_mask(dsm, row0, row1)
        log(f"  water {water[row0:row1].mean():.1%} of strip, level {level:.2f} m NAP  ({time.time()-t0:.0f}s)")
        dsm[water] = level
        dtm[water] = level
        dsm.flush()
        dtm.flush()
        t1 = time.time()
        n_holes = fill_small_holes(dsm) + fill_small_holes(dtm)
        dsm.flush(); dtm.flush()
        log(f"  filled {n_holes} small holes in {time.time()-t1:.0f}s")
        t2 = time.time()
        n_dsm = fill_nearest(dsm)
        n_dtm = fill_nearest(dtm)
        dsm.flush(); dtm.flush()
        log(f"  nearest-filled remaining nodata  dsm {n_dsm}  dtm {n_dtm}  in {time.time()-t2:.0f}s")
        water_mm = np.memmap(water_path, dtype=np.uint8, mode="w+", shape=water.shape)
        water_mm[:] = water.astype(np.uint8)
        water_mm.flush()
        del water_mm
        surf_path.write_text(json.dumps({"H": H05, "W": W05, "level": float(level), "ready": True}))
        prog_path.unlink(missing_ok=True)
        for stale in (WORK / f"{strip}_v4.u8", WORK / f"{strip}_v5.u8",
                      WORK / f"{strip}_cov.f2", WORK / f"{strip}_mag.f2"):
            stale.unlink(missing_ok=True)
        log("  surfaces saved")

    fps = to_utm(fetch_footprints(bounds))
    geoms = [g for g, _ in fps]
    tree = STRtree(geoms) if geoms else None

    v4_path = WORK / f"{strip}_v4.u8"
    v5_path = WORK / f"{strip}_v5.u8"
    cov_path = WORK / f"{strip}_cov.f2"
    mag_path = WORK / f"{strip}_mag.f2"
    shapes_ok = (_nbytes(v4_path, h * w) and _nbytes(v5_path, h * w)
                 and _nbytes(cov_path, h * w * 2) and _nbytes(mag_path, h * w * 2))
    resumed = None
    if prog_path.exists() and shapes_ok:
        resumed = _load_progress(prog_path, h, w)
    if resumed is None:
        v4 = np.memmap(v4_path, dtype=np.uint8, mode="w+", shape=(h, w))
        v5 = np.memmap(v5_path, dtype=np.uint8, mode="w+", shape=(h, w))
        cov_m = np.memmap(cov_path, dtype=np.float16, mode="w+", shape=(h, w))
        mag_m = np.memmap(mag_path, dtype=np.float16, mode="w+", shape=(h, w))
        v4[:] = 255
        v5[:] = 255
        v4.flush(); v5.flush(); cov_m.flush(); mag_m.flush()
        acc = dict(pixels=0, ign=0, unk_d=0, water=0, a=0, b=0, c=0,
                   cls=np.zeros(3, np.int64))
        c = 0
        _save_progress(prog_path, c, h, w, acc)
        log("  label maps initialised")
    else:
        c, acc = resumed
        v4 = np.memmap(v4_path, dtype=np.uint8, mode="r+", shape=(h, w))
        v5 = np.memmap(v5_path, dtype=np.uint8, mode="r+", shape=(h, w))
        cov_m = np.memmap(cov_path, dtype=np.float16, mode="r+", shape=(h, w))
        mag_m = np.memmap(mag_path, dtype=np.float16, mode="r+", shape=(h, w))
        log(f"  resume columns at {c}/{w}")
    src_rows = canon_src_rows(h, geom["flip"], pad_px)
    if src_rows.size != bh:
        raise RuntimeError(f"canonical rows {src_rows.size} != padded height {bh}")
    inside = (src_rows >= 0) & (src_rows < h)
    sel = np.flatnonzero(inside)
    sr = src_rows[sel]
    br = sr + pad_px

    t_block = time.time()
    while c < w:
        c0 = c
        c1 = min(w, c + BLOCK)
        cb0 = max(0, c0 - HALO)
        cb1 = min(w, c1 + HALO)
        bw = cb1 - cb0
        dsm_b = bilinear_block(dsm, cb0, bw, bh)
        dtm_b = bilinear_block(dtm, cb0, bw, bh)
        water_b = nearest_block(water, cb0, bw, bh)
        x0 = west0 + cb0 * PX
        x1 = west0 + cb1 * PX
        y1 = north0
        y0 = north0 - bh * PX
        fp = np.zeros((bh, bw), np.uint8)
        if tree is not None:
            hits = tree.query(box(x0, y0, x1, y1))
            if len(hits):
                shapes = [(geoms[int(i)], 1) for i in np.atleast_1d(hits)]
                fp = rasterize(shapes, out_shape=(bh, bw),
                               transform=from_origin(x0, y1, PX, PX),
                               fill=0, dtype=np.uint8)
        fpb = fp.astype(bool)
        fp1 = binary_dilation(fpb, iterations=4)   # 1 m
        fp2 = binary_dilation(fpb, iterations=8)   # 2 m
        ground = dtm_b.copy()
        z_b = ground.copy()
        use = fp1 & np.isfinite(dsm_b)
        z_b[use] = dsm_b[use]
        ndsm = dsm_b - ground
        core_cl = (ndsm > 3.0) & ~fp1 & np.isfinite(ndsm) & ~water_b
        clutter = binary_dilation(core_cl, iterations=8) & ~water_b

        j0 = c0 - cb0
        j1 = j0 + (c1 - c0)
        # canonical reorder, core columns only. block row of source row s is s+pad
        zf = dsm_b[src_rows + pad_px, j0:j1]
        zb = z_b[src_rows + pad_px, j0:j1]
        cols = np.arange(c0, c1)
        gg = intercept[cols][None, :] + slope[cols][None, :] * src_rows[:, None]
        r_f = np.hypot(geom["H"] - zf, gg)
        a_f = np.arctan2(gg, geom["H"] - zf)
        r_b = np.hypot(geom["H"] - zb, gg)
        a_b = np.arctan2(gg, geom["H"] - zb)
        bad = ~np.isfinite(zf) | ~np.isfinite(gg)
        r_f = r_f.copy(); a_f = a_f.copy()
        r_b = r_b.copy(); a_b = a_b.copy()
        r_f[bad] = np.nan
        a_f[bad] = np.nan
        badb = ~np.isfinite(zb) | ~np.isfinite(gg)
        r_b[badb] = np.nan
        a_b[badb] = np.nan
        cov_f, unk_f, _ = own_range_grid(r_f, a_f, axis=0, nan_margin=NAN_MARGIN)
        del r_f, a_f
        cov_b, unk_b, _ = own_range_grid(r_b, a_b, axis=0, nan_margin=NAN_MARGIN)
        disp, unk_d = own_range_displacement_grid(
            r_b, a_b, axis=0, nan_margin=NAN_MARGIN, step=0.1, pixel_m=PX)
        del r_b, a_b
        lab_f = labels_from_coverage(cov_f, unk_f)
        lab_b = labels_from_coverage(cov_b, unk_b)
        mag = layover_magnitude(cov_b, disp)

        lb = lab_b[sel]
        lf = lab_f[sel]
        ign_a = unk_b[sel]
        water_c = water_b[br, j0:j1]
        ign_b = clutter[br, j0:j1]
        ign_c = (lb != lf) & ~water_c & ~ign_a
        m = lb.copy()
        m[ign_a | ign_b | ign_c] = 255
        _assign_rows(v4, sr, c0, c1, m)
        ud = unk_d[sel]
        m5 = m.copy()
        m5[fp2[br, j0:j1] | ud] = 255
        _assign_rows(v5, sr, c0, c1, m5)
        _assign_rows(cov_m, sr, c0, c1, cov_b[sel].astype(np.float16))
        mg = mag[sel].astype(np.float32)
        mg[~np.isfinite(mg) | (mg < 0)] = 0
        mg[m5 != 1] = 0
        _assign_rows(mag_m, sr, c0, c1, mg.astype(np.float16))

        acc["pixels"] += m.size
        acc["ign"] += int((m == 255).sum())
        acc["unk_d"] += int(ud.sum())
        acc["water"] += int(water_c.sum())
        acc["a"] += int(ign_a.sum())
        acc["b"] += int(ign_b.sum())
        acc["c"] += int(ign_c.sum())
        for cls in (0, 1, 2):
            acc["cls"][cls] += int((m == cls).sum())
        c = c1
        if (c // BLOCK) % 4 == 0 or c >= w:
            v4.flush(); v5.flush(); cov_m.flush(); mag_m.flush()
            _save_progress(prog_path, c, h, w, acc)
            dt = time.time() - t_block
            log(f"  columns {c}/{w}  {dt:.1f}s  "
                f"ign {acc['ign']/max(acc['pixels'],1):.1%}  "
                f"lay {acc['cls'][1]/max(acc['pixels'],1):.1%}")
            t_block = time.time()

    v4.flush(); v5.flush(); cov_m.flush(); mag_m.flush()
    tot = max(acc["pixels"], 1)
    full = dict(
        ignored=acc["ign"] / tot,
        unk_d=acc["unk_d"] / tot,
        water=acc["water"] / tot,
        frac_a=acc["a"] / tot,
        frac_b=acc["b"] / tot,
        frac_c=acc["c"] / tot,
        shares=(acc["cls"] / max(acc["cls"].sum(), 1)).tolist(),
    )
    log(f"  full strip ignored {full['ignored']:.1%} water {full['water']:.1%} "
        f"unk_d {full['unk_d']:.2%}  shares {np.round(full['shares'], 3).tolist()}")
    tile_stats = cut_tiles(strip, tiles, grid, geom, v4, v5, cov_m, mag_m)
    ratios = ratio_check(strip, fps, grid, geom, tiles)
    del v4, v5, cov_m, mag_m, dsm, dtm
    for p in (dsm_path, dtm_path, water_path, surf_path, prog_path,
              v4_path, v5_path, cov_path, mag_path):
        p.unlink(missing_ok=True)
    log(f"  cleared intermediates for {strip}")
    med = float(np.median(ratios)) if ratios else None
    p10 = float(np.percentile(ratios, 10)) if ratios else None
    p90 = float(np.percentile(ratios, 90)) if ratios else None
    log(f"  magnitude/L  n={len(ratios)} median {med} p10 {p10} p90 {p90}")
    return dict(full=full, ratios=ratios, ratio_median=med, ratio_p10=p10, ratio_p90=p90,
                **tile_stats)


def cut_tiles(strip, tiles, grid, geom, v4, v5, cov_m, mag_m) -> dict:
    PACK.mkdir(parents=True, exist_ok=True)
    counts = {s: 0 for s in ("train", "val", "test")}
    dropped_empty = 0
    over60 = 0
    over60_m4 = 0
    h = grid["height"]
    edges = np.linspace(-50, 30, 8001)
    thirds = {k: {"l": np.zeros(8000, np.int64), "n": np.zeros(8000, np.int64)}
              for k in ("near", "middle", "far")}
    mag_edges = np.linspace(0, 500, 5001)
    mag_hist = {s: np.zeros(5000, np.int64) for s in counts}
    mag_max = {s: 0.0 for s in counts}
    mag_sum = {s: 0.0 for s in counts}
    mag_n = {s: 0 for s in counts}
    below1 = {s: 0 for s in counts}
    cls = np.zeros(3, np.int64)
    cls5 = np.zeros(3, np.int64)
    ign = 0
    tot = 0
    bad_mag = 0
    manifest = []
    for t in tiles:
        gr, gc = t["gr"], t["gc"]
        m4 = np.array(v4[gr:gr + 512, gc:gc + 512])
        m5 = np.array(v5[gr:gr + 512, gc:gc + 512])
        cv = np.array(cov_m[gr:gr + 512, gc:gc + 512])
        mg = np.array(mag_m[gr:gr + 512, gc:gc + 512], dtype=np.float32)
        if geom["flip"]:
            m4, m5 = np.flipud(m4), np.flipud(m5)
            cv, mg = np.flipud(cv), np.flipud(mg)
        if not (m5 != 255).any():
            dropped_empty += 1
            continue
        if (m5 == 255).mean() > 0.60:
            over60 += 1
        if (m4 == 255).mean() > 0.60:
            over60_m4 += 1
        # magnitude rules, enforced again at the tile
        mg = np.nan_to_num(mg, nan=0.0, posinf=0.0, neginf=0.0)
        mg[mg < 0] = 0
        mg[m5 != 1] = 0
        if not np.isfinite(mg).all() or (mg[m5 != 1] != 0).any() or (mg < 0).any():
            bad_mag += 1
        z = np.load(t["path"])
        hh = z["image"][0].astype(np.float32)
        ign += int((m4 == 255).sum())
        tot += m4.size
        for c in (0, 1, 2):
            cls[c] += int((m4 == c).sum())
            cls5[c] += int((m5 == c).sum())
        rows = np.arange(512)
        src = t["gr"] + (511 - rows if geom["flip"] else rows)
        # distance from the sensor, in source rows. On a north-looking strip the
        # sensor is at the south edge, so a raw source-row fraction would swap
        # the near and far thirds.
        dist = ((h - 1) - src) if geom["flip"] else src
        frac = dist / h
        key_i = np.where(frac < 1 / 3, 0, np.where(frac < 2 / 3, 1, 2))
        key_name = ("near", "middle", "far")
        for ki, name in enumerate(key_name):
            sel = key_i == ki
            if not sel.any():
                continue
            hh_s = hh[sel]
            m_s = m4[sel]
            thirds[name]["l"] += np.histogram(hh_s[m_s == 1], bins=edges)[0]
            thirds[name]["n"] += np.histogram(hh_s[m_s == 0], bins=edges)[0]
        lay = mg[m5 == 1]
        if lay.size:
            mag_hist[t["split"]] += np.histogram(np.clip(lay, 0, 499.999), bins=mag_edges)[0]
            mag_max[t["split"]] = max(mag_max[t["split"]], float(lay.max()))
            mag_sum[t["split"]] += float(lay.sum())
            mag_n[t["split"]] += int(lay.size)
            below1[t["split"]] += int((lay < 1).sum())
        rel = PACK / t["split"] / t["name"]
        rel.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(rel, mask=m5.astype(np.uint8),
                            coverage=cv.astype(np.float16),
                            magnitude=mg.astype(np.float16))
        md5 = hashlib.md5(np.ascontiguousarray(z["image"]).tobytes()).hexdigest()
        manifest.append(dict(split=t["split"], name=t["name"], v2_split=t["split"],
                             v2_name=t["name"], image_md5=md5, label_version=LABEL_VERSION))
        dst = TILES / t["split"] / t["name"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        raw = z["meta"]
        if isinstance(raw, np.ndarray):
            raw = raw.item()
        if isinstance(raw, bytes):
            raw = raw.decode()
        try:
            meta = json.loads(raw)
        except Exception:
            meta = {"v2_meta": str(raw)}
        meta["label_version"] = LABEL_VERSION
        np.savez(dst, image=z["image"], mask=m5.astype(np.uint8), incidence=z["incidence"],
                 look=z["look"], coverage=cv.astype(np.float16),
                 magnitude=mg.astype(np.float16), meta=np.array(json.dumps(meta)))
        counts[t["split"]] += 1
    if bad_mag:
        raise RuntimeError(f"{bad_mag} tiles failed the magnitude rules")

    def hist_median(hist, edges_):
        c = np.cumsum(hist)
        if c[-1] == 0:
            return None
        i = int(np.searchsorted(c, (c[-1] + 1) // 2))
        i = min(i, len(edges_) - 2)
        return float(edges_[i])

    contrast = {}
    for name, d in thirds.items():
        ml, mn = hist_median(d["l"], edges), hist_median(d["n"], edges)
        contrast[name] = None if ml is None or mn is None else ml - mn
    shares = (cls / max(cls.sum(), 1)).tolist()
    shares5 = (cls5 / max(cls5.sum(), 1)).tolist()
    log(f"  kept {sum(counts.values())} empty {dropped_empty} "
        f">60% v5 {over60} >60% v4-mask {over60_m4}")
    log(f"  tile m4 shares {np.round(shares, 3).tolist()} ignored {ign/max(tot,1):.1%}")
    log(f"  layover-nominal dB near {contrast['near']} middle {contrast['middle']} far {contrast['far']}")
    mag_stats = {}
    for s in counts:
        hist = mag_hist[s]
        mag_stats[s] = dict(
            n=mag_n[s],
            mean=(mag_sum[s] / mag_n[s]) if mag_n[s] else None,
            median=hist_median(hist, mag_edges),
            p90=_hist_pct(hist, mag_edges, 90),
            p99=_hist_pct(hist, mag_edges, 99),
            max=mag_max[s] if mag_n[s] else None,
            below_1m=(below1[s] / mag_n[s]) if mag_n[s] else None,
        )
        log(f"  mag {s} {mag_stats[s]}")
    return dict(counts=counts, over60=over60, over60_m4=over60_m4, empty=dropped_empty,
                shares_m4=shares, shares_m5=shares5, ignored_tiles=ign / max(tot, 1),
                contrast=contrast, manifest=manifest, mag_stats=mag_stats)


def _hist_pct(hist, edges, pct):
    c = np.cumsum(hist)
    if c[-1] == 0:
        return None
    i = int(np.searchsorted(c, c[-1] * pct / 100.0))
    i = min(max(i, 0), len(edges) - 2)
    return float(edges[i])


def ratio_check(strip, fps, grid, geom, tiles) -> list[float]:
    """Median magnitude / L on flat roofs, in front of the near wall."""
    if not fps:
        return []
    geoms = [g for g, _ in fps]
    attrs = [a for _, a in fps]
    tree = STRtree(geoms)
    H = geom["H"]
    used = set()
    ratios = []
    examples = []
    for t in tiles:
        pack_path = PACK / t["split"] / t["name"]
        if not pack_path.exists():
            continue
        z = np.load(pack_path)
        mask = z["mask"]
        mag = z["magnitude"].astype(np.float32)
        inc = np.load(t["path"])["incidence"].astype(np.float64)
        x0 = grid["west"] + t["gc"] * PX
        y_north = grid["north"] - t["gr"] * PX
        # source-orientation window, then flip to stored (sensor at top) if needed
        tf = from_origin(x0, y_north, PX, PX)
        minx, maxx = x0, x0 + 512 * PX
        maxy, miny = y_north, y_north - 512 * PX
        hits = tree.query(box(minx, miny, maxx, maxy))
        for i in np.atleast_1d(hits):
            i = int(i)
            a = attrs[i]
            if a["id"] in used:
                continue
            h70, hmax, mv = a["roof70"], a["roofmax"], a["ground"]
            if h70 is None or hmax is None or mv is None:
                continue
            if not (math.isfinite(h70) and math.isfinite(hmax) and math.isfinite(mv)):
                continue
            if (hmax - h70) >= 2.0:
                continue
            height = h70 - mv
            if not (15.0 <= height <= 400.0):
                continue
            fp = rasterize([(geoms[i], 1)], out_shape=(512, 512), transform=tf, fill=0, dtype=np.uint8)
            if geom["flip"]:
                fp = np.flipud(fp)
            cols = np.flatnonzero(fp.any(0))
            if cols.size < 4:
                continue
            near = np.argmax(fp, 0)
            far = 511 - np.argmax(fp[::-1], 0)
            # keep columns where the footprint actually is
            near_c = near[cols]
            far_c = far[cols]
            g = H * np.tan(np.deg2rad(inc[near_c, cols]))
            disc = g * g - 2 * H * height + height * height
            ok = disc > 1
            if ok.sum() < 4:
                continue
            L = g[ok] - np.sqrt(disc[ok])
            cols_k = cols[ok]
            near_k = near_c[ok]
            far_k = far_c[ok]
            depth = (far_k - near_k + 1) * PX
            keep = (depth > L) & (L > 5) & (L < 110) & (near_k * PX >= L)
            if keep.sum() < 4:
                continue
            cols_k, near_k, L = cols_k[keep], near_k[keep], L[keep]
            mags = []
            for col, nr, Lc in zip(cols_k, near_k, L):
                dist = (nr - np.arange(512)) * PX
                sel = (dist >= 2.0) & (dist <= Lc) & (mask[:, col] == 1)
                if sel.any():
                    mags.append(mag[sel, col])
            if not mags:
                continue
            mags = np.concatenate(mags)
            if mags.size < 30:
                continue
            Lmed = float(np.median(L))
            ratio = float(np.median(mags) / Lmed)
            used.add(a["id"])
            ratios.append(ratio)
            if len(examples) < 8:
                examples.append((a["id"], round(height, 1), round(Lmed, 1),
                                 round(float(np.median(mags)), 1), round(ratio, 3), t["name"]))
    for e in examples:
        log(f"  example h={e[1]} L={e[2]} mag={e[3]} ratio={e[4]} {e[5]} {e[0][-12:]}")
    return ratios


def _write_manifest(rows: list[dict]) -> None:
    PACK.mkdir(parents=True, exist_ok=True)
    with (PACK / "manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "name", "v2_split", "v2_name", "image_md5", "label_version"])
        w.writeheader()
        w.writerows(rows)
    (PACK / "classes.json").write_text(json.dumps(["nominal", "layover", "shadow"]))


def _stamp_path(strip: str) -> Path:
    return OUT / "strips" / f"{strip}.json"


def _save_stamp(strip: str, st: dict) -> None:
    p = _stamp_path(strip)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: st[k] for k in st if k != "manifest"}
    payload["manifest"] = st["manifest"]
    payload["strip"] = strip
    p.write_text(json.dumps(payload, indent=2, default=str))


def _load_stamp(strip: str) -> dict:
    return json.loads(_stamp_path(strip).read_text())


def _index_selfcheck() -> None:
    a = np.zeros((6, 8), np.uint8)
    rows = np.array([4, 1, 3])
    m = np.arange(12, dtype=np.uint8).reshape(3, 4)
    _assign_rows(a, rows, 2, 6, m)
    assert a[4, 2] == 0 and a[1, 5] == 7 and a[3, 2] == 8, a


def main():
    _index_selfcheck()
    OUT.mkdir(parents=True, exist_ok=True)
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    only = args[0] if args else None
    if only and only not in GEOM:
        raise SystemExit(f"unknown strip {only}")
    if only:
        strips = [only]
    else:
        strips = [K_ID] + [s for s in STRIPS if s != K_ID]
    all_manifest = []
    summary = {}
    for strip in strips:
        if _stamp_path(strip).exists() and "--redo" not in sys.argv:
            st = _load_stamp(strip)
            log(f"skip finished {strip}  ratio {st.get('ratio_median')}")
        else:
            st = process_strip(strip)
            _save_stamp(strip, st)
        summary[strip] = {k: st[k] for k in st if k != "manifest"}
        all_manifest.extend(st["manifest"])
        _write_manifest(all_manifest)
        (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        # Gate on K before any later strip, when this invocation is the full run
        # or an explicit K run.
        if strip == K_ID:
            med = st.get("ratio_median")
            n = len(st.get("ratios") or [])
            log(f"GATE K magnitude/L n={n} median={med} p10={st.get('ratio_p10')} p90={st.get('ratio_p90')}")
            if n >= 8 and (med is None or not (0.8 <= med <= 1.2)):
                log("STOP: median magnitude/L outside 0.8-1.2")
                sys.exit(2)
            if n < 8:
                log("  few wide flat roofs on K; geometry gate uses whatever qualified, not a stop")
            if only:
                return
    TILES.mkdir(parents=True, exist_ok=True)
    shutil.copy(V2 / "norm_stats.json", TILES / "norm_stats.json")
    log(f"manifest {len(all_manifest)} tiles")


if __name__ == "__main__":
    main()
