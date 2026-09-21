#!/usr/bin/env python3
"""Tiled SAR + layover/shadow dataset for semantic segmentation.

Does not train a model. Masks are computed from the DSM and two geometry
parameters only — SAR brightness never enters the labels.

    python catalog.py
    python download.py --ahn-aoi
    python dataset.py --one-strip          # inspect first
    python dataset.py --strips ID ID       # named strips (local files or --fetch-sar)
    python dataset.py --max-strips 12      # geographic sample
    python dataset.py --masks-only         # cache fold-test masks for all 202
    python dataset.py                      # tile every strip that is on disk / fetchable
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw
from rasterio.transform import Affine, from_origin, xy
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window, from_bounds as window_from_bounds

from catalog import LOCAL_KEY, open_strip
from dsm_geometry import (
    CLASS_ACTIVE,
    CLASS_NOMINAL,
    CLASS_PASSIVE,
    CLASS_SHADOW,
    fill_nodata,
    mask_from_dsm,
)
from pipeline import percentile_stretch
from splits import BlockGrid, assign_blocks, assert_no_train_test_overlap

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUT = ROOT / "output"
MASK_DIR = OUTPUT / "masks"
SET_DIR = OUTPUT / "dataset"
PUBLIC = ROOT.parent / "public" / "dataset"

TILE_SIZE = 512
GSD_M = 0.25
TILE_M = TILE_SIZE * GSD_M  # 128 m
TRAIN_STRIDE = 256  # 50% overlap
EVAL_STRIDE = 512  # no overlap
BLOCK_M = 1000.0
DSM_POSTING = 1.0
DSM_BUFFER_M = 150.0
LOG_EPS = 1e-6
BANDS = (1, 2, 3, 4)  # HH, HV, VH, VV
ACTIVE_RGB = np.array([212, 120, 58], dtype=np.float32)
PASSIVE_RGB = np.array([140, 84, 64], dtype=np.float32)
SHADOW_RGB = np.array([91, 143, 191], dtype=np.float32)
KEEP_LOCAL = {LOCAL_KEY}


def log(msg: str) -> None:
    print(msg, flush=True)


def default_dsm() -> Path:
    aoi = DATA / "ahn_dsm_1m_aoi.tif"
    if aoi.exists() and aoi.stat().st_size > 1_000_000:
        return aoi
    return DATA / "ahn_dsm_1m.tif"


def load_catalog(path: Path | None = None) -> list[dict]:
    path = path or (OUTPUT / "strip_catalog.json")
    if not path.exists():
        raise SystemExit(f"catalog missing: {path}  (python catalog.py)")
    rows = json.loads(path.read_text())
    for row in rows:
        if "incidence_deg" not in row or row["incidence_deg"] is None:
            raise SystemExit(f"FAIL {row.get('strip_id')}: incidence missing")
        if row.get("orientation_flag") not in (0, 1):
            raise SystemExit(f"FAIL {row.get('strip_id')}: look azimuth missing")
    return rows


def union_bounds(rows: list[dict]) -> tuple[float, float, float, float]:
    xs = [b for r in rows for b in (r["bounds"][0], r["bounds"][2])]
    ys = [b for r in rows for b in (r["bounds"][1], r["bounds"][3])]
    return min(xs), min(ys), max(xs), max(ys)


def affine_from_list(values) -> Affine:
    a, b, c, d, e, f = values
    return Affine(a, b, c, d, e, f)


def tile_windows(height: int, width: int, size: int, stride: int):
    if height < size or width < size:
        return
    last_r = height - size
    last_c = width - size
    r = 0
    while True:
        c = 0
        while True:
            yield r, c
            if c == last_c:
                break
            c = min(c + stride, last_c)
        if r == last_r:
            break
        r = min(r + stride, last_r)


def log_scale(arr: np.ndarray) -> tuple[np.ndarray, str]:
    """log10(x+eps) on linear amplitude. MAG-POL is already dB — leave it."""
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return arr, "empty"
    if finite.min() < 0:
        return arr.astype(np.float32), "db"
    return np.log10(np.clip(arr, 0, None) + LOG_EPS).astype(np.float32), "log10"


def canonicalise(
    image: np.ndarray,
    mask: np.ndarray,
    orientation_flag: int,
    enabled: bool = True,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Flip south-facing (rows) so layover sits on the same side as north-facing."""
    if enabled and orientation_flag == 1:
        return np.flip(image, axis=-2).copy(), np.flip(mask, axis=-2).copy(), True
    return image, mask, False


def drop_reason(
    image: np.ndarray,
    mask: np.ndarray,
    dsm_valid: np.ndarray,
    cap_zero: float = 1.0,
) -> str | None:
    if not np.isfinite(image).all():
        return "sar_nodata"
    if not dsm_valid.all():
        return "dsm_nodata"
    layover = float(np.mean((mask == CLASS_ACTIVE) | (mask == CLASS_PASSIVE)))
    if layover == 0.0 and cap_zero < 1.0:
        return "zero_layover"
    return None


def window_bounds(transform: Affine, row: int, col: int, size: int):
    west, north = xy(transform, row, col, offset="ul")
    east, south = xy(transform, row + size, col + size, offset="ul")
    if south > north:
        south, north = north, south
    if west > east:
        west, east = east, west
    return float(west), float(south), float(east), float(north)


def prepare_dsm_for_strip(ahn_path: Path, row: dict):
    """Windowed read of the AHN, reprojected to the strip CRS at 1 m."""
    from pyproj import Transformer

    west, south, east, north = row["bounds"]
    west -= DSM_BUFFER_M
    south -= DSM_BUFFER_M
    east += DSM_BUFFER_M
    north += DSM_BUFFER_M
    width = int(round((east - west) / DSM_POSTING))
    height = int(round((north - south) / DSM_POSTING))
    transform = from_origin(west, north, DSM_POSTING, DSM_POSTING)
    dest = np.full((height, width), np.nan, dtype=np.float32)
    valid = np.zeros((height, width), dtype=np.uint8)
    with rasterio.open(ahn_path) as ahn:
        to_ahn = Transformer.from_crs(row["crs"], ahn.crs, always_xy=True)
        corners = [
            to_ahn.transform(x, y)
            for x, y in ((west, south), (east, south), (east, north), (west, north))
        ]
        ax = [c[0] for c in corners]
        ay = [c[1] for c in corners]
        win = window_from_bounds(min(ax), min(ay), max(ax), max(ay), transform=ahn.transform)
        win = win.round_offsets().round_lengths().intersection(Window(0, 0, ahn.width, ahn.height))
        if win.width <= 0 or win.height <= 0:
            raise RuntimeError(f"AHN does not cover {row['strip_id']}")
        z = ahn.read(1, window=win)
        src_t = rasterio.windows.transform(win, ahn.transform)
        ok = np.isfinite(z) & (np.abs(z) < 1.0e6)
        if ahn.nodata is not None and np.isfinite(ahn.nodata):
            ok &= z != ahn.nodata
        src = np.where(ok, z, np.nan).astype(np.float32)
        reproject(
            source=src,
            destination=dest,
            src_transform=src_t,
            src_crs=ahn.crs,
            src_nodata=np.nan,
            dst_transform=transform,
            dst_crs=row["crs"],
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
        reproject(
            source=ok.astype(np.uint8),
            destination=valid,
            src_transform=src_t,
            src_crs=ahn.crs,
            dst_transform=transform,
            dst_crs=row["crs"],
            resampling=Resampling.nearest,
        )
    filled = fill_nodata(dest, nodata=None)
    return filled, valid.astype(bool), transform


def warp_to_strip(arr, src_t, src_crs, row: dict, resampling, dtype):
    out = np.zeros((row["height"], row["width"]), dtype=dtype)
    reproject(
        source=arr,
        destination=out,
        src_transform=src_t,
        src_crs=src_crs,
        dst_transform=affine_from_list(row["transform"]),
        dst_crs=row["crs"],
        resampling=resampling,
    )
    return out


def write_mask_geotiff(path: Path, mask: np.ndarray, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "count": 1,
        "height": row["height"],
        "width": row["width"],
        "crs": row["crs"],
        "transform": affine_from_list(row["transform"]),
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(mask, 1)
        dst.update_tags(
            incidence_deg=str(row["incidence_deg"]),
            look_azimuth_deg=str(row["look_azimuth_deg"]),
            orientation_flag=str(row["orientation_flag"]),
            incidence_field=row["incidence_field"],
        )


def generate_mask(row: dict, ahn_path: Path, cache_dir: Path) -> tuple[Path, np.ndarray, np.ndarray]:
    """DSM + incidence + look azimuth. Never reads SAR pixels."""
    path = cache_dir / f"{row['strip_id']}.tif"
    valid_path = cache_dir / f"{row['strip_id']}_valid.tif"
    if path.exists() and valid_path.exists():
        with rasterio.open(path) as src:
            mask = src.read(1)
        with rasterio.open(valid_path) as src:
            valid = src.read(1).astype(bool)
        return path, mask, valid
    log(
        f"  mask {row['strip_id']}  θ={row['incidence_deg']:.3f}°  "
        f"look={row['look_azimuth_deg']:.0f}°  ori={row['orientation_flag']}"
    )
    z, valid_1m, t_1m = prepare_dsm_for_strip(ahn_path, row)
    mask_1m = mask_from_dsm(z, row["incidence_deg"], row["look_azimuth_deg"], DSM_POSTING)
    mask = warp_to_strip(mask_1m, t_1m, row["crs"], row, Resampling.nearest, np.uint8)
    valid = warp_to_strip(
        valid_1m.astype(np.uint8), t_1m, row["crs"], row, Resampling.nearest, np.uint8
    ).astype(bool)
    write_mask_geotiff(path, mask, row)
    write_mask_geotiff(valid_path, valid.astype(np.uint8), row)
    return path, mask, valid


def class_fractions(mask: np.ndarray) -> dict[str, float]:
    n = mask.size
    return {
        "frac_nominal": float((mask == CLASS_NOMINAL).mean()) if n else 0,
        "frac_active": float((mask == CLASS_ACTIVE).mean()) if n else 0,
        "frac_passive": float((mask == CLASS_PASSIVE).mean()) if n else 0,
        "frac_shadow": float((mask == CLASS_SHADOW).mean()) if n else 0,
        "frac_layover": float(((mask == CLASS_ACTIVE) | (mask == CLASS_PASSIVE)).mean()) if n else 0,
    }


def sar_rgb(image: np.ndarray) -> Image.Image:
    gray = percentile_stretch(image[0])
    return Image.fromarray(np.stack([gray, gray, gray], axis=-1))


def overlay_rgb(image: np.ndarray, mask: np.ndarray) -> Image.Image:
    hh = image[0]
    gray = percentile_stretch(hh)
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
    rgb[mask == CLASS_PASSIVE] = (1 - 0.45) * rgb[mask == CLASS_PASSIVE] + 0.45 * PASSIVE_RGB
    rgb[mask == CLASS_ACTIVE] = (1 - 0.55) * rgb[mask == CLASS_ACTIVE] + 0.55 * ACTIVE_RGB
    rgb[mask == CLASS_SHADOW] = (1 - 0.42) * rgb[mask == CLASS_SHADOW] + 0.42 * SHADOW_RGB
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def overlay_tile(image: np.ndarray, mask: np.ndarray, path: Path, title: str) -> None:
    img = overlay_rgb(image, mask)
    draw = ImageDraw.Draw(img)
    draw.text((8, TILE_SIZE - 18), title, fill=(232, 230, 225))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "PNG", optimize=True)


def plot_split_map(grid: BlockGrid, mapping: dict, tiles: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    colours = {"train": "#8b9098", "val": "#b7c4ce", "test": "#d4783a"}
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#0b0c0e")
    ax.set_facecolor("#0b0c0e")
    for block, split in mapping.items():
        west, south, east, north = grid.block_bounds(block)
        ax.add_patch(
            Rectangle(
                (west, south),
                east - west,
                north - south,
                facecolor=colours[split],
                edgecolor="#e8e6e1",
                linewidth=0.2,
                alpha=0.7,
            )
        )
    for rec in tiles:
        w, s, e, n = rec["bbox"]
        ax.add_patch(
            Rectangle((w, s), e - w, n - s, fill=False, edgecolor="#e8e6e1", linewidth=0.2, alpha=0.35)
        )
    ax.set_aspect("equal")
    ax.tick_params(colors="#8b9098")
    for spine in ax.spines.values():
        spine.set_color("#8b9098")
    ax.set_title("1 km blocks  ·  west train → val → east test", color="#e8e6e1")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_layover_hist(values: list[float], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3.4), facecolor="#0b0c0e")
    ax.set_facecolor("#0b0c0e")
    ax.hist(values, bins=20, range=(0, 1), color="#d4783a", edgecolor="#0b0c0e")
    ax.set_xlabel("per-tile layover fraction", color="#e8e6e1")
    ax.set_ylabel("tiles", color="#e8e6e1")
    ax.tick_params(colors="#8b9098")
    for spine in ax.spines.values():
        spine.set_color("#8b9098")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)


def write_report(stats: dict, path: Path) -> None:
    lines = [
        "# Training dataset report",
        "",
        "## Geometry (verified per strip)",
        "",
        f"- Incidence field: `{stats['incidence_field']}` — from vertical, not grazing.",
        f"- Incidence range: {stats['incidence_min']:.3f}° – {stats['incidence_max']:.3f}° (90−x would be {90-stats['incidence_max']:.3f}° – {90-stats['incidence_min']:.3f}°).",
        f"- Look azimuth: orientation flag 0 → 0° (north), 1 → 180° (south). Canonicalise={'on' if stats['canonicalise'] else 'off'}.",
        "",
        "## Tiles",
        "",
        f"- Tile size: {TILE_SIZE} px ({TILE_M:.0f} m) at {GSD_M} m.",
        f"- Train stride {TRAIN_STRIDE} px (50% overlap). Val/test stride {EVAL_STRIDE} px (no overlap).",
        f"- Strips processed: {stats['n_strips']} (north {stats['n_north']}, south {stats['n_south']}).",
        f"- Masks cached: {stats.get('n_masks', stats['n_strips'])}.",
        "",
        "| split | tiles |",
        "|-------|------:|",
    ]
    for split in ("train", "val", "test"):
        lines.append(f"| {split} | {stats['counts'][split]} |")
    lines += [
        "",
        "## Per-class pixel fraction",
        "",
        "| split | nominal | active | passive | shadow | any layover |",
        "|-------|--------:|-------:|--------:|-------:|------------:|",
    ]
    for split in ("train", "val", "test"):
        f = stats["class_frac"][split]
        lines.append(
            f"| {split} | {f['frac_nominal']:.4f} | {f['frac_active']:.4f} | "
            f"{f['frac_passive']:.4f} | {f['frac_shadow']:.4f} | {f['frac_layover']:.4f} |"
        )
    lines += [
        "",
        "## Dropped tiles",
        "",
        "| reason | count |",
        "|--------|------:|",
    ]
    for reason, n in sorted(stats["dropped"].items()):
        lines.append(f"| {reason} | {n} |")
    lines += [
        "",
        f"- Train/test bbox intersection: **none** (asserted).",
        f"- Log domain: `{stats['log_kind']}` (MAG-POL magnitude is stored as dB; linear bands would get log10(x+eps)).",
        f"- Normalisation JSON written from the **train** split only.",
        "",
        "## Layover fraction per tile",
        "",
        f"- min {stats['layover_min']:.4f}  median {stats['layover_med']:.4f}  max {stats['layover_max']:.4f}",
        "",
        "Twenty overlay PNGs are in `dataset/overlays/`. Copper = active layover, umber = passive, blue = shadow.",
        "Canonicalise check: `dataset/overlays/canonicalise_pair.png` — south as-recorded, south flipped, north unflipped. Layover must point the same way in the last two.",
        "",
        "## Strips",
        "",
        "| strip | look | θ (vertical) | 90−x |",
        "|-------|-----:|-------------:|-----:|",
    ]
    for s in stats.get("strips", []):
        lines.append(
            f"| {s['strip_id']} | {s['look_azimuth_deg']:.0f}° | {s['incidence_deg']:.3f}° | {s['incidence_90_minus_raw_deg']:.3f}° |"
        )
    lines.append("")
    n_cat = stats.get("n_catalog")
    if n_cat:
        lines += [
            "## Coverage",
            "",
            f"- Catalogued MAG-POL strips: {n_cat} (north {stats.get('n_catalog_north', '?')}, south {stats.get('n_catalog_south', '?')}).",
            f"- Masks cached: {stats.get('n_masks', 0)}.",
            "- Tiling all 202 would be ~70–115 GB of compressed tiles. This run tiled a north+south pair on the same ground so canonicalise can be checked. `python dataset.py --masks-only` caches a mask for every strip without reading SAR; `--fetch-sar --release-sar` tiles the rest one strip at a time.",
            "",
        ]
    path.write_text("\n".join(lines) + "\n")


def copy_public(set_dir: Path, extra: dict | None = None) -> None:
    PUBLIC.mkdir(parents=True, exist_ok=True)
    for name in ("REPORT.md", "manifest.csv", "norm_stats.json", "split_map.png", "layover_hist.png"):
        src = set_dir / name
        if src.exists():
            shutil.copy2(src, PUBLIC / name)
    ov = set_dir / "overlays"
    dest = PUBLIC / "overlays"
    dest.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        for old in dest.glob("*.png"):
            if ov.exists() and (ov / old.name).exists():
                continue
            # drop stale overlays not regenerated this run
        for png in dest.glob("*.png"):
            if not ov.exists() or not (ov / png.name).exists():
                png.unlink()
    if ov.exists():
        for png in ov.glob("*.png"):
            shutil.copy2(png, dest / png.name)
    bundled = ROOT.parent / "src" / "lib" / "dataset-report.json"
    report = extra.copy() if extra else {}
    report.update(
        {
            "report_markdown": (set_dir / "REPORT.md").read_text() if (set_dir / "REPORT.md").exists() else "",
            "norm": json.loads((set_dir / "norm_stats.json").read_text())
            if (set_dir / "norm_stats.json").exists()
            else {},
            "overlays": [
                f"/dataset/overlays/{p.name}"
                for p in sorted((set_dir / "overlays").glob("*.png"))
                if not p.name.startswith("canonicalise") and not p.name.endswith("-sar.png")
            ]
            if (set_dir / "overlays").exists()
            else [],
            "split_map": "/dataset/split_map.png",
            "layover_hist": "/dataset/layover_hist.png" if (set_dir / "layover_hist.png").exists() else None,
            "canonicalise_pair": "/dataset/overlays/canonicalise_pair.png"
            if (set_dir / "overlays" / "canonicalise_pair.png").exists()
            else None,
            "manifest": "/dataset/manifest.csv",
        }
    )
    if (set_dir / "manifest.csv").exists():
        counts = {"train": 0, "val": 0, "test": 0}
        with (set_dir / "manifest.csv").open() as fh:
            for rec in csv.DictReader(fh):
                counts[rec["split"]] = counts.get(rec["split"], 0) + 1
        report["counts"] = counts
    bundled.write_text(json.dumps(report, indent=2))


def select_strips(rows: list[dict], one_strip: bool, max_strips: int | None, strip_ids: list[str] | None):
    if strip_ids:
        want = set(strip_ids)
        picked = [r for r in rows if r["strip_id"] in want]
        missing = want - {r["strip_id"] for r in picked}
        if missing:
            raise SystemExit(f"unknown strip ids: {missing}")
        return picked
    if one_strip:
        for r in rows:
            if r["strip_id"] == LOCAL_KEY:
                return [r]
        return rows[:1]
    if max_strips is None or max_strips >= len(rows):
        return rows
    ranked = sorted(rows, key=lambda r: (r["bounds"][0] + r["bounds"][2]) / 2)
    step = max(1, len(ranked) // max_strips)
    picked = ranked[::step][:max_strips]
    return picked


def local_sar_path(strip_id: str) -> Path:
    from download import magpol_local_path

    return magpol_local_path(strip_id)


def ensure_strip(strip_id: str, fetch: bool) -> None:
    path = local_sar_path(strip_id)
    if path.exists() and path.stat().st_size > 1_000_000:
        return
    if not fetch:
        return
    from download import download_strip

    log(f"  fetch {strip_id}")
    download_strip(strip_id, path)


def release_strip(strip_id: str) -> None:
    if strip_id in KEEP_LOCAL:
        return
    path = local_sar_path(strip_id)
    if path.exists():
        log(f"  release {path.name}")
        path.unlink()


def reset_tile_dir(set_dir: Path) -> None:
    for split in ("train", "val", "test"):
        d = set_dir / split
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    ov = set_dir / "overlays"
    if ov.exists():
        shutil.rmtree(ov)


def write_one_tile(rec: dict, row: dict, set_dir: Path) -> dict:
    tile_id = f"{rec['strip_id']}_{rec['row']:05d}_{rec['col']:05d}"
    look = np.full((TILE_SIZE, TILE_SIZE), row["orientation_flag"], dtype=np.uint8)
    out = set_dir / rec["split"] / f"{tile_id}.npz"
    np.savez_compressed(
        out,
        image=rec["image"].astype(np.float32),
        mask=rec["mask"].astype(np.uint8),
        look=look,
        meta=np.array(
            json.dumps(
                {
                    "strip_id": rec["strip_id"],
                    "bbox": rec["bbox"],
                    "crs": row["crs"],
                    "original_orientation": row["orientation_flag"],
                    "look_azimuth_deg": row["look_azimuth_deg"],
                    "flipped": rec["flipped"],
                    "incidence_deg": row["incidence_deg"],
                    "incidence_field": row["incidence_field"],
                    "incidence_raw_deg": row["incidence_raw_deg"],
                }
            )
        ),
    )
    entry = {
        "tile_id": tile_id,
        "split": rec["split"],
        "source_strip": rec["strip_id"],
        "bbox_west": rec["bbox"][0],
        "bbox_south": rec["bbox"][1],
        "bbox_east": rec["bbox"][2],
        "bbox_north": rec["bbox"][3],
        "orientation_flag": row["orientation_flag"],
        "look_azimuth_deg": row["look_azimuth_deg"],
        "incidence_deg": row["incidence_deg"],
        "flipped": rec["flipped"],
        **rec["frac"],
        "path": str(out.relative_to(set_dir)),
    }
    rec["tile_id"] = tile_id
    rec["path"] = out
    rec["orientation_flag"] = row["orientation_flag"]
    rec.pop("image", None)
    rec.pop("mask", None)
    return entry


class RunningNorm:
    def __init__(self) -> None:
        self.sums = np.zeros(4, dtype=np.float64)
        self.sq = np.zeros(4, dtype=np.float64)
        self.n = 0

    def update(self, image: np.ndarray, split: str) -> None:
        if split != "train":
            return
        x = image.reshape(4, -1)
        finite = np.isfinite(x)
        self.sums += np.where(finite, x, 0).sum(axis=1)
        self.sq += np.where(finite, x**2, 0).sum(axis=1)
        self.n += int(finite[0].sum())

    def dump(self) -> dict:
        mean = (self.sums / max(self.n, 1)).tolist()
        var = self.sq / max(self.n, 1) - np.array(mean) ** 2
        std = np.sqrt(np.clip(var, 1e-12, None)).tolist()
        return {
            "bands": ["HH", "HV", "VH", "VV"],
            "mean": mean,
            "std": std,
            "n_pixels": self.n,
            "split": "train",
            "note": "computed after log-domain conversion; apply in the training loop",
        }


def process_strip_tiles(
    row: dict,
    mask: np.ndarray,
    valid: np.ndarray,
    grid: BlockGrid,
    block_split: dict,
    canonicalise_flag: bool,
    set_dir: Path,
    norm: RunningNorm,
    cap_zero: float = 1.0,
) -> tuple[list[dict], dict[str, int], str]:
    dropped = {"sar_nodata": 0, "dsm_nodata": 0, "block_boundary": 0, "zero_layover": 0, "no_split": 0}
    kept: list[dict] = []
    transform = affine_from_list(row["transform"])
    log_kind = "db"
    with open_strip(row["strip_id"]) as src:
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
                img, kind = log_scale(img)
                log_kind = kind
                img, m, flipped = canonicalise(img, m, row["orientation_flag"], canonicalise_flag)
                reason = drop_reason(img, m, v, cap_zero=1.0)
                if reason:
                    dropped[reason] += 1
                    continue
                frac = class_fractions(m)
                rec = {
                    "row": gr,
                    "col": c,
                    "bbox": bbox,
                    "split": split,
                    "image": img,
                    "mask": m,
                    "flipped": flipped,
                    "frac": frac,
                    "strip_id": row["strip_id"],
                }
                norm.update(img, split)
                write_one_tile(rec, row, set_dir)
                kept.append(rec)
    return kept, dropped, log_kind


def cap_zero_layover(tiles: list[dict], cap: float, rng: np.random.Generator) -> tuple[list[dict], int]:
    if cap >= 1.0:
        return tiles, 0
    zeros = [t for t in tiles if t["frac"]["frac_layover"] == 0.0]
    rest = [t for t in tiles if t["frac"]["frac_layover"] > 0.0]
    max_zeros = int(math.floor(cap / max(1e-9, 1 - cap) * len(rest))) if cap < 1 else len(zeros)
    rng.shuffle(zeros)
    dropped_list = zeros[max_zeros:]
    for rec in dropped_list:
        path = rec.get("path")
        if path and Path(path).exists():
            Path(path).unlink()
    return rest + zeros[:max_zeros], len(dropped_list)


def disk_used_gb(path: Path) -> float:
    total = 0
    if not path.exists():
        return 0.0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total / (1024**3)


def render_samples(tiles: list[dict], set_dir: Path, n: int = 20) -> list[dict]:
    rng = np.random.default_rng(0)
    by_split = {"train": [], "val": [], "test": []}
    for t in tiles:
        by_split[t["split"]].append(t)
    picked = []
    remaining = n
    for split in ("train", "val", "test"):
        take = min(len(by_split[split]), max(1, remaining // 3))
        if by_split[split]:
            idx = rng.choice(len(by_split[split]), size=min(take, len(by_split[split])), replace=False)
            picked.extend(by_split[split][i] for i in np.atleast_1d(idx))
    pool = tiles[:]
    rng.shuffle(pool)
    for t in pool:
        if len(picked) >= n:
            break
        if t not in picked:
            picked.append(t)
    ov = set_dir / "overlays"
    ov.mkdir(parents=True, exist_ok=True)
    meta = []
    for i, rec in enumerate(picked[:n]):
        data = np.load(rec["path"])
        image = data["image"]
        mask = data["mask"]
        name = f"{i:02d}_{rec['split']}_{rec['tile_id'][-20:]}.png"
        overlay_tile(
            image,
            mask,
            ov / name,
            f"{rec['split']}  {rec['strip_id'][:15]}  flip={int(rec['flipped'])}",
        )
        sar_name = name.replace(".png", "-sar.png")
        sar_rgb(image).save(ov / sar_name, "PNG", optimize=True)
        meta.append(
            {
                "src": f"/dataset/overlays/{name}",
                "sar_src": f"/dataset/overlays/{sar_name}",
                "split": rec["split"],
                "strip_id": rec["strip_id"],
                "flipped": bool(rec["flipped"]),
                "frac_layover": rec["frac"]["frac_layover"],
                "orientation_flag": rec.get("orientation_flag"),
            }
        )
    return meta


def render_canonicalise_pair(tiles: list[dict], set_dir: Path) -> str | None:
    south = [t for t in tiles if t.get("orientation_flag") == 1 and t["frac"]["frac_layover"] >= 0.12]
    north = [t for t in tiles if t.get("orientation_flag") == 0 and t["frac"]["frac_layover"] >= 0.12]
    south.sort(key=lambda t: -t["frac"]["frac_layover"])
    north.sort(key=lambda t: -t["frac"]["frac_layover"])
    if not south:
        return None
    s = south[0]
    data = np.load(s["path"])
    flipped_img, flipped_mask = data["image"], data["mask"]
    raw_img = np.flip(flipped_img, axis=-2)
    raw_mask = np.flip(flipped_mask, axis=-2)
    panels = [
        (overlay_rgb(raw_img, raw_mask), "south as recorded"),
        (overlay_rgb(flipped_img, flipped_mask), "south flipped (canonical)"),
    ]
    if north:
        n = north[0]
        nd = np.load(n["path"])
        panels.append((overlay_rgb(nd["image"], nd["mask"]), "north unflipped (canonical)"))
    gap = 8
    w = TILE_SIZE
    h = TILE_SIZE + 28
    canvas = Image.new("RGB", (len(panels) * w + (len(panels) - 1) * gap, h), (11, 12, 14))
    draw = ImageDraw.Draw(canvas)
    for i, (im, title) in enumerate(panels):
        x = i * (w + gap)
        canvas.paste(im, (x, 0))
        draw.text((x + 8, TILE_SIZE + 6), title, fill=(232, 230, 225))
    path = set_dir / "overlays" / "canonicalise_pair.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, "PNG", optimize=True)
    return str(path)


def write_masks_index(cache_dir: Path, rows: list[dict]) -> None:
    path = cache_dir / "INDEX.csv"
    fields = [
        "strip_id",
        "orientation_flag",
        "look_azimuth_deg",
        "incidence_deg",
        "incidence_field",
        "path",
        "valid_path",
    ]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in rows:
            mask_p = cache_dir / f"{row['strip_id']}.tif"
            valid_p = cache_dir / f"{row['strip_id']}_valid.tif"
            if not mask_p.exists():
                continue
            w.writerow(
                {
                    "strip_id": row["strip_id"],
                    "orientation_flag": row["orientation_flag"],
                    "look_azimuth_deg": row["look_azimuth_deg"],
                    "incidence_deg": row["incidence_deg"],
                    "incidence_field": row["incidence_field"],
                    "path": str(mask_p.relative_to(OUTPUT)),
                    "valid_path": str(valid_p.relative_to(OUTPUT)) if valid_p.exists() else "",
                }
            )


def generate_all_masks(rows: list[dict], ahn_path: Path) -> int:
    import gc

    MASK_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for i, row in enumerate(rows, 1):
        log(f"=== mask {i}/{len(rows)} {row['strip_id']} ===")
        try:
            generate_mask(row, ahn_path, MASK_DIR)
            n += 1
        except Exception as exc:
            log(f"  SKIP mask {row['strip_id']}: {exc}")
        gc.collect()
        if i % 5 == 0 or i == len(rows):
            write_masks_index(MASK_DIR, rows)
    write_masks_index(MASK_DIR, rows)
    log(f"=== masks cached {n}/{len(rows)} in {MASK_DIR} ===")
    bundled = ROOT.parent / "src" / "lib" / "dataset-report.json"
    if bundled.exists():
        data = json.loads(bundled.read_text())
        data["n_masks"] = n
        bundled.write_text(json.dumps(data, indent=2))
    return n


def run(args: argparse.Namespace) -> None:
    rows = load_catalog()
    picked = select_strips(rows, args.one_strip, args.max_strips, args.strips)
    log(f"=== strips {len(picked)} / {len(rows)}  canonicalise={args.canonicalise} ===")
    for r in picked:
        log(
            f"  {r['strip_id']}  ori={r['orientation_flag']} look={r['look_azimuth_deg']:.0f} "
            f"θ={r['incidence_deg']:.3f}° field={r['incidence_field']}"
        )

    ahn_path = Path(args.dsm) if args.dsm else default_dsm()
    if not ahn_path.exists():
        raise SystemExit(f"AHN DSM missing: {ahn_path}")
    log(f"  dsm {ahn_path}")

    if args.masks_only:
        generate_all_masks(picked, ahn_path)
        return

    west, south, east, north = union_bounds(picked)
    grid = BlockGrid(
        origin_x=math.floor(west / BLOCK_M) * BLOCK_M,
        origin_y=math.floor(south / BLOCK_M) * BLOCK_M,
        size=BLOCK_M,
    )
    covered = set()
    for r in picked:
        b = r["bounds"]
        ix0, iy0 = grid.block_id(b[0] + 1, b[1] + 1)
        ix1, iy1 = grid.block_id(b[2] - 1, b[3] - 1)
        for ix in range(min(ix0, ix1), max(ix0, ix1) + 1):
            for iy in range(min(iy0, iy1), max(iy0, iy1) + 1):
                covered.add((ix, iy))
    block_split = assign_blocks(sorted(covered), (args.train_frac, args.val_frac, args.test_frac))
    log(
        f"  blocks {len(block_split)}  "
        + ", ".join(f"{s}={sum(v == s for v in block_split.values())}" for s in ("train", "val", "test"))
    )

    MASK_DIR.mkdir(parents=True, exist_ok=True)
    SET_DIR.mkdir(parents=True, exist_ok=True)
    reset_tile_dir(SET_DIR)
    all_tiles: list[dict] = []
    dropped_total = {"sar_nodata": 0, "dsm_nodata": 0, "block_boundary": 0, "zero_layover": 0, "no_split": 0}
    log_kind = "db"
    rng = np.random.default_rng(args.seed)
    norm = RunningNorm()
    rows_by_id = {r["strip_id"]: r for r in picked}
    n_masks = 0

    for i, row in enumerate(picked, 1):
        if args.disk_budget_gb and disk_used_gb(SET_DIR) >= args.disk_budget_gb:
            log(f"  disk budget {args.disk_budget_gb} GB reached, stopping")
            break
        log(f"=== {i}/{len(picked)} {row['strip_id']} ===")
        ensure_strip(row["strip_id"], fetch=args.fetch_sar)
        _, mask, valid = generate_mask(row, ahn_path, MASK_DIR)
        n_masks += 1
        tiles, dropped, kind = process_strip_tiles(
            row, mask, valid, grid, block_split, args.canonicalise, SET_DIR, norm, cap_zero=1.0
        )
        del mask, valid
        log_kind = kind
        for k, v in dropped.items():
            dropped_total[k] = dropped_total.get(k, 0) + v
        all_tiles.extend(tiles)
        log(f"  kept {len(tiles)}  dropped {dropped}")
        if args.release_sar:
            release_strip(row["strip_id"])

    all_tiles, n_cap = cap_zero_layover(all_tiles, args.cap_zero_layover, rng)
    dropped_total["zero_layover"] += n_cap

    if not all_tiles:
        raise SystemExit("FAIL: no tiles kept")

    manifest = []
    for rec in all_tiles:
        row = rows_by_id[rec["strip_id"]]
        manifest.append(
            {
                "tile_id": rec["tile_id"],
                "split": rec["split"],
                "source_strip": rec["strip_id"],
                "bbox_west": rec["bbox"][0],
                "bbox_south": rec["bbox"][1],
                "bbox_east": rec["bbox"][2],
                "bbox_north": rec["bbox"][3],
                "orientation_flag": row["orientation_flag"],
                "look_azimuth_deg": row["look_azimuth_deg"],
                "incidence_deg": row["incidence_deg"],
                "flipped": rec["flipped"],
                **rec["frac"],
                "path": str(Path(rec["path"]).relative_to(SET_DIR)),
            }
        )

    train_b = [t["bbox"] for t in all_tiles if t["split"] == "train"]
    test_b = [t["bbox"] for t in all_tiles if t["split"] == "test"]
    assert_no_train_test_overlap(train_b, test_b)
    log("  train/test bbox overlap: none")

    man_path = SET_DIR / "manifest.csv"
    fields = list(manifest[0].keys())
    with man_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(manifest)

    counts = {s: sum(1 for t in all_tiles if t["split"] == s) for s in ("train", "val", "test")}
    class_frac = {}
    for split in ("train", "val", "test"):
        acc = {k: 0.0 for k in ("frac_nominal", "frac_active", "frac_passive", "frac_shadow", "frac_layover")}
        n = 0
        for t in all_tiles:
            if t["split"] != split:
                continue
            for k in acc:
                acc[k] += t["frac"][k]
            n += 1
        class_frac[split] = {k: (acc[k] / n if n else 0.0) for k in acc}

    lay = [t["frac"]["frac_layover"] for t in all_tiles] or [0.0]
    hist, edges = np.histogram(lay, bins=10, range=(0.0, 1.0))
    stats = {
        "incidence_field": picked[0]["incidence_field"],
        "incidence_min": min(r["incidence_deg"] for r in picked),
        "incidence_max": max(r["incidence_deg"] for r in picked),
        "canonicalise": args.canonicalise,
        "n_strips": len({t["strip_id"] for t in all_tiles}),
        "n_north": sum(1 for r in picked if r["orientation_flag"] == 0),
        "n_south": sum(1 for r in picked if r["orientation_flag"] == 1),
        "n_masks": n_masks,
        "n_catalog": len(rows),
        "n_catalog_north": sum(1 for r in rows if r["orientation_flag"] == 0),
        "n_catalog_south": sum(1 for r in rows if r["orientation_flag"] == 1),
        "counts": counts,
        "class_frac": class_frac,
        "dropped": dropped_total,
        "layover_min": float(min(lay)),
        "layover_med": float(np.median(lay)),
        "layover_max": float(max(lay)),
        "log_kind": log_kind,
        "strips": [
            {
                "strip_id": r["strip_id"],
                "orientation_flag": r["orientation_flag"],
                "look_azimuth_deg": r["look_azimuth_deg"],
                "incidence_deg": r["incidence_deg"],
                "incidence_90_minus_raw_deg": r["incidence_90_minus_raw_deg"],
                "incidence_field": r["incidence_field"],
            }
            for r in picked
        ],
    }
    dumped = norm.dump()
    dumped["log_kind"] = log_kind
    (SET_DIR / "norm_stats.json").write_text(json.dumps(dumped, indent=2))
    write_report(stats, SET_DIR / "REPORT.md")
    plot_split_map(grid, block_split, all_tiles, SET_DIR / "split_map.png")
    plot_layover_hist(lay, SET_DIR / "layover_hist.png")
    overlay_meta = render_samples(all_tiles, SET_DIR, n=20)
    pair = render_canonicalise_pair(all_tiles, SET_DIR)
    extra = {
        **stats,
        "norm": dumped,
        "overlay_meta": overlay_meta,
        "train_test_overlap": False,
        "layover_hist_bins": [
            {"lo": float(edges[i]), "hi": float(edges[i + 1]), "n": int(hist[i])} for i in range(len(hist))
        ],
        "canonicalise_pair_on_disk": bool(pair),
    }
    copy_public(SET_DIR, extra=extra)
    log(f"=== done tiles={sum(counts.values())} {counts} ===")
    log(f"  {SET_DIR / 'REPORT.md'}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--one-strip", action="store_true", help="Process the local MAG-POL strip only.")
    p.add_argument("--max-strips", type=int, default=None)
    p.add_argument("--strips", nargs="*", default=None)
    p.add_argument("--dsm", default=None, help="AHN GeoTIFF. Default: AOI mosaic if present, else one-strip DSM.")
    p.add_argument("--canonicalise", dest="canonicalise", action="store_true", default=True)
    p.add_argument("--no-canonicalise", dest="canonicalise", action="store_false")
    p.add_argument("--cap-zero-layover", type=float, default=1.0)
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--masks-only", action="store_true", help="Cache per-strip DSM masks, no tiling.")
    p.add_argument("--fetch-sar", action="store_true", default=False, help="Download missing MAG-POL from S3.")
    p.add_argument("--release-sar", action="store_true", help="Delete fetched MAG-POL after tiling (keeps the original local strip).")
    p.add_argument("--disk-budget-gb", type=float, default=20.0)
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
