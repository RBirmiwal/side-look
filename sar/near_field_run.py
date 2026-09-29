#!/usr/bin/env python3
"""Near-field mask for one airborne strip. Does not touch output/dataset."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling

from acquire_geometry import ecef_to_geodetic, enu_basis
from dataset import (
    BLOCK_M,
    DSM_POSTING,
    OUTPUT,
    TILE_SIZE,
    _covered_blocks,
    affine_from_list,
    default_dsm,
    load_catalog,
    log,
    prepare_dsm_for_strip,
    process_strip_tiles,
    reset_tile_dir,
    select_strips,
    union_bounds,
    warp_to_strip,
    write_mask_geotiff,
    RunningNorm,
)
from dsm_geometry import CLASS_ACTIVE, CLASS_NOMINAL, CLASS_SHADOW
from near_field import (
    add_undulation,
    fit_flight_line,
    look_azimuth_from_track,
    mask_from_flight,
    south_looking,
    undulation_at,
)
from splits import BlockGrid, assign_blocks

ROOT = Path(__file__).resolve().parent
STRIP = "20190822074237_20190822074521"
GRID_STRIP = "20190804111224_20190804111453"
FAR_FIELD = OUTPUT / "dataset_look180_asread"
OUT = OUTPUT / "dataset_near_field"
MASK_DIR = OUTPUT / "masks_near_field"


def _pixel_spacing(path: Path) -> dict:
    with rasterio.open(path) as src:
        t = src.transform
        return {
            "path": path.name,
            "height": src.height,
            "width": src.width,
            "col_m": math.hypot(t.a, t.d),
            "row_m": math.hypot(t.b, t.e),
            "range_swath_m": src.height * math.hypot(t.b, t.e),
            "along_m": src.width * math.hypot(t.a, t.d),
        }


def _load_collect(path: Path) -> dict:
    with rasterio.open(path) as src:
        return json.loads(src.tags()["TIFFTAG_IMAGEDESCRIPTION"])["collect"]


def _enu_track(collect: dict) -> tuple[np.ndarray, dict]:
    target = np.array(collect["image"]["center_pixel"]["target_position"], dtype=np.float64)
    lat, lon, height = ecef_to_geodetic(target)
    basis = enu_basis(lat, lon)
    pos = np.array([sv["position"] for sv in collect["state"]["state_vectors"]], dtype=np.float64)
    enu = (basis @ (pos - target).T).T
    return enu, {"lat": lat, "lon": lon, "height": height, "target": target}


def _contrast(paths: list[Path], row_lo: int | None = None, row_hi: int | None = None) -> dict:
    sums = {CLASS_NOMINAL: 0.0, CLASS_ACTIVE: 0.0, CLASS_SHADOW: 0.0}
    counts = {CLASS_NOMINAL: 0, CLASS_ACTIVE: 0, CLASS_SHADOW: 0}
    n = 0
    for path in paths:
        row = int(path.stem.split("_")[-2])
        center = row + TILE_SIZE / 2
        if row_lo is not None and center < row_lo:
            continue
        if row_hi is not None and center >= row_hi:
            continue
        z = np.load(path)
        hh = z["image"][0]
        mask = z["mask"]
        finite = np.isfinite(hh)
        n += 1
        for cls in sums:
            sel = finite & (mask == cls)
            sums[cls] += float(hh[sel].sum())
            counts[cls] += int(sel.sum())
    def mean(cls):
        return sums[cls] / counts[cls] if counts[cls] else float("nan")
    nom, act, sh = mean(CLASS_NOMINAL), mean(CLASS_ACTIVE), mean(CLASS_SHADOW)
    return {
        "n_tiles": n,
        "shadow_minus_nominal_db": sh - nom,
        "active_minus_nominal_db": act - nom,
    }


def main() -> None:
    log("=== pixel spacing (GeoTIFF affine) ===")
    for path in (
        ROOT / "data" / "sar_strip.tif",
        ROOT / "data" / f"SN6_AOI_11_Rotterdam_SAR-MAG-POL_{STRIP}.tif",
    ):
        info = _pixel_spacing(path)
        log(
            f"  {info['path']}  {info['height']} x {info['width']}  "
            f"row {info['row_m']:.4f} m  col {info['col_m']:.4f} m  "
            f"range swath {info['range_swath_m']:.2f} m  along-track {info['along_m']:.1f} m"
        )

    rows = load_catalog()
    row = select_strips(rows, False, None, [STRIP])[0]
    sar_path = ROOT / "data" / f"SN6_AOI_11_Rotterdam_SAR-MAG-POL_{STRIP}.tif"
    collect = _load_collect(sar_path)
    enu, geo = _enu_track(collect)
    fit = fit_flight_line(enu)
    pointing = collect["radar"]["pointing"]
    look = look_azimuth_from_track(fit["direction_enu"], pointing)
    # Height of the line where it is broadside to the centre target (ENU origin).
    t_foot = -float(np.dot(fit["origin_enu"], fit["direction_enu"]))
    foot = fit["origin_enu"] + t_foot * fit["direction_enu"]
    log("=== flight line ===")
    log(
        f"  residual rms {fit['residual_rms_m']:.3f} m  max {fit['residual_max_m']:.3f} m  "
        f"n {len(enu)}"
    )
    log(
        f"  mean platform height above centre target {fit['height_above_origin_m']:.2f} m  "
        f"broadside foot height {foot[2]:.2f} m"
    )
    log(f"  radar.pointing {pointing}  geometric look {look:.6f}°  south_looking {south_looking(look)}")
    bin_width = float(collect["image"]["range_resolution"])
    log(f"  slant-range bin {bin_width:.6f} m = collect.image.range_resolution")

    from pyproj import Transformer

    to_rd = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)
    rd_x, rd_y = to_rd.transform(geo["lon"], geo["lat"])
    ahn_path = default_dsm()
    with rasterio.open(ahn_path) as src:
        nap = float(next(src.sample([(rd_x, rd_y)]))[0])
    n_geoid = undulation_at(geo["lon"], geo["lat"])
    log("=== vertical datum ===")
    log(f"  centre target ellipsoidal height {geo['height']:.3f} m")
    log(f"  AHN NAP at that point {nap:.3f} m  NLGEO2018 N {n_geoid:.3f} m")
    log(f"  DSM ellipsoidal (NAP+N) {nap + n_geoid:.3f} m  minus target {nap + n_geoid - geo['height']:.3f} m")

    log("=== DSM + near-field mask ===")
    z_nap, valid, transform = prepare_dsm_for_strip(ahn_path, row)
    z_ell = add_undulation(z_nap, transform, row["crs"])
    mask_1m = mask_from_flight(
        z_ell,
        transform,
        row["crs"],
        look,
        fit["origin_enu"],
        fit["direction_enu"],
        geo["lat"],
        geo["lon"],
        geo["height"],
        bin_width,
    )
    mask = warp_to_strip(mask_1m.astype(np.uint8), transform, row["crs"], row, Resampling.nearest, np.uint8)
    valid_img = warp_to_strip(valid.astype(np.uint8), transform, row["crs"], row, Resampling.nearest, np.uint8).astype(bool)
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    tagged = dict(row)
    tagged["look_azimuth_deg"] = look
    tagged["theta_mode"] = "near-field"
    tagged["incidence_deg"] = float(collect["image"]["center_pixel"]["incidence_angle"])
    write_mask_geotiff(MASK_DIR / f"{STRIP}.tif", mask, tagged)
    log(f"  wrote {MASK_DIR / f'{STRIP}.tif'}  classes " +
        " ".join(f"{c}:{float((mask==c).mean()):.3f}" for c in range(4)))

    grid_rows = select_strips(rows, False, None, [STRIP, GRID_STRIP])
    west, south, east, north = union_bounds(grid_rows)
    grid = BlockGrid(
        origin_x=math.floor(west / BLOCK_M) * BLOCK_M,
        origin_y=math.floor(south / BLOCK_M) * BLOCK_M,
        size=BLOCK_M,
    )
    block_split = assign_blocks(sorted(_covered_blocks(grid, grid_rows)), (0.70, 0.15, 0.15))
    OUT.mkdir(parents=True, exist_ok=True)
    reset_tile_dir(OUT)
    tiles, dropped, _ = process_strip_tiles(
        tagged,
        mask,
        valid_img,
        grid,
        block_split,
        True,
        OUT,
        RunningNorm(),
        cap_zero=1.0,
        flip_override=south_looking(look),
    )
    log(f"  kept {len(tiles)}  dropped {dropped}  flip {south_looking(look)}")
    log(f"  out {OUT}")

    near_paths = sorted(OUT.rglob("*.npz"))
    far_paths = sorted(p for p in FAR_FIELD.rglob("*.npz") if p.name.startswith(STRIP))
    with rasterio.open(sar_path) as src:
        height = src.height
    thirds = {
        "near": (0, height / 3),
        "middle": (height / 3, 2 * height / 3),
        "far": (2 * height / 3, height + 1),
    }
    log("=== PHYSICS HH dB ===")
    for label, paths in (("near-field", near_paths), ("far-field look180 as-read", far_paths)):
        overall = _contrast(paths)
        log(
            f"  {label}  all  shadow-nominal {overall['shadow_minus_nominal_db']:+.3f}  "
            f"active-nominal {overall['active_minus_nominal_db']:+.3f}  tiles {overall['n_tiles']}"
        )
        for name, (lo, hi) in thirds.items():
            part = _contrast(paths, lo, hi)
            log(
                f"  {label}  {name:6}  shadow-nominal {part['shadow_minus_nominal_db']:+.3f}  "
                f"active-nominal {part['active_minus_nominal_db']:+.3f}  tiles {part['n_tiles']}"
            )


if __name__ == "__main__":
    main()
