#!/usr/bin/env python3
"""Compare a SNAP layover/shadow GeoTIFF to the near-field mask.

Not run when staging prints NOT READY. SNAP codes, if a product is produced:

    0 neither, 1 layover, 2 shadow, 3 layover and shadow

Near-field codes stay 0 nominal, 1 active, 2 passive, 3 shadow.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject


def _section(title: str) -> None:
    print(f"\n## {title}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--magpol", type=Path, required=True)
    parser.add_argument("--snap", type=Path, required=True)
    parser.add_argument("--ours", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    with rasterio.open(args.magpol) as src:
        mag_crs, mag_transform = src.crs, src.transform
        mag_shape = (src.height, src.width)
    with rasterio.open(args.snap) as src:
        snap = src.read(1)
        snap_crs, snap_transform = src.crs, src.transform
        snap_shape = (src.height, src.width)
        snap_names = list(src.descriptions)
    same_grid = (
        snap_shape == mag_shape
        and snap_crs == mag_crs
        and snap_transform.almost_equals(mag_transform, precision=1e-3)
    )
    lines = []
    lines.append("# SNAP layover/shadow vs near-field\n")
    lines.append("## 1. Alignment\n")
    if not same_grid:
        lines.append("NOT ALIGNED\n")
        lines.append(
            f"SNAP grid {snap_shape} {snap_crs} is not the MAG-POL grid {mag_shape} {mag_crs}.\n"
        )
        lines.append(f"SNAP bands: {snap_names}\n")
        lines.append("\n## 2. Class agreement\n")
        lines.append("Not interpreted. Section 1 is NOT ALIGNED.\n")
        lines.append("\n## 3. Brightness\n")
        lines.append("Not interpreted. Section 1 is NOT ALIGNED.\n")
    else:
        with rasterio.open(args.ours) as src:
            ours = np.zeros(mag_shape, dtype=np.uint8)
            reproject(
                src.read(1),
                ours,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=mag_transform,
                dst_crs=mag_crs,
                resampling=Resampling.nearest,
            )
        lines.append("ALIGNED\n")
        lines.append(f"SNAP bands: {snap_names}\n")
        valid = snap >= 0
        lines.append("\n## 2. Class agreement\n")
        for code, name in ((1, "layover"), (2, "shadow"), (3, "both")):
            sel = valid & (snap == code)
            lines.append(
                f"SNAP {code} {name}: {int(sel.sum())} px, "
                f"our modal class {int(np.bincount(ours[sel]).argmax()) if sel.any() else -1}\n"
            )
        lines.append("\n## 3. Brightness\n")
        lines.append("HH contrast is computed only after alignment. Grid matches; "
                     "see the rendered crops for the sign of shadow.\n")
    args.out.write_text("".join(lines))
    print(args.out.read_text())


if __name__ == "__main__":
    main()
