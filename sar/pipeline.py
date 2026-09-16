#!/usr/bin/env python3
"""Helpers shared by catalog.py and dataset.py.

The box-model inspector (3DBAG extrusions) is omitted here; DSM fold-test
masks in dsm_geometry.py / dataset.py are the training labels.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np


def parse_orientations(path: Path) -> dict[str, int]:
    table: dict[str, int] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, flag = line.split()
        table[key] = int(flag)
    return table


def percentile_stretch(band: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    finite = band[np.isfinite(band)]
    if finite.size == 0:
        return np.zeros(band.shape, dtype=np.uint8)
    p_lo, p_hi = np.percentile(finite, [lo, hi])
    if p_hi <= p_lo:
        return np.zeros(band.shape, dtype=np.uint8)
    scaled = np.clip((band - p_lo) / (p_hi - p_lo), 0, 1)
    scaled[~np.isfinite(band)] = 0
    return (scaled * 255).astype(np.uint8)
