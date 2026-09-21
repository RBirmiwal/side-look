"""Closed-form DSM scan tests. No real imagery — a synthetic box and ramps."""

from __future__ import annotations

import math

import numpy as np
import pytest

from dsm_geometry import (
    CLASS_ACTIVE,
    CLASS_PASSIVE,
    CLASS_SHADOW,
    mask_from_dsm,
    scan_range_lines,
)


H = 12.0
PIXEL = 1.0
# Box occupies columns [C0, C1) on a flat floor. Rows are uniform.
C0 = 60
C1 = 80
ROWS = 24
COLS = 160


def _box_dsm() -> np.ndarray:
    z = np.zeros((ROWS, COLS), dtype=np.float64)
    z[:, C0:C1] = H
    return z


def _layover_extent(theta: float) -> float:
    return H / math.tan(math.radians(theta))


def _shadow_extent(theta: float) -> float:
    return H * math.tan(math.radians(theta))


@pytest.mark.parametrize("theta", [30.0, 45.0, 60.0])
def test_box_extents_along_range(theta: float):
    """Layover extends L in front of the near wall; shadow S behind the far wall.

    scan_range_lines walks +col as away-from-sensor, so the near wall is C0.
    """
    mask = scan_range_lines(_box_dsm(), theta, PIXEL)
    line = mask[ROWS // 2]
    L = _layover_extent(theta)
    S = _shadow_extent(theta)
    if theta == 45.0:
        assert L == pytest.approx(H)
        assert S == pytest.approx(H)

    lay = np.where((line == CLASS_ACTIVE) | (line == CLASS_PASSIVE))[0]
    sh = np.where(line == CLASS_SHADOW)[0]
    assert lay.size > 0, "expected layover on the box"
    assert sh.size > 0, "expected shadow behind the box"

    # In front of the near wall, not behind the far wall.
    assert lay.min() == pytest.approx(C0 - L, abs=1.5)
    assert lay.max() < C1 + 1
    # Shadow starts at the far wall and reaches S metres behind it.
    assert sh.min() >= C0
    assert sh.max() == pytest.approx((C1 - 1) + S, abs=1.5)

    # Passive layover lives in front of the box; active lives on the roof edge.
    front = line[:C0]
    roof = line[C0:C1]
    behind = line[C1:]
    assert np.any(front == CLASS_PASSIVE)
    assert np.any(roof == CLASS_ACTIVE)
    assert not np.any(front == CLASS_SHADOW)
    assert not np.any(behind == CLASS_ACTIVE)
    assert not np.any(behind == CLASS_PASSIVE)


@pytest.mark.parametrize("theta", [30.0, 45.0, 60.0])
def test_flat_ground_is_clean(theta: float):
    z = np.zeros((16, 64), dtype=np.float64)
    mask = scan_range_lines(z, theta, PIXEL)
    assert np.all(mask == 0)


@pytest.mark.parametrize("theta", [30.0, 45.0, 60.0])
def test_shallow_slope_no_layover(theta: float):
    """Front-facing slope flatter than incidence: r still increases along range."""
    cols = 80
    g = (np.arange(cols) + 0.5) * PIXEL
    slope = 0.5 * math.tan(math.radians(theta))
    z = np.broadcast_to(slope * g, (12, cols)).copy()
    mask = scan_range_lines(z, theta, PIXEL)
    assert not np.any((mask == CLASS_ACTIVE) | (mask == CLASS_PASSIVE))


@pytest.mark.parametrize("theta", [30.0, 45.0, 60.0])
def test_steep_slope_has_layover(theta: float):
    """Front-facing slope steeper than incidence: r decreases, active layover."""
    cols = 80
    g = (np.arange(cols) + 0.5) * PIXEL
    slope = 2.0 * math.tan(math.radians(theta))
    z = np.broadcast_to(slope * g, (12, cols)).copy()
    mask = scan_range_lines(z, theta, PIXEL)
    assert np.any(mask == CLASS_ACTIVE)


@pytest.mark.parametrize("theta", [30.0, 45.0, 60.0])
def test_reverse_azimuth_mirrors(theta: float):
    """Look 90° (east) vs 270° (west): layover/shadow swap sides of the box."""
    z = _box_dsm()
    east = mask_from_dsm(z, theta, 90.0, PIXEL)
    west = mask_from_dsm(z, theta, 270.0, PIXEL)
    line_e = east[ROWS // 2]
    line_w = west[ROWS // 2]

    lay_e = np.where((line_e == CLASS_ACTIVE) | (line_e == CLASS_PASSIVE))[0]
    lay_w = np.where((line_w == CLASS_ACTIVE) | (line_w == CLASS_PASSIVE))[0]
    sh_e = np.where(line_e == CLASS_SHADOW)[0]
    sh_w = np.where(line_w == CLASS_SHADOW)[0]

    # East look: sensor west of the box, layover west, shadow east.
    assert lay_e.min() < C0
    assert sh_e.max() >= C1 - 1
    # West look: the other way around.
    assert lay_w.max() >= C1 - 1
    assert sh_w.min() < C0
    # Pattern is mirrored (sides swap). Pixel-center indexing is not a
    # perfect fliplr because the first cell of a scan has no predecessor.
    assert lay_e.min() < C0 <= lay_w.max()
    assert sh_w.min() < C0 <= sh_e.max()


def test_shadow_outranks_layover():
    """A cell that would be both layover and shadow is coded shadow."""
    # Two boxes: a tall one in front, a short one sitting in its shadow.
    z = np.zeros((12, 100), dtype=np.float64)
    z[:, 30:40] = 20.0
    z[:, 50:55] = 4.0
    mask = scan_range_lines(z, 45.0, PIXEL)
    # The short box is ~10 m behind a 20 m wall → well inside h*tan(45)=20 m shadow.
    assert np.all(mask[:, 50:55] == CLASS_SHADOW)


def test_south_facing_look_uses_rows():
    """look_azimuth=180 (south): layover is north of the box (smaller row index)."""
    z = np.zeros((120, 24), dtype=np.float64)
    r0, r1 = 50, 70
    z[r0:r1, :] = H
    mask = mask_from_dsm(z, 45.0, 180.0, PIXEL)
    col = mask[:, 12]
    lay = np.where((col == CLASS_ACTIVE) | (col == CLASS_PASSIVE))[0]
    sh = np.where(col == CLASS_SHADOW)[0]
    assert lay.min() < r0
    assert sh.max() > r1 - 1
