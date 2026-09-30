"""Closed-form near-field layover and shadow. No imagery."""

from __future__ import annotations

import math

import numpy as np

from dsm_geometry import CLASS_ACTIVE, CLASS_PASSIVE, CLASS_SHADOW, scan_range_lines
from near_field import (
    canonical_flip,
    cardinal_look,
    fit_flight_line,
    layover_extent_m,
    look_azimuth_from_track,
    profile_slant_alpha,
    scan_near_field,
    shadow_extent_m,
)


def _box(cols, c0, c1, height):
    z = np.zeros((4, cols), dtype=np.float64)
    z[:, c0:c1] = height
    return z


def test_right_look_of_an_eastward_track_is_south():
    direction = np.array([1.0, 0.0, 0.0])
    assert look_azimuth_from_track(direction, "right") == 180.0
    assert look_azimuth_from_track(np.array([0.0, 1.0, 0.0]), "right") == 90.0


def test_canonical_flip_puts_the_sensor_on_row_zero():
    assert canonical_flip(180.0) is False
    assert canonical_flip(180.18) is False
    assert canonical_flip(0.0) is True
    assert canonical_flip(359.6) is True


def test_cardinal_look_skips_anything_past_five_degrees():
    assert cardinal_look(180.4) == 180.0
    assert cardinal_look(359.2) == 0.0
    assert cardinal_look(5.0) == 0.0
    assert cardinal_look(5.1) is None
    assert cardinal_look(90.0) is None


def test_flight_line_residuals_on_a_straight_track():
    t = np.linspace(0, 1000, 21)
    pts = np.stack([t, np.zeros_like(t), np.full_like(t, 1600.0)], axis=1)
    fit = fit_flight_line(pts)
    assert fit["residual_rms_m"] < 1e-6
    assert fit["residual_max_m"] < 1e-6
    assert fit["height_above_origin_m"] == 1600.0
    assert fit["direction_enu"][0] > 0.0


def test_box_extents_match_closed_form():
    """One box at a known distance. Scan must reproduce the analytic extents."""
    pixel = 1.0
    platform = 400.0
    building = 30.0
    c0, c1, cols = 200, 220, 320
    z = _box(cols, c0, c1, building)
    g = (np.arange(cols) + 0.5) * pixel
    slant, alpha = profile_slant_alpha(z, g, platform)
    mask = scan_near_field(slant, alpha, bin_width_m=0.25)
    line = mask[0]
    # Walls are the pixel edges. Centres sit 0.5 m inside.
    lay_m = layover_extent_m(c0 * pixel, building, platform)
    sh_m = shadow_extent_m(c1 * pixel, building, platform)
    lay = np.where((line == CLASS_ACTIVE) | (line == CLASS_PASSIVE))[0]
    sh = np.where(line == CLASS_SHADOW)[0]
    assert abs(lay.min() - (c0 - lay_m)) < 3.0
    assert abs(sh.max() - ((c1 - 1) + sh_m)) < 3.0
    assert np.any(line[c0:c1] == CLASS_ACTIVE)
    assert np.any(line[:c0] == CLASS_PASSIVE)
    assert not np.any(line[:c0] == CLASS_SHADOW)


def test_far_field_limit_matches_constant_incidence():
    """Very high platform, building at 45°. Near-field scan matches the far-field one."""
    pixel = 1.0
    height = 200_000.0
    building = 20.0
    cols = 80
    c0, c1 = 30, 50
    z = _box(cols, c0, c1, building)
    # Place the near wall at ground range = platform height so incidence is 45°.
    g = height + (np.arange(cols) - c0) * pixel
    slant, alpha = profile_slant_alpha(z, g, height)
    near = scan_near_field(slant, alpha, bin_width_m=pixel * math.sin(math.radians(45.0)))
    far = scan_range_lines(z, 45.0, pixel)
    # Extents, not a pixel-identical raster: the two range definitions differ by < 1 px here.
    for mask in (near, far):
        line = mask[0]
        lay = np.where((line == CLASS_ACTIVE) | (line == CLASS_PASSIVE))[0]
        sh = np.where(line == CLASS_SHADOW)[0]
        assert abs(lay.min() - (c0 - building)) <= 2
        assert abs(sh.max() - ((c1 - 1) + building)) <= 2


def test_layover_is_longer_at_near_range():
    platform = 2000.0
    building = 50.0
    pixel = 1.0

    def extent(g_wall):
        cols = int(g_wall + 80)
        c0 = int(g_wall)
        c1 = c0 + 20
        z = _box(cols, c0, c1, building)
        g = (np.arange(cols) + 0.5) * pixel
        slant, alpha = profile_slant_alpha(z, g, platform)
        line = scan_near_field(slant, alpha, bin_width_m=0.25)[0]
        lay = np.where((line == CLASS_ACTIVE) | (line == CLASS_PASSIVE))[0]
        return c0 - int(lay.min())

    assert extent(600) > extent(1400)
