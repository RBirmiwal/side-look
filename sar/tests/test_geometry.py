"""Closed-form checks for layover / shadow extents on a square building."""

from __future__ import annotations

import math

import pytest
from shapely.geometry import box

from geometry import (
    GeographicCRSError,
    combine_masks,
    ensure_projected_crs,
    ground_range_unit,
    layover_extent_m,
    layover_region,
    look_azimuth_from_orientation,
    shadow_extent_m,
    shadow_region,
)


H = 12.0
SIDE = 8.0
SQUARE = box(0.0, 0.0, SIDE, SIDE)


def _hull_bounds_along_range(geom, look_azimuth_deg: float):
    """Return (min, max) of hull coordinates along the away-from-sensor axis."""
    ux, uy = ground_range_unit(look_azimuth_deg)
    xs, ys = geom.exterior.coords.xy
    dots = [x * ux + y * uy for x, y in zip(xs, ys)]
    return min(dots), max(dots)


@pytest.mark.parametrize("theta", [30.0, 45.0, 60.0])
def test_extents_match_closed_form(theta: float):
    layover = layover_extent_m(H, theta)
    shadow = shadow_extent_m(H, theta)
    assert layover == pytest.approx(H / math.tan(math.radians(theta)))
    assert shadow == pytest.approx(H * math.tan(math.radians(theta)))
    if theta == 45.0:
        assert layover == pytest.approx(H)
        assert shadow == pytest.approx(H)


@pytest.mark.parametrize("theta", [30.0, 45.0, 60.0])
@pytest.mark.parametrize("look_azimuth", [0.0, 90.0, 180.0, 270.0])
def test_square_building_extents(theta: float, look_azimuth: float):
    """Layover extends L from the near edge; shadow extends S from the far edge."""
    L = H / math.tan(math.radians(theta))
    S = H * math.tan(math.radians(theta))
    ux, uy = ground_range_unit(look_azimuth)

    # Square corners projected onto ground-range (away from sensor).
    corners = [(0.0, 0.0), (SIDE, 0.0), (SIDE, SIDE), (0.0, SIDE)]
    dots = [x * ux + y * uy for x, y in corners]
    near_edge = min(dots)  # toward the sensor
    far_edge = max(dots)  # away from the sensor

    lay = layover_region(SQUARE, H, theta, look_azimuth)
    shad = shadow_region(SQUARE, H, theta, look_azimuth)

    lay_min, lay_max = _hull_bounds_along_range(lay, look_azimuth)
    sh_min, sh_max = _hull_bounds_along_range(shad, look_azimuth)

    # Toward-sensor tip of layover is near_edge - L.
    assert lay_min == pytest.approx(near_edge - L, abs=1e-9)
    # Far side of the layover hull still includes the building's far edge.
    assert lay_max == pytest.approx(far_edge, abs=1e-9)

    # Away-from-sensor tip of shadow is far_edge + S.
    assert sh_max == pytest.approx(far_edge + S, abs=1e-9)
    # Near side of the shadow hull still includes the building's near edge.
    assert sh_min == pytest.approx(near_edge, abs=1e-9)


def test_look_azimuth_from_orientation():
    assert look_azimuth_from_orientation(0) == 0.0
    assert look_azimuth_from_orientation(1) == 180.0
    with pytest.raises(ValueError):
        look_azimuth_from_orientation(2)


def test_geographic_crs_is_rejected():
    with pytest.raises(GeographicCRSError):
        ensure_projected_crs("EPSG:4326")
    with pytest.raises(GeographicCRSError):
        ensure_projected_crs(None)
    parsed = ensure_projected_crs("EPSG:28992")
    assert parsed.to_epsg() == 28992


def test_layover_dominates_overlap():
    lay = layover_region(SQUARE, H, 45.0, 0.0)
    shad = shadow_region(SQUARE, H, 45.0, 0.0)
    # Both hulls contain the building, so they overlap there.
    assert lay.intersects(shad)
    lay_u, shad_u = combine_masks([lay], [shad])
    assert shad_u.intersection(lay_u).area == pytest.approx(0.0, abs=1e-8)
