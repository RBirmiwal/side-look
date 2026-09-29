"""Synthetic check of look azimuth derived from positions, not a flag."""

from __future__ import annotations

import numpy as np

from acquire_geometry import (
    angular_separation_deg,
    ecef_to_geodetic,
    enu_offset_to_ecef,
    geodetic_to_ecef,
    geometry_from_positions,
)


def test_platform_due_north_at_45_deg_elevation():
    lat, lon, height = 51.92, 4.48, 0.0
    target = geodetic_to_ecef(lat, lon, height)
    # 1000 m north and 1000 m up: elevation 45°, so incidence from vertical is 45°.
    platform = target + enu_offset_to_ecef(0.0, 1000.0, 1000.0, lat, lon)
    out = geometry_from_positions(target, platform)
    assert angular_separation_deg(out["bearing_to_sensor_deg"], 0.0) < 1e-8
    assert angular_separation_deg(out["look_azimuth_deg"], 180.0) < 1e-8
    assert abs(out["incidence_deg"] - 45.0) < 1e-8
    back_lat, back_lon, back_h = ecef_to_geodetic(target)
    assert abs(back_lat - lat) < 1e-8
    assert abs(back_lon - lon) < 1e-8
    assert abs(back_h - height) < 1e-4
    assert np.isfinite(out["slant_range_m"])
