"""Look azimuth and incidence from platform and target positions.

Does not read the SpaceNet orientation flag or trust the name of an angle
field. Positions are ECEF metres. Incidence is from vertical.

    bearing_to_sensor = atan2(E, N)          compass toward the platform
    look_azimuth      = (bearing_to_sensor + 180) mod 360
    incidence         = atan2(sqrt(E^2 + N^2), U)
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
from pyproj import Transformer

_TO_GEODETIC = Transformer.from_crs("EPSG:4978", "EPSG:4979", always_xy=True)
_TO_ECEF = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)

Vec3 = Sequence[float]


def ecef_to_geodetic(xyz: Vec3) -> tuple[float, float, float]:
    """Return latitude, longitude (degrees) and ellipsoidal height (metres)."""
    lon, lat, height = _TO_GEODETIC.transform(float(xyz[0]), float(xyz[1]), float(xyz[2]))
    return float(lat), float(lon), float(height)


def geodetic_to_ecef(lat_deg: float, lon_deg: float, height_m: float) -> np.ndarray:
    x, y, z = _TO_ECEF.transform(lon_deg, lat_deg, height_m)
    return np.array([x, y, z], dtype=np.float64)


def enu_basis(lat_deg: float, lon_deg: float) -> np.ndarray:
    """Rows are east, north, up unit vectors in ECEF. Shape (3, 3)."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sl, cl = math.sin(lat), math.cos(lat)
    so, co = math.sin(lon), math.cos(lon)
    return np.array(
        [
            [-so, co, 0.0],
            [-sl * co, -sl * so, cl],
            [cl * co, cl * so, sl],
        ],
        dtype=np.float64,
    )


def ecef_offset_to_enu(delta_ecef: Vec3, lat_deg: float, lon_deg: float) -> np.ndarray:
    return enu_basis(lat_deg, lon_deg) @ np.asarray(delta_ecef, dtype=np.float64)


def enu_offset_to_ecef(east: float, north: float, up: float, lat_deg: float, lon_deg: float) -> np.ndarray:
    return enu_basis(lat_deg, lon_deg).T @ np.array([east, north, up], dtype=np.float64)


def _wrap360(deg: float) -> float:
    return float(deg % 360.0)


def geometry_from_positions(target_ecef: Vec3, platform_ecef: Vec3) -> dict:
    """Local geometry of one platform position relative to one target.

    A platform due north of the target at 45° elevation returns
    bearing_to_sensor = 0, look_azimuth = 180, incidence = 45.
    """
    target = np.asarray(target_ecef, dtype=np.float64)
    platform = np.asarray(platform_ecef, dtype=np.float64)
    lat, lon, height = ecef_to_geodetic(target)
    east, north, up = ecef_offset_to_enu(platform - target, lat, lon)
    horizontal = math.hypot(float(east), float(north))
    bearing = _wrap360(math.degrees(math.atan2(float(east), float(north))))
    look = _wrap360(bearing + 180.0)
    incidence = math.degrees(math.atan2(horizontal, float(up)))
    return {
        "lat_deg": lat,
        "lon_deg": lon,
        "height_m": height,
        "east_m": float(east),
        "north_m": float(north),
        "up_m": float(up),
        "slant_range_m": float(np.linalg.norm(platform - target)),
        "ground_range_m": horizontal,
        "bearing_to_sensor_deg": bearing,
        "look_azimuth_deg": look,
        "incidence_deg": incidence,
    }


def mean_platform(state_vectors: Iterable[dict]) -> tuple[np.ndarray, dict]:
    """Mean ECEF position and the spread of the vectors about that mean."""
    positions = np.array([sv["position"] for sv in state_vectors], dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) == 0:
        raise ValueError("state vectors have no ECEF positions")
    mean = positions.mean(axis=0)
    delta = positions - mean
    dist = np.linalg.norm(delta, axis=1)
    return mean, {
        "n": int(len(positions)),
        "max_m": float(dist.max()),
        "rms_m": float(np.sqrt(np.mean(dist**2))),
        "std_xyz_m": delta.std(axis=0).tolist(),
    }


def angular_separation_deg(a: float, b: float) -> float:
    """Smallest absolute difference between two compass bearings, in degrees."""
    d = abs((float(a) - float(b)) % 360.0)
    return float(min(d, 360.0 - d))


def nearest_cardinal_deg(azimuth: float) -> tuple[float, float]:
    """Return (nearest of 0/180, signed error in degrees, -180..180)."""
    az = _wrap360(azimuth)
    candidates = (0.0, 180.0)
    best = min(candidates, key=lambda c: angular_separation_deg(az, c))
    err = (az - best + 180.0) % 360.0 - 180.0
    return best, float(err)
