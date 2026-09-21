"""Layover and shadow geometry for a side-looking SAR collect.

Rotterdam is treated as flat: each building is a vertical extrusion of its
footprint, so layover and shadow reduce to two translations in the ground plane.

Ground-range convention (metres, projected CRS):
    look_azimuth_deg  compass bearing the sensor looks toward, clockwise from
                      grid north
    unit vector away from the sensor (ground range):
                      (sin(look_azimuth), cos(look_azimuth))
    layover           translation toward the sensor (negative of that vector)
    shadow            translation away from the sensor (positive of that vector)

Extents for height h and incidence theta (from vertical):
    layover  = h / tan(theta)   metres toward the sensor from the near edge
    shadow   = h * tan(theta)   metres away from the sensor from the far edge
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple

from pyproj import CRS
from shapely.affinity import translate
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

XY = Tuple[float, float]


class GeographicCRSError(ValueError):
    """Raised when a geographic (degree) CRS would silently corrupt metre math."""


def ensure_projected_crs(crs) -> CRS:
    """Return a projected CRS, or raise if the CRS is geographic / missing."""
    if crs is None:
        raise GeographicCRSError(
            "CRS is missing. Reproject building footprints to a metre grid "
            "(EPSG:28992) before mixing heights with coordinates."
        )
    parsed = CRS.from_user_input(crs)
    if parsed.is_geographic:
        raise GeographicCRSError(
            f"CRS is geographic ({parsed.to_string()}). Building heights are in "
            "metres; translating degree-coordinates by metres is silently absurd. "
            "Reproject to EPSG:28992 (or another projected metre grid) first."
        )
    return parsed


def look_azimuth_from_orientation(orientation_flag: int) -> float:
    """Map SpaceNet 6 SummaryData orientation to look azimuth.

    0 = north-facing (sensor looks toward grid north) -> 0 deg
    1 = south-facing (sensor looks toward grid south) -> 180 deg
    """
    if orientation_flag == 0:
        return 0.0
    if orientation_flag == 1:
        return 180.0
    raise ValueError(
        f"orientation_flag must be 0 (north-facing) or 1 (south-facing), got {orientation_flag!r}"
    )


def ground_range_unit(look_azimuth_deg: float) -> XY:
    """Unit vector along ground range, pointing away from the sensor."""
    az = math.radians(look_azimuth_deg)
    return (math.sin(az), math.cos(az))


def layover_extent_m(height_m: float, incidence_deg: float) -> float:
    if incidence_deg <= 0.0 or incidence_deg >= 90.0:
        raise ValueError(f"incidence_deg must be in (0, 90), got {incidence_deg}")
    return height_m / math.tan(math.radians(incidence_deg))


def shadow_extent_m(height_m: float, incidence_deg: float) -> float:
    if incidence_deg <= 0.0 or incidence_deg >= 90.0:
        raise ValueError(f"incidence_deg must be in (0, 90), got {incidence_deg}")
    return height_m * math.tan(math.radians(incidence_deg))


def translate_xy(geom: BaseGeometry, offset: XY) -> BaseGeometry:
    return translate(geom, xoff=offset[0], yoff=offset[1])


def swept_hull(geom: BaseGeometry, offset: XY) -> BaseGeometry:
    """Convex hull of the original polygon unioned with its translation.

    This is the region swept by sliding the footprint along `offset`.
    """
    if geom.is_empty:
        return geom
    moved = translate_xy(geom, offset)
    return unary_union([geom, moved]).convex_hull


def layover_region(
    footprint: BaseGeometry,
    height_m: float,
    incidence_deg: float,
    look_azimuth_deg: float,
) -> BaseGeometry:
    """Area swept toward the sensor from a building footprint of height h."""
    ux, uy = ground_range_unit(look_azimuth_deg)
    dist = layover_extent_m(height_m, incidence_deg)
    return swept_hull(footprint, (-ux * dist, -uy * dist))


def shadow_region(
    footprint: BaseGeometry,
    height_m: float,
    incidence_deg: float,
    look_azimuth_deg: float,
) -> BaseGeometry:
    """Area swept away from the sensor from a building footprint of height h."""
    ux, uy = ground_range_unit(look_azimuth_deg)
    dist = shadow_extent_m(height_m, incidence_deg)
    return swept_hull(footprint, (ux * dist, uy * dist))


def combine_masks(
    layover_geoms: Iterable[BaseGeometry],
    shadow_geoms: Iterable[BaseGeometry],
) -> Tuple[BaseGeometry, BaseGeometry]:
    """Union layover, union shadow, subtract layover from shadow.

    Layover returns dominate, so overlapping shadow is removed.
    """
    layover = unary_union([g for g in layover_geoms if g is not None and not g.is_empty])
    shadow = unary_union([g for g in shadow_geoms if g is not None and not g.is_empty])
    if layover.is_empty:
        return layover, shadow
    shadow = shadow.difference(layover)
    return layover, shadow


def project_axis(points: Sequence[XY], axis: XY) -> Tuple[float, float]:
    """Min/max scalar projection of points onto a unit axis."""
    dots = [p[0] * axis[0] + p[1] * axis[1] for p in points]
    return min(dots), max(dots)
