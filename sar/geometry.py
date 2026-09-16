"""Layover and shadow for a side-looking SAR collect.
layover = h / tan(θ) toward sensor; shadow = h * tan(θ) away.
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
    pass

def ensure_projected_crs(crs) -> CRS:
    if crs is None:
        raise GeographicCRSError("CRS is missing. Reproject to EPSG:28992 first.")
    parsed = CRS.from_user_input(crs)
    if parsed.is_geographic:
        raise GeographicCRSError(f"CRS is geographic ({parsed.to_string()}). Reproject to EPSG:28992 first.")
    return parsed

def look_azimuth_from_orientation(orientation_flag: int) -> float:
    if orientation_flag == 0:
        return 0.0
    if orientation_flag == 1:
        return 180.0
    raise ValueError(f"orientation_flag must be 0 or 1, got {orientation_flag!r}")

def ground_range_unit(look_azimuth_deg: float) -> XY:
    az = math.radians(look_azimuth_deg)
    return (math.sin(az), math.cos(az))

def layover_extent_m(height_m: float, incidence_deg: float) -> float:
    if not (0.0 < incidence_deg < 90.0):
        raise ValueError(f"incidence_deg must be in (0, 90), got {incidence_deg}")
    return height_m / math.tan(math.radians(incidence_deg))

def shadow_extent_m(height_m: float, incidence_deg: float) -> float:
    if not (0.0 < incidence_deg < 90.0):
        raise ValueError(f"incidence_deg must be in (0, 90), got {incidence_deg}")
    return height_m * math.tan(math.radians(incidence_deg))

def translate_xy(geom: BaseGeometry, offset: XY) -> BaseGeometry:
    return translate(geom, xoff=offset[0], yoff=offset[1])

def swept_hull(geom: BaseGeometry, offset: XY) -> BaseGeometry:
    if geom.is_empty:
        return geom
    moved = translate_xy(geom, offset)
    return unary_union([geom, moved]).convex_hull

def layover_region(footprint, height_m, incidence_deg, look_azimuth_deg):
    ux, uy = ground_range_unit(look_azimuth_deg)
    dist = layover_extent_m(height_m, incidence_deg)
    return swept_hull(footprint, (-ux * dist, -uy * dist))

def shadow_region(footprint, height_m, incidence_deg, look_azimuth_deg):
    ux, uy = ground_range_unit(look_azimuth_deg)
    dist = shadow_extent_m(height_m, incidence_deg)
    return swept_hull(footprint, (ux * dist, uy * dist))

def combine_masks(layover_geoms: Iterable[BaseGeometry], shadow_geoms: Iterable[BaseGeometry]):
    layover = unary_union([g for g in layover_geoms if g is not None and not g.is_empty])
    shadow = unary_union([g for g in shadow_geoms if g is not None and not g.is_empty])
    if layover.is_empty:
        return layover, shadow
    return layover, shadow.difference(layover)
