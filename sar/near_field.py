"""Near-field layover and shadow for an airborne straight-line collect.

The platform is an aircraft a couple of kilometres up, not a satellite.
Incidence is not constant across the swath, so the fold test does not take a
theta. Each DSM cell is classified from its perpendicular distance to the
fitted flight line and its off-nadir angle.

Scan each line perpendicular to the track, moving away from it:

    active layover   r[i] < max(r[:i])
    shadow           alpha[i] < max(alpha[:i])
    passive layover  shares a slant-range bin with an active cell

Shadow wins ties. Class codes match the far-field mask.

Slant-range bins use the product range resolution (metres), not
pixel_size * sin(theta). There is no single theta to put in that formula.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from rasterio.transform import Affine, xy

from acquire_geometry import enu_basis, geodetic_to_ecef
from dsm_geometry import (
    CLASS_ACTIVE,
    CLASS_PASSIVE,
    CLASS_SHADOW,
    _near_azimuth,
    _passive_layover,
    _shifted_running_max,
    mask_from_dsm,
)

GEOID = Path(__file__).resolve().parent / "data" / "nl_nsgi_nlgeo2018.tif"


def look_azimuth_from_track(direction_enu: np.ndarray, pointing: str) -> float:
    """Azimuth the sensor looks, perpendicular to the track.

    `direction_enu` points along the flight in the direction of time.
    `pointing` is Capella `collect.radar.pointing` (`right` or `left`).
    """
    east, north = float(direction_enu[0]), float(direction_enu[1])
    heading = math.degrees(math.atan2(east, north)) % 360.0
    side = pointing.strip().lower()
    if side == "right":
        return (heading + 90.0) % 360.0
    if side == "left":
        return (heading - 90.0) % 360.0
    raise ValueError(f"radar.pointing must be right or left, got {pointing!r}")


def south_looking(look_azimuth_deg: float) -> bool:
    """True when the sensor looks nearer south than north. Canonicalise flips these."""
    az = look_azimuth_deg % 360.0
    to_south = min(abs(az - 180.0), 360.0 - abs(az - 180.0))
    to_north = min(az, 360.0 - az)
    return to_south < to_north


def fit_flight_line(positions_enu: np.ndarray) -> dict:
    """Straight line through platform positions in a local ENU frame.

    Direction follows the first principal axis, oriented with increasing time
    (the rows of `positions_enu` are in time order).
    """
    pts = np.asarray(positions_enu, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2:
        raise ValueError("need at least two ENU positions")
    origin = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - origin, full_matrices=False)
    direction = vt[0].copy()
    if float(np.dot(pts[-1] - pts[0], direction)) < 0.0:
        direction = -direction
    direction /= np.linalg.norm(direction)
    delta = pts - origin
    along = delta @ direction
    perp = delta - along[:, None] * direction
    dist = np.linalg.norm(perp, axis=1)
    return {
        "origin_enu": origin,
        "direction_enu": direction,
        "residual_rms_m": float(np.sqrt(np.mean(dist**2))),
        "residual_max_m": float(dist.max()),
        "height_above_origin_m": float(origin[2]),
    }


def perpendicular_geometry(
    east: np.ndarray,
    north: np.ndarray,
    up: np.ndarray,
    origin_enu: np.ndarray,
    direction_enu: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Slant range to the flight line and off-nadir angle, in radians.

    Off-nadir is the angle at the line between straight down and the point.
    """
    delta = np.stack(
        [
            np.asarray(east, dtype=np.float64) - origin_enu[0],
            np.asarray(north, dtype=np.float64) - origin_enu[1],
            np.asarray(up, dtype=np.float64) - origin_enu[2],
        ],
        axis=-1,
    )
    along = np.sum(delta * direction_enu, axis=-1)
    perp = delta - along[..., None] * direction_enu
    slant = np.linalg.norm(perp, axis=-1)
    vertical_drop = -perp[..., 2]
    horizontal = np.sqrt(np.maximum(perp[..., 0] ** 2 + perp[..., 1] ** 2, 0.0))
    alpha = np.arctan2(horizontal, vertical_drop)
    return slant, alpha


def scan_near_field(slant: np.ndarray, alpha: np.ndarray, bin_width_m: float) -> np.ndarray:
    """Classify a cross-track scan. Axis 1 increases away from the flight line."""
    if bin_width_m <= 0.0:
        raise ValueError(f"bin_width_m must be > 0, got {bin_width_m}")
    r = np.asarray(slant, dtype=np.float64)
    a = np.asarray(alpha, dtype=np.float64)
    r_prefix = _shifted_running_max(r, axis=1)
    a_prefix = _shifted_running_max(a, axis=1)
    active = r < r_prefix
    shadow = a < a_prefix
    active[:, 0] = False
    shadow[:, 0] = False
    bins = np.floor(r / bin_width_m).astype(np.int64)
    passive = _passive_layover(bins, active)
    mask = np.zeros(r.shape, dtype=np.uint8)
    mask[passive] = CLASS_PASSIVE
    mask[active] = CLASS_ACTIVE
    mask[shadow] = CLASS_SHADOW
    return mask


def orient_range(arr: np.ndarray, look_azimuth_deg: float):
    """Same cardinal rotation as the far-field mask, without filling nodata."""
    az = look_azimuth_deg % 360.0
    if _near_azimuth(az, 90.0):
        return arr, (lambda m: m)
    if _near_azimuth(az, 270.0):
        return np.fliplr(arr), np.fliplr
    if _near_azimuth(az, 180.0):
        return np.rot90(arr, k=1), (lambda m: np.rot90(m, k=-1))
    if _near_azimuth(az, 0.0):
        return np.rot90(arr, k=-1), (lambda m: np.rot90(m, k=1))
    raise ValueError(
        f"look {look_azimuth_deg:.3f}° is more than 0.5° off a cardinal; "
        "refusing a bilinear rotate for the near-field grid"
    )


def profile_slant_alpha(z: np.ndarray, ground_range_m: np.ndarray, platform_height_m: float):
    """Cross-section helper: flight line at cross-track 0, height `platform_height_m`."""
    z = np.asarray(z, dtype=np.float64)
    g = np.asarray(ground_range_m, dtype=np.float64)
    dz = platform_height_m - z
    slant = np.hypot(g, dz)
    alpha = np.arctan2(g, dz)
    return slant, alpha


def layover_extent_m(ground_range_m: float, building_height_m: float, platform_height_m: float) -> float:
    """Ground metres the roof folds toward the track, from the near wall.

    Closed form: the ground point with the same slant range as the near roof edge.
    """
    g = float(ground_range_m)
    h = float(building_height_m)
    height = float(platform_height_m)
    inside = g * g - 2.0 * height * h + h * h
    if inside <= 0.0:
        return g
    return g - math.sqrt(inside)


def shadow_extent_m(ground_range_m: float, building_height_m: float, platform_height_m: float) -> float:
    """Ground metres of shadow beyond the far wall.

    The grazing ray off the far roof edge meets z=0 at g * H / (H - h).
    """
    g = float(ground_range_m)
    h = float(building_height_m)
    height = float(platform_height_m)
    if height <= h:
        raise ValueError("platform must be above the roof")
    return g * h / (height - h)


def undulation_at(lon: float, lat: float, geoid_path: Path | None = None) -> float:
    """NLGEO2018 undulation N, metres. Ellipsoidal height = NAP + N."""
    import rasterio

    path = Path(geoid_path) if geoid_path else GEOID
    with rasterio.open(path) as src:
        val = next(src.sample([(lon, lat)]))
    n = float(val[0])
    if not np.isfinite(n):
        raise RuntimeError(f"geoid nodata at {lon}, {lat}")
    return n


def add_undulation(z_nap: np.ndarray, transform: Affine, crs: str, geoid_path: Path | None = None) -> np.ndarray:
    """Add NLGEO2018 to a north-up NAP raster. Geoid is smooth, so it is sampled coarsely."""
    from pyproj import Transformer
    from scipy.interpolate import RegularGridInterpolator

    path = Path(geoid_path) if geoid_path else GEOID
    step = 40
    rows = np.arange(0, z_nap.shape[0], step)
    cols = np.arange(0, z_nap.shape[1], step)
    if rows[-1] != z_nap.shape[0] - 1:
        rows = np.append(rows, z_nap.shape[0] - 1)
    if cols[-1] != z_nap.shape[1] - 1:
        cols = np.append(cols, z_nap.shape[1] - 1)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    xs, ys = xy(transform, rr, cc, offset="center")
    to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = to_ll.transform(np.asarray(xs), np.asarray(ys))
    import rasterio

    with rasterio.open(path) as src:
        samples = np.array([v[0] for v in src.sample(zip(lon.ravel(), lat.ravel()))], dtype=np.float64)
    samples = samples.reshape(len(rows), len(cols))
    interp = RegularGridInterpolator(
        (rows.astype(np.float64), cols.astype(np.float64)),
        samples,
        bounds_error=False,
        fill_value=None,
    )
    jj, ii = np.meshgrid(np.arange(z_nap.shape[1]), np.arange(z_nap.shape[0]))
    n = interp(np.stack([ii.ravel(), jj.ravel()], axis=1)).reshape(z_nap.shape)
    return np.asarray(z_nap, dtype=np.float64) + n


def pixel_grids(transform: Affine, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    cols = np.arange(width)
    rows = np.arange(height)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    xs, ys = xy(transform, rr, cc, offset="center")
    return np.asarray(xs, dtype=np.float64).reshape(height, width), np.asarray(ys, dtype=np.float64).reshape(height, width)


def enu_from_map(
    easting: np.ndarray,
    northing: np.ndarray,
    up: np.ndarray,
    crs: str,
    lat0: float,
    lon0: float,
    height0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map coordinates plus ellipsoidal height to ENU at the centre target."""
    from pyproj import Transformer

    to_ecef = Transformer.from_crs(crs, "EPSG:4978", always_xy=True)
    x, y, z = to_ecef.transform(np.asarray(easting), np.asarray(northing), np.asarray(up))
    shape = np.shape(up)
    x = np.asarray(x, dtype=np.float64).reshape(shape)
    y = np.asarray(y, dtype=np.float64).reshape(shape)
    z = np.asarray(z, dtype=np.float64).reshape(shape)
    target = geodetic_to_ecef(lat0, lon0, height0)
    basis = enu_basis(lat0, lon0)
    delta = np.stack([x - target[0], y - target[1], z - target[2]], axis=0)
    enu = np.tensordot(basis, delta, axes=(1, 0))
    return enu[0], enu[1], enu[2]


def mask_from_flight(
    z_ellips: np.ndarray,
    transform: Affine,
    crs: str,
    look_azimuth_deg: float,
    origin_enu: np.ndarray,
    direction_enu: np.ndarray,
    lat0: float,
    lon0: float,
    height0: float,
    bin_width_m: float,
) -> np.ndarray:
    """North-up DSM (row 0 = north) to a near-field layover/shadow mask."""
    xs, ys = pixel_grids(transform, z_ellips.shape[0], z_ellips.shape[1])
    east, north, up = enu_from_map(xs, ys, z_ellips, crs, lat0, lon0, height0)
    slant, alpha = perpendicular_geometry(east, north, up, origin_enu, direction_enu)
    slant_o, _ = orient_range(slant, look_azimuth_deg)
    alpha_o, restore = orient_range(alpha, look_azimuth_deg)
    # Axis 1 must run away from the track: off-nadir increases.
    if float(np.nanmean(alpha_o[:, -1]) - np.nanmean(alpha_o[:, 0])) < 0.0:
        raise RuntimeError(
            f"look {look_azimuth_deg:.3f}° scans toward the track, not away from it"
        )
    mask_o = scan_near_field(slant_o, alpha_o, bin_width_m)
    return restore(mask_o)


def far_field_reference(z: np.ndarray, incidence_deg: float, look_azimuth_deg: float, pixel_size_m: float) -> np.ndarray:
    return mask_from_dsm(z, incidence_deg, look_azimuth_deg, pixel_size_m)
