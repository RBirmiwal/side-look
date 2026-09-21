"""Layover and shadow from a DSM via far-field SAR geometry.

Far-field approximation: one incidence angle for the entire array. Acceptable
for a single flat strip (Rotterdam, ~700 m in ground range). Image brightness
plays no part — only surface height and viewing geometry.

For each cell:
    g      ground-range coordinate, increasing away from the sensor
    z      surface height
    theta  incidence angle from vertical

    slant range:   r   = g * sin(theta) - z * cos(theta)
    shadow proxy:  psi = z + g * cot(theta)

Walk outward along each line parallel to ground range. Running maxima are
shifted by one so a cell is not compared against itself
(numpy.maximum.accumulate of the prefix).

Class codes
    0  nominal
    1  active layover   r[i] < max(r[:i])   — roof / steep face that folded forward
    2  passive layover  same slant-range bin as an active cell, itself monotonic
                        — the ground the roof landed on
    3  shadow           psi[i] < max(psi[:i])
                        takes precedence: a shadowed cell has no return at all

Look azimuth is the compass bearing the sensor looks toward, clockwise from
grid north. Ground range is rotated (or flipped, for cardinal looks) so that
axis 1 increases away from the sensor, scanned, then rotated back. Height is
rotated with bilinear interpolation; the mask returns with nearest neighbour.
"""

from __future__ import annotations

import math
from typing import Callable, Tuple

import numpy as np

CLASS_NOMINAL = 0
CLASS_ACTIVE = 1
CLASS_PASSIVE = 2
CLASS_SHADOW = 3

CLASS_NAMES = {
    CLASS_NOMINAL: "nominal",
    CLASS_ACTIVE: "active layover",
    CLASS_PASSIVE: "passive layover",
    CLASS_SHADOW: "shadow",
}


def slant_range(g: np.ndarray, z: np.ndarray, incidence_deg: float) -> np.ndarray:
    theta = math.radians(incidence_deg)
    return g * math.sin(theta) - z * math.cos(theta)


def shadow_proxy(g: np.ndarray, z: np.ndarray, incidence_deg: float) -> np.ndarray:
    theta = math.radians(incidence_deg)
    return z + g * (math.cos(theta) / math.sin(theta))


def _shifted_running_max(arr: np.ndarray, axis: int = 1) -> np.ndarray:
    """max(arr[:i]) along `axis`, with 0 at the first cell (nothing in front)."""
    acc = np.maximum.accumulate(arr, axis=axis)
    out = np.empty_like(arr)
    if axis == 1:
        out[:, 0] = -np.inf
        out[:, 1:] = acc[:, :-1]
    elif axis == 0:
        out[0, :] = -np.inf
        out[1:, :] = acc[:-1, :]
    else:
        raise ValueError("axis must be 0 or 1")
    return out


def _passive_layover(bins: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Mark cells that share a slant-range bin with an active cell on the same line."""
    passive = np.zeros_like(active)
    for i in range(bins.shape[0]):
        row_active = active[i]
        if not np.any(row_active):
            continue
        infected = np.unique(bins[i, row_active])
        passive[i] = np.isin(bins[i], infected) & ~row_active
    return passive


def scan_range_lines(
    z: np.ndarray,
    incidence_deg: float,
    pixel_size_m: float,
) -> np.ndarray:
    """Classify a DSM whose axis 1 already increases away from the sensor.

    Parameters
    ----------
    z:
        Height raster, float, no NaNs (fill nodata first).
    incidence_deg:
        Incidence from vertical, in (0, 90).
    pixel_size_m:
        Ground posting along axis 1.
    """
    if not (0.0 < incidence_deg < 90.0):
        raise ValueError(f"incidence_deg must be in (0, 90), got {incidence_deg}")
    if pixel_size_m <= 0.0:
        raise ValueError(f"pixel_size_m must be > 0, got {pixel_size_m}")

    z = np.asarray(z, dtype=np.float64)
    nrows, ncols = z.shape
    g = (np.arange(ncols, dtype=np.float64) + 0.5) * pixel_size_m
    g = np.broadcast_to(g, z.shape)

    theta = math.radians(incidence_deg)
    sin_t = math.sin(theta)
    cos_t = math.cos(theta)
    r = g * sin_t - z * cos_t
    psi = z + g * (cos_t / sin_t)

    r_prefix = _shifted_running_max(r, axis=1)
    psi_prefix = _shifted_running_max(psi, axis=1)

    active = r < r_prefix
    shadow = psi < psi_prefix
    # First column has nothing in front.
    active[:, 0] = False
    shadow[:, 0] = False

    bin_width = pixel_size_m * sin_t
    bins = np.floor(r / bin_width).astype(np.int64)
    passive = _passive_layover(bins, active)

    mask = np.zeros(z.shape, dtype=np.uint8)
    mask[passive] = CLASS_PASSIVE
    mask[active] = CLASS_ACTIVE
    mask[shadow] = CLASS_SHADOW  # shadow takes precedence
    return mask


def _near_azimuth(az: float, target: float, tol: float = 0.5) -> bool:
    d = abs((az - target) % 360.0)
    return min(d, 360.0 - d) < tol


def _orient_range_along_cols(
    z: np.ndarray, look_azimuth_deg: float
) -> Tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """Rotate/flip so axis 1 increases away from the sensor.

    Cardinal looks (0/90/180/270) use exact transpose/flip — required for the
    closed-form tests. Other azimuths use bilinear rotation.
    """
    az = look_azimuth_deg % 360.0

    if _near_azimuth(az, 90.0):
        return z, (lambda m: m)
    if _near_azimuth(az, 270.0):
        return np.fliplr(z), np.fliplr
    if _near_azimuth(az, 180.0):
        # +row (south) becomes +col
        return np.rot90(z, k=1), (lambda m: np.rot90(m, k=-1))
    if _near_azimuth(az, 0.0):
        # -row (north) becomes +col
        return np.rot90(z, k=-1), (lambda m: np.rot90(m, k=1))

    from scipy.ndimage import rotate

    # Image convention: +col east, +row south. Away vector (dcol, drow) =
    # (sin az, -cos az). CCW rotation by (90 - az) maps that onto +col.
    angle = 90.0 - az
    orig_shape = z.shape
    rotated = rotate(
        z,
        angle=angle,
        reshape=True,
        order=1,
        mode="constant",
        cval=np.nan,
    )

    def restore(mask: np.ndarray) -> np.ndarray:
        back = rotate(
            mask.astype(np.float32),
            angle=-angle,
            reshape=True,
            order=0,
            mode="constant",
            cval=0.0,
        )
        return _center_crop_or_pad(back, orig_shape).astype(mask.dtype)

    return rotated, restore


def _center_crop_or_pad(arr: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=arr.dtype)
    sr = min(arr.shape[0], shape[0])
    sc = min(arr.shape[1], shape[1])
    r0a = (arr.shape[0] - sr) // 2
    c0a = (arr.shape[1] - sc) // 2
    r0o = (shape[0] - sr) // 2
    c0o = (shape[1] - sc) // 2
    out[r0o : r0o + sr, c0o : c0o + sc] = arr[r0a : r0a + sr, c0a : c0a + sc]
    return out


def fill_nodata(
    z: np.ndarray,
    nodata: float | None = None,
    max_search_distance: float = 64.0,
) -> np.ndarray:
    """Interpolate nodata so running-max scans are not poisoned."""
    from rasterio.fill import fillnodata

    arr = np.array(z, dtype=np.float64, copy=True)
    valid = np.isfinite(arr)
    if nodata is not None and np.isfinite(nodata):
        valid &= arr != nodata
    # AHN sometimes uses huge IEEE sentinels.
    valid &= np.abs(arr) < 1.0e6
    if np.all(valid):
        return arr
    if not np.any(valid):
        raise ValueError("DSM has no valid pixels to fill from")

    work = arr.copy()
    work[~valid] = 0.0
    fillnodata(
        work,
        mask=valid.astype(np.uint8) * 255,
        max_search_distance=float(max_search_distance),
        smoothing_iterations=0,
    )
    still = ~np.isfinite(work) | (np.abs(work) >= 1.0e6)
    if np.any(still):
        work[still] = float(np.median(arr[valid]))
    return work


def mask_from_dsm(
    z: np.ndarray,
    incidence_deg: float,
    look_azimuth_deg: float,
    pixel_size_m: float,
) -> np.ndarray:
    """Layover/shadow mask for a north-up DSM (row 0 = north, col 0 = west).

    `look_azimuth_deg` and `incidence_deg` are explicit parameters — the
    debugging loop is changing them, not editing constants.
    """
    oriented, restore = _orient_range_along_cols(z, look_azimuth_deg)
    if np.isnan(oriented).any():
        oriented = fill_nodata(oriented, nodata=None)
    mask = scan_range_lines(oriented, incidence_deg, pixel_size_m)
    return restore(mask)


def mask_is_trivial(mask: np.ndarray) -> bool:
    """True if the mask is all one class — usually a nodata or load failure."""
    return int(np.unique(mask).size) <= 1
