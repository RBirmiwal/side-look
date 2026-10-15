"""
Layover / shadow labels for the SpaceNet 6 Capella airborne SAR, under two
candidate rules for where the image draws things that stand above the ground.

The product JSON says the image was focused onto SRTM GL1 on a UTM grid
(image_geometry type=surface, z.dem_name=SrtmGl1), and the SpaceNet 6 paper
says the imagery was ortho-rectified with SRTM. On strip
20190822074237_20190822074521, though, labels built on that rule did not line
up with the image, and the tallest structures looked bright on their own
footprints. So both rules are implemented and the real image decides.

RULE A, "lean": project_line / project_grid
    A point off the reference surface (roof, wall) is drawn where the
    REFERENCE surface (SRTM) has the same closest-approach range to the
    flight line. Roofs are drawn h/tan(theta) toward the sensor; the dark
    region behind a building starts inside its own footprint.

RULE B, "in place": own_range_line / own_range_grid
    Every cell is drawn where it stands, and shows whatever the radar
    received at that cell's OWN closest-approach range. Roofs stay on their
    footprints; the ground in front of a building also shows the wall and
    roof (they share its range); shadow is the geometric shadow behind the
    back wall, except where another building's wall or roof shares the
    range and fills it. Without the filling, this is the v2 labelling.

Both return "coverage" per pixel: how many visible surfaces share it.

    coverage < 0.5          shadow    nothing sends energy to this pixel
    0.5 <= coverage < 1.5   nominal   exactly one surface
    coverage >= 1.5         layover   two or more surfaces summed

Coverage is also the severity ("magnitude") target: 3 typically means ground
+ wall + roof in one pixel.

INPUTS, frame-agnostic on purpose
Per-cell quantities that near_field.py already computes, on one grid, in
CANONICAL orientation (sensor at index 0 of the range axis):

    r      closest-approach distance from the flight line to the DSM surface
           point of each cell (metres)
    alpha  off-nadir angle of that DSM point (any monotonic angle unit);
           grows away from the sensor over flat ground
    r_ref  (rule A only) closest-approach distance to the REFERENCE-surface
           point of the same cell (SRTM GL1, ellipsoidal, same frame as r)

`line_geometry()` builds r and alpha for a straight level track, for tests.

Pad the DSM at least ~150 m beyond both range edges of the image before
calling, then crop. Content can project in from outside: layover from beyond
the far edge, shadow from beyond the near edge.
"""

from __future__ import annotations

import numpy as np

NOMINAL, LAYOVER, SHADOW, UNKNOWN = 0, 1, 2, 255
CLASS_NAMES = ["nominal", "layover", "shadow"]


# ----------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------
def line_geometry(y, z, H):
    """
    Straight, level flight line in a local frame.

    y : horizontal distance from the nadir ground track (m), increasing away
        from the sensor
    z : height of the surface point (m), same vertical frame as H
    H : platform height (m)

    Returns (r, alpha) with alpha = off-nadir angle in radians.
    """
    y = np.asarray(y, float)
    z = np.asarray(z, float)
    return np.hypot(H - z, y), np.arctan2(y, H - z)


# ----------------------------------------------------------------------
# interval coverage
# ----------------------------------------------------------------------
def _coverage_from_intervals(lo, hi, n):
    """
    Total overlap, in cell widths, of intervals [lo, hi] (fractional cell
    index) with each of n cells, cell j spanning [j - 0.5, j + 0.5].

    Uses C(x) = sum_k max(0, x - lo_k) - max(0, x - hi_k), the covered length
    up to x, evaluated at every cell edge with sorted prefix sums.
    """
    if lo.size == 0:
        return np.zeros(n)
    edges = np.arange(n + 1) - 0.5

    def ramp_sum(points):
        p = np.sort(points)
        csum = np.concatenate([[0.0], np.cumsum(p)])
        k = np.searchsorted(p, edges, side="left")      # count of p < edge
        return k * edges - csum[k]

    c = ramp_sum(lo) - ramp_sum(hi)
    return np.diff(c)


# ----------------------------------------------------------------------
# one cross-track line
# ----------------------------------------------------------------------
def _visible_segments(r, alpha, nan_margin=0, angle_tol=1e-12):
    """
    Occlusion along one cross-track line, shared by both rules.

    Returns a dict with
      visible       (n,)   DSM cell visible from the sensor
      uncertain     (n,)   NaN cells and cells within nan_margin of one
      r_start/r_end (n-1,) closest-approach range span of the visible part of
                           each segment k -> k+1
      use           (n-1,) segment has a visible part
      seg_unknown   (n-1,) segment touches an uncertain cell
    """
    r = np.asarray(r, float)
    alpha = np.asarray(alpha, float)
    n = r.size
    idx = np.arange(n, dtype=float)

    bad = ~(np.isfinite(r) & np.isfinite(alpha))

    # ---- visibility: visible iff alpha >= max alpha of all closer cells ----
    a = np.where(bad, -np.inf, alpha)
    run = np.maximum.accumulate(a)
    prev_max = np.concatenate([[-np.inf], run[:-1]])     # max over cells before i
    visible = (a >= prev_max - angle_tol) & ~bad

    # cells that missing DSM could affect: after a gap a hidden occluder could
    # shadow them; before a gap, layover from the missing surface could land
    # on them
    uncertain = bad.copy()
    if nan_margin > 0 and bad.any():
        last_bad = np.maximum.accumulate(np.where(bad, idx, -np.inf))
        next_bad = np.minimum.accumulate(np.where(bad, idx, np.inf)[::-1])[::-1]
        uncertain |= ((idx - last_bad) <= nan_margin) | ((next_bad - idx) <= nan_margin)

    # ---- segments between consecutive cells -------------------------------
    a0, a1 = alpha[:-1], alpha[1:]
    r0, r1 = r[:-1], r[1:]
    v0, v1 = visible[:-1], visible[1:]

    # A straight segment's off-nadir angle is monotonic along it. If its far
    # end is hidden, the whole segment is (back slope or inside a shadow).
    # If its far end is visible and near end hidden, it emerges from shadow
    # where its angle reaches the occlusion threshold.
    use = v1.copy()
    t0 = np.zeros(n - 1)
    emerge = use & ~v0
    with np.errstate(invalid="ignore", divide="ignore"):
        thr = prev_max[1:]
        t_star = (thr - a0) / (a1 - a0)
    t0[emerge] = np.clip(t_star[emerge], 0.0, 1.0)

    return dict(visible=visible, uncertain=uncertain,
                r_start=r0 + t0 * (r1 - r0), r_end=r1.copy(),
                use=use, seg_unknown=uncertain[:-1] | uncertain[1:])


def _accumulate(p0, p1, use, seg_unknown, n):
    """Coverage and unknown flags on n cells from segment index spans."""
    lo = np.clip(np.minimum(p0, p1), -0.5, n - 0.5)
    hi = np.clip(np.maximum(p0, p1), -0.5, n - 0.5)
    good = use & ~seg_unknown & (hi > lo)
    coverage = _coverage_from_intervals(lo[good], hi[good], n)

    unk_lo = np.where(np.isfinite(lo), lo, np.nan)[seg_unknown]
    unk_hi = np.where(np.isfinite(hi), hi, np.nan)[seg_unknown]
    keep = np.isfinite(unk_lo) & np.isfinite(unk_hi)
    unknown_cov = _coverage_from_intervals(unk_lo[keep], unk_hi[keep], n)
    return coverage, unknown_cov > 0


def project_line(r, alpha, r_ref, nan_margin=0, angle_tol=1e-12):
    """
    RULE A ("lean"). Project one cross-track line of DSM into image space.

    r, alpha : (n,) DSM surface closest-approach range and off-nadir angle
    r_ref    : (n,) reference-surface closest-approach range at the same cells;
               must increase along the line (true for any SRTM-like surface)
    nan_margin : cells after a NaN whose visibility is treated as unknown,
               because the missing DSM could have hidden a tall occluder

    Returns
      coverage : (n,) surfaces per image pixel
      unknown  : (n,) bool, pixel touched by projections of unknown surface
      visible  : (n,) bool, DSM cell visible from the sensor (ground space)
    """
    r = np.asarray(r, float)
    r_ref = np.asarray(r_ref, float)
    n = r.size
    idx = np.arange(n, dtype=float)

    # reference range must be strictly increasing to be invertible
    rr = r_ref.copy()
    if not np.all(np.isfinite(rr)):
        raise ValueError("r_ref must be finite everywhere on the line")
    if np.any(np.diff(rr) <= 0):
        # an SRTM surface steeper than the look angle; does not happen in
        # Rotterdam but keep the mapping usable
        rr = np.maximum.accumulate(rr + np.arange(n) * 1e-9)

    s = _visible_segments(r, alpha, nan_margin, angle_tol)

    # map closest-approach range to fractional image index on the reference
    def to_index(rv):
        return np.interp(rv, rr, idx, left=-np.inf, right=np.inf)

    coverage, unk = _accumulate(to_index(s["r_start"]), to_index(s["r_end"]),
                                s["use"], s["seg_unknown"], n)
    return coverage, unk | s["uncertain"], s["visible"]


def own_range_line(r, alpha, nan_margin=0, step=0.05, angle_tol=1e-12):
    """
    RULE B ("in place"). Each cell shows the visible surfaces at its OWN
    closest-approach range.

    The visible surface pieces of the line are binned by range on a fine
    grid (step metres), which gives layers(range): how many visible surfaces
    the radar sees at that range. Each cell's coverage is layers(r of that
    cell). A visible cell counts itself; a hidden cell does not, so it is
    shadow unless something else shares its range.

    Returns (coverage, unknown, visible), each (n,), like project_line.
    """
    r = np.asarray(r, float)
    n = r.size
    s = _visible_segments(r, alpha, nan_margin, angle_tol)
    fin = np.isfinite(r)
    if not fin.any():
        return np.zeros(n), np.ones(n, bool), s["visible"]

    r0 = np.nanmin(r) - 2 * step
    m = int(np.ceil((np.nanmax(r) + 2 * step - r0) / step)) + 1

    def to_bin(rv):
        return (rv - r0) / step

    layers, unk_bins = _accumulate(to_bin(s["r_start"]), to_bin(s["r_end"]),
                                   s["use"], s["seg_unknown"], m)

    coverage = np.zeros(n)
    unknown = s["uncertain"].copy()
    b = to_bin(r[fin])
    grid = np.arange(m, dtype=float)
    coverage[fin] = np.interp(b, grid, layers)
    unknown[fin] |= unk_bins[np.clip(np.rint(b).astype(int), 0, m - 1)]
    return coverage, unknown, s["visible"]


# ----------------------------------------------------------------------
# a whole grid
# ----------------------------------------------------------------------
def project_grid(r, alpha, r_ref, axis=0, nan_margin=0):
    """
    RULE A. Apply project_line to every cross-track line of a 2D grid.

    axis : the RANGE axis (canonical orientation: index 0 nearest the sensor).
    Returns (coverage float32, unknown bool, visible bool), same shape as r.
    """
    r = np.asarray(r)
    if axis == 1:
        cov, unk, vis = project_grid(r.T, np.asarray(alpha).T, np.asarray(r_ref).T, 0, nan_margin)
        return cov.T, unk.T, vis.T

    cov = np.zeros(r.shape, np.float32)
    unk = np.zeros(r.shape, bool)
    vis = np.zeros(r.shape, bool)
    for j in range(r.shape[1]):
        c, u, v = project_line(r[:, j], alpha[:, j], r_ref[:, j], nan_margin)
        cov[:, j], unk[:, j], vis[:, j] = c, u, v
    return cov, unk, vis


def own_range_grid(r, alpha, axis=0, nan_margin=0, step=0.05):
    """
    RULE B over a 2D grid. axis is the RANGE axis (index 0 nearest the sensor).
    Returns (coverage float32, unknown bool, visible bool), same shape as r.
    """
    r = np.asarray(r)
    if axis == 1:
        cov, unk, vis = own_range_grid(r.T, np.asarray(alpha).T, 0, nan_margin, step)
        return cov.T, unk.T, vis.T

    cov = np.zeros(r.shape, np.float32)
    unk = np.zeros(r.shape, bool)
    vis = np.zeros(r.shape, bool)
    for j in range(r.shape[1]):
        c, u, v = own_range_line(r[:, j], alpha[:, j], nan_margin, step)
        cov[:, j], unk[:, j], vis[:, j] = c, u, v
    return cov, unk, vis


def labels_from_coverage(coverage, unknown=None, shadow_below=0.5, layover_from=1.5):
    """0 nominal, 1 layover, 2 shadow, 255 unknown."""
    out = np.full(coverage.shape, NOMINAL, np.uint8)
    out[coverage >= layover_from] = LAYOVER
    out[coverage < shadow_below] = SHADOW
    if unknown is not None:
        out[unknown] = UNKNOWN
    return out


# ----------------------------------------------------------------------
# registration check against the real image
# ----------------------------------------------------------------------
def _xcorr_peak(a, b, search):
    """Shift (dy, dx) to apply to b to best match a, and the peak value."""
    a = np.nan_to_num(a - np.nanmean(a))
    b = np.nan_to_num(b - np.nanmean(b))
    H, W = a.shape
    A = np.fft.rfft2(a, s=(2 * H, 2 * W))
    B = np.fft.rfft2(b, s=(2 * H, 2 * W))
    c = np.fft.irfft2(A * np.conj(B), s=(2 * H, 2 * W))
    c = np.fft.fftshift(c) / (a.size * a.std() * b.std() + 1e-12)
    cy, cx = H, W
    win = c[cy - search:cy + search + 1, cx - search:cx + search + 1]
    iy, ix = np.unravel_index(np.argmax(win), win.shape)
    return int(iy - search), int(ix - search), float(win[iy, ix])


def registration_check(hh_db, labels, size=1024, n_windows=8, search=300, along_axis=1,
                       range_bands=1, valid=None, use=("layover", "shadow"),
                       min_labelled=0.02):
    """
    Cross-correlate a signed label image (+1 layover, -1 shadow, 0 otherwise)
    with HH in dB, in windows spread along the strip.

    The peak tells how far the labels sit from what the image shows. A
    positive dy means the labels must move AWAY from the sensor (along the
    range axis, toward larger index) to match the image.

    range_bands : 1 = one band of windows at mid range (original behaviour);
                  3 = near / middle / far bands, so a range-dependent offset
                  shows up
    valid       : optional bool mask; False pixels are ignored (water,
                  clutter, unknown)
    use         : which classes enter the signed image. ("shadow",) uses only
                  shadow, whose near edge is where the two rules differ most.

    Returns a list of (band, window_centre, dy, dx, peak) and the median
    (dy, dx, peak) over all kept windows.
    """
    s = np.zeros(labels.shape, np.float32)
    if "layover" in use:
        s[labels == LAYOVER] = 1.0
    if "shadow" in use:
        s[labels == SHADOW] = -1.0
    ignore = labels == UNKNOWN
    if valid is not None:
        ignore |= ~np.asarray(valid, bool)
    s[ignore] = 0.0
    hh = np.where(ignore, np.nan, hh_db).astype(np.float32)

    L = hh.shape[along_axis]
    other = hh.shape[1 - along_axis]
    h = min(size, other // range_bands if range_bands > 1 else other)
    rows = []
    for band in range(range_bands):
        if range_bands == 1:
            o0 = (other - h) // 2
        else:
            centre = int((band + 0.5) * other / range_bands)
            o0 = int(np.clip(centre - h // 2, 0, other - h))
        for w in range(n_windows):
            c = int((w + 0.5) * L / n_windows)
            a0 = max(0, c - size // 2)
            a1 = min(L, a0 + size)
            if along_axis == 1:
                sl = (slice(o0, o0 + h), slice(a0, a1))
            else:
                sl = (slice(a0, a1), slice(o0, o0 + h))
            hw, sw = hh[sl], s[sl]
            if np.isfinite(hw).mean() < 0.5 or (sw != 0).mean() < min_labelled:
                continue
            srch = min(search, min(hw.shape) // 2 - 1)
            dy, dx, pk = _xcorr_peak(hw, sw, srch)
            if along_axis == 0:
                dy, dx = dx, dy
            rows.append((band, c, dy, dx, pk))
    if not rows:
        return rows, (np.nan, np.nan, np.nan)
    arr = np.array([r[2:] for r in rows], float)
    return rows, tuple(np.median(arr, axis=0))
