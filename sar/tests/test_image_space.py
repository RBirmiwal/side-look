"""
Closed-form tests for image_space.py, rule A ("lean", project_line) and
rule B ("in place", own_range_line).

Geometry matches the real strips: aircraft H = 1614 m above the ground,
swath from ~24 to ~43 deg incidence, 0.25 m image cells.

For a box building, near wall at ground distance g, depth w, height h, on flat
ground imaged on a flat reference surface, a point (y, z) appears at
y' = sqrt(y^2 - 2 H z + z^2). So in the IMAGE:

    roof near edge      y_top    = sqrt(g^2 - 2Hh + h^2)
    roof far edge       roof_end = sqrt((g+w)^2 - 2Hh + h^2)
    ground reappears    shadow_end = (g+w) H / (H - h)

    [y_top, g)            ground + wall + roof   coverage 3   (wide building)
    [g, roof_end)         roof only              coverage 1
    [roof_end, shadow_end) nothing               coverage 0   <- image shadow
                                                  starts INSIDE the footprint

Run:  python test_image_space.py
"""

import sys
from pathlib import Path

import numpy as np

# works beside image_space.py or one level below it (sar/tests/)
_here = Path(__file__).resolve().parent
for _p in (_here, _here.parent):
    if (_p / "image_space.py").exists():
        sys.path.insert(0, str(_p))
        break

from image_space import (LAYOVER, NOMINAL, SHADOW, UNKNOWN, labels_from_coverage,  # noqa: E402
                         layover_magnitude, line_geometry, own_range_displacement,
                         own_range_displacement_grid, own_range_grid, own_range_line,
                         project_grid, project_line, registration_check)

H = 1614.0
DY = 0.25
Y = np.arange(730.0, 1500.0, DY)


def ground_line(*buildings, ref_height=0.0, H=H, y=Y):
    z = np.zeros_like(y)
    for g, w, h in buildings:
        z[(y >= g) & (y < g + w)] = h
    r, a = line_geometry(y, z, H)
    r_ref, _ = line_geometry(y, np.full_like(y, ref_height), H)
    return z, r, a, r_ref


def mean_cov(y, cov, lo, hi, pad=2 * DY):
    sel = (y > lo + pad) & (y < hi - pad)
    assert sel.any(), (lo, hi)
    return float(cov[sel].mean()), float(cov[sel].min()), float(cov[sel].max())


def edge(y, mask):
    """first y where mask becomes true"""
    i = np.argmax(mask)
    return y[i] if mask.any() else np.nan


def test_flat_ground():
    _, r, a, rr = ground_line()
    cov, unk, vis = project_line(r, a, rr)
    inner = cov[1:-1]
    assert np.allclose(inner, 1.0, atol=1e-9), (inner.min(), inner.max())
    assert vis.all() and not unk.any()
    print(f"  flat ground: coverage {inner.min():.6f}..{inner.max():.6f}  (expect 1)")


def test_wide_building():
    g, w, h = 1100.0, 40.0, 20.0
    _, r, a, rr = ground_line((g, w, h))
    cov, _, _ = project_line(r, a, rr)
    far = g + w - DY                        # last roof sample
    y_top = np.sqrt(g**2 - 2 * H * h + h**2)
    roof_end = np.sqrt(far**2 - 2 * H * h + h**2)
    shadow_end = far * H / (H - h)
    assert roof_end > g, "test needs a wide building"

    zones = [("in front", Y[0], y_top, 1), ("ground+wall+roof", y_top, g, 3),
             ("roof only", g, roof_end, 1), ("image shadow", roof_end, shadow_end, 0),
             ("behind", shadow_end, Y[-1], 1)]
    for name, lo, hi, want in zones:
        m, mn, mx = mean_cov(Y, cov, lo, hi)
        print(f"  wide  {name:<18} [{lo:8.2f}, {hi:8.2f})  coverage {m:5.3f}  (expect {want})")
        assert abs(m - want) < 0.02 and mn > want - 0.05 and mx < want + 0.05, (name, m, mn, mx)

    lab = labels_from_coverage(cov)
    s_lo = edge(Y, lab == SHADOW)
    s_hi = Y[np.where(lab == SHADOW)[0].max()]
    print(f"  wide  image shadow {s_lo:.2f}..{s_hi:.2f}  vs closed form {roof_end:.2f}..{shadow_end:.2f}")
    assert abs(s_lo - roof_end) <= 2 * DY and abs(s_hi - shadow_end) <= 2 * DY
    assert s_lo < g + w, "image shadow must start inside the footprint"


def test_narrow_building():
    g, w, h = 1100.0, 5.0, 20.0
    _, r, a, rr = ground_line((g, w, h))
    cov, _, _ = project_line(r, a, rr)
    far = g + w - DY
    y_top = np.sqrt(g**2 - 2 * H * h + h**2)
    roof_end = np.sqrt(far**2 - 2 * H * h + h**2)
    shadow_end = far * H / (H - h)
    assert roof_end < g
    zones = [("ground+wall+roof", y_top, roof_end, 3), ("ground+wall", roof_end, g, 2),
             ("image shadow", g, shadow_end, 0)]
    for name, lo, hi, want in zones:
        m, mn, mx = mean_cov(Y, cov, lo, hi)
        print(f"  narrow {name:<17} [{lo:8.2f}, {hi:8.2f})  coverage {m:5.3f}  (expect {want})")
        assert abs(m - want) < 0.02, (name, m)


def test_reference_surface_offset():
    """Reference 5 m above true ground: ground shifts AWAY from the sensor."""
    zr = 5.0
    g, w, h = 1100.0, 40.0, 20.0
    _, r, a, rr = ground_line((g, w, h), ref_height=zr)
    cov, _, _ = project_line(r, a, rr)
    # roof near edge lands where the 5 m reference surface has the same range
    r_top = np.hypot(H - h, g)
    y_top = np.sqrt(r_top**2 - (H - zr) ** 2)
    first_lay = edge(Y, cov >= 1.5)
    shift_ground = np.sqrt(1300.0**2 + 2 * H * zr - zr**2) - 1300.0
    print(f"  ref +5 m  layover starts {first_lay:.2f}, closed form {y_top:.2f};"
          f"  ground displaced {shift_ground:.2f} m away from sensor")
    assert abs(first_lay - y_top) <= 2 * DY
    m, _, _ = mean_cov(Y, cov, 1200, 1450)
    assert abs(m - 1) < 1e-6


def test_far_field_limit():
    """H -> infinity at fixed incidence recovers h/tan and h*tan."""
    Hf, th, h, w = 1.0e6, np.radians(35.0), 20.0, 60.0
    y0 = Hf * np.tan(th)
    y = np.arange(y0 - 150, y0 + 150, DY)
    g = y0 - 20
    _, r, a, rr = ground_line((g, w, h), H=Hf, y=y)
    cov, _, _ = project_line(r, a, rr)
    lab = labels_from_coverage(cov)
    lay = y[lab == LAYOVER]
    shd = y[lab == SHADOW]
    lay_len = lay.max() - lay.min() + DY
    shd_len = shd.max() - shd.min() + DY
    want_lay = h / np.tan(th)
    want_shd = h * np.tan(th) + h / np.tan(th)
    print(f"  far field  layover {lay_len:.2f} m (h/tan = {want_lay:.2f});"
          f"  image shadow {shd_len:.2f} m (h*tan + h/tan = {want_shd:.2f})")
    assert abs(lay_len - want_lay) < 0.6 and abs(shd_len - want_shd) < 0.6


def test_two_buildings_fill_geometric_shadow():
    """
    The hypothesis behind the shadow failure: a rear building's layover lands
    on the front building's geometric shadow, so ground-space 'shadow' cells
    are bright in the image.
    """
    front, rear = (1100.0, 15.0, 20.0), (1135.0, 15.0, 20.0)
    z, r, a, rr = ground_line(front, rear)
    cov, _, vis = project_line(r, a, rr)
    ground_shadow = (~vis) & (z == 0)            # what v2 labels call shadow
    lab = labels_from_coverage(cov)
    front_shadow = ground_shadow & (Y < rear[0])  # between the buildings
    rear_shadow = ground_shadow & (Y >= rear[0])  # behind the rear building
    lit_front = (cov[front_shadow] >= 0.5).mean()
    lit_rear = (cov[rear_shadow] >= 0.5).mean()
    print(f"  two buildings: front building's geometric shadow, {front_shadow.sum()} cells:"
          f" {lit_front:.0%} receive returns in the image"
          f" ({(lab[front_shadow] == LAYOVER).mean():.0%} layover)")
    print(f"                 rear building's geometric shadow, {rear_shadow.sum()} cells:"
          f" {lit_rear:.0%} receive returns (genuinely dark)")
    assert lit_front > 0.9 and lit_rear < 0.05


def test_shadow_exit_has_no_gap():
    g, w, h = 1100.0, 40.0, 20.0
    _, r, a, rr = ground_line((g, w, h))
    cov, _, _ = project_line(r, a, rr)
    shadow_end = (g + w - DY) * H / (H - h)
    after = cov[(Y > shadow_end + DY) & (Y < shadow_end + 3)]
    print(f"  pixels just past the shadow: coverage {after.min():.3f}..{after.max():.3f}")
    assert after.min() > 0.99


def test_nan_gap():
    _, r, a, rr = ground_line((1100.0, 40.0, 20.0))
    r[2000:2040] = np.nan
    a[2000:2040] = np.nan
    cov, unk, _ = project_line(r, a, rr, nan_margin=100)
    assert unk[1900:2140].all(), "gap and margin must be unknown"
    assert not unk[:1800].any()
    lab = labels_from_coverage(cov, unk)
    assert (lab[2000:2040] == UNKNOWN).all()
    print(f"  NaN gap: {unk.sum()} unknown pixels, no crash")


def test_registration_recovers_shift():
    rng = np.random.default_rng(0)
    rows, cols = 1200, 6000
    lab = np.zeros((rows, cols), np.uint8)
    for c in range(80, cols - 200, 160):
        r0 = int(rng.integers(300, 800))
        lab[r0 - 100:r0, c:c + 90] = LAYOVER
        lab[r0 + 20:r0 + 70, c:c + 90] = SHADOW
    hh = rng.normal(-11, 3, (rows, cols)).astype(np.float32)
    hh[lab == LAYOVER] += 6
    hh[lab == SHADOW] -= 8
    hh = np.roll(hh, (37, -11), axis=(0, 1))        # image 37 rows further away
    _, (dy, dx, pk) = registration_check(hh, lab)
    print(f"  registration: recovered shift ({dy:+.0f}, {dx:+.0f}), peak {pk:.2f}  (expect +37, -11)")
    assert (dy, dx) == (37, -11)


# ----------------------------------------------------------------------
# rule B: every cell drawn where it stands, showing its own range
# ----------------------------------------------------------------------
def test_in_place_flat_ground():
    _, r, a, _ = ground_line()
    cov, unk, vis = own_range_line(r, a)
    inner = cov[2:-2]
    assert np.allclose(inner, 1.0, atol=1e-6), (inner.min(), inner.max())
    assert not unk.any()
    print(f"  in place, flat ground: coverage {inner.min():.6f}..{inner.max():.6f}  (expect 1)")


def test_in_place_wide_building():
    """
    Rule B on a 40 m deep, 20 m tall box. Closed form:
      ground in front, from y_top to the wall      ground + wall + roof = 3
      roof, from the wall to y_r                   roof + wall + ground = 3
          (y_r: roof point at the same range as the wall foot)
      rest of the roof                             1
      geometric shadow, back wall to shadow_end    0
    This is the v2 layout: passive, active, nominal roof, shadow.
    """
    g, w, h = 1100.0, 40.0, 20.0
    _, r, a, _ = ground_line((g, w, h))
    cov, _, _ = own_range_line(r, a)
    y_top = np.sqrt(g**2 - 2 * H * h + h**2)
    y_r = np.sqrt(g**2 + 2 * H * h - h**2)
    far = g + w - DY
    shadow_end = far * H / (H - h)
    zones = [("in front", Y[0], y_top, 1), ("ground in front", y_top, g, 3),
             ("roof, near part", g, y_r, 3), ("roof, far part", y_r, g + w, 1),
             ("geometric shadow", g + w, shadow_end, 0), ("behind", shadow_end, Y[-1], 1)]
    for name, lo, hi, want in zones:
        m, mn, mx = mean_cov(Y, cov, lo, hi)
        print(f"  in place {name:<17} [{lo:8.2f}, {hi:8.2f})  coverage {m:5.3f}  (expect {want})")
        assert abs(m - want) < 0.02 and mn > want - 0.05 and mx < want + 0.05, (name, m, mn, mx)
    lab = labels_from_coverage(cov)
    s_idx = np.where(lab == SHADOW)[0]
    print(f"  in place: shadow {Y[s_idx.min()]:.2f}..{Y[s_idx.max()]:.2f}"
          f"  vs back wall {g + w:.2f} .. {shadow_end:.2f}")
    assert abs(Y[s_idx.min()] - (g + w)) <= 2 * DY, "shadow starts at the back wall"
    assert abs(Y[s_idx.max()] - shadow_end) <= 2 * DY


def test_rules_differ_where_expected():
    """Same box: rule A darkens the far footprint, rule B keeps the roof there."""
    g, w, h = 1100.0, 40.0, 20.0
    _, r, a, rr = ground_line((g, w, h))
    lab_a = labels_from_coverage(project_line(r, a, rr)[0])
    lab_b = labels_from_coverage(own_range_line(r, a)[0])
    fp_far = (Y > g + w / 2) & (Y < g + w - 2 * DY)
    front = (Y > g - 25) & (Y < g - 5)
    print(f"  far half of footprint: rule A {np.mean(lab_a[fp_far] == SHADOW):.0%} shadow,"
          f" rule B {np.mean(lab_b[fp_far] == SHADOW):.0%} shadow")
    print(f"  25..5 m in front:      rule A {np.mean(lab_a[front] == LAYOVER):.0%} layover,"
          f" rule B {np.mean(lab_b[front] == LAYOVER):.0%} layover")
    assert np.mean(lab_a[fp_far] == SHADOW) > 0.9 and np.mean(lab_b[fp_far] == SHADOW) == 0
    assert np.mean(lab_a[front] == LAYOVER) > 0.9 and np.mean(lab_b[front] == LAYOVER) > 0.9


def test_in_place_neighbour_fills_shadow():
    """
    Rule B with two buildings 20 m apart: the front building's geometric
    shadow shares its range with the rear building's wall, so it is not dark.
    v2 called it shadow. The rear building's shadow has nothing behind it.
    """
    front, rear = (1100.0, 15.0, 20.0), (1135.0, 15.0, 20.0)
    z, r, a, _ = ground_line(front, rear)
    cov, _, vis = own_range_line(r, a)
    lab = labels_from_coverage(cov)
    geo = (~vis) & (z == 0)
    f_sh = geo & (Y < rear[0])
    r_sh = geo & (Y >= rear[0])
    print(f"  in place, two buildings: front geometric shadow {np.mean(lab[f_sh] == SHADOW):.0%} dark"
          f" ({np.mean(lab[f_sh] == LAYOVER):.0%} layover, filled by the rear wall);"
          f" rear {np.mean(lab[r_sh] == SHADOW):.0%} dark")
    assert np.mean(lab[f_sh] == SHADOW) < 0.05 and np.mean(lab[r_sh] == SHADOW) > 0.95


def test_in_place_nan_gap_and_grid():
    z, r, a, _ = ground_line((1100.0, 40.0, 20.0))
    r[2000:2040] = np.nan
    a[2000:2040] = np.nan
    cov, unk, _ = own_range_line(r, a, nan_margin=100)
    assert unk[1900:2140].all() and not unk[:1700].any()
    R = np.stack([r, r], axis=1)
    A = np.stack([a, a], axis=1)
    cg, ug, _ = own_range_grid(R, A, axis=0, nan_margin=100)
    assert np.allclose(cg[:, 1], cov, atol=1e-5) and (ug[:, 0] == unk).all()
    cg1, _, _ = own_range_grid(R.T, A.T, axis=1, nan_margin=100)
    assert np.allclose(cg1.T, cg, atol=1e-6)
    print(f"  in place NaN gap: {unk.sum()} unknown pixels; grid matches line on both axes")


def test_registration_bands_and_shadow_only():
    rng = np.random.default_rng(1)
    rows, cols = 1800, 4000
    lab = np.zeros((rows, cols), np.uint8)
    for c in range(60, cols - 200, 150):
        for r0 in (300, 900, 1500):
            r0 += int(rng.integers(-60, 60))
            lab[r0 - 80:r0, c:c + 80] = LAYOVER
            lab[r0 + 20:r0 + 70, c:c + 80] = SHADOW
    hh = rng.normal(-11, 3, (rows, cols)).astype(np.float32)
    hh[lab == LAYOVER] += 6
    hh[lab == SHADOW] -= 8
    hh = np.roll(hh, (12, 0), axis=(0, 1))
    res, (dy, dx, pk) = registration_check(hh, lab, size=512, n_windows=4, search=100,
                                           range_bands=3)
    bands = sorted({b for b, *_ in res})
    res_s, (dys, dxs, _) = registration_check(hh, lab, size=512, n_windows=4, search=100,
                                              range_bands=3, use=("shadow",))
    print(f"  registration, 3 bands {bands}: ({dy:+.0f}, {dx:+.0f}); shadow only ({dys:+.0f}, {dxs:+.0f})"
          f"  (expect +12, 0)")
    assert bands == [0, 1, 2] and (dy, dx) == (12, 0) and (dys, dxs) == (12, 0)


# ----------------------------------------------------------------------
# layover magnitude: how far the echoes in a pixel have come from
# ----------------------------------------------------------------------
def test_magnitude_flat_ground():
    _, r, a, _ = ground_line()
    d, unk = own_range_displacement(r, a, pixel_m=DY)
    print(f"  magnitude, flat ground: max {d[5:-5].max():.2f} m  (expect under 1, the bin size)")
    assert d[5:-5].max() < 1.0 and not unk.any()


def test_magnitude_wide_building():
    """40 m deep, 20 m tall: the roof's echo reaches the whole strip in front."""
    g, w, h = 1100.0, 40.0, 20.0
    z, r, a, _ = ground_line((g, w, h))
    d, _ = own_range_displacement(r, a, pixel_m=DY)
    cov = own_range_line(r, a)[0]
    y_top = np.sqrt(g**2 - 2 * H * h + h**2)
    front = (Y > y_top + 1) & (Y < g - 1)
    want = np.sqrt(Y[front] ** 2 + 2 * H * h - h**2) - Y[front]      # distance to the roof point
    err = np.abs(d[front] - want)
    shadow_end = (g + w - DY) * H / (H - h)
    shadow = (Y > g + w + 1) & (Y < shadow_end - 1)
    behind = Y > shadow_end + 2
    print(f"  magnitude, wide building: front {d[front].min():.1f}-{d[front].max():.1f} m vs closed form "
          f"{want.min():.1f}-{want.max():.1f} m (max err {err.max():.2f}); shadow {d[shadow].max():.2f}; "
          f"behind {d[behind].max():.2f}")
    assert err.max() < 0.6 and d[shadow].max() == 0 and d[behind].max() < 1.0
    mag = layover_magnitude(cov, d)
    assert (mag[cov < 1.5] == 0).all() and np.allclose(mag[front], d[front])


def test_magnitude_narrow_building():
    """
    5 m deep, 20 m tall. Near the wall only the wall mixes in, so the
    magnitude is the distance to the wall; further out the roof's echo also
    lands, about h / tan(theta) away. On the roof, the ground echo in front
    lands, the same distance back.
    """
    g, w, h = 1100.0, 5.0, 20.0
    z, r, a, _ = ground_line((g, w, h))
    d, _ = own_range_displacement(r, a, pixel_m=DY)
    y_top = np.sqrt(g**2 - 2 * H * h + h**2)
    split = np.sqrt((g + w - DY) ** 2 - 2 * H * h + h**2)           # where the roof echo starts
    near = (Y > split + 1) & (Y < g - 1)
    far = (Y > y_top + 1) & (Y < split - 1)
    roof = (Y > g + 0.5) & (Y < g + w - 0.75)
    e_near = np.abs(d[near] - (g - DY / 2 - Y[near])).max()
    e_far = np.abs(d[far] - (np.sqrt(Y[far] ** 2 + 2 * H * h - h**2) - Y[far])).max()
    e_roof = np.abs(d[roof] - (Y[roof] - np.sqrt(Y[roof] ** 2 - 2 * H * h + h**2))).max()
    print(f"  magnitude, narrow building: near the wall err {e_near:.2f} m (ramp to {d[near].max():.1f} m), "
          f"roof-echo zone err {e_far:.2f} m (~{d[far].mean():.1f} m), on the roof err {e_roof:.2f} m")
    assert e_near < 0.6 and e_far < 0.6 and e_roof < 0.6


def test_magnitude_grid_and_nan():
    _, r, a, _ = ground_line((1100.0, 40.0, 20.0))
    r[2000:2040] = np.nan
    a[2000:2040] = np.nan
    d, unk = own_range_displacement(r, a, nan_margin=100, pixel_m=DY)
    assert unk[1900:2140].all() and np.isfinite(d).all()
    R, A = np.stack([r, r], 1), np.stack([a, a], 1)
    dg, ug = own_range_displacement_grid(R, A, axis=0, nan_margin=100, pixel_m=DY)
    assert np.allclose(dg[:, 0], d, atol=1e-4) and (ug[:, 1] == unk).all()
    print(f"  magnitude grid matches the line; NaN gap -> {unk.sum()} unknown pixels")


if __name__ == "__main__":
    import sys
    import traceback
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        print(t.__name__)
        try:
            t()
        except AssertionError:
            failed += 1
            print("  FAIL")
            traceback.print_exc(limit=2)
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
