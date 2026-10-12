"""
Closed-form tests for image_space.py.

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
                         line_geometry, project_line, registration_check)

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
