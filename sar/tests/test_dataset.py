"""Tiling / canonicalise / nodata rules. Synthetic arrays only."""

from __future__ import annotations

import numpy as np

from dsm_geometry import CLASS_ACTIVE, CLASS_SHADOW, mask_from_dsm
from dataset import (
    LOG_EPS,
    RunningNorm,
    canonicalise,
    drop_reason,
    log_scale,
    tile_windows,
)


def test_canonicalise_flip_makes_layover_agree():
    """South-facing layover is toward north (small row). Flip rows → matches north-facing."""
    z = np.zeros((80, 40), dtype=np.float64)
    z[30:50, :] = 12.0
    south = mask_from_dsm(z, 45.0, 180.0, 1.0)
    north = mask_from_dsm(z, 45.0, 0.0, 1.0)
    image = np.zeros((4, 80, 40), dtype=np.float32)
    flipped_img, flipped_mask, was_flipped = canonicalise(image, south, orientation_flag=1)
    assert was_flipped
    _, unflipped, was = canonicalise(image, north, orientation_flag=0)
    assert not was
    # After flipping the south mask, layover mass sits on the same side as north.
    lay_s = np.where((flipped_mask == CLASS_ACTIVE).any(axis=1))[0]
    lay_n = np.where((unflipped == CLASS_ACTIVE).any(axis=1))[0]
    assert lay_s.size and lay_n.size
    assert np.sign(lay_s.mean() - 40) == np.sign(lay_n.mean() - 40)


def test_nodata_and_empty_reasons():
    good = np.ones((4, 8, 8), dtype=np.float32)
    mask = np.zeros((8, 8), dtype=np.uint8)
    valid = np.ones((8, 8), dtype=bool)
    assert drop_reason(good, mask, valid) is None
    bad = good.copy()
    bad[0, 0, 0] = np.nan
    assert drop_reason(bad, mask, valid) == "sar_nodata"
    valid2 = valid.copy()
    valid2[1, 1] = False
    assert drop_reason(good, mask, valid2) == "dsm_nodata"
    assert drop_reason(good, mask, valid, cap_zero=0.0) == "zero_layover"


def test_log_scale_skips_db():
    db = np.array([-10.0, 5.0], dtype=np.float32)
    out, kind = log_scale(db)
    assert kind == "db"
    np.testing.assert_array_equal(out, db)
    lin = np.array([1.0, 100.0], dtype=np.float32)
    out, kind = log_scale(lin)
    assert kind == "log10"
    np.testing.assert_allclose(out, np.log10(lin + LOG_EPS))


def test_tile_windows_overlap():
    wins = list(tile_windows(1000, 2000, size=512, stride=256))
    assert wins[0] == (0, 0)
    assert (256, 0) in wins
    # val/test: no overlap
    no = list(tile_windows(1000, 2000, size=512, stride=512))
    rows = sorted({r for r, c in no})
    assert rows[0] == 0
    # last window is pinned to height - size (488), not a full extra stride
    assert rows[-1] == 1000 - 512


def test_running_norm_is_train_only():
    n = RunningNorm()
    train = np.ones((4, 4, 4), dtype=np.float32) * 2.0
    val = np.ones((4, 4, 4), dtype=np.float32) * 99.0
    n.update(train, "train")
    n.update(val, "val")
    out = n.dump()
    assert out["split"] == "train"
    np.testing.assert_allclose(out["mean"], [2, 2, 2, 2])
    assert out["n_pixels"] == 16
