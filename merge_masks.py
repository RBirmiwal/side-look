#!/usr/bin/env python3
"""
Rebuild a relabelled dataset (v4, v5, ...) on the cluster from the v2 tiles
you already have plus a small "mask pack" from Grok, instead of copying
~10 GB of tiles.

The new datasets use exactly the v2 images (same tiles, grid, split,
orientation); only the labels changed. The pack holds just the new labels:

    v5_masks/
      manifest.csv     split,name,v2_split,v2_name,image_md5,label_version
      classes.json     ["nominal", "layover", "shadow"]
      train/<name>.npz  mask      uint8   (512,512)  0/1/2, 255 = ignore
      val/...           coverage  float16 (512,512)
      test/...          magnitude float16 (512,512)  metres, optional
                        (layover pixels: displacement; every other labelled
                        pixel 0; if one tile has it, every tile must)

    python merge_masks.py --v2 side-look-tiles-v2 --pack v5_masks \
        --out side-look-tiles-v5

Every v2 image is checked against the md5 Grok recorded, so a mismatch
between your v2 copy and Grok's stops the merge before anything is written.
--check-only runs the checks and writes nothing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

MAX_MAGNITUDE_M = 1000.0     # garbage guard only; a 200 m tower at 24 deg can exceed 400 m


def image_md5(arr: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()


def new_meta(old, label_version: str) -> str:
    try:
        text = old.item() if isinstance(old, np.ndarray) else old
        if isinstance(text, bytes):
            text = text.decode()
        meta = json.loads(text)
        if not isinstance(meta, dict):
            raise ValueError
    except Exception:
        meta = {"v2_meta": str(old)}
    meta["label_version"] = label_version
    return json.dumps(meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", type=Path, required=True, help="v2 tile directory")
    ap.add_argument("--pack", type=Path, required=True, help="unpacked v4 mask pack")
    ap.add_argument("--out", type=Path, required=True, help="v4 tile directory to write")
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    rows = list(csv.DictReader((args.pack / "manifest.csv").open()))
    need = {"split", "name", "v2_split", "v2_name", "image_md5", "label_version"}
    if not rows or not need <= set(rows[0]):
        sys.exit(f"manifest.csv must have columns {sorted(need)}")
    classes = json.loads((args.pack / "classes.json").read_text())
    nc = len(classes)
    lay_ids = [classes.index(c) for c in ("layover", "active", "passive") if c in classes]
    print(f"{len(rows)} tiles in the pack, classes {classes}")
    has_mag = set()

    # ---- pass 1: check everything, write nothing -------------------------
    problems = []
    for r in rows:
        src = args.v2 / r["v2_split"] / r["v2_name"]
        lab = args.pack / r["split"] / r["name"]
        if not src.exists():
            problems.append(f"missing v2 tile {src}")
            continue
        if not lab.exists():
            problems.append(f"missing pack file {lab}")
            continue
        img = np.load(src)["image"]
        if image_md5(img) != r["image_md5"]:
            problems.append(f"image md5 differs from Grok's for {src}")
            continue
        m = np.load(lab)
        mask, cov = m["mask"], m["coverage"]
        if mask.shape != img.shape[1:] or cov.shape != img.shape[1:]:
            problems.append(f"shape mismatch {lab}: mask {mask.shape}, coverage {cov.shape}, "
                            f"image {img.shape}")
            continue
        bad = (mask != 255) & (mask >= nc)
        if bad.any():
            problems.append(f"{lab}: mask value {int(mask[bad].max())} with {nc} classes")
        has_mag.add("magnitude" in m.files)
        if "magnitude" in m.files:
            mag = m["magnitude"].astype(np.float32)
            labelled = mask != 255
            off = labelled & ~np.isin(mask, lay_ids)
            if mag.shape != mask.shape:
                problems.append(f"{lab}: magnitude {mag.shape}, mask {mask.shape}")
            elif not np.isfinite(mag[labelled]).all():
                problems.append(f"{lab}: magnitude has NaN/inf on labelled pixels")
            elif (mag[labelled] < 0).any() or (mag[labelled] > MAX_MAGNITUDE_M).any():
                problems.append(f"{lab}: magnitude outside 0..{MAX_MAGNITUDE_M:g} m "
                                f"(min {mag[labelled].min():.1f}, max {mag[labelled].max():.1f})")
            elif (mag[off] != 0).any():
                problems.append(f"{lab}: magnitude non-zero on {int((mag[off] != 0).sum())} "
                                f"labelled non-layover pixels (must be 0 there)")
    if len(has_mag) > 1:
        problems.append("some pack files have 'magnitude' and some do not")
    if problems:
        print(f"\n{len(problems)} problem(s); nothing written. First ones:")
        for p in problems[:10]:
            print("  " + p)
        sys.exit(1)
    with_mag = has_mag == {True}
    print("all v2 images match Grok's md5; masks and coverage have the right shape"
          + ("; magnitude present and valid in every tile" if with_mag else "; no magnitude"))
    if args.check_only:
        return

    # ---- pass 2: write ---------------------------------------------------
    counts, ignored, total = Counter(), 0, 0
    mag_vals = []
    for r in rows:
        z = np.load(args.v2 / r["v2_split"] / r["v2_name"])
        m = np.load(args.pack / r["split"] / r["name"])
        out = {k: z[k] for k in z.files if k not in ("mask", "coverage", "magnitude", "meta")}
        out["mask"] = m["mask"].astype(np.uint8)
        out["coverage"] = m["coverage"].astype(np.float16)
        if with_mag:
            mag = np.nan_to_num(m["magnitude"].astype(np.float32))
            mag[out["mask"] == 255] = 0.0
            out["magnitude"] = mag.astype(np.float16)
            if r["split"] == "train":
                v = mag[np.isin(out["mask"], lay_ids)]
                mag_vals.append(v[:: max(1, v.size // 2000)])
        out["meta"] = new_meta(z["meta"] if "meta" in z.files else "{}", r["label_version"])
        dst = args.out / r["split"] / r["name"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez(dst, **out)
        counts[r["split"]] += 1
        ignored += int((out["mask"] == 255).sum())
        total += out["mask"].size

    shutil.copy(args.pack / "classes.json", args.out / "classes.json")
    shutil.copy(args.v2 / "norm_stats.json", args.out / "norm_stats.json")
    for extra in ["manifest.csv", *sorted(p.name for p in args.pack.glob("REPORT*.md"))]:
        if (args.pack / extra).exists():
            shutil.copy(args.pack / extra, args.out / extra)
    print(f"wrote {sum(counts.values())} tiles to {args.out}: "
          + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
          + f"; ignored pixels {ignored / max(total, 1):.1%}")
    if mag_vals:
        v = np.concatenate(mag_vals)
        if v.size:
            q = np.percentile(v, [10, 50, 90, 99])
            print(f"train layover magnitude (sampled): mean {v.mean():.1f} m, "
                  f"p10 {q[0]:.1f}, median {q[1]:.1f}, p90 {q[2]:.1f}, p99 {q[3]:.1f} m")


if __name__ == "__main__":
    main()
