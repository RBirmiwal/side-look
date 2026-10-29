#!/usr/bin/env python3
"""Sheet, report, merge check, and the two release archives for dataset v5."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v5_build import (  # noqa: E402
    GEOM, K_ID, LABEL_VERSION, OUT, PACK, PX, ROOT, STRIPS, TILES, V2, V4_REF,
    log, rd_bounds, strip_grid, tiles_of, to_utm, fetch_footprints,
)

SHEET = OUT / "magnitude_sheet.png"


def stamps():
    rows = []
    for s in STRIPS:
        p = OUT / "strips" / f"{s}.json"
        if not p.exists():
            raise SystemExit(f"missing stamp {p}")
        rows.append(json.loads(p.read_text()))
    return rows


def pooled_ratio(sts):
    ratios = []
    for st in sts:
        ratios.extend(st.get("ratios") or [])
    ratios = np.array(ratios, float)
    if ratios.size > 200:
        rng = np.random.default_rng(1)
        ratios = rng.choice(ratios, 200, replace=False)
    if ratios.size == 0:
        return dict(n=0, median=None, p10=None, p90=None)
    return dict(n=int(ratios.size), median=float(np.median(ratios)),
                p10=float(np.percentile(ratios, 10)), p90=float(np.percentile(ratios, 90)))


def mag_by_split():
    """Layover-pixel magnitude stats, every kept tile, no double count beyond the tile list."""
    edges = np.linspace(0, 500, 5001)
    hist = {s: np.zeros(5000, np.int64) for s in ("train", "val", "test")}
    total = {s: 0 for s in hist}
    below = {s: 0 for s in hist}
    sm = {s: 0.0 for s in hist}
    mx = {s: 0.0 for s in hist}
    cls = {s: np.zeros(4, np.int64) for s in hist}  # 0,1,2,255
    for split in hist:
        for p in (PACK / split).glob("*.npz"):
            z = np.load(p)
            m = z["mask"]
            mag = z["magnitude"].astype(np.float32)
            for i, v in enumerate((0, 1, 2, 255)):
                cls[split][i] += int((m == v).sum())
            lay = mag[m == 1]
            if lay.size == 0:
                continue
            hist[split] += np.histogram(np.clip(lay, 0, 499.999), bins=edges)[0]
            total[split] += int(lay.size)
            below[split] += int((lay < 1).sum())
            sm[split] += float(lay.sum())
            mx[split] = max(mx[split], float(lay.max()))
    out = {}
    for s in hist:
        c = np.cumsum(hist[s])
        def pct(q):
            if c[-1] == 0:
                return None
            i = int(np.searchsorted(c, c[-1] * q))
            return float(edges[min(i, len(edges) - 2)])
        out[s] = dict(n=total[s],
                      mean=(sm[s] / total[s]) if total[s] else None,
                      median=pct(0.5), p90=pct(0.9), p99=pct(0.99),
                      max=mx[s] if total[s] else None,
                      below_1m=(below[s] / total[s]) if total[s] else None,
                      counts={k: int(v) for k, v in zip(("nominal", "layover", "shadow", "ignore"), cls[s])})
    return out


def _hh_rgb(hh):
    lo, hi = np.percentile(hh, [2, 98])
    x = np.clip((hh - lo) / max(hi - lo, 1e-3), 0, 1)
    g = (x * 255).astype(np.uint8)
    return np.dstack([g, g, g])


def _overlay(base, color, alpha):
    out = base.astype(np.float32)
    m = alpha[..., None]
    out = out * (1 - m) + color.astype(np.float32) * m
    return out.astype(np.uint8)


def pick_tiles(sts):
    """3 val/test tiles with the tallest flat-ish roofs, 3 random with >= 3% layover."""
    tall = {}
    for strip in STRIPS:
        grid = strip_grid(strip)
        fps = to_utm(fetch_footprints(rd_bounds(grid, 250)))
        tiles = [t for t in tiles_of(strip) if t["split"] in ("val", "test")]
        if not tiles or not fps:
            continue
        bounds = np.array([g.bounds for g, _ in fps])
        heights = []
        for _, a in fps:
            h70, mv = a.get("roof70"), a.get("ground")
            if h70 is None or mv is None:
                heights.append(-1.0)
            else:
                heights.append(float(h70) - float(mv))
        heights = np.array(heights)
        for t in tiles:
            x0 = grid["west"] + t["gc"] * PX
            yN = grid["north"] - t["gr"] * PX
            m = ((bounds[:, 2] >= x0) & (bounds[:, 0] <= x0 + 128)
                 & (bounds[:, 3] >= yN - 128) & (bounds[:, 1] <= yN))
            h = float(heights[m].max()) if m.any() else -1
            tall[(t["split"], t["name"])] = (h, strip)
    # layover fraction
    info = []
    for split in ("val", "test"):
        for p in (PACK / split).glob("*.npz"):
            m = np.load(p)["mask"]
            lay = float((m == 1).mean())
            h, strip = tall.get((split, p.name), (-1, p.name.split("_")[0]))
            info.append(dict(split=split, name=p.name, lay=lay, height=h, strip=strip, path=p))
    info.sort(key=lambda r: r["height"], reverse=True)
    chosen = info[:3]
    pool = [r for r in info if r["lay"] >= 0.03 and r["name"] not in {c["name"] for c in chosen}]
    rng = np.random.default_rng(0)
    if len(pool) >= 3:
        extra = list(rng.choice(pool, 3, replace=False))
    else:
        extra = pool
    return chosen, extra


def draw_sheet(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import viridis

    fig, axes = plt.subplots(len(rows), 3, figsize=(10.2, 3.15 * len(rows)), dpi=110)
    if len(rows) == 1:
        axes = np.array([axes])
    norm = Normalize(vmin=0, vmax=60)
    last_im = None
    for i, r in enumerate(rows):
        z = np.load(r["path"])
        # pack has no HH; it lives on the v2 tile
        v2 = V2 / r["split"] / r["name"]
        hh = np.load(v2)["image"][0].astype(np.float32)
        base = _hh_rgb(hh)
        mask = z["mask"]
        mag = z["magnitude"].astype(np.float32)
        axes[i, 0].imshow(base)
        # mask: layover copper, shadow steel-blue, 255 and nominal clear
        color = np.zeros_like(base)
        alpha = np.zeros(mask.shape, np.float32)
        color[mask == 1] = (184, 115, 51)
        alpha[mask == 1] = 0.72
        color[mask == 2] = (70, 110, 170)
        alpha[mask == 2] = 0.65
        axes[i, 1].imshow(_overlay(base, color, alpha))
        show = np.ma.masked_where((mag <= 0) | (mask == 255), mag)
        last_im = axes[i, 2].imshow(base)
        last_im = axes[i, 2].imshow(show, cmap=viridis, norm=norm, alpha=0.9)
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
        kind = "tall" if r.get("kind") == "tall" else "random"
        axes[i, 0].set_ylabel(f"{kind}\n{r['split']}\n{r['height']:.0f} m", fontsize=8)
        axes[i, 0].set_title("HH" if i == 0 else "", fontsize=10)
        axes[i, 1].set_title("mask" if i == 0 else "", fontsize=10)
        axes[i, 2].set_title("magnitude" if i == 0 else "", fontsize=10)
    fig.colorbar(last_im, ax=axes[:, 2], fraction=0.046, pad=0.04, label="metres")
    fig.suptitle("Layover magnitude. Sensor at the top. Ignore and 0 m left clear.", fontsize=11, y=1.002)
    fig.tight_layout()
    SHEET.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SHEET)
    log(f"sheet {SHEET}")


def write_report(sts, ratio, mag):
    lines = []
    w = lines.append
    w("# Dataset v5")
    w("")
    w("Same 15 strips as v4. Rule B (`own_range_grid`) on a buildings-only scene,")
    w("AHN4 DTM as the ground, then 3DBAG footprints dilated 2 m set to 255, plus a")
    w("per-pixel layover magnitude in metres. v2 was not modified. Images, incidence,")
    w("look, grid and split come from the v2 tiles. `label_version` is `v5-agreed-mag`.")
    w("")
    w("The v4 mask rasters are not on this machine, so this is not a pixel-for-pixel")
    w("compare. The table below is the v4 recipe (mask before the footprint 255) scored")
    w("on the v2 tiles the same way REPORT_v4.md scored them: overlapping train tiles")
    w("count twice, water stays labelled, ignored pixels stay out of the brightness.")
    w("")
    w("## Choices")
    w("")
    w("- AHN4 `dsm_05m` and `dtm_05m` from the PDOK WCS, 0.5 m, bilinear onto the 0.25 m")
    w("  MAG-POL grid in EPSG:32631. Each mosaic was checked against a fresh WCS chip.")
    w("- Range pad 250 m (1000 px). `nan_margin` 400 cells (100 m), same count as v4.")
    w("- Water: DSM nodata components of at least 2500 m² inside the strip, filled with")
    w("  the median of the edge below 2 m NAP (else the 10th percentile). The pad is not water.")
    w("- Holes under 100 m²: median of a 2 px ring on the 0.5 m grid. Whatever nodata is")
    w("  left, DSM and DTM, is then filled from the nearest finite cell. That is what")
    w("  makes the unknown mask empty, which is what v4 reported. The DSM is not copied")
    w("  into DTM holes: that would paste the roof in as the ground.")
    w("- Footprints are 3DBAG v20250903 `pand` polygons (gpkg tiles), not the sidelook-v2")
    w("  index. Heights are `lod22_2d` `b3_h_70p` / `b3_h_max` in metres NAP. The ratio")
    w("  check uses height = `b3_h_70p − b3_h_maaiveld`.")
    w("- Dilations at 0.25 m: 4 px (1 m) around footprints before the building surface is")
    w("  pasted, 8 px (2 m) for clutter and for the v5 footprint zone.")
    w("- Ground range is `H tan(incidence)` from the v2 tiles, one linear fit per column.")
    w("  Columns with no tile use the strip's median fit, so the full strip can be scored.")
    w("  H is the v2 platform height. The fit slope is about 0.25 m per source row, which")
    w("  is one ground-range metre per ground metre.")
    w("- A tile is kept if it has at least one labelled pixel, including tiles that are")
    w("  more than 60% ignored. v4 dropped those.")
    w("- Magnitude is `layover_magnitude` at step 0.1 m and pixel size 0.25 m. It is 0")
    w("  off layover and 0 wherever the mask is 255. Coverage is the buildings-only")
    w("  coverage, including on ignored pixels.")
    w("")
    w("## Against v4, on the v2 tiles")
    w("")
    w("| strip | v4 ignored | v5 full-strip ignored | v4 near dB | v5 near | v4 middle | v5 middle | v4 drop >60% | v5 >60% (kept) |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for st in sts:
        s = st["strip"]
        ref = V4_REF[s]
        w(f"| `{s[-14:]}` | {ref['ignored']:.1%} | {st['full']['ignored']:.1%} | "
          f"{ref['near']:+.2f} | {st['contrast']['near']:+.2f} | {ref['middle']:+.2f} | "
          f"{st['contrast']['middle']:+.2f} | {ref['drop']} | {st['over60']} |")
    w("")
    w("Ignored on the full strip is higher than v4 on several strips. The two label sets")
    w("were not built from the same 3DBAG extract, and v5 fills DTM holes from the nearest")
    w("ground cell rather than whatever v4 had cached. Water on K is 17.0% against v4's")
    w("16.7%. K's middle-third contrast is +3.30 dB against v4's +3.32. The near third")
    w("is about half a decibel higher. Unknown from the projection is under 0.1% on K")
    w(f"(v4 reported 0). K unknown fraction {next(s['full']['unk_d'] for s in sts if s['strip']==K_ID):.3%}.")
    w("")
    w("## Magnitude / L")
    w("")
    w("Flat roofs (`b3_h_max − b3_h_70p < 2`), height at least 15 m, deeper along range")
    w("than `layover_len_m`, with the zone from 2 m to L in front of the near wall inside")
    w("one tile. L is `g − sqrt(g² − 2Hh + h²)` with g from the tile incidence at the wall.")
    w("")
    w(f"n = {ratio['n']} (capped at 200), median {ratio['median']:.3f}, "
      f"p10 {ratio['p10']:.3f}, p90 {ratio['p90']:.3f}." if ratio["median"] is not None else "no buildings qualified")
    w("")
    w("A units mix-up (pixels taken as metres, or the range axis reversed) would put the")
    w("median near 4, 0.25 or 0. The gate is the median, 0.8–1.2. A single building can")
    w("sit well below that when its cadastral near edge is not the tall wall: the roof")
    w("echo then lands inside the footprint, which v5 marks 255, and the only labelled")
    w("layover in front of that edge is a short band. Those buildings pull the 10th")
    w("percentile down. They are not a scale error. Kop van Zuid, where the tall wall")
    w("is the near wall, is the check that the metres are right.")
    w("")
    w("## Magnitude by split, layover pixels")
    w("")
    w("| split | n | mean | median | p90 | p99 | max | share < 1 m |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|")
    for s in ("train", "val", "test"):
        d = mag[s]
        if not d["n"]:
            w(f"| {s} | 0 | | | | | | |")
            continue
        w(f"| {s} | {d['n']} | {d['mean']:.1f} | {d['median']:.1f} | {d['p90']:.1f} | "
          f"{d['p99']:.1f} | {d['max']:.1f} | {d['below_1m']:.1%} |")
    w("")
    w("## Class shares of non-ignored pixels")
    w("")
    w("| split | nominal | layover | shadow | ignored |")
    w("|---|---:|---:|---:|---:|")
    for s in ("train", "val", "test"):
        c = mag[s]["counts"]
        lab = c["nominal"] + c["layover"] + c["shadow"]
        tot = lab + c["ignore"]
        if lab == 0:
            continue
        w(f"| {s} | {c['nominal']/lab:.3f} | {c['layover']/lab:.3f} | {c['shadow']/lab:.3f} | {c['ignore']/tot:.1%} |")
    w("")
    kept = sum(sum(st["counts"].values()) for st in sts)
    over = sum(st["over60"] for st in sts)
    w(f"Kept tiles {kept}. Tiles more than 60% ignored, and kept: {over}.")
    w("")
    w("## Inputs")
    w("")
    w("- AHN WCS `https://service.pdok.nl/rws/ahn/wcs/v1_0` coverages `dsm_05m`, `dtm_05m`, CRS EPSG:28992.")
    w("- 3DBAG v20250903 `https://data.3dbag.nl/v20250903/tiles/9/{x}/{y}/9-{x}-{y}.gpkg.gz`.")
    w("- v2 tiles for the image, the incidence and the grid. No SAR image was re-downloaded.")
    w("")
    text = "\n".join(lines) + "\n"
    (OUT / "REPORT_v5.md").write_text(text)
    (PACK / "REPORT_v5.md").write_text(text)
    log("wrote REPORT_v5.md")
    return text


def main():
    sts = stamps()
    ratio = pooled_ratio(sts)
    log(f"pooled magnitude/L {ratio}")
    if ratio["median"] is None or not (0.8 <= ratio["median"] <= 1.2):
        raise SystemExit("STOP: pooled median magnitude/L outside 0.8-1.2")
    mag = mag_by_split()
    log(f"mag { {s: {k: mag[s][k] for k in ('n','mean','median','p90','p99','max','below_1m')} for s in mag} }")
    tall, rand = pick_tiles(sts)
    for r in tall:
        r["kind"] = "tall"
    for r in rand:
        r["kind"] = "random"
    rows = tall + rand
    log("sheet tiles " + ", ".join(f"{r['kind']} {r['split']} {r['name']} h={r['height']:.0f} lay={r['lay']:.1%}" for r in rows))
    draw_sheet(rows)
    write_report(sts, ratio, mag)
    # classes + norm
    (PACK / "classes.json").write_text(json.dumps(["nominal", "layover", "shadow"]))
    TILES.mkdir(parents=True, exist_ok=True)
    if not (TILES / "norm_stats.json").exists():
        shutil.copy(V2 / "norm_stats.json", TILES / "norm_stats.json")
    shutil.copy(PACK / "classes.json", TILES / "classes.json")
    shutil.copy(OUT / "REPORT_v5.md", TILES / "REPORT_v5.md")
    if (PACK / "manifest.csv").exists():
        shutil.copy(PACK / "manifest.csv", TILES / "manifest.csv")


if __name__ == "__main__":
    main()
