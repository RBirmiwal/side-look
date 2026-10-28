#!/usr/bin/env python3
"""
Train the Side Look U-Net: layover / shadow segmentation and, optionally,
layover MAGNITUDE (how far the echoes in each pixel have been displaced, in
metres).

    python train.py --data side-look-tiles-v5 --overfit                    # 2-min sanity test
    python train.py --data side-look-tiles-v5 --out runs/v5                # classes only
    python train.py --data side-look-tiles-v5 --out runs/v5_mag --magnitude
    python train.py --data side-look-tiles-v5 --out runs/v5_mag_ft --magnitude \
        --init runs/v4/best.pt --lr 1e-4                                   # fine-tune
    python train.py --data side-look-tiles-v5 --eval runs/v5_mag/best.pt

Tiles: image float16 (4,512,512) dB, mask uint8, incidence float16 (512,512)
degrees, look uint8, meta JSON; with --magnitude also `magnitude` float16
(512,512) metres (0 off layover). Canonical orientation: sensor at row 0.

Classes come from <data>/classes.json (index 0 must be nominal). A dataset
without that file is a v2 dataset with the four original classes.

Mask value 255 means IGNORE: no label is claimed for that pixel. Ignored
pixels are left out of every loss, the class weights, every score and the
baselines.

MAGNITUDE (--magnitude): the network gets one extra output channel, read as
metres through softplus x --mag-scale. It is trained with a smooth-L1 loss on
every labelled pixel (target 0 off layover), with layover pixels weighted by
--mag-layover-weight. It is scored on true-layover pixels: mean absolute error
in metres, bias, and skill against a constant baseline that predicts the
training-set mean layover magnitude everywhere (skill 0 = no better than that
constant, 1 = perfect).

--init loads any earlier checkpoint whose tensors fit (for example the
segmentation-only v4 model) and keeps its class head, so the magnitude head
is the only part that starts from scratch.

Pixel accuracy is never reported: all-nominal scores ~77% and means nothing.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from unet import UNet

V2_CLASSES = ["nominal", "active", "passive", "shadow"]
IGNORE = 255
MAG_BINS = [(0.5, 10.0), (10.0, 25.0), (25.0, 50.0), (50.0, float("inf"))]

# Set in main() from <data>/classes.json; v2 datasets have no such file and
# keep the original four classes. Index 0 must be nominal.
CLASSES = list(V2_CLASSES)
EVAL_CLASSES = [1, 2, 3]
COLOURS = {"nominal": (0, 0, 0, 0), "active": (0.85, 0.45, 0.15, 0.6),
           "layover": (0.85, 0.45, 0.15, 0.6), "passive": (0.55, 0.40, 0.22, 0.6),
           "shadow": (0.2, 0.45, 0.85, 0.6)}


def load_classes(data: Path):
    p = data / "classes.json"
    if not p.exists():
        return list(V2_CLASSES)
    names = json.loads(p.read_text())
    if not isinstance(names, list) or not names or names[0] != "nominal":
        raise ValueError(f"{p}: expected a JSON list starting with 'nominal', got {names!r}")
    if "shadow" not in names or not ({"layover", "active"} & set(names)):
        raise ValueError(f"{p}: needs 'shadow' and 'layover' (or 'active') for the baseline")
    return names


def bright_dark_ids():
    """Class ids the brightness baseline predicts: bright -> layover, dark -> shadow."""
    bright = CLASSES.index("layover") if "layover" in CLASSES else CLASSES.index("active")
    return bright, CLASSES.index("shadow")


def layover_ids():
    """Class ids that count as layover for the magnitude scores."""
    return [CLASSES.index(c) for c in ("layover", "active", "passive") if c in CLASSES]


def make_scaler(enabled: bool):
    try:                                        # torch >= 2.4
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):         # torch 2.2 / 2.3
        return torch.cuda.amp.GradScaler(enabled=enabled)


# ----------------------------------------------------------------------
class Tiles(Dataset):
    """Returns (x, mask, magnitude); magnitude is all zeros unless requested."""

    def __init__(self, root: Path, split: str, mean, std, augment: bool,
                 inc_stats: tuple[float, float] | None = None, magnitude: bool = False):
        self.paths = sorted((root / split).glob("*.npz"))
        if not self.paths:
            raise FileNotFoundError(f"no .npz in {root / split}")
        self.mean = mean[:, None, None]
        self.std = std[:, None, None]
        self.augment = augment
        self.inc_stats = inc_stats
        self.magnitude = magnitude

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        z = np.load(self.paths[i])
        img = torch.from_numpy(z["image"].astype(np.float32))
        msk = torch.from_numpy(z["mask"].astype(np.int64))
        bad = (msk != IGNORE) & (msk >= len(CLASSES))
        if bool(bad.any()):
            raise ValueError(f"{self.paths[i].name}: mask value {int(msk[bad].max())} but only "
                             f"{len(CLASSES)} classes (255 is the only other value allowed: ignore)")
        if self.magnitude:
            if "magnitude" not in z.files:
                raise KeyError(f"{self.paths[i].name} has no 'magnitude' array; "
                               f"--magnitude needs a dataset built with it")
            mag = torch.from_numpy(np.nan_to_num(z["magnitude"].astype(np.float32)))
        else:
            mag = torch.zeros(msk.shape, dtype=torch.float32)

        inc = None
        if self.inc_stats is not None:
            if "incidence" not in z.files:
                raise KeyError(f"{self.paths[i].name} has no 'incidence' array")
            inc = torch.from_numpy(z["incidence"].astype(np.float32))[None]

        # Left-right only (along-track mirror). Vertical flips or rotations
        # would put layover on the wrong side -- tiles are canonicalised so
        # the sensor is always at row 0. Incidence varies along rows, so a
        # column flip leaves it physically valid, but it must flip too.
        if self.augment and torch.rand(()) < 0.5:
            img, msk, mag = img.flip(-1), msk.flip(-1), mag.flip(-1)
            if inc is not None:
                inc = inc.flip(-1)

        if self.augment:
            img = img + torch.randn(img.shape[0], 1, 1) * 0.5   # per-band gain jitter

        x = (img - self.mean) / self.std
        if inc is not None:
            m, s = self.inc_stats
            x = torch.cat([x, (inc - m) / s], dim=0)
        return x, msk, mag


def sample_stats(paths, n, rng, want_incidence: bool, want_magnitude: bool = False):
    """Class frequencies, incidence mean/std and mean layover magnitude from a RANDOM train sample."""
    idx = rng.choice(len(paths), size=min(n, len(paths)), replace=False)
    nc = len(CLASSES)
    counts = np.zeros(nc)
    n_ignored, n_total = 0, 0
    inc, mag_sum, mag_n = [], 0.0, 0
    lay = np.array(layover_ids())
    for i in idx:
        z = np.load(paths[i])
        m = z["mask"].ravel()
        keep = m != IGNORE
        n_ignored += int((~keep).sum())
        n_total += m.size
        counts += np.bincount(m[keep], minlength=nc)[:nc]
        if want_incidence:
            inc.append(z["incidence"][::8, ::8].astype(np.float32).ravel())
        if want_magnitude:
            if "magnitude" not in z.files:
                raise KeyError(f"{Path(paths[i]).name} has no 'magnitude' array; "
                               f"--magnitude needs a dataset built with it")
            sel = np.isin(m, lay)
            v = np.nan_to_num(z["magnitude"].astype(np.float32).ravel()[sel])
            mag_sum += float(v.sum())
            mag_n += int(v.size)
    if counts.sum() == 0:
        raise ValueError("every sampled pixel is ignored (255)")
    freq = counts / counts.sum()
    inc_stats = None
    if want_incidence:
        v = np.concatenate(inc)
        inc_stats = (float(v.mean()), float(v.std()))
    mag_mean = mag_sum / mag_n if mag_n else None
    return freq, inc_stats, n_ignored / max(n_total, 1), mag_mean


# ----------------------------------------------------------------------
def dice_loss(logits, target, classes=None, eps=1.0):
    """Soft Dice on the scored classes, over non-ignored pixels only."""
    classes = EVAL_CLASSES if classes is None else classes
    valid = (target != IGNORE)
    t = torch.where(valid, target, torch.zeros_like(target))
    v = valid[:, None].float()
    prob = logits.float().softmax(1) * v
    oh = F.one_hot(t, logits.shape[1]).permute(0, 3, 1, 2).float() * v
    loss = 0.0
    for c in classes:
        p, t = prob[:, c], oh[:, c]
        inter = (p * t).sum((1, 2))
        denom = p.sum((1, 2)) + t.sum((1, 2))
        loss = loss + (1 - (2 * inter + eps) / (denom + eps)).mean()
    return loss / len(classes)


def seg_loss(logits, target, ce):
    """0.5 weighted CE + 0.5 Dice. Ignored pixels (255) contribute nothing."""
    if not bool((target != IGNORE).any()):
        return logits.sum() * 0.0                 # keeps the graph, zero gradient
    return 0.5 * ce(logits, target) + 0.5 * dice_loss(logits, target)


def split_output(out, n_classes, mag_scale):
    """Network output -> (class logits, magnitude in metres or None)."""
    if out.shape[1] == n_classes:
        return out, None
    return out[:, :n_classes], F.softplus(out[:, n_classes].float()) * mag_scale


def mag_loss(pred_m, target_m, mask, mag_scale, layover_weight):
    """Smooth-L1 in units of mag_scale over labelled pixels; layover pixels weighted up."""
    valid = mask != IGNORE
    if not bool(valid.any()):
        return pred_m.sum() * 0.0
    w = torch.ones_like(pred_m)
    for c in layover_ids():
        w = torch.where(mask == c, torch.full_like(w, layover_weight), w)
    w = w * valid
    err = F.smooth_l1_loss(pred_m / mag_scale, target_m.float() / mag_scale, reduction="none")
    return (err * w).sum() / w.sum()


def total_loss(out, y, mag_t, ce, args):
    logits, mag_p = split_output(out, len(CLASSES), args.mag_scale)
    loss = seg_loss(logits, y, ce)
    if mag_p is not None:
        loss = loss + args.mag_lambda * mag_loss(mag_p, mag_t, y, args.mag_scale,
                                                 args.mag_layover_weight)
    return loss


@torch.no_grad()
def confusion(pred, target, n=None):
    n = len(CLASSES) if n is None else n
    valid = target != IGNORE
    k = (target[valid] * n + pred[valid]).flatten()
    return torch.bincount(k, minlength=n * n).reshape(n, n)     # rows = truth


def scores(cm):
    cm = cm.double()
    tp = cm.diag()
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    iou = (tp / (tp + fp + fn).clamp(min=1)).cpu().numpy()
    prec = (tp / (tp + fp).clamp(min=1)).cpu().numpy()
    rec = (tp / (tp + fn).clamp(min=1)).cpu().numpy()
    return iou, prec, rec


class MagStats:
    """Accumulates magnitude errors on true-layover pixels (and all labelled pixels)."""

    def __init__(self, constant: float | None):
        self.c = constant
        self.n = 0
        self.abs = 0.0
        self.err = 0.0
        self.abs_const = 0.0
        self.n_all = 0
        self.abs_all = 0.0
        self.bins = {b: [0.0, 0] for b in MAG_BINS}

    @torch.no_grad()
    def add(self, pred_m, target_m, mask):
        valid = mask != IGNORE
        lay = torch.zeros_like(valid)
        for c in layover_ids():
            lay |= mask == c
        e = pred_m.float() - target_m.float()
        self.n_all += int(valid.sum())
        self.abs_all += float(e[valid].abs().sum())
        if bool(lay.any()):
            el, tl = e[lay], target_m.float()[lay]
            self.n += int(lay.sum())
            self.abs += float(el.abs().sum())
            self.err += float(el.sum())
            if self.c is not None:
                self.abs_const += float((tl - self.c).abs().sum())
            for lo, hi in MAG_BINS:
                s = (tl >= lo) & (tl < hi)
                self.bins[(lo, hi)][0] += float(el[s].abs().sum())
                self.bins[(lo, hi)][1] += int(s.sum())

    def summary(self):
        if self.n == 0:
            return {"mae_layover_m": None, "bias_layover_m": None, "mae_all_m": None,
                    "mae_constant_m": None, "skill": None, "n_layover_px": 0, "mae_by_bin_m": {}}
        mae = self.abs / self.n
        mae_c = self.abs_const / self.n if self.c is not None else None
        return {"mae_layover_m": mae, "bias_layover_m": self.err / self.n,
                "mae_all_m": self.abs_all / max(self.n_all, 1),
                "mae_constant_m": mae_c,
                "skill": (1.0 - mae / mae_c) if mae_c else None,
                "n_layover_px": self.n,
                "mae_by_bin_m": {f"{lo:g}-{hi:g}": (s / k if k else None)
                                 for (lo, hi), (s, k) in self.bins.items()}}


@torch.no_grad()
def evaluate(model, loader, device, args, constant=None):
    model.eval()
    nc = len(CLASSES)
    cm = torch.zeros(nc, nc, dtype=torch.long, device=device)
    ms = MagStats(constant)
    for x, y, mag_t in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            out = model(x)
        logits, mag_p = split_output(out, nc, args.mag_scale)
        cm += confusion(logits.argmax(1), y)
        if mag_p is not None:
            ms.add(mag_p, mag_t.to(device, non_blocking=True), y)
    return cm, (ms.summary() if args.magnitude else None)


def selection_score(miou, mag, select):
    """What 'best' means: mIoU(rare), magnitude skill, or their average."""
    skill = mag["skill"] if mag and mag.get("skill") is not None else None
    if select == "miou" or skill is None:
        return miou
    if select == "mag":
        return skill
    return 0.5 * miou + 0.5 * skill


# ----------------------------------------------------------------------
def fit_brightness_baseline(paths, rng, n_tiles=150, n_px=500_000):
    """HH > t_hi -> layover (active in v2), HH < t_lo -> shadow. Thresholds maximise train IoU."""
    idx = rng.choice(len(paths), size=min(n_tiles, len(paths)), replace=False)
    hh, m = [], []
    for i in idx:
        z = np.load(paths[i])
        hh.append(z["image"][0].astype(np.float32).ravel())
        m.append(z["mask"].ravel())
    hh, m = np.concatenate(hh), np.concatenate(m)
    keep = m != IGNORE
    hh, m = hh[keep], m[keep]
    sel = rng.choice(hh.size, size=min(n_px, hh.size), replace=False)
    hh, m = hh[sel], m[sel]

    def best(cls, cands, above):
        g = m == cls
        out = (None, -1.0)
        for t in cands:
            p = hh > t if above else hh < t
            u = (p | g).sum()
            iou = (p & g).sum() / u if u else 0.0
            if iou > out[1]:
                out = (float(t), float(iou))
        return out[0]

    bright, dark = bright_dark_ids()
    t_hi = best(bright, np.percentile(hh, np.arange(50, 99.5, 0.5)), above=True)
    t_lo = best(dark, np.percentile(hh, np.arange(0.5, 50, 0.5)), above=False)
    return t_hi, t_lo


def eval_brightness_baseline(paths, t_hi, t_lo):
    nc = len(CLASSES)
    bright, dark = bright_dark_ids()
    cm = np.zeros((nc, nc), np.int64)
    for p in paths:
        z = np.load(p)
        hh = z["image"][0].astype(np.float32)
        y = z["mask"].astype(np.int64)
        pred = np.zeros_like(y)
        pred[hh > t_hi] = bright
        pred[hh < t_lo] = dark
        valid = y != IGNORE
        cm += np.bincount((y * nc + pred)[valid], minlength=nc * nc).reshape(nc, nc)
    return torch.from_numpy(cm)


# ----------------------------------------------------------------------
def save_overlays(model, loader, device, out: Path, tag: str, args, n=10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap([COLOURS.get(c, (0.6, 0.2, 0.6, 0.6)) for c in CLASSES])
    vmax = len(CLASSES) - 1
    out.mkdir(parents=True, exist_ok=True)
    model.eval()
    done = 0
    with torch.no_grad():
        for x, y, mag_t in loader:
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                o = model(x.to(device))
            logits, mag_p = split_output(o, len(CLASSES), args.mag_scale)
            p = logits.argmax(1).cpu()
            mag_p = mag_p.float().cpu() if mag_p is not None else None
            for i in range(x.shape[0]):
                if done >= n:
                    return
                hh = x[i, 0].float().numpy()
                lo, hi = np.percentile(hh, [2, 98])
                cols = 5 if mag_p is not None else 3
                fig, ax = plt.subplots(1, cols, figsize=(5 * cols, 5.2))
                ax[0].imshow(hh, cmap="gray", vmin=lo, vmax=hi)
                ax[0].set_title("HH (sensor at top)")
                lab = np.ma.masked_equal(y[i].numpy(), IGNORE)
                lab_title = "label (ignored pixels left clear)" if np.ma.is_masked(lab) else "label"
                for a, m, t in ((ax[1], lab, lab_title), (ax[2], p[i].numpy(), "prediction")):
                    a.imshow(hh, cmap="gray", vmin=lo, vmax=hi)
                    a.imshow(m, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
                    a.set_title(t)
                if mag_p is not None:
                    ign = y[i].numpy() == IGNORE
                    top = max(10.0, float(np.percentile(mag_t[i].numpy()[~ign], 99.5))
                              if (~ign).any() else 10.0)
                    for a, m, t in ((ax[3], mag_t[i].numpy(), "true magnitude (m)"),
                                    (ax[4], mag_p[i].numpy(), "predicted magnitude (m)")):
                        a.imshow(hh, cmap="gray", vmin=lo, vmax=hi)
                        im = a.imshow(np.ma.masked_where(ign | (m < 1.0), m), cmap="magma",
                                      vmin=0, vmax=top, interpolation="nearest")
                        a.set_title(t)
                    fig.colorbar(im, ax=ax[4], fraction=0.046, pad=0.04)
                for a in ax:
                    a.axis("off")
                fig.tight_layout()
                fig.savefig(out / f"{tag}_{done:02d}.png", dpi=85)
                plt.close(fig)
                done += 1


def print_table(title, rows):
    print(f"\n{title}")
    print(f"{'':>22}" + "".join(f"{c:>10}" for c in CLASSES) + f"{'mIoU rare':>11}")
    for name, (iou, prec, rec) in rows:
        miou = float(np.mean([iou[c] for c in EVAL_CLASSES]))
        print(f"{name + ' IoU':>22}" + "".join(f"{v:>10.3f}" for v in iou) + f"{miou:>11.3f}")
        print(f"{'  precision':>22}" + "".join(f"{v:>10.3f}" for v in prec))
        print(f"{'  recall':>22}" + "".join(f"{v:>10.3f}" for v in rec))


def print_mag(title, mag):
    if not mag or mag["mae_layover_m"] is None:
        print(f"\n{title}: no layover pixels")
        return
    c = mag["mae_constant_m"]
    print(f"\n{title} magnitude, on true-layover pixels ({mag['n_layover_px']:,} px):")
    print(f"  U-Net MAE {mag['mae_layover_m']:.2f} m   bias {mag['bias_layover_m']:+.2f} m"
          + (f"   constant-baseline MAE {c:.2f} m   skill {mag['skill']:+.3f}" if c else ""))
    print("  MAE by true magnitude: " + "   ".join(
        f"{k} m: {v:.2f}" if v is not None else f"{k} m: -" for k, v in mag["mae_by_bin_m"].items()))


def init_from(model, path, device):
    """Copy every tensor of an earlier checkpoint that fits; keep the old class rows of the head."""
    st = torch.load(path, map_location=device)
    old = st.get("model", st)
    old_classes = list(st.get("classes", V2_CLASSES))
    new = model.state_dict()
    copied, partial, skipped = 0, [], []
    for k, v in new.items():
        if k not in old:
            skipped.append(k)
            continue
        o = old[k]
        if k.startswith("head."):
            # the head's rows are classes: copy them only when the class list
            # is the same (a 4-class v2 head has the same shape as a 3-class +
            # magnitude head, but its rows mean different things)
            if old_classes == CLASSES and o.shape[1:] == v.shape[1:]:
                n = min(o.shape[0], len(CLASSES))
                new[k][:n] = o[:n]
                partial.append(k)
            else:
                skipped.append(k)
        elif o.shape == v.shape:
            new[k] = o.clone()
            copied += 1
        else:
            skipped.append(k)
    body = [k for k in new if not k.startswith("head.")]
    if copied < len(body) // 2:
        raise ValueError(f"--init {path}: only {copied} of {len(body)} tensors fit. Use the "
                         f"checkpoint's --base ({st.get('base', '?')}) and the same "
                         f"--use-incidence setting ({st.get('use_incidence', False)}).")
    model.load_state_dict(new)
    head = ("class rows copied" if partial else
            f"new (checkpoint classes {old_classes} differ from {CLASSES})")
    print(f"init from {path}: {copied} of {len(body)} body tensors copied; head {head}"
          + (f"; left new: {[k for k in skipped if not k.startswith('head.')]}"
             if any(not k.startswith("head.") for k in skipped) else ""))


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("side-look-tiles-v2"))
    ap.add_argument("--out", type=Path, default=Path("runs/v2_base"))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--class-weights", choices=["inverse", "sqrt", "none"], default="sqrt",
                    help="CE weighting. Dice already targets the rare classes, so full "
                         "inverse frequency double-corrects and over-predicts them.")
    ap.add_argument("--use-incidence", action="store_true",
                    help="add per-pixel incidence as a 5th input channel")
    ap.add_argument("--magnitude", action="store_true",
                    help="also predict layover magnitude (metres) from the tiles' 'magnitude' array")
    ap.add_argument("--mag-scale", type=float, default=10.0,
                    help="metres per unit of the magnitude output and of its loss")
    ap.add_argument("--mag-lambda", type=float, default=1.0, help="weight of the magnitude loss")
    ap.add_argument("--mag-layover-weight", type=float, default=10.0,
                    help="how much more a layover pixel counts in the magnitude loss")
    ap.add_argument("--select", choices=["auto", "miou", "mag", "both"], default="auto",
                    help="best checkpoint by mIoU(rare), magnitude skill, or their average "
                         "(auto: average with --magnitude, else mIoU)")
    ap.add_argument("--init", type=Path, default=None,
                    help="start from an earlier checkpoint (fine-tuning); tensors that fit are copied")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overfit", action="store_true")
    ap.add_argument("--eval", type=Path, default=None, help="evaluate a checkpoint only")
    ap.add_argument("--resume", action="store_true",
                    help="continue from <out>/last.pt (written every epoch)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    global CLASSES, EVAL_CLASSES
    CLASSES = load_classes(args.data)
    EVAL_CLASSES = list(range(1, len(CLASSES)))
    print(f"classes: {CLASSES}  (scored: {[CLASSES[c] for c in EVAL_CLASSES]})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    stats = json.loads((args.data / "norm_stats.json").read_text())
    mean = torch.tensor(stats["mean"], dtype=torch.float32)
    std = torch.tensor(stats["std"], dtype=torch.float32)

    # when evaluating, take the channel and head config from the checkpoint
    ckpt = None
    if args.eval:
        ckpt = torch.load(args.eval, map_location="cpu")
        args.use_incidence = bool(ckpt.get("use_incidence", False))
        args.magnitude = bool(ckpt.get("magnitude", False))
        args.mag_scale = float(ckpt.get("mag_scale", args.mag_scale))
        args.base = int(ckpt.get("base", args.base))
        ck_classes = ckpt.get("classes", V2_CLASSES)
        if list(ck_classes) != CLASSES:
            raise ValueError(f"checkpoint was trained on classes {ck_classes} "
                             f"but {args.data} has {CLASSES}")
    select = args.select if args.select != "auto" else ("both" if args.magnitude else "miou")

    train_paths = sorted((args.data / "train").glob("*.npz"))
    freq, inc_stats, ignored, mag_const = sample_stats(train_paths, 400, rng, args.use_incidence,
                                                       args.magnitude)
    if ckpt is not None and ckpt.get("inc_stats"):
        inc_stats = tuple(ckpt["inc_stats"])
    if ckpt is not None and ckpt.get("mag_constant") is not None:
        mag_const = float(ckpt["mag_constant"])

    train_ds = Tiles(args.data, "train", mean, std, True, inc_stats, args.magnitude)
    val_ds = Tiles(args.data, "val", mean, std, False, inc_stats, args.magnitude)
    test_ds = Tiles(args.data, "test", mean, std, False, inc_stats, args.magnitude)
    print(f"train {len(train_ds)}   val {len(val_ds)}   test {len(test_ds)}")

    dl = dict(num_workers=args.workers, pin_memory=device.type == "cuda",
              persistent_workers=args.workers > 0)
    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True, **dl)
    val_ld = DataLoader(val_ds, batch_size=args.batch, shuffle=False, **dl)
    test_ld = DataLoader(test_ds, batch_size=args.batch, shuffle=False, **dl)

    cin = 5 if args.use_incidence else 4
    n_out = len(CLASSES) + (1 if args.magnitude else 0)
    model = UNet(cin=cin, classes=n_out, base=args.base).to(device)
    print(f"input channels {cin}   outputs {n_out}"
          + (" (classes + magnitude)" if args.magnitude else "")
          + f"   params {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    if args.init is not None and ckpt is None and not args.resume:
        init_from(model, args.init, device)

    f = np.clip(freq, 1e-6, None)
    w = {"inverse": 1.0 / f, "sqrt": 1.0 / np.sqrt(f), "none": np.ones_like(f)}[args.class_weights]
    w = w / w.mean()
    print("class freq (random train sample)", np.round(freq, 4).tolist(),
          f"  ignored (255): {ignored:.1%} of pixels")
    print(f"ce weights ({args.class_weights:>7})          ", np.round(w, 2).tolist())
    if inc_stats:
        print(f"incidence mean {inc_stats[0]:.2f} deg  std {inc_stats[1]:.2f}")
    if args.magnitude:
        print(f"magnitude: mean on train layover pixels {mag_const:.2f} m (the constant baseline); "
              f"scale {args.mag_scale:g} m, loss weight {args.mag_lambda:g}, "
              f"layover weight {args.mag_layover_weight:g}; best checkpoint by {select}")
    ce = torch.nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device),
                                   ignore_index=IGNORE)

    # ---- overfit one batch --------------------------------------------
    if args.overfit:
        # take the first batch (of up to 50) that contains every class, so
        # every IoU printed below means something
        it = iter(train_ld)
        for _ in range(50):
            try:
                x, y, mag_t = next(it)
            except StopIteration:
                break
            if all(bool((y == c).any()) for c in range(len(CLASSES))):
                break
        counts = {c: int((y == i).sum()) for i, c in enumerate(CLASSES)}
        print("\noverfit batch pixels: "
              + ", ".join(f"{c} {n:,}" for c, n in counts.items())
              + f", ignored {int((y == IGNORE).sum()):,}")
        x, y, mag_t = x.to(device), y.to(device), mag_t.to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        print("overfitting one batch - loss must collapse toward zero")
        print("(n/a = class absent from this batch and never predicted)\n")
        for step in range(200):
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = total_loss(out, y, mag_t, ce, args)
            loss.backward()
            opt.step()
            if step % 20 == 0 or step == 199:
                logits, mag_p = split_output(out, len(CLASSES), args.mag_scale)
                cm = confusion(logits.argmax(1), y)
                iou, _, _ = scores(cm)
                seen = ((cm.sum(0) + cm.sum(1)) > 0).tolist()
                extra = ""
                if mag_p is not None:
                    ms = MagStats(mag_const)
                    ms.add(mag_p.detach(), mag_t, y)
                    s = ms.summary()
                    if s["mae_layover_m"] is not None:
                        extra = f"   magnitude MAE {s['mae_layover_m']:.2f} m"
                print(f"  step {step:3d}  loss {loss.item():.4f}   "
                      + " ".join(f"{c}={v:.2f}" if seen[i] else f"{c}=n/a"
                                 for i, (c, v) in enumerate(zip(CLASSES, iou))) + extra)
        return

    # ---- train -------------------------------------------------------
    if ckpt is None:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        scaler = make_scaler(device.type == "cuda")

        best, best_epoch, start_epoch = -1e9, -1, 0
        last_path = args.out / "last.pt"
        if args.resume:
            if not last_path.exists():
                raise FileNotFoundError(f"--resume given but {last_path} does not exist")
            st = torch.load(last_path, map_location=device)
            if list(st.get("classes", V2_CLASSES)) != CLASSES:
                raise ValueError(f"{last_path} was trained on {st.get('classes', V2_CLASSES)}, "
                                 f"dataset has {CLASSES}; use a fresh --out")
            if bool(st.get("magnitude", False)) != args.magnitude:
                raise ValueError(f"{last_path} magnitude={st.get('magnitude', False)} but this run "
                                 f"has --magnitude={args.magnitude}; use a fresh --out")
            model.load_state_dict(st["model"])
            opt.load_state_dict(st["opt"])
            sched.load_state_dict(st["sched"])
            scaler.load_state_dict(st["scaler"])
            start_epoch = st["epoch"] + 1
            best, best_epoch = st["best"], st["best_epoch"]
            print(f"resumed from epoch {st['epoch']}  (best score {best:.3f} at epoch {best_epoch})")

        new_log = not (args.resume and (args.out / "log.csv").exists())
        log = (args.out / "log.csv").open("w" if new_log else "a", newline="")
        writer = csv.writer(log)
        if new_log:
            writer.writerow(["epoch", "train_loss", "lr", *[f"iou_{c}" for c in CLASSES],
                             "miou_rare", "mag_mae_layover_m", "mag_skill", "score", "seconds"])

        common = {"use_incidence": args.use_incidence,
                  "inc_stats": list(inc_stats) if inc_stats else None,
                  "base": args.base, "classes": CLASSES, "magnitude": args.magnitude,
                  "mag_scale": args.mag_scale, "mag_constant": mag_const}
        for epoch in range(start_epoch, args.epochs):
            model.train()
            t0, total, n = time.time(), 0.0, 0
            for x, y, mag_t in train_ld:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                mag_t = mag_t.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", enabled=device.type == "cuda"):
                    out = model(x)
                    loss = total_loss(out, y, mag_t, ce, args)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                total += loss.item() * x.shape[0]
                n += x.shape[0]
            sched.step()

            cm, mag = evaluate(model, val_ld, device, args, mag_const)
            iou, _, _ = scores(cm)
            miou = float(np.mean([iou[c] for c in EVAL_CLASSES]))
            score = selection_score(miou, mag, select)
            secs = time.time() - t0
            mag_txt = ""
            if mag and mag["mae_layover_m"] is not None:
                mag_txt = (f"  mag MAE {mag['mae_layover_m']:.2f} m"
                           + (f" skill {mag['skill']:+.3f}" if mag["skill"] is not None else ""))
            print(f"epoch {epoch:3d}  loss {total/max(n,1):.4f}  "
                  + "  ".join(f"{c}={iou[i]:.3f}" for i, c in enumerate(CLASSES))
                  + f"  mIoU(rare) {miou:.3f}{mag_txt}  [{secs:.0f}s]")
            writer.writerow([epoch, f"{total/max(n,1):.5f}", f"{sched.get_last_lr()[0]:.2e}",
                             *[f"{v:.4f}" for v in iou], f"{miou:.4f}",
                             "" if not mag or mag["mae_layover_m"] is None else f"{mag['mae_layover_m']:.3f}",
                             "" if not mag or mag["skill"] is None else f"{mag['skill']:.4f}",
                             f"{score:.4f}", f"{secs:.0f}"])
            log.flush()

            if score > best:
                best, best_epoch = score, epoch
                torch.save({"model": model.state_dict(), "epoch": epoch, "miou": miou,
                            "score": score, "select": select, **common,
                            "args": {k: str(v) for k, v in vars(args).items()}},
                           args.out / "best.pt")
            elif epoch - best_epoch >= args.patience:
                print(f"\nno val improvement for {args.patience} epochs - stopping")
                break

            # full training state every epoch, written atomically so an
            # interruption mid-save cannot corrupt the previous copy
            tmp = last_path.with_suffix(".tmp")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                        "epoch": epoch, "best": best, "best_epoch": best_epoch, **common}, tmp)
            tmp.replace(last_path)
        log.close()
        print(f"\nbest val score ({select}) {best:.3f} at epoch {best_epoch}")
        ckpt = torch.load(args.out / "best.pt", map_location="cpu")

    # ---- final evaluation ---------------------------------------------
    model = UNet(cin=cin, classes=n_out, base=int(ckpt.get("base", args.base))).to(device)
    model.load_state_dict(ckpt["model"])

    cm_val, mag_val = evaluate(model, val_ld, device, args, mag_const)
    cm_test, mag_test = evaluate(model, test_ld, device, args, mag_const)
    val_s, test_s = scores(cm_val), scores(cm_test)

    t_hi, t_lo = fit_brightness_baseline(train_paths, rng)
    base_val = scores(eval_brightness_baseline(val_ds.paths, t_hi, t_lo))
    base_test = scores(eval_brightness_baseline(test_ds.paths, t_hi, t_lo))

    b_id, d_id = bright_dark_ids()
    print(f"\nbrightness baseline: HH > {t_hi:.2f} dB -> {CLASSES[b_id]}, "
          f"HH < {t_lo:.2f} dB -> {CLASSES[d_id]}")
    print_table("VALIDATION", [("U-Net", val_s), ("baseline", base_val)])
    print_table("TEST", [("U-Net", test_s), ("baseline", base_test)])
    if args.magnitude:
        print_mag("VALIDATION", mag_val)
        print_mag("TEST", mag_test)

    def pack(s):
        iou, prec, rec = s
        return {"iou": dict(zip(CLASSES, map(float, iou))),
                "precision": dict(zip(CLASSES, map(float, prec))),
                "recall": dict(zip(CLASSES, map(float, rec))),
                "miou_rare": float(np.mean([iou[c] for c in EVAL_CLASSES])),
                **({"miou_123": float(np.mean([iou[c] for c in EVAL_CLASSES]))}
                   if len(CLASSES) == 4 else {})}

    metrics = {
        "checkpoint_epoch": int(ckpt["epoch"]),
        "classes": CLASSES,
        "use_incidence": args.use_incidence,
        "val": pack(val_s), "test": pack(test_s),
        "baseline": {"t_hi_db": t_hi, "t_lo_db": t_lo,
                     "val": pack(base_val), "test": pack(base_test)},
        "tiles": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
    }
    if args.magnitude:
        metrics["magnitude"] = {"scale_m": args.mag_scale, "constant_baseline_m": mag_const,
                                "val": mag_val, "test": mag_test}
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    save_overlays(model, val_ld, device, args.out / "overlays", "val", args)
    save_overlays(model, test_ld, device, args.out / "overlays", "test", args)
    print(f"\nwrote {args.out/'metrics.json'} and overlays in {args.out/'overlays'}")
    print("IoU measures agreement with the labels. Pixels marked 255 (ignore) are left out"
          "\nof training and of every score above, including the baselines.")


if __name__ == "__main__":
    main()
