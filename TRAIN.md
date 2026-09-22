# Side Look — codebase and U-Net training

How the repo is laid out, what a tile contains, and how to train a 4-class U-Net on it.

The pipeline does **not** train anything. It writes tiles. This note is the training side of [`DATASET.md`](DATASET.md).

Current data (two strips, north + south on the same ground):

| split | tiles | stride |
|-------|------:|--------|
| train | 383 | 256 px (50% overlap) |
| val | 21 | 512 px (no overlap) |
| test | 5 | 512 px (no overlap) |

409 tiles, ~1.5 GB. Val and test are small until more strips are tiled further east. Do not tune on test.

---

## Codebase

```
side-look/
  DATASET.md                 what the tiles are and why
  GETTING_STARTED.md         laptop setup, 5-strip disk budget
  TRAIN.md                   this file
  sar/
    catalog.py               202 MAG-POL strips → output/strip_catalog.json
    incidence.py             Capella incidence from vertical (not 90−θ)
    download.py              public S3 MAG-POL + PDOK AHN DSM
    dsm_geometry.py          DSM fold-test → classes 0–3
    splits.py                1 km blocks, west→east 70/15/15
    dataset.py               tile, write .npz, REPORT.md, overlays
    pipeline.py / dsm_pipeline.py
                             one-strip inspector (not the training set)
    tests/                   closed-form layover / shadow / incidence / splits
    output/
      strip_catalog.json     every strip: bounds, θ, look
      dataset/
        train|val|test/*.npz
        manifest.csv
        norm_stats.json      train-only mean/std, do not bake into files
        REPORT.md
        SHA256SUMS
        overlays/            HH SAR and mask PNGs (not training inputs)
```

Data flow:

```
SpaceNet MAG-POL (S3)          AHN DSM (PDOK WCS)
        │                            │
        │                     dsm_geometry.mask_from_dsm
        │                     θ + look only — SAR brightness never enters
        │                            │
        └──────── dataset.py ────────┘
                     │
        geographic split (splits.py) + canonical flip
                     │
              .npz  image (4,512,512) dB
                    mask  (512,512)   uint8
                    look  (512,512)   uint8
                    meta  JSON
```

| module | you touch it when |
|--------|-------------------|
| `incidence.py` | the Capella field or the 90−θ bug |
| `dsm_geometry.py` | the fold-test (layover / shadow definition) |
| `splits.py` | the 70/15/15 cut or block size |
| `dataset.py` | tile size, stride, drops, `.npz` keys |
| `download.py` | S3 / AHN. `SIDE_LOOK_INSECURE_SSL=1` if a corp proxy breaks TLS |

Labels are geometry. A U-Net that memorizes brightness without matching the copper layover overlay has not learned the task this set defines.

---

## What one tile is

```python
import json
import numpy as np

z = np.load("train/<tile>.npz")
image = z["image"]                 # float32 (4, 512, 512)  HH, HV, VH, VV in dB
mask  = z["mask"]                  # uint8   (512, 512)     0 nom / 1 active / 2 passive / 3 shadow
look  = z["look"]                  # uint8   (512, 512)     original orientation flag, constant
meta  = json.loads(str(z["meta"]))
```

| class | name | meaning |
|------:|------|---------|
| 0 | nominal | neither layover nor shadow |
| 1 | active layover | roof / face that folded toward the sensor |
| 2 | passive layover | ground that roof landed on |
| 3 | shadow | no return; wins if both would apply |

South-facing strips are already row-flipped so layover sits on the same side of the tile as north-facing. `look` still says which way the satellite was facing (0 north, 1 south).

Normalise in the loader, from **train** stats only (`sar/output/dataset/norm_stats.json`):

| band | mean (dB) | std |
|------|----------:|----:|
| HH | −11.202 | 5.091 |
| HV | −14.126 | 2.909 |
| VH | −13.484 | 3.356 |
| VV | −10.771 | 5.099 |

```text
x = (image - mean[:, None, None]) / std[:, None, None]
```

Do not recompute mean/std on val or test. Do not `log10` MAG-POL; it is already dB.

Train pixels on this run: nominal 0.695, active 0.150, passive 0.087, shadow 0.069. Overall accuracy is a bad metric — predicting all nominal scores ~70%.

---

## U-Net

A standard encoder–decoder is the right first model.

| | |
|--|--|
| input | 4 channels, 512×512, z-scored dB |
| output | 4 logits, same spatial size |
| depth | 4 downs is enough at 512 (32 → 512 channels, or start at 16 if VRAM is tight) |
| norm | GroupNorm or BatchNorm; GroupNorm is safer at batch 2–4 |
| activation | ReLU |
| upsample | bilinear + conv, or transposed conv |
| optional 5th input | `look` as float 0/1, broadcast. Skip it for the first run; canonicalise already aligned the geometry |

Loss: weighted cross-entropy, weights inverse to **train** frequency, plus Dice on classes 1–3 so the rare classes are not ignored.

```text
freq   = [0.695, 0.150, 0.087, 0.069]
weight = (1 / freq) / mean(1 / freq)     # ~ [0.19, 0.90, 1.55, 1.95] before you rescale
```

A simple mix that works: `0.5 * CE(weight) + 0.5 * (1 - mean Dice over classes 1,2,3)`.

Do not weight by the val/test fractions. Those splits are layover-heavier than train (any-layover 0.24 train, 0.34 val, 0.41 test) because this two-strip cut is west-heavy. That is a split artefact, not a reason to rebalance the loss on val.

Augmentation that does **not** break the label:

- flips **left–right** (along-track). Layover stays on the same range side.
- small gain/noise on the dB image (±1 dB, speckle-like). Do not touch the mask.
- **no** vertical flip, **no** 90° rotation. That moves layover to the wrong side of the building. Canonicalise exists so you do not need that rotation.

Train tiles overlap 50%. Neighbouring tiles are not independent. Shuffle them; do not report train IoU as a result. Val and test do not overlap.

Optimiser: AdamW, lr `1e-3` or `3e-4`, weight decay `1e-4`. Cosine decay over ~40–80 epochs. Batch 4 at 512×512×4 fits a 16 GB GPU in fp16; batch 2 if it does not. Early-stop on **mean IoU of classes 1, 2, 3** on val, not on pixel accuracy.

With 383 train tiles this will overfit. That is expected. The useful check is whether val IoU on active layover and shadow is above a constant-class baseline, and whether the prediction points the same way as the copper overlay. More strips (`--max-strips 5`, then more) matter more than a fancier decoder.

---

## Minimal loader and step

PyTorch is not a pipeline dependency. Install it only in the training env.

```python
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path("sar/output/dataset")
STATS = json.loads((ROOT / "norm_stats.json").read_text())
MEAN = torch.tensor(STATS["mean"], dtype=torch.float32)[:, None, None]
STD = torch.tensor(STATS["std"], dtype=torch.float32)[:, None, None]


class Tiles(Dataset):
    def __init__(self, split: str):
        self.paths = sorted((ROOT / split).glob("*.npz"))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        z = np.load(self.paths[i])
        image = torch.from_numpy(z["image"].astype("float32"))  # (4, 512, 512)
        mask = torch.from_numpy(z["mask"].astype("int64"))      # (512, 512)
        if split_is_train(self.paths[i]):
            if torch.rand(1).item() < 0.5:
                image = image.flip(-1)
                mask = mask.flip(-1)
        return (image - MEAN) / STD, mask


def split_is_train(path: Path) -> bool:
    return path.parent.name == "train"
```

U-Net sketch (4 levels). Swap in `segmentation_models_pytorch.Unet(in_channels=4, classes=4)` if you want a stock one.

```python
import torch.nn as nn


class Conv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.GroupNorm(8, cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.GroupNorm(8, cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, cin=4, classes=4, base=32):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]
        self.down = nn.ModuleList([Conv(cin if i == 0 else c[i - 1], c[i]) for i in range(4)])
        self.pool = nn.MaxPool2d(2)
        self.mid = Conv(c[3], c[4])
        self.up = nn.ModuleList([Conv(c[i] + c[i + 1], c[i]) for i in range(3, -1, -1)])
        self.head = nn.Conv2d(c[0], classes, 1)

    def forward(self, x):
        skips = []
        for conv in self.down:
            x = conv(x)
            skips.append(x)
            x = self.pool(x)
        x = self.mid(x)
        for conv, skip in zip(self.up, reversed(skips)):
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = conv(torch.cat([x, skip], dim=1))
        return self.head(x)
```

```python
FREQ = torch.tensor([0.6950, 0.1499, 0.0865, 0.0685])
WEIGHT = (1.0 / FREQ)
WEIGHT = WEIGHT / WEIGHT.mean()

model = UNet().cuda()
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
ce = torch.nn.CrossEntropyLoss(weight=WEIGHT.cuda())

for epoch in range(60):
    model.train()
    for x, y in train_loader:          # x (N,4,512,512), y (N,512,512)
        x, y = x.cuda(), y.cuda()
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = ce(logits, y)
        loss.backward()
        opt.step()
```

Score val with per-class IoU. Ignore class 0 when you decide whether it learned layover:

```python
@torch.no_grad()
def iou(logits, target, classes=4):
    pred = logits.argmax(1)
    out = []
    for c in range(classes):
        tp = ((pred == c) & (target == c)).sum()
        den = ((pred == c) | (target == c)).sum()
        out.append((tp / den.clamp(min=1)).item())
    return out  # [nom, active, passive, shadow]
```

---

## What to trust

- Train and test boxes do not intersect (asserted in `dataset.py`). Same building can still appear in two **train** tiles because of the 256 px stride. That is overlap, not leakage.
- A strip-level split would leak: the same roof is in many collects. The 1 km west→east blocks are there to stop that. Do not reshuffle tiles at random into test.
- 5 test tiles cannot support a paper number. Tile more strips before you quote test IoU.
- If the prediction’s layover points the opposite way from `overlays/*` mask PNGs, look at canonicalise and incidence before you change the U-Net. `90 − θ` puts layover on the wrong side of every building.

Rebuild (PNNL TLS intercept):

```bash
cd sar
SIDE_LOOK_INSECURE_SSL=1 python download.py          # one strip + AHN over it
python dataset.py --strips \
  20190804111224_20190804111453 \
  20190822074237_20190822074521 \
  --fetch-sar --release-sar --disk-budget-gb 12
```

`--max-strips 5` is the next step when val/test are too small to stop on.
