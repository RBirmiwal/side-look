# Side Look tiles (Rotterdam MAG-POL)

Private tiled training set: 512×512 MAG-POL patches with DSM fold-test layover/shadow masks.

| split | tiles |
|-------|------:|
| train | 383 |
| val | 21 |
| test | 5 |
| **total** | **409** |

Two SpaceNet 6 Capella MAG-POL strips (north + south on the same ground). Labels never use SAR brightness.

## Files

- `train/*.npz` `val/*.npz` `test/*.npz`
- `manifest.csv` — bbox, split, θ, look, class fractions
- `norm_stats.json` — train-only mean/std of HH/HV/VH/VV (dB)
- `SHA256SUMS` — sha256 of every `.npz`

## Load a tile

```python
import numpy as np, json
z = np.load("train/<tile_id>.npz")
image = z["image"]          # float32 (4, 512, 512)  HH, HV, VH, VV in dB
mask  = z["mask"]           # uint8  (512, 512)      0 nom / 1 active / 2 passive / 3 shadow
look  = z["look"]           # uint8  (512, 512)      original orientation flag
meta  = json.loads(str(z["meta"]))
```

Apply `norm_stats.json` in the training loop. Do not z-score into the files.

## Download

```bash
huggingface-cli login
huggingface-cli download RBirmiwal/side-look-tiles --repo-type dataset --local-dir side-look-tiles
cd side-look-tiles
sha256sum -c SHA256SUMS
```

Pipeline and catalog: https://github.com/RBirmiwal/side-look
