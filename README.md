# Side Look

Layover and shadow masks for Capella X-band MAG-POL over Rotterdam (SpaceNet 6), plus a tiled semantic-segmentation dataset.

Masks come from an AHN lidar DSM and two geometry parameters only. SAR brightness never enters the labels.

- Incidence: `collect.image.center_pixel.incidence_angle` (from vertical, not grazing)
- Look: SpaceNet orientation `0` → 0° north, `1` → 180° south
- Classes: `0` nominal, `1` active layover, `2` passive layover, `3` shadow (shadow wins)
- South-facing tiles are row-flipped so layover sits on the same side as north-facing

## Clone (code)

```bash
git clone https://github.com/RBirmiwal/side-look.git
cd side-look
python3 -m venv .venv && source .venv/bin/activate
pip install -r sar/requirements.txt awscli
```

macOS: `brew install gdal` first. Ubuntu: `sudo apt install gdal-bin libgdal-dev python3-venv`.

See [DATASET.md](DATASET.md) and [GETTING_STARTED.md](GETTING_STARTED.md).

## Tiles (private Hugging Face)

`.npz` tiles are **not** in git. They live in a private dataset:

```bash
huggingface-cli login
huggingface-cli download RBirmiwal/side-look-tiles --repo-type dataset --local-dir side-look-tiles
cd side-look-tiles
sha256sum -c SHA256SUMS
```

Python:

```python
from huggingface_hub import snapshot_download
snapshot_download("RBirmiwal/side-look-tiles", repo_type="dataset", local_dir="side-look-tiles")
```

This run: 409 tiles (383 train / 21 val / 5 test), ~1.5 GB. Keys: `image` (4,512,512) float32 HH/HV/VH/VV in dB, `mask` uint8, `look` original orientation, `meta` JSON. Apply `norm_stats.json` in the training loop.

## Rebuild (do not pull all 202 MAG-POL strips)

Each source GeoTIFF is ~3 GB. `--release-sar` deletes a strip after tiling.

```bash
cd sar
python download.py --ahn-aoi
python dataset.py --max-strips 5 --fetch-sar --release-sar --disk-budget-gb 12
```
