# Side Look

Layover and shadow masks for Capella X-band MAG-POL over Rotterdam (SpaceNet 6), plus a tiled semantic-segmentation dataset.

Masks come from an AHN lidar DSM and two geometry parameters only. SAR brightness never enters the labels.

- Incidence: `collect.image.center_pixel.incidence_angle` (from vertical, not grazing)
- Look: SpaceNet orientation `0` → 0° north, `1` → 180° south
- Classes: `0` nominal, `1` active layover, `2` passive layover, `3` shadow (shadow wins)
- South-facing tiles are row-flipped so layover sits on the same side as north-facing

## Clone

```bash
git clone https://github.com/RBirmiwal/side-look.git
cd side-look
git pull
python3 -m venv .venv && source .venv/bin/activate
pip install -r sar/requirements.txt awscli
```

macOS: `brew install gdal` first. Ubuntu: `sudo apt install gdal-bin libgdal-dev python3-venv`.

## Start with 5 strips (do not pull all 202)

Each MAG-POL GeoTIFF is ~3 GB. 202 of them is ~600 GB. `--release-sar` deletes a strip after tiling, so **peak disk is one 3 GB file + the tiles**.

```bash
cd sar
python download.py --ahn-aoi    # city DSM, a few hundred MB
python dataset.py --max-strips 5 --fetch-sar --release-sar --disk-budget-gb 12
```

`--max-strips 5` walks the catalog west→east and keeps five. To force a north+south pair on the same ground (canonicalise check) plus three neighbours:

```bash
python dataset.py --strips \
  20190804111224_20190804111453 \
  20190822074237_20190822074521 \
  20190804111851_20190804112030 \
  20190804113009_20190804113242 \
  20190822074720_20190822075013 \
  --fetch-sar --release-sar --disk-budget-gb 12
```

| thing | disk |
|-------|------|
| one MAG-POL (temporary) | ~3 GB |
| AHN city DSM | ~0.3–0.5 GB |
| tiles from 5 strips | ~2–4 GB |
| 202 MAG-POL kept on disk | ~600 GB — skip |

Then one strip only (smallest smoke test):

```bash
python download.py              # first MAG-POL + AHN over that strip
python dataset.py --one-strip
```

`.npz` keys: `image` (4,512,512) float32 HH/HV/VH/VV in dB, `mask` uint8, `look` original orientation, `meta` JSON. Apply `sar/output/dataset/norm_stats.json` in the training loop — do not z-score into the files.

See [GETTING_STARTED.md](GETTING_STARTED.md).
