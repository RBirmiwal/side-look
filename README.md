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
python3 -m venv .venv && source .venv/bin/activate
pip install -r sar/requirements.txt
pip install awscli
```

macOS: `brew install gdal` first. Ubuntu: `sudo apt install gdal-bin libgdal-dev python3-venv`.

```bash
cd sar
python -m pytest
```

## Data (public, not in this repo)

```text
s3://spacenet-dataset/AOIs/AOI_11_Rotterdam/SAR-MAG-POL/
s3://spacenet-dataset/AOIs/AOI_11_Rotterdam/SummaryData/SAR_orientations.txt
```

No AWS account. Each MAG-POL GeoTIFF is ~3 GB.

```bash
cd sar
python download.py                                          # one strip + AHN + 3DBAG
python catalog.py                                           # 202-strip metadata
python download.py --ahn-aoi                                # AHN over all footprints
python download.py --strip 20190822074237_20190822074521    # extra strip
python dataset.py --one-strip
python dataset.py --masks-only
python dataset.py --max-strips 12 --fetch-sar --release-sar
```

`.npz` keys: `image` (4,512,512) float32 HH/HV/VH/VV in dB, `mask` uint8, `look` original orientation, `meta` JSON. Apply `sar/output/dataset/norm_stats.json` in the training loop — do not z-score into the files.

Example strips: south `20190804111224_20190804111453` (θ ≈ 33.616°), north `20190822074237_20190822074521` (θ ≈ 34.613°).

See [GETTING_STARTED.md](GETTING_STARTED.md) and [sar/README.md](sar/README.md).
