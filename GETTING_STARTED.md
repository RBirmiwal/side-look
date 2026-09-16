# Side Look — laptop setup

Two pieces: the **pipeline** (this folder) and the **data** (public SpaceNet 6 + Dutch AHN lidar). The live preview in chat is the inspector; this is what you run to build tiles.

Nothing here needs an AWS account. SpaceNet 6 MAG-POL is a public bucket.

## 1. Unzip and Python

You want Python 3.10+ and [GDAL/rasterio](https://rasterio.readthedocs.io/). On macOS, Homebrew GDAL first is the least painful path:

```bash
brew install gdal
python3 -m venv .venv
source .venv/bin/activate
pip install -r sar/requirements.txt
pip install awscli          # only used with --no-sign-request
```

On Ubuntu: `sudo apt install gdal-bin libgdal-dev python3-venv` then the same venv.

```bash
cd sar
python -m pytest            # geometry + split + incidence tests, no SAR required
```

## 2. Pull the data

SpaceNet 6, Rotterdam, MAG-POL (HH/HV/VH/VV, already in dB):

```text
s3://spacenet-dataset/AOIs/AOI_11_Rotterdam/SAR-MAG-POL/
s3://spacenet-dataset/AOIs/AOI_11_Rotterdam/SummaryData/SAR_orientations.txt
```

AHN DSM (buildings included) comes from PDOK WCS — no login.

```bash
cd sar
python download.py                 # one MAG-POL strip + AHN over that strip + 3DBAG
python download.py --ahn-aoi       # AHN covering all 202 strip footprints (~15 × 9 km)
python download.py --strip 20190822074237_20190822074521   # extra strip by id
```

A MAG-POL GeoTIFF is ~3 GB. 202 of them will not fit on a small disk. Pull what you need, tile, delete (`--release-sar`).

List every strip:

```bash
aws s3 ls s3://spacenet-dataset/AOIs/AOI_11_Rotterdam/SAR-MAG-POL/ --no-sign-request
```

The catalog in `sar/output/strip_catalog.json` already has CRS, shape, incidence, and look for all 202. Rebuilding it:

```bash
python catalog.py
```

Incidence field (verified, from vertical, not grazing):

```text
collect.image.center_pixel.incidence_angle
```

Look azimuth is the SpaceNet orientation flag: `0` → 0° (north-facing), `1` → 180° (south-facing). Missing either number fails the run — it will not guess.

## 3. Build tiles

```bash
python dataset.py --one-strip
# then, with more MAG-POL on disk:
python dataset.py --strips 20190804111224_20190804111453 20190822074237_20190822074521
# or walk the catalog, download each strip, tile, delete:
python dataset.py --max-strips 12 --fetch-sar --release-sar
python dataset.py --masks-only          # DSM fold-test masks for all 202, no SAR pixels
```

Outputs land in `sar/output/dataset/`:

| file | what |
|------|------|
| `train/*.npz` `val/*.npz` `test/*.npz` | tiles |
| `manifest.csv` | bbox, split, θ, look, class fractions |
| `norm_stats.json` | train-only mean/std of HH/HV/VH/VV |
| `REPORT.md` | validation report |
| `overlays/` | 20 PNG overlays + canonicalise pair |

Each `.npz`:

```python
import numpy as np, json
z = np.load("tile.npz")
image = z["image"]   # float32 (4, 512, 512)  HH, HV, VH, VV in dB
mask  = z["mask"]    # uint8  (512, 512)      0 nom / 1 active / 2 passive / 3 shadow
look  = z["look"]    # uint8  (512, 512)      original orientation flag
meta  = json.loads(str(z["meta"]))
```

Do **not** z-score into the files. Apply `norm_stats.json` in the training loop. MAG-POL is already dB — the pipeline does not `log10` it.

Splits are 1 km geographic blocks, west → train, then val, then east → test. Not random, not by strip. Tiles that sit on a block edge are dropped.

South-facing tiles are flipped so layover sits on the same side as north-facing. `--no-canonicalise` leaves them as recorded.

## 4. Disk reality

| thing | size |
|-------|------|
| one MAG-POL strip | ~3 GB |
| one 512² tile `.npz` | ~3.5 MB |
| 409 tiles (this run, 2 strips) | ~1.5 GB |
| all 202 tiled | tens of GB of tiles + 600 GB of source if you kept every strip |

`--disk-budget-gb 20` stops tiling when the dataset directory grows past that. Masks are small; cache those for every strip with `--masks-only`.

## 5. Viewer (optional)

The Inspector / Dataset pages are a TanStack + React app. From the unzipped root, with Node 22:

```bash
npm install
npm run dev
```

It reads `public/dataset/` (report, split map, overlays) and `public/overlays/` (the four inspector crops). It does not load the `.npz` files.

## 6. What is already in this zip

- Pipeline source and tests
- `SAR_orientations.txt` and the 202-strip catalog (incidence + look already joined)
- Validation report, manifest, train-only norm stats, split map, 20 overlays, canonicalise pair
- A few example `.npz` tiles (not the full 409 — regenerate those)

The two example strips used here:

- `20190804111224_20190804111453` south-facing, θ ≈ 33.616°
- `20190822074237_20190822074521` north-facing, θ ≈ 34.613°
