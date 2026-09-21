# Side Look dataset

What this is, why it is built this way, and what you actually have on disk.

This is a **tiled semantic-segmentation set** for SAR layover and shadow over Rotterdam. It does not train a model. Labels never see SAR brightness — only a lidar DSM and two geometry numbers per strip.

Companion files:

- [`GETTING_STARTED.md`](public/GETTING_STARTED.md) — how to run it on a laptop
- [`sar/output/dataset/REPORT.md`](sar/output/dataset/REPORT.md) — numbers from the current tiling run
- [`sar/output/dataset/manifest.csv`](sar/output/dataset/manifest.csv) — one row per kept tile

---

## 1. Problem

A SAR image of a building is not a map of the building. Tall structure folds **toward the sensor** (layover) and hides ground **away from the sensor** (shadow). How far those effects reach is a function of height and incidence:

```
layover distance  ≈  h / tan(θ)     toward the sensor
shadow distance   ≈  h · tan(θ)     away from the sensor
```

`θ` is incidence from **vertical**, not grazing. Using `90 − θ` puts layover on the wrong side of every building. That was checked, not assumed.

SpaceNet 6 Capella MAG-POL over Rotterdam images the same city from both looks: north-facing (`orientation_flag = 0`, look azimuth 0°) and south-facing (`flag = 1`, look 180°). The same roof therefore appears in many strips, folded in opposite directions.

The dataset turns that geometry into 512×512 patches with a four-class mask so a network can learn layover/shadow from polarimetric SAR.

---

## 2. What was built

| piece | role |
|-------|------|
| `sar/catalog.py` | 202 MAG-POL strips: CRS, shape, bounds, incidence, look |
| `sar/incidence.py` | Capella field `collect.image.center_pixel.incidence_angle` (from vertical) |
| `sar/dsm_geometry.py` | DSM fold-test → classes 0–3 |
| `sar/splits.py` | 1 km geographic blocks, west→east train/val/test |
| `sar/download.py` | public S3 MAG-POL + PDOK AHN DSM |
| `sar/dataset.py` | tile, cache masks, write `.npz`, report, overlays |
| Inspector (`/`) | four crops: DSM mask vs 3DBAG boxes, flip, incidence-as-grazing |
| Dataset (`/dataset`) | this run’s stats, split map, 20 tiles as **HH SAR \| mask** |

No ESA SNAP. No PyTorch. Masks are numpy `maximum.accumulate` along ground range.

---

## 3. Source data

**SAR** — SpaceNet 6, public, no AWS account:

```
s3://spacenet-dataset/AOIs/AOI_11_Rotterdam/SAR-MAG-POL/
s3://spacenet-dataset/AOIs/AOI_11_Rotterdam/SummaryData/SAR_orientations.txt
```

Four bands, in order: **HH, HV, VH, VV**. Stored as **dB** (negative finite values). The pipeline does **not** `log10` MAG-POL.

**DSM** — AHN 0.5 m from PDOK WCS, resampled to 1 m, EPSG:28992, warped to each strip (32631). Buildings are in the surface. Water and holes are nodata and drop the tile.

**Look** comes only from the SpaceNet orientation file. **Incidence** comes only from the Capella JSON tag named above. Missing either number fails the strip — nothing is guessed.

Catalogued: **202** MAG-POL strips (102 north, 100 south). Incidence on this AOI is about **33.4°–36.6°**.

---

## 4. Mask (the labels)

Far-field scan of the DSM. One `θ` for the whole strip (Rotterdam ground range is ~700 m; that is acceptable).

Along each line of ground range `g`, increasing **away from the sensor**, with height `z`:

```
slant range   r   = g · sin(θ) − z · cos(θ)
shadow proxy  ψ   = z + g · cot(θ)
```

Walk outward. A cell is compared to the running max of cells **in front of it** (shifted accumulate, not including itself).

| code | name | rule |
|-----:|------|------|
| 0 | nominal | neither layover nor shadow |
| 1 | active layover | `r[i] < max(r[:i])` — the roof / steep face that folded forward |
| 2 | passive layover | same slant-range bin as an active cell, itself monotonic — the ground the roof landed on |
| 3 | shadow | `ψ[i] < max(ψ[:i])` — **wins** if both would apply (no return) |

Image brightness is not an input. A bright layover blob in the SAR does not move a label.

---

## 5. Tiles

| | |
|--|--|
| size | 512 px = **128 m** at 0.25 m GSD |
| train stride | 256 px (50% overlap) |
| val / test stride | 512 px (no overlap) |
| CRS of bboxes | strip CRS (UTM 31N, EPSG:32631) |

A window is **dropped** if:

- SAR has nodata in the patch
- DSM has nodata in the patch
- the 128 m box sits on a 1 km block edge (so a train tile cannot leak across the cut)
- (optional) layover fraction is zero, if you cap that — currently **kept**; a detector needs negatives

All-nominal tiles are kept.

---

## 6. Splits

**Geographic, not random, not by strip.**

The union of strip footprints is cut into a **1 km BlockGrid**. Blocks are ordered west → east. About **70% / 15% / 15%** of blocks become train / val / test.

SpaceNet strips overlap: the same building is in many collects from both looks. A strip-level split would put that building in train and test. A tile belongs to the block that contains its **centre**.

Asserted on every run: train and test bounding boxes do not intersect.

This two-strip run is **west-heavy**, so almost everything landed in train (383 / 21 / 5). More strips further east will fill val and test. The 70/15/15 ratio is a parameter (`--train-frac` etc.), not baked into the mask cache.

---

## 7. Canonical look

South-facing strips are **flipped** (`numpy.flip` on the row axis) so layover sits on the **same side of the tile** as north-facing. The network then sees one orientation.

The original north/south flag is still stored in the `look` channel (constant over the tile). `--no-canonicalise` leaves south-facing as recorded.

Check image: `public/dataset/overlays/canonicalise_pair.png` — south as-recorded, south flipped, north unflipped. The last two must agree.

---

## 8. File format

Each tile is one compressed archive:

```python
import numpy as np, json
z = np.load("tile.npz")
image = z["image"]          # float32 (4, 512, 512)  HH, HV, VH, VV in dB
mask  = z["mask"]           # uint8  (512, 512)      0 / 1 / 2 / 3
look  = z["look"]           # uint8  (512, 512)      original orientation flag (0 or 1)
meta  = json.loads(str(z["meta"]))
```

Do **not** z-score into the files. Apply [`norm_stats.json`](public/dataset/norm_stats.json) in the training loop. Stats are **train-only** (this run):

| band | mean (dB) | std |
|------|----------:|----:|
| HH | −11.202 | 5.091 |
| HV | −14.126 | 2.909 |
| VH | −13.484 | 3.356 |
| VV | −10.771 | 5.099 |

`manifest.csv` has bbox, split, strip id, θ, look, flipped, and per-class fractions for every kept tile.

---

## 9. This run (what the Dataset page shows)

Two overlapping strips on the same ground, so canonicalise can be checked:

| strip | look | θ (from vertical) |
|-------|------|------------------:|
| `20190804111224_20190804111453` | south (180°) | 33.616° |
| `20190822074237_20190822074521` | north (0°) | 34.613° |

| split | tiles |
|-------|------:|
| train | 383 |
| val | 21 |
| test | 5 |
| **total** | **409** |

| class (train pixels) | fraction |
|----------------------|---------:|
| nominal | 0.695 |
| active layover | 0.150 |
| passive layover | 0.087 |
| shadow | 0.069 |
| any layover (active + passive) | 0.236 |

Dropped windows: 855 SAR-nodata, 1,414 DSM-nodata, 828 block-boundary. Zero-layover tiles were **not** dropped.

Per-tile layover fraction (the number on each Dataset card, e.g. `train 0.41`): share of pixels that are active or passive layover. This run: min 0.007, median 0.243, max 0.586. Shadow is not in that number.

Each Dataset card is **HH SAR** (2–98% stretch, no paint) next to the **mask on SAR** (copper active, umber passive, blue shadow).

---

## 10. How many tiles will I have?

Not a fixed count per strip. A MAG-POL collect is ~40k × 3k px. Before drops that is ~2,000 train windows and ~550 eval windows. After nodata and block edges you keep on the order of **~200 tiles per strip**.

| run | strips | tiles (est.) | `.npz` on disk |
|-----|-------:|-------------:|----------------|
| this page | 2 | **409** | 1.5 GB (~3.6 MB/tile) |
| `--max-strips 5` | 5 | **~1,000** | ~4 GB |
| all MAG-POL | 202 | **~41,000** | ~150 GB |

One source strip is ~3 GB. 202 of those will not fit; tile then delete:

```bash
cd sar
python dataset.py --max-strips 5 --fetch-sar --release-sar --disk-budget-gb 12
```

`--masks-only` caches a DSM mask for every catalogued strip without reading SAR (small). Changing tile size or the 70/15/15 cut **retiles from those masks**. Changing incidence or look **rebuilds the masks**.

---

## 11. What is not in the labels

- SAR intensity / polarimetry (inputs only)
- 3DBAG box extrusions (inspector comparison only; training masks are DSM)
- SNAP / orbit / DEM-in-radar processing
- a trained model

If layover in the image and the copper overlay disagree, the overlay is the DSM fold-test at that strip’s θ and look — that is the ground truth this set defines.
