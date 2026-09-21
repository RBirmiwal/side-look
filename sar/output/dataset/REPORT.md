# Training dataset report

## Geometry (verified per strip)

- Incidence field: `collect.image.center_pixel.incidence_angle` — from vertical, not grazing.
- Incidence range: 33.616° – 34.613° (90−x would be 55.387° – 56.384°).
- Look azimuth: orientation flag 0 → 0° (north), 1 → 180° (south). Canonicalise=on.

## Tiles

- Tile size: 512 px (128 m) at 0.25 m.
- Train stride 256 px (50% overlap). Val/test stride 512 px (no overlap).
- Strips processed: 2 (north 1, south 1).
- Masks cached: 2.

| split | tiles |
|-------|------:|
| train | 383 |
| val | 21 |
| test | 5 |

## Per-class pixel fraction

| split | nominal | active | passive | shadow | any layover |
|-------|--------:|-------:|--------:|-------:|------------:|
| train | 0.6950 | 0.1499 | 0.0865 | 0.0685 | 0.2364 |
| val | 0.5607 | 0.2288 | 0.1160 | 0.0946 | 0.3447 |
| test | 0.4733 | 0.2709 | 0.1366 | 0.1192 | 0.4075 |

## Dropped tiles

| reason | count |
|--------|------:|
| block_boundary | 828 |
| dsm_nodata | 1414 |
| no_split | 0 |
| sar_nodata | 855 |
| zero_layover | 0 |

- Train/test bbox intersection: **none** (asserted).
- Log domain: `db` (MAG-POL magnitude is stored as dB; linear bands would get log10(x+eps)).
- Normalisation JSON written from the **train** split only.

## Layover fraction per tile

- min 0.0072  median 0.2430  max 0.5864

Twenty overlay PNGs are in `dataset/overlays/`. Copper = active layover, umber = passive, blue = shadow.
Canonicalise check: `dataset/overlays/canonicalise_pair.png` — south as-recorded, south flipped, north unflipped. Layover must point the same way in the last two.

## Strips

| strip | look | θ (vertical) | 90−x |
|-------|-----:|-------------:|-----:|
| 20190804111224_20190804111453 | 180° | 33.616° | 56.384° |
| 20190822074237_20190822074521 | 0° | 34.613° | 55.387° |

## Coverage

- Catalogued MAG-POL strips: 202 (north 102, south 100).
- Masks cached: 2 (the two tiled strips). `python dataset.py --masks-only` caches the rest without reading SAR.
- Tiling all 202 would be ~70–115 GB of compressed tiles. This run tiled a north+south pair on the same ground so canonicalise can be checked. `--fetch-sar --release-sar` tiles further strips one at a time.

