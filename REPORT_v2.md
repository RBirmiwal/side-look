# Dataset v2

Near-field layover/shadow labels and a geometric look. The v1 409-tile set was not modified.

## What changed

- Look azimuth is perpendicular to the fitted flight line, on the `collect.radar.pointing` side. The SpaceNet orientation flag is not used. On these collects the flag is backwards: flag 0 looks south, flag 1 looks north.
- Masks use the near-field fold test in NEAR_FIELD.md. No single theta. Slant-range bin is `collect.image.range_resolution`. DSM heights are NAP plus NLGEO2018.
- Canonical orientation: sensor at the top of the tile (row 0), layover toward row 0. A north-up product already has that for a south-looking sensor. North-looking strips are flipped. Image, mask, and incidence flip together.
- Tile size, strides, the west-to-east 1 km block split, drop rules, and class codes are unchanged from v1.

A fitted look more than 5° from 0 or 180 is skipped rather than rotated.
No selected strip was more than 5° off north or south.

Strips that were within 5° of north or south but missed the +2.0 dB gate are not in the tiles. On each of them the other side of the mask is darker, so this is not a reversed look.

- `20190822071501_20190822071737` look 180.222°. Active − nominal +1.480 dB, shadow − nominal -1.190 dB. Opposite side of the same tiles: active − nominal -0.389 dB. Flight-line look 180.222° matches the velocity right-look (179.9°) and the mean position look (180.4°). Active pixels are on the bright side; the other side of the tile scores -0.39 dB. It misses +2.0, so it is not in the set.
- `20190823070415_20190823070716` look 180.184°. Active − nominal +1.572 dB, shadow − nominal -0.536 dB. Opposite side of the same tiles: active − nominal +0.092 dB. Flight-line look 180.184°. Shadow is darker than nominal (-0.54 dB) and the other side of the tile is only +0.09 dB, so the look is not reversed. Active misses +2.0, so the strip is not in the set.
- `20190823083559_20190823083940` look 0.480°. Active − nominal +1.409 dB, shadow − nominal n/a. Opposite side of the same tiles: active − nominal +0.283 dB. Look 0.480°. The other side scores +0.28 dB, so the look is not reversed. Active misses +2.0.
- `20190823101748_20190823102106` look 180.405°. Active − nominal +0.218 dB, shadow − nominal n/a. Opposite side of the same tiles: active − nominal -0.685 dB. Look 180.405°. The other side scores -0.69 dB, so the look is not reversed. Active misses +2.0.

## Strips

18 collects (10 looking north, 8 looking south), spread across the Rotterdam AOI so the eastern blocks are imaged by more than one swath. The two v1 strips are included; both were already on disk.

| strip | flag | look ° | snap | height m | rms m | near ° | centre ° | far ° | flip |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `20190804111224_20190804111453` | 1 | 359.620 | 0 | 1617.2 | 30.21 | 25.21 | 34.21 | 41.62 | True |
| `20190804143846_20190804144102` | 0 | 180.077 | 180 | 1627.6 | 12.73 | 24.67 | 34.24 | 42.02 | False |
| `20190822071025_20190822071321` | 1 | 0.401 | 0 | 1621.6 | 4.21 | 25.00 | 34.59 | 42.34 | True |
| `20190822074237_20190822074521` | 0 | 180.180 | 180 | 1614.0 | 8.06 | 24.42 | 34.65 | 42.84 | False |
| `20190822083256_20190822083600` | 1 | 0.398 | 0 | 1627.8 | 8.99 | 24.96 | 34.64 | 42.58 | True |
| `20190822084226_20190822084526` | 0 | 180.149 | 180 | 1619.4 | 10.00 | 24.42 | 34.59 | 42.79 | False |
| `20190822085151_20190822085445` | 0 | 180.197 | 180 | 1620.5 | 9.75 | 24.35 | 34.60 | 42.86 | False |
| `20190822090458_20190822090745` | 0 | 180.134 | 180 | 1619.2 | 7.35 | 24.44 | 34.61 | 42.76 | False |
| `20190822133333_20190822133635` | 1 | 0.468 | 0 | 1615.4 | 6.69 | 25.00 | 34.63 | 42.44 | True |
| `20190822134328_20190822134645` | 1 | 0.430 | 0 | 1616.8 | 5.64 | 25.00 | 34.60 | 42.40 | True |
| `20190822162246_20190822162529` | 0 | 180.186 | 180 | 1620.7 | 5.98 | 24.48 | 34.59 | 42.75 | False |
| `20190823065021_20190823065333` | 1 | 0.460 | 0 | 1611.5 | 3.56 | 24.85 | 34.63 | 42.54 | True |
| `20190823072627_20190823072925` | 1 | 0.457 | 0 | 1619.0 | 3.99 | 24.99 | 34.64 | 42.44 | True |
| `20190823083119_20190823083425` | 0 | 180.158 | 180 | 1790.5 | 8.13 | 24.58 | 34.71 | 42.81 | False |
| `20190823085011_20190823085317` | 0 | 180.109 | 180 | 1783.1 | 7.67 | 24.49 | 34.67 | 42.86 | False |
| `20190823124221_20190823124528` | 1 | 0.491 | 0 | 1784.9 | 11.37 | 25.13 | 34.65 | 42.42 | True |
| `20190823125228_20190823125530` | 1 | 0.431 | 0 | 1787.2 | 7.50 | 25.13 | 34.65 | 42.39 | True |
| `20190823134629_20190823134924` | 1 | 0.486 | 0 | 1790.8 | 8.08 | 25.10 | 34.62 | 42.39 | True |

## Tiles

| split | tiles |
|---|---:|
| train | 3772 |
| val | 138 |
| test | 59 |

Total 3969. Image is float16 dB. `incidence` is its own float16 key, degrees, not a fifth band. `look` is 1 if the tile was flipped. `label_version` is `v2-nearfield`.

## Class fractions

| split | nominal | active | passive | shadow |
|---|---:|---:|---:|---:|
| train | 0.7656 | 0.1242 | 0.0546 | 0.0556 |
| val | 0.5521 | 0.2523 | 0.0932 | 0.1023 |
| test | 0.5075 | 0.2806 | 0.1107 | 0.1012 |

## Acceptance

Active layover minus nominal HH must be greater than +2.0 dB on every strip. Shadow is reported and does not block.

| strip | tiles | active − nominal | shadow − nominal | active mean row |
|---|---:|---:|---:|---:|
| `20190804111224_20190804111453` | 197 | +3.501 | +0.491 | 247.3 |
| `20190804143846_20190804144102` | 183 | +3.252 | +0.516 | 253.4 |
| `20190822071025_20190822071321` | 75 | +2.933 | -0.907 | 254.4 |
| `20190822074237_20190822074521` | 212 | +2.981 | +0.449 | 252.9 |
| `20190822083256_20190822083600` | 81 | +3.660 | -0.087 | 261.0 |
| `20190822084226_20190822084526` | 447 | +3.114 | +0.288 | 238.6 |
| `20190822085151_20190822085445` | 310 | +2.720 | +0.769 | 253.0 |
| `20190822090458_20190822090745` | 203 | +2.089 | +0.383 | 252.9 |
| `20190822133333_20190822133635` | 264 | +3.252 | +0.269 | 251.7 |
| `20190822134328_20190822134645` | 231 | +2.541 | -0.639 | 250.5 |
| `20190822162246_20190822162529` | 443 | +3.740 | +0.148 | 239.6 |
| `20190823065021_20190823065333` | 271 | +3.493 | +0.218 | 256.2 |
| `20190823072627_20190823072925` | 51 | +2.081 | -0.136 | 259.2 |
| `20190823083119_20190823083425` | 421 | +3.670 | +0.471 | 238.0 |
| `20190823085011_20190823085317` | 251 | +3.873 | +0.892 | 254.3 |
| `20190823124221_20190823124528` | 257 | +2.705 | -0.838 | 249.8 |
| `20190823125228_20190823125530` | 66 | +2.633 | -0.880 | 261.1 |
| `20190823134629_20190823134924` | 6 | +2.158 | +0.870 | 266.1 |

Overall: active − nominal +3.135 dB, shadow − nominal +0.161 dB, active mean row 249.8 (tile centre is 256; layover should sit toward row 0).

Train/test bounding boxes intersecting: **0**. Train/val: 0. Val/test: 0.

Overlays: 20 PNGs in `overlays/`, covering test, train, val and look sides N, S.

## Normalisation

Train split only. Image bands. Incidence is not in these stats.

| band | mean | std |
|---|---:|---:|
| HH | -11.3101 | 5.2171 |
| HV | -14.2813 | 2.8380 |
| VH | -13.6988 | 3.2671 |
| VV | -10.8514 | 5.2773 |

Pixels: 988807168.

## Against v1, shared strips

v1 used the orientation flag as the look, which is reversed, and a single theta. v2 uses the flight line. The flip itself matches v1 on these two strips: flag 1 (actually looking north) was flipped then, and is flipped now because the sensor is on the south edge.

| strip | set | active − nominal | shadow − nominal |
|---|---|---:|---:|
| `20190804111224_20190804111453` | v2 | +3.501 | +0.491 |
| `20190804111224_20190804111453` | v1 | +1.518 | +2.834 |
| `20190822074237_20190822074521` | v2 | +2.981 | +0.449 |
| `20190822074237_20190822074521` | v1 | +1.320 | +2.482 |

## Download

Same layout as v1: `train/`, `val/`, `test/` of `.npz`, plus `manifest.csv`, `norm_stats.json`, `SHA256SUMS`, and this report. The archive is split because a GitHub release file cannot exceed 2 GB. `cat` the parts, then extract. Do not unzip.

```bash
mkdir -p side-look-tiles-v2 && cd side-look-tiles-v2
base=https://github.com/RBirmiwal/side-look/releases/download/tiles-v2
curl -L -O "$base/side-look-tiles-v2.tar.00"
curl -L -O "$base/side-look-tiles-v2.tar.01"
curl -L -O "$base/side-look-tiles-v2.tar.02"
curl -L -O "$base/side-look-tiles-v2.tar.03"
cat side-look-tiles-v2.tar.* | tar -xf -
```

3969 tiles: train 3772, val 138, test 59. v1 (`output/dataset`, 409 tiles) is unchanged.

