# Layover magnitude

A per-pixel target for how far the echoes in a layover pixel have been moved, in metres. The functions are in `sar/image_space.py`. The tiles are dataset v5, release `v5`. A v2 tile cannot supply this target. The heights come from AHN4.

## What the number is

For each cell, the largest horizontal distance along the range line between that cell and any visible surface whose echo shares its closest-approach range. On a layover pixel that is how far the farthest echo in the pixel came from. Off layover it is 0.

```
own_range_displacement_grid(r, alpha, axis, nan_margin, step, pixel_m)
    -> displacement_m float32, unknown bool
layover_magnitude(coverage, displacement)
    -> float32, the displacement where coverage >= 1.5, else 0
```

`r` and `alpha` are the DSM surface, one value per cell: closest-approach range to the flight line, and the off-nadir angle of that surface point. Canonical orientation, sensor at index 0. This is the same pair `own_range_grid` already takes for rule B. `step` is 0.1 m and `pixel_m` is the cell size, 0.25 m on these strips.

Two closed forms, from the tests (`H = 1614` m, a 20 m box):

| where | what the magnitude is |
|---|---|
| Ground in front of a wide roof | About `L`, the distance to the roof point at the same range. The test measured 29.1–29.9 m against 28.8–29.5 m. |
| Ground just in front of a narrow wall | The distance to the wall. It ramps up as you move away. The test reached 23.4 m before the roof echo took over at about 30 m. |
| On the roof of a narrow building | The same distance back, to the ground point whose range matches the roof. |
| Shadow with nothing at that range | 0 |
| Flat ground | Under 1 m, which is the bin size, not a real shift. |

`L` here is the near-field length `g − √(g² − 2·H·h + h²)`, the same form as [NEAR_FIELD.md](NEAR_FIELD.md). On a wide building the raster is nearly constant and close to `L`. On a narrow building it is not: a single `L` misses the ramp.

`sar/tests/test_image_space.py` covers both rules and the magnitude. It prints 19/19 passed. Rules A and B were not changed.

## v5

v5 is the v4 recipe plus this magnitude. Rule B (`own_range_grid`) on a buildings-only scene, AHN4 DTM as the ground, then 3DBAG footprints dilated 2 m set to 255, so training only sees pixels where the leaning rule and the in-place rule agree. Same 15 strips as v4. The v4 tiles were not on disk, so the scene was rebuilt from the public inputs: PDOK WCS `dsm_05m` and `dtm_05m`, 3DBAG v20250903, and the v2 tiles for the image, the incidence and the grid. v2 was not modified. `label_version` is `v5-agreed-mag`.

3,503 tiles: 3,348 train, 106 val, 49 test. Every v2 tile of those 15 strips is kept, including 963 that are more than 60% ignored. v4 dropped those. The mask pack is the GitHub release `v5` (`v5_masks.tar`, 61 MB, and `v5_build_code.tar`, 1.4 MB). Per-strip numbers are in [REPORT_v5.md](sar/output/v5/REPORT_v5.md).

On 54 flat roofs (`b3_h_max − b3_h_70p < 2` m, height at least 15 m, deeper along range than `L`, the zone from 2 m to `L` in front of the near wall inside one tile) the median of magnitude / `L` is 0.822. The 10th percentile is 0.382 and the 90th is 0.968. Kop van Zuid is 0.875. The median is inside 0.8–1.2, so this is not a pixel/metre mix-up and not a reversed range axis. The low 10th percentile is buildings whose tall wall is not the cadastral near edge: the roof echo then falls inside the footprint, which is 255, and the labelled layover in front of that edge is a short band.

| split | layover pixels | mean | median | p90 | p99 | max | share < 1 m |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 6,172,613 | 11.7 | 8.3 | 24.1 | 71.6 | 84.9 | 0.8% |
| val | 1,647,876 | 17.9 | 15.6 | 29.3 | 73.3 | 117.9 | 0.0% |
| test | 550,894 | 14.3 | 14.6 | 22.6 | 37.1 | 62.9 | 0.0% |

After the footprint is marked 255, train layover is 1.2% of non-ignored pixels. v4's was 5.1%. Rule B calls the near part of the roof layover, and that roof is inside the footprint zone.

Kop van Zuid (`…23072925`) is +2.92 / +3.30 dB, layover minus nominal, near and middle, against v4's +2.90 / +3.32. Six strips are not that close. Near third: `…04111453` +2.77 against +1.20, `…22133635` +3.16 against +1.38, `…22085445` +2.44 against +1.06. Middle third: `…22084526` +3.47 against +2.04, `…23085317` +3.59 against +1.92, `…23083425` +4.57 against +10.16. v4's +10.16 was already an outlier. The clutter mask is rebuilt and the 3DBAG extract is not the one v4 used, so the nominal class is not the same set of pixels.

```bash
base=https://github.com/RBirmiwal/side-look/releases/download/v5
curl -L -O "$base/v5_masks.tar"
curl -L -O "$base/v5_build_code.tar"
tar -xf v5_masks.tar
python3 merge_masks.py --v2 side-look-tiles-v2 --pack v5_masks --out side-look-tiles-v5
```

The merge checked every v2 image md5 and reported magnitude present and valid in every tile. `train.py` can score the extra channel. It was not run here. There is no PyTorch on this machine.

## The v2 tiles are not a substitute

A v2 `.npz` holds:

| key | what it is |
|---|---|
| `image` | HH, HV, VH, VV, float16 dB, `(4, 512, 512)` |
| `mask` | uint8. `0` nominal, `1` active layover, `2` passive layover, `3` shadow. The near-field fold, already collapsed to a class. |
| `incidence` | float16 degrees, the look angle of the grid. It increases with row. It is not the off-nadir angle of the roof. |
| `look` | 1 if the tile was flipped so the sensor sits at row 0. |
| `meta` | JSON. One `platform_height_m` for the strip, about 1,611–1,791 m, plus the bbox, the flight-line look, and `label_version` `v2-nearfield`. |

There is no height raster. The DSM that produced the mask was not written into the tile. Incidence and `platform_height_m` fix the ground range of a flat surface. They do not fix `r` and `alpha` of a roof. Many height fields map to the same four-class mask, and the displacement is the quantity that was discarded when the mask was written.

The building index ([SIDELOOK.md](SIDELOOK.md), release `sidelook-v2`) does not fill the gap. It has a footprint and two heights per pand, and one `layover_len_m` per building per strip. That length is the closed form at the near wall. It is one number, not a raster. Using it as the target would be a different dataset:

- In front of a wide building it is close to the magnitude, because the magnitude is about `L` there.
- In front of a narrow building it is wrong. The magnitude ramps with distance to the wall, then jumps to the roof point. One `L` cannot show that.
- The v2 mask is not the v4 mask. v4 is rule B, buildings only, three classes plus ignore. v2 is the near-field fold, four classes, nothing ignored. The two rules disagree on the footprint, which is why v5 marks footprints 255.
- Containers, cranes and ships are in the v2 mask and not in 3DBAG. A rear wall that fills the shadow in front of it is also invisible to a per-building length.
- v2 has no `coverage`. v5 keeps the buildings-only coverage, including on ignored pixels.

v2 is the right source for the images, the incidence, the flight height, and the grid. It is not a source for the magnitude. The index is the right source for the check that a real magnitude should come out near `L` on wide buildings. It is not the target.
