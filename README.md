# Side Look — technical status

Layover and shadow labels for Capella X-band MAG-POL over Rotterdam (SpaceNet 6). The labels come from an AHN lidar DSM and the flight line. SAR brightness is used only as a check, never as a label.

The label set to train on next is **dataset v4** ([V4.md](V4.md)). v2 is unchanged and is still the published tile release. v1 is kept. v4 is not a release yet: the tiles are not in this repo. Six of them are shown as pictures in [V4.md](V4.md).

## Settled

| fact | status |
|---|---|
| Platform | Aircraft, about 1.6–1.8 km up, not a satellite. |
| Look | Perpendicular to the fitted flight line, on the `collect.radar.pointing` side. The SpaceNet orientation flag is backwards on every strip (flag 0 looks south, flag 1 looks north). |
| Incidence | From vertical. It matches `incidence_angle`. The complement is about 21° off. Incidence runs about 24° to 43° across a strip, so one theta per strip is the wrong model. |
| Heights | AHN is NAP. Platform positions are ellipsoidal. Ellipsoidal height = NAP + NLGEO2018. On the 22 Aug strip that difference is +1.37 m. |
| Mask | Near-field fold. No single theta. Slant-range bin is `collect.image.range_resolution` (0.361 m). Shadow wins ties. |
| Classes | `0` nominal, `1` active layover, `2` passive layover, `3` shadow. |
| Canonical tile | Sensor at the top (row 0), layover toward row 0. North-looking strips are flipped. Image, mask, and incidence flip together. |
| v1 | 409 tiles. Look taken from the flag, so it is reversed. One theta per strip. Not modified. |
| v2 | 3969 tiles: 3772 train / 138 val / 59 test. `label_version` is `v2-nearfield`. |
| v4 | 3042 tiles from 15 of those 18 strips: 2918 train / 95 val / 29 test. Rule B, buildings only, AHN4 DTM. `0` nominal, `1` layover, `2` shadow, `255` ignore. `label_version` is `v4-inplace-dtm-ign`. |

Every v2 strip that was tiled has active layover brighter than nominal by more than **+2.0 dB** in HH. Overall active − nominal is **+3.14 dB**. Train, val, and test bounding boxes do not intersect. A fitted look more than 5° from north or south is skipped rather than rotated. None of the selected strips were that far off.

Four strips passed the angle test and failed the +2.0 dB gate. On each of them the other side of the mask is darker, so the look is not reversed. They are not in v2. Names and numbers are in [REPORT_v2.md](REPORT_v2.md).

## Not settled

**Shadow is not dark on the v2 labels.** Overall shadow − nominal HH on v2 is **+0.16 dB**. It does not block a strip. On the 22 Aug strip only the near-range third is slightly negative (−0.19 dB); the far third is about +1.2 dB. A new placement of the roof does not fix it: the stacked shadow profile on both strips in [DECIDE.md](DECIDE.md) is already at the background in the first bin. On the v4 buildings-only labels, P and K shadow is about −1 dB ([V4.md](V4.md)). That is a different mask. It is darker than nominal, and it is not the open-water floor.

**Where the roof goes is settled.** The image-space projection on the 22 Aug strip failed both gates: the labels sat 28–48 px too close to the sensor, and shadow was −0.04 dB ([IMAGE_SPACE.md](IMAGE_SPACE.md)). Flat ground on that strip is already at the map, and the bright roofs sit on the lidar footprint rather than on the SRTM outline ([SHIFT_CHECK.md](SHIFT_CHECK.md)). Two placement rules were then scored against five label sets (v2, and each rule on the full DSM and on buildings only). None of those sets passed the old gate. Layover is bright enough in the near and middle thirds. Shadow is not dark enough in any set, and the registration on the specified mask kept no windows because a 30 m clutter buffer removes the buildings. On the footprints, the SRTM projection is still too close to the sensor and the in-place rule is within a few metres the other way. Detail is in [RULE_CHECK.md](RULE_CHECK.md).

The follow-up in [DECIDE.md](DECIDE.md) is the decision. On Kop van Zuid the pixels the two rules disagree on — A calls them shadow, B leaves them on the footprint — match land (−0.04 dB), and the agreed no-echo beside buildings is −1.52 dB. That is rule B. The port strip does not reach the dark-no-echo bar (Z is only −1.22 dB, against water at −1.86), so its formal call is raised no-echo rather than in-place, and the same histogram says the roofs are not the no-echo population. Nothing supports the SRTM lean. The layover in front of the wall tracks `L_B` from the AHN edge height (slope 0.89 on K, n = 159). The one clean ~52 m column is 1:1.

**The opposite-look check does not show a jump.** [OPPOSITE.md](OPPOSITE.md) uses only the radar and the roof heights. No label, and `image_space.py` was not run. Kop van Zuid is not in any opposite pair: south-looking strips stop at northing 5749729, and Kop van Zuid (`…23072925`) is 5751335–5752097. The city centre is further north. The only overlap is a port band 567 m tall, northing 5749162–5749729. Six roofs there are at least 30 m above the AHN4 DTM. There are not fifteen. The pair is south `20190822074237_20190822074521` (look 180.18°) and north `20190823065021_20190823065333` (look 0.46°), 6.70 km². One building is inside both swaths with both leaned positions also inside, and it stays on the footprint (+4.87 dB from the north, +2.45 dB from the south, both brighter than the outline shifted by `L`). None of the six is dark on the footprint in both looks and bright on that outline in both. The other five are cut by the swath: the tall roofs sit on the north edge of every south-looking strip, so the leaned position toward that aircraft falls outside the image. Where a bright return is imaged from both sides, it is on the footprint. Along-track registration is within about 1 m on most windows. Range often does not lock. Nothing that locked is a 100 m shift.

[V4_CHECK.md](V4_CHECK.md) is the attempt to build that label. Ground is the AHN4 DTM, not the 5th-percentile surface, and the mask is `own_range_grid` on buildings only. The gate was layover at least +1.5 dB above nominal in the near and middle thirds, on every known non-water pixel, on both strips. Kop van Zuid passes (+2.16 / +2.62). The port strip does not (+1.18 / +1.04). The roof test is between the two bars (+0.87 dB, 70% of buildings positive), so it would not have switched on the roof-nominal edit either. No tiles were built. v2 was not modified.

[V4.md](V4.md) is the follow-up. Pixels where a non-building object shapes the return are marked 255 and left out of the contrast. On that mask the port strip clears +1.5 dB (+1.68 near, +1.52 middle, full strip). Fifteen of the eighteen v2 strips were tiled (3042 tiles). Three were left out for missing the +1.0 dB tile gate. The single-batch overfit runs. The full training run does not fit this machine.

**SNAP was not an independent check.** ESA SNAP 14.0.0 is installed and can open the 22 Aug SLC (`20190822074237_20190822074521`). The product is not a radar-geometry SLC:

- `image_geometry.type` is `surface` (`surface_type` `affine`). The pixels already sit on an EPSG:32631 grid at 0.25 m. Columns are not slant-range samples.
- There is no `terrain_models` field. The only height tag is `image_geometry.z.dem_name = SrtmGl1`.
- Missing: `range_to_first_sample`, `delta_range_sample`, `delta_line_time`, `first_line_time`, `state.direction`.
- The file is one complex band per polarisation (`complex_int16`, HH 2485 × 47228), not a 4-band stack.
- SNAP still marks the file SLC because the name contains `SLC`, then fills slant range with the missing-metadata sentinel **99999 m**. Distance from the first state vector to pixel (0, 0) is **1813 m**.

Staging printed `NOT READY` and stopped. No DSM was exported, the layover graph was not run, and there is no SNAP-versus-near-field comparison. A run on the 99999 m sentinel would not test the labels.

**Look is cardinal, not exact.** Flight-line azimuths are within 0.5° of 0° or 180°, so the mask uses an exact flip. Position-derived azimuths on the two geometry strips are up to about 3.7° off cardinal. That residual is reported, not rotated into the tiles.

## What to train on

v4 tiles are float16 dB, 4 bands, copied from v2, plus `incidence`, `look`, and a float16 `coverage` (surfaces per pixel). The mask is 0 nominal, 1 layover, 2 shadow, 255 ignore. Ignore is left out of the loss and out of every score. `label_version` is `v4-inplace-dtm-ign`. Norm stats are the v2 train stats, because the images did not change. The tiles themselves are not in a release. Counts, the per-strip gate, and the class mix are in [V4.md](V4.md) and [REPORT_v4.md](REPORT_v4.md). Val and test are much richer in layover than train (28% and 36%, against 5% on train). That is the v2 geographic split, not a new one.

v2, for the images and for anything still training on the near-field labels, is the same four dB bands plus a separate `incidence` key in degrees. Do not fold incidence into the image normalisation. `look` is 1 if the tile was flipped. Norm stats are train-only:

| band | mean | std |
|---|---:|---:|
| HH | −11.31 | 5.22 |
| HV | −14.28 | 2.84 |
| VH | −13.70 | 3.27 |
| VV | −10.85 | 5.28 |

Download is a split tar on the GitHub release `tiles-v2` (a release file cannot exceed 2 GB). `cat` the parts, then extract. Do not `unzip`.

```bash
mkdir -p side-look-tiles-v2 && cd side-look-tiles-v2
base=https://github.com/RBirmiwal/side-look/releases/download/tiles-v2
curl -L -O "$base/side-look-tiles-v2.tar.00"
curl -L -O "$base/side-look-tiles-v2.tar.01"
curl -L -O "$base/side-look-tiles-v2.tar.02"
curl -L -O "$base/side-look-tiles-v2.tar.03"
cat side-look-tiles-v2.tar.* | tar -xf -
```

v1 (409 tiles, `sar/output/dataset`) is a different release and a different label definition. Do not mix the two.

## Where the detail is

| doc | what it answers |
|---|---|
| [REPORT_v2.md](REPORT_v2.md) | Strips, geometry table, split counts, class fractions, the +2 dB gate, v1 comparison, download. |
| [NEAR_FIELD.md](NEAR_FIELD.md) | Flight-line fit, datum, fold test, closed forms, one-strip physics against far-field look 180. |
| [GEOMETRY.md](GEOMETRY.md) | Why the flag is backwards, why incidence is from vertical, swath width (707 m and 801 m are full widths). |
| [DATASET.md](DATASET.md) | Original v1 dataset note. Its look paragraph is the old flag convention. |
| [SNAP.md](SNAP.md) | Why the SNAP layover check did not run. The SLC is surface geometry; slant range fell back to 99999 m. |
| [IMAGE_SPACE.md](IMAGE_SPACE.md) | Image-space projection onto SRTM. Registration and the brightness gate both failed. No tiles. |
| [SHIFT_CHECK.md](SHIFT_CHECK.md) | Where this strip puts flat ground and buildings. Middle water is at the map. SLC matches MAG-POL. Bright roofs sit on the lidar footprint. |
| [RULE_CHECK.md](RULE_CHECK.md) | SRTM projection against labels that stay on the map column. Neither passes the old gate. Shadow is still not dark. No tiles. |
| [DECIDE.md](DECIDE.md) | The roof stays on the footprint (rule B). Layover length tracks `L_B`. |
| [OPPOSITE.md](OPPOSITE.md) | Same roofs from north and south, no label. One stays, none jumps. Kop van Zuid is not in the overlap. |
| [V4_CHECK.md](V4_CHECK.md) | Rule B with the AHN4 DTM, scored on every known dry pixel. The port strip misses +1.5 dB. No tiles. |
| [V4.md](V4.md) | The same labels with clutter marked ignore. The port strip clears +1.5 dB. Fifteen strips tiled. |
| [REPORT_v4.md](REPORT_v4.md) | Per-strip gate, ignored fraction, class shares, coverage. |
| [TRAIN.md](TRAIN.md) | U-Net recipe: 4-channel dB, train-only norm, no vertical flip, stop on mean IoU of the rare classes. |

Code for the v2 labels is `sar/near_field.py` and `sar/dataset_v2.py`. The SNAP gate is `capella_snap_stage.py`. It exits 2 on this product.
