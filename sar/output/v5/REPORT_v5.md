# Dataset v5

Same 15 strips as v4. Rule B (`own_range_grid`) on a buildings-only scene,
AHN4 DTM as the ground, then 3DBAG footprints dilated 2 m set to 255, plus a
per-pixel layover magnitude in metres. v2 was not modified. Images, incidence,
look, grid and split come from the v2 tiles. `label_version` is `v5-agreed-mag`.

The v4 mask rasters are not on this machine, so this is not a pixel-for-pixel
compare. The table below is the v4 recipe (mask before the footprint 255) scored
on the v2 tiles the same way REPORT_v4.md scored them: overlapping train tiles
count twice, water stays labelled, ignored pixels stay out of the brightness.

## Choices

- AHN4 `dsm_05m` and `dtm_05m` from the PDOK WCS, 0.5 m, bilinear onto the 0.25 m
  MAG-POL grid in EPSG:32631. Each mosaic was checked against a fresh WCS chip.
- Range pad 250 m (1000 px). `nan_margin` 400 cells (100 m), same count as v4.
- Water: DSM nodata components of at least 2500 m² inside the strip, filled with
  the median of the edge below 2 m NAP (else the 10th percentile). The pad is not water.
- Holes under 100 m²: median of a 2 px ring on the 0.5 m grid. Whatever nodata is
  left, DSM and DTM, is then filled from the nearest finite cell. That is what
  makes the unknown mask empty, which is what v4 reported. The DSM is not copied
  into DTM holes: that would paste the roof in as the ground.
- Footprints are 3DBAG v20250903 `pand` polygons (gpkg tiles), not the sidelook-v2
  index. Heights are `lod22_2d` `b3_h_70p` / `b3_h_max` in metres NAP. The ratio
  check uses height = `b3_h_70p − b3_h_maaiveld`.
- Dilations at 0.25 m: 4 px (1 m) around footprints before the building surface is
  pasted, 8 px (2 m) for clutter and for the v5 footprint zone.
- Ground range is `H tan(incidence)` from the v2 tiles, one linear fit per column.
  Columns with no tile use the strip's median fit, so the full strip can be scored.
  H is the v2 platform height. The fit slope is about 0.25 m per source row, which
  is one ground-range metre per ground metre.
- A tile is kept if it has at least one labelled pixel, including tiles that are
  more than 60% ignored. v4 dropped those.
- Magnitude is `layover_magnitude` at step 0.1 m and pixel size 0.25 m. It is 0
  off layover and 0 wherever the mask is 255. Coverage is the buildings-only
  coverage, including on ignored pixels.

## Against v4, on the v2 tiles

| strip | v4 ignored | v5 full-strip ignored | v4 near dB | v5 near | v4 middle | v5 middle | v4 drop >60% | v5 >60% (kept) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `20190822074521` | 37.0% | 38.8% | +1.72 | +1.51 | +1.34 | +1.09 | 24 | 72 |
| `20190823072925` | 29.2% | 33.6% | +2.90 | +2.92 | +3.32 | +3.30 | 7 | 20 |
| `20190804111453` | 33.3% | 38.1% | +1.20 | +2.77 | +2.54 | +2.86 | 16 | 83 |
| `20190804144102` | 37.6% | 39.4% | +1.92 | +2.84 | +1.84 | +1.82 | 27 | 63 |
| `20190822071321` | 21.4% | 23.4% | +2.74 | +2.98 | +3.88 | +4.03 | 8 | 35 |
| `20190822084526` | 47.7% | 44.5% | +4.82 | +5.00 | +2.04 | +3.47 | 67 | 79 |
| `20190822133635` | 35.6% | 39.3% | +1.38 | +3.16 | +2.18 | +2.03 | 31 | 100 |
| `20190822083600` | 19.9% | 20.3% | +1.52 | +1.18 | +3.42 | +3.17 | 2 | 27 |
| `20190822085445` | 42.5% | 42.2% | +1.06 | +2.44 | +1.62 | +1.55 | 49 | 64 |
| `20190822134645` | 33.6% | 36.4% | +1.48 | +1.54 | +1.44 | +0.96 | 34 | 79 |
| `20190822162529` | 47.2% | 44.2% | +6.06 | +5.10 | +3.34 | +3.71 | 65 | 79 |
| `20190823065333` | 37.6% | 41.0% | +1.56 | +2.04 | +2.52 | +2.62 | 32 | 103 |
| `20190823083425` | 49.0% | 44.8% | +3.14 | +3.86 | +10.16 | +4.57 | 64 | 71 |
| `20190823085317` | 38.1% | 39.7% | +1.94 | +1.86 | +1.92 | +3.59 | 30 | 58 |
| `20190823125530` | 20.6% | 21.7% | +2.42 | +2.32 | +3.84 | +3.89 | 5 | 30 |

Near is the third of the strip closest to the aircraft. The first pass of the scorer used source-row fraction, which swaps near and far on a north-looking strip. The v5 columns above are corrected; the middle third does not move. Kop van Zuid (`…23072925`) is then +2.92 / +3.30 against v4's +2.90 / +3.32.

The strips that are still not close: `…04111453` near (+2.77 against +1.20), `…22133635` near (+3.16 against +1.38), `…22085445` near (+2.44 against +1.06), `…22084526` middle (+3.47 against +2.04), `…23085317` middle (+3.59 against +1.92), and `…23083425` middle (+4.57 against +10.16). v4's +10.16 was already an outlier in REPORT_v4.md. These are not the same 3DBAG extract, and the clutter mask is rebuilt, so the nominal class is not the same set of pixels. Ignored fraction and the middle third of K are the numbers that say the scene is the same one.

Class shares below are after footprints are marked 255. v4's train layover share was 0.051 of non-ignored pixels. v5's is 0.012, because rule B calls the near part of the roof layover and that roof is inside the footprint zone.

## Magnitude / L

Flat roofs (`b3_h_max − b3_h_70p < 2`), height at least 15 m, deeper along range
than `layover_len_m`, with the zone from 2 m to L in front of the near wall inside
one tile. L is `g − sqrt(g² − 2Hh + h²)` with g from the tile incidence at the wall.

n = 54 (capped at 200), median 0.822, p10 0.382, p90 0.968.

A units mix-up (pixels taken as metres, or the range axis reversed) would put the
median near 4, 0.25 or 0. The gate is the median, 0.8–1.2. A single building can
sit well below that when its cadastral near edge is not the tall wall: the roof
echo then lands inside the footprint, which v5 marks 255, and the only labelled
layover in front of that edge is a short band. Those buildings pull the 10th
percentile down. They are not a scale error. Kop van Zuid, where the tall wall
is the near wall, is the check that the metres are right.

## Magnitude by split, layover pixels

| split | n | mean | median | p90 | p99 | max | share < 1 m |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 6172613 | 11.7 | 8.3 | 24.1 | 71.6 | 84.9 | 0.8% |
| val | 1647876 | 17.9 | 15.6 | 29.3 | 73.3 | 117.9 | 0.0% |
| test | 550894 | 14.3 | 14.6 | 22.6 | 37.1 | 62.9 | 0.0% |

## Class shares of non-ignored pixels

| split | nominal | layover | shadow | ignored |
|---|---:|---:|---:|---:|
| train | 0.986 | 0.012 | 0.002 | 41.2% |
| val | 0.792 | 0.173 | 0.035 | 65.8% |
| test | 0.823 | 0.154 | 0.022 | 72.2% |

Kept tiles 3503. Tiles more than 60% ignored, and kept: 963.

## Inputs

- AHN WCS `https://service.pdok.nl/rws/ahn/wcs/v1_0` coverages `dsm_05m`, `dtm_05m`, CRS EPSG:28992.
- 3DBAG v20250903 `https://data.3dbag.nl/v20250903/tiles/9/{x}/{y}/9-{x}-{y}.gpkg.gz`.
- v2 tiles for the image, the incidence and the grid. No SAR image was re-downloaded.

