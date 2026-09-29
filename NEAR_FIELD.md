# Near-field fold test

The platform is an aircraft about 1.6 km up, not a satellite. Incidence runs from about 24° to 43° across a strip, so one theta is the wrong model. The SpaceNet orientation flag is also backwards: flag 0 on this strip means the sensor sits north of the scene and looks south.

This note replaces the far-field fold test for that airborne geometry. The original 409-tile dataset was not modified. Masks and tiles below are only for `20190822074237_20190822074521`.

Code: [`sar/near_field.py`](sar/near_field.py), runner [`sar/near_field_run.py`](sar/near_field_run.py), tests [`sar/tests/test_near_field.py`](sar/tests/test_near_field.py).

## Range extent

The GeoTIFF affine is exactly 0.25 m on both axes, axis-aligned. The product JSON quotes 0.25008 m and 0.25017 m. The raster the mask is painted on is 0.25 m.

| strip | pixels (row × col) | spacing | range swath | along-track |
|---|---:|---:|---:|---:|
| `20190804111224…` | 2829 × 42724 | 0.25 m | **707.25 m** | 10.68 km |
| `20190822074237…` | 3204 × 47260 | 0.25 m | **801.00 m** | 11.82 km |

707 m and 801 m are the full near-to-far width of the short axis, not a distance from the centre. Half-swath is about 354 m and 400 m. That matches DATASET.md (~3k px × 0.25 m ≈ 750 m).

## Flight line

328 state vectors, ENU at the centre target. A straight line is the first principal axis, oriented with time.

| | |
|---|---:|
| residual rms | 8.060 m |
| residual max | 19.547 m |
| mean height above the centre target | 1614.04 m |
| height at the broadside foot | 1614.11 m |
| `radar.pointing` | right |
| look, perpendicular to the line, right side | **180.180°** |

180.180° is within 0.5° of south, so the mask uses an exact flip, not a bilinear rotate. The SpaceNet flag on this strip is 0. The old pipeline treated that as look 0 (north). Tiles are canonicalised from the geometric look (`flipped: true` in the npz meta). The stored `look` channel is still the flag (0). The azimuth that was used is `meta.look_azimuth_deg`.

## Vertical datum

AHN heights are NAP. Platform positions are ellipsoidal. Undulation is NLGEO2018. pyproj's EPSG:7415 → EPSG:4937 path does not apply the geoid here; the grid is `sar/data/nl_nsgi_nlgeo2018.tif` (gitignored, from the PROJ CDN). Ellipsoidal height = NAP + N.

| | height |
|---|---:|
| centre target, ellipsoidal | 45.962 m |
| AHN NAP at that point | 3.689 m |
| NLGEO2018 N | 43.642 m |
| DSM ellipsoidal | 47.331 m |
| DSM − target | **+1.369 m** |

Within a few metres.

## Fold test

No theta. For each DSM cell:

```
r     = perpendicular distance to the flight line
alpha = off-nadir angle from the flight line to the cell
```

Scan each line perpendicular to the track, moving away from it:

| class | code | test |
|---|---:|---|
| nominal | 0 | none of the tests below |
| active layover | 1 | `r[i] < max(r[:i])` |
| passive layover | 2 | shares a slant-range bin with an active cell |
| shadow | 3 | `alpha[i] < max(alpha[:i])` — wins ties |

Slant-range bin width is `collect.image.range_resolution` = **0.360710 m**. The far-field default was `pixel_size * sin(theta)`. There is no single theta to put in that formula.

Closed form, platform height `H` above flat ground, building height `h`, near wall at ground range `g`:

```
layover = g - sqrt(g² - 2·H·h + h²)     toward the track, if the root is real
shadow  = g · h / (H - h)               beyond the far wall
```

A building nearer the track has a longer layover than the same building farther out. With `H` very large and the building at a fixed incidence, the extents match the existing far-field scan.

## One strip vs look-180 far-field

Same 212 tiles. Thirds are by the tile-centre row. Row 0 is near range. HH dB.

| | shadow − nominal | active − nominal | tiles |
|---|---:|---:|---:|
| near-field, all | **+0.449** | **+2.981** | 212 |
| near-field, near | −0.194 | +2.159 | 50 |
| near-field, middle | +0.148 | +2.869 | 95 |
| near-field, far | +1.231 | +4.260 | 67 |
| far-field look 180, θ as-read, all | **+0.419** | **+3.082** | 212 |
| far-field look 180, near | −0.178 | +2.205 | 50 |
| far-field look 180, middle | +0.167 | +2.948 | 95 |
| far-field look 180, far | +1.346 | +4.125 | 67 |

Finite-pixel class fractions (nominal / active / passive / shadow):

| | nominal | active | passive | shadow |
|---|---:|---:|---:|---:|
| near-field | 0.710 | 0.153 | 0.065 | 0.073 |
| far-field look 180 | 0.689 | 0.151 | 0.087 | 0.073 |

Once the look is south, switching from a constant theta to the flight line barely moves the brightness. Active layover is clearly bright. Shadow is not clearly dark. Only the near third is slightly negative. The far third is worse.

The far-field comparison tiles were stored unflipped, because `--look-azimuth` does not change the orientation flag. The figures flip that mask so the HH lines up with the near-field tiles. Left to right: HH, far-field look 180, near-field.

- [01024_20480](sar/output/near_field_compare/20190822074237_20190822074521_01024_20480_pair.png)
- [01536_16384](sar/output/near_field_compare/20190822074237_20190822074521_01536_16384_pair.png)
- [00512_43520](sar/output/near_field_compare/20190822074237_20190822074521_00512_43520_pair.png)
- [01792_16384](sar/output/near_field_compare/20190822074237_20190822074521_01792_16384_pair.png)

Outputs, separate from `output/dataset`:

- `sar/output/masks_near_field/` (not in git; GeoTIFF)
- `sar/output/dataset_near_field/` (212 npz, not in git)
- `sar/output/near_field_compare/` (the four PNGs above)
