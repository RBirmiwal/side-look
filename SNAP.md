# SNAP layover check

Independent check of the near-field labels with ESA SNAP's layover/shadow mask, on one strip: `20190822074237_20190822074521`.

It did not run. The SpaceNet SLC is not a radar-geometry product, so SAR-Simulation has nothing to simulate. Dataset v2 was not modified.

## Install

SNAP 14.0.0, headless, no root:

```bash
sh esa-snap_sentinel_linux-14.0.0.sh -q -dir $HOME/esa-snap
```

`VERSION.txt` is `14.0.0`. Heap in `$HOME/esa-snap/bin/gpt.vmoptions` is `-Xmx2816m`, 70% of 3.84 GB.

`gpt -h SAR-Simulation` and `gpt -h Terrain-Correction` both succeed. Every parameter name in [snap_layover.xml](snap_layover.xml) is one of those operators: `externalDEMFile`, `externalDEMNoDataValue`, `externalDEMApplyEGM`, `demResamplingMethod`, `saveLayoverShadowMask`, `pixelSpacingInMeter`, `mapProjection`, `alignToStandardGrid`, `standardGridOriginX`, `standardGridOriginY`, `sourceBands`. The subset uses `region` and `copyMetadata`.

## The SLC

From `s3://spacenet-dataset/AOIs/AOI_11_Rotterdam/SAR-SLC/`. One GeoTIFF per polarisation, not one multi-band file.

| file | size |
|---|---:|
| `CAPELLA_ARL_SM_SLC_HH_…20190822074237_20190822074521.tif` | 384 MB |
| HV | 417 MB |
| VH | 410 MB |
| VV | 388 MB |

HH was the file opened. **1 band, `complex_int16`, 2485 × 47228**, EPSG:32631. Each polarisation has its own `.json` and `_extended.json`.

## What the metadata actually is

`collect.image.image_geometry`:

| field | value |
|---|---|
| `type` | `surface` |
| `surface_type` | `affine` |
| `z.dem_name` | `SrtmGl1` |
| `terrain_models` | absent, anywhere in the product |

The geotransform is a UTM grid. Column spacing is 0.25 m on the ground. That axis is not slant range.

SNAP's Capella reader (`CapellaProductDirectory` in the 14.0.0 `sar-io` jar) only builds a slant-range-time tie-point grid when `image_geometry.range_to_first_sample` is present. These are missing:

- `collect.image.image_geometry.range_to_first_sample`
- `collect.image.image_geometry.delta_range_sample`
- `collect.image.image_geometry.delta_line_time`
- `collect.image.image_geometry.first_line_time`
- `collect.state.direction`

`pixel_spacing_column` is not a stand-in for `delta_range_sample`. On this product it is map spacing.

## What SNAP did anyway

The reader treats any filename containing `SLC` as radar geometry. It opened the extended JSON. A 64×64 subset wrote a BEAM-DIMAP product with bands `i_HH`, `q_HH`, and a virtual intensity.

Because `range_to_first_sample` is absent, `slant_range_to_first_pixel` stays at the missing-metadata sentinel **99999 m**. The subset operator then rewrites it to the far sample of that window:

```
99999 + 63 × 0.25011978187 = 100014.76 m
```

Distance from the first state vector to pixel (0, 0), at 46 m ellipsoidal height, is **1813 m**. Orbit vectors themselves were read correctly (position arrays became `x_pos` / `y_pos` / `z_pos`). PRF is 984.25 Hz. Line time was inferred from `start_timestamp` and `center_pixel.center_time` (0.00347 s), not from `delta_line_time`. `PASS` is `UNKNOWN` because `state.direction` is absent. `is_terrain_corrected` is 0, so SNAP does not notice that the raster is already on a map grid.

## Stop

```bash
python capella_snap_stage.py \
  --tif <SLC HH tif> \
  --out snap_in/20190822074237_20190822074521 \
  --magpol <matching MAG-POL tif>
```

Exit code 2. It prints `NOT READY` and the missing fields above, plus `image_geometry.type surface` and `terrain_models ABSENT`.

Not done, on purpose:

- no AHN DSM export
- no `gpt snap_layover.xml`
- no flat-height sweep (`--flat-height` at centre-target ±10 m and ±25 m)
- no `compare_snap.py`
- no crops

A layover run on the 99999 m sentinel would not test the near-field labels. The graph and the comparison script are in the repo so the same command can be used on a product that actually has a slant-range axis.

## Code

| file | role |
|---|---|
| [capella_snap_stage.py](capella_snap_stage.py) | Readiness gate. Exits 2 on this strip. |
| [snap_layover.xml](snap_layover.xml) | Subset, SAR-Simulation on the real DSM, Terrain-Correction onto the MAG-POL grid using the flat DEM. Not executed. |
| [compare_snap.py](compare_snap.py) | Would refuse to interpret class agreement if the grids do not match. Not executed. |
