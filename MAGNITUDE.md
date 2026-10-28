# Layover magnitude

A per-pixel target for how far the echoes in a layover pixel have been moved, in metres. The functions are in `sar/image_space.py`. No v5 tiles were built. The v2 tiles cannot supply this target.

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

## What v5 asked for, and why it was not built

v5 was specified as the v4 labels plus this magnitude, with 3DBAG footprints dilated 2 m set to 255 so training only sees pixels where the leaning rule and the in-place rule agree. Same 15 strips, same buildings-only scene, same AHN4 DTM ground, same tiles and grid as v4. `label_version` would have been `v5-agreed-mag`.

That scene is not on this machine. `sar/data` has the orientation notes and one JSON. There is no AHN DSM, no AHN4 DTM, and no 3DBAG. The v4 tiles were never released (3,042 of them; see [V4.md](V4.md)), so there is no v4 mask to copy and nothing to run the "recomputed class equals the v4 mask" check against. The v2 release is the images and the near-field mask only. v2 and v4 were not modified.

`train.py` can score the extra channel (smooth-L1 on the metres, MAE on true-layover pixels against a constant baseline). It was not run. There is no v5 tile directory, and this machine has no PyTorch.

## The v2 tiles are not a substitute

A v2 `.npz` holds:

| key | what it is |
|---|---|
| `image` | HH, HV, VH, VV, float16 dB, `(4, 512, 512)` |
| `mask` | uint8. `0` nominal, `1` active layover, `2` passive layover, `3` shadow. The near-field fold, already collapsed to a class. |
| `incidence` | float16 degrees, the look angle of the grid. It increases with row. It is not the off-nadir angle of the roof. |
| `look` | 1 if the tile was flipped so the sensor sits at row 0. |
| `meta` | JSON. One `platform_height_m` for the strip, about 1,611–1,791 m, plus the bbox, the flight-line look, and `label_version` `v2-nearfield`. |

There is no height raster. The DSM that produced the mask was not written into the tile. Incidence and `platform_height_m` fix the ground range of a flat surface. They do not fix `r` and `alpha` of a roof. Many height fields map to the same four-class mask, and the displacement is the quantity that was discarded when the mask was written. A ratio near 4 or near 0, which the planned check was meant to catch, cannot even be computed from these arrays.

The building index ([SIDELOOK.md](SIDELOOK.md), release `sidelook-v2`) does not fill the gap. It has a footprint and two heights per pand, and one `layover_len_m` per building per strip. That length is the closed form at the near wall. It is one number, not a raster. Using it as the target would be a different dataset:

- In front of a wide building it is close to the magnitude, because the magnitude is about `L` there.
- In front of a narrow building it is wrong. The magnitude ramps with distance to the wall, then jumps to the roof point. One `L` cannot show that.
- The v2 mask is not the v4 mask. v4 is rule B, buildings only, three classes plus ignore. v2 is the near-field fold, four classes, nothing ignored. The two rules disagree on the footprint, which is why v5 was going to mark footprints 255. Painting `L` onto v2 class 1 and class 2 would put a number on pixels v5 would have left unlabelled, and would miss pixels v5 would have called layover.
- Containers, cranes and ships are in the v2 mask and not in 3DBAG. A rear wall that fills the shadow in front of it is also invisible to a per-building length.
- v2 has no `coverage`. v5 was going to keep the v4 coverage unchanged.

v2 is the right source for the images, the incidence, the flight height, and the grid. It is not a source for the magnitude. The index is the right source for the check that a real magnitude should come out near `L` on wide buildings. It is not the target.
