# Acquisition geometry

Read-only check of the two strips that were tiled. No masks were regenerated. The dataset on disk is unchanged.

The mask generator takes look azimuth from the SpaceNet orientation flag (`0` → 0°, `1` → 180°) and incidence from `collect.image.center_pixel.incidence_angle`. Brightness checks had shadow labels landing on bright wall returns, which suggested the look was 180° out. This note derives both numbers from positions instead of trusting the flag or the field name.

Code: [`sar/acquire_geometry.py`](sar/acquire_geometry.py). A platform due north of a target at 45° elevation must return bearing 0°, look azimuth 180°, incidence 45°. That test is `sar/tests/test_acquire_geometry.py`.

```
bearing_to_sensor = atan2(E, N)                 compass from target toward platform
look_azimuth      = (bearing_to_sensor + 180) mod 360
incidence         = atan2(sqrt(E² + N²), U)     degrees from vertical
```

East, north, and up are the local ENU frame at the target (WGS84).

## Where the metadata lives

There is no separate `*_extended.json` on disk. The Capella product JSON is the GeoTIFF `IMAGEDESCRIPTION` tag.

| | |
|---|---|
| platform state | `collect.state.state_vectors[]` — `time`, ECEF `position` (m), `velocity`. Coordinate system `ecef`, source `real_time`. |
| target | `collect.image.center_pixel.target_position` — one ECEF point. No corner or edge targets. |
| platform name | `collect.platform = capellaspace/metasensing`. Speed is about 76 m/s. This is the aircraft, not an orbiting satellite. |

State-vector counts: 298 on the 4 Aug strip, 328 on the 22 Aug strip. The aircraft moves about 11–12 km during a strip. Spread about the mean position is 5.6 km and 6.2 km (max), 3.3 km and 3.6 km (rms).

## Fields that mention an angle

Recursive walk of both JSON documents. Nothing named grazing, layover, shadow, heading, or look.

| path | `20190804111224_20190804111453` | `20190822074237_20190822074521` |
|---|---:|---:|
| `collect.image.center_pixel.incidence_angle` | 33.6155957963679 | 34.613175285729675 |
| `collect.image.center_pixel.squint_angle` | 0.9856668718125832 | 1.3410269019624224 |
| `collect.radar.pointing` | right | right |
| `collect.image.azimuth_looks` / `range_looks` | 1 / 1 | 1 / 1 |
| `collect.image.azimuth_resolution` | 0.223 m | 0.226 m |
| antenna `azimuth_beamwidth` | 0.0953 rad | 0.0953 rad |
| antenna `elevation_beamwidth` | 0.744 rad | 0.744 rad |

`collect.pointing[]` is an attitude quaternion at each state time, not a look angle. `radar.pointing = right` is the same string on both strips. It is not an azimuth.

## What the positions say

Target = center-pixel ECEF. Primary platform = mean of the state-vector positions. Also reported: the single state vector nearest `center_pixel.center_time`, because incidence is a center-pixel quantity and the track is 11 km long.

| | `20190804111224…` flag 1 | `20190822074237…` flag 0 |
|---|---:|---:|
| azimuth the pipeline uses | 180 | 0 |
| geometric look, mean platform | **359.1366157056177** | **177.77061439867362** |
| geometric look, at center time | 358.1672327794457 | 176.33343857143643 |
| mean look vs nearest of 0°/180° | 0.863° from 0 | 2.229° from 180 |
| center-time look vs nearest of 0°/180° | 1.833° from 0 | **3.667° from 180** |
| `incidence_angle` field | 33.6155957963679 | 34.613175285729675 |
| geometric incidence, mean platform | 34.121634 | 34.610207 |
| geometric incidence, center time | 33.814405 | 34.444641 |
| `90 − incidence_angle` | 56.384404 | 55.386825 |
| track heading (velocity, ENU) | 270.49° | 89.61° |
| right-look from that heading | 0.49° | 179.61° |

Target geodetic positions: 51.88818° N, 4.41904° E, h 44.6 m, and 51.88607° N, 4.42633° E, h 46.0 m.

## Answers

**Look.** The geometric azimuth does not match the pipeline. It is the opposite direction. Separation from the flag is 179.14° and 177.77° using the mean platform. The flags are swapped: the strip labeled south (flag 1) is looking north, and the strip labeled north (flag 0) is looking south.

This confirms the brightness test on `20190822074237_20190822074521`. Forcing look 180 fitted the imagery; the flag said 0. The positions say 177.77° (mean) or 176.33° (center time).

These are not exact cardinal looks. The mean on the 22 Aug strip is 2.23° off 180. The center-time azimuth on that strip is **176.33343857143643°, which is 3.67° off 180**. The 4 Aug strip stays within 2° of 0 (mean 359.14°, center time 358.17°).

Squint in the JSON (0.99° and 1.34°) does not equal that offset. The right-look implied by the velocity heading (0.49° and 179.61°) also disagrees with the position-derived look by about 2°. No field in the file resolves the residual. It is reported, not folded into the azimuth.

**Incidence.** The geometric incidence matches `incidence_angle`, not its complement. The largest gap is 0.51°. The complement is about 21° off. The field is from vertical.

**Opposite looks.** Yes. Mean azimuths differ by 181.37°. Headings are 270.5° and 89.6°, and both collects are right-looking, which is the same fact. The pipeline kept the two strips 180° apart and flipped the absolute sense of both.

## Incidence is not constant across the strip

Capella stores no edge target. Near and far range below are the two ends of the short image axis at mid-azimuth (the range direction: 707 m and 801 m on the ground), converted to ECEF at the center target’s ellipsoid height. Incidence is from the center-time platform.

| strip | near | center field | far | spread |
|---|---:|---:|---:|---:|
| `20190804111224…` | 24.341° | 33.616° | 41.592° | **17.25°** |
| `20190822074237…` | 23.700° | 34.613° | 43.036° | **19.34°** |

One θ for the whole strip is the average of a swath that runs from about 24° to about 42°. The along-track ends of the image, seen from one platform position, are a different number (incidence ~74°) and are not the swath edges.

## What the mask generator can accept

`mask_from_dsm` takes any look azimuth. Within 0.5° of 0/90/180/270 it uses an exact flip or `rot90`. Any other azimuth, including 2° or 3.7° off cardinal, uses a bilinear rotation. The dataset driver does not pass the geometric azimuth. It passes 0 or 180 from the flag, unless `--look-azimuth` is set.
