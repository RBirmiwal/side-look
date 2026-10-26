# sidelook v2

Radar tiles of Rotterdam and an index of the buildings in them. The tiles are
the v2 set, unchanged: SpaceNet 6 Capella X-band airborne SAR, August 2019,
0.25 m, four polarisations stored as decibels. The labels were drawn from
AHN4 lidar, 3DBAG footprints and the flight line of each collect. Nothing in
this package rewrites a tile or a label.

## Install

Python 3.10 or newer. The reader needs numpy, pandas, pyarrow, shapely,
pyproj and matplotlib.

```bash
pip install numpy pandas pyarrow shapely pyproj matplotlib
cd sidelook_v2
PYTHONPATH=. python -m sidelook query --min-height 20 --out tall.csv
PYTHONPATH=. pytest
```

## Five lines

```python
import sidelook
ds = sidelook.open("sidelook_v2")
tall = ds.buildings().sort_values("height_m", ascending=False).iloc[0]
chip = ds.chip(tall.building_id, size=256)
ds.plot_chip(chip, out="tallest.png")
```

`examples/quickstart.ipynb` is the same thing. A building id may be the full
3DBAG identificatie (`NL.IMBAG.Pand.…`) or just the numeric suffix.

`buildings(min_height=30)` is empty on the lite sample. The tallest building
in those 36 tiles is 29.8 m. The full set does have buildings over 30 m.

## What you can ask for

| call | returns |
|---|---|
| `ds.buildings(min_height=, max_height=, bbox=, bbox_crs=, strip=, split=)` | a table of buildings |
| `ds.building(building_id)` | one row, with the footprint polygon |
| `ds.near(lon, lat, radius_m=50)` | buildings within the radius, nearest first |
| `ds.tiles_for(building_id, split=, strip=)` | tiles that contain it |
| `ds.tile(tile_id)` | image `(4, 512, 512)` float32 dB, mask, incidence, affine |
| `ds.chip(building_id, size=256, strip=, split=, include_layover=True)` | a smaller window, stitched if it crosses a tile |
| `ds.plot_chip(chip, out=None)` | HH, HH with the mask, HH with the footprint |
| `ds.to_geotiff(chip_or_tile, path)` | a north-up GeoTIFF |

Command line, from this directory:

```bash
python -m sidelook query --min-height 30 --bbox E0 N0 E1 N1 --out buildings.csv
python -m sidelook near --lon 4.48 --lat 51.90 --radius 100
python -m sidelook chip --id <building_id> --size 256 --out chip.npz --png chip.png
python -m sidelook tile --id <building_id> --out-dir tiles_for_building/
```

`--bbox` is east, north, east, north in EPSG:32631 unless you pass
`--bbox-crs`. `tile` copies the original `.npz` and writes a GeoTIFF beside it.

## The image

Each tile is 512 pixels, 128 m, pixel 0.25 m. Bands are HH, HV, VH, VV, in
dB. `incidence` is a separate array, in degrees, and is not a fifth band.
The mask is `uint8`:

| code | class |
|---:|---|
| 0 | nominal |
| 1 | active layover (the roof folded toward the aircraft) |
| 2 | passive layover (ground in front of the wall, in the same fold) |
| 3 | shadow |
| 255 | no data, only on a chip that runs off the strip |

Splits are west-to-east 1 km blocks: train, then val, then test, with no
shared ground between them. Train tiles overlap by half a tile. Val and test
do not. Normalisation constants for the four bands, from the train split
only, are in `index/norm_stats.json`.

## Orientation

Every stored tile has the aircraft at row 0. Layover in the label points
toward that edge.

A south-looking aircraft is north of the scene. A north-up image already has
it at row 0, so those tiles are not flipped. A north-looking aircraft is
south of the scene, so the north-up image has it on the last row. Those
tiles are flipped upside down relative to north, and the affine has a
positive northing step. `flipped` records which is which. `to_geotiff` turns
a flipped tile back so that north is up.

The look in this index is the flight-line azimuth, not the SpaceNet
orientation flag. On these collects the flag is the wrong way round: flag 0
looks south, flag 1 looks north. Do not use the flag.

## The index

Geometries are WKB in EPSG:32631, the same polygons in
`index/sidelook_v2.gpkg` for QGIS. `tiles` and `buildings` are the polygon
layers. `building_strips` and `building_tiles` are tables without geometry.
`centroid_lon` and `centroid_lat` are WGS84.

`tiles`: tile id, split, strip, flight-line look, flipped, the tile origin
in the strip's canonical grid (`strip_row0`, `strip_col0`, for stitching),
size, pixel size, the six-number rasterio affine of the stored tile (flip
already included), footprint, incidence range, layover share (active plus
passive) and shadow share.

`buildings`: 3DBAG pand identificatie, footprint, centroid, area, DTM ground
in metres NAP, roof at `b3_h_70p` and `b3_h_max`, height as roof 70p minus
ground, year built when 3DBAG has it, and how many strips and tiles see it.
Every 3DBAG building that meets a v2 tile is included, whatever its height.

`building_strips`: one row per strip that sees the building. Incidence is
taken at the footprint, from the tile that contains the centroid.
`layover_len_m` is the near-field fold from the wall that faces the
aircraft: ground range from that incidence and the strip's flight height,
then the closed-form fold. `shadow_len_m` is the same geometry off the far
wall. `fraction_inside_strip` is the share of the footprint that lies on
tiles of that strip.

`building_tiles`: fraction of the footprint on that tile, the pixel bounding
box of the footprint (`row0`, `col0`, `row1`, `col1`, half-open, and it may
fall outside 0–512), the same box for the footprint plus the layover shift,
and `fully_inside`.

A chip is centred on the footprint. With layover included it is shifted
toward the aircraft by half `layover_len_m`, so the footprint and the fold
both sit in the window, and then snapped to the strip grid (half a pixel at
most). If the window crosses a tile edge it is stitched from the tiles of
that strip using `strip_row0` and `strip_col0`. Pixels off the strip are NaN
in the image and 255 in the mask, and `partial` is set. If you do not name a
strip, the one where the building sits furthest from the strip edge is used.

## What the labels get wrong

Shadow is not a dark class here. Across the training tiles, shadow minus
nominal HH is about +0.16 dB, and on several strips the shadow class is
brighter than nominal. Do not expect a shadow pixel to look dark.

The lidar is 2020–2022. The radar is August 2019. Containers, ships and
cranes in the port moved. A bright return with no building, or a building
with no return, is often that.

Anything that is not a building is nominal. Cranes, ships, containers, cars
and bridges are not labelled.

Whether a roof return sits on the footprint or leans toward the aircraft is
not settled. The mask follows the footprint. Treat that boundary as the
least certain one in the label.

## Files

```
sidelook_v2/
  README.md
  LICENSES.md
  tiles/{train,val,test}/<tile>.npz
  index/tiles.parquet
  index/buildings.parquet
  index/building_strips.parquet
  index/building_tiles.parquet
  index/sidelook_v2.gpkg
  index/norm_stats.json
  index/classes.json
  sidelook/
  examples/quickstart.ipynb
  examples/overlay_check.png
  tests/
```

`examples/overlay_check.png` is the registration check: chips with the
footprint drawn, including flipped strips and chips that had to be stitched.
The outline is supposed to sit on the building.

## This copy

This is the lite sample: 36 tiles and the buildings that meet them, so the
reader can be tried before the full set. The tallest building here is 29.8 m,
so a 30 m cut returns no rows. Fractions in `building_strips` are
against these tiles, not against the full strips. The full set has 3,969
tiles and 7,917 buildings.

