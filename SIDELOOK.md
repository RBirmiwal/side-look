# sidelook v2

How a building is attached to the v2 tiles. The tiles themselves are unchanged: 3,969 chips of SpaceNet 6 Capella MAG-POL, August 2019, 0.25 m, near-field labels. This note is the geometry in the building index that sits on top of them. Download is the GitHub release `sidelook-v2`. The label definition is [REPORT_v2.md](REPORT_v2.md) and [NEAR_FIELD.md](NEAR_FIELD.md).

The index has 7,917 buildings, 16,900 building–strip rows, and 29,500 building–tile rows.

## A building is a footprint and two heights

It is not a mesh, and it is not a copy of the mask.

The polygon is the 3DBAG `pand` footprint (`identificatie`, `NL.IMBAG.Pand.<number>`). It is stored as WKB in EPSG:32631. `centroid_lon` and `centroid_lat` are that polygon’s centroid in WGS84. `area_m2` is its planar area in the UTM metres. A pand with more than one ring is a `MultiPolygon`. QGIS can open the same polygons in `index/sidelook_v2.gpkg` (layer `buildings`, MultiPolygon, EPSG:32631). `building_strips` and `building_tiles` are tables with no geometry.

Roof heights come from 3DBAG `lod22_2d`, in metres NAP:

| column | source |
|---|---|
| `roof_70p_m` | `b3_h_70p`, the 70th percentile of the roof. If a pand has more than one lod row, the maximum. |
| `roof_max_m` | `b3_h_max`, the same way. |
| `ground_m` | Median of the AHN4 DTM (1 m, NAP) on the footprint. If the footprint has no DTM samples, the median of a 20 m ring around it. Null if that also fails. |
| `height_m` | `roof_70p_m − ground_m`. Null when `ground_m` is null. |
| `year_built` | `oorspronkelijkbouwjaar`, when 3DBAG has it. |

`height_m` is exactly that subtraction. It is not `b3_h_nok` (the ridge) and not `b3_h_maaiveld` (3DBAG’s own ground). A known roof, `NL.IMBAG.Pand.0599100000431693`, is 32.77 m this way (roof 70p 36.19 m, DTM 3.42 m). The ridge is about a metre and a half lower in the difference, which is why an earlier note quoted ~31 m.

Every 3DBAG pand whose footprint intersects a v2 tile is included, at any height. A sliver counts: `fraction_inside_tile` goes down to about 10⁻⁶. Buildings that miss every tile are not in the index.

## The mask is not this polygon

The `uint8` mask in each `.npz` is the near-field fold of the AHN DSM, drawn before this index existed. A pixel is class 1 because a DSM cell has a shorter slant range than a cell closer to the aircraft, not because it falls inside the 3DBAG ring. Classes are `0` nominal, `1` active layover, `2` passive layover, `3` shadow. Non-buildings stay nominal. Filling the pand polygon would be a different label (that is the v4 experiment, not these tiles).

So the ring in the index and the copper in the mask are two geometries. Draw the ring on the image if you want the map footprint. Do not read the mask edge as the cadastral wall, and do not move the stored ring by `layover_len_m`. The ring is where the footprint is on the map.

## Ground range of the walls

Each strip that touches the building has one row in `building_strips`: the flight-line look, an incidence, a layover length, a shadow length, and the fraction of the footprint that lands on that strip’s tiles.

Incidence is taken at the footprint centroid. The tile is the one of that strip whose stored footprint contains the centroid. If several do (train tiles overlap by half), it is the one with the largest share of the building. If none does, it is the largest share, and the centroid’s row is brought back onto that tile. On the stored tile, row 0 is the near edge and incidence increases with row:

```
θ = incidence_min + (row / 512) · (incidence_max − incidence_min)
```

`row` is the rasterio row of the centroid through the tile affine. The flip is already in that affine.

Platform height `H` is the strip’s flight height above the scene-centre target, the “height m” column in [REPORT_v2.md](REPORT_v2.md). About 1,611–1,791 m. Ground range of the centroid is `g_c = H · tan θ`.

The aircraft-facing direction depends on the flight-line look, not on the SpaceNet flag. In east and north:

```
away from the aircraft = (sin look, cos look)
toward the aircraft    = (−sin look, −cos look)
```

Look near 180° means the aircraft is north of the scene, so “away” is south. Look near 0° means the aircraft is south, so “away” is north.

For each exterior vertex, `d = (vertex − centroid) · away`. The near wall is the minimum `d` (closest to the aircraft). The far wall is the maximum.

```
g_near = max(1, g_c + min d)
g_far  = max(1, g_c + max d)
```

`h = height_m`. The lengths are the closed forms in [NEAR_FIELD.md](NEAR_FIELD.md), evaluated at those two walls rather than at the centroid:

```
layover_len_m = g_near − √(g_near² − 2·H·h + h²)     toward the aircraft, from the near wall
shadow_len_m  = g_far · h / (H − h)                  past the far wall
```

If the expression under the square root is not positive, the layover length is `g_near`. If `h` is missing or not positive, both lengths are null. Using the centroid instead of the near wall is short by a fraction of a metre on a house and by more on a large footprint. The wall form is what the index stores.

`fraction_inside_strip` is `area(footprint ∩ strip tiles) / area(footprint)`. In the full index the strip is every v2 tile of that collect. The lite release (36 tiles) recomputes this fraction, and the per-tile fractions, against only the tiles in that archive. Incidence and the two lengths are the full-strip values in both.

## Where the footprint sits in a tile

`building_tiles` is one row per tile the footprint meets.

`fraction_inside_tile` is `area(footprint ∩ tile) / area(footprint)`. `fully_inside` is that fraction at least 0.999. A building can be fully inside one tile and still meet the overlapping train tile next to it.

The pixel box is the axis-aligned box of the exterior ring in the stored tile, half-open, rasterio convention (column, row of the pixel corner):

```
col0 = ⌊min col⌋    col1 = ⌈max col⌉
row0 = ⌊min row⌋    row1 = ⌈max row⌉
```

It is allowed to fall outside `0…512`. That is a footprint crossing the tile edge, not a bad affine. The same box in `lay_row0`, `lay_col0`, `lay_row1`, `lay_col1` is taken on the union of two polygons: the footprint, and the footprint translated toward the aircraft by the full `layover_len_m`. That union is the footprint plus the fold in front of it. Only the box is stored.

The tile affine is six numbers, rasterio order `(a, b, c, d, e, f)`, mapping the corner of pixel `(col, row)` to easting and northing in EPSG:32631. The flip is already in it. `strip_col0` and `strip_row0` are the tile’s origin on the strip’s canonical grid, which is what makes two tiles stitch.

## Cutting a chip

`Dataset.chip` does not lean the building. It cuts a square out of the strip and reports the map footprint in that square.

1. The centre starts at the footprint centroid.
2. With `include_layover` (the default), the centre moves toward the aircraft by `layover_len_m / 2`, so the footprint and the fold are pulled into the same window. The polygon itself stays where it was.
3. That point is converted with the strip affine and the window is snapped onto the strip grid: `col0 = round(col − size/2)`, and the same for the row. The snap is at most half a pixel.
4. Tiles of that strip are copied in where their canonical windows overlap the chip. Train tiles overlap, so more than one tile can cover a pixel; the pixels agree. Pixels with no tile are NaN in the image and 255 in the mask, and `partial` is set.

The strip affine is the tile affine shifted back to strip pixel `(0, 0)`:

```
c_s = c − a·strip_col0 − b·strip_row0
f_s = f − d·strip_col0 − e·strip_row0
```

`footprint_px` is the original pand polygon through the chip affine. `plot_chip` draws that ring. The aircraft is row 0 of the chip, because the tile is stored that way. A 25 m bar is 100 pixels.

If you do not name a strip, the one used is the strip whose tile union contains the centroid furthest from the edge. Passing `split` restricts both the strip choice and the tiles that may be stitched.

`to_geotiff` writes the chip or tile north-up. A flipped tile has a positive northing step (`e > 0`, row 0 at the south). The writer reverses those rows and the affine so row 0 is north. The `.npz` on disk is left as stored.

On the full index, a 256-pixel chip with the layover shift falls inside one tile for 3,454 buildings, crosses a tile edge and has to be stitched for 469, runs off the near edge of the strip for 3,977 (`partial`), and has no tile under the window for 17.

## Orientation

Every stored tile has the aircraft at row 0.

A south-looking aircraft (look about 180°) is north of the scene. A north-up image already has it at row 0, so those tiles are not flipped and the northing step is −0.25 m. A north-looking aircraft (look about 0°) is south of the scene. Those tiles are flipped, and the northing step is +0.25 m. `flipped` is that bit. The SpaceNet orientation flag is the wrong way round on these collects: flag 0 looks south, flag 1 looks north.

## What the geometry does not decide

`layover_len_m` is a length on the ground, from the near wall, for a flat roof of height `h` at the centroid’s incidence. It is not a measurement of where the bright pixels are.

The opposite-look test ([OPPOSITE.md](OPPOSITE.md)) did not show those roofs jumping to the leaned outline. [DECIDE.md](DECIDE.md) keeps the roof on the footprint. The chip follows that: the ring is the pand polygon, and the window is merely shifted so the fold in front of the wall is not cut off. The mask is still the DSM fold, so a mask boundary and the ring can disagree by the width of the roof structure the DSM saw and the pand ring did not.

Shadow length is the same kind of number. On the v2 tiles, shadow minus nominal HH is about +0.16 dB. A pixel in class 3 is not reliably dark.

The DTM and the roof heights are lidar from 2020–2022. The radar is August 2019. Containers, ships, and cranes moved. A bright return with no building, or a building with no return, is often that.
