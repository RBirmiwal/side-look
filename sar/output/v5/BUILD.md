# Rebuilding the v5 labels

v5 is the v4 scene plus a layover magnitude, with 3DBAG footprints dilated 2 m set to 255. The images are the v2 tiles. v2 is not modified.

## Inputs

- AHN4 DSM and DTM, PDOK WCS `https://service.pdok.nl/rws/ahn/wcs/v1_0`, coverages `dsm_05m` and `dtm_05m`, CRS EPSG:28992, 0.5 m. A chip is `GetCoverage` version 1.0.0, `FORMAT=image/tiff`.
- 3DBAG v20250903, `https://data.3dbag.nl/v20250903/tiles/9/{x}/{y}/9-{x}-{y}.gpkg.gz`.
- v2 tiles: image, incidence, look, and the strip grid. Release `tiles-v2`.

## Commands

From the repo, with the v2 tiles at `/tmp/v2tiles`:

```
python3 sar/v5_build.py
python3 sar/v5_finish.py
python3 merge_masks.py --v2 /tmp/v2tiles --pack v5_masks --out /tmp/v5_rebuilt
```

`v5_build.py` takes one strip id as an argument, and `--redo` to rebuild a finished strip. It writes `sar/output/v5/strips/<id>.json`, the mask pack under `v5_masks/`, and full tiles under `side-look-tiles-v5/`. Range pad is 250 m. `nan_margin` is 400 cells at 0.25 m. Pixel size passed to the magnitude is 0.25 m.

The closed-form tests are `python3 sar/tests/test_image_space.py`.
