# Capella Open Data over the Netherlands

SICD products in the Capella Open Data STAC collection `capella-open-data-sicd`, 1,510 items, fetched 2026-11-09 from `s3://capella-open-data` (unsigned) and the catalogue at `https://capella-open-data.s3.us-west-2.amazonaws.com/stac/catalog.json`. Licence CC BY 4.0.

A collect is listed when its SICD footprint polygon intersects the European Netherlands. The outline is the Natural Earth `NLD` polygon shipped in `johan/world.geo.json` (14 vertices). Caribbean Netherlands is not in that outline, and AHN does not cover it. Four further SICDs fall in the same longitude and latitude box and do not intersect the outline: two over Antwerp, one over Brussels, one sliding-spotlight over Belgium.

## The five collects

Sorted by date. Incidence is `view:incidence_angle` at the scene centre. Look side is `sar:observation_direction`. Resolution is azimuth / slant range / ground range, in metres. Products are the sibling folders on the same date that share the collect's time window. The SICD size is the `.ntf` object size.

| date | collect | mode | pol | inc | look | orbit | resolution m | products | SICD | where |
|---|---|---|---|---:|---|---|---|---|---:|---|
| 2023-01-19 | `155a9365-218f-4a65-aa11-48f37a8351ca` | stripmap | HH | 49.9 | left | descending | 1.37 / 0.66 / 0.87 | GEC, GEO, SICD, SLC | 541 MB | Amsterdam |
| 2023-03-08 | `72aaa515-51c3-4322-8c42-fa882321e4a5` | spotlight | HH | 42.8 | right | ascending | 0.05 / 0.32 / 0.47 | CPHD, GEC, GEO, SICD, SIDD, SLC | 4.92 GB | west of Rotterdam, not the skyline |
| 2023-05-25 | `07925c73-c9dc-4013-9119-c452eb91e2b6` | stripmap | HH | 45.5 | left | ascending | 1.21 / 0.66 / 0.93 | GEC, GEO, SICD, SLC | 3.32 GB | Amsterdam, includes Zuidas |
| 2024-11-22 | `3c91dbdd-3404-43d6-ac0b-83269c2eff3b` | spotlight | HH | 43.3 | right | ascending | 0.04 / 0.27 / 0.39 | CPHD, GEC, GEO, SICD, SIDD, SLC | 7.99 GB | Amsterdam centre |
| 2024-11-24 | `46fefdf2-d58c-4a52-943f-491bbf1e8482` | spotlight | HH | 49.3 | right | ascending | 0.04 / 0.27 / 0.35 | GEC, GEO, SICD, SLC | 8.67 GB | The Hague, includes Hoftoren |

Platforms are capella-6, capella-8, capella-6, capella-15, capella-15, in that order. Every polarisation list is HH only. Full columns, including the S3 key, are in [inventory.csv](inventory.csv).

## Totals

5 collects. 1,510 SICD items in the collection, of which 5 meet the outline. 3 spotlight, 2 stripmap. All HH. SICD bytes sum to 25,445,576,072 (23.7 GiB). No Dutch SICD is under 2 GB except the 19 January stripmap (540,913,732 bytes).

There is no Rotterdam spotlight over the tall buildings. The 8 March spotlight's footprint is 4.237–4.320°E, 51.863–51.915°N. De Rotterdam and the Maastoren are near 4.49°E and are outside it.

## US cities

Fewer than 10 Dutch SICDs, so the same 1,510 items were counted over the United States. A footprint counts for a city when its centroid is inside longitude −125 to −66 and latitude 24 to 50, and the city is the most populous GeoNames place (cities with population at least 15,000, feature class P, country US) within 30 km of that centroid. 460 centroids are in that box. 353 fall within 30 km of such a place. 107 do not and are not in the table.

| SICDs | city |
|---:|---|
| 60 | Miami, FL |
| 45 | Long Beach, CA |
| 43 | Charleston, SC |
| 36 | Los Angeles, CA |
| 23 | Savannah, GA |
| 18 | Santa Clarita, CA |
| 15 | El Centro, CA |
| 12 | Lancaster, CA |
| 9 | Eugene, OR |
| 7 | Pasadena, TX |

## Pilot

The published collect is the 24 November 2024 Hague spotlight, `46fefdf2-d58c-4a52-943f-491bbf1e8482`, product `CAPELLA_C15_SP_SICD_HH_20241124220442_20241124220523`. It is the spotlight whose footprint contains Hoftoren and Het Strijkijzer. The release is `capella-pilot`. The SICD folder on S3 contains only the `.ntf`. There is no extended metadata JSON.
