#!/usr/bin/env python3
"""Decide rule A (lean) against rule B (in place) on the real HH image.

Does not modify dataset v2 or image_space.py. Writes only under
sar/output/decide/. Does not build tiles or change any labels that already
exist under rule_check/.

Range axis is image row. Sensor-at-row-0 strips (P) are already canonical.
Strips imaged from the south (K, and the north-bank strip) are flipped so
project_grid sees index 0 nearest the sensor, then flipped back onto the
north-up MAG grid.
"""

from __future__ import annotations

import argparse
import gc
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.windows import Window, transform as window_transform
from shapely.geometry import Polygon
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from acquire_geometry import ecef_to_geodetic, enu_basis  # noqa: E402
from image_space import labels_from_coverage, own_range_grid, project_grid  # noqa: E402
from image_space_run import (  # noqa: E402
    PAD_M,
    PAD_PX,
    load_hgt,
    prepare_ahn,
    sample_srtm,
    upsample,
)
from near_field import (  # noqa: E402
    add_undulation,
    enu_from_map,
    fit_flight_line,
    look_azimuth_from_track,
    perpendicular_geometry,
)
from rule_check import fill_small_holes  # noqa: E402
from shift_check import gpkg_wkb, local_ground_5th  # noqa: E402

GSD = 0.25
OUT = ROOT / "output" / "decide"
BAG = ROOT / "data" / "3dbag"
AHN = ROOT / "data" / "ahn_dsm_1m_aoi.tif"
HGT_PATH = ROOT / "data" / "srtm" / "N51E004.hgt.gz"
GEOID = ROOT / "data" / "nl_nsgi_nlgeo2018.tif"
NAN_MARGIN = 400
CHUNK_COLS = 800
H_EDGE_MIN = 8.0
EDGES = np.linspace(-40.0, 20.0, 3001)  # 0.02 dB

P_ID = "20190822074237_20190822074521"
K_ID = "20190823072627_20190823072925"
N_ID = "20190823073526_20190823073827"
MAGS = {
    "P": ROOT / "data" / f"SN6_AOI_11_Rotterdam_SAR-MAG-POL_{P_ID}.tif",
    "K": Path("/tmp/towers/city.tif"),
    "N": Path("/tmp/decide/north.tif"),
}


def log(msg: str) -> None:
    print(msg, flush=True)


def dump(obj, path: Path) -> None:
    def conv(o):
        if isinstance(o, dict):
            return {str(k): conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        if isinstance(o, (np.floating, float)):
            x = float(o)
            return None if not np.isfinite(x) else x
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return conv(o.tolist())
        return o

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(conv(obj), indent=2))


def med_from(hist: np.ndarray) -> float:
    tot = int(hist.sum())
    if tot < 30:
        return float("nan")
    c = np.cumsum(hist)
    i = int(np.searchsorted(c, tot / 2.0))
    i = min(i, EDGES.size - 2)
    return float(0.5 * (EDGES[i] + EDGES[i + 1]))


def _hist_add(hist, values):
    v = np.asarray(values, np.float32)
    v = v[np.isfinite(v)]
    if v.size:
        hist += np.histogram(v, bins=EDGES)[0]


def flight_of(mag: Path):
    with rasterio.open(mag) as src:
        desc = json.loads(src.tags()["TIFFTAG_IMAGEDESCRIPTION"])
        t = src.transform
        H, W = src.height, src.width
        if src.crs and src.crs.to_epsg() not in (32631, None):
            log(f"  unexpected crs {src.crs}")
    collect = desc["collect"]
    target = np.array(collect["image"]["center_pixel"]["target_position"], np.float64)
    lat, lon, height = ecef_to_geodetic(target)
    basis = enu_basis(lat, lon)
    pos = np.array([sv["position"] for sv in collect["state"]["state_vectors"]], np.float64)
    enu = (basis @ (pos - target).T).T
    fit = fit_flight_line(enu)
    look = look_azimuth_from_track(fit["direction_enu"], collect["radar"]["pointing"])
    flight = {
        "origin_enu": np.asarray(fit["origin_enu"], np.float64),
        "direction_enu": np.asarray(fit["direction_enu"], np.float64),
        "height_m": float(fit["height_above_origin_m"]),
        "look_deg": float(look),
    }
    geo = {"lat": float(lat), "lon": float(lon), "height": float(height)}
    grid = {
        "west": float(t.c), "north": float(t.f), "height": int(H), "width": int(W),
        "transform": t,
        "south": float(t.f + t.e * H), "east": float(t.c + t.a * W),
    }
    log(f"  grid {H} x {W}  west {grid['west']:.1f} north {grid['north']:.1f}  "
        f"look {flight['look_deg']:.3f}°  platform {flight['height_m']:.1f} m")
    return flight, geo, grid


def sensor_at_row0(flight, geo, grid) -> bool:
    """True when off-nadir grows with file row, i.e. the sensor is at row 0."""
    H, W = grid["height"], grid["width"]
    rows = np.array([H // 5, 4 * H // 5])
    xs = np.full(2, grid["west"] + (W // 2 + 0.5) * GSD)
    ys = grid["north"] - (rows + 0.5) * GSD
    z = np.full(2, geo["height"])
    e, n, u = enu_from_map(xs, ys, z, "EPSG:32631", geo["lat"], geo["lon"], geo["height"])
    _, alpha = perpendicular_geometry(e, n, u, flight["origin_enu"], flight["direction_enu"])
    ok = bool(alpha[1] > alpha[0])
    log(f"  alpha row {rows[0]} {np.degrees(alpha[0]):.2f}°  row {rows[1]} {np.degrees(alpha[1]):.2f}°  "
        f"sensor_at_row0 {ok}")
    return ok


def toward_sensor(sens0: bool) -> int:
    """File-row step that walks toward the sensor."""
    return -1 if sens0 else 1


def _erode(poly):
    g = poly if poly.is_valid else poly.buffer(0)
    if g.is_empty:
        return None
    e = g.buffer(-2.0)
    if e.is_empty:
        return None
    if e.geom_type == "GeometryCollection":
        parts = [p for p in e.geoms if p.geom_type in ("Polygon", "MultiPolygon") and not p.is_empty]
        if not parts:
            return None
        e = unary_union(parts)
    if e.geom_type == "MultiPolygon":
        e = max(e.geoms, key=lambda p: p.area)
    if e.geom_type != "Polygon" or e.is_empty:
        return None
    return e


def load_buildings(grid) -> list[dict]:
    from pyproj import Transformer
    from shapely import wkb

    to_utm = Transformer.from_crs("EPSG:28992", "EPSG:32631", always_xy=True).transform
    to_rd = Transformer.from_crs("EPSG:32631", "EPSG:28992", always_xy=True)
    rd = [to_rd.transform(x, y) for x, y in (
        (grid["west"], grid["south"]), (grid["east"], grid["south"]),
        (grid["east"], grid["north"]), (grid["west"], grid["north"]),
    )]
    xmin, xmax = min(p[0] for p in rd) - 30, max(p[0] for p in rd) + 30
    ymin, ymax = min(p[1] for p in rd) - 30, max(p[1] for p in rd) + 30
    found = {}
    q = """SELECT p.identificatie, p.b3_h_nok, p.b3_h_maaiveld, p.geom
           FROM pand p, rtree_pand_geom r
           WHERE p.fid = r.id AND r.maxx >= ? AND r.minx <= ? AND r.maxy >= ? AND r.miny <= ?"""
    t0 = time.time()
    for gpkg in sorted(BAG.glob("*.gpkg")):
        if gpkg.name.endswith(".gz"):
            continue
        con = sqlite3.connect(f"file:{gpkg}?mode=ro", uri=True)
        try:
            rows = con.execute(q, (xmin, xmax, ymin, ymax)).fetchall()
        except Exception as exc:
            log(f"  skip {gpkg.name}: {exc}")
            con.close()
            continue
        con.close()
        for ident, hnok, hground, blob in rows:
            if blob is None or hnok is None or hground is None:
                continue
            try:
                geom = shp_transform_safe(to_utm, gpkg_wkb(blob), wkb)
            except Exception:
                continue
            if geom is None or geom.is_empty or geom.geom_type not in ("Polygon", "MultiPolygon"):
                continue
            if geom.geom_type == "MultiPolygon":
                geom = max(geom.geoms, key=lambda g: g.area)
            if not geom.is_valid:
                geom = geom.buffer(0)
                if geom.geom_type != "Polygon" or geom.is_empty:
                    continue
            h = float(hnok) - float(hground)
            area = float(geom.area)
            if area < 8.0:
                continue
            key = str(ident)
            prev = found.get(key)
            if prev is None or area > prev["area"]:
                found[key] = {"id": key, "h_bag": h, "area": area, "poly": geom}
    buildings = list(found.values())
    log(f"  footprints in bbox: {len(buildings)}  ({time.time()-t0:.1f}s)")
    return buildings


def shp_transform_safe(to_utm, raw, wkb_mod):
    from shapely.ops import transform as shp_transform
    return shp_transform(to_utm, wkb_mod.loads(raw))


def _sample_geoid(lon, lat) -> np.ndarray:
    lon = np.asarray(lon, np.float64).ravel()
    lat = np.asarray(lat, np.float64).ravel()
    with rasterio.open(GEOID) as src:
        return np.array([float(v[0]) for v in src.sample(zip(lon, lat))], np.float64)


def _px_bbox(poly, grid):
    minx, miny, maxx, maxy = poly.bounds
    c0 = int(np.floor((minx - grid["west"]) / GSD))
    c1 = int(np.ceil((maxx - grid["west"]) / GSD))
    r0 = int(np.floor((grid["north"] - maxy) / GSD))
    r1 = int(np.ceil((grid["north"] - miny) / GSD))
    return r0, r1, c0, c1


def _raster_poly(poly, r0, r1, c0, c1, grid):
    H, W = grid["height"], grid["width"]
    rr0, rr1 = max(0, r0), min(H, r1)
    cc0, cc1 = max(0, c0), min(W, c1)
    if rr1 <= rr0 or cc1 <= cc0:
        return None
    wt = from_origin(grid["west"] + cc0 * GSD, grid["north"] - rr0 * GSD, GSD, GSD)
    m = rasterize([(poly, 1)], out_shape=(rr1 - rr0, cc1 - cc0), transform=wt, fill=0, dtype=np.uint8)
    return m, rr0, cc0


def _near_back(mask, r0, sens0):
    """File row of the sensor-facing edge and the far edge. -1 if the column is empty."""
    has = mask.any(0)
    if sens0:
        near = r0 + np.argmax(mask, axis=0)
        back = r0 + (mask.shape[0] - 1 - np.argmax(mask[::-1], axis=0))
    else:
        near = r0 + (mask.shape[0] - 1 - np.argmax(mask[::-1], axis=0))
        back = r0 + np.argmax(mask, axis=0)
    near = near.astype(np.int32)
    back = back.astype(np.int32)
    near[~has] = -1
    back[~has] = -1
    return near, back


def index_walls(buildings, grid, sens0) -> None:
    """Per-column near (sensor-facing) and back file rows. Drops nothing."""
    t0 = time.time()
    n_ok = 0
    for b in buildings:
        b["ero"] = _erode(b["poly"])
        got = _raster_poly(b["poly"], *_px_bbox(b["poly"], grid), grid)
        if got is None:
            b["near"] = None
            continue
        mask, r0, c0 = got
        near, back = _near_back(mask.astype(bool), r0, sens0)
        b["c0"] = int(c0)
        b["r0_box"] = int(r0)
        b["near"] = near
        b["back"] = back
        b["col0"] = int(c0)
        b["col1"] = int(c0 + near.size)
        n_ok += 1
    log(f"  near-wall index {n_ok}/{len(buildings)}  ({time.time()-t0:.1f}s)")


def annotate(buildings, prep_path: Path, grid, flight, geo, h90: bool = True) -> None:
    """s, theta, and (optionally) h90 of nDSM inside the footprint. 1 m prep grid."""
    from pyproj import Transformer

    prep = np.load(prep_path)
    z = prep["z_nap"]
    ground = prep["ground"]
    north, west = float(prep["north"]), float(prep["west"])
    to_ll = Transformer.from_crs("EPSG:32631", "EPSG:4326", always_xy=True)
    hgt = load_hgt(HGT_PATH)
    if h90:
        t0 = time.time()
        for b in buildings:
            poly = b["poly"]
            minx, miny, maxx, maxy = poly.bounds
            c0 = max(0, int(np.floor(minx - west)))
            c1 = min(z.shape[1], int(np.ceil(maxx - west)) + 1)
            r0 = max(0, int(np.floor(north - maxy)))
            r1 = min(z.shape[0], int(np.ceil(north - miny)) + 1)
            b["h90"] = float("nan")
            b["gnd"] = float("nan")
            if c1 - c0 >= 2 and r1 - r0 >= 2:
                wt = from_origin(west + c0, north - r0, 1.0, 1.0)
                sub_z = z[r0:r1, c0:c1]
                sub_g = ground[r0:r1, c0:c1]
                m = rasterize([(poly, 1)], out_shape=sub_z.shape, transform=wt, fill=0, dtype=np.uint8)
                inside = m.astype(bool) & np.isfinite(sub_z) & np.isfinite(sub_g)
                if int(inside.sum()) >= 8:
                    b["h90"] = float(np.percentile(sub_z[inside] - sub_g[inside], 90))
                    b["gnd"] = float(np.median(sub_g[inside]))
        log(f"  h90 {time.time()-t0:.1f}s")
    else:
        for b in buildings:
            b["h90"] = float("nan")
            b["gnd"] = float("nan")
    xs = np.array([b["poly"].centroid.x for b in buildings], np.float64)
    ys = np.array([b["poly"].centroid.y for b in buildings], np.float64)
    rr = np.clip(np.rint(north - ys).astype(int), 0, ground.shape[0] - 1)
    cc = np.clip(np.rint(xs - west).astype(int), 0, ground.shape[1] - 1)
    g_cent = ground[rr, cc].astype(np.float64)
    gnd = np.array([
        b["gnd"] if np.isfinite(b["gnd"]) else g_cent[i] for i, b in enumerate(buildings)
    ], np.float64)
    lon, lat = to_ll.transform(xs, ys)
    lon = np.asarray(lon, np.float64)
    lat = np.asarray(lat, np.float64)
    srtm = np.asarray(sample_srtm(lon.reshape(1, -1), lat.reshape(1, -1), hgt, orthometric=False), np.float64).reshape(-1)
    ngeo = _sample_geoid(lon, lat)
    g_ell = gnd + ngeo
    s = srtm - g_ell
    e, n, u = enu_from_map(xs, ys, g_ell, "EPSG:32631", geo["lat"], geo["lon"], geo["height"])
    slant, alpha = perpendicular_geometry(e, n, u, flight["origin_enu"], flight["direction_enu"])
    theta = np.degrees(alpha)
    for i, b in enumerate(buildings):
        b["s"] = float(s[i])
        b["theta"] = float(theta[i])
        b["gnd_ell"] = float(g_ell[i])
        b["slant"] = float(slant[i])
        b["h"] = float(b["h90"]) if np.isfinite(b["h90"]) else float(b["h_bag"])
        th = np.radians(b["theta"]) if np.isfinite(b["theta"]) else np.nan
        extra = max(0.0, b["s"]) / np.tan(th) if np.isfinite(th) and th > 0.05 else 0.0
        b["clear_m"] = 6.0 + (float(extra) if np.isfinite(extra) else 0.0)
    log(f"  s finite {int(np.isfinite(s).sum())}/{len(buildings)}  "
        f"median s {np.nanmedian(s):.2f} m  theta {np.nanmedian(theta):.1f}°")
    del z, ground, prep


def _open_masks(prep_path: Path):
    from scipy.ndimage import distance_transform_edt

    prep = np.load(prep_path)
    shore = prep["water"].astype(bool)
    # 1 m posting: dist > 30 cells is 30 m from any shore.
    dist = distance_transform_edt(shore)
    open_w = shore & (dist > 30.0)
    del dist
    return shore, open_w, float(prep["north"]), float(prep["west"])


def _sample_mask(mask, rows, cols, n1, w1, north, west):
    y = north - (rows + 0.5) * GSD
    x = west + (cols + 0.5) * GSD
    r = np.rint((n1 - y) - 0.5).astype(np.int32)
    c = np.rint((x - w1) - 0.5).astype(np.int32)
    r = np.clip(r, 0, mask.shape[0] - 1)
    c = np.clip(c, 0, mask.shape[1] - 1)
    return mask[r][:, c]


def part1(name: str, mag: Path, prep_path: Path, a_cov: Path, a_lab: Path, b_cov: Path, b_lab: Path,
          max_chunks: int = 0):
    """Footprint test. One pass. Shore water out of T/Z/L. Halo columns not counted."""
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    flight, geo, grid = flight_of(mag)
    sens0 = sensor_at_row0(flight, geo, grid)
    H, W = grid["height"], grid["width"]
    buildings = load_buildings(grid)
    index_walls(buildings, grid, sens0)
    # h90 for every footprint is only needed for the height bins of buildings
    # that actually contribute T. s and theta are needed up front (clearance).
    annotate(buildings, prep_path, grid, flight, geo, h90=False)
    by_id = {b["id"]: b for b in buildings}
    shore_1m, open_1m, n1, w1 = _open_masks(prep_path)
    log(f"  shore {shore_1m.mean():.3%}  open water {open_1m.mean():.3%}")

    thirds = {"near": (0, H // 3), "middle": (H // 3, 2 * H // 3), "far": (2 * H // 3, H)}
    keys = ("T", "Z", "W", "L")
    tnames = list(thirds) + ["all"]
    hists = {k: {t: np.zeros(EDGES.size - 1, np.int64) for t in tnames} for k in keys}
    bh: dict[str, np.ndarray] = {}

    def canon_of(file_rows):
        return file_rows if sens0 else (H - 1) - file_rows

    # column buckets so a chunk does not scan every building
    buckets: dict[int, list] = {}
    for b in buildings:
        if b.get("near") is None:
            continue
        k0 = max(0, b["col0"] // 512)
        k1 = max(0, (b["col1"] - 1) // 512)
        for k in range(k0, k1 + 1):
            buckets.setdefault(k, []).append(b)

    t0 = time.time()
    n_chunk = 0
    with rasterio.open(mag) as mag_ds, rasterio.open(a_cov) as ac, rasterio.open(a_lab) as al, \
            rasterio.open(b_cov) as bc, rasterio.open(b_lab) as bl:
        if (ac.height, ac.width) != (H, W):
            raise RuntimeError(f"A coverage shape {(ac.height, ac.width)} != MAG {(H, W)}")
        for col0 in range(0, W, 1024):
            if max_chunks and n_chunk >= max_chunks:
                break
            n_chunk += 1
            cw = min(1024, W - col0)
            halo = 12
            c_lo = max(0, col0 - halo)
            c_hi = min(W, col0 + cw + halo)
            ww = c_hi - c_lo
            win = Window(c_lo, 0, ww, H)
            hh = mag_ds.read(1, window=win).astype(np.float32)
            A = ac.read(1, window=win).astype(np.float32)
            B = bc.read(1, window=win).astype(np.float32)
            Al = al.read(1, window=win)
            Bl = bl.read(1, window=win)
            rows = np.arange(H)
            cols = np.arange(c_lo, c_hi)
            shore = _sample_mask(shore_1m, rows, cols, n1, w1, grid["north"], grid["west"])
            opened = _sample_mask(open_1m, rows, cols, n1, w1, grid["north"], grid["west"])
            known = (Al != 255) & (Bl != 255) & np.isfinite(A) & np.isfinite(B) & np.isfinite(hh)
            wt = window_transform(win, grid["transform"])
            k0, k1 = c_lo // 512, (c_hi - 1) // 512
            overlapping = []
            seen = set()
            for k in range(k0, k1 + 1):
                for b in buckets.get(k, ()):
                    if b["id"] in seen:
                        continue
                    if b["col1"] > c_lo and b["col0"] < c_hi:
                        seen.add(b["id"])
                        overlapping.append(b)
            shapes = [(b["poly"], 1) for b in overlapping]
            fp = rasterize(shapes, out_shape=(H, ww), transform=wt, fill=0, dtype=np.uint8) if shapes else np.zeros((H, ww), np.uint8)
            outside = fp == 0
            noecho = outside & (A < 0.5) & (B < 0.5) & known & ~shore
            from scipy.ndimage import distance_transform_edt
            zin = distance_transform_edt(noecho) >= 8.0
            nominal = (A >= 0.5) & (A < 1.5) & (B >= 0.5) & (B < 1.5) & known & outside & ~shore
            wmask = opened & np.isfinite(hh) & outside
            sl = slice(col0 - c_lo, col0 - c_lo + cw)
            crow = canon_of(rows)

            def acc(mask, key, _hh=hh, _crow=crow, _sl=sl):
                sub_h = _hh[:, _sl]
                sub_m = mask[:, _sl]
                _hist_add(hists[key]["all"], sub_h[sub_m])
                for tname, (lo, hi) in thirds.items():
                    sel = (_crow >= lo) & (_crow < hi)
                    _hist_add(hists[key][tname], sub_h[sel][sub_m[sel]])

            acc(zin, "Z")
            acc(wmask, "W")
            acc(nominal, "L")

            # T. Zero the halo before either histogram so a building that
            # straddles a chunk boundary is counted once.
            agree = (A < 0.5) & (B >= 0.5) & known & ~shore
            agree[:, :sl.start] = False
            agree[:, sl.stop:] = False
            clear_default = 6.0 / GSD
            for b in overlapping:
                if b.get("ero") is None or b.get("near") is None:
                    continue
                if not np.isfinite(b["theta"]) or not np.isfinite(b["s"]):
                    continue
                lc0 = max(c_lo, b["col0"])
                lc1 = min(c_hi, b["col1"])
                # core columns only
                lc0 = max(lc0, col0)
                lc1 = min(lc1, col0 + cw)
                if lc1 <= lc0:
                    continue
                near_abs = b["near"][lc0 - b["col0"]: lc1 - b["col0"]]
                if not np.any(near_abs >= 0):
                    continue
                eb = _px_bbox(b["ero"], grid)
                got = _raster_poly(b["ero"], eb[0], eb[1], lc0, lc1, grid)
                if got is None:
                    continue
                emo, er0, ec0 = got
                emo = emo.astype(bool)
                if ec0 != lc0 or emo.shape[1] != lc1 - lc0:
                    # bbox clip at the image edge
                    pass
                rh, rw = emo.shape
                file_rows = np.arange(er0, er0 + rh)
                near_c = np.where(near_abs >= 0, canon_of(near_abs), -10_000)
                row_c = canon_of(file_rows)
                # near array covers lc0:lc1; emo columns start at ec0
                if ec0 != lc0:
                    continue
                row_ok = row_c[:, None] >= (near_c[None, :] + b["clear_m"] / GSD)
                sel = emo & row_ok & (near_abs[None, :] >= 0)
                sub = agree[er0:er0 + rh, lc0 - c_lo:lc0 - c_lo + rw]
                if sub.shape != sel.shape:
                    continue
                take = sel & sub
                if not take.any():
                    continue
                vals = hh[er0:er0 + rh, lc0 - c_lo:lc0 - c_lo + rw][take]
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    continue
                hist = bh.get(b["id"])
                if hist is None:
                    hist = np.zeros(EDGES.size - 1, np.int32)
                    bh[b["id"]] = hist
                hist += np.histogram(vals, bins=EDGES)[0].astype(np.int32)
                # global T, split by the canonical third of each pixel
                rr, cc = np.nonzero(take)
                abs_r = er0 + rr
                pix = hh[er0:er0 + rh, lc0 - c_lo:lc0 - c_lo + rw][take]
                cr = crow[abs_r]
                _hist_add(hists["T"]["all"], pix)
                for tname, (lo, hi) in thirds.items():
                    m = (cr >= lo) & (cr < hi)
                    if m.any():
                        _hist_add(hists["T"][tname], pix[m])
            if col0 % 8192 == 0:
                log(f"  col {col0}/{W}  {time.time()-t0:.0f}s  buildings with T {len(bh)}")
            del hh, A, B, Al, Bl, fp, zin, agree
            gc.collect()

    def pack(key):
        row = {}
        for tname, hist in hists[key].items():
            row[tname] = {"median_db": med_from(hist), "n": int(hist.sum())}
        return row

    table = {k: pack(k) for k in keys}
    for key, row in table.items():
        for tname, cell in row.items():
            base = table["L"][tname]["median_db"]
            md = cell["median_db"]
            cell["minus_L"] = (md - base) if md is not None and base is not None and np.isfinite(md) and np.isfinite(base) else float("nan")

    Lmed = table["L"]["all"]["median_db"]
    Zmed = table["Z"]["all"]["median_db"]
    Tmed = table["T"]["all"]["median_db"]
    Wmed = table["W"]["all"]["median_db"]
    # h90 only for buildings we will report
    need = [by_id[i] for i, h in bh.items() if int(h.sum()) >= 200 and i in by_id]
    if need and not max_chunks:
        _h90_some(need, prep_path)
        for b in need:
            b["h"] = float(b["h90"]) if np.isfinite(b["h90"]) else float(b["h_bag"])

    per = []
    t_index = []
    for bid, hist in bh.items():
        n = int(hist.sum())
        b = by_id.get(bid)
        if b is None:
            continue
        rec = {
            "id": bid, "n": n, "T_db": med_from(hist) if n >= 30 else float("nan"),
            "h": b["h"], "h_bag": b["h_bag"], "h90": b.get("h90", float("nan")),
            "theta": b["theta"], "s": b["s"],
        }
        if np.isfinite(rec["T_db"]) and np.isfinite(Lmed):
            rec["T_minus_L"] = rec["T_db"] - Lmed
        else:
            rec["T_minus_L"] = float("nan")
        t_index.append({"id": bid, "n": n, "T_db": rec["T_db"], "T_minus_L": rec["T_minus_L"],
                        "h": b["h"], "h_bag": b["h_bag"], "theta": b["theta"], "s": b["s"]})
        if n >= 200 and np.isfinite(rec["T_minus_L"]):
            per.append(rec)

    zl = (Zmed - Lmed) if np.isfinite(Zmed) and np.isfinite(Lmed) else float("nan")
    tl = (Tmed - Lmed) if np.isfinite(Tmed) and np.isfinite(Lmed) else float("nan")
    tz = (Tmed - Zmed) if np.isfinite(Tmed) and np.isfinite(Zmed) else float("nan")

    def frac_closer(records):
        if not records or not np.isfinite(zl):
            return {"n": 0, "closer_to_Z": None, "closer_to_0": None, "median_T_minus_L": None}
        v = np.array([r["T_minus_L"] for r in records], float)
        dz = np.abs(v - zl)
        d0 = np.abs(v - 0.0)
        return {
            "n": int(v.size),
            "closer_to_Z": float(np.mean(dz < d0)),
            "closer_to_0": float(np.mean(d0 < dz)),
            "tie": float(np.mean(dz == d0)),
            "median_T_minus_L": float(np.median(v)),
        }

    def in_bin(p, lo, hi):
        return np.isfinite(p["h"]) and lo <= p["h"] < hi

    bins = {
        "all": frac_closer(per),
        "h_10_20": frac_closer([p for p in per if in_bin(p, 10, 20)]),
        "h_20_40": frac_closer([p for p in per if in_bin(p, 20, 40)]),
        "h_gt_40": frac_closer([p for p in per if np.isfinite(p["h"]) and p["h"] >= 40]),
    }
    if np.isfinite(zl) and zl <= -1.5 and np.isfinite(tz) and abs(tz) <= 0.5:
        verdict = "LEAN"
    elif np.isfinite(zl) and zl <= -1.5 and np.isfinite(tl) and tl >= -0.5:
        verdict = "IN_PLACE"
    elif np.isfinite(zl) and zl > -1.5:
        verdict = "RAISED_NO_ECHO"
    else:
        verdict = "BETWEEN"
    result = {
        "strip": name,
        "sensor_at_row0": sens0,
        "look_deg": flight["look_deg"],
        "partial_chunks": int(max_chunks),
        "table": table,
        "Z_minus_L": zl,
        "T_minus_L": tl,
        "T_minus_Z": tz,
        "W_minus_L": (Wmed - Lmed) if np.isfinite(Wmed) and np.isfinite(Lmed) else float("nan"),
        "buildings_with_any_T": len(bh),
        "buildings_with_200_T": len(per),
        "bins": bins,
        "verdict_rule": verdict,
        "definitions": {
            "T": "eroded 2 m footprint, >= (6 m + max(0,s)/tan theta) behind the near wall, A<0.5, B>=0.5, known, finite HH, not shore water",
            "Z": "outside footprints, A<0.5 and B<0.5 for every pixel within 2 m, known, finite HH, not shore water",
            "W": "open water >30 m from shore at 1 m, finite HH, outside footprints",
            "L": "both coverages nominal, outside footprints, known, finite HH, not shore water",
            "thirds": "canonical range, sensor at index 0",
        },
    }
    dump(result, out / "part1.json")
    dump({"buildings": sorted(t_index, key=lambda r: -r["n"])}, out / "t_index.json")
    _plot_hist(out / "part1_hist.png", per, zl, name, verdict, bins)
    if not max_chunks:
        ranked = sorted(bh.items(), key=lambda kv: -int(kv[1].sum()))[:8]
        _crops_part1(out, mag, grid, sens0, by_id, ranked, Lmed, shore_1m, n1, w1, a_cov, a_lab, b_cov, b_lab, buildings)
    log(f"  {name} {verdict}  T-L {tl:+.2f}  Z-L {zl:+.2f}  T-Z {tz:+.2f}  "
        f"nT {table['T']['all']['n']}  buildings>200 {len(per)}")
    log(f"  wrote {out / 'part1.json'}  ({time.time()-t0:.0f}s)")
    return result


def _h90_some(buildings, prep_path: Path) -> None:
    prep = np.load(prep_path)
    z = prep["z_nap"]
    ground = prep["ground"]
    north, west = float(prep["north"]), float(prep["west"])
    for b in buildings:
        poly = b["poly"]
        minx, miny, maxx, maxy = poly.bounds
        c0 = max(0, int(np.floor(minx - west)))
        c1 = min(z.shape[1], int(np.ceil(maxx - west)) + 1)
        r0 = max(0, int(np.floor(north - maxy)))
        r1 = min(z.shape[0], int(np.ceil(north - miny)) + 1)
        b["h90"] = float("nan")
        if c1 - c0 < 2 or r1 - r0 < 2:
            continue
        wt = from_origin(west + c0, north - r0, 1.0, 1.0)
        sub_z = z[r0:r1, c0:c1]
        sub_g = ground[r0:r1, c0:c1]
        m = rasterize([(poly, 1)], out_shape=sub_z.shape, transform=wt, fill=0, dtype=np.uint8)
        inside = m.astype(bool) & np.isfinite(sub_z) & np.isfinite(sub_g)
        if int(inside.sum()) >= 8:
            b["h90"] = float(np.percentile((sub_z - sub_g)[inside], 90))
    log(f"  h90 on {len(buildings)} buildings with >=200 T")


def _plot_hist(path, per, zl, name, verdict, bins):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    if per:
        vals = np.array([p["T_minus_L"] for p in per], float)
        ax.hist(vals, bins=40, color="0.35", alpha=0.9)
        ax.axvline(0.0, color="0.1", lw=1.0, label="0 (same as land)")
        if np.isfinite(zl):
            ax.axvline(zl, color="C0", lw=1.2, label=f"Z − L = {zl:.2f} dB")
        closer = bins["all"].get("closer_to_Z")
        ax.set_xlabel("per-building median T − L (dB)")
        ax.set_ylabel("buildings (≥ 200 T pixels)")
        ax.set_title(f"{name}  {verdict}   closer to Z−L than to 0: {closer}")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _crops_part1(out, mag, grid, sens0, by_id, ranked, Lmed, shore_1m, n1, w1, a_cov, a_lab, b_cov, b_lab, buildings):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import distance_transform_edt

    H = grid["height"]
    crops = []
    with rasterio.open(mag) as mag_ds, rasterio.open(a_cov) as ac, rasterio.open(a_lab) as al, \
            rasterio.open(b_cov) as bc, rasterio.open(b_lab) as bl:
        for bid, hist in ranked:
            b = by_id[bid]
            minx, miny, maxx, maxy = b["poly"].bounds
            pad = 40.0
            c0 = max(0, int((minx - pad - grid["west"]) / GSD))
            c1 = min(grid["width"], int((maxx + pad - grid["west"]) / GSD) + 1)
            r0 = max(0, int((grid["north"] - (maxy + pad)) / GSD))
            r1 = min(H, int((grid["north"] - (miny - pad)) / GSD) + 1)
            if c1 - c0 < 16 or r1 - r0 < 16:
                continue
            win = Window(c0, r0, c1 - c0, r1 - r0)
            hh = mag_ds.read(1, window=win).astype(np.float32)
            A = ac.read(1, window=win)
            B = bc.read(1, window=win)
            Al = al.read(1, window=win)
            Bl = bl.read(1, window=win)
            rows = np.arange(r0, r1)
            cols = np.arange(c0, c1)
            shore = _sample_mask(shore_1m, rows, cols, n1, w1, grid["north"], grid["west"])
            wt = window_transform(win, grid["transform"])
            fp = rasterize([(b["poly"], 1)], out_shape=hh.shape, transform=wt, fill=0, dtype=np.uint8)
            ero = rasterize([(b["ero"], 1)], out_shape=hh.shape, transform=wt, fill=0, dtype=np.uint8) if b["ero"] is not None else np.zeros(hh.shape, np.uint8)
            if fp.any() and b.get("near") is not None:
                loc_c0 = max(c0, b["col0"])
                loc_c1 = min(c1, b["col1"])
                tmask = np.zeros(hh.shape, bool)
                if loc_c1 > loc_c0:
                    near_abs = b["near"][loc_c0 - b["col0"]: loc_c1 - b["col0"]]
                    file_rows = np.arange(r0, r1)
                    row_c = file_rows if sens0 else (H - 1) - file_rows
                    near_c = np.where(
                        near_abs >= 0,
                        near_abs if sens0 else (H - 1) - near_abs,
                        -10_000,
                    )
                    # near_c is per column; broadcast against rows
                    row_ok = row_c[:, None] >= (near_c[None, :] + b["clear_m"] / GSD)
                    slc = slice(loc_c0 - c0, loc_c1 - c0)
                    has = near_abs >= 0
                    tmask[:, slc] = ero[:, slc].astype(bool) & row_ok & has[None, :]
            else:
                tmask = np.zeros(hh.shape, bool)
            known = (Al != 255) & (Bl != 255) & np.isfinite(A) & np.isfinite(B) & np.isfinite(hh)
            tmask &= (A < 0.5) & (B >= 0.5) & known & ~shore
            shapes = [
                (bb["poly"], 1) for bb in buildings
                if bb.get("near") is not None and bb["col1"] > c0 and bb["col0"] < c1
            ]
            fp_all = rasterize(shapes, out_shape=hh.shape, transform=wt, fill=0, dtype=np.uint8) if shapes else np.zeros(hh.shape, np.uint8)
            noecho = (fp_all == 0) & (A < 0.5) & (B < 0.5) & known & ~shore
            zmask = distance_transform_edt(noecho) >= 8.0
            show, tm, zm, fp_s = hh, tmask, zmask, fp
            if not sens0:
                show, tm, zm, fp_s = show[::-1], tm[::-1], zm[::-1], fp_s[::-1]
            fig, ax = plt.subplots(figsize=(5.2, 5.6))
            v = show[np.isfinite(show)]
            lo, hi = (np.percentile(v, [2, 98]) if v.size else (0, 1))
            ax.imshow(show, cmap="gray", vmin=lo, vmax=hi, origin="upper")
            if fp_s.any():
                ax.contour(fp_s.astype(float), levels=[0.5], colors="white", linewidths=0.8)
            if tm.any():
                ax.contour(tm.astype(float), levels=[0.5], colors="magenta", linewidths=0.7)
            if zm.any():
                ax.contour(zm.astype(float), levels=[0.5], colors="deepskyblue", linewidths=0.4)
            dT = med_from(hist) - Lmed if np.isfinite(Lmed) else float("nan")
            ax.set_title(f"{bid}\nh {b['h']:.0f} m  θ {b['theta']:.0f}°  T−L {dT:+.2f} dB  n {int(hist.sum())}", fontsize=8)
            ax.text(0.02, 0.98, "sensor", transform=ax.transAxes, color="white", fontsize=7, va="top")
            ax.axis("off")
            fig.tight_layout()
            path = out / f"crop_{bid}.png"
            fig.savefig(path, dpi=110)
            plt.close(fig)
            crops.append({"id": bid, "file": path.name, "T_minus_L": dT, "n": int(hist.sum()),
                          "h": b["h"], "h_bag": b["h_bag"], "theta": b["theta"]})
            log(f"    crop {path.name}  T-L {dT:+.2f}  n {int(hist.sum())}")
    dump(crops, out / "crops.json")


# ---------------------------------------------------------------------------
# prep + project (K, and a partial north strip for part 1b)
# ---------------------------------------------------------------------------

def prepare_ahn_fast(west: float, south: float, east: float, north: float):
    """prepare_ahn, but water components are selected with bincount.

    The stock loop is O(components × pixels). A strip with a ragged AHN edge
    has enough nodata components to stall for hours.
    """
    from scipy.ndimage import binary_dilation, label
    from rasterio.warp import Resampling, reproject
    from rasterio.windows import from_bounds

    from image_space_run import AHN as AHN_PATH, WATER_MIN_M2

    height = int(np.ceil((north - south) / 1.0)) + 2
    width = int(np.ceil((east - west) / 1.0)) + 2
    transform = from_origin(west, north, 1.0, 1.0)
    dest = np.full((height, width), np.nan, dtype=np.float32)
    valid = np.zeros((height, width), dtype=np.uint8)
    with rasterio.open(AHN_PATH) as ahn:
        from pyproj import Transformer

        to_ahn = Transformer.from_crs("EPSG:32631", ahn.crs, always_xy=True)
        corners = [to_ahn.transform(x, y) for x, y in (
            (west - 3, south - 3), (east + 3, south - 3), (east + 3, north + 3), (west - 3, north + 3)
        )]
        ax = [c[0] for c in corners]
        ay = [c[1] for c in corners]
        win = from_bounds(min(ax), min(ay), max(ax), max(ay), transform=ahn.transform)
        win = win.round_offsets().round_lengths().intersection(Window(0, 0, ahn.width, ahn.height))
        z = ahn.read(1, window=win).astype(np.float32)
        src_t = rasterio.windows.transform(win, ahn.transform)
        ok = np.isfinite(z) & (np.abs(z) < 1.0e5)
        src = np.where(ok, z, np.nan).astype(np.float32)
        reproject(src, dest, src_transform=src_t, src_crs=ahn.crs, src_nodata=np.nan,
                  dst_transform=transform, dst_crs="EPSG:32631", dst_nodata=np.nan,
                  resampling=Resampling.bilinear)
        reproject(ok.astype(np.uint8), valid, src_transform=src_t, src_crs=ahn.crs,
                  dst_transform=transform, dst_crs="EPSG:32631", resampling=Resampling.nearest)
        log(f"  AHN window {z.shape} -> {dest.shape}")
        del z, src, ok
    nodata = valid == 0
    comp, ncomp = label(nodata)
    counts = np.bincount(comp.ravel(), minlength=ncomp + 1)
    big = np.flatnonzero(counts >= WATER_MIN_M2)
    big = big[big != 0]
    water = np.isin(comp, big)
    log(f"  nodata components {ncomp}  water components {big.size}")
    del comp
    edge = binary_dilation(water, iterations=2) & ~water & (valid == 1)
    border = dest[edge]
    border = border[np.isfinite(border)]
    low = border[border < 2.0]
    if low.size > 100:
        level = float(np.median(low))
        how = "median of valid pixels within 2 m of water and below 2 m NAP"
    elif border.size:
        level = float(np.percentile(border, 10))
        how = "10th percentile of the water edge (nothing below 2 m NAP)"
    else:
        level = 0.0
        how = "fallback 0 m NAP; no water edge"
    filled = dest.copy()
    filled[water] = level
    filled[nodata & ~water] = np.nan
    log(f"  water level {level:.3f} m NAP ({how})")
    log(f"  water {water.mean():.3%} of the 1 m grid, other nodata {(nodata & ~water).mean():.3%}")
    return filled, water, transform, level


def stage_prep(name: str, mag: Path) -> Path:
    """Same surface prep as rule_check: pad 200 m in range, hole fill, water, ground."""
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "prep.npz"
    if dest.exists() and (out / "prep_meta.json").exists():
        log(f"prep exists {dest}")
        return dest
    flight, geo, g = flight_of(mag)
    log(f"AHN 1 m, padded {PAD_M:.0f} m in range, water fill")
    z, water, t1, level = prepare_ahn_fast(g["west"], g["south"] - PAD_M, g["east"], g["north"] + PAD_M)
    log(f"  1 m grid {z.shape}  finite {np.isfinite(z).mean():.3%}")
    z, holes = fill_small_holes(z, water)
    log("local ground, 5th percentile within 50 m")
    ground = local_ground_5th(z, 50)
    miss = ~np.isfinite(ground) & np.isfinite(z)
    ground = ground.copy()
    ground[miss] = z[miss]
    log(f"  ground finite {np.isfinite(ground).mean():.3%}")
    fp = _load_footprints_1m(z.shape, t1)
    from scipy.ndimage import binary_dilation

    fp_dil = binary_dilation(fp.astype(bool), iterations=1)
    z_bldg = ground.copy()
    use = fp_dil & np.isfinite(z)
    z_bldg[use] = z[use]
    z_bldg[water & ~fp_dil] = level
    ndsm = (z - ground).astype(np.float32)
    clutter_core = (ndsm > 3.0) & ~fp.astype(bool) & np.isfinite(ndsm)
    log(f"  footprints {fp.mean():.3%}  clutter core {clutter_core.mean():.3%}")
    log("NLGEO2018")
    z_ell = add_undulation(np.where(np.isfinite(z), z, 0.0).astype(np.float64), t1, "EPSG:32631")
    z_ell = z_ell.astype(np.float32)
    z_ell[~np.isfinite(z)] = np.nan
    zb_ell = add_undulation(np.where(np.isfinite(z_bldg), z_bldg, 0.0).astype(np.float64), t1, "EPSG:32631")
    zb_ell = zb_ell.astype(np.float32)
    zb_ell[~np.isfinite(z_bldg)] = np.nan
    np.savez_compressed(
        dest,
        z_ell=z_ell, zb_ell=zb_ell,
        water=water.astype(np.uint8), fp=fp.astype(np.uint8),
        clutter_core=clutter_core.astype(np.uint8),
        ground=ground.astype(np.float32), z_nap=z.astype(np.float32),
        level=np.float32(level),
        west=np.float64(g["west"]), north=np.float64(float(t1.f)),
    )
    dump({
        "water_level_nap_m": level, "holes": holes, "shape_1m": list(z.shape),
        "grid_north": float(t1.f), "grid_west": float(g["west"]),
        "fp_fraction": float(fp.mean()), "clutter_core_fraction": float(clutter_core.mean()),
        "finite_dsm": float(np.isfinite(z).mean()),
    }, out / "prep_meta.json")
    log(f"wrote {dest}")
    return dest


def _load_footprints_1m(shape, transform):
    from pyproj import Transformer
    from shapely import wkb
    from shapely.ops import transform as shp_transform

    to_utm = Transformer.from_crs("EPSG:28992", "EPSG:32631", always_xy=True).transform
    to_rd = Transformer.from_crs("EPSG:32631", "EPSG:28992", always_xy=True)
    h, w = shape
    west, north = float(transform.c), float(transform.f)
    south = north - h
    east = west + w
    rd = [to_rd.transform(x, y) for x, y in ((west, south), (east, south), (east, north), (west, north))]
    xmin, xmax = min(p[0] for p in rd) - 40, max(p[0] for p in rd) + 40
    ymin, ymax = min(p[1] for p in rd) - 40, max(p[1] for p in rd) + 40
    shapes = []
    q = """SELECT p.geom FROM pand p, rtree_pand_geom r
           WHERE p.fid = r.id AND r.maxx >= ? AND r.minx <= ? AND r.maxy >= ? AND r.miny <= ?"""
    for gpkg in sorted(BAG.glob("*.gpkg")):
        if gpkg.name.endswith(".gz"):
            continue
        con = sqlite3.connect(f"file:{gpkg}?mode=ro", uri=True)
        try:
            cur = con.execute(q, (xmin, xmax, ymin, ymax))
        except Exception:
            con.close()
            continue
        for (blob,) in cur:
            if blob is None:
                continue
            try:
                geom = shp_transform(to_utm, wkb.loads(gpkg_wkb(blob)))
            except Exception:
                continue
            if not geom.is_empty:
                shapes.append((geom, 1))
        con.close()
    log(f"  rasterizing {len(shapes)} footprints on {h} x {w}")
    if not shapes:
        return np.zeros(shape, np.uint8)
    return rasterize(shapes, out_shape=shape, transform=transform, fill=0, dtype=np.uint8, all_touched=True)


def _open_pair(path_lab, path_cov, transform, height, width, resume: bool):
    profile = dict(
        driver="GTiff", height=height, width=width, count=1, crs="EPSG:32631",
        transform=transform, compress="deflate", tiled=True, blockxsize=512, blockysize=512,
    )
    mode = "r+" if resume else "w"
    lab = rasterio.open(path_lab, mode, dtype="uint8", nodata=255, predictor=2, **profile)
    cov = rasterio.open(path_cov, mode, dtype="float32", nodata=np.nan, predictor=3, **profile)
    return lab, cov


def stage_project(name: str, mag: Path, prep_path: Path, products=("A_full", "B_full", "B_bldg"),
                  limit_chunks: int = 0, col_ranges=None) -> None:
    """Project onto the MAG grid. Flips the range axis when the sensor is south."""
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    log(f"project {name} {products}")
    prep = np.load(prep_path)
    z_ell = np.asarray(prep["z_ell"])
    zb_ell = np.asarray(prep["zb_ell"]) if "B_bldg" in products else None
    flight, geo, g = flight_of(mag)
    sens0 = sensor_at_row0(flight, geo, g)
    full_h = g["height"] + 2 * PAD_PX
    progress_path = out / "chunks_done.json"
    # A partial col_ranges run is not the full-strip resume file.
    if col_ranges is None:
        done = set(json.loads(progress_path.read_text())) if progress_path.exists() else set()
        resume = bool(done)
    else:
        done = set()
        resume = (out / f"{products[0]}_cov.tif").exists()
    datasets = {}
    for pname in products:
        lab_p, cov_p = out / f"{pname}_lab.tif", out / f"{pname}_cov.tif"
        datasets[pname] = _open_pair(lab_p, cov_p, g["transform"], g["height"], g["width"], resume and lab_p.exists())
    unk_counts = {n: 0 for n in products}
    unk_path = out / "unk_counts.json"
    if unk_path.exists() and resume:
        unk_counts.update(json.loads(unk_path.read_text()))

    from pyproj import Transformer
    to_ll = Transformer.from_crs("EPSG:32631", "EPSG:4326", always_xy=True)
    hgt = load_hgt(HGT_PATH)
    north = float(prep["north"])
    t0 = time.time()
    n_done = 0
    col_iter = list(range(0, g["width"], CHUNK_COLS))
    if col_ranges:
        col_iter = []
        for a, b in col_ranges:
            a = max(0, int(a))
            b = min(g["width"], int(b))
            col_iter.extend(range(a, b, CHUNK_COLS))
    for col0 in col_iter:
        if col0 in done:
            continue
        if limit_chunks and n_done >= limit_chunks:
            break
        cw = min(CHUNK_COLS, g["width"] - col0)
        t1 = time.time()
        log(f"chunk {col0}:{col0+cw}")
        z_full = upsample(z_ell, 0, col0, full_h, cw, order=1)
        z_bldg = upsample(zb_ell, 0, col0, full_h, cw, order=1) if zb_ell is not None else None
        cols = np.arange(col0, col0 + cw)
        xs = np.broadcast_to(g["west"] + (cols[None, :] + 0.5) * GSD, (full_h, cw)).copy()
        ys = np.broadcast_to(north - (np.arange(full_h)[:, None] + 0.5) * GSD, (full_h, cw)).copy()
        lon, lat = to_ll.transform(xs, ys)
        zref = sample_srtm(lon, lat, hgt, orthometric=False)
        del lon, lat
        east, north_enu, up = enu_from_map(xs, ys, zref, "EPSG:32631", geo["lat"], geo["lon"], geo["height"])
        r_ref, alpha_ref = perpendicular_geometry(east, north_enu, up, flight["origin_enu"], flight["direction_enu"])
        del east, north_enu, up, zref
        if not np.isfinite(r_ref).all():
            raise RuntimeError("r_ref has NaN")

        def slant_of(zsurf):
            e, n, u = enu_from_map(xs, ys, zsurf, "EPSG:32631", geo["lat"], geo["lon"], geo["height"])
            return perpendicular_geometry(e, n, u, flight["origin_enu"], flight["direction_enu"])

        r_f, a_f = slant_of(z_full)
        del z_full
        if z_bldg is not None:
            r_b, a_b = slant_of(z_bldg)
            del z_bldg
        else:
            r_b = a_b = None
        del xs, ys
        # Canonical: index 0 nearest the sensor. File row 0 is north.
        if not sens0:
            r_f, a_f, r_ref = np.flipud(r_f), np.flipud(a_f), np.flipud(r_ref)
            if r_b is not None:
                r_b, a_b = np.flipud(r_b), np.flipud(a_b)
        dr = float(np.nanmedian(r_ref[-40:, :]) - np.nanmedian(r_ref[:40, :]))
        if not np.isfinite(dr) or dr <= 0:
            raise RuntimeError(f"canonical r_ref does not increase ({dr})")
        if n_done == 0:
            log(f"  canonical r_ref far-near {dr:.1f} m  "
                f"surface alpha near {np.degrees(np.nanmedian(a_f[:40])):.2f}° "
                f"far {np.degrees(np.nanmedian(a_f[-40:])):.2f}°")

        def to_file(arr):
            img = arr[PAD_PX:PAD_PX + g["height"]]
            return img if sens0 else img[::-1]

        win = Window(col0, 0, cw, g["height"])
        made = {}
        if "A_full" in products:
            made["A_full"] = project_grid(r_f, a_f, r_ref, axis=0, nan_margin=NAN_MARGIN)
        if "B_full" in products:
            made["B_full"] = own_range_grid(r_f, a_f, axis=0, nan_margin=NAN_MARGIN)
        del r_f, a_f, r_ref
        if "B_bldg" in products:
            made["B_bldg"] = own_range_grid(r_b, a_b, axis=0, nan_margin=NAN_MARGIN)
        del r_b, a_b
        gc.collect()
        for pname, (cov, unk, _vis) in made.items():
            cov_f = to_file(cov)
            unk_f = to_file(unk)
            lab = labels_from_coverage(cov_f, unk_f)
            datasets[pname][0].write(lab.astype(np.uint8), 1, window=win)
            datasets[pname][1].write(cov_f.astype(np.float32), 1, window=win)
            unk_counts[pname] += int(np.asarray(unk_f).sum())
        del made
        gc.collect()
        n_done += 1
        if col_ranges is None:
            done.add(col0)
            progress_path.write_text(json.dumps(sorted(done)))
            unk_path.write_text(json.dumps(unk_counts))
        log(f"  chunk {time.time()-t1:.1f}s  total {time.time()-t0:.0f}s")
    for lab, cov in datasets.values():
        lab.close()
        cov.close()
    if col_ranges is None and not limit_chunks:
        n_pix = g["height"] * g["width"]
        frac = {k: v / n_pix for k, v in unk_counts.items()}
        log("unknown fraction " + "  ".join(f"{k} {v:.3%}" for k, v in frac.items()))
        dump({"counts": unk_counts, "fraction": frac, "n": n_pix, "sensor_at_row0": sens0}, out / "unknown.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--limit-chunks", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    stage = args.stage
    if stage == "part1-p":
        rc = ROOT / "output" / "rule_check"
        part1("P", MAGS["P"], rc / "prep.npz", rc / "A_full_cov.tif", rc / "A_full_lab.tif",
              rc / "B_full_cov.tif", rc / "B_full_lab.tif", max_chunks=args.limit_chunks)
    elif stage == "part1-k":
        k = OUT / "K"
        part1("K", MAGS["K"], k / "prep.npz", k / "A_full_cov.tif", k / "A_full_lab.tif",
              k / "B_full_cov.tif", k / "B_full_lab.tif", max_chunks=args.limit_chunks)
    elif stage == "prep-k":
        stage_prep("K", MAGS["K"])
    elif stage == "prep-n":
        stage_prep("N", MAGS["N"])
    elif stage == "project-k":
        stage_project("K", MAGS["K"], OUT / "K" / "prep.npz", limit_chunks=args.limit_chunks)
    else:
        from decide_lay import dispatch
        dispatch(stage, args.limit_chunks)


if __name__ == "__main__":
    main()
