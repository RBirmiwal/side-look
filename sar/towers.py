#!/usr/bin/env python3
"""Tall-building crops on the strip that covers De Rotterdam and the Maastoren.

The MAG-POL is /tmp/towers/city.tif (20190823072627_20190823072925). Writes
only under sar/output/rule_check/.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from shapely.geometry import Polygon

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from acquire_geometry import ecef_to_geodetic, enu_basis  # noqa: E402
from image_space_run import load_hgt, sample_srtm  # noqa: E402
from near_field import (  # noqa: E402
    add_undulation,
    enu_from_map,
    fit_flight_line,
    look_azimuth_from_track,
    perpendicular_geometry,
)
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

TIF = Path("/tmp/towers/city.tif")
AHN = Path("/workspace/sar/data/ahn_dsm_1m_aoi.tif")
HGT = Path("/workspace/sar/data/srtm/N51E004.hgt.gz")
OUT = Path("/workspace/sar/output/rule_check")
GSD = 0.25


def log(msg):
    print(msg, flush=True)


def fetch_box(xmin, ymin, xmax, ymax):
    """All pand features in an RD bbox."""
    feats = []
    url = (
        "https://api.3dbag.nl/collections/pand/items"
        f"?bbox={xmin},{ymin},{xmax},{ymax}&limit=100"
    )
    pages = 0
    while url and pages < 8:
        pages += 1
        req = urllib.request.Request(url, headers={"User-Agent": "sar/1.0", "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=90) as r:
            d = json.load(r)
        tr = (d.get("metadata") or {}).get("transform")
        for f in d.get("features", []):
            if tr and "transform" not in f:
                f["transform"] = tr
            feats.append(f)
        url = None
        for link in d.get("links", []):
            if link.get("rel") == "next":
                url = link["href"]
        if len(d.get("features", [])) < 100:
            url = None
    return feats


def footprint_rd(feat):
    """Ground ring in RD from a CityJSON feature. Returns (polygon_xy list, h, area)."""
    # vertices may be relative to a transform stored beside the feature
    tr = feat.get("transform")
    verts = feat["vertices"]
    co = next(iter(feat["CityObjects"].values()))
    a = co["attributes"]
    h = float(a["b3_h_nok"] - a["b3_h_maaiveld"])
    area = a.get("b3_opp_grond") or 0
    ident = next(iter(feat["CityObjects"]))
    geom = co["geometry"][0]["boundaries"]

    def ring(node):
        if node and isinstance(node[0], int):
            return node
        return ring(node[0])

    idx = ring(geom)
    if tr:
        sc, tl = tr["scale"], tr["translate"]
        pts = [(verts[i][0] * sc[0] + tl[0], verts[i][1] * sc[1] + tl[1]) for i in idx]
    else:
        pts = [(verts[i][0], verts[i][1]) for i in idx]
    return ident, pts, h, float(area)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pyproj import Transformer

    desc = json.loads(rasterio.open(TIF).tags()["TIFFTAG_IMAGEDESCRIPTION"])
    collect = desc["collect"]
    target = np.array(collect["image"]["center_pixel"]["target_position"], float)
    lat, lon, height = ecef_to_geodetic(target)
    basis = enu_basis(lat, lon)
    pos = np.array([sv["position"] for sv in collect["state"]["state_vectors"]], float)
    enu = (basis @ (pos - target).T).T
    fit = fit_flight_line(enu)
    look = look_azimuth_from_track(fit["direction_enu"], collect["radar"]["pointing"])
    flight = {"origin_enu": fit["origin_enu"], "direction_enu": fit["direction_enu"]}
    geo = {"lat": float(lat), "lon": float(lon), "height": float(height)}
    log(f"look {look:.2f}  platform {fit['height_above_origin_m']:.0f} m")

    with rasterio.open(TIF) as src:
        H, W = src.height, src.width
        mag_t = src.transform
        west, north = float(mag_t.c), float(mag_t.f)
    # which way is the sensor? alpha should grow away from it
    rows = np.array([H // 5, 4 * H // 5])
    col = W // 2
    xs = np.full(2, west + (col + 0.5) * GSD)
    ys = north - (rows + 0.5) * GSD
    z = np.full(2, height)
    e, n, u = enu_from_map(xs, ys, z, "EPSG:32631", geo["lat"], geo["lon"], geo["height"])
    _, alpha = perpendicular_geometry(e, n, u, flight["origin_enu"], flight["direction_enu"])
    sensor_at_row0 = bool(alpha[1] > alpha[0])
    log(f"alpha row {rows[0]} {np.degrees(alpha[0]):.2f}  row {rows[1]} {np.degrees(alpha[1]):.2f}  "
        f"sensor_at_row0 {sensor_at_row0}")

    to_utm = Transformer.from_crs("EPSG:28992", "EPSG:32631", always_xy=True)
    to_rd = Transformer.from_crs("EPSG:32631", "EPSG:28992", always_xy=True)
    to_ll = Transformer.from_crs("EPSG:32631", "EPSG:4326", always_xy=True)

    # two search boxes in RD, generous
    boxes = {
        "De Rotterdam area": (92850, 435350, 93450, 435950),
        "Maastoren area": (93250, 435750, 93750, 436350),
    }
    buildings = []
    for name, box in boxes.items():
        log(f"fetch {name}")
        feats = fetch_box(*box)
        log(f"  {len(feats)} features")
        # the collection endpoint returns features whose vertices are absolute
        # only when the wrapper carries the transform. items?bbox returns
        # per-feature vertices that are already scaled in some deployments and
        # relative in others. Handle both.
        for feat in feats:
            # bbox response stores transform inside metadata of the collection
            # document, copied onto each feature by us below if present
            try:
                ident, pts, h, area = footprint_rd(feat)
            except Exception:
                continue
            if h < 40 or len(pts) < 4:
                continue
            utm = [to_utm.transform(x, y) for x, y in pts]
            buildings.append({"id": ident, "h_bag": h, "area": area, "utm": utm, "where": name})
        log(f"  over 40 m: {sum(1 for b in buildings if b['where']==name)}")

    # AHN max within 300 m of the named points, to confirm the candidates
    named = {"De Rotterdam": (4.4881, 51.9061), "Maastoren": (4.4931, 51.9094)}
    confirm = {}
    with rasterio.open(AHN) as ahn:
        to_ahn = Transformer.from_crs("EPSG:32631", ahn.crs, always_xy=True)
        for name, (lon_p, lat_p) in named.items():
            x, y = Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=True).transform(lon_p, lat_p)
            ax, ay = to_ahn.transform(x, y)
            # 300 m window
            win_t = from_origin(ax - 300, ay + 300, 1.0, 1.0)
            dest = np.full((600, 600), np.nan, np.float32)
            reproject(
                rasterio.band(ahn, 1), dest, src_transform=ahn.transform, src_crs=ahn.crs,
                dst_transform=win_t, dst_crs=ahn.crs, resampling=Resampling.bilinear,
            )
            confirm[name] = {
                "utm": [x, y], "rd": list(to_rd.transform(x, y)),
                "ahn_max": None if not np.isfinite(dest).any() else float(np.nanmax(dest)),
                "ahn_p95": None if not np.isfinite(dest).any() else float(np.nanpercentile(dest, 95)),
            }
            log(f"  {name} AHN max within 300 m {confirm[name]['ahn_max']}")

    if not buildings:
        log("NO buildings over 40 m returned")
        (OUT / "towers.json").write_text(json.dumps({"confirm": confirm, "buildings": []}, indent=2))
        return

    # keep the largest-volume ones plus anything near the named points
    for b in buildings:
        xs = [p[0] for p in b["utm"]]
        ys = [p[1] for p in b["utm"]]
        b["cx"] = float(np.mean(xs))
        b["cy"] = float(np.mean(ys))
        b["volume"] = b["h_bag"] * b["area"]
    buildings.sort(key=lambda b: -b["volume"])
    keep = buildings[:4]
    log("kept " + ", ".join(f"{b['id'].split('.')[-1]} h{b['h_bag']:.0f} A{b['area']:.0f}" for b in keep))

    hgt = load_hgt(HGT)
    results = []
    with rasterio.open(TIF) as src, rasterio.open(AHN) as ahn:
        to_ahn = Transformer.from_crs("EPSG:32631", ahn.crs, always_xy=True)
        for i, b in enumerate(keep):
            col = int(round((b["cx"] - west) / GSD))
            row = int(round((north - b["cy"]) / GSD))
            if not sensor_at_row0:
                # we will flip the crop; row index in the file stays as stored
                pass
            half = int(300 / GSD)
            r0 = max(0, row - half)
            c0 = max(0, col - half)
            r1 = min(H, row + half)
            c1 = min(W, col + half)
            hh = src.read(1, window=Window(c0, r0, c1 - c0, r1 - r0)).astype(np.float32)
            # local AHN on the crop, 1 m
            xs = [p[0] for p in b["utm"]]
            ys = [p[1] for p in b["utm"]]
            pad = 80
            wx0, wx1 = min(xs) - pad, max(xs) + pad
            wy0, wy1 = min(ys) - pad, max(ys) + pad
            aw, ah = int(wx1 - wx0) + 2, int(wy1 - wy0) + 2
            ahn_t = from_origin(wx0, wy1, 1.0, 1.0)
            dsm = np.full((ah, aw), np.nan, np.float32)
            reproject(
                rasterio.band(ahn, 1), dsm, src_transform=ahn.transform, src_crs=ahn.crs,
                dst_transform=ahn_t, dst_crs="EPSG:32631", resampling=Resampling.bilinear,
                dst_nodata=np.nan,
            )
            finite = dsm[np.isfinite(dsm)]
            if finite.size < 100:
                log(f"  {b['id']} no AHN")
                continue
            poly = Polygon(b["utm"])
            if not poly.is_valid:
                poly = poly.buffer(0)
            fp = rasterize([(poly, 1)], out_shape=dsm.shape, transform=ahn_t, fill=0, dtype=np.uint8)
            inside = fp.astype(bool) & np.isfinite(dsm)
            if inside.sum() < 10:
                log(f"  {b['id']} footprint misses AHN")
                continue
            outside = (~fp.astype(bool)) & np.isfinite(dsm)
            if outside.sum() > 80:
                ground_level = float(np.percentile(dsm[outside], 10))
            else:
                ground_level = float(np.percentile(finite, 5))
            ndsm = dsm - ground_level
            h70 = float(np.percentile(ndsm[inside], 70))
            h = float(np.percentile(ndsm[inside], 90))
            hmax = float(np.nanmax(ndsm[inside]))
            # depth along range. range is image row.
            # polygon pixel rows
            rs, cs = np.where(inside)
            # map 1 m northing to image row
            northings = wy1 - (rs + 0.5)
            eastings = wx0 + (cs + 0.5)
            img_rows = (north - northings) / GSD
            depth = float(img_rows.max() - img_rows.min()) * GSD
            # theta and s at the centroid
            gx, gy = b["cx"], b["cy"]
            lon_c, lat_c = to_ll.transform(gx, gy)
            # NLGEO2018 at the centroid. add_undulation on a 1x1 grid returns NaN.
            with rasterio.open(Path("/workspace/sar/data/nl_nsgi_nlgeo2018.tif")) as geoid:
                n_geo = float(next(geoid.sample([(lon_c, lat_c)]))[0])
            g_ell = ground_level + n_geo
            ortho = float(sample_srtm(np.array([[lon_c]]), np.array([[lat_c]]), hgt, orthometric=True)[0, 0])
            from pyproj import Transformer as T
            egm = T.from_pipeline(
                "+proj=pipeline +step +proj=vgridshift "
                "+grids=/workspace/sar/data/geoid/us_nga_egm96_15.tif +multiplier=1"
            )
            _, _, n_egm = egm.transform(lon_c, lat_c, 0.0)
            srtm = ortho + float(n_egm)
            s = srtm - g_ell
            e, n, u = enu_from_map(np.array([gx]), np.array([gy]), np.array([g_ell]),
                                   "EPSG:32631", geo["lat"], geo["lon"], geo["height"])
            slant, alpha_c = perpendicular_geometry(e, n, u, flight["origin_enu"], flight["direction_enu"])
            theta = float(np.degrees(alpha_c[0]))
            shift = (h - s) / np.tan(np.radians(theta))
            shadow = h * np.tan(np.radians(theta))
            log(f"  {b['id'].split('.')[-1]} h70 {h:.1f} bag {b['h_bag']:.1f} depth {depth:.0f} "
                f"s {s:+.1f} theta {theta:.1f} shift {shift:.0f} shadow {shadow:.0f} "
                f"row {row} col {col}")
            if depth < 25:
                log("    skipped, footprint shallower than 25 m along range")
                continue

            # draw on the HH crop. coordinates: image row/col
            def to_rc(poly_utm, drow=0.0):
                pts = np.array(poly_utm)
                cc = (pts[:, 0] - west) / GSD - c0
                rr = (north - pts[:, 1]) / GSD - r0 + drow
                return np.c_[cc, rr]

            sign = -1.0 if sensor_at_row0 else 1.0  # toward the sensor in +row if sensor is at the bottom
            # drow is added to image row. toward sensor: decrease row if sensor_at_row0
            d_roof = sign * shift / GSD
            # shadow band: from the far side, length `shadow`, away from sensor
            d_shadow = -sign * shadow / GSD

            show = hh
            extent_row0 = r0
            if not sensor_at_row0:
                show = hh[::-1]
                # flipping rows: a point at image row r becomes (r1-1-r) - r0 in the flipped array
                # easier to flip and remap
            fig, ax = plt.subplots(figsize=(7.2, 7.2))
            v = show[np.isfinite(show)]
            lo, hi = np.percentile(v, [2, 98]) if v.size else (0, 1)
            ax.imshow(show, cmap="gray", vmin=lo, vmax=hi)
            def draw(pts, color, lw=1.0):
                ax.plot(np.r_[pts[:, 0], pts[0, 0]], np.r_[pts[:, 1], pts[0, 1]], color=color, lw=lw)

            roof_utm = [(x, y + sign * shift) for x, y in b["utm"]]  # WRONG if range isn't north
            # range is image row, which is north-south only if the image is north-up.
            # This strip is north-up 0.25 m. Moving toward the sensor is ±northing.
            # sensor_at_row0 means sensor is at the north edge, toward sensor = +northing.
            dn = (shift if sensor_at_row0 else -shift)
            roof = [(x, y + dn) for x, y in b["utm"]]
            # shadow is away from the sensor, from the far wall. Approximate by shifting
            # the whole footprint away, and draw both the footprint and the shifted copy
            # so the band between them is the shadow.
            ds = -dn / shift * shadow if shift else 0
            shad = [(x, y + ds) for x, y in b["utm"]]

            def pix(poly):
                pts = np.array(poly)
                cc = (pts[:, 0] - west) / GSD - c0
                rr = (north - pts[:, 1]) / GSD - r0
                if not sensor_at_row0:
                    rr = (hh.shape[0] - 1) - rr
                return np.c_[cc, rr]

            draw(pix(b["utm"]), "white", 1.2)
            draw(pix(roof), "cyan", 1.0)
            draw(pix(shad), "deepskyblue", 0.8)
            ax.set_xlim(0, hh.shape[1])
            ax.set_ylim(hh.shape[0], 0)
            title = (f"h90 {h:.0f} m  max {hmax:.0f}  bag {b['h_bag']:.0f}  "
                     f"s {s:+.1f}  θ {theta:.0f}°  depth {depth:.0f} m\n"
                     f"white footprint  cyan rule A roof ({shift:.0f} m toward sensor)  "
                     f"blue geometric shadow ({shadow:.0f} m)")
            ax.set_title(title, fontsize=8)
            ax.axis("off")
            fig.tight_layout()
            path = OUT / f"tower_{i}_{b['id'].split('.')[-1]}.png"
            fig.savefig(path, dpi=110)
            plt.close(fig)
            log(f"    wrote {path.name}")
            results.append({
                "id": b["id"], "where": b["where"], "h90_ndsm": h, "h70_ndsm": h70,
                "hmax_ndsm": hmax, "h_bag": b["h_bag"],
                "area_m2": b["area"], "depth_m": depth, "s_m": s, "theta_deg": theta,
                "roof_shift_m": shift, "shadow_m": shadow, "row": row, "col": col,
                "file": path.name, "sensor_at_row0": sensor_at_row0,
            })
    (OUT / "towers.json").write_text(json.dumps({"confirm_ahn": confirm, "look_deg": look,
                                                 "buildings": results}, indent=2))
    log(f"wrote {OUT / 'towers.json'}")


if __name__ == "__main__":
    main()
