"""Pick the 50 Teesta test tiles for hand labelling and render their images."""
import base64, io, json, glob, re, random
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
from shapely.geometry import box
from scipy.ndimage import median_filter, uniform_filter
from PIL import Image
from common import ROOT, DATA, DER, CRS, grid, teesta_mask

TS = 256
OUT = ROOT / "evaluation" / "hand_labels"; OUT.mkdir(parents=True, exist_ok=True)
t10, H, W = grid(10)
rng = random.Random(20261106)

mask30 = teesta_mask(30)
mask10 = np.kron(mask30, np.ones((3, 3), bool))[:H, :W]
with rasterio.open(DER / "dist_river_30m.tif") as s:
    dist30 = s.read(1)
dates = sorted(pd.read_csv(DATA / "scenes_core.csv").date.unique())


def read_db(date, pol):
    with rasterio.open(DER / "s1_10m" / f"{date}_{pol}.tif") as s:
        a = s.read(1).astype(np.float32)
    a[a == -32768] = np.nan
    return a / 100


cands = []
for r in range(0, H - TS, TS):
    for c in range(0, W - TS, TS):
        if mask10[r:r + TS, c:c + TS].mean() < 0.99:
            continue
        rc, cc = (r + TS // 2) // 3, (c + TS // 2) // 3
        if not np.isfinite(dist30[rc, cc]) or dist30[rc, cc] > 3000:
            continue
        cands.append((r, c))
print("candidate tile locations:", len(cands))

chosen, used = [], set()
extra = {"2020-06-07": 1, "2025-08-16": 1}
for date in dates:
    vv = read_db(date, "VV")
    thr = -15.0
    vv = median_filter(np.nan_to_num(vv, nan=0.0), size=5) + np.where(np.isfinite(vv), 0, np.nan)
    info = []
    for r, c in cands:
        if (r, c) in used:
            continue
        tile = vv[r:r + TS, c:c + TS]
        if not np.isfinite(tile).all():
            continue
        info.append((r, c, float((tile < thr).mean())))
    mixed = [x for x in info if 0.10 <= x[2] <= 0.60]
    little = [x for x in info if 0.02 <= x[2] < 0.10]
    want = [("mixed", mixed, 4), ("little water", little, 1), ("random", info, 1)]
    if date in extra:
        want[0] = ("mixed", mixed, 5)
    picked = []
    for kind, pool, n in want:
        pool = pool[:]; rng.shuffle(pool)
        for r, c, f in pool:
            if len([p for p in picked if p[3] == kind]) >= n:
                break
            if (r, c) in used or any(np.hypot(r - p[0], c - p[1]) * 10 < 6000 for p in picked):
                continue
            picked.append((r, c, f, kind)); used.add((r, c))
    target = sum(n for _, _, n in want)
    pool = info[:]; rng.shuffle(pool)
    for r, c, f in pool:
        if len(picked) >= target:
            break
        if (r, c) in used or any(np.hypot(r - p[0], c - p[1]) * 10 < 6000 for p in picked):
            continue
        picked.append((r, c, f, "random (top-up)")); used.add((r, c))
    print(f"{date}: threshold {thr:.1f} dB, {len(info)} usable locations -> {len(picked)} tiles", flush=True)
    chosen += [dict(date=date, row=r, col=c, kind=k, rough_water=f) for r, c, f, k in picked]
    del vv

tiles = pd.DataFrame(chosen)
tiles["tile_id"] = [f"T{i+1:02d}_{d.replace('-', '')}" for i, d in enumerate(tiles.date)]
tiles["x0"] = t10.c + tiles.col * 10; tiles["y1"] = t10.f - tiles.row * 10
tiles["x1"] = tiles.x0 + TS * 10; tiles["y0"] = tiles.y1 - TS * 10
order = list(range(len(tiles))); rng.shuffle(order)
groups = ["ABCD"[i % 4] for i in range(len(tiles))]
tiles["group"] = [groups[order.index(i)] for i in range(len(tiles))]
dbl = rng.sample(range(len(tiles)), 8)
tiles["second_group"] = ["ABCD"[("ABCD".index(tiles.group[i]) + 1) % 4] if i in dbl else "" for i in range(len(tiles))]
print(tiles.groupby("date").size().to_dict(), tiles.kind.value_counts().to_dict())


def to_jpeg(rgb, q=92):
    b = io.BytesIO(); Image.fromarray(rgb).save(b, "JPEG", quality=q); return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def to_png(g):
    b = io.BytesIO(); Image.fromarray(g).save(b, "PNG", optimize=True); return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def lee(db, size=5, enl=4.4):
    p = 10 ** (db / 10); m = uniform_filter(p, size); m2 = uniform_filter(p * p, size)
    var = np.maximum(m2 - m * m, 0); cu2 = 1 / enl
    w = np.clip((var - m * m * cu2) / np.maximum(var * (1 + cu2), 1e-12), 0, 1)
    return 10 * np.log10(np.maximum(m + w * (p - m), 1e-6))


def stretch(a, lo, hi, gamma=1.0):
    return (np.clip((a - lo) / (hi - lo), 0, 1) ** gamma * 255).astype(np.uint8)


gran = {}
for f in glob.glob(str(DATA / "hls_flood" / "*.tif")):
    m = re.match(r".*(HLS\.(L30|S30)\.T\w+\.(\d{7})T\d+\.v2\.0)\.(\w+)\.tif$", f)
    gran.setdefault(m.group(1), {})[m.group(4)] = f


def granule_date(g):
    import datetime as dt
    y, doy = int(g.split(".")[3][:4]), int(g.split(".")[3][4:7])
    return dt.date(y, 1, 1) + dt.timedelta(doy - 1)


def hls_on_tile(path, ttf, nodata):
    out = np.full((TS, TS), nodata, np.float32)
    with rasterio.open(path) as s:
        reproject(rasterio.band(s, 1), out, src_nodata=nodata, dst_transform=ttf, dst_crs=CRS, dst_nodata=nodata,
                  resampling=Resampling.nearest)
    return out


hand_src = rasterio.open(DER / "hand_dem_30m.tif")
records = []
for _, tl in tiles.iterrows():
    win = Window(tl.col, tl.row, TS, TS)
    with rasterio.open(DER / "s1_10m" / f"{tl.date}_VV.tif") as s:
        vv = s.read(1, window=win) / 100
    with rasterio.open(DER / "s1_10m" / f"{tl.date}_VH.tif") as s:
        vh = s.read(1, window=win) / 100
    rec = dict(id=tl.tile_id, date=tl.date, group=tl.group, second=tl.second_group,
               bbox32645=[tl.x0, tl.y0, tl.x1, tl.y1],
               smooth=to_jpeg(np.dstack([stretch(lee(vv), -25, -3), stretch(lee(vh), -30, -10), stretch(lee(vv) - lee(vh), 2, 14)])),
               radar=to_jpeg(np.dstack([stretch(vv, -25, -3), stretch(vh, -30, -10), stretch(vv - vh, 2, 14)])),
               vv=to_jpeg(stretch(vv, -25, 0)))
    ttf = from_origin(tl.x0, tl.y1, 10, 10)
    hand = np.zeros((TS, TS), np.float32)
    reproject(rasterio.band(hand_src, 1), hand, dst_transform=ttf, dst_crs=CRS, resampling=Resampling.bilinear)
    rec["hand"] = to_png(stretch(np.nan_to_num(hand), 0, 10))
    import datetime as dt
    d0 = dt.date.fromisoformat(tl.date); best = None
    for g, b in gran.items():
        if abs((granule_date(g) - d0).days) > 3 or "Fmask" not in b:
            continue
        fm = hls_on_tile(b["Fmask"], ttf, 255)
        clear = (fm != 255) & ((fm.astype(np.uint8) & 0b1110) == 0)
        if best is None or clear.mean() > best[1]:
            best = (g, clear.mean())
    if best and best[1] >= 0.10:
        g, cf = best; b = gran[g]
        nir, sw = ("B05", "B06") if ".L30." in g else ("B8A", "B11")
        band = {k: hls_on_tile(b[k], ttf, -9999) * 1e-4 for k in ("B02", "B03", "B04", nir, sw)}
        rec["optical"] = to_jpeg(np.dstack([stretch(band["B04"], 0, 0.25, 0.8), stretch(band["B03"], 0, 0.25, 0.8), stretch(band["B02"], 0, 0.25, 0.8)]))
        rec["water_view"] = to_jpeg(np.dstack([stretch(band[sw], 0, 0.4), stretch(band[nir], 0, 0.45), stretch(band["B03"], 0, 0.25)]))
        rec["optical_note"] = f"{'Landsat' if '.L30.' in g else 'Sentinel-2'} {granule_date(g)}, {100*cf:.0f}% clear"
    else:
        rec["optical_note"] = "no clear optical image within 3 days"
    records.append(rec)
    print(tl.tile_id, tl.kind, f"rough water {tl.rough_water:.2f}", "|", rec["optical_note"], flush=True)

tiles.to_csv(OUT / "tiles.csv", index=False)
gpd.GeoDataFrame(tiles.drop(columns=["x0", "x1", "y0", "y1"]),
                 geometry=[box(r.x0, r.y0, r.x1, r.y1) for r in tiles.itertuples()], crs=CRS).to_crs(4326) \
    .to_file(OUT / "tiles.geojson", driver="GeoJSON")
json.dump(records, open(OUT / "tiles_payload.json", "w"))
print("tiles with optical:", sum("optical" in r for r in records), "of", len(records))
