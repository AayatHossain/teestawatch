"""Yearly active channel, bank lines, transects and erosion rates from HLS."""
import glob, json, re
from collections import defaultdict
import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import shapes, rasterize
from shapely.geometry import shape, LineString, Point
from shapely.ops import linemerge, unary_union
from scipy import ndimage
from skimage.morphology import disk, binary_closing
from common import ROOT, DATA, DER, CRS, RIVER, grid, teesta_mask, write

YEARS = list(range(2016, 2027))
OUT = ROOT / "outputs" / "erosion"; OUT.mkdir(parents=True, exist_ok=True)
CH = DER / "channel"; CH.mkdir(exist_ok=True)
t, h, w = grid(30)
inside = teesta_mask(30)
dline = ndimage.distance_transform_edt(~rasterize(RIVER.geometry, out_shape=(h, w), transform=t, fill=0, default_value=1, dtype="uint8").astype(bool)) * 30
corridor = inside & (dline <= 3000)
near = dline <= 1500
CLOUDY = 0b11110


def on_grid(path, nodata, dtype):
    dst = np.full((h, w), nodata, dtype)
    with rasterio.open(path) as s:
        reproject(rasterio.band(s, 1), dst, src_nodata=nodata, dst_transform=t, dst_crs=CRS, dst_nodata=nodata, resampling=Resampling.nearest)
    return dst


gran = defaultdict(dict)
for f in glob.glob(str(DATA / "hls" / "*.tif")):
    m = re.match(r".*(HLS\.(L30|S30)\.T\w+\.(\d{4})\d{3}T\d+\.v2\.0)\.(\w+)\.tif$", f)
    gran[m.group(1)][m.group(4)] = f
by_year = defaultdict(list)
for g, b in gran.items():
    nir, sw = ("B05", "B06") if ".L30." in g else ("B8A", "B11")
    if {"B03", "B04", nir, sw, "Fmask"} <= b.keys():
        by_year[int(g.split(".")[3][:4])].append((b["B03"], b["B04"], b[nir], b[sw], b["Fmask"]))


def channel_of(cls):
    wet = (cls == 1); act = corridor & ((cls == 1) | (cls == 2))
    act = binary_closing(act, disk(2))
    lab, n = ndimage.label(act)
    wl, wn = ndimage.label(wet & corridor)
    wsz = ndimage.sum(wet, wl, range(1, wn + 1)); wnear = ndimage.maximum(near, wl, range(1, wn + 1))
    river_water = np.isin(wl, 1 + np.flatnonzero((wsz * 900 >= 5e4) & (wnear > 0)))
    keep = np.unique(lab[river_water]); keep = keep[keep > 0]
    ch = np.isin(lab, keep)
    ch = ndimage.binary_fill_holes(binary_closing(ch, disk(5))) & inside
    lab, n = ndimage.label(ch)
    if n:
        sz = ndimage.sum(ch, lab, range(1, n + 1))
        ch = np.isin(lab, 1 + np.flatnonzero(sz * 900 >= 5e5))
    return ch


classes, channels, observed = {}, {}, {}
for y in YEARS:
    nw = np.zeros((h, w), np.uint16); ns = np.zeros((h, w), np.uint16); n = np.zeros((h, w), np.uint16)
    for fg, fr, fn, fs, fm in by_year[y]:
        q = on_grid(fm, 255, np.uint8)
        g, r, nr, s = (on_grid(p, -9999, np.int16).astype(np.float32) for p in (fg, fr, fn, fs))
        ok = (q != 255) & ((q & CLOUDY) == 0) & (g > -9999) & (r > -9999) & (nr > -9999) & (s > -9999) & ((g + s) > 0) & ((nr + r) > 0)
        mndwi = np.where(ok, (g - s) / np.where(ok, g + s, 1), 0); ndvi = np.where(ok, (nr - r) / np.where(ok, nr + r, 1), 0)
        wat = ok & (mndwi > 0); snd = ok & ~wat & (ndvi < 0.15) & (r * 1e-4 > 0.10)
        nw += wat; ns += snd; n += ok
    obs = (n >= 2) & inside
    cls = np.full((h, w), 255, np.uint8)
    cls[obs] = 0
    cls[obs & (ns >= 0.5 * n)] = 2
    cls[obs & (nw >= 0.5 * n)] = 1
    ch = channel_of(cls)
    classes[y], channels[y], observed[y] = cls, ch, obs
    write(CH / f"class_{y}.tif", cls, nodata=255); write(CH / f"channel_{y}.tif", ch.astype(np.uint8))
    print(f"{y}: images {len(by_year[y]):3d} | observed {100*obs.sum()/inside.sum():5.1f}% | water {(cls==1).sum()*9e-4:6.1f} km2 | sand {(cls==2).sum()*9e-4:6.1f} km2 | channel {ch.sum()*9e-4:6.1f} km2", flush=True)

loss = []
for i, y in enumerate(YEARS):
    ch = channels[y]
    poly = unary_union([shape(gm) for gm, v in shapes(ch.astype(np.uint8), mask=ch, transform=t) if v == 1])
    gpd.GeoDataFrame({"year": [y], "channel_km2": [round(ch.sum() * 9e-4, 2)]}, geometry=[poly.boundary], crs=CRS) \
        .to_crs(4326).to_file(OUT / f"banks_{y}.geojson", driver="GeoJSON")
    if i:
        both = observed[y] & observed[YEARS[i - 1]]
        lost = ch & ~channels[YEARS[i - 1]] & both; gained = channels[YEARS[i - 1]] & ~ch & both
        loss.append(dict(year=y, land_lost_ha=round(float(lost.sum() * 0.09), 1), land_gained_ha=round(float(gained.sum() * 0.09), 1),
                         observed_share=round(float(both.sum() / inside.sum()), 3)))
json.dump({"method": "HLS dry-season water+sand active channel; land lost = channel(y) minus channel(y-1) on pixels observed in both years",
           "unit": "ha", "area": "Teesta in Bangladesh (OSM border), Brahmaputra side excluded", "years": loss},
          open(OUT / "loss_by_year.json", "w"), indent=1)
print("land lost (ha):", {r["year"]: r["land_lost_ha"] for r in loss})

line = linemerge(RIVER.geometry.union_all())
if line.geom_type == "MultiLineString":
    line = max(line.geoms, key=lambda g: g.length)
ref = line.simplify(300)
rows = []
for k, s0 in enumerate(np.arange(0, ref.length, 250.0)):
    p = ref.interpolate(s0); q = ref.interpolate(min(s0 + 50, ref.length)); q0 = ref.interpolate(max(s0 - 50, 0))
    dx, dy = q.x - q0.x, q.y - q0.y; L = np.hypot(dx, dy)
    if L == 0:
        continue
    nx, ny = -dy / L, dx / L
    tr = LineString([(p.x - 4000 * nx, p.y - 4000 * ny), (p.x + 4000 * nx, p.y + 4000 * ny)])
    ds = np.arange(-4000, 4001, 15.0)
    xs, ys = p.x + ds * nx, p.y + ds * ny
    cols = ((xs - t.c) / 30).astype(int); rws = ((t.f - ys) / 30).astype(int)
    okp = (cols >= 0) & (cols < w) & (rws >= 0) & (rws < h)
    rec = dict(transect=k, chainage_m=round(float(s0)), geometry=tr)
    for y in YEARS:
        on = np.zeros(len(ds), bool); seen = np.zeros(len(ds), bool)
        on[okp] = channels[y][rws[okp], cols[okp]]; seen[okp] = observed[y][rws[okp], cols[okp]]
        for side, sel in (("left", ds < 0), ("right", ds > 0)):
            hit = on & sel
            rec[f"{side}_{y}"] = float(np.abs(ds[hit]).max()) if hit.any() and seen[sel].mean() > 0.8 else np.nan
    rows.append(rec)
tr = gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS)


def slope(yrs, vals):
    yrs, vals = np.array(yrs, float), np.array(vals, float); ok = np.isfinite(vals)
    if ok.sum() < 4:
        return np.nan, np.nan
    b, a = np.polyfit(yrs[ok], vals[ok], 1)
    return b, a


bt = []
for side in ("left", "right"):
    rates, preds = [], []
    for _, r in tr.iterrows():
        rate, _ = slope(YEARS, [r[f"{side}_{y}"] for y in YEARS])
        rates.append(rate)
        b, a = slope([y for y in YEARS if y <= 2022], [r[f"{side}_{y}"] for y in YEARS if y <= 2022])
        obs25, last22 = r[f"{side}_2025"], r[f"{side}_2022"]
        if np.isfinite(b) and np.isfinite(obs25) and np.isfinite(last22):
            bt.append(dict(side=side, transect=r.transect, pred=a + b * 2025, nochange=last22, obs=obs25))
    tr[f"{side}_rate_m_per_yr"] = np.round(rates, 1)
bt = pd.DataFrame(bt)
res = {"fit_years": "2016-2022", "predict_year": 2025, "n_bank_points": int(len(bt)),
       "linear_trend_MAE_m": round(float((bt.pred - bt.obs).abs().mean()), 1), "linear_trend_median_AE_m": round(float((bt.pred - bt.obs).abs().median()), 1),
       "no_change_MAE_m": round(float((bt.nochange - bt.obs).abs().mean()), 1), "no_change_median_AE_m": round(float((bt.nochange - bt.obs).abs().median()), 1),
       "note": "bank position = outermost active-channel distance from the smoothed centre line on each side of a 250 m transect"}
json.dump(res, open(OUT / "backtest.json", "w"), indent=1)
keep = ["transect", "chainage_m", "left_rate_m_per_yr", "right_rate_m_per_yr"] + [f"{s}_{y}" for s in ("left", "right") for y in YEARS]
tr[keep + ["geometry"]].to_crs(4326).to_file(OUT / "transects.geojson", driver="GeoJSON")
print(f"transects: {len(tr)} | median outward rate left {np.nanmedian(tr.left_rate_m_per_yr):.1f}, right {np.nanmedian(tr.right_rate_m_per_yr):.1f} m/yr")
print("backtest:", res)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(18, 10))
for a_, y in zip(ax, (2016, 2021, 2026)):
    c = classes[y]; img = np.ones((h, w, 3)) * 0.92
    img[inside] = (0.75, 0.82, 0.70); img[c == 2] = (0.93, 0.85, 0.6); img[c == 1] = (0.15, 0.35, 0.8)
    edge = channels[y] & ~ndimage.binary_erosion(channels[y]); img[edge] = (0.85, 0.1, 0.1)
    a_.imshow(img[400:3725, 0:2600]); a_.set_title(f"{y}: water (blue), sand (beige), bank line (red)"); a_.axis("off")
plt.tight_layout(); plt.savefig(ROOT / "outputs" / "quicklook" / "channel_banks.png", dpi=90)
