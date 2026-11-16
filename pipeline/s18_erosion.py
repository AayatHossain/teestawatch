"""Land newly reached by the river each year, 2016-2026."""
import json
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio.features import shapes
from shapely.geometry import shape, LineString
from shapely.ops import linemerge, unary_union
from scipy import ndimage
from common import ROOT, DER, CRS, RIVER, grid, teesta_mask, write

YEARS = list(range(2016, 2027))
OUT = ROOT / "outputs" / "erosion"
t, h, w = grid(30)
inside = teesta_mask(30)
chan = {y: rasterio.open(DER / "channel" / f"channel_{y}.tif").read(1).astype(bool) for y in YEARS}
corr, acc = {}, np.zeros((h, w), bool)
for y in YEARS:
    acc = ndimage.binary_fill_holes(acc | chan[y]) & inside
    corr[y] = acc.copy()
    write(DER / "channel" / f"corridor_{y}.tif", acc.astype(np.uint8))
loss = [dict(year=y, newly_eroded_ha=round(float((corr[y] & ~corr[YEARS[i - 1]]).sum() * 0.09), 1),
             corridor_km2=round(float(corr[y].sum() * 9e-4), 1), active_channel_km2=round(float(chan[y].sum() * 9e-4), 1))
        for i, y in enumerate(YEARS) if i]
json.dump({"method": "land reached by the river for the first time in the 2016-2026 record (corridor(y) - corridor(y-1)); "
                     "active channel = dry-season water + bare sand (HLS); see pipeline/s18_erosion.py",
           "unit": "ha", "area": "Teesta in Bangladesh (OSM border), Brahmaputra side excluded", "years": loss},
          open(OUT / "loss_by_year.json", "w"), indent=1)
print("newly eroded land (ha/yr):", {r["year"]: r["newly_eroded_ha"] for r in loss})
for y in YEARS:
    poly = unary_union([shape(g) for g, v in shapes(corr[y].astype(np.uint8), mask=corr[y], transform=t) if v == 1])
    gpd.GeoDataFrame({"year": [y], "corridor_km2": [round(corr[y].sum() * 9e-4, 1)]}, geometry=[poly.exterior if poly.geom_type == "Polygon" else poly.boundary],
                     crs=CRS).to_crs(4326).to_file(OUT / f"corridor_{y}.geojson", driver="GeoJSON")

line = linemerge(RIVER.geometry.union_all())
if line.geom_type == "MultiLineString":
    line = max(line.geoms, key=lambda g: g.length)
ref = line.simplify(300)
rows = []
ds = np.arange(-4000, 4001, 15.0)
for k, s0 in enumerate(np.arange(0, ref.length, 250.0)):
    p = ref.interpolate(s0); q = ref.interpolate(min(s0 + 50, ref.length)); q0 = ref.interpolate(max(s0 - 50, 0))
    dx, dy = q.x - q0.x, q.y - q0.y; L = np.hypot(dx, dy)
    if L == 0:
        continue
    nx, ny = -dy / L, dx / L
    xs, ys = p.x + ds * nx, p.y + ds * ny
    cols = ((xs - t.c) / 30).astype(int); rws = ((t.f - ys) / 30).astype(int)
    okp = (cols >= 0) & (cols < w) & (rws >= 0) & (rws < h)
    ins = np.zeros(len(ds), bool); ins[okp] = inside[rws[okp], cols[okp]]
    rec = dict(transect=k, chainage_m=round(float(s0)), cx=p.x, cy=p.y, nx=nx, ny=ny,
               geometry=LineString([(xs[0], ys[0]), (xs[-1], ys[-1])]))
    for y in YEARS:
        on = np.zeros(len(ds), bool); on[okp] = corr[y][rws[okp], cols[okp]]
        for side, sel in (("left", ds < 0), ("right", ds > 0)):
            hit = on & sel
            v = float(np.abs(ds[hit]).max()) if hit.any() else np.nan
            if np.isfinite(v):
                beyond = sel & (np.abs(ds) > v) & (np.abs(ds) <= v + 300)
                if not ins[beyond].all():
                    v = np.nan
            rec[f"{side}_{y}"] = v
    rows.append(rec)
tr = gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS)


def fit(yrs, vals):
    yrs, vals = np.array(yrs, float), np.array(vals, float); ok = np.isfinite(vals)
    return np.polyfit(yrs[ok], vals[ok], 1) if ok.sum() >= 4 else (np.nan, np.nan)


bt = []
for side in ("left", "right"):
    rate = []
    for _, r in tr.iterrows():
        rate.append(fit(YEARS, [r[f"{side}_{y}"] for y in YEARS])[0])
        b, a = fit([y for y in YEARS if y <= 2022], [r[f"{side}_{y}"] for y in YEARS if y <= 2022])
        o, l = r[f"{side}_2025"], r[f"{side}_2022"]
        if np.isfinite(b) and np.isfinite(o) and np.isfinite(l):
            bt.append(dict(pred=a + b * 2025, nochange=l, obs=o))
    tr[f"{side}_rate_m_per_yr"] = np.round(rate, 1)
bt = pd.DataFrame(bt)
res = {"fit_years": "2016-2022", "predict_year": 2025, "n_bank_points": int(len(bt)),
       "linear_trend_MAE_m": round(float((bt.pred - bt.obs).abs().mean()), 1),
       "linear_trend_median_AE_m": round(float((bt.pred - bt.obs).abs().median()), 1),
       "no_change_MAE_m": round(float((bt.nochange - bt.obs).abs().mean()), 1),
       "no_change_median_AE_m": round(float((bt.nochange - bt.obs).abs().median()), 1),
       "share_within_100m_trend": round(float(((bt.pred - bt.obs).abs() <= 100).mean()), 3),
       "share_within_100m_nochange": round(float(((bt.nochange - bt.obs).abs() <= 100).mean()), 3),
       "note": "bank = outermost corridor edge (land the river has reached since 2016) per side of 250 m transects"}
json.dump(res, open(OUT / "backtest.json", "w"), indent=1)
keep = ["transect", "chainage_m", "left_rate_m_per_yr", "right_rate_m_per_yr"] + [f"{s}_{y}" for s in ("left", "right") for y in YEARS]
tr[keep + ["geometry"]].to_crs(4326).to_file(OUT / "transects.geojson", driver="GeoJSON")
tr[["transect", "chainage_m", "cx", "cy", "nx", "ny", "left_rate_m_per_yr", "right_rate_m_per_yr", "left_2026", "right_2026"]].to_csv(DER / "transects_32645.csv", index=False)
r_all = pd.concat([tr.left_rate_m_per_yr, tr.right_rate_m_per_yr])
print(f"transects {len(tr)} | outward corridor rate: median {np.nanmedian(r_all):.1f} m/yr, 90th pct {np.nanpercentile(r_all, 90):.1f} m/yr, max {np.nanmax(r_all):.0f} m/yr")
print("backtest:", res)
