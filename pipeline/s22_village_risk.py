"""Settlement clusters and their first-passage probability of river arrival."""
import json
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from scipy import ndimage
from scipy.stats import norm
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from common import ROOT, DATA, DER, CRS, grid, teesta_mask

OUT = ROOT / "outputs" / "villages"; OUT.mkdir(parents=True, exist_ok=True)
t, h, w = grid(30); inside = teesta_mask(30)
YEARS = list(range(2016, 2027))
rd = lambda f: rasterio.open(DER / "channel" / f).read(1).astype(bool)


def dist_to(mask):
    return ndimage.distance_transform_edt(~mask) * 30.0


def sample(arr, x, y):
    c = np.clip(((x - t.c) / 30).astype(int), 0, w - 1); r = np.clip(((t.f - y) / 30).astype(int), 0, h - 1)
    return arr[r, c]


def first_passage(d, mu, sig, tt):
    d = np.maximum(d, 0.0); mu = np.maximum(mu, 0.0); sig = np.maximum(sig, 5.0); s = sig * np.sqrt(tt)
    a = norm.cdf((mu * tt - d) / s)
    b = np.exp(np.minimum(2 * mu * d / sig ** 2 + norm.logcdf(-(mu * tt + d) / s), 0))
    return np.clip(a + b, 0, 1)


tr = pd.read_csv(DER / "transects_32645.csv")
trg = gpd.read_file(ROOT / "outputs" / "erosion" / "transects.geojson")
edge = {s: trg[[f"{s}_{y}" for y in YEARS]].to_numpy(float) for s in ("left", "right")}


def side_params(years):
    out = {}
    for s in ("left", "right"):
        e = edge[s][:, [YEARS.index(y) for y in years]]
        inc = np.diff(e, axis=1)
        if years[0] == 2016:
            inc = inc[:, 1:]
        mu = np.full(len(e), np.nan); sg = np.full(len(e), np.nan)
        for i in range(len(e)):
            pool = inc[max(0, i - 2):i + 3].ravel(); pool = pool[np.isfinite(pool)]
            if len(pool) >= 3:
                mu[i], sg[i] = max(pool.mean(), 0), max(pool.std(), 5.0)
        mu = pd.Series(mu).fillna(np.nanmedian(mu)).to_numpy(); sg = pd.Series(sg).fillna(np.nanmedian(sg)).to_numpy()
        out[s] = (mu, sg)
    return out


tree = cKDTree(tr[["cx", "cy"]].to_numpy())


def params_at(x, y, prm):
    _, i = tree.query(np.c_[x, y])
    side = np.where((x - tr.cx.values[i]) * tr.nx.values[i] + (y - tr.cy.values[i]) * tr.ny.values[i] < 0, "left", "right")
    mu = np.where(side == "left", prm["left"][0][i], prm["right"][0][i]); sg = np.where(side == "left", prm["left"][1][i], prm["right"][1][i])
    return mu, sg, i, side


rng = np.random.default_rng(0)
corr19, corr25 = rd("corridor_2019.tif"), rd("corridor_2025.tif")
_bl = pd.read_parquet(DATA / "buildings" / "buildings_aoi.parquet")
_bg = gpd.GeoSeries(gpd.points_from_xy(_bl.lon, _bl.lat), crs=4326).to_crs(CRS)
built = np.zeros((h, w), bool)
_c = ((_bg.x.values - t.c) / 30).astype(int); _r = ((t.f - _bg.y.values) / 30).astype(int); _ok = (_c >= 0) & (_c < w) & (_r >= 0) & (_r < h)
built[_r[_ok], _c[_ok]] = True
active = lambda y: np.isin(rasterio.open(DER / "channel" / f"class_{y}.tif").read(1), (1, 2)) & rd(f"channel_{y}.tif") & ~built
d19 = dist_to(active(2019))
cand = np.flatnonzero((inside & ~corr19 & (d19 <= 3000)).ravel())
pick = rng.choice(cand, size=min(40000, len(cand)), replace=False)
r_, c_ = np.unravel_index(pick, (h, w)); px, py = t.c + (c_ + 0.5) * 30, t.f - (r_ + 0.5) * 30
p19 = side_params([2016, 2017, 2018, 2019])
mu, sg, _, _ = params_at(px, py, p19)
prob = first_passage(d19[r_, c_], mu, sg, 6.0)
hit = corr25[r_, c_].astype(float)
from sklearn.isotonic import IsotonicRegression
_, _, tid, _ = params_at(px, py, p19)
calib_half = (tid // 20) % 2 == 0
iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(prob[calib_half], hit[calib_half])
ev = ~calib_half
prob_cal = iso.predict(prob[ev])
bs_raw_ev = float(np.mean((prob[ev] - hit[ev]) ** 2)); bs_cal_ev = float(np.mean((prob_cal - hit[ev]) ** 2))
bs_clim_ev = float(np.mean((hit[calib_half].mean() - hit[ev]) ** 2))
base = np.full_like(prob, hit.mean())
bs, bs0 = float(np.mean((prob - hit) ** 2)), float(np.mean((base - hit) ** 2))
bins = np.array([0, .01, .05, .1, .2, .4, .6, .8, 1.0001])
rel = []
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (prob >= lo) & (prob < hi)
    if m.sum():
        rel.append(dict(bin=f"{lo:.2f}-{min(hi,1):.2f}", n=int(m.sum()), mean_predicted=round(float(prob[m].mean()), 3), observed=round(float(hit[m].mean()), 3)))
from sklearn.metrics import roc_auc_score
bt = dict(fit_years="2016-2019", predict="reached by 2025 (t = 6 yr)", n_points=int(len(prob)), observed_rate=round(float(hit.mean()), 4),
          brier=round(bs, 4), brier_climatology=round(bs0, 4), brier_skill_score=round(1 - bs / bs0, 3),
          auc=round(float(roc_auc_score(hit, prob)), 3), reliability=rel,
          recalibrated=dict(how="isotonic map learned on alternate 5 km stretches, scored on the other stretches",
                            n_eval=int(ev.sum()), brier_raw=round(bs_raw_ev, 4), brier_recalibrated=round(bs_cal_ev, 4),
                            brier_climatology=round(bs_clim_ev, 4), brier_skill_score=round(1 - bs_cal_ev / bs_clim_ev, 3),
                            auc=round(float(roc_auc_score(hit[ev], prob_cal)), 3),
                            reliability=[dict(bin=f"{lo:.2f}-{min(hi,1):.2f}", n=int(m.sum()), mean_predicted=round(float(prob_cal[m].mean()), 3), observed=round(float(hit[ev][m].mean()), 3))
                                         for lo, hi in zip(bins[:-1], bins[1:]) for m in [(prob_cal >= lo) & (prob_cal < hi)] if m.sum()]))
import pickle; pickle.dump(iso, open(DER / "arrival_isotonic.pkl", "wb"))
json.dump(bt, open(OUT / "backtest.json", "w"), indent=1)
print("backtest:", {k: v for k, v in bt.items() if k not in ("reliability", "recalibrated")})
print("recalibrated:", {k: v for k, v in bt["recalibrated"].items() if k != "reliability"})
for r in rel:
    print("   ", r)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.figure(figsize=(5, 5)); plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.plot([r["mean_predicted"] for r in rel], [r["observed"] for r in rel], "o-", label=f"raw (Brier {bs:.3f})")
_rc = bt["recalibrated"]["reliability"]
plt.plot([r["mean_predicted"] for r in _rc], [r["observed"] for r in _rc], "s-", label=f"recalibrated, held-out stretches (Brier {bs_cal_ev:.3f})")
plt.xlabel("predicted P(river reaches the point by 2025)"); plt.ylabel("observed share reached"); plt.title("Reliability, backtest 2019 -> 2025")
plt.legend(); plt.tight_layout(); plt.savefig(ROOT / "outputs" / "quicklook" / "reliability.png", dpi=110)

b = pd.read_parquet(DATA / "buildings" / "buildings_aoi.parquet")
b = b[b.bf_source == "google"].reset_index(drop=True)
g = gpd.GeoDataFrame(b, geometry=gpd.points_from_xy(b.lon, b.lat), crs=4326).to_crs(CRS)
x, y = g.geometry.x.values, g.geometry.y.values
d26 = dist_to(active(2026))
g["dist_m"] = np.round(sample(d26, x, y), 0)
g = g[sample(inside, x, y)].reset_index(drop=True); x, y = g.geometry.x.values, g.geometry.y.values
pnow = side_params(YEARS)
g["rate_m_yr"], g["sigma"], g["transect"], g["side"] = params_at(x, y, pnow)
for tt in (1, 3, 5):
    g[f"p{tt}"] = np.round(first_passage(g.dist_m.values, g.rate_m_yr.values, g.sigma.values, float(tt)), 4)
g["years_to_arrival"] = np.where(g.rate_m_yr >= 0.5, g.dist_m / g.rate_m_yr.clip(lower=0.5), np.inf)
lab = DBSCAN(eps=60, min_samples=10).fit_predict(np.c_[x, y]); g["settlement"] = lab
print(f"houses (Google) in the Teesta area: {len(g):,}; settlements: {lab.max() + 1}; houses outside settlements: {(lab < 0).sum():,}")
osm = gpd.read_file(DATA / "aoi" / "villages_osm.geojson").to_crs(CRS)
otree = cKDTree(np.c_[osm.geometry.x, osm.geometry.y])
rows = []
for sid, grp in g[g.settlement >= 0].groupby("settlement"):
    cx, cy = grp.geometry.x.mean(), grp.geometry.y.mean()
    dd, j = otree.query([cx, cy])
    name = osm.name.iloc[j] if dd <= 2000 else f"Settlement {sid}"
    k = grp.dist_m.idxmin(); d, rate = float(grp.dist_m[k]), float(grp.rate_m_yr[k])
    yrs = d / rate if rate >= 0.5 else float("inf")
    cls = "High" if (d < 400 or yrs < 2) else "Medium" if (d < 1000 or yrs < 5) else "Low"
    lon, lat = gpd.GeoSeries([grp.geometry.iloc[0].__class__(cx, cy)], crs=CRS).to_crs(4326).iloc[0].coords[0]
    rows.append(dict(settlement_id=int(sid), name=name, name_source="OSM (nearest place within 2 km)" if dd <= 2000 else "unnamed cluster",
                     lon=round(lon, 5), lat=round(lat, 5), houses=int(len(grp)), nearest_house_dist_m=round(d), erosion_rate_m_yr=round(rate, 1),
                     years_to_arrival=None if not np.isfinite(yrs) else round(yrs, 1), risk_class=cls,
                     p_reach_1yr=round(float(grp.p1[k]), 4), p_reach_3yr=round(float(grp.p3[k]), 4), p_reach_5yr=round(float(grp.p5[k]), 4),
                     houses_p5_over_50pct=int((grp.p5 > 0.5).sum()), expected_houses_reached_5yr=round(float(grp.p5.sum()), 1)))
v = pd.DataFrame(rows).sort_values(["risk_class", "p_reach_5yr"], key=lambda s: s.map({"High": 0, "Medium": 1, "Low": 2}) if s.name == "risk_class" else -s)
json.dump({"method": "see pipeline/s22_village_risk.py", "classes": "High: d<400 m or <2 yr; Medium: d<1 km or <5 yr; Low: otherwise",
           "summary": v.risk_class.value_counts().to_dict(), "settlements": v.to_dict(orient="records")},
          open(OUT / "villages_risk.json", "w"), indent=1, default=float)
hr = g[g.p5 >= 0.05].copy(); hr["settlement_name"] = hr.settlement.map(dict(zip(v.settlement_id, v.name)))
hr[["dist_m", "rate_m_yr", "years_to_arrival", "p1", "p3", "p5", "settlement_name", "geometry"]].replace([np.inf], np.nan).to_crs(4326) \
    .to_file(OUT / "houses_at_risk.geojson", driver="GeoJSON")
print("settlement classes:", v.risk_class.value_counts().to_dict())
print(f"houses with P(reached within 5 yr) >= 5 %: {len(hr):,}; >= 50 %: {(g.p5 >= 0.5).sum():,}; expected houses reached in 5 yr: {g.p5.sum():,.0f} (raw model)")
print(v.head(10)[["name", "houses", "nearest_house_dist_m", "erosion_rate_m_yr", "years_to_arrival", "risk_class", "p_reach_5yr"]].to_string(index=False))
