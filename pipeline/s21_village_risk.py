"""Settlement clusters and their first-passage probability of river arrival."""
import json
import pickle

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.stats import norm
from sklearn.cluster import DBSCAN
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from common import CRS, DATA, DER, ROOT, grid, teesta_mask

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PIXEL_M = 30
YEARS = list(range(2016, 2027))
BACKTEST_FIT_YEARS = [2016, 2017, 2018, 2019]
BACKTEST_HORIZON_YR = 6.0              # 2019 -> 2025
BACKTEST_POINTS = 40000
BACKTEST_MAX_DIST_M = 3000
MIN_SIGMA_M = 5.0                      # floor on the yearly spread of bank movement
MIN_RATE_M_YR = 0.5                    # slower banks are treated as stable
NEIGHBOUR_TRANSECTS = 2                # rates are pooled over +-2 transects (+-500 m)
SETTLEMENT_EPS_M, SETTLEMENT_MIN_HOMES = 60, 10
OSM_NAME_RADIUS_M = 2000
HIGH = dict(dist_m=400, years=2)
MEDIUM = dict(dist_m=1000, years=5)
HOUSE_EXPORT_MIN_P5 = 0.05

OUT = ROOT / "outputs" / "villages"; OUT.mkdir(parents=True, exist_ok=True)
QUICKLOOK = ROOT / "outputs" / "quicklook"; QUICKLOOK.mkdir(parents=True, exist_ok=True)
t, h, w = grid(PIXEL_M)
inside = teesta_mask(PIXEL_M)


def read_mask(name):
    return rasterio.open(DER / "channel" / name).read(1).astype(bool)


def distance_to(mask):
    return ndimage.distance_transform_edt(~mask) * PIXEL_M


def cell_index(x, y):
    return ((t.f - y) / PIXEL_M).astype(int), ((x - t.c) / PIXEL_M).astype(int)


def sample(arr, x, y):
    r, c = cell_index(x, y)
    return arr[np.clip(r, 0, h - 1), np.clip(c, 0, w - 1)]


def first_passage(dist, rate, sigma, years):
    """P(a bank drifting at `rate` with spread `sigma` covers `dist` within `years`): inverse-Gaussian CDF."""
    dist = np.maximum(dist, 0.0); rate = np.maximum(rate, 0.0); sigma = np.maximum(sigma, MIN_SIGMA_M)
    s = sigma * np.sqrt(years)
    a = norm.cdf((rate * years - dist) / s)
    b = np.exp(np.minimum(2 * rate * dist / sigma ** 2 + norm.logcdf(-(rate * years + dist) / s), 0))
    return np.clip(a + b, 0, 1)


transects = pd.read_csv(DER / "transects_32645.csv")
bank_edges = gpd.read_file(ROOT / "outputs" / "erosion" / "transects.geojson")
edge = {side: bank_edges[[f"{side}_{y}" for y in YEARS]].to_numpy(float) for side in ("left", "right")}
transect_tree = cKDTree(transects[["cx", "cy"]].to_numpy())


def bank_rates(years):
    """Mean and spread of yearly outward bank movement per transect and side."""
    out = {}
    for side in ("left", "right"):
        moves = np.diff(edge[side][:, [YEARS.index(y) for y in years]], axis=1)
        if years[0] == 2016:
            moves = moves[:, 1:]  # corridor_2016 is one dry season; later corridors accumulate, so 2016->2017 is skipped
        rate = np.full(len(moves), np.nan); sigma = np.full(len(moves), np.nan)
        for i in range(len(moves)):
            pool = moves[max(0, i - NEIGHBOUR_TRANSECTS):i + NEIGHBOUR_TRANSECTS + 1].ravel()
            pool = pool[np.isfinite(pool)]
            if len(pool) >= 3:
                rate[i], sigma[i] = max(pool.mean(), 0), max(pool.std(), MIN_SIGMA_M)
        out[side] = (pd.Series(rate).fillna(np.nanmedian(rate)).to_numpy(), pd.Series(sigma).fillna(np.nanmedian(sigma)).to_numpy())
    return out


def rates_at(x, y, rates):
    """Rate and spread of the nearest transect, on the bank side the point lies on."""
    _, i = transect_tree.query(np.c_[x, y])
    along_normal = (x - transects.cx.values[i]) * transects.nx.values[i] + (y - transects.cy.values[i]) * transects.ny.values[i]
    side = np.where(along_normal < 0, "left", "right")
    rate = np.where(side == "left", rates["left"][0][i], rates["right"][0][i])
    sigma = np.where(side == "left", rates["left"][1][i], rates["right"][1][i])
    return rate, sigma, i, side


def load_buildings(google_only=False):
    b = pd.read_parquet(DATA / "buildings" / "buildings_aoi.parquet")
    if google_only:
        b = b[b.bf_source == "google"].reset_index(drop=True)
    return gpd.GeoDataFrame(b, geometry=gpd.points_from_xy(b.lon, b.lat), crs=4326).to_crs(CRS)


def built_cells(buildings):
    r, c = cell_index(buildings.geometry.x.values, buildings.geometry.y.values)
    ok = (r >= 0) & (r < h) & (c >= 0) & (c < w)
    built = np.zeros((h, w), bool); built[r[ok], c[ok]] = True
    return built


def active_channel(year, built):
    water_or_sand = np.isin(rasterio.open(DER / "channel" / f"class_{year}.tif").read(1), (1, 2))
    return water_or_sand & read_mask(f"channel_{year}.tif") & ~built


def reliability(pred, hit):
    bins = [0, .01, .05, .1, .2, .4, .6, .8, 1.0001]
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (pred >= lo) & (pred < hi)
        if m.sum():
            rows.append(dict(bin=f"{lo:.2f}-{min(hi, 1):.2f}", n=int(m.sum()), mean_predicted=round(float(pred[m].mean()), 3),
                             observed=round(float(hit[m].mean()), 3)))
    return rows


def brier(pred, hit):
    return float(np.mean((pred - hit) ** 2))


def backtest(built):
    """Fit bank rates on 2016-2019, predict which land the river reaches by 2025, score on random points."""
    rng = np.random.default_rng(0)
    reached_2019, reached_2025 = read_mask("corridor_2019.tif"), read_mask("corridor_2025.tif")
    dist_2019 = distance_to(active_channel(2019, built))
    candidates = np.flatnonzero((inside & ~reached_2019 & (dist_2019 <= BACKTEST_MAX_DIST_M)).ravel())
    pick = rng.choice(candidates, size=min(BACKTEST_POINTS, len(candidates)), replace=False)
    r, c = np.unravel_index(pick, (h, w))
    px, py = t.c + (c + 0.5) * PIXEL_M, t.f - (r + 0.5) * PIXEL_M
    rate, sigma, transect_id, _ = rates_at(px, py, bank_rates(BACKTEST_FIT_YEARS))
    prob = first_passage(dist_2019[r, c], rate, sigma, BACKTEST_HORIZON_YR)
    hit = reached_2025[r, c].astype(float)

    calib = (transect_id // 20) % 2 == 0  # alternate 5 km stretches
    held = ~calib
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(prob[calib], hit[calib])
    prob_cal = iso.predict(prob[held])
    climatology = np.full_like(prob, hit.mean())
    bs, bs0 = brier(prob, hit), brier(climatology, hit)
    bs_raw, bs_cal = brier(prob[held], hit[held]), brier(prob_cal, hit[held])
    bs_clim = float(np.mean((hit[calib].mean() - hit[held]) ** 2))
    result = dict(
        fit_years="2016-2019", predict="reached by 2025 (t = 6 yr)", n_points=int(len(prob)), observed_rate=round(float(hit.mean()), 4),
        brier=round(bs, 4), brier_climatology=round(bs0, 4), brier_skill_score=round(1 - bs / bs0, 3),
        auc=round(float(roc_auc_score(hit, prob)), 3), reliability=reliability(prob, hit),
        recalibrated=dict(how="isotonic map learned on alternate 5 km stretches, scored on the other stretches",
                          n_eval=int(held.sum()), brier_raw=round(bs_raw, 4), brier_recalibrated=round(bs_cal, 4),
                          brier_climatology=round(bs_clim, 4), brier_skill_score=round(1 - bs_cal / bs_clim, 3),
                          auc=round(float(roc_auc_score(hit[held], prob_cal)), 3), reliability=reliability(prob_cal, hit[held])))
    pickle.dump(iso, open(DER / "arrival_isotonic.pkl", "wb"))
    json.dump(result, open(OUT / "backtest.json", "w"), indent=1)
    plot_reliability(result, bs, bs_cal)
    print("backtest:", {k: v for k, v in result.items() if k not in ("reliability", "recalibrated")})
    return result


def plot_reliability(result, bs, bs_cal):
    plt.figure(figsize=(5, 5)); plt.plot([0, 1], [0, 1], "k--", lw=1)
    for rows, style, label in ((result["reliability"], "o-", f"raw (Brier {bs:.3f})"),
                               (result["recalibrated"]["reliability"], "s-", f"recalibrated, held-out stretches (Brier {bs_cal:.3f})")):
        plt.plot([r["mean_predicted"] for r in rows], [r["observed"] for r in rows], style, label=label)
    plt.xlabel("predicted P(river reaches the point by 2025)"); plt.ylabel("observed share reached")
    plt.title("Reliability, backtest 2019 -> 2025"); plt.legend(); plt.tight_layout()
    plt.savefig(QUICKLOOK / "reliability.png", dpi=110)


def risk_class(dist_m, years):
    if dist_m < HIGH["dist_m"] or years < HIGH["years"]:
        return "High"
    if dist_m < MEDIUM["dist_m"] or years < MEDIUM["years"]:
        return "Medium"
    return "Low"


def score_homes(built):
    homes = load_buildings(google_only=True)
    x, y = homes.geometry.x.values, homes.geometry.y.values
    homes["dist_m"] = np.round(sample(distance_to(active_channel(2026, built)), x, y), 0)
    homes = homes[sample(inside, x, y)].reset_index(drop=True)
    x, y = homes.geometry.x.values, homes.geometry.y.values
    homes["rate_m_yr"], homes["sigma"], homes["transect"], homes["side"] = rates_at(x, y, bank_rates(YEARS))
    for years in (1, 3, 5):
        homes[f"p{years}"] = np.round(first_passage(homes.dist_m.values, homes.rate_m_yr.values, homes.sigma.values, float(years)), 4)
    homes["years_to_arrival"] = np.where(homes.rate_m_yr >= MIN_RATE_M_YR, homes.dist_m / homes.rate_m_yr.clip(lower=MIN_RATE_M_YR), np.inf)
    homes["settlement"] = DBSCAN(eps=SETTLEMENT_EPS_M, min_samples=SETTLEMENT_MIN_HOMES).fit_predict(np.c_[x, y])
    print(f"homes in the Teesta area: {len(homes):,}; settlements: {homes.settlement.max() + 1}")
    return homes


def summarise_settlements(homes):
    osm = gpd.read_file(DATA / "aoi" / "villages_osm.geojson").to_crs(CRS)
    osm_tree = cKDTree(np.c_[osm.geometry.x, osm.geometry.y])
    rows = []
    for sid, grp in homes[homes.settlement >= 0].groupby("settlement"):
        cx, cy = grp.geometry.x.mean(), grp.geometry.y.mean()
        osm_dist, j = osm_tree.query([cx, cy])
        named = osm_dist <= OSM_NAME_RADIUS_M
        k = grp.dist_m.idxmin()
        dist, rate = float(grp.dist_m[k]), float(grp.rate_m_yr[k])
        years = dist / rate if rate >= MIN_RATE_M_YR else float("inf")
        lon, lat = gpd.points_from_xy([cx], [cy], crs=CRS).to_crs(4326)[0].coords[0]
        rows.append(dict(settlement_id=int(sid), name=osm.name.iloc[j] if named else f"Settlement {sid}",
                         name_source="OSM (nearest place within 2 km)" if named else "unnamed cluster",
                         lon=round(lon, 5), lat=round(lat, 5), houses=int(len(grp)), nearest_house_dist_m=round(dist),
                         erosion_rate_m_yr=round(rate, 1), years_to_arrival=round(years, 1) if np.isfinite(years) else None,
                         risk_class=risk_class(dist, years), p_reach_1yr=round(float(grp.p1[k]), 4),
                         p_reach_3yr=round(float(grp.p3[k]), 4), p_reach_5yr=round(float(grp.p5[k]), 4),
                         houses_p5_over_50pct=int((grp.p5 > 0.5).sum()), expected_houses_reached_5yr=round(float(grp.p5.sum()), 1)))
    order = {"High": 0, "Medium": 1, "Low": 2}
    return pd.DataFrame(rows).sort_values(["risk_class", "p_reach_5yr"], key=lambda s: s.map(order) if s.name == "risk_class" else -s)


built = built_cells(load_buildings())
backtest(built)
homes = score_homes(built)
settlements = summarise_settlements(homes)
json.dump({"method": "see pipeline/s21_village_risk.py", "classes": "High: d<400 m or <2 yr; Medium: d<1 km or <5 yr; Low: otherwise",
           "summary": settlements.risk_class.value_counts().to_dict(), "settlements": settlements.to_dict(orient="records")},
          open(OUT / "villages_risk.json", "w"), indent=1, default=float)
at_risk = homes[homes.p5 >= HOUSE_EXPORT_MIN_P5].copy()
at_risk["settlement_name"] = at_risk.settlement.map(dict(zip(settlements.settlement_id, settlements.name)))
cols = ["dist_m", "rate_m_yr", "years_to_arrival", "p1", "p3", "p5", "settlement_name", "geometry"]
at_risk[cols].replace([np.inf], np.nan).to_crs(4326).to_file(OUT / "houses_at_risk.geojson", driver="GeoJSON")
print("settlement classes:", settlements.risk_class.value_counts().to_dict())
print(f"homes with P(reached within 5 yr) >= 5 %: {len(at_risk):,}; >= 50 %: {(homes.p5 >= 0.5).sum():,}")
