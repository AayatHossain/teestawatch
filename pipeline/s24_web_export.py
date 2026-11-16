"""Export maps, villages and headline numbers as PNG overlays, GeoJSON and stats.json for the website."""
import glob
import json
import os
from pathlib import Path

import numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform, transform_bounds
from PIL import Image
from pyproj import Transformer
from scipy import ndimage as ndi

from common import CRS, DATA, DER, ROOT, grid, teesta_mask

WEB = Path(os.environ.get("TW_WEB_DATA", ROOT / "web_data"))
for sub in ("flood", "river"):
    (WEB / sub).mkdir(parents=True, exist_ok=True)
RES3857 = 25.0
corners_all = {}


def to_png(arr, src_t, colors, path, nodata=0):
    h, w = arr.shape
    left, bottom, right, top = rasterio.transform.array_bounds(h, w, src_t)
    dt, dw, dh = calculate_default_transform(CRS, "EPSG:3857", w, h, left, bottom, right, top, resolution=RES3857)
    out = np.zeros((dh, dw), np.uint8)
    reproject(arr, out, src_transform=src_t, src_crs=CRS, src_nodata=None, dst_transform=dt, dst_crs="EPSG:3857", resampling=Resampling.nearest)
    rgba = np.zeros((dh, dw, 4), np.uint8)
    for k, c in colors.items():
        rgba[out == k] = c
    Image.fromarray(rgba, "RGBA").save(path, optimize=True)
    l, b, r, t = rasterio.transform.array_bounds(dh, dw, dt)
    lon0, lat0, lon1, lat1 = transform_bounds("EPSG:3857", "EPSG:4326", l, b, r, t)
    return [[lon0, lat1], [lon1, lat1], [lon1, lat0], [lon0, lat0]]  # MapLibre image order: TL, TR, BR, BL


t10, H, W = grid(10)
EVENTS = {"flood_2017_08": ("2017-07-17", "2017-08-10", "August 2017"), "flood_2020_07": ("2020-06-07", "2020-07-13", "July 2020"),
          "flood_2024_07": ("2024-06-10", "2024-07-04", "July 2024"), "flood_2025_08": ("2025-07-23", "2025-08-16", "August 2025")}
flood_stats = []
for ev, (b, a, label) in EVENTS.items():
    def water(d):
        p = rasterio.open(DER / "flood_prob" / f"{d}_tgft_prob.tif").read(1)
        return (p != 255) & (p >= 127)
    wb, wa = water(b), water(a)
    cls = np.zeros((H, W), np.uint8); cls[wb & wa] = 1; cls[wa & ~wb] = 2
    h2, w2 = H // 2, W // 2
    c2 = cls[:h2 * 2, :w2 * 2].reshape(h2, 2, w2, 2).max((1, 3))
    t20 = rasterio.transform.Affine(20, 0, t10.c, 0, -20, t10.f)
    corners_all[f"flood/{ev}.png"] = to_png(c2, t20, {1: (37, 99, 235, 190), 2: (34, 211, 238, 235)}, WEB / "flood" / f"{ev}.png")
    cb = wb[:h2 * 2, :w2 * 2].reshape(h2, 2, w2, 2).max((1, 3)).astype(np.uint8)
    corners_all[f"flood/{ev}_before.png"] = to_png(cb, t20, {1: (37, 99, 235, 190)}, WEB / "flood" / f"{ev}_before.png")
    st = json.load(open(ROOT / "outputs" / "flood" / ev / "stats.json"))
    flood_stats.append(dict(id=ev, label=label, before=b, after=a, water_before_km2=round(st["water_before_km2"]), water_after_km2=round(st["water_after_km2"]),
                            new_water_km2=round(st["new_water_km2"]), image=f"data/flood/{ev}.png"))
    print(ev, flood_stats[-1], flush=True)

t30, h30, w30 = grid(30)
YEARS = list(range(2016, 2027))
c2016 = rasterio.open(DER / "channel" / "corridor_2016.tif").read(1).astype(bool)
for y in YEARS:
    ch = rasterio.open(DER / "channel" / f"channel_{y}.tif").read(1).astype(bool)
    co = rasterio.open(DER / "channel" / f"corridor_{y}.tif").read(1).astype(bool)
    cls = np.zeros((h30, w30), np.uint8); cls[co & ~c2016 & ~ch] = 2; cls[ch] = 1
    corners_all[f"river/{y}.png"] = to_png(cls, t30, {1: (56, 189, 248, 200), 2: (249, 115, 22, 215)}, WEB / "river" / f"{y}.png")
ch26 = rasterio.open(DER / "channel" / "channel_2026.tif").read(1).astype(np.uint8)
corners_all["river/channel_2026.png"] = to_png(ch26, t30, {1: (56, 189, 248, 170)}, WEB / "river" / "channel_2026.png")
risk = rasterio.open(DER / "channel" / "erosion_risk_2027.tif").read(1)
rk = np.where(risk == 255, 0, risk).astype(np.uint8)
corners_all["river/forecast_2027.png"] = to_png(rk, t30, {3: (239, 68, 68, 245), 2: (250, 204, 21, 220)}, WEB / "river" / "forecast_2027.png")
# display-only halo so thin risk strips stay visible when zoomed out
halo = np.zeros_like(rk); hi = ndi.binary_dilation(rk == 3, iterations=4); md = ndi.binary_dilation(rk >= 2, iterations=4)
halo[md] = 2; halo[hi] = 3
corners_all["river/forecast_2027_halo.png"] = to_png(halo, t30, {3: (239, 68, 68, 95), 2: (250, 204, 21, 70)}, WEB / "river" / "forecast_2027_halo.png")
dens = ndi.uniform_filter((rk == 3).astype(np.float32), 500)
fr, fc_ = np.unravel_index(np.argmax(dens), dens.shape)
FOCUS = list(Transformer.from_crs(CRS, 4326, always_xy=True).transform(t30.c + (fc_ + 0.5) * 30, t30.f - (fr + 0.5) * 30))
loss = json.load(open(ROOT / "outputs" / "erosion" / "loss_by_year.json"))["years"]
fc = json.load(open(ROOT / "outputs" / "erosion" / "forecast_2027.json"))
line = gpd.read_file(ROOT / "outputs" / "erosion" / "banks_2026.geojson").to_crs(4326)
line["geometry"] = line.to_crs(CRS).simplify(30).to_crs(4326)
line[["geometry"]].to_file(WEB / "river_line.geojson", driver="GeoJSON", COORDINATE_PRECISION=5)

vr = json.load(open(ROOT / "outputs" / "villages" / "villages_risk.json"))
s = pd.DataFrame(vr["settlements"])
g = gpd.GeoDataFrame(s, geometry=gpd.points_from_xy(s.lon, s.lat), crs=4326)
for L, col in (("ADM4", "union"), ("ADM3", "upazila"), ("ADM2", "district")):
    adm = gpd.read_file(DATA / "admin" / f"bgd_{L}.geojson")[["shapeName", "geometry"]].rename(columns={"shapeName": col}).to_crs(4326)
    g = gpd.sjoin_nearest(g.to_crs(CRS), adm.to_crs(CRS), how="left", max_distance=3000).drop(columns="index_right").to_crs(4326)
    g = g[~g.index.duplicated()]
named = g.name_source != "unnamed cluster"
g["display_name"] = np.where(named, g["name"], g["union"].fillna("Teesta") + " settlement")
rk = g.groupby("display_name")["houses"].rank(method="first", ascending=False).astype(int)
cnt = g.groupby("display_name")["houses"].transform("size")
g["display_name"] = np.where(cnt > 1, g["display_name"] + " " + rk.astype(str), g["display_name"])
keep = ["settlement_id", "display_name", "union", "upazila", "district", "houses", "risk_class", "p_reach_1yr", "p_reach_3yr", "p_reach_5yr",
        "nearest_house_dist_m", "years_to_arrival", "expected_houses_reached_5yr"]
v = g[keep + ["geometry"]].rename(columns={"settlement_id": "id", "display_name": "name", "risk_class": "risk", "p_reach_1yr": "p1",
                                           "p_reach_3yr": "p3", "p_reach_5yr": "p5", "nearest_house_dist_m": "dist_m",
                                           "years_to_arrival": "years", "expected_houses_reached_5yr": "exp5"})
for c in ("p1", "p3", "p5"):
    v[c] = v[c].round(3)
v["years"] = v["years"].round(1); v["exp5"] = v["exp5"].round(1)
v = v.fillna({"union": "", "upazila": "", "district": ""})
v.to_file(WEB / "villages.geojson", driver="GeoJSON", COORDINATE_PRECISION=5)
hz = gpd.read_file(ROOT / "outputs" / "villages" / "houses_at_risk.geojson")[["p5", "geometry"]]
hz["p5"] = hz["p5"].round(2)
hz.to_file(WEB / "houses.geojson", driver="GeoJSON", COORDINATE_PRECISION=5)
print("villages", len(v), "| named by OSM", int(named.sum()), "| districts", v.district.value_counts().head(6).to_dict(), flush=True)

def houses_in_teesta():
    bb = pd.read_parquet(DATA / "buildings" / "buildings_aoi.parquet"); bb = bb[bb.bf_source == "google"]
    pts = gpd.GeoSeries(gpd.points_from_xy(bb.lon, bb.lat), crs=4326).to_crs(CRS)
    m = teesta_mask(30); cols = ((pts.x - t30.c) / 30).astype(int).values; rws = ((t30.f - pts.y) / 30).astype(int).values
    ok = (rws >= 0) & (rws < h30) & (cols >= 0) & (cols < w30)
    return int(m[rws[ok], cols[ok]].sum())


he = json.load(open(ROOT / "outputs" / "report" / "hand_label_eval.json"))["all"]["TG-FloodNet"]
met = json.load(open(ROOT / "results" / "metrics.json"))
stats = {
    "area_km2": 1609, "years_of_data": "2016–2026", "satellite_images": None,
    "houses_analysed": houses_in_teesta(),
    "settlements": len(v), "risk_counts": vr["summary"],
    "houses_p5_over_50": int((hz.p5 >= 0.5).sum()), "houses_p5_over_5": int(len(hz)),
    "expected_houses_reached_5yr": round(float(hz.p5.sum())), "houses_in_settlements": int(s.houses.sum()),
    "floods": flood_stats,
    "land_lost_by_year": [dict(year=r["year"], ha=round(r["newly_eroded_ha"])) for r in loss],
    "land_lost_total_ha": round(sum(r["newly_eroded_ha"] for r in loss)),
    "forecast_2027": dict(high_ha=round(fc["high_ha"]), medium_ha=round(fc["medium_ha"]), expected_ha=round(fc["expected_newly_eroded_ha"]), focus=[round(FOCUS[0], 4), round(FOCUS[1], 4)]),
    "accuracy": {
        "flood_f1": round(he["f1"], 3), "flood_iou": round(he["iou"], 3),
        "flood_hand_tiles": 50,
        "erosion_auc": [round(met["erosion_forecast"][y]["auc"], 2) for y in ("2025", "2026")],
        "erosion_x_chance": [round(met["erosion_forecast"][y]["times_chance"]) for y in ("2025", "2026")],
        "village_risk_auc": met["village_risk"]["auc"],
    },
    "overlays": corners_all,
}
stats["satellite_images"] = len(glob.glob(str(DATA / "hls" / "*Fmask.tif"))) + len(glob.glob(str(DATA / "hls_flood" / "*Fmask.tif"))) + 16
json.dump(stats, open(WEB / "stats.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print({k: stats[k] for k in ("houses_analysed", "settlements", "risk_counts", "houses_p5_over_50", "expected_houses_reached_5yr", "land_lost_total_ha", "forecast_2027", "accuracy", "satellite_images")})
