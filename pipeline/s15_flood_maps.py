"""Flood maps for the four events with the fine-tuned TG-FloodNet ensemble."""
import importlib.util
import json
import os
import time

import geopandas as gpd
import numpy as np
import rasterio
import torch
from rasterio.features import shapes
from rasterio.warp import Resampling, calculate_default_transform, reproject
from shapely.geometry import shape

from common import CRS, DER, MODELS, ROOT, WEIGHTS, grid, teesta_mask, write

GAMMA_TO_SIGMA_DB = -1.0  # HyP3 RTC is gamma0, the training data is sigma0 (cos 37 deg)
WIN, OVL, THR = 512, 64, 0.5
PIXEL_KM2 = 1e-4  # one 10 m pixel
OUT = ROOT / "outputs" / "flood"; OUT.mkdir(parents=True, exist_ok=True)
PROB = DER / "flood_prob"; PROB.mkdir(exist_ok=True)
EVENTS = {"flood_2017_08": ("2017-07-17", "2017-08-10"), "flood_2020_07": ("2020-06-07", "2020-07-13"),
          "flood_2024_07": ("2024-06-10", "2024-07-04"), "flood_2025_08": ("2025-07-23", "2025-08-16")}

spec = importlib.util.spec_from_file_location("tg", MODELS / "flood" / "tgfloodnet.py")
tg = importlib.util.module_from_spec(spec); spec.loader.exec_module(tg)
DEV = tg.DEV
norm = json.load(open(WEIGHTS / "flood" / "norm.json"))
MEAN = np.array(norm["mean"], np.float32)[:, None, None]; STD = np.array(norm["std"], np.float32)[:, None, None]
models = []
for s in (0, 1, 2):
    m = tg.TGFloodNet(pretrained=False).to(DEV).eval()
    m.load_state_dict(torch.load(WEIGHTS / "flood" / f"seed{s}.pt", map_location=DEV))
    models.append(m)

t10, H, W = grid(10)
mask = np.kron(teesta_mask(30), np.ones((3, 3), bool))[:H, :W]


def terrain10(name):
    out = np.zeros((H, W), np.float32)
    with rasterio.open(DER / name) as s:
        reproject(rasterio.band(s, 1), out, dst_transform=t10, dst_crs=CRS, resampling=Resampling.bilinear)
    return np.nan_to_num(out)


hand = ((np.log1p(np.clip(terrain10("hand_dem_30m.tif"), 0, 100)) - MEAN[3, 0, 0]) / STD[3, 0, 0]).astype(np.float32)
dist = ((np.log1p(np.clip(terrain10("dist_stream_30m.tif"), 0, 10000)) - MEAN[4, 0, 0]) / STD[4, 0, 0]).astype(np.float32)


def read_db(date, pol):
    with rasterio.open(DER / "s1_10m" / f"{date}_{pol}.tif") as s:
        a = s.read(1)
    return a.astype(np.float32) / 100, a != -32768


@torch.no_grad()
def predict_date(date):
    vv, ok1 = read_db(date, "VV"); vh, ok2 = read_db(date, "VH")
    ok = ok1 & ok2 & mask
    vv = np.clip(vv + GAMMA_TO_SIGMA_DB, *norm["clip"]); vh = np.clip(vh + GAMMA_TO_SIGMA_DB, *norm["clip"])

    def window(sr0, sr1, sc0, sc1):
        a, b = vv[sr0:sr1, sc0:sc1], vh[sr0:sr1, sc0:sc1]
        z = np.stack([(a - MEAN[0, 0, 0]) / STD[0, 0, 0], (b - MEAN[1, 0, 0]) / STD[1, 0, 0], (a - b - MEAN[2, 0, 0]) / STD[2, 0, 0],
                      hand[sr0:sr1, sc0:sc1], dist[sr0:sr1, sc0:sc1]]).astype(np.float32)
        z[:, ~ok[sr0:sr1, sc0:sc1]] = 0
        return z

    prob = np.zeros((H, W), np.float32); step = WIN - 2 * OVL
    batch, coords = [], []

    def flush():
        xb = torch.from_numpy(np.stack(batch)).to(DEV)
        with torch.autocast(DEV.type, enabled=DEV.type == "cuda"):
            p = sum(torch.sigmoid(m(xb)["water"].float()) for m in models) / len(models)
        for (r0, c0), pi in zip(coords, p[:, 0].cpu().numpy()):
            h, w = min(step, H - r0), min(step, W - c0)
            prob[r0:r0 + h, c0:c0 + w] = pi[OVL:OVL + h, OVL:OVL + w]
        batch.clear(); coords.clear()

    for r0 in range(0, H, step):
        for c0 in range(0, W, step):
            if not ok[r0:r0 + step, c0:c0 + step].any():
                continue
            wr, wc = r0 - OVL, c0 - OVL
            win = np.zeros((5, WIN, WIN), np.float32)
            sr0, sc0 = max(wr, 0), max(wc, 0); sr1, sc1 = min(wr + WIN, H), min(wc + WIN, W)
            win[:, sr0 - wr:sr1 - wr, sc0 - wc:sc1 - wc] = window(sr0, sr1, sc0, sc1)
            batch.append(win); coords.append((r0, c0))
            if len(batch) == 4:
                flush()
    if batch:
        flush()
    prob[~ok] = np.nan
    return prob


def to_4326(arr10, path, nodata=255):
    dst_t, dw, dh = calculate_default_transform(CRS, "EPSG:4326", W, H, *rasterio.transform.array_bounds(H, W, t10))
    out = np.full((dh, dw), nodata, np.uint8)
    reproject(arr10, out, src_transform=t10, src_crs=CRS, src_nodata=nodata, dst_transform=dst_t, dst_crs="EPSG:4326",
              dst_nodata=nodata, resampling=Resampling.nearest)
    with rasterio.open(path, "w", driver="GTiff", height=dh, width=dw, count=1, dtype="uint8", crs="EPSG:4326",
                       transform=dst_t, nodata=nodata, compress="deflate", tiled=True, blockxsize=256, blockysize=256) as d:
        d.write(out, 1)
        d.write_colormap(1, {0: (0, 0, 0, 0), 1: (30, 120, 255, 255)})


water = {}
for date in sorted({d for pair in EVENTS.values() for d in pair}):
    f = PROB / f"{date}_tgft_prob.tif"
    t0 = time.time()
    if f.exists():
        with rasterio.open(f) as s:
            p8 = s.read(1)
    else:
        prob = predict_date(date)
        p8 = np.where(np.isfinite(prob), np.round(np.nan_to_num(prob) * 254), 255).astype(np.uint8)
        write(f, p8, res=10, nodata=255, dtype="uint8")
    water[date] = np.where(p8 != 255, (p8 >= round(THR * 254)).astype(np.uint8), 255).astype(np.uint8)
    print(f"{date}: water {(water[date] == 1).sum() * PIXEL_KM2:.1f} km2 ({time.time() - t0:.0f}s)", flush=True)

summary = {}
for ev, (b, a) in EVENTS.items():
    d = OUT / ev; d.mkdir(exist_ok=True)
    wb, wa = water[b], water[a]
    valid = (wb != 255) & (wa != 255)
    new = valid & (wa == 1) & (wb == 0)
    to_4326(wb, d / "water_before.tif"); to_4326(wa, d / "water_after.tif")
    gdf = gpd.GeoDataFrame(geometry=[shape(g) for g, v in shapes(new.astype(np.uint8), mask=new, transform=t10) if v == 1], crs=CRS)
    gdf = gdf[gdf.area >= 5000].copy()
    gdf["area_ha"] = (gdf.area / 1e4).round(2)
    gdf["geometry"] = gdf.simplify(5)
    gdf.to_crs(4326).to_file(d / "new_water.geojson", driver="GeoJSON")
    st = {"event": ev, "before_date": b, "after_date": a, "model": "TG-FloodNet, 3-model ensemble",
          "area_mapped_km2": round(valid.sum() * PIXEL_KM2, 1),
          "water_before_km2": round(((wb == 1) & valid).sum() * PIXEL_KM2, 2), "water_after_km2": round(((wa == 1) & valid).sum() * PIXEL_KM2, 2),
          "new_water_km2": round(new.sum() * PIXEL_KM2, 2), "new_water_polygons": int(len(gdf))}
    json.dump(st, open(d / "stats.json", "w"), indent=1)
    summary[ev] = st
    print(f"{ev}: before {st['water_before_km2']} km2, after {st['water_after_km2']} km2, new {st['new_water_km2']} km2", flush=True)
json.dump(summary, open(OUT / "flood_summary.json", "w"), indent=1)
