"""2027 erosion risk map from the ConvGRU U-Net ensemble trained on all years 2018-2026."""
import importlib.util
import json
import os

import geopandas as gpd
import numpy as np
import rasterio
import torch
from rasterio.features import shapes
from shapely.geometry import shape

from common import CRS, DATA, DER, MODELS, ROOT, WEIGHTS, grid, write

HIGH, MEDIUM = 0.664, 0.221  # F1-best threshold on 2024 and one third of it
os.environ.setdefault("TW_EROSION_DATA", str(DATA / "erosion_pack"))
os.environ.setdefault("TW_OUT", str(ROOT / "cache" / "erosion_infer"))
spec = importlib.util.spec_from_file_location("convgru", MODELS / "erosion" / "convgru_unet.py")
cg = importlib.util.module_from_spec(spec); spec.loader.exec_module(cg)

probs = []
for s in (0, 1, 2):
    net = cg.Net().to(cg.DEV)
    net.load_state_dict(torch.load(WEIGHTS / "erosion" / f"seed{s}.pt", map_location=cg.DEV))
    probs.append(np.round(cg.predict(net, 2027)[1] * 255))  # 8-bit probabilities, as stored by the training run
risk_p = np.round(np.mean(probs, 0)) / 255

zn = cg.zone(2026)
cls = np.zeros(cg.INSIDE.shape, np.uint8); cls[zn] = 1
cls[zn & (risk_p >= MEDIUM)] = 2; cls[zn & (risk_p >= HIGH)] = 3

r0, c0 = map(int, np.load(DATA / "erosion_pack" / "stacks.npz")["origin"])
t30, h30, w30 = grid(30)
full = np.full((h30, w30), 255, np.uint8); full[r0:r0 + cls.shape[0], c0:c0 + cls.shape[1]] = np.where(cg.INSIDE, cls, 255)
(DER / "channel").mkdir(exist_ok=True)
write(DER / "channel" / "erosion_risk_2027.tif", full, nodata=255)

tt = rasterio.transform.Affine(t30.a, 0, t30.c + c0 * 30, 0, t30.e, t30.f - r0 * 30)
rows = [dict(risk={3: "High", 2: "Medium"}[int(v)], geometry=shape(g)) for g, v in shapes(cls, mask=cls >= 2, transform=tt) if v >= 2]
gdf = gpd.GeoDataFrame(rows, crs=CRS); gdf["area_ha"] = (gdf.area / 1e4).round(2)
out = ROOT / "outputs" / "erosion"; out.mkdir(parents=True, exist_ok=True)
gdf.to_crs(4326).to_file(out / "forecast_2027.geojson", driver="GeoJSON")
summary = {"year": 2027, "model": "ConvGRU U-Net, 3-model ensemble",
           "high_ha": round(float((cls == 3).sum() * 0.09), 1), "medium_ha": round(float((cls == 2).sum() * 0.09), 1),
           "expected_newly_eroded_ha": round(float(risk_p[zn].sum() * 0.09), 1),
           "classes": {"High": f"p >= {HIGH}", "Medium": f"p >= {MEDIUM}", "Low": "rest of the 1 km band"},
           "test_scores": json.load(open(ROOT / "results" / "metrics.json"))["erosion_forecast"]}
json.dump(summary, open(out / "forecast_2027.json", "w"), indent=1)
print({k: summary[k] for k in ("high_ha", "medium_ha", "expected_newly_eroded_ha")})
