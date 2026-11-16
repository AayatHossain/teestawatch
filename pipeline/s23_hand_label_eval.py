"""Independent test on the 50 hand-labelled Teesta tiles (256 x 256 px at 10 m), never used in training."""
import glob
import importlib.util
import json
import os

import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.warp import Resampling, reproject

from common import CRS, DATA, DER, MODELS, ROOT, WEIGHTS, grid, teesta_mask

N, WIN = 256, 512
LAB = ROOT / "evaluation" / "hand_labels"
OUT = ROOT / "outputs" / "report"; OUT.mkdir(parents=True, exist_ok=True)
t10, H, W = grid(10)
mask10 = np.kron(teesta_mask(30), np.ones((3, 3), bool))[:H, :W]

tiles = pd.read_csv(LAB / "tiles.csv").set_index("tile_id")
labels = {}
for f in sorted(LAB.glob("labels_part*.json")):
    d = json.load(open(f, encoding="utf-8"))
    for tid, v in d["tiles"].items():
        runs = [p.split(":") for p in v["rle"].split(",") if p]
        labels[tid] = (np.concatenate([np.full(int(c), int(a), np.uint8) for a, c in runs]).reshape(N, N), d["part"])
assert len(labels) == 50, len(labels)
order = sorted(labels, key=lambda t: tiles.loc[t, "date"])

os.environ.setdefault("TW_OUT", str(ROOT / "cache" / "infer"))
spec = importlib.util.spec_from_file_location("tg", MODELS / "flood" / "tgfloodnet.py")
tg = importlib.util.module_from_spec(spec); spec.loader.exec_module(tg)
DEV = tg.DEV
norm = json.load(open(WEIGHTS / "flood" / "norm.json"))
M, S = np.array(norm["mean"], np.float32), np.array(norm["std"], np.float32)
nets = []
for s in (0, 1, 2):
    m = tg.TGFloodNet().to(DEV).eval()
    m.load_state_dict(torch.load(WEIGHTS / "flood" / f"seed{s}.pt", map_location=DEV))
    nets.append(m)


def terrain10(name):
    out = np.zeros((H, W), np.float32)
    with rasterio.open(DER / name) as s:
        reproject(rasterio.band(s, 1), out, dst_transform=t10, dst_crs=CRS, resampling=Resampling.bilinear)
    return np.nan_to_num(out)


HAND = np.log1p(np.clip(terrain10("hand_dem_30m.tif"), 0, 100))
DIST = np.log1p(np.clip(terrain10("dist_stream_30m.tif"), 0, 10000))
RADAR = {}


def window(date, r, c):
    if date not in RADAR:
        RADAR.clear()
        RADAR[date] = {p: rasterio.open(DER / "s1_10m" / f"{date}_{p}.tif").read(1) for p in ("VV", "VH")}
    r0, c0 = r - (WIN - N) // 2, c - (WIN - N) // 2
    x = np.zeros((5, WIN, WIN), np.float32); okw = np.zeros((WIN, WIN), bool)
    a0, b0, a1, b1 = max(r0, 0), max(c0, 0), min(r0 + WIN, H), min(c0 + WIN, W)
    vv, vh = RADAR[date]["VV"][a0:a1, b0:b1], RADAR[date]["VH"][a0:a1, b0:b1]
    ok = (vv != -32768) & (vh != -32768) & mask10[a0:a1, b0:b1]
    a = np.clip(vv / 100 - 1.0, *norm["clip"]); b = np.clip(vh / 100 - 1.0, *norm["clip"])
    ch = [a, b, a - b, HAND[a0:a1, b0:b1], DIST[a0:a1, b0:b1]]
    z = np.stack([(ch[i] - M[i]) / S[i] for i in range(5)]).astype(np.float32); z[:, ~ok] = 0
    x[:, a0 - r0:a1 - r0, b0 - c0:b1 - c0] = z; okw[a0 - r0:a1 - r0, b0 - c0:b1 - c0] = ok
    o = (WIN - N) // 2
    return x, okw[o:o + N, o:o + N]


def dswx_s1(date, r, c):
    tt = rasterio.transform.from_origin(t10.c + c * 10, t10.f - r * 10, 10, 10)
    out = np.full((N, N), 255, np.uint8)
    for f in sorted(glob.glob(str(DATA / "opera_dswx_s1" / f"*_{date.replace('-', '')}T*_B01_WTR.tif"))):
        a = np.full((N, N), 255, np.uint8)
        with rasterio.open(f) as s:
            reproject(rasterio.band(s, 1), a, src_nodata=255, dst_transform=tt, dst_crs=CRS, dst_nodata=255, resampling=Resampling.nearest)
        fill = (out == 255) & (a != 255); out[fill] = a[fill]
    w = np.full((N, N), 255, np.uint8); w[out == 0] = 0; w[np.isin(out, [1, 3])] = 1
    return w


rows, CNT = [], {}
o = (WIN - N) // 2
with torch.no_grad():
    for tid in order:
        lab, part = labels[tid]; r = tiles.loc[tid]; rr, cc = int(r.row), int(r.col)
        x, ok = window(r.date, rr, cc)
        with torch.autocast(DEV.type, enabled=DEV.type == "cuda"):
            p = sum(torch.sigmoid(n(torch.from_numpy(x[None]).to(DEV))["water"].float()) for n in nets) / len(nets)
        maps = {"TG-FloodNet": np.where(ok, (p[0, 0, o:o + N, o:o + N].cpu().numpy() >= 0.5).astype(np.uint8), 255)}
        if r.date.startswith("2025"):
            maps["NASA DSWx-S1"] = dswx_s1(r.date, rr, cc)
        for name, mp in maps.items():
            v = (lab != 2) & (mp != 255); pr = mp[v] == 1; t = lab[v] == 1
            c = np.array([(pr & t).sum(), (pr & ~t).sum(), (~pr & t).sum(), (~pr & ~t).sum()], np.int64)
            CNT[(name, tid)] = c
            rows.append(dict(tile=tid, date=r.date, labeller=part, method=name, tp=c[0], fp=c[1], fn=c[2], tn=c[3],
                             iou=c[0] / max(c[:3].sum(), 1)))
df = pd.DataFrame(rows); df.to_csv(OUT / "hand_label_eval_tiles.csv", index=False)


def scores(c):
    tp, fp, fn, tn = c
    return dict(iou=tp / max(tp + fp + fn, 1), f1=2 * tp / max(2 * tp + fp + fn, 1), precision=tp / max(tp + fp, 1),
                recall=tp / max(tp + fn, 1), accuracy=(tp + tn) / max(c.sum(), 1))


rng = np.random.default_rng(0)
res = {}
for subset, sel in (("all", lambda t: True), ("2025", lambda t: tiles.loc[t, "date"].startswith("2025"))):
    tids = [t for t in order if sel(t)]
    res[subset] = {"n_tiles": len(tids)}
    for name in ("TG-FloodNet", "NASA DSWx-S1"):
        tm = [t for t in tids if (name, t) in CNT]
        if not tm:
            continue
        C = np.stack([CNT[(name, t)] for t in tm])
        boot = [scores(C[rng.integers(0, len(tm), len(tm))].sum(0))["iou"] for _ in range(2000)]
        m = scores(C.sum(0))
        m.update(n_tiles=len(tm), iou_ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))])
        res[subset][name] = {k: round(float(v), 4) if not isinstance(v, (list, int)) else v for k, v in m.items()}
json.dump(res, open(OUT / "hand_label_eval.json", "w"), indent=1)
for subset, r in res.items():
    for name, m in r.items():
        if name != "n_tiles":
            print(f"{subset:4s} {name:13s} tiles {m['n_tiles']:2d}  IoU {m['iou']:.3f}  F1 {m['f1']:.3f}  P {m['precision']:.3f}  R {m['recall']:.3f}")
