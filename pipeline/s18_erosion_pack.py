"""Yearly dry-season feature stacks for the erosion forecast model."""
import glob, re, json
from collections import defaultdict
import numpy as np, rasterio
from rasterio.warp import reproject, Resampling
from common import CLOUDY, DATA, DER, CRS, grid, on_grid, teesta_mask

YEARS = list(range(2016, 2027))
OUT = DATA / "erosion_pack"; OUT.mkdir(exist_ok=True)
t, h, w = grid(30)
inside = teesta_mask(30)
rr, cc = np.where(inside)
r0, r1, c0, c1 = rr.min() // 32 * 32, min(h, (rr.max() // 32 + 1) * 32), cc.min() // 32 * 32, min(w, (cc.max() // 32 + 1) * 32)
S = np.s_[r0:r1, c0:c1]


gran = defaultdict(dict)
for f in glob.glob(str(DATA / "hls" / "*.tif")):
    m = re.match(r".*(HLS\.(L30|S30)\.T\w+\.(\d{4})\d{3}T\d+\.v2\.0)\.(\w+)\.tif$", f)
    gran[m.group(1)][m.group(4)] = f
by_year = defaultdict(list)
for g, b in gran.items():
    nir, sw = ("B05", "B06") if ".L30." in g else ("B8A", "B11")
    if {"B03", "B04", nir, sw, "Fmask"} <= b.keys():
        by_year[int(g.split(".")[3][:4])].append((b["B03"], b["B04"], b[nir], b[sw], b["Fmask"]))

feats, chan, corr, nobs = [], [], [], []
for y in YEARS:
    nw = np.zeros((h, w), np.float32); ns = np.zeros_like(nw); sm = np.zeros_like(nw); sv = np.zeros_like(nw); n = np.zeros_like(nw)
    for fg, fr, fn, fs, fm in by_year[y]:
        q = on_grid(fm, 255, np.uint8)
        g, r, nr, s = (on_grid(p, -9999, np.int16).astype(np.float32) for p in (fg, fr, fn, fs))
        ok = (q != 255) & ((q & CLOUDY) == 0) & (g > -9999) & (r > -9999) & (nr > -9999) & (s > -9999) & ((g + s) > 0) & ((nr + r) > 0)
        mndwi = np.where(ok, (g - s) / np.where(ok, g + s, 1), 0); ndvi = np.where(ok, (nr - r) / np.where(ok, nr + r, 1), 0)
        wat = ok & (mndwi > 0); snd = ok & ~wat & (ndvi < 0.15) & (r * 1e-4 > 0.10)
        nw += wat; ns += snd; sm += mndwi; sv += ndvi; n += ok
    k = np.maximum(n, 1)
    f = np.stack([nw / k * 250, ns / k * 250, (np.clip(sm / k, -1, 1) + 1) * 125, (np.clip(sv / k, -1, 1) + 1) * 125])
    f[:, n == 0] = 0
    feats.append(np.round(f[(slice(None),) + S]).astype(np.uint8)); nobs.append(np.minimum(n, 255).astype(np.uint8)[S])
    chan.append(rasterio.open(DER / "channel" / f"channel_{y}.tif").read(1)[S].astype(bool))
    corr.append(rasterio.open(DER / "channel" / f"corridor_{y}.tif").read(1)[S].astype(bool))
    print(f"{y}: {len(by_year[y])} images, observed {(nobs[-1][inside[S]] >= 2).mean():.2f}, channel {chan[-1].sum()*9e-4:.0f} km2, "
          f"corridor {corr[-1].sum()*9e-4:.0f} km2", flush=True)

stat = np.stack([on_grid(DER / n_, -9999, np.float32, Resampling.bilinear)[S] for n_ in ("hand_dem_30m.tif", "dem_30m.tif", "dist_river_30m.tif")])
stat = np.nan_to_num(np.where(stat < -9000, 0, stat))
np.savez_compressed(OUT / "stacks.npz", years=np.array(YEARS), feats=np.stack(feats), nobs=np.stack(nobs), channel=np.stack(chan),
                    corridor=np.stack(corr), static=stat.astype(np.float32), inside=inside[S], origin=np.array([r0, c0]))
er = {YEARS[i]: round(float((corr[i] & ~corr[i - 1]).sum() * 0.09), 1) for i in range(1, len(YEARS))}
print("crop", (r1 - r0, c1 - c0), "| newly eroded ha by year:", er)
