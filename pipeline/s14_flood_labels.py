"""Teesta fine-tuning labels from NASA HLS optical water, one map per radar date."""
import glob, json, re, datetime as dt
from collections import defaultdict
import numpy as np, rasterio
from rasterio.warp import reproject, Resampling
from common import DATA, DER, CRS, grid, teesta_mask, write

DATES = ["2017-07-17", "2017-08-10", "2020-06-07", "2020-07-13", "2024-06-10", "2024-07-04", "2025-07-23", "2025-08-16"]
OUT = DER / "ft_labels"; OUT.mkdir(exist_ok=True)
t, h, w = grid(30)
inside = teesta_mask(30)
CLOUDY = 0b11110


def on_grid(path, nodata, dtype):
    dst = np.full((h, w), nodata, dtype)
    with rasterio.open(path) as s:
        reproject(rasterio.band(s, 1), dst, src_nodata=nodata, dst_transform=t, dst_crs=CRS, dst_nodata=nodata,
                  resampling=Resampling.nearest)
    return dst


gran = defaultdict(dict)
for f in glob.glob(str(DATA / "hls_flood" / "*.tif")):
    m = re.match(r".*(HLS\.(L30|S30)\.T\w+\.(\d{7})T\d+\.v2\.0)\.(\w+)\.tif$", f)
    if m:
        gran[m.group(1)][m.group(4)] = f


def hls_label(date):
    c = dt.date.fromisoformat(date); cand = []
    for g, b in gran.items():
        d = dt.datetime.strptime(g.split(".")[3][:7], "%Y%j").date()
        swir = "B06" if ".L30." in g else "B11"
        if abs((d - c).days) <= 3 and {"B03", swir, "Fmask"} <= b.keys():
            cand.append((abs((d - c).days), g, b["B03"], b[swir], b["Fmask"]))
    lab = np.full((h, w), 255, np.uint8); used = []
    for dd, g, fg, fs, fm in sorted(cand):
        gg = on_grid(fg, -9999, np.int16).astype(np.float32); ss = on_grid(fs, -9999, np.int16).astype(np.float32)
        q = on_grid(fm, 255, np.uint8)
        ok = (q != 255) & ((q & CLOUDY) == 0) & (gg != -9999) & (ss != -9999) & ((gg + ss) > 0) & (lab == 255)
        lab[ok] = ((gg[ok] - ss[ok]) / (gg[ok] + ss[ok]) > 0).astype(np.uint8)
        used.append(f"{g} (dt {dd} d, +{ok.sum() * 9e-4:.0f} km2 clear)")
    lab[~inside] = 255
    return lab, used


def dswx_hls_label(date):
    c = dt.date.fromisoformat(date); lab = np.full((h, w), 255, np.uint8); n = 0
    files = sorted(glob.glob(str(DATA / "opera_dswx_hls" / "*_B01_WTR.tif")),
                   key=lambda f: abs((dt.datetime.strptime(re.search(r"_(\d{8})T", f).group(1), "%Y%m%d").date() - c).days))
    for f in files:
        d = dt.datetime.strptime(re.search(r"_(\d{8})T", f).group(1), "%Y%m%d").date()
        if abs((d - c).days) > 3:
            break
        a = on_grid(f, 255, np.uint8); fill = (lab == 255) & (a <= 2)
        lab[fill] = (a[fill] >= 1).astype(np.uint8); n += 1
    lab[~inside] = 255
    return lab, n


summary = {}
for date in DATES:
    lab, used = hls_label(date)
    write(OUT / f"label_{date}.tif", lab, nodata=255)
    k = (lab != 255)
    row = dict(hls_granules=used, labelled_km2=round(k.sum() * 9e-4, 1), teesta_km2=round(inside.sum() * 9e-4, 1),
               labelled_share=round(k.sum() / inside.sum(), 3), water_share=round(float((lab[k] == 1).mean()) if k.any() else 0, 3))
    if date >= "2024":
        ref, n = dswx_hls_label(date)
        v = k & (ref != 255)
        if v.any():
            p, r = lab[v] == 1, ref[v] == 1
            tp, fp, fn = (p & r).sum(), (p & ~r).sum(), (~p & r).sum()
            row["vs_dswx_hls"] = dict(granules=n, compared_km2=round(v.sum() * 9e-4, 1), iou=round(tp / max(tp + fp + fn, 1), 3),
                                      accuracy=round(float((p == r).mean()), 3))
    summary[date] = row
    print(date, {k_: v_ for k_, v_ in row.items() if k_ != "hls_granules"}, flush=True)
json.dump(summary, open(OUT / "summary.json", "w"), indent=1)
