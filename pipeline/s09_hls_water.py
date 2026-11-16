"""Yearly dry-season water maps from NASA HLS."""
import glob, re, sys
from collections import defaultdict
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from common import DATA, DER, CRS, grid, aoi_mask, write

OUT = DER / "water_hls"; OUT.mkdir(exist_ok=True)
t, h, w = grid(30)
inside = aoi_mask(30)
CLOUDY = 0b11110


def on_grid(path, nodata, dtype):
    dst = np.full((h, w), nodata, dtype)
    with rasterio.open(path) as s:
        reproject(rasterio.band(s, 1), dst, src_nodata=nodata, dst_transform=t, dst_crs=CRS,
                  dst_nodata=nodata, resampling=Resampling.nearest)
    return dst


granules = defaultdict(dict)
for f in glob.glob(str(DATA / "hls" / "*.tif")):
    m = re.match(r".*(HLS\.(L30|S30)\.T\w+\.(\d{4})\d{3}T\d+\.v2\.0)\.(\w+)\.tif$", f)
    granules[m.group(1)][m.group(4)] = f
by_year = defaultdict(list)
for g, bands in granules.items():
    swir = "B06" if ".L30." in g else "B11"
    if {"B03", swir, "Fmask"} <= bands.keys():
        by_year[int(g.split(".")[3][:4])].append((bands["B03"], bands[swir], bands["Fmask"]))

summary = []
for y in sorted(by_year):
    wet = np.zeros((h, w), np.uint16); n = np.zeros((h, w), np.uint16)
    for fg, fs, fm in by_year[y]:
        g = on_grid(fg, -9999, np.int16).astype(np.float32)
        s = on_grid(fs, -9999, np.int16).astype(np.float32)
        q = on_grid(fm, 255, np.uint8)
        ok = (q != 255) & ((q & CLOUDY) == 0) & (g != -9999) & (s != -9999) & ((g + s) > 0)
        mndwi = np.where(ok, (g - s) / np.where(ok, g + s, 1), 0)
        wet += (ok & (mndwi > 0)); n += ok
    freq = np.where(n > 0, np.round(100 * wet / np.maximum(n, 1)), 255).astype(np.uint8)
    water = np.where(n >= 3, (freq >= 50).astype(np.uint8), 255).astype(np.uint8)
    for a in (freq, water):
        a[~inside] = 255
    nobs = np.minimum(n, 254).astype(np.uint8); nobs[~inside] = 255
    write(OUT / f"water_freq_{y}.tif", freq, nodata=255)
    write(OUT / f"nobs_{y}.tif", nobs, nodata=255)
    write(OUT / f"water_{y}.tif", water, nodata=255)
    valid = inside & (water != 255)
    row = (y, len(by_year[y]), 100 * valid.sum() / inside.sum(), (water[valid] == 1).sum() * 900 / 1e6, np.median(n[inside]))
    summary.append(row)
    print("%d  images %3d  AOI covered %5.1f%%  water %6.1f km2  median clear looks %.0f" % row, flush=True)
