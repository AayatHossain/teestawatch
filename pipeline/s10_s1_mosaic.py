"""Mosaic each radar date onto the common 10 m grid in dB."""
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from common import DATA, DER, CRS, grid, aoi_mask, write

OUT = DER / "s1_10m"; OUT.mkdir(exist_ok=True)
t, h, w = grid(10)
inside = aoi_mask(10)
scenes = pd.read_csv(DATA / "scenes_core.csv")
for date, grp in scenes.groupby("date"):
    for pol in ("VV", "VH"):
        out = OUT / f"{date}_{pol}.tif"
        if out.exists():
            continue
        acc = np.zeros((h, w), np.float32); n = np.zeros((h, w), np.uint8)
        for sc in grp.scene:
            buf = np.zeros((h, w), np.float32)
            with rasterio.open(DATA / "s1rtc" / f"{sc}_{pol}.tif") as s:
                reproject(rasterio.band(s, 1), buf, src_nodata=0, dst_transform=t, dst_crs=CRS, dst_nodata=0,
                          resampling=Resampling.bilinear)
            ok = buf > 0
            acc[ok] += buf[ok]; n[ok] += 1
        db = np.full((h, w), -32768, np.int16)
        ok = (n > 0) & inside
        db[ok] = np.round(10 * np.log10(np.maximum(acc[ok] / n[ok], 1e-6)) * 100).astype(np.int16)
        write(out, db, res=10, nodata=-32768, dtype="int16")
        print(f"{date} {pol}: AOI covered {100*ok.sum()/inside.sum():.1f}%  median {np.median(db[ok])/100:.1f} dB", flush=True)
