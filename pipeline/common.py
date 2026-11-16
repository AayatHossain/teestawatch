"""Shared paths, the common analysis grid (EPSG:32645) and raster helpers."""
import math
from pathlib import Path
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DER = DATA / "derived"
DER.mkdir(parents=True, exist_ok=True)
MODELS = ROOT / "models"
WEIGHTS = ROOT / "weights"
RUNS = ROOT / "runs"
CRS = "EPSG:32645"
AOI = gpd.read_file(DATA / "aoi" / "teesta_aoi_32645.geojson").to_crs(CRS)
GRID_AOI = gpd.read_file(DATA / "aoi" / "teesta_aoi_32645_unclipped.geojson").to_crs(CRS)
RIVER = gpd.read_file(DATA / "aoi" / "teesta_river_osm.geojson").to_crs(CRS)
CLOUDY = 0b11110  # HLS Fmask bits 1-4: cloud, adjacent cloud, cloud shadow, snow/ice


def grid(res=30):
    # bounds snap to 30 m at every resolution, so 10 m cells nest exactly 3 x 3 inside 30 m cells
    x0, y0, x1, y1 = GRID_AOI.total_bounds
    x0, y0 = math.floor(x0 / 30) * 30, math.floor(y0 / 30) * 30
    x1, y1 = math.ceil(x1 / 30) * 30, math.ceil(y1 / 30) * 30
    return from_origin(x0, y1, res, res), int(round((y1 - y0) / res)), int(round((x1 - x0) / res))


def aoi_mask(res=30):
    t, h, w = grid(res)
    return rasterize(AOI.geometry, out_shape=(h, w), transform=t, fill=0, default_value=1, dtype="uint8").astype(bool)


def write(path, arr, res=30, nodata=None, dtype=None):
    t, h, w = grid(res)
    dtype = dtype or arr.dtype
    prof = dict(driver="GTiff", height=h, width=w, count=1, dtype=dtype, crs=CRS, transform=t, nodata=nodata,
                compress="deflate", tiled=True, blockxsize=256, blockysize=256,
                predictor=3 if np.dtype(dtype).kind == "f" else 2)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype(dtype), 1)


def teesta_mask(res=30, brahma_gap=5000):
    from scipy import ndimage
    t, h, w = grid(res)
    bra = gpd.read_file(DATA / "aoi" / "brahmaputra_osm.geojson").to_crs(CRS)
    rast = lambda g: rasterize(g.geometry, out_shape=(h, w), transform=t, fill=0, default_value=1, dtype="uint8").astype(bool)
    d_t = ndimage.distance_transform_edt(~rast(RIVER)) * res
    d_b = ndimage.distance_transform_edt(~rast(bra)) * res
    return aoi_mask(res) & (d_t < d_b) & (d_b > brahma_gap)


def on_grid(path, nodata, dtype, resampling=Resampling.nearest, res=30):
    t, h, w = grid(res)
    dst = np.full((h, w), nodata, dtype)
    with rasterio.open(path) as src:
        reproject(rasterio.band(src, 1), dst, src_nodata=nodata, dst_transform=t, dst_crs=CRS, dst_nodata=nodata,
                  resampling=resampling)
    return dst
