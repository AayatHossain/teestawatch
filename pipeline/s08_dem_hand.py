"""NASADEM mosaic, height above nearest drainage and distance to the river on the 30 m grid."""
import glob
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize
from scipy import ndimage
from scipy.spatial import cKDTree
from common import DATA, DER, CRS, RIVER, grid, aoi_mask, write

t, h, w = grid(30)
srcs = [rasterio.open(f) for f in sorted(glob.glob(str(DATA / "nasadem" / "*.hgt")))]
mos, mos_t = merge(srcs, nodata=-32768)
dem = np.full((h, w), np.nan, np.float32)
reproject(mos[0].astype(np.float32), dem, src_transform=mos_t, src_crs="EPSG:4326", src_nodata=-32768,
          dst_transform=t, dst_crs=CRS, dst_nodata=np.nan, resampling=Resampling.bilinear)
inside = aoi_mask(30)
print("DEM inside AOI: min %.1f  median %.1f  max %.1f m" % tuple(np.nanpercentile(dem[inside], [0, 50, 100])))

line = rasterize(RIVER.geometry, out_shape=(h, w), transform=t, fill=0, default_value=1, dtype="uint8").astype(bool)
rr, cc = np.nonzero(line)
win = 16
raw = np.array([np.nanpercentile(dem[max(r - win, 0):r + win + 1, max(c - win, 0):c + win + 1], 10) for r, c in zip(rr, cc)])
tree = cKDTree(np.c_[rr, cc])
riv = np.array([np.median(raw[idx]) for idx in tree.query_ball_point(np.c_[rr, cc], r=2000 / 30)])
print("river centreline pixels:", len(rr), " river elevation range %.1f to %.1f m" % (riv.min(), riv.max()))

riv_img = np.full((h, w), np.nan, np.float32)
riv_img[rr, cc] = riv
dist, (ir, ic) = ndimage.distance_transform_edt(~line, return_indices=True)
hand = dem - riv_img[ir, ic]
dist = (dist * 30).astype(np.float32)
for a in (dem, hand, dist):
    a[~inside] = np.nan
write(DER / "dem_30m.tif", dem, nodata=np.nan, dtype="float32")
write(DER / "hand_30m.tif", hand, nodata=np.nan, dtype="float32")
write(DER / "dist_river_30m.tif", dist, nodata=np.nan, dtype="float32")
print("height above river inside AOI: p5 %.1f  p50 %.1f  p95 %.1f m" % tuple(np.nanpercentile(hand[inside], [5, 50, 95])))
