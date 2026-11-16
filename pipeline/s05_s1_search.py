"""Search Sentinel-1 IW GRD scenes over the study area and record their coverage."""
import asf_search as asf, geopandas as gpd, pandas as pd
from shapely.geometry import shape
aoi = gpd.read_file("data/aoi/teesta_aoi.geojson")
aoi_u = aoi.to_crs(32645).geometry.iloc[0]
wkt = aoi.geometry.iloc[0].convex_hull.simplify(0.01).wkt
res = asf.geo_search(platform=asf.PLATFORM.SENTINEL1, intersectsWith=wkt, start="2017-01-01", end="2026-11-06",
                     beamMode=asf.BEAMMODE.IW, processingLevel=asf.PRODUCT_TYPE.GRD_HD)
rows = []
for r in res:
    p = r.properties
    fp = gpd.GeoSeries([shape(r.geometry)], crs=4326).to_crs(32645).iloc[0]
    rows.append(dict(scene=p["sceneName"], time=p["startTime"], date=p["startTime"][:10], platform=p["platform"],
                     rel_orbit=p["pathNumber"], frame=p["frameNumber"], direction=p["flightDirection"],
                     pol=p["polarization"], aoi_cover=round(fp.intersection(aoi_u).area / aoi_u.area, 3)))
df = pd.DataFrame(rows).sort_values("time")
df.to_csv("data/s1_all_scenes.csv", index=False)
print("scenes:", len(df))
print(df.groupby(["rel_orbit", "direction"]).agg(n=("scene", "size"), cover_med=("aoi_cover", "median"),
      first=("date", "min"), last=("date", "max")).sort_values("n", ascending=False).to_string())
