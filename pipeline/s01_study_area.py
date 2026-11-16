"""Study area: the Teesta channel inside Bangladesh from OSM, buffered 6 km."""
import requests, geopandas as gpd
from shapely.geometry import LineString
from shapely.ops import unary_union, linemerge

UA = {"User-Agent": "TeestaWatch-research/0.1"}
q = """[out:json][timeout:180];
area["ISO3166-1"="BD"][admin_level=2]->.bd;
way["waterway"="river"]["name"~"^(Teesta|Tista|Teesta River|তিস্তা|তিস্তা নদী)$",i](area.bd);
out geom;"""
r = requests.post("https://overpass-api.de/api/interpreter", data={"data": q}, timeout=240, headers=UA)
r.raise_for_status()
ways = r.json()["elements"]
lines = [LineString([(p["lon"], p["lat"]) for p in w["geometry"]]) for w in ways]
print("ways:", len(ways), sorted({w["tags"].get("name") for w in ways}))
river = gpd.GeoDataFrame(geometry=[linemerge(unary_union(lines))], crs=4326)
ru = river.to_crs(32645)
print("river length km:", round(ru.length.iloc[0] / 1e3, 1), "bounds:", river.total_bounds.round(3))
river.to_file("data/aoi/teesta_river_osm.geojson", driver="GeoJSON")
aoi_u = ru.buffer(6000).simplify(200)
aoi = gpd.GeoDataFrame({"name": ["Teesta AOI (river + 6 km)"]}, geometry=aoi_u, crs=32645)
aoi.to_file("data/aoi/teesta_aoi_32645.geojson", driver="GeoJSON")
aoi.to_crs(4326).to_file("data/aoi/teesta_aoi.geojson", driver="GeoJSON")
print("AOI area km2:", round(aoi.area.iloc[0] / 1e6), "bbox:", aoi.to_crs(4326).total_bounds.round(3))
