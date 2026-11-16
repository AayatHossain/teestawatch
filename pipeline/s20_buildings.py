"""Google Open Buildings footprints for the study area as points."""
import time
import duckdb, geopandas as gpd, pandas as pd
from common import AOI, DATA, ROOT

U = "https://data.source.coop/vida/google-microsoft-open-buildings/geoparquet/by_country/country_iso=BGD/BGD.parquet"
OUT = DATA / "buildings"; OUT.mkdir(exist_ok=True)
DUCK = ROOT / "cache" / "duckdb"; DUCK.mkdir(parents=True, exist_ok=True)
w, s, e, n = AOI.to_crs(4326).total_bounds
con = duckdb.connect(); con.execute("INSTALL httpfs; LOAD httpfs; SET home_directory='" + str(DUCK) + "';")
t0 = time.time()
df = con.execute(f"""SELECT bf_source, confidence, area_in_meters,
        (bbox.xmin + bbox.xmax) / 2 AS lon, (bbox.ymin + bbox.ymax) / 2 AS lat
     FROM read_parquet('{U}')
     WHERE bbox.xmax >= {w} AND bbox.xmin <= {e} AND bbox.ymax >= {s} AND bbox.ymin <= {n}""").df()
print(f"fetched {len(df):,} buildings in the AOI box in {time.time()-t0:.0f}s; sources: {df.bf_source.value_counts().to_dict()}", flush=True)
g = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs=4326)
g = g[g.within(AOI.to_crs(4326).union_all())].reset_index(drop=True)
g.to_parquet(OUT / "buildings_aoi.parquet")
print(f"inside the Bangladesh AOI: {len(g):,} buildings; by source: {g.bf_source.value_counts().to_dict()}; "
      f"median area {g.area_in_meters.median():.0f} m2", flush=True)
