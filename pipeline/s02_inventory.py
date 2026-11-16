"""Count the NASA Earthdata granules available over the study area."""
import earthaccess, collections
earthaccess.login(strategy="netrc")
bbox = (88.837, 25.345, 89.747, 26.365)
def inv(short, temporal, **kw):
    res = earthaccess.search_data(short_name=short, bounding_box=bbox, temporal=temporal, **kw)
    sizes = [g.size for g in res]
    tiles = collections.Counter(g["meta"]["native-id"].split("_")[3] if "OPERA" in short else g["meta"]["native-id"].split(".")[2] for g in res)
    print(f"{short:24s} n={len(res):5d} total={sum(sizes)/1024:7.1f} GB tiles={dict(tiles)}")
    return res
inv("OPERA_L3_DSWX-HLS_V1", ("2023-04-01", "2026-11-05"))
inv("OPERA_L3_DSWX-S1_V1", ("2023-09-01", "2026-11-05"))
inv("NASADEM_HGT", None)
for s in ["HLSL30", "HLSS30"]:
    tot = 0
    for y in range(2013, 2027):
        try:
            res = earthaccess.search_data(short_name=s, version="2.0", bounding_box=bbox, temporal=(f"{y}-01-01", f"{y}-03-31"), cloud_cover=(0, 20))
        except Exception as e:
            res = []
        tot += len(res)
        print(s, y, len(res), round(sum(g.size for g in res)/1024, 1), "GB")
