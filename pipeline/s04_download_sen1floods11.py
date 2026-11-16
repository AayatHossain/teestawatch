"""Download the Sentinel-1 parts of Sen1Floods11 v1.1."""
import requests, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "sen1floods11"
LOG = ROOT / "logs" / "download_sen1floods11.log"
PREFIXES = ["v1.1/splits/", "v1.1/Sen1Floods11_Metadata.geojson",
            "v1.1/data/flood_events/HandLabeled/S1Hand/", "v1.1/data/flood_events/HandLabeled/LabelHand/",
            "v1.1/data/flood_events/HandLabeled/JRCWaterHand/", "v1.1/data/flood_events/HandLabeled/S1OtsuLabelHand/",
            "v1.1/data/flood_events/WeaklyLabeled/S1OtsuLabelWeak/", "v1.1/data/flood_events/WeaklyLabeled/S2IndexLabelWeak/",
            "v1.1/data/flood_events/WeaklyLabeled/S1Weak/"]
def log(m):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {m}"; print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f: f.write(line + "\n")
def listing(prefix):
    tok, items = None, []
    while True:
        p = {"prefix": prefix, "fields": "items(name,size),nextPageToken", "maxResults": 1000}
        if tok: p["pageToken"] = tok
        r = requests.get("https://storage.googleapis.com/storage/v1/b/sen1floods11/o", params=p, timeout=60).json()
        items += r.get("items", []); tok = r.get("nextPageToken")
        if not tok: return items
def get(item):
    dest = OUT / item["name"].removeprefix("v1.1/")
    if dest.exists() and dest.stat().st_size == int(item["size"]): return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    for a in range(5):
        try:
            r = requests.get("https://storage.googleapis.com/sen1floods11/" + item["name"], timeout=120); r.raise_for_status()
            tmp = dest.with_suffix(dest.suffix + ".part"); tmp.write_bytes(r.content); tmp.replace(dest)
            return len(r.content)
        except Exception as e:
            time.sleep(5 * (a + 1))
    raise RuntimeError(item["name"])
items = [it for p in PREFIXES for it in listing(p) if not it["name"].endswith("/")]
log(f"{len(items)} files, {sum(int(i['size']) for i in items)/1e9:.2f} GB")
n, b, t0 = 0, 0, time.time()
with ThreadPoolExecutor(8) as ex:
    for fu in as_completed([ex.submit(get, it) for it in items]):
        try: b += fu.result()
        except Exception as e: log(f"ERROR {e}")
        n += 1
        if n % 500 == 0 or n == len(items): log(f"{n}/{len(items)}  {b/1e9:.2f} GB  {b/1e6/max(time.time()-t0,1):.1f} MB/s")
log("ALL DONE")
