"""Download NASA HLS, OPERA DSWx and NASADEM granules, cropped to the study area."""
import os, sys, time, zipfile, json, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import earthaccess
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask

ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / "cache" / "tmp" / "dl"
TMP.mkdir(parents=True, exist_ok=True)
AOI = gpd.read_file(ROOT / "data" / "aoi" / "teesta_aoi_32645.geojson")
BBOX = tuple(gpd.read_file(ROOT / "data" / "aoi" / "teesta_aoi.geojson").total_bounds.round(3))
LOG = ROOT / "logs" / "download_nasa.log"
lock = threading.Lock()


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    with lock:
        print(line, flush=True)
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")


earthaccess.login(strategy="netrc")
SESSION = earthaccess.get_requests_https_session()


def fetch(url, dest):
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(5):
        try:
            with SESSION.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            tmp.replace(dest)
            return dest
        except Exception as e:
            log(f"  retry {attempt+1}/5 {url.split('/')[-1]}: {e}")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"failed: {url}")


def crop_to_aoi(src_path, out_path):
    with rasterio.open(src_path) as src:
        geom = AOI.to_crs(src.crs).geometry
        try:
            data, transform = rio_mask(src, geom, crop=True, filled=True)
        except ValueError:
            return False
        prof = src.profile.copy()
        prof.update(driver="GTiff", height=data.shape[1], width=data.shape[2], transform=transform,
                    compress="deflate", tiled=True, blockxsize=256, blockysize=256, predictor=2 if data.dtype.kind in "iu" else 1)
        prof.pop("photometric", None)
        tmp = out_path.with_suffix(".tmp.tif")
        with rasterio.open(tmp, "w", **prof) as dst:
            dst.write(data)
            for i in range(1, src.count + 1):
                cmap = None
                try:
                    cmap = src.colormap(i)
                except ValueError:
                    pass
                if cmap:
                    dst.write_colormap(i, cmap)
            dst.update_tags(**src.tags(), source_file=Path(src_path).name)
    tmp.replace(out_path)
    return True


def process_url(url, out_dir):
    name = url.split("/")[-1]
    out = out_dir / name
    skip = out_dir / (name + ".nooverlap")
    if out.exists() or skip.exists():
        return "skip", 0
    raw = TMP / name
    fetch(url, raw)
    size = raw.stat().st_size
    ok = crop_to_aoi(raw, out)
    if not ok:
        skip.touch()
    raw.unlink(missing_ok=True)
    return ("ok" if ok else "nooverlap"), size


def run_urls(tag, urls, out_dir, workers=6):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.txt").write_text("\n".join(urls), encoding="utf-8")
    log(f"[{tag}] {len(urls)} files -> {out_dir}")
    done, nbytes, t0 = 0, 0, time.time()
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(process_url, u, out_dir): u for u in urls}
        for fu in as_completed(futs):
            try:
                status, size = fu.result()
            except Exception as e:
                log(f"[{tag}] ERROR {futs[fu].split('/')[-1]}: {e}")
                continue
            done += 1
            nbytes += size
            if done % 25 == 0 or done == len(urls):
                el = time.time() - t0
                log(f"[{tag}] {done}/{len(urls)}  downloaded {nbytes/1e9:.2f} GB  {nbytes/1e6/max(el,1):.1f} MB/s")
    log(f"[{tag}] finished")


def links(granules, suffixes):
    out = []
    for g in granules:
        for l in g.data_links(access="external"):
            if any(l.endswith(s) for s in suffixes):
                out.append(l)
    return out


def do_nasadem():
    out_dir = ROOT / "data" / "nasadem"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = earthaccess.search_data(short_name="NASADEM_HGT", bounding_box=BBOX)
    for g in res:
        for url in g.data_links(access="external"):
            if not url.endswith(".zip"):
                continue
            z = TMP / url.split("/")[-1]
            hgt_name = z.stem.split("_")[-1] + ".hgt"
            if (out_dir / hgt_name).exists():
                continue
            fetch(url, z)
            with zipfile.ZipFile(z) as zf:
                member = [m for m in zf.namelist() if m.endswith(".hgt")][0]
                (out_dir / hgt_name).write_bytes(zf.read(member))
            z.unlink()
            log(f"[nasadem] {hgt_name}")
    log("[nasadem] finished (mosaic + reprojection happens in preprocessing)")


def do_dswx(short, tag, start):
    res = earthaccess.search_data(short_name=short, bounding_box=BBOX, temporal=(start, time.strftime("%Y-%m-%d")))
    run_urls(tag, links(res, ["_B01_WTR.tif"]), ROOT / "data" / tag)


def do_hls():
    urls = []
    for short, bands in [("HLSL30", [".B03.tif", ".B04.tif", ".B05.tif", ".B06.tif", ".Fmask.tif"]),
                         ("HLSS30", [".B03.tif", ".B04.tif", ".B8A.tif", ".B11.tif", ".Fmask.tif"])]:
        for y in range(2013, 2027):
            try:
                res = earthaccess.search_data(short_name=short, bounding_box=BBOX,
                                              temporal=(f"{y}-01-01", f"{y}-03-31"), cloud_cover=(0, 20))
            except Exception:
                res = []
            urls += links(res, bands)
    run_urls("hls", urls, ROOT / "data" / "hls")


FLOOD_DATES = ["2017-07-17", "2017-08-10", "2020-06-07", "2020-07-13", "2024-06-10", "2024-07-04", "2025-07-23", "2025-08-16"]


def do_hls_flood():
    import datetime as dt
    urls = []
    for d in FLOOD_DATES:
        c = dt.date.fromisoformat(d)
        for short, bands in [("HLSL30", [".B02.tif", ".B03.tif", ".B04.tif", ".B05.tif", ".B06.tif", ".Fmask.tif"]),
                             ("HLSS30", [".B02.tif", ".B03.tif", ".B04.tif", ".B8A.tif", ".B11.tif", ".Fmask.tif"])]:
            try:
                res = earthaccess.search_data(short_name=short, bounding_box=BBOX, cloud_cover=(0, 94),
                                              temporal=(str(c - dt.timedelta(3)), str(c + dt.timedelta(3))))
            except Exception:
                res = []
            urls += links(res, bands)
    run_urls("hls_flood", sorted(set(urls)), ROOT / "data" / "hls_flood")


if __name__ == "__main__":
    products = sys.argv[1:] or ["nasadem", "dswx_s1", "dswx_hls", "hls"]
    for p in products:
        if p == "nasadem":
            do_nasadem()
        elif p == "dswx_s1":
            do_dswx("OPERA_L3_DSWX-S1_V1", "opera_dswx_s1", "2023-09-01")
        elif p == "dswx_hls":
            do_dswx("OPERA_L3_DSWX-HLS_V1", "opera_dswx_hls", "2023-04-01")
        elif p == "hls":
            do_hls()
        elif p == "hls_flood":
            do_hls_flood()
    log("ALL DONE: " + " ".join(products))
