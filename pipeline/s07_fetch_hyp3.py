"""Fetch finished HyP3 RTC jobs and keep VV and VH cropped to the study area."""
import os, sys, time, zipfile
from pathlib import Path
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
import hyp3_sdk as sdk

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "s1rtc"; OUT.mkdir(parents=True, exist_ok=True)
TMP = ROOT / "cache" / "tmp" / "hyp3"; TMP.mkdir(parents=True, exist_ok=True)
LOG = ROOT / "logs" / "hyp3_fetch.log"
AOI = gpd.read_file(ROOT / "data" / "aoi" / "teesta_aoi_32645.geojson")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "3")
os.environ.setdefault("CPL_VSIL_CURL_CHUNK_SIZE", "4194304")


def log(m):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {m}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def crop(src_path, out_path):
    with rasterio.open(src_path) as src:
        geom = AOI.to_crs(src.crs).geometry
        data, transform = rio_mask(src, geom, crop=True, filled=True, nodata=0)
        prof = src.profile.copy()
    prof.update(driver="GTiff", height=data.shape[1], width=data.shape[2], transform=transform, nodata=0,
                compress="deflate", predictor=3, tiled=True, blockxsize=256, blockysize=256)
    tmp = out_path.with_suffix(".tmp.tif")
    with rasterio.open(tmp, "w", **prof) as dst:
        dst.write(data)
    tmp.replace(out_path)


def outputs(scene):
    return OUT / f"{scene}_VV.tif", OUT / f"{scene}_VH.tif"


def handle(job):
    scene = job.job_parameters["granules"][0]
    vv_out, vh_out = outputs(scene)
    if vv_out.exists() and vh_out.exists():
        return "skip"
    f = job.files[0]
    url, zname = f["url"], f["filename"]
    stem = zname[:-4]
    t0 = time.time()
    zpath = TMP / zname
    if zpath.exists() and not zipfile.is_zipfile(zpath):
        log(f"removing incomplete zip {zname} ({zpath.stat().st_size/1e9:.2f} GB)")
        zpath.unlink()
    if not zpath.exists():
        job.download_files(TMP)
    if not zipfile.is_zipfile(zpath):
        zpath.unlink()
        raise RuntimeError("downloaded zip is incomplete, will retry next pass")
    size = zpath.stat().st_size
    with zipfile.ZipFile(zpath) as zf:
        for pol, out in [("VV", vv_out), ("VH", vh_out)]:
            member = f"{stem}/{stem}_{pol}.tif"
            zf.extract(member, TMP)
            crop(TMP / member, out)
            (TMP / member).unlink()
    zpath.unlink()
    try:
        (TMP / stem).rmdir()
    except OSError:
        pass
    log(f"zip ok  {scene}  {size/1e9:.2f} GB in {time.time()-t0:.0f}s")
    return "zip"


if __name__ == "__main__":
    hyp3 = sdk.HyP3()
    while True:
        import pandas as pd
        core = set(pd.read_csv(ROOT / "data" / "scenes_core.csv").scene)
        jobs = [j for j in hyp3.find_jobs(name="teesta", job_type="RTC_GAMMA") if j.job_parameters["granules"][0] in core]
        st = {}
        for j in jobs:
            st[j.status_code] = st.get(j.status_code, 0) + 1
        pending = 0
        for j in jobs:
            if j.succeeded():
                try:
                    handle(j)
                except Exception as e:
                    log(f"ERROR {j.job_parameters['granules'][0]}: {e}")
                    pending += 1
            elif j.failed():
                log(f"JOB FAILED {j.job_parameters['granules'][0]}")
            elif j.expired():
                log(f"JOB EXPIRED {j.job_parameters['granules'][0]}")
            else:
                pending += 1
        n_have = sum(all(o.exists() for o in outputs(j.job_parameters["granules"][0])) for j in jobs)
        log(f"status {st}  cropped {n_have}/{len(jobs)}")
        if pending == 0:
            log("ALL DONE")
            break
        time.sleep(300)
