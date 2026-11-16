"""Terrain inputs for TG-FloodNet, computed the same way for Sen1Floods11 and the Teesta."""
import sys, json, math, zipfile, glob, time, shutil, warnings
from pathlib import Path
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
from scipy import ndimage

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
S1F = ROOT / "data" / "sen1floods11"
TILES = ROOT / "cache" / "tmp" / "dem_s1f11"
WORK = ROOT / "cache" / "tmp" / "terrain_work"
STREAM_CELLS = 1000
MARGIN = 0.05


def terrain_from_dem(dem, transform, nodata=-32768):
    if not hasattr(np, "in1d"):
        np.in1d = np.isin
    from pysheds.grid import Grid
    WORK.mkdir(parents=True, exist_ok=True)
    tif = WORK / f"dem_{time.time_ns()}.tif"
    d = dem.astype(np.float32)
    miss = (d == nodata) | ~np.isfinite(d)
    if miss.all():
        z = np.zeros_like(d); return z, z.copy()
    d[miss] = np.nanmin(d[~miss]) - 1
    with rasterio.open(tif, "w", driver="GTiff", height=d.shape[0], width=d.shape[1], count=1, dtype="float32",
                       crs="EPSG:4326", transform=transform, nodata=-9999) as dst:
        dst.write(d, 1)
    g = Grid.from_raster(str(tif)); r = g.read_raster(str(tif))
    filled = g.resolve_flats(g.fill_depressions(g.fill_pits(r)))
    fdir = g.flowdir(filled)
    acc = g.accumulation(fdir)
    streams = np.asarray(acc) > STREAM_CELLS
    hand = np.asarray(g.compute_hand(fdir, r, acc > STREAM_CELLS), dtype=np.float32)
    tif.unlink(missing_ok=True)
    hand = np.where(np.isfinite(hand), hand, np.nan)
    hand[miss] = 0
    if np.isnan(hand).any():
        idx = ndimage.distance_transform_edt(np.isnan(hand), return_distances=False, return_indices=True)
        hand = hand[tuple(idx)]
    hand = np.clip(hand, 0, None)
    lat = transform.f + transform.e * d.shape[0] / 2
    dy = abs(transform.e) * 111320.0; dx = transform.a * 111320.0 * math.cos(math.radians(lat))
    stream_or_sea = streams | miss
    dist = ndimage.distance_transform_edt(~stream_or_sea, sampling=(dy, dx)).astype(np.float32) if stream_or_sea.any() \
        else np.full(d.shape, 5000, np.float32)
    return hand.astype(np.float32), dist


def to_grid(arr, src_t, src_crs, dst_shape, dst_t, dst_crs):
    out = np.zeros(dst_shape, np.float32)
    reproject(arr, out, src_transform=src_t, src_crs=src_crs, dst_transform=dst_t, dst_crs=dst_crs,
              resampling=Resampling.bilinear)
    return out


def download():
    import earthaccess
    earthaccess.login(strategy="netrc")
    sess = earthaccess.get_requests_https_session()
    TILES.mkdir(parents=True, exist_ok=True)
    tiles = json.load(open(S1F / "chip_bounds.json"))["tiles"]
    got, none, nbytes = 0, 0, 0
    for la, lo in tiles:
        name = f"{'n' if la >= 0 else 's'}{abs(la):02d}{'e' if lo >= 0 else 'w'}{abs(lo):03d}"
        if (TILES / f"{name}.hgt").exists() or (TILES / f"{name}.none").exists():
            continue
        res = earthaccess.search_data(short_name="NASADEM_HGT", bounding_box=(lo + 0.45, la + 0.45, lo + 0.55, la + 0.55))
        urls = [u for g in res for u in g.data_links(access="external") if u.endswith(".zip") and name in u.lower()]
        if not urls:
            (TILES / f"{name}.none").touch(); none += 1; continue
        z = TILES / urls[0].split("/")[-1]
        for k in range(5):
            try:
                with sess.get(urls[0], stream=True, timeout=120) as r:
                    r.raise_for_status()
                    with open(z, "wb") as f:
                        for ch in r.iter_content(1 << 20):
                            f.write(ch)
                break
            except Exception as e:
                print("retry", name, e); time.sleep(5 * (k + 1))
        nbytes += z.stat().st_size
        with zipfile.ZipFile(z) as zf:
            m = [x for x in zf.namelist() if x.endswith(".hgt")][0]
            (TILES / f"{name}.hgt").write_bytes(zf.read(m))
        z.unlink(); got += 1
        print(f"{name} ok ({got} downloaded, {none} without tile, {nbytes/1e9:.2f} GB)", flush=True)
    print(f"done: {got} tiles downloaded ({nbytes/1e9:.2f} GB), {none} tiles have no NASADEM (ocean)")


def s1f11():
    out_dir = S1F / "terrain"; out_dir.mkdir(exist_ok=True)
    srcs = {Path(f).stem: rasterio.open(f) for f in glob.glob(str(TILES / "*.hgt"))}
    chips = json.load(open(S1F / "chip_bounds.json"))["chips"]
    t0 = time.time(); stats = []
    for i, c in enumerate(chips):
        out = out_dir / f"{c['chip']}_Terrain.tif"
        if out.exists():
            continue
        w, s, e, n = c["w"] - MARGIN, c["s"] - MARGIN, c["e"] + MARGIN, c["n"] + MARGIN
        use = [srcs[k] for k in srcs if (lambda b: b.left < e and b.right > w and b.bottom < n and b.top > s)(srcs[k].bounds)]
        res = 1 / 3600
        if use:
            dem, dt = merge(use, bounds=(w, s, e, n), res=res, nodata=-32768)
            dem = dem[0]
        else:
            dt = from_origin(w, n, res, res)
            dem = np.full((int(round((n - s) / res)), int(round((e - w) / res))), -32768, np.int16)
        hand, dist = terrain_from_dem(dem, dt)
        with rasterio.open(S1F / "data" / "flood_events" / "HandLabeled" / "S1Hand" / f"{c['chip']}_S1Hand.tif") as ref:
            shp, rt, rc = (ref.height, ref.width), ref.transform, ref.crs
        h2 = to_grid(hand, dt, "EPSG:4326", shp, rt, rc); d2 = to_grid(dist, dt, "EPSG:4326", shp, rt, rc)
        with rasterio.open(out, "w", driver="GTiff", height=shp[0], width=shp[1], count=2, dtype="float32", crs=rc,
                           transform=rt, compress="deflate", predictor=3, tiled=True, blockxsize=256, blockysize=256) as dst:
            dst.write(np.stack([h2, d2]))
            dst.set_band_description(1, "HAND_m"); dst.set_band_description(2, "dist_to_stream_m")
        stats.append((c["chip"], float(np.median(h2)), float(np.median(d2)), len(use)))
        if (i + 1) % 25 == 0:
            print(f"{i+1}/{len(chips)}  {time.time()-t0:.0f}s  last: HAND median {stats[-1][1]:.1f} m, dist median {stats[-1][2]:.0f} m", flush=True)
    print("done", len(list(out_dir.glob("*_Terrain.tif"))), "chips")


def teesta():
    sys.path.insert(0, str(ROOT / "src"))
    from common import DER, CRS, grid, aoi_mask, write
    srcs = [rasterio.open(f) for f in sorted(glob.glob(str(ROOT / "data" / "nasadem" / "*.hgt")))]
    dem, dt = merge(srcs, nodata=-32768)
    hand, dist = terrain_from_dem(dem[0], dt)
    t, h, w = grid(30); inside = aoi_mask(30)
    h2 = to_grid(hand, dt, "EPSG:4326", (h, w), t, CRS); d2 = to_grid(dist, dt, "EPSG:4326", (h, w), t, CRS)
    h2[~inside] = np.nan; d2[~inside] = np.nan
    write(DER / "hand_dem_30m.tif", h2, nodata=np.nan, dtype="float32")
    write(DER / "dist_stream_30m.tif", d2, nodata=np.nan, dtype="float32")
    print("Teesta HAND (pysheds) p5/p50/p95: %.1f / %.1f / %.1f m" % tuple(np.nanpercentile(h2[inside], [5, 50, 95])),
          "| distance to stream p50 %.0f m" % np.nanpercentile(d2[inside], 50))


if __name__ == "__main__":
    {"download": download, "s1f11": s1f11, "teesta": teesta,
     "clean": lambda: shutil.rmtree(TILES, ignore_errors=True)}[sys.argv[1]]()
