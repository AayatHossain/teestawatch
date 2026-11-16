"""Pack the Teesta fine-tuning dataset with a spatial block split."""
import glob, json, re, shutil, datetime as dt
import numpy as np, pandas as pd, rasterio
from rasterio.warp import reproject, Resampling
from common import CRS, DATA, DER, MODELS, ROOT, RUNS, grid, teesta_mask

OUT = DATA / "ft_teesta"; OUT.mkdir(exist_ok=True)
TRAIN = ["2017-07-17", "2020-06-07", "2024-06-10"]
TEST = ["2025-07-23", "2025-08-16"]
BLOCK = 1536
t10, H, W = grid(10); t30, h30, w30 = grid(30)
m30 = teesta_mask(30)
inside = np.kron(m30, np.ones((3, 3), bool))[:H, :W]
rr, cc = np.where(inside)
r0, r1 = (rr.min() // 512) * 512, min(H, (rr.max() // 512 + 1) * 512)
c0, c1 = (cc.min() // 512) * 512, min(W, (cc.max() // 512 + 1) * 512)
del rr, cc
S = np.s_[r0:r1, c0:c1]
print(f"crop rows {r0}:{r1} cols {c0}:{c1} -> {r1-r0} x {c1-c0}", flush=True)


def up(a30):
    return np.kron(a30, np.ones((3, 3), a30.dtype))[:H, :W][S]


def on30(path, nodata=255):
    d = np.full((h30, w30), nodata, np.uint8)
    with rasterio.open(path) as s:
        reproject(rasterio.band(s, 1), d, src_nodata=nodata, dst_transform=t30, dst_crs=CRS, dst_nodata=nodata, resampling=Resampling.nearest)
    return d


def terrain10(name):
    out = np.zeros((H, W), np.float32)
    with rasterio.open(DER / name) as s:
        reproject(rasterio.band(s, 1), out, dst_transform=t10, dst_crs=CRS, resampling=Resampling.bilinear)
    return np.nan_to_num(out)[S]


def ddate(f):
    return dt.datetime.strptime(re.search(r"_(\d{8})T", f).group(1), "%Y%m%d").date()


def dswx(folder, date, keep):
    c = dt.date.fromisoformat(date); lab = np.full((h30, w30), 255, np.uint8)
    for f in sorted(glob.glob(str(DATA / folder / "*_B01_WTR.tif")), key=lambda f: abs((ddate(f) - c).days)):
        if abs((ddate(f) - c).days) > keep:
            break
        a = on30(f); fill = (lab == 255) & (a < 250)
        lab[fill] = a[fill]
    return lab


hand = np.log1p(np.clip(terrain10("hand_dem_30m.tif"), 0, 100)).astype(np.float16)
dist = np.log1p(np.clip(terrain10("dist_stream_30m.tif"), 0, 10000)).astype(np.float16)
ins = inside[S]
br, bc = np.meshgrid(np.arange(r1 - r0) // BLOCK, np.arange(c1 - c0) // BLOCK, indexing="ij")
occ = sorted({(int(a), int(b)) for a, b in zip(br[ins], bc[ins])}, key=lambda k: (k[0] + k[1], k[0]))
vset = {k for i, k in enumerate(occ) if i % 4 == 2}
block = np.zeros(ins.shape, np.uint8)
for a, b in vset:
    block[a * BLOCK:(a + 1) * BLOCK, b * BLOCK:(b + 1) * BLOCK] = 1
notile = np.ones_like(ins)
for _, r in pd.read_csv(ROOT / "evaluation" / "hand_labels" / "tiles.csv").iterrows():
    a, b = int(r.row) - r0, int(r.col) - c0
    notile[max(a, 0):max(a + 256, 0), max(b, 0):max(b + 256, 0)] = False
np.savez_compressed(OUT / "static.npz", hand=hand, dist=dist, inside=ins, block=block, notile=notile,
                    origin=np.array([r0, c0]), block_size=BLOCK)
print("static: inside", ins.sum() * 1e-4, "km2 | validation-block share", round(float(block[ins].mean()), 3),
      "| tile px", int((~notile).sum()), flush=True)

info = {}
for date in TRAIN + TEST:
    rec = {}
    pol = {}
    for p in ("VV", "VH"):
        with rasterio.open(DER / "s1_10m" / f"{date}_{p}.tif") as s:
            a = s.read(1)[S]
        pol[p] = np.where(a != -32768, a - 100, -32768).astype(np.int16)
    ok = (pol["VV"] != -32768) & (pol["VH"] != -32768) & ins
    rec["radar_km2"] = round(ok.sum() * 1e-4, 1)
    arrs = dict(vv=pol["VV"], vh=pol["VH"])
    lf = DER / "ft_labels" / f"label_{date}.tif"
    lab30 = rasterio.open(lf).read(1)
    if date >= "2023-04":
        hcls = dswx("opera_dswx_hls", date, 3)
        rec["partial_water_set_unknown_km2"] = round(float(((lab30 != 255) & (hcls == 2)).sum() * 9e-4), 1)
        lab30 = np.where(hcls == 2, 255, lab30).astype(np.uint8)
    lab = up(lab30); lab[~ok] = 255
    if (lab != 255).any():
        arrs["lab"] = lab
        rec.update(label_km2=round((lab != 255).sum() * 1e-4, 1), label_water_share=round(float((lab[lab != 255] == 1).mean()), 3))
        if date in TRAIN:
            v = (lab != 255) & notile
            rec["train_km2"] = round((v & (block == 0)).sum() * 1e-4, 1); rec["val_km2"] = round((v & (block == 1)).sum() * 1e-4, 1)
    if date in TEST:
        scls = dswx("opera_dswx_s1", date, 0)
        ref = np.full_like(scls, 255); ref[scls == 0] = 0; ref[np.isin(scls, [1, 3])] = 1
        ref = up(ref); ref[~ok] = 255
        arrs["s1ref"] = ref
        rec["dswx_s1_km2"] = round((ref != 255).sum() * 1e-4, 1); rec["dswx_s1_water_share"] = round(float((ref[ref != 255] == 1).mean()), 3)
    np.savez_compressed(OUT / f"date_{date}.npz", **arrs)
    info[date] = rec
    print(date, rec, flush=True)

json.dump(dict(train=TRAIN, test=TEST, origin=[int(r0), int(c0)], shape=[int(r1 - r0), int(c1 - c0)], dates=info),
          open(OUT / "info.json", "w"), indent=1)
stage1 = RUNS / "tgflood"
shutil.copy(MODELS / "flood" / "tgfloodnet.py", OUT / "tgfloodnet.py")
shutil.copy(stage1 / "norm.json", OUT / "norm.json")
for s in (0, 1, 2):
    shutil.copy(stage1 / "runs" / f"seed{s}" / "weights.pt", OUT / f"tg_seed{s}.pt")
print("packed:", sorted(p.name for p in OUT.rglob("*") if p.is_file()))
