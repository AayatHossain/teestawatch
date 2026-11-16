"""ConvGRU U-Net: next-year river channel and erosion forecast from past dry seasons (HLS, 30 m).

Train on target years 2018-2023, select on 2024, test on 2025-2026, forecast 2027.
TW_FINAL=1 retrains on all years 2018-2026 for a fixed number of steps and writes the 2027 forecast.
Env: TW_SMOKE=1, TW_OUT, TW_EROSION_DATA (folder with stacks.npz), TW_FINAL.
"""
import copy
import csv
import glob
import json
import math
import os
import platform
import random
import time
from pathlib import Path

import numpy as np

SMOKE = os.environ.get("TW_SMOKE") == "1"
FINAL = os.environ.get("TW_FINAL") == "1"
NAME = "convgru_final" if FINAL else "convgru"
OUT = Path(os.environ.get("TW_OUT", f"/kaggle/working/{NAME}")); OUT.mkdir(parents=True, exist_ok=True)
BC = Path(os.environ.get("TW_EROSION_DATA") or Path(sorted(glob.glob("/kaggle/input/**/stacks.npz", recursive=True))[0]).parent)
SEEDS = [0] if SMOKE else [0, 1, 2]
STEPS = int(os.environ.get("TW_STEPS", 4 if SMOKE else (1000 if FINAL else 2000)))
EVAL_EVERY = 10 ** 9 if FINAL else (2 if SMOKE else 250)
BATCH = 2 if SMOKE else 8
TILE, T_MAX = 256, 5
LR, WD, WARM = 3e-4, 1e-4, (1 if SMOKE else 100)
TRAIN_Y = list(range(2018, 2027)) if FINAL else [2018, 2019, 2020, 2021, 2022, 2023]
VAL_Y, TEST_Y, FORECAST_Y = 2024, [2025, 2026], 2027
ZONE_M = 1000
LOGF = open(OUT / "log.txt", "a", encoding="utf-8")


def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from scipy import ndimage
from scipy.stats import rankdata

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

z = np.load(BC / "stacks.npz")
YEARS = [int(y) for y in z["years"]]; Y0 = YEARS[0]
FEATS, NOBS, CHAN, CORR = z["feats"], z["nobs"], z["channel"], z["corridor"]
INSIDE = z["inside"]
st = z["static"].astype(np.float32)
st[0] = np.log1p(np.clip(st[0], 0, 100)); st[2] = np.log1p(np.clip(st[2], 0, 10000))
for k in range(3):
    st[k] = (st[k] - st[k][INSIDE].mean()) / (st[k][INSIDE].std() + 1e-6)
st[:, ~INSIDE] = 0
STATIC = st
if SMOKE:
    rr, cc = np.where(CHAN[0]); r0, c0 = int(np.median(rr)) - 384, int(np.median(cc)) - 384
    sl = np.s_[r0:r0 + 768, c0:c0 + 768]
    FEATS, NOBS, CHAN, CORR = FEATS[(slice(None), slice(None)) + sl], NOBS[(slice(None),) + sl], CHAN[(slice(None),) + sl], CORR[(slice(None),) + sl]
    INSIDE, STATIC = INSIDE[sl], STATIC[(slice(None),) + sl]
H, W = INSIDE.shape
YI = lambda y: y - Y0


def zone(y):
    """Land the river has not reached yet, within ZONE_M of it."""
    c = CORR[YI(y)]
    return INSIDE & ~c & (ndimage.distance_transform_edt(~c) * 30 <= ZONE_M)


def stack(Y, r=0, c=0, h=None, w=None):
    h = h or H; w = w or W
    x = np.zeros((T_MAX, 7, h, w), np.float32); m = np.zeros(T_MAX, bool)
    for j, yy in enumerate(range(Y - T_MAX, Y)):
        if yy < Y0 or yy > YEARS[-1]:
            continue
        i = YI(yy); s = np.s_[r:r + h, c:c + w]
        f = FEATS[i][(slice(None),) + s].astype(np.float32)
        x[j, 0] = f[0] / 250; x[j, 1] = f[1] / 250; x[j, 2] = f[2] / 125 - 1; x[j, 3] = f[3] / 125 - 1
        x[j, 4] = CHAN[i][s]; x[j, 5] = CORR[i][s]; x[j, 6] = NOBS[i][s] >= 2
        m[j] = True
    return x, m, np.arange(T_MAX, 0, -1)


def targets(Y, r=0, c=0, h=None, w=None):
    h = h or H; w = w or W; s = np.s_[r:r + h, c:c + w]
    ins = INSIDE[s]
    yc = np.where(ins, CHAN[YI(Y)][s], -1).astype(np.int8)
    ye = np.where(ins & ~CORR[YI(Y - 1)][s], CORR[YI(Y)][s] & ~CORR[YI(Y - 1)][s], -1).astype(np.int8)
    return yc, ye


CENTRES = {Y: (np.argwhere(zone(Y - 1)), np.argwhere(INSIDE)) for Y in TRAIN_Y}


def sample(Y):
    zr, ar = CENTRES[Y]
    p = zr[np.random.randint(len(zr))] if random.random() < 0.8 else ar[np.random.randint(len(ar))]
    r = int(np.clip(p[0] - TILE // 2 + np.random.randint(-64, 65), 0, H - TILE)); c = int(np.clip(p[1] - TILE // 2 + np.random.randint(-64, 65), 0, W - TILE))
    x, m, rel = stack(Y, r, c, TILE, TILE); s = STATIC[:, r:r + TILE, c:c + TILE].copy(); yc, ye = targets(Y, r, c, TILE, TILE)
    k = random.randint(0, 3); fl = random.random() < 0.5
    aug = lambda a: np.ascontiguousarray(np.flip(np.rot90(a, k, (-2, -1)), -1) if fl else np.rot90(a, k, (-2, -1)))
    return aug(x), m, rel, aug(s), aug(yc), aug(ye)


class ConvBNReLU(nn.Sequential):
    def __init__(self, cin, cout, k=3):
        super().__init__(nn.Conv2d(cin, cout, k, padding=k // 2, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class DecoderBlock(nn.Module):
    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.c1 = ConvBNReLU(cin + cskip, cout); self.c2 = ConvBNReLU(cout, cout)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], 1)
        return self.c2(self.c1(x))


class Encoder(nn.Module):
    def __init__(self, cin):
        super().__init__()
        r = torchvision.models.resnet34(weights=None if SMOKE and not os.environ.get("TW_PRETRAINED") else torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
        c1 = nn.Conv2d(cin, 64, 7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            c1.weight[:, :3] = r.conv1.weight; c1.weight[:, 3:] = r.conv1.weight.mean(1, keepdim=True)
        self.stem = nn.Sequential(c1, r.bn1, r.relu); self.pool = r.maxpool
        self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4

    def forward(self, x):
        s0 = self.stem(x); s1 = self.l1(self.pool(s0)); s2 = self.l2(s1); s3 = self.l3(s2); s4 = self.l4(s3)
        return [s0, s1, s2, s3, s4]


class ConvGRU(nn.Module):
    def __init__(self, c, hid):
        super().__init__()
        self.hid = hid
        self.zr = nn.Conv2d(c + hid, 2 * hid, 3, padding=1); self.hh = nn.Conv2d(c + hid, hid, 3, padding=1)

    def forward(self, xs, mask):
        B, T = xs.shape[:2]; h = xs.new_zeros(B, self.hid, *xs.shape[3:])
        for t in range(T):
            x = xs[:, t]
            z_, r = torch.sigmoid(self.zr(torch.cat([x, h], 1))).chunk(2, 1)
            hn = (1 - z_) * h + z_ * torch.tanh(self.hh(torch.cat([x, r * h], 1)))
            m = mask[:, t].view(B, 1, 1, 1).to(h.dtype); h = m * hn + (1 - m) * h
        return h


class Temporal(nn.Module):
    def __init__(self, c):
        super().__init__()
        hid = min(c // 2, 128)
        self.gru = ConvGRU(c, hid); self.proj = nn.Conv2d(hid, c, 1)

    def forward(self, f, mask):
        return self.proj(self.gru(f, mask))


CH = [64, 64, 128, 256, 512]


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder(10)
        self.temp = nn.ModuleList([Temporal(c) for c in CH])
        dec = [(512, 256, 256), (256, 128, 128), (128, 64, 64), (64, 64, 32), (32, 0, 16)]
        self.dec = nn.ModuleList([DecoderBlock(*d) for d in dec])
        self.head_c = nn.Conv2d(16, 1, 1)
        self.head_e = nn.Sequential(ConvBNReLU(16, 16), nn.Conv2d(16, 1, 1))
        nn.init.constant_(self.head_e[-1].bias, -4.0)

    def forward(self, x, mask, rel, s):
        B, T = x.shape[:2]
        x = torch.cat([x, s.unsqueeze(1).expand(B, T, *s.shape[1:])], 2)
        feats = self.enc(x.flatten(0, 1))
        sk = [t(f.view(B, T, *f.shape[1:]), mask) for t, f in zip(self.temp, feats)]
        y = sk[4]
        for i, d in enumerate(self.dec):
            y = d(y, sk[3 - i] if i < 4 else None)
        return {"channel": self.head_c(y), "erosion": self.head_e(y)}


def dice(logit, t, v):
    p = torch.sigmoid(logit) * v; t = t * v
    return (1 - (2 * (p * t).sum((1, 2, 3)) + 1) / (p.sum((1, 2, 3)) + t.sum((1, 2, 3)) + 1)).mean()


def focal(logit, t, v, g=2.0, a=0.75):
    p = torch.sigmoid(logit); ce = F.binary_cross_entropy_with_logits(logit, t, reduction="none")
    pt = p * t + (1 - p) * (1 - t); w = a * t + (1 - a) * (1 - t)
    return (w * (1 - pt) ** g * ce * v).sum() / v.sum().clamp(min=1)


def loss_fn(out, yc, ye):
    vc = (yc >= 0).float(); tc = yc.clamp(min=0).float(); ve = (ye >= 0).float(); te = ye.clamp(min=0).float()
    lc = (F.binary_cross_entropy_with_logits(out["channel"], tc, reduction="none") * vc).sum() / vc.sum().clamp(min=1) + dice(out["channel"], tc, vc)
    return 0.5 * lc + focal(out["erosion"], te, ve) + dice(out["erosion"], te, ve)


def auc_ap(score, y):
    score = np.asarray(score, np.float64); y = np.asarray(y, bool)
    npos, nneg = y.sum(), (~y).sum()
    if npos == 0 or nneg == 0:
        return float("nan"), float("nan")
    ys = y[np.argsort(-score, kind="mergesort")]
    auc = (rankdata(score)[y].sum() - npos * (npos + 1) / 2) / (npos * nneg)
    prec = np.cumsum(ys) / np.arange(1, len(ys) + 1)
    return float(auc), float(prec[ys].mean())


def prf(pred, y):
    tp = int((pred & y).sum()); fp = int((pred & ~y).sum()); fn = int((~pred & y).sum())
    return dict(f1=2 * tp / max(2 * tp + fp + fn, 1), precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1), iou=tp / max(tp + fp + fn, 1))


def best_thr(score, y):
    grid = np.unique(np.quantile(score, np.linspace(0.5, 0.9995, 200)))
    return float(grid[int(np.argmax([prf(score >= t, y)["f1"] for t in grid]))])


def eroded(Y):
    return CORR[YI(Y)] & ~CORR[YI(Y - 1)]


def score_year(Y, pc, pe, thr):
    zn = zone(Y - 1); ye = eroded(Y)[zn]; e = pe[zn]
    auc, ap = auc_ap(e, ye)
    r = dict(erosion_auc=auc, erosion_ap=ap, erosion_base_rate=float(ye.mean()), expected_ha=float(e.sum() * 0.09))
    r.update({f"erosion_{k}": v for k, v in prf(e >= thr, ye).items()}); r["erosion_threshold"] = thr
    r["channel_iou"] = prf(pc[INSIDE] >= 0.5, CHAN[YI(Y)][INSIDE])["iou"]
    return {k: round(float(v), 5) for k, v in r.items()}


@torch.no_grad()
def predict(model, Y, tile=TILE, step=128):
    model.eval()
    x, m, rel = stack(Y)
    acc_c = np.zeros((H, W), np.float32); acc_e = np.zeros((H, W), np.float32); cnt = np.zeros((H, W), np.float32)
    wgt = np.outer(np.hanning(tile + 2)[1:-1], np.hanning(tile + 2)[1:-1]).astype(np.float32) + 1e-3
    pos = [(r, c) for r in list(range(0, max(H - tile, 0) + 1, step)) + [max(H - tile, 0)]
           for c in list(range(0, max(W - tile, 0) + 1, step)) + [max(W - tile, 0)]]
    pos = [p for p in dict.fromkeys(pos) if INSIDE[p[0]:p[0] + tile, p[1]:p[1] + tile].any()]
    mt = torch.from_numpy(m).to(DEV)[None]; rt = torch.from_numpy(rel).to(DEV)[None]
    for i in range(0, len(pos), 8):
        ps = pos[i:i + 8]
        xb = torch.from_numpy(np.stack([x[:, :, r:r + tile, c:c + tile] for r, c in ps])).to(DEV)
        sb = torch.from_numpy(np.stack([STATIC[:, r:r + tile, c:c + tile] for r, c in ps])).to(DEV)
        with torch.autocast(DEV.type, enabled=DEV.type == "cuda"):
            o = model(xb, mt.expand(len(ps), -1), rt.expand(len(ps), -1), sb)
        pcs = torch.sigmoid(o["channel"].float())[:, 0].cpu().numpy(); pes = torch.sigmoid(o["erosion"].float())[:, 0].cpu().numpy()
        for (r, c), a, b in zip(ps, pcs, pes):
            acc_c[r:r + tile, c:c + tile] += a * wgt; acc_e[r:r + tile, c:c + tile] += b * wgt; cnt[r:r + tile, c:c + tile] += wgt
    cnt = np.maximum(cnt, 1e-6)
    pc, pe = acc_c / cnt, acc_e / cnt
    pc[~INSIDE] = 0; pe[~INSIDE] = 0
    if Y - 1 <= YEARS[-1]:
        pe[CORR[YI(Y - 1)]] = 0
    return pc, pe


def run(seed):
    rd = OUT / "runs" / f"seed{seed}"
    if (rd / "metrics.json").exists():
        log(f"seed {seed}: done already"); return json.load(open(rd / "metrics.json"))
    rd.mkdir(parents=True, exist_ok=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = Net().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / WARM if s < WARM else 0.5 * (1 + math.cos(math.pi * (s - WARM) / max(STEPS - WARM, 1))))
    scaler = torch.amp.GradScaler(enabled=DEV.type == "cuda")
    best, best_state, best_step, hist, tl, t0 = -1, None, 0, [], 0.0, time.time()
    for step in range(1, STEPS + 1):
        model.train()
        b = [sample(random.choice(TRAIN_Y)) for _ in range(BATCH)]
        x, m, rel, s, yc, ye = (torch.from_numpy(np.stack(a)).to(DEV) for a in zip(*b))
        with torch.autocast(DEV.type, enabled=DEV.type == "cuda"):
            out = model(x, m, rel, s)
        loss = loss_fn({k: v.float() for k, v in out.items()}, yc[:, None], ye[:, None])
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        tl += loss.item()
        if step % EVAL_EVERY == 0 or (step == STEPS and not FINAL):
            _, pe = predict(model, VAL_Y)
            zn = zone(VAL_Y - 1); auc, ap = auc_ap(pe[zn], eroded(VAL_Y)[zn])
            hist.append(dict(step=step, train_loss=tl / EVAL_EVERY, val_erosion_ap=ap, val_erosion_auc=auc, sec=round(time.time() - t0))); tl = 0.0
            if ap > best:
                best, best_step, best_state = ap, step, copy.deepcopy(model.state_dict())
            log(f"seed {seed} step {step}/{STEPS} val AP {ap:.4f} AUC {auc:.4f} best {best:.4f}@{best_step}")
    best_state = best_state or copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state); torch.save(best_state, rd / "weights.pt")
    if hist:
        with open(rd / "history.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(hist[0])); w.writeheader(); w.writerows(hist)
    probs = {Y: predict(model, Y) for Y in [VAL_Y] + TEST_Y + [FORECAST_Y]}
    zn = zone(VAL_Y - 1)
    thr = best_thr(probs[VAL_Y][1][zn], eroded(VAL_Y)[zn])
    res = dict(seed=seed, final=FINAL, best_step=best_step or STEPS, train_sec=round(time.time() - t0),
               params=sum(p.numel() for p in model.parameters()),
               scores={str(Y): score_year(Y, *probs[Y], thr) for Y in [VAL_Y] + TEST_Y})
    json.dump(res, open(rd / "metrics.json", "w"), indent=1)
    np.savez_compressed(rd / "probs.npz", **{f"{t}_{Y}": (probs[Y][i] * 255).round().astype(np.uint8)
                                            for Y in probs for i, t in enumerate(("channel", "erosion"))})
    log(f"seed {seed} done ({res['train_sec'] / 60:.1f} min)")
    return res


if __name__ == "__main__":
    log(f"{NAME} start smoke={SMOKE} final={FINAL} device={DEV}")
    with open(OUT / "versions.txt", "w") as f:
        f.write(f"python {platform.python_version()}\ntorch {torch.__version__}\ntorchvision {torchvision.__version__}\nnumpy {np.__version__}\n")
    json.dump(dict(seeds=SEEDS, steps=STEPS, batch=BATCH, tile=TILE, t_max=T_MAX, lr=LR, wd=WD, warmup=WARM, final=FINAL,
                   train_years=TRAIN_Y, val_year=VAL_Y, test_years=TEST_Y, forecast_year=FORECAST_Y, zone_m=ZONE_M),
              open(OUT / "config.json", "w"), indent=1)
    for seed in SEEDS:
        run(seed)
    P = [np.load(OUT / "runs" / f"seed{s}" / "probs.npz") for s in SEEDS]
    mean = lambda key: np.mean([p[key].astype(np.float32) / 255 for p in P], 0)
    zn = zone(VAL_Y - 1)
    thr = best_thr(mean(f"erosion_{VAL_Y}")[zn], eroded(VAL_Y)[zn])
    ens = {str(Y): score_year(Y, mean(f"channel_{Y}"), mean(f"erosion_{Y}"), thr) for Y in [VAL_Y] + TEST_Y}
    json.dump(ens, open(OUT / "ensemble.json", "w"), indent=1)
    np.savez_compressed(OUT / f"forecast_{FORECAST_Y}.npz", channel=(mean(f"channel_{FORECAST_Y}") * 255).round().astype(np.uint8),
                        erosion=(mean(f"erosion_{FORECAST_Y}") * 255).round().astype(np.uint8), erosion_threshold=thr)
    for Y, m in ens.items():
        log(f"{Y}: erosion AUC {m['erosion_auc']:.3f} AP {m['erosion_ap']:.3f} F1 {m['erosion_f1']:.3f} | channel IoU {m['channel_iou']:.3f}")
    log("done")
