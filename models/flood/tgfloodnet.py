"""TG-FloodNet: terrain-gated ResNet-34 U-Net for SAR flood mapping, trained on Sen1Floods11."""
import os, sys, json, time, math, random, glob, csv, shutil, subprocess, platform, io
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np

ABL = sorted(set(filter(None, os.environ.get("TW_ABLATE", "").split(","))))
assert set(ABL) <= {"a1", "a2", "a3a", "a3b", "a4", "a5b", "ds"}, ABL
NAME = "tgflood" + ("_no_" + "_".join(ABL) if ABL else "")
SMOKE = os.environ.get("TW_SMOKE") == "1"
ON_KAGGLE = Path("/kaggle/working").exists()
OUT = Path(os.environ.get("TW_OUT", f"/kaggle/working/{NAME}"))
DATA = Path(os.environ.get("TW_DATA", "/tmp/sen1floods11"))
IN_CH = 5
CHANNELS = ["VV", "VH", "VV-VH", "log1p_HAND", "log1p_dist_stream"]
_TERR = {}


def terrain_path(chip):
    if not _TERR:
        roots = [os.environ["TW_TERRAIN"]] if os.environ.get("TW_TERRAIN") else ["/kaggle/input"]
        for r in roots:
            for p in glob.glob(os.path.join(r, "**", "*_Terrain.tif"), recursive=True):
                _TERR[os.path.basename(p).replace("_Terrain.tif", "")] = p
        assert _TERR, f"no terrain tifs found under {roots}"
    return _TERR[chip]
SEEDS = [0] if SMOKE else [0, 1, 2]
EPOCHS = int(os.environ.get("TW_EPOCHS", 1 if SMOKE else 120))
WARMUP = 0 if SMOKE else 3
BATCH = 4 if SMOKE else 8
LR, WD = 5e-4, 1e-4
CLIP = (-40.0, 10.0)
TIME_BUDGET_H = 11.0
T_START = time.time()
TRAIN_FRAC = [0.0]
OUT.mkdir(parents=True, exist_ok=True)
LOGF = open(OUT / "log.txt", "a", encoding="utf-8")


def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True)
    LOGF.write(s + "\n"); LOGF.flush()


def pip(pkg, mod=None):
    try:
        __import__(mod or pkg)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)


pip("rasterio")
import rasterio
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SPLITS = {"train": "flood_train_data.csv", "valid": "flood_valid_data.csv",
          "test": "flood_test_data.csv", "bolivia": "flood_bolivia_data.csv"}


def download():
    import urllib.request, urllib.parse
    need = [f"v1.1/splits/flood_handlabeled/{v}" for v in SPLITS.values()]
    for sub in ("S1Hand", "LabelHand"):
        tok = None
        while True:
            q = {"prefix": f"v1.1/data/flood_events/HandLabeled/{sub}/", "fields": "items(name),nextPageToken"}
            if tok:
                q["pageToken"] = tok
            r = json.load(urllib.request.urlopen("https://storage.googleapis.com/storage/v1/b/sen1floods11/o?" + urllib.parse.urlencode(q)))
            need += [i["name"] for i in r.get("items", []) if i["name"].endswith(".tif")]
            tok = r.get("nextPageToken")
            if not tok:
                break

    def get(name):
        dst = DATA / name.replace("v1.1/", "", 1)
        if dst.exists():
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        url = "https://storage.googleapis.com/sen1floods11/" + urllib.parse.quote(name)
        for k in range(5):
            try:
                tmp = dst.with_suffix(".part")
                urllib.request.urlretrieve(url, tmp); tmp.replace(dst); return
            except Exception:
                time.sleep(3 * (k + 1))
        raise RuntimeError(name)

    with ThreadPoolExecutor(16) as ex:
        list(ex.map(get, need))
    log(f"data ready: {len(need)} files in {DATA}")


def read_split(split):
    rows = list(csv.reader(open(DATA / "splits" / "flood_handlabeled" / SPLITS[split])))
    xs, ys, ids, oks = [], [], [], []
    for s1, lab in rows:
        with rasterio.open(DATA / "data" / "flood_events" / "HandLabeled" / "S1Hand" / s1.strip()) as f:
            a = f.read().astype(np.float32)
        with rasterio.open(DATA / "data" / "flood_events" / "HandLabeled" / "LabelHand" / lab.strip()) as f:
            y = f.read(1).astype(np.int8)
        ok = np.isfinite(a).all(0)
        a = np.clip(np.nan_to_num(a, nan=CLIP[0]), *CLIP)
        x = np.stack([a[0], a[1], a[0] - a[1]])
        with rasterio.open(terrain_path(s1.strip().replace("_S1Hand.tif", ""))) as f:
            tr = f.read().astype(np.float32)
        tr = np.nan_to_num(tr, nan=0.0)
        x = np.concatenate([x, np.log1p(np.clip(tr[0:1], 0, 100)), np.log1p(np.clip(tr[1:2], 0, 10000))])
        y[~ok] = -1
        xs.append(x); ys.append(y); oks.append(ok); ids.append(s1.strip().replace("_S1Hand.tif", ""))
    return np.stack(xs), np.stack(ys), ids, np.stack(oks)


def load_all():
    if not (DATA / "splits" / "flood_handlabeled" / SPLITS["test"]).exists():
        download()
    d = {}
    for s in SPLITS:
        x, y, ids, ok = read_split(s)
        if SMOKE:
            n = {"train": 16, "valid": 8, "test": 8, "bolivia": 4}[s]
            x, y, ids, ok = x[:n], y[:n], ids[:n], ok[:n]
        d[s] = [x, y, ids, ok]
    m = d["train"][3]
    mean = np.array([d["train"][0][:, c][m].mean() for c in range(IN_CH)], np.float32)
    std = np.array([d["train"][0][:, c][m].std() for c in range(IN_CH)], np.float32)
    for s in d:
        x, ok = d[s][0], d[s][3]
        x = (x - mean[None, :, None, None]) / std[None, :, None, None]
        d[s][0] = np.where(ok[:, None], x, 0).astype(np.float32)
        d[s] = d[s][:3]
    sizes = {s: len(d[s][2]) for s in d}
    log("chips:", sizes, "mean", mean.round(3).tolist(), "std", std.round(3).tolist())
    if not SMOKE:
        assert sizes == {"train": 252, "valid": 89, "test": 90, "bolivia": 15}, sizes
    return d, mean, std


class ConvBNReLU(nn.Sequential):
    def __init__(self, cin, cout, k=3):
        super().__init__(nn.Conv2d(cin, cout, k, padding=k // 2, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class DecoderBlock(nn.Module):
    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.c1 = ConvBNReLU(cin + cskip, cout)
        self.c2 = ConvBNReLU(cout, cout)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], 1)
        return self.c2(self.c1(x))


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        r = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)
        self.pool = r.maxpool
        self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.l1(self.pool(s0))
        s2 = self.l2(s1)
        s3 = self.l3(s2)
        s4 = self.l4(s3)
        return [s0, s1, s2, s3, s4]


class UNetR34(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder()
        self.d4 = DecoderBlock(512, 256, 256)
        self.d3 = DecoderBlock(256, 128, 128)
        self.d2 = DecoderBlock(128, 64, 64)
        self.d1 = DecoderBlock(64, 64, 32)
        self.d0 = DecoderBlock(32, 0, 16)
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        s0, s1, s2, s3, s4 = self.enc(x)
        y = self.d4(s4, s3)
        y = self.d3(y, s2)
        y = self.d2(y, s1)
        y = self.d1(y, s0)
        y = self.d0(y)
        return {"water": self.head(y)}

def small_init(conv, std=1e-3):
    nn.init.normal_(conv.weight, std=std)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)


class SE(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, max(c // r, 4), 1), nn.ReLU(inplace=True),
                                nn.Conv2d(max(c // r, 4), c, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x)


class PolStem(nn.Module):
    def __init__(self, cin=3, cb=16):
        super().__init__()
        self.br = nn.ModuleList([nn.Sequential(ConvBNReLU(1, cb), ConvBNReLU(cb, cb)) for _ in range(cin)])
        self.se = SE(cin * cb)
        self.out = nn.Conv2d(cin * cb, cin, 1); small_init(self.out)

    def forward(self, x):
        f = torch.cat([b(x[:, i:i + 1]) for i, b in enumerate(self.br)], 1)
        return x + self.out(self.se(f))


class TerrainBranch(nn.Module):
    def __init__(self, cin=2, chs=(64, 64, 128, 256)):
        super().__init__()
        self.blocks, self.gates, c = nn.ModuleList(), nn.ModuleList(), cin
        for co in chs:
            self.blocks.append(nn.Sequential(nn.Conv2d(c, co, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(co),
                                             nn.ReLU(inplace=True), ConvBNReLU(co, co)))
            g = nn.Conv2d(co, co, 1); small_init(g); self.gates.append(g); c = co

    def forward(self, t):
        gs, f = [], t
        for b, g in zip(self.blocks, self.gates):
            f = b(f); gs.append(2 * torch.sigmoid(g(f)))
        return gs


class ASPP(nn.Module):
    def __init__(self, cin, c=256, rates=(2, 4, 8)):
        super().__init__()
        self.br = nn.ModuleList([ConvBNReLU(cin, c, 1)] + [nn.Sequential(
            nn.Conv2d(cin, c, 3, padding=r, dilation=r, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True)) for r in rates])
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(cin, c, 1), nn.ReLU(inplace=True))
        self.proj = ConvBNReLU(c * (len(rates) + 2), c, 1)

    def forward(self, x):
        fs = [b(x) for b in self.br] + [F.interpolate(self.pool(x), size=x.shape[2:], mode="nearest")]
        return self.proj(torch.cat(fs, 1))


class GlobalAttn(nn.Module):
    def __init__(self, cin, c=256, layers=2, heads=8, grid=16):
        super().__init__()
        self.inp = nn.Conv2d(cin, c, 1)
        self.pos = nn.Parameter(torch.zeros(1, c, grid, grid)); nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(c, heads, 4 * c, dropout=0.1, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)

    def forward(self, x):
        f = self.inp(x); h, w = f.shape[2:]
        pos = self.pos if (h, w) == tuple(self.pos.shape[2:]) else F.interpolate(self.pos, size=(h, w), mode="bilinear", align_corners=False)
        f = f + pos; b, c = f.shape[:2]
        z = self.tf(f.flatten(2).transpose(1, 2))
        return z.transpose(1, 2).reshape(b, c, h, w)


class Context(nn.Module):
    def __init__(self, cin=512, aspp=True, attn=True):
        super().__init__()
        self.aspp = ASPP(cin) if aspp else None
        self.attn = GlobalAttn(cin) if attn else None
        self.fuse = nn.Conv2d(256 * (int(aspp) + int(attn)), cin, 1); small_init(self.fuse, 1e-2)

    def forward(self, x):
        fs = [m(x) for m in (self.aspp, self.attn) if m is not None]
        return x + self.fuse(torch.cat(fs, 1))


class AttnGate(nn.Module):
    def __init__(self, cg, cx, ci):
        super().__init__()
        self.wg = nn.Sequential(nn.Conv2d(cg, ci, 1, bias=False), nn.BatchNorm2d(ci))
        self.wx = nn.Sequential(nn.Conv2d(cx, ci, 1, bias=False), nn.BatchNorm2d(ci))
        self.psi = nn.Sequential(nn.Conv2d(ci, 1, 1), nn.Sigmoid())

    def forward(self, g, x):
        g = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=False)
        return x * self.psi(F.relu(self.wg(g) + self.wx(x)))


class CBAM(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.mlp = nn.Sequential(nn.Conv2d(c, max(c // r, 4), 1), nn.ReLU(inplace=True), nn.Conv2d(max(c // r, 4), c, 1))
        self.sp = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x):
        a = x * torch.sigmoid(self.mlp(F.adaptive_avg_pool2d(x, 1)) + self.mlp(F.adaptive_max_pool2d(x, 1)))
        a = a * torch.sigmoid(self.sp(torch.cat([a.mean(1, keepdim=True), a.amax(1, keepdim=True)], 1)))
        return x + a


class GatedDecoderBlock(nn.Module):
    def __init__(self, cin, cskip, cout, gated=True):
        super().__init__()
        self.ag = AttnGate(cin, cskip, max(cskip // 2, 16)) if (cskip and gated) else None
        self.c1 = ConvBNReLU(cin + cskip, cout); self.c2 = ConvBNReLU(cout, cout)
        self.cbam = CBAM(cout) if gated else nn.Identity()

    def forward(self, x, skip=None, tgate=None):
        if skip is not None:
            if self.ag is not None:
                skip = self.ag(x, skip)
            if tgate is not None:
                skip = skip * tgate
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], 1)
        return self.cbam(self.c2(self.c1(x)))


class TGFloodNet(nn.Module):
    def __init__(self, abl=ABL, p_drop_terrain=0.3):
        super().__init__()
        self.abl, self.p_drop = set(abl), p_drop_terrain
        self.stem = None if "a1" in self.abl else PolStem()
        self.enc = Encoder()
        self.terrain = None if "a2" in self.abl else TerrainBranch()
        a, b = "a3a" not in self.abl, "a3b" not in self.abl
        self.ctx = Context(512, a, b) if (a or b) else None
        g = "a4" not in self.abl
        self.d4 = GatedDecoderBlock(512, 256, 256, g)
        self.d3 = GatedDecoderBlock(256, 128, 128, g)
        self.d2 = GatedDecoderBlock(128, 64, 64, g)
        self.d1 = GatedDecoderBlock(64, 64, 32, g)
        self.d0 = GatedDecoderBlock(32, 0, 16, g)
        self.head = nn.Conv2d(16, 1, 1)
        self.edge = None if "a5b" in self.abl else nn.Conv2d(16, 1, 1)
        self.ds = "ds" not in self.abl
        if self.ds:
            self.aux8, self.aux4 = nn.Conv2d(128, 1, 1), nn.Conv2d(64, 1, 1)

    def forward(self, x, terrain=None):
        if x.shape[1] == 5:
            x, terrain = x[:, :3], x[:, 3:]
        if self.stem is not None:
            x = self.stem(x)
        s0, s1, s2, s3, s4 = self.enc(x)
        if self.ctx is not None:
            s4 = self.ctx(s4)
        tg = [None] * 4
        if self.terrain is not None and terrain is not None and not (self.training and random.random() < self.p_drop):
            tg = self.terrain(terrain)
        y = self.d4(s4, s3, tg[3])
        y8 = self.d3(y, s2, tg[2])
        y4 = self.d2(y8, s1, tg[1])
        y = self.d1(y4, s0, tg[0])
        y = self.d0(y)
        out = {"water": self.head(y)}
        if self.edge is not None:
            out["edge"] = self.edge(y)
        if self.ds and self.training:
            out["aux8"], out["aux4"] = self.aux8(y8), self.aux4(y4)
        return out


def build_model():
    return TGFloodNet()


def dice_loss(logit, y, valid):
    p = torch.sigmoid(logit) * valid
    t = y * valid
    inter = (p * t).sum((1, 2, 3)); den = p.sum((1, 2, 3)) + t.sum((1, 2, 3))
    return (1 - (2 * inter + 1) / (den + 1)).mean()


def seg_loss(logit, t, valid):
    if logit.shape[-1] != t.shape[-1]:
        logit = F.interpolate(logit, size=t.shape[2:], mode="bilinear", align_corners=False)
    bce = (F.binary_cross_entropy_with_logits(logit, t, reduction="none") * valid).sum() / valid.sum().clamp(min=1)
    return bce + dice_loss(logit, t, valid)


def lovasz_grad(gt_sorted):
    gts = gt_sorted.sum()
    inter = gts - gt_sorted.cumsum(0); union = gts + (1 - gt_sorted).cumsum(0)
    j = 1.0 - inter / union
    if len(gt_sorted) > 1:
        j[1:] = j[1:] - j[:-1]
    return j


def lovasz_hinge(logit, t, valid):
    losses = []
    for lg, tt, vv in zip(logit.flatten(1), t.flatten(1), valid.flatten(1)):
        m = vv > 0
        lg, tt = lg[m], tt[m]
        if tt.numel() == 0:
            continue
        err = 1 - lg * (2 * tt - 1)
        es, perm = torch.sort(err, descending=True)
        losses.append(torch.dot(F.relu(es), lovasz_grad(tt[perm])))
    return torch.stack(losses).mean() if losses else logit.sum() * 0


def edge_target(t, valid):
    edge = (F.max_pool2d(t, 3, 1, 1) + F.max_pool2d(-t, 3, 1, 1)).clamp(0, 1)
    return edge, -F.max_pool2d(-valid, 3, 1, 1)


def loss_fn(out, y):
    valid = (y >= 0).float(); t = y.clamp(min=0).float()
    w_lov = 0.5 * min(max((TRAIN_FRAC[0] - 0.3) / 0.3, 0.0), 1.0)
    loss = seg_loss(out["water"], t, valid)
    if w_lov > 0:
        loss = loss + w_lov * lovasz_hinge(out["water"], t, valid)
    if "edge" in out:
        e, ve = edge_target(t, valid)
        pw = torch.tensor(5.0, device=t.device)
        loss = loss + 0.5 * (F.binary_cross_entropy_with_logits(out["edge"], e, pos_weight=pw, reduction="none") * ve).sum() / ve.sum().clamp(min=1)
    if "aux8" in out:
        loss = loss + 0.3 * seg_loss(out["aux8"], t, valid) + 0.2 * seg_loss(out["aux4"], t, valid)
    return loss


def augment(x, y):
    k = random.randint(0, 3)
    x, y = np.rot90(x, k, (1, 2)), np.rot90(y, k, (0, 1))
    if random.random() < 0.5:
        x, y = x[:, :, ::-1], y[:, ::-1]
    if random.random() < 0.5:
        x, y = x[:, ::-1], y[::-1]
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


@torch.no_grad()
def predict(model, x, tta=False, bs=8):
    model.eval(); out = []
    for i in range(0, len(x), bs):
        xb = torch.from_numpy(x[i:i + bs]).to(DEV)
        with torch.autocast(DEV.type, enabled=DEV.type == "cuda"):
            if not tta:
                p = torch.sigmoid(model(xb)["water"].float())
            else:
                p = 0
                for k in range(4):
                    for fl in (False, True):
                        xi = torch.rot90(xb, k, (2, 3)); xi = xi.flip(3) if fl else xi
                        pi = torch.sigmoid(model(xi)["water"].float())
                        pi = pi.flip(3) if fl else pi
                        p = p + torch.rot90(pi, -k, (2, 3))
                p = p / 8
        out.append(p[:, 0].cpu().numpy())
    return np.concatenate(out)


def counts(prob, y, thr):
    v = y >= 0; pr = prob >= thr; t = y == 1
    tp = int((pr & t & v).sum()); fp = int((pr & ~t & v).sum()); fn = int((~pr & t & v).sum()); tn = int((~pr & ~t & v).sum())
    return tp, fp, fn, tn


def metrics(tp, fp, fn, tn):
    n = tp + fp + fn + tn
    iou = tp / max(tp + fp + fn, 1); prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1); oa = (tp + tn) / max(n, 1)
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / max(n * n, 1)
    kappa = (oa - pe) / max(1 - pe, 1e-9)
    return dict(iou=iou, f1=f1, precision=prec, recall=rec, oa=oa, kappa=kappa)


def evaluate(prob, y, thr):
    m = metrics(*counts(prob, y, thr))
    per = []
    for i in range(len(y)):
        tp, fp, fn, tn = counts(prob[i], y[i], thr)
        per.append(tp / (tp + fp + fn) if tp + fp + fn > 0 else float("nan"))
    m["mean_chip_iou"] = float(np.nanmean(per)) if np.isfinite(per).any() else float("nan")
    return m, per


def run_seed(seed, d, prior_dirs):
    rd = OUT / "runs" / f"seed{seed}"
    if (rd / "preds.npz").exists():
        log(f"seed {seed}: already done"); return
    for p in prior_dirs:
        src = p / "runs" / f"seed{seed}"
        if (src / "preds.npz").exists():
            shutil.copytree(src, rd, dirs_exist_ok=True); log(f"seed {seed}: copied from {src}"); return
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    rd.mkdir(parents=True, exist_ok=True)
    model = build_model().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    xtr, ytr, _ = d["train"]
    steps = math.ceil(len(xtr) / BATCH)
    total = EPOCHS * steps
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / (WARMUP * steps) if s < WARMUP * steps
                                              else 0.5 * (1 + math.cos(math.pi * (s - WARMUP * steps) / max(total - WARMUP * steps, 1))))
    scaler = torch.amp.GradScaler(enabled=DEV.type == "cuda")
    best, hist, t0 = -1, [], time.time()
    for ep in range(EPOCHS):
        model.train(); order = np.random.permutation(len(xtr)); tl = 0
        for i in range(steps):
            idx = order[i * BATCH:(i + 1) * BATCH]
            bx, by = zip(*[augment(xtr[j], ytr[j]) for j in idx])
            xb = torch.from_numpy(np.stack(bx)).to(DEV); yb = torch.from_numpy(np.stack(by)).to(DEV)[:, None]
            TRAIN_FRAC[0] = (ep * steps + i) / total
            with torch.autocast(DEV.type, enabled=DEV.type == "cuda"):
                out = model(xb)
            loss = loss_fn({k: v.float() for k, v in out.items()}, yb)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tl += loss.item()
        pv = predict(model, d["valid"][0])
        viou = metrics(*counts(pv, d["valid"][1], 0.5))["iou"]
        hist.append(dict(epoch=ep + 1, train_loss=tl / steps, val_iou=viou, lr=opt.param_groups[0]["lr"], sec=time.time() - t0))
        if viou > best:
            best = viou; torch.save(model.state_dict(), rd / "weights.pt")
        if (ep + 1) % 10 == 0 or ep == 0 or ep == EPOCHS - 1:
            log(f"seed {seed} ep {ep+1}/{EPOCHS} loss {tl/steps:.4f} val_iou {viou:.4f} best {best:.4f} {time.time()-t0:.0f}s")
    train_sec = time.time() - t0
    with open(rd / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hist[0])); w.writeheader(); w.writerows(hist)
    model.load_state_dict(torch.load(rd / "weights.pt", map_location=DEV))
    res, conf, per_rows, probs = {}, {}, [], {}
    for tta in (False, True):
        tag = "tta" if tta else "plain"
        p = {s: predict(model, d[s][0], tta) for s in ("valid", "test", "bolivia")}
        grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
        thr_v = float(grid[np.argmax([metrics(*counts(p["valid"], d["valid"][1], t))["iou"] for t in grid])])
        for thr_name, thr in (("t050", 0.5), ("tval", thr_v)):
            for s in ("valid", "test", "bolivia"):
                m, per = evaluate(p[s], d[s][1], thr)
                res[f"{tag}/{thr_name}/{s}"] = dict(m, threshold=thr)
                conf[f"{tag}/{thr_name}/{s}"] = dict(zip(("tp", "fp", "fn", "tn"), counts(p[s], d[s][1], thr)))
                if thr_name == "t050":
                    per_rows += [dict(mode=tag, split=s, chip=c, iou=v) for c, v in zip(d[s][2], per)]
        probs[tag] = p
    t1 = time.time(); predict(model, d["test"][0]); inf_ms = 1000 * (time.time() - t1) / len(d["test"][0])
    json.dump(res, open(rd / "metrics.json", "w"), indent=1)
    json.dump(conf, open(rd / "confusion.json", "w"), indent=1)
    with open(rd / "per_chip.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["mode", "split", "chip", "iou"]); w.writeheader(); w.writerows(per_rows)
    json.dump(dict(name=NAME, ablate=ABL, seed=seed, epochs=EPOCHS, batch=BATCH, lr=LR, wd=WD, warmup=WARMUP, clip=CLIP,
                   best_val_iou_t050=best, train_sec=train_sec, infer_ms_per_chip=inf_ms, smoke=SMOKE,
                   params=sum(p.numel() for p in model.parameters())), open(rd / "config.json", "w"), indent=1)
    np.savez_compressed(rd / "preds.npz", **{f"{tag}_{s}": (probs[tag][s] * 255).round().astype(np.uint8)
                                            for tag in probs for s in ("test", "bolivia")},
                        test_ids=np.array(d["test"][2]), bolivia_ids=np.array(d["bolivia"][2]))
    r = res["plain/t050/test"]
    log(f"seed {seed} done: test IoU {r['iou']:.4f} F1 {r['f1']:.4f} | bolivia IoU {res['plain/t050/bolivia']['iou']:.4f} | {train_sec/60:.1f} min")


def efficiency():
    m = build_model().to(DEV).eval()
    params = sum(p.numel() for p in m.parameters())
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as fc, torch.no_grad():
            m(torch.zeros(1, IN_CH, 512, 512, device=DEV))
        gflops = fc.get_total_flops() / 2e9
    except Exception as e:
        gflops = float("nan")
    (OUT / "efficiency.txt").write_text(f"params {params}\nGMACs_512 {gflops:.2f}\n")
    log(f"params {params/1e6:.2f} M, {gflops:.1f} GMACs at 512 px")


def summarise():
    rows = []
    for rd in sorted((OUT / "runs").glob("seed*")):
        if not (rd / "metrics.json").exists():
            continue
        for k, m in json.load(open(rd / "metrics.json")).items():
            rows.append(dict(seed=rd.name, key=k, **{x: m[x] for x in ("iou", "f1", "precision", "recall", "oa", "kappa", "mean_chip_iou", "threshold")}))
    if not rows:
        return
    with open(OUT / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    lines = [f"# {NAME} summary (mean ± sd over {len({r['seed'] for r in rows})} seeds)", "",
             "| mode/threshold/split | IoU | F1 | precision | recall | kappa |", "|---|---|---|---|---|---|"]
    for k in dict.fromkeys(r["key"] for r in rows):
        sel = [r for r in rows if r["key"] == k]
        f = lambda x: f"{np.mean([r[x] for r in sel]):.4f} ± {np.std([r[x] for r in sel]):.4f}"
        lines.append(f"| {k} | {f('iou')} | {f('f1')} | {f('precision')} | {f('recall')} | {f('kappa')} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    log("\n" + "\n".join(lines))


if __name__ == "__main__":
    log(f"{NAME} start  smoke={SMOKE}  device={DEV}  out={OUT}")
    with open(OUT / "versions.txt", "w") as f:
        f.write(f"python {platform.python_version()}\ntorch {torch.__version__}\ntorchvision {torchvision.__version__}\n"
                f"numpy {np.__version__}\nrasterio {rasterio.__version__}\n"
                f"gpu {torch.cuda.get_device_name(0) if DEV.type == 'cuda' else 'cpu'}\n")
    d, mean, std = load_all()
    json.dump(dict(mean=mean.tolist(), std=std.tolist(), channels=CHANNELS, clip=CLIP), open(OUT / "norm.json", "w"))
    efficiency()
    prior = [Path(p).parent.parent for p in glob.glob(f"/kaggle/input/**/{NAME}/runs/seed*/preds.npz", recursive=True)] if ON_KAGGLE else []
    for s in SEEDS:
        el = (time.time() - T_START) / 3600
        done_secs = [json.load(open(c))["train_sec"] for c in (OUT / "runs").glob("seed*/config.json")]
        need = (max(done_secs) / 3600 * 1.15) if done_secs else 0
        if el + need > TIME_BUDGET_H:
            log(f"seed {s}: skipped, time budget ({el:.1f} h used, {need:.1f} h needed)"); continue
        run_seed(s, d, prior)
    summarise()
    log("ALL DONE")
