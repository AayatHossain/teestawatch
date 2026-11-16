"""Fine-tune TG-FloodNet on the Teesta with optical labels and Sen1Floods11 replay."""
import os, sys, json, time, math, random, glob, csv, platform, copy, importlib.util
from pathlib import Path
import numpy as np

NAME = "ft_teesta"
SMOKE = os.environ.get("TW_SMOKE") == "1"
OUT = Path(os.environ.get("TW_OUT", f"/kaggle/working/{NAME}"))
OUT.mkdir(parents=True, exist_ok=True)
FT = Path(os.environ.get("TW_FT") or Path(sorted(glob.glob("/kaggle/input/**/static.npz", recursive=True))[0]).parent)
KINDS = ["tg"]
SEEDS = [0] if SMOKE else [0, 1, 2]
STEPS = int(os.environ.get("TW_STEPS", 6 if SMOKE else 1500))
EVAL_EVERY = 3 if SMOKE else 100
BT, BR = (1, 1) if SMOKE else (4, 4)
LR, WD, WARM = 1e-4, 1e-4, (2 if SMOKE else 50)
WIN, CLIP = 512, (-40.0, 10.0)
TIME_BUDGET_H = 11.0
T0 = time.time()
LOGF = open(OUT / "log.txt", "a", encoding="utf-8")


def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()


def load_mod(name, file):
    os.environ["TW_OUT"] = str(OUT / "_mod" / name)
    spec = importlib.util.spec_from_file_location(name, file); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


tg = load_mod("tg", FT / "tgfloodnet.py")
tg.TRAIN_FRAC[0] = 1.0
MOD = {"tg": tg}
import torch
DEV = tg.DEV
norm = json.load(open(FT / "norm.json"))
M = np.array(norm["mean"], np.float32); SD = np.array(norm["std"], np.float32)
info = json.load(open(FT / "info.json"))
TRAIN, TEST = info["train"], info["test"]


def pad(a, fill):
    H, W = a.shape; ph, pw = (-H) % WIN, (-W) % WIN
    return np.pad(a, ((0, ph), (0, pw)), constant_values=fill) if ph or pw else a


st = np.load(FT / "static.npz")
HAND = pad(st["hand"].astype(np.float32), 0); DIST = pad(st["dist"].astype(np.float32), 0)
INSIDE = pad(st["inside"], False); BLOCK = pad(st["block"], 255); NOTILE = pad(st["notile"], True)
HP, WP = INSIDE.shape
D = {}
for d in TRAIN + TEST:
    z = np.load(FT / f"date_{d}.npz")
    D[d] = {k: pad(z[k], -32768 if k in ("vv", "vh") else 255) for k in z.files}
log(f"Teesta grid {HP} x {WP}; dates {list(D)}")


def window(d, r, c):
    vv = D[d]["vv"][r:r + WIN, c:c + WIN]; vh = D[d]["vh"][r:r + WIN, c:c + WIN]
    ok = (vv != -32768) & (vh != -32768) & INSIDE[r:r + WIN, c:c + WIN]
    a = np.clip(vv / 100.0, *CLIP); b = np.clip(vh / 100.0, *CLIP)
    x = np.stack([(a - M[0]) / SD[0], (b - M[1]) / SD[1], (a - b - M[2]) / SD[2],
                  (HAND[r:r + WIN, c:c + WIN] - M[3]) / SD[3], (DIST[r:r + WIN, c:c + WIN] - M[4]) / SD[4]]).astype(np.float32)
    x[:, ~ok] = 0
    return x, ok


def target(arr, r, c, sel):
    a = arr[r:r + WIN, c:c + WIN]; s = sel[r:r + WIN, c:c + WIN]
    y = np.full(a.shape, -1, np.int8); y[(a == 1) & s] = 1; y[(a == 0) & s] = 0
    return y


def box_count(mask, step, min_share):
    ii = np.zeros((HP + 1, WP + 1), np.int32); ii[1:, 1:] = mask.cumsum(0, dtype=np.int32).cumsum(1, dtype=np.int32)
    out = []
    for r in range(0, HP - WIN + 1, step):
        for c in range(0, WP - WIN + 1, step):
            n = ii[r + WIN, c + WIN] - ii[r, c + WIN] - ii[r + WIN, c] + ii[r, c]
            if n >= min_share * WIN * WIN:
                out.append((r, c))
    return out


SEL_TR = NOTILE & (BLOCK == 0); SEL_VA = NOTILE & (BLOCK == 1); SEL_ALL = INSIDE
train_w, val_w = [], []
for d in TRAIN:
    lab = D[d]["lab"] != 255
    train_w += [(d, r, c) for r, c in box_count(lab & SEL_TR, 128, 0.15)]
    val_w += [(d, r, c) for r, c in box_count(lab & SEL_VA, WIN, 0.05)]
TESTSETS = {"hls_2025-08-16": [("2025-08-16", r, c) for r, c in box_count((D["2025-08-16"]["lab"] != 255), WIN, 0.01)]}
for d in TEST:
    TESTSETS[f"dswxs1_{d}"] = [(d, r, c) for r, c in box_count(INSIDE & (D[d]["s1ref"] != 255), WIN, 1e-4)]
REF = {"hls_2025-08-16": "lab"} | {f"dswxs1_{d}": "s1ref" for d in TEST}
if SMOKE:
    random.seed(0); train_w = random.sample(train_w, 20); val_w = val_w[:4]; TESTSETS = {k: v[:4] for k, v in TESTSETS.items()}
log(f"windows: train {len(train_w)} (stride 128) | val {len(val_w)} | " + " | ".join(f"{k} {len(v)}" for k, v in TESTSETS.items()))


@torch.no_grad()
def predict_windows(model, wins, bs=8):
    model.eval(); out = []
    for i in range(0, len(wins), bs):
        xb = torch.from_numpy(np.stack([window(*w)[0] for w in wins[i:i + bs]])).to(DEV)
        with torch.autocast(DEV.type, enabled=DEV.type == "cuda"):
            p = torch.sigmoid(model(xb)["water"].float())
        out.append((p[:, 0] * 255).round().byte().cpu().numpy())
    return np.concatenate(out)


def ys_for(name, wins):
    if name == "val":
        return np.stack([target(D[d]["lab"], r, c, SEL_VA) for d, r, c in wins])
    return np.stack([target(D[d][REF[name]], r, c, SEL_ALL) for d, r, c in wins])


Y_VAL = ys_for("val", val_w)
Y_TEST = {k: ys_for(k, v) for k, v in TESTSETS.items()}
GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)


def score(prob8, y, thr):
    tp, fp, fn, tn = tg.counts(prob8.astype(np.float32) / 255, y, thr)
    m = tg.metrics(tp, fp, fn, tn)
    m.update(pred_water_km2=(tp + fp) * 1e-4, ref_water_km2=(tp + fn) * 1e-4, compared_km2=(tp + fp + fn + tn) * 1e-4)
    m["area_ratio"] = m["pred_water_km2"] / max(m["ref_water_km2"], 1e-9)
    return {k: round(float(v), 4) for k, v in m.items()}


def val_iou(model):
    p = predict_windows(model, val_w)
    return score(p, Y_VAL, 0.5)["iou"], p


S, _, _ = tg.load_all()
XR, YR = S["train"][0], S["train"][1]


def eval_all(model, thr_val=None):
    vi, pv = val_iou(model)
    tv = thr_val if thr_val is not None else float(GRID[np.argmax([score(pv, Y_VAL, t)["iou"] for t in GRID])])
    res, probs = {"val/t050": score(pv, Y_VAL, 0.5), "val_threshold": tv}, {}
    for k, wins in TESTSETS.items():
        p = predict_windows(model, wins); probs[k] = p
        res[f"{k}/t050"] = score(p, Y_TEST[k], 0.5); res[f"{k}/tval"] = score(p, Y_TEST[k], tv)
    for s in ("test", "bolivia"):
        p = tg.predict(model, S[s][0])
        res[f"s1f11_{s}/t050"] = {k: round(float(v), 4) for k, v in tg.metrics(*tg.counts(p, S[s][1], 0.5)).items()}
    return res, probs


def run(kind, seed):
    rd = OUT / "runs" / f"{kind}_seed{seed}"
    if (rd / "metrics.json").exists():
        log(f"{kind} seed {seed}: done already"); return json.load(open(rd / "metrics.json")), None
    rd.mkdir(parents=True, exist_ok=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    mod = MOD[kind]; model = mod.build_model().to(DEV)
    model.load_state_dict(torch.load(FT / f"{kind}_seed{seed}.pt", map_location=DEV))
    before, pb = eval_all(model)
    log(f"{kind} seed {seed} BEFORE: val {before['val/t050']['iou']:.4f} | " + " | ".join(
        f"{k} IoU {before[k + '/t050']['iou']:.3f} x{before[k + '/t050']['area_ratio']:.2f}" for k in TESTSETS))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / WARM if s < WARM else 0.5 * (1 + math.cos(math.pi * (s - WARM) / max(STEPS - WARM, 1))))
    scaler = torch.amp.GradScaler(enabled=DEV.type == "cuda")
    best, best_state, hist, t0, tl = before["val/t050"]["iou"], copy.deepcopy(model.state_dict()), [], time.time(), 0.0
    best_step = 0
    for step in range(1, STEPS + 1):
        model.train()
        bx, by = [], []
        for d, r, c in random.sample(train_w, BT):
            x, _ = window(d, r, c); x, y = mod.augment(x, target(D[d]["lab"], r, c, SEL_TR)); bx.append(x); by.append(y)
        for j in np.random.randint(0, len(XR), BR):
            x, y = mod.augment(XR[j], YR[j]); bx.append(x); by.append(y)
        xb = torch.from_numpy(np.stack(bx)).to(DEV); yb = torch.from_numpy(np.stack(by)).to(DEV)[:, None]
        with torch.autocast(DEV.type, enabled=DEV.type == "cuda"):
            out = model(xb)
        loss = mod.loss_fn({k: v.float() for k, v in out.items()}, yb)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        tl += loss.item()
        if step % EVAL_EVERY == 0 or step == STEPS:
            vi, _ = val_iou(model)
            hist.append(dict(step=step, train_loss=tl / EVAL_EVERY, val_iou=vi, lr=opt.param_groups[0]["lr"], sec=round(time.time() - t0)))
            tl = 0.0
            if vi > best:
                best, best_step, best_state = vi, step, copy.deepcopy(model.state_dict())
            log(f"{kind} seed {seed} step {step}/{STEPS} val IoU {vi:.4f} best {best:.4f}@{best_step} {time.time()-t0:.0f}s")
    train_sec = time.time() - t0
    model.load_state_dict(best_state); torch.save(best_state, rd / "weights.pt")
    after, pa = eval_all(model)
    with open(rd / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hist[0])); w.writeheader(); w.writerows(hist)
    res = dict(kind=kind, seed=seed, best_step=best_step, train_sec=round(train_sec), before=before, after=after)
    json.dump(res, open(rd / "metrics.json", "w"), indent=1)
    np.savez_compressed(rd / "test_probs.npz", **{f"before_{k}": pb[k] for k in pb}, **{f"after_{k}": pa[k] for k in pa})
    log(f"{kind} seed {seed} AFTER (best step {best_step}): val {after['val/t050']['iou']:.4f} | " + " | ".join(
        f"{k} IoU {after[k + '/t050']['iou']:.3f} x{after[k + '/t050']['area_ratio']:.2f}" for k in TESTSETS)
        + f" | s1f11 test {after['s1f11_test/t050']['iou']:.3f} bolivia {after['s1f11_bolivia/t050']['iou']:.3f} | {train_sec/60:.1f} min")
    return res, (pb, pa)


def summarise(results, ens):
    keys = [f"{k}/t050" for k in TESTSETS] + [f"{k}/tval" for k in TESTSETS] + ["val/t050", "s1f11_test/t050", "s1f11_bolivia/t050"]
    L = [f"# {NAME} summary (mean ± sd over seeds; IoU, F1 and water-area ratio vs reference)", "",
         "| model | stage | set | IoU | F1 | precision | recall | area ratio |", "|---|---|---|---|---|---|---|---|"]
    for kind in KINDS:
        rs = [r for r in results if r["kind"] == kind]
        for stage in ("before", "after"):
            for k in keys:
                v = [r[stage][k] for r in rs if k in r[stage]]
                if not v:
                    continue
                f = lambda x: f"{np.mean([a[x] for a in v]):.3f} ± {np.std([a[x] for a in v]):.3f}" if x in v[0] else "-"
                L.append(f"| {kind} | {stage} | {k} | {f('iou')} | {f('f1')} | {f('precision')} | {f('recall')} | {f('area_ratio')} |")
    L += ["", "## 3-seed ensembles (mean probability, threshold 0.5)", "", "| model | stage | set | IoU | F1 | area ratio |", "|---|---|---|---|---|---|"]
    for (kind, stage, k), m in ens.items():
        L.append(f"| {kind} | {stage} | {k} | {m['iou']:.3f} | {m['f1']:.3f} | {m['area_ratio']:.2f} |")
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    log("\n" + "\n".join(L))


if __name__ == "__main__":
    log(f"{NAME} start smoke={SMOKE} device={DEV} FT={FT}")
    with open(OUT / "versions.txt", "w") as f:
        f.write(f"python {platform.python_version()}\ntorch {torch.__version__}\nnumpy {np.__version__}\n"
                f"gpu {torch.cuda.get_device_name(0) if DEV.type == 'cuda' else 'cpu'}\n")
    json.dump(dict(kinds=KINDS, seeds=SEEDS, steps=STEPS, batch_teesta=BT, batch_replay=BR, lr=LR, wd=WD, warmup=WARM,
                   eval_every=EVAL_EVERY, train_windows=len(train_w), val_windows=len(val_w),
                   test_windows={k: len(v) for k, v in TESTSETS.items()}, info=info["dates"]), open(OUT / "config.json", "w"), indent=1)
    results, probs = [], {}
    for seed in SEEDS:
        for kind in KINDS:
            el = (time.time() - T0) / 3600
            done = [r["train_sec"] for r in results]
            if done and el + 1.3 * max(done) / 3600 > TIME_BUDGET_H:
                log(f"{kind} seed {seed}: skipped (time budget)"); continue
            r, p = run(kind, seed); results.append(r)
            if p:
                probs[(kind, seed)] = p
    ens = {}
    for kind in KINDS:
        ps = [probs[(k, s)] for (k, s) in probs if k == kind]
        if not ps:
            continue
        for i, stage in enumerate(("before", "after")):
            for k in TESTSETS:
                m = np.mean([p[i][k].astype(np.float32) for p in ps], 0)
                ens[(kind, stage, k)] = score(m.round().astype(np.uint8), Y_TEST[k], 0.5)
    json.dump({f"{a}/{b}/{c}": v for (a, b, c), v in ens.items()}, open(OUT / "ensemble.json", "w"), indent=1)
    summarise(results, ens)
    log("ALL DONE")
