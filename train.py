import argparse, json, os, random, time

import numpy as np
import torch
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader

from data import CRTS
from model import CATFormer, Refiner, loss_fn

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--out", default="runs/catformer")
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--bs", type=int, default=4)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--T", type=int, default=3)
ap.add_argument("--max_samples", type=int, default=0)
ap.add_argument("--stage2_epochs", type=int, default=0)
ap.add_argument("--steps", type=int, default=4)
ap.add_argument("--model", default="catformer", choices=["catformer", "uncrtaints"])
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"

ds = CRTS(args.root, T=args.T)
keys = sorted(ds.keys)
if args.max_samples:
    keys = keys[: args.max_samples]
# whole-tile split, fixed independently of --seed: test 20% / val 10% / train 70% of tiles
tiles = sorted({k[:2] for k in keys})
random.Random(1234).shuffle(tiles)
n_te, n_va = max(1, round(0.2 * len(tiles))), max(1, round(0.1 * len(tiles)))
te_t, va_t = set(tiles[:n_te]), set(tiles[n_te:n_te + n_va])
test_keys = [k for k in keys if k[:2] in te_t]
val_keys = [k for k in keys if k[:2] in va_t]
train_keys = [k for k in keys if k[:2] not in te_t | va_t]
json.dump(dict(test=sorted(map(list, te_t)), val=sorted(map(list, va_t)),
               train=sorted(map(list, set(tiles) - te_t - va_t))), open(f"{args.out}/split.json", "w"), indent=1)
print(f"tiles: {len(tiles)} (train {len(tiles)-n_te-n_va} / val {n_va} / test {n_te}); "
      f"samples: train {len(train_keys)} val {len(val_keys)} test {len(test_keys)}", flush=True)

def preload(ks):
    from concurrent.futures import ThreadPoolExecutor
    sub = CRTS.__new__(CRTS); sub.__dict__.update(ds.__dict__); sub.keys = ks
    with ThreadPoolExecutor(16) as ex:
        return list(ex.map(sub.__getitem__, range(len(ks))))

t0 = time.time()
train, val, test = preload(train_keys), preload(val_keys), preload(test_keys)
print(f"preloaded in {time.time()-t0:.0f}s", flush=True)


def augment(b):
    k = random.randint(0, 3); f = random.random() < 0.5
    for n in ("s2", "s1", "cloud", "y"):
        b[n] = torch.rot90(b[n], k, (-2, -1))
        if f:
            b[n] = b[n].flip(-1)
    return b


def metrics(pred, y):
    p, t = pred.cpu().numpy(), y.cpu().numpy()
    out = []
    for a, b in zip(p, t):
        mse = ((a - b) ** 2).mean()
        psnr = 10 * np.log10(1.0 / max(mse, 1e-10))
        ssim = structural_similarity(a[[3, 2, 1]], b[[3, 2, 1]], channel_axis=0, data_range=1.0)
        cos = (a * b).sum(0) / (np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0) + 1e-8)
        sam = np.degrees(np.arccos(np.clip(cos, -1, 1))).mean()
        out.append((np.abs(a - b).mean(), psnr, ssim, sam))
    return out


def summarize(rows):
    r = np.array(rows)
    return dict(MAE=float(r[:, 0].mean()), PSNR=float(r[:, 1].mean()),
                SSIM_rgb=float(r[:, 2].mean()), SAM=float(r[:, 3].mean()))


# baselines on val
def baselines(data):
    base = {"least_cloudy_input": [], "mean_of_inputs": []}
    for b in data:
        lc = b["cloud"].mean((1, 2, 3)).argmin()
        base["least_cloudy_input"] += metrics(b["s2"][lc][None], b["y"][None])
        base["mean_of_inputs"] += metrics(b["s2"].mean(0)[None], b["y"][None])
    return {k: summarize(v) for k, v in base.items()}


base = baselines(val)
base_test = baselines(test)
print("baselines_val", json.dumps(base), flush=True)
print("baselines_test", json.dumps(base_test), flush=True)

if args.model == "uncrtaints":
    from baselines import UnCRtainTSWrapper
    model = UnCRtainTSWrapper().to(dev)
else:
    model = CATFormer().to(dev)
print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
steps = args.epochs * max(1, len(train) // args.bs)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.05)
collate = lambda bs: {k: torch.stack([b[k] for b in bs]) for k in bs[0]}

best, hist = None, []
for ep in range(args.epochs):
    model.train(); random.shuffle(train)
    tl = []
    for i in range(0, len(train) - args.bs + 1, args.bs):
        b = collate([augment(dict(x)) for x in train[i:i + args.bs]])
        b = {k: v.to(dev) for k, v in b.items()}
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            mean, logvar, _ = model(b["s2"], b["s1"], b["cloud"], b["tpos"])
        occl = b["cloud"].min(1).values
        loss, l1, nll = loss_fn(mean.float(), logvar.float(), b["y"], occl)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); tl.append(l1.item())
    model.eval(); rows = []
    with torch.no_grad():
        for i in range(0, len(val), args.bs):
            b = collate(val[i:i + args.bs]); b = {k: v.to(dev) for k, v in b.items()}
            mean, _, _ = model(b["s2"], b["s1"], b["cloud"], b["tpos"])
            rows += metrics(mean.float(), b["y"])
    m = summarize(rows); m["epoch"] = ep; m["train_L1"] = float(np.mean(tl)); hist.append(m)
    print(json.dumps(m), flush=True)
    if best is None or m["PSNR"] > best["PSNR"]:
        best = m; torch.save(model.state_dict(), f"{args.out}/best.pt")

print("BEST", json.dumps(best), flush=True)


def evaluate(data, predict):
    rows = []
    with torch.no_grad():
        for i in range(0, len(data), args.bs):
            b = collate(data[i:i + args.bs]); b = {k: v.to(dev) for k, v in b.items()}
            rows += metrics(predict(b).float(), b["y"])
    return summarize(rows)


# test tiles are scored exactly once, with the checkpoint chosen on val
model.load_state_dict(torch.load(f"{args.out}/best.pt")); model.eval()
test1 = evaluate(test, lambda b: model(b["s2"], b["s1"], b["cloud"], b["tpos"])[0])
print("TEST_S1", json.dumps(test1), flush=True)

# ---- stage 2: flow-matching refiner on top of the frozen best stage-1 model ----
best2, hist2, test2 = None, [], None
if args.stage2_epochs:
    model.load_state_dict(torch.load(f"{args.out}/best.pt")); model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    def stage1(b):
        mean, logvar, _ = model(b["s2"], b["s1"], b["cloud"], b["tpos"])
        cond = Refiner.make_cond(mean.float(), logvar.float(), b["cloud"].min(1).values, b["s1"].mean(1))
        return mean.float(), cond

    ref = Refiner().to(dev)
    print(f"refiner params: {sum(p.numel() for p in ref.parameters())/1e6:.2f}M", flush=True)
    opt2 = torch.optim.AdamW(ref.parameters(), lr=args.lr, weight_decay=0.05)
    steps2 = args.stage2_epochs * max(1, len(train) // args.bs)
    sched2 = torch.optim.lr_scheduler.OneCycleLR(opt2, args.lr, total_steps=steps2, pct_start=0.05)
    for ep in range(args.stage2_epochs):
        ref.train(); random.shuffle(train); tl = []
        for i in range(0, len(train) - args.bs + 1, args.bs):
            b = collate([augment(dict(x)) for x in train[i:i + args.bs]])
            b = {k: v.to(dev) for k, v in b.items()}
            with torch.no_grad():
                m, cond = stage1(b)
            loss = ref.train_loss(m, b["y"], cond)
            opt2.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(ref.parameters(), 1.0)
            opt2.step(); sched2.step(); tl.append(loss.item())
        ref.eval(); rows = []
        with torch.no_grad():
            for i in range(0, len(val), args.bs):
                b = collate(val[i:i + args.bs]); b = {k: v.to(dev) for k, v in b.items()}
                m, cond = stage1(b)
                rows += metrics(ref.sample(m, cond, args.steps), b["y"])
        m2 = summarize(rows); m2["epoch"] = ep; m2["train_loss"] = float(np.mean(tl)); hist2.append(m2)
        print("S2", json.dumps(m2), flush=True)
        if best2 is None or m2["PSNR"] > best2["PSNR"]:
            best2 = m2; torch.save(ref.state_dict(), f"{args.out}/refiner_best.pt")
    print("BEST_S2", json.dumps(best2), flush=True)
    ref.load_state_dict(torch.load(f"{args.out}/refiner_best.pt")); ref.eval()
    test2 = evaluate(test, lambda b: ref.sample(*stage1(b), args.steps))
    print("TEST_S2", json.dumps(test2), flush=True)

json.dump(dict(baselines_val=base, baselines_test=base_test, best_val=best, test_stage1=test1,
               best_val_stage2=best2, test_stage2=test2, history=hist, history_stage2=hist2,
               n_train=len(train), n_val=len(val), n_test=len(test), args=vars(args)),
          open(f"{args.out}/results.json", "w"), indent=1)
