"""Qualitative figure on the test tile: 4 best + 2 worst locations by CATFormer+refiner PSNR."""
import argparse, json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from baselines import UnCRtainTSWrapper
from data import CRTS
from model import CATFormer, Refiner

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/asiaWest")
ap.add_argument("--cat", default="runs/full_catformer_v2_s0")
ap.add_argument("--unc", default="runs/full_uncrtaints_s0")
ap.add_argument("--out", default="figures/test_predictions.png")
ap.add_argument("--steps", type=int, default=4)
args = ap.parse_args()
torch.set_num_threads(16)

test_tiles = {tuple(t) for t in json.load(open(f"{args.cat}/split.json"))["test"]}
ds = CRTS(args.root, T=3)
ds.keys = [k for k in sorted(ds.keys) if k[:2] in test_tiles]
print("test locations:", len(ds), flush=True)

cat = CATFormer(); cat.load_state_dict(torch.load(f"{args.cat}/best.pt", map_location="cpu")); cat.eval()
ref = Refiner(); ref.load_state_dict(torch.load(f"{args.cat}/refiner_best.pt", map_location="cpu")); ref.eval()
unc = UnCRtainTSWrapper(); unc.load_state_dict(torch.load(f"{args.unc}/best.pt", map_location="cpu")); unc.eval()


def psnr(a, b):
    return float(10 * np.log10(1.0 / max(((a - b) ** 2).mean(), 1e-10)))


rows = []
with torch.no_grad():
    for i in range(len(ds)):
        b = {k: v[None] for k, v in ds[i].items()}
        mean, logvar, _ = cat(b["s2"], b["s1"], b["cloud"], b["tpos"])
        cond = Refiner.make_cond(mean, logvar, b["cloud"].min(1).values, b["s1"].mean(1))
        pred_ref = ref.sample(mean, cond, args.steps)[0].numpy()
        pred_unc = unc(b["s2"], b["s1"], b["cloud"], b["tpos"])[0][0].numpy()
        lc = int(b["cloud"][0].mean((1, 2, 3)).argmin())
        y = b["y"][0].numpy()
        rows.append(dict(i=i, key=ds.keys[i], s2=b["s2"][0].numpy(), s1=b["s1"][0].numpy(),
                         cloud=b["cloud"][0, :, 0].numpy(), tpos=b["tpos"][0].numpy(), y=y,
                         base=b["s2"][0, lc].numpy(), unc=pred_unc, ref=pred_ref,
                         std=np.exp(0.5 * logvar[0].numpy()).mean(0),
                         p_base=psnr(b["s2"][0, lc].numpy(), y), p_unc=psnr(pred_unc, y), p_ref=psnr(pred_ref, y)))
        if i % 50 == 0:
            print(i, flush=True)

order = sorted(rows, key=lambda r: -r["p_ref"])
pick = order[:4] + order[-2:]
print("picked:", [(r["key"], round(r["p_ref"], 2)) for r in pick], flush=True)


def rgb(x):
    return np.clip(x[[3, 2, 1]].transpose(1, 2, 0) * 3.5, 0, 1)


cols = ["Input t1", "Input t2", "Input t3", "SAR VV", "Least-cloudy", "UnCRtainTS",
        "CATFormer + refiner", "Target", "Uncertainty (σ)"]
fig, ax = plt.subplots(len(pick), len(cols), figsize=(2.1 * len(cols), 2.3 * len(pick) + 0.6))
vmax = np.percentile(np.concatenate([r["std"].ravel() for r in pick]), 99)
for r_i, r in enumerate(pick):
    near = int(np.abs(r["tpos"]).argmin())  # SAR from the input date closest to the target
    panels = [rgb(r["s2"][0]), rgb(r["s2"][1]), rgb(r["s2"][2]), r["s1"][near, 0],
              rgb(r["base"]), rgb(r["unc"]), rgb(r["ref"]), rgb(r["y"]), r["std"]]
    for c, img in enumerate(panels):
        a = ax[r_i, c]
        if c == 3:
            a.imshow(img, cmap="gray", vmin=0, vmax=1)
        elif c == 8:
            im = a.imshow(img, cmap="magma", vmin=0, vmax=vmax)
        else:
            a.imshow(img)
        a.set_xticks([]); a.set_yticks([])
        if r_i == 0:
            a.set_title(cols[c], fontsize=9)
    for c in range(3):
        ax[r_i, c].set_xlabel(f"{r['tpos'][c]:+.1f} mo, cloud {r['cloud'][c].mean():.0%}", fontsize=7)
    ax[r_i, 3].set_xlabel(f"date {r['tpos'][near]:+.1f} mo", fontsize=7)
    ax[r_i, 4].set_xlabel(f"{r['p_base']:.2f} dB", fontsize=8)
    ax[r_i, 5].set_xlabel(f"{r['p_unc']:.2f} dB", fontsize=8)
    ax[r_i, 6].set_xlabel(f"{r['p_ref']:.2f} dB", fontsize=8, fontweight="bold")
    tag = "best" if r_i < 4 else "worst"
    ax[r_i, 0].set_ylabel(f"{tag} #{r_i + 1 if r_i < 4 else r_i - 3}\npatch {r['key'][2]}", fontsize=8)
fig.colorbar(im, ax=ax[:, 8], fraction=0.05, pad=0.02)
fig.suptitle(f"Test tile {pick[0]['key'][0]}/{pick[0]['key'][1]} (held out). Rows: 4 best and 2 worst of "
             f"{len(rows)} locations by CATFormer+refiner PSNR. RGB = B4/B3/B2 ×3.5", fontsize=10)
import os; os.makedirs(os.path.dirname(args.out), exist_ok=True)
plt.savefig(args.out, dpi=130, bbox_inches="tight")
json.dump([dict(key=list(r["key"]), psnr_base=r["p_base"], psnr_unc=r["p_unc"], psnr_ref=r["p_ref"]) for r in order],
          open(args.out.replace(".png", "_psnr.json"), "w"), indent=1)
print("FIG_DONE", args.out, "mean PSNR base/unc/ref:",
      round(np.mean([r["p_base"] for r in rows]), 2), round(np.mean([r["p_unc"] for r in rows]), 2),
      round(np.mean([r["p_ref"] for r in rows]), 2))
