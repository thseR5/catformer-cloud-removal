"""Export RGB PNGs for every test location: the 3 cloudy inputs, the CATFormer+refiner prediction
and the ground truth, one folder per location, plus index.csv with per-location metrics."""
import argparse, csv, json, os

import numpy as np
import torch
from PIL import Image

from data import CRTS
from model import CATFormer, Refiner

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/asiaWest")
ap.add_argument("--cat", default="runs/full_catformer_v2_s0")
ap.add_argument("--out", default="figures/test_pngs")
ap.add_argument("--steps", type=int, default=4)
args = ap.parse_args()
torch.set_num_threads(16)

test_tiles = {tuple(t) for t in json.load(open(f"{args.cat}/split.json"))["test"]}
ds = CRTS(args.root, T=3)
ds.keys = [k for k in sorted(ds.keys) if k[:2] in test_tiles]
cat = CATFormer(); cat.load_state_dict(torch.load(f"{args.cat}/best.pt", map_location="cpu")); cat.eval()
ref = Refiner(); ref.load_state_dict(torch.load(f"{args.cat}/refiner_best.pt", map_location="cpu")); ref.eval()


def to_png(x, path):
    """13-band reflectance in [0,1] -> 8-bit true colour (B4/B3/B2), same x3.5 stretch as the figures."""
    rgb = np.clip(x[[3, 2, 1]].transpose(1, 2, 0) * 3.5, 0, 1)
    Image.fromarray((rgb * 255).round().astype(np.uint8)).save(path)


os.makedirs(args.out, exist_ok=True)
with open(f"{args.out}/index.csv", "w", newline="") as fh, torch.no_grad():
    w = csv.writer(fh)
    w.writerow(["folder", "roi", "tile", "patch", "psnr", "mae", "input_months_from_target", "input_cloud_frac"])
    for i in range(len(ds)):
        b = {k: v[None] for k, v in ds[i].items()}
        mean, logvar, _ = cat(b["s2"], b["s1"], b["cloud"], b["tpos"])
        cond = Refiner.make_cond(mean, logvar, b["cloud"].min(1).values, b["s1"].mean(1))
        pred = ref.sample(mean, cond, args.steps)[0].numpy()
        y = b["y"][0].numpy()
        roi, tile, patch = ds.keys[i]
        d = f"{args.out}/{roi}_{tile}_patch{patch:03d}"
        os.makedirs(d, exist_ok=True)
        for t in range(3):
            to_png(b["s2"][0, t].numpy(), f"{d}/input_t{t + 1}.png")
        to_png(pred, f"{d}/prediction.png")
        to_png(y, f"{d}/ground_truth.png")
        mse = ((pred - y) ** 2).mean()
        w.writerow([os.path.basename(d), roi, tile, patch, round(float(10 * np.log10(1 / max(mse, 1e-10))), 3),
                    round(float(np.abs(pred - y).mean()), 5),
                    " ".join(f"{v:+.2f}" for v in b["tpos"][0].tolist()),
                    " ".join(f"{v:.2f}" for v in b["cloud"][0].mean((1, 2, 3)).tolist())])
        if i % 50 == 0:
            print(i, flush=True)
print("EXPORT_DONE", len(ds), "locations")
