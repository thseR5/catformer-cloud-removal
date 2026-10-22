"""EDA for SEN12MS-CR-TS samples. Writes PNG figures + summary.json to --out."""
import argparse, json, os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import cloud_score, index_files, norm_s1, norm_s2, read

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--out", default="eda")
ap.add_argument("--max_locs", type=int, default=150)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]

idx = index_files(args.root)
locs = sorted(idx)
print("locations found:", len(locs))
full = [k for k in locs if len(idx[k]["s2"]) == 30 and len(idx[k]["s1"]) == 30]
use = (full or [k for k in locs if len(idx[k]["s2"]) >= 10])[: args.max_locs]

rows, s2_px, s1_px, nan_s1 = [], [], [], 0
for k in use:
    for sensor in ("s2", "s1"):
        for t, (p, date) in idx[k][sensor].items():
            x = read(p)
            r = dict(roi=k[0], tile=k[1], patch=k[2], sensor=sensor, t=t, date=date,
                     shape="x".join(map(str, x.shape)))
            if sensor == "s2":
                n = norm_s2(x)
                c = cloud_score(n)
                r.update(cloud_cov=float((c > 0.5).mean()), cloud_mean=float(c.mean()),
                         zero_frac=float((x == 0).all(0).mean()))
                s2_px.append(x[:, ::16, ::16].reshape(13, -1))
            else:
                nan_s1 += int(np.isnan(x).sum())
                s1_px.append(x[:, ::16, ::16].reshape(2, -1))
            rows.append(r)
df = pd.DataFrame(rows)
df.to_csv(f"{args.out}/frames.csv", index=False)
s2df = df[df.sensor == "s2"]
s2_px, s1_px = np.concatenate(s2_px, 1), np.concatenate(s1_px, 1)

# 1. cloud coverage distribution
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
ax[0].hist(s2df.cloud_cov, bins=40, color="steelblue"); ax[0].set(title="Per-frame cloud coverage", xlabel="fraction cloudy", ylabel="frames")
s2df.groupby("t").cloud_cov.mean().plot(ax=ax[1], marker="o"); ax[1].set(title="Mean cloud coverage by time index", xlabel="t (0-29)")
clear = s2df.groupby(["roi", "tile", "patch"]).cloud_cov.apply(lambda c: (c < 0.05).sum())
ax[2].hist(clear, bins=range(0, 32), color="seagreen"); ax[2].set(title="Clear frames (<5% cloud) per location", xlabel="# clear frames of 30")
plt.tight_layout(); plt.savefig(f"{args.out}/01_cloud_coverage.png", dpi=120); plt.close()

# 2. acquisition dates
d = pd.to_datetime(s2df.date)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
d.dt.month.value_counts().sort_index().plot.bar(ax=ax[0]); ax[0].set(title="S2 acquisitions by month")
s2df.assign(m=d.dt.month).groupby("m").cloud_cov.mean().plot.bar(ax=ax[1], color="gray"); ax[1].set(title="Mean cloud coverage by month")
plt.tight_layout(); plt.savefig(f"{args.out}/02_dates.png", dpi=120); plt.close()

# 3. band distributions
fig, ax = plt.subplots(1, 2, figsize=(16, 4))
ax[0].boxplot([np.clip(b, 0, 12000) for b in s2_px], tick_labels=BANDS, showfliers=False); ax[0].set(title="S2 reflectance per band (raw DN)")
for i, n in enumerate(["VV", "VH"]):
    v = s1_px[i]; ax[1].hist(v[np.isfinite(v)], bins=100, alpha=.6, label=n)
ax[1].legend(); ax[1].set(title="S1 backscatter (dB)")
plt.tight_layout(); plt.savefig(f"{args.out}/03_band_distributions.png", dpi=120); plt.close()

# 4. band correlation (S2 + S1)
pairs = []
for k in use[:60]:
    for t in sorted(set(idx[k]["s2"]) & set(idx[k]["s1"])):
        a, b = read(idx[k]["s2"][t][0]), read(idx[k]["s1"][t][0])
        if cloud_score(norm_s2(a)).mean() < 0.05:  # clear frames only, so optics reflect the surface
            pairs.append(np.concatenate([a, b], 0)[:, ::16, ::16].reshape(15, -1))
allb = np.concatenate(pairs, 1); ok = np.isfinite(allb).all(0)
corr = np.corrcoef(allb[:, ok])
fig, ax = plt.subplots(figsize=(8, 7)); im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
lab = BANDS + ["VV", "VH"]; ax.set_xticks(range(15), lab, rotation=90); ax.set_yticks(range(15), lab)
plt.colorbar(im); ax.set_title("Pixel correlation across bands (clear, co-registered frames)"); plt.tight_layout()
plt.savefig(f"{args.out}/04_band_correlation.png", dpi=120); plt.close()

# 5. time-series examples: RGB + VV for 3 locations
def rgb(x):
    return np.clip(norm_s2(x)[[3, 2, 1]].transpose(1, 2, 0) * 3.5, 0, 1)
ex = use[:: max(1, len(use) // 3)][:3]
fig, ax = plt.subplots(2 * len(ex), 10, figsize=(20, 4.3 * len(ex)))
for r, k in enumerate(ex):
    for j, t in enumerate(sorted(idx[k]["s2"])[::3][:10]):
        p, date = idx[k]["s2"][t]; x = read(p)
        ax[2 * r, j].imshow(rgb(x)); ax[2 * r, j].set_title(f"t{t} {date}\ncloud {cloud_score(norm_s2(x)).mean():.2f}", fontsize=7)
        if t in idx[k]["s1"]:
            ax[2 * r + 1, j].imshow(norm_s1(read(idx[k]["s1"][t][0]))[0], cmap="gray")
for a in ax.ravel(): a.axis("off")
plt.suptitle("S2 RGB (rows 1,3,5) and S1 VV (rows 2,4,6), every 3rd time step"); plt.tight_layout()
plt.savefig(f"{args.out}/05_timeseries_examples.png", dpi=100); plt.close()

# 6. cloud score map sanity check
k = ex[0]; ts = sorted(idx[k]["s2"], key=lambda t: -s2df[(s2df.patch == k[2]) & (s2df.t == t)].cloud_cov.max())[:4]
fig, ax = plt.subplots(2, 4, figsize=(14, 7))
for j, t in enumerate(ts):
    x = read(idx[k]["s2"][t][0]); ax[0, j].imshow(rgb(x)); ax[1, j].imshow(cloud_score(norm_s2(x)), vmin=0, vmax=1)
for a in ax.ravel(): a.axis("off")
plt.suptitle("Cloudiest frames and their cloud-probability maps"); plt.tight_layout()
plt.savefig(f"{args.out}/06_cloud_maps.png", dpi=100); plt.close()

summary = dict(
    locations_indexed=len(locs), locations_complete_30=len(full), locations_analysed=len(use),
    frames_s2=int(len(s2df)), frames_s1=int((df.sensor == "s1").sum()),
    shapes=dict(Counter(df["shape"])),
    date_range=[str(d.min().date()), str(d.max().date())],
    cloud_cov_mean=float(s2df.cloud_cov.mean()), cloud_cov_median=float(s2df.cloud_cov.median()),
    frac_frames_clear_lt5pct=float((s2df.cloud_cov < 0.05).mean()),
    frac_frames_overcast_gt90pct=float((s2df.cloud_cov > 0.9).mean()),
    locations_with_no_clear_frame=int((clear == 0).sum()),
    clear_frames_per_location_mean=float(clear.mean()),
    s1_nan_values=nan_s1, s2_nodata_pixel_frac=float(s2df.zero_frac.mean()),
    s2_band_mean_dn=dict(zip(BANDS, map(float, s2_px.mean(1)))),
    s1_mean_db=dict(VV=float(np.nanmean(s1_px[0])), VH=float(np.nanmean(s1_px[1]))),
)
json.dump(summary, open(f"{args.out}/summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
