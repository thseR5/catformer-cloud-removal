"""SEN12MS-CR-TS loader.

Layout: <region>/<ROI>/<tile>/{S1,S2}/<t>/{s1,s2}_<ROI>_<tile>_ImgNo_<t>_<date>_patch_<p>.tif
A sample = one (ROI, tile, patch) location with its time series.
"""
import glob, os, re
from collections import defaultdict

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

PAT = re.compile(r"(s[12])_(ROIs\d+)_(\d+)_ImgNo_(\d+)_(\d{4}-\d{2}-\d{2})_patch_(\d+)\.tif$")


def index_files(root):
    """Return {(roi, tile, patch): {"s1": {t: (path, date)}, "s2": {...}}}."""
    idx = defaultdict(lambda: {"s1": {}, "s2": {}})
    for p in glob.glob(os.path.join(root, "**", "*.tif"), recursive=True):
        m = PAT.search(os.path.basename(p))
        if not m:
            continue
        sensor, roi, tile, t, date, patch = m.groups()
        idx[(roi, tile, int(patch))][sensor][int(t)] = (p, date)
    return idx


def read(path):
    with rasterio.open(path) as f:
        return f.read().astype(np.float32)


def norm_s2(x):
    return np.clip(x, 0, 10000) / 10000.0


def norm_s1(x):
    # VV in [-25, 0] dB, VH in [-32.5, 0] dB, as in the dataset paper
    x = np.nan_to_num(x, nan=-25.0)
    lo = np.array([-25.0, -32.5], np.float32)[:, None, None]
    return (np.clip(x, lo, 0) - lo) / -lo


_DETECTOR = None


def cloud_score(s2n):
    """Per-pixel cloud probability (0..1) from s2cloudless on the 13 normalised S2 bands.
    (A brightness heuristic was tried first; it scored bright bare soil as cloud.)"""
    global _DETECTOR
    if _DETECTOR is None:
        from s2cloudless import S2PixelCloudDetector
        _DETECTOR = S2PixelCloudDetector(all_bands=True, average_over=4, dilation_size=2)
    return _DETECTOR.get_cloud_probability_maps(s2n.transpose(1, 2, 0)[None])[0].astype(np.float32)


def cloud_cache_path(p):
    return p[:-4] + "_cld.npy"


def cached_cloud(p):
    """Cloud map for S2 file p, from the uint8 cache written by cache_clouds.py if present."""
    c = cloud_cache_path(p)
    if os.path.exists(c):
        return np.load(c).astype(np.float32) / 255.0
    return cloud_score(norm_s2(read(p)))


class CRTS(Dataset):
    """Inputs: T cloudy S2 frames + co-registered S1 + cloud maps; target: clearest frame
    not among the inputs (standard SEN12MS-CR-TS seq2point protocol)."""

    def __init__(self, root, keys=None, T=3, size=256):
        idx = index_files(root)
        full = {k: v for k, v in idx.items() if len(v["s2"]) >= T + 1 and len(v["s1"]) >= T + 1}
        self.idx, self.keys = full, keys or sorted(full)
        self.T, self.size = T, size
        self._plan = {}

    def plan(self, k):
        """Pick target = least cloudy frame, inputs = T cloudiest-but-informative others nearest in time."""
        if k in self._plan:
            return self._plan[k]
        ts = sorted(set(self.idx[k]["s2"]) & set(self.idx[k]["s1"]))
        cov = {t: float(cached_cloud(self.idx[k]["s2"][t][0]).mean()) for t in ts}
        tgt = min(ts, key=cov.get)
        rest = sorted([t for t in ts if t != tgt], key=lambda t: abs(t - tgt))[: self.T]
        self._plan[k] = (sorted(rest), tgt, cov)
        return self._plan[k]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        k = self.keys[i]
        ins, tgt, cov = self.plan(k)
        s2 = np.stack([norm_s2(read(self.idx[k]["s2"][t][0])) for t in ins])
        s1 = np.stack([norm_s1(read(self.idx[k]["s1"][t][0])) for t in ins])
        cm = np.stack([cached_cloud(self.idx[k]["s2"][t][0]) for t in ins])[:, None]
        y = norm_s2(read(self.idx[k]["s2"][tgt][0]))
        # relative acquisition time in months (real dates; revisits are irregular)
        d = lambda t: np.datetime64(self.idx[k]["s2"][t][1])
        tpos = np.array([(d(t) - d(tgt)).astype(int) / 30.0 for t in ins], np.float32)
        return dict(s2=torch.from_numpy(s2), s1=torch.from_numpy(s1), cloud=torch.from_numpy(cm),
                    y=torch.from_numpy(y), tpos=torch.from_numpy(tpos),
                    tgt_cov=torch.tensor(cov[tgt]))
