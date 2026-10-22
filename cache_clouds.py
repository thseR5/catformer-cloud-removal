"""Precompute s2cloudless maps for every S2 patch once (uint8, next to each .tif)."""
import glob, os, sys
from multiprocessing import Pool

import numpy as np

from data import cloud_cache_path, cloud_score, norm_s2, read


def work(p):
    c = cloud_cache_path(p)
    if os.path.exists(c):
        return 0
    try:
        np.save(c, (cloud_score(norm_s2(read(p))) * 255).round().astype(np.uint8))
        return 0
    except Exception as e:
        print("BAD", p, e, flush=True)
        return 1


if __name__ == "__main__":
    fs = sorted(glob.glob(os.path.join(sys.argv[1], "**", "S2", "**", "*.tif"), recursive=True))
    print("S2 files:", len(fs), flush=True)
    with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 16) as pool:
        bad = 0
        for i, b in enumerate(pool.imap_unordered(work, fs, chunksize=32)):
            bad += b
            if i % 5000 == 0:
                print(i, "done", flush=True)
    print("CACHE_DONE files", len(fs), "unreadable", bad)
