# CATFormer: cloud removal on SEN12MS-CR-TS

Multi-temporal SAR-optical cloud removal for Sentinel-2, trained on
[SEN12MS-CR-TS](https://arxiv.org/abs/2201.09613). Work in progress.

## Model

**Stage 1, CATFormer** (`model.py`): a U-Net over 3 cloudy Sentinel-2 dates plus co-registered Sentinel-1.
- SAR-gated stem: radar features enter the optical stream through a gate driven by the cloud map.
- Cloud-biased temporal attention at every scale: attention logits get `-beta_h * cloud(t, pixel)`.
- Time encoding uses real acquisition dates.
- Residual-on-composite decoding: predicts a correction to an attention-weighted blend of the inputs.
- Per-pixel, per-band aleatoric uncertainty.
- Loss: occlusion-weighted L1 + Gaussian NLL + SSIM + SAM.

**Stage 2, refiner** (`model.py: Refiner`): a rectified-flow bridge from the Stage 1 estimate to the clear image, conditioned on the Stage 1 uncertainty; 4 sampling steps.

## Files

| File | Purpose |
|---|---|
| `data.py` | dataset indexing, normalisation, s2cloudless cloud maps (cached), sample selection |
| `model.py` | CATFormer, losses, refiner |
| `baselines.py` | UnCRtainTS (official code) behind the same interface |
| `train.py` | whole-tile train/val/test split, baselines, both stages, single final test scoring |
| `cache_clouds.py` | precompute cloud maps |
| `eda.py` | exploratory data analysis figures |
| `*.sbatch` | Slurm job scripts used on the cluster |

## Usage

```bash
git clone https://github.com/PatrickTUM/UnCRtainTS uncrtaints_repo   # needed for baselines.py
pip install torch rasterio s2cloudless scikit-image pandas matplotlib
python cache_clouds.py data/asiaWest 32
python train.py --root data/asiaWest --model catformer --epochs 40 --stage2_epochs 20
python train.py --root data/asiaWest --model uncrtaints --epochs 40 --lr 1e-3
```

## Results (preliminary)

Region asiaWest of SEN12MS-CR-TS (7 tiles, 1,920 locations). The split is by whole tile: 5 train / 1 val / 1 test.
Inputs are 3 S2 + S1 dates, and the target is the clearest valid date. The table shows test tile `ROIs1970/112` (270 locations), scored once with the checkpoint chosen on val. Seed 0 only.

| Method | Params | PSNR ↑ | SSIM (RGB) ↑ | MAE ↓ | SAM ↓ |
|---|---|---|---|---|---|
| Mean of inputs | – | 23.29 | 0.721 | 0.0767 | 11.82° |
| Least-cloudy input | – | 30.80 | 0.840 | 0.0330 | **7.90°** |
| UnCRtainTS (retrained here) | 0.57M | 32.53 | 0.920 | 0.0218 | 9.75° |
| CATFormer, Stage 1 | 5.5M | 32.49 | 0.933 | 0.0210 | 9.72° |
| **CATFormer + refiner** | 8.8M | **32.72** | **0.936** | **0.0196** | 9.09° |

Caveats:
- Single seed and a single test tile, so the gaps between the learned models may be within noise. Runs for seeds 1–2 are in progress.
- Every learned model is worse than the least-cloudy baseline on SAM (spectral fidelity).
- The val tile is nearly cloud-free, so it is a weak signal for choosing checkpoints.
- These numbers are not directly comparable to published results, which use other regions and splits.
