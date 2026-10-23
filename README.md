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
Inputs are 3 S2 + S1 dates, and the target is the clearest valid date. The table shows test tile `ROIs1970/112` (270 locations), scored once with the checkpoint chosen on val. Values are mean ± std over 3 seeds.

| Method | Params | PSNR ↑ | SSIM (RGB) ↑ | MAE ↓ | SAM ↓ |
|---|---|---|---|---|---|
| Least-cloudy input | – | 30.80 | 0.840 | 0.0330 | **7.90°** |
| UnCRtainTS (retrained here) | 0.57M | 32.69 ± 0.27 | 0.920 ± 0.004 | 0.0209 ± 0.0011 | 9.75 ± 0.47° |
| CATFormer, Stage 1 | 5.5M | 33.10 ± 0.67 | **0.933 ± 0.003** | 0.0200 ± 0.0010 | 9.66 ± 0.35° |
| CATFormer + refiner | 8.8M | **33.12 ± 0.60** | 0.933 ± 0.003 | **0.0195 ± 0.0004** | 9.39 ± 0.33° |

Findings:
- SSIM: every CATFormer seed beats every UnCRtainTS seed (+0.012 on average).
- PSNR is +0.42 dB on average, but that is within the seed spread, so it is not a reliable gain yet.
- The refiner adds almost nothing over Stage 1. Its best val epoch was 1–13 of 20, because the val tile is nearly cloud-free and is a weak signal for choosing checkpoints.
- Every learned model is worse than the least-cloudy baseline on SAM (spectral fidelity).
- These numbers come from one test tile and are not directly comparable to published results, which use other regions and splits.

`viz.py` generates a qualitative figure (best and worst test locations, all methods, uncertainty map).
