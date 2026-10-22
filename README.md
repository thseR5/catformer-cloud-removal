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

## Status

Preliminary results exist only on a single tile, with validation patches taken from the training tile, so they are not reported here. Held-out-tile results are pending.
