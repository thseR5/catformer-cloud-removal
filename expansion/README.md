# AsiaWest → AsiaEast transfer experiment

This pipeline starts from `runs/full_catformer_v2_s0/best.pt`. It never overwrites
October checkpoints. The old checkpoints contain weights only, so adaptation
starts a new optimizer at learning rate 3e-5. Seed 0 is preselected, not chosen
using AsiaEast test results. The refiner is deferred because its earlier gains
were inconsistent.

## Comparisons

1. `zero_shot`: unchanged AsiaWest-trained CATFormer on both regions' test sets.
2. `finetune`: 20 epochs using AsiaEast training locations plus a fixed sample
   of AsiaWest training locations (up to 25% of the mixed training set).
3. `ssl_finetune`: start independently from the same original weights; 8 epochs
   of masked observed-image reconstruction, then the same supervised fine-tuning.

SSL hides shared 32-pixel blocks in every optical input date, supplies the paired
SAR and remaining optical context, recenters timestamps on the selected observed
date, and reconstructs only its originally clear, nonmissing hidden pixels.
The supervised reference is not used for the SSL loss. Its validation loss uses
fixed masks on held-out validation observations. Test data never enters either
training objective or checkpoint selection.

The SSL branch gets additional training compute; this initial comparison does
not establish a benefit over an equal-compute supervised baseline. A later
compute-matched control and multiple seeds are required for that claim.

## Splits and quality controls

The exact October AsiaWest tile assignments are preserved. AsiaEast is split by
whole ROI with fixed seed 1234 before loading any model results. Files without
CRS, geographic overlap across new splits/regions, or empty splits stop the run.

Each sample has a complete reference with <=5% detected cloud and three paired,
finite nearest-in-actual-time input dates with <50% missing optical pixels each.
Thus, the preparation protocol is stricter than the October loader. Recomputed
zero-shot and adapted results are comparable to one another; directly comparing
them with the October table would conflate data-selection changes and training.
A low-cloud reference is not guaranteed to be genuinely cloud-free.

`expansion/prepared/audit.json` records exclusions, splits, overlap checks and the
original checkpoint checksum. A successful preparation produces an immutable
manifest shared by all three jobs. Each result includes per-location metrics,
least-cloudy baselines and both East transfer and West retention scores.

## Embeddings

Every mode saves the pooled final encoder features with location/region/split
labels. The comparison figure projects test features using one PCA basis fitted
only on the original model's West training features. Embedding clustering is
exploratory and must not be treated as proof of generalisation. Regional score
differences also reflect differences in cloud cover and scene difficulty.

## Files and execution

- `cross_region.py`: quality audit, fixed manifests, inference, masking, adaptation,
  resumable training-state checkpoints, metrics and embeddings.
- `test_transfer.py`: synthetic CPU smoke test for masking, gradients and checkpoints.
- `summarize_transfer.py`: comparison tables and a common-basis embedding figure.
- `prepare_asiaeast.sbatch`: extraction, cloud caches and data preparation.
- `{zero_shot,finetune,ssl_finetune,summary}.sbatch`: sequential experiment jobs.
- The Slurm files target the current HPC account and `/scratch/tadi.reddy.ug24/cloudrm`;
  update those paths, account, and software environment before reusing them on
  another cluster. The one-off download-wait controller used for this run is
  intentionally not included because it contains a run-specific process ID.

Results are written under `runs/transfer_asiaeast_s0/`.

Requested GPU time limits are 1h + 4h + 6h. These are caps rather than estimates.
An interrupted experiment should be inspected and resumed deliberately from its
saved training state; the runner refuses to overwrite a nonempty run directory.
