# Task 1 -- Inductive Biases and Feature Representations

Dataset: **Oxford-IIIT Pets** (37 classes). Backbones: torchvision ResNet-50
(`IMAGENET1K_V2`), torchvision ViT-B/16 (`IMAGENET1K_V1`), OpenCLIP ViT-B/32
(`pretrained="openai"`). All seeds fixed to **6304**.

## Requirements

This code needs a GPU-capable machine with internet access (to download
pretrained weights and the dataset on first run). It will not run to
completion on a CPU-only, no-internet sandbox -- it was developed and
smoke-tested there using synthetic tensors only (see `tests/`), not the
real dataset.

```bash
pip install -r requirements.txt
```

## One-time setup

1. **AdaIN decoder weights.** The shape/texture cue-conflict step (Step 3)
   uses AdaIN style transfer (Huang & Belongie, 2017). The VGG encoder is
   downloaded automatically via torchvision, but the decoder is a
   generative network that must be *trained* to invert VGG features into
   pixels -- training it from scratch is out of scope for this assignment,
   so we load the publicly released checkpoint from the reference
   implementation:

   - Repo: https://github.com/naoto0804/pytorch-AdaIN
   - Direct download (GitHub release asset, no manual browser step needed):
     ```bash
     wget -q https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/decoder.pth \
          -O checkpoints/decoder.pth
     ```
   - Place it at: `task1/checkpoints/decoder.pth`

   `models/adain_net.py` raises a clear `FileNotFoundError` (rather than
   silently generating garbage images) if this file is missing.

2. **Build the data splits and cue conflicts** (downloads Oxford-IIIT Pets
   automatically):

   ```bash
   cd task1
   python data/make_subset.py --config configs/config.yaml
   python data/make_cue_conflicts.py --config configs/config.yaml
   ```

   This writes:
   - `results/trainval_split_seed6304.json` -- stratified 80/20 split
   - `results/test_subset_ids.json` -- class-balanced 500-image test subset
   - `results/cue_conflict_manifest.json` + `data/cue_conflicts/*.png` --
     accepted cue-conflict images and their metadata

3. **Run the full pipeline:**

   ```bash
   python scripts/run_task1.py --config configs/config.yaml
   ```

   This extracts features, trains one linear head per backbone, runs every
   required intervention, and writes all report tables to `results/*.csv`
   and figures to `results/figures/*.png`.

## Experimental-design choices (yours to justify in the report)

These are fixed in `configs/config.yaml`, not hardcoded in scripts, so they
can be changed and rerun:

- **Dataset:** Oxford-IIIT Pets (fine-grained, 37 classes) rather than
  STL-10.
- **Additional color intervention:** fixed 90-degree hue rotation in HSV
  space (`interventions.color.hue_rotation_degrees`). Chosen over palette
  transfer / class-swapped statistics because it is fully controlled,
  exactly preserves luminance and geometry, and isolates "does the model
  care which hue a surface has" from any texture confound.
- **Cue-conflict class pairs and style strength:** 5 pairs sampled with
  seed 6304 from all `C(37, 2)` pairs; AdaIN `alpha=1.0` (full style
  strength). See `results/cue_conflict_manifest.json` for the realized
  pairs.
- **Cue-conflict rejection rule** (fixed *before* any model sees the
  images -- see `data/make_cue_conflicts.py::passes_rejection_rule`):
  reject if the output is near-uniform (`pixel_std < 0.03`) or if its edge
  energy (mean squared Laplacian response) falls outside
  `[0.15, 6.0] x` the content image's edge energy.
- **Representation-visualization method:** t-SNE (`configs/config.yaml:
  representation.method`), perplexity 30. Switch to UMAP by changing that
  one field.

## Repository layout

```
task1/
  configs/config.yaml       single source of truth for every seed/hyperparameter
  data/
    dataset.py               PetsSubset: common [0,1] 224x224 image loader
    split_utils.py            stratified split + class-balanced subset (pure functions)
    make_subset.py            builds the 80/20 split and the 500-image test subset
    make_cue_conflicts.py     AdaIN cue-conflict generation + rejection rule
    transforms.py              grayscale, hue rotation, translation, patch shuffle
  models/
    backbones.py               ResNet-50 / ViT-B-16 / CLIP-ViT-B-32 wrappers
    linear_probe.py            AdamW linear-head training with early stopping
    adain_net.py                AdaIN encoder/decoder + style transfer
  analysis/
    metrics.py                 top1, macro-F1, mean-max-confidence, consistency, cosine sim
    evaluate_bias.py           Step 1-5 report tables
    feature_similarity.py      I_T cosine-stability formula
    representation.py          joint t-SNE/UMAP plotting
  scripts/run_task1.py        end-to-end orchestration (Steps 0-6)
  tests/test_pipeline.py      synthetic-tensor smoke tests (no GPU/dataset needed)
  checkpoints/                 place decoder.pth here (not committed)
  results/                     all CSV/JSON outputs + results/figures/*.png
```

## Attribution

- AdaIN architecture and pretrained decoder: Huang & Belongie (2017),
  weights from https://github.com/naoto0804/pytorch-AdaIN.
- Backbones and ImageNet weights: `torchvision.models`.
- CLIP: `open_clip_torch`, `ViT-B-32`, `pretrained="openai"`.
- All linear-probe training, transform, metric, and evaluation code in
  this repository was written for this assignment (with the exception of
  the AdaIN decoder weights noted above).

## Reproducibility

Every stochastic step (splits, subset selection, class-pair sampling,
patch permutations, linear-head training, t-SNE/UMAP) is seeded with 6304
and reads that seed from `configs/config.yaml`. No test-set or
CIFAR-100-equivalent label ever influences training or checkpoint
selection.
