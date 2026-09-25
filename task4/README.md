# Task 4 -- Open-Set Recognition on CIFAR-10 (known) / CIFAR-100 (unknown, eval-only)

Seed **6304** throughout. Known classes: all ten CIFAR-10 classes. Unknowns
(evaluation-only, never seen during training/selection): two fixed groups of
eight CIFAR-100 fine classes each (800 images/group):

- **Near**: bus, pickup_truck, motorcycle, tractor, wolf, fox, leopard, camel
- **Far**: bottle, bowl, chair, clock, keyboard, mushroom, sunflower, wardrobe

## What this implements

Three trained models, one frozen-model score comparison, and one common
evaluation/failure-analysis pipeline:

| Step | What |
|------|------|
| Vanilla | 10-class ResNet-18-CIFAR, plain cross-entropy, random init |
| Post-hoc scores | MSP, MLS, Energy, Mahalanobis -- all read off Vanilla's frozen logits/features |
| GCSC | Identical recipe to Vanilla + RandAugment(2, 9) |
| PROSER | Fine-tuned from Vanilla's checkpoint; classifier placeholders + manifold-mixup data placeholders |

Backbone: a CIFAR-appropriate ResNet-18 -- the standard torchvision resnet18
architecture with its 7x7/stride-2 stem replaced by a 3x3/stride-1 conv and
the initial max-pool removed, operating natively on 32x32 images (per the
assignment). Trained from random initialization for every method (unlike
Task 2/3's ImageNet fine-tuning) -- Vanilla and GCSC from scratch, PROSER
fine-tuned from the selected Vanilla checkpoint.

## Repo layout

```
task4/
  configs/config.yaml        seed, paths, every hyperparameter (one file, all methods)
  data/
    cifar10.py                loading + transforms (CIFAR-10 mean/std used for CIFAR-100 too, see docstring)
    cifar100_unknowns.py      fixed near/far unknown groups -- evaluation-only, loaded nowhere else
    make_splits.py            stratified 90/10 split, seed 6304, saved once and reused
  models/
    resnet_cifar.py            CIFAR ResNet-18, exposes forward_pre/forward_post (split at layer2/layer3)
    classifier_head.py          linear head, optional "dummy classifier" columns for PROSER
  methods/
    vanilla.py, gcsc.py         each just supplies its augmentation transform
    manifold_mixup.py           different-class pairing + Beta(2,2) interpolation mechanics
    proser.py                   PROSER training loop, both placeholder losses, placeholder detection score
  scores/
    msp.py, mls.py, energy.py, mahalanobis.py    the four required post-hoc scores
  train.py                     unified entry point for all three methods
  extract_outputs.py            caches features/logits for a frozen checkpoint (CIFAR-10 + fixed CIFAR-100 groups)
  evaluation/
    metrics.py                  AUROC (known vs. near/far/all)
    thresholds.py                95th-percentile-of-val calibration, FPR@95TPR
    failure_analysis.py          incorrectly-accepted-unknown inspection
  evaluate_osr.py               builds every required table/figure from cache/
  cache/                        saved features/logits only (never raw images)
  tests/test_pipeline.py        synthetic-tensor smoke test -- run this FIRST
```

**Design choice**: uses one `config.yaml` with a section per method rather
than the suggested separate `vanilla.yaml`/`gcsc.yaml`/`proser.yaml`/
`rpl.yaml` files, so every setting for every method being compared sits
in one place. Documented here rather than left implicit; same convention
as Tasks 2 and 3.

## How to run (in order)

```bash
# 0. sanity check -- seconds, no real data needed
python tests/test_pipeline.py

# 1. Vanilla and GCSC, from scratch (100 epochs each, cosine LR, early-stop-free
#    per the assignment -- checkpoint is simply the best-val-accuracy epoch)
python train.py --method vanilla
python train.py --method gcsc

# 2. PROSER, fine-tuned from Vanilla (50 epochs)
python train.py --method proser --vanilla_checkpoint checkpoints/vanilla.pt

# 3. Cache features/logits for every method over CIFAR-10 train/val/test and
#    the fixed CIFAR-100 near/far groups (checkpoint already frozen at this point)
python extract_outputs.py --method vanilla
python extract_outputs.py --method gcsc
python extract_outputs.py --method proser

# 4. Build every required table/figure from the cached features/logits
python evaluate_osr.py
```

CIFAR-10/100 download automatically via `torchvision.datasets` the first time
`dataset.root` is a writable directory with `download=True` -- no manual
Kaggle-dataset attachment needed, unlike Task 2/3's PACS.

## What each result answers

- **`results/scores_comparison_vanilla.csv`** -- MSP/MLS/Energy/Mahalanobis on
  the one frozen Vanilla model: AUROC (known vs. near/far/all) and the
  validation-calibrated operating point (achieved test acceptance rate,
  near/far rejection rates, FPR@95TPR). Answers what each score captures and
  which is most informative for near vs. far unknowns.
- **`results/models_comparison.csv`** -- Vanilla / GCSC / PROSER, closed-set
  accuracy + the same near/far/all AUROC and calibrated metrics, using MLS as
  the common score, plus an extra PROSER row using its own placeholder-based
  detection score. Answers whether stronger augmentation (GCSC) or learned
  placeholders (PROSER) trade closed-set accuracy for better rejection, or
  vice versa.
- **`results/score_distributions.png`** -- compact 3-panel figure (MSP, MLS,
  Mahalanobis), known-test vs. near vs. far score histograms on the Vanilla
  model.
- **`results/failures_near.csv` / `failures_far.csv`** -- CIFAR-100 examples
  the Vanilla+MLS threshold incorrectly accepted as a known class, with the
  true unknown class, predicted CIFAR-10 class, score, and threshold.
  `confusions_near.csv` / `confusions_far.csv` aggregate these into
  (unknown_class -> predicted_class) counts, for separating semantically
  plausible confusions (e.g. wolf -> dog) from surprising ones.

## Design choices not fully pinned down by the assignment

- **Mahalanobis's shared covariance**: estimated as one pooled diagonal
  variance across all classes' *centered* (per-class-mean-subtracted)
  unaugmented training features -- the standard tied-covariance construction
  (e.g. Lee et al. 2018), not a separate per-class covariance. `+1e-6` on
  every diagonal entry, per the assignment.
- **CIFAR-100 normalization**: unknown-evaluation images are normalized with
  CIFAR-10's mean/std (not CIFAR-100's own), since the model only ever saw
  CIFAR-10-normalized inputs during training -- using CIFAR-100 statistics
  would confound "is this an unknown" with "is this normalized differently."
  Documented in `data/cifar10.py`'s module docstring.
- **PROSER's classifier-placeholder and data-placeholder losses**: the
  assignment explicitly leaves the exact construction to "the paper/reference
  code." Implemented here as: (1) ordinary cross-entropy over all K+5 logits
  with the true label (keeps the true class on top, dummies included); (2)
  after masking the true class to -inf, collapse the five dummy logits to one
  virtual "reject" column via max and train that column to outrank the
  remaining K-1 classes (beta=1 weight on this second term); (3) for a
  manifold-mixed feature between two different-class examples, the same
  max-collapsed dummy column is trained to outrank all K known classes
  (gamma=0.1 weight, applied to the second half of each mini-batch per the
  assignment's explicit half/half split). This is a direct, literal reading
  of the assignment's prose description of the mechanism, not a line-for-line
  port of the official PROSER codebase (not available at build time) -- full
  reasoning in `methods/proser.py`'s module docstring.
- **PROSER's placeholder-based detection score**: `u(x) = max(dummy logits) -
  max(known logits)` -- the margin by which the strongest dummy classifier
  beats the strongest known-class response. Larger margin = the network
  itself would rather flag this as unknown-like than commit to a known class.
- **Manifold-mixup pairing**: different-class partners within a batch are
  found via a resampled random permutation with an explicit linear-search
  fallback for any residual same-class collisions (`methods/manifold_mixup.py`)
  -- guarantees zero same-class mixup pairs without requiring a slower exact
  bipartite matching.

## Attribution

- ResNet-18 architecture: `torchvision.models.resnet18`, stem modified per
  the assignment's explicit instruction (3x3/stride-1 conv1, no maxpool).
  Trained from random initialization (`weights=None`), not ImageNet-pretrained.
- MSP: Hendrycks & Gimpel (2017). MLS / "a good closed-set classifier":
  Vaze et al. (2022). Energy score: Liu et al. (2020). Mahalanobis-distance
  OOD detection: Lee et al. (2018) (cited in the assignment's optional
  readings' broader context; the tied-diagonal-covariance construction here
  follows that paper's standard approach). PROSER (classifier + data
  placeholders via manifold mixup): Zhou et al. (2021) -- see the
  implementation-notes section above and `methods/proser.py` for exactly
  which parts are a literal reading of the assignment's description versus
  the paper's own equations. All code in this directory was written from
  scratch for this assignment.

## Reproducibility

Every stochastic step (train/val split, model init, batch order, mixup
pairing/interpolation, Mahalanobis fit) is seeded with **6304**. The
train/val split is saved to `results/cifar10_seed6304_split.json` on first
use and reloaded (not recomputed) afterward, so Vanilla, GCSC, and PROSER all
select checkpoints against the exact same validation set.
