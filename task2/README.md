# Task 2 -- Unsupervised Domain Adaptation on PACS

Seed **6304** is used throughout (dataset split, batch sampling, model
init, domain-separability probe). Source domains: **photo, art_painting,
cartoon**. Target domain (unlabeled during training, held out for
evaluation only): **sketch**.

## What this implements

Four methods, all sharing one backbone, one training loop, and one
evaluation pipeline -- only the per-step loss differs:

| Method       | Loss                                                        |
|--------------|--------------------------------------------------------------|
| Source-only  | Cross-entropy on labeled source examples only (ERM baseline) |
| DAN          | Source CE + lambda_MMD * MMD^2(source features, target features) |
| DANN         | Source CE + domain-adversarial loss via a Gradient Reversal Layer |
| CDAN         | Same as DANN, but the discriminator sees g(x) = vec(f(x) (x) p(x)) -- the outer product of features and predicted class probabilities |

Backbone: ResNet-18, ImageNet-pretrained. Per the assignment, it is
**fully fine-tuned** for every method (unlike Task 1's frozen probing) --
the only thing frozen is BatchNorm's running mean/variance, which stay at
their pretrained ImageNet values for every method so that adaptation
can't "cheat" by just re-normalizing statistics toward the target domain.
gamma/beta remain trainable. This is implemented in
`models/backbone.py::set_train_with_frozen_bn`, called instead of
`model.train()` every epoch.

## Getting the PACS dataset

PACS isn't bundled with torchvision, so you need to attach it as a
Kaggle dataset the same way you did Oxford-IIIT Pets in Task 1. Two
public Kaggle mirrors that carry the standard photo/art_painting/cartoon/
sketch folder structure:

- https://www.kaggle.com/datasets/ma3ple/pacs-dataset
- https://www.kaggle.com/datasets/nickfratto/pacs-dataset

On the notebook's right sidebar: **Add Input -> search "PACS" -> Add**.
Whichever one you use, `shared/pacs.py::locate_pacs_domains()` searches
recursively under `/kaggle/input` for four folders named
`photo`/`art_painting`/`cartoon`/`sketch` (case- and separator-insensitive
-- it also accepts "art painting" or "artpainting"), each containing 7
class subfolders (dog, elephant, giraffe, guitar, horse, house, person).
If it can't find all four, it raises an error listing exactly what it did
find, so a wrong/incomplete dataset attachment fails loudly instead of
silently training on the wrong thing. You do not need to manually move or
rename anything -- just attach the dataset and point `dataset.root` in
`configs/config.yaml` at `/kaggle/input` (the provided notebook does this
via the same glob-based auto-discovery pattern used in Task 1).

## Repo layout

```
task2/
  configs/config.yaml       single source of truth: seed, paths, every hyperparameter
  models/
    backbone.py              ResNet-18 + the BatchNorm-freeze policy
    classifier_head.py        512 -> 7 linear head
    domain_discriminator.py   gradient reversal layer, alpha schedule, discriminator, CDAN conditioning
  methods/
    source_only.py, dan.py, dann.py, cdan.py    one compute_loss() each, common interface
  evaluation/
    metrics.py                 accuracy / macro-F1 extraction
    domain_separability.py     the domain-fingerprint diagnostic (Step 5)
    class_analysis.py          per-class comparison + dominant confusions (Step 4/6)
  tests/test_pipeline.py       synthetic-tensor smoke test -- run this FIRST on Kaggle
  train.py                     unified training loop, used by every method
  evaluate_final.py            builds every results/*.csv from the saved checkpoints
shared/
  pacs.py                      PACS loading + auto-discovery (used by task2/ and task3/)
  pacs_protocol.py              the one train/val split + domain-balanced batching, shared with task3/
  mmd.py                        multi-kernel MMD, shared verbatim with task3/'s DAN-DG
```

**Design choice**: uses one `config.yaml` with a section per method rather
than the suggested separate `base.yaml`/`source_only.yaml`/`dan.yaml`/
`dann.yaml`/`cdan.yaml` files, so every setting for every method being
compared sits in one place. Documented here rather than left implicit;
Task 3 follows the same convention.

**Design choice**: the train/val split is saved to
`task2/results/pacs_sketch_seed6304_split.json` (generated once by
`train.py` on its first run) rather than to a `shared/splits/` location.
Task 3 loads this exact file directly (see `task3/README.md`), which
guarantees byte-identical reuse rather than relying on two code paths
writing to a shared location staying in sync.

## How to run (in order)

```bash
# 0. sanity check -- takes seconds, catches bugs before spending GPU time
python tests/test_pipeline.py

# 1. train all four methods (each trains from a fresh ImageNet init,
#    early-stopping on mean_source_macro_f1 across the 3 source val sets)
python train.py --method source_only
python train.py --method dan
python train.py --method dann
python train.py --method cdan

# 2. controlled design study (Task 2's required ablation): sweep DAN's
#    lambda_MMD over {0.1, 1.0, 10.0}. lambda_MMD=1.0 IS the main DAN run
#    above, so only the two extra points need training:
python train.py --method dan --lambda_mmd 0.1  --tag _lam0.1
python train.py --method dan --lambda_mmd 10.0 --tag _lam10.0

# 3. build every comparison table/figure input from the saved checkpoints
python evaluate_final.py
```

Each `train.py` call writes `checkpoints/<method><tag>.pt` and
`results/train_history_<method><tag>.json`. `evaluate_final.py` is
read-only over those checkpoints, so it can be re-run any number of times
while writing the report without retraining anything.

## What each result answers

- **`results/main_comparison.csv`** -- per-source-domain accuracy/macro-F1,
  mean-source accuracy/F1, target (sketch) accuracy/F1, and the domain-
  separability score, one row per method. This is the headline table:
  does adaptation improve target accuracy over Source-only, and at what
  cost (if any) to source performance?
- **`results/per_class_<method>_vs_source_only.csv`** -- per-class target
  accuracy for DAN/DANN/CDAN next to Source-only, sorted by improvement.
  Answers whether an aggregate gain is broad or concentrated in a couple
  of classes.
- **`results/confusions_<method>_<class>.csv`** -- for the single most-
  improved and most-degraded class under each method, the top wrong
  predicted classes. Answers *why* that class moved.
- **`results/controlled_study_dan_lambda_mmd.csv`** -- the same headline
  metrics as lambda_MMD varies. Answers whether stronger domain alignment
  keeps helping, plateaus, or starts trading off source accuracy for
  target accuracy (the classic DAN/DANN over-regularization failure
  mode).
- **Domain-separability score** (in both tables above) -- accuracy of a
  logistic-regression probe trying to tell source-val features apart from
  target features, on frozen features from the trained model. 50% =
  indistinguishable (the representation carries no recoverable domain
  signal); higher = a domain "fingerprint" survives. This is a
  *diagnostic*, not a target-accuracy metric -- per the assignment's own
  warning, a lower score is evidence alignment worked, not proof that
  class-discriminative information was preserved (that's what
  target_accuracy is for).

## Design choices not fully pinned down by the assignment

- **`steps_per_epoch`**: defined as the largest source domain's
  `len(train_split) // source_per_domain_batch`, so the biggest source
  domain is seen exactly once per epoch; smaller domains cycle more than
  once via `infinite_loader`. Documented in `train.py::train_one_method`.
- **Domain-balanced batches**: implemented as one independent DataLoader
  per domain (each infinitely cycling), combined by
  `shared/pacs_protocol.py::SourceTargetBatchIterator`, rather than a
  single custom multi-domain `Sampler` -- simpler to verify class-by-class
  (see `tests/test_pipeline.py`).
- **CDAN**: run with `detach_features=False, detach_probs=False` (no
  entropy conditioning), per the assignment's explicit instruction for the
  required implementation; the detach flags exist in
  `models/domain_discriminator.py::cdan_conditioning` only as an unused
  optional hook.
- **Gradient clipping** (`torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)`,
  applied to every trainable parameter right before `optimizer.step()` in
  `train.py::train_one_method`): added identically for all four methods
  after observing that DANN's and CDAN's adversarial objective could
  produce very large gradients once the domain discriminator became highly
  confident, destabilizing the backbone for several subsequent epochs
  (`domain_loss` spiking from order-1 values into the hundreds, with
  `mean_source_macro_f1` collapsing correspondingly). This is a standard,
  widely used remedy for this class of adversarial-training instability,
  not a per-method tweak: it is applied uniformly across Source-only, DAN,
  DANN, and CDAN to keep the training pipeline identical across methods,
  per the assignment's requirement. In practice Source-only's and DAN's
  gradients already sit well under this bound, so clipping does not change
  their results -- it only stabilizes DANN/CDAN.
- **Non-finite loss guard**: if a single batch's loss is non-finite
  (observed during DANN/CDAN's adversarial training), that batch's update
  is skipped entirely (logged with a `[warning]` line) rather than letting
  a corrupted gradient reach the optimizer. A second, independent safety
  net alongside gradient clipping, applied identically to every method; a
  no-op for source_only/DAN.

## Attribution

- ResNet-18 architecture and ImageNet-pretrained weights: `torchvision.models.resnet18` (`IMAGENET1K_V1`), used as-is via the standard torchvision API.
- MMD kernel formulation (multi-kernel RBF, median-heuristic bandwidth) and the gradient-reversal-layer / DANN formulation follow the standard descriptions in Long et al. (Learning Transferable Features with Deep Adaptation Networks) and Ganin & Lempitsky (Domain-Adversarial Training of Neural Networks); CDAN's conditioning follows Long et al. (Conditional Adversarial Domain Adaptation). All code implementing these ideas (`shared/mmd.py`, `models/domain_discriminator.py`, `methods/*.py`) was written from scratch for this assignment, not copied from any reference implementation.
- All orchestration, data-loading, splitting, evaluation, and analysis code (`shared/pacs*.py`, `train.py`, `evaluate_final.py`, `evaluation/*.py`, `tests/test_pipeline.py`) is original to this submission.

## Reproducibility

Every stochastic step (train/val split, batch sampling order, model
weight init beyond the pretrained ImageNet backbone, the
domain-separability probe's train/test split) is seeded with **6304** or
a fixed offset of it. The train/val split is saved to
`results/pacs_sketch_seed6304_split.json` the first time it's computed and
reloaded (not recomputed) on every subsequent run, so all four methods
and the controlled study train/validate on the exact same split.
