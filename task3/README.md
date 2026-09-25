# Task 3 -- Domain Generalization on PACS

Seed **6304** throughout. Source domains (labeled, always visible):
**photo, art_painting, cartoon**. Target domain: **sketch** -- completely
unavailable to any Task 3 training, diagnostic, checkpoint-selection, or
hyperparameter-selection step. Sketch is loaded in exactly one script in
this whole task, `evaluate_sketch.py`, and only after every other decision
below has been made and frozen.

## Prerequisite: Task 2 must be run first

Task 3 reuses two things from Task 2 unchanged, per the assignment:

1. **The exact source train/val split** -- `task3/train.py` loads
   `../task2/results/pacs_sketch_seed6304_split.json` rather than
   recomputing a split itself.
2. **The Source-only checkpoint as the ERM baseline** -- `train.py --method erm`
   loads `../task2/checkpoints/source_only.pt` and does not retrain it.

So `task2/` must exist as a sibling folder to `task3/` with `train.py --method source_only`
already having been run at least once. The provided Kaggle notebook
(`task3_kaggle_clean.ipynb`) attaches Task 2's completed code+results as a
second input for exactly this reason.

## What this implements

Three methods, all evaluated identically on source validation and, at the
very end, on Sketch:

| Method  | What it does |
|---------|--------------|
| ERM     | Task 2's Source-only checkpoint, loaded unchanged (not retrained) |
| DAN-DG  | ERM loss + average MMD over all 3 unordered pairs of *observed* source domains (never touches Sketch) -- an adaptation of Task 2's DAN that removes source-domain differences instead of source-target differences |
| SAM     | Standard, non-adaptive Sharpness-Aware Minimization (rho=0.05): seeks source-loss minima that stay low under a worst-case local perturbation, rather than aligning any domains explicitly |

Backbone: the identical ResNet-18 + frozen-BatchNorm policy as Task 2
(`models/backbone.py`, duplicated byte-for-byte from `task2/models/backbone.py`
so that Task 2's checkpoint loads into it with no remapping -- see that
file's docstring for why duplication rather than a cross-task import is
the deliberate choice here).

## Repo layout

```
task3/
  configs/config.yaml         single source of truth (see note on layout below)
  models/
    backbone.py                duplicated from task2/ -- state-dict compatible
    classifier_head.py         duplicated from task2/
  methods/
    erm.py                      documents the ERM objective + the Task-2-checkpoint loader
    dan_dg.py                   pairwise source MMD, reuses shared/mmd.py verbatim
    sam.py                      two-pass sharpness-aware update
  selection/
    source_validation.py        mean-source-macro-F1 checkpoint-selection metric
  evaluation/
    domain_metrics.py           per-domain/mean/worst source metrics + the one Sketch-eval function
    source_domain_separability.py   3-way (Photo/Art/Cartoon) logistic-regression probe, chance=33.3%
    sharpness.py                 the local sharpness diagnostic, shared by all 3 models
    class_analysis.py            per-class Sketch comparison vs ERM (duplicated from task2/)
  tests/test_pipeline.py         synthetic-tensor smoke test -- run this FIRST on Kaggle
  train.py                       trains dan_dg/sam, or loads the ERM checkpoint
  evaluate_sketch.py              THE ONLY script that loads Sketch
```

**Design choice**: like Task 2, this uses one `config.yaml` with a section
per method rather than the suggested separate `erm.yaml`/`dan_dg.yaml`/`sam.yaml`
files, so every setting for every method being compared sits in one place.
Documented here rather than left implicit.

## How to run (in order)

```bash
# 0. sanity check -- catches bugs before spending GPU time, no Task 2 needed
python tests/test_pipeline.py

# 1. ERM: loads Task 2's checkpoint, does not train
python train.py --method erm

# 2. DAN-DG and SAM: real training, source domains only
python train.py --method dan_dg
python train.py --method sam

# 3. controlled design study -- config's controlled_study.method picks
#    which sweep this looks for. Default here is DAN-DG's lambda_dg;
#    lambda_dg=1.0 IS the main run above, so only the two extra points
#    need training:
python train.py --method dan_dg --lambda_dg 0.1  --tag _lam0.1
python train.py --method dan_dg --lambda_dg 10.0 --tag _lam10.0

# (to run the SAM rho sweep instead, set controlled_study.method: sam in
#  configs/config.yaml, then:)
#    python train.py --method sam --rho 0.01 --tag _rho0.01
#    python train.py --method sam --rho 0.1  --tag _rho0.1

# 4. THE ONLY SCRIPT THAT LOADS SKETCH -- run only after every decision
#    above is finished and frozen
python evaluate_sketch.py
```

Each `train.py` call writes `checkpoints/<method><tag>.pt` and
`results/train_history_<method><tag>.json`. `evaluate_sketch.py` is
read-only over those checkpoints and can be re-run any number of times
while writing the report.

## What each result answers

- **`results/main_comparison.csv`** -- per-source-domain accuracy/macro-F1,
  mean AND worst source accuracy/F1 (the assignment asks for both, since a
  strong mean can hide one weak domain), Sketch accuracy/F1, the Sketch
  accuracy change relative to ERM, source-domain separability, and the
  sharpness diagnostic -- one row per method. The headline table: does
  removing source-domain differences (DAN-DG) or seeking flatter minima
  (SAM) generalize to an unseen domain better than plain ERM?
- **`results/per_class_<method>_vs_erm.csv`** -- per-class Sketch accuracy
  for DAN-DG/SAM next to ERM, sorted by improvement. Answers whether an
  aggregate Sketch change is broad or concentrated in a few classes.
- **`results/confusions_<method>_<class>.csv`** -- dominant wrong
  predictions for the single most-improved and most-degraded class per
  method. Answers *why* that class moved.
- **`results/controlled_study_dan_dg.csv`** (or `_sam.csv`) -- headline
  metrics as lambda_dg (or rho) varies. Answers whether stronger source
  alignment (or a larger sharpness radius) keeps helping Sketch, plateaus,
  or starts trading off source performance the way Task 2's DAN sweep did.
- **`source_domain_separability`** -- held-out accuracy of a
  logistic-regression probe trying to tell Photo/Art Painting/Cartoon
  features apart. 33.3% = chance (no recoverable source-domain signal);
  higher = a domain "fingerprint" survives. Per the assignment's warning,
  a LOWER score is evidence of stronger invariance across observed
  sources, not proof that Sketch performance or even class-discriminative
  information improved -- DAN-DG could shrink this score by discarding
  useful structure along with domain structure.
- **`delta_sharp`** -- the loss increase after one normalized
  gradient-ascent step of radius 0.05 on a FIXED 96-example batch (32 per
  source domain, same batch for every model). Lower = locally flatter
  under this specific perturbation. This is a standardized local
  diagnostic, not proof that a model's entire loss landscape is flatter
  everywhere.

## Design choices not fully pinned down by the assignment

- **Sketch is never even opened as an ImageFolder during training.**
  `train.py::prepare_source_only_data` calls `shared/pacs.py`'s
  `load_domain_pair()` only for the three source domains, deliberately
  avoiding `load_all_domain_pairs()` (which would also instantiate a
  Sketch `ImageFolder`, even if nothing then reads a Sketch image through
  the model). This is a stricter reading of "no Sketch image may be
  loaded" than merely "never forward a Sketch image through the network."
- **SAM's optimizer interaction**: after the ascent step and the second
  backward pass, `optimizer.step()` is called once, applying the
  perturbed-point gradient (sitting in `.grad` from the second backward
  pass) to the restored, unperturbed parameters -- the standard SAM/AdamW
  interaction pattern from the reference implementations of Foret et al. (2021).
- **`steps_per_epoch`**: same definition as Task 2 -- the largest source
  domain's `len(train_split) // source_per_domain_batch`, documented in
  `train.py::train_one_method`.

## Attribution

- ResNet-18 architecture and ImageNet-pretrained weights: `torchvision.models.resnet18` (`IMAGENET1K_V1`), used as-is.
- MMD kernel formulation: identical to Task 2's, following the standard description in Long et al. (Learning Transferable Features with Deep Adaptation Networks); adapted here to align pairs of source domains rather than source-target, per the assignment's DAN-DG specification.
- SAM formulation follows the two-step ascent/descent procedure described in Foret et al. (2021), Sharpness-Aware Minimization for Efficiently Improving Generalization.
- All code (`methods/*.py`, `evaluation/*.py`, `train.py`, `evaluate_sketch.py`, `tests/test_pipeline.py`) was written from scratch for this assignment, not copied from any reference SAM or DAN-DG implementation.

## Reproducibility

Every stochastic step (model init beyond the pretrained ImageNet backbone,
domain-balanced batch sampling order, the source-domain-separability
probe's train/test split, the fixed sharpness batch) is seeded with
**6304** or a fixed offset of it. The source train/val split is loaded
verbatim from Task 2's saved file rather than recomputed, so ERM, DAN-DG,
SAM, and the controlled study all validate on the exact same held-out
examples.
