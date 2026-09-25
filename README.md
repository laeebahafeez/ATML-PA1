# Beyond IID: Inductive Biases, Domain Adaptation/Generalization, and Open-Set Recognition

Programming Assignment 1 (EE-5102 / CS-6304). Four tasks, one seed (**6304**) fixed
throughout every stochastic step (splits, initialization, batch order, augmentation,
mixup pairing, Mahalanobis fit).

- **Task 1** -- Inductive biases of ResNet-50, ViT-B/16, and zero-shot CLIP ViT-B/32
  under color, shape/texture, translation, and patch-shuffle interventions.
- **Task 2** -- Domain adaptation on PACS (Photo/Art/Cartoon source domains, Sketch
  target): Source-only ERM, DAN (MMD), DANN, CDAN.
- **Task 3** -- Domain generalization on PACS (Sketch held out entirely from training
  and model selection): ERM, DAN-DG (target-free pairwise-source MMD), SAM.
- **Task 4** -- Open-set recognition on CIFAR-10 (known) vs. fixed CIFAR-100 near/far
  unknown groups (evaluation-only): Vanilla, GCSC, PROSER, with four post-hoc scores
  (MSP, MLS, Energy, Mahalanobis).

## Repository layout

```
pa1-beyond-iid/
  requirements.txt
  shared/            PACS loading + protocol + MMD, used by both task2/ and task3/
  task1/ .. task4/   code, configs, README, and results/ (small CSV/JSON/PNG only)
  report/figures/    figures referenced by the PDF report
```

Each `task{1..4}/README.md` documents that task's design choices, attribution, and
exact run order in detail. Each `task{1..4}/<task>_kaggle_clean.ipynb` is the notebook
actually used to produce the committed results -- all four tasks require a GPU
(trained from scratch or fine-tuned; Task 1 only needs a GPU for speed, not
correctness) and were run on Kaggle (T4 x2).

There is no top-level `common/` directory: no utility in this codebase is genuinely
shared across all four tasks (each task's `evaluation/metrics.py` etc. is scoped to
that task's own metrics). `shared/` holds only the PACS-specific code Task 2 and
Task 3 both depend on, per the assignment's suggested structure for those two tasks.

One structural note: the PACS train/val split
(`pacs_sketch_seed6304_split.json`) is saved to `task2/results/` rather than
`shared/splits/`, because that is the path the tested, verified code actually reads
and writes (Task 3's `evaluate_sketch.py` and model-selection code load Task 2's
split from `../task2/results/`). This matches what was actually run and verified on
Kaggle; moving the file without touching the code would desynchronize the repo from
the results it documents.

## Reproducing each task

All four tasks were run on Kaggle (GPU T4 x2, Internet on). For each task:

1. Attach the task's code as a Kaggle Dataset input (zip the `task{N}/` folder --
   and `shared/` too, for Task 2/3 -- as siblings, matching this repo's layout).
2. Open `task{N}/task{N}_kaggle_clean.ipynb`, attach the required inputs listed in
   its first cell, and run cells top to bottom.
3. Trained checkpoints and cached features are intentionally **not** committed to
   this repository (see `.gitignore`) -- they are large (Task 2: ~250MB, Task 3:
   ~208MB, Task 4: ~125MB) and fully regenerable from the committed code + configs
   + seed 6304. The small, machine-readable results (CSVs, JSONs, the Task 4
   score-distribution figure) that every reported number traces back to **are**
   committed, under each task's `results/`.

Local setup (for smoke tests / non-GPU-bound code only):
```bash
pip install -r requirements.txt
python task4/tests/test_pipeline.py   # example: any task's smoke test needs no GPU
```

## AI usage attribution

Per the course's AI Usage Policy: code in this repository was written with
implementation assistance from Claude (Anthropic) -- architecture decisions, method
implementations, debugging, and Kaggle-environment troubleshooting were done
interactively with AI assistance, and every line was reviewed and is understood by
the author. The accompanying PDF report's language, analysis, and conclusions were
written entirely by the author without generative-AI assistance, per the assignment's
explicit prohibition on AI-drafted report content. External-method attributions
(DAN, DANN, CDAN, SAM, PROSER, MSP, MLS, Energy, Mahalanobis-distance OOD detection,
AdaIN) are documented in each task's own README.

## Report

The PDF report (8 pages, NeurIPS format) is submitted separately per the course's
LMS submission requirements; a link to this repository is included at the end of
its abstract.
