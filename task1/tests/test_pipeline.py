"""
Synthetic-data smoke tests for the torch-free parts of the Task 1 pipeline.

These run with only numpy + pandas (no torch/torchvision/open_clip/
scikit-learn required), which is what is available in a CPU-only sandbox
with no internet access. They check:

    - split_utils: stratified split proportions, no train/val overlap,
      determinism given the fixed seed, class-balanced subset sizing and
      documented imbalance handling.
    - metrics: softmax normalization, top-1 accuracy, mean-max-confidence,
      prediction consistency, cosine similarity.
    - evaluate_bias: shape-bias / coverage arithmetic, color-bias deltas,
      translation-curve aggregation, patch-shuffle accuracy drop.
    - a dependency-free re-derivation of the patch-shuffle index math used
      in data/transforms.py (block decomposition + permutation +
      recomposition), to check that logic without needing torch installed.

What this suite does NOT exercise (needs torch/torchvision/open_clip/GPU,
run on your own machine): models/backbones.py, models/adain_net.py,
models/linear_probe.py, the torch-tensor versions of data/transforms.py,
and scripts/run_task1.py end-to-end.

Run: python tests/test_pipeline.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))

from split_utils import class_balanced_subset, stratified_train_val_split  # noqa: E402
from metrics import (  # noqa: E402
    cosine_similarity_paired, mean_max_confidence, prediction_consistency,
    softmax_np, top1_accuracy,
)
from evaluate_bias import (  # noqa: E402
    classify_cue_conflict_prediction, color_bias_report,
    patch_shuffle_report, shape_bias_report, translation_curve_report,
)

PASS = 0
FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [ok] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}")


# --------------------------------------------------------------------------- #
print("split_utils.stratified_train_val_split")
labels = np.array([c for c in range(10) for _ in range(20)])  # 10 classes x 20
train_idx, val_idx = stratified_train_val_split(labels, val_fraction=0.2, seed=6304)
check("no overlap between train and val", set(train_idx).isdisjoint(set(val_idx)))
check("train+val covers every example", len(train_idx) + len(val_idx) == len(labels))
check("~80/20 split", abs(len(train_idx) - 160) <= 10 and abs(len(val_idx) - 40) <= 10)
for c in range(10):
    n_val_c = sum(1 for i in val_idx if labels[i] == c)
    check(f"class {c} gets ~20% in val (stratified)", n_val_c == 4)
train_idx2, val_idx2 = stratified_train_val_split(labels, val_fraction=0.2, seed=6304)
check("deterministic given fixed seed", train_idx == train_idx2 and val_idx == val_idx2)

# --------------------------------------------------------------------------- #
print("\nsplit_utils.class_balanced_subset")
# 37 pseudo-classes, uneven pool sizes, one deliberately scarce class (id 5)
rng = np.random.default_rng(0)
labels2 = []
for c in range(37):
    n = 5 if c == 5 else 50
    labels2 += [c] * n
labels2 = np.array(labels2)
selected, per_class_count, imbalance = class_balanced_subset(labels2, target_size=500, seed=6304)
check("selected size <= target (scarce class truncates)", len(selected) <= 500)
check("scarce class documented in imbalance notes", 5 in imbalance)
check("scarce class capped at its availability", per_class_count[5] == 5)
check("no duplicate indices", len(selected) == len(set(selected)))
selected2, _, _ = class_balanced_subset(labels2, target_size=500, seed=6304)
check("deterministic given fixed seed", selected == selected2)

# --------------------------------------------------------------------------- #
print("\nmetrics")
logits = np.array([[2.0, 1.0, 0.1], [0.1, 0.2, 3.0]])
probs = softmax_np(logits)
check("softmax rows sum to 1", np.allclose(probs.sum(axis=1), 1.0))
preds = probs.argmax(axis=1)
labels3 = np.array([0, 2])
check("top1 accuracy = 1.0 on a perfect match", top1_accuracy(preds, labels3) == 1.0)
check("top1 accuracy = 0.0 on a total miss", top1_accuracy(preds, np.array([1, 0])) == 0.0)
check("mean max confidence in (0, 1]", 0 < mean_max_confidence(probs) <= 1.0)
check("consistency = 1.0 for identical predictions",
      prediction_consistency([0, 1, 2], [0, 1, 2]) == 1.0)
check("consistency = 0.0 for fully different predictions",
      prediction_consistency([0, 1, 2], [1, 2, 0]) == 0.0)

a = np.array([[1.0, 0.0], [0.0, 1.0]])
b = np.array([[1.0, 0.0], [1.0, 0.0]])
sims = cosine_similarity_paired(a, b)
check("cosine sim of identical vectors is 1", np.isclose(sims[0], 1.0))
check("cosine sim of orthogonal vectors is 0", np.isclose(sims[1], 0.0))

# --------------------------------------------------------------------------- #
print("\nevaluate_bias: shape-bias / coverage arithmetic")
# 4 cue conflicts: content classes [0,0,1,1], style classes [1,1,0,0]
manifest = [
    {"content_class_idx": 0, "style_class_idx": 1},
    {"content_class_idx": 0, "style_class_idx": 1},
    {"content_class_idx": 1, "style_class_idx": 0},
    {"content_class_idx": 1, "style_class_idx": 0},
]
# model predicts: shape(0), shape(0), texture(0==style of item 2), other(2)
#   item 0: pred=0, content=0, style=1 -> matches content -> shape
#   item 1: pred=0, content=0, style=1 -> matches content -> shape
#   item 2: pred=0, content=1, style=0 -> matches style   -> texture
#   item 3: pred=2, content=1, style=0 -> matches neither -> other
preds_by_model = {"toy_model": [0, 0, 0, 2]}
df = shape_bias_report(preds_by_model, manifest)
row = df.iloc[0]
check("n_shape counted correctly", row["n_shape"] == 2)
check("n_texture counted correctly", row["n_texture"] == 1)
check("n_other counted correctly", row["n_other"] == 1)
check("shape_bias_pct = 100 * shape/(shape+texture)",
      np.isclose(row["shape_bias_pct"], 100.0 * 2 / 3))
check("coverage_pct = 100 * (shape+texture)/total",
      np.isclose(row["coverage_pct"], 75.0))
check("classify_cue_conflict_prediction: shape", classify_cue_conflict_prediction(0, 0, 1) == "shape")
check("classify_cue_conflict_prediction: texture", classify_cue_conflict_prediction(1, 0, 1) == "texture")
check("classify_cue_conflict_prediction: other", classify_cue_conflict_prediction(2, 0, 1) == "other")

# --------------------------------------------------------------------------- #
print("\nevaluate_bias: color bias deltas")
labels4 = np.array([0, 1, 0, 1])
clean_logits = {"m": np.array([[3, 0], [0, 3], [3, 0], [0, 3]], dtype=float)}
gray_logits = {"m": {"grayscale": np.array([[3, 0], [3, 0], [3, 0], [0, 3]], dtype=float)}}
df = color_bias_report(clean_logits, gray_logits, labels4)
row = df.iloc[0]
check("clean accuracy computed correctly", np.isclose(row["clean_accuracy"], 1.0))
check("transformed accuracy computed correctly (1 flipped)", np.isclose(row["transformed_accuracy"], 0.75))
check("accuracy_change = transformed - clean", np.isclose(row["accuracy_change"], -0.25))
check("prediction_consistency reflects the 1 flip", np.isclose(row["prediction_consistency"], 0.75))

# --------------------------------------------------------------------------- #
print("\nevaluate_bias: translation curve aggregation")
labels5 = np.array([0, 1, 0, 1])
clean = np.array([[3, 0], [0, 3], [3, 0], [0, 3]], dtype=float)
by_px = {
    0: {"up": clean},
    8: {"up": clean, "down": clean, "left": np.array([[0, 3], [0, 3], [3, 0], [0, 3]], dtype=float),
        "right": clean},
}
df = translation_curve_report(by_px, clean, labels5)
row8 = df[df.pixels == 8].iloc[0]
check("translation accuracy averages across directions",
      np.isclose(row8["accuracy"], (1.0 + 1.0 + 0.75 + 1.0) / 4))
check("translation consistency averages across directions",
      np.isclose(row8["consistency"], (1.0 + 1.0 + 0.75 + 1.0) / 4))

# --------------------------------------------------------------------------- #
print("\nevaluate_bias: patch-shuffle accuracy drop")
clean_by_model = {"m": clean}
shuffled_by_model = {"m": np.array([[3, 0], [0, 3], [0, 3], [0, 3]], dtype=float)}
df = patch_shuffle_report(clean_by_model, shuffled_by_model, labels5)
row = df.iloc[0]
check("accuracy_drop = clean_acc - shuffled_acc", np.isclose(row["accuracy_drop"], 0.25))

# --------------------------------------------------------------------------- #
print("\npatch-shuffle index math (dependency-free re-derivation of "
      "data/transforms.py::patch_shuffle, using numpy instead of torch)")


def numpy_patch_shuffle_roundtrip(img, grid, perm):
    c, h, w = img.shape
    ph, pw = h // grid, w // grid
    # decompose into patches exactly like transforms.py's unfold/unfold/view
    patches = img.reshape(c, grid, ph, grid, pw).transpose(0, 1, 3, 2, 4)  # [c,gr,gc,ph,pw]
    patches = patches.reshape(c, grid * grid, ph, pw)
    shuffled = patches[:, perm, :, :]
    shuffled = shuffled.reshape(c, grid, grid, ph, pw)
    shuffled = shuffled.transpose(0, 1, 3, 2, 4).reshape(c, h, w)
    return shuffled


img = np.arange(3 * 8 * 8, dtype=float).reshape(3, 8, 8)
identity_perm = np.arange(16)
recon = numpy_patch_shuffle_roundtrip(img, grid=4, perm=identity_perm)
check("identity permutation exactly reconstructs the image", np.allclose(recon, img))

perm = np.array([1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])  # swap patch 0 and 1
shuffled = numpy_patch_shuffle_roundtrip(img, grid=4, perm=perm)
# patch (0,0) [top-left 2x2 block] should now contain what was patch (0,1)
ph, pw = 2, 2
original_patch1 = img[:, 0:ph, pw:2 * pw]
check("non-identity permutation relocates the correct patch",
      np.allclose(shuffled[:, 0:ph, 0:pw], original_patch1))
check("shuffled image is pixel-permutation of original (same multiset of values)",
      np.allclose(np.sort(shuffled.ravel()), np.sort(img.ravel())))

# --------------------------------------------------------------------------- #
print(f"\n{PASS} passed, {FAIL} failed")
if FAIL:
    sys.exit(1)
