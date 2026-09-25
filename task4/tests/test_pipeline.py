"""
Synthetic-tensor smoke test for Task 4 -- run this FIRST on Kaggle, before
downloading CIFAR-10/100 or spending any real GPU training time. No real data
needed; every check below uses small random tensors matching the shapes
real data would produce.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "methods"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scores"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "evaluation"))

from resnet_cifar import ResNet18Cifar  # noqa: E402
from classifier_head import ClassifierHead  # noqa: E402
import manifold_mixup  # noqa: E402
import proser  # noqa: E402
import msp, mls, energy, mahalanobis  # noqa: E402
import metrics, thresholds, failure_analysis  # noqa: E402

passed = 0
failed = 0


def check(name: str, cond: bool):
    global passed, failed
    if cond:
        print(f"  [ok] {name}")
        passed += 1
    else:
        print(f"  [FAIL] {name}")
        failed += 1


print("Backbone: CIFAR stem + forward_pre/forward_post consistency")
backbone = ResNet18Cifar()
x = torch.randn(4, 3, 32, 32)
with torch.no_grad():
    feat_direct = backbone(x)
    feat_split = backbone.forward_post(backbone.forward_pre(x))
check("output feature dim is 512", feat_direct.shape == (4, 512))
check("forward() matches forward_post(forward_pre())",
      torch.allclose(feat_direct, feat_split, atol=1e-5))
h = backbone.forward_pre(x)
check("post-layer2 feature map has 128 channels at 16x16 (32 -> layer1(s1) -> layer2(s2))",
      tuple(h.shape[1:]) == (128, 16, 16))

print("\nClassifierHead: known/dummy logit slicing")
head = ClassifierHead(512, num_classes=10, num_dummy=5)
feats = torch.randn(4, 512)
logits = head(feats)
check("logits shape is [B, 15]", logits.shape == (4, 15))
check("known_logits() returns first 10 columns",
      torch.equal(head.known_logits(logits), logits[:, :10]))
check("dummy_logits() returns last 5 columns",
      torch.equal(head.dummy_logits(logits), logits[:, 10:]))

print("\nManifold mixup: different-class pairing")
labels = torch.tensor([0, 0, 1, 1, 2, 3, 0, 1])
gen = torch.Generator().manual_seed(6304)
perm = manifold_mixup.pair_different_class_indices(labels, gen)
check("no example is paired with a same-class partner",
      bool(torch.all(labels[perm] != labels)))
check("every entry of perm is a valid index into the batch",
      bool(torch.all((perm >= 0) & (perm < len(labels)))))

print("\nPROSER: placeholder losses and detection score")
K, D = 10, 5
all_logits = torch.randn(6, K + D)
y = torch.randint(0, K, (6,))
cls_loss = proser.classifier_placeholder_loss(all_logits, y, K, beta=1.0)
check("classifier_placeholder_loss is finite", torch.isfinite(cls_loss).item())
mixed_logits = torch.randn(6, K + D)
data_loss = proser.data_placeholder_loss(mixed_logits, K)
check("data_placeholder_loss is finite", torch.isfinite(data_loss).item())
ph_score = proser.placeholder_detection_score(all_logits.numpy(), K)
check("placeholder_detection_score shape matches batch size", ph_score.shape == (6,))
check("placeholder_detection_score returns a numpy array (the real caller, "
      "evaluate_osr.py, always passes numpy read back from .npz cache -- "
      "this is the exact type mismatch that broke evaluate_osr.py before this fix)",
      isinstance(ph_score, np.ndarray))

# targeted check: making the true class logit enormous should drive L2 down
# (since after masking y, dummy still has to beat only the OTHER K-1 classes,
# but L1 should be trivially near-zero and total loss should be small overall)
easy_logits = torch.full((1, K + D), -10.0)
easy_logits[0, 3] = 10.0  # true class 3 dominates everything
easy_y = torch.tensor([3])
easy_loss = proser.classifier_placeholder_loss(easy_logits, easy_y, K, beta=1.0)
hard_logits = torch.full((1, K + D), 10.0)  # every known class rivals the true class
hard_logits[0, K:] = -10.0  # dummies are weak
hard_loss = proser.classifier_placeholder_loss(hard_logits, easy_y, K, beta=1.0)
check("classifier-placeholder loss is lower when the true class clearly dominates",
      easy_loss.item() < hard_loss.item())

print("\nPost-hoc scores: MSP, MLS, Energy basic sanity")
confident_logits = np.array([[10.0] + [0.0] * 9])
uniform_logits = np.array([[0.0] * 10])
check("MSP: confident prediction has lower u(x) than uniform",
      msp.score(confident_logits)[0] < msp.score(uniform_logits)[0])
check("MLS: higher max logit gives lower (more negative) u(x)",
      mls.score(confident_logits)[0] < mls.score(uniform_logits)[0])
check("Energy: higher logits overall give lower u(x)",
      energy.score(confident_logits)[0] < energy.score(uniform_logits)[0])

print("\nMahalanobis: fit + score")
rng = np.random.default_rng(6304)
feats_c0 = rng.normal(loc=0.0, scale=0.1, size=(50, 4))
feats_c1 = rng.normal(loc=5.0, scale=0.1, size=(50, 4))
train_feats = np.concatenate([feats_c0, feats_c1])
train_labels = np.array([0] * 50 + [1] * 50)
fitted = mahalanobis.fit(train_feats, train_labels, diag_eps=1e-6)
near_c0 = np.array([[0.05, -0.02, 0.01, 0.0]])
far_point = np.array([[50.0, 50.0, 50.0, 50.0]])
s_near = mahalanobis.score(near_c0, fitted)[0]
s_far = mahalanobis.score(far_point, fitted)[0]
check("a point near a class mean has low Mahalanobis distance", s_near < 5.0)
check("a point far from every class mean has much higher Mahalanobis distance",
      s_far > s_near * 100)

print("\nEvaluation: AUROC and calibrated threshold")
known = rng.normal(0, 1, 200)
unknown_perfect = known.max() + 1 + rng.normal(0, 0.01, 200)
unknown_random = rng.normal(0, 1, 200)
check("AUROC is ~1.0 for perfectly separated scores",
      metrics.auroc_known_vs_unknown(known, unknown_perfect) > 0.99)
check("AUROC is ~0.5 for identically-distributed scores",
      abs(metrics.auroc_known_vs_unknown(known, unknown_random) - 0.5) < 0.15)

val_known = rng.normal(0, 1, 1000)
tau = thresholds.calibrate_threshold(val_known, percentile=95.0)
check("threshold accepts ~95% of the calibration set itself",
      abs(thresholds.acceptance_rate(val_known, tau) - 0.95) < 0.02)

report = thresholds.calibrated_report(val_known, known, unknown_perfect, unknown_random, 95.0)
check("near(=unknown_perfect)/perfectly-separated-unknown acceptance rate is ~0 "
      "(correctly rejected)", report["fpr_at_95tpr_near"] < 0.02)
check("far(=unknown_random)/same-distribution-as-known acceptance rate is close to "
      "95% (statistically indistinguishable from known, so the threshold can't "
      "reject it any better than it accepts known examples)",
      abs(report["fpr_at_95tpr_far"] - 0.95) < 0.1)

print("\nFailure analysis: incorrectly-accepted filtering")
fake_logits = np.tile(np.array([5.0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), (4, 1))
fake_scores = np.array([-6.0, -1.0, 10.0, 20.0])  # first two below a tau of 0.0
fake_names = np.array(["wolf", "fox", "chair", "clock"])
rows = failure_analysis.incorrectly_accepted(fake_logits, fake_scores, fake_names,
                                              threshold=0.0, num_classes=10, top_n=10)
check("only examples with score <= threshold are returned", len(rows) == 2)
check("returned rows are sorted by score ascending",
      list(rows["score"]) == sorted(rows["score"].tolist()))
check("predicted_class is correctly read off the known-logit argmax",
      set(rows["predicted_class"]) == {"airplane"})

print(f"\n{passed} passed, {failed} failed")
