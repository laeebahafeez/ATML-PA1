"""
Smoke tests for Task 3's core math and mechanics, using only synthetic
tensors -- no GPU, no real PACS data, no Task 2 checkpoint required.
Run this FIRST on Kaggle before spending GPU time on real training.

Run: python tests/test_pipeline.py
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "shared"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "methods"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "evaluation"))

from mmd import mmd2  # noqa: E402
from backbone import ResNet18Backbone, set_train_with_frozen_bn  # noqa: E402
from classifier_head import ClassifierHead  # noqa: E402
import dan_dg  # noqa: E402
import sam  # noqa: E402
import erm  # noqa: E402
from sharpness import build_fixed_sharpness_batch, compute_sharpness  # noqa: E402
from source_domain_separability import source_domain_separability_score  # noqa: E402
from class_analysis import per_class_accuracy, per_class_comparison, top_confusions  # noqa: E402

torch.manual_seed(0)
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


TOY_CFG = {
    "mmd_dg": {"lambda_dg_main": 1.0, "bandwidth_multipliers": [0.5, 1.0, 2.0]},
    "sam": {"rho_main": 0.05},
}

# --------------------------------------------------------------------------- #
print("Backbone + BatchNorm freeze (shared design with Task 2)")
bb = ResNet18Backbone()
head = ClassifierHead(512, 7)
set_train_with_frozen_bn(bb)
bn_modules = [m for m in bb.modules() if isinstance(m, nn.BatchNorm2d)]
check("BatchNorm modules exist", len(bn_modules) > 0)
check("BatchNorm modules are frozen (eval) under set_train_with_frozen_bn",
      all(not m.training for m in bn_modules))

# --------------------------------------------------------------------------- #
print("\nDAN-DG: no-sketch domain_ids (only 0/1/2) and pairwise MMD")
images = torch.randn(24, 3, 32, 32)  # tiny fake images, not real 224x224 -- only shape/logic under test
labels = torch.randint(0, 7, (24,))
domain_ids = torch.cat([torch.full((8,), d) for d in range(3)])
check("domain_ids only ever take values in {0,1,2} (no target row=3)",
      set(domain_ids.tolist()) <= {0, 1, 2})

# a tiny fake backbone/head pair for fast CPU testing (not the real ResNet-18 --
# just needs to be a valid nn.Module producing [B, 512] features)
tiny_backbone = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 512))
tiny_head = nn.Linear(512, 7)
optimizer = torch.optim.AdamW(list(tiny_backbone.parameters()) + list(tiny_head.parameters()), lr=1e-3)
logs = dan_dg.train_step(tiny_backbone, tiny_head, images, labels, domain_ids, TOY_CFG, optimizer)
check("dan_dg.train_step returns finite cls_loss", np.isfinite(logs["cls_loss"]))
check("dan_dg.train_step returns finite mmd_dg_term", np.isfinite(logs["mmd_dg_term"]))
check("dan_dg.train_step's mmd term is non-negative (MMD^2 by construction)",
      logs["mmd_dg_term"] >= -1e-6)

# --------------------------------------------------------------------------- #
print("\nSAM: perturb-restore correctness")
tiny_backbone2 = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 512))
tiny_head2 = nn.Linear(512, 7)
params_before = [p.detach().clone() for p in
                  list(tiny_backbone2.parameters()) + list(tiny_head2.parameters())]
optimizer2 = torch.optim.SGD(
    list(tiny_backbone2.parameters()) + list(tiny_head2.parameters()), lr=0.01)
logs_sam = sam.train_step(tiny_backbone2, tiny_head2, images, labels, domain_ids, TOY_CFG, optimizer2)
params_after = list(tiny_backbone2.parameters()) + list(tiny_head2.parameters())
check("SAM step returns finite losses", np.isfinite(logs_sam["cls_loss"]) and
      np.isfinite(logs_sam["cls_loss_at_perturbed_point"]))
check("SAM step actually changed the parameters (optimizer.step ran)",
      any(not torch.allclose(a, b.detach()) for a, b in zip(params_before, params_after)))
check("SAM reports the requested rho", abs(logs_sam["rho"] - TOY_CFG["sam"]["rho_main"]) < 1e-9)

# rho override
logs_sam2 = sam.train_step(tiny_backbone2, tiny_head2, images, labels, domain_ids, TOY_CFG,
                             optimizer2, rho_override=0.2)
check("SAM rho_override is honored", abs(logs_sam2["rho"] - 0.2) < 1e-9)

# --------------------------------------------------------------------------- #
print("\nSharpness diagnostic: no side effects + qualitative direction")
tiny_backbone3 = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 512))
tiny_head3 = nn.Linear(512, 7)
params_snapshot = [p.detach().clone() for p in
                    list(tiny_backbone3.parameters()) + list(tiny_head3.parameters())]
result = compute_sharpness(tiny_backbone3, tiny_head3, images, labels,
                            torch.device("cpu"), rho=0.05)
params_after_sharp = list(tiny_backbone3.parameters()) + list(tiny_head3.parameters())
check("compute_sharpness restores parameters exactly (no side effects)",
      all(torch.allclose(a, b.detach(), atol=1e-6) for a, b in zip(params_snapshot, params_after_sharp)))
check("compute_sharpness reports a finite delta_sharp", np.isfinite(result["delta_sharp"]))
check("compute_sharpness's perturbation norm matches rho",
      abs(result["grad_norm"] * (0.05 / (result["grad_norm"] + 1e-12)) - 0.05) < 1e-6)

# --------------------------------------------------------------------------- #
print("\nFixed sharpness batch construction")


class _FakeDataset:
    def __init__(self, n, seed):
        rng = np.random.default_rng(seed)
        self.x = torch.tensor(rng.normal(size=(n, 3, 4, 4)), dtype=torch.float32)
        self.y = torch.tensor(rng.integers(0, 7, size=n), dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], int(self.y[i])


fake_val = {"photo": _FakeDataset(100, 1), "art_painting": _FakeDataset(100, 2),
            "cartoon": _FakeDataset(100, 3)}
imgs, lbls = build_fixed_sharpness_batch(fake_val, seed=6304, n_per_domain=32)
check("fixed sharpness batch has 32 examples per domain (96 total)", imgs.shape[0] == 96)
imgs2, lbls2 = build_fixed_sharpness_batch(fake_val, seed=6304, n_per_domain=32)
check("fixed sharpness batch is deterministic across calls (same seed)",
      torch.equal(imgs, imgs2) and torch.equal(lbls, lbls2))

# --------------------------------------------------------------------------- #
print("\nSource-domain separability: chance level with random labels")
rng = np.random.default_rng(6304)
feats_by_domain = {d: rng.normal(size=(200, 16)) for d in ["photo", "art_painting", "cartoon"]}
sep = source_domain_separability_score(feats_by_domain, seed=6304, test_fraction=0.3, C=1.0)
check("chance_level is 1/3 for 3 domains", abs(sep["chance_level"] - 1.0 / 3) < 1e-9)
check("held-out accuracy on RANDOM features is near chance (< 0.55)",
      sep["held_out_accuracy"] < 0.55)

# now make domains trivially separable (shifted means) -- should score much higher
feats_sep = {
    "photo": rng.normal(loc=0, size=(200, 16)),
    "art_painting": rng.normal(loc=20, size=(200, 16)),
    "cartoon": rng.normal(loc=-20, size=(200, 16)),
}
sep2 = source_domain_separability_score(feats_sep, seed=6304, test_fraction=0.3, C=1.0)
check("held-out accuracy on TRIVIALLY separable features is near 1.0",
      sep2["held_out_accuracy"] > 0.95)

# --------------------------------------------------------------------------- #
print("\nERM checkpoint loader error message")
try:
    erm.load_source_only_checkpoint("/nonexistent/path/source_only.pt")
    check("load_source_only_checkpoint raises on missing file", False)
except FileNotFoundError as e:
    check("load_source_only_checkpoint raises a clear FileNotFoundError",
          "Task 2" in str(e))

# --------------------------------------------------------------------------- #
print("\nPer-class analysis (reused from Task 2, same logic)")
class_names = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
gt = np.array([0, 0, 1, 1, 2, 2, 2])
baseline_preds = np.array([0, 1, 1, 1, 0, 2, 2])
method_preds = np.array([0, 0, 1, 0, 2, 2, 2])
acc = per_class_accuracy(baseline_preds, gt, class_names)
check("per_class_accuracy computes 'dog' correctly", abs(acc["dog"] - 0.5) < 1e-9)
cmp = per_class_comparison(method_preds, gt, baseline_preds, gt, class_names, "DAN-DG", "ERM")
check("per_class_comparison uses ERM as the default baseline label",
      "ERM_accuracy" in cmp.columns)
conf = top_confusions(baseline_preds, gt, class_names, "dog")
check("top_confusions finds the one wrong dog prediction", len(conf) == 1)

# --------------------------------------------------------------------------- #
print(f"\n{PASS} passed, {FAIL} failed")
if FAIL:
    sys.exit(1)
