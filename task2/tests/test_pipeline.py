"""
Smoke tests for Task 2's core math, using only synthetic tensors -- no
GPU, no real PACS data. Run this FIRST on Kaggle (takes seconds) before
spending GPU time on the real training runs.

Run: python tests/test_pipeline.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "shared"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "evaluation"))

from mmd import mmd2, median_heuristic_bandwidth  # noqa: E402
from domain_discriminator import (  # noqa: E402
    DomainDiscriminator, cdan_conditioning, gradient_reversal, grad_reversal_alpha,
)
from backbone import ResNet18Backbone, set_train_with_frozen_bn  # noqa: E402
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


# --------------------------------------------------------------------------- #
print("MMD")
# use a larger sample size than the other toy tests specifically here --
# the off-diagonal-mean MMD^2 estimator is a difference of near-cancelling
# terms and has real finite-sample variance that can occasionally dip
# slightly negative or bounce above a small fixed threshold at n=64; n=256
# keeps that noise well under control without changing what's being tested.
x_same_a = torch.randn(256, 16)
x_same_b = x_same_a + torch.randn(256, 16) * 1e-3  # near-identical distribution
mmd_same = mmd2(x_same_a, x_same_b).item()

x_diff = torch.randn(256, 16) * 5 + 10  # very different distribution (shifted+scaled)
mmd_diff = mmd2(x_same_a, x_diff).item()

# the property that actually matters is RELATIVE: near-identical distributions
# score far lower than clearly different ones -- an absolute cutoff on
# mmd_same alone is sensitive to sample size/dimensionality/RNG and isn't
# the real claim being tested.
check("MMD between near-identical distributions is much smaller than between different ones",
      mmd_same < 0.1 * mmd_diff)
check("MMD between clearly different distributions is larger than near-identical case",
      mmd_diff > mmd_same)

mmd_symmetric_a = mmd2(x_same_a, x_diff).item()
mmd_symmetric_b = mmd2(x_diff, x_same_a).item()
check("MMD is symmetric", abs(mmd_symmetric_a - mmd_symmetric_b) < 1e-5)

bw = median_heuristic_bandwidth(torch.cat([x_same_a, x_diff])).item()
check("median heuristic bandwidth is positive", bw > 0)

# --------------------------------------------------------------------------- #
print("\nGradient reversal")
alpha0 = grad_reversal_alpha(0.0)
alpha1 = grad_reversal_alpha(1.0)
alpha_half = grad_reversal_alpha(0.5)
check("alpha(0) == 0", abs(alpha0 - 0.0) < 1e-6)
check("alpha(1) close to 1 (fully ramped up)", alpha1 > 0.999)
check("alpha is monotonically increasing", alpha0 < alpha_half < alpha1)

alpha_scaled = grad_reversal_alpha(1.0, max_strength=0.25)
check("max_strength scales the fully-ramped-up value", abs(alpha_scaled - 0.25 * alpha1) < 1e-6)

x = torch.randn(8, 4, requires_grad=True)
y = gradient_reversal(x, alpha=1.0)
check("gradient_reversal forward pass is identity", torch.allclose(x.detach(), y.detach()))
loss = y.sum()
loss.backward()
check("gradient_reversal negates the gradient (alpha=1)",
      torch.allclose(x.grad, -torch.ones_like(x)))

x2 = torch.randn(8, 4, requires_grad=True)
y2 = gradient_reversal(x2, alpha=0.5)
y2.sum().backward()
check("gradient_reversal scales the negated gradient by alpha",
      torch.allclose(x2.grad, -0.5 * torch.ones_like(x2)))

# --------------------------------------------------------------------------- #
print("\nCDAN conditioning")
feats = torch.randn(5, 8)
probs = torch.softmax(torch.randn(5, 3), dim=1)
g = cdan_conditioning(feats, probs)
check("cdan_conditioning output shape is [B, feature_dim * num_classes]",
      g.shape == (5, 8 * 3))
# spot-check one example's outer product manually
manual = torch.outer(feats[0], probs[0]).flatten()
check("cdan_conditioning matches a manual outer product for one example",
      torch.allclose(g[0], manual, atol=1e-5))

g_detached = cdan_conditioning(feats, probs, detach_features=True)
check("detach_features actually detaches", not g_detached.requires_grad or feats.requires_grad is False)

# --------------------------------------------------------------------------- #
print("\nDomain discriminator shapes")
disc_dann = DomainDiscriminator(input_dim=512)
out = disc_dann(torch.randn(10, 512))
check("DANN discriminator outputs 2 logits per example", out.shape == (10, 2))

disc_cdan = DomainDiscriminator(input_dim=512 * 7)
out2 = disc_cdan(torch.randn(10, 512 * 7))
check("CDAN discriminator (bigger input) outputs 2 logits per example", out2.shape == (10, 2))

# --------------------------------------------------------------------------- #
print("\nBatchNorm freeze policy")
bb = ResNet18Backbone()
set_train_with_frozen_bn(bb)
bn_modules = [m for m in bb.modules() if isinstance(m, torch.nn.BatchNorm2d)]
check("at least one BatchNorm2d module exists in ResNet-18", len(bn_modules) > 0)
check("every BatchNorm2d module is in eval mode after set_train_with_frozen_bn",
      all(not m.training for m in bn_modules))
non_bn_training = [m for m in bb.modules() if not isinstance(m, torch.nn.BatchNorm2d)
                   and len(list(m.children())) == 0]
check("non-BatchNorm leaf modules are still in train mode",
      any(m.training for m in non_bn_training if hasattr(m, "training")))

running_mean_before = bn_modules[0].running_mean.clone()
bb.train()  # sanity: WITHOUT the freeze helper, BN stats WOULD update
set_train_with_frozen_bn(bb)
with torch.no_grad():
    _ = bb(torch.randn(4, 3, 224, 224))
check("running_mean does not change with BN frozen",
      torch.allclose(running_mean_before, bn_modules[0].running_mean))

# --------------------------------------------------------------------------- #
print("\nPer-class analysis (class_analysis.py)")
# only 3 classes actually appear in this toy example -- using the full
# 7-name list here would leave guitar/horse/house/person with zero examples,
# so per_class_accuracy correctly reports NaN for them (documented behavior,
# same convention as Task 1's class-balanced-subset handling), and NaN rows
# always sort last regardless of ascending/descending, which made the
# iloc[-1] check below compare against NaN instead of the real worst class.
# Restricting class_names to the classes actually present avoids that
# entirely without changing what per_class_comparison itself does.
class_names = ["dog", "elephant", "giraffe"]
labels = np.array([0, 0, 1, 1, 2, 2, 2])
baseline_preds = np.array([0, 1, 1, 1, 0, 2, 2])  # class 0: 1/2 right, class1: 2/2, class2: 2/3
method_preds = np.array([0, 0, 1, 0, 2, 2, 2])    # class 0: 2/2, class1: 1/2, class2: 3/3

acc = per_class_accuracy(baseline_preds, labels, class_names)
check("per_class_accuracy computes class 'dog' correctly", abs(acc["dog"] - 0.5) < 1e-9)
check("per_class_accuracy computes class 'elephant' correctly", abs(acc["elephant"] - 1.0) < 1e-9)

cmp = per_class_comparison(method_preds, labels, baseline_preds, labels, class_names, "Method")
check("per_class_comparison flags dog as improved", cmp.loc["dog", "change"] > 0)
check("per_class_comparison flags elephant as degraded", cmp.loc["elephant", "change"] < 0)
check("per_class_comparison sorted with largest improvement first",
      cmp["change"].iloc[0] >= cmp["change"].iloc[-1])

conf = top_confusions(baseline_preds, labels, class_names, for_class="dog")
check("top_confusions finds the one wrong dog prediction (predicted elephant)",
      len(conf) == 1 and conf.iloc[0]["predicted_class"] == "elephant")

# --------------------------------------------------------------------------- #
print(f"\n{PASS} passed, {FAIL} failed")
if FAIL:
    sys.exit(1)
