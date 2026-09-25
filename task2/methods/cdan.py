"""
CDAN -- class-conditional adversarial alignment.

Identical to DANN except the domain discriminator sees
g(x) = vec(f(x) (x) p(x)) -- the outer product of the 512-d feature and
the softmax class-probability vector -- instead of f(x) alone. Same
discriminator hidden width/activation/dropout, same gradient-reversal
schedule, same loss weight as DANN. Per the assignment: "Do not use
entropy conditioning or detach f or p in the required implementation."
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
from domain_discriminator import (  # noqa: E402
    cdan_conditioning, gradient_reversal, grad_reversal_alpha,
)


def compute_loss(backbone, head, images, labels, domain_ids, n_source,
                  progress, cfg, discriminator=None) -> Tuple[torch.Tensor, Dict]:
    assert discriminator is not None, "CDAN requires a domain discriminator"
    n_source_domains = len(cfg["dataset"]["source_domains"])
    cdan_cfg = cfg["cdan"]

    feats = backbone(images)
    logits_all = head(feats)
    cls_loss = F.cross_entropy(logits_all[:n_source], labels)

    probs_all = F.softmax(logits_all, dim=1)
    g = cdan_conditioning(
        feats, probs_all,
        detach_features=cdan_cfg["detach_features"],
        detach_probs=cdan_cfg["detach_probs"],
    )

    # CDAN's main comparison uses the same full-strength schedule as DANN's
    # main comparison (the controlled design study only ever sweeps DAN's
    # lambda_mmd or DANN's max strength, never CDAN's).
    alpha = grad_reversal_alpha(progress, max_strength=1.0)
    reversed_g = gradient_reversal(g, alpha)
    domain_logits = discriminator(reversed_g)
    domain_labels = (domain_ids >= n_source_domains).long()
    domain_loss = F.cross_entropy(domain_logits, domain_labels)

    total = cls_loss + cdan_cfg["loss_weight"] * domain_loss

    with torch.no_grad():
        domain_acc = (domain_logits.argmax(dim=1) == domain_labels).float().mean().item()

    return total, {
        "cls_loss": cls_loss.item(),
        "domain_loss": domain_loss.item(),
        "domain_acc": domain_acc,
        "alpha": alpha,
    }
