"""
DANN -- adversarial domain alignment via gradient reversal.

    L = L_cls(source only) + L_domain(source + target)

The domain discriminator sees a gradient-reversal layer applied to the
512-d feature: forward pass is identity, backward pass negates (and
scales by alpha(p)) the gradient flowing back into the backbone, so the
backbone is pushed toward features the discriminator cannot tell apart by
domain, while the discriminator itself is still trained normally (via its
own un-reversed forward path for its own parameter updates -- achieved
simply by the reversal happening between the backbone and the
discriminator, so the discriminator's own gradients w.r.t. its own
weights are unaffected).
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
from domain_discriminator import gradient_reversal, grad_reversal_alpha  # noqa: E402


def compute_loss(backbone, head, images, labels, domain_ids, n_source,
                  progress, cfg, discriminator=None) -> Tuple[torch.Tensor, Dict]:
    assert discriminator is not None, "DANN requires a domain discriminator"
    n_source_domains = len(cfg["dataset"]["source_domains"])

    feats = backbone(images)
    logits = head(feats[:n_source])
    cls_loss = F.cross_entropy(logits, labels)

    alpha = grad_reversal_alpha(progress, cfg["dann"]["max_grad_reversal_strength_main"])
    reversed_feats = gradient_reversal(feats, alpha)
    domain_logits = discriminator(reversed_feats)
    # binary domain label: 0 for any source domain, 1 for target
    domain_labels = (domain_ids >= n_source_domains).long()
    domain_loss = F.cross_entropy(domain_logits, domain_labels)

    loss_weight = cfg["dann"]["loss_weight"]
    total = cls_loss + loss_weight * domain_loss

    with torch.no_grad():
        domain_acc = (domain_logits.argmax(dim=1) == domain_labels).float().mean().item()

    return total, {
        "cls_loss": cls_loss.item(),
        "domain_loss": domain_loss.item(),
        "domain_acc": domain_acc,
        "alpha": alpha,
    }
