"""
DAN -- MMD alignment between labeled source and unlabeled target features.

    L_DAN = L_cls + lambda_mmd * MMD^2(source_features, target_features)

Applied to the 512-d feature immediately before the classifier head, using
the shared multi-kernel MMD implementation (shared/mmd.py) so Task 3's
DAN-DG uses an identical discrepancy measure.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "shared"))
from mmd import mmd2  # noqa: E402


def compute_loss(backbone, head, images, labels, domain_ids, n_source,
                  progress, cfg, discriminator=None) -> Tuple[torch.Tensor, Dict]:
    feats = backbone(images)
    source_feats, target_feats = feats[:n_source], feats[n_source:]
    logits = head(source_feats)

    cls_loss = F.cross_entropy(logits, labels)
    discrepancy = mmd2(source_feats, target_feats,
                        bandwidth_multipliers=cfg["mmd"]["bandwidth_multipliers"])
    lambda_mmd = cfg["mmd"]["lambda_mmd_main"]
    total = cls_loss + lambda_mmd * discrepancy

    return total, {
        "cls_loss": cls_loss.item(),
        "mmd": discrepancy.item(),
        "lambda_mmd": lambda_mmd,
    }
