"""
DAN-DG: pairwise source-domain MMD alignment, an adaptation of Task 2's DAN
that NEVER accesses the target (Sketch) domain -- it aligns every unordered
pair of the three observed source domains instead:

    L_DAN-DG = L_ERM + (lambda_dg / 3) * sum_{e < e'} MMD^2(F(X_e), F(X_e'))

Uses shared/mmd.py verbatim (same multi-kernel RBF construction, same
median-heuristic bandwidth, same {0.5, 1, 2} multipliers as Task 2's DAN),
per the assignment's instruction to reuse the identical discrepancy
mechanism so the role of target access can later be examined without
changing the measure itself.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "shared"))
from mmd import mmd2  # noqa: E402


def train_step(backbone, head, images, labels, domain_ids, cfg, optimizer,
                lambda_dg_override=None) -> dict:
    optimizer.zero_grad()
    feats = backbone(images)
    logits = head(feats)
    cls_loss = F.cross_entropy(logits, labels)

    lambda_dg = (lambda_dg_override if lambda_dg_override is not None
                 else cfg["mmd_dg"]["lambda_dg_main"])
    mults = cfg["mmd_dg"]["bandwidth_multipliers"]

    # three unordered pairs among domain ids {0, 1, 2} = {photo, art_painting, cartoon}
    pairs = [(0, 1), (0, 2), (1, 2)]
    mmd_terms = []
    for a, b in pairs:
        feats_a = feats[domain_ids == a]
        feats_b = feats[domain_ids == b]
        if feats_a.shape[0] < 2 or feats_b.shape[0] < 2:
            # degenerate batch (shouldn't happen with domain-balanced sampling,
            # but guard against it rather than crash mid-training)
            continue
        mmd_terms.append(mmd2(feats_a, feats_b, bandwidth_multipliers=mults))
    mmd_term = torch.stack(mmd_terms).mean() if mmd_terms else torch.zeros(
        (), device=images.device)

    loss = cls_loss + lambda_dg * mmd_term
    loss.backward()
    optimizer.step()
    return {
        "cls_loss": cls_loss.item(),
        "mmd_dg_term": float(mmd_term.item()) if torch.is_tensor(mmd_term) else float(mmd_term),
        "lambda_dg": float(lambda_dg),
        "total_loss": loss.item(),
    }
