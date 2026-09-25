"""
Local sharpness diagnostic, per the assignment's exact formula:

    delta_sharp = L_val(theta + eps) - L_val(theta),
    eps = rho * grad_theta(L_val) / ||grad_theta(L_val)||_2

Applied to the SAME fixed 96-example batch (32 from each source domain,
sampled once with seed 6304) for ERM, DAN-DG, and SAM, so the three numbers
are directly comparable. This is a standardized LOCAL diagnostic, not proof
that one model's entire loss landscape is flatter -- the assignment is
explicit about this limitation.

Note this is intentionally the SAME perturbation mechanics as
methods/sam.py's ascent step (normalized gradient, radius rho), but here it
is a read-only measurement after training: the model's parameters are
perturbed and then immediately restored, and no optimizer step follows.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def build_fixed_sharpness_batch(
    source_val_datasets: Dict[str, object], seed: int = 6304, n_per_domain: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Deterministically samples n_per_domain examples from EACH source
    validation set (sorted domain order for reproducibility) and
    concatenates them into one fixed batch, reused for every model."""
    rng = np.random.default_rng(seed)
    images, labels = [], []
    for domain in sorted(source_val_datasets.keys()):
        ds = source_val_datasets[domain]
        n = len(ds)
        take = min(n_per_domain, n)
        idx = rng.choice(n, size=take, replace=False)
        for i in idx:
            x, y = ds[int(i)]
            images.append(x)
            labels.append(y)
    return torch.stack(images), torch.tensor(labels, dtype=torch.long)


def compute_sharpness(backbone, head, images: torch.Tensor, labels: torch.Tensor,
                       device, rho: float = 0.05) -> dict:
    """Places the model in EVAL mode (per the assignment), perturbs every
    trainable backbone+head parameter along the normalized loss gradient,
    measures the loss increase, then restores the original parameters
    exactly (no side effects on the passed-in model)."""
    backbone.eval()
    head.eval()
    images, labels = images.to(device), labels.to(device)

    params = [p for p in list(backbone.parameters()) + list(head.parameters())
              if p.requires_grad]

    logits = head(backbone(images))
    loss_clean = F.cross_entropy(logits, labels)
    grads = torch.autograd.grad(loss_clean, params, retain_graph=False, create_graph=False)

    flat_norm = torch.norm(torch.stack([g.norm(2) for g in grads]), 2)
    scale = rho / (flat_norm + 1e-12)
    epsilons = [g * scale for g in grads]

    with torch.no_grad():
        for p, eps in zip(params, epsilons):
            p.add_(eps)
        logits_perturbed = head(backbone(images))
        loss_perturbed = F.cross_entropy(logits_perturbed, labels)
        for p, eps in zip(params, epsilons):
            p.sub_(eps)  # restore exactly -- this function must have no side effects

    return {
        "loss_clean": float(loss_clean.item()),
        "loss_perturbed": float(loss_perturbed.item()),
        "delta_sharp": float(loss_perturbed.item() - loss_clean.item()),
        "grad_norm": float(flat_norm.item()),
        "rho": float(rho),
    }
