"""
Standard, non-adaptive Sharpness-Aware Minimization (Foret et al. 2021):

    min_theta max_{||eps||_2 <= rho} L_ERM(theta + eps)

Per the assignment: "For each source batch, SAM first finds a normalized
ascent perturbation and then updates the original parameters using the loss
at the perturbed point, requiring two forward/backward passes." This is why
SAM needs the train_step(..., optimizer) interface (see methods/__init__.py)
instead of Task 2's simpler compute-loss-and-let-train.py-call-backward
pattern: no single scalar loss captures a two-pass update.

Sequence per step:
  1. Forward/backward at theta -> gradient g.
  2. eps = rho * g / ||g||_2 (global L2 norm over ALL trainable backbone+head
     parameters, flattened -- not per-parameter).
  3. theta <- theta + eps (in place).
  4. Forward/backward AGAIN at theta+eps, using the SAME batch -> gradient g'.
  5. theta <- theta - eps (restore the original, unperturbed parameters).
  6. optimizer.step() applies g' (the perturbed-point gradient) to the now-
     restored original theta -- this is the actual SAM update.

The shared frozen-BatchNorm policy (models/backbone.py::set_train_with_frozen_bn)
must be applied before calling this in every epoch, exactly as for ERM and
DAN-DG -- BatchNorm modules stay in eval() for both forward passes here
since nothing in this function calls .train() again mid-step.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def train_step(backbone, head, images, labels, domain_ids, cfg, optimizer,
                rho_override=None) -> dict:
    rho = rho_override if rho_override is not None else cfg["sam"]["rho_main"]
    params = [p for p in list(backbone.parameters()) + list(head.parameters())
              if p.requires_grad]

    # ---- pass 1: gradient at the current point theta ----
    optimizer.zero_grad()
    logits1 = head(backbone(images))
    loss1 = F.cross_entropy(logits1, labels)
    loss1.backward()

    grads = [p.grad.detach().clone() if p.grad is not None else None for p in params]
    grad_norm = torch.norm(
        torch.stack([g.norm(2) for g in grads if g is not None]), 2
    )
    scale = rho / (grad_norm + 1e-12)

    # ---- ascent step: theta <- theta + eps ----
    epsilons = []
    with torch.no_grad():
        for p, g in zip(params, grads):
            if g is None:
                epsilons.append(None)
                continue
            eps = g * scale
            p.add_(eps)
            epsilons.append(eps)

    # ---- pass 2: gradient at the perturbed point theta + eps, same batch ----
    optimizer.zero_grad()
    logits2 = head(backbone(images))
    loss2 = F.cross_entropy(logits2, labels)
    loss2.backward()

    # ---- restore: theta <- theta - eps (back to the original parameters) ----
    with torch.no_grad():
        for p, eps in zip(params, epsilons):
            if eps is not None:
                p.sub_(eps)

    # ---- apply the optimizer update using the PERTURBED-point gradient
    #      (currently sitting in p.grad from pass 2) to the restored theta ----
    optimizer.step()

    return {
        "cls_loss": loss1.item(),
        "cls_loss_at_perturbed_point": loss2.item(),
        "grad_norm": grad_norm.item(),
        "rho": float(rho),
        "total_loss": loss1.item(),
    }
