"""
Manifold mixup for PROSER's data placeholders.

"For two examples from different classes, let h_i = phi_pre(x_i), h_j =
phi_pre(x_j), where phi_pre denotes the network up to layer2. Sample
lambda ~ Beta(2,2) and construct h~ = lambda*h_i + (1-lambda)*h_j, y_i != y_j."

This module only implements the mixing mechanics (pairing different-class
examples within a batch and interpolating their post-layer2 feature maps);
what the mixed feature is trained toward (the dummy classifiers) lives in
proser.py, since that is PROSER-specific, not a generic mixup utility.
"""
from __future__ import annotations

from typing import Tuple

import torch


def sample_lambda(alpha: float, beta: float, device) -> float:
    dist = torch.distributions.Beta(torch.tensor(float(alpha)), torch.tensor(float(beta)))
    return float(dist.sample().to(device))


def pair_different_class_indices(labels: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """
    Returns an index array `perm` (values in [0, B), NOT guaranteed to be a
    strict one-to-one permutation once collision resolution kicks in -- some
    source images may be reused as the mixup partner for more than one
    example, which is harmless for mixup) such that labels[perm[i]] !=
    labels[i] for every i.

    Implementation: start from a random permutation; resample indices where
    the partner's label collides with the original (bounded retries, then
    fall back to an explicit linear search for any residual collisions) --
    CIFAR-10 batches of 128 over 10 classes make same-class collisions rare,
    so this converges in a handful of iterations in practice, but the
    explicit fallback guarantees zero silent same-class pairs regardless.
    """
    B = labels.shape[0]
    idx = torch.arange(B, device=labels.device)
    perm = torch.randperm(B, generator=generator, device=labels.device)

    for _ in range(10):
        collide = labels[perm] == labels[idx]
        if not collide.any():
            break
        n_collide = int(collide.sum().item())
        resample = torch.randperm(B, generator=generator, device=labels.device)[:n_collide]
        perm[collide] = resample

    # explicit fallback for any residual collisions (extremely rare): swap with
    # the next index that has a different label
    collide = labels[perm] == labels[idx]
    if collide.any():
        collide_positions = torch.nonzero(collide, as_tuple=True)[0].tolist()
        for i in collide_positions:
            for j in range(B):
                if labels[j].item() != labels[i].item():
                    perm[i] = j
                    break
    return perm


def manifold_mixup_pairs(
    images: torch.Tensor, labels: torch.Tensor, alpha: float, beta: float,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Returns (images_i, images_j, labels_j_for_reference, lam) where images_i is
    the original batch order and images_j is a different-class-paired
    permutation of it, ready for the caller to run phi_pre on both and mix.
    labels_j_for_reference is returned only for bookkeeping/asserts, since the
    mixed example's target is neither y_i nor y_j (it targets a dummy
    classifier -- see proser.py).
    """
    perm = pair_different_class_indices(labels, generator)
    assert torch.all(labels[perm] != labels), "manifold mixup paired same-class examples"
    lam = sample_lambda(alpha, beta, images.device)
    return images, images[perm], labels[perm], lam
