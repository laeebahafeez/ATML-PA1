"""
Multi-kernel Maximum Mean Discrepancy (MMD), shared verbatim between
Task 2's DAN (source vs. unlabeled target) and Task 3's DAN-DG (pairwise
among observed source domains), per the assignment's instruction to
"Use the same MMD implementation and kernel construction as Task 2 so
that the role of target access can later be examined without changing
the discrepancy measure."

Kernel: k(x, y) = exp(-||x - y||^2 / bandwidth), summed over three
bandwidths equal to {0.5, 1, 2} times the median pairwise squared
Euclidean distance among all features in the current combined batch (the
standard "median heuristic," computed fresh per batch, per the
assignment's exact wording).

MMD^2(X, Y) = E[k(x, x')] - 2 E[k(x, y)] + E[k(y, y')]
"""
from __future__ import annotations

from typing import Sequence

import torch


def _pairwise_sq_dists(x: torch.Tensor) -> torch.Tensor:
    sq_norms = (x ** 2).sum(dim=1)
    d2 = sq_norms.unsqueeze(1) + sq_norms.unsqueeze(0) - 2.0 * (x @ x.t())
    return d2.clamp(min=0.0)


def median_heuristic_bandwidth(combined: torch.Tensor) -> torch.Tensor:
    """Median of the pairwise squared distances among all rows of
    `combined`, excluding the zero self-distances on the diagonal."""
    with torch.no_grad():
        d2 = _pairwise_sq_dists(combined)
        n = d2.shape[0]
        off_diag_mask = ~torch.eye(n, dtype=torch.bool, device=d2.device)
        med = d2[off_diag_mask].median()
    return med.clamp(min=1e-8)


def multi_kernel_gram(combined: torch.Tensor, bandwidth_multipliers: Sequence[float]) -> torch.Tensor:
    """Sum of RBF kernels (one per multiplier) over the full combined
    [n+m, n+m] Gram matrix, using the fresh-per-batch median heuristic."""
    base_bw = median_heuristic_bandwidth(combined)
    d2 = _pairwise_sq_dists(combined)
    gram = torch.zeros_like(d2)
    for mult in bandwidth_multipliers:
        bw = mult * base_bw
        gram = gram + torch.exp(-d2 / bw)
    return gram


def mmd2(x: torch.Tensor, y: torch.Tensor,
          bandwidth_multipliers: Sequence[float] = (0.5, 1.0, 2.0)) -> torch.Tensor:
    """
    Squared MMD between two sets of feature vectors x [n, d] and y [m, d],
    using the kernel trick (never constructing an explicit feature map).
    """
    n, m = x.shape[0], y.shape[0]
    combined = torch.cat([x, y], dim=0)
    gram = multi_kernel_gram(combined, bandwidth_multipliers)

    k_xx = gram[:n, :n]
    k_yy = gram[n:, n:]
    k_xy = gram[:n, n:]

    # exclude self-similarity terms (diagonal of k_xx, k_yy) for an
    # unbiased-ish estimate; with n, m typically >= 8 this has negligible
    # practical effect but is the more standard convention.
    def off_diag_mean(k: torch.Tensor) -> torch.Tensor:
        size = k.shape[0]
        if size <= 1:
            return k.mean()
        mask = ~torch.eye(size, dtype=torch.bool, device=k.device)
        return k[mask].mean()

    return off_diag_mean(k_xx) - 2.0 * k_xy.mean() + off_diag_mean(k_yy)
