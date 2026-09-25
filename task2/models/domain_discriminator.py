"""
Gradient-reversal layer and domain discriminator shared by DANN and CDAN.

DANN discriminator input: f(x)  -- the 512-d backbone feature.
CDAN discriminator input: g(x) = vec(f(x) (x) p(x))  -- outer product of the
    512-d feature and the num_classes-d softmax probability vector,
    flattened. The discriminator architecture itself (hidden width,
    activation, dropout, output layer) and the gradient-reversal schedule
    are identical between DANN and CDAN, per the assignment ("Feed g(x) to
    a discriminator with the same hidden width, activation, dropout,
    gradient-reversal schedule, and loss weight used for DANN").
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.alpha * grad_output, None


def gradient_reversal(x: torch.Tensor, alpha: float) -> torch.Tensor:
    return GradientReversalFunction.apply(x, alpha)


def grad_reversal_alpha(progress: float, max_strength: float = 1.0) -> float:
    """
    alpha(p) = 2 / (1 + exp(-10p)) - 1,  p in [0, 1] = training progress.

    `max_strength` rescales the fully-ramped-up value (used by the
    controlled design study that varies "maximum gradient-reversal
    strength" over {0.25, 0.5, 1} while keeping the same S-shaped schedule
    shape, per the assignment's Task 2 Step 6 options).
    """
    import math
    base = 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0
    return max_strength * base


class DomainDiscriminator(nn.Module):
    """256-unit hidden layer, ReLU, dropout 0.5, 2-class output -- used
    verbatim by both DANN (input dim = feature_dim) and CDAN (input dim =
    feature_dim * num_classes, from the flattened outer product)."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def cdan_conditioning(features: torch.Tensor, probs: torch.Tensor,
                       detach_features: bool = False, detach_probs: bool = False) -> torch.Tensor:
    """
    g(x) = vec(f (x) p): outer product of the feature vector and the
    class-probability vector, flattened to a single vector per example.

    Per the assignment: "Do not use entropy conditioning or detach f or p
    in the required implementation" -- both detach flags default to False
    and the main comparison must be run with them False; they exist only
    so the (optional/off-by-default) alternative can be tried if desired.
    """
    if detach_features:
        features = features.detach()
    if detach_probs:
        probs = probs.detach()
    b, d = features.shape
    c = probs.shape[1]
    outer = torch.bmm(features.unsqueeze(2), probs.unsqueeze(1))  # [B, d, c]
    return outer.view(b, d * c)
