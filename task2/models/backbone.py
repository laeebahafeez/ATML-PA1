"""
ResNet-18 backbone for Task 2 (and reused unchanged by Task 3).

Unlike Task 1, this backbone is FULLY FINE-TUNED (not frozen) for every
method -- the assignment only freezes BatchNorm's running statistics, not
the convolutional weights.

BatchNorm policy (identical for every method in Tasks 2 and 3, straight
from the assignment):
    "freeze all BatchNorm running means and variances at their pretrained
     ImageNet values for every method... The BatchNorm scale and bias
     parameters (gamma and beta) remain trainable. In PyTorch, after
     calling model.train(), place only the BatchNorm modules in evaluation
     mode so their running statistics are not updated; do not place the
     complete model in evaluation mode."
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tv_models


class ResNet18Backbone(nn.Module):
    """Everything up to (and including) global average pooling -> 512-d feature."""

    feature_dim = 512

    def __init__(self):
        super().__init__()
        weights = tv_models.ResNet18_Weights.IMAGENET1K_V1
        net = tv_models.resnet18(weights=weights)
        self.stem = nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool,
            net.layer1, net.layer2, net.layer3, net.layer4,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.pool(x).flatten(1)
        return x


def set_train_with_frozen_bn(model: nn.Module) -> None:
    """
    Call this INSTEAD of model.train() during training for every method in
    Tasks 2 and 3. Puts the whole model in train mode (so dropout etc. in
    a domain discriminator behaves correctly, and gamma/beta keep
    receiving gradients) but forces every BatchNorm module specifically
    back into eval mode, so its running_mean/running_var stay frozen at
    the pretrained ImageNet values instead of drifting toward whatever
    source/target mixture happens to be in the current batch.
    """
    model.train()
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eval()


def freeze_bn_running_stats_permanently(model: nn.Module) -> None:
    """
    Extra safety net: disable running-stat updates at the module level too
    (track_running_stats-style guard), so even a stray model.train() call
    elsewhere in the code can't accidentally start updating BN statistics.
    This does NOT affect gamma/beta's requires_grad.
    """
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.momentum = 0.0  # running stats become invariant to eval-mode updates too
