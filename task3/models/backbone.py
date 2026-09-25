"""
ResNet-18 backbone for Task 3, byte-identical to task2/models/backbone.py.

Duplicated rather than imported across the task2/task3 boundary -- same
rationale as shared/pacs_protocol.py's duplicated stratified-split function:
shared/ is the intentional home for anything both tasks need, and task2/
and task3/ are otherwise meant to be independent. This duplication is also
what makes loading Task 2's Source-only checkpoint into a Task 3
ResNet18Backbone instance safe: PyTorch state_dicts key on parameter names
within the class, not on which file defines the class, so an identical
class definition here loads Task 2's checkpoint without any remapping.

Fully fine-tuned for every Task 3 method (ERM is loaded, not retrained; DAN-DG
and SAM both fine-tune every parameter) -- only BatchNorm's running
mean/variance are frozen, exactly as in Task 2.
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
    """Call instead of model.train() for every Task 3 method. See
    task2/models/backbone.py for the full rationale -- identical policy."""
    model.train()
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eval()


def freeze_bn_running_stats_permanently(model: nn.Module) -> None:
    """Extra safety net, identical to task2/models/backbone.py."""
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.momentum = 0.0
