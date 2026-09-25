"""
CIFAR-appropriate ResNet-18 for Task 4.

Per the assignment: "Use the same CIFAR-appropriate ResNet-18 architecture
throughout Task 4 ... Replace the ImageNet 7x7, stride-2 first convolution
with a 3x3, stride-1 convolution and remove the initial max-pooling layer.
Operate on the original 32x32 images."

Implemented by taking torchvision's standard resnet18 architecture (random
init, weights=None -- Task 4 trains from scratch, unlike Task 2/3's ImageNet
fine-tuning) and patching exactly the two components the assignment names,
rather than hand-rolling BasicBlocks from scratch: this keeps the block/layer
structure identical to the well-known, widely-used "CIFAR ResNet-18" recipe
while being provably faithful to torchvision's implementation everywhere else.

Exposes `forward_pre` / `forward_post`, split at the layer2/layer3 boundary,
because PROSER's manifold-mixup data placeholders require mixing features
"after layer2 and before layer3" for two different-class examples and then
continuing the forward pass on the mixed feature map.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tv_models


class ResNet18Cifar(nn.Module):
    """Everything up to (and including) global average pooling -> 512-d feature."""

    feature_dim = 512

    def __init__(self):
        super().__init__()
        net = tv_models.resnet18(weights=None, num_classes=10)  # num_classes unused: fc discarded

        # --- the assignment's two required stem changes ---
        net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        net.maxpool = nn.Identity()

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward_pre(self, x: torch.Tensor) -> torch.Tensor:
        """Stem -> layer1 -> layer2. Returns the conv feature MAP (not flattened),
        i.e. h = phi_pre(x) in the assignment's manifold-mixup notation."""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x

    def forward_post(self, h: torch.Tensor) -> torch.Tensor:
        """layer3 -> layer4 -> avgpool -> flatten. Takes a (possibly mixed)
        post-layer2 feature map and returns the 512-d penultimate feature."""
        x = self.layer3(h)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_post(self.forward_pre(x))
