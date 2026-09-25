"""Seven-class linear classifier head on top of the 512-d ResNet-18 feature."""
from __future__ import annotations

import torch.nn as nn


class ClassifierHead(nn.Linear):
    def __init__(self, feature_dim: int = 512, num_classes: int = 7):
        super().__init__(feature_dim, num_classes)
