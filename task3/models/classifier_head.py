"""Linear classifier head, byte-identical to task2/models/classifier_head.py
(duplicated for the same state-dict-compatibility reason as backbone.py)."""
from __future__ import annotations

import torch.nn as nn


class ClassifierHead(nn.Linear):
    def __init__(self, feature_dim: int = 512, num_classes: int = 7):
        super().__init__(feature_dim, num_classes)
