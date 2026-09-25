"""
Classifier head for Task 4: a single linear layer, 512-d feature -> logits.

For Vanilla/GCSC, `num_dummy=0` and this is an ordinary 10-way linear
classifier. For PROSER, `num_dummy=5` appends five randomly initialized
"dummy classifier" output units to the SAME linear layer -- columns
[0:10] are the known-class logits, columns [10:15] are the dummy-classifier
logits. Keeping them as one nn.Linear (rather than a separate module) matches
the PROSER paper's placeholder formulation, where the dummy classifiers are
just extra rows of the same final weight matrix, trained jointly with the
known-class rows.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ClassifierHead(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int, num_dummy: int = 0):
        super().__init__()
        self.num_classes = num_classes
        self.num_dummy = num_dummy
        self.fc = nn.Linear(feature_dim, num_classes + num_dummy)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """Returns logits of shape [B, num_classes + num_dummy]."""
        return self.fc(feats)

    def known_logits(self, all_logits: torch.Tensor) -> torch.Tensor:
        return all_logits[:, : self.num_classes]

    def dummy_logits(self, all_logits: torch.Tensor) -> torch.Tensor:
        assert self.num_dummy > 0, "This head has no dummy classifiers."
        return all_logits[:, self.num_classes :]
