"""
Vanilla closed-set baseline: ordinary 10-class cross-entropy on CIFAR-10, no
augmentation beyond the assignment's shared recipe (random crop+pad+hflip).

Trained by the shared epoch loop in train.py -- this module only supplies the
method-specific piece (the augmentation transform), matching Task 2/3's
pattern of thin per-method modules plugged into one unified training loop.
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
from cifar10 import build_train_transform  # noqa: E402


def build_transform(cfg: dict):
    aug = cfg["augmentation"]
    return build_train_transform(
        crop_padding=aug["random_crop_padding"],
        crop_size=aug["random_crop_size"],
        hflip=aug["random_hflip"],
        extra=None,
    )
