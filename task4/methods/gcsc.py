"""
GCSC -- "strong closed-set classifier": exactly the Vanilla recipe with one
change, RandAugment(num_ops=2, magnitude=9) inserted after crop+flip and
before tensor conversion/normalization. Tests whether an augmentation-induced
change in closed-set accuracy is accompanied by better rejection (Task 4
Research Question 3).
"""
from __future__ import annotations

import sys
import os

import torchvision.transforms as T

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
from cifar10 import build_train_transform  # noqa: E402


def build_transform(cfg: dict):
    aug = cfg["augmentation"]
    gcsc_cfg = cfg["gcsc"]
    rand_augment = T.RandAugment(
        num_ops=gcsc_cfg["randaugment_num_ops"],
        magnitude=gcsc_cfg["randaugment_magnitude"],
    )
    return build_train_transform(
        crop_padding=aug["random_crop_padding"],
        crop_size=aug["random_crop_size"],
        hflip=aug["random_hflip"],
        extra=[rand_augment],
    )
