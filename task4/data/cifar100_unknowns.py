"""
Fixed, evaluation-only CIFAR-100 unknown sets for Task 4.

Loaded ONLY by evaluate_osr.py, after every training/checkpoint/score/threshold
decision has been frozen -- never by train.py, extract_outputs.py's train/val/
test-CIFAR-10 path, or any selection logic. The assignment is explicit that
CIFAR-100 training images may not be used at all, and CIFAR-100 test images
may only be used for final evaluation.

Near-unknown group (visually/semantically closer to a CIFAR-10 class):
  bus, pickup_truck, motorcycle, tractor, wolf, fox, leopard, camel
Far-unknown group (less overlap with the known label space):
  bottle, bowl, chair, clock, keyboard, mushroom, sunflower, wardrobe

Each group is exactly 800 images: all CIFAR-100 TEST images (100/class) of
these 8 classes, nothing added or subtracted. The grouping is fixed by the
assignment and must not be revisited after seeing results.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100

from cifar10 import CIFAR10_MEAN, CIFAR10_STD, build_eval_transform  # noqa: E402

NEAR_CLASSES = ["bus", "pickup_truck", "motorcycle", "tractor",
                "wolf", "fox", "leopard", "camel"]
FAR_CLASSES = ["bottle", "bowl", "chair", "clock",
               "keyboard", "mushroom", "sunflower", "wardrobe"]
EXPECTED_PER_GROUP = 800  # 8 classes x 100 CIFAR-100 test images/class


class _IndexSubsetDataset(Dataset):
    """Thin wrapper so callers get (image, fine_class_name) rather than
    CIFAR100's raw integer fine-label id, which is easy to confuse with
    CIFAR-10's own 0-9 label space downstream."""

    def __init__(self, base: CIFAR100, indices: List[int], idx_to_name: dict):
        self.base = base
        self.indices = indices
        self.idx_to_name = idx_to_name

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        real_idx = self.indices[i]
        image, fine_label = self.base[real_idx]
        return image, self.idx_to_name[fine_label]


def _select_group(base: CIFAR100, class_names: List[str]) -> _IndexSubsetDataset:
    name_to_idx = {name: i for i, name in enumerate(base.classes)}
    missing = [c for c in class_names if c not in name_to_idx]
    assert not missing, (
        f"Class name(s) {missing} not found in CIFAR-100's fine-label set. "
        f"Check spelling against torchvision's CIFAR100(...).classes."
    )
    wanted_idx = {name_to_idx[c] for c in class_names}
    idx_to_name = {name_to_idx[c]: c for c in class_names}
    targets = base.targets if hasattr(base, "targets") else [t for _, t in base]
    indices = [i for i, t in enumerate(targets) if t in wanted_idx]
    assert len(indices) == EXPECTED_PER_GROUP, (
        f"Expected {EXPECTED_PER_GROUP} images for classes {class_names}, "
        f"found {len(indices)}. CIFAR-100's test set may be malformed or "
        f"class names may not match torchvision's fine-label spelling."
    )
    return _IndexSubsetDataset(base, indices, idx_to_name)


def load_unknown_groups(root: str, download: bool = True
                         ) -> Tuple[_IndexSubsetDataset, _IndexSubsetDataset]:
    """Returns (near_dataset, far_dataset), each yielding (image, fine_class_name),
    normalized with CIFAR-10's mean/std (see cifar10.py's module docstring for why)."""
    transform = build_eval_transform()  # CIFAR10_MEAN/STD, no augmentation
    base = CIFAR100(root=root, train=False, download=download, transform=transform)
    near_ds = _select_group(base, NEAR_CLASSES)
    far_ds = _select_group(base, FAR_CLASSES)
    return near_ds, far_ds
