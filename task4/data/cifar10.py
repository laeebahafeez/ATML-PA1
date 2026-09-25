"""
CIFAR-10 loading for Task 4.

CIFAR-10/100 are downloaded directly via torchvision (no manual Kaggle-dataset
hunting needed, unlike Task 2/3's PACS) -- set `download=True` and point `root`
at a writable directory.

Normalization: CIFAR-10 mean/std are used for BOTH CIFAR-10 and the CIFAR-100
unknown-evaluation images (see cifar100_unknowns.py). This is deliberate: the
model was trained exclusively on CIFAR-10-normalized inputs, so evaluating
unknowns under the same normalization is the only fair, apples-to-apples
comparison -- using CIFAR-100's own statistics would silently shift the input
distribution the model sees, confounding "is this an unknown" with "is this
normalized differently."
"""
from __future__ import annotations

from typing import Tuple

import torchvision.transforms as T
from torchvision.datasets import CIFAR10

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def build_train_transform(crop_padding: int = 4, crop_size: int = 32,
                           hflip: bool = True, extra: list | None = None) -> T.Compose:
    """`extra` is where GCSC inserts RandAugment -- after crop+flip, before
    ToTensor/Normalize, per the assignment's explicit ordering."""
    ops = [T.RandomCrop(crop_size, padding=crop_padding)]
    if hflip:
        ops.append(T.RandomHorizontalFlip())
    if extra:
        ops.extend(extra)
    ops.extend([T.ToTensor(), T.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    return T.Compose(ops)


def build_eval_transform() -> T.Compose:
    return T.Compose([T.ToTensor(), T.Normalize(CIFAR10_MEAN, CIFAR10_STD)])


def build_unaugmented_transform() -> T.Compose:
    """No crop/flip/RandAugment at all -- used for Mahalanobis's class-mean/
    covariance estimation, which the assignment specifies must use
    'unaugmented CIFAR-10 training features.'"""
    return build_eval_transform()


def load_cifar10(root: str, train_transform: T.Compose, eval_transform: T.Compose,
                  download: bool = True) -> Tuple[CIFAR10, CIFAR10, CIFAR10]:
    """
    Returns (train_ds_with_train_transform, train_ds_with_eval_transform, test_ds).

    Two dataset objects wrap the SAME underlying training files (one with
    train-time augmentation, one without) so that a single stratified index
    split can select "train" rows from the augmented view and produce an
    unaugmented view of the same files whenever needed (e.g. Mahalanobis
    stats, or inspecting a raw training image) without loading the data twice
    from disk in different Python objects with mismatched indices.
    """
    train_aug_ds = CIFAR10(root=root, train=True, download=download, transform=train_transform)
    train_eval_ds = CIFAR10(root=root, train=True, download=download, transform=eval_transform)
    test_ds = CIFAR10(root=root, train=False, download=download, transform=eval_transform)
    return train_aug_ds, train_eval_ds, test_ds
