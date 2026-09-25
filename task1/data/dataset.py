"""
Thin dataset wrapper that returns a common, un-normalized [0, 1] 224x224
RGB tensor for a given set of Oxford-IIIT Pets indices, plus the label and
the original dataset index (needed so patch_shuffle can seed
deterministically per-image).
"""
from __future__ import annotations

from typing import List, Sequence

import torch
from torch.utils.data import Dataset
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms.functional import resize, to_tensor


class PetsSubset(Dataset):
    def __init__(self, root: str, split: str, indices: Sequence[int], image_size: int = 224):
        self.base = OxfordIIITPet(root=root, split=split, target_types="category",
                                   download=True)
        self.indices = list(indices)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        base_idx = self.indices[i]
        img, label = self.base[base_idx]
        img = resize(img, [self.image_size, self.image_size])
        x01 = to_tensor(img)
        if x01.shape[0] == 1:  # a few Pets images are grayscale; replicate to RGB
            x01 = x01.expand(3, -1, -1)
        return x01, label, base_idx

    @property
    def classes(self) -> List[str]:
        return self.base.classes


def collate_common(batch):
    xs, ys, idxs = zip(*batch)
    return torch.stack(xs), torch.tensor(ys), torch.tensor(idxs)
