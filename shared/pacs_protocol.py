"""
The one shared PACS protocol used identically by Task 2 and Task 3, per
the assignment's instruction to "Reuse the same source splits across both
tasks."

Provides:
  - a stratified 80/20 train/val split per source domain (seed 6304)
  - a JSON save/load of that split so Task 3 can load *exactly* the same
    split without recomputing it
  - domain-balanced batch iteration: one training step pulls a fixed
    number of examples from each source domain (and, for adaptation
    methods, from the target domain), which is implemented here as
    separate per-domain DataLoaders combined by an infinite-cycling
    iterator, rather than a single custom multi-domain Sampler -- simpler
    to get right and to debug.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Iterator, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from pacs import EXPECTED_CLASSES, load_all_domain_pairs  # noqa: E402


def stratified_split_indices(
    labels: List[int], val_fraction: float = 0.2, seed: int = 6304
) -> Tuple[List[int], List[int]]:
    """Same algorithm as task1/data/split_utils.py, duplicated here (not
    imported across task boundaries) since shared/ is the intentional home
    for anything Task 2 and Task 3 both need."""
    labels_arr = np.asarray(labels)
    rng = np.random.default_rng(seed)
    train_idx: List[int] = []
    val_idx: List[int] = []
    for c in sorted(set(labels_arr.tolist())):
        idx = np.where(labels_arr == c)[0].copy()
        rng.shuffle(idx)
        n_val = int(round(len(idx) * val_fraction))
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    train_idx.sort()
    val_idx.sort()
    return train_idx, val_idx


def build_source_splits(
    source_domains: List[str], root: str, resize_size: int, crop_size: int,
    val_fraction: float, seed: int,
) -> Tuple[Dict[str, tuple], Dict[str, Dict[str, list]]]:
    """
    Returns:
      domain_pairs: {domain: (train_transform_dataset, eval_transform_dataset)}
                     for ALL four PACS domains (source + target)
      split_record: {domain: {"train_indices": [...], "val_indices": [...]}}
                     for the three SOURCE domains only
    """
    domain_pairs = load_all_domain_pairs(root, resize_size, crop_size)
    split_record: Dict[str, Dict[str, list]] = {}
    for domain in source_domains:
        _, eval_ds = domain_pairs[domain]
        labels = [s[1] for s in eval_ds.samples]
        train_idx, val_idx = stratified_split_indices(labels, val_fraction, seed)
        split_record[domain] = {"train_indices": train_idx, "val_indices": val_idx}
    return domain_pairs, split_record


def save_split(split_record: dict, path: str, seed: int, val_fraction: float) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump({
            "seed": seed,
            "val_fraction": val_fraction,
            "classes": EXPECTED_CLASSES,
            "domains": split_record,
        }, f, indent=2)


def load_split(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def make_source_subsets(
    domain_pairs: Dict[str, tuple], split_record: Dict[str, Dict[str, list]]
) -> Dict[str, Dict[str, Subset]]:
    """{domain: {"train": Subset(train-aug), "val": Subset(eval-preproc)}}"""
    out: Dict[str, Dict[str, Subset]] = {}
    for domain, idx in split_record.items():
        train_ds, eval_ds = domain_pairs[domain]
        out[domain] = {
            "train": Subset(train_ds, idx["train_indices"]),
            "val": Subset(eval_ds, idx["val_indices"]),
        }
    return out


def infinite_loader(loader: DataLoader) -> Iterator:
    """Cycle a DataLoader forever, reshuffling each pass (since DataLoader's
    own iterator raises StopIteration at the end of one epoch)."""
    while True:
        for batch in loader:
            yield batch


def make_domain_loader(dataset, batch_size: int, seed: int, shuffle: bool = True) -> DataLoader:
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                       drop_last=shuffle, num_workers=2, generator=gen if shuffle else None)


class SourceTargetBatchIterator:
    """
    Produces one combined training batch per call to `next()`:
        images: [3*source_per_domain + target_batch_size, C, H, W]
        labels: [3*source_per_domain]  (target has no labels -- unsupervised)
        domain_ids: [total]  (0/1/2 for the three source domains, 3 for target)
    Only source images carry class labels; only source images should ever
    contribute to the classification loss, matching the assignment's
    "Only source examples contribute to the class loss" instruction for
    DANN (and, by extension, DAN/CDAN's classification term).
    """

    def __init__(self, source_train_subsets: Dict[str, Subset], target_dataset,
                 source_domains: List[str], source_per_domain: int,
                 target_batch_size: int, seed: int, use_target: bool = True):
        self.source_domains = source_domains
        self.use_target = use_target
        self.source_iters = {
            d: infinite_loader(make_domain_loader(source_train_subsets[d], source_per_domain,
                                                   seed=seed + i))
            for i, d in enumerate(source_domains)
        }
        if use_target:
            self.target_iter = infinite_loader(
                make_domain_loader(target_dataset, target_batch_size, seed=seed + 100)
            )

    def __iter__(self):
        return self

    def __next__(self):
        all_images, all_labels, all_domain_ids = [], [], []
        for domain_id, d in enumerate(self.source_domains):
            x, y = next(self.source_iters[d])
            all_images.append(x)
            all_labels.append(y)
            all_domain_ids.append(torch.full((x.shape[0],), domain_id, dtype=torch.long))
        n_source = sum(t.shape[0] for t in all_labels)

        if self.use_target:
            xt, _ = next(self.target_iter)  # target labels are NEVER used in training
            all_images.append(xt)
            all_domain_ids.append(torch.full((xt.shape[0],), len(self.source_domains),
                                              dtype=torch.long))

        images = torch.cat(all_images, dim=0)
        labels = torch.cat(all_labels, dim=0)
        domain_ids = torch.cat(all_domain_ids, dim=0)
        return images, labels, domain_ids, n_source
