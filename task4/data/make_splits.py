"""
Stratified 90/10 train/val split of CIFAR-10's official training partition,
seed 6304. Saved once and reused verbatim by every Task 4 method (vanilla,
gcsc, proser) so all three select checkpoints against the exact same
validation set -- same rationale as Task 2/3's shared pacs_sketch split.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np


def stratified_split_indices(
    labels: List[int], val_fraction: float = 0.1, seed: int = 6304
) -> Tuple[List[int], List[int]]:
    """Same algorithm as task1/task2/task3's split utilities, duplicated here
    (task4/ is otherwise self-contained -- CIFAR has nothing in common with
    PACS worth sharing across the task2/3 <-> task4 boundary)."""
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


def save_split(train_idx: List[int], val_idx: List[int], path: str,
                seed: int, val_fraction: float) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump({
            "seed": seed,
            "val_fraction": val_fraction,
            "train_indices": train_idx,
            "val_indices": val_idx,
        }, f, indent=2)


def load_split(path: str) -> Dict[str, list]:
    with open(path) as f:
        return json.load(f)


def get_or_build_split(train_labels: List[int], path: str, seed: int,
                        val_fraction: float) -> Tuple[List[int], List[int]]:
    """Build once, then reuse verbatim on every subsequent call (e.g. gcsc's
    and proser's training runs within the same or later sessions) -- mirrors
    Task 2's prepare_data() pattern."""
    if os.path.exists(path):
        record = load_split(path)
        assert record["seed"] == seed, (
            f"Existing split at {path!r} was built with seed {record['seed']}, "
            f"not the requested {seed} -- refusing to silently mix splits."
        )
        return record["train_indices"], record["val_indices"]
    train_idx, val_idx = stratified_split_indices(train_labels, val_fraction, seed)
    save_split(train_idx, val_idx, path, seed, val_fraction)
    return train_idx, val_idx
