"""
Reusable, seed-deterministic splitting utilities shared by make_subset.py
and make_cue_conflicts.py. Kept dependency-free (numpy only) so they are
trivially unit-testable without touching the actual dataset.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


def stratified_train_val_split(
    labels: Sequence[int], val_fraction: float = 0.2, seed: int = 6304
) -> Tuple[List[int], List[int]]:
    """
    Stratified split: within every class, the same `val_fraction` of
    examples goes to validation, using a single seeded RNG so the result
    is fully reproducible and independent of input ordering.
    """
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    train_idx: List[int] = []
    val_idx: List[int] = []
    for c in sorted(set(labels.tolist())):
        idx = np.where(labels == c)[0]
        idx = idx.copy()
        rng.shuffle(idx)
        n_val = int(round(len(idx) * val_fraction))
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    train_idx.sort()
    val_idx.sort()
    return train_idx, val_idx


def class_balanced_subset(
    labels: Sequence[int], target_size: int, seed: int = 6304
) -> Tuple[List[int], Dict[int, int], Dict[int, str]]:
    """
    Select a class-balanced subset of `target_size` total examples.

    Returns:
        selected_indices: sorted list of selected indices into `labels`
        per_class_count:  {class_id: number_selected}
        imbalance_notes:  {class_id: note} for any class with fewer
                           available examples than its target allocation
    """
    labels = np.asarray(labels)
    classes = sorted(set(labels.tolist()))
    n_classes = len(classes)
    base = target_size // n_classes
    remainder = target_size - base * n_classes

    rng = np.random.default_rng(seed)
    # Randomly choose which `remainder` classes receive one extra image,
    # deterministically given the fixed seed.
    extra_classes = set(rng.choice(classes, size=remainder, replace=False).tolist())

    selected: List[int] = []
    per_class_count: Dict[int, int] = {}
    imbalance_notes: Dict[int, str] = {}

    for c in classes:
        idx = np.where(labels == c)[0].copy()
        rng.shuffle(idx)
        want = base + (1 if c in extra_classes else 0)
        take = min(want, len(idx))
        if take < want:
            imbalance_notes[c] = (
                f"class {c} has only {len(idx)} available examples; "
                f"wanted {want}, used all {take}."
            )
        selected.extend(idx[:take].tolist())
        per_class_count[c] = take

    selected.sort()
    return selected, per_class_count, imbalance_notes
