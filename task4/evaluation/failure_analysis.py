"""
Incorrectly-accepted-unknown inspection: for a fixed model/score/threshold,
find CIFAR-100 unknown examples with u(x) <= tau (i.e. the classifier would
accept and confidently label them as a known CIFAR-10 class), and record
their true CIFAR-100 fine class, predicted CIFAR-10 class, score, and
threshold -- exactly the fields the assignment's Required Evidence asks for.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def incorrectly_accepted(logits: np.ndarray, scores: np.ndarray,
                          fine_class_names: np.ndarray, threshold: float,
                          num_classes: int, top_n: int = 10) -> pd.DataFrame:
    """Returns up to `top_n` rows, sorted by score ascending (the most
    confidently-accepted -- i.e. most surprising -- failures first)."""
    known_logits = logits[:, :num_classes]
    accepted_mask = scores <= threshold
    if not accepted_mask.any():
        return pd.DataFrame(columns=["unknown_class", "predicted_class", "score", "threshold"])

    idx = np.nonzero(accepted_mask)[0]
    preds = known_logits[idx].argmax(axis=1)
    rows = pd.DataFrame({
        "unknown_class": fine_class_names[idx],
        "predicted_class": [CIFAR10_CLASSES[p] for p in preds],
        "score": scores[idx],
        "threshold": threshold,
    }).sort_values("score", ascending=True).reset_index(drop=True)
    return rows.head(top_n)


def summarize_confusions(rows: pd.DataFrame) -> pd.DataFrame:
    """Which (unknown_class -> predicted_class) pairs recur most often among
    the incorrectly-accepted examples -- helps separate 'semantically
    plausible' confusions (e.g. wolf -> dog) from surprising ones."""
    if rows.empty:
        return rows
    return (rows.groupby(["unknown_class", "predicted_class"])
            .size().reset_index(name="count").sort_values("count", ascending=False))
