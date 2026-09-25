"""Low-level, dependency-light metric primitives reused across every Task 1 step."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def softmax_np(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    logits = logits - logits.max(axis=axis, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=axis, keepdims=True)


def top1_accuracy(preds: Sequence[int], labels: Sequence[int]) -> float:
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    return float((preds == labels).mean())


def macro_f1(preds: Sequence[int], labels: Sequence[int]) -> float:
    from sklearn.metrics import f1_score  # lazy import: keeps this module
    # importable (for the torch-free parts of the test suite) even where
    # scikit-learn is not installed.
    return float(f1_score(labels, preds, average="macro"))


def mean_max_confidence(probs: np.ndarray) -> float:
    """probs: [N, C] softmax probabilities."""
    return float(probs.max(axis=1).mean())


def prediction_consistency(preds_a: Sequence[int], preds_b: Sequence[int]) -> float:
    """Fraction of examples whose predicted class is unchanged between two
    prediction arrays (e.g. clean vs. transformed)."""
    preds_a = np.asarray(preds_a)
    preds_b = np.asarray(preds_b)
    assert preds_a.shape == preds_b.shape
    return float((preds_a == preds_b).mean())


def cosine_similarity_paired(feats_a: np.ndarray, feats_b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between two [N, D] feature matrices."""
    a = feats_a / (np.linalg.norm(feats_a, axis=1, keepdims=True) + 1e-12)
    b = feats_b / (np.linalg.norm(feats_b, axis=1, keepdims=True) + 1e-12)
    return (a * b).sum(axis=1)
