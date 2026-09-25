"""
AUROC for known-vs-unknown, given an unknownness score u(x) where LARGER
means more novel (the convention shared by every score in scores/*.py).

known label = 0, unknown label = 1, so AUROC = P(u(unknown) > u(known)) --
the standard OSR/OOD-detection convention: higher AUROC means the score
ranks unknowns above knowns more often.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def auroc_known_vs_unknown(known_scores: np.ndarray, unknown_scores: np.ndarray) -> float:
    labels = np.concatenate([np.zeros(len(known_scores)), np.ones(len(unknown_scores))])
    scores = np.concatenate([known_scores, unknown_scores])
    return float(roc_auc_score(labels, scores))


def roc_points(known_scores: np.ndarray, unknown_scores: np.ndarray):
    labels = np.concatenate([np.zeros(len(known_scores)), np.ones(len(unknown_scores))])
    scores = np.concatenate([known_scores, unknown_scores])
    fpr, tpr, thresholds = roc_curve(labels, scores)
    return fpr, tpr, thresholds


def all_auroc_variants(known_scores: np.ndarray, near_scores: np.ndarray,
                        far_scores: np.ndarray) -> Dict[str, float]:
    all_unknown_scores = np.concatenate([near_scores, far_scores])
    return {
        "auroc_known_vs_near": auroc_known_vs_unknown(known_scores, near_scores),
        "auroc_known_vs_far": auroc_known_vs_unknown(known_scores, far_scores),
        "auroc_known_vs_all": auroc_known_vs_unknown(known_scores, all_unknown_scores),
    }
