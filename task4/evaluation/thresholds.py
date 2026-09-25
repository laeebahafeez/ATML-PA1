"""
Validation-calibrated rejection threshold.

Per the assignment: "choose a threshold equal to the 95th percentile of
unknownness on the CIFAR-10 validation set; accept x when u(x) <= tau. This
uses known validation data only and aims to accept 95% of known examples.
Report the achieved CIFAR-10 test acceptance rate and the near- and
far-unknown rejection rates. Equivalently, report FPR@95TPR as the fraction
of unknown examples incorrectly accepted under this convention."

TPR here = fraction of KNOWN examples accepted (95% by calibration target on
val; measured achieved rate is reported on the held-out CIFAR-10 test set).
FPR@95TPR = fraction of unknown examples that are incorrectly accepted
(scored <= tau) at that same operating point.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def calibrate_threshold(val_known_scores: np.ndarray, percentile: float = 95.0) -> float:
    return float(np.percentile(val_known_scores, percentile))


def acceptance_rate(scores: np.ndarray, threshold: float) -> float:
    """Fraction of examples with u(x) <= threshold, i.e. accepted as known."""
    return float(np.mean(scores <= threshold))


def calibrated_report(val_known_scores: np.ndarray, test_known_scores: np.ndarray,
                       near_scores: np.ndarray, far_scores: np.ndarray,
                       percentile: float = 95.0) -> Dict[str, float]:
    tau = calibrate_threshold(val_known_scores, percentile)
    test_accept = acceptance_rate(test_known_scores, tau)
    near_accept = acceptance_rate(near_scores, tau)  # == FPR@95TPR for near
    far_accept = acceptance_rate(far_scores, tau)    # == FPR@95TPR for far
    all_unknown_scores = np.concatenate([near_scores, far_scores])
    all_accept = acceptance_rate(all_unknown_scores, tau)
    return {
        "threshold_tau": tau,
        "cifar10_test_acceptance_rate": test_accept,
        "near_unknown_rejection_rate": 1.0 - near_accept,
        "far_unknown_rejection_rate": 1.0 - far_accept,
        "fpr_at_95tpr_near": near_accept,
        "fpr_at_95tpr_far": far_accept,
        "fpr_at_95tpr_all": all_accept,
    }
