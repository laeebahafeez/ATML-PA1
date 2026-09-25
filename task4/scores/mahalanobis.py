"""
u_Mahalanobis(x) = min_c (f(x)-mu_c)^T Sigma^-1 (f(x)-mu_c).

Per the assignment: "estimate class means mu_c and one shared diagonal
covariance Sigma from unaugmented CIFAR-10 training features, adding 1e-6 to
every diagonal entry." One shared covariance across all 10 classes (not
per-class), and diagonal only (not full covariance) -- both explicit in the
assignment's formula and wording.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def fit(train_features: np.ndarray, train_labels: np.ndarray, diag_eps: float = 1e-6
        ) -> Dict[str, np.ndarray]:
    """train_features: [N, D] unaugmented CIFAR-10 TRAIN features from the
    frozen vanilla model. Returns {"class_means": [K, D], "inv_diag_cov": [D]}."""
    classes = np.sort(np.unique(train_labels))
    D = train_features.shape[1]
    class_means = np.zeros((len(classes), D), dtype=np.float64)

    # Pooled within-class diagonal variance: center each example by its own
    # class mean first, then estimate one shared diagonal covariance across
    # all (centered) examples -- this is what "one shared diagonal covariance"
    # from the training features means (the standard tied-covariance LDA/
    # Mahalanobis-OOD construction, e.g. Lee et al. 2018).
    centered = np.zeros_like(train_features, dtype=np.float64)
    for i, c in enumerate(classes):
        mask = train_labels == c
        mu_c = train_features[mask].mean(axis=0)
        class_means[i] = mu_c
        centered[mask] = train_features[mask] - mu_c

    var_diag = centered.var(axis=0) + diag_eps
    inv_diag_cov = 1.0 / var_diag
    return {"class_means": class_means, "inv_diag_cov": inv_diag_cov, "classes": classes}


def score(features: np.ndarray, fitted: Dict[str, np.ndarray]) -> np.ndarray:
    """features: [N, D]. Returns u(x) = min_c Mahalanobis^2 distance to class c."""
    means = fitted["class_means"]  # [K, D]
    inv_diag = fitted["inv_diag_cov"]  # [D]
    # squared Mahalanobis distance to each class, looped over the (small, K=10)
    # class dimension rather than materializing an [N, K, D] tensor
    num_classes = means.shape[0]
    dist2 = np.empty((features.shape[0], num_classes), dtype=np.float64)
    for c in range(num_classes):
        diff = features - means[c]  # [N, D]
        dist2[:, c] = np.sum((diff * diff) * inv_diag, axis=1)
    return dist2.min(axis=1)
