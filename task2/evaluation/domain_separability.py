"""
Domain-separability diagnostic (Task 2 Step 5 / Task 3 Step 4's analog):
freeze the backbone, collect equal numbers of source-validation and
target features, and see how well a simple linear classifier can still
tell them apart. 50% = indistinguishable; higher = a domain "fingerprint"
still survives in the representation.

This is a DIAGNOSTIC, not a target-recognition metric -- per the
assignment's explicit warning, a lower score here is evidence that domain
information is harder to recover, NOT proof that class-discriminative
information was preserved.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def domain_separability_score(
    source_val_features: np.ndarray, target_features: np.ndarray,
    seed: int = 6304, test_fraction: float = 0.3, C: float = 1.0,
) -> dict:
    n = min(len(source_val_features), len(target_features))
    rng = np.random.default_rng(seed)
    src_idx = rng.choice(len(source_val_features), size=n, replace=False)
    tgt_idx = rng.choice(len(target_features), size=n, replace=False)

    X = np.concatenate([source_val_features[src_idx], target_features[tgt_idx]], axis=0)
    y = np.concatenate([np.zeros(n), np.ones(n)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_fraction, random_state=seed, stratify=y,
    )
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000)
    clf.fit(X_train, y_train)
    held_out_acc = float(clf.score(X_test, y_test))

    return {
        "n_per_class": n,
        "held_out_accuracy": held_out_acc,
        "chance_level": 0.5,
    }
