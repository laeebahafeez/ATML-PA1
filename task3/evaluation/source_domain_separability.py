"""
Source-domain separability diagnostic (Task 3's analog of Task 2's
source-vs-target probe, but 3-WAY among the observed source domains
themselves): freeze the backbone, collect balanced Photo/Art Painting/
Cartoon validation features, and see how well a multinomial logistic
regression can still tell which source domain each one came from.

Chance level = 1/3 (33.3%), not 50% as in Task 2's binary version.
A lower score is evidence of stronger cross-source invariance -- but per
the assignment's explicit warning, this does NOT by itself establish that
class-discriminative information was preserved or that Sketch performance
improved; DAN-DG could reduce this score by discarding class structure.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def source_domain_separability_score(
    features_by_domain: Dict[str, np.ndarray],
    seed: int = 6304, test_fraction: float = 0.3, C: float = 1.0,
) -> dict:
    domains = sorted(features_by_domain.keys())
    n = min(len(features_by_domain[d]) for d in domains)
    rng = np.random.default_rng(seed)

    X_parts, y_parts = [], []
    for label_id, d in enumerate(domains):
        idx = rng.choice(len(features_by_domain[d]), size=n, replace=False)
        X_parts.append(features_by_domain[d][idx])
        y_parts.append(np.full(n, label_id))
    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_fraction, random_state=seed, stratify=y,
    )
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000)
    clf.fit(X_train, y_train)
    held_out_acc = float(clf.score(X_test, y_test))

    return {
        "domains": domains,
        "n_per_domain": n,
        "held_out_accuracy": held_out_acc,
        "chance_level": 1.0 / len(domains),
    }
