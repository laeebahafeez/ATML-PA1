"""u_Energy(x) = -log sum_k exp(z_k(x)). Larger = more novel. Uses ALL logits
(not just the max), via scipy's numerically stable logsumexp."""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def score(logits: np.ndarray) -> np.ndarray:
    return -logsumexp(logits, axis=1)
