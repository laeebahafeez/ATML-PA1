"""u_MSP(x) = 1 - max_k p_k(x), p = softmax(logits). Larger = more novel."""
from __future__ import annotations

import numpy as np
from scipy.special import softmax


def score(logits: np.ndarray) -> np.ndarray:
    probs = softmax(logits, axis=1)
    return 1.0 - probs.max(axis=1)
