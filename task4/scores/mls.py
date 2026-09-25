"""u_MLS(x) = -max_k z_k(x) (Maximum Logit Score). Larger = more novel.
Retains absolute logit magnitude, unlike MSP's normalized-confidence view."""
from __future__ import annotations

import numpy as np


def score(logits: np.ndarray) -> np.ndarray:
    return -logits.max(axis=1)
