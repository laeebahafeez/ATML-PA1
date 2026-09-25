"""
Step 6 (Representation Analysis): cosine stability of backbone features
under each intervention.

    I_T = (1/N) * sum_i  f(x_i)^T f(T(x_i)) / (||f(x_i)|| * ||f(T(x_i))||)

Required for: grayscale, cue conflict, translation, and patch shuffling.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from metrics import cosine_similarity_paired


def representation_stability(
    clean_feats: np.ndarray, transformed_feats: np.ndarray
) -> float:
    """I_T for one (model, transform) pair. clean_feats and transformed_feats
    must be row-aligned: row i of each corresponds to the same underlying
    image / cue-conflict pair."""
    assert clean_feats.shape == transformed_feats.shape
    sims = cosine_similarity_paired(clean_feats, transformed_feats)
    return float(sims.mean())


def representation_stability_report(
    feats_by_model_and_transform: Dict[str, Dict[str, np.ndarray]],
    clean_feats_by_model: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    feats_by_model_and_transform: {model: {transform_name: transformed_feats}}
    clean_feats_by_model: {model: clean_feats}, row-aligned with each transform.

    Note: for the cue-conflict condition there is no single "clean"
    counterpart in the usual sense; pass the matching *content* image's
    clean features as `clean_feats_by_model["<model>"]` for that call so
    I_T measures stability relative to the untransformed content image.
    """
    rows = []
    for model_name, transforms in feats_by_model_and_transform.items():
        clean_feats = clean_feats_by_model[model_name]
        for transform_name, t_feats in transforms.items():
            it = representation_stability(clean_feats, t_feats)
            rows.append({"model": model_name, "transform": transform_name, "I_T": it})
    return pd.DataFrame(rows)
