"""
Step-level evaluation functions for Task 1 (Steps 1-5).

Every function here is pure: it takes already-computed predictions/logits
(produced by scripts/run_task1.py, which does the actual forward passes)
and returns a small, report-ready dictionary or DataFrame. Keeping model
inference out of this file makes every metric independently unit-testable.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from metrics import (
    macro_f1,
    mean_max_confidence,
    prediction_consistency,
    softmax_np,
    top1_accuracy,
)


# --------------------------------------------------------------------------- #
# Step 1: Clean baseline
# --------------------------------------------------------------------------- #

def clean_baseline_report(logits_by_model: Dict[str, np.ndarray],
                           labels: np.ndarray) -> pd.DataFrame:
    """logits_by_model: {"resnet50": [...], "vit_b16": [...], "clip_head": [...],
    "clip_zeroshot": [...]} each [N, C]."""
    rows = []
    for name, logits in logits_by_model.items():
        probs = softmax_np(logits)
        preds = probs.argmax(axis=1)
        rows.append({
            "model": name,
            "top1_accuracy": top1_accuracy(preds, labels),
            "macro_f1": macro_f1(preds, labels),
            "mean_max_confidence": mean_max_confidence(probs),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Step 2: Color bias (grayscale / hue rotation)
# --------------------------------------------------------------------------- #

def color_bias_report(
    clean_logits_by_model: Dict[str, np.ndarray],
    transformed_logits_by_model: Dict[str, Dict[str, np.ndarray]],
    labels: np.ndarray,
) -> pd.DataFrame:
    """
    transformed_logits_by_model: {model_name: {"grayscale": logits, "hue_rotation": logits}}
    Reports absolute accuracy under the transform AND the change relative to
    that same model's own clean baseline, plus prediction consistency.
    """
    rows = []
    for model_name, clean_logits in clean_logits_by_model.items():
        clean_preds = softmax_np(clean_logits).argmax(axis=1)
        clean_acc = top1_accuracy(clean_preds, labels)
        for transform_name, t_logits in transformed_logits_by_model.get(model_name, {}).items():
            t_preds = softmax_np(t_logits).argmax(axis=1)
            t_acc = top1_accuracy(t_preds, labels)
            rows.append({
                "model": model_name,
                "transform": transform_name,
                "clean_accuracy": clean_acc,
                "transformed_accuracy": t_acc,
                "accuracy_change": t_acc - clean_acc,
                "prediction_consistency": prediction_consistency(clean_preds, t_preds),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Step 3: Shape vs. texture (cue conflicts)
# --------------------------------------------------------------------------- #

def classify_cue_conflict_prediction(pred_class: int, content_class: int,
                                      style_class: int) -> str:
    if pred_class == content_class:
        return "shape"
    if pred_class == style_class:
        return "texture"
    return "other"


def shape_bias_report(
    predictions_by_model: Dict[str, List[int]],
    manifest: List[dict],
) -> pd.DataFrame:
    """
    predictions_by_model: {model_name: [pred_class_for_each_conflict_in_manifest_order]}
    manifest: list of dicts with 'content_class_idx' and 'style_class_idx'
              (accepted conflicts only, in the same order as predictions).
    """
    content = np.array([m["content_class_idx"] for m in manifest])
    style = np.array([m["style_class_idx"] for m in manifest])

    rows = []
    for model_name, preds in predictions_by_model.items():
        preds = np.asarray(preds)
        labels_kind = [
            classify_cue_conflict_prediction(p, c, s)
            for p, c, s in zip(preds, content, style)
        ]
        n_shape = labels_kind.count("shape")
        n_texture = labels_kind.count("texture")
        n_other = labels_kind.count("other")
        n_total = len(labels_kind)
        denom = n_shape + n_texture
        shape_bias = 100.0 * n_shape / denom if denom > 0 else float("nan")
        coverage = 100.0 * denom / n_total if n_total > 0 else float("nan")
        rows.append({
            "model": model_name,
            "n_shape": n_shape,
            "n_texture": n_texture,
            "n_other": n_other,
            "n_total": n_total,
            "shape_bias_pct": shape_bias,
            "coverage_pct": coverage,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Step 4: Translation
# --------------------------------------------------------------------------- #

def translation_curve_report(
    logits_by_pixels_direction: Dict[int, Dict[str, np.ndarray]],
    clean_logits: np.ndarray,
    labels: np.ndarray,
) -> pd.DataFrame:
    """
    logits_by_pixels_direction: {pixels: {direction: logits [N, C]}}
    For each displacement, averages accuracy and consistency (vs. clean
    predictions) across the four cardinal directions.
    """
    clean_preds = softmax_np(clean_logits).argmax(axis=1)
    rows = []
    for pixels, per_direction in sorted(logits_by_pixels_direction.items()):
        accs, cons = [], []
        for direction, logits in per_direction.items():
            preds = softmax_np(logits).argmax(axis=1)
            accs.append(top1_accuracy(preds, labels))
            cons.append(prediction_consistency(clean_preds, preds))
        rows.append({
            "pixels": pixels,
            "accuracy": float(np.mean(accs)),
            "consistency": float(np.mean(cons)),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Step 5: Patch structure
# --------------------------------------------------------------------------- #

def patch_shuffle_report(
    clean_logits_by_model: Dict[str, np.ndarray],
    shuffled_logits_by_model: Dict[str, np.ndarray],
    labels: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for model_name, clean_logits in clean_logits_by_model.items():
        clean_preds = softmax_np(clean_logits).argmax(axis=1)
        clean_acc = top1_accuracy(clean_preds, labels)
        shuf_logits = shuffled_logits_by_model[model_name]
        shuf_preds = softmax_np(shuf_logits).argmax(axis=1)
        shuf_acc = top1_accuracy(shuf_preds, labels)
        rows.append({
            "model": model_name,
            "clean_accuracy": clean_acc,
            "shuffled_accuracy": shuf_acc,
            "accuracy_drop": clean_acc - shuf_acc,
            "prediction_consistency": prediction_consistency(clean_preds, shuf_preds),
        })
    return pd.DataFrame(rows)
