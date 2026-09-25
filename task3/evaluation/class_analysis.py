"""
Per-class Sketch analysis: does an aggregate Sketch accuracy change hide
class-specific improvement or degradation relative to ERM? Byte-identical
logic to task2/evaluation/class_analysis.py (duplicated across the task2/
task3 boundary for the same independence reason as models/backbone.py),
applied here to DAN-DG/SAM vs. the shared ERM baseline instead of DAN/DANN/
CDAN vs. Source-only.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def per_class_accuracy(preds: np.ndarray, labels: np.ndarray, class_names: List[str]) -> pd.Series:
    accs = {}
    for c, name in enumerate(class_names):
        mask = labels == c
        accs[name] = float((preds[mask] == labels[mask]).mean()) if mask.sum() > 0 else float("nan")
    return pd.Series(accs)


def per_class_comparison(
    method_preds: np.ndarray, method_labels: np.ndarray,
    baseline_preds: np.ndarray, baseline_labels: np.ndarray,
    class_names: List[str], method_name: str, baseline_name: str = "ERM",
) -> pd.DataFrame:
    assert np.array_equal(method_labels, baseline_labels), (
        "method and baseline must be evaluated on the identical, identically "
        "ordered Sketch set for a valid per-class comparison"
    )
    method_acc = per_class_accuracy(method_preds, method_labels, class_names)
    baseline_acc = per_class_accuracy(baseline_preds, baseline_labels, class_names)
    df = pd.DataFrame({
        f"{baseline_name}_accuracy": baseline_acc,
        f"{method_name}_accuracy": method_acc,
    })
    df["change"] = df[f"{method_name}_accuracy"] - df[f"{baseline_name}_accuracy"]
    return df.sort_values("change", ascending=False)


def top_confusions(preds: np.ndarray, labels: np.ndarray, class_names: List[str],
                    for_class: str, top_k: int = 3) -> pd.DataFrame:
    """Most common WRONG predicted classes for true class `for_class`."""
    c = class_names.index(for_class)
    mask = (labels == c) & (preds != c)
    wrong_preds = preds[mask]
    if len(wrong_preds) == 0:
        return pd.DataFrame(columns=["predicted_class", "count"])
    values, counts = np.unique(wrong_preds, return_counts=True)
    order = np.argsort(-counts)[:top_k]
    return pd.DataFrame({
        "predicted_class": [class_names[values[i]] for i in order],
        "count": [int(counts[i]) for i in order],
    })
