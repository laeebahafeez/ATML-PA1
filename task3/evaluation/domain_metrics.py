"""
Core prediction/metric extraction for Task 3: per-source-domain accuracy and
macro-F1 (plus mean AND worst-domain, per the assignment's explicit request
for both), and the final Sketch evaluation -- the only metric in this file
that touches target data, and only ever called from evaluate_sketch.py.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader


@torch.no_grad()
def predict_on_dataset(backbone, head, dataset, device, batch_size: int = 64
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (preds, labels, features) over an entire (eval-transform) dataset."""
    backbone.eval()
    head.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    all_preds, all_labels, all_feats = [], [], []
    for x, y in loader:
        x = x.to(device)
        feats = backbone(x)
        logits = head(feats)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_labels.append(y.numpy())
        all_feats.append(feats.cpu().numpy())
    return (np.concatenate(all_preds), np.concatenate(all_labels),
            np.concatenate(all_feats))


def accuracy_and_macro_f1(preds: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    acc = float((preds == labels).mean())
    mf1 = float(f1_score(labels, preds, average="macro"))
    return acc, mf1


def evaluate_source_domains(backbone, head, source_val_datasets: Dict[str, object],
                             device) -> dict:
    """
    Per-source-domain accuracy/macro-F1, their mean, AND their worst (min)
    value -- the assignment explicitly asks for both, since "strong
    mean-source performance can hide a weak source domain."
    """
    per_domain = {}
    for domain, ds in source_val_datasets.items():
        preds, labels, _ = predict_on_dataset(backbone, head, ds, device)
        acc, mf1 = accuracy_and_macro_f1(preds, labels)
        per_domain[domain] = {"accuracy": acc, "macro_f1": mf1}

    accs = [v["accuracy"] for v in per_domain.values()]
    f1s = [v["macro_f1"] for v in per_domain.values()]
    return {
        "per_domain": per_domain,
        "mean_accuracy": float(np.mean(accs)),
        "mean_macro_f1": float(np.mean(f1s)),
        "worst_accuracy": float(np.min(accs)),
        "worst_macro_f1": float(np.min(f1s)),
    }


def evaluate_sketch(backbone, head, sketch_dataset, device) -> dict:
    """The ONLY function in evaluation/ that touches Sketch. Callers must
    only invoke this from evaluate_sketch.py, after every Task 3 training,
    checkpoint-selection, and hyperparameter decision has been frozen."""
    preds, labels, feats = predict_on_dataset(backbone, head, sketch_dataset, device)
    acc, mf1 = accuracy_and_macro_f1(preds, labels)
    return {
        "accuracy": acc, "macro_f1": mf1,
        "preds": preds, "labels": labels, "features": feats,
    }
