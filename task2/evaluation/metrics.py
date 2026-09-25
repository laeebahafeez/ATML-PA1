"""
Core prediction/metric extraction, reused by evaluate_final.py to build
every required comparison table.
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


def evaluate_method_on_all_domains(
    backbone, head, source_val_datasets: Dict[str, object], target_dataset,
    device,
) -> dict:
    """
    source_val_datasets: {domain: eval-transform Subset/dataset}
    Returns a dict with per-source-domain acc/F1, mean, target acc/F1, and
    the raw target preds/labels (for downstream per-class analysis).
    """
    per_source = {}
    for domain, ds in source_val_datasets.items():
        preds, labels, _ = predict_on_dataset(backbone, head, ds, device)
        acc, mf1 = accuracy_and_macro_f1(preds, labels)
        per_source[domain] = {"accuracy": acc, "macro_f1": mf1}

    mean_source_acc = float(np.mean([v["accuracy"] for v in per_source.values()]))
    mean_source_f1 = float(np.mean([v["macro_f1"] for v in per_source.values()]))

    target_preds, target_labels, target_feats = predict_on_dataset(
        backbone, head, target_dataset, device)
    target_acc, target_f1 = accuracy_and_macro_f1(target_preds, target_labels)

    return {
        "per_source": per_source,
        "mean_source_accuracy": mean_source_acc,
        "mean_source_macro_f1": mean_source_f1,
        "target_accuracy": target_acc,
        "target_macro_f1": target_f1,
        "target_preds": target_preds,
        "target_labels": target_labels,
        "target_features": target_feats,
    }
