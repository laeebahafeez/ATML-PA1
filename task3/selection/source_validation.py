"""
Checkpoint-selection metric used INSIDE the train.py epoch loop for DAN-DG
and SAM: mean macro-F1 across the three source validation domains, per the
assignment's "Select every checkpoint using mean macro-F1 across the three
source validation domains." Kept separate from evaluation/domain_metrics.py
(which also reports worst-domain and Sketch) so it's obvious at a glance
that only source, never target, data can influence this decision.
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "evaluation"))
from domain_metrics import evaluate_source_domains  # noqa: E402


def mean_source_macro_f1(backbone, head, source_val_datasets, device) -> float:
    result = evaluate_source_domains(backbone, head, source_val_datasets, device)
    return result["mean_macro_f1"]
