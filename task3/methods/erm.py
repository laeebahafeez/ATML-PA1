"""
Source-only Empirical Risk Minimization.

Per the assignment: "The Source-only model in this task is also the ERM
baseline for Task 3 and should not be retrained differently... load its
saved checkpoint rather than retraining it under a different configuration."
train.py therefore does NOT call train_step() for method="erm" by default --
it calls load_source_only_checkpoint() instead. train_step() is kept here
(a) to document the exact ERM objective in one place, matching the other
two methods' modules, and (b) so tests/test_pipeline.py can exercise it on
synthetic data without touching any real checkpoint.

    L_ERM = (1/3) * sum_{e in {P,A,C}} R_e(theta)

Domain-balanced batches (8 examples per source domain, per
shared/pacs_protocol.py) already give every source domain equal influence,
so this reduces to a single cross-entropy term over the whole batch.
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F


def train_step(backbone, head, images, labels, domain_ids, cfg, optimizer) -> dict:
    optimizer.zero_grad()
    feats = backbone(images)
    logits = head(feats)
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    optimizer.step()
    return {"cls_loss": loss.item(), "total_loss": loss.item()}


def load_source_only_checkpoint(path: str, map_location=None) -> dict:
    """Loads Task 2's checkpoints/source_only.pt. Raises a clear error
    (rather than a raw FileNotFoundError several frames deep) if Task 2
    hasn't been run yet -- this checkpoint is a hard prerequisite for all
    of Task 3, not just the ERM row."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Task 2's Source-only checkpoint not found at {path!r}. "
            "Task 3 reuses this checkpoint as its ERM baseline rather than "
            "retraining -- run `python train.py --method source_only` in "
            "task2/ first, or check task2_paths.source_only_checkpoint in "
            "task3/configs/config.yaml."
        )
    return torch.load(path, map_location=map_location)
