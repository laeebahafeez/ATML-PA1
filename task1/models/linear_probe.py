"""
Linear classifier head training on pre-extracted, frozen backbone features.

Because the backbone is frozen, features can be extracted once and cached;
training the head is then just logistic regression via SGD, which is why
this is implemented directly on feature tensors rather than re-running the
backbone every epoch.

Spec (identical for every backbone, per the assignment):
    AdamW, lr=1e-3, weight_decay=1e-4, up to 50 epochs, early stopping
    after 5 epochs without improved validation accuracy, seed 6304.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ProbeConfig:
    max_epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 5
    seed: int = 6304
    batch_size: int = 128


def train_linear_probe(
    train_feats: torch.Tensor, train_labels: torch.Tensor,
    val_feats: torch.Tensor, val_labels: torch.Tensor,
    num_classes: int, cfg: ProbeConfig,
) -> nn.Linear:
    torch.manual_seed(cfg.seed)
    feature_dim = train_feats.shape[1]
    head = nn.Linear(feature_dim, num_classes)
    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg.lr,
                                   weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    n = train_feats.shape[0]
    best_val_acc = -1.0
    best_state = copy.deepcopy(head.state_dict())
    epochs_without_improvement = 0
    gen = torch.Generator().manual_seed(cfg.seed)

    for epoch in range(cfg.max_epochs):
        head.train()
        perm = torch.randperm(n, generator=gen)
        for start in range(0, n, cfg.batch_size):
            idx = perm[start:start + cfg.batch_size]
            optimizer.zero_grad()
            logits = head(train_feats[idx])
            loss = criterion(logits, train_labels[idx])
            loss.backward()
            optimizer.step()

        head.eval()
        with torch.no_grad():
            val_logits = head(val_feats)
            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == val_labels).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(head.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.patience:
                break

    head.load_state_dict(best_state)
    head.eval()
    return head
