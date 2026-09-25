"""
Unified training entry point for Task 4.

Usage:
    python train.py --method vanilla
    python train.py --method gcsc
    python train.py --method proser --vanilla_checkpoint checkpoints/vanilla.pt

Vanilla and GCSC share one plain cross-entropy training loop (only the
augmentation transform differs -- see methods/vanilla.py, methods/gcsc.py).
PROSER is structurally different (fine-tunes from a Vanilla checkpoint with
two loss terms and a batch split) and lives in its own training loop in
methods/proser.py; this file just wires up its inputs and calls it.

CIFAR-100 is never imported by this file, directly or indirectly -- only
CIFAR-10 (data/cifar10.py, data/make_splits.py). Enforced by construction,
not just convention.
"""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "methods"))

from cifar10 import load_cifar10, build_eval_transform  # noqa: E402
from make_splits import get_or_build_split  # noqa: E402
from resnet_cifar import ResNet18Cifar  # noqa: E402
from classifier_head import ClassifierHead  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def prepare_data(cfg: dict, train_transform):
    train_aug_ds, train_eval_ds, test_ds = load_cifar10(
        root=cfg["dataset"]["root"], train_transform=train_transform,
        eval_transform=build_eval_transform(), download=True,
    )
    labels = train_aug_ds.targets  # identical ordering/labels for both train views
    train_idx, val_idx = get_or_build_split(
        labels, cfg["split"]["save_path"], cfg["split"]["seed"], cfg["split"]["val_fraction"])
    train_subset = Subset(train_aug_ds, train_idx)
    val_subset = Subset(train_eval_ds, val_idx)  # val gets NO augmentation
    return train_subset, val_subset, test_ds, train_eval_ds


@torch.no_grad()
def evaluate_accuracy(backbone, head, loader: DataLoader, num_classes: int) -> float:
    backbone.eval()
    head.eval()
    all_preds, all_labels = [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = head(backbone(x))[:, :num_classes]  # ignore any dummy columns
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_labels.append(y.cpu().numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    return float((preds == labels).mean())


def build_model_bundle(cfg: dict, num_dummy: int = 0):
    backbone = ResNet18Cifar().to(DEVICE)
    head = ClassifierHead(cfg["backbone"]["feature_dim"], cfg["dataset"]["num_classes"],
                           num_dummy=num_dummy).to(DEVICE)
    return backbone, head


def train_vanilla_or_gcsc(method: str, cfg: dict):
    torch.manual_seed(cfg["training"]["seed"])
    method_module = importlib.import_module(method)
    train_transform = method_module.build_transform(cfg)

    train_subset, val_subset, _, _ = prepare_data(cfg, train_transform)
    train_loader = DataLoader(train_subset, batch_size=cfg["training"]["batch_size"],
                               shuffle=True, num_workers=2, drop_last=True,
                               generator=torch.Generator().manual_seed(cfg["training"]["seed"]))
    val_loader = DataLoader(val_subset, batch_size=256, shuffle=False, num_workers=2)

    backbone, head = build_model_bundle(cfg)
    params = list(backbone.parameters()) + list(head.parameters())
    optimizer = torch.optim.SGD(params, lr=cfg["training"]["lr"],
                                 momentum=cfg["training"]["momentum"],
                                 weight_decay=cfg["training"]["weight_decay"])
    epochs = cfg["training"]["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = -1.0
    best_state = None
    history = []

    for epoch in range(epochs):
        backbone.train()
        head.train()
        epoch_losses = []
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = head(backbone(x))
            loss = nn.functional.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        scheduler.step()

        val_acc = evaluate_accuracy(backbone, head, val_loader, cfg["dataset"]["num_classes"])
        mean_loss = float(np.mean(epoch_losses))
        history.append({"epoch": epoch, "train_loss": mean_loss, "val_accuracy": val_acc,
                         "lr": scheduler.get_last_lr()[0]})
        print(f"[{method}] epoch {epoch:03d}  train_loss={mean_loss:.4f}  "
              f"val_accuracy={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                "backbone": copy.deepcopy(backbone.state_dict()),
                "head": copy.deepcopy(head.state_dict()),
                "epoch": epoch, "val_accuracy": val_acc,
            }

    backbone.load_state_dict(best_state["backbone"])
    head.load_state_dict(best_state["head"])
    return backbone, head, best_state, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["vanilla", "gcsc", "proser"])
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--vanilla_checkpoint", default="checkpoints/vanilla.pt",
                         help="required for --method proser")
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    if args.method in ("vanilla", "gcsc"):
        backbone, head, best_state, history = train_vanilla_or_gcsc(args.method, cfg)
        torch.save({"backbone": backbone.state_dict(), "head": head.state_dict(),
                    "best_state_meta": {k: v for k, v in best_state.items()
                                         if k not in ("backbone", "head")}},
                   f"checkpoints/{args.method}.pt")
        with open(f"results/train_history_{args.method}.json", "w") as f:
            json.dump(history, f, indent=2)
        print(f"Saved checkpoints/{args.method}.pt and "
              f"results/train_history_{args.method}.json")
        return

    # method == "proser"
    import proser  # noqa: E402
    assert os.path.exists(args.vanilla_checkpoint), (
        f"PROSER must be initialized from a trained Vanilla checkpoint, but "
        f"{args.vanilla_checkpoint!r} does not exist. Run "
        f"`python train.py --method vanilla` first."
    )
    backbone, head, best_state, history = proser.train_proser(
        cfg, args.vanilla_checkpoint, prepare_data_fn=prepare_data,
        build_model_bundle_fn=build_model_bundle, evaluate_accuracy_fn=evaluate_accuracy,
        device=DEVICE,
    )
    torch.save({"backbone": backbone.state_dict(), "head": head.state_dict(),
                "num_dummy": head.num_dummy,
                "best_state_meta": {k: v for k, v in best_state.items()
                                     if k not in ("backbone", "head")}},
               "checkpoints/proser.pt")
    with open("results/train_history_proser.json", "w") as f:
        json.dump(history, f, indent=2)
    print("Saved checkpoints/proser.pt and results/train_history_proser.json")


if __name__ == "__main__":
    main()
