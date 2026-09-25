"""
Extract and cache penultimate features + logits for a frozen, already-selected
checkpoint (vanilla / gcsc / proser), over CIFAR-10 train/val/test and the
fixed CIFAR-100 near/far unknown groups.

Per the assignment's Step 1: "Freeze the selected checkpoint and extract its
penultimate feature f(x) and logits z(x) ... for the CIFAR-10 training,
validation, and test examples and the fixed CIFAR-100 evaluation examples."
This is pure forward-pass caching against an already-frozen checkpoint -- it
does not train, select, or threshold anything, so loading CIFAR-100 images
here does not violate "no CIFAR-100 image may influence training, checkpoint
selection, score definition, or threshold selection" (those decisions were
already made before this script ever runs). Score computation, AUROC, and
threshold calibration all happen later, in evaluate_osr.py.

Usage:
    python extract_outputs.py --method vanilla
    python extract_outputs.py --method gcsc
    python extract_outputs.py --method proser
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))

from cifar10 import load_cifar10, build_eval_transform, CIFAR10_CLASSES  # noqa: E402
from cifar100_unknowns import load_unknown_groups  # noqa: E402
from make_splits import get_or_build_split  # noqa: E402
from resnet_cifar import ResNet18Cifar  # noqa: E402
from classifier_head import ClassifierHead  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def extract_known(backbone, head, loader) -> dict:
    backbone.eval()
    head.eval()
    feats_all, logits_all, labels_all = [], [], []
    for x, y in loader:
        x = x.to(DEVICE)
        f = backbone(x)
        z = head(f)
        feats_all.append(f.cpu().numpy())
        logits_all.append(z.cpu().numpy())
        labels_all.append(y.numpy())
    return {
        "features": np.concatenate(feats_all),
        "logits": np.concatenate(logits_all),
        "labels": np.concatenate(labels_all),
    }


@torch.no_grad()
def extract_unknown(backbone, head, dataset) -> dict:
    """dataset yields (image, fine_class_name) -- see cifar100_unknowns.py."""
    backbone.eval()
    head.eval()
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=2,
                         collate_fn=lambda batch: (
                             torch.stack([b[0] for b in batch]), [b[1] for b in batch]))
    feats_all, logits_all, names_all = [], [], []
    for x, names in loader:
        x = x.to(DEVICE)
        f = backbone(x)
        z = head(f)
        feats_all.append(f.cpu().numpy())
        logits_all.append(z.cpu().numpy())
        names_all.extend(names)
    return {
        "features": np.concatenate(feats_all),
        "logits": np.concatenate(logits_all),
        "fine_class_names": np.array(names_all),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["vanilla", "gcsc", "proser"])
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.makedirs(cfg["evaluation"]["cache_path"], exist_ok=True)

    ckpt_path = f"checkpoints/{args.method}.pt"
    assert os.path.exists(ckpt_path), f"{ckpt_path!r} not found -- train this method first."
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    num_dummy = ckpt.get("num_dummy", 0)

    backbone = ResNet18Cifar().to(DEVICE)
    head = ClassifierHead(cfg["backbone"]["feature_dim"], cfg["dataset"]["num_classes"],
                           num_dummy=num_dummy).to(DEVICE)
    backbone.load_state_dict(ckpt["backbone"])
    head.load_state_dict(ckpt["head"])

    eval_transform = build_eval_transform()
    _, train_eval_ds, test_ds = load_cifar10(
        root=cfg["dataset"]["root"], train_transform=eval_transform,
        eval_transform=eval_transform, download=True)
    train_idx, val_idx = get_or_build_split(
        train_eval_ds.targets, cfg["split"]["save_path"],
        cfg["split"]["seed"], cfg["split"]["val_fraction"])
    train_subset = Subset(train_eval_ds, train_idx)  # UNAUGMENTED -- required for Mahalanobis fit
    val_subset = Subset(train_eval_ds, val_idx)

    train_loader = DataLoader(train_subset, batch_size=256, shuffle=False, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=256, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)

    near_ds, far_ds = load_unknown_groups(root=cfg["dataset"]["root"], download=True)

    print(f"[{args.method}] extracting CIFAR-10 train (unaugmented)...")
    train_out = extract_known(backbone, head, train_loader)
    print(f"[{args.method}] extracting CIFAR-10 val...")
    val_out = extract_known(backbone, head, val_loader)
    print(f"[{args.method}] extracting CIFAR-10 test...")
    test_out = extract_known(backbone, head, test_loader)
    print(f"[{args.method}] extracting CIFAR-100 near-unknown group...")
    near_out = extract_unknown(backbone, head, near_ds)
    print(f"[{args.method}] extracting CIFAR-100 far-unknown group...")
    far_out = extract_unknown(backbone, head, far_ds)

    cache_dir = cfg["evaluation"]["cache_path"]
    for split_name, out in [("train", train_out), ("val", val_out), ("test", test_out),
                             ("near", near_out), ("far", far_out)]:
        path = os.path.join(cache_dir, f"{args.method}_{split_name}.npz")
        np.savez(path, **out)
        print(f"Saved {path}  (features={out['features'].shape}, logits={out['logits'].shape})")

    print(f"Done. CIFAR-10 classes (for reference): {CIFAR10_CLASSES}")


if __name__ == "__main__":
    main()
