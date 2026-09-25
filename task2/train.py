"""
Unified training loop for Task 2. Every method (source_only, dan, dann,
cdan) runs through this exact same loop -- only the per-step loss
function differs (methods/<name>.py::compute_loss) -- per the assignment's
"all methods must run through the same training and evaluation pipeline."

Usage:
    python train.py --method source_only --config configs/config.yaml
    python train.py --method dan          --config configs/config.yaml
    python train.py --method dann         --config configs/config.yaml
    python train.py --method cdan         --config configs/config.yaml
    python train.py --method dan --lambda_mmd 0.1 --tag lam0.1   (controlled study)
"""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "methods"))

from pacs_protocol import (  # noqa: E402
    SourceTargetBatchIterator, build_source_splits, load_split,
    make_domain_loader, make_source_subsets, save_split,
)
from backbone import ResNet18Backbone, set_train_with_frozen_bn  # noqa: E402
from classifier_head import ClassifierHead  # noqa: E402
from domain_discriminator import DomainDiscriminator  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

METHODS_NEEDING_TARGET = {"dan", "dann", "cdan"}
METHODS_NEEDING_DISCRIMINATOR = {"dann", "cdan"}


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def prepare_data(cfg: dict):
    split_path = cfg["split"]["save_path"]
    domain_pairs, split_record = build_source_splits(
        cfg["dataset"]["source_domains"], cfg["dataset"]["root"],
        cfg["dataset"]["resize_size"], cfg["dataset"]["image_size"],
        cfg["split"]["val_fraction"], cfg["split"]["seed"],
    )
    if not os.path.exists(split_path):
        save_split(split_record, split_path, cfg["split"]["seed"], cfg["split"]["val_fraction"])
    else:
        # Reuse the already-saved split verbatim (e.g. if this is a second
        # method run in the same session, or Task 3 loading Task 2's split).
        split_record = load_split(split_path)["domains"]

    source_subsets = make_source_subsets(domain_pairs, split_record)
    target_train_ds, target_eval_ds = domain_pairs[cfg["dataset"]["target_domain"]]
    return source_subsets, target_train_ds, target_eval_ds


def build_model_bundle(method: str, cfg: dict):
    backbone = ResNet18Backbone().to(DEVICE)
    head = ClassifierHead(cfg["backbone"]["feature_dim"], cfg["dataset"]["num_classes"]).to(DEVICE)
    discriminator = None
    if method in METHODS_NEEDING_DISCRIMINATOR:
        if method == "dann":
            in_dim = cfg["backbone"]["feature_dim"]
            hidden = cfg["dann"]["discriminator_hidden"]
            dropout = cfg["dann"]["discriminator_dropout"]
        else:  # cdan
            in_dim = cfg["backbone"]["feature_dim"] * cfg["dataset"]["num_classes"]
            hidden = cfg["cdan"]["discriminator_hidden"]
            dropout = cfg["cdan"]["discriminator_dropout"]
        discriminator = DomainDiscriminator(in_dim, hidden, dropout).to(DEVICE)
    return backbone, head, discriminator


@torch.no_grad()
def evaluate_on_loader(backbone, head, loader: DataLoader):
    backbone.eval()
    head.eval()
    all_preds, all_labels = [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = head(backbone(x))
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_labels.append(y.cpu().numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    acc = float((preds == labels).mean())
    macro_f1 = float(f1_score(labels, preds, average="macro"))
    return acc, macro_f1, preds, labels


def train_one_method(method: str, cfg: dict, tag: str = "", lambda_mmd_override=None,
                      max_grad_reversal_override=None, source_only_checkpoint=None):
    """
    Trains `method` and returns the trained (backbone, head, discriminator)
    plus a dict of per-source-domain (acc, macro_f1) validation results
    logged at the best (early-stopped) checkpoint.

    `source_only_checkpoint`: if given (a state dict), used to INITIALIZE
    dan/dann/cdan from the same starting point as source-only rather than a
    fresh ImageNet init -- NOT used by default, since the assignment
    specifies each method trains from the standard ImageNet initialization;
    kept as an optional hook, unused in the main comparison.
    """
    torch.manual_seed(cfg["training"]["seed"])
    method_module = importlib.import_module(method)

    source_subsets, target_train_ds, target_eval_ds = prepare_data(cfg)
    source_domains = cfg["dataset"]["source_domains"]

    use_target = method in METHODS_NEEDING_TARGET
    batch_iter = SourceTargetBatchIterator(
        source_train_subsets={d: source_subsets[d]["train"] for d in source_domains},
        target_dataset=target_train_ds,
        source_domains=source_domains,
        source_per_domain=cfg["training"]["source_per_domain_batch"],
        target_batch_size=cfg["training"]["target_batch_size"],
        seed=cfg["training"]["seed"],
        use_target=use_target,
    )

    val_loaders = {
        d: make_domain_loader(source_subsets[d]["val"], batch_size=64,
                               seed=cfg["training"]["seed"], shuffle=False)
        for d in source_domains
    }

    backbone, head, discriminator = build_model_bundle(method, cfg)
    if source_only_checkpoint is not None:
        backbone.load_state_dict(source_only_checkpoint["backbone"])
        head.load_state_dict(source_only_checkpoint["head"])

    params = list(backbone.parameters()) + list(head.parameters())
    if discriminator is not None:
        params += list(discriminator.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg["training"]["lr"],
                                   weight_decay=cfg["training"]["weight_decay"])

    # steps_per_epoch: enough steps for the LARGEST source-domain training
    # split to be fully seen once per epoch (smaller domains simply cycle
    # more than once) -- a design choice documented in task2/README.md,
    # since the assignment does not fix an exact definition.
    steps_per_epoch = max(
        len(source_subsets[d]["train"]) // cfg["training"]["source_per_domain_batch"]
        for d in source_domains
    )
    steps_per_epoch = max(steps_per_epoch, 1)
    max_epochs = cfg["training"]["max_epochs"]
    total_planned_steps = steps_per_epoch * max_epochs

    # allow the controlled design study to override method-specific knobs
    # without touching the main cfg dict used elsewhere
    run_cfg = copy.deepcopy(cfg)
    if lambda_mmd_override is not None:
        run_cfg["mmd"]["lambda_mmd_main"] = lambda_mmd_override
    if max_grad_reversal_override is not None:
        run_cfg["dann"]["max_grad_reversal_strength_main"] = max_grad_reversal_override

    best_mean_f1 = -1.0
    best_state = None
    epochs_without_improvement = 0
    history = []
    global_step = 0

    for epoch in range(max_epochs):
        set_train_with_frozen_bn(backbone)
        head.train()
        if discriminator is not None:
            discriminator.train()

        epoch_logs = []
        for _ in range(steps_per_epoch):
            images, labels, domain_ids, n_source = next(batch_iter)
            images, labels, domain_ids = images.to(DEVICE), labels.to(DEVICE), domain_ids.to(DEVICE)
            progress = min(global_step / max(total_planned_steps, 1), 1.0)

            optimizer.zero_grad()
            loss, logs = method_module.compute_loss(
                backbone, head, images, labels, domain_ids, n_source,
                progress, run_cfg, discriminator=discriminator,
            )
            if not torch.isfinite(loss):
                # A second, independent safety net alongside gradient clipping
                # below: if a single batch's loss is already non-finite before
                # backward() even runs (observed during DANN/CDAN's adversarial
                # training when the discriminator's logits blow up), skip this
                # step entirely rather than let a corrupted gradient reach the
                # optimizer. Applied identically to every method for pipeline
                # consistency; a no-op for source_only/DAN, whose losses never
                # exhibited this failure mode.
                print(f"  [warning] non-finite loss ({loss.item()}) at global_step "
                      f"{global_step} -- skipping this batch's update")
                global_step += 1
                continue
            loss.backward()
            # Gradient clipping, applied identically for every method (source_only,
            # DAN, DANN, CDAN) to keep the training pipeline uniform per the
            # assignment's "keep ... optimizer ... fixed across methods." DANN and
            # CDAN's adversarial objective can produce very large gradients once the
            # domain discriminator becomes highly confident (observed empirically:
            # domain_loss spiking into the hundreds and destabilizing the backbone
            # for several subsequent epochs); clipping the global gradient norm is
            # the standard remedy for this class of adversarial-training instability.
            # source_only and DAN's gradients are already well within this bound in
            # practice, so clipping is a no-op for them and does not change their
            # results -- this is purely a DANN/CDAN stability fix applied uniformly.
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            logs["total_loss"] = loss.item()
            epoch_logs.append(logs)
            global_step += 1

        # ---- validation / checkpoint selection ----
        per_domain = {}
        for d in source_domains:
            acc, mf1, _, _ = evaluate_on_loader(backbone, head, val_loaders[d])
            per_domain[d] = {"accuracy": acc, "macro_f1": mf1}
        mean_f1 = float(np.mean([per_domain[d]["macro_f1"] for d in source_domains]))

        # extremely unlikely, but if every single batch this epoch was skipped
        # for a non-finite loss, epoch_logs is empty -- fall back to an empty
        # dict rather than crash on epoch_logs[0]
        mean_logs = ({k: float(np.mean([l[k] for l in epoch_logs if k in l]))
                      for k in epoch_logs[0] if isinstance(epoch_logs[0][k], (int, float))}
                     if epoch_logs else {})
        history.append({"epoch": epoch, "mean_source_macro_f1": mean_f1,
                         "per_domain": per_domain, **mean_logs})
        print(f"[{method}{tag}] epoch {epoch:02d}  mean_source_macro_f1={mean_f1:.4f}  "
              f"train_logs={mean_logs}")

        if mean_f1 > best_mean_f1:
            best_mean_f1 = mean_f1
            best_state = {
                "backbone": copy.deepcopy(backbone.state_dict()),
                "head": copy.deepcopy(head.state_dict()),
                "discriminator": copy.deepcopy(discriminator.state_dict()) if discriminator else None,
                "epoch": epoch,
                "mean_source_macro_f1": mean_f1,
                "per_domain": per_domain,
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg["training"]["early_stop_patience"]:
                print(f"[{method}{tag}] early stopping at epoch {epoch} "
                      f"(best={best_mean_f1:.4f} at epoch {best_state['epoch']})")
                break

    backbone.load_state_dict(best_state["backbone"])
    head.load_state_dict(best_state["head"])
    if discriminator is not None and best_state["discriminator"] is not None:
        discriminator.load_state_dict(best_state["discriminator"])

    return backbone, head, discriminator, best_state, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True,
                         choices=["source_only", "dan", "dann", "cdan"])
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--tag", default="")
    parser.add_argument("--lambda_mmd", type=float, default=None,
                         help="override for the DAN lambda_MMD controlled study")
    parser.add_argument("--max_grad_reversal", type=float, default=None,
                         help="override for the DANN strength controlled study")
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    source_only_ckpt = None
    backbone, head, discriminator, best_state, history = train_one_method(
        args.method, cfg, tag=args.tag,
        lambda_mmd_override=args.lambda_mmd,
        max_grad_reversal_override=args.max_grad_reversal,
    )

    ckpt_name = f"{args.method}{args.tag}"
    torch.save(
        {"backbone": backbone.state_dict(), "head": head.state_dict(),
         "discriminator": discriminator.state_dict() if discriminator else None,
         "best_state_meta": {k: v for k, v in best_state.items()
                              if k not in ("backbone", "head", "discriminator")}},
        f"checkpoints/{ckpt_name}.pt",
    )
    with open(f"results/train_history_{ckpt_name}.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved checkpoints/{ckpt_name}.pt and results/train_history_{ckpt_name}.json")


if __name__ == "__main__":
    main()
