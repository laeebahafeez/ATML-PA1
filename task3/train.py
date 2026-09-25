"""
Unified training entry point for Task 3.

Usage:
    python train.py --method erm       # loads Task 2's Source-only checkpoint, does NOT retrain
    python train.py --method dan_dg    # pairwise source-domain MMD alignment
    python train.py --method sam       # sharpness-aware minimization
    python train.py --method dan_dg --lambda_dg 0.1  --tag _lam0.1   (controlled study)
    python train.py --method sam      --rho 0.01     --tag _rho0.01  (controlled study)

Deliberately never imports or loads anything from the Sketch domain: only
the three source domains are located and opened via shared/pacs.py's
lower-level load_domain_pair() (NOT load_all_domain_pairs(), which would
also instantiate a Sketch ImageFolder). Sketch is loaded in exactly one
place in this whole task -- evaluate_sketch.py -- and only after every
Task 3 config/checkpoint decision below has been made.
"""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "methods"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "evaluation"))

from pacs import locate_pacs_domains, load_domain_pair  # noqa: E402
from pacs_protocol import load_split, make_source_subsets, SourceTargetBatchIterator  # noqa: E402
from backbone import ResNet18Backbone, set_train_with_frozen_bn  # noqa: E402
from classifier_head import ClassifierHead  # noqa: E402
from domain_metrics import evaluate_source_domains  # noqa: E402
import erm  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def prepare_source_only_data(cfg: dict):
    """Loads ONLY the three source domains (never Sketch) and reuses Task 2's
    saved split verbatim -- does not recompute it, per the assignment's
    "reuse the same source splits across both tasks.\""""
    source_domains = cfg["dataset"]["source_domains"]
    paths = locate_pacs_domains(cfg["dataset"]["root"])  # confirms all 4 exist on disk;
    # does not open/read any Sketch image file -- see module docstring
    domain_pairs = {
        d: load_domain_pair(paths[d], cfg["dataset"]["resize_size"], cfg["dataset"]["image_size"])
        for d in source_domains
    }

    split_path = cfg["task2_paths"]["split_path"]
    if not os.path.exists(split_path):
        raise FileNotFoundError(
            f"Task 2's saved split not found at {split_path!r}. Task 3 must reuse "
            "Task 2's exact source train/val split rather than recomputing one -- "
            "run Task 2's train.py at least once first (it saves this file on its "
            "first run), or check task2_paths.split_path in configs/config.yaml."
        )
    split_record = load_split(split_path)["domains"]
    return make_source_subsets(domain_pairs, split_record)


def build_model_bundle(cfg: dict):
    backbone = ResNet18Backbone().to(DEVICE)
    head = ClassifierHead(cfg["backbone"]["feature_dim"], cfg["dataset"]["num_classes"]).to(DEVICE)
    return backbone, head


def train_one_method(method: str, cfg: dict, tag: str = "",
                      lambda_dg_override=None, rho_override=None):
    assert method in ("dan_dg", "sam"), "erm is handled directly in main() (load, not train)"
    torch.manual_seed(cfg["training"]["seed"])
    method_module = importlib.import_module(method)

    source_subsets = prepare_source_only_data(cfg)
    source_domains = cfg["dataset"]["source_domains"]

    batch_iter = SourceTargetBatchIterator(
        source_train_subsets={d: source_subsets[d]["train"] for d in source_domains},
        target_dataset=None,
        source_domains=source_domains,
        source_per_domain=cfg["training"]["source_per_domain_batch"],
        target_batch_size=0,
        seed=cfg["training"]["seed"],
        use_target=False,   # Task 3 NEVER draws a target/Sketch batch
    )
    val_datasets = {d: source_subsets[d]["val"] for d in source_domains}

    backbone, head = build_model_bundle(cfg)
    params = list(backbone.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg["training"]["lr"],
                                   weight_decay=cfg["training"]["weight_decay"])

    steps_per_epoch = max(
        len(source_subsets[d]["train"]) // cfg["training"]["source_per_domain_batch"]
        for d in source_domains
    )
    steps_per_epoch = max(steps_per_epoch, 1)
    max_epochs = cfg["training"]["max_epochs"]

    best_mean_f1 = -1.0
    best_state = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(max_epochs):
        set_train_with_frozen_bn(backbone)
        head.train()

        epoch_logs = []
        for _ in range(steps_per_epoch):
            images, labels, domain_ids, _n_source = next(batch_iter)
            images, labels, domain_ids = images.to(DEVICE), labels.to(DEVICE), domain_ids.to(DEVICE)

            if method == "dan_dg":
                logs = method_module.train_step(
                    backbone, head, images, labels, domain_ids, cfg, optimizer,
                    lambda_dg_override=lambda_dg_override,
                )
            else:  # sam
                logs = method_module.train_step(
                    backbone, head, images, labels, domain_ids, cfg, optimizer,
                    rho_override=rho_override,
                )
            epoch_logs.append(logs)

        val_result = evaluate_source_domains(backbone, head, val_datasets, DEVICE)
        mean_f1 = val_result["mean_macro_f1"]

        mean_logs = {k: float(sum(l[k] for l in epoch_logs if k in l) / len(epoch_logs))
                     for k in epoch_logs[0] if isinstance(epoch_logs[0][k], (int, float))}
        history.append({
            "epoch": epoch, "mean_source_macro_f1": mean_f1,
            "worst_source_macro_f1": val_result["worst_macro_f1"],
            "per_domain": val_result["per_domain"], **mean_logs,
        })
        print(f"[{method}{tag}] epoch {epoch:02d}  mean_source_macro_f1={mean_f1:.4f}  "
              f"worst_source_macro_f1={val_result['worst_macro_f1']:.4f}  train_logs={mean_logs}")

        if mean_f1 > best_mean_f1:
            best_mean_f1 = mean_f1
            best_state = {
                "backbone": copy.deepcopy(backbone.state_dict()),
                "head": copy.deepcopy(head.state_dict()),
                "epoch": epoch,
                "mean_source_macro_f1": mean_f1,
                "val_result": val_result,
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
    return backbone, head, best_state, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["erm", "dan_dg", "sam"])
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--tag", default="")
    parser.add_argument("--lambda_dg", type=float, default=None,
                         help="override for the DAN-DG lambda_dg controlled study")
    parser.add_argument("--rho", type=float, default=None,
                         help="override for the SAM rho controlled study")
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    if args.method == "erm":
        ckpt = erm.load_source_only_checkpoint(
            cfg["task2_paths"]["source_only_checkpoint"], map_location=DEVICE)
        backbone, head = build_model_bundle(cfg)
        backbone.load_state_dict(ckpt["backbone"])
        head.load_state_dict(ckpt["head"])
        backbone.eval()
        head.eval()

        source_subsets = prepare_source_only_data(cfg)
        val_datasets = {d: source_subsets[d]["val"] for d in cfg["dataset"]["source_domains"]}
        val_result = evaluate_source_domains(backbone, head, val_datasets, DEVICE)

        ckpt_name = f"erm{args.tag}"
        torch.save({"backbone": backbone.state_dict(), "head": head.state_dict(),
                    "val_result": val_result}, f"checkpoints/{ckpt_name}.pt")
        with open(f"results/train_history_{ckpt_name}.json", "w") as f:
            json.dump({
                "note": "ERM is Task 2's Source-only checkpoint, loaded unchanged (not retrained), "
                        "per the assignment's explicit instruction.",
                "source_checkpoint": cfg["task2_paths"]["source_only_checkpoint"],
                "val_result": val_result,
            }, f, indent=2)
        print(f"Loaded Task 2's Source-only checkpoint as the Task 3 ERM baseline. "
              f"Saved checkpoints/{ckpt_name}.pt")
        return

    backbone, head, best_state, history = train_one_method(
        args.method, cfg, tag=args.tag,
        lambda_dg_override=args.lambda_dg, rho_override=args.rho,
    )
    ckpt_name = f"{args.method}{args.tag}"
    torch.save({
        "backbone": backbone.state_dict(), "head": head.state_dict(),
        "best_state_meta": {k: v for k, v in best_state.items() if k not in ("backbone", "head")},
    }, f"checkpoints/{ckpt_name}.pt")
    with open(f"results/train_history_{ckpt_name}.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved checkpoints/{ckpt_name}.pt and results/train_history_{ckpt_name}.json")


if __name__ == "__main__":
    main()
