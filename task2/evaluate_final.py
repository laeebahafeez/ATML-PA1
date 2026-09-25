"""
Common evaluation across Source-only, DAN, DANN, CDAN (+ the controlled
lambda_MMD study), run ONLY after every checkpoint has been produced by
train.py. This script never trains anything itself -- it loads frozen
checkpoints, so it can be re-run cheaply as often as needed while writing
the report.

Prerequisite (run once each, in order -- source_only first since dan/dann/
cdan don't strictly depend on it but the assignment's Task 3 does):
    python train.py --method source_only
    python train.py --method dan
    python train.py --method dann
    python train.py --method cdan
    # controlled design study (Task 2 Step 6, DAN lambda_MMD sweep):
    # tags must exactly match f"_lam{value}" for the given float, since
    # evaluate_final.py reconstructs the same string to find the checkpoint
    python train.py --method dan --lambda_mmd 0.1 --tag _lam0.1
    python train.py --method dan --lambda_mmd 10.0 --tag _lam10.0

Then:
    python evaluate_final.py --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "evaluation"))

from pacs_protocol import build_source_splits, load_split, make_source_subsets  # noqa: E402
from backbone import ResNet18Backbone  # noqa: E402
from classifier_head import ClassifierHead  # noqa: E402
from metrics import evaluate_method_on_all_domains, predict_on_dataset  # noqa: E402
from domain_separability import domain_separability_score  # noqa: E402
from class_analysis import per_class_comparison, top_confusions  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAIN_METHODS = ["source_only", "dan", "dann", "cdan"]
DISPLAY_NAMES = {"source_only": "Source-only", "dan": "DAN", "dann": "DANN", "cdan": "CDAN"}


def load_checkpoint_bundle(name: str, cfg: dict):
    path = f"checkpoints/{name}.pt"
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, map_location=DEVICE)
    backbone = ResNet18Backbone().to(DEVICE)
    head = ClassifierHead(cfg["backbone"]["feature_dim"], cfg["dataset"]["num_classes"]).to(DEVICE)
    backbone.load_state_dict(ckpt["backbone"])
    head.load_state_dict(ckpt["head"])
    backbone.eval()
    head.eval()
    return backbone, head


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    os.makedirs("results", exist_ok=True)
    source_domains = cfg["dataset"]["source_domains"]
    class_names = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]

    domain_pairs, split_record = build_source_splits(
        source_domains, cfg["dataset"]["root"], cfg["dataset"]["resize_size"],
        cfg["dataset"]["image_size"], cfg["split"]["val_fraction"], cfg["split"]["seed"],
    )
    split_record = load_split(cfg["split"]["save_path"])["domains"]
    source_subsets = make_source_subsets(domain_pairs, split_record)
    source_val = {d: source_subsets[d]["val"] for d in source_domains}
    _, target_eval_ds = domain_pairs[cfg["dataset"]["target_domain"]]

    # ---- Main comparison table ------------------------------------------
    rows = []
    results_by_method = {}
    for name in MAIN_METHODS:
        bundle = load_checkpoint_bundle(name, cfg)
        if bundle is None:
            print(f"WARNING: checkpoints/{name}.pt not found, skipping {name}.")
            continue
        backbone, head = bundle
        res = evaluate_method_on_all_domains(backbone, head, source_val, target_eval_ds, DEVICE)
        results_by_method[name] = res

        src_val_feats = np.concatenate(
            [predict_on_dataset(backbone, head, source_val[d], DEVICE)[2] for d in source_domains]
        )
        sep = domain_separability_score(
            src_val_feats, res["target_features"],
            seed=cfg["domain_separability"]["seed"],
            test_fraction=cfg["domain_separability"]["test_fraction"],
            C=cfg["domain_separability"]["logistic_regression_C"],
        )

        row = {"method": DISPLAY_NAMES[name]}
        for d in source_domains:
            row[f"{d}_accuracy"] = res["per_source"][d]["accuracy"]
            row[f"{d}_macro_f1"] = res["per_source"][d]["macro_f1"]
        row["mean_source_accuracy"] = res["mean_source_accuracy"]
        row["mean_source_macro_f1"] = res["mean_source_macro_f1"]
        row["target_accuracy"] = res["target_accuracy"]
        row["target_macro_f1"] = res["target_macro_f1"]
        row["domain_separability"] = sep["held_out_accuracy"]
        rows.append(row)

    main_table = pd.DataFrame(rows)
    if "target_accuracy" in main_table.columns and "source_only" in results_by_method:
        baseline_acc = main_table.loc[main_table.method == "Source-only", "target_accuracy"].iloc[0]
        main_table["target_accuracy_change_vs_source_only"] = main_table["target_accuracy"] - baseline_acc
    main_table.to_csv("results/main_comparison.csv", index=False)
    print("=== Main comparison (Source-only / DAN / DANN / CDAN) ===")
    print(main_table)

    # ---- Per-class analysis vs. Source-only ------------------------------
    if "source_only" in results_by_method:
        baseline = results_by_method["source_only"]
        for name in ["dan", "dann", "cdan"]:
            if name not in results_by_method:
                continue
            res = results_by_method[name]
            df = per_class_comparison(
                res["target_preds"], res["target_labels"],
                baseline["target_preds"], baseline["target_labels"],
                class_names, DISPLAY_NAMES[name],
            )
            df.to_csv(f"results/per_class_{name}_vs_source_only.csv")
            print(f"\n=== Per-class target accuracy: {DISPLAY_NAMES[name]} vs Source-only ===")
            print(df)

            # dominant confusions for the single most-degraded and
            # most-improved class, as a starting point for failure analysis
            most_degraded = df["change"].idxmin()
            most_improved = df["change"].idxmax()
            for cls in {most_degraded, most_improved}:
                conf = top_confusions(res["target_preds"], res["target_labels"], class_names, cls)
                conf.to_csv(f"results/confusions_{name}_{cls}.csv", index=False)
                print(f"-- top confusions for class '{cls}' under {DISPLAY_NAMES[name]} --")
                print(conf)

    # ---- Controlled design study: DAN lambda_MMD sweep --------------------
    print("\n=== Controlled study: DAN lambda_MMD sweep ===")
    lam_rows = []
    for lam in cfg["controlled_study"]["lambda_mmd_values"]:
        tag = f"_lam{lam}"
        name = f"dan{tag}" if lam != cfg["mmd"]["lambda_mmd_main"] else "dan"
        bundle = load_checkpoint_bundle(name, cfg)
        if bundle is None:
            print(f"WARNING: checkpoints/{name}.pt not found for lambda_mmd={lam}, skipping.")
            continue
        backbone, head = bundle
        res = evaluate_method_on_all_domains(backbone, head, source_val, target_eval_ds, DEVICE)
        src_val_feats = np.concatenate(
            [predict_on_dataset(backbone, head, source_val[d], DEVICE)[2] for d in source_domains]
        )
        sep = domain_separability_score(
            src_val_feats, res["target_features"],
            seed=cfg["domain_separability"]["seed"],
            test_fraction=cfg["domain_separability"]["test_fraction"],
            C=cfg["domain_separability"]["logistic_regression_C"],
        )
        lam_rows.append({
            "lambda_mmd": lam,
            "mean_source_accuracy": res["mean_source_accuracy"],
            "mean_source_macro_f1": res["mean_source_macro_f1"],
            "target_accuracy": res["target_accuracy"],
            "target_macro_f1": res["target_macro_f1"],
            "domain_separability": sep["held_out_accuracy"],
        })
    lam_df = pd.DataFrame(lam_rows)
    lam_df.to_csv("results/controlled_study_dan_lambda_mmd.csv", index=False)
    print(lam_df)

    print("\nDone. All Task 2 required evidence is in task2/results/.")


if __name__ == "__main__":
    main()
