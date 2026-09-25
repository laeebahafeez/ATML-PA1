"""
THE ONLY SCRIPT IN TASK 3 THAT LOADS SKETCH. Run this only after every
training run, checkpoint, and controlled-study setting below is finished
and frozen -- per the assignment, Task 3's learning procedure, source-side
diagnostics, checkpoint selection, and hyperparameter selection may never
see a Sketch image, and Task 2's Sketch results must not be used to revise
any Task 3 decision.

Prerequisite (run once each in task3/, in order):
    python train.py --method erm       # loads Task 2's Source-only checkpoint
    python train.py --method dan_dg
    python train.py --method sam
    # controlled study (config's controlled_study.method decides which of
    # these two sweeps this script looks for):
    python train.py --method dan_dg --lambda_dg 0.1  --tag _lam0.1
    python train.py --method dan_dg --lambda_dg 10.0 --tag _lam10.0
    # -- or, if controlled_study.method is "sam" instead --
    python train.py --method sam --rho 0.01 --tag _rho0.01
    python train.py --method sam --rho 0.1  --tag _rho0.1

Then:
    python evaluate_sketch.py --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "evaluation"))

from pacs import locate_pacs_domains, load_domain_pair, EXPECTED_CLASSES  # noqa: E402
from backbone import ResNet18Backbone  # noqa: E402
from classifier_head import ClassifierHead  # noqa: E402
from domain_metrics import evaluate_source_domains, evaluate_sketch, predict_on_dataset  # noqa: E402
from source_domain_separability import source_domain_separability_score  # noqa: E402
from sharpness import build_fixed_sharpness_batch, compute_sharpness  # noqa: E402
from class_analysis import per_class_comparison, top_confusions  # noqa: E402

from train import prepare_source_only_data, build_model_bundle  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAIN_METHODS = ["erm", "dan_dg", "sam"]
DISPLAY_NAMES = {"erm": "ERM", "dan_dg": "DAN-DG", "sam": "SAM"}


def load_checkpoint_bundle(name: str, cfg: dict):
    path = f"checkpoints/{name}.pt"
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, map_location=DEVICE)
    backbone, head = build_model_bundle(cfg)
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
    class_names = EXPECTED_CLASSES

    # ---- the ONE place Sketch is loaded in all of Task 3 ----
    paths = locate_pacs_domains(cfg["dataset"]["root"])
    _, sketch_eval_ds = load_domain_pair(
        paths[cfg["dataset"]["target_domain"]],
        cfg["dataset"]["resize_size"], cfg["dataset"]["image_size"],
    )

    source_subsets = prepare_source_only_data(cfg)
    source_val = {d: source_subsets[d]["val"] for d in source_domains}
    sharp_images, sharp_labels = build_fixed_sharpness_batch(
        source_val, seed=cfg["sharpness"]["seed"], n_per_domain=cfg["sharpness"]["n_per_domain"],
    )

    # ---- main comparison table ----
    rows = []
    results_by_method = {}
    for name in MAIN_METHODS:
        bundle = load_checkpoint_bundle(name, cfg)
        if bundle is None:
            print(f"WARNING: checkpoints/{name}.pt not found, skipping {name}.")
            continue
        backbone, head = bundle
        src_result = evaluate_source_domains(backbone, head, source_val, DEVICE)
        sketch_result = evaluate_sketch(backbone, head, sketch_eval_ds, DEVICE)
        results_by_method[name] = {"src": src_result, "sketch": sketch_result}

        feats_by_domain = {}
        for d in source_domains:
            _, _, feats = predict_on_dataset(backbone, head, source_val[d], DEVICE)
            feats_by_domain[d] = feats
        sep = source_domain_separability_score(
            feats_by_domain,
            seed=cfg["source_domain_separability"]["seed"],
            test_fraction=cfg["source_domain_separability"]["test_fraction"],
            C=cfg["source_domain_separability"]["logistic_regression_C"],
        )
        sharp = compute_sharpness(backbone, head, sharp_images, sharp_labels, DEVICE,
                                   rho=cfg["sharpness"]["rho"])

        row = {"method": DISPLAY_NAMES[name]}
        for d in source_domains:
            row[f"{d}_accuracy"] = src_result["per_domain"][d]["accuracy"]
            row[f"{d}_macro_f1"] = src_result["per_domain"][d]["macro_f1"]
        row["mean_source_accuracy"] = src_result["mean_accuracy"]
        row["mean_source_macro_f1"] = src_result["mean_macro_f1"]
        row["worst_source_accuracy"] = src_result["worst_accuracy"]
        row["worst_source_macro_f1"] = src_result["worst_macro_f1"]
        row["sketch_accuracy"] = sketch_result["accuracy"]
        row["sketch_macro_f1"] = sketch_result["macro_f1"]
        row["source_domain_separability"] = sep["held_out_accuracy"]
        row["delta_sharp"] = sharp["delta_sharp"]
        rows.append(row)

    main_table = pd.DataFrame(rows)
    if "erm" in results_by_method and "sketch_accuracy" in main_table.columns:
        baseline_acc = main_table.loc[main_table.method == "ERM", "sketch_accuracy"].iloc[0]
        main_table["sketch_accuracy_change_vs_erm"] = main_table["sketch_accuracy"] - baseline_acc
    main_table.to_csv("results/main_comparison.csv", index=False)
    print("=== Main comparison (ERM / DAN-DG / SAM) ===")
    print(main_table)

    # ---- per-class Sketch analysis vs. ERM ----
    if "erm" in results_by_method:
        baseline = results_by_method["erm"]["sketch"]
        for name in ["dan_dg", "sam"]:
            if name not in results_by_method:
                continue
            res = results_by_method[name]["sketch"]
            df = per_class_comparison(
                res["preds"], res["labels"], baseline["preds"], baseline["labels"],
                class_names, DISPLAY_NAMES[name],
            )
            df.to_csv(f"results/per_class_{name}_vs_erm.csv")
            print(f"\n=== Per-class Sketch accuracy: {DISPLAY_NAMES[name]} vs ERM ===")
            print(df)

            most_degraded = df["change"].idxmin()
            most_improved = df["change"].idxmax()
            for cls in {most_degraded, most_improved}:
                conf = top_confusions(res["preds"], res["labels"], class_names, cls)
                conf.to_csv(f"results/confusions_{name}_{cls}.csv", index=False)
                print(f"-- top confusions for class '{cls}' under {DISPLAY_NAMES[name]} --")
                print(conf)

    # ---- controlled design study ----
    method = cfg["controlled_study"]["method"]
    print(f"\n=== Controlled study: {method} sweep ===")
    sweep_rows = []
    if method == "dan_dg":
        main_value = cfg["mmd_dg"]["lambda_dg_main"]
        values = cfg["controlled_study"]["lambda_dg_values"]
        param_key, tag_prefix = "lambda_dg", "_lam"
    else:  # sam
        main_value = cfg["sam"]["rho_main"]
        values = cfg["controlled_study"]["sam_rho_values"]
        param_key, tag_prefix = "rho", "_rho"

    for value in values:
        tag = f"{tag_prefix}{value}"
        name = method if value == main_value else f"{method}{tag}"
        bundle = load_checkpoint_bundle(name, cfg)
        if bundle is None:
            print(f"WARNING: checkpoints/{name}.pt not found for {param_key}={value}, skipping.")
            continue
        backbone, head = bundle
        src_result = evaluate_source_domains(backbone, head, source_val, DEVICE)
        sketch_result = evaluate_sketch(backbone, head, sketch_eval_ds, DEVICE)
        feats_by_domain = {}
        for d in source_domains:
            _, _, feats = predict_on_dataset(backbone, head, source_val[d], DEVICE)
            feats_by_domain[d] = feats
        sep = source_domain_separability_score(
            feats_by_domain, seed=cfg["source_domain_separability"]["seed"],
            test_fraction=cfg["source_domain_separability"]["test_fraction"],
            C=cfg["source_domain_separability"]["logistic_regression_C"],
        )
        sharp = compute_sharpness(backbone, head, sharp_images, sharp_labels, DEVICE,
                                   rho=cfg["sharpness"]["rho"])
        sweep_rows.append({
            param_key: value,
            "mean_source_accuracy": src_result["mean_accuracy"],
            "mean_source_macro_f1": src_result["mean_macro_f1"],
            "worst_source_macro_f1": src_result["worst_macro_f1"],
            "sketch_accuracy": sketch_result["accuracy"],
            "sketch_macro_f1": sketch_result["macro_f1"],
            "source_domain_separability": sep["held_out_accuracy"],
            "delta_sharp": sharp["delta_sharp"],
        })
    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(f"results/controlled_study_{method}.csv", index=False)
    print(sweep_df)

    print("\nDone. All Task 3 required evidence is in task3/results/.")


if __name__ == "__main__":
    main()
