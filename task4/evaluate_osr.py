"""
Task 4's required-evidence builder. Reads ONLY the cached features/logits
produced by extract_outputs.py (cache/*.npz) -- this is where every score,
AUROC, and threshold actually gets computed and is therefore the one place
where the assignment's "no CIFAR-100 image may influence ... score
definitions, or threshold selection" is enforced by construction: this script
never trains anything or touches a raw image, only the frozen numeric
features/logits already extracted from already-selected checkpoints.

Produces, in results/:
  - scores_comparison_vanilla.csv       (MSP/MLS/Energy/Mahalanobis x AUROC + calibrated rejection)
  - models_comparison.csv               (Vanilla/GCSC/PROSER x CSA + near/far OSR, MLS + PROSER-placeholder row)
  - score_distributions.png             (compact 3-panel figure: MSP, MLS, Mahalanobis)
  - failures_near.csv, failures_far.csv (>=3 incorrectly-accepted examples each, under the vanilla MLS threshold)
  - confusions_near.csv, confusions_far.csv

Usage:
    python evaluate_osr.py
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scores"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "evaluation"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "methods"))

import msp as msp_score  # noqa: E402
import mls as mls_score  # noqa: E402
import energy as energy_score  # noqa: E402
import mahalanobis as mahalanobis_score  # noqa: E402
from metrics import all_auroc_variants  # noqa: E402
from thresholds import calibrated_report, calibrate_threshold  # noqa: E402
from failure_analysis import incorrectly_accepted, summarize_confusions, CIFAR10_CLASSES  # noqa: E402
from proser import placeholder_detection_score  # noqa: E402


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_cache(method: str, split: str, cache_dir: str) -> dict:
    path = os.path.join(cache_dir, f"{method}_{split}.npz")
    assert os.path.exists(path), (
        f"{path!r} not found -- run `python extract_outputs.py --method {method}` first."
    )
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def closed_set_accuracy(logits: np.ndarray, labels: np.ndarray, num_classes: int) -> float:
    preds = logits[:, :num_classes].argmax(axis=1)
    return float((preds == labels).mean())


SCORE_FNS = {"msp": msp_score.score, "mls": mls_score.score, "energy": energy_score.score}


def compute_all_post_hoc_scores(logits: np.ndarray) -> dict:
    return {name: fn(logits) for name, fn in SCORE_FNS.items()}


def main():
    cfg = load_config("configs/config.yaml")
    cache_dir = cfg["evaluation"]["cache_path"]
    results_dir = cfg["evaluation"]["save_path"]
    os.makedirs(results_dir, exist_ok=True)
    num_classes = cfg["dataset"]["num_classes"]
    percentile = cfg["evaluation"]["threshold_percentile"]
    diag_eps = cfg["mahalanobis"]["diag_eps"]

    # ================= Table 1: MSP / MLS / Energy / Mahalanobis on frozen Vanilla =================
    van_train = load_cache("vanilla", "train", cache_dir)
    van_val = load_cache("vanilla", "val", cache_dir)
    van_test = load_cache("vanilla", "test", cache_dir)
    van_near = load_cache("vanilla", "near", cache_dir)
    van_far = load_cache("vanilla", "far", cache_dir)

    mahal_fit = mahalanobis_score.fit(van_train["features"], van_train["labels"], diag_eps)

    def vanilla_scores(split_data: dict) -> dict:
        s = compute_all_post_hoc_scores(split_data["logits"])
        s["mahalanobis"] = mahalanobis_score.score(split_data["features"], mahal_fit)
        return s

    van_val_scores = vanilla_scores(van_val)
    van_test_scores = vanilla_scores(van_test)
    van_near_scores = vanilla_scores(van_near)
    van_far_scores = vanilla_scores(van_far)

    rows = []
    for score_name in ["msp", "mls", "energy", "mahalanobis"]:
        auroc = all_auroc_variants(van_test_scores[score_name], van_near_scores[score_name],
                                    van_far_scores[score_name])
        calib = calibrated_report(van_val_scores[score_name], van_test_scores[score_name],
                                   van_near_scores[score_name], van_far_scores[score_name],
                                   percentile)
        rows.append({"score": score_name, **auroc, **calib})
    scores_comparison = pd.DataFrame(rows)
    scores_comparison.to_csv(os.path.join(results_dir, "scores_comparison_vanilla.csv"), index=False)
    print("=== Scores comparison (frozen Vanilla) ===")
    print(scores_comparison)

    # vanilla MLS threshold, reused below for the required failure analysis
    vanilla_mls_tau = calibrate_threshold(van_val_scores["mls"], percentile)

    # ================= Table 2: Vanilla / GCSC / PROSER, MLS (+ PROSER placeholder row) =================
    model_rows = []
    per_method_mls = {}
    for method in ["vanilla", "gcsc", "proser"]:
        val = load_cache(method, "val", cache_dir)
        test = load_cache(method, "test", cache_dir)
        near = load_cache(method, "near", cache_dir)
        far = load_cache(method, "far", cache_dir)

        csa = closed_set_accuracy(test["logits"], test["labels"], num_classes)

        val_mls = mls_score.score(val["logits"])
        test_mls = mls_score.score(test["logits"])
        near_mls = mls_score.score(near["logits"])
        far_mls = mls_score.score(far["logits"])
        per_method_mls[method] = {"val": val, "test": test, "near": near, "far": far}

        auroc = all_auroc_variants(test_mls, near_mls, far_mls)
        calib = calibrated_report(val_mls, test_mls, near_mls, far_mls, percentile)
        model_rows.append({"method": method.upper(), "score": "MLS",
                            "closed_set_accuracy": csa, **auroc, **calib})

        if method == "proser":
            val_ph = placeholder_detection_score(val["logits"], num_classes)
            test_ph = placeholder_detection_score(test["logits"], num_classes)
            near_ph = placeholder_detection_score(near["logits"], num_classes)
            far_ph = placeholder_detection_score(far["logits"], num_classes)
            auroc_ph = all_auroc_variants(test_ph, near_ph, far_ph)
            calib_ph = calibrated_report(val_ph, test_ph, near_ph, far_ph, percentile)
            model_rows.append({"method": "PROSER", "score": "placeholder",
                                "closed_set_accuracy": csa, **auroc_ph, **calib_ph})

    models_comparison = pd.DataFrame(model_rows)
    models_comparison.to_csv(os.path.join(results_dir, "models_comparison.csv"), index=False)
    print("\n=== Models comparison (Vanilla / GCSC / PROSER) ===")
    print(models_comparison)

    # ================= Compact 3-panel score-distribution figure (MSP, MLS, Mahalanobis; Vanilla) =================
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, score_name in zip(axes, ["msp", "mls", "mahalanobis"]):
        ax.hist(van_test_scores[score_name], bins=40, alpha=0.5, density=True, label="known (test)")
        ax.hist(van_near_scores[score_name], bins=40, alpha=0.5, density=True, label="near unknown")
        ax.hist(van_far_scores[score_name], bins=40, alpha=0.5, density=True, label="far unknown")
        ax.set_title(score_name.upper())
        ax.set_xlabel("unknownness score u(x)")
        ax.legend(fontsize=8)
    fig.suptitle("Vanilla model: score distributions, known vs. near/far unknown")
    fig.tight_layout()
    fig_path = os.path.join(results_dir, "score_distributions.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved {fig_path}")

    # ================= Failure analysis: vanilla MLS threshold, near + far =================
    near_fail = incorrectly_accepted(van_near["logits"], van_near_scores["mls"],
                                      van_near["fine_class_names"], vanilla_mls_tau,
                                      num_classes, top_n=10)
    far_fail = incorrectly_accepted(van_far["logits"], van_far_scores["mls"],
                                     van_far["fine_class_names"], vanilla_mls_tau,
                                     num_classes, top_n=10)
    near_fail.to_csv(os.path.join(results_dir, "failures_near.csv"), index=False)
    far_fail.to_csv(os.path.join(results_dir, "failures_far.csv"), index=False)
    summarize_confusions(near_fail).to_csv(os.path.join(results_dir, "confusions_near.csv"), index=False)
    summarize_confusions(far_fail).to_csv(os.path.join(results_dir, "confusions_far.csv"), index=False)

    print(f"\n=== Near-unknown failures under vanilla MLS (tau={vanilla_mls_tau:.4f}) ===")
    print(near_fail if len(near_fail) else "(none accepted at this threshold)")
    print(f"\n=== Far-unknown failures under vanilla MLS (tau={vanilla_mls_tau:.4f}) ===")
    print(far_fail if len(far_fail) else "(none accepted at this threshold)")

    print("\nDone. All Task 4 required evidence is in task4/results/.")


if __name__ == "__main__":
    main()
