"""
End-to-end Task 1 pipeline: extract features, train linear heads, run every
required intervention, and write all report tables + figures to results/.

Prerequisites (run once, in order):
    python data/make_subset.py --config configs/config.yaml
    python data/make_cue_conflicts.py --config configs/config.yaml
    (place task1/checkpoints/decoder.pth -- see README.md)

Then:
    python scripts/run_task1.py --config configs/config.yaml

This script requires a GPU-capable machine with internet access for the
first run (to download ImageNet/OpenAI-CLIP pretrained weights and the
Oxford-IIIT Pets dataset). It is organized as a sequence of clearly
labeled steps mirroring the assignment's Steps 1-6 so any one step can be
re-run independently by commenting out the others.
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
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))

from dataset import PetsSubset, collate_common  # noqa: E402
from transforms import to_grayscale, hue_rotate, translate_cardinal, patch_shuffle  # noqa: E402
from backbones import BACKBONE_NAMES, build_backbone  # noqa: E402
from linear_probe import ProbeConfig, train_linear_probe  # noqa: E402
from evaluate_bias import (  # noqa: E402
    clean_baseline_report, color_bias_report, shape_bias_report,
    translation_curve_report, patch_shuffle_report,
)
from representation import plot_clean_vs_transformed  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# Feature / logit extraction
# --------------------------------------------------------------------------- #

@torch.no_grad()
def extract(backbone, loader, transform_fn=None, image_id_aware=False):
    """Run a frozen backbone (+ head, if attached) over a loader, optionally
    applying `transform_fn` to each batch of un-normalized [0,1] images first.

    transform_fn signature is either f(x01) or, if image_id_aware, f(x01, ids)
    (needed for patch_shuffle, which must seed per dataset index).
    """
    all_feats, all_logits, all_labels, all_ids = [], [], [], []
    for x01, y, idx in loader:
        x01 = x01.to(DEVICE)
        if transform_fn is not None:
            x01 = transform_fn(x01, idx.tolist()) if image_id_aware else transform_fn(x01)
        feats = backbone.features(x01)
        all_feats.append(feats.cpu().numpy())
        if backbone.head is not None:
            logits = backbone.logits_from_features(feats)
            all_logits.append(logits.cpu().numpy())
        all_labels.append(y.numpy())
        all_ids.append(idx.numpy())
    out = {
        "feats": np.concatenate(all_feats),
        "labels": np.concatenate(all_labels),
        "ids": np.concatenate(all_ids),
    }
    if all_logits:
        out["logits"] = np.concatenate(all_logits)
    return out


def make_loader(ds, batch_size=64, shuffle=False):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                       collate_fn=collate_common, num_workers=2)


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #

def step0_prepare_backbones(cfg, classes):
    with open("results/trainval_split_seed6304.json") as f:
        split = json.load(f)
    train_ds = PetsSubset(cfg["dataset"]["root"], "trainval", split["train_indices"],
                           cfg["dataset"]["image_size"])
    val_ds = PetsSubset(cfg["dataset"]["root"], "trainval", split["val_indices"],
                         cfg["dataset"]["image_size"])
    train_loader = make_loader(train_ds, batch_size=cfg["linear_probe"]["batch_size"])
    val_loader = make_loader(val_ds, batch_size=cfg["linear_probe"]["batch_size"])

    lp = cfg["linear_probe"]
    probe_cfg = ProbeConfig(
        max_epochs=lp["max_epochs"],
        lr=lp["lr"],
        weight_decay=lp["weight_decay"],
        patience=lp["early_stop_patience"],
        seed=lp["seed"],
        batch_size=lp["batch_size"],
    )
    backbones = {}
    for name in BACKBONE_NAMES:
        print(f"[backbone={name}] extracting train/val features...")
        bb = build_backbone(name).to(DEVICE)
        bb.attach_head(len(classes)).to(DEVICE)
        train_out = extract(bb, train_loader)
        val_out = extract(bb, val_loader)

        print(f"[backbone={name}] training linear head...")
        head = train_linear_probe(
            torch.tensor(train_out["feats"]), torch.tensor(train_out["labels"]),
            torch.tensor(val_out["feats"]), torch.tensor(val_out["labels"]),
            num_classes=len(classes), cfg=probe_cfg,
        )
        bb.head = head.to(DEVICE)
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(head.state_dict(), f"checkpoints/{name}_linear_head.pt")
        backbones[name] = bb
    return backbones


def load_test_subset(cfg):
    with open(cfg["test_subset"]["save_path"]) as f:
        subset = json.load(f)
    test_ds = PetsSubset(cfg["dataset"]["root"], "test", subset["dataset_indices"],
                          cfg["dataset"]["image_size"])
    return test_ds, subset["classes"]


def model_variant_logits(backbones, loader, transform_fn=None, image_id_aware=False,
                          clip_class_names=None, clip_prompt=None):
    """Returns {"resnet50_head": logits, "vit_b16_head": logits,
    "clip_head": logits, "clip_zeroshot": logits}, plus features per backbone."""
    results = {}
    feats_by_backbone = {}
    for name, bb in backbones.items():
        out = extract(bb, loader, transform_fn, image_id_aware)
        results[f"{name}_head"] = out["logits"]
        feats_by_backbone[name] = out["feats"]

    if "clip_vitb32" in backbones and clip_class_names is not None:
        clip_bb = backbones["clip_vitb32"]
        zs_logits = []
        for x01, y, idx in loader:
            x01 = x01.to(DEVICE)
            if transform_fn is not None:
                x01 = transform_fn(x01, idx.tolist()) if image_id_aware else transform_fn(x01)
            with torch.no_grad():
                zs_logits.append(
                    clip_bb.zero_shot_logits(x01, clip_class_names, clip_prompt).cpu().numpy()
                )
        results["clip_zeroshot"] = np.concatenate(zs_logits)
    return results, feats_by_backbone


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["seed"])
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    test_ds, classes = load_test_subset(cfg)
    backbones = step0_prepare_backbones(cfg, classes)
    test_loader = make_loader(test_ds, batch_size=32)
    prompt = cfg["backbones"]["clip_vitb32"]["zero_shot_prompt_template"]

    # ---- Step 1: Clean baseline -----------------------------------------
    print("Step 1: clean baseline")
    clean_logits, clean_feats = model_variant_logits(
        backbones, test_loader, clip_class_names=classes, clip_prompt=prompt
    )
    labels = np.array([test_ds[i][1] for i in range(len(test_ds))])
    df = clean_baseline_report(clean_logits, labels)
    df.to_csv("results/step1_clean_baseline.csv", index=False)
    print(df)

    # ---- Step 2: Color bias (grayscale + hue rotation) --------------------
    print("Step 2: color bias")
    gray_logits, gray_feats = model_variant_logits(
        backbones, test_loader, transform_fn=to_grayscale,
        clip_class_names=classes, clip_prompt=prompt,
    )
    hue_deg = cfg["interventions"]["color"]["hue_rotation_degrees"]
    hue_logits, hue_feats = model_variant_logits(
        backbones, test_loader,
        transform_fn=lambda x: hue_rotate(x, hue_deg),
        clip_class_names=classes, clip_prompt=prompt,
    )
    clean_by_model = {k: v for k, v in clean_logits.items()}
    transformed_by_model = {
        k: {"grayscale": gray_logits[k], "hue_rotation": hue_logits[k]}
        for k in clean_logits
    }
    df = color_bias_report(clean_by_model, transformed_by_model, labels)
    df.to_csv("results/step2_color_bias.csv", index=False)
    print(df)

    # ---- Step 3: Shape vs. texture (cue conflicts) -------------------------
    print("Step 3: shape vs texture")
    with open("results/cue_conflict_manifest.json") as f:
        manifest_data = json.load(f)
    manifest = manifest_data["conflicts"]
    cue_conflict_feats = {}
    cue_conflict_content_feats = {}
    if len(manifest) == 0:
        print("No accepted cue conflicts found -- run data/make_cue_conflicts.py first.")
    else:
        from PIL import Image
        from torchvision.datasets import OxfordIIITPet
        from torchvision.transforms.functional import resize, to_tensor

        imgs = torch.stack([to_tensor(Image.open(m["image_path"]).convert("RGB"))
                             for m in manifest]).to(DEVICE)

        # Load the corresponding *unstylized* content image for each accepted
        # conflict, from the same trainval pool used by make_cue_conflicts.py,
        # so Step 6 can measure representation stability relative to it.
        content_pool = OxfordIIITPet(root=cfg["dataset"]["root"], split="trainval",
                                      target_types="category", download=True)
        content_imgs = torch.stack([
            to_tensor(resize(content_pool[m["content_dataset_index"]][0],
                              [cfg["dataset"]["image_size"]] * 2))
            for m in manifest
        ]).to(DEVICE)

        preds_by_model = {}
        for name, bb in backbones.items():
            with torch.no_grad():
                feats = bb.features(imgs)
                content_feats = bb.features(content_imgs)
                logits = bb.logits_from_features(feats)
            preds_by_model[f"{name}_head"] = logits.argmax(dim=1).cpu().numpy()
            cue_conflict_feats[name] = feats.cpu().numpy()
            cue_conflict_content_feats[name] = content_feats.cpu().numpy()
        if "clip_vitb32" in backbones:
            clip_bb = backbones["clip_vitb32"]
            with torch.no_grad():
                zs_logits = clip_bb.zero_shot_logits(imgs, classes, prompt)
            preds_by_model["clip_zeroshot"] = zs_logits.argmax(dim=1).cpu().numpy()

        df = shape_bias_report(preds_by_model, manifest)

        # --- Export per-image cue-conflict predictions for qualitative examples ---
        import pandas as pd

        def _classify(pred_idx, content_idx, style_idx):
            if pred_idx == content_idx:
                return "shape"
            if pred_idx == style_idx:
                return "texture"
            return "other"

        rows = []
        for i, m in enumerate(manifest):
            row = {
                "pair_id": m["pair_id"],
                "direction": m["direction"],
                "image_path": m["image_path"],
                "content_class_name": m["content_class_name"],
                "style_class_name": m["style_class_name"],
            }
            for model_name, preds in preds_by_model.items():
                pred_idx = int(preds[i])
                row[f"{model_name}_pred"] = classes[pred_idx]
                row[f"{model_name}_label"] = _classify(pred_idx, m["content_class_idx"], m["style_class_idx"])
            rows.append(row)

        pd.DataFrame(rows).to_csv("results/cue_conflict_predictions.csv", index=False)

        df.to_csv("results/step3_shape_bias.csv", index=False)
        print(df)

    # ---- Step 4: Translation ------------------------------------------------
    print("Step 4: translation")
    pixels_list = cfg["interventions"]["translation"]["pixels"]
    directions = cfg["interventions"]["translation"]["directions"]
    max_px = max(pixels_list)
    translation_logits = {}  # {pixels: {direction: {model: logits}}}
    translation_feats_max_px = {}  # {direction: {backbone: feats}}, at max_px only
    for px in pixels_list:
        translation_logits[px] = {}
        dirs_to_run = ["up"] if px == 0 else directions  # 0px is direction-invariant
        for d in dirs_to_run:
            logits_d, feats_d = model_variant_logits(
                backbones, test_loader,
                transform_fn=lambda x, px=px, d=d: translate_cardinal(x, px, d),
                clip_class_names=classes, clip_prompt=prompt,
            )
            translation_logits[px][d] = logits_d
            if px == max_px:
                translation_feats_max_px[d] = feats_d
        if px == 0:  # replicate the single 0px result across all direction keys
            for d in directions[1:]:
                translation_logits[px][d] = translation_logits[px]["up"]

    for model_key in clean_logits:
        per_px_dir = {px: {d: translation_logits[px][d][model_key] for d in translation_logits[px]}
                      for px in pixels_list}
        df = translation_curve_report(per_px_dir, clean_logits[model_key], labels)
        df.to_csv(f"results/step4_translation_{model_key}.csv", index=False)
        print(f"-- {model_key} --")
        print(df)

    # ---- Step 5: Patch structure --------------------------------------------
    print("Step 5: patch shuffle")
    grid = cfg["interventions"]["patch_shuffle"]["grid"]
    shuffle_seed = cfg["interventions"]["patch_shuffle"]["seed"]
    shuffle_logits, shuffle_feats = model_variant_logits(
        backbones, test_loader,
        transform_fn=lambda x, ids: patch_shuffle(x, grid=grid, seed=shuffle_seed,
                                                    image_ids=ids),
        image_id_aware=True,
        clip_class_names=classes, clip_prompt=prompt,
    )
    df = patch_shuffle_report(clean_by_model, shuffle_logits, labels)
    df.to_csv("results/step5_patch_shuffle.csv", index=False)
    print(df)

    # ---- Step 6: Representation stability + t-SNE/UMAP ----------------------
    # Required for exactly these four interventions: grayscale, cue conflict,
    # translation, and patch shuffling (assignment Step 6).
    print("Step 6: representation analysis")
    from metrics import cosine_similarity_paired

    stability_rows = []
    for name in backbones:
        # grayscale and patch-shuffle: direct clean-vs-transformed pairing
        for transform_name, t_feats in [
            ("grayscale", gray_feats[name]),
            ("patch_shuffle", shuffle_feats[name]),
        ]:
            it = float(cosine_similarity_paired(clean_feats[name], t_feats).mean())
            stability_rows.append({"model": name, "transform": transform_name, "I_T": it})

        # translation at the largest displacement, averaged over the four
        # cardinal directions (each direction's I_T is computed separately,
        # then averaged, rather than averaging feature vectors first).
        per_direction_it = [
            float(cosine_similarity_paired(clean_feats[name],
                                            translation_feats_max_px[d][name]).mean())
            for d in directions
        ]
        stability_rows.append({
            "model": name, "transform": f"translation_{max_px}px",
            "I_T": float(np.mean(per_direction_it)),
        })

        # cue conflict: paired against each conflict's own (unstylized)
        # content image, not the fixed 500-image test subset.
        if name in cue_conflict_feats:
            it = float(cosine_similarity_paired(
                cue_conflict_content_feats[name], cue_conflict_feats[name]
            ).mean())
            stability_rows.append({"model": name, "transform": "cue_conflict", "I_T": it})

    stability_df = pd.DataFrame(stability_rows)
    stability_df.to_csv("results/step6_representation_stability.csv", index=False)
    print(stability_df)

    method = cfg["representation"]["method"]
    n_vis = min(cfg["representation"]["n_images_per_condition"], len(test_ds))
    rng = np.random.default_rng(cfg["representation"]["seed"])
    vis_idx = rng.choice(len(test_ds), size=n_vis, replace=False)
    vis_labels = labels[vis_idx]
    method_kwargs = cfg["representation"][method]

    for name in backbones:
        for transform_name, t_feats in [
            ("grayscale", gray_feats[name]),
            (f"translation_{max_px}px", translation_feats_max_px["up"][name]),
            ("patch_shuffle", shuffle_feats[name]),
        ]:
            plot_clean_vs_transformed(
                clean_feats[name][vis_idx], t_feats[vis_idx], vis_labels, classes,
                method=method, title=f"{name}: clean vs {transform_name}",
                save_path=f"results/figures/{name}_{transform_name}_{method}.png",
                seed=cfg["representation"]["seed"], **method_kwargs,
            )
        if name in cue_conflict_feats:
            cc_content_labels = np.array([m["content_class_idx"] for m in manifest])
            plot_clean_vs_transformed(
                cue_conflict_content_feats[name], cue_conflict_feats[name],
                cc_content_labels, classes,
                method=method, title=f"{name}: content vs cue-conflict",
                save_path=f"results/figures/{name}_cue_conflict_{method}.png",
                seed=cfg["representation"]["seed"], **method_kwargs,
            )
    print("Saved representation figures to results/figures/")
    print("\nDone. All Task 1 required evidence is in task1/results/.")


if __name__ == "__main__":
    main()
