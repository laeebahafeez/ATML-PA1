"""
Generate shape-vs-texture cue-conflict images for Task 1, Step 3.

Protocol (matches the assignment spec exactly):
  - Select >=5 unordered class pairs (A, B), seed 6304.
  - For each pair, generate both directions when feasible:
        shape/content = A, texture/style = B
        shape/content = B, texture/style = A
  - Use AdaIN (models/adain_net.py) to fuse the content image's shape with
    the style image's texture.
  - Define a VISUAL rejection rule *before* any model ever sees the
    images, and apply it purely from pixel statistics -- never from a
    classifier's prediction ("Do not use model predictions to decide
    which images to retain").
  - Keep generating until >=200 valid (accepted) conflicts are produced,
    balanced across class pairs and directions as closely as possible.
  - Record accepted and rejected counts.

Content/style source images are drawn from the "trainval" partition (not
the frozen 500-image test-evaluation subset), so the evaluation subset
stays reserved for the clean/color/translation/patch-shuffle experiments.

Run:
    python data/make_cue_conflicts.py --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms.functional import to_tensor, resize

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
from adain_net import AdaINStyleTransfer  # noqa: E402


# --------------------------------------------------------------------------- #
# Rejection rule (fixed BEFORE looking at any model prediction)
# --------------------------------------------------------------------------- #

_LAPLACIAN_KERNEL = torch.tensor(
    [[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]
).view(1, 1, 3, 3)


def _edge_energy(x01: torch.Tensor) -> torch.Tensor:
    """Mean squared Laplacian response on the luma channel: a simple,
    fixed, prediction-free proxy for 'how much shape/edge structure remains'."""
    gray = (x01 * torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)).sum(1, keepdim=True)
    edges = F.conv2d(gray, _LAPLACIAN_KERNEL, padding=1)
    return (edges ** 2).mean(dim=[1, 2, 3])


def passes_rejection_rule(
    generated01: torch.Tensor, content01: torch.Tensor, rule: dict
) -> Tuple[bool, dict]:
    """Returns (accepted, diagnostics). Applied identically to every image,
    fixed thresholds taken straight from configs/config.yaml."""
    pixel_std = generated01.std().item()
    gen_energy = _edge_energy(generated01).item()
    content_energy = _edge_energy(content01).item() + 1e-8
    ratio = gen_energy / content_energy

    ok_std = pixel_std >= rule["min_pixel_std"]
    ok_ratio = rule["min_edge_energy_ratio"] <= ratio <= rule["max_edge_energy_ratio"]
    accepted = bool(ok_std and ok_ratio)
    diagnostics = {"pixel_std": pixel_std, "edge_energy_ratio": ratio}
    return accepted, diagnostics


# --------------------------------------------------------------------------- #
# Main generation loop
# --------------------------------------------------------------------------- #

def load_image(dataset: OxfordIIITPet, idx: int, size: int) -> torch.Tensor:
    img, _ = dataset[idx]  # PIL image, RGB
    img = resize(img, [size, size])
    return to_tensor(img)  # [3, H, W] in [0, 1]


def select_class_pairs(classes: List[str], num_pairs: int, seed: int) -> List[Tuple[int, int]]:
    rng = np.random.default_rng(seed)
    all_pairs = list(itertools.combinations(range(len(classes)), 2))
    chosen_idx = rng.choice(len(all_pairs), size=num_pairs, replace=False)
    return [all_pairs[i] for i in chosen_idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--out_dir", default="data/cue_conflicts")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cc_cfg = cfg["interventions"]["cue_conflict"]
    seed = cc_cfg["seed"]
    alpha = cc_cfg["style_strength_alpha"]
    n_pairs = cc_cfg["num_class_pairs"]
    per_pair_dir = cc_cfg["images_per_pair_per_direction"]
    min_valid = cc_cfg["min_valid_conflicts"]
    rule = cc_cfg["rejection_rule"]
    img_size = cfg["dataset"]["image_size"]

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print("Loading Oxford-IIIT Pets 'trainval' partition as the "
          "content/style source pool...")
    pool = OxfordIIITPet(root=cfg["dataset"]["root"], split="trainval",
                          target_types="category", download=True)
    labels = np.asarray(pool._labels)
    classes = pool.classes

    pairs = select_class_pairs(classes, n_pairs, seed)
    print("Selected class pairs:")
    for a, b in pairs:
        print(f"  {classes[a]}  <->  {classes[b]}")

    style_net = AdaINStyleTransfer(cfg["adain"]["decoder_weights"])

    rng = np.random.default_rng(seed)
    manifest: List[dict] = []
    n_accepted = 0
    n_rejected = 0
    max_attempts_multiplier = 3  # generate extra candidates to survive rejection

    for pair_id, (a, b) in enumerate(pairs):
        for direction, (content_cls, style_cls) in enumerate([(a, b), (b, a)]):
            content_pool = np.where(labels == content_cls)[0]
            style_pool = np.where(labels == style_cls)[0]

            attempts = 0
            accepted_this_cell = 0
            max_attempts = per_pair_dir * max_attempts_multiplier

            while accepted_this_cell < per_pair_dir and attempts < max_attempts:
                c_idx = int(rng.choice(content_pool))
                s_idx = int(rng.choice(style_pool))
                content_img = load_image(pool, c_idx, img_size).unsqueeze(0)
                style_img = load_image(pool, s_idx, img_size).unsqueeze(0)

                generated = style_net.stylize(content_img, style_img, alpha=alpha)
                accepted, diag = passes_rejection_rule(generated[0], content_img[0], rule)
                attempts += 1

                fname = f"pair{pair_id}_dir{direction}_{'accepted' if accepted else 'rejected'}_{attempts}.png"
                out_path = os.path.join(args.out_dir, fname)

                record = {
                    "pair_id": pair_id,
                    "direction": direction,
                    "content_class_idx": content_cls,
                    "content_class_name": classes[content_cls],
                    "style_class_idx": style_cls,
                    "style_class_name": classes[style_cls],
                    "content_dataset_index": c_idx,
                    "style_dataset_index": s_idx,
                    "accepted": accepted,
                    "diagnostics": diag,
                    "image_path": out_path,
                }

                if accepted:
                    img_np = (generated[0].permute(1, 2, 0).numpy() * 255).astype("uint8")
                    Image.fromarray(img_np).save(out_path)
                    manifest.append(record)
                    accepted_this_cell += 1
                    n_accepted += 1
                else:
                    n_rejected += 1
                    # rejected images are not written to disk, only counted,
                    # keeping the accepted/rejected tally auditable without
                    # bloating the repository.

            if accepted_this_cell < per_pair_dir:
                print(f"WARNING: pair {pair_id} direction {direction} "
                      f"({classes[content_cls]}/{classes[style_cls]}) only produced "
                      f"{accepted_this_cell}/{per_pair_dir} valid conflicts after "
                      f"{attempts} attempts.")

    print(f"\nTotal accepted: {n_accepted}   Total rejected: {n_rejected}   "
          f"(target >= {min_valid})")
    if n_accepted < min_valid:
        print("WARNING: fewer than the required 200 valid conflicts were produced. "
              "Consider raising images_per_pair_per_direction or "
              "max_attempts_multiplier, or relaxing (and re-fixing in advance) "
              "the rejection thresholds before re-running.")

    with open("results/cue_conflict_manifest.json", "w") as f:
        json.dump(
            {
                "seed": seed,
                "alpha": alpha,
                "class_pairs": [[classes[a], classes[b]] for a, b in pairs],
                "rejection_rule": rule,
                "n_accepted": n_accepted,
                "n_rejected": n_rejected,
                "conflicts": manifest,
            },
            f,
            indent=2,
        )
    print("Saved manifest to results/cue_conflict_manifest.json")


if __name__ == "__main__":
    main()
