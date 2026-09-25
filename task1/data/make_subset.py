"""
Build the reproducible Task 1 data splits for Oxford-IIIT Pets:

  1. A stratified 80/20 train/validation split of the official "trainval"
     partition (seed 6304), used to train and select each linear head.
  2. A class-balanced subset of 500 images from the official "test"
     partition (seed 6304), used for every clean/intervention evaluation.
     Image identifiers (dataset indices + filenames) are saved so the
     exact same 500 images are reused throughout the whole task.

Run:
    python data/make_subset.py --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import json
import os

import yaml
from torchvision.datasets import OxfordIIITPet

from split_utils import class_balanced_subset, stratified_train_val_split


def _labels_of(dataset: OxfordIIITPet):
    # OxfordIIITPet stores per-example category labels in `_labels`
    # (avoids decoding every image just to read its class).
    return list(dataset._labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    root = cfg["dataset"]["root"]
    seed = cfg["split"]["seed"]
    val_fraction = cfg["split"]["val_fraction"]
    test_size = cfg["test_subset"]["size"]
    test_seed = cfg["test_subset"]["seed"]

    os.makedirs("results", exist_ok=True)
    os.makedirs(root, exist_ok=True)

    print("Loading Oxford-IIIT Pets 'trainval' split (downloads on first run)...")
    trainval = OxfordIIITPet(root=root, split="trainval", target_types="category",
                              download=True)
    trainval_labels = _labels_of(trainval)
    classes = trainval.classes

    train_idx, val_idx = stratified_train_val_split(trainval_labels, val_fraction, seed)
    print(f"trainval={len(trainval_labels)}  -> train={len(train_idx)}  val={len(val_idx)}")

    with open("results/trainval_split_seed6304.json", "w") as f:
        json.dump(
            {
                "seed": seed,
                "val_fraction": val_fraction,
                "num_classes": len(classes),
                "classes": classes,
                "train_indices": train_idx,
                "val_indices": val_idx,
            },
            f,
            indent=2,
        )

    print("Loading Oxford-IIIT Pets 'test' split...")
    test = OxfordIIITPet(root=root, split="test", target_types="category", download=True)
    test_labels = _labels_of(test)

    selected, per_class_count, imbalance_notes = class_balanced_subset(
        test_labels, test_size, test_seed
    )
    print(f"selected {len(selected)} / requested {test_size} test images "
          f"across {len(per_class_count)} classes")
    if imbalance_notes:
        print("Class imbalance encountered:")
        for c, note in imbalance_notes.items():
            print(f"  - {note}")

    # Save both the dataset indices and the underlying filenames so the
    # subset can be re-identified even if the dataset object is rebuilt.
    filenames = [os.path.basename(str(test._images[i])) for i in selected]
    labels_for_selected = [int(test_labels[i]) for i in selected]

    with open(cfg["test_subset"]["save_path"], "w") as f:
        json.dump(
            {
                "seed": test_seed,
                "requested_size": test_size,
                "actual_size": len(selected),
                "per_class_count": per_class_count,
                "imbalance_notes": imbalance_notes,
                "dataset_indices": selected,
                "filenames": filenames,
                "labels": labels_for_selected,
                "classes": test.classes,
            },
            f,
            indent=2,
        )
    print(f"Saved test subset identifiers to {cfg['test_subset']['save_path']}")


if __name__ == "__main__":
    main()
