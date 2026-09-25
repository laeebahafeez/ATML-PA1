"""
PACS dataset loading, shared between Task 2 (UDA) and Task 3 (DG).

Handles two real-world annoyances discovered while building this:
  1. PACS is not bundled with torchvision, so it has to be found wherever
     it was attached/uploaded, at an unpredictable path (same lesson as
     Task 1's Kaggle-input auto-discovery).
  2. Folder-naming conventions differ slightly across public PACS mirrors
     (e.g. "art_painting" vs "art painting" vs "artpainting").

Both problems are solved once, here, so task2/ and task3/ never have to
think about them again.
"""
from __future__ import annotations

import os
import re
from typing import Dict, List

from torchvision import transforms as T
from torchvision.datasets import ImageFolder

DOMAIN_ALIASES = {
    "photo": ["photo", "photos"],
    "art_painting": ["art_painting", "art painting", "artpainting", "art-painting"],
    "cartoon": ["cartoon", "cartoons"],
    "sketch": ["sketch", "sketches"],
}

EXPECTED_CLASSES = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _normalize_name(name: str) -> str:
    return re.sub(r"[\s_\-]+", "", name.strip().lower())


def locate_pacs_domains(search_root: str) -> Dict[str, str]:
    """
    Recursively search `search_root` for the four PACS domain folders,
    tolerating naming variation. Returns {canonical_domain: absolute_path}.

    Fails loudly (rather than silently proceeding with a partial or wrong
    match) if any of the four domains cannot be found, listing exactly
    what was found so the failure is easy to diagnose.
    """
    alias_lookup = {}
    for canonical, aliases in DOMAIN_ALIASES.items():
        for a in aliases:
            alias_lookup[_normalize_name(a)] = canonical

    found: Dict[str, str] = {}
    for dirpath, dirnames, _ in os.walk(search_root):
        for d in dirnames:
            key = _normalize_name(d)
            if key in alias_lookup and alias_lookup[key] not in found:
                candidate = os.path.join(dirpath, d)
                try:
                    subdirs = [s for s in os.listdir(candidate)
                               if os.path.isdir(os.path.join(candidate, s))]
                except OSError:
                    continue
                # a genuine PACS domain folder has ~7 class subfolders;
                # tolerate a little slack rather than requiring exactly 7
                # in case of an unusual mirror layout.
                if len(subdirs) >= 5:
                    found[alias_lookup[key]] = candidate

    missing = set(DOMAIN_ALIASES) - set(found)
    if missing:
        raise FileNotFoundError(
            f"Could not locate PACS domain folder(s) {sorted(missing)} under "
            f"{search_root!r}. Found so far: {found}.\n"
            "Make sure a PACS dataset (photo/art_painting/cartoon/sketch, each "
            "with 7 class subfolders: dog, elephant, giraffe, guitar, horse, "
            "house, person) is attached or extracted somewhere under this root."
        )
    return found


def build_transform(resize_size: int, crop_size: int, train: bool) -> T.Compose:
    if train:
        return T.Compose([
            T.Resize((resize_size, resize_size)),
            T.RandomCrop(crop_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return T.Compose([
        T.Resize((resize_size, resize_size)),
        T.CenterCrop(crop_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_domain_imagefolder(domain_path: str, transform: T.Compose) -> ImageFolder:
    """
    torchvision's ImageFolder sorts both class-folder names and filenames
    within each class internally, so `.samples` order (and therefore any
    integer indices saved against it) is deterministic across machines and
    reruns -- safe to use for reproducible splits.
    """
    ds = ImageFolder(domain_path, transform=transform)
    if len(ds.classes) != len(EXPECTED_CLASSES):
        raise ValueError(
            f"{domain_path} has {len(ds.classes)} class subfolders "
            f"({ds.classes}); expected {len(EXPECTED_CLASSES)} "
            f"({EXPECTED_CLASSES}). Check that this is really a PACS domain folder."
        )
    return ds


def load_domain_pair(domain_path: str, resize_size: int, crop_size: int):
    """
    Returns (train_transform_dataset, eval_transform_dataset) -- two
    ImageFolder instances over the SAME underlying files, differing only in
    which transform is applied. Needed because a stratified train/val split
    of one domain requires the train portion to get train-time augmentation
    and the val portion to get eval-time preprocessing, from files that
    otherwise live in the same folder.

    Both instances enumerate files in the same (sorted) order, so indices
    computed from one apply identically to the other.
    """
    train_ds = load_domain_imagefolder(
        domain_path, build_transform(resize_size, crop_size, train=True))
    eval_ds = load_domain_imagefolder(
        domain_path, build_transform(resize_size, crop_size, train=False))
    return train_ds, eval_ds


def load_all_domain_pairs(root: str, resize_size: int, crop_size: int) -> Dict[str, tuple]:
    """{domain: (train_transform_dataset, eval_transform_dataset)} for all four domains."""
    paths = locate_pacs_domains(root)
    return {domain: load_domain_pair(path, resize_size, crop_size)
            for domain, path in paths.items()}
