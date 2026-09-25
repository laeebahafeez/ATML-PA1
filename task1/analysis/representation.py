"""
Step 6 (Representation Analysis): joint t-SNE / UMAP visualization of clean
vs. transformed features, per backbone.

For each backbone, ONE 2-D projection is fit on the combined clean +
transformed features (as required -- "fit one two-dimensional projection
to the combined clean and transformed features so that both conditions
appear in the same space"). Color encodes ground-truth class; marker
style distinguishes clean from transformed examples. Coordinates are not
comparable across backbones, since each backbone gets its own projection.
"""
from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

MARKERS = {"clean": "o", "transformed": "x"}


def fit_projection(features: np.ndarray, method: str = "tsne", seed: int = 6304,
                    perplexity: int = 30, max_iter: int = 1000,
                    n_neighbors: int = 15, min_dist: float = 0.1) -> np.ndarray:
    """Keyword names here (perplexity, max_iter, n_neighbors, min_dist) match
    configs/config.yaml's representation.tsne / representation.umap sections
    exactly, so `**cfg["representation"][method]` can be forwarded straight
    through from scripts/run_task1.py without renaming anything.

    Note: scikit-learn renamed TSNE's `n_iter` to `max_iter` in version 1.5;
    this uses the current name so it works on any recent scikit-learn."""
    if method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(
            n_components=2, perplexity=perplexity, max_iter=max_iter,
            random_state=seed, init="pca",
        )
    elif method == "umap":
        import umap
        reducer = umap.UMAP(
            n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
            random_state=seed,
        )
    else:
        raise ValueError(f"unknown method {method!r}; use 'tsne' or 'umap'")
    return reducer.fit_transform(features)


def plot_clean_vs_transformed(
    clean_feats: np.ndarray,
    transformed_feats: np.ndarray,
    labels: Sequence[int],
    class_names: Sequence[str],
    method: str,
    title: str,
    save_path: str,
    seed: int = 6304,
    **method_kwargs,
) -> None:
    """
    clean_feats, transformed_feats: [N, D], row-aligned (same underlying
    images / cue-conflict content classes in the same order).
    labels: ground-truth class id for each of the N rows (shared by both
    conditions since the object identity does not change).
    """
    combined = np.concatenate([clean_feats, transformed_feats], axis=0)
    proj = fit_projection(combined, method=method, seed=seed, **method_kwargs)
    n = clean_feats.shape[0]
    proj_clean, proj_transformed = proj[:n], proj[n:]

    labels = np.asarray(labels)
    unique_labels = sorted(set(labels.tolist()))
    cmap = plt.get_cmap("tab20", max(len(unique_labels), 1))

    fig, ax = plt.subplots(figsize=(7, 6))
    for i, cls in enumerate(unique_labels):
        mask = labels == cls
        color = cmap(i)
        ax.scatter(proj_clean[mask, 0], proj_clean[mask, 1], color=[color],
                   marker=MARKERS["clean"], s=18, alpha=0.8,
                   label=f"{class_names[cls]} (clean)" if i < 10 else None)
        ax.scatter(proj_transformed[mask, 0], proj_transformed[mask, 1], color=[color],
                   marker=MARKERS["transformed"], s=28, alpha=0.8,
                   label=f"{class_names[cls]} (transformed)" if i < 10 else None)

    ax.set_title(title)
    ax.set_xlabel(f"{method.upper()} dim 1")
    ax.set_ylabel(f"{method.upper()} dim 2")
    ax.legend(fontsize=6, ncol=2, loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
