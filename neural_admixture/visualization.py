"""
Visualization utilities for Neural ADMIXTURE.

Provides:
  - PCA scatter plots with learnt F-matrix centroids
  - Stacked bar plots for ancestry proportions (Q)
  - Training loss curves
  - Multi-head Q comparison panels
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
from typing import Dict, List, Optional, Tuple, Union


# Consistent color palette for up to 12 clusters
_PALETTE = [
    "#E64B35", "#4DBBD5", "#00A087", "#3C5488",
    "#F39B7F", "#8491B4", "#91D1C2", "#DC9E82",
    "#7E6148", "#B09C85", "#E377C2", "#BCBD22",
]


def _get_colors(k: int) -> List[str]:
    if k <= len(_PALETTE):
        return _PALETTE[:k]
    cmap = plt.cm.get_cmap("tab20", k)
    return [mcolors.rgb2hex(cmap(i)) for i in range(k)]


# ---------------------------------------------------------------------------
# PCA projection with centroids
# ---------------------------------------------------------------------------

def plot_pca_with_centroids(
    X: np.ndarray,
    F_est: np.ndarray,
    labels: Optional[np.ndarray] = None,
    label_names: Optional[Dict[int, str]] = None,
    F_gt: Optional[np.ndarray] = None,
    n_components: int = 2,
    figsize: Tuple[float, float] = (10, 8),
    title: str = "PCA Projection with Learnt Centroids",
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Project training data onto the first 2 PCs and overlay F-matrix centroids.

    Parameters
    ----------
    X : (N, M) genotype matrix.
    F_est : (K, M) estimated allele frequency matrix from the decoder.
    labels : (N,) integer population labels for coloring data points.
    label_names : maps integer label -> display name.
    F_gt : optional (K, M) ground-truth allele frequencies to plot alongside.
    n_components : PCA dimensions (only first 2 are plotted).
    figsize : figure size.
    title : plot title.
    ax : existing axes to draw on (creates new figure if None).
    """
    pca = PCA(n_components=n_components)
    Z = pca.fit_transform(X)

    F_proj = pca.transform(F_est)

    show = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if labels is not None:
        unique_labels = sorted(set(labels))
        colors = _get_colors(len(unique_labels))
        for i, lab in enumerate(unique_labels):
            mask = labels == lab
            name = label_names[lab] if label_names else str(lab)
            ax.scatter(
                Z[mask, 0], Z[mask, 1],
                c=colors[i], label=name, alpha=0.4, s=8, edgecolors="none",
            )
    else:
        ax.scatter(Z[:, 0], Z[:, 1], c="gray", alpha=0.3, s=8, edgecolors="none")

    K = F_est.shape[0]
    centroid_colors = _get_colors(K)
    ax.scatter(
        F_proj[:, 0], F_proj[:, 1],
        c=centroid_colors[:K], marker="*", s=300, edgecolors="black",
        linewidths=1.2, zorder=5, label="Learnt centroids",
    )

    if F_gt is not None:
        Fgt_proj = pca.transform(F_gt)
        ax.scatter(
            Fgt_proj[:, 0], Fgt_proj[:, 1],
            c=centroid_colors[:K], marker="D", s=120, edgecolors="black",
            linewidths=1.2, zorder=5, label="GT centroids",
        )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8, markerscale=1.5)

    fig.tight_layout()
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Stacked bar plot for Q
# ---------------------------------------------------------------------------

def plot_admixture_barplot(
    Q: np.ndarray,
    labels: Optional[np.ndarray] = None,
    label_names: Optional[Dict[int, str]] = None,
    sort_by_label: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    title: str = "Ancestry Proportions (Q)",
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Stacked vertical bar plot of ancestry proportions.

    Each vertical bar is one individual; color segments show fractional
    ancestry for each cluster. Individuals are grouped by population label.

    Parameters
    ----------
    Q : (N, K) ancestry proportion matrix.
    labels : (N,) integer population labels for grouping.
    label_names : maps integer label -> display name.
    sort_by_label : if True, sort individuals by label then by dominant cluster.
    figsize : figure size (auto-scaled if None).
    title : plot title.
    ax : existing axes (creates new figure if None).
    """
    N, K = Q.shape
    colors = _get_colors(K)

    if figsize is None:
        figsize = (max(10, N * 0.02), 3)

    if sort_by_label and labels is not None:
        dominant = Q.argmax(axis=1)
        order = np.lexsort((dominant, labels))
    elif labels is not None:
        order = np.argsort(labels)
    else:
        order = np.arange(N)

    Q_sorted = Q[order]
    labels_sorted = labels[order] if labels is not None else None

    show = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    x = np.arange(N)
    bottom = np.zeros(N)
    for k_idx in range(K):
        ax.bar(
            x, Q_sorted[:, k_idx], bottom=bottom,
            width=1.0, color=colors[k_idx], edgecolor="none",
            label=f"K={k_idx + 1}",
        )
        bottom += Q_sorted[:, k_idx]

    if labels_sorted is not None:
        unique_labels = []
        positions = []
        prev_label = None
        for i, lab in enumerate(labels_sorted):
            if lab != prev_label:
                if prev_label is not None:
                    unique_labels.append(prev_label)
                    positions.append((start_pos + i - 1) / 2.0)
                    ax.axvline(x=i - 0.5, color="black", linewidth=0.5)
                start_pos = i
                prev_label = lab
        unique_labels.append(prev_label)
        positions.append((start_pos + N - 1) / 2.0)

        tick_labels = [
            label_names[l] if label_names else str(l) for l in unique_labels
        ]
        ax.set_xticks(positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
    else:
        ax.set_xticks([])

    ax.set_xlim(-0.5, N - 0.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Ancestry fraction")
    ax.set_title(title)

    fig.tight_layout()
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Multi-head Q comparison
# ---------------------------------------------------------------------------

def plot_multihead_barplots(
    Qs: List[np.ndarray],
    k_values: List[int],
    labels: Optional[np.ndarray] = None,
    label_names: Optional[Dict[int, str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    suptitle: str = "Multi-head Ancestry Proportions",
) -> plt.Figure:
    """
    Side-by-side stacked bar plots for each head in multi-head mode.

    Parameters
    ----------
    Qs : list of (N, K_h) ancestry matrices, one per head.
    k_values : list of K values corresponding to each head.
    labels : (N,) integer population labels.
    label_names : maps integer label -> display name.
    """
    H = len(Qs)
    N = Qs[0].shape[0]
    if figsize is None:
        figsize = (max(12, N * 0.02), 2.5 * H)

    fig, axes = plt.subplots(H, 1, figsize=figsize, sharex=True)
    if H == 1:
        axes = [axes]

    for i, (Q, k, ax) in enumerate(zip(Qs, k_values, axes)):
        plot_admixture_barplot(
            Q, labels=labels, label_names=label_names,
            title=f"K = {k}", ax=ax,
        )

    fig.suptitle(suptitle, fontsize=14, y=1.01)
    fig.tight_layout()
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Training loss curves
# ---------------------------------------------------------------------------

def plot_training_history(
    history: List[Dict[str, float]],
    figsize: Tuple[float, float] = (8, 5),
    title: str = "Training Loss",
) -> plt.Figure:
    """
    Plot train (and optionally validation) loss over epochs.

    Parameters
    ----------
    history : list of dicts from Trainer.history, each with at least
        'epoch' and 'train_loss', optionally 'val_loss'.
    """
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(epochs, train_loss, label="Train loss", linewidth=2)

    if "val_loss" in history[0]:
        val_loss = [h["val_loss"] for h in history]
        ax.plot(epochs, val_loss, label="Val loss", linewidth=2, linestyle="--")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (BCE)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()
    return fig
