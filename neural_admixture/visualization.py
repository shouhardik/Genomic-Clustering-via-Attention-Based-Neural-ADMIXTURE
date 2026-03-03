"""
Visualization utilities for Neural ADMIXTURE.

Provides:
  - PCA scatter plots with learnt F-matrix centroids
  - Stacked bar plots for ancestry proportions (Q)
  - Population-level ancestry heatmap
  - Training loss curves
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from typing import Dict, List, Optional, Tuple


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
# Stacked bar plot for Q (improved for large N)
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
        figsize = (max(14, min(N * 0.015, 24)), 4.5)

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
                    ax.axvline(x=i - 0.5, color="black", linewidth=0.8)
                start_pos = i
                prev_label = lab
        unique_labels.append(prev_label)
        positions.append((start_pos + N - 1) / 2.0)

        tick_labels = [
            label_names[l] if label_names else str(l) for l in unique_labels
        ]
        ax.set_xticks(positions)
        ax.set_xticklabels(tick_labels, fontsize=10, fontweight="bold")
    else:
        ax.set_xticks([])

    patches = [mpatches.Patch(color=colors[k], label=f"Cluster {k+1}")
               for k in range(K)]
    ax.legend(handles=patches, loc="upper right", fontsize=8,
              ncol=min(K, 6), framealpha=0.9, title="Ancestry")

    ax.set_xlim(-0.5, N - 0.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Ancestry fraction", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.tick_params(axis="y", labelsize=9)

    fig.tight_layout()
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Population-level ancestry heatmap
# ---------------------------------------------------------------------------

def plot_ancestry_heatmap(
    Q: np.ndarray,
    labels: np.ndarray,
    label_names: Optional[Dict[int, str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: str = "Mean Ancestry Proportions per Population",
) -> plt.Figure:
    """
    Heatmap showing mean ancestry proportions for each population.

    Much cleaner than per-individual barplots for large sample sizes.
    Rows = populations, columns = ancestry clusters, cell values = mean Q.

    Parameters
    ----------
    Q : (N, K) ancestry proportion matrix.
    labels : (N,) integer population labels.
    label_names : maps integer label -> display name.
    """
    K = Q.shape[1]
    unique_labels = sorted(set(labels))
    n_pops = len(unique_labels)

    pop_names = [label_names[l] if label_names else str(l) for l in unique_labels]
    mean_Q = np.zeros((n_pops, K))
    pop_counts = np.zeros(n_pops, dtype=int)
    for i, lab in enumerate(unique_labels):
        mask = labels == lab
        mean_Q[i] = Q[mask].mean(axis=0)
        pop_counts[i] = mask.sum()

    if figsize is None:
        figsize = (max(6, K * 1.2), max(4, n_pops * 0.6))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mean_Q, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)

    for i in range(n_pops):
        for j in range(K):
            val = mean_Q[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=10, color=color, fontweight="bold")

    ax.set_xticks(range(K))
    ax.set_xticklabels([f"Cluster {k+1}" for k in range(K)], fontsize=10)
    ax.set_yticks(range(n_pops))
    ax.set_yticklabels(
        [f"{name}  (n={pop_counts[i]})" for i, name in enumerate(pop_names)],
        fontsize=10,
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Ancestry cluster", fontsize=11)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Mean ancestry fraction", fontsize=10)

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
