# visualization.py
"""Centralized visualization module.

All plotting functions live here.  Each function:
- accepts data as dicts / numpy arrays / DataFrames (never raw model objects)
- saves figures to *save_dir* in both PNG (300 dpi) and PDF formats
- uses seaborn styling for visual consistency
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

logger = logging.getLogger(__name__)

# ── Global style ────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
_PALETTE = sns.color_palette("Set2")


# ── Helpers ─────────────────────────────────────────────────────────────────

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save_fig(fig: Figure, save_dir: str, name: str) -> None:
    """Save a figure as both PNG and PDF."""
    _ensure_dir(save_dir)
    for ext in ("png", "pdf"):
        out = os.path.join(save_dir, f"{name}.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        logger.info("Saved figure: %s", out)
    plt.close(fig)


# ── Training-curve plots ───────────────────────────────────────────────────

def plot_train_accuracy_curve(
    trajectory: List[Dict],
    save_dir: str,
) -> None:
    """Plot training accuracy vs. epoch.

    Parameters
    ----------
    trajectory : list[dict]
        Each dict must contain ``epoch`` and ``train_acc`` keys.
    save_dir : str
        Directory to write output figures.
    """
    epochs = [r["epoch"] for r in trajectory]
    accs = [r["train_acc"] for r in trajectory]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, accs, marker="o", markersize=3, color=_PALETTE[0],
            linewidth=1.5, label="Train Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Train Accuracy Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_fig(fig, save_dir, "train_accuracy_curve")


def plot_test_accuracy_curve(
    trajectory: List[Dict],
    save_dir: str,
) -> None:
    """Plot test accuracy vs. epoch."""
    epochs = [r["epoch"] for r in trajectory]
    accs = [r["target_acc"] for r in trajectory]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, accs, marker="s", markersize=3, color=_PALETTE[1],
            linewidth=1.5, label="Test Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Test Accuracy Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_fig(fig, save_dir, "test_accuracy_curve")


def plot_loss_curve(
    trajectory: List[Dict],
    save_dir: str,
) -> None:
    """Plot training loss vs. epoch."""
    epochs = [r["epoch"] for r in trajectory]
    losses = [r["target_loss"] for r in trajectory]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, losses, marker="^", markersize=3, color=_PALETTE[2],
            linewidth=1.5, label="Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_fig(fig, save_dir, "loss_curve")


def plot_asr_curve(
    mia_metrics: Dict,
    save_dir: str,
) -> None:
    """Plot MIA evaluation metrics as a grouped bar chart.

    Since ASR is typically evaluated once (post-training), this generates
    a summary bar chart of all attack metrics.

    Parameters
    ----------
    mia_metrics : dict
        Must contain keys: asr, auc_roc, precision, recall, f1.
    save_dir : str
        Output directory.
    """
    metric_names = ["ASR (%)", "AUC-ROC", "Precision", "Recall", "F1"]
    values = [
        mia_metrics["asr"],
        mia_metrics["auc_roc"] * 100,  # scale to %
        mia_metrics["precision"] * 100,
        mia_metrics["recall"] * 100,
        mia_metrics["f1"] * 100,
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(metric_names, values, color=_PALETTE[:len(values)],
                  edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Value (%)")
    ax.set_title("Membership Inference Attack — Summary Metrics")
    ax.set_ylim(0, 105)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}", ha="center", va="bottom", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    _save_fig(fig, save_dir, "asr_summary")


# ── Explainability: PCA / t-SNE ────────────────────────────────────────────

def plot_pca_2d(
    features: np.ndarray,
    labels: np.ndarray,
    save_dir: str,
    title: str = "PCA 2-D Projection — Member vs. Non-Member",
) -> None:
    """2-D PCA scatter of MIA feature vectors, coloured by membership.

    Parameters
    ----------
    features : np.ndarray, shape (n_samples, n_features)
        Concatenated member + non-member feature vectors.
    labels : np.ndarray, shape (n_samples,)
        Binary membership labels (1 = member, 0 = non-member).
    save_dir : str
        Output directory.
    title : str
        Plot title.
    """
    pca = PCA(n_components=2)
    proj = pca.fit_transform(features)

    fig, ax = plt.subplots(figsize=(10, 8))
    for lab, name, colour in [(1, "Member", _PALETTE[0]),
                               (0, "Non-Member", _PALETTE[1])]:
        mask = labels == lab
        ax.scatter(proj[mask, 0], proj[mask, 1], s=8, alpha=0.5,
                   color=colour, label=name)
    ax.set_xlabel(f"PC-1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC-2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title(title)
    ax.legend(markerscale=3)
    ax.grid(True, alpha=0.3)
    _save_fig(fig, save_dir, "pca_2d")


def plot_pca_3d(
    features: np.ndarray,
    labels: np.ndarray,
    save_dir: str,
    title: str = "PCA 3-D Projection — Member vs. Non-Member",
) -> None:
    """3-D PCA scatter of MIA feature vectors."""
    pca = PCA(n_components=3)
    proj = pca.fit_transform(features)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    for lab, name, colour in [(1, "Member", _PALETTE[0]),
                               (0, "Non-Member", _PALETTE[1])]:
        mask = labels == lab
        ax.scatter(proj[mask, 0], proj[mask, 1], proj[mask, 2],
                   s=8, alpha=0.4, color=colour, label=name)
    ax.set_xlabel(f"PC-1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC-2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_zlabel(f"PC-3 ({pca.explained_variance_ratio_[2]:.1%})")
    ax.set_title(title)
    ax.legend(markerscale=3)
    _save_fig(fig, save_dir, "pca_3d")


def plot_tsne_2d(
    features: np.ndarray,
    labels: np.ndarray,
    save_dir: str,
    perplexity: float = 30.0,
    title: str = "t-SNE 2-D Projection — Member vs. Non-Member",
) -> None:
    """2-D t-SNE scatter of MIA feature vectors.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix.
    labels : np.ndarray
        Binary membership labels.
    save_dir : str
        Output directory.
    perplexity : float
        t-SNE perplexity parameter.
    title : str
        Plot title.
    """
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=1000,
                random_state=42)
    proj = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(10, 8))
    for lab, name, colour in [(1, "Member", _PALETTE[0]),
                               (0, "Non-Member", _PALETTE[1])]:
        mask = labels == lab
        ax.scatter(proj[mask, 0], proj[mask, 1], s=8, alpha=0.5,
                   color=colour, label=name)
    ax.set_xlabel("t-SNE Dim 1")
    ax.set_ylabel("t-SNE Dim 2")
    ax.set_title(title)
    ax.legend(markerscale=3)
    ax.grid(True, alpha=0.3)
    _save_fig(fig, save_dir, "tsne_2d")


# ── Grad-CAM visualization ─────────────────────────────────────────────────

def plot_gradcam(
    images: np.ndarray,
    cam_maps: np.ndarray,
    preds: Sequence[int],
    save_dir: str,
    max_images: int = 8,
) -> None:
    """Overlay Grad-CAM heatmaps on input images.

    Parameters
    ----------
    images : np.ndarray, shape (N, H, W, 3)
        Normalized images in [0, 1] range, channel-last.
    cam_maps : np.ndarray, shape (N, H, W)
        Grad-CAM activation maps.
    preds : sequence of int
        Predicted class indices.
    save_dir : str
        Output directory.
    max_images : int
        Maximum number of images to display.
    """
    n = min(max_images, len(images))
    fig, axes = plt.subplots(2, n // 2, figsize=(3 * (n // 2), 6))
    axes = axes.flatten()
    for i in range(n):
        ax = axes[i]
        img = images[i]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        ax.imshow(img)
        ax.imshow(cam_maps[i], cmap="jet", alpha=0.5)
        ax.set_title(f"Pred: {preds[i]}")
        ax.axis("off")
    fig.suptitle("Grad-CAM Interpretability", fontsize=14)
    _save_fig(fig, save_dir, "gradcam_visualization")


# ── Combined training trajectory plot ──────────────────────────────────────

def plot_training_trajectory(
    trajectory: List[Dict],
    save_dir: str,
) -> None:
    """Combined plot showing train/test accuracy on one figure."""
    epochs = [r["epoch"] for r in trajectory]
    train_accs = [r["train_acc"] for r in trajectory]
    test_accs = [r["target_acc"] for r in trajectory]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_accs, marker="o", markersize=3, color=_PALETTE[0],
            linewidth=1.5, label="Train Acc")
    ax.plot(epochs, test_accs, marker="s", markersize=3, color=_PALETTE[1],
            linewidth=1.5, label="Test Acc")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Training Trajectory")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_fig(fig, save_dir, "training_trajectory")
