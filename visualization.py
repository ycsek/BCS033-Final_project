# visualization.py
"""Centralized visualization module.

All plotting functions live here.  Each function:
- accepts data as dicts / numpy arrays / DataFrames (never raw model objects)
- saves figures to *save_dir* as PNG (300 dpi)
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
    """Save a figure as PNG only."""
    _ensure_dir(save_dir)
    out = os.path.join(save_dir, f"{name}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    logger.info("Saved figure: %s", out)
    plt.close(fig)


# ── Training-curve plots ───────────────────────────────────────────────────

def plot_accuracy_curve(
    trajectory: List[Dict],
    save_dir: str,
) -> None:
    """Plot training and test accuracy on a single figure.

    Parameters
    ----------
    trajectory : list[dict]
        Each dict must contain ``epoch``, ``train_acc``, and ``target_acc``.
    save_dir : str
        Directory to write output figures.
    """
    epochs = [r["epoch"] for r in trajectory]
    train_accs = [r["train_acc"] for r in trajectory]
    test_accs = [r["target_acc"] for r in trajectory]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_accs, marker="o", markersize=3, color=_PALETTE[0],
            linewidth=1.5, label="Train Accuracy")
    ax.plot(epochs, test_accs, marker="s", markersize=3, color=_PALETTE[1],
            linewidth=1.5, label="Test Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Training & Test Accuracy Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_fig(fig, save_dir, "accuracy_curve")


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
    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=1000,
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


# ── Calibration: Reliability Diagram ──────────────────────────────────────

def plot_reliability_diagram(
    prob_true: np.ndarray,
    prob_pred: np.ndarray,
    save_dir: str,
) -> None:
    """Plot a reliability diagram (calibration curve).

    Visualizes the deviation between the model's predicted confidence
    and the observed (empirical) accuracy for each confidence bin,
    alongside a perfect-calibration reference line.

    Parameters
    ----------
    prob_true : np.ndarray, shape (n_bins,)
        Fraction of positives (empirical accuracy) in each bin, as
        returned by ``sklearn.calibration.calibration_curve``.
    prob_pred : np.ndarray, shape (n_bins,)
        Mean predicted probability (confidence) in each bin.
    save_dir : str
        Directory to write the output figure.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Perfect calibration reference
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1.2,
            label="Perfect Calibration")

    # Model calibration curve
    ax.plot(prob_pred, prob_true, marker="o", markersize=5,
            color=_PALETTE[0], linewidth=1.8, label="Model")

    # Shade the gap between model and perfect calibration
    ax.fill_between(prob_pred, prob_pred, prob_true, alpha=0.15,
                    color=_PALETTE[0])

    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability Diagram — Model Calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    _save_fig(fig, save_dir, "reliability_diagram")


# ── ROC & Precision-Recall curves ─────────────────────────────────────────

def plot_roc_pr_curves(
    fpr: np.ndarray,
    tpr: np.ndarray,
    precision: np.ndarray,
    recall: np.ndarray,
    save_dir: str,
) -> None:
    """Plot ROC and Precision-Recall curves side by side.

    Produces a 1×2 subplot figure summarizing the MIA attack
    performance from two complementary perspectives.

    Parameters
    ----------
    fpr : np.ndarray, shape (n_thresholds,)
        False positive rates from ``sklearn.metrics.roc_curve``.
    tpr : np.ndarray, shape (n_thresholds,)
        True positive rates from ``sklearn.metrics.roc_curve``.
    precision : np.ndarray, shape (n_thresholds_pr,)
        Precision values from ``sklearn.metrics.precision_recall_curve``.
    recall : np.ndarray, shape (n_thresholds_pr,)
        Recall values from ``sklearn.metrics.precision_recall_curve``.
    save_dir : str
        Directory to write the output figure.
    """
    # Compute AUC for the ROC curve
    auc_val = np.trapz(tpr, fpr)

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        "Membership Inference Attack — ROC & Precision-Recall Curves",
        fontsize=15, fontweight="bold", y=1.01,
    )

    # ── Left: ROC Curve ─────────────────────────────────────────
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray",
                linewidth=1.0, label="Random Classifier")
    ax_roc.plot(fpr, tpr, color=_PALETTE[0], linewidth=1.8,
                label=f"MIA Attack (AUC = {auc_val:.4f})")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve — MIA Attack")
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1.02)
    ax_roc.legend(loc="lower right")
    ax_roc.grid(True, alpha=0.3)

    # ── Right: Precision-Recall Curve ───────────────────────────
    ax_pr.plot(recall, precision, color=_PALETTE[1], linewidth=1.8,
               label="MIA Attack")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curve — MIA Attack")
    ax_pr.set_xlim(0, 1)
    ax_pr.set_ylim(0, 1.02)
    ax_pr.legend(loc="upper right")
    ax_pr.grid(True, alpha=0.3)

    fig.tight_layout()
    _save_fig(fig, save_dir, "mia_roc_pr_curves")
