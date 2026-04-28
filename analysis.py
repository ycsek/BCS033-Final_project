# analysis.py
"""Post-training analysis: Grad-CAM, ECE, top-k accuracy.

PSNR / SSIM evaluation has been **removed** per project requirements.
All plotting is delegated to ``visualization.py``.
"""

from __future__ import annotations

import json
import logging
import os
from types import SimpleNamespace
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from visualization import plot_gradcam, plot_training_trajectory

logger = logging.getLogger(__name__)


# ── Grad-CAM ────────────────────────────────────────────────────────────────

class GradCAM:
    """Gradient-weighted Class Activation Mapping.

    Parameters
    ----------
    model : nn.Module
        Target model (must support gradient computation on the target layer).
    target_layer : nn.Module
        Convolutional layer to extract activations from.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        target_layer.register_forward_hook(self._save_activation)

    def _save_activation(
        self, module: nn.Module, inp: tuple, output: torch.Tensor
    ) -> None:
        self.activations = output

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate Grad-CAM activation maps.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Input images, shape ``(B, C, H, W)``.
        target_class : torch.Tensor or None
            Target class indices; if ``None``, uses argmax predictions.

        Returns
        -------
        torch.Tensor
            CAM maps of shape ``(B, H', W')``, normalized to [0, 1].
        """
        self.model.zero_grad()
        input_tensor = input_tensor.requires_grad_(True)
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1)
        score = output[range(len(target_class)), target_class]
        gradients = torch.autograd.grad(
            score, self.activations, torch.ones_like(score)
        )[0]

        gradients = gradients.mean(dim=[2, 3], keepdim=True)
        cam = F.relu(self.activations * gradients).mean(dim=1)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam


# ── Expected Calibration Error ──────────────────────────────────────────────

def compute_ece(
    preds: torch.Tensor, labels: torch.Tensor, n_bins: int = 15
) -> float:
    """Compute the Expected Calibration Error (ECE).

    Parameters
    ----------
    preds : torch.Tensor
        Softmax probability matrix, shape ``(N, C)``.
    labels : torch.Tensor
        Ground-truth labels, shape ``(N,)``.
    n_bins : int
        Number of calibration bins.

    Returns
    -------
    float
        ECE value in [0, 1].
    """
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences, predictions = torch.max(preds, 1)
    accuracies = predictions.eq(labels)

    ece = torch.zeros(1, device=preds.device)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(
            bin_upper.item()
        )
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin -
                             accuracy_in_bin) * prop_in_bin
    return ece.item()


# ── Main analysis entry point ──────────────────────────────────────────────

def run_analysis(
    log_dir: str,
    target_model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    cfg: SimpleNamespace,
) -> Dict[str, float]:
    """Run post-training analysis: top-k accuracy, ECE, Grad-CAM.

    Saves Grad-CAM and training-trajectory visualizations to the figures
    directory inside *log_dir*.

    Parameters
    ----------
    log_dir : str
        Path to the current experiment log directory.
    target_model : nn.Module
        Trained target model.
    test_loader : DataLoader
        Test data loader.
    device : torch.device
        Compute device.
    cfg : SimpleNamespace
        Full experiment configuration.

    Returns
    -------
    dict
        Additional metrics (test_loss, top1_acc, top5_acc, ece).
    """
    target_model.eval()
    additional_metrics: Dict[str, float] = {}

    # ── Test-set metrics ────────────────────────────────────────────
    test_loss = 0.0
    top1_correct, top5_correct, total = 0, 0, 0
    criterion = nn.CrossEntropyLoss()

    all_probs, all_labels = [], []
    k_val = 5

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = target_model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            probs = F.softmax(outputs, dim=1)
            all_probs.append(probs)
            all_labels.append(targets)

            k_val = min(5, outputs.size(1))
            _, pred = outputs.topk(k_val, 1, True, True)

            top1_correct += pred[:, 0].eq(targets).sum().item()
            top5_correct += (
                pred.eq(targets.view(-1, 1).expand_as(pred)).sum().item()
            )
            total += targets.size(0)

    all_probs_t = torch.cat(all_probs)
    all_labels_t = torch.cat(all_labels)
    ece_val = compute_ece(all_probs_t, all_labels_t)

    additional_metrics.update(
        {
            "test_loss": test_loss / len(test_loader),
            "top1_acc": 100.0 * top1_correct / total,
            "top5_acc": 100.0 * top5_correct / total,
            "ece": ece_val,
        }
    )

    # ── Grad-CAM ────────────────────────────────────────────────────
    figures_dir = os.path.join(log_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    batch = next(iter(test_loader))
    images = batch[0].to(device)[:16]
    labels_batch = batch[1].to(device)[:16]

    # Build a clean (Opacus-free) copy for Grad-CAM so that no
    # GradSampleModule backward hooks interfere with autograd.grad.
    from dp import get_target_model as _build_model

    raw_model = (
        target_model._module
        if hasattr(target_model, "_module")
        else target_model
    )
    clean_model = _build_model(
        num_classes=raw_model.fc.out_features, device=device
    )
    clean_model.load_state_dict(raw_model.state_dict())
    clean_model.eval()

    target_layer = dict(clean_model.named_modules())["layer4"]
    gradcam = GradCAM(clean_model, target_layer)

    with torch.no_grad():
        preds = target_model(images).argmax(dim=1)

    cam_maps = gradcam.generate(images, preds)
    cam_maps = F.interpolate(
        cam_maps.unsqueeze(1), size=(32, 32), mode="bilinear",
        align_corners=False,
    ).squeeze(1)

    # Convert to numpy for the visualisation function
    images_np = images.detach().cpu().permute(0, 2, 3, 1).numpy()
    cam_np = cam_maps.detach().cpu().numpy()
    preds_list = preds.cpu().tolist()
    plot_gradcam(images_np, cam_np, preds_list, figures_dir)

    results_path = os.path.join(log_dir, "results.json")
    if os.path.isfile(results_path):
        with open(results_path, "r", encoding="utf-8") as fh:
            results = json.load(fh)
        plot_training_trajectory(results["trajectory"], figures_dir)

    if os.path.isfile(results_path):
        with open(results_path, "r", encoding="utf-8") as fh:
            results = json.load(fh)
        results["additional_metrics"] = additional_metrics
        with open(results_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=4, ensure_ascii=False)

    logger.info(
        "Analysis complete — Test Loss: %.4f | Top-1 Acc: %.2f%% | Top-%d Acc: %.2f%% | ECE: %.4f",
        additional_metrics["test_loss"],
        additional_metrics["top1_acc"],
        k_val,
        additional_metrics["top5_acc"],
        ece_val,
    )

    return additional_metrics
