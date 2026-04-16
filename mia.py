# mia.py
"""Membership Inference Attack (MIA) module.

Extracts strong feature vectors from a target model's outputs and trains
an XGBoost binary classifier to distinguish members from non-members.

Returns evaluation metrics **and** the raw features / fitted model so
they can be used for explainability (PCA/t-SNE) and model persistence.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


# ── Feature extraction ──────────────────────────────────────────────────────

def extract_strong_features(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    is_member: bool,
    num_classes: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract rich feature vectors from a target model for MIA.

    For each sample the feature vector consists of:
    - Sorted softmax probabilities  (num_classes)
    - Sorted raw logits             (num_classes)
    - Per-sample cross-entropy loss (1)
    - One-hot encoded true label    (num_classes)
    - Prediction entropy            (1)

    Parameters
    ----------
    model : nn.Module
        Target model in eval mode.
    dataloader : DataLoader
        Data loader for member or non-member data.
    is_member : bool
        ``True`` if the data are training (member) samples.
    num_classes : int
        Number of classes in the dataset.
    device : torch.device
        Compute device.

    Returns
    -------
    features : np.ndarray, shape (N, feature_dim)
    labels : np.ndarray, shape (N,)
        Binary membership labels (1 = member, 0 = non-member).
    """
    model.eval()
    features, labels = [], []
    criterion = nn.CrossEntropyLoss(reduction="none")
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            probs = softmax(outputs)
            sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
            sorted_logits, _ = torch.sort(outputs, dim=1, descending=True)

            loss_values = criterion(outputs, targets).unsqueeze(1)
            one_hot_labels = F.one_hot(
                targets, num_classes=num_classes
            ).float()
            entropy = -torch.sum(
                probs * torch.log(probs + 1e-10), dim=1, keepdim=True
            )

            batch_features = torch.cat(
                (sorted_probs, sorted_logits, loss_values,
                 one_hot_labels, entropy),
                dim=1,
            )
            features.append(batch_features.cpu().numpy())
            lbl = (
                np.ones(inputs.size(0))
                if is_member
                else np.zeros(inputs.size(0))
            )
            labels.append(lbl)

    return np.vstack(features), np.concatenate(labels)


# ── Utility ─────────────────────────────────────────────────────────────────

def _get_tpr_at_fpr(
    fpr: np.ndarray, tpr: np.ndarray, target_fpr: float
) -> float:
    """Return TPR at a given FPR threshold using the ROC curve."""
    idx = np.where(fpr <= target_fpr)[0][-1]
    return float(tpr[idx])


# ── Main evaluation entry point ────────────────────────────────────────────

def evaluate_mia_vulnerability(
    target_model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    num_classes: int,
    device: torch.device,
    cfg: SimpleNamespace,
) -> Dict[str, Any]:
    """Run a full MIA evaluation using XGBoost.

    Parameters
    ----------
    target_model : nn.Module
        Trained target model.
    train_loader, test_loader : DataLoader
        Member and non-member data loaders.
    num_classes : int
        Number of dataset classes.
    device : torch.device
        Compute device.
    cfg : SimpleNamespace
        Full experiment configuration (attack hypers read from ``cfg.attack``).

    Returns
    -------
    dict
        Keys:
        - Standard metrics: asr, auc_roc, precision, recall, f1
        - Low-FPR TPR: tpr_0_1, tpr_1, tpr_5
        - Raw data: features (np.ndarray), labels (np.ndarray)
        - Fitted model: attack_model (XGBClassifier)
    """
    # ── Extract features ────────────────────────────────────────────
    feat_in, lab_in = extract_strong_features(
        target_model, train_loader, True, num_classes, device
    )
    feat_out, lab_out = extract_strong_features(
        target_model, test_loader, False, num_classes, device
    )

    # Balance classes
    min_len = min(len(feat_in), len(feat_out))
    X = np.vstack((feat_in[:min_len], feat_out[:min_len]))
    Y = np.concatenate((lab_in[:min_len], lab_out[:min_len]))

    # Train / test split (80/20)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split = int(0.8 * len(X))
    train_idx, test_idx = indices[:split], indices[split:]

    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]

    # ── Train XGBoost attack model ──────────────────────────────────
    atk = cfg.attack
    logger.info(
        "Training XGBoost attack model (feature dim: %d) ...", X_train.shape[1]
    )
    attack_model = XGBClassifier(
        n_estimators=atk.n_estimators,
        max_depth=atk.max_depth,
        learning_rate=atk.learning_rate,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    attack_model.fit(X_train, Y_train)

    # ── Evaluate ────────────────────────────────────────────────────
    test_probs = attack_model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs > 0.5).astype(int)

    asr = 100.0 * (test_preds == Y_test).mean()
    auc_roc = roc_auc_score(Y_test, test_probs)
    precision, recall, f1, _ = precision_recall_fscore_support(
        Y_test, test_preds, average="binary", zero_division=0
    )

    fpr, tpr, _ = roc_curve(Y_test, test_probs)
    tpr_0_1 = _get_tpr_at_fpr(fpr, tpr, 0.001) * 100.0
    tpr_1 = _get_tpr_at_fpr(fpr, tpr, 0.01) * 100.0
    tpr_5 = _get_tpr_at_fpr(fpr, tpr, 0.05) * 100.0

    return {
        # Metrics
        "asr": asr,
        "auc_roc": auc_roc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tpr_0_1": tpr_0_1,
        "tpr_1": tpr_1,
        "tpr_5": tpr_5,
        # Raw data for explainability
        "features": X,
        "labels": Y,
        # Fitted attack model for persistence
        "attack_model": attack_model,
    }
