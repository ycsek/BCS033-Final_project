"""Membership Inference Attack (MIA) module."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


def extract_strong_features(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    is_member: bool,
    num_classes: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
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
            is_correct = (outputs.argmax(dim=1) ==
                          targets).float().unsqueeze(1)
            one_hot_labels = F.one_hot(
                targets, num_classes=num_classes
            ).float()
            entropy = -torch.sum(
                probs * torch.log(probs + 1e-10), dim=1, keepdim=True
            )

            batch_features = torch.cat(
                (sorted_probs, sorted_logits, loss_values, is_correct,
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


def _get_tpr_at_fpr(
    fpr: np.ndarray, tpr: np.ndarray, target_fpr: float
) -> float:
    idx = np.where(fpr <= target_fpr)[0][-1]
    return float(tpr[idx])


def evaluate_mia_vulnerability(
    target_model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    num_classes: int,
    device: torch.device,
    cfg: SimpleNamespace,
) -> Dict[str, Any]:
    # ── Extract features ────────────────────────────────────────────
    feat_in, lab_in = extract_strong_features(
        target_model, train_loader, True, num_classes, device
    )
    feat_out, lab_out = extract_strong_features(
        target_model, test_loader, False, num_classes, device
    )

    # Balance classes and shuffle
    min_len = min(len(feat_in), len(feat_out))

    np.random.seed(42)
    idx_in = np.random.permutation(len(feat_in))[:min_len]
    idx_out = np.random.permutation(len(feat_out))[:min_len]

    X = np.vstack((feat_in[idx_in], feat_out[idx_out]))
    Y = np.concatenate((lab_in[idx_in], lab_out[idx_out]))

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
    pr_precision, pr_recall, _ = precision_recall_curve(Y_test, test_probs)
    tpr_0_1 = _get_tpr_at_fpr(fpr, tpr, 0.001) * 100.0
    tpr_1 = _get_tpr_at_fpr(fpr, tpr, 0.01) * 100.0
    tpr_5 = _get_tpr_at_fpr(fpr, tpr, 0.05) * 100.0

    return {
        "asr": asr,
        "auc_roc": auc_roc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tpr_0_1": tpr_0_1,
        "tpr_1": tpr_1,
        "tpr_5": tpr_5,
        "fpr": fpr,
        "tpr": tpr,
        "pr_precision": pr_precision,
        "pr_recall": pr_recall,
        "features": X,
        "labels": Y,
        "attack_model": attack_model,
    }
