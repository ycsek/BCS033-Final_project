# mia.py
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, roc_curve
from xgboost import XGBClassifier


def extract_strong_features(model, dataloader, is_member, num_classes, device):
    model.eval()
    features, labels = [], []
    criterion = nn.CrossEntropyLoss(reduction='none')
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
                targets, num_classes=num_classes).float()
            entropy = -torch.sum(probs * torch.log(probs +
                                 1e-10), dim=1, keepdim=True)

            batch_features = torch.cat(
                (sorted_probs, sorted_logits, loss_values, one_hot_labels, entropy), dim=1)
            features.append(batch_features.cpu().numpy())
            lbl = np.ones(inputs.size(0)) if is_member else np.zeros(
                inputs.size(0))
            labels.append(lbl)
    return np.vstack(features), np.concatenate(labels)


def get_tpr_at_fpr(fpr, tpr, target_fpr):
    idx = np.where(fpr <= target_fpr)[0][-1]
    return tpr[idx]


def evaluate_mia_vulnerability(target_model, train_loader, test_loader, num_classes, device, args):
    feat_in, lab_in = extract_strong_features(
        target_model, train_loader, True, num_classes, device)
    feat_out, lab_out = extract_strong_features(
        target_model, test_loader, False, num_classes, device)

    min_len = min(len(feat_in), len(feat_out))
    X = np.vstack((feat_in[:min_len], feat_out[:min_len]))
    Y = np.concatenate((lab_in[:min_len], lab_out[:min_len]))

    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split = int(0.8 * len(X))
    train_idx, test_idx = indices[:split], indices[split:]

    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]

    print(f"\n XGBoost Attack Model (Feature Dim: {X_train.shape[1]})...")
    attack_model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    attack_model.fit(X_train, Y_train)

    test_probs = attack_model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs > 0.5).astype(int)

    final_test_asr = 100. * (test_preds == Y_test).mean()
    auc_roc = roc_auc_score(Y_test, test_probs)
    precision, recall, f1, _ = precision_recall_fscore_support(
        Y_test, test_preds, average='binary', zero_division=0)

    # TPR @ low FPR
    fpr, tpr, thresholds = roc_curve(Y_test, test_probs)
    tpr_0_1 = get_tpr_at_fpr(fpr, tpr, 0.001) * 100.0
    tpr_1 = get_tpr_at_fpr(fpr, tpr, 0.01) * 100.0
    tpr_5 = get_tpr_at_fpr(fpr, tpr, 0.05) * 100.0

    return {
        "asr": final_test_asr,
        "auc_roc": auc_roc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tpr_0_1": tpr_0_1,
        "tpr_1": tpr_1,
        "tpr_5": tpr_5
    }
