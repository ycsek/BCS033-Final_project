# mia.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support  # 新增

# MIA Attack Model Definition, simple MLP
class AttackModel(nn.Module):
    def __init__(self, num_classes):
        super(AttackModel, self).__init__()
        input_dim = num_classes * 2 + 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# Get features for MIA attack model, White-box setting
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
            loss_values = criterion(outputs, targets).unsqueeze(1)
            one_hot_labels = F.one_hot(
                targets, num_classes=num_classes).float()
            entropy = -torch.sum(probs * torch.log(probs +
                                 1e-10), dim=1, keepdim=True)
            batch_features = torch.cat(
                (sorted_probs, loss_values, one_hot_labels, entropy), dim=1)
            features.append(batch_features.cpu().numpy())
            lbl = np.ones(inputs.size(0)) if is_member else np.zeros(
                inputs.size(0))
            labels.append(lbl)
    return np.vstack(features), np.concatenate(labels)


def evaluate_mia_vulnerability(target_model, train_loader, test_loader, num_classes, device, args):
    # Extract features
    feat_in, lab_in = extract_strong_features(
        target_model, train_loader, True, num_classes, device)
    feat_out, lab_out = extract_strong_features(
        target_model, test_loader, False, num_classes, device)

    # Balance dataset
    min_len = min(len(feat_in), len(feat_out))
    X = np.vstack((feat_in[:min_len], feat_out[:min_len]))
    Y = np.concatenate((lab_in[:min_len], lab_out[:min_len]))

    # Train/test split
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split = int(0.8 * len(X))
    train_idx, test_idx = indices[:split], indices[split:]

    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]

    # DataLoaders
    train_tensor = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                 torch.tensor(Y_train, dtype=torch.float32).unsqueeze(1))
    test_tensor = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                torch.tensor(Y_test, dtype=torch.float32).unsqueeze(1))

    attack_train_loader = DataLoader(
        train_tensor, batch_size=512, shuffle=True)
    attack_test_loader = DataLoader(test_tensor, batch_size=512, shuffle=False)

    # Train attack model
    attack_model = AttackModel(num_classes).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(attack_model.parameters(),
                           lr=args.mia_lr, weight_decay=1e-4)

    logger_msg = "MIA Probing (with proper train/test split)"
    pbar = tqdm(range(args.mia_epochs), desc=logger_msg,
                leave=True, dynamic_ncols=False)

    for epoch in pbar:
        attack_model.train()
        for inputs, targets in attack_train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = attack_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    # Final evaluation on test set with full metrics
    attack_model.eval()
    test_probs = []
    test_labels = []
    with torch.no_grad():
        for inputs, targets in attack_test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = attack_model(inputs)
            test_probs.append(outputs.cpu().numpy())
            test_labels.append(targets.cpu().numpy().flatten())

    test_probs = np.concatenate(test_probs).flatten()
    test_labels = np.concatenate(test_labels)

    # Compute metrics
    final_test_asr = 100. * ((test_probs > 0.5) == test_labels).mean()
    auc_roc = roc_auc_score(test_labels, test_probs)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, (test_probs > 0.5).astype(int), average='binary', zero_division=0)

    pbar.set_postfix({
        'ASR': f"{final_test_asr:.2f}%",
        'AUC': f"{auc_roc:.4f}",
        'Prec': f"{precision:.4f}",
        'Rec': f"{recall:.4f}"
    })

    return {
        "asr": final_test_asr,
        "auc_roc": auc_roc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
