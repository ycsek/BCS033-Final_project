# mia.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F


class AttackModel(nn.Module):
    def __init__(self, num_classes):
        super(AttackModel, self).__init__()
        input_dim = num_classes * 2 + 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


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
    """提取当前 Target Model 的特征，训练一个新的攻击模型，返回 ASR"""
    feat_in, lab_in = extract_strong_features(
        target_model, train_loader, True, num_classes, device)
    feat_out, lab_out = extract_strong_features(
        target_model, test_loader, False, num_classes, device)

    min_len = min(len(feat_in), len(feat_out))
    X = np.vstack((feat_in[:min_len], feat_out[:min_len]))
    Y = np.concatenate((lab_in[:min_len], lab_out[:min_len]))

    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32).unsqueeze(1)
    attack_loader = DataLoader(TensorDataset(
        X_tensor, Y_tensor), batch_size=512, shuffle=True)

    # 每次评估必须重新初始化攻击模型，防止过去 epoch 的权重残留污染测试结果
    attack_model = AttackModel(num_classes).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(attack_model.parameters(),
                           lr=args.mia_lr, weight_decay=1e-4)

    final_asr = 0.0
    pbar = tqdm(range(args.mia_epochs), desc="MIA Probing", leave=True)

    for epoch in pbar:
        attack_model.train()
        correct, total = 0, 0
        for inputs, targets in attack_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = attack_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            preds = (outputs > 0.5).float()
            total += targets.size(0)
            correct += (preds == targets).sum().item()

        final_asr = 100. * correct / total
        pbar.set_postfix({'asr': f"{final_asr:.2f}%"})

    return final_asr
