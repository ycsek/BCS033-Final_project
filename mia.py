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
        # 输入特征维度 = 排序后的置信度(K) + Loss(1) + 真实标签的One-hot编码(K) = 2K + 1
        input_dim = num_classes * 2 + 1

        # 增加网络容量，并引入 Dropout 防止攻击模型自身过拟合
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def extract_strong_features(model, dataloader, is_member, num_classes, device):
    """提取用于高强度 MIA 的复合特征"""
    model.eval()
    features = []
    labels = []

    # 不进行 reduction，以获取每个样本的独立 Loss
    criterion = nn.CrossEntropyLoss(reduction='none')
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            # 1. 置信度特征 (按降序排列)
            probs = softmax(outputs)
            sorted_probs, _ = torch.sort(probs, dim=1, descending=True)

            # 2. 损失特征
            loss_values = criterion(outputs, targets).unsqueeze(1)

            # 3. 标签特征 (One-hot)
            one_hot_labels = F.one_hot(
                targets, num_classes=num_classes).float()

            # 将三者拼接成强大的特征向量: [Sorted_Probs, Loss, One_Hot]
            batch_features = torch.cat(
                (sorted_probs, loss_values, one_hot_labels), dim=1)

            features.append(batch_features.cpu().numpy())
            lbl = np.ones(inputs.size(0)) if is_member else np.zeros(
                inputs.size(0))
            labels.append(lbl)

    return np.vstack(features), np.concatenate(labels)


def train_mia_model(target_model, train_loader, test_loader, num_classes, device, args, logger):
    logger.info("--- Extracting Strong Features for MIA ---")
    feat_in, lab_in = extract_strong_features(
        target_model, train_loader, True, num_classes, device)
    feat_out, lab_out = extract_strong_features(
        target_model, test_loader, False, num_classes, device)

    # 保持正负样本平衡
    min_len = min(len(feat_in), len(feat_out))
    X = np.vstack((feat_in[:min_len], feat_out[:min_len]))
    Y = np.concatenate((lab_in[:min_len], lab_out[:min_len]))

    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32).unsqueeze(1)

    # 使用更大的 Batch Size 加速攻击模型的收敛
    attack_loader = DataLoader(TensorDataset(
        X_tensor, Y_tensor), batch_size=256, shuffle=True)

    attack_model = AttackModel(num_classes).to(device)
    criterion = nn.BCELoss()
    # 使用 args 中的 mia_lr，并加入轻微的 weight_decay
    optimizer = optim.Adam(attack_model.parameters(),
                           lr=args.mia_lr, weight_decay=1e-5)

    logger.info("--- Powerful MIA Classifier Training Started ---")
    for epoch in range(args.mia_epochs):
        attack_model.train()
        running_loss = 0.0
        correct, total = 0, 0

        pbar = tqdm(attack_loader,
                    desc=f"MIA Epoch {epoch+1}/{args.mia_epochs}", leave=False)
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = attack_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = (outputs > 0.5).float()
            total += targets.size(0)
            correct += (preds == targets).sum().item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        epoch_loss = running_loss / len(attack_loader)
        asr = 100. * correct / total

        logger.info(
            f"MIA Epoch {epoch+1} | Loss: {epoch_loss:.4f} | ASR: {asr:.2f}%")
        logger.log_mia_epoch(epoch+1, epoch_loss, asr)

    return asr
