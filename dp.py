# dp.py
import torch
import torch.nn as nn
from torchvision.models import resnet50
from opacus.validators import ModuleValidator
from tqdm import tqdm


def get_target_model(num_classes, device):
    model = resnet50(num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                            stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    if not ModuleValidator.is_valid(model):
        model = ModuleValidator.fix(model)
    return model.to(device)


def train_target_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    pbar = tqdm(train_loader, desc="Target Train", leave=True)
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    return running_loss / len(train_loader)


def evaluate_model(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total
