# dp.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from tqdm import tqdm


def get_target_model(num_classes, device):
    model = resnet18(num_classes=num_classes)
    if not ModuleValidator.is_valid(model):
        model = ModuleValidator.fix(model)
    return model.to(device)


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


def train_target_model(model, train_loader, test_loader, device, args, logger):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    privacy_engine = None
    if args.use_dp:
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=args.noise_multiplier,
            max_grad_norm=args.max_grad_norm,
        )
        logger.info("--- DP-SGD Target Training Started ---")
    else:
        logger.info("--- Standard Target Training Started ---")

    for epoch in range(args.target_epochs):
        model.train()
        running_loss = 0.0

        # 使用 tqdm 包装 train_loader
        pbar = tqdm(
            train_loader, desc=f"Target Epoch {epoch+1}/{args.target_epochs}", leave=True)
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = evaluate_model(model, test_loader, device)

        epsilon = None
        if args.use_dp:
            epsilon = privacy_engine.get_epsilon(1e-5)
            logger.info(
                f"Target Epoch {epoch+1} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}% | ε: {epsilon:.2f}")
        else:
            logger.info(
                f"Target Epoch {epoch+1} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")

        logger.log_target_epoch(epoch+1, epoch_loss, epoch_acc, epsilon)

    return model
