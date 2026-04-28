# dp.py
"""Target model definition and training utilities.

Uses a ResNet-50 adapted for 32×32 inputs with GroupNorm (required by
Opacus / DP-SGD).  Includes patched forward methods for in-place
operation compatibility.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from opacus.validators import ModuleValidator
from torchvision.models import resnet50
from torchvision.models.resnet import BasicBlock, Bottleneck
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ── Patch ResNet forward methods for Opacus compatibility ───────────────────
# Opacus requires no in-place operations.  The default torchvision
# Bottleneck/BasicBlock use ``+= identity`` which is in-place on older
# versions.  We patch the forward to use ``out = out + identity``.

def _patched_bottleneck_forward(self, x: torch.Tensor) -> torch.Tensor:
    identity = x
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)
    out = self.conv3(out)
    out = self.bn3(out)
    if self.downsample is not None:
        identity = self.downsample(x)
    out = out + identity
    out = self.relu(out)
    return out


def _patched_basicblock_forward(self, x: torch.Tensor) -> torch.Tensor:
    identity = x
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)
    if self.downsample is not None:
        identity = self.downsample(x)
    out = out + identity
    out = self.relu(out)
    return out


Bottleneck.forward = _patched_bottleneck_forward
BasicBlock.forward = _patched_basicblock_forward 


# ── Model factory ──────────────────────────────────────────────────────────

def get_target_model(num_classes: int, device: torch.device) -> nn.Module:
    """Create a ResNet-50 adapted for 32×32 images with GroupNorm.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    device : torch.device
        Target device.

    Returns
    -------
    nn.Module
        Model moved to *device*, ready for Opacus wrapping.
    """
    model = resnet50(num_classes=num_classes)

    # Adapt for 32×32 input (remove aggressive downsampling)
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = nn.Identity()

    # BatchNorm → GroupNorm (required by Opacus)
    if not ModuleValidator.is_valid(model):
        model = ModuleValidator.fix(model)

    # Ensure no in-place operations remain
    for module in model.modules():
        if hasattr(module, "inplace"):
            module.inplace = False

    logger.info("Target model created: ResNet-50 (%d classes)", num_classes)
    return model.to(device)


# ── Training ───────────────────────────────────────────────────────────────

def train_target_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int = 1,
    total_epochs: int = 1,
) -> float:
    """Train the target model for one epoch.

    Parameters
    ----------
    model : nn.Module
        Target model.
    train_loader : DataLoader
        Training data loader (may be a BatchMemoryManager wrapper).
    optimizer : Optimizer
        Optimizer (possibly wrapped by Opacus).
    criterion : nn.Module
        Loss function.
    device : torch.device
        Compute device.
    epoch, total_epochs : int
        Current / total epoch numbers (for progress display).

    Returns
    -------
    float
        Average training loss over the epoch.
    """
    model.train()
    running_loss = 0.0

    pbar_desc = f"Epoch [{epoch}/{total_epochs}] Train"
    pbar = tqdm(train_loader, desc=pbar_desc, leave=True, dynamic_ncols=False)

    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return running_loss / len(train_loader)


# ── Evaluation ─────────────────────────────────────────────────────────────

def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """Evaluate model accuracy.

    Parameters
    ----------
    model : nn.Module
        Target model.
    dataloader : DataLoader
        Evaluation data loader.
    device : torch.device
        Compute device.

    Returns
    -------
    float
        Accuracy in percent.
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total
