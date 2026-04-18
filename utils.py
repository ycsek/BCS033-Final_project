# utils.py
"""Data loading utilities.

Provides ``get_dataloaders`` which returns train, test, and eval-train
data loaders for the supported datasets.
"""

from __future__ import annotations

import logging
import os

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def get_dataloaders(
    dataset_name: str = "CIFAR10",
    batch_size: int = 256,
    num_workers: int = 8,
    use_aug: bool = False,
    data_root: str = "/home/syc/experiments/data",
):
    """Create train, test, and eval-train data loaders.

    Parameters
    ----------
    dataset_name : str
        One of ``CIFAR10``, ``SVHN``, ``MNIST``, ``CelebA``.
    batch_size : int
        Mini-batch size.
    num_workers : int
        Number of data-loading workers.
    use_aug : bool
        Whether to apply data augmentation to training data.
    data_root : str
        Root directory for downloaded datasets.

    Returns
    -------
    tuple
        ``(train_loader, test_loader, eval_train_loader, num_classes)``

    Raises
    ------
    ValueError
        If *dataset_name* is not supported.
    """
    os.makedirs(data_root, exist_ok=True)

    if dataset_name == "CIFAR10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        if use_aug:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train_dataset = torchvision.datasets.CIFAR10(
            root=data_root, train=True, download=True, transform=train_transform)
        test_dataset = torchvision.datasets.CIFAR10(
            root=data_root, train=False, download=True, transform=test_transform)
        eval_train_dataset = torchvision.datasets.CIFAR10(
            root=data_root, train=True, download=True, transform=test_transform)
        num_classes = 10

    elif dataset_name == "SVHN":
        mean, std = (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)
        train_transform = transforms.Compose([
            transforms.RandomCrop(
                32, padding=4) if use_aug else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train_dataset = torchvision.datasets.SVHN(
            root=data_root, split='train', download=True, transform=train_transform)
        test_dataset = torchvision.datasets.SVHN(
            root=data_root, split='test', download=True, transform=test_transform)
        eval_train_dataset = torchvision.datasets.SVHN(
            root=data_root, split='train', download=True, transform=test_transform)
        num_classes = 10

    elif dataset_name == "MNIST":
        mean, std = (0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)
        train_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomCrop(
                32, padding=2) if use_aug else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train_dataset = torchvision.datasets.MNIST(
            root=data_root, train=True, download=True, transform=train_transform)
        test_dataset = torchvision.datasets.MNIST(
            root=data_root, train=False, download=True, transform=test_transform)
        eval_train_dataset = torchvision.datasets.MNIST(
            root=data_root, train=True, download=True, transform=test_transform)
        num_classes = 10

    elif dataset_name == "CelebA":
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        train_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip() if use_aug else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        def target_transform(target):
            return target[20]

        train_dataset = torchvision.datasets.CelebA(
            root=data_root, split='train', download=False,
            transform=train_transform, target_type='attr',
            target_transform=target_transform)
        test_dataset = torchvision.datasets.CelebA(
            root=data_root, split='test', download=False,
            transform=test_transform, target_type='attr',
            target_transform=target_transform)
        eval_train_dataset = torchvision.datasets.CelebA(
            root=data_root, split='train', download=False,
            transform=test_transform, target_type='attr',
            target_transform=target_transform)
        num_classes = 2
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": True,
        "prefetch_factor": 2,
    }
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)
    eval_train_loader = DataLoader(
        eval_train_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)

    logger.info(
        "Loaded %s — train: %d, test: %d, num_classes: %d",
        dataset_name, len(train_dataset), len(test_dataset), num_classes,
    )
    return train_loader, test_loader, eval_train_loader, num_classes
