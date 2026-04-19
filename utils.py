# utils.py
"""Data loading utilities.

Provides ``get_dataloaders`` which returns train, test, and eval-train
data loaders for the supported datasets.
"""

from __future__ import annotations

import logging
import os

import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class CelebADataset(Dataset):
    """Lightweight CelebA loader that reads directly from a local directory.

    This avoids ``torchvision.datasets.CelebA``'s strict integrity checks
    which fail when the dataset was downloaded / extracted manually.

    Expected directory layout under *root*::

        root/
            img_align_celeba/
                000001.jpg
                000002.jpg
                ...
            list_eval_partition.txt
            list_attr_celeba.txt
    """

    _SPLIT_MAP = {"train": 0, "valid": 1, "test": 2}

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform=None,
        target_transform=None,
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        split_id = self._SPLIT_MAP[split]

        # ── Read partition file (CSV) ─────────────────────────────
        partition_path = os.path.join(root, "list_eval_partition.csv")
        filenames, partitions = [], []
        with open(partition_path, "r", encoding="utf-8") as f:
            header = f.readline()               # skip CSV header
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    filenames.append(parts[0].strip())
                    partitions.append(int(parts[1].strip()))

        # ── Read attribute file (CSV) ─────────────────────────────
        attr_path = os.path.join(root, "list_attr_celeba.csv")
        attr_map: dict[str, list[int]] = {}
        with open(attr_path, "r", encoding="utf-8") as f:
            header = f.readline()               # skip CSV header
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 2:
                    continue
                fname = parts[0].strip()
                # CelebA attrs are -1/1; convert to 0/1
                attrs = [max(0, int(v.strip())) for v in parts[1:]]
                attr_map[fname] = attrs

        # ── Locate image directory ────────────────────────────────
        img_dir = os.path.join(root, "img_align_celeba")
        # Handle common double-nested extraction:
        #   img_align_celeba/img_align_celeba/*.jpg
        nested = os.path.join(img_dir, "img_align_celeba")
        if os.path.isdir(nested):
            img_dir = nested

        # ── Filter by split ───────────────────────────────────────
        self.samples: list[tuple[str, list[int]]] = []
        missing = 0
        for fname, pid in zip(filenames, partitions):
            if pid == split_id and fname in attr_map:
                img_path = os.path.join(img_dir, fname)
                if os.path.isfile(img_path):
                    self.samples.append((img_path, attr_map[fname]))
                else:
                    missing += 1

        if missing > 0:
            logger.warning(
                "CelebADataset [%s]: %d images not found in %s — skipped",
                split, missing, img_dir,
            )
        logger.info(
            "CelebADataset [%s]: %d samples from %s",
            split, len(self.samples), img_dir,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        img_path, attrs = self.samples[index]
        img = Image.open(img_path).convert("RGB")
        target = torch.tensor(attrs, dtype=torch.long)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


def get_dataloaders(
    dataset_name: str = "CIFAR10",
    batch_size: int = 256,
    num_workers: int = 8,
    use_aug: bool = False,
    data_root: str = "/home/syc/experiments/data",
    celebA_path: str | None = "/home/syc/experiments/data/celeba",
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
        # Determine the root directory for CelebA dataset. Use provided celebA_path if given, otherwise fall back to data_root.
        celeb_root = celebA_path if celebA_path is not None else data_root
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

        train_dataset = CelebADataset(
            root=celeb_root, split='train',
            transform=train_transform,
            target_transform=target_transform)
        test_dataset = CelebADataset(
            root=celeb_root, split='test',
            transform=test_transform,
            target_transform=target_transform)
        eval_train_dataset = CelebADataset(
            root=celeb_root, split='train',
            transform=test_transform,
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
