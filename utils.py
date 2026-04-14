# utils.py
import os
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_dataloaders(dataset_name="CIFAR10", batch_size=256, num_workers=8, use_aug=False):
    DATA_ROOT = '/home/syc/experiments/data'
    os.makedirs(DATA_ROOT, exist_ok=True)

    if dataset_name == "CIFAR10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        if use_aug:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:
            # 关闭增强，诱导过拟合
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        train_dataset = torchvision.datasets.CIFAR10(
            root=DATA_ROOT, train=True, download=True, transform=train_transform)
        test_dataset = torchvision.datasets.CIFAR10(
            root=DATA_ROOT, train=False, download=True, transform=test_transform)
        num_classes = 10

    elif dataset_name == "SVHN":
        mean, std = (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)
        train_transform = transforms.Compose([
            transforms.RandomCrop(
                32, padding=4) if use_aug else transforms.Identity(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        train_dataset = torchvision.datasets.SVHN(
            root=DATA_ROOT, split='train', download=True, transform=train_transform)
        test_dataset = torchvision.datasets.SVHN(
            root=DATA_ROOT, split='test', download=True, transform=test_transform)
        num_classes = 10

    elif dataset_name == "MNIST":
        mean, std = (0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)
        train_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomCrop(
                32, padding=2) if use_aug else transforms.Identity(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        train_dataset = torchvision.datasets.MNIST(
            root=DATA_ROOT, train=True, download=True, transform=train_transform)
        test_dataset = torchvision.datasets.MNIST(
            root=DATA_ROOT, train=False, download=True, transform=test_transform)
        num_classes = 10

    elif dataset_name == "CelebA":
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        train_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip() if use_aug else transforms.Identity(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        def target_transform(target): return target[20]
        train_dataset = torchvision.datasets.CelebA(
            root=DATA_ROOT, split='train', download=True, transform=train_transform, target_type='attr', target_transform=target_transform)
        test_dataset = torchvision.datasets.CelebA(
            root=DATA_ROOT, split='test', download=True, transform=test_transform, target_type='attr', target_transform=target_transform)
        num_classes = 2
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    loader_kwargs = {'num_workers': num_workers,
                     'pin_memory': True, 'prefetch_factor': 2}
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)

    return train_loader, test_loader, num_classes
