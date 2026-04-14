# utils.py
import os
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_dataloaders(dataset_name="SVHN", batch_size=64):
    # ==========================================
    # 定义全局数据存放路径
    # ==========================================
    DATA_ROOT = '/home/syc/experiments/data'
    os.makedirs(DATA_ROOT, exist_ok=True)  # 确保服务器上的这个目录存在

    # 基础的图像预处理：调整为统一的 32x32 尺寸并归一化
    base_transform_list = [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

    if dataset_name == "SVHN":
        transform = transforms.Compose(base_transform_list)
        train_dataset = torchvision.datasets.SVHN(
            root=DATA_ROOT, split='train', download=True, transform=transform)
        test_dataset = torchvision.datasets.SVHN(
            root=DATA_ROOT, split='test', download=True, transform=transform)
        num_classes = 10

    elif dataset_name == "CIFAR10":
        transform = transforms.Compose(base_transform_list)
        train_dataset = torchvision.datasets.CIFAR10(
            root=DATA_ROOT, train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(
            root=DATA_ROOT, train=False, download=True, transform=transform)
        num_classes = 10

    elif dataset_name == "MNIST":
        # MNIST 是 28x28 单通道，ResNet18 需要 3 通道输入
        mnist_transform_list = [
            transforms.Resize((32, 32)),
            transforms.Grayscale(num_output_channels=3),  # 强制转换为3通道
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        transform = transforms.Compose(mnist_transform_list)
        train_dataset = torchvision.datasets.MNIST(
            root=DATA_ROOT, train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(
            root=DATA_ROOT, train=False, download=True, transform=transform)
        num_classes = 10

    elif dataset_name == "CelebA":
        transform = transforms.Compose(base_transform_list)
        # 依然以提取 'Male' 属性为例进行二分类
        def target_transform(target): return target[20]
        train_dataset = torchvision.datasets.CelebA(
            root=DATA_ROOT, split='train', download=True, transform=transform, target_type='attr', target_transform=target_transform)
        test_dataset = torchvision.datasets.CelebA(
            root=DATA_ROOT, split='test', download=True, transform=transform, target_type='attr', target_transform=target_transform)
        num_classes = 2

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, num_classes
