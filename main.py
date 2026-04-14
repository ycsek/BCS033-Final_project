# main.py
import argparse
import torch
from utils import get_dataloaders
from dp import get_target_model, train_target_model
from mia import train_mia_model
from logger import ExperimentLogger


def parse_args():
    parser = argparse.ArgumentParser(
        description="DP-SGD and Membership Inference Attack Experiment")

    # 基础参数 (根据需求进行了扩展)
    parser.add_argument('--dataset', type=str, default='SVHN',
                        choices=['SVHN', 'CelebA', 'CIFAR10', 'MNIST'],
                        help='Dataset to use for training and attack')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for data loading')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate for target model')
    parser.add_argument('--mia_lr', type=float, default=0.001,
                        help='Learning rate for MIA model')
    parser.add_argument('--target_epochs', type=int, default=10,
                        help='Number of epochs to train target model')
    parser.add_argument('--mia_epochs', type=int, default=15,
                        help='Number of epochs to train MIA model')

    # DP 参数
    parser.add_argument('--use_dp', action='store_true',
                        help='Enable DP-SGD defense')
    parser.add_argument('--max_grad_norm', type=float,
                        default=1.0, help='Clipping bound C for DP-SGD')
    parser.add_argument('--noise_multiplier', type=float,
                        default=1.2, help='Noise multiplier Sigma for DP-SGD')

    return parser.parse_args()


def main():
    args = parse_args()
    logger = ExperimentLogger(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Experiment Arguments: {args}")

    # 1. 加载数据
    train_loader, test_loader, num_classes = get_dataloaders(
        args.dataset, args.batch_size)

    # 2. 训练目标模型
    target_model = get_target_model(num_classes, device)
    target_model = train_target_model(
        model=target_model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        args=args,
        logger=logger
    )

    # 3. 执行成员推理攻击
    train_mia_model(
        target_model=target_model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_classes=num_classes,
        device=device,
        args=args,
        logger=logger
    )

    # 4. 保存所有 JSON 记录
    logger.save_results()


if __name__ == "__main__":
    main()
