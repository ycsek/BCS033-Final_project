# main.py
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
from utils import get_dataloaders
from dp import get_target_model, train_target_epoch, evaluate_model
from mia import evaluate_mia_vulnerability
from logger import ExperimentLogger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dynamic DP-SGD and MIA Evaluation")

    parser.add_argument('--dataset', type=str, default='CIFAR10',
                        choices=['SVHN', 'CelebA', 'CIFAR10', 'MNIST'])
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--mia_lr', type=float,
                        default=0.005)  # 适当调大 MIA 学习率以加速单次探测
    parser.add_argument('--target_epochs', type=int, default=100)
    parser.add_argument('--mia_epochs', type=int,
                        default=10)  # 每次探测只需较少的 epoch 即可收敛

    parser.add_argument('--use_dp', action='store_true')
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--noise_multiplier', type=float, default=1.2)

    return parser.parse_args()


def main():
    args = parse_args()
    logger = ExperimentLogger(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device} | DP Enabled: {args.use_dp}")

    train_loader, test_loader, num_classes = get_dataloaders(
        args.dataset, args.batch_size)

    # 1. 初始化目标模型与优化器
    target_model = get_target_model(num_classes, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(target_model.parameters(),
                          lr=args.lr, momentum=0.9, weight_decay=5e-4)

    # 2. 初始化差分隐私引擎
    privacy_engine = None
    if args.use_dp:
        privacy_engine = PrivacyEngine()
        target_model, optimizer, train_loader = privacy_engine.make_private(
            module=target_model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=args.noise_multiplier,
            max_grad_norm=args.max_grad_norm,
        )

    # 3. 动态轨迹主循环 (Dynamic Trajectory Loop)
    for epoch in range(1, args.target_epochs + 1):
        logger.info(
            f"\n{'='*20} Target Epoch {epoch}/{args.target_epochs} {'='*20}")

        # 步骤 A: 训练目标模型一个 Epoch
        target_loss = train_target_epoch(
            target_model, train_loader, optimizer, criterion, device)

        # 步骤 B: 评估目标模型 Utility (ACC)
        target_acc = evaluate_model(target_model, test_loader, device)

        # 步骤 C: 评估当前状态下的隐私脆弱性 (ASR)
        asr = evaluate_mia_vulnerability(
            target_model, train_loader, test_loader, num_classes, device, args)

        # 步骤 D: 记录与输出
        epsilon = privacy_engine.get_epsilon(1e-5) if args.use_dp else None

        log_msg = f"Epoch {epoch} | Target Loss: {target_loss:.4f} | Target ACC: {target_acc:.2f}% | MIA ASR: {asr:.2f}%"
        if epsilon:
            log_msg += f" | ε: {epsilon:.2f}"
        logger.info(log_msg)

        logger.log_epoch_metrics(epoch, target_loss, target_acc, asr, epsilon)

    logger.save_results()


if __name__ == "__main__":
    main()
