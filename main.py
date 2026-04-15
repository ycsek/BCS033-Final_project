# main.py
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.backends import cudnn
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from utils import get_dataloaders
from dp import get_target_model, train_target_epoch, evaluate_model
from mia import evaluate_mia_vulnerability
from logger import ExperimentLogger
from analysis import run_analysis


def parse_args():
    parser = argparse.ArgumentParser(
        description="Static DP-SGD and MIA Evaluation")
    parser.add_argument('--device', type=str, default='cuda:0',
                        help="Device to use for training")
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                        choices=['SVHN', 'CelebA', 'CIFAR10', 'MNIST'])
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--target_epochs', type=int, default=200)
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--use_aug', action='store_true',
                        help="Enable data augmentation")
    parser.add_argument('--weight_decay', type=float,
                        default=0.0, help="Set to 0 to induce overfitting")

    parser.add_argument('--use_dp', action='store_true')
    parser.add_argument('--max_grad_norm', type=float, default=1.2)
    parser.add_argument('--noise_multiplier', type=float, default=1.0)
    parser.add_argument('--max_physical_batch_size', type=int, default=128)

    return parser.parse_args()


def main():
    args = parse_args()
    logger = ExperimentLogger(args)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True

    logger.info(
        f"Using device: {device} | DP Enabled: {args.use_dp} | Augmentation: {args.use_aug}")

    # 解包新加入的 eval_train_loader
    train_loader, test_loader, eval_train_loader, num_classes = get_dataloaders(
        args.dataset, args.batch_size, args.num_workers, args.use_aug)

    target_model = get_target_model(num_classes, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(target_model.parameters(
    ), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

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

    # ================= Phase 1: Training Target Model =================
    logger.info("\n" + "="*20 + " Phase 1: Training Target Model " + "="*20)
    for epoch in range(1, args.target_epochs + 1):
        if args.use_dp:
            with BatchMemoryManager(
                data_loader=train_loader,
                max_physical_batch_size=args.max_physical_batch_size,
                optimizer=optimizer
            ) as memory_safe_data_loader:
                target_loss = train_target_epoch(
                    target_model, memory_safe_data_loader, optimizer, criterion, device, epoch, args.target_epochs)
        else:
            target_loss = train_target_epoch(
                target_model, train_loader, optimizer, criterion, device, epoch, args.target_epochs)

        # 评估训练准确率时，使用 eval_train_loader 保证评价客观
        train_acc = evaluate_model(target_model, eval_train_loader, device)
        target_acc = evaluate_model(target_model, test_loader, device)
        epsilon = privacy_engine.get_epsilon(1e-5) if args.use_dp else None

        log_msg = (f"Target Epoch {epoch}/{args.target_epochs} | "
                   f"Loss: {target_loss:.4f} | "
                   f"Train ACC: {train_acc:.2f}% | "
                   f"Test ACC: {target_acc:.2f}%")
        if epsilon:
            log_msg += f" | ε: {epsilon:.2f}"
        logger.info(log_msg)
        logger.log_epoch_metrics(
            epoch, target_loss, train_acc, target_acc, epsilon)

    # ================= Phase 2: MIA Vulnerability Evaluation =================
    logger.info("\n" + "="*20 +
                " Phase 2: Evaluating MIA Vulnerability " + "="*20)
    target_model.eval()

    # 强制传入 eval_train_loader 防止特征泄露
    mia_results = evaluate_mia_vulnerability(
        target_model, eval_train_loader, test_loader, num_classes, device, args)

    logger.info(
        f"Final MIA ASR: {mia_results['asr']:.2f}% | AUC-ROC: {mia_results['auc_roc']:.4f}")
    logger.info(
        f"Precision: {mia_results['precision']:.4f} | Recall: {mia_results['recall']:.4f}")
    logger.info(f"TPR @ FPR=0.1%: {mia_results['tpr_0_1']:.2f}%")
    logger.info(f"TPR @ FPR=1.0%: {mia_results['tpr_1']:.2f}%")
    logger.info(f"TPR @ FPR=5.0%: {mia_results['tpr_5']:.2f}%")

    logger.log_final_metrics(mia_results)
    logger.save_results()

    # ================= Phase 3: Enhanced Metrics & Interpretability =================
    logger.info("\n" + "="*20 +
                " Phase 3: Enhanced Metrics & Interpretability Analysis " + "="*20)
    run_analysis(logger.log_dir, target_model, test_loader, device, args)


if __name__ == "__main__":
    main()
