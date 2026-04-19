"""Static DP-SGD and MIA Evaluation — Main Orchestrator.

Pipeline phases:
  1. Train a ResNet-50 target model (optionally with DP-SGD defence).
  2. Evaluate membership inference attack (MIA) vulnerability via XGBoost.
  3. Run explainability analysis (PCA, t-SNE, Grad-CAM) and save figures.
  4. Persist model checkpoints and structured results.
"""

from __future__ import annotations

import logging
import os

import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from torch.backends import cudnn

from analysis import run_analysis
from config import load_config_with_cli
from dp import evaluate_model, get_target_model, train_target_epoch
from logger import ExperimentLogger
from mia import evaluate_mia_vulnerability
from visualization import (
    plot_accuracy_curve,
    plot_asr_curve,
    plot_loss_curve,
    plot_pca_2d,
    plot_reliability_diagram,
    plot_roc_pr_curves,
    plot_tsne_2d,
)

log = logging.getLogger(__name__)


def main() -> None:
    """Run the full experiment pipeline."""

    # ── 0. Configuration ────────────────────────────────────────────
    cfg = load_config_with_cli()
    exp_logger = ExperimentLogger(cfg)
    device = torch.device(
        cfg.training.device
        if torch.cuda.is_available()
        else "cpu"
    )
    cudnn.benchmark = True

    exp_logger.info(
        f"Device: {device} | DP: {cfg.dp.use_dp} | "
        f"Augmentation: {cfg.training.use_aug}"
    )

    # ── 1. Data loading ─────────────────────────────────────────────
    from utils import get_dataloaders

    train_subset_size = getattr(cfg.training, "train_subset_size", None)

    train_loader, test_loader, eval_train_loader, num_classes = get_dataloaders(
        dataset_name=cfg.training.dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        use_aug=cfg.training.use_aug,
        data_root=cfg.paths.data_root,
        celebA_path=getattr(cfg.paths, "celebA_path", None),
        train_subset_size=train_subset_size,
    )

    # ── 2. Model, loss, optimizer ───────────────────────────────────
    target_model = get_target_model(num_classes, device)
    criterion = nn.CrossEntropyLoss()

    # [修改核心 1] 换用 Adam 优化器。
    # Adam 能在小数据集上极速收敛并发生严重过拟合，它是我们拉高 MIA ASR 基准的最佳武器。
    optimizer = optim.Adam(
        target_model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    # ── 3. Privacy engine (fixed epsilon) ───────────────────────────
    privacy_engine = None
    if cfg.dp.use_dp:
        privacy_engine = PrivacyEngine()
        target_model, optimizer, train_loader = (
            privacy_engine.make_private_with_epsilon(
                module=target_model,
                optimizer=optimizer,
                data_loader=train_loader,
                target_epsilon=cfg.dp.target_epsilon,
                target_delta=cfg.dp.delta,
                epochs=cfg.training.target_epochs,
                max_grad_norm=cfg.dp.max_grad_norm,
            )
        )
        exp_logger.info(
            f"DP-SGD enabled — target ε={cfg.dp.target_epsilon}, "
            f"δ={cfg.dp.delta}"
        )

    # [修改核心 2] 增加学习率调度器。
    # 强制学习率在训练后期平滑降至 0，迫使模型将 Train Loss 压榨到极致（完全死记硬背）。
    # 注意：为了防止破坏 Opacus 内部结构，Scheduler 必须在 make_private 之后声明！
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.target_epochs
    )

    # ═══════════════════ Phase 1: Training ══════════════════════════
    exp_logger.info(
        "\n" + "=" * 20 + " Phase 1: Training Target Model " + "=" * 20
    )
    exp_logger.log_training_start()

    for epoch in range(1, cfg.training.target_epochs + 1):
        if cfg.dp.use_dp:
            with BatchMemoryManager(
                data_loader=train_loader,
                max_physical_batch_size=cfg.dp.max_physical_batch_size,
                optimizer=optimizer,
            ) as memory_safe_loader:
                target_loss = train_target_epoch(
                    target_model, memory_safe_loader, optimizer, criterion,
                    device, epoch, cfg.training.target_epochs,
                )
        else:
            target_loss = train_target_epoch(
                target_model, train_loader, optimizer, criterion,
                device, epoch, cfg.training.target_epochs,
            )

        # 步进调度器，更新下一轮的 Learning Rate
        scheduler.step()

        # Evaluate
        train_acc = evaluate_model(target_model, eval_train_loader, device)
        target_acc = evaluate_model(target_model, test_loader, device)
        epsilon = (
            privacy_engine.get_epsilon(cfg.dp.delta)
            if cfg.dp.use_dp
            else None
        )

        current_lr = scheduler.get_last_lr()[0]

        log_msg = (
            f"Target Epoch {epoch}/{cfg.training.target_epochs} | "
            f"LR: {current_lr:.6f} | "
            f"Loss: {target_loss:.4f} | "
            f"Train ACC: {train_acc:.2f}% | "
            f"Test ACC: {target_acc:.2f}%"
        )
        if epsilon is not None:
            log_msg += f" | ε: {epsilon:.2f}"
        exp_logger.info(log_msg)
        exp_logger.log_epoch_metrics(
            epoch, target_loss, train_acc, target_acc, epsilon)

    exp_logger.log_training_end()

    # Record final ε
    if cfg.dp.use_dp and privacy_engine is not None:
        final_eps = privacy_engine.get_epsilon(cfg.dp.delta)
        exp_logger.log_final_epsilon(final_eps)

    # ═══════════════════ Phase 2: MIA Evaluation ════════════════════
    exp_logger.info(
        "\n" + "=" * 20
        + " Phase 2: Evaluating MIA Vulnerability " + "=" * 20
    )
    target_model.eval()

    mia_results = evaluate_mia_vulnerability(
        target_model, eval_train_loader, test_loader, num_classes, device, cfg,
    )

    exp_logger.info(
        f"Final MIA ASR: {mia_results['asr']:.2f}% | "
        f"AUC-ROC: {mia_results['auc_roc']:.4f}"
    )
    exp_logger.info(
        f"Precision: {mia_results['precision']:.4f} | "
        f"Recall: {mia_results['recall']:.4f} | "
        f"F1: {mia_results['f1']:.4f}"
    )
    exp_logger.info(f"TPR @ FPR=0.1%: {mia_results['tpr_0_1']:.2f}%")
    exp_logger.info(f"TPR @ FPR=1.0%: {mia_results['tpr_1']:.2f}%")
    exp_logger.info(f"TPR @ FPR=5.0%: {mia_results['tpr_5']:.2f}%")

    # Store metrics (without raw numpy arrays / model objects)
    serializable_metrics = {
        k: v for k, v in mia_results.items()
        if k not in ("features", "labels", "attack_model",
                     "fpr", "tpr", "pr_precision", "pr_recall")
    }
    exp_logger.log_final_metrics(serializable_metrics)
    exp_logger.save_results()

    # ═══════════════════ Phase 3: Explainability ════════════════════
    exp_logger.info(
        "\n" + "=" * 20
        + " Phase 3: Explainability & Visualization " + "=" * 20
    )

    figures_dir = os.path.join(exp_logger.log_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # 3a — Training curves
    trajectory = exp_logger.results["trajectory"]
    plot_accuracy_curve(trajectory, figures_dir)
    plot_loss_curve(trajectory, figures_dir)
    plot_asr_curve(serializable_metrics, figures_dir)

    # 3b — PCA / t-SNE on MIA features (member vs. non-member)
    mia_features = mia_results["features"]
    mia_labels = mia_results["labels"]

    exp_logger.info("Generating PCA 2-D projection...")
    plot_pca_2d(mia_features, mia_labels, figures_dir)
    exp_logger.info("Generating t-SNE 2-D projection...")
    plot_tsne_2d(mia_features, mia_labels, figures_dir)

    # 3c — ROC & Precision-Recall curves
    exp_logger.info("Generating ROC & Precision-Recall curves...")
    plot_roc_pr_curves(
        mia_results["fpr"], mia_results["tpr"],
        mia_results["pr_precision"], mia_results["pr_recall"],
        figures_dir,
    )

    # 3d — Reliability diagram (calibration curve)
    exp_logger.info("Generating reliability diagram...")
    from sklearn.calibration import calibration_curve
    # Use the attack model's predicted probabilities on all data
    attack_probs = mia_results["attack_model"].predict_proba(
        mia_results["features"]
    )[:, 1]
    prob_true, prob_pred = calibration_curve(
        mia_results["labels"], attack_probs, n_bins=15
    )
    plot_reliability_diagram(prob_true, prob_pred, figures_dir)

    # 3e — Grad-CAM + ECE + top-k (via analysis module)
    run_analysis(exp_logger.log_dir, target_model, test_loader, device, cfg)

    # ═══════════════════ Phase 4: Model Persistence ═════════════════
    exp_logger.info(
        "\n" + "=" * 20 + " Phase 4: Saving Model Checkpoints " + "=" * 20
    )

    ckpt_dir = getattr(cfg.paths, "checkpoint_dir", "./checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    from datetime import datetime

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save target model weights
    base_model = (
        target_model._module
        if hasattr(target_model, "_module")
        else target_model
    )
    target_path = os.path.join(ckpt_dir, f"target_model_{ts}.pt")
    torch.save(base_model.state_dict(), target_path)
    exp_logger.info(f"Target model saved: {target_path}")

    # Save XGBoost attack model
    attack_path = os.path.join(ckpt_dir, f"attack_model_{ts}.joblib")
    joblib.dump(mia_results["attack_model"], attack_path)
    exp_logger.info(f"Attack model saved: {attack_path}")

    exp_logger.info("\n>>> Experiment complete <<<")


if __name__ == "__main__":
    main()
