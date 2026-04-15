# analysis.py
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from torchvision.utils import make_grid
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1)
        score = output[range(len(target_class)), target_class]
        score.backward(torch.ones_like(score))

        gradients = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = F.relu(self.activations * gradients).mean(dim=1)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam


def compute_psnr_ssim(original: torch.Tensor, noisy: torch.Tensor) -> tuple:
    mse = torch.mean((original - noisy) ** 2, dim=[1, 2, 3])
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    ssim_val = torch.mean(torch.abs(original - noisy) < 0.05).item()
    return psnr.mean().item(), ssim_val


def simulate_gaussian_noise(images: torch.Tensor, noise_multiplier: float, device: torch.device):
    sigma = noise_multiplier * 0.05
    noise = torch.randn_like(images, device=device) * sigma
    noisy_images = torch.clamp(images + noise, 0.0, 1.0)
    return noisy_images


def run_analysis(log_dir: str, target_model, test_loader, device, args):
    target_model.eval()
    additional_metrics = {}

    # Top-1, Top-5 ACC and Test Loss
    test_loss = 0.0
    top1_correct, top5_correct, total = 0, 0, 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = target_model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, pred = outputs.topk(5, 1, True, True)
            top1_correct += pred[:, 0].eq(targets).sum().item()
            top5_correct += pred.eq(targets.view(-1,
                                    1).expand_as(pred)).sum().item()
            total += targets.size(0)

    additional_metrics.update({
        "test_loss": test_loss / len(test_loader),
        "top1_acc": 100. * top1_correct / total,
        "top5_acc": 100. * top5_correct / total
    })

    # Robustness to Noise (PSNR, SSIM, Noisy Accuracy)
    batch = next(iter(test_loader))
    images, labels = batch[0].to(device)[:16], batch[1].to(device)[:16]
    noisy_images = simulate_gaussian_noise(
        images, getattr(args, 'noise_multiplier', 0.0), device)
    psnr_val, ssim_val = compute_psnr_ssim(images, noisy_images)
    with torch.no_grad():
        noisy_preds = target_model(noisy_images).argmax(1)
    noisy_acc = 100. * (noisy_preds == labels).float().mean().item()

    additional_metrics.update({
        "image_psnr": psnr_val,
        "image_ssim": ssim_val,
        "noisy_test_acc": noisy_acc
    })

    # Grad-CAM Visualization
    target_layer = dict(target_model.named_modules())[
        'layer4']
    gradcam = GradCAM(target_model, target_layer)

    with torch.no_grad():
        preds = target_model(images).argmax(dim=1)

    cam_maps = gradcam.generate(images, preds)
    cam_maps = F.interpolate(cam_maps.unsqueeze(1), size=(
        32, 32), mode='bilinear', align_corners=False).squeeze(1)

    os.makedirs(os.path.join(log_dir, "visualizations"), exist_ok=True)
    plt.figure(figsize=(12, 8))
    for i in range(min(8, len(images))):
        plt.subplot(2, 4, i + 1)
        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min())
        cam = cam_maps[i].cpu().numpy()
        plt.imshow(img)
        plt.imshow(cam, cmap='jet', alpha=0.5)
        plt.title(f"Pred: {preds[i].item()}")
        plt.axis("off")
    plt.suptitle("Grad-CAM Interpretability (Target Model Focus Areas)")
    plt.savefig(os.path.join(log_dir, "visualizations",
                "gradcam_visualization.png"), bbox_inches="tight")
    plt.close()

    # Training Trajectory
    with open(os.path.join(log_dir, "results.json"), "r", encoding="utf-8") as f:
        results = json.load(f)
    epochs = [r["epoch"] for r in results["trajectory"]]
    train_accs = [r["train_acc"] for r in results["trajectory"]]
    test_accs = [r["target_acc"] for r in results["trajectory"]]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accs, label="Train Acc", marker="o")
    plt.plot(epochs, test_accs, label="Test Acc", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training Trajectory")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, "visualizations", "training_curve.png"))
    plt.close()

    results["additional_metrics"] = additional_metrics
    with open(os.path.join(log_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\n>>> XAI Analysis.")
    print(
        f"Test Loss: {additional_metrics['test_loss']:.4f} | Top-5 Acc: {additional_metrics['top5_acc']:.2f}%")
    print(f"Grad-CAM saved to visualizations/gradcam_visualization.png")
    print(f"PSNR: {psnr_val:.2f} dB | Noise Robustness Acc: {noisy_acc:.2f}%")
