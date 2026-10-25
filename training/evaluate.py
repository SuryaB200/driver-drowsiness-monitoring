"""Model evaluation, metrics generation, latency benchmarking, and chart plotting on held-out test set."""
import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_fscore_support,
    accuracy_score
)
import matplotlib
matplotlib.use('Agg') # Headless plotting
import matplotlib.pyplot as plt
import seaborn as sns

from backend.config import CHECKPOINTS_DIR, RESULTS_DIR, PLOTS_DIR, DATASETS_DIR
from backend.models.mobilenet import load_checkpoint

TEST_DIR = DATASETS_DIR / "processed" / "test"
TRAIN_HISTORY_PATH = RESULTS_DIR / "training_history.json"


def plot_training_curves(history: Dict[str, Any]):
    """Generates loss and accuracy curves from genuine training history."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    epochs = history.get("epochs", [])
    if not epochs:
        return

    # 1. Loss curve
    plt.figure(figsize=(7, 4.5), dpi=150)
    plt.plot(epochs, history["train_loss"], 'o-', color='#2457A7', label='Training Loss', linewidth=2)
    plt.plot(epochs, history["val_loss"], 's--', color='#C43D3D', label='Validation Loss', linewidth=2)
    plt.title('MobileNetV3 Training & Validation Loss', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('CrossEntropy Loss', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "loss_curves.png")
    plt.close()

    # 2. Accuracy curve
    plt.figure(figsize=(7, 4.5), dpi=150)
    train_acc_pct = [acc * 100 for acc in history["train_acc"]]
    val_acc_pct = [acc * 100 for acc in history["val_acc"]]
    plt.plot(epochs, train_acc_pct, 'o-', color='#287D52', label='Training Accuracy', linewidth=2)
    plt.plot(epochs, val_acc_pct, 's--', color='#B7791F', label='Validation Accuracy', linewidth=2)
    plt.title('MobileNetV3 Training & Validation Accuracy', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Accuracy (%)', fontsize=11)
    plt.ylim(0, 105)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "accuracy_curves.png")
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, class_names: list):
    """Plots and saves normalized confusion matrix."""
    plt.figure(figsize=(5.5, 4.5), dpi=150)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        annot_kws={"size": 13, "weight": "bold"}
    )
    plt.title('Test Set Confusion Matrix', fontsize=12, fontweight='bold', pad=10)
    plt.xlabel('Predicted Label', fontsize=11)
    plt.ylabel('Ground Truth Label', fontsize=11)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "confusion_matrix.png")
    plt.close()


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float):
    """Plots ROC Curve and AUC metric."""
    plt.figure(figsize=(6, 4.5), dpi=150)
    plt.plot(fpr, tpr, color='#2457A7', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='#697586', linestyle='--', lw=1.5, label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('Receiver Operating Characteristic (ROC)', fontsize=12, fontweight='bold', pad=10)
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "roc_curve.png")
    plt.close()


def run_model_evaluation(checkpoint_name: str = "best_mobilenet_v3.pth") -> Dict[str, Any]:
    """Runs rigorous evaluation on the untouched held-out test split."""
    ckpt_path = str(CHECKPOINTS_DIR / checkpoint_name)
    if not os.path.exists(ckpt_path):
        return {"error": f"Checkpoint '{ckpt_path}' not found. Please train model first."}

    if not TEST_DIR.exists():
        return {"error": f"Test dataset directory '{TEST_DIR}' not found."}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_checkpoint(ckpt_path, device=str(device))
    if model is None:
        return {"error": "Failed to load model checkpoint"}

    # Test transforms
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(str(TEST_DIR), transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    class_names = test_dataset.classes

    all_labels = []
    all_preds = []
    all_probs = []
    latencies = []

    model.eval()

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            start_t = time.perf_counter()
            outputs = model(images)
            lat = (time.perf_counter() - start_t) * 1000.0
            latencies.append(lat)

            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            pred = int(np.argmax(probs))

            all_labels.append(labels.item())
            all_preds.append(pred)
            all_probs.append(probs[1] if len(probs) > 1 else probs[0])

    # Convert to numpy arrays
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)

    # Compute genuine classification metrics
    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    clf_report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)

    # ROC & AUC
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = float(auc(fpr, tpr))

    # Latency benchmarks
    mean_lat = float(np.mean(latencies))
    median_lat = float(np.median(latencies))
    p95_lat = float(np.percentile(latencies, 95))

    # Generate and save plots
    plot_confusion_matrix(cm, class_names)
    plot_roc_curve(fpr, tpr, roc_auc)

    if TRAIN_HISTORY_PATH.exists():
        try:
            with open(TRAIN_HISTORY_PATH, "r", encoding="utf-8") as f:
                hist_data = json.load(f)
                plot_training_curves(hist_data.get("history", {}))
        except Exception as e:
            print(f"Notice: Failed to plot training curves: {e}")

    metrics_payload = {
        "evaluation_timestamp": time.time(),
        "model_architecture": "MobileNetV3-Small",
        "device_tested": str(device),
        "total_test_samples": len(test_dataset),
        "class_names": class_names,
        "test_accuracy": round(acc, 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "auc_roc": round(roc_auc, 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": clf_report,
        "latency_benchmarks_ms": {
            "mean": round(mean_lat, 2),
            "median": round(median_lat, 2),
            "p95": round(p95_lat, 2)
        },
        "plots_saved": {
            "confusion_matrix": "/plots/confusion_matrix.png",
            "roc_curve": "/plots/roc_curve.png",
            "loss_curves": "/plots/loss_curves.png",
            "accuracy_curves": "/plots/accuracy_curves.png"
        }
    }

    # Save to results/metrics.json
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    return metrics_payload


if __name__ == "__main__":
    res = run_model_evaluation()
    print(json.dumps(res, indent=2))
