"""Training script for MobileNetV3 drowsiness detection classifier."""
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from backend.models.mobilenet import DrowsinessMobileNetV3, save_checkpoint
from backend.config import CHECKPOINTS_DIR, DATASETS_DIR
from training.prepare_dataset import prepare_benchmark_dataset, check_dataset_status

PROCESSED_DIR = DATASETS_DIR / "processed"


def train_mobilenet_v3(epochs: int = 6,
                       batch_size: int = 32,
                       lr: float = 0.0001,
                       use_aug: bool = True,
                       checkpoint_name: str = "best_mobilenet_v3.pth",
                       progress_callback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Executes training of MobileNetV3 on subject-independent dataset split.
    Saves best model checkpoint.
    """
    # 1. Verify dataset readiness
    status = check_dataset_status()
    if not status["processed_splits"]["ready"]:
        print("Dataset not ready. Generating benchmark splits...")
        prepare_benchmark_dataset()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 4))
    print(f"Starting training on device: {device} (threads: {torch.get_num_threads()})")

    # 2. Transforms with data augmentation for training only
    train_transforms = [
        transforms.Resize((224, 224)),
    ]
    if use_aug:
        train_transforms.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15)
        ])
    train_transforms.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(str(PROCESSED_DIR / "train"), transform=transforms.Compose(train_transforms))
    val_dataset = datasets.ImageFolder(str(PROCESSED_DIR / "val"), transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(f"Loaded classes: {class_names} | Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # 3. Model, Optimizer, Loss, Scheduler
    model = DrowsinessMobileNetV3(num_classes=num_classes, pretrained=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_val_loss = float('inf')
    best_val_acc = 0.0
    history = {
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "lr": []
    }

    if progress_callback is not None:
        progress_callback["is_training"] = True
        progress_callback["total_epochs"] = epochs
        progress_callback["status"] = f"Training on {device}..."

    ckpt_path = str(CHECKPOINTS_DIR / checkpoint_name)

    start_total_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        running_train_loss = 0.0
        train_correct = 0
        total_train = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            total_train += labels.size(0)

            if (batch_idx + 1) % 40 == 0 or (batch_idx + 1) == len(train_loader):
                batch_acc = (train_correct / total_train) * 100.0
                print(f"  [Epoch {epoch:02d}/{epochs:02d}] Step {batch_idx+1:03d}/{len(train_loader):03d} - Loss: {loss.item():.4f} - Running Acc: {batch_acc:.1f}%")

        epoch_train_loss = running_train_loss / total_train if total_train > 0 else 0.0
        epoch_train_acc = train_correct / total_train if total_train > 0 else 0.0

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        val_correct = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                total_val += labels.size(0)

        epoch_val_loss = running_val_loss / total_val if total_val > 0 else 0.0
        epoch_val_acc = val_correct / total_val if total_val > 0 else 0.0

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(epoch_val_loss)

        epoch_time = time.time() - epoch_start

        # Record history
        history["epochs"].append(epoch)
        history["train_loss"].append(round(epoch_train_loss, 4))
        history["val_loss"].append(round(epoch_val_loss, 4))
        history["train_acc"].append(round(epoch_train_acc, 4))
        history["val_acc"].append(round(epoch_val_acc, 4))
        history["lr"].append(current_lr)

        log_str = (f"Epoch [{epoch:02d}/{epochs:02d}] "
                   f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}% | "
                   f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.2f}% | "
                   f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")
        print(log_str)

        if progress_callback is not None:
            progress_callback["current_epoch"] = epoch
            progress_callback["train_loss"] = round(epoch_train_loss, 4)
            progress_callback["val_loss"] = round(epoch_val_loss, 4)
            progress_callback["train_acc"] = round(epoch_train_acc, 4)
            progress_callback["val_acc"] = round(epoch_val_acc, 4)
            progress_callback["logs"].append(log_str)

        # Save best checkpoint according to validation loss
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_val_acc = epoch_val_acc
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_loss=epoch_val_loss,
                val_acc=epoch_val_acc,
                filepath=ckpt_path,
                class_names=class_names
            )
            print(f" --> Saved improved checkpoint to {ckpt_path}")

    total_elapsed = time.time() - start_total_time

    if progress_callback is not None:
        progress_callback["is_training"] = False
        progress_callback["status"] = "Completed"

    result = {
        "status": "success",
        "checkpoint_path": ckpt_path,
        "best_val_loss": round(best_val_loss, 4),
        "best_val_acc": round(best_val_acc, 4),
        "total_elapsed_sec": round(total_elapsed, 1),
        "history": history
    }

    # Save history to results/training_history.json
    results_file = Path(DATASETS_DIR).parent / "results" / "training_history.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "w", encoding="utf-8") as f:
        import json
        json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    train_mobilenet_v3(epochs=6, batch_size=32)
