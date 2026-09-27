"""
Training & evaluation script for Agricultural Crops Image Classification (MSc level).
Includes:
- Transfer learning
- Class-balanced sampling + heavy augmentation (data enrichment)
- Early stopping, learning-rate scheduling
- Full metrics: accuracy, precision, recall, F1, confusion matrix
- Model checkpointing
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# local imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import create_dataloaders, CLASS_NAMES, NUM_CLASSES, analyze_dataset
from models import create_model, count_parameters


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    for images, labels in tqdm(loader, desc="Val", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix – Agricultural Crop Classification")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return cm


def plot_training_curves(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"], label="Val")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def run_training(
    zip_path: str = "/home/workdir/attachments/Crop.zip",
    output_dir: str = "/home/workdir/artifacts/Agricultural_Crop_Classification/results",
    model_name: str = "resnet18",
    epochs: int = 15,
    batch_size: int = 8,
    lr: float = 1e-4,
    img_size: int = 160,
    max_per_class: int = 80,
    freeze_backbone: bool = False,
    device: str = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path("/home/workdir/artifacts/Agricultural_Crop_Classification/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Dataset analysis
    stats = analyze_dataset(zip_path)
    print("Dataset stats:", stats)
    with open(output_dir / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    train_loader, val_loader, info = create_dataloaders(
        zip_path=zip_path,
        batch_size=batch_size,
        img_size=img_size,
        max_per_class=max_per_class,
        val_split=0.2,
        num_workers=0,
    )
    print(f"Train: {info['train_size']}, Val: {info['val_size']}")

    model = create_model(model_name, NUM_CLASSES, pretrained=True, freeze_backbone=freeze_backbone)
    model = model.to(device)
    print(f"Model: {model_name}, Trainable params: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc = 0.0
    best_path = models_dir / f"best_{model_name}.pth"
    patience_counter = 0
    max_patience = 5

    start = time.time()
    for epoch in range(1, epochs + 1):
        print(f"\n=== Epoch {epoch}/{epochs} ===")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, preds, labels = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_acc)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "class_names": CLASS_NAMES,
                "model_name": model_name,
                "img_size": img_size,
            }, best_path)
            print(f"  -> New best model saved (acc={best_acc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print("Early stopping triggered.")
                break

    elapsed = time.time() - start
    print(f"\nTraining finished in {elapsed/60:.1f} min. Best Val Acc: {best_acc:.4f}")

    # Final evaluation with best model
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    _, final_acc, preds, labels = evaluate(model, val_loader, criterion, device)

    # Metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    report = classification_report(labels, preds, target_names=CLASS_NAMES, zero_division=0)
    print("\nClassification Report:\n", report)

    cm = plot_confusion_matrix(labels, preds, CLASS_NAMES, output_dir / "confusion_matrix.png")
    plot_training_curves(history, output_dir / "training_curves.png")

    results = {
        "model_name": model_name,
        "best_val_acc": float(best_acc),
        "final_val_acc": float(final_acc),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "epochs_trained": epoch,
        "training_time_min": elapsed / 60,
        "class_names": CLASS_NAMES,
        "dataset_stats": stats,
        "history": history,
        "classification_report": report,
        "timestamp": datetime.now().isoformat(),
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(output_dir / "classification_report.txt", "w") as f:
        f.write(report)

    print(f"\nResults saved to {output_dir}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--img-size", type=int, default=160)
    parser.add_argument("--max-per-class", type=int, default=80)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()

    run_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        model_name=args.model,
        img_size=args.img_size,
        max_per_class=args.max_per_class,
        freeze_backbone=args.freeze,
    )
