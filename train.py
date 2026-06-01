'''
train.py — 訓練Training 和 驗證validation loop

Usage：
    python train.py --dataset white_black [--resume checkpoints/white_black/checkpoint_epoch10.pth]  # 有檢查點

The script：
  1. train 訓練圖資料源：data/train/<variant> 和 validation 驗證圖資料源：data/val/<variants>
  2. Trains the CNN for cfg.EPOCHS epochs with validation after each epoch  # 訓練共 cfg.EPOCHS 個 epochs 且 每個 epochs 後會驗證一次
  3. Saves a checkpoint every epoch and keeps the best model (lowest val loss on its own variant)
  4. Writes logs to logs/ and saves metrics to outputs/<variant>/epoch_metrics.xlsx
'''

import argparse
import logging
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import config as cfg
from dataset import MyDataset, train_transform
from model import CNN
from utils import load_checkpoint, save_checkpoint, save_epoch_metrics_to_excel
from val import validate_cross


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# One-epoch helpers
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Run one training epoch and return (avg_loss, accuracy %)."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples
    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train digit-recognition CNN")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=cfg.DATASET_VARIANTS,
        default=cfg.DATASET_VARIANTS[0],
        help="Dataset variant to train on.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="CHECKPOINT",
        help="Path to a checkpoint to resume training from.",
    )
    args = parser.parse_args()
    dataset_variant = args.dataset

    device = cfg.DEVICE
    log_path = os.path.join(cfg.LOG_DIR, f"train_{dataset_variant}.log")
    logger = setup_logging(log_path)
    logger.info("Using device: %s", device)

    # Datasets & loaders
    train_dir = os.path.join(cfg.TRAIN_DIR, dataset_variant)
    train_dataset = MyDataset(train_dir, transform=train_transform)

    use_persistent = cfg.NUM_WORKERS > 0
    common_loader_kwargs = {
        "batch_size": cfg.BATCH_SIZE,
        "num_workers": cfg.NUM_WORKERS,
        "pin_memory": True,
        "persistent_workers": use_persistent,
        "prefetch_factor": 4 if use_persistent else None,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **common_loader_kwargs)

    # Model, loss, optimiser
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

    start_epoch = 0
    best_val_loss = float("inf")
    checkpoint_dir = os.path.join(cfg.CHECKPOINT_DIR, dataset_variant)
    output_dir = os.path.join(cfg.OUTPUT_DIR, dataset_variant)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Resume from checkpoint if requested
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("Resuming from checkpoint: %s", args.resume)
            ckpt = load_checkpoint(args.resume, model, optimizer)
            start_epoch = ckpt.get("epoch", 0)
            best_val_loss = ckpt.get("val_loss", float("inf"))
            logger.info("Resumed at epoch %d, best val_loss=%.4f", start_epoch, best_val_loss)
        else:
            logger.warning("Checkpoint not found: %s — starting from scratch.", args.resume)

    # Training loop
    epoch_metrics_rows = []

    for epoch in range(start_epoch + 1, cfg.EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate_cross(
            model,
            criterion,
            device,
            cfg.VAL_DIR,
            cfg.DATASET_VARIANTS,
            common_loader_kwargs,
        )
        epoch_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
        }
        for variant in cfg.DATASET_VARIANTS:
            epoch_row[f"val_{variant}_loss"] = val_metrics[variant]["loss"]
            epoch_row[f"val_{variant}_acc"] = val_metrics[variant]["acc"]
        epoch_metrics_rows.append(epoch_row)

        val_log_parts = [
            f"{variant}: Loss {val_metrics[variant]['loss']:.4f} Acc {val_metrics[variant]['acc']:.2f}%"
            for variant in cfg.DATASET_VARIANTS
        ]
        logger.info(
            "Epoch [%d/%d] | Train Loss: %.4f | Train Acc: %.2f%% | Val (%s)",
            epoch,
            cfg.EPOCHS,
            train_loss,
            train_acc,
            " | ".join(val_log_parts),
        )

        # Save per-epoch checkpoint
        checkpoint_path = save_checkpoint(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_metrics[dataset_variant]["loss"],
                "val_acc": val_metrics[dataset_variant]["acc"],
            },
            checkpoint_dir,
            f"checkpoint_epoch{epoch}.pth",
        )
        logger.info("Checkpoint saved: %s", checkpoint_path)

        # Keep best model
        if val_metrics[dataset_variant]["loss"] < best_val_loss:
            best_val_loss = val_metrics[dataset_variant]["loss"]
            best_path = save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_metrics[dataset_variant]["loss"],
                    "val_acc": val_metrics[dataset_variant]["acc"],
                },
                checkpoint_dir,
                "best_model.pth",
            )
            logger.info(
                "New best model saved: %s (val_loss=%.4f)",
                best_path,
                best_val_loss,
            )

    metrics_excel_path = os.path.join(output_dir, "epoch_metrics.xlsx")
    headers = ["epoch", "train_loss", "train_acc"]
    for variant in cfg.DATASET_VARIANTS:
        headers.append(f"val_{variant}_loss")
        headers.append(f"val_{variant}_acc")
    save_epoch_metrics_to_excel(epoch_metrics_rows, metrics_excel_path, headers=headers)
    logger.info("Epoch metrics saved to %s", metrics_excel_path)


if __name__ == "__main__":
    main()
