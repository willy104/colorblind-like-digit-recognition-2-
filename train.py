"""train.py — Training and validation loop for the digit-recognition CNN.

Usage:
    python train.py [--resume checkpoints/checkpoint_epoch10.pth]

The script:
  1. Loads train and validation datasets from data/train and data/val.
  2. Trains the CNN for cfg.EPOCHS epochs with validation after each epoch.
  3. Saves a checkpoint every epoch and keeps the best model (lowest val loss).
  4. Writes logs to logs/train.log and saves loss/accuracy plots to outputs/.
"""

import argparse
import logging
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import config as cfg
from dataset import MyDataset, train_transform, eval_transform
from model import CNN
from utils import load_checkpoint, plot_curves, save_checkpoint, save_epoch_metrics_to_excel


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

os.makedirs(cfg.LOG_DIR, exist_ok=True)
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(cfg.LOG_DIR, "train.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


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
        images, labels = images.to(device), labels.to(device)

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


def validate(model, loader, criterion, device):
    """Run validation and return (avg_loss, accuracy %)."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

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
        "--resume",
        type=str,
        default=None,
        metavar="CHECKPOINT",
        help="Path to a checkpoint to resume training from.",
    )
    args = parser.parse_args()

    device = cfg.DEVICE
    logger.info("Using device: %s", device)

    # Datasets & loaders
    train_dataset = MyDataset(cfg.TRAIN_DIR, transform=train_transform)
    val_dataset = MyDataset(cfg.VAL_DIR, transform=eval_transform)
    test_dataset = MyDataset(cfg.TEST_DIR, transform=eval_transform)

    use_persistent = cfg.NUM_WORKERS > 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=use_persistent,
        prefetch_factor=2 if use_persistent else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=use_persistent,
        prefetch_factor=2 if use_persistent else None,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=use_persistent,
        prefetch_factor=2 if use_persistent else None,
    )

    # Model, loss, optimiser
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

    start_epoch = 0
    best_val_loss = float("inf")

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
    train_losses, val_losses = [], []
    test_losses = []
    train_accs, val_accs = [], []
    test_accs = []
    epoch_metrics_rows = []

    for epoch in range(start_epoch + 1, cfg.EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        test_loss, test_acc = validate(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        test_accs.append(test_acc)
        epoch_metrics_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
            }
        )

        logger.info(
            "Epoch [%d/%d] | Train Loss: %.4f | Train Acc: %.2f%% | "
            "Val Loss: %.4f | Val Acc: %.2f%% | "
            "Test Loss: %.4f | Test Acc: %.2f%%",
            epoch,
            cfg.EPOCHS,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            test_loss,
            test_acc,
        )

        # Save per-epoch checkpoint
        checkpoint_path = save_checkpoint(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
            },
            cfg.CHECKPOINT_DIR,
            f"checkpoint_epoch{epoch}.pth",
        )
        logger.info("Checkpoint saved: %s", checkpoint_path)

        # Keep best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                cfg.CHECKPOINT_DIR,
                "best_model.pth",
            )
            logger.info("New best model saved: %s (val_loss=%.4f)", best_path, best_val_loss)

    # Save loss/accuracy plots
    plot_curves(
        train_losses,
        val_losses,
        train_accs,
        val_accs,
        cfg.OUTPUT_DIR,
        test_losses=test_losses,
        test_accs=test_accs,
    )
    metrics_excel_path = os.path.join(cfg.OUTPUT_DIR, "epoch_metrics.xlsx")
    save_epoch_metrics_to_excel(epoch_metrics_rows, metrics_excel_path)
    logger.info("Training curves saved to %s/", cfg.OUTPUT_DIR)
    logger.info("Epoch metrics saved to %s", metrics_excel_path)


if __name__ == "__main__":
    main()
