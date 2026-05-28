"""train.py — multi-domain training with per-epoch probe/validation metrics."""

import argparse
import logging
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import config as cfg
from dataset import MyDataset, train_transform, eval_transform
from model import CNN
from utils import load_checkpoint, plot_curves, save_checkpoint, save_epoch_metrics_to_excel
from val import eval_loop


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_or_create_subset_filenames(root_dir, split_file_path, sample_size, seed):
    os.makedirs(os.path.dirname(split_file_path), exist_ok=True)
    all_files = sorted([f for f in os.listdir(root_dir) if f.lower().endswith(".png")])
    if len(all_files) < sample_size:
        raise ValueError(
            f"Not enough images in {root_dir}. "
            f"Need {sample_size}, found {len(all_files)}."
        )

    if os.path.isfile(split_file_path):
        with open(split_file_path, "r", encoding="utf-8") as file_obj:
            selected = [line.strip() for line in file_obj if line.strip()]
        missing = [name for name in selected if name not in all_files]
        if missing:
            raise ValueError(
                f"Split file {split_file_path} contains files that do not exist in {root_dir}."
            )
        return selected

    rng = random.Random(seed)
    selected = sorted(rng.sample(all_files, sample_size))
    with open(split_file_path, "w", encoding="utf-8") as file_obj:
        file_obj.write("\n".join(selected) + "\n")
    return selected


def make_loader(dataset, shuffle):
    use_persistent = cfg.NUM_WORKERS > 0
    loader_kwargs = {
        "batch_size": cfg.BATCH_SIZE,
        "num_workers": cfg.NUM_WORKERS,
        "pin_memory": True,
        "persistent_workers": use_persistent,
        "prefetch_factor": 2 if use_persistent else None,
    }
    return DataLoader(dataset, shuffle=shuffle, **loader_kwargs)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Run one training epoch and return (avg_loss, accuracy %)."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

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


def setup_logger(domain):
    os.makedirs(cfg.LOG_DIR, exist_ok=True)
    logger = logging.getLogger(f"train_{domain}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(os.path.join(cfg.LOG_DIR, f"train_{domain}.log"))
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def main():
    parser = argparse.ArgumentParser(description="Train digit-recognition CNN")
    parser.add_argument(
        "--domain",
        type=str,
        default="A",
        choices=cfg.DOMAINS,
        help="Training domain: A, B, or C.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="CHECKPOINT",
        help="Path to a checkpoint to resume training from.",
    )
    args = parser.parse_args()

    domain = args.domain.upper()
    logger = setup_logger(domain)
    set_seed(cfg.SEED)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)

    device = cfg.DEVICE
    logger.info("Using device: %s", device)
    logger.info("Training domain: %s", domain)

    train_dir = cfg.get_domain_split_dir(domain, "train")

    splits_dir = os.path.join(cfg.OUTPUT_DIR, "splits")

    train_probe_files = get_or_create_subset_filenames(
        train_dir,
        os.path.join(splits_dir, f"train_probe_{domain}.txt"),
        cfg.SUBSET_SIZE,
        cfg.SEED,
    )
    val_subset_files = {}
    for val_domain in cfg.DOMAINS:
        val_dir = cfg.get_domain_split_dir(val_domain, "val")
        val_subset_files[val_domain] = get_or_create_subset_filenames(
            val_dir,
            os.path.join(splits_dir, f"val_{val_domain}.txt"),
            cfg.SUBSET_SIZE,
            cfg.SEED,
        )

    train_dataset = MyDataset(train_dir, transform=train_transform)
    train_probe_dataset = MyDataset(train_dir, transform=eval_transform, file_list=train_probe_files)
    val_datasets = {
        val_domain: MyDataset(
            cfg.get_domain_split_dir(val_domain, "val"),
            transform=eval_transform,
            file_list=val_subset_files[val_domain],
        )
        for val_domain in cfg.DOMAINS
    }

    train_loader = make_loader(train_dataset, shuffle=True)
    train_probe_loader = make_loader(train_probe_dataset, shuffle=False)
    val_loaders = {
        val_domain: make_loader(dataset, shuffle=False)
        for val_domain, dataset in val_datasets.items()
    }

    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

    start_epoch = 0
    best_val_loss = float("inf")

    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("Resuming from checkpoint: %s", args.resume)
            ckpt = load_checkpoint(args.resume, model, optimizer)
            start_epoch = ckpt.get("epoch", 0)
            best_val_loss = ckpt.get("val_loss", float("inf"))
            logger.info("Resumed at epoch %d, best val_loss=%.4f", start_epoch, best_val_loss)
        else:
            logger.warning("Checkpoint not found: %s — starting from scratch.", args.resume)

    train_losses, selected_val_losses = [], []
    train_accs, selected_val_accs = [], []
    epoch_metrics_rows = []

    for epoch in range(start_epoch + 1, cfg.EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        train_probe_loss, train_probe_acc = eval_loop(model, train_probe_loader, criterion, device)

        val_metrics = {}
        for val_domain in cfg.DOMAINS:
            val_loss, val_acc = eval_loop(model, val_loaders[val_domain], criterion, device)
            val_metrics[val_domain] = {"loss": val_loss, "acc": val_acc}

        selected_val_loss = val_metrics[domain]["loss"]
        selected_val_acc = val_metrics[domain]["acc"]

        train_losses.append(train_loss)
        selected_val_losses.append(selected_val_loss)
        train_accs.append(train_acc)
        selected_val_accs.append(selected_val_acc)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_probe_loss": train_probe_loss,
            "train_probe_acc": train_probe_acc,
        }
        for val_domain in cfg.DOMAINS:
            row[f"val{val_domain}_loss"] = val_metrics[val_domain]["loss"]
            row[f"val{val_domain}_acc"] = val_metrics[val_domain]["acc"]
        epoch_metrics_rows.append(row)

        val_log_parts = [
            f"Val{val_domain} Loss: {row[f'val{val_domain}_loss']:.4f} | "
            f"Val{val_domain} Acc: {row[f'val{val_domain}_acc']:.2f}%"
            for val_domain in cfg.DOMAINS
        ]
        logger.info(
            (
                "Epoch [%d/%d] | Train Loss: %.4f | Train Acc: %.2f%% | "
                "Train-Probe Loss: %.4f | Train-Probe Acc: %.2f%% | "
                "%s"
            ),
            epoch,
            cfg.EPOCHS,
            train_loss,
            train_acc,
            train_probe_loss,
            train_probe_acc,
            " | ".join(val_log_parts),
        )

        checkpoint_path = save_checkpoint(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": selected_val_loss,
                "val_acc": selected_val_acc,
                "domain": domain,
            },
            cfg.CHECKPOINT_DIR,
            f"checkpoint_{domain}_epoch{epoch}.pth",
        )
        logger.info("Checkpoint saved: %s", checkpoint_path)

        if selected_val_loss < best_val_loss:
            best_val_loss = selected_val_loss
            best_path = save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": selected_val_loss,
                    "val_acc": selected_val_acc,
                    "domain": domain,
                },
                cfg.CHECKPOINT_DIR,
                f"best_model_{domain}.pth",
            )
            logger.info("New best model saved: %s (val_loss=%.4f)", best_path, best_val_loss)

    plot_curves(
        train_losses,
        selected_val_losses,
        train_accs,
        selected_val_accs,
        cfg.OUTPUT_DIR,
    )
    metrics_excel_path = os.path.join(cfg.OUTPUT_DIR, "epoch_metrics.xlsx")
    save_epoch_metrics_to_excel(epoch_metrics_rows, metrics_excel_path)
    logger.info("Training curves saved to %s/", cfg.OUTPUT_DIR)
    logger.info("Epoch metrics saved to %s", metrics_excel_path)


if __name__ == "__main__":
    main()
