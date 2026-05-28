"""test.py — Evaluate checkpoint on A/B/C full test sets."""

import argparse
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from torch.utils.data import DataLoader

import config as cfg
from dataset import MyDataset, eval_transform
from model import CNN
from utils import load_checkpoint
from val import evaluate


def setup_logger(domain):
    os.makedirs(cfg.LOG_DIR, exist_ok=True)
    logger = logging.getLogger("test")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(os.path.join(cfg.LOG_DIR, f"test_{domain}.log"))
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def make_loader(dataset):
    use_persistent = cfg.NUM_WORKERS > 0
    return DataLoader(
        dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=use_persistent,
        prefetch_factor=2 if use_persistent else None,
    )


def collect_predictions(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)


def main():
    parser = argparse.ArgumentParser(description="Evaluate digit-recognition CNN on A/B/C test sets")
    parser.add_argument(
        "--domain",
        type=str,
        default="A",
        choices=cfg.DOMAINS,
        help="Model domain used for default checkpoint naming.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the model checkpoint to evaluate.",
    )
    args = parser.parse_args()

    domain = args.domain.upper()
    logger = setup_logger(domain)

    checkpoint_path = args.checkpoint or os.path.join(cfg.CHECKPOINT_DIR, f"best_model_{domain}.pth")

    device = cfg.DEVICE
    logger.info("Using device: %s", device)
    logger.info("Model domain: %s", domain)

    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()

    if not os.path.isfile(checkpoint_path):
        logger.error("Checkpoint not found: %s", checkpoint_path)
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info("Loading checkpoint: %s", checkpoint_path)
    load_checkpoint(checkpoint_path, model)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    for test_domain in cfg.DOMAINS:
        test_dataset = MyDataset(cfg.get_domain_split_dir(test_domain, "test"), transform=eval_transform)
        test_loader = make_loader(test_dataset)

        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        all_labels, all_preds = collect_predictions(model, test_loader, device)

        logger.info("[%s] Test Loss: %.4f | Test Accuracy: %.2f%%", test_domain, test_loss, test_acc)

        report = classification_report(all_labels, all_preds, digits=4)
        logger.info("[%s] Classification Report:\n%s", test_domain, report)

        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        fig, ax = plt.subplots(figsize=(8, 8))
        disp.plot(ax=ax, colorbar=False)
        plt.title(f"Confusion Matrix ({test_domain})")
        cm_path = os.path.join(cfg.OUTPUT_DIR, f"confusion_matrix_{test_domain}.png")
        plt.savefig(cm_path, bbox_inches="tight")
        plt.close(fig)
        logger.info("[%s] Confusion matrix saved to %s", test_domain, cm_path)


if __name__ == "__main__":
    main()
