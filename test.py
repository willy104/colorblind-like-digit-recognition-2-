"""test.py — Evaluate the best model on the test set.

Usage:
    python test.py [--checkpoint checkpoints/best_model.pth]

The script:
  1. Loads the test dataset from data/test.
  2. Loads model weights from the specified checkpoint (default: best_model.pth).
  3. Computes overall accuracy and per-class classification report.
  4. Saves a confusion matrix image to outputs/confusion_matrix.png.
  5. Writes a test log to logs/test.log.
"""

import argparse
import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import config as cfg
from dataset import MyDataset, eval_transform
from model import CNN
from utils import load_checkpoint


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

os.makedirs(cfg.LOG_DIR, exist_ok=True)
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(cfg.LOG_DIR, "test.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main evaluation routine
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate digit-recognition CNN on test set")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(cfg.CHECKPOINT_DIR, "best_model.pth"),
        help="Path to the model checkpoint to evaluate.",
    )
    args = parser.parse_args()

    device = cfg.DEVICE
    logger.info("Using device: %s", device)

    # Dataset & loader
    test_dataset = MyDataset(cfg.TEST_DIR, transform=eval_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
    )

    # Model
    model = CNN().to(device)

    if not os.path.isfile(args.checkpoint):
        logger.error("Checkpoint not found: %s", args.checkpoint)
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    logger.info("Loading checkpoint: %s", args.checkpoint)
    load_checkpoint(args.checkpoint, model)

    # Inference
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Accuracy
    accuracy = (all_preds == all_labels).mean() * 100
    logger.info("Test Accuracy: %.2f%%", accuracy)

    # Classification report
    report = classification_report(all_labels, all_preds, digits=4)
    logger.info("Classification Report:\n%s", report)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, colorbar=False)
    plt.title("Confusion Matrix")
    cm_path = os.path.join(cfg.OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()
    logger.info("Confusion matrix saved to %s", cm_path)


if __name__ == "__main__":
    main()
