"""test.py — Evaluate the best model on the test set.

Usage:
    python test.py --dataset white_black [--checkpoint checkpoints/white_black/best_model.pth]

The script:
  1. Loads the test datasets from data/test/<variants>.
  2. Loads model weights from the specified checkpoint (default: best_model.pth for the variant).
  3. Computes accuracy for each variant.
  4. Writes a test log to logs/test_<variant>.log.
"""

import argparse
import logging
import os

import torch
from torch.utils.data import DataLoader

import config as cfg
from dataset import MyDataset, eval_transform
from model import CNN
from utils import load_checkpoint


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


def evaluate(model, loader, device):
    """Compute accuracy (%) for a dataloader."""
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    if total_samples == 0:
        return 0.0
    return 100.0 * total_correct / total_samples


# ---------------------------------------------------------------------------
# Main evaluation routine
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate digit-recognition CNN on test set")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=cfg.DATASET_VARIANTS,
        default=cfg.DATASET_VARIANTS[0],
        help="Dataset variant used to select the model checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the model checkpoint to evaluate.",
    )
    args = parser.parse_args()
    dataset_variant = args.dataset
    if args.checkpoint is None:
        args.checkpoint = os.path.join(cfg.CHECKPOINT_DIR, dataset_variant, "best_model.pth")

    device = cfg.DEVICE
    logger = setup_logging(os.path.join(cfg.LOG_DIR, f"test_{dataset_variant}.log"))
    logger.info("Using device: %s", device)

    use_persistent = cfg.NUM_WORKERS > 0
    common_loader_kwargs = {
        "batch_size": cfg.BATCH_SIZE,
        "num_workers": cfg.NUM_WORKERS,
        "pin_memory": True,
        "persistent_workers": use_persistent,
        "prefetch_factor": 2 if use_persistent else None,
    }

    # Model
    model = CNN().to(device)

    if not os.path.isfile(args.checkpoint):
        logger.error("Checkpoint not found: %s", args.checkpoint)
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    logger.info("Loading checkpoint: %s", args.checkpoint)
    load_checkpoint(args.checkpoint, model)

    for variant in cfg.DATASET_VARIANTS:
        variant_dir = os.path.join(cfg.TEST_DIR, variant)
        test_dataset = MyDataset(variant_dir, transform=eval_transform)
        test_loader = DataLoader(test_dataset, shuffle=False, **common_loader_kwargs)
        accuracy = evaluate(model, test_loader, device)
        logger.info("Test Accuracy (%s): %.2f%%", variant, accuracy)


if __name__ == "__main__":
    main()
