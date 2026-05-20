"""infer.py — Single-image inference CLI for the digit-recognition CNN.

Usage:
    python infer.py --image path/to/image.png
                    [--checkpoint checkpoints/best_model.pth]

The predicted digit (0–9) is printed to stdout.
"""

import argparse
import os

import torch
from PIL import Image

import config as cfg
from dataset import eval_transform
from model import CNN
from utils import load_checkpoint


def predict(image_path, checkpoint_path, device):
    """Load model, preprocess a single image, and return the predicted label."""
    # Load model
    model = CNN().to(device)

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    load_checkpoint(checkpoint_path, model)
    model.eval()

    # Preprocess image
    image = Image.open(image_path).convert("RGB")
    tensor = eval_transform(image).unsqueeze(0).to(device)  # (1, C, H, W)

    with torch.no_grad():
        output = model(tensor)
        label = output.argmax(dim=1).item()

    return label


def main():
    parser = argparse.ArgumentParser(description="Predict digit class from a single image")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input image (PNG/JPEG).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(cfg.CHECKPOINT_DIR, "best_model.pth"),
        help="Path to the model checkpoint.",
    )
    args = parser.parse_args()

    device = cfg.DEVICE
    label = predict(args.image, args.checkpoint, device)
    print(f"預測類別 (0-9): {label}")


if __name__ == "__main__":
    main()
