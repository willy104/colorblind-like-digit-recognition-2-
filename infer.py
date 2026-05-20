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
    valid_exts = (".png", ".jpg", ".jpeg")

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not image_path.lower().endswith(valid_exts):
        raise ValueError("Unsupported image format. Please use PNG/JPEG.")

    # Load model
    model = CNN().to(device)

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    load_checkpoint(checkpoint_path, model, map_location=device)
    model.eval()

    # Preprocess image
    with Image.open(image_path) as img:
        image = img.convert("RGB")
    tensor = eval_transform(image).unsqueeze(0).to(device)  # (1, C, H, W)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)
        top_k = min(3, probs.shape[1])
        top_values, top_indices = torch.topk(probs, k=top_k, dim=1)
        top_preds = list(
            zip(
                top_indices.squeeze(0).tolist(),
                top_values.squeeze(0).tolist(),
            )
        )

    return pred_idx.item(), confidence.item(), top_preds


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
    print(f"Using device: {device}")
    label, confidence, top_preds = predict(args.image, args.checkpoint, device)
    print(f"預測類別 (0-9): {label}")
    print(f"預測機率: {confidence:.4f}")
    print("Top-3 預測:")
    for rank, (cls_idx, prob) in enumerate(top_preds, start=1):
        print(f"  {rank}. 類別 {cls_idx}: {prob:.4f}")


if __name__ == "__main__":
    main()
