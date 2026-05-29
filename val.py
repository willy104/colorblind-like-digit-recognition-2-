"""
val.py — Cross-validation utilities for digit-recognition CNN.

Provides helpers to validate a model across multiple dataset variants.
"""

import os

import torch
from torch.utils.data import DataLoader

from dataset import MyDataset, eval_transform


def validate(model, loader, criterion, device):
    """Run validation and return (avg_loss, accuracy %)."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples
    return avg_loss, accuracy


def validate_cross(model, criterion, device, val_root, variants, loader_kwargs):
    """Validate on multiple dataset variants under val_root."""
    metrics = {}
    for variant in variants:
        variant_dir = os.path.join(val_root, variant)
        dataset = MyDataset(variant_dir, transform=eval_transform)
        loader = DataLoader(dataset, shuffle=False, **loader_kwargs)
        loss, acc = validate(model, loader, criterion, device)
        metrics[variant] = {"loss": loss, "acc": acc}
    return metrics
