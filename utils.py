import os
import torch
import matplotlib.pyplot as plt
from openpyxl import Workbook


def save_checkpoint(state, checkpoint_dir, filename):
    """Save a training checkpoint to *checkpoint_dir/filename*."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    torch.save(state, path)
    return path


def load_checkpoint(path, model, optimizer=None):
    """Load a checkpoint and restore model (and optionally optimizer) state.

    Returns:
        dict: The full checkpoint dictionary (contains 'epoch', 'val_loss', etc.)
    """
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def plot_curves(
    train_losses,
    val_losses,
    train_accs,
    val_accs,
    output_dir,
    test_losses=None,
    test_accs=None,
):
    """Plot and save loss and accuracy curves to *output_dir*."""
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    # Loss curve
    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    if test_losses is not None:
        plt.plot(epochs, test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation vs Test Loss" if test_losses is not None else "Training vs Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()

    # Accuracy curve
    plt.figure()
    plt.plot(epochs, train_accs, label="Train Accuracy")
    plt.plot(epochs, val_accs, label="Val Accuracy")
    if test_accs is not None:
        plt.plot(epochs, test_accs, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(
        "Training vs Validation vs Test Accuracy"
        if test_accs is not None
        else "Training vs Validation Accuracy"
    )
    plt.legend()
    plt.savefig(os.path.join(output_dir, "accuracy_curve.png"))
    plt.close()


def save_epoch_metrics_to_excel(rows, output_path):
    """Save epoch metrics rows to an Excel file."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "epoch_metrics"
    worksheet.append([
        "epoch",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "test_loss",
        "test_acc",
    ])
    for row in rows:
        worksheet.append([
            row["epoch"],
            row["train_loss"],
            row["train_acc"],
            row["val_loss"],
            row["val_acc"],
            row["test_loss"],
            row["test_acc"],
        ])
    workbook.save(output_path)
