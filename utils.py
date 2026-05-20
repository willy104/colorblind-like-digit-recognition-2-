import os
import torch
import matplotlib.pyplot as plt


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


def plot_curves(train_losses, val_losses, train_accs, val_accs, output_dir):
    """Plot and save loss and accuracy curves to *output_dir*."""
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    # Loss curve
    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()

    # Accuracy curve
    plt.figure()
    plt.plot(epochs, train_accs, label="Train Accuracy")
    plt.plot(epochs, val_accs, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "accuracy_curve.png"))
    plt.close()
