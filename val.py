import torch


def evaluate(model, loader, criterion, device):
    """Run evaluation and return (avg_loss, accuracy %)."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples
    return avg_loss, accuracy


def eval_loop(model, loader, criterion, device):
    """Alias of evaluate() for callers that prefer eval_loop naming."""
    return evaluate(model, loader, criterion, device)
