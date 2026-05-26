import os
import torch

# Image settings
IMAGE_SIZE = 256
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)

# Training hyperparameters
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 5e-4
NUM_CLASSES = 10
AVG_EVERY = 20  # log average loss/accuracy every N batches

# Data paths (relative to project root)
TRAIN_DIR = os.path.join("data", "train")
VAL_DIR = os.path.join("data", "val")
TEST_DIR = os.path.join("data", "test")

# Output directories
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"
OUTPUT_DIR = "outputs"

# DataLoader settings
NUM_WORKERS = 4

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
