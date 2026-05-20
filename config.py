import os
import torch

# Project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

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

# Data paths
# Priority:
# 1) TRAIN_DIR / VAL_DIR / TEST_DIR env vars
# 2) DATA_ROOT env var + split name
# 3) <project>/data/<split>
DATA_ROOT = os.getenv("DATA_ROOT", os.path.join(PROJECT_ROOT, "data"))
TRAIN_DIR = os.path.abspath(os.path.expanduser(os.getenv("TRAIN_DIR", os.path.join(DATA_ROOT, "train"))))
VAL_DIR = os.path.abspath(os.path.expanduser(os.getenv("VAL_DIR", os.path.join(DATA_ROOT, "val"))))
TEST_DIR = os.path.abspath(os.path.expanduser(os.getenv("TEST_DIR", os.path.join(DATA_ROOT, "test"))))

# Output directories
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

# DataLoader settings
NUM_WORKERS = os.cpu_count() or 4

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
