import os
import torch

# Image settings
IMAGE_SIZE = 256  # 圖片大小 256*256
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)
'''
此變換通常應用於輸入圖像，然後再將它們輸入到神經網路中。
說明：
transforms.ToTensor()：此變換將 PIL 影像或 NumPy ndarray 轉換為 PyTorch 張量。
並將像素值從圖像的典型範圍 [0,255] 縮放到浮點張量的 [0.0,1.0]。
transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))：此變換將影像張量的像素值標準化。
兩個參數：mean(平均)和std（標準差），在這裡我們設定mean=(0.5,0.5,0.5)和std=(0.5,0.5,0.5)。
所應用的標準化公式為 (x - 平均值) / 標準差。
設定mean=(0.5,0.5,0.5)和std=(0.5,0.5,0.5)的原因：
標準化會將像素值從[0.0,1.0]範圍轉換為[-1.0,1.0]，
讓範圍介於負值和正值之間，是因為後續的激活函數選擇為 LeakyReLU ，負值和正值會有明顯差異對比，
如此一來在神經網路架構的訓練能提升穩定性和效能。
'''

# Training hyperparameters
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 5e-4
NUM_CLASSES = 10  # 最後輸出為 0~9 十種
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
'''
每個 worker 預先載入 NUM_WORKERS 個 batch 放在 queue 裡等主行程來讀取，GPU 訓練時比較不容易「等資料」，配合後續 prefetch。
最終共預先準備 prefetch_factor * NUM_WORKERS 個 batch
'''

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # DEVICE 使用 GPU
