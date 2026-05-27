'''
建立一個指令腳本可辨識「單張圖」

指令 Usage：
    python infer.py --image path/to/image.png
                    [--checkpoint checkpoints/best_model.pth]  # 指定模型的 checkpoint 檔案

執行結果輸出：
腳本會將預測的數字（0~9）直接印出
'''

import argparse
import os

import torch
from PIL import Image

import config as cfg
from dataset import eval_transform
from model import CNN
from utils import load_checkpoint


def predict(image_path, checkpoint_path, device):
    # Load model, image 預處理, 回傳預測值
    
    # Load model
    model = CNN().to(device)

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    load_checkpoint(checkpoint_path, model)
    model.eval()  # predict 時不會更新權重

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
