import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import config as cfg


class MyDataset(Dataset):
    '''
    讀取訓練圖做分類並標記 label
    訓練圖的檔名：digit_X_NNNNNN.png ( X：(0~9) )
    e.g. digit_3_000123.png -> label=3
    '''

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [
            f for f in os.listdir(root_dir) if f.lower().endswith(".png")
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        # Parse label from filename: digit_X_NNNNNN.png -> label=X
        parts = os.path.basename(img_name).split("_")
        if len(parts) < 2:
            raise ValueError(
                f"Unexpected filename format '{img_name}'. "
                "Expected 'digit_X_NNNNNN.png'."
            )
        label = int(parts[1])

        if self.transform:
            image = self.transform(image)
        return image, label


# Shared transforms 訓練圖預處理
train_transform = transforms.Compose([
    transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(cfg.MEAN, cfg.STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(cfg.MEAN, cfg.STD),
])
