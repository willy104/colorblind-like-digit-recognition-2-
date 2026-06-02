# Colorblind-like Digit Recognition

本專案將原本的 Google Colab + Drive 架構重構成可在 **本機 Windows/Linux** 執行的完整 PyTorch 專案。

## 專案資料夾結構

```
project/
│
├── data/
│   ├── train/          # 訓練圖片（digit_X_NNNNNN.png 格式）
│   │   ├── white_black/
│   │   ├── rainbow_bw/
│   │   └── bw_rainbow/
│   ├── val/            # 驗證圖片
│   │   ├── white_black/
│   │   ├── rainbow_bw/
│   │   └── bw_rainbow/
│   └── test/           # 測試圖片
│       ├── white_black/
│       ├── rainbow_bw/
│       └── bw_rainbow/
│
├── checkpoints/        # 每個 epoch 的 checkpoint 與最佳模型
├── logs/               # 訓練與測試 log 記錄
├── outputs/            # 指標彙整 Excel
│
├── config.py           # 超參數與路徑集中設定
├── dataset.py          # 自訂 Dataset 與資料前處理
├── model.py            # CNN 模型架構（ConvBlock + BatchNorm + AdaptiveAvgPool）
├── train.py            # 訓練流程（含交叉驗證與 checkpoint）
├── val.py              # 驗證流程（交叉驗證）
├── test.py             # 測試流程（交叉測試）
├── infer.py            # 單圖推論 CLI
├── utils.py            # 工具函式（checkpoint 存取、Excel 指標輸出）
├── requirements.txt    # Python 套件需求清單
└── README.md           # 本說明檔
```

## 快速開始

### 1. 安裝套件

```bash
pip install -r requirements.txt
```

### 2. 準備資料

將圖片放入對應資料夾，檔名格式必須為 `digit_X_NNNNNN.png`（例如 `digit_3_000123.png`）：

```
data/train/white_black/   ← 訓練集 (約 52000 張)
data/train/rainbow_bw/    ← 訓練集 (約 52000 張)
data/train/bw_rainbow/    ← 訓練集 (約 52000 張)
data/val/white_black/     ← 驗證集 (約 8000 張)
data/val/rainbow_bw/      ← 驗證集 (約 8000 張)
data/val/bw_rainbow/      ← 驗證集 (約 8000 張)
data/test/white_black/    ← 測試集 (約 10000 張)
data/test/rainbow_bw/     ← 測試集 (約 10000 張)
data/test/bw_rainbow/     ← 測試集 (約 10000 張)
```

若資料不放在專案內的 `data/`，可設定環境變數 `DATA_ROOT` 指向本機資料根目錄（其下仍需 `train/val/test` 與三種分類資料夾）：

```bash
# macOS/Linux
DATA_ROOT=/path/to/local/data python train.py --dataset white_black

# Windows PowerShell
$env:DATA_ROOT="C:\你的\資料路徑"; python train.py --dataset white_black
```

若資料原在 Google Drive，可使用 `rclone` 或直接複製同步到本機。

### 3. 訓練

```bash
# 全新訓練（指定模型對應資料集）
python train.py --dataset white_black

# 從 checkpoint 續訓
python train.py --dataset white_black --resume checkpoints/white_black/checkpoint_epoch10.pth
```

訓練過程中：
- 每個 epoch 自動儲存 checkpoint 至 `checkpoints/<variant>/`
- 驗證 loss 最低時更新 `checkpoints/<variant>/best_model.pth`
- 每個 epoch 會做三種驗證資料集的交叉驗證
- Log 輸出至終端與 `logs/train_<variant>.log`
- 每個 epoch 的 train 與交叉 val 指標會彙整輸出至 `outputs/<variant>/epoch_metrics.xlsx`

### 4. 測試

```bash
python test.py --dataset white_black [--checkpoint checkpoints/white_black/best_model.pth]
```

輸出：
- 三種測試資料集的 Accuracy
- Log 儲存至 `logs/test_<variant>.log`

### 5. 單圖推論

```bash
python infer.py --image path/to/example.png --dataset white_black \
                [--checkpoint checkpoints/white_black/best_model.pth]
```

輸出範例：

```
預測類別 (0-9): 3
```

## 超參數設定

所有超參數集中於 `config.py`，常用設定如下：

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `IMAGE_SIZE` | 256 | 輸入圖片尺寸 |
| `BATCH_SIZE` | 16 | 每個 batch 的樣本數 |
| `EPOCHS` | 20 | 訓練總 epoch 數 |
| `LEARNING_RATE` | 5e-4 | Adam 優化器學習率 |
| `NUM_WORKERS` | CPU 核心數 | DataLoader 工作程序數 |

## 模型架構

CNN 包含：
- 5 個 `ConvBlock`（卷積 + BatchNorm + LeakyReLU），前 4 個附 MaxPool2d
- `AdaptiveAvgPool2d(1,1)` 全域平均池化（取代龐大的 Flatten+Linear）
- 兩層 FC（含 Dropout 0.5）最終輸出 10 類

## GPU 支援

程式自動偵測 CUDA；若無 GPU，則改用 CPU 運算（於 `config.py` 中的 `DEVICE` 設定）。
