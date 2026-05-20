# Colorblind-like Digit Recognition

本專案將原本的 Google Colab + Drive 架構重構成可在 **本機 Windows/Linux** 執行的完整 PyTorch 專案。

## 專案資料夾結構

```
project/
│
├── data/
│   ├── train/          # 訓練圖片（digit_X_NNNNNN.png 格式）
│   ├── val/            # 驗證圖片
│   └── test/           # 測試圖片
│
├── checkpoints/        # 每個 epoch 的 checkpoint 與最佳模型
├── logs/               # 訓練與測試 log 記錄
├── outputs/            # 曲線圖與混淆矩陣圖
│
├── config.py           # 超參數與路徑集中設定
├── dataset.py          # 自訂 Dataset 與資料前處理
├── model.py            # CNN 模型架構（ConvBlock + BatchNorm + AdaptiveAvgPool）
├── train.py            # 訓練流程（含驗證、checkpoint、繪圖等）
├── test.py             # 測試流程（載入最佳模型、混淆矩陣、分類報告）
├── infer.py            # 單圖推論 CLI
├── utils.py            # 工具函式（checkpoint 存取、繪圖）
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
data/train/   ← 訓練集
data/val/     ← 驗證集
data/test/    ← 測試集
```

若資料在雲端，建議先同步到專案 `data/`，可用內建腳本：

```bash
python sync_cloud_data.py \
  --source /path/to/cloud_mounted_dataset \
  --target ./data
```

`--source` 需包含 `train/`、`val/`、`test/` 三個子資料夾。  
腳本只會同步 `.png`，並檢查檔名是否符合 `digit_[0-9]_NNNNNN.png`（`[0-9]` 為類別，`NNNNNN` 為 6 位數字）。

你也可以不複製到 `data/`，直接讓訓練程式讀「已掛載」的雲端目錄（見下方環境變數設定）。

### 3. 訓練

```bash
# 全新訓練
python train.py

# 從 checkpoint 續訓
python train.py --resume checkpoints/checkpoint_epoch10.pth
```

訓練過程中：
- 每個 epoch 自動儲存 checkpoint 至 `checkpoints/`
- 驗證 loss 最低時更新 `checkpoints/best_model.pth`
- Log 輸出至終端與 `logs/train.log`
- 訓練結束後，損失與準確度曲線圖儲存至 `outputs/`

### 4. 測試

```bash
python test.py [--checkpoint checkpoints/best_model.pth]
```

輸出：
- 整體 Accuracy
- 各類別 precision / recall / F1 分類報告
- 混淆矩陣圖（儲存於 `outputs/confusion_matrix.png`）
- Log 儲存至 `logs/test.log`

### 5. 單圖推論

```bash
python infer.py --image path/to/example.png \
                [--checkpoint checkpoints/best_model.pth]
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

### 資料路徑（支援雲端掛載路徑）

程式支援以下環境變數（優先順序由高到低）：

1. `TRAIN_DIR` / `VAL_DIR` / `TEST_DIR`
2. `DATA_ROOT`（程式會自動接上 `train|val|test`）
3. 預設：`<project>/data/train|val|test`

範例（直接讀取掛載在本機的雲端目錄）：

```bash
export DATA_ROOT=/mnt/cloud/digit_dataset
python train.py
```

## 模型架構

CNN 包含：
- 5 個 `ConvBlock`（卷積 + BatchNorm + LeakyReLU），前 4 個附 MaxPool2d
- `AdaptiveAvgPool2d(1,1)` 全域平均池化（取代龐大的 Flatten+Linear）
- 兩層 FC（含 Dropout 0.5）最終輸出 10 類

## GPU 支援

程式自動偵測 CUDA；若無 GPU，則改用 CPU 運算（於 `config.py` 中的 `DEVICE` 設定）。
