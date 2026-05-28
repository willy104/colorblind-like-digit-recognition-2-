# Colorblind-like Digit Recognition

本專案將原本的 Google Colab + Drive 架構重構成可在 **本機 Windows/Linux** 執行的完整 PyTorch 專案。

## 專案資料夾結構

```
project/
│
├── data/
│   ├── A/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── B/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── C/
│       ├── train/
│       ├── val/
│       └── test/
│
├── checkpoints/        # 每個 epoch 的 checkpoint 與最佳模型
├── logs/               # 訓練與測試 log 記錄
├── outputs/            # 曲線圖與混淆矩陣圖
│
├── config.py           # 超參數與路徑集中設定
├── dataset.py          # 自訂 Dataset 與資料前處理
├── model.py            # CNN 模型架構（ConvBlock + BatchNorm + AdaptiveAvgPool）
├── train.py            # 訓練流程（含 train-probe + A/B/C 驗證）
├── val.py              # 通用 evaluate/eval_loop
├── test.py             # 測試流程（載入最佳模型、A/B/C test 評估）
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
data/A/train, data/A/val, data/A/test
data/B/train, data/B/val, data/B/test
data/C/train, data/C/val, data/C/test
```

若資料原在 Google Drive，可使用 `rclone` 或直接複製同步到本機。

### 3. 訓練

```bash
# 訓練 domain A/B/C 各自模型
python train.py --domain A
python train.py --domain B
python train.py --domain C

# 從 checkpoint 續訓
python train.py --domain A --resume checkpoints/checkpoint_A_epoch10.pth
```

訓練過程中：
- 每個 epoch 自動儲存 checkpoint 至 `checkpoints/checkpoint_<domain>_epoch{n}.pth`
- 以「訓練 domain 的 val 子集」loss 最低更新 `checkpoints/best_model_<domain>.pth`
- Log 輸出至終端與 `logs/train_<domain>.log`
- 每個 epoch 會記錄 train、train-probe(固定 2000) 與 valA/valB/valC(各固定 2000) 的 loss/acc
- 固定子集檔名清單儲存於 `outputs/splits/`
- 每個 epoch 指標輸出至 `outputs/epoch_metrics.xlsx`

### 4. 測試

```bash
python test.py --domain A
# 或指定 checkpoint
python test.py --checkpoint checkpoints/best_model_A.pth
```

輸出：
- A/B/C 三個 domain 各自的 Accuracy 與 Loss（full test set）
- 各 domain 分類報告
- 各 domain 混淆矩陣（`outputs/confusion_matrix_A.png` 等）
- Log 儲存至 `logs/test_<domain>.log`

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

## 模型架構

CNN 包含：
- 5 個 `ConvBlock`（卷積 + BatchNorm + LeakyReLU），前 4 個附 MaxPool2d
- `AdaptiveAvgPool2d(1,1)` 全域平均池化（取代龐大的 Flatten+Linear）
- 兩層 FC（含 Dropout 0.5）最終輸出 10 類

## GPU 支援

程式自動偵測 CUDA；若無 GPU，則改用 CPU 運算（於 `config.py` 中的 `DEVICE` 設定）。
