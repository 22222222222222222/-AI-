# AlexNet 完整教學實作

這是一個完整的 AlexNet 深度學習教學項目，使用 PyTorch 實現了經典的 AlexNet 架構，並在 CIFAR-10 數據集上進行訓練和測試。本項目特別適合深度學習初學者理解卷積神經網路的工作原理。

## 🔥 AlexNet 的歷史背景和重要性

### 歷史背景
- **發表時間**: 2012年
- **論文**: "ImageNet Classification with Deep Convolutional Neural Networks"
- **作者**: Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton
- **重要性**: 在 ImageNet ILSVRC-2012 競賽中獲得冠軍，top-5 錯誤率僅為 15.3%，遠超第二名的 26.2%

### 革命性貢獻
1. **證明了深度學習的威力**: 首次在大規模視覺識別任務中展現出深度神經網路的巨大潛力
2. **ReLU 激活函數**: 首次在 CNN 中大規模使用 ReLU，解決了梯度消失問題
3. **GPU 加速**: 展示了 GPU 在深度學習訓練中的重要作用
4. **數據增強**: 系統性地使用了數據增強技術防止過擬合
5. **Dropout 技術**: 在全連接層中使用 Dropout 防止過擬合

## 🏗️ 模型架構詳細解釋

### 整體架構
AlexNet 包含 **8 個學習層**：
- **5 個卷積層** (Convolutional Layers)
- **3 個全連接層** (Fully Connected Layers)

### 詳細層級結構

```
輸入: 3×32×32 (CIFAR-10 圖片)
    ↓
第一卷積層: Conv2d(3→64, kernel=5×5) + ReLU + MaxPool(2×2)
    ↓ 64×16×16
第二卷積層: Conv2d(64→192, kernel=3×3) + ReLU + MaxPool(2×2)
    ↓ 192×8×8
第三卷積層: Conv2d(192→384, kernel=3×3) + ReLU
    ↓ 384×8×8
第四卷積層: Conv2d(384→256, kernel=3×3) + ReLU
    ↓ 256×8×8
第五卷積層: Conv2d(256→256, kernel=3×3) + ReLU + MaxPool(2×2)
    ↓ 256×4×4
自適應平均池化: AdaptiveAvgPool2d(2×2)
    ↓ 256×2×2 = 1024
展平 (Flatten)
    ↓ 1024
第一全連接層: Linear(1024→1024) + ReLU + Dropout(0.5)
    ↓ 1024
第二全連接層: Linear(1024→512) + ReLU
    ↓ 512
輸出層: Linear(512→10)
    ↓ 10 (CIFAR-10 類別數)
```

### 關鍵技術特點

1. **ReLU 激活函數**
   ```python
   f(x) = max(0, x)
   ```
   - 優點：解決梯度消失、計算簡單、收斂快速
   - 相比 Sigmoid/Tanh：避免飽和區域的梯度消失問題

2. **局部響應歸一化 (LRN)**
   - 原始 AlexNet 使用，本實現為簡化未包含
   - 作用：增強模型的泛化能力

3. **Dropout 正規化**
   ```python
   nn.Dropout(0.5)  # 隨機將 50% 的神經元輸出設為 0
   ```
   - 防止過擬合
   - 僅在訓練時啟用，測試時關閉

4. **數據增強**
   - 隨機水平翻轉
   - 隨機裁剪和填充
   - 隨機旋轉
   - 色彩抖動

## 📁 項目結構

```
alexnet_tutorial/
├── README.md                 # 本說明文件
├── requirements.txt          # Python 依賴包
├── alexnet_model.py          # AlexNet 模型實現
├── train.py                  # 訓練腳本
├── inference.py              # 推理腳本
├── data_loader.py            # 數據加載器
├── utils.py                  # 工具函數
└── outputs/                  # 訓練輸出目錄 (自動創建)
    ├── checkpoints/          # 模型檢查點
    ├── logs/                 # 訓練日誌
    └── training_curves.png   # 訓練曲線圖
```

## 🚀 快速開始

### 1. 環境準備

#### 創建虛擬環境 (推薦)
```bash
# 使用 conda
conda create -n alexnet python=3.8
conda activate alexnet

# 或使用 venv
python -m venv alexnet_env
source alexnet_env/bin/activate  # Linux/Mac
# alexnet_env\Scripts\activate     # Windows
```

#### 安裝依賴
```bash
cd alexnet_tutorial
pip install -r requirements.txt
```

### 2. 數據集準備

程序會自動下載 CIFAR-10 數據集到 `./data` 目錄：
- **訓練集**: 50,000 張圖片
- **測試集**: 10,000 張圖片
- **類別**: 10 類 (飛機、汽車、鳥類、貓、鹿、狗、蛙、馬、船、卡車)
- **圖片大小**: 32×32 彩色圖片

### 3. 模型訓練

#### 基本訓練
```bash
python train.py
```

#### 自定義參數訓練
```bash
python train.py \
    --batch_size 64 \
    --epochs 100 \
    --learning_rate 0.001 \
    --output_dir ./my_outputs
```

#### 從檢查點恢復訓練
```bash
python train.py --resume
```

### 4. 模型推理

#### 在測試集上評估
```bash
python inference.py \
    --model_path ./outputs/checkpoints/best_model.pth \
    --evaluate_test
```

#### 預測單張圖片
```bash
python inference.py \
    --model_path ./outputs/checkpoints/best_model.pth \
    --image_path /path/to/your/image.jpg
```

## 📊 訓練監控

### TensorBoard 可視化
```bash
# 啟動 TensorBoard
tensorboard --logdir ./outputs/logs

# 在瀏覽器中打開 http://localhost:6006
```

可以監控的指標：
- 訓練/驗證損失
- 訓練/驗證準確率
- 學習率變化

### 訓練輸出文件
- `checkpoints/best_model.pth`: 最佳模型權重
- `checkpoints/latest.pth`: 最新檢查點
- `logs/training_log.json`: 訓練歷史記錄
- `training_curves.png`: 訓練曲線圖

## 🔧 文件功能說明

### alexnet_model.py
- **功能**: AlexNet 模型架構實現
- **特點**: 
  - 詳細的中文註解
  - 適配 CIFAR-10 的網路結構
  - 權重初始化
  - 模型測試功能

**關鍵類和函數**:
```python
class AlexNet(nn.Module)  # 主要模型類
def alexnet(num_classes=10, pretrained=False)  # 模型構建函數
```

### data_loader.py
- **功能**: CIFAR-10 數據集加載和預處理
- **特點**:
  - 數據增強策略
  - 標準化處理
  - 可視化工具
  - 統計信息計算

**關鍵函數**:
```python
get_cifar10_dataloaders()  # 獲取數據加載器
visualize_dataset_samples()  # 可視化樣本
get_cifar10_transforms()  # 獲取數據變換
```

### train.py
- **功能**: 完整的訓練循環
- **特點**:
  - 命令行參數支持
  - 檢查點保存/恢復
  - 早停機制
  - TensorBoard 日誌

**主要功能**:
```python
train_one_epoch()  # 訓練一個 epoch
validate()  # 驗證模型
main()  # 主訓練函數
```

### inference.py
- **功能**: 模型推理和評估
- **特點**:
  - 單張圖片預測
  - 測試集批量評估
  - 混淆矩陣繪製
  - 結果可視化

**主要功能**:
```python
predict_single_image()  # 單張圖片預測
evaluate_test_set()  # 測試集評估
visualize_prediction()  # 結果可視化
```

### utils.py
- **功能**: 工具函數集合
- **特點**:
  - 模型保存/加載
  - 訓練曲線繪製
  - 準確率計算
  - 早停機制

**實用工具**:
```python
save_model() / load_model()  # 模型 I/O
plot_training_history()  # 繪製訓練曲線
AverageMeter()  # 平均值計算器
EarlyStopping()  # 早停機制
```

## ⚙️ 訓練參數詳解

### 基本參數
- `--batch_size`: 批次大小 (預設: 128)
  - 較大的批次: 更穩定的梯度，但需要更多記憶體
  - 較小的批次: 更多的梯度更新，可能更快收斂

- `--epochs`: 訓練輪數 (預設: 50)
  - 根據收斂情況調整
  - 建議配合早停機制使用

- `--learning_rate`: 學習率 (預設: 0.01)
  - 太大: 可能發散或震盪
  - 太小: 收斂太慢

### 優化器參數
- `--momentum`: SGD 動量 (預設: 0.9)
  - 幫助穿越局部最小值
  - 加速收斂

- `--weight_decay`: 權重衰減 (預設: 5e-4)
  - L2 正規化係數
  - 防止過擬合

### 學習率調度
- `--lr_step_size`: 學習率衰減步長 (預設: 20)
- `--lr_gamma`: 學習率衰減因子 (預設: 0.1)

## 🎯 參數調優建議

### 學習率調優
1. **學習率過大的表現**:
   - 損失劇烈震盪
   - 無法收斂或發散

2. **學習率過小的表現**:
   - 收斂極慢
   - 容易卡在局部最小值

3. **推薦策略**:
   ```python
   # 學習率範圍測試
   learning_rates = [0.1, 0.01, 0.001, 0.0001]
   ```

### 批次大小選擇
- **小批次 (32-64)**: 適合記憶體受限的環境
- **中批次 (128-256)**: 平衡效果，推薦使用
- **大批次 (512+)**: 需要大記憶體，可能需要調整學習率

### 數據增強策略
```python
# 當前使用的增強策略
transforms.RandomHorizontalFlip(p=0.5)      # 隨機水平翻轉
transforms.RandomCrop(32, padding=4)        # 隨機裁剪
transforms.RandomRotation(degrees=10)       # 隨機旋轉
transforms.ColorJitter(...)                 # 色彩抖動
```

## 📈 預期訓練結果

### 性能指標
- **CIFAR-10 測試準確率**: 70-80%
- **訓練時間**: 約 1-2 小時 (GPU) / 6-10 小時 (CPU)
- **模型大小**: 約 60MB

### 訓練曲線特徵
1. **初期** (1-10 epochs): 快速下降
2. **中期** (10-30 epochs): 緩慢改善
3. **後期** (30+ epochs): 趨於穩定

### 常見問題和解決方案

#### 1. 過擬合
**表現**: 訓練準確率持續上升，驗證準確率下降
**解決方案**:
- 增加 Dropout 比例
- 增強數據增強
- 早停機制
- 權重衰減

#### 2. 欠擬合
**表現**: 訓練和驗證準確率都很低
**解決方案**:
- 增加模型複雜度
- 降低正規化強度
- 增加訓練時間

#### 3. 梯度爆炸
**表現**: 損失變為 NaN 或無窮大
**解決方案**:
- 降低學習率
- 梯度剪裁
- 檢查權重初始化

## 💡 進階實驗建議

### 1. 架構改進實驗
```python
# 嘗試不同的卷積核大小
nn.Conv2d(3, 64, kernel_size=3)  # 替代 5x5
nn.Conv2d(3, 64, kernel_size=7)  # 更大的感受野

# 嘗試批次正規化
nn.BatchNorm2d(64)  # 在卷積層後添加

# 嘗試不同的池化策略
nn.AvgPool2d(2, 2)  # 平均池化替代最大池化
```

### 2. 優化器對比實驗
```python
# Adam 優化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

# AdamW 優化器
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# SGD with Nesterov momentum
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
```

### 3. 學習率調度策略
```python
# 餘弦退火
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

# 指數衰減
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

# 自適應調整
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
```

## 🔍 性能分析和調試

### 1. 模型複雜度分析
```python
from utils import count_parameters, print_model_summary

model = alexnet(num_classes=10)
print_model_summary(model)
```

### 2. 訓練過程監控
- 監控 GPU 記憶體使用: `nvidia-smi`
- 監控訓練速度: 每秒處理的樣本數
- 監控梯度範數: 檢測梯度爆炸/消失

### 3. 錯誤分析
```python
# 查看預測錯誤的樣本
python inference.py --model_path best_model.pth --evaluate_test
```

## 📚 學習資源推薦

### 經典論文
1. **AlexNet 原論文**: "ImageNet Classification with Deep Convolutional Neural Networks" (2012)
2. **Dropout**: "Improving neural networks by preventing co-adaptation of feature detectors" (2012)
3. **批次正規化**: "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift" (2015)

### 在線課程
1. **CS231n**: Stanford 的卷積神經網路課程
2. **Deep Learning Specialization**: Coursera 上的深度學習專項課程
3. **PyTorch 官方教程**: https://pytorch.org/tutorials/

### 書籍推薦
1. **"Deep Learning"** by Ian Goodfellow, Yoshua Bengio, and Aaron Courville
2. **"Hands-On Machine Learning"** by Aurélien Géron
3. **"Deep Learning with PyTorch"** by Eli Stevens, Luca Antiga, and Thomas Viehmann

### 實作練習建議
1. **嘗試其他數據集**: MNIST, Fashion-MNIST, CIFAR-100
2. **實現其他經典架構**: VGG, ResNet, DenseNet
3. **遷移學習實驗**: 使用預訓練模型
4. **模型壓縮**: 剪枝、量化、知識蒸餾

## ❓ 常見問題 (FAQ)

### Q: 為什麼選擇 CIFAR-10 而不是 ImageNet？
A: CIFAR-10 的優勢：
- 數據集較小，下載和訓練更快
- 適合教學和快速實驗
- 不需要大量的計算資源
- 原理和 ImageNet 相同，但更易於理解

### Q: 能否在 CPU 上訓練？
A: 可以，但效率較低：
- GPU 訓練: 1-2 小時
- CPU 訓練: 6-10 小時
- 建議先在 CPU 上進行小規模測試

### Q: 如何提高模型準確率？
A: 幾種策略：
1. 增加訓練時間 (更多 epochs)
2. 調整學習率和優化器
3. 增強數據增強策略
4. 使用更復雜的模型架構
5. 集成學習 (ensemble)

### Q: 出現記憶體不足怎麼辦？
A: 解決方案：
1. 減小批次大小 (`--batch_size`)
2. 減少數據加載進程 (`--num_workers`)
3. 使用混合精度訓練
4. 梯度累積

### Q: 如何解讀訓練曲線？
A: 健康的訓練曲線特徵：
- 損失持續下降
- 訓練和驗證準確率都在提升
- 驗證指標沒有大幅度落後於訓練指標

## 🤝 貢獻和反饋

歡迎提出建議和改進：
1. 發現 bug 請提交 issue
2. 有改進建議請提交 pull request
3. 有問題可以在 discussion 中討論

## 📄 授權協議

本項目採用 MIT 授權協議，詳見 LICENSE 文件。

---

**開始你的深度學習之旅吧！** 🚀

如果這個教程對你有幫助，請給個 ⭐ 支持一下！