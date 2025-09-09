# Windows環境快速安裝指南

## 第一步：創建工作目錄
1. 在桌面建立一個新資料夾，命名為 `AI學習`
2. 開啟命令提示字元（按 Win+R，輸入 `cmd`，按 Enter）
3. 切換到工作目錄：
```cmd
cd Desktop\AI學習
```

## 第二步：建立虛擬環境
```cmd
# 建立虛擬環境
python -m venv ai_env

# 啟動虛擬環境
ai_env\Scripts\activate

# 你應該會看到提示符前面出現 (ai_env)
```

## 第三步：一鍵安裝所有套件
```cmd
# 更新pip
python -m pip install --upgrade pip

# 安裝所有必要套件
pip install numpy pandas matplotlib scikit-learn tensorflow jupyter ipykernel

# 如果上面指令失敗，可以分別安裝：
pip install numpy
pip install pandas  
pip install matplotlib
pip install scikit-learn
pip install tensorflow
pip install jupyter
```

## 第四步：驗證安裝
下載並執行測試檔案：
```cmd
python environment_test.py
```

## 第五步：開始學習
```cmd
# 執行第一個神經網路程式
python first_neural_network.py

# 或開啟Jupyter Notebook
jupyter notebook
```

## 常見安裝問題解決

### 問題1：找不到Python
**錯誤訊息：** `'python' 不是內部或外部命令`

**解決方法：**
1. 重新安裝Python，確保勾選「Add Python to PATH」
2. 或手動添加Python到系統路徑
3. 重新開啟命令提示字元

### 問題2：pip安裝失敗
**錯誤訊息：** 網路連線錯誤或超時

**解決方法：**
```cmd
# 使用國內鏡像源
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ tensorflow
pip install -i https://pypi.douban.com/simple/ numpy pandas matplotlib
```

### 問題3：TensorFlow安裝失敗
**解決方法：**
```cmd
# 先升級pip和setuptools
python -m pip install --upgrade pip setuptools

# 安裝Microsoft Visual C++ Redistributable
# 從微軟官網下載並安裝

# 重新安裝TensorFlow
pip install tensorflow
```

### 問題4：權限錯誤
**解決方法：**
1. 以系統管理員身份執行命令提示字元
2. 或使用用戶安裝：
```cmd
pip install --user tensorflow
```

## 推薦的資料夾結構
```
Desktop\
└── AI學習\
    ├── ai_env\                 # 虛擬環境
    ├── environment_test.py     # 環境測試
    ├── first_neural_network.py # 第一個神經網路
    ├── projects\               # 你的項目
    │   ├── mnist\
    │   ├── cat_dog\
    │   └── sentiment\
    └── datasets\               # 資料集存放處
```

## 完整的一鍵安裝腳本
將以下內容保存為 `quick_setup.bat`：

```batch
@echo off
echo ========================================
echo AI學習環境自動安裝腳本
echo ========================================

echo 正在檢查Python...
python --version
if errorlevel 1 (
    echo 錯誤：找不到Python，請先安裝Python
    pause
    exit /b 1
)

echo 正在建立虛擬環境...
python -m venv ai_env

echo 正在啟動虛擬環境...
call ai_env\Scripts\activate.bat

echo 正在更新pip...
python -m pip install --upgrade pip

echo 正在安裝套件...
pip install numpy pandas matplotlib scikit-learn tensorflow jupyter ipykernel

echo 正在驗證安裝...
python -c "import tensorflow as tf; print('TensorFlow版本:', tf.__version__)"

echo ========================================
echo 安裝完成！
echo 請執行以下命令開始使用：
echo ai_env\Scripts\activate
echo python first_neural_network.py
echo ========================================
pause
```

## 啟動腳本
將以下內容保存為 `start_ai.bat`：

```batch
@echo off
echo 啟動AI學習環境...
call ai_env\Scripts\activate.bat
echo 環境已啟動！你可以開始執行Python程式了。
echo.
echo 建議執行的程式：
echo 1. python environment_test.py     ^(測試環境^)
echo 2. python first_neural_network.py ^(第一個神經網路^)
echo 3. jupyter notebook               ^(開啟Jupyter^)
echo.
cmd /k
```

使用方法：
1. 雙擊 `quick_setup.bat` 進行一鍵安裝
2. 以後每次學習時雙擊 `start_ai.bat` 啟動環境

這樣就能確保最高的成功率和最簡單的使用體驗！