@echo off
chcp 65001 > nul
echo ========================================
echo 🧠 AI神經網路學習環境一鍵安裝
echo ========================================
echo.

echo 📋 檢查系統需求...
python --version > nul 2>&1
if errorlevel 1 (
    echo ❌ 錯誤：找不到Python
    echo 請先從 https://www.python.org/downloads/ 下載並安裝Python
    echo 安裝時務必勾選 "Add Python to PATH"
    pause
    exit /b 1
)

echo ✅ Python已安裝
python --version

echo.
echo 🔄 建立虛擬環境...
if exist ai_env (
    echo ⚠️  虛擬環境已存在，跳過建立
) else (
    python -m venv ai_env
    echo ✅ 虛擬環境建立完成
)

echo.
echo 🔄 啟動虛擬環境...
call ai_env\Scripts\activate.bat

echo.
echo 🔄 更新pip...
python -m pip install --upgrade pip

echo.
echo 📦 安裝AI學習套件...
echo 這可能需要幾分鐘時間，請耐心等待...
pip install -r requirements.txt

if errorlevel 1 (
    echo ⚠️  使用國內鏡像重試安裝...
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ -r requirements.txt
)

echo.
echo 🧪 測試安裝結果...
python environment_test.py

echo.
echo ========================================
echo 🎉 安裝完成！
echo ========================================
echo.
echo 📚 接下來你可以：
echo 1. python first_neural_network.py     (訓練第一個神經網路)
echo 2. python cat_dog_classifier.py       (貓狗圖像分類)
echo 3. jupyter notebook                    (開啟Jupyter筆記本)
echo.
echo 💡 重要提醒：
echo - 每次使用前請先執行: ai_env\Scripts\activate
echo - 遇到問題請查看「神經網路入門教程.md」
echo - 或執行 environment_test.py 檢查環境
echo.
pause