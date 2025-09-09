@echo off
chcp 65001 > nul
echo 🧠 啟動AI學習環境...

if not exist ai_env (
    echo ❌ 找不到虛擬環境，請先執行 install.bat
    pause
    exit /b 1
)

call ai_env\Scripts\activate.bat

echo ✅ AI學習環境已啟動！
echo.
echo 📚 可用的學習程式：
echo.
echo 🔬 環境測試：
echo    python environment_test.py
echo.
echo 🎯 初學者項目：
echo    python first_neural_network.py
echo.
echo 🖼️  進階項目：
echo    python cat_dog_classifier.py
echo.
echo 📓 開發環境：
echo    jupyter notebook
echo.
echo 📖 說明文檔：
echo    神經網路入門教程.md
echo    Windows快速安裝指南.md
echo.
echo 💡 提示：輸入指令後按Enter執行
echo ❌ 退出：輸入 exit 或直接關閉視窗
echo.

cmd /k