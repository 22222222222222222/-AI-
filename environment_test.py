"""
神經網路環境測試腳本
運行此腳本來檢查你的Python環境是否正確設置
"""

import sys
import subprocess

def check_python_version():
    """檢查Python版本"""
    version = sys.version_info
    print(f"Python 版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 8:
        print("✅ Python版本符合要求")
        return True
    else:
        print("❌ Python版本過舊，建議使用Python 3.8或更新版本")
        return False

def check_package(package_name, import_name=None):
    """檢查套件是否安裝"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✅ {package_name} 已安裝")
        return True
    except ImportError:
        print(f"❌ {package_name} 未安裝")
        return False

def install_missing_packages():
    """安裝缺失的套件"""
    packages = [
        'numpy',
        'pandas', 
        'matplotlib',
        'scikit-learn',
        'tensorflow',
        'jupyter'
    ]
    
    missing_packages = []
    
    for package in packages:
        if not check_package(package):
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n發現缺失套件: {', '.join(missing_packages)}")
        install = input("是否要自動安裝這些套件？(y/n): ")
        
        if install.lower() in ['y', 'yes', '是']:
            for package in missing_packages:
                print(f"正在安裝 {package}...")
                try:
                    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                    print(f"✅ {package} 安裝成功")
                except subprocess.CalledProcessError:
                    print(f"❌ {package} 安裝失敗")
        else:
            print("請手動安裝缺失的套件：")
            for package in missing_packages:
                print(f"pip install {package}")

def test_tensorflow():
    """測試TensorFlow功能"""
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow 版本: {tf.__version__}")
        
        # 檢查GPU支援
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ 發現 {len(gpus)} 個GPU設備")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
        else:
            print("ℹ️  未發現GPU設備，將使用CPU運算")
        
        # 簡單測試
        hello = tf.constant('Hello, TensorFlow!')
        print(f"✅ TensorFlow測試通過: {hello.numpy().decode()}")
        return True
        
    except Exception as e:
        print(f"❌ TensorFlow測試失敗: {str(e)}")
        return False

def test_basic_ml():
    """測試基礎機器學習功能"""
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        
        print("正在進行基礎機器學習測試...")
        
        # 創建測試資料
        X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 訓練模型
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
        
        print(f"✅ 基礎機器學習測試通過，準確率: {accuracy:.2f}")
        return True
        
    except Exception as e:
        print(f"❌ 基礎機器學習測試失敗: {str(e)}")
        return False

def main():
    """主函數"""
    print("=" * 50)
    print("神經網路環境檢查工具")
    print("=" * 50)
    
    # 檢查Python版本
    python_ok = check_python_version()
    print()
    
    # 檢查必要套件
    print("檢查必要套件...")
    essential_packages = [
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('matplotlib', 'matplotlib'),
        ('scikit-learn', 'sklearn'),
        ('tensorflow', 'tensorflow'),
        ('jupyter', 'jupyter')
    ]
    
    packages_ok = True
    for package_name, import_name in essential_packages:
        if not check_package(package_name, import_name):
            packages_ok = False
    
    print()
    
    if not packages_ok:
        install_missing_packages()
        print()
    
    # 測試TensorFlow
    print("測試TensorFlow...")
    tf_ok = test_tensorflow()
    print()
    
    # 測試基礎機器學習
    print("測試基礎機器學習...")
    ml_ok = test_basic_ml()
    print()
    
    # 總結
    print("=" * 50)
    print("環境檢查結果總結")
    print("=" * 50)
    
    if python_ok and packages_ok and tf_ok and ml_ok:
        print("🎉 恭喜！你的環境已經完全設置好了！")
        print("現在你可以開始學習神經網路了。")
        print()
        print("建議下一步：")
        print("1. 執行 first_neural_network.py")
        print("2. 開啟 Jupyter Notebook: jupyter notebook")
        print("3. 閱讀 神經網路入門教程.md")
    else:
        print("⚠️  環境設置還未完成，請解決上述問題後重新執行此腳本。")
        print()
        print("常見問題解決方法：")
        print("1. 確保使用了虛擬環境")
        print("2. 更新pip: python -m pip install --upgrade pip")
        print("3. 重新安裝問題套件: pip uninstall [套件名] && pip install [套件名]")
    
    print("\n如果仍有問題，請查看教程中的「常見問題解決」章節。")

if __name__ == "__main__":
    main()