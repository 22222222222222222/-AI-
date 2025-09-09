"""
第一個神經網路：手寫數字識別
這個程式會訓練一個神經網路來識別手寫數字（0-9）
適合完全初學者，包含詳細的中文註解
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

# 設定中文字體（如果系統支援）
try:
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
except:
    print("注意：無法設定中文字體，圖表可能顯示方框")

def load_and_prepare_data():
    """載入並準備MNIST資料集"""
    print("🔄 正在載入MNIST手寫數字資料集...")
    
    # 載入資料集（第一次執行會自動下載）
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    print(f"✅ 資料載入完成！")
    print(f"   訓練圖片數量: {x_train.shape[0]}")
    print(f"   測試圖片數量: {x_test.shape[0]}")
    print(f"   圖片尺寸: {x_train.shape[1]} x {x_train.shape[2]} 像素")
    
    # 顯示一些範例圖片
    plt.figure(figsize=(10, 4))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x_train[i], cmap='gray')
        plt.title(f'數字: {y_train[i]}')
        plt.axis('off')
    plt.suptitle('MNIST資料集範例')
    plt.tight_layout()
    plt.savefig('mnist_examples.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 資料正規化：將像素值從0-255縮放到0-1
    print("🔄 正在進行資料預處理...")
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    print("✅ 資料預處理完成！")
    return (x_train, y_train), (x_test, y_test)

def create_model():
    """建立神經網路模型"""
    print("🔄 正在建立神經網路模型...")
    
    model = keras.Sequential([
        # 輸入層：將28x28的圖片展平成784個數字
        keras.layers.Flatten(input_shape=(28, 28), name='flatten'),
        
        # 隱藏層1：128個神經元，使用ReLU啟動函數
        keras.layers.Dense(128, activation='relu', name='hidden1'),
        
        # Dropout層：隨機丟棄20%的連接，防止過度擬合
        keras.layers.Dropout(0.2, name='dropout'),
        
        # 輸出層：10個神經元（對應0-9），使用softmax得到機率分布
        keras.layers.Dense(10, activation='softmax', name='output')
    ])
    
    # 編譯模型：設定優化器、損失函數和評估指標
    model.compile(
        optimizer='adam',                    # Adam優化器（自適應學習率）
        loss='sparse_categorical_crossentropy', # 多分類交叉熵損失
        metrics=['accuracy']                 # 準確率作為評估指標
    )
    
    print("✅ 模型建立完成！")
    print("\n📊 模型架構:")
    model.summary()
    
    return model

def train_model(model, x_train, y_train):
    """訓練模型"""
    print("\n🚀 開始訓練模型...")
    print("這可能需要幾分鐘時間，請耐心等待...")
    
    # 設定訓練參數
    EPOCHS = 5          # 訓練週期數
    BATCH_SIZE = 32     # 批次大小
    VALIDATION_SPLIT = 0.2  # 驗證集比例
    
    # 開始訓練
    history = model.fit(
        x_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        verbose=1,  # 顯示詳細的訓練過程
        callbacks=[
            # 可以添加回調函數，如早停、模型檢查點等
        ]
    )
    
    print("✅ 模型訓練完成！")
    return history

def evaluate_model(model, x_test, y_test):
    """評估模型性能"""
    print("\n📈 正在評估模型性能...")
    
    # 在測試集上評估
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    
    print(f"✅ 評估完成！")
    print(f"   測試損失: {test_loss:.4f}")
    print(f"   測試準確率: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # 進行預測
    predictions = model.predict(x_test[:10], verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    actual_classes = y_test[:10]
    
    print(f"\n🔍 前10個預測結果:")
    correct_count = 0
    for i in range(10):
        is_correct = predicted_classes[i] == actual_classes[i]
        if is_correct:
            correct_count += 1
        status = "✅" if is_correct else "❌"
        confidence = predictions[i][predicted_classes[i]] * 100
        print(f"   {status} 實際: {actual_classes[i]}, 預測: {predicted_classes[i]} (信心度: {confidence:.1f}%)")
    
    print(f"\n前10個預測準確率: {correct_count}/10 ({correct_count*10}%)")
    
    return test_accuracy, predictions[:10], actual_classes, predicted_classes

def visualize_results(history, test_accuracy, x_test, y_test, predictions, actual_classes, predicted_classes):
    """視覺化結果"""
    print("\n📊 正在生成結果圖表...")
    
    # 創建圖表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 訓練過程 - 損失
    axes[0, 0].plot(history.history['loss'], label='訓練損失', linewidth=2)
    axes[0, 0].plot(history.history['val_loss'], label='驗證損失', linewidth=2)
    axes[0, 0].set_title('模型損失變化', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('週期')
    axes[0, 0].set_ylabel('損失')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 訓練過程 - 準確率
    axes[0, 1].plot(history.history['accuracy'], label='訓練準確率', linewidth=2)
    axes[0, 1].plot(history.history['val_accuracy'], label='驗證準確率', linewidth=2)
    axes[0, 1].set_title('模型準確率變化', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('週期')
    axes[0, 1].set_ylabel('準確率')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 預測結果範例
    for i in range(8):
        row = i // 4
        col = i % 4
        ax = plt.subplot(4, 4, 9 + i)
        plt.imshow(x_test[i], cmap='gray')
        
        is_correct = predicted_classes[i] == actual_classes[i]
        color = 'green' if is_correct else 'red'
        confidence = predictions[i][predicted_classes[i]] * 100
        
        plt.title(f'實際:{actual_classes[i]} 預測:{predicted_classes[i]}\n信心度:{confidence:.1f}%', 
                 color=color, fontsize=10)
        plt.axis('off')
    
    # 4. 總結信息
    axes[1, 1].text(0.1, 0.8, f'最終測試準確率: {test_accuracy:.4f}', fontsize=16, fontweight='bold')
    axes[1, 1].text(0.1, 0.6, f'準確率百分比: {test_accuracy*100:.2f}%', fontsize=16)
    axes[1, 1].text(0.1, 0.4, f'訓練週期: {len(history.history["loss"])}', fontsize=14)
    axes[1, 1].text(0.1, 0.2, f'模型參數數量: {sum([np.prod(v.get_shape()) for v in model.trainable_variables]):,}', fontsize=14)
    axes[1, 1].set_title('訓練總結', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('neural_network_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ 圖表已保存為 'neural_network_results.png'")

def save_model(model):
    """保存訓練好的模型"""
    model_path = 'my_first_neural_network.h5'
    model.save(model_path)
    print(f"✅ 模型已保存為 '{model_path}'")
    print("你可以稍後使用以下程式碼載入模型：")
    print("model = keras.models.load_model('my_first_neural_network.h5')")

def main():
    """主函數"""
    print("=" * 60)
    print("🧠 歡迎使用神經網路入門教程 - 手寫數字識別")
    print("=" * 60)
    print("這個程式將帶你完成第一個神經網路項目！")
    print()
    
    try:
        # 設定隨機種子，確保結果可重現
        tf.random.set_seed(42)
        np.random.seed(42)
        
        # 檢查TensorFlow版本
        print(f"🔧 TensorFlow 版本: {tf.__version__}")
        
        # 檢查是否有GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"🚀 發現GPU: {len(gpus)}個設備")
        else:
            print("💻 使用CPU運算")
        print()
        
        # 步驟1: 載入並準備資料
        (x_train, y_train), (x_test, y_test) = load_and_prepare_data()
        
        # 步驟2: 建立模型
        model = create_model()
        
        # 步驟3: 訓練模型
        history = train_model(model, x_train, y_train)
        
        # 步驟4: 評估模型
        test_accuracy, predictions, actual_classes, predicted_classes = evaluate_model(model, x_test, y_test)
        
        # 步驟5: 視覺化結果
        visualize_results(history, test_accuracy, x_test, y_test, predictions, actual_classes, predicted_classes)
        
        # 步驟6: 保存模型
        save_model(model)
        
        print("\n" + "=" * 60)
        print("🎉 恭喜！你已經成功訓練了第一個神經網路！")
        print("=" * 60)
        print("📚 接下來你可以：")
        print("1. 嘗試修改模型架構（增加層數、改變神經元數量）")
        print("2. 調整訓練參數（學習率、批次大小、訓練週期）")
        print("3. 學習其他類型的神經網路（CNN、RNN等）")
        print("4. 嘗試其他資料集（CIFAR-10、Fashion-MNIST等）")
        print()
        print("🔗 相關檔案：")
        print("- neural_network_results.png: 訓練結果圖表")
        print("- mnist_examples.png: 資料集範例")
        print("- my_first_neural_network.h5: 訓練好的模型")
        
    except Exception as e:
        print(f"❌ 程式執行時發生錯誤: {str(e)}")
        print("\n🔧 可能的解決方法：")
        print("1. 確保所有必要套件都已安裝")
        print("2. 執行 environment_test.py 檢查環境")
        print("3. 查看教程中的「常見問題解決」章節")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()