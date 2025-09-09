"""
實戰項目：貓狗圖像分類器
使用卷積神經網路(CNN)來區分貓和狗的圖片
這是一個經典的電腦視覺入門項目
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

# 設定中文字體
try:
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
except:
    print("注意：無法設定中文字體")

def prepare_data():
    """準備貓狗分類資料"""
    print("🔄 正在準備貓狗分類資料...")
    print("   我們將使用CIFAR-10資料集中的貓和狗圖片")
    
    # 載入CIFAR-10資料集
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    
    # CIFAR-10類別說明
    class_names = ['飛機', '汽車', '鳥', '貓', '鹿', '狗', '青蛙', '馬', '船', '卡車']
    
    print(f"✅ CIFAR-10資料載入完成")
    print(f"   原始訓練資料: {x_train.shape}")
    print(f"   類別: {class_names}")
    
    # 只保留貓(類別3)和狗(類別5)的資料
    cat_indices_train = np.where(y_train.flatten() == 3)[0]  # 貓
    dog_indices_train = np.where(y_train.flatten() == 5)[0]  # 狗
    
    cat_indices_test = np.where(y_test.flatten() == 3)[0]
    dog_indices_test = np.where(y_test.flatten() == 5)[0]
    
    # 合併貓狗索引
    catdog_indices_train = np.concatenate([cat_indices_train, dog_indices_train])
    catdog_indices_test = np.concatenate([cat_indices_test, dog_indices_test])
    
    # 提取貓狗資料
    x_train_catdog = x_train[catdog_indices_train]
    y_train_catdog = y_train[catdog_indices_train]
    x_test_catdog = x_test[catdog_indices_test]
    y_test_catdog = y_test[catdog_indices_test]
    
    # 重新標記：貓=0, 狗=1
    y_train_binary = (y_train_catdog.flatten() == 5).astype(int)  # 狗=1
    y_test_binary = (y_test_catdog.flatten() == 5).astype(int)
    
    # 資料正規化
    x_train_normalized = x_train_catdog.astype('float32') / 255.0
    x_test_normalized = x_test_catdog.astype('float32') / 255.0
    
    print(f"✅ 貓狗資料準備完成")
    print(f"   訓練資料: {x_train_normalized.shape}")
    print(f"   測試資料: {x_test_normalized.shape}")
    print(f"   貓的圖片數量 (訓練): {np.sum(y_train_binary == 0)}")
    print(f"   狗的圖片數量 (訓練): {np.sum(y_train_binary == 1)}")
    
    # 顯示範例圖片
    display_samples(x_train_normalized, y_train_binary)
    
    return (x_train_normalized, y_train_binary), (x_test_normalized, y_test_binary)

def display_samples(x_data, y_data):
    """顯示資料範例"""
    plt.figure(figsize=(12, 8))
    
    # 顯示貓的範例
    cat_indices = np.where(y_data == 0)[0][:8]
    dog_indices = np.where(y_data == 1)[0][:8]
    
    for i in range(8):
        # 貓的圖片
        plt.subplot(4, 4, i + 1)
        plt.imshow(x_data[cat_indices[i]])
        plt.title('貓', color='blue')
        plt.axis('off')
        
        # 狗的圖片
        plt.subplot(4, 4, i + 9)
        plt.imshow(x_data[dog_indices[i]])
        plt.title('狗', color='red')
        plt.axis('off')
    
    plt.suptitle('CIFAR-10 貓狗資料範例', fontsize=16)
    plt.tight_layout()
    plt.savefig('cat_dog_samples.png', dpi=150, bbox_inches='tight')
    plt.show()

def create_cnn_model():
    """建立卷積神經網路模型"""
    print("🔄 正在建立CNN模型...")
    
    model = keras.Sequential([
        # 第一個卷積層
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3), name='conv1'),
        keras.layers.MaxPooling2D((2, 2), name='pool1'),
        
        # 第二個卷積層
        keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv2'),
        keras.layers.MaxPooling2D((2, 2), name='pool2'),
        
        # 第三個卷積層
        keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv3'),
        
        # 展平層
        keras.layers.Flatten(name='flatten'),
        
        # 全連接層
        keras.layers.Dense(64, activation='relu', name='dense1'),
        keras.layers.Dropout(0.5, name='dropout'),
        
        # 輸出層（二元分類）
        keras.layers.Dense(1, activation='sigmoid', name='output')
    ])
    
    # 編譯模型
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',  # 二元分類損失
        metrics=['accuracy']
    )
    
    print("✅ CNN模型建立完成！")
    print("\n📊 模型架構:")
    model.summary()
    
    return model

def train_cnn_model(model, x_train, y_train):
    """訓練CNN模型"""
    print("\n🚀 開始訓練貓狗分類器...")
    print("CNN訓練通常需要較長時間，請耐心等待...")
    
    # 資料增強（可選）
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,      # 隨機旋轉
        width_shift_range=0.2,  # 水平移動
        height_shift_range=0.2, # 垂直移動
        horizontal_flip=True,   # 水平翻轉
        zoom_range=0.2,         # 隨機縮放
        fill_mode='nearest'
    )
    
    # 訓練參數
    EPOCHS = 15
    BATCH_SIZE = 32
    VALIDATION_SPLIT = 0.2
    
    # 回調函數
    callbacks = [
        keras.callbacks.EarlyStopping(
            patience=3, 
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            factor=0.5, 
            patience=2, 
            verbose=1
        )
    ]
    
    # 開始訓練
    history = model.fit(
        x_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks,
        verbose=1
    )
    
    print("✅ 模型訓練完成！")
    return history

def evaluate_cnn_model(model, x_test, y_test):
    """評估CNN模型"""
    print("\n📈 正在評估貓狗分類器性能...")
    
    # 評估模型
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    
    print(f"✅ 評估完成！")
    print(f"   測試損失: {test_loss:.4f}")
    print(f"   測試準確率: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # 進行預測
    predictions = model.predict(x_test, verbose=0)
    predicted_classes = (predictions > 0.5).astype(int).flatten()
    
    # 計算混淆矩陣
    from sklearn.metrics import confusion_matrix, classification_report
    
    cm = confusion_matrix(y_test, predicted_classes)
    
    print(f"\n🔍 詳細分析:")
    print(f"   正確分類的貓: {cm[0,0]}")
    print(f"   誤分為狗的貓: {cm[0,1]}")
    print(f"   誤分為貓的狗: {cm[1,0]}")
    print(f"   正確分類的狗: {cm[1,1]}")
    
    # 顯示預測範例
    display_predictions(x_test, y_test, predictions, predicted_classes)
    
    return test_accuracy, predictions, predicted_classes, cm

def display_predictions(x_test, y_test, predictions, predicted_classes):
    """顯示預測結果"""
    plt.figure(figsize=(15, 10))
    
    # 顯示一些正確和錯誤的預測
    correct_indices = np.where(predicted_classes == y_test)[0]
    incorrect_indices = np.where(predicted_classes != y_test)[0]
    
    # 正確預測範例
    for i in range(8):
        if i < len(correct_indices):
            idx = correct_indices[i]
            plt.subplot(4, 4, i + 1)
            plt.imshow(x_test[idx])
            actual = '貓' if y_test[idx] == 0 else '狗'
            confidence = predictions[idx][0] if y_test[idx] == 1 else 1 - predictions[idx][0]
            plt.title(f'✅ {actual}\n信心度: {confidence:.2f}', color='green')
            plt.axis('off')
    
    # 錯誤預測範例
    for i in range(8):
        if i < len(incorrect_indices):
            idx = incorrect_indices[i]
            plt.subplot(4, 4, i + 9)
            plt.imshow(x_test[idx])
            actual = '貓' if y_test[idx] == 0 else '狗'
            predicted = '貓' if predicted_classes[idx] == 0 else '狗'
            confidence = predictions[idx][0] if predicted_classes[idx] == 1 else 1 - predictions[idx][0]
            plt.title(f'❌ 實際:{actual} 預測:{predicted}\n信心度: {confidence:.2f}', color='red')
            plt.axis('off')
    
    plt.suptitle('貓狗分類預測結果 (上兩排:正確預測, 下兩排:錯誤預測)', fontsize=14)
    plt.tight_layout()
    plt.savefig('cat_dog_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()

def visualize_training_results(history, test_accuracy, cm):
    """視覺化訓練結果"""
    print("\n📊 正在生成結果圖表...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 訓練損失
    axes[0, 0].plot(history.history['loss'], label='訓練損失', linewidth=2)
    axes[0, 0].plot(history.history['val_loss'], label='驗證損失', linewidth=2)
    axes[0, 0].set_title('貓狗分類器 - 損失變化', fontweight='bold')
    axes[0, 0].set_xlabel('週期')
    axes[0, 0].set_ylabel('損失')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 訓練準確率
    axes[0, 1].plot(history.history['accuracy'], label='訓練準確率', linewidth=2)
    axes[0, 1].plot(history.history['val_accuracy'], label='驗證準確率', linewidth=2)
    axes[0, 1].set_title('貓狗分類器 - 準確率變化', fontweight='bold')
    axes[0, 1].set_xlabel('週期')
    axes[0, 1].set_ylabel('準確率')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 混淆矩陣
    im = axes[1, 0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[1, 0].set_title('混淆矩陣', fontweight='bold')
    tick_marks = np.arange(2)
    axes[1, 0].set_xticks(tick_marks)
    axes[1, 0].set_yticks(tick_marks)
    axes[1, 0].set_xticklabels(['貓', '狗'])
    axes[1, 0].set_yticklabels(['貓', '狗'])
    axes[1, 0].set_xlabel('預測標籤')
    axes[1, 0].set_ylabel('真實標籤')
    
    # 在混淆矩陣中顯示數值
    for i in range(2):
        for j in range(2):
            axes[1, 0].text(j, i, str(cm[i, j]), ha="center", va="center", color="red", fontweight='bold')
    
    # 4. 總結信息
    axes[1, 1].text(0.1, 0.8, f'最終測試準確率: {test_accuracy:.4f}', fontsize=16, fontweight='bold')
    axes[1, 1].text(0.1, 0.6, f'準確率百分比: {test_accuracy*100:.2f}%', fontsize=16)
    axes[1, 1].text(0.1, 0.4, f'訓練週期: {len(history.history["loss"])}', fontsize=14)
    axes[1, 1].text(0.1, 0.2, '模型類型: 卷積神經網路 (CNN)', fontsize=14)
    axes[1, 1].set_title('訓練總結', fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('cat_dog_training_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ 圖表已保存")

def main():
    """主函數"""
    print("=" * 60)
    print("🐱🐶 貓狗圖像分類器 - 卷積神經網路實戰")
    print("=" * 60)
    print("這個項目將教你使用CNN來區分貓和狗的圖片！")
    print()
    
    try:
        # 設定隨機種子
        tf.random.set_seed(42)
        np.random.seed(42)
        
        # 檢查環境
        print(f"🔧 TensorFlow 版本: {tf.__version__}")
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"🚀 發現GPU: {len(gpus)}個設備")
        else:
            print("💻 使用CPU運算（CNN訓練可能較慢）")
        print()
        
        # 步驟1: 準備資料
        (x_train, y_train), (x_test, y_test) = prepare_data()
        
        # 步驟2: 建立CNN模型
        model = create_cnn_model()
        
        # 步驟3: 訓練模型
        history = train_cnn_model(model, x_train, y_train)
        
        # 步驟4: 評估模型
        test_accuracy, predictions, predicted_classes, cm = evaluate_cnn_model(model, x_test, y_test)
        
        # 步驟5: 視覺化結果
        visualize_training_results(history, test_accuracy, cm)
        
        # 步驟6: 保存模型
        model.save('cat_dog_classifier.h5')
        print("✅ 模型已保存為 'cat_dog_classifier.h5'")
        
        print("\n" + "=" * 60)
        print("🎉 恭喜！你已經成功訓練了貓狗分類器！")
        print("=" * 60)
        print("📊 性能總結:")
        print(f"   最終準確率: {test_accuracy*100:.2f}%")
        print(f"   正確分類的貓: {cm[0,0]}")
        print(f"   正確分類的狗: {cm[1,1]}")
        print()
        print("📚 學習要點:")
        print("1. CNN比普通神經網路更適合圖像處理")
        print("2. 卷積層能提取圖像特徵")
        print("3. 池化層能減少參數數量")
        print("4. Dropout能防止過度擬合")
        print()
        print("🔗 生成的檔案:")
        print("- cat_dog_samples.png: 資料範例")
        print("- cat_dog_predictions.png: 預測結果")
        print("- cat_dog_training_results.png: 訓練過程")
        print("- cat_dog_classifier.h5: 訓練好的模型")
        
    except Exception as e:
        print(f"❌ 程式執行時發生錯誤: {str(e)}")
        print("\n🔧 可能的解決方法：")
        print("1. 確保記憶體足夠（建議8GB以上）")
        print("2. 如果CPU運算太慢，考慮減少EPOCHS")
        print("3. 檢查網路連線（需要下載資料集）")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()