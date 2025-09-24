"""
AlexNet 推理腳本
加載訓練好的模型，對單張圖片或測試集進行預測
顯示預測結果和準確率分析
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import json

# 導入自定義模組
from alexnet_model import alexnet
from data_loader import get_cifar10_dataloaders, CIFAR10_CLASSES, CIFAR10_CLASSES_EN
from utils import get_device, calculate_accuracy


def load_trained_model(model_path, num_classes=10, device=None):
    """
    加載訓練好的模型
    
    Args:
        model_path (str): 模型文件路徑
        num_classes (int): 分類類別數
        device: 計算設備
        
    Returns:
        nn.Module: 加載的模型
    """
    if device is None:
        device = get_device()
    
    # 創建模型
    model = alexnet(num_classes=num_classes)
    
    # 加載檢查點
    if os.path.exists(model_path):
        print(f"正在加載模型: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        # 根據檢查點格式加載權重
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"模型信息:")
            print(f"  - 訓練輪數: {checkpoint.get('epoch', 'Unknown')}")
            print(f"  - 驗證準確率: {checkpoint.get('accuracy', 'Unknown'):.4f}")
            print(f"  - 驗證損失: {checkpoint.get('loss', 'Unknown'):.4f}")
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(device)
        model.eval()
        print("模型加載成功！")
    else:
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    return model


def preprocess_image(image_path):
    """
    預處理單張圖片
    
    Args:
        image_path (str): 圖片路徑
        
    Returns:
        torch.Tensor: 預處理後的圖片張量
    """
    # 定義預處理變換（與訓練時保持一致）
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # 調整大小到32x32
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # 加載和預處理圖片
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # 添加批次維度
    
    return image_tensor, image


def predict_single_image(model, image_tensor, device):
    """
    對單張圖片進行預測
    
    Args:
        model (nn.Module): 訓練好的模型
        image_tensor (torch.Tensor): 預處理後的圖片張量
        device: 計算設備
        
    Returns:
        tuple: (predicted_class, confidence, all_probabilities)
    """
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        # 前向傳播
        outputs = model(image_tensor)
        
        # 計算概率
        probabilities = F.softmax(outputs, dim=1)
        
        # 獲取預測結果
        confidence, predicted = torch.max(probabilities, 1)
        predicted_class = predicted.item()
        confidence = confidence.item()
    
    return predicted_class, confidence, probabilities.cpu().numpy()[0]


def visualize_prediction(image, predicted_class, confidence, probabilities, save_path=None):
    """
    可視化預測結果
    
    Args:
        image (PIL.Image): 原始圖片
        predicted_class (int): 預測類別
        confidence (float): 預測信心度
        probabilities (np.array): 所有類別的概率
        save_path (str, optional): 保存路徑
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 顯示原始圖片
    ax1.imshow(image)
    ax1.set_title(f'預測結果: {CIFAR10_CLASSES[predicted_class]}\n信心度: {confidence:.4f}', 
                 fontsize=14)
    ax1.axis('off')
    
    # 顯示所有類別的概率分布
    classes = CIFAR10_CLASSES
    y_pos = np.arange(len(classes))
    
    bars = ax2.barh(y_pos, probabilities)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(classes)
    ax2.set_xlabel('概率')
    ax2.set_title('各類別預測概率')
    ax2.grid(axis='x', alpha=0.3)
    
    # 高亮最高概率的類別
    bars[predicted_class].set_color('red')
    
    # 添加數值標籤
    for i, prob in enumerate(probabilities):
        ax2.text(prob + 0.01, i, f'{prob:.3f}', va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"預測結果圖片已保存至: {save_path}")
    
    plt.show()


def evaluate_test_set(model, test_loader, device, num_samples=None):
    """
    在測試集上評估模型性能
    
    Args:
        model (nn.Module): 訓練好的模型
        test_loader (DataLoader): 測試數據加載器
        device: 計算設備
        num_samples (int, optional): 評估樣本數量限制
        
    Returns:
        dict: 評估結果字典
    """
    model.eval()
    
    total_correct = 0
    total_samples = 0
    class_correct = [0] * 10
    class_total = [0] * 10
    all_predictions = []
    all_labels = []
    
    print("正在評估測試集...")
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            if num_samples and total_samples >= num_samples:
                break
                
            data, target = data.to(device), target.to(device)
            
            # 前向傳播
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            
            # 統計準確率
            total_samples += target.size(0)
            correct = (predicted == target)
            total_correct += correct.sum().item()
            
            # 統計每個類別的準確率
            for i in range(target.size(0)):
                label = target[i].item()
                class_correct[label] += correct[i].item()
                class_total[label] += 1
            
            # 保存預測結果
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    
    # 計算總體準確率
    overall_accuracy = total_correct / total_samples
    
    # 計算每個類別的準確率
    class_accuracies = []
    for i in range(10):
        if class_total[i] > 0:
            acc = class_correct[i] / class_total[i]
            class_accuracies.append(acc)
        else:
            class_accuracies.append(0.0)
    
    # 創建結果字典
    results = {
        'overall_accuracy': overall_accuracy,
        'total_samples': total_samples,
        'class_accuracies': class_accuracies,
        'predictions': all_predictions,
        'labels': all_labels
    }
    
    return results


def print_evaluation_results(results):
    """
    打印評估結果
    
    Args:
        results (dict): 評估結果字典
    """
    print(f"\n=== 測試集評估結果 ===")
    print(f"總樣本數: {results['total_samples']}")
    print(f"總體準確率: {results['overall_accuracy']:.4f}")
    
    print(f"\n各類別準確率:")
    for i, acc in enumerate(results['class_accuracies']):
        print(f"  {CIFAR10_CLASSES[i]:>6}: {acc:.4f}")
    
    # 找出表現最好和最差的類別
    best_class = np.argmax(results['class_accuracies'])
    worst_class = np.argmin(results['class_accuracies'])
    
    print(f"\n表現最好的類別: {CIFAR10_CLASSES[best_class]} ({results['class_accuracies'][best_class]:.4f})")
    print(f"表現最差的類別: {CIFAR10_CLASSES[worst_class]} ({results['class_accuracies'][worst_class]:.4f})")


def plot_confusion_matrix(results, save_path=None):
    """
    繪製混淆矩陣
    
    Args:
        results (dict): 評估結果字典
        save_path (str, optional): 保存路徑
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    # 計算混淆矩陣
    cm = confusion_matrix(results['labels'], results['predictions'])
    
    # 繪製混淆矩陣
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CIFAR10_CLASSES,
                yticklabels=CIFAR10_CLASSES)
    plt.title('混淆矩陣')
    plt.xlabel('預測標籤')
    plt.ylabel('真實標籤')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩陣已保存至: {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='AlexNet 推理腳本')
    parser.add_argument('--model_path', type=str, required=True, help='訓練好的模型路徑')
    parser.add_argument('--image_path', type=str, help='要預測的單張圖片路徑')
    parser.add_argument('--evaluate_test', action='store_true', help='在測試集上評估模型')
    parser.add_argument('--batch_size', type=int, default=128, help='測試批次大小')
    parser.add_argument('--num_samples', type=int, help='限制評估樣本數量')
    parser.add_argument('--output_dir', type=str, default='./inference_results', help='輸出目錄')
    
    args = parser.parse_args()
    
    # 創建輸出目錄
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 獲取設備
    device = get_device()
    
    # 加載模型
    model = load_trained_model(args.model_path, device=device)
    
    if args.image_path:
        # 單張圖片預測
        print(f"\n正在預測圖片: {args.image_path}")
        
        try:
            # 預處理圖片
            image_tensor, original_image = preprocess_image(args.image_path)
            
            # 進行預測
            predicted_class, confidence, probabilities = predict_single_image(
                model, image_tensor, device
            )
            
            # 打印結果
            print(f"\n預測結果:")
            print(f"  類別: {CIFAR10_CLASSES[predicted_class]}")
            print(f"  信心度: {confidence:.4f}")
            
            # 可視化結果
            save_path = os.path.join(args.output_dir, 'prediction_result.png')
            visualize_prediction(original_image, predicted_class, confidence, 
                               probabilities, save_path)
            
        except Exception as e:
            print(f"圖片預測失敗: {e}")
    
    if args.evaluate_test:
        # 測試集評估
        print(f"\n正在加載測試數據...")
        _, test_loader, _, _ = get_cifar10_dataloaders(
            batch_size=args.batch_size,
            download=False
        )
        
        # 評估模型
        results = evaluate_test_set(model, test_loader, device, args.num_samples)
        
        # 打印結果
        print_evaluation_results(results)
        
        # 保存結果
        results_path = os.path.join(args.output_dir, 'evaluation_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            # 將numpy數組轉換為列表以便JSON序列化
            json_results = {
                'overall_accuracy': results['overall_accuracy'],
                'total_samples': results['total_samples'],
                'class_accuracies': results['class_accuracies']
            }
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n評估結果已保存至: {results_path}")
        
        # 繪製混淆矩陣（如果安裝了sklearn和seaborn）
        try:
            cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
            plot_confusion_matrix(results, cm_path)
        except ImportError:
            print("sklearn或seaborn未安裝，跳過混淆矩陣繪製")
    
    if not args.image_path and not args.evaluate_test:
        print("請指定 --image_path 進行單張圖片預測，或使用 --evaluate_test 在測試集上評估")


if __name__ == "__main__":
    main()