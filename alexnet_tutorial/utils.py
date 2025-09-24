"""
工具函數模組
包含模型保存/加載、訓練可視化、準確率計算等輔助函數
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import json


def save_model(model, optimizer, epoch, loss, accuracy, save_path):
    """
    保存模型檢查點
    
    Args:
        model (nn.Module): 要保存的模型
        optimizer (torch.optim.Optimizer): 優化器
        epoch (int): 當前訓練輪數
        loss (float): 當前損失值
        accuracy (float): 當前準確率
        save_path (str): 保存路徑
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # 確保保存目錄存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    torch.save(checkpoint, save_path)
    print(f"模型已保存至: {save_path}")


def load_model(model, optimizer, load_path, device):
    """
    加載模型檢查點
    
    Args:
        model (nn.Module): 要加載權重的模型
        optimizer (torch.optim.Optimizer): 優化器
        load_path (str): 檢查點文件路徑
        device (torch.device): 設備
        
    Returns:
        tuple: (start_epoch, best_loss, best_accuracy)
    """
    if os.path.exists(load_path):
        print(f"加載模型檢查點: {load_path}")
        checkpoint = torch.load(load_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['loss']
        best_accuracy = checkpoint['accuracy']
        
        print(f"檢查點信息:")
        print(f"  - 訓練輪數: {checkpoint['epoch']}")
        print(f"  - 損失: {best_loss:.4f}")
        print(f"  - 準確率: {best_accuracy:.4f}")
        print(f"  - 保存時間: {checkpoint.get('timestamp', 'Unknown')}")
        
        return start_epoch, best_loss, best_accuracy
    else:
        print(f"檢查點文件不存在: {load_path}")
        return 0, float('inf'), 0.0


def calculate_accuracy(outputs, labels):
    """
    計算準確率
    
    Args:
        outputs (torch.Tensor): 模型輸出，形狀為 (batch_size, num_classes)
        labels (torch.Tensor): 真實標籤，形狀為 (batch_size,)
        
    Returns:
        float: 準確率（0-1之間）
    """
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return correct / total


def plot_training_history(train_losses, train_accuracies, val_losses=None, val_accuracies=None, save_path=None):
    """
    繪製訓練歷史曲線
    
    Args:
        train_losses (list): 訓練損失列表
        train_accuracies (list): 訓練準確率列表
        val_losses (list, optional): 驗證損失列表
        val_accuracies (list, optional): 驗證準確率列表
        save_path (str, optional): 保存圖片的路徑
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 繪製損失曲線
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='訓練損失', linewidth=2)
    if val_losses:
        ax1.plot(epochs, val_losses, 'r-', label='驗證損失', linewidth=2)
    ax1.set_title('訓練過程 - 損失變化', fontsize=14)
    ax1.set_xlabel('訓練輪數', fontsize=12)
    ax1.set_ylabel('損失值', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 繪製準確率曲線
    ax2.plot(epochs, train_accuracies, 'b-', label='訓練準確率', linewidth=2)
    if val_accuracies:
        ax2.plot(epochs, val_accuracies, 'r-', label='驗證準確率', linewidth=2)
    ax2.set_title('訓練過程 - 準確率變化', fontsize=14)
    ax2.set_xlabel('訓練輪數', fontsize=12)
    ax2.set_ylabel('準確率', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"訓練曲線已保存至: {save_path}")
    
    plt.show()


def save_training_log(train_losses, train_accuracies, val_losses=None, val_accuracies=None, 
                     config=None, save_path="training_log.json"):
    """
    保存訓練日誌到JSON文件
    
    Args:
        train_losses (list): 訓練損失列表
        train_accuracies (list): 訓練準確率列表
        val_losses (list, optional): 驗證損失列表
        val_accuracies (list, optional): 驗證準確率列表
        config (dict, optional): 訓練配置字典
        save_path (str): 保存路徑
    """
    log_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'config': config,
        'epochs': len(train_losses),
        'best_train_accuracy': max(train_accuracies) if train_accuracies else 0,
        'best_val_accuracy': max(val_accuracies) if val_accuracies else 0
    }
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    print(f"訓練日誌已保存至: {save_path}")


def get_device():
    """
    獲取可用的計算設備
    
    Returns:
        torch.device: 可用的設備（cuda或cpu）
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU內存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device('cpu')
        print("使用CPU進行訓練")
    
    return device


def count_parameters(model):
    """
    計算模型參數總數
    
    Args:
        model (nn.Module): PyTorch模型
        
    Returns:
        tuple: (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"模型參數統計:")
    print(f"  - 總參數數: {total_params:,}")
    print(f"  - 可訓練參數數: {trainable_params:,}")
    
    return total_params, trainable_params


def print_model_summary(model, input_size=(3, 32, 32)):
    """
    打印模型摘要信息
    
    Args:
        model (nn.Module): PyTorch模型
        input_size (tuple): 輸入張量大小
    """
    print("=== 模型架構摘要 ===")
    print(model)
    print("\n=== 模型參數統計 ===")
    count_parameters(model)
    
    # 測試前向傳播
    try:
        dummy_input = torch.randn(1, *input_size)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"\n=== 輸入輸出形狀 ===")
        print(f"輸入形狀: {dummy_input.shape}")
        print(f"輸出形狀: {output.shape}")
    except Exception as e:
        print(f"前向傳播測試失敗: {e}")


class AverageMeter:
    """
    平均值計算器，用於統計訓練過程中的指標
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置統計"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        """
        更新統計
        
        Args:
            val: 新的值
            n: 樣本數量
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """
    早停機制，防止過擬合
    """
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        """
        Args:
            patience (int): 容忍的沒有改善的輪數
            min_delta (float): 認為是改善的最小變化
            restore_best_weights (bool): 是否恢復最佳權重
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss, model):
        """
        檢查是否應該早停
        
        Args:
            val_loss (float): 當前驗證損失
            model (nn.Module): 模型
            
        Returns:
            bool: 是否應該早停
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        """保存最佳模型權重"""
        self.best_weights = model.state_dict().copy()


def set_seed(seed=42):
    """
    設置隨機種子，確保實驗的可重現性
    
    Args:
        seed (int): 隨機種子
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # 確保CUDNN的確定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"隨機種子已設置為: {seed}")


if __name__ == "__main__":
    # 測試工具函數
    print("=== 工具函數模組測試 ===")
    
    # 測試設備檢測
    device = get_device()
    
    # 測試隨機種子設置
    set_seed(42)
    
    # 測試平均值計算器
    meter = AverageMeter()
    for i in range(10):
        meter.update(i, 1)
    print(f"\n平均值計算器測試: {meter.avg}")
    
    # 測試早停機制
    early_stopping = EarlyStopping(patience=3)
    print(f"早停機制初始化完成，容忍輪數: {early_stopping.patience}")
    
    print("\n工具函數模組測試完成！")