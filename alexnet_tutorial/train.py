"""
AlexNet 訓練腳本
完整的訓練循環，支持CIFAR-10數據集訓練
包含損失函數、優化器設置、訓練過程可視化和模型保存
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import time
from tqdm import tqdm

# 導入自定義模組
from alexnet_model import alexnet
from data_loader import get_cifar10_dataloaders, CIFAR10_CLASSES
from utils import (
    save_model, load_model, calculate_accuracy, plot_training_history,
    save_training_log, get_device, count_parameters, print_model_summary,
    AverageMeter, EarlyStopping, set_seed
)


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    訓練一個epoch
    
    Args:
        model (nn.Module): 模型
        train_loader (DataLoader): 訓練數據加載器
        criterion: 損失函數
        optimizer: 優化器
        device: 計算設備
        epoch (int): 當前epoch
        
    Returns:
        tuple: (平均損失, 平均準確率)
    """
    model.train()  # 設置為訓練模式
    
    # 統計指標
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    # 進度條
    pbar = tqdm(train_loader, desc=f'訓練 Epoch {epoch}')
    
    for batch_idx, (data, target) in enumerate(pbar):
        # 將數據移動到指定設備
        data, target = data.to(device), target.to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向傳播
        output = model(data)
        
        # 計算損失
        loss = criterion(output, target)
        
        # 反向傳播
        loss.backward()
        
        # 更新參數
        optimizer.step()
        
        # 計算準確率
        accuracy = calculate_accuracy(output, target)
        
        # 更新統計指標
        loss_meter.update(loss.item(), data.size(0))
        acc_meter.update(accuracy, data.size(0))
        
        # 更新進度條
        pbar.set_postfix({
            'Loss': f'{loss_meter.avg:.4f}',
            'Acc': f'{acc_meter.avg:.4f}'
        })
    
    return loss_meter.avg, acc_meter.avg


def validate(model, val_loader, criterion, device):
    """
    驗證模型
    
    Args:
        model (nn.Module): 模型
        val_loader (DataLoader): 驗證數據加載器
        criterion: 損失函數
        device: 計算設備
        
    Returns:
        tuple: (平均損失, 平均準確率)
    """
    model.eval()  # 設置為評估模式
    
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    with torch.no_grad():  # 關閉梯度計算
        pbar = tqdm(val_loader, desc='驗證')
        
        for data, target in pbar:
            # 將數據移動到指定設備
            data, target = data.to(device), target.to(device)
            
            # 前向傳播
            output = model(data)
            
            # 計算損失
            loss = criterion(output, target)
            
            # 計算準確率
            accuracy = calculate_accuracy(output, target)
            
            # 更新統計指標
            loss_meter.update(loss.item(), data.size(0))
            acc_meter.update(accuracy, data.size(0))
            
            # 更新進度條
            pbar.set_postfix({
                'Loss': f'{loss_meter.avg:.4f}',
                'Acc': f'{acc_meter.avg:.4f}'
            })
    
    return loss_meter.avg, acc_meter.avg


def main(args):
    """
    主訓練函數
    
    Args:
        args: 命令行參數
    """
    print("=== AlexNet CIFAR-10 訓練開始 ===")
    
    # 設置隨機種子
    set_seed(args.seed)
    
    # 獲取計算設備
    device = get_device()
    
    # 創建輸出目錄
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)
    
    # 創建TensorBoard writer
    writer = SummaryWriter(os.path.join(args.output_dir, 'logs'))
    
    # 獲取數據加載器
    print("\n正在加載數據集...")
    train_loader, test_loader, train_dataset, test_dataset = get_cifar10_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=True
    )
    
    # 創建模型
    print("\n正在創建模型...")
    model = alexnet(num_classes=10)
    model = model.to(device)
    
    # 打印模型摘要
    print_model_summary(model)
    
    # 定義損失函數
    criterion = nn.CrossEntropyLoss()
    
    # 定義優化器
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # 學習率調度器
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_step_size,
        gamma=args.lr_gamma
    )
    
    # 早停機制
    early_stopping = EarlyStopping(
        patience=args.patience,
        min_delta=0.001,
        restore_best_weights=True
    )
    
    # 加載檢查點（如果存在）
    start_epoch = 0
    best_val_accuracy = 0.0
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    checkpoint_path = os.path.join(args.output_dir, 'checkpoints', 'latest.pth')
    if args.resume and os.path.exists(checkpoint_path):
        start_epoch, _, best_val_accuracy = load_model(model, optimizer, checkpoint_path, device)
        # 加載訓練歷史（如果存在）
        try:
            import json
            log_path = os.path.join(args.output_dir, 'logs', 'training_log.json')
            if os.path.exists(log_path):
                with open(log_path, 'r', encoding='utf-8') as f:
                    log_data = json.load(f)
                    train_losses = log_data.get('train_losses', [])
                    train_accuracies = log_data.get('train_accuracies', [])
                    val_losses = log_data.get('val_losses', [])
                    val_accuracies = log_data.get('val_accuracies', [])
        except Exception as e:
            print(f"無法加載訓練歷史: {e}")
    
    print(f"\n開始訓練，從第 {start_epoch + 1} 輪開始")
    print(f"訓練配置:")
    print(f"  - 學習率: {args.learning_rate}")
    print(f"  - 批次大小: {args.batch_size}")
    print(f"  - 訓練輪數: {args.epochs}")
    print(f"  - 優化器: SGD (momentum={args.momentum}, weight_decay={args.weight_decay})")
    print(f"  - 設備: {device}")
    
    # 訓練循環
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"當前學習率: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 記錄開始時間
        epoch_start_time = time.time()
        
        # 訓練一個epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch + 1
        )
        
        # 驗證
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        
        # 更新學習率
        scheduler.step()
        
        # 記錄指標
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # 記錄到TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # 計算epoch時間
        epoch_time = time.time() - epoch_start_time
        
        # 打印訓練結果
        print(f"\n訓練結果:")
        print(f"  - 訓練損失: {train_loss:.4f}, 訓練準確率: {train_acc:.4f}")
        print(f"  - 驗證損失: {val_loss:.4f}, 驗證準確率: {val_acc:.4f}")
        print(f"  - 耗時: {epoch_time:.2f}秒")
        
        # 保存最佳模型
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_model_path = os.path.join(args.output_dir, 'checkpoints', 'best_model.pth')
            save_model(model, optimizer, epoch, val_loss, val_acc, best_model_path)
            print(f"  - 新的最佳驗證準確率: {best_val_accuracy:.4f}")
        
        # 保存最新檢查點
        save_model(model, optimizer, epoch, val_loss, val_acc, checkpoint_path)
        
        # 早停檢查
        if early_stopping(val_loss, model):
            print(f"\n早停觸發！在第 {epoch + 1} 輪停止訓練")
            break
    
    # 訓練完成
    print(f"\n{'='*50}")
    print("訓練完成！")
    print(f"最佳驗證準確率: {best_val_accuracy:.4f}")
    
    # 保存訓練日誌
    config = {
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'lr_step_size': args.lr_step_size,
        'lr_gamma': args.lr_gamma,
        'seed': args.seed
    }
    
    log_path = os.path.join(args.output_dir, 'logs', 'training_log.json')
    save_training_log(train_losses, train_accuracies, val_losses, val_accuracies, config, log_path)
    
    # 繪製訓練曲線
    plot_path = os.path.join(args.output_dir, 'training_curves.png')
    plot_training_history(train_losses, train_accuracies, val_losses, val_accuracies, plot_path)
    
    # 關閉TensorBoard writer
    writer.close()
    
    print(f"\n所有輸出文件已保存至: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AlexNet CIFAR-10 訓練腳本')
    
    # 基本參數
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小 (default: 128)')
    parser.add_argument('--epochs', type=int, default=50, help='訓練輪數 (default: 50)')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='學習率 (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD動量 (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='權重衰減 (default: 5e-4)')
    
    # 學習率調度
    parser.add_argument('--lr_step_size', type=int, default=20, help='學習率衰減步長 (default: 20)')
    parser.add_argument('--lr_gamma', type=float, default=0.1, help='學習率衰減因子 (default: 0.1)')
    
    # 數據加載
    parser.add_argument('--num_workers', type=int, default=4, help='數據加載進程數 (default: 4)')
    
    # 輸出和日誌
    parser.add_argument('--output_dir', type=str, default='./outputs', help='輸出目錄 (default: ./outputs)')
    parser.add_argument('--resume', action='store_true', help='從檢查點恢復訓練')
    
    # 早停和其他
    parser.add_argument('--patience', type=int, default=10, help='早停耐心值 (default: 10)')
    parser.add_argument('--seed', type=int, default=42, help='隨機種子 (default: 42)')
    
    args = parser.parse_args()
    
    # 開始訓練
    main(args)