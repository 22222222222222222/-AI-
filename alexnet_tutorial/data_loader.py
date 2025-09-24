"""
CIFAR-10 數據集加載器
包含數據加載、預處理和數據增強功能
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


# CIFAR-10 類別名稱（中文）
CIFAR10_CLASSES = [
    '飛機', '汽車', '鳥類', '貓', '鹿',
    '狗', '蛙', '馬', '船', '卡車'
]

# CIFAR-10 類別名稱（英文）
CIFAR10_CLASSES_EN = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def get_cifar10_transforms(is_training=True):
    """
    獲取CIFAR-10數據集的預處理變換
    
    Args:
        is_training (bool): 是否為訓練模式
        
    Returns:
        torchvision.transforms.Compose: 數據變換組合
    """
    if is_training:
        # 訓練時的數據增強
        transform = transforms.Compose([
            # 隨機水平翻轉，增加數據多樣性
            transforms.RandomHorizontalFlip(p=0.5),
            
            # 隨機裁剪並填充，模擬平移變換
            transforms.RandomCrop(32, padding=4),
            
            # 隨機旋轉，增強模型的旋轉不變性
            transforms.RandomRotation(degrees=10),
            
            # 色彩抖動，增強顏色變化的魯棒性
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            
            # 轉換為張量
            transforms.ToTensor(),
            
            # 標準化：使用CIFAR-10的均值和標準差
            # 計算自全數據集的統計信息
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # RGB三通道的均值
                std=[0.229, 0.224, 0.225]    # RGB三通道的標準差
            )
        ])
    else:
        # 測試時只進行基本的預處理
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    return transform


def get_cifar10_dataloaders(batch_size=128, num_workers=4, download=True):
    """
    獲取CIFAR-10數據集的DataLoader
    
    Args:
        batch_size (int): 批次大小
        num_workers (int): 數據加載進程數
        download (bool): 是否下載數據集
        
    Returns:
        tuple: (train_loader, test_loader, train_dataset, test_dataset)
    """
    
    # 獲取數據變換
    train_transform = get_cifar10_transforms(is_training=True)
    test_transform = get_cifar10_transforms(is_training=False)
    
    # 創建訓練數據集
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=download,
        transform=train_transform
    )
    
    # 創建測試數據集
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=download,
        transform=test_transform
    )
    
    # 創建數據加載器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,          # 訓練時打亂數據
        num_workers=num_workers,
        pin_memory=True        # 將數據固定在內存中，加速GPU傳輸
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,         # 測試時不打亂數據
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"訓練集大小: {len(train_dataset)}")
    print(f"測試集大小: {len(test_dataset)}")
    print(f"批次大小: {batch_size}")
    print(f"訓練批次數: {len(train_loader)}")
    print(f"測試批次數: {len(test_loader)}")
    
    return train_loader, test_loader, train_dataset, test_dataset


def imshow(img, title=None):
    """
    顯示圖片的工具函數
    
    Args:
        img (torch.Tensor): 圖片張量
        title (str): 圖片標題
    """
    # 反標準化
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    
    img = img * std[:, None, None] + mean[:, None, None]
    img = torch.clamp(img, 0, 1)
    
    # 轉換為numpy數組
    npimg = img.numpy()
    
    # 調整維度順序從CHW到HWC
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if title:
        plt.title(title)
    plt.axis('off')


def visualize_dataset_samples(dataloader, num_samples=8):
    """
    可視化數據集樣本
    
    Args:
        dataloader (DataLoader): 數據加載器
        num_samples (int): 要顯示的樣本數量
    """
    # 獲取一個批次的數據
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    
    # 創建子圖
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle('CIFAR-10 數據集樣本', fontsize=16)
    
    for i in range(min(num_samples, 8)):
        row = i // 4
        col = i % 4
        
        # 顯示圖片
        ax = axes[row, col]
        imshow(images[i])
        
        # 設置標題
        label_idx = labels[i].item()
        title = f'{CIFAR10_CLASSES[label_idx]}'
        ax.set_title(title, fontsize=12)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def calculate_dataset_statistics(dataset):
    """
    計算數據集的統計信息（均值和標準差）
    
    Args:
        dataset: PyTorch數據集
        
    Returns:
        tuple: (mean, std) 每個通道的均值和標準差
    """
    # 創建數據加載器
    dataloader = DataLoader(dataset, batch_size=100, shuffle=False)
    
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = 0
    
    print("計算數據集統計信息...")
    
    for data, _ in dataloader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        total_samples += batch_samples
    
    mean /= total_samples
    std /= total_samples
    
    print(f"均值: {mean}")
    print(f"標準差: {std}")
    
    return mean, std


def get_class_distribution(dataset):
    """
    獲取數據集的類別分布
    
    Args:
        dataset: PyTorch數據集
        
    Returns:
        dict: 類別分布字典
    """
    class_counts = {}
    
    for _, label in dataset:
        label = label.item() if hasattr(label, 'item') else label
        class_counts[label] = class_counts.get(label, 0) + 1
    
    # 打印類別分布
    print("\n=== 類別分布 ===")
    for class_idx, count in sorted(class_counts.items()):
        class_name = CIFAR10_CLASSES[class_idx]
        percentage = count / len(dataset) * 100
        print(f"{class_name}: {count} 張 ({percentage:.1f}%)")
    
    return class_counts


if __name__ == "__main__":
    # 測試數據加載器
    print("=== CIFAR-10 數據加載器測試 ===")
    
    # 獲取數據加載器
    train_loader, test_loader, train_dataset, test_dataset = get_cifar10_dataloaders(
        batch_size=32, 
        num_workers=2,
        download=True
    )
    
    # 顯示類別分布
    print("\n訓練集類別分布:")
    get_class_distribution(train_dataset)
    
    print("\n測試集類別分布:")
    get_class_distribution(test_dataset)
    
    # 測試一個批次的數據
    print("\n=== 數據批次測試 ===")
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    
    print(f"圖片批次形狀: {images.shape}")
    print(f"標籤批次形狀: {labels.shape}")
    print(f"圖片數據類型: {images.dtype}")
    print(f"圖片數值範圍: [{images.min():.3f}, {images.max():.3f}]")
    
    # 顯示一些樣本（如果在支持matplotlib的環境中）
    try:
        import matplotlib.pyplot as plt
        print("\n顯示數據集樣本...")
        visualize_dataset_samples(train_loader)
    except ImportError:
        print("matplotlib未安裝，跳過可視化")
    
    print("\n數據加載器測試完成！")