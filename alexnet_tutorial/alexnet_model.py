"""
AlexNet 模型實現
完整實現了2012年ImageNet競賽獲勝的AlexNet架構
包含詳細的中文註解說明每個層的作用
"""

import torch
import torch.nn as nn


class AlexNet(nn.Module):
    """
    AlexNet 神經網路模型
    
    原始論文: "ImageNet Classification with Deep Convolutional Neural Networks"
    作者: Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton
    
    架構特點:
    - 8層網路：5個卷積層 + 3個全連接層
    - 使用ReLU激活函數（首次在CNN中大規模使用）
    - 使用Dropout防止過擬合
    - 使用局部響應歸一化（LRN）
    - 使用數據增強技術
    """
    
    def __init__(self, num_classes=10):
        """
        初始化AlexNet模型
        
        Args:
            num_classes (int): 分類類別數，CIFAR-10為10類
        """
        super(AlexNet, self).__init__()
        
        # 特徵提取部分（卷積層）
        self.features = nn.Sequential(
            # 第一個卷積層
            # 輸入: 3x32x32 (CIFAR-10), 輸出: 64x8x8
            # 原始AlexNet使用11x11卷積核，這裡調整為5x5適配CIFAR-10
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),  # ReLU激活函數，inplace=True節省內存
            nn.MaxPool2d(kernel_size=2, stride=2),  # 最大池化，降採樣
            
            # 第二個卷積層
            # 輸入: 64x16x16, 輸出: 192x8x8
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第三個卷積層
            # 輸入: 192x8x8, 輸出: 384x8x8
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            # 第四個卷積層
            # 輸入: 384x8x8, 輸出: 256x8x8
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            # 第五個卷積層
            # 輸入: 256x8x8, 輸出: 256x4x4
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # 自適應平均池化，確保輸出固定大小
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        
        # 分類器部分（全連接層）
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),  # Dropout層，防止過擬合，隨機將50%的神經元設為0
            
            # 第一個全連接層: 256*2*2 -> 1024
            nn.Linear(256 * 2 * 2, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            # 第二個全連接層: 1024 -> 512
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            
            # 輸出層: 512 -> num_classes
            nn.Linear(512, num_classes),
        )
        
        # 初始化權重
        self._initialize_weights()
    
    def forward(self, x):
        """
        前向傳播
        
        Args:
            x (torch.Tensor): 輸入張量，形狀為 (batch_size, 3, 32, 32)
            
        Returns:
            torch.Tensor: 輸出張量，形狀為 (batch_size, num_classes)
        """
        # 通過特徵提取層
        x = self.features(x)
        
        # 通過自適應池化層
        x = self.avgpool(x)
        
        # 展平張量，為全連接層做準備
        x = torch.flatten(x, 1)
        
        # 通過分類器
        x = self.classifier(x)
        
        return x
    
    def _initialize_weights(self):
        """
        權重初始化
        使用正態分布初始化卷積層和全連接層的權重
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 卷積層權重初始化
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 全連接層權重初始化
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)


def alexnet(num_classes=10, pretrained=False):
    """
    創建AlexNet模型的便捷函數
    
    Args:
        num_classes (int): 分類類別數
        pretrained (bool): 是否使用預訓練權重（暫不支持）
        
    Returns:
        AlexNet: AlexNet模型實例
    """
    model = AlexNet(num_classes=num_classes)
    
    if pretrained:
        # 注意：這裡可以加載預訓練權重
        # 由於是教學版本，暫時不實現
        print("警告：預訓練權重暫不支持")
    
    return model


if __name__ == "__main__":
    # 測試模型
    print("=== AlexNet 模型測試 ===")
    
    # 創建模型
    model = alexnet(num_classes=10)
    print(f"模型參數總數: {sum(p.numel() for p in model.parameters()):,}")
    
    # 創建測試輸入
    batch_size = 4
    test_input = torch.randn(batch_size, 3, 32, 32)
    print(f"輸入張量形狀: {test_input.shape}")
    
    # 前向傳播測試
    with torch.no_grad():
        output = model(test_input)
        print(f"輸出張量形狀: {output.shape}")
        print(f"輸出範例: {output[0]}")
    
    print("模型測試完成！")