import numpy as np
import cv2
import random
from typing import Dict
import torch

class RockAugmentation:
    """适用于岩土CT图像的自定义数据增强"""
    
    def __init__(self, output_size=(1024, 1024)):
        self.output_size = output_size
        
    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        输入: {'image': (H,W,3), 'mask': (H,W)}
        输出: 增强后的同结构字典
        """
        image, mask = sample['image'], sample['mask']
        
        # 随机水平/垂直翻转 (概率50%)
        if random.random() > 0.5:
            image = cv2.flip(image, 0)  # 垂直翻转
            mask = cv2.flip(mask, 0)
        if random.random() > 0.5:
            image = cv2.flip(image, 1)  # 水平翻转
            mask = cv2.flip(mask, 1)
            
        # 随机旋转 (-15° ~ 15°)
        angle = random.uniform(-15, 15)
        center = (image.shape[1]//2, image.shape[0]//2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, rot_mat, self.output_size, flags=cv2.INTER_LINEAR)
        mask = cv2.warpAffine(mask, rot_mat, self.output_size, flags=cv2.INTER_NEAREST)
        
        # 随机调整亮度/对比度 (适用于CT图像)
        alpha = random.uniform(0.8, 1.2)  # 对比度系数
        beta = random.uniform(-10, 10)     # 亮度增量
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        
        return {'image': image, 'mask': mask}

class ToTensor:
    """将numpy数组转换为PyTorch张量"""
    
    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        image, mask = sample['image'], sample['mask']
        
        # 调整维度顺序: (H,W,C) -> (C,H,W)
        image = torch.from_numpy(image.transpose((2, 0, 1))).float() / 255.0
        mask = torch.from_numpy(mask).long()
        
        return {'image': image, 'mask': mask}