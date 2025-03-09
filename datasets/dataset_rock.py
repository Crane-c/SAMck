# datasets/dataset_rock.py
import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from einops import repeat
from icecream import ic

def random_rot_flip(image, label):
    '''
    随机执行90度倍数的旋转和镜像翻转组合增强
    '''
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    # 仅允许水平翻转，禁止垂直翻转
    axis = 1
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

class RandomGenerator(object):
    def __init__(self, output_size, low_res):
        self.output_size = output_size
        self.low_res = low_res

    def __call__(self, sample):
        try:
            image, label = sample['image'], sample['label'] # image:(1024, 1024, 3) label:(1024, 1024)

            # ==================== 维度安全检查 ====================
            if image.shape[:2] != label.shape:
                raise ValueError(f"图像和标签尺寸不匹配: image {image.shape}, label {label.shape}")

            # 随机增强
            if random.random() > 0.5:
                image, label = random_rot_flip(image, label)
            elif random.random() > 0.5:
                image, label = random_rotate(image, label)
        
            image = image.transpose(2, 0, 1) 
            c, h, w = image.shape

        
            # 尺寸调整（兼容多通道）
            if (h, w) != self.output_size:
                img_zoom = (1, self.output_size[0]/h, self.output_size[1]/w)

                image = zoom(image, img_zoom, order=1)  
                label = zoom(label, (self.output_size[0]/h, self.output_size[1]/w), order=0)


            # ====================  生成低分辨率标签 ====================
            label_h, label_w = label.shape
            low_res_label = zoom(label, (self.low_res[0]/label_h, self.low_res[1]/label_w), order=0)

            image = torch.from_numpy(image.astype(np.float32))  # 此时 image 范围仍为 0-255       
            label = torch.from_numpy(label.astype(np.int64))
            low_res_label = torch.from_numpy(low_res_label)
        
            sample = {
                'image': image,               
                'label': label,                
                'low_res_label': low_res_label,
                'case_id': sample.get('case_id', 'unknown')
            }

        except Exception as e:
            print(f"处理样本 {sample.get('case_id', 'unknown')} 失败: {e}")
            # 返回包含所有键的空样本
            return {
                'image': torch.Tensor(),
                'label': torch.Tensor(),
                'low_res_label': torch.Tensor(),
                'case_id': sample.get('case_id', 'unknown')
            }

        return sample
    
class Rock_dataset(Dataset):
    """加载岩土CT数据集（支持NPZ和HDF5格式）"""
    def __init__(self, base_dir, list_dir, split, transform=None):
        """
        Args:
            base_dir: 数据根目录
            list_dir: 包含 train.txt/val.txt 的目录
            split: 数据集类型（train/val/test）
            transform: 数据增强变换
        """
        self.transform = transform
        self.split = split
        self.data_dir = base_dir
        
        # 加载样本列表
        list_path = os.path.join(list_dir, f"{split}.txt")
        if not os.path.exists(list_path):
            raise FileNotFoundError(f"列表文件 {list_path} 不存在")
        
        with open(list_path, 'r') as f:
            self.sample_list = [line.strip() for line in f.readlines()]
        
    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case_id = self.sample_list[idx]
        sample = {'case_id': case_id}
        # case_name = self.sample_list[idx].strip('\n')

        try:
            # ==================== 数据加载 ====================
            if self.split == "train":
                # 加载NPZ训练数据 (H, W, 3)
                data_path = os.path.join(self.data_dir, f"{case_id}.npz")
                with np.load(data_path) as data:
                    image = data['image'].astype(np.float32)  # 确保float32
                    label = data['label'].astype(np.int64)    # 避免索引溢出
            else:
                # 加载HDF5验证/测试数据 (1, H, W, 3)
                data_path = os.path.join(self.data_dir, f"{case_id}.h5")
                with h5py.File(data_path, 'r') as f:
                    image = f['image'][:].squeeze(0).astype(np.float32)  # 去除批次维度
                    label = f['label'][:].squeeze(0).astype(np.int64)

        # Input dim should be consistent
        # Since the channel dimension of nature image is 3, that of medical image should also be 3
            # ==================== 维度统一 ====================
            # 确保图像为 (H, W, 3)，标签为 (H, W)
            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"图像 {case_id} 的维度错误: {image.shape}")
            if label.ndim != 2:
                raise ValueError(f"标签 {case_id} 的维度错误: {label.shape}")
            
            # ==================== 转换到PyTorch格式 ====================
            sample = {
                'image': image,  # (H, W, 3) → 后续transform转换为 (C, H, W)
                'label': label,
                'low_res_label': np.zeros_like(label),  # 临时占位
                'case_id': case_id
            }
            
            # ==================== 数据增强 ====================
            if self.transform:
                sample = self.transform(sample)
            
            # ==================== 最终维度检查 ====================
            if 'image' not in sample or 'label' not in sample:
                raise RuntimeError(f"数据增强后样本 {case_id} 缺少必要字段")
                
        except Exception as e:
            print(f"加载样本 {case_id} 失败: {e}")
            # 返回空样本并跳过（需在DataLoader中设置collate_fn处理）
            return {
            'image': torch.Tensor(),
            'label': torch.Tensor(),
            'low_res_label': torch.Tensor(),
            'case_id': case_id
        }
        
        return sample
    
        # sample = {'image': image, 'label': label}
        # if self.transform:
        #     sample = self.transform(sample)
        # sample['case_name'] = self.sample_list[idx].strip('\n')
        # return sample