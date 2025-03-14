# preprocess_data.py
import os
import argparse
import numpy as np
import cv2
import tifffile
import h5py
from glob import glob
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ======================== 参数配置 ========================
parser = argparse.ArgumentParser()
parser.add_argument('--src_dir', type=str, 
                    default='./data/Rock/RawData',
                    help='原始数据目录，包含images和masks子目录')
parser.add_argument('--dst_dir', type=str,
                    default='./data/Rock/Processed',
                    help='预处理结果保存目录')
parser.add_argument('--img_size', type=int,
                    default=1024,
                    help='输出图像尺寸（正方形）')
parser.add_argument('--test_ratio', type=float,
                    default=0.2,
                    help='验证集比例 (0.0~1.0)')
parser.add_argument('--ct_min', type=float,
                    default=0,
                    help='CT值归一化下限(HU)')
parser.add_argument('--ct_max', type=float,
                    default=255,
                    help='CT值归一化上限(HU)')
parser.add_argument('--use_normalize', action='store_true',
                    help='是否进行CT值归一化')
args = parser.parse_args()

# ======================== 预处理函数 ========================
def process_slice(image_path, label_path):
    """处理单个岩土CT切片，适配SAM输入规范"""
    image = tifffile.imread(image_path)    # (H, W) 原始CT值
    mask = np.array(Image.open(label_path)) # (H, W) 原始标签

    # 调整尺寸（保持插值策略）
    image = cv2.resize(image, (args.img_size, args.img_size), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (args.img_size, args.img_size), interpolation=cv2.INTER_NEAREST)  # 保持标签的离散性

    # 标准化处理（适配CT值和SAM模型要求）
    if args.use_normalize:
        # 步骤1: CT值裁剪（防止异常值干扰）
        image = np.clip(image, args.ct_min, args.ct_max + 1e-8)  # 典型值：args.ct_min=-1000, args.ct_max=2000
        
        # 步骤2: 归一化到 [0, 1]
        image = (image - args.ct_min) / (args.ct_max - args.ct_min)
        
        # 步骤3: SAM标准归一化（[0,1] -> 均值0.5/方差0.5）
        # 记得验证它这个步骤是否必要
        # image = (image - 0.5) / 0.5  # 最终范围 [-1, 1]

    # 转换为三通道（兼容SAM的输入格式）
    image_rgb = np.stack([image]*3, axis=-1)  # (H, W, 3)

    # 数值范围验证（调试用）
    '''
    if args.debug:
        print(f"图像范围: [{image_rgb.min():.2f}, {image_rgb.max():.2f}]")
        print(f"标签类别: {np.unique(mask)}")
    '''
    
    return image_rgb.astype(np.float32), mask.astype(np.uint8)

def preprocess_train_image(image_files, label_files):
    """处理训练集（保存为NPZ）"""
    os.makedirs(f"{args.dst_dir}/train_npz", exist_ok=True)
    
    for img_path, lbl_path in tqdm(zip(image_files, label_files), desc="处理训练集"):
        # 从文件名中提取案例ID（例如5-10101.tif → 5_10101）
        case_id = os.path.basename(img_path).split('.')[0].replace('-', '_')
        
        # 处理并保存
        image, mask = process_slice(img_path, lbl_path)
        np.savez(
            f"{args.dst_dir}/train_npz/{case_id}.npz",
            image=image,
            label=mask
        )

def preprocess_valid_image(image_files, label_files):
    """处理验证集（保存为HDF5）"""
    os.makedirs(f"{args.dst_dir}/test_vol_h5", exist_ok=True)
    
    # 遍历所有验证集样本
    for img_path, lbl_path in tqdm(zip(image_files, label_files), desc="处理验证集"):
        # 提取案例ID（例如5-10101.tif → 5_10101）
        case_id = os.path.basename(img_path).split('.')[0].replace('-', '_')
        
        # 处理切片
        image, mask = process_slice(img_path, lbl_path)
        
        # 保存为HDF5（每个案例单独保存，即使只有单切片）
        with h5py.File(f"{args.dst_dir}/test_vol_h5/{case_id}.h5", 'w') as f:
            f.create_dataset('image', data=image[np.newaxis, ...], dtype='float32')  # (1, H, W, 3)
            f.create_dataset('label', data=mask[np.newaxis, ...], dtype='uint8')     # (1, H, W)

# ======================== 主执行逻辑 ========================
if __name__ == "__main__":
    # 获取所有图像和标注文件（确保一一对应）
    image_files = sorted(glob(f"{args.src_dir}/images/*.tif"))
    label_files = sorted([f.replace('images', 'masks').replace('.tif', '_Simple Segmentation.png') for f in image_files])
    
    # 验证文件存在性
    missing = [lbl for lbl in label_files if not os.path.exists(lbl)]
    if missing:
        raise FileNotFoundError(f"缺失标注文件: {missing[:3]}... (共{len(missing)}个)")
    
    # 划分训练集/验证集
    train_img, val_img, train_lbl, val_lbl = train_test_split(
        image_files, label_files, 
        test_size=args.test_ratio,
        random_state=42
    )
    
    # 执行预处理
    preprocess_train_image(train_img, train_lbl)
    preprocess_valid_image(val_img, val_lbl)
    
    print(f"预处理完成！结果保存在: {args.dst_dir}")