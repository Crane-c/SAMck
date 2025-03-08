# datasets/dataset_rock.py
import os
import numpy as np
import h5py
from glob import glob
from torch.utils.data import Dataset
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from einops import repeat

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
        image, label = sample['image'], sample['label']

        # 随机增强
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
            # image, label = simulate_mineral_fracture(image, label)  # 模拟矿物裂缝
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
            # image = add_geological_noise(image)  # 添加地质噪声（如CT伪影）
        # 尺寸调整
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=1)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        # 生成低分辨率标签
        label_h, label_w = label.shape
        low_res_label = zoom(label, (self.low_res[0] / label_h, self.low_res[1] / label_w), order=0)

        # 转换为张量并适配通道
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        image = repeat(image, 'c h w -> (repeat c) h w', repeat=3)
        label = torch.from_numpy(label.astype(np.float32))
        low_res_label = torch.from_numpy(low_res_label.astype(np.float32))
        sample = {'image': image, 'label': label.long(), 'low_res_label': low_res_label.long()}
        return sample
    
class RockDataset(Dataset):
    """加载岩土CT数据集（支持NPZ和HDF5格式）"""
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir
        
    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case_name = self.sample_list[idx].strip('\n')

        if self.split == "train":
            data_path = os.path.join(self.data_dir, case_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            data_path = os.path.join(self.data_dir, f"{case_name}.h5")
            data = h5py.File(data_path, 'r')
            image, label = data['image'][:], data['label'][:]

        # Input dim should be consistent
        # Since the channel dimension of nature image is 3, that of medical image should also be 3

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample