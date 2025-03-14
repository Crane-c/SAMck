import os
import sys
from tqdm import tqdm
import logging
import numpy as np
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from utils import test_single_volume
from importlib import import_module
from segment_anything import sam_model_registry
from datasets.dataset_rock import Rock_dataset

from icecream import ic


class_to_name = {
    0: 'solid',
    1: 'liquid',
    2: 'air', 
    3: 'background'
}


def inference(args, multimask_output, db_config, model, test_save_path=None):
    db_test = db_config['Dataset'](base_dir=args.volume_path, list_dir=args.list_dir, split='test_vol')
    testloader = DataLoader(db_test, batch_size=args.batch_size, shuffle=False, num_workers=1)
    logging.info(f'测试样本数量: {len(db_test)}, 批次数量: {len(testloader)}')
    model.eval()
    metric_list = np.zeros((args.num_classes, 3))  # 存储每类的dice、hd95、iou

    with torch.no_grad(), torch.cuda.amp.autocast():
        for batch_idx, sample in tqdm(enumerate(testloader), total=len(testloader)):
            # 数据准备
            args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            images = sample['image'].to(args.device)    # (B, C, H, W)
            labels = sample['label'].to(args.device)           # (B, H, W)
            case_ids = sample['case_id']
            
            # 逐样本处理
            for i in range(images.shape[0]):
                # 调用单样本测试
                metric_i = test_single_volume(
                    image=images[i].unsqueeze(0),  # (1, C, H, W)
                    label=labels[i],               # (H, W)
                    net=model,
                    classes=args.num_classes,
                    input_size=[args.input_size, args.input_size],    # SAM输入尺寸（如1024）
                    patch_size=[args.img_size, args.img_size],    # 分块推理尺寸（适配显存）
                    multimask_output=multimask_output,        # 岩土任务通常单mask输出
                    test_save_path=test_save_path,
                    case=case_ids[i],
                    z_spacing=db_config['z_spacing']
                )
                metric_list += np.array(metric_i)
    
    metric_list = metric_list / len(db_test)
    for cls in range(args.num_classes - 1):
        cls_name = class_to_name[cls]
        dice = metric_list[cls][0]
        hd95 = metric_list[cls][1]
        iou = metric_list[cls][2]
        logging.info(f"[相态分类] {cls_name} | Dice: {dice:.4f} | HD95: {hd95:.2f}mm | IoU: {iou:.4f}")

    # 全局统计
    valid_metrics = metric_list[:3]
    avg_dice = np.mean(valid_metrics[:, 0])  # 排除背景
    avg_hd95 = np.mean(valid_metrics[:, 1])
    avg_iou = np.mean(valid_metrics[:, 2])
    logging.info(f"[综合指标] 平均Dice: {avg_dice:.4f} | 平均HD95: {avg_hd95:.2f}mm | 平均IoU: {avg_iou:.4f}")
    logging.info("Testing Finished!")
    return 1


def config_to_dict(config):
    items_dict = {}
    with open(config, 'r') as f:
        items = f.readlines()
    for i in range(len(items)):
        key, value = items[i].strip().split(': ')
        items_dict[key] = value
    return items_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='The config file provided by the trained model')
    parser.add_argument('--volume_path', type=str, default='./data/Rock/Processed/test_vol_h5/')
    parser.add_argument('--dataset', type=str, default='Rock', help='Dataset name')
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--list_dir', type=str, default='./lists/lists_Rock/', help='list_dir')
    parser.add_argument('--output_dir', type=str, default='./output/test')
    parser.add_argument('--img_size', type=int, default=1024, help='Input image size of the network')
    parser.add_argument('--input_size', type=int, default=224, help='The input size for training SAM model')
    parser.add_argument('--batch_size', type=int, default=2, help='The bstch size for training SAM model')
    parser.add_argument('--seed', type=int,
                        default=1234, help='random seed')
    parser.add_argument('--is_savenii', action='store_true', help='Whether to save results during inference')
    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    parser.add_argument('--ckpt', type=str, default='checkpoints/sam_vit_b_01ec64.pth',
                        help='Pretrained checkpoint')
    parser.add_argument('--lora_ckpt', type=str, default='output/sam/results/Rock_1024_pretrain_vit_b_30k_epo15_bs2_lr0.0001/epoch_14.pth', help='The checkpoint from LoRA')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='Select one vit model')
    parser.add_argument('--rank', type=int, default=4, help='Rank for LoRA adaptation')
    parser.add_argument('--module', type=str, default='sam_lora_image_encoder')

    args = parser.parse_args()

    if args.config is not None:
        # overwtite default configurations with config file\
        config_dict = config_to_dict(args.config)
        for key in config_dict:
            setattr(args, key, config_dict[key])

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'Rock': {
            'Dataset': Rock_dataset,
            'volume_path': args.volume_path,
            'list_dir': args.list_dir,
            'num_classes': args.num_classes,
            'z_spacing': 1
        }
    }
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # register model
    sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                                    num_classes=args.num_classes,
                                                                    checkpoint=args.ckpt, pixel_mean=[0, 0, 0],
                                                                    pixel_std=[1, 1, 1])
    
    pkg = import_module(args.module)
    net = pkg.LoRA_Sam(sam, args.rank).cuda()

    assert args.lora_ckpt is not None
    net.load_lora_parameters(args.lora_ckpt)

    if args.num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False

    # initialize log
    log_folder = os.path.join(args.output_dir, 'test_log')
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/' + 'log.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    if args.is_savenii:
        test_save_path = os.path.join(args.output_dir, 'predictions')
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, multimask_output, dataset_config[dataset_name], net, test_save_path)
