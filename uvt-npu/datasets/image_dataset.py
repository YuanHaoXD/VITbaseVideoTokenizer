"""D-2 · csv 图像数据集（05 篇 §4 D-2 卡）。

设计原则：与 datasets/video_dataset.py（D-1）**同构**——
  - csv 索引（'path' 列必需，'label' 列可选）；
  - null 冒烟机制原样模仿（csv_file 以 'null' 开头 = 假数据，'null128' = 128 个样本）；
  - 输出契约（冻结）：{"video": Tensor[3,1,H,W], "is_video": False}
    图像 = 单锚帧视频（时间维 T=1，即 §0 张量约定的"纯图像 T1=1"），下游零特判。
  - 训练增广：RandomResizedCrop(crop_size, scale=(0.8,1.0)) + RandomHorizontalFlip（任务卡指定）；
    eval：Resize + CenterCrop（与 video_dataset 的 eval_tfm 一致）。

实现注记：
  - 额外附带 "gt"（与 "video" 同一张量）/"path"/"label" 三个键：video_dataset 的训练主键是 'gt'
    （LARP trainer 读 data['gt']），只给 'video' 会迫使 trainer 特判图像源，违背"下游零特判"。
    冻结契约是输出的**下界**（超集不违约）；该键名分歧已上报任务书（见交付报告契约问题①）。
  - 像素值域 [0,1] float32，与 video_dataset 一致（Resize/Crop 直接作用于 float tensor）。
"""
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import (CenterCrop, RandomHorizontalFlip,
                                    RandomResizedCrop, Resize)

from datasets import register


def ImageTransform(crop_size=256, eval_tfm=False, rand_flip='yes'):
    """训练：RandomResizedCrop(256,(0.8,1.0))+HFlip（任务卡冻结）；eval：Resize+CenterCrop。"""
    if eval_tfm:
        return transforms.Compose(
            [Resize(size=crop_size, antialias=True), CenterCrop(crop_size)])
    tfm_list = [RandomResizedCrop(crop_size, scale=(0.8, 1.0), antialias=True)]
    if rand_flip != 'no':
        tfm_list.append(RandomHorizontalFlip())
    return transforms.Compose(tfm_list)


@register('image_dataset')
class ImageDataset(Dataset):
    """csv 图像集：__getitem__ -> {"video": [3,1,H,W], "is_video": False, ...}。

    Args:
        root_path: 数据根目录；csv 里的相对路径基于它拼接。
        crop_size: 输出边长（默认 256，SigLIP2 固定分辨率权重的训练分辨率）。
        split:     'train'（增广）或 'test'（确定性预处理）。
        csv_file:  csv 索引文件；以 'null' 开头进入冒烟模式（模仿 video_dataset 机制）。
        rand_flip: 'yes'/'no'，训练期水平翻转开关。
    """

    def __init__(self, root_path='', crop_size=256, split='train', csv_file='',
                 rand_flip='yes'):
        self.crop_size = crop_size
        self.split = split
        self.csv_file = csv_file

        assert split in ('train', 'test'), f'Unknown split: {split}'
        self.cur_tfm = ImageTransform(
            crop_size=crop_size, eval_tfm=(split == 'test'), rand_flip=rand_flip)

        if csv_file.lower().startswith('null'):  # 冒烟假数据集（无磁盘 IO、无网络）
            num = 128 if csv_file.lower().startswith('null128') else 32 * 7000
            self.fake = True
            self.img_list = ['' for _ in range(num)]
            self.labels = [i % 1000 for i in range(num)]  # 假装 ImageNet 千类
            return

        self.fake = False
        csv_path = csv_file if os.path.isabs(csv_file) else os.path.join(root_path, csv_file)
        csv_data = pd.read_csv(csv_path)
        assert 'path' in csv_data, f"{csv_path} 缺少 'path' 列"
        paths = csv_data['path'].tolist()
        self.img_list = [
            p if os.path.isabs(p) else os.path.join(root_path, p) for p in paths]
        if 'label' in csv_data:
            self.labels = [int(x) for x in csv_data['label'].tolist()]
        else:
            self.labels = [-1] * len(self.img_list)

    def __len__(self):
        return len(self.img_list)

    def _load_image(self, idx):
        """返回 float32 [3,H,W]，值域 [0,1]。"""
        if self.fake:
            arr = torch.randint(0, 256, (3, self.crop_size, self.crop_size),
                                dtype=torch.uint8)
            return arr.float() / 255.
        img = Image.open(self.img_list[idx]).convert('RGB')
        arr = torch.from_numpy(np.array(img, dtype=np.uint8))  # [H,W,3]
        return arr.permute(2, 0, 1).float() / 255.

    def __getitem__(self, idx):
        img = self._load_image(idx)          # [3,H,W]
        img = self.cur_tfm(img)              # [3,crop,crop]
        video = img.unsqueeze(1)             # [3,1,H,W]：图像=单锚帧视频（§0 约定）
        return {
            'video': video,                  # 冻结契约键
            'is_video': False,               # 冻结契约键：蒸馏 vid 项屏蔽依据
            'gt': video,                     # 与 D-1 主键对齐（trainer 统一读 'gt'）
            'path': self.img_list[idx] if not self.fake else 'fake_path',
            'label': self.labels[idx],
        }
