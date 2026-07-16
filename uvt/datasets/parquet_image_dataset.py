"""D-2b · HF parquet 图像数据集(ImageNet-1k parquet 直读)。

动机:ImageNet-1k 在本机是 HuggingFace parquet 分片(`train-*.parquet`,294 片
×~4358 行≈128 万),每行 `image={bytes: JPEG字节, path: 文件名}` + `label: int`。
`datasets/image_dataset.py`(D-2)读的是 CSV+磁盘 JPEG,不适用;本类直读 parquet,
**免抽取、可扩全量**,输出与 D-2 完全相同的冻结契约,下游(JointLoader/TR-2)零改动。

输出契约(与 D-2 逐键一致):
    {"video": Tensor[3,1,H,W], "is_video": False, "gt": <同 video>,
     "path": "<shard>#<row>", "label": int}
预处理复用 D-2 的 `ImageTransform`(训练 RandomResizedCrop+HFlip / eval Resize+CenterCrop),
保证与 CSV 图像路径的增广/值域([0,1] float32)完全一致。

加载策略:
  - 索引 `(shard_idx, row)` 由各分片 parquet **元数据**(`num_rows`)构建,不整读分片——秒级。
  - `in_memory=False`(默认):单分片 LRU(缓存最近访问的一片 DataFrame)。适合顺序/子集;
    全量 + DistributedSampler 全排列洗牌会跨片抖动(见下 `shard_shuffle` 说明)。
  - `in_memory=True`:__init__ 预载选中分片的两列进内存。子集验证首选(2 片≈360MB,快且无抖动)。
  - `max_shards>0` / `max_samples>0`:截取子集(短程验证用)。

已知边界:部分 ImageNet 图为灰度('L')/CMYK → `.convert('RGB')` 统一为 3 通道。

⚠️ 全量训练的跨片洗牌抖动:`in_memory=False` 下若 sampler 全排列,单片缓存会被反复换出。
全量阶段方案(已实现):`rank_shard=True` + `in_memory=True` —— 每 rank 只持有 files[rank::
world_size](294 片 8 卡≈37 片/rank≈16GB/rank,主机内存充足),rank 内本地洗牌无跨片抖动;
DataLoader 侧配 JointLoader `pre_sharded: true` 跳过 DistributedSampler(否则二次分片)。
"""
import glob as _glob
import io
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import Dataset

from datasets import register
from datasets.image_dataset import ImageTransform  # 复用 D-2 冻结预处理


def _resolve_rank(rank, world_size):
    """(rank, world_size) 缺省时从 torch.distributed 读；不可用则退 (0, 1)。"""
    if rank is not None and world_size is not None:
        return int(rank), int(world_size)
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


@register('parquet_image_dataset')
class ParquetImageDataset(Dataset):
    """HF parquet 图像集:__getitem__ -> D-2 冻结契约 dict。

    Args:
        parquet_dir: parquet 分片所在目录。
        file_glob:   分片匹配(默认 'train-*.parquet';eval 用 'test-*.parquet' 等)。
        crop_size:   输出边长(默认 256,SigLIP2 固定分辨率)。
        split:       'train'(增广)/ 'test'(确定性预处理)。
        rand_flip:   'yes'/'no',训练水平翻转开关。
        max_shards:  >0 时只用前 N 个分片(子集验证)。
        max_samples: >0 时全局样本上限(子集验证)。
        in_memory:   True 时预载选中分片两列进内存(子集验证首选,无跨片抖动)。
        image_col/label_col: 列名(HF imagenet-1k 为 'image'/'label')。
        rank_shard:  True 时按 rank 步进切片(files[rank::world_size]),每 rank 只索引/
                     预载自己那 ~1/world_size 份分片。**全量 8 卡 DDP 高效加载的关键**:
                     配合 in_memory=True,每 rank 常驻 ~1/N 分片(8 卡×37 片≈16GB/rank),
                     且 DataLoader 侧必须 opt-in 跳过 DistributedSampler(见 JointLoader
                     `pre_sharded`),否则会二次分片、每 rank 只训到 1/N² 数据。
        rank/world_size: rank_shard 的切片坐标;缺省从 torch.distributed 读,不可用退 (0,1)。
    """

    def __init__(self, parquet_dir, file_glob='train-*.parquet', crop_size=256,
                 split='train', rand_flip='yes', max_shards=0, max_samples=0,
                 in_memory=False, image_col='image', label_col='label',
                 rank_shard=False, rank=None, world_size=None):
        assert split in ('train', 'test'), f'Unknown split: {split}'
        self.crop_size = crop_size
        self.image_col = image_col
        self.label_col = label_col
        self.in_memory = in_memory

        files = sorted(_glob.glob(os.path.join(parquet_dir, file_glob)))
        assert files, f'无 parquet 分片:{os.path.join(parquet_dir, file_glob)}'
        if max_shards > 0:
            files = files[:max_shards]
        # rank 分片:各 rank 只持有 files[rank::world_size]。在 max_shards 截取之后做,
        # 从而各 rank 分片两两无交集、并集 = 上一步选中的全集(见 tests/test_parquet_rank_shard)。
        if rank_shard:
            self.rank, self.world_size = _resolve_rank(rank, world_size)
            files = files[self.rank::self.world_size]
            assert files, (
                f'rank {self.rank}/{self.world_size} 分片后无分片——'
                f'world_size 超过分片数({max_shards or "全部"})?')
        else:
            self.rank, self.world_size = 0, 1
        self.rank_shard = rank_shard
        self.files = files

        # 索引:全局 idx -> (分片下标, 分片内行号)。用元数据取行数,不整读分片。
        self._index = []
        for si, f in enumerate(files):
            n = pq.ParquetFile(f).metadata.num_rows
            for r in range(n):
                self._index.append((si, r))
                if max_samples > 0 and len(self._index) >= max_samples:
                    break
            if max_samples > 0 and len(self._index) >= max_samples:
                break

        self.cur_tfm = ImageTransform(
            crop_size=crop_size, eval_tfm=(split == 'test'), rand_flip=rand_flip)

        self._cache_si = -1        # 单分片 LRU
        self._cache_df = None
        if in_memory:
            self._shards = [
                pd.read_parquet(f, columns=[image_col, label_col]) for f in files]

    def __len__(self):
        return len(self._index)

    def _shard(self, si):
        if self.in_memory:
            return self._shards[si]
        if si != self._cache_si:
            self._cache_df = pd.read_parquet(
                self.files[si], columns=[self.image_col, self.label_col])
            self._cache_si = si
        return self._cache_df

    def _load_image(self, si, row):
        """返回 (float32 [3,H,W] 值域[0,1], label:int)。"""
        rec = self._shard(si).iloc[row]
        img = Image.open(io.BytesIO(rec[self.image_col]['bytes'])).convert('RGB')
        arr = torch.from_numpy(np.array(img, dtype=np.uint8))  # [H,W,3]
        return arr.permute(2, 0, 1).float() / 255., int(rec[self.label_col])

    def __getitem__(self, idx):
        si, row = self._index[idx]
        img, label = self._load_image(si, row)       # [3,H,W]
        img = self.cur_tfm(img)                       # [3,crop,crop]
        video = img.unsqueeze(1)                      # [3,1,H,W]:图像=单锚帧视频(§0)
        return {
            'video': video,                           # 冻结契约键
            'is_video': False,                        # 冻结契约键:蒸馏 vid 项屏蔽依据
            'gt': video,                              # 与 D-1 主键对齐
            'path': f'{os.path.basename(self.files[si])}#{row}',
            'label': label,
        }
