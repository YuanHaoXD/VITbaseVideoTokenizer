"""D-2b · rank 分片契约测试:ParquetImageDataset(rank_shard=True) 全量 DDP 加载。

验证 8 卡 DDP 全量方案的核心不变量(用显式注入 rank/world_size 模拟,不真起 DDP):
  - 各 rank 分到的分片两两**无交集**、并集 == 全集(不漏样本、不重样本);
  - 各 rank __len__ 合理(> 0,且总和 == 单进程全集 __len__);
  - 输出冻结契约键不变(与 D-2 逐键一致)。

数据依赖:ImageNet-1k parquet(默认路径,可用环境变量 UVT_IMAGENET_PARQUET 覆盖)。
无数据 / 无 pyarrow 时自动 skip——CPU 开发机与 CI 不阻塞。
"""
import glob
import os

import pytest

pytest.importorskip("pyarrow")

PQ_DIR = os.environ.get(
    "UVT_IMAGENET_PARQUET",
    "/home/ma-user/work/dataset/yh222/Datasets/imagenet-1k/data",
)

# 只取前 6 个分片做子集验证(足够覆盖 world_size=2 的分片切分逻辑,且快)。
MAX_SHARDS = 6


def _has_data():
    return len(glob.glob(os.path.join(PQ_DIR, "train-*.parquet"))) >= 2


pytestmark = pytest.mark.skipif(
    not _has_data(), reason=f"无 ImageNet parquet(需≥2 分片):{PQ_DIR}")


def _make(rank_shard=False, rank=None, world_size=None, **kw):
    import datasets  # 仓库内本地 datasets 包
    spec = {"name": "parquet_image_dataset",
            "args": {"parquet_dir": PQ_DIR, "crop_size": 64,
                     "max_shards": MAX_SHARDS, "rank_shard": rank_shard,
                     "rank": rank, "world_size": world_size, **kw}}
    return datasets.make(spec)


def _shard_names(ds):
    """该 dataset 实际持有的分片文件名集合(rank 分片后的子集)。"""
    return {os.path.basename(f) for f in ds.files}


def test_rank_shards_disjoint_and_cover_all():
    """world_size=2:rank0/rank1 分片无交集、并集 == 全集(前 MAX_SHARDS 片)。"""
    full = _make(rank_shard=False)                       # 单进程全集(对照)
    r0 = _make(rank_shard=True, rank=0, world_size=2)
    r1 = _make(rank_shard=True, rank=1, world_size=2)

    s_full, s0, s1 = _shard_names(full), _shard_names(r0), _shard_names(r1)
    assert s0 & s1 == set(), f"rank0/rank1 分片有交集:{s0 & s1}"   # 无交集
    assert s0 | s1 == s_full, "rank0∪rank1 分片 != 全集"           # 并集 == 全集
    # files[rank::world_size] 步进切分:rank0 取偶数位、rank1 取奇数位
    assert len(s0) == (MAX_SHARDS + 1) // 2
    assert len(s1) == MAX_SHARDS // 2


def test_rank_lens_sum_to_full():
    """各 rank __len__ 均 > 0,且总和 == 单进程全集 __len__(样本不漏不重)。"""
    full = _make(rank_shard=False)
    r0 = _make(rank_shard=True, rank=0, world_size=2)
    r1 = _make(rank_shard=True, rank=1, world_size=2)
    assert len(r0) > 0 and len(r1) > 0
    assert len(r0) + len(r1) == len(full)


def test_rank_shard_frozen_contract_unchanged():
    """rank 分片后单样本仍满足 D-2 冻结契约(键/形状/值域/类型)。"""
    import torch
    r0 = _make(rank_shard=True, rank=0, world_size=2)
    b = r0[0]
    assert set(b.keys()) == {"video", "is_video", "gt", "path", "label"}
    assert tuple(b["video"].shape) == (3, 1, 64, 64)
    assert b["video"].dtype == torch.float32
    assert 0.0 <= float(b["video"].min()) and float(b["video"].max()) <= 1.0
    assert b["is_video"] is False
    assert torch.equal(b["gt"], b["video"])
    assert isinstance(b["label"], int)


def test_default_off_is_single_process_fullset():
    """不开 rank_shard(默认):行为与改动前一致——持有全部分片。"""
    full = _make()                                        # rank_shard 默认 False
    assert _shard_names(full) == {
        os.path.basename(f)
        for f in sorted(glob.glob(os.path.join(PQ_DIR, "train-*.parquet")))[:MAX_SHARDS]}


def test_world_size3_partition():
    """奇数 world_size=3:三 rank 分片仍两两无交集、并集 == 全集。"""
    parts = [_shard_names(_make(rank_shard=True, rank=r, world_size=3))
             for r in range(3)]
    union = set().union(*parts)
    assert sum(len(p) for p in parts) == len(union) == MAX_SHARDS   # 无重叠 → 计数==并集
