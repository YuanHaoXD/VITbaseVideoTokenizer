"""D-2b 契约测试:parquet_image_dataset 输出与 D-2 冻结契约一致。

数据依赖:ImageNet-1k parquet(默认路径见下,可用环境变量 UVT_IMAGENET_PARQUET 覆盖)。
无数据 / 无 pyarrow 时自动 skip——CPU 开发机与 CI 不阻塞。
"""
import glob
import os

import pytest
import torch

pytest.importorskip("pyarrow")

PQ_DIR = os.environ.get(
    "UVT_IMAGENET_PARQUET",
    "/home/ma-user/work/dataset/yh222/Datasets/imagenet-1k/data",
)


def _has_data():
    return bool(glob.glob(os.path.join(PQ_DIR, "train-*.parquet")))


pytestmark = pytest.mark.skipif(
    not _has_data(), reason=f"无 ImageNet parquet:{PQ_DIR}")


def _make(**kw):
    import datasets  # 仓库内本地 datasets 包
    spec = {"name": "parquet_image_dataset",
            "args": {"parquet_dir": PQ_DIR, "crop_size": 64,
                     "max_shards": 1, "max_samples": 4, **kw}}
    return datasets.make(spec)


def test_frozen_contract_keys_and_shapes():
    ds = _make()
    assert len(ds) == 4
    b = ds[0]
    assert set(b.keys()) == {"video", "is_video", "gt", "path", "label"}
    assert tuple(b["video"].shape) == (3, 1, 64, 64)   # [3,1,H,W] 单锚帧
    assert b["video"].dtype == torch.float32
    assert 0.0 <= float(b["video"].min()) and float(b["video"].max()) <= 1.0
    assert b["is_video"] is False
    assert torch.equal(b["gt"], b["video"])            # gt 与 video 同张量
    assert isinstance(b["label"], int)


def test_collate_batches():
    from torch.utils.data import DataLoader
    ds = _make()
    bb = next(iter(DataLoader(ds, batch_size=2, num_workers=0)))
    assert tuple(bb["video"].shape) == (2, 3, 1, 64, 64)
    assert bb["is_video"].tolist() == [False, False]


def test_in_memory_matches_lazy():
    """in_memory 与惰性单分片缓存对同一 idx 应产出同 label(同数据源)。"""
    lazy = _make(in_memory=False)
    mem = _make(in_memory=True)
    assert [lazy[i]["label"] for i in range(4)] == [mem[i]["label"] for i in range(4)]
