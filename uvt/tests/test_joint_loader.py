"""D-3 验收测试：datasets/joint_loader.py。

- test_joint_determinism：同 seed 两次实例化 → 源调度序与样本内容完全一致（配对消融 D13 的命门）。
- test_ratio_composition：每 epoch 内各源出现次数满足 ratio 配比。
- test_pure_batch：每步 batch 均为单源纯 batch（不混装）。
- test_trace_file：sampling_trace.json 落盘且与实际调度一致。

纯离线单进程（DDP 分片路径由 8 卡冒烟覆盖，见 TR-1 验收），只依赖 torch。
"""
import json

import torch
from torch.utils.data import Dataset

from datasets.joint_loader import JointLoader


class _IndexDataset(Dataset):
    """确定性哑数据集：样本值 = base + idx（值域可区分源，且内容与全局 RNG 无关）。"""

    def __init__(self, base, n, is_video):
        self.base, self.n, self.is_video = base, n, is_video

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {'gt': torch.tensor(self.base + idx, dtype=torch.long),
                'is_video': self.is_video}


def _make_sources():
    # 源 0：视频（bs=4, ratio=3）；源 1：图像（bs=8, ratio=1）→ 窗长 4，窗内 3 视频 1 图像
    return [
        {'dataset': _IndexDataset(0, 48, True), 'batch_size': 4, 'ratio': 3, 'name': 'vid'},
        {'dataset': _IndexDataset(1000, 32, False), 'batch_size': 8, 'ratio': 1, 'name': 'img'},
    ]


def _collect(loader, epochs=2):
    """返回 (源序列, 样本内容序列)。源由值域判别：>=1000 为图像源。"""
    src_seq, content = [], []
    for _ in range(epochs):
        for batch in loader:
            v = batch['gt']
            src_seq.append(1 if int(v[0].item()) >= 1000 else 0)
            content.append(v.tolist())
    return src_seq, content


def test_joint_determinism(tmp_path):
    l1 = JointLoader(_make_sources(), seed=7, trace_path=str(tmp_path / 't1.json'))
    l2 = JointLoader(_make_sources(), seed=7, trace_path=str(tmp_path / 't2.json'))
    s1, c1 = _collect(l1)
    s2, c2 = _collect(l2)
    assert s1 == s2, '同 seed 两次实例化的源调度序必须一致'
    assert c1 == c2, '同 seed 两次实例化的样本序必须一致（含源内洗牌）'

    # 反向 sanity：不同 seed 应产生不同序（16 步/epoch 下碰撞概率可忽略）
    l3 = JointLoader(_make_sources(), seed=8, trace_path=None)
    s3, c3 = _collect(l3)
    assert (s1, c1) != (s3, c3), '不同 seed 不应复现同一采样序'


def test_ratio_composition():
    loader = JointLoader(_make_sources(), seed=0, trace_path=None)
    # len(vid_loader)=12, ratio=3 → 4 cycles；len(img_loader)=4, ratio=1 → 4 cycles
    # steps_per_epoch = 4 cycles × 窗长 4 = 16
    assert len(loader) == 16
    src_seq, _ = _collect(loader, epochs=1)
    assert src_seq.count(0) == 12 and src_seq.count(1) == 4
    # 窗内配比严格：每个窗长 4 的窗口恰好 3 视频 1 图像
    for w in range(0, 16, 4):
        window = src_seq[w:w + 4]
        assert window.count(0) == 3 and window.count(1) == 1


def test_pure_batch():
    loader = JointLoader(_make_sources(), seed=1, trace_path=None)
    for batch in loader:
        v = batch['gt']
        is_img = v >= 1000
        assert bool(is_img.all()) or bool((~is_img).all()), '每步必须是单源纯 batch'
        # batch 大小随源而定：视频 4 / 图像 8
        assert v.shape[0] == (8 if bool(is_img.all()) else 4)
        # is_video 标记与源一致（D-1/D-2 契约字段）
        assert bool(batch['is_video'].all()) == (not bool(is_img.all()))


def test_trace_file(tmp_path):
    trace_path = tmp_path / 'sampling_trace.json'
    loader = JointLoader(_make_sources(), seed=42, trace_path=str(trace_path))
    src_seq, _ = _collect(loader, epochs=1)

    assert trace_path.exists(), 'sampling_trace.json 必须落盘'
    with open(trace_path) as f:
        trace = json.load(f)
    assert trace['seed'] == 42
    assert trace['steps_per_epoch'] == 16
    assert [s['name'] for s in trace['sources']] == ['vid', 'img']
    assert trace['epochs']['0'] == src_seq, 'trace 记录的调度序必须与实际产出一致'
    # 迭代完一个 epoch 后自动推进 epoch → trace 追加了下一 epoch 的调度
    with open(trace_path) as f:
        trace2 = json.load(f)
    assert '1' in trace2['epochs']
