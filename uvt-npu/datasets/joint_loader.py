"""D-3 · 图像/视频多源联合 loader（OmniTokenizer VideoData 设计思想 → 无 Lightning 重写）。

设计来源：OmniTokenizer data.py 的 VideoData——每源独立 batch_size + sample_ratio 交替供批。
本实现只借其"多源交替"思想，不引入 pytorch-lightning（硬约束）。

三条纪律（05 篇 D-3 卡 + 02 篇 §2.3）：
  1. 每个 step 产出**单源纯 batch**（不混装：视频/图像张量形状不同、tubelet mask 不同）；
  2. **确定性**：源调度序列由 (seed, epoch) 完全决定，从而任一 step 取哪个源由 (seed, step)
     完全确定；序列落盘 sampling_trace.json——配对消融（README D13）依赖同数据序；
  3. DDP 兼容：分布式环境下每源用 DistributedSampler 按 rank 分片（sampler seed=构造 seed），
     单进程用固定种子 torch.Generator 洗牌——两种路径同 seed 下均可复现。

调度算法：
  - 每个"循环窗"包含 ratio_i 次源 i（例：图像 ratio=1、视频 ratio=3 → 窗长 4，窗内 3 视频 1 图像）；
  - 窗内顺序用 random.Random(f"{seed}:{epoch}") 洗牌（窗内洗牌保证任意前缀的配比误差 < 1 窗）；
  - 每 epoch 步数 = 窗长 × min_i(每源可供批数 // ratio_i)（受限源恰好消费一遍，长源欠采样，
    跨 epoch 由 sampler 重洗弥补）。

批大小语义：sources[i]['batch_size'] 是**每 rank** 的 micro-batch 大小（DDP 全局 batch =
batch_size × world_size × grad_accumulates），与 base_trainer 的"全局 batch ÷ tot_gpus"约定
不同——联合训练各源 batch 不同，除法语义在此处会引入隐蔽的整除坑，故改为显式 per-rank。
"""
import json
import os
import random as pyrandom

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


class JointLoader:
    """多源交替联合 loader（可迭代对象，len = 每 epoch 步数）。

    Args:
        sources: list[dict]，每源 {"dataset": Dataset, "batch_size": int, "ratio": int}，
                 可选 "name"（trace 里的源名，缺省用 dataset 类名+下标）。
        seed:    采样确定性种子（配对消融的命门：两臂共用同一 seed → 同数据序）。
        num_workers: 每源 DataLoader 的 worker 数。
        trace_path:  sampling_trace.json 落盘路径；None = 不落盘（单测用）。
        drop_last:   源内 loader 是否丢尾批（默认 True，保证 batch 形状恒定）。
        pin_memory:  透传 DataLoader。
    """

    def __init__(self, sources, seed=0, num_workers=0,
                 trace_path='sampling_trace.json', drop_last=True, pin_memory=False):
        assert len(sources) >= 1, 'JointLoader 至少需要一个源'
        self.seed = int(seed)
        self.trace_path = trace_path
        self.epoch = 0
        self._auto_epoch = True   # 外部一旦调用 set_epoch 即交出 epoch 推进权

        if dist.is_available() and dist.is_initialized():
            self.world_size, self.rank = dist.get_world_size(), dist.get_rank()
        else:
            self.world_size, self.rank = 1, 0

        self.names, self.ratios, self.loaders, self.samplers = [], [], [], []
        for i, src in enumerate(sources):
            ds, bs = src['dataset'], int(src['batch_size'])
            ratio = int(src['ratio'])
            assert ratio >= 1, f"源 {i} 的 ratio 必须为正整数，得到 {src['ratio']}"
            self.names.append(src.get('name', f'{type(ds).__name__}_{i}'))
            self.ratios.append(ratio)

            if self.world_size > 1:
                sampler = DistributedSampler(
                    ds, num_replicas=self.world_size, rank=self.rank,
                    shuffle=True, seed=self.seed, drop_last=drop_last)
                generator = None
                shuffle = False
            else:
                sampler = None
                # 单进程：固定种子 generator 驱动 RandomSampler，两次实例化同序
                generator = torch.Generator()
                generator.manual_seed(self.seed + i)
                shuffle = True
            loader = DataLoader(
                ds, batch_size=bs, sampler=sampler, shuffle=shuffle,
                generator=generator, num_workers=num_workers,
                drop_last=drop_last, pin_memory=pin_memory,
                persistent_workers=(num_workers > 0))
            self.loaders.append(loader)
            self.samplers.append(sampler)

        self._window = sum(self.ratios)
        cycles = max(1, min(len(ld) // r for ld, r in zip(self.loaders, self.ratios)))
        self.steps_per_epoch = cycles * self._window
        self._exhaust_count = [0] * len(self.loaders)

        self._schedule = self._build_schedule(self.epoch)
        self._write_trace()

    # ------------------------------------------------------------------ 调度
    def _build_schedule(self, epoch):
        """源 id 序列 [steps_per_epoch]，由 (seed, epoch) 完全决定。"""
        rng = pyrandom.Random(f'{self.seed}:{epoch}')
        template = [i for i, r in enumerate(self.ratios) for _ in range(r)]
        schedule = []
        for _ in range(self.steps_per_epoch // self._window):
            window = list(template)
            rng.shuffle(window)
            schedule.extend(window)
        return schedule

    def source_of_step(self, step, epoch=None):
        """契约接口：(seed, step) → 源 id（供配对消融校验两臂数据序一致）。"""
        epoch = self.epoch if epoch is None else epoch
        schedule = self._schedule if epoch == self.epoch else self._build_schedule(epoch)
        return schedule[step % self.steps_per_epoch]

    def set_epoch(self, epoch):
        """外部（trainer）每 epoch 调用：重排调度序 + 通知各源 sampler + 追加 trace。"""
        self._auto_epoch = False
        self._apply_epoch(int(epoch))

    def _apply_epoch(self, epoch):
        self.epoch = epoch
        for sampler in self.samplers:
            if sampler is not None:
                sampler.set_epoch(epoch)
        self._schedule = self._build_schedule(epoch)
        self._write_trace()

    # ------------------------------------------------------------------ trace
    def _write_trace(self):
        """rank 0 落盘调度序（原子替换写，追加式按 epoch 记录）。"""
        if self.trace_path is None or self.rank != 0:
            return
        trace = {}
        if os.path.exists(self.trace_path):
            try:
                with open(self.trace_path, 'r') as f:
                    trace = json.load(f)
            except (json.JSONDecodeError, OSError):
                trace = {}  # 损坏的旧 trace 直接重建（trace 是审计副本，不是状态源）
        trace.setdefault('seed', self.seed)
        trace.setdefault('sources', [
            {'name': n, 'ratio': r, 'batch_size': ld.batch_size, 'num_batches': len(ld)}
            for n, r, ld in zip(self.names, self.ratios, self.loaders)])
        trace.setdefault('steps_per_epoch', self.steps_per_epoch)
        trace.setdefault('epochs', {})
        trace['epochs'][str(self.epoch)] = self._schedule
        tmp = self.trace_path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(trace, f)
        os.replace(tmp, self.trace_path)

    # ------------------------------------------------------------------ 迭代
    def __len__(self):
        return self.steps_per_epoch

    def __iter__(self):
        iters = [iter(ld) for ld in self.loaders]
        for src in self._schedule:
            try:
                batch = next(iters[src])
            except StopIteration:
                # 长源在 epoch 内耗尽（ratio 与数据量不匹配时）：确定性重开。
                # DDP 用递增伪 epoch 重洗；单进程 generator 状态自然续走——两者均可复现。
                self._exhaust_count[src] += 1
                if self.samplers[src] is not None:
                    self.samplers[src].set_epoch(
                        self.epoch * 10007 + self._exhaust_count[src])
                iters[src] = iter(self.loaders[src])
                batch = next(iters[src])
            yield batch
        if self._auto_epoch:
            # 未接 trainer 的 set_epoch 时自动推进（保证多轮迭代也完全确定）
            self._apply_epoch(self.epoch + 1)
