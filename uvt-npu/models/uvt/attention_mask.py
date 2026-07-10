"""注意力 mask 工厂（ADR-2）。

约定：token 序列按 [B, T1, N, D] 展平成 [B, T1*N, D]，时间位 t 的 N 个空间 token 连续排布。
时间位 0 = 锚帧（图像输入时只有锚位）。mask 以"折叠后的时间位"为单位（ADR-2）。

三种模式（对应 Hydra-X 表 1 与 Phase B/P2 消融）：
  full    : 全时空双向
  causal  : 全历史因果（tk <= tq）
  tubelet : 位 t 只见 {t-1, t}；锚位只见自己（帧内始终全双向）
"""
from functools import lru_cache

import torch

VALID_KINDS = ("full", "causal", "tubelet")


def make_time_ids(T1: int, N: int, device: torch.device) -> torch.Tensor:
    """每个 token 的时间位 id，形状 [T1*N]。"""
    return torch.arange(T1, device=device).repeat_interleave(N)


def make_bool_mask(time_ids: torch.Tensor, kind: str) -> torch.Tensor:
    """[S,S] bool，True=允许注意。"""
    assert kind in VALID_KINDS, f"unknown attn kind: {kind}"
    tq, tk = time_ids[:, None], time_ids[None, :]
    if kind == "full":
        return torch.ones(len(time_ids), len(time_ids), dtype=torch.bool, device=time_ids.device)
    if kind == "causal":
        return tk <= tq
    # tubelet
    return (tk == tq) | (tk == tq - 1)


@lru_cache(maxsize=128)
def _cached_bias(T1: int, N: int, kind: str, device_str: str, dtype_str: str) -> torch.Tensor:
    """加性 bias [1,1,S,S]（0=允许 / dtype最小值=禁止），可直接传 SDPA attn_mask。
    按 (T1,N,kind,device,dtype) 缓存——BlockMask/大mask 反复构造是常见性能坑。"""
    device = torch.device(device_str)
    dtype = getattr(torch, dtype_str)
    time_ids = make_time_ids(T1, N, device)
    allowed = make_bool_mask(time_ids, kind)
    neg = torch.finfo(dtype).min  # 不用 -inf：fp16/bf16 下 softmax 全 -inf 行会出 NaN
    bias = torch.where(allowed, torch.zeros((), device=device, dtype=dtype),
                       torch.full((), neg, device=device, dtype=dtype))
    return bias[None, None]  # [1,1,S,S]


def attn_bias(T1: int, N: int, kind: str, device: torch.device, dtype: torch.dtype):
    """训练主入口。full 模式返回 None（走 SDPA 无 mask 快速路径）。"""
    if kind == "full":
        return None
    return _cached_bias(T1, N, kind, str(device), str(dtype).split(".")[-1])
