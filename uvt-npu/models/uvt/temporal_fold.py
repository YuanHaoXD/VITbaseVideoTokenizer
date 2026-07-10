"""层级时间折叠/展开（ADR-3 / ADR-4'）。

- 折叠 = 相邻两个时间位在通道维拼接 + Linear(2D→D)，等价 Conv3d(kernel_t=2, stride_t=2)；
- 锚帧（时间位 0）**永不参与折叠**（ADR-4'：锚帧隔离，主动偏离 Hydra-X 的 zero-pad 方案，
  与 OmniTokenizer/Wan-VAE/LeanVAE 的行业标准一致；zero-pad 对照臂见 P2-pre 消融）；
- 纯图像输入（只有锚位）直通。
"""
import torch
import torch.nn as nn
from einops import rearrange


class TemporalFold2x(nn.Module):
    """[B, 1+T, N, D] -> [B, 1+T/2, N, D]"""

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(2 * dim, dim)
        # 初始化为"两帧平均"附近：保护预训练特征分布，避免折叠层随机初始化冲毁下游 block 的输入统计
        with torch.no_grad():
            eye = torch.eye(dim)
            self.proj.weight.copy_(torch.cat([eye, eye], dim=1) * 0.5)
            self.proj.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        anchor, frames = x[:, :1], x[:, 1:]
        if frames.shape[1] == 0:          # 纯图像：直通
            return x
        assert frames.shape[1] % 2 == 0, f"非锚帧数 {frames.shape[1]} 必须能被 2 整除"
        f = rearrange(frames, "b (t two) n d -> b t n (two d)", two=2)
        return torch.cat([anchor, self.proj(f)], dim=1)


class TemporalUnfold2x(nn.Module):
    """[B, 1+T, N, D] -> [B, 1+2T, N, D]（decoder 与 Decompressor 共用的逆算子）"""

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, 2 * dim)
        with torch.no_grad():             # 初始化为"复制两份"：与 Fold 的平均初始化互逆
            eye = torch.eye(dim)
            self.proj.weight.copy_(torch.cat([eye, eye], dim=0))
            self.proj.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        anchor, frames = x[:, :1], x[:, 1:]
        if frames.shape[1] == 0:
            return x
        f = rearrange(self.proj(frames), "b t n (two d) -> b (t two) n d", two=2)
        return torch.cat([anchor, f], dim=1)
