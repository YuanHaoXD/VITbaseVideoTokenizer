"""Decompressor · 训练期 4× 时间上采样（M-9，张量流见 01 §1 步骤⑩ / 01 §2.8）。

职责：把 Sem-ViT 非锚帧输出 s[:,1:] 从 (T1z-1) 个时间位升回 16 帧，
与 InternVideo 视频教师逐帧对齐（L-2 蒸馏 vid 项）。**只在训练图上存在，
checkpoint 不导出**（M-10 的 state_dict post-hook 剔除本模块）。

【契约裁决（任务书 M-9，架构方 v1.x 验收）】
  - **输出 STUDENT_DIM(D=1152)，不投教师维度**——投教师维度的职责归 L-2 的 head_vid
    （投影职责单一化：与 img_patch/img_pool 三项一致；且 Stage-3「冻 decoder 训 Sem 侧」
    时冻结归属清晰）。故 01 §2.8 文字「末端 Linear(1152→D_teacher)」为 stale，本类不持该 Linear。
  - 接口：`Decompressor(dim, num_heads).forward(s_frames:[B,T1z-1,N,D]) -> [B,4*(T1z-1),N,D]`。

【实现选择（M-9 卡「二选一」的后半）】**不调用 TemporalUnfold2x.forward**：其锚帧直通逻辑
  （`x[:,:1]` 切片 + `x[:,1:]` 才上采样）对无锚位输入会误伤——本模块输入已是 s[:,1:]
  （锚位已剥离），复用它会误把首位当锚帧直通而非参与上采样。故直接用 proj+rearrange
  内联实现两级 2× 上采样（`_TemporalUnfold2xNoAnchor`），结构 = 两级 (Unfold → UVTBlock(full mask))，
  4 位 → 8 → 16。

【注意力模式】full（双向）：蒸馏上采样无因果约束，全双向注意力（M-9 卡「定 full」）。
  full 走 SDPA 无 mask 快速路径（M-1）。rope_dims=0（训练期附件，无需时间 RoPE；M-9 卡「无 RoPE 皆可」）。
"""
import torch
import torch.nn as nn
from einops import rearrange

from .attention_mask import VALID_KINDS, attn_bias
from .blocks import UVTBlock


class _TemporalUnfold2xNoAnchor(nn.Module):
    """[B,T,N,D] -> [B,2T,N,D]：镜像 TemporalUnfold2x 的 proj+rearrange，**去掉锚帧直通逻辑**。

    Decompressor 输入是 s[:,1:]（锚位已剥离），复用 TemporalUnfold2x 会把首位误当锚帧直通。
    近恒等初始化（复制两份），与 TemporalUnfold2x 互逆、保护 Sem-ViT 输出分布。
    """

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, 2 * dim)
        with torch.no_grad():            # 初始化为「复制两份」，与 Fold 的平均初始化互逆
            eye = torch.eye(dim)
            self.proj.weight.copy_(torch.cat([eye, eye], dim=0))
            self.proj.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(self.proj(x), "b t n (two d) -> b (t two) n d", two=2)


class Decompressor(nn.Module):
    """训练期 4× 时间上采样（蒸馏专用，checkpoint 不导出）。"""

    def __init__(self, dim: int = 1152, num_heads: int = 16, rope_dims: int = 0):
        super().__init__()
        # 两级 2× 上采样：4 位 → 8 → 16（T1z-1=4 时输出 16 帧，与 InternVideo 16 帧逐帧对齐）。
        # UVTBlock 随机初始化（蒸馏附件无需继承 SigLIP2 权重）；full mask + rope_dims=0。
        self.up1 = _TemporalUnfold2xNoAnchor(dim)
        self.block1 = UVTBlock(dim, num_heads, rope_dims)
        self.up2 = _TemporalUnfold2xNoAnchor(dim)
        self.block2 = UVTBlock(dim, num_heads, rope_dims)
        self.use_rope = rope_dims > 0

    def _run_block(self, block: UVTBlock, x: torch.Tensor, attn_mode: str) -> torch.Tensor:
        """[B,T,N,D] 展平 [B,S,D] 过一个 block 再复原（full 走 SDPA 无 mask 快速路径）。"""
        B, T, N, D = x.shape
        bias = attn_bias(T, N, attn_mode, x.device, x.dtype)
        tids = None                         # rope_dims=0：不上时间 RoPE
        y = block(x.reshape(B, T * N, D), bias, tids)
        return y.reshape(B, T, N, D)

    def forward(self, s_frames: torch.Tensor, attn_mode: str = "full") -> torch.Tensor:
        """s_frames:[B,T1z-1,N,D] -> [B,4*(T1z-1),N,D]（D=STUDENT_DIM，不投教师维度）。"""
        assert attn_mode in VALID_KINDS, f"unknown attn_mode: {attn_mode}"
        x = self._run_block(self.block1, self.up1(s_frames), attn_mode)   # T → 2T
        x = self._run_block(self.block2, self.up2(x), attn_mode)          # 2T → 4T
        return x
