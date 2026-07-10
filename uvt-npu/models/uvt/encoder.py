"""GenViT · 生成编码器（M-6，张量流见 01 §1 步骤①~⑤）。

流程：逐帧 patchify（SigLIP2 embeddings，含学习式空间 PE，ADR-8 默认保留）
  → 按 cfg.fold_positions 在指定 block 之前插入 TemporalFold2x（ADR-3）
  → gen blocks（联合时空注意力 + tubelet 因果 mask，ADR-2）。

约定（任务书 §0 / M-6 卡）：
  - 输入像素 [B,3,H,W]（图像，入口升维 [B,3,1,H,W]，ADR-4'：图像=只有锚帧）
    或 [B,3,1+T,H,W]，T % 4 == 0；
  - 输出 h [B, 1+T/4, N, D]；
  - fold_positions 语义：位置 i = "第 i 个 block 之前折叠"，0=输入处，
    len(blocks)=末块之后（与 decoder 的 unfold_positions=27 语义镜像）；
  - 折叠次数固定 2（4× 时间压缩，01 §2.2 层级式 2×+2×）；
  - 每次折叠后 T1 变化，必须重取 attn_bias 与 time_ids——M-1 有 lru_cache，
    逐 block 重复调用零成本。
"""
import torch
import torch.nn as nn
from einops import rearrange

from .attention_mask import VALID_KINDS, attn_bias, make_time_ids
from .siglip_backbone import BackboneParts
from .temporal_fold import TemporalFold2x


class GenViT(nn.Module):
    def __init__(self, parts: BackboneParts, cfg):
        super().__init__()
        # embeddings/gen_blocks 由本模块独占注册（parts 各字段与消费方一一对应，无重复注册）。
        self.embeddings = parts.embeddings
        self.blocks = nn.ModuleList(parts.gen_blocks)  # M-4 交付的是普通 list，需包 ModuleList 注册参数
        dim = parts.gen_blocks[0].attn.dim

        fp = tuple(cfg.fold_positions)
        # ADR-3：折叠次数固定 2（4× 时间压缩）；位置合法域 [0, len(blocks)]，要求升序。
        assert len(fp) == 2, f"fold_positions 必须恰为 2 个（4×=2×2 层级折叠，ADR-3），实得 {fp}"
        assert all(0 <= p <= len(self.blocks) for p in fp), \
            f"fold_positions {fp} 越界（合法域 [0, {len(self.blocks)}]）"
        assert fp == tuple(sorted(fp)), f"fold_positions 必须升序，实得 {fp}"
        self.fold_positions = fp
        # folds[j] 与 fold_positions[j] 一一对应；近恒等初始化在 TemporalFold2x 内（ADR-4'）。
        self.folds = nn.ModuleList([TemporalFold2x(dim) for _ in fp])

        # ADR-8 消融臂：rope_dims=0 时全程 time_ids=None（block 内亦有同款开关，双保险）。
        self.use_rope = cfg.rope_dims > 0

    def _run_block(self, block, x: torch.Tensor, attn_mode: str) -> torch.Tensor:
        """[B,T1,N,D] 展平 [B,S,D] 过一个 block 再复原。bias/time_ids 按当前 T1 现取（M-1 缓存）。"""
        B, T1, N, D = x.shape
        bias = attn_bias(T1, N, attn_mode, x.device, x.dtype)
        tids = make_time_ids(T1, N, x.device) if self.use_rope else None
        y = block(x.reshape(B, T1 * N, D), bias, tids)
        return y.reshape(B, T1, N, D)

    def forward(self, video: torch.Tensor, attn_mode: str = "tubelet") -> torch.Tensor:
        assert attn_mode in VALID_KINDS, f"unknown attn_mode: {attn_mode}"
        if video.dim() == 4:
            # 图像入口升维 [B,3,1,H,W]（ADR-4'：图像=单锚帧视频，下游零特判）。
            video = video.unsqueeze(2)
        B, _, F, H, W = video.shape
        assert (F - 1) % 4 == 0, f"帧数须为 1+T 且 T%4==0（两次 2× 折叠），实得 F={F}"

        # ①空间 patchify：逐帧过 SigLIP2 embeddings，(B·F) 折进 batch（M-6 卡实现要点②）。
        frames = rearrange(video, "b c f h w -> (b f) c h w")
        tok = self.embeddings(frames)                      # [(B·F), N, D]
        x = rearrange(tok, "(b f) n d -> b f n d", b=B)    # [B, T1=F, N, D]

        # ②~⑤ 折叠位循环：位置 i 的折叠发生在第 i 个 block 之前（0=输入处）。
        for i, block in enumerate(self.blocks):
            for j, pos in enumerate(self.fold_positions):
                if pos == i:
                    x = self.folds[j](x)                   # 锚帧永不折叠（ADR-4'，Fold 内保证）
            x = self._run_block(block, x, attn_mode)
        for j, pos in enumerate(self.fold_positions):
            if pos == len(self.blocks):                    # 末块之后折叠（语义与 decoder 的 27 镜像）
                x = self.folds[j](x)

        # 出口 sanity：h [B, 1+T/4, N, D]。
        assert x.shape[1] == 1 + (F - 1) // 4, \
            f"折叠后时间位 {x.shape[1]} != 1+{(F - 1) // 4}（fold_positions 配置或输入帧数有误）"
        return x
