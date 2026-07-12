"""PixelDecoder · 像素解码器（M-8，张量流见 01 §1 步骤⑦ / 01 §2.7）。

结构：in_proj Linear(c_latent→D)（自持，见 M-5 裁决）→ 27 层 ViT block
  → 按 cfg.unfold_positions 在指定 block 之前插入 TemporalUnfold2x（镜像 encoder
  的折叠点）→ 锚帧/非锚帧独立像素头 Linear(D→3·p·p) + rearrange 回像素。

约定（M-8 卡）：
  - 消费**物理** latent z_phys（非规范化；规范化往返由 GSB/调用方负责，ADR-5：
    decoder 物理上永远消费反归一化的 z）；
  - unfold_positions 语义：位置 i = "第 i 个 block 之前展开"，len(blocks)（默认 27）
    =末块之后——与 encoder fold_positions 的 0=输入处 镜像；
  - 双像素头：锚帧与非锚帧独立 Linear（镜像 OmniTokenizer 的 first-frame 分头设计，
    ADR-4' 锚帧隔离在输出端的对应物）；
  - 初始化：dec_blocks 用 SigLIP2 全 27 层权重复制（01 §2.7）；对照臂
    cfg.decoder_init="random" 时改用随机初始化的同构 block（P1 消融）。
"""
import torch
import torch.nn as nn
from einops import rearrange

from .attention_mask import VALID_KINDS, attn_bias, make_time_ids
from .blocks import UVTBlock
from .siglip_backbone import BackboneParts
from .temporal_fold import TemporalUnfold2x


class PixelDecoder(nn.Module):
    def __init__(self, parts: BackboneParts, cfg):
        super().__init__()
        dim = parts.dec_blocks[0].attn.dim
        num_heads = parts.dec_blocks[0].attn.num_heads

        # 01 §2.7 初始化裁决：默认 SigLIP2 权重复制；random 臂由 config 控制（P1 消融，禁硬编码）。
        if cfg.decoder_init == "siglip":
            blocks = parts.dec_blocks
        elif cfg.decoder_init == "random":
            blocks = [UVTBlock(dim, num_heads, cfg.rope_dims) for _ in parts.dec_blocks]
        else:
            raise ValueError(f"decoder_init 必须是 siglip/random，实得 {cfg.decoder_init}")
        self.blocks = nn.ModuleList(blocks)

        up = tuple(cfg.unfold_positions)
        assert len(up) == 2, f"unfold_positions 必须恰为 2 个（镜像 2×2 折叠），实得 {up}"
        assert all(0 <= p <= len(self.blocks) for p in up), \
            f"unfold_positions {up} 越界（合法域 [0, {len(self.blocks)}]，上界=末块之后）"
        assert up == tuple(sorted(up)), f"unfold_positions 必须升序，实得 {up}"
        self.unfold_positions = up
        self.unfolds = nn.ModuleList([TemporalUnfold2x(dim) for _ in up])

        # 反投影 64→D：decoder 自持（M-5 裁决，与 SemViT 的 in_proj 互相独立）。
        self.in_proj = nn.Linear(cfg.c_latent, dim)

        # patch 尺寸从骨干 embeddings 读取（全案协议 p=16，§0；不硬编码以兼容 tiny 臂）。
        p = getattr(parts.embeddings, "patch_size", None)
        if p is None:
            p = parts.embeddings.patch_embedding.kernel_size[0]
        self.patch_size = int(p)

        # 末端规范化（第 15 号修复，docs/08 §6.5 / docs/06 §6.8）：末 block 输出是
        # SigLIP2 初始化残差流（真权重下尺度 O(10²)），直入随机初始化像素头会产出
        # ±44 的 x_hat（目标 [0,1]）。SigLIP2 原序 encoder→post_layernorm→head，
        # Sem-ViT 已镜像，decoder 此处补齐（新建 LN，不与 sem_vit.post_ln 共享参数）。
        self.final_ln = nn.LayerNorm(dim)

        # 锚帧/非锚帧独立像素头（随机初始化，01 §2.7）。展开已把时间位还原到逐帧，
        # 故两头输出维一致均为 3·p·p（M-8 卡），无需时间 patch 因子。
        self.head_anchor = nn.Linear(dim, 3 * self.patch_size * self.patch_size)
        self.head_frame = nn.Linear(dim, 3 * self.patch_size * self.patch_size)
        # 像素头校准初始化（第 15 号修复配套）：目标值域 [0,1]、均值 ~0.5，
        # weight 小尺度 + bias=0.5 使初始输出 ≈ 灰图（L1 起点 ~0.25，梯度方向干净）。
        for head in (self.head_anchor, self.head_frame):
            nn.init.normal_(head.weight, std=0.02)
            nn.init.constant_(head.bias, 0.5)

        # ADR-8 消融臂：rope_dims=0 → time_ids=None。
        self.use_rope = cfg.rope_dims > 0

    def _run_block(self, block, x: torch.Tensor, attn_mode: str) -> torch.Tensor:
        """[B,T1,N,D] 展平过 block 再复原；每次展开后 T1 变化，bias/time_ids 现取（M-1 缓存）。"""
        B, T1, N, D = x.shape
        bias = attn_bias(T1, N, attn_mode, x.device, x.dtype)
        tids = make_time_ids(T1, N, x.device) if self.use_rope else None
        y = block(x.reshape(B, T1 * N, D), bias, tids)
        return y.reshape(B, T1, N, D)

    def forward(self, z_phys: torch.Tensor, hw, attn_mode: str = "tubelet") -> torch.Tensor:
        """z_phys:[B,T1z,N,c_latent] -> x̂:[B,3,1+T,H,W]（图像时 [B,3,1,H,W]）。"""
        assert attn_mode in VALID_KINDS, f"unknown attn_mode: {attn_mode}"
        H, W = hw
        p = self.patch_size
        gh, gw = H // p, W // p
        B, T1z, N, _ = z_phys.shape
        assert N == gh * gw, f"N={N} 必须等于 (H/{p})*(W/{p})={gh * gw}（M-8 卡形状契约）"

        x = self.in_proj(z_phys)                            # [B,T1z,N,D]

        # 展开位循环：位置 i 的展开发生在第 i 个 block 之前；len(blocks)=末块之后。
        for i, block in enumerate(self.blocks):
            for j, pos in enumerate(self.unfold_positions):
                if pos == i:
                    x = self.unfolds[j](x)                  # 锚帧直通（ADR-4'，Unfold 内保证）
            x = self._run_block(block, x, attn_mode)
        for j, pos in enumerate(self.unfold_positions):
            if pos == len(self.blocks):
                x = self.unfolds[j](x)

        # 末端规范化后进像素头（第 15 号修复：镜像 SigLIP2 encoder→post_ln→head 原序）。
        x = self.final_ln(x)

        # 双像素头：锚帧与非锚帧独立线性投影（T1z=1 时 frames 切片为空，cat 天然退化）。
        pix_anchor = self.head_anchor(x[:, :1])             # [B,1,N,3pp]
        pix_frames = self.head_frame(x[:, 1:])              # [B,T,N,3pp]
        pix = torch.cat([pix_anchor, pix_frames], dim=1)    # [B,1+T,N,3pp]

        # patch → 像素：N=(gh·gw) 网格按行优先展开（与 SigLIP2 patchify 的 flatten 顺序一致）。
        x_hat = rearrange(
            pix, "b f (gh gw) (c ph pw) -> b c f (gh ph) (gw pw)",
            gh=gh, gw=gw, c=3, ph=p, pw=p,
        )
        return x_hat                                        # [B,3,1+T,H,W]
