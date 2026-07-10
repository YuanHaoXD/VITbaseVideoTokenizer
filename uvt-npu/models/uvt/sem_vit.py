"""SemViT · 语义读出支路（M-7，张量流见 01 §1 步骤⑧⑨）。

结构：in_proj Linear(c_latent→D)（M-5 裁决：反投影拆到消费方，Sem-ViT 与 decoder
各自独立持有，避免 Stage-3 "冻 decoder 训 Sem 侧" 在共享投影上打架）
  → sem blocks（SigLIP2 第 gen_depth+1..27 层，tubelet 因果注意力）
  → post_ln → MAP 注意力池化（ADR-9：对齐 SigLIP2 池化嵌入，zero-shot 的关键）。

接口约定（M-7 卡）：
  - forward 消费**规范化** latent z_can（ADR-5：Stage 3 起 ẑ=(z-m)/s，之前直通）；
  - s_pool 只对锚帧 token s[:,0] 过 map_head——图像级语义接口，zero-shot 用它；
  - forward_pair 是 STI 双路由（01 §2.10）：拼 T1=2 用 tubelet mask，位 0=源只见
    自己（因果保证由 mask 天然给出）；对照组 Indep 由评测脚本对两图分别调 forward
    路由，**本类不单写 Indep 分支**。
"""
import torch
import torch.nn as nn

from .attention_mask import VALID_KINDS, attn_bias, make_time_ids
from .siglip_backbone import BackboneParts


class SemViT(nn.Module):
    def __init__(self, parts: BackboneParts, cfg):
        super().__init__()
        dim = parts.post_ln.normalized_shape[0]
        # 反投影 64→D：Sem-ViT 自持（M-5 裁决，HYDRA W_unproj^und 上标语义）。
        self.in_proj = nn.Linear(cfg.c_latent, dim)
        self.blocks = nn.ModuleList(parts.sem_blocks)  # M-4 交付普通 list，包 ModuleList 注册
        self.post_ln = parts.post_ln
        assert parts.map_head is not None, \
            "SemViT 需要 MAP 池化头（ADR-9 zero-shot 依赖）；请用 vision_use_head=True 的骨干"
        self.map_head = parts.map_head
        # ADR-8 消融臂：rope_dims=0 → time_ids=None。
        self.use_rope = cfg.rope_dims > 0

    def forward(self, z_can: torch.Tensor, attn_mode: str = "tubelet"):
        """z_can:[B,T1,N,c_latent] -> (s:[B,T1,N,D], s_pool:[B,D])。"""
        assert attn_mode in VALID_KINDS, f"unknown attn_mode: {attn_mode}"
        B, T1, N, _ = z_can.shape
        x = self.in_proj(z_can)                            # [B,T1,N,D]
        D = x.shape[-1]

        # Sem 支路内 T1 不再变化，bias/time_ids 取一次即可（M-1 有缓存）。
        bias = attn_bias(T1, N, attn_mode, x.device, x.dtype)
        tids = make_time_ids(T1, N, x.device) if self.use_rope else None
        x = x.reshape(B, T1 * N, D)
        for block in self.blocks:
            x = block(x, bias, tids)
        x = self.post_ln(x)                                # SigLIP2 原序：encoder→post_ln→head
        s = x.reshape(B, T1, N, D)

        # ADR-9：s_pool 只池化锚帧 token（图像级语义接口）。
        # SiglipMultiheadAttentionPoolingHead: [B,N,D] -> [B,D]。
        s_pool = self.map_head(s[:, 0])
        return s, s_pool

    def forward_pair(self, z_src: torch.Tensor, z_tgt: torch.Tensor):
        """STI 联合路由（01 §2.10）：各 [B,1,N,c_latent] 拼成 T1=2 过 tubelet。

        tubelet mask 下位 0（源）只见自己 → s_src 与单独 forward(z_src) 的 s[:,0]
        逐位相等（验收 test_sti_causality）；位 1（目标）可见源+自己。
        mask 固定 tubelet——因果性是 STI 结论成立的前提，不随 cfg.attn_mode 走。
        """
        assert z_src.shape[1] == 1 and z_tgt.shape[1] == 1, \
            f"forward_pair 只接受单时间位输入，实得 {z_src.shape[1]}/{z_tgt.shape[1]}"
        pair = torch.cat([z_src, z_tgt], dim=1)            # [B,2,N,c_latent]
        s, _ = self.forward(pair, attn_mode="tubelet")
        return s[:, 0], s[:, 1]                            # (s_src, s_tgt) 各 [B,N,D]
