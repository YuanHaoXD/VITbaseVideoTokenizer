"""SigLIP2 兼容的 transformer block（M-3）——全模型的原子构件。

设计目标：一块既能**深拷贝继承 SigLIP2 预训练权重**（Gen/Sem-ViT、decoder 的初始化），
又能挂载自定义 mask（M-1 的加性 bias）与**时间 RoPE**（ADR-8）。

关键纪律（任务书 §2 M-3）：
  - 不直接调用 HF `SiglipEncoderLayer.forward`：其 mask 语义/注意力后端随 transformers
    版本漂移，自持 q/k/v/out/ln/mlp 权重才数值可控；
  - 注意力统一走 `F.scaled_dot_product_attention`（与 M-1 的加性 bias 天然对接，规避
    flex_attention 依赖，见 §0 禁止事项）；
  - pre-LN 结构与 SigLIP2 逐位对齐：ln1→attn→残差, ln2→mlp→残差。

张量约定：进本模块前已展平为 [B, S, D]（S = T1*N，时间位内空间 token 连续，见 §0）。
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

# ADR-8：时间 RoPE 的分频底数。冻结签名只暴露 rope_dims，故底数作模块常量 + 实例属性
# （可在构造后覆写 `attn.rope_theta` 做消融），而非埋进 forward 里的魔数。10000 为 RoPE 惯例。
ROPE_THETA = 10000.0


class UVTAttention(nn.Module):
    """多头自注意力：加性 attn_bias + 可选时间 RoPE（仅每头前 rope_dims 维旋转）。

    rope_dims=0 或 time_ids=None 时退化为无位置注意力（消融臂开关，ADR-8）。
    """

    def __init__(self, dim: int, num_heads: int, rope_dims: int = 32):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} 必须能被 num_heads {num_heads} 整除"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # So400m: 1152/16 = 72
        assert rope_dims % 2 == 0, "rope_dims 必须为偶数（rotate-half 成对旋转）"
        assert rope_dims <= self.head_dim, f"rope_dims {rope_dims} 不能超过 head_dim {self.head_dim}"
        self.rope_dims = rope_dims
        self.rope_theta = ROPE_THETA  # ADR-8 决策点，暴露为可覆写属性而非硬编码

        # 与 SigLIP2 `self_attn.{q,k,v,out}_proj` 逐一对应（含 bias），from_siglip 深拷贝替换。
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        if rope_dims > 0:
            # inv_freq: [rope_dims/2]，1/θ^(2i/rope_dims)。persistent=False：不写入 state_dict，
            # 避免污染权重迁移（from_siglip 的 allclose 校验）与 checkpoint 键集。
            inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, rope_dims, 2).float() / rope_dims))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """rotate-half 惯例（LLaMA/HF 同款）：[x1, x2] -> [-x2, x1]。"""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def _rope_tables(self, time_ids: torch.Tensor, dtype: torch.dtype, device: torch.device):
        """由时间位 id 生成 cos/sin 表 [1,1,S,rope_dims]。角度=time_id×inv_freq。
        在 float32 下算三角函数再转回目标 dtype（bf16/fp16 精度保护）。"""
        inv_freq = self.inv_freq.to(device=device)
        freqs = torch.outer(time_ids.to(torch.float32), inv_freq)  # [S, rope_dims/2]
        emb = torch.cat([freqs, freqs], dim=-1)                    # [S, rope_dims]
        cos = emb.cos().to(dtype)[None, None]                      # [1,1,S,rope_dims]
        sin = emb.sin().to(dtype)[None, None]
        return cos, sin

    def _apply_rope(self, t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """仅对每头前 rope_dims 维做 1D rotary，其余维（head_dim-rope_dims）原样直通。"""
        t_rot, t_pass = t[..., : self.rope_dims], t[..., self.rope_dims:]
        t_rot = t_rot * cos + self._rotate_half(t_rot) * sin
        return torch.cat([t_rot, t_pass], dim=-1)

    def forward(self, x: torch.Tensor, attn_bias, time_ids) -> torch.Tensor:
        # x: [B, S, D]；attn_bias: [1,1,S,S] 或 None（来自 M-1）；time_ids: [S] 或 None。
        B, S, _ = x.shape
        # [B,S,D] -> [B,H,S,head_dim]，头切分方式与 SigLIP2 SiglipAttention 一致。
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        if time_ids is not None and self.rope_dims > 0:
            cos, sin = self._rope_tables(time_ids, x.dtype, x.device)
            q = self._apply_rope(q, cos, sin)
            k = self._apply_rope(k, cos, sin)

        # SDPA 默认 scale=head_dim**-0.5，与 SigLIP2 的 self.scale 等价；attn_bias 为加性 mask。
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)  # [B,H,S,head_dim]
        out = out.transpose(1, 2).reshape(B, S, self.dim)
        return self.out_proj(out)


class UVTBlock(nn.Module):
    """pre-LN transformer block：x + attn(ln1(x)); x + mlp(ln2(x))。

    子模块命名（layer_norm1/layer_norm2/mlp/attn）刻意对齐 SigLIP2，便于 from_siglip 深拷贝替换。
    """

    def __init__(self, dim: int, num_heads: int, rope_dims: int = 32,
                 mlp_ratio: float = 4.0, layer_norm_eps: float = 1e-6):
        super().__init__()
        self.attn = UVTAttention(dim, num_heads, rope_dims)
        self.layer_norm1 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(dim, eps=layer_norm_eps)
        # 默认 MLP：仅用于"非 from_siglip"的新建 block（tiny 测试臂）；from_siglip 会整体替换。
        # gelu(tanh 近似) 对齐 SigLIP2 的 gelu_pytorch_tanh。
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor, attn_bias, time_ids) -> torch.Tensor:
        x = x + self.attn(self.layer_norm1(x), attn_bias, time_ids)
        x = x + self.mlp(self.layer_norm2(x))
        return x

    @classmethod
    def from_siglip(cls, layer, rope_dims: int = 32) -> "UVTBlock":
        """从 HF `SiglipEncoderLayer` 深拷贝权重构造。

        深拷贝而非引用/浅赋值：每个 UVTBlock 必须自持独立权重（Gen/Sem/dec 三份互不共享，
        否则 Stage-2/3 的分组冻结会在共享张量上互相打架，见任务书 §2.5 / 01 §2.7）。
        不调用 layer.forward——只搬运权重张量，注意力/mask 语义由本类自持实现。

        rope_dims：架构方契约修订 2026-07-07（任务书 M-3 卡 v1.1）——原冻结签名缺此参数，
        导致 ADR-8 的 rope_dims=0 消融臂无法经此路径构造，现补为可选参（默认值不变）。
        """
        attn = layer.self_attn
        dim = attn.q_proj.in_features                      # 从权重反推宽度，不依赖易漂移的属性名
        num_heads = getattr(attn, "num_heads", None)
        if num_heads is None:                              # 老/新版本属性兜底
            num_heads = attn.config.num_attention_heads
        block = cls(dim, num_heads, rope_dims)

        block.attn.q_proj = copy.deepcopy(attn.q_proj)
        block.attn.k_proj = copy.deepcopy(attn.k_proj)
        block.attn.v_proj = copy.deepcopy(attn.v_proj)
        block.attn.out_proj = copy.deepcopy(attn.out_proj)
        block.layer_norm1 = copy.deepcopy(layer.layer_norm1)
        block.layer_norm2 = copy.deepcopy(layer.layer_norm2)
        block.mlp = copy.deepcopy(layer.mlp)
        return block
