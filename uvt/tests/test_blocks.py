"""M-3 验收测试：models/uvt/blocks.py（UVTAttention / UVTBlock）。

- test_block_weight_transfer：from_siglip 深拷贝的 UVTBlock 与原 SigLIP 层在
  time_ids=None、bias=None 下对同一输入 allclose(atol=1e-5)。
- test_rope_shift_equivariance：整体平移 time_ids 不改变注意力（RoPE 相对位置性质）。

离线：用随机初始化的 tiny SiglipVisionModel（不下载权重）。transformers 导入置于函数内。
"""
import pytest
import torch

from models.uvt.blocks import UVTAttention, UVTBlock


def _tiny_siglip_layer():
    """构造一个随机初始化的 tiny SigLIP 视觉层（第 0 层），离线可用。"""
    from transformers import SiglipVisionConfig, SiglipVisionModel

    config = SiglipVisionConfig(
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=128,
        image_size=64,
        patch_size=16,
    )
    vision = SiglipVisionModel(config)
    # transformers ≤5.12 包一层 .vision_model；5.13+ 移除（直接挂属性）。两者择一。
    vision = vision.vision_model if hasattr(vision, "vision_model") else vision
    vision.eval()
    return vision.encoder.layers[0]


def _run_siglip_layer(layer, x):
    """兼容不同 transformers 版本：forward 可能返回 tensor 或 tuple。"""
    out = layer(x, attention_mask=None)
    return out[0] if isinstance(out, tuple) else out


def test_block_weight_transfer():
    torch.manual_seed(0)
    layer = _tiny_siglip_layer()
    block = UVTBlock.from_siglip(layer)
    block.eval()

    x = torch.randn(2, 10, 64)  # [B, S, D]
    with torch.no_grad():
        ref = _run_siglip_layer(layer, x)
        got = block(x, None, None)  # 无 mask、无时间 RoPE：应与原层数值一致

    assert got.shape == ref.shape == (2, 10, 64)
    assert torch.allclose(got, ref, atol=1e-5), \
        f"权重迁移后输出偏差过大: max|Δ|={ (got - ref).abs().max().item() }"


def test_rope_shift_equivariance():
    """RoPE 只依赖相对位置：整体平移 time_ids，block 输出应不变。"""
    torch.manual_seed(0)
    # head_dim=32，rope_dims=16 → 同时覆盖旋转维与直通维两条路径。
    block = UVTBlock(dim=64, num_heads=2, rope_dims=16)
    block.eval()

    B, T1, N = 2, 4, 3
    S = T1 * N
    x = torch.randn(B, S, 64)
    time_ids = torch.arange(T1).repeat_interleave(N)  # [S]，与 M-1 make_time_ids 一致

    with torch.no_grad():
        out0 = block(x, None, time_ids)
        out_shift = block(x, None, time_ids + 5)  # 整体平移常量

    assert torch.allclose(out0, out_shift, atol=1e-5), \
        f"time_ids 整体平移改变了输出: max|Δ|={ (out0 - out_shift).abs().max().item() }"


def test_rope_actually_applied():
    """健壮性：time_ids 非平凡变化应改变输出（确认 RoPE 真的生效，而非被跳过）。"""
    torch.manual_seed(0)
    block = UVTBlock(dim=64, num_heads=2, rope_dims=16)
    block.eval()
    x = torch.randn(1, 6, 64)
    ids_a = torch.tensor([0, 0, 0, 1, 1, 1])
    ids_b = torch.tensor([0, 1, 2, 3, 4, 5])
    with torch.no_grad():
        out_none = block(x, None, None)
        out_a = block(x, None, ids_a)
        out_b = block(x, None, ids_b)
    assert not torch.allclose(out_none, out_a, atol=1e-5)
    assert not torch.allclose(out_a, out_b, atol=1e-5)


def test_attn_bias_masks_attention():
    """加性 bias（-inf 项）应真正屏蔽被禁 token（与 M-1 的 SDPA 路径对接）。"""
    torch.manual_seed(0)
    attn = UVTAttention(dim=64, num_heads=2, rope_dims=0)
    attn.eval()
    x = torch.randn(1, 4, 64)
    # 位置 0 只允许看自己，其余全禁；对比 full 应产生不同输出。
    neg = torch.finfo(x.dtype).min
    bias = torch.full((1, 1, 4, 4), 0.0)
    bias_masked = bias.clone()
    bias_masked[0, 0, 0, 1:] = neg
    with torch.no_grad():
        out_full = attn(x, bias, None)
        out_masked = attn(x, bias_masked, None)
    assert not torch.allclose(out_full[:, 0], out_masked[:, 0], atol=1e-5)


def test_rope_dims_zero_skips():
    """rope_dims=0（消融臂）：即便传入 time_ids 也不应改变输出。"""
    torch.manual_seed(0)
    block = UVTBlock(dim=64, num_heads=2, rope_dims=0)
    block.eval()
    x = torch.randn(1, 6, 64)
    ids = torch.tensor([0, 1, 2, 3, 4, 5])
    with torch.no_grad():
        out_none = block(x, None, None)
        out_ids = block(x, None, ids)
    assert torch.allclose(out_none, out_ids, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
