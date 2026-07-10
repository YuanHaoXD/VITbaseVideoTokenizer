"""M-5 验收测试：models/uvt/gsb.py。

- test_gsb_roundtrip：规范化往返 from_canonical(to_canonical(z)) ≈ z。
- test_kl_zero_at_standard_normal：μ=0, ρ=0 时 KL≈0。

纯离线，只依赖 torch。
"""
import pytest
import torch

from models.uvt.gsb import GSB


def test_gsb_roundtrip():
    torch.manual_seed(0)
    gsb = GSB(d_model=32, c_latent=8)
    z = torch.randn(2, 3, 5, 8)  # [B, T1, N, c_latent]

    # normalize=False：两个方向都是直通。
    assert torch.allclose(gsb.to_canonical(z), z)
    assert torch.allclose(gsb.from_canonical(z), z)

    # normalize=True + 非平凡通道统计：往返应还原。
    gsb.z_mean.copy_(torch.randn(8))
    gsb.z_std.copy_(torch.rand(8) + 0.5)  # 保证 > 0
    gsb.normalize = True
    z_can = gsb.to_canonical(z)
    z_back = gsb.from_canonical(z_can)
    assert not torch.allclose(z_can, z), "normalize=True 时规范化应真的改变数值"
    assert torch.allclose(z_back, z, atol=1e-5), \
        f"规范化往返未还原: max|Δ|={ (z_back - z).abs().max().item() }"


def test_kl_zero_at_standard_normal():
    """把 proj 权重/偏置清零 → μ=0, ρ=0 → KL = -0.5·mean(1+0-0-1) = 0。"""
    torch.manual_seed(0)
    gsb = GSB(d_model=32, c_latent=8)
    with torch.no_grad():
        gsb.proj.weight.zero_()
        gsb.proj.bias.zero_()

    h = torch.randn(4, 2, 6, 32)
    z, mu, kl = gsb.compress(h)

    assert torch.allclose(mu, torch.zeros_like(mu)), "μ 应恒为 0"
    assert abs(kl.item()) < 1e-6, f"标准正态处 KL 应≈0，实得 {kl.item()}"
    assert z.shape == (4, 2, 6, 8)


def test_compress_shapes_and_finite():
    """健壮性：压缩输出形状与有限性（ρ clamp 防溢出）。"""
    torch.manual_seed(0)
    gsb = GSB(d_model=32, c_latent=8)
    h = torch.randn(2, 5, 4, 32) * 100.0  # 大幅值也不应产生 NaN/Inf
    z, mu, kl = gsb.compress(h)
    assert z.shape == mu.shape == (2, 5, 4, 8)
    assert torch.isfinite(z).all() and torch.isfinite(mu).all() and torch.isfinite(kl).all()

    # 确认 GSB 无反投影方法（M-5 裁决：expand/unproj 已删，归各消费方）。
    assert not hasattr(gsb, "expand")
    assert not hasattr(gsb, "unproj")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
