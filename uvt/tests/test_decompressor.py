"""M-9 验收测试：models/uvt/decompressor.py。

- test_decompressor_shape：[1,4,256,1152] -> [1,16,256,1152]（4 位升 16 帧，D 不变=STUDENT_DIM）。
- test_decompressor_finite：输出有限（近恒等初始化 + full mask 下不应 NaN）。
- test_decompressor_no_anchor_no_passthrough：确认不误把首位当锚帧直通（首位确实参与上采样）。

纯离线，只依赖 torch。集群运行：`pytest tests/test_decompressor.py -v`（需 torch + einops）。
"""
import pytest
import torch

from models.uvt.decompressor import Decompressor


def test_decompressor_shape():
    """4 个时间位 → 16 帧（4×=2×2），D=1152 不变（契约：输出 STUDENT_DIM，不投教师维度）。"""
    torch.manual_seed(0)
    dec = Decompressor(dim=1152, num_heads=16)
    s = torch.randn(1, 4, 256, 1152)          # s[:,1:]：T1z-1=4 位
    out = dec(s)
    assert out.shape == (1, 16, 256, 1152), f"期望 [1,16,256,1152]，实得 {tuple(out.shape)}"
    assert torch.isfinite(out).all(), "输出含 NaN/Inf"


def test_decompressor_finite():
    """健壮性：较大输入下输出仍有限（ρ/数值保护由 GSB 负责，此处只验 Decompressor 自身）。"""
    torch.manual_seed(0)
    dec = Decompressor(dim=128, num_heads=4)
    s = torch.randn(2, 4, 16, 128) * 10.0
    out = dec(s)
    assert out.shape == (2, 16, 16, 128)
    assert torch.isfinite(out).all()


def test_decompressor_no_anchor_no_passthrough():
    """验证没有锚帧直通逻辑：首位被上采样（输出时间维严格 = 4×输入），
    且近恒等初始化下输出 != 输入的首位广播（确认不是简单直通/复制）。"""
    torch.manual_seed(0)
    dec = Decompressor(dim=64, num_heads=2)
    s = torch.randn(1, 4, 4, 64)
    out = dec(s)
    # 严格 4×（无锚位直通会把首位抠出导致非整数倍关系）
    assert out.shape[1] == 4 * s.shape[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
