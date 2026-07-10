"""M-8 验收测试：models/uvt/decoder.py（PixelDecoder）。

任务书 §2 M-8 卡验收项：
- test_decoder_shapes：latent [2,5,16,c]（1+4 位）→ 像素 [2,3,17,64,64]
- test_image_roundtrip_shape：图像 latent [2,1,16,c] → 像素 [2,3,1,64,64] 形状闭环

离线：tiny 骨干，禁下载权重。UVTConfig 未实现（M-10），用 SimpleNamespace 伪造
（decoder 消费 cfg.decoder_init / unfold_positions / c_latent / rope_dims）。

tiny 几何：6 层全深拷贝为 dec_blocks；hidden_size=64 → D=64；patch=16 → N=16；
unfold_positions=(0,3)：pos0 在输入处展开 5→9，pos3 在第 3 块前展开 9→17。
"""
from types import SimpleNamespace

import pytest
import torch

from models.uvt.decoder import PixelDecoder
from models.uvt.siglip_backbone import load_siglip_parts


def _tiny_cfg(**over):
    """decoder 消费 decoder_init / unfold_positions / c_latent / rope_dims。"""
    base = dict(
        decoder_init="siglip",
        unfold_positions=(0, 3),
        c_latent=64,
        rope_dims=0,                # ADR-8 消融臂
        attn_mode="tubelet",
        tiny=True,
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_decoder_shapes():
    """视频 latent [2,5,16,64]（1+4 位）→ 两次 2× 展开 → 像素 [2,3,17,64,64]。"""
    torch.manual_seed(0)
    parts = load_siglip_parts(tiny=True, gen_depth=3)   # dec_blocks 恒为 6 层（全深拷贝）
    dec = PixelDecoder(parts, _tiny_cfg())
    dec.eval()

    z_phys = torch.randn(2, 5, 16, 64)                  # [B, T1z=1+4, N=16, c_latent=64]
    with torch.no_grad():
        x_hat = dec(z_phys, hw=(64, 64))
    # 两次 2× 展开：1+4 → 1+8=9 → 1+16=17 帧；patch=16 还原 64×64
    assert x_hat.shape == (2, 3, 17, 64, 64), \
        f"视频像素重建形状错: {tuple(x_hat.shape)}"


def test_image_roundtrip_shape():
    """图像 latent [2,1,16,64] → 像素 [2,3,1,64,64]。
    单锚位时 TemporalUnfold2x 的 frames 切片为空 → 两次展开均 no-op（ADR-4' 锚帧隔离）。"""
    torch.manual_seed(0)
    parts = load_siglip_parts(tiny=True, gen_depth=3)
    dec = PixelDecoder(parts, _tiny_cfg())
    dec.eval()

    z_img = torch.randn(2, 1, 16, 64)
    with torch.no_grad():
        x_hat = dec(z_img, hw=(64, 64))
    assert x_hat.shape == (2, 3, 1, 64, 64), \
        f"图像像素重建形状错（形状闭环失败）: {tuple(x_hat.shape)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
