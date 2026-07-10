"""M-6 验收测试：models/uvt/encoder.py（GenViT）。

任务书 §2 M-6 卡验收项：
- test_encoder_shapes：视频 [2,3,17,64,64] → h [2,1+4,16,D]；纯图像 → h.shape[1]==1
- test_fold_position_config：(0,4) 配置下第 4 块前折叠生效（hook 检查各 block 输入 T1）

离线：tiny 骨干（load_siglip_parts(tiny=True)），禁下载权重。
UVTConfig 尚未实现（M-10 进行中），用 SimpleNamespace 伪造所需 cfg 字段
（fold_positions / rope_dims；encoder 仅读这两个）。

tiny 几何：image_size=64, patch=16 → 网格 4×4 → N=16；hidden_size=64 → D=64；
num_hidden_layers=6，默认 gen_depth=13 夹断到 6（与 test_backbone 一致）。
"""
from types import SimpleNamespace

import pytest
import torch

from models.uvt.encoder import GenViT
from models.uvt.siglip_backbone import load_siglip_parts


def _tiny_cfg(**over):
    """encoder 只消费 cfg.fold_positions 与 cfg.rope_dims；其余字段为对齐 UVTConfig 预留。"""
    base = dict(
        fold_positions=(0, 3),
        rope_dims=0,           # 关 RoPE：tiny 单测简化路径（ADR-8 消融臂）
        c_latent=64,
        attn_mode="tubelet",
        tiny=True,
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_encoder_shapes():
    """视频 17 帧 → 两次 2× 折叠 → 1+T/4 = 5 位；纯图像入口升维后只剩锚位。"""
    torch.manual_seed(0)
    parts = load_siglip_parts(tiny=True)               # 6 gen blocks（默认 gen_depth 夹断）
    enc = GenViT(parts, _tiny_cfg())
    enc.eval()

    # 视频 [B,3,1+T,H,W]：T=16，T%4==0 → 折叠两次到 1+T/4 = 5 位
    video = torch.randn(2, 3, 17, 64, 64)
    with torch.no_grad():
        h = enc(video)
    # tiny: hidden_size=64（D），image_size=64/patch=16 → N=(64/16)²=16
    assert h.shape == (2, 5, 16, 64), f"视频 latent 形状错: {tuple(h.shape)}"
    assert h.shape[1] == 1 + (17 - 1) // 4, "时间位应为 1+T/4（两次 2× 折叠）"

    # 纯图像 [B,3,H,W]：入口升维 [B,3,1,H,W]，折叠对单锚位是 no-op
    image = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        h_img = enc(image)
    assert h_img.shape[1] == 1, f"纯图像应只剩 1 个时间位，实得 {h_img.shape[1]}"
    assert h_img.shape == (2, 1, 16, 64), f"图像 latent 形状错: {tuple(h_img.shape)}"


def test_fold_position_config():
    """fold_positions=(0,4)：pos0 在输入处折叠（17→9），pos4 在第 4 块前折叠（9→5）。
    用 forward_pre_hook 抓各 block 输入 [B, T1·N, D]，T1 = shape[1] // N。
    期望：block 0-3 见 T1=9，block 4-5 见 T1=5。"""
    torch.manual_seed(0)
    parts = load_siglip_parts(tiny=True)               # 6 gen blocks
    cfg = _tiny_cfg(fold_positions=(0, 4))
    enc = GenViT(parts, cfg)
    enc.eval()

    N = 16
    seen = {}

    def make_hook(idx):
        def hook(_module, inp):
            x = inp[0]                                  # [B, T1*N, D]（block 展平输入）
            seen[idx] = x.shape[1] // N
        return hook

    handles = [blk.register_forward_pre_hook(make_hook(i))
               for i, blk in enumerate(enc.blocks)]

    video = torch.randn(1, 3, 17, 64, 64)
    try:
        with torch.no_grad():
            enc(video)
    finally:
        for h in handles:
            h.remove()

    # pos0 折叠在 block 0 之前生效：所有 block 都看不到 T1=17
    assert seen[0] == 9, f"pos0 应在 block0 前折叠 17→9，实得 T1={seen[0]}"
    # block 0-3 见 T1=9（pos4 折叠尚未发生）
    pre = [seen[i] for i in range(0, 4)]
    assert all(t == 9 for t in pre), f"block 0-3 应见 T1=9，实得 {pre}"
    # block 4-5 见 T1=5（pos4 折叠后）
    post = [seen[i] for i in range(4, 6)]
    assert all(t == 5 for t in post), f"block 4-5 应见 T1=5，实得 {post}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
