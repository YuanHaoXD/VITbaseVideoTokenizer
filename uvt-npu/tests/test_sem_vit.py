"""M-7 验收测试：models/uvt/sem_vit.py（SemViT）。

任务书 §2 M-7 卡验收项：
- test_sti_causality：forward_pair(z_src, z_tgt) 的 s_src 与单独 forward(z_src)
  的 s[:,0] allclose（tubelet mask 下位 0=源只见自己，证明源侧无目标信息泄漏；
  03 篇 STI 指标③的单测版）。

离线：tiny 骨干，禁下载权重。UVTConfig 未实现（M-10），用 SimpleNamespace 伪造
（SemViT 只消费 cfg.c_latent 与 cfg.rope_dims）。

tiny 几何：6 层；用 gen_depth=3 切分 → gen=3, sem=3（sem_blocks 必须非空，
否则 SemViT 无法构造有意义输出）。hidden_size=64 → D=64；N=16。
"""
from types import SimpleNamespace

import pytest
import torch

from models.uvt.sem_vit import SemViT
from models.uvt.siglip_backbone import load_siglip_parts


def _tiny_cfg():
    """SemViT 只消费 cfg.c_latent 与 cfg.rope_dims；attn_mode 由 forward 参数控制。"""
    return SimpleNamespace(c_latent=64, rope_dims=0, attn_mode="tubelet", tiny=True)


def test_sti_causality():
    """STI 联合路由下，源侧输出 s_src 应与单独 forward 源图的 s[:,0] 逐位相等——
    tubelet mask 保证位 0（源）只见自己，源侧无目标信息泄漏。"""
    torch.manual_seed(0)
    # gen_depth=3：6 层 tiny 切成 gen=3, sem=3（sem_blocks 非空）。
    parts = load_siglip_parts(tiny=True, gen_depth=3)
    sem = SemViT(parts, _tiny_cfg())
    sem.eval()

    B, N, c = 2, 16, 64
    z_src = torch.randn(B, 1, N, c)
    z_tgt = torch.randn(B, 1, N, c)

    with torch.no_grad():
        s_src_pair, s_tgt_pair = sem.forward_pair(z_src, z_tgt)   # 各 [B,N,D]
        s_single, s_pool_single = sem.forward(z_src)              # s [B,1,N,D]

    # forward_pair 单时间位输入断言已由模块内 assert 保证；此处校验输出形状。
    assert s_src_pair.shape == s_tgt_pair.shape == (B, N, 64), \
        f"forward_pair 输出形状错: src={tuple(s_src_pair.shape)} tgt={tuple(s_tgt_pair.shape)}"
    assert s_single[:, 0].shape == (B, N, 64), \
        f"forward 单图 s[:,0] 形状错: {tuple(s_single[:, 0].shape)}"

    # 核心断言：源侧无泄漏（位 0 只见自己）
    assert torch.allclose(s_src_pair, s_single[:, 0], atol=1e-5), \
        f"源侧泄漏：s_src 与单独 forward 的 s[:,0] 偏差 " \
        f"max|Δ|={(s_src_pair - s_single[:, 0]).abs().max().item()}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
