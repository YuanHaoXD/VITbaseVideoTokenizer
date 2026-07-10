"""M-4 验收测试：models/uvt/siglip_backbone.py。

- test_load_split：层数分配 13/14/27（非 tiny 路径，monkeypatch from_pretrained 返回
  一个 27 层/1152 宽的随机小模型，离线跑通全部断言与切分算术）。
- test_tiny_offline：无网络环境构建成功（tiny=True）。

均离线，不下载任何权重。transformers 导入置于函数内。
"""
import pytest
import torch.nn as nn

from models.uvt.blocks import UVTBlock
from models.uvt.siglip_backbone import BackboneParts, load_siglip_parts


def test_load_split(monkeypatch):
    """非 tiny 路径：13/14/27 切分 + 断言。用 monkeypatch 造一个满足断言（27 层、
    1152 宽）的随机小模型顶替 from_pretrained，故无需下载真权重。"""
    import transformers
    from transformers import SiglipVisionConfig, SiglipVisionModel

    # 满足 load_siglip_parts 的两处断言（27 层、hidden_size==1152），其余尺寸压到最小。
    fake_config = SiglipVisionConfig(
        hidden_size=1152,
        num_hidden_layers=27,
        num_attention_heads=16,
        intermediate_size=64,
        image_size=64,
        patch_size=16,
    )

    def fake_from_pretrained(*args, **kwargs):
        return SiglipVisionModel(fake_config)

    monkeypatch.setattr(transformers.SiglipVisionModel, "from_pretrained",
                        staticmethod(fake_from_pretrained))

    parts = load_siglip_parts(gen_depth=13, tiny=False)

    assert isinstance(parts, BackboneParts)
    assert len(parts.gen_blocks) == 13
    assert len(parts.sem_blocks) == 14
    assert len(parts.dec_blocks) == 27
    assert all(isinstance(b, UVTBlock) for b in parts.gen_blocks + parts.sem_blocks + parts.dec_blocks)
    assert isinstance(parts.embeddings, nn.Module)
    assert isinstance(parts.post_ln, nn.Module)
    assert isinstance(parts.map_head, nn.Module)


def test_tiny_offline():
    """tiny=True 离线构建成功；验证切分算术与 dec_blocks 的独立深拷贝。"""
    parts = load_siglip_parts(tiny=True, gen_depth=3)

    # 6 层 tiny：gen=3, sem=3, dec=6。
    assert len(parts.gen_blocks) == 3
    assert len(parts.sem_blocks) == 3
    assert len(parts.dec_blocks) == 6
    assert isinstance(parts.embeddings, nn.Module)
    assert isinstance(parts.post_ln, nn.Module)
    assert isinstance(parts.map_head, nn.Module)

    # dec_blocks 与 gen_blocks 是同源的**独立**深拷贝：数值相等但存储互不共享。
    import torch
    w_gen = parts.gen_blocks[0].attn.q_proj.weight
    w_dec = parts.dec_blocks[0].attn.q_proj.weight
    assert w_gen.data_ptr() != w_dec.data_ptr(), "dec_blocks 必须是独立深拷贝，不能与 gen 共享存储"
    assert torch.allclose(w_gen, w_dec), "同源层的 gen/dec 深拷贝初值应相等"


def test_tiny_default_gen_depth_clamped():
    """默认 gen_depth=13 > tiny 的 6 层时不应崩溃（切分被夹到实际层数）。"""
    parts = load_siglip_parts(tiny=True)  # gen_depth 默认 13
    assert len(parts.gen_blocks) == 6
    assert len(parts.sem_blocks) == 0
    assert len(parts.dec_blocks) == 6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
