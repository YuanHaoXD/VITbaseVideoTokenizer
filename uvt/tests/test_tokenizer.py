"""M-10 验收测试：models/uvt/uvt_tokenizer.py。

- test_forward_tiny：tiny 配置，视频 [2,3,17,64,64] 与图像 [2,3,64,64] 各一遍，
  断言全部输出键存在、形状正确、数值有限；decomp_out 仅视频非 None。
- test_stage_freeze：set_stage 1/2/3 后 requires_grad 集合分别 == 全参 / decoder / sem_vit。

集群运行（需 torch + transformers + einops + huggingface_hub）：
    pytest tests/test_tokenizer.py -v

Windows 开发机无 torch，本文件仅 py_compile 校验语法（见交付报告）。
"""
import pytest
import torch

from models.uvt.uvt_tokenizer import UVTConfig, UVTTokenizer


def _tiny_cfg():
    """tiny 一致配置。

    【已上报 bug】UVTConfig 默认 unfold_positions=(21,27) 是为**全量 27 层** decoder 标定的；
    tiny 下 dec_blocks 仅 6 层（合法域 [0,6]），(21,27) 越界 → PixelDecoder.__init__ 断言炸。
    故 tiny 测试须显式给 6 层合法的 unfold_positions（此处取 (3,6)：block3 前展开 + 末块后展开，
    2×2 还原 5→9→17）。fold_positions=(0,6) 在 tiny 下本就合法（[0,6] 内），无需改。
    本测试不擅自改 UVTConfig 默认（冻结字段），仅在此构造一致 tiny 配置。
    """
    return UVTConfig(tiny=True, unfold_positions=(3, 6))


def _new_tiny_model():
    torch.manual_seed(0)
    return UVTTokenizer(_tiny_cfg())


def _assert_finite_dict(out: dict, label: str) -> None:
    """断言 out 中所有 tensor 值有限；decomp_out 为 None 时跳过。"""
    for k in ("x_hat", "z", "mu", "kl", "h", "s", "s_pool", "mu_proj"):
        v = out[k]
        assert torch.is_tensor(v), f"[{label}] {k} 应为 tensor"
        assert torch.isfinite(v).all(), f"[{label}] {k} 含 NaN/Inf"
    # kl 是标量 tensor，单独已覆盖。
    if out.get("decomp_out") is not None:
        assert torch.isfinite(out["decomp_out"]).all(), f"[{label}] decomp_out 含 NaN/Inf"


def test_forward_tiny_video():
    """视频 [2,3,17,64,64]：tiny 下 N=(64/16)^2=16，D=64。17 帧 → 1+4 时间位。"""
    model = _new_tiny_model().eval()
    video = torch.rand(2, 3, 17, 64, 64)       # [0,1] 区间（与 decoder 输出语义一致）
    with torch.no_grad():
        out = model(video)

    assert out["x_hat"].shape == (2, 3, 17, 64, 64)
    assert out["h"].shape == (2, 5, 16, 64)    # 1+4 时间位
    assert out["z"].shape == out["mu"].shape == (2, 5, 16, 64)
    assert out["s"].shape == (2, 5, 16, 64)
    assert out["s_pool"].shape == (2, 64)
    assert out["mu_proj"].shape == (2, 5, 16, 64)   # 与 h 同形（L_cos 按特征维 D 求余弦）
    # decomp_out 仅视频：s[:,1:] 为 4 位 → 4×=16 帧
    assert out["decomp_out"] is not None
    assert out["decomp_out"].shape == (2, 16, 16, 64)
    _assert_finite_dict(out, "video")


def test_forward_tiny_image():
    """图像 [2,3,64,64]：入口升维 [2,3,1,64,64] → 1 时间位；decomp_out=None。"""
    model = _new_tiny_model().eval()
    image = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        out = model(image)

    assert out["x_hat"].shape == (2, 3, 1, 64, 64)
    assert out["h"].shape == (2, 1, 16, 64)
    assert out["z"].shape == out["mu"].shape == (2, 1, 16, 64)
    assert out["s"].shape == (2, 1, 16, 64)
    assert out["s_pool"].shape == (2, 64)
    assert out["decomp_out"] is None, "图像 batch（F=1）必须 decomp_out=None（契约⑥）"
    _assert_finite_dict(out, "image")


def test_forward_train_training_mode_has_grad():
    """训练态 forward → forward_train：输出可反传（x_hat 与 decomp_out 皆有梯度图）。"""
    model = _new_tiny_model().train()
    video = torch.rand(1, 3, 17, 64, 64)
    out = model(video)
    loss = out["x_hat"].float().mean()
    if out["decomp_out"] is not None:
        loss = loss + out["decomp_out"].float().mean()
    loss.backward()      # 不报错即说明 forward_train 训练图完整
    assert out["x_hat"].requires_grad


def _trainable_ptrs(m):
    return {p.data_ptr() for p in m.parameters() if p.requires_grad}


def test_stage_freeze():
    """set_stage 1/2/3：可训参数集合分别 == 全参 / decoder / sem_vit（§2.9 表）。"""
    model = _new_tiny_model()
    total = {p.data_ptr() for p in model.parameters()}
    assert len(total) > 0

    # Stage 1：全部可训
    model.set_stage(1)
    assert _trainable_ptrs(model) == total

    # Stage 2：仅 decoder（§2.9「仅 decoder」，encoder/GSB 冻结）
    model.set_stage(2)
    dec_ptrs = {p.data_ptr() for p in model.decoder.parameters()}
    assert _trainable_ptrs(model) == dec_ptrs, "Stage 2 可训参数应恰为 decoder 参数"

    # Stage 3：仅 Sem-ViT（含 map_head 子模块）
    model.set_stage(3)
    sem_ptrs = {p.data_ptr() for p in model.sem_vit.parameters()}
    assert _trainable_ptrs(model) == sem_ptrs, "Stage 3 可训参数应恰为 sem_vit 参数"


def test_state_dict_excludes_decompressor():
    """state_dict post-hook：剔除 Decompressor.* 键（训练期专用，checkpoint 不导出）。"""
    model = _new_tiny_model().eval()
    sd = model.state_dict()
    decompressor_keys = [k for k in sd if "decompressor." in k]
    assert decompressor_keys == [], f"state_dict 不应含 decompressor 键，实得 {decompressor_keys}"
    # 其余核心部件应在
    assert any(k.startswith("encoder.") for k in sd)
    assert any(k.startswith("gsb.") for k in sd)
    assert any(k.startswith("decoder.") for k in sd)
    assert any(k.startswith("sem_vit.") for k in sd)
    # gsb.normalize 持久化 buffer 应在（契约⑤的实际形态）
    assert any("gsb." in k and "_normalize_flag" in k for k in sd), \
        "gsb._normalize_flag（normalize 持久化 buffer）应进 state_dict"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
