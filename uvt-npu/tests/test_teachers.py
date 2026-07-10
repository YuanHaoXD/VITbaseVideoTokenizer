"""T-1/T-2 验收测试：teachers/{siglip2_teacher,internvideo_teacher}.py。

- tiny SigLIP2 教师：形状/冻结/无梯度/文本塔（离线，需 transformers）。
- MockTeacher：形状/确定性/冻结（纯 torch 离线）。
- golden 骨架：test_siglip2_golden（vs 官方 pipeline fixture，cos>0.999）与
  internvideo golden——fixture 约定 tests/fixtures/，无 fixture/无权重时 skip。
"""
from pathlib import Path

import pytest
import torch

FIXTURES = Path(__file__).parent / "fixtures"


def _has_transformers() -> bool:
    try:
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


needs_transformers = pytest.mark.skipif(not _has_transformers(),
                                        reason="环境无 transformers（教师依赖，§0 约定 6）")


# ======================================================================================
# T-1 SigLIP2Teacher（tiny 离线路径）
# ======================================================================================
@needs_transformers
def test_siglip2_tiny_forward():
    """tiny 模式离线构建；forward 形状 (patch[B,N,D_t], pooled[B,D_t])，输出无梯度。"""
    from teachers.siglip2_teacher import SigLIP2Teacher

    teacher = SigLIP2Teacher(tiny=True)
    imgs = torch.rand(2, 3, 64, 64)  # tiny 的 image_size=64
    patch, pooled = teacher(imgs)

    n = (64 // 16) ** 2  # 16 个空间 token
    assert patch.shape == (2, n, teacher.dim)
    assert pooled.shape == (2, teacher.dim)
    assert torch.isfinite(patch).all() and torch.isfinite(pooled).all()
    # 教师冻结语义：参数不可训 + 输出不在计算图上 + train() 不可把它切回训练态。
    assert all(not p.requires_grad for p in teacher.parameters())
    assert patch.grad_fn is None and pooled.grad_fn is None
    teacher.train()
    assert not teacher.training, "教师 train() 必须被覆盖为常驻 eval"


@needs_transformers
def test_siglip2_tiny_rejects_wrong_size():
    """尺寸与 processor 期望不符必须报错（resize 归 E-1 protocols，教师不做插值）。"""
    from teachers.siglip2_teacher import SigLIP2Teacher

    teacher = SigLIP2Teacher(tiny=True)
    with pytest.raises(AssertionError):
        teacher(torch.rand(1, 3, 32, 32))


@needs_transformers
def test_siglip2_tiny_encode_text():
    """文本塔：K 条 prompt → [K,D_t]，确定性（同 prompt 同输出）。"""
    from teachers.siglip2_teacher import SigLIP2Teacher

    teacher = SigLIP2Teacher(tiny=True)
    prompts = ["a photo of a cat", "a photo of a dog", "a video of rain"]
    t1 = teacher.encode_text(prompts)
    t2 = teacher.encode_text(prompts)
    assert t1.shape == (3, teacher.dim)
    assert torch.allclose(t1, t2), "同 prompt 两次编码必须一致（zero-shot 可复现性）"
    assert t1.grad_fn is None


@needs_transformers
def test_siglip2_norm_from_processor():
    """归一化常数必须来自官方 processor 配置（buffer 已注册），不是手写值的旁证测试。"""
    from transformers import SiglipImageProcessor

    from teachers.siglip2_teacher import SigLIP2Teacher

    teacher = SigLIP2Teacher(tiny=True)
    ip = SiglipImageProcessor()  # tiny 路径的同源官方默认
    assert torch.allclose(teacher._mean.flatten(), torch.tensor(ip.image_mean))
    assert torch.allclose(teacher._std.flatten(), torch.tensor(ip.image_std))


# ======================================================================================
# T-2 MockTeacher（真教师需权重+网络，单测一律走 Mock）
# ======================================================================================
def test_mock_teacher_shapes_and_determinism():
    """[B,3,16,H,W] → [B,16,N_t,D_t]；同输入同输出；跨实例（同 seed）也一致。"""
    from teachers.internvideo_teacher import MockTeacher

    mock = MockTeacher(dim=32, spatial_tokens=16, frames=16, seed=0)
    video = torch.rand(2, 3, 16, 64, 64)
    f1 = mock(video)
    f2 = mock(video)
    assert f1.shape == (2, 16, 16, 32)
    assert torch.allclose(f1, f2), "MockTeacher 必须确定性（golden 骨架依赖）"
    assert f1.grad_fn is None
    assert torch.isfinite(f1).all()

    mock_b = MockTeacher(dim=32, spatial_tokens=16, frames=16, seed=0)
    assert torch.allclose(f1, mock_b(video)), "同 seed 跨实例必须一致"

    # 输入相关性：换输入输出必须变（否则测不出对齐头在学东西）。
    assert not torch.allclose(f1, mock(torch.rand(2, 3, 16, 64, 64)))


def test_mock_teacher_frozen_and_dim():
    from teachers.internvideo_teacher import MockTeacher

    mock = MockTeacher(dim=24, spatial_tokens=4)
    assert mock.dim == 24
    assert all(not p.requires_grad for p in mock.parameters())
    mock.train()
    assert not mock.training


def test_mock_teacher_feeds_distill_loss():
    """契约闭环：MockTeacher 输出可直接喂 L-2 的 vid 项（含网格对齐路径）。"""
    from losses.distill import DistillLoss

    from teachers.internvideo_teacher import MockTeacher

    mock = MockTeacher(dim=12, spatial_tokens=4, frames=16)   # 教师网格 2×2
    t_vid = mock(torch.rand(2, 3, 16, 32, 32))                # [2,16,4,12]
    loss_mod = DistillLoss(student_dim=8, teacher_img_dim=12, teacher_vid_dim=12)
    s = torch.randn(2, 3, 16, 8)
    s_pool = torch.randn(2, 8)
    decomp = torch.randn(2, 16, 16, 8, requires_grad=True)    # 学生网格 4×4 → 对齐到 2×2
    d = loss_mod(s, s_pool, decomp, torch.randn(2, 16, 12), torch.randn(2, 12),
                 t_vid, torch.ones(2, dtype=torch.bool))
    assert torch.isfinite(d["vid"])
    d["vid"].backward()
    assert decomp.grad is not None


def test_internvideo_config_teacher_id_switch():
    """config 字段 teacher_id 可切换主选/备胎（不加载权重，只验配置面契约）。"""
    from teachers.internvideo_teacher import InternVideoTeacherConfig

    cfg = InternVideoTeacherConfig()
    assert "InternVideo-Next" in cfg.teacher_id
    assert cfg.fallback_id == "OpenGVLab/InternVideo2-Stage2_1B-224p-f4"
    cfg2 = InternVideoTeacherConfig(teacher_id="OpenGVLab/InternVideo2-Stage2_1B-224p-f4")
    assert "Stage2" in cfg2.teacher_id


# ======================================================================================
# golden 骨架（P0-golden 首日生成 fixture 入库后启用；无网络/无权重时跳过）
# ======================================================================================
@pytest.mark.skipif(not (FIXTURES / "siglip2_golden.pt").exists(),
                    reason="缺 tests/fixtures/siglip2_golden.pt（P0-golden 用官方 pipeline 生成入库）")
@needs_transformers
def test_siglip2_golden():
    """T-1 验收：固定输入 vs 官方 pipeline 输出 fixture，cos>0.999。

    fixture 生成方（P0-golden 脚本）约定字段：
      {"images_01": [B,3,256,256] float in [0,1],
       "patch_feats": 官方 AutoProcessor+AutoModel 路径的 last_hidden_state,
       "pooled":      同路径 pooler_output,
       "model_id":    生成时的 checkpoint id}
    """
    import torch.nn.functional as F

    from teachers.siglip2_teacher import SigLIP2Teacher

    blob = torch.load(FIXTURES / "siglip2_golden.pt")
    teacher = SigLIP2Teacher(model_id=blob["model_id"])  # 需本地缓存权重，否则此行触发下载
    patch, pooled = teacher(blob["images_01"])

    cos_patch = F.cosine_similarity(patch.flatten(1), blob["patch_feats"].flatten(1), dim=-1)
    cos_pool = F.cosine_similarity(pooled, blob["pooled"], dim=-1)
    assert cos_patch.min().item() > 0.999, f"patch 特征偏离官方 pipeline: {cos_patch.min().item()}"
    assert cos_pool.min().item() > 0.999, f"池化嵌入偏离官方 pipeline: {cos_pool.min().item()}"


@pytest.mark.skipif(not (FIXTURES / "internvideo_golden.pt").exists(),
                    reason="缺 tests/fixtures/internvideo_golden.pt（T-2 有真实 API 不确定性，"
                           "P0-golden 首日对官方 repo 校准后生成）")
@needs_transformers
def test_internvideo_golden():
    """T-2 验收：固定 clip vs 官方推理输出 fixture（同时核销 internvideo_teacher.UNCERTAINTIES）。

    fixture 约定字段：{"video_01": [B,3,16,H,W], "feats": [B,16,N_t,D_t], "teacher_id": str}
    """
    import torch.nn.functional as F

    from teachers.internvideo_teacher import InternVideoTeacher, InternVideoTeacherConfig

    blob = torch.load(FIXTURES / "internvideo_golden.pt")
    teacher = InternVideoTeacher(InternVideoTeacherConfig(teacher_id=blob["teacher_id"]))
    feats = teacher(blob["video_01"])
    assert feats.shape == blob["feats"].shape
    cos = F.cosine_similarity(feats.flatten(1), blob["feats"].flatten(1), dim=-1)
    assert cos.min().item() > 0.999


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
