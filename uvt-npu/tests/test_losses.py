"""L-1/L-2/L-3 验收测试：losses/{recon,distill,gan}.py。

- test_recon_loss_perfect      : x_hat==x 时 l1≈0（LPIPS 项另测，无 lpips 库时跳过）。
- test_distill_masking         : is_video 全 False 时 vid 项无梯度（head_vid.grad is None）。
- test_disc_forward            : 真假样本 logits 形状/有限性（我方协议 17 帧 + 锚帧隔离 patchify）。
- test_gan_alternate_100_steps : tiny 配置 G/D 交替 100 步 loss 不 NaN。
- golden test 骨架             : fixture 约定 tests/fixtures/，缺 fixture 时 skip。

除 lpips 相关外全部离线、只依赖 torch。
"""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from losses.distill import DistillLoss
from losses.gan import GANLossConfig, TransformerDiscriminator, UVTGANLoss
from losses.recon import recon_loss

FIXTURES = Path(__file__).parent / "fixtures"


def _has_lpips() -> bool:
    try:
        import lpips  # noqa: F401
        return True
    except ImportError:
        return False


def _recon_cfg(**over):
    """最小 cfg 替身：显式给全五个权重字段（硬约束：权重从 cfg 读）。"""
    base = dict(l1_weight=1.0, lpips_weight=0.0, kl_weight=1e-6,
                cos_weight=1.0, use_cos_consistency=False)
    base.update(over)
    return SimpleNamespace(**base)


# ======================================================================================
# L-1 recon
# ======================================================================================
def test_recon_loss_perfect():
    """x_hat==x → l1≈0；kl 由 out 传入并按权重计入。"""
    torch.manual_seed(0)
    x = torch.rand(2, 3, 5, 32, 32)  # [B,3,1+T,H,W]
    out = {"x_hat": x.clone(), "kl": torch.zeros(())}
    d = recon_loss(x, out, _recon_cfg())
    assert set(d) == {"l1", "lpips", "kl", "cos_consistency", "total"}
    assert d["l1"].item() < 1e-8, f"完美重建 l1 应≈0，实得 {d['l1'].item()}"
    assert d["lpips"].item() == 0.0  # lpips_weight=0 时不触发 lpips 库
    assert d["total"].item() < 1e-7


@pytest.mark.skipif(not _has_lpips(), reason="环境无 lpips 库（recon 的感知项依赖）")
def test_recon_loss_perfect_lpips():
    """x_hat==x 时 lpips≈0（需 lpips 库；VGG 要求空间边长 ≥64 取 64）。"""
    torch.manual_seed(0)
    x = torch.rand(1, 3, 2, 64, 64)
    out = {"x_hat": x.clone(), "kl": torch.zeros(())}
    d = recon_loss(x, out, _recon_cfg(lpips_weight=1.0))
    assert d["lpips"].item() < 1e-6, f"完美重建 lpips 应≈0，实得 {d['lpips'].item()}"


def test_recon_cos_consistency():
    """L_cos = mean(1−cos(mu_proj, h))：同向=0，反向=2；开关关时恒 0。"""
    torch.manual_seed(0)
    x = torch.rand(2, 3, 1, 16, 16)
    h = torch.randn(2, 1, 4, 8)
    out = {"x_hat": x.clone(), "kl": torch.zeros(()), "mu_proj": h.clone(), "h": h}
    cfg = _recon_cfg(use_cos_consistency=True)
    assert recon_loss(x, out, cfg)["cos_consistency"].item() < 1e-6

    out["mu_proj"] = -h  # 反向 → 1−(−1)=2
    val = recon_loss(x, out, cfg)["cos_consistency"].item()
    assert abs(val - 2.0) < 1e-5

    cfg_off = _recon_cfg(use_cos_consistency=False)
    assert recon_loss(x, out, cfg_off)["cos_consistency"].item() == 0.0


def test_recon_image_4d_input():
    """图像 [B,3,H,W] 入口升维（§0 张量约定），不特判崩溃。"""
    x = torch.rand(2, 3, 16, 16)
    d = recon_loss(x, {"x_hat": x.clone(), "kl": torch.zeros(())}, _recon_cfg())
    assert d["l1"].item() < 1e-8


# ======================================================================================
# L-2 distill
# ======================================================================================
def _distill_inputs(B=2, T1=3, N=4, D=8, Dt=12, Nt=4, is_video=None):
    torch.manual_seed(0)
    s = torch.randn(B, T1, N, D, requires_grad=True)
    s_pool = torch.randn(B, D, requires_grad=True)
    decomp = torch.randn(B, 16, N, D, requires_grad=True)
    t_patch = torch.randn(B, N, Dt)
    t_pool = torch.randn(B, Dt)
    t_vid = torch.randn(B, 16, Nt, Dt)
    if is_video is None:
        is_video = torch.ones(B, dtype=torch.bool)
    return s, s_pool, decomp, t_patch, t_pool, t_vid, is_video


def test_distill_masking():
    """验收核心：is_video 全 False → vid 项为 0 且 head_vid 参数梯度为 None。"""
    loss_mod = DistillLoss(student_dim=8, teacher_img_dim=12, teacher_vid_dim=12)
    s, s_pool, decomp, t_patch, t_pool, t_vid, _ = _distill_inputs()
    is_video = torch.zeros(2, dtype=torch.bool)

    d = loss_mod(s, s_pool, decomp, t_patch, t_pool, t_vid, is_video)
    assert d["vid"].item() == 0.0
    d["total"].backward()

    # vid 支路完全未被触碰：对齐头无梯度、decomp_out 无梯度。
    assert loss_mod.head_vid.weight.grad is None, "纯图像 batch 下 head_vid 不得收到梯度"
    assert decomp.grad is None, "纯图像 batch 下 decomp_out 不得收到梯度"
    # 图像两项照常回传。
    assert loss_mod.head_img_patch.weight.grad is not None
    assert loss_mod.head_img_pool.weight.grad is not None
    assert s.grad is not None and s_pool.grad is not None


def test_distill_mixed_batch():
    """混合 batch：vid 项只在视频样本上取平均，梯度有限。"""
    loss_mod = DistillLoss(student_dim=8, teacher_img_dim=12, teacher_vid_dim=12)
    is_video = torch.tensor([True, False])
    s, s_pool, decomp, t_patch, t_pool, t_vid, _ = _distill_inputs()
    d = loss_mod(s, s_pool, decomp, t_patch, t_pool, t_vid, is_video)
    assert torch.isfinite(d["total"])
    d["total"].backward()
    assert loss_mod.head_vid.weight.grad is not None
    assert torch.isfinite(loss_mod.head_vid.weight.grad).all()
    # 图像样本（样本 1）的 decomp 梯度应为 0（被掩码），视频样本（样本 0）非零。
    assert decomp.grad is not None
    assert decomp.grad[1].abs().max().item() == 0.0
    assert decomp.grad[0].abs().max().item() > 0.0


def test_distill_grid_align():
    """ADR-7：decomp 空间网格(16 token=4×4) 与教师(4 token=2×2) 不一致时双线性对齐后可算。"""
    loss_mod = DistillLoss(student_dim=8, teacher_img_dim=12, teacher_vid_dim=12)
    s, s_pool, _, t_patch, t_pool, _, is_video = _distill_inputs()
    decomp = torch.randn(2, 16, 16, 8, requires_grad=True)   # N_dec=16
    t_vid = torch.randn(2, 16, 4, 12)                        # N_t=4
    d = loss_mod(s, s_pool, decomp, t_patch, t_pool, t_vid, is_video)
    assert torch.isfinite(d["vid"]) and d["vid"].item() > 0.0


def test_distill_identity_head_when_dims_match():
    """学生=教师维度时对齐头退化为 Identity（免参数，05 §3 L-2 裁决）。"""
    import torch.nn as nn
    loss_mod = DistillLoss(student_dim=8, teacher_img_dim=8, teacher_vid_dim=12)
    assert isinstance(loss_mod.head_img_patch, nn.Identity)
    assert isinstance(loss_mod.head_vid, nn.Linear)


# ======================================================================================
# L-3 gan
# ======================================================================================
def _tiny_gan_cfg(**over):
    base = dict(disc_hidden_size=32, disc_n_heads=2, disc_n_layers=2,
                disc_input_size=16, disc_tran_temporal_patch_size=4,
                disc_tran_patch_size=16, disc_frame_num=16,
                lecam_weight=1e-3, d_update_freq=5, disc_start=0)
    base.update(over)
    return GANLossConfig(**base)


def test_disc_forward():
    """真假样本 logits [B,1] 且有限（我方协议：17 帧 = 1 锚 + 16，锚帧隔离 patchify）。"""
    torch.manual_seed(0)
    disc = TransformerDiscriminator(hidden_size=32, n_heads=2, n_layers=2, input_size=32,
                                    temporal_patch_size=4, patch_size=16, frame_num=16)
    # 时间 token 数 = 1(锚) + 16/4 = 5
    assert disc.token_t == 5
    real = torch.rand(2, 3, 17, 32, 32)
    fake = torch.rand(2, 3, 17, 32, 32)
    lr, lf = disc(real), disc(fake)
    assert lr.shape == lf.shape == (2, 1)
    assert torch.isfinite(lr).all() and torch.isfinite(lf).all()


def test_d_update_freq():
    """d_update_freq=5：仅每 5 步更新一次 D（损失详情 §1.3 稳定器）。"""
    gan = UVTGANLoss(_tiny_gan_cfg())
    flags = [gan.should_update_d(i) for i in range(10)]
    assert flags == [True, False, False, False, False, True, False, False, False, False]


def test_gan_alternate_100_steps():
    """tiny 配置 G/D 交替 100 步无 NaN（L-3 验收）。G 侧用一个可学习像素张量模拟 decoder。"""
    torch.manual_seed(0)
    gan = UVTGANLoss(_tiny_gan_cfg())
    real = torch.rand(2, 3, 17, 16, 16)
    fake_param = torch.rand(2, 3, 17, 16, 16, requires_grad=True)
    # 优化器语义（05 §3 L-3 ③）：D 侧 Adam(0.5,0.9)，lr=G lr×dis_lr_multiplier。
    g_lr = 1e-3
    opt_g = torch.optim.Adam([fake_param], lr=g_lr)
    opt_d = torch.optim.Adam(gan.discriminator.parameters(),
                             lr=g_lr * gan.cfg.dis_lr_multiplier,
                             betas=gan.cfg.dis_adam_betas)
    for step in range(100):
        g_loss, g_info = gan.generator_loss(fake_param, global_step=step)
        opt_g.zero_grad()
        g_loss.backward()
        opt_g.step()
        assert torch.isfinite(g_loss), f"step {step}: G loss NaN/Inf"

        if gan.should_update_d(step):
            d_loss, d_info = gan.discriminator_loss(real, fake_param.detach(), global_step=step)
            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()
            assert torch.isfinite(d_loss), f"step {step}: D loss NaN/Inf"
    # LeCam EMA 应已被更新（非初始 0）。
    assert gan.lecam_ema_real.item() != 0.0 or gan.lecam_ema_fake.item() != 0.0


def test_gan_disc_start_gating():
    """global_step < disc_start 时 G/D 两侧均为 0（adopt_weight 门控）。"""
    gan = UVTGANLoss(_tiny_gan_cfg(disc_start=1000))
    x = torch.rand(1, 3, 17, 16, 16)
    g_loss, _ = gan.generator_loss(x, global_step=10)
    d_loss, _ = gan.discriminator_loss(x, x, global_step=10)
    assert g_loss.item() == 0.0 and d_loss.item() == 0.0


# ======================================================================================
# golden 骨架（P0-golden 首日生成 fixture 后启用）
# ======================================================================================
@pytest.mark.skipif(not (FIXTURES / "recon_golden.pt").exists(),
                    reason="缺 tests/fixtures/recon_golden.pt（P0-golden 生成入库后启用）")
def test_recon_golden():
    """固定输入的 recon 各分项 vs 入库值（防有人改权重语义/归一化不自知）。"""
    blob = torch.load(FIXTURES / "recon_golden.pt")
    d = recon_loss(blob["x"], blob["out"], SimpleNamespace(**blob["cfg"]))
    for k, v in blob["expected"].items():
        assert abs(d[k].item() - v) < 1e-5, f"{k}: {d[k].item()} vs golden {v}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
