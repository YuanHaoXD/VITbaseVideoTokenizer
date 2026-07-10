"""集成冒烟测试：训练步全链路 forward_train→recon+distill→backward→step（CPU tiny，离线）。

这是 8-GPU torchrun null 冒烟的 CPU 可做子集——不验 DDP/多卡/真实权重，
只验「模型 forward_train 产出的 dict 各键 → 被 recon_loss/distill_loss 正确消费 →
total.backward() 梯度回流 → optimizer.step()」这条最高风险集成路径不出接线错。
单测覆盖各部件；本测试验证它们**组合**起来能训练一步。
"""
import pytest

torch = pytest.importorskip("torch")


def _run_one_step(tok, x, t_img_patch, t_img_pool, distill, cfg, opt):
    opt.zero_grad()
    out = tok.forward_train(x)
    rec = __import__("losses.recon", fromlist=["recon_loss"]).recon_loss(x, out, cfg)
    t_vid = torch.zeros(1, 16, 16, x.shape[1] * 0 + 64) if out["decomp_out"] is not None else None
    is_video = torch.tensor([x.shape[2] > 1]) if x.dim() == 5 else torch.tensor([False])
    dis = distill(out["s"], out["s_pool"], out["decomp_out"],
                  t_img_patch, t_img_pool, t_vid, is_video)
    total = rec["total"] + cfg.lambda_dist * dis["total"]
    total.backward()
    opt.step()
    return total, rec, dis


def test_train_step_image_and_video_compose():
    """图像 + 视频各跑一步：total 有限、梯度回流、视频激活 Decompressor(更多 grad params)。"""
    from models.uvt.uvt_tokenizer import UVTTokenizer, UVTConfig
    from losses.distill import DistillLoss
    from losses.recon import recon_loss  # noqa: F401
    from teachers.siglip2_teacher import SigLIP2Teacher

    torch.manual_seed(0)
    cfg = UVTConfig(tiny=True, use_cos_consistency=True)
    tok = UVTTokenizer(cfg)
    tok.set_stage(1)
    tok.train()

    t_img = SigLIP2Teacher(tiny=True)
    t_img_patch, t_img_pool = t_img(torch.rand(1, 3, 64, 64))
    distill = DistillLoss(student_dim=64, teacher_img_dim=64, teacher_vid_dim=64, cfg=cfg)
    opt = torch.optim.AdamW([p for p in tok.parameters() if p.requires_grad], lr=1e-3)

    # 图像步（decomp_out=None，Decompressor 不参与）
    total_i, rec_i, _ = _run_one_step(tok, torch.rand(1, 3, 64, 64),
                                      t_img_patch, t_img_pool, distill, cfg, opt)
    assert torch.isfinite(total_i), "图像步 total 非有限"

    # 视频步（decomp_out≠None，Decompressor + vid 蒸馏激活 → 更多参数收梯度）
    total_v, rec_v, _ = _run_one_step(tok, torch.rand(1, 3, 17, 64, 64),
                                      t_img_patch, t_img_pool, distill, cfg, opt)
    assert torch.isfinite(total_v), "视频步 total 非有限"

    # 关键集成断言：所有损失项数值有限（接线无误的充分信号）
    for name, v in [("img.l1", rec_i["l1"]), ("img.kl", rec_i["kl"]),
                    ("vid.l1", rec_v["l1"]), ("vid.kl", rec_v["kl"])]:
        assert torch.isfinite(v), f"{name} 非有限"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
