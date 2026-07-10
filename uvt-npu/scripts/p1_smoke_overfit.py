"""P1-smoke Gate（04 §1 Phase 1 第一项）：固定 batch overfit，500 步，断言 PSNR>30。

为什么：null 冒烟只证明"不崩 + loss 下降"；本脚本固定一个 batch 反复训，验证训练管线
真能把重建推到 PSNR>30（梯度回流、loss 组装、优化器全链路在真实学习）。P1=图像 tokenizer，
故用图像 batch（F=1，轻量）。单卡 NPU。生产模型（tiny=false + 真 SigLIP2）。

用法：
  source scripts/env_npu.sh
  ASCEND_RT_VISIBLE_DEVICES=0 $PYBIN scripts/p1_smoke_overfit.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import accel  # triton shim + import torch_npu（必须在 import torch 前，本模块首行即此）
import torch

SIG = "/cache/VITbaseVideoTokenizer/models/siglip2-so400m-patch16-256"
STEPS = int(os.environ.get("STEPS", 500))
BS = int(os.environ.get("BS", 4))        # 256² 视频显存吃紧,图像 bs=4
SIZE = int(os.environ.get("SIZE", 256))  # so400m 位置编码原生 256²,不能改(会崩)
LR = float(os.environ.get("LR", 1e-3))   # overfit 用更高 lr(解码器像素头随机初始化)

dev = accel.device()
torch.manual_seed(0)

from models.uvt.uvt_tokenizer import UVTTokenizer, UVTConfig
from losses.recon import recon_loss
from losses.distill import DistillLoss
from teachers.siglip2_teacher import SigLIP2Teacher

cfg = UVTConfig(model_name=SIG, tiny=False,
                lpips_weight=float(os.environ.get("LPIPS_W", "1.0")))
tok = UVTTokenizer(cfg).to(dev)
tok.set_stage(1)
tok.train()
print(f"[model] #params={sum(p.numel() for p in tok.parameters())/1e6:.1f}M  device={dev}", flush=True)

t_img = SigLIP2Teacher(model_id=SIG, tiny=False).to(dev).eval()
for p in t_img.parameters():
    p.requires_grad_(False)
distill = DistillLoss(student_dim=1152, teacher_img_dim=1152, teacher_vid_dim=1152, cfg=cfg).to(dev)

opt = torch.optim.AdamW([p for p in tok.parameters() if p.requires_grad],
                        lr=LR, betas=(0.9, 0.95), weight_decay=0.05)

# 固定图像 batch（P1=image）。recon 驱动 PSNR，distill img 项同时训。
# x 用 5D [B,3,F,H,W]（F=1）与模型 x_hat 输出对齐，避免相减广播错位。
x = torch.rand(BS, 3, 1, SIZE, SIZE, device=dev)
is_video = torch.zeros(BS, dtype=torch.bool, device=dev)

amp = torch.bfloat16
amp_on = os.environ.get("AMP", "1") != "0"   # AMP=0 跑 fp32,排除 bf16 精度上限
for step in range(STEPS):
    opt.zero_grad()
    with torch.autocast(device_type=accel.BACKEND, dtype=amp, enabled=amp_on):
        # SAMPLE=0 用确定性 z(=μ,无重参数噪声),隔离"解码器能否重建"——排除每步随机 z 记不住的干扰
        _sample = os.environ.get("SAMPLE", "1") != "0"
        out = tok._forward_core(x, sample=_sample)  # [B,3,1,256,256]
        rec = recon_loss(x, out, cfg)
        lam = 0.0 if os.environ.get("DIST", "1") == "0" else 0.5  # DIST=0 纯重建(排除蒸馏拉扯)
        if lam > 0:
            t_patch, t_pool = t_img(x[:, :, 0])   # SigLIP2 教师 eat [B,3,H,W]（锚帧）
            d = distill(out['s'], out['s_pool'], out.get('decomp_out'),
                        t_patch, t_pool, None, is_video)
            loss = rec['total'] + lam * d['total']
        else:
            d = {'total': torch.zeros((), device=x.device)}
            loss = rec['total']
    loss.backward()
    opt.step()
    if step % 25 == 0 or step == STEPS - 1:
        with torch.no_grad():
            xh = out['x_hat'].float().clamp(0, 1)
            mse = ((xh - x.float()) ** 2).mean()
            psnr = (-10 * torch.log10(mse.clamp_min(1e-10))).item()
        print(f"step {step:3d}: loss={loss.item():.4f} psnr={psnr:5.2f} "
              f"(l1={rec['l1'].item():.3f} lpips={rec['lpips'].item():.3f} "
              f"cos={rec['cos_consistency'].item():.3f} dist={d['total'].item():.3f})", flush=True)

print(f"\nFINAL PSNR={psnr:.2f}  ->  "
      f"{'PASS (Gate PSNR>30)' if psnr > 30 else 'FAIL (未达 30;按 R11 拿到曲线后重标阈值)'}", flush=True)
