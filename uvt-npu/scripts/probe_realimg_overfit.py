"""真图 overfit(§9-4 收尾):从 ImageNet parquet 抽 N 张真实图,固定 batch overfit。
验证 #15 修复在自然图下:①μ 不塌缩(编码器区分样本)②重建能收敛(排除 torch.rand 噪声伪影)。
用法: source scripts/env_npu.sh; ASCEND_RT_VISIBLE_DEVICES=0 $PYBIN scripts/probe_realimg_overfit.py
"""
import os, sys, io
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import accel
import torch, torch.nn.functional as F
import pyarrow.parquet as pq
from PIL import Image
import numpy as np

SIG = os.environ.get("UVT_SIGLIP",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), os.pardir,
                 "models", "siglip2-so400m-patch16-256"))
PARQUET = os.environ.get("IMAGENET_PARQUET",
    "/home/ma-user/work/dataset/yh222/datasets/imagenet-1k/data/train-00000-of-00294.parquet")
BS = int(os.environ.get("BS", 4)); SIZE = 256
STEPS = int(os.environ.get("STEPS", 500)); LR = float(os.environ.get("LR", 1e-3))
SAMPLE = os.environ.get("SAMPLE", "0") != "0"
LPIPS_W = float(os.environ.get("LPIPS_W", "0.0"))   # 默认纯L1(最干净);置1看含lpips
dev = accel.device(); torch.manual_seed(0)

# --- 抽 BS 张真图,resize+centercrop 256,[0,1] ---
tbl = pq.ParquetFile(PARQUET).read_row_group(0).to_pylist()
imgs = []
for i in range(BS):
    im = Image.open(io.BytesIO(tbl[i*37]['image']['bytes'])).convert('RGB')  # 隔开取,增加多样性
    # resize 短边到256 + center crop
    w,h = im.size; s = 256/min(w,h)
    im = im.resize((round(w*s), round(h*s)), Image.BICUBIC)
    w,h = im.size; l,t = (w-256)//2,(h-256)//2
    im = im.crop((l,t,l+256,t+256))
    arr = torch.from_numpy(np.array(im,dtype=np.uint8)).permute(2,0,1).float()/255.
    imgs.append(arr)
x = torch.stack(imgs).unsqueeze(2).to(dev)  # [BS,3,1,256,256]
print(f"[data] {BS} 张真图 x: mean={x.mean():.3f} std={x.std():.3f} range=[{x.min():.2f},{x.max():.2f}]", flush=True)

from models.uvt.uvt_tokenizer import UVTTokenizer, UVTConfig
from losses.recon import recon_loss
cfg = UVTConfig(model_name=SIG, tiny=False, lpips_weight=LPIPS_W)
tok = UVTTokenizer(cfg).to(dev); tok.set_stage(1); tok.train()

# --- 先探针:真图 μ 是否分化 ---
with torch.no_grad():
    enc = tok.encode(x, sample=False); mu = enc["mu"].flatten(1)
    cm = F.cosine_similarity(mu[:,None,:], mu[None,:,:], dim=-1)
    off = cm[torch.triu_indices(BS,BS,offset=1).unbind()]
    print(f"[真图 μ 塌缩检查] 两两余弦 mean={off.mean():.4f} min={off.min():.4f} max={off.max():.4f} (噪声图曾0.67,自然图应更低/更分化)", flush=True)

opt = torch.optim.AdamW([p for p in tok.parameters() if p.requires_grad], lr=LR, betas=(0.9,0.95), weight_decay=0.0)
for step in range(STEPS):
    opt.zero_grad()
    with torch.autocast(device_type=accel.BACKEND, dtype=torch.bfloat16):
        out = tok._forward_core(x, sample=SAMPLE)
        rec = recon_loss(x, out, cfg)
        loss = rec['total']
    loss.backward(); opt.step()
    if step % 50 == 0 or step == STEPS-1:
        with torch.no_grad():
            xh = out['x_hat'].float().clamp(0,1); mse = ((xh-x.float())**2).mean()
            psnr = (-10*torch.log10(mse.clamp_min(1e-10))).item()
        print(f"step {step:3d}: loss={loss.item():.4f} psnr={psnr:5.2f} (l1={rec['l1'].item():.3f} lpips={rec['lpips'].item():.3f})", flush=True)
print(f"\nFINAL PSNR={psnr:.2f}  ({'PASS>30 真图重建健康✓' if psnr>30 else 'note: 阈值/步数待调'})", flush=True)
