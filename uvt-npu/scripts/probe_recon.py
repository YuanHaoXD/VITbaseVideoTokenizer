"""重建路径结构探针:定位 overfit 不收敛是"编码器 μ 塌缩"还是"解码器重建不了"。

量四件事(确定性 z=μ,排除重参数噪声):
  1. 4 张不同图 → μ 之间的余弦(>>0.9 = 塌缩,编码器没区分开)
  2. z=μ 的值域/规模
  3. decode(z) → x_hat 的值域 + 4 张图 x_hat 之间的多样性(都一样=解码器或 z 问题)
  4. L1(x,x_hat) 反传后,解码器像素头 vs 编码器的梯度范数(像素头≈0=梯度没到)

单卡。用法: source scripts/env_npu.sh; ASCEND_RT_VISIBLE_DEVICES=0 $PYBIN scripts/probe_recon.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import accel
import torch
import torch.nn.functional as F

SIG = "/cache/VITbaseVideoTokenizer/models/siglip2-so400m-patch16-256"
dev = accel.device(); torch.manual_seed(0)
from models.uvt.uvt_tokenizer import UVTTokenizer, UVTConfig

cfg = UVTConfig(model_name=SIG, tiny=False)
tok = UVTTokenizer(cfg).to(dev); tok.set_stage(1); tok.train()
x = torch.rand(4, 3, 1, 256, 256, device=dev)   # 4 张不同的随机图

# 1) 编码 → μ(确定性)
with torch.no_grad():
    enc = tok.encode(x, sample=False)
    mu = enc["mu"]      # [4,1,256,64]
    z = enc["z"]
    print(f"[μ] shape={tuple(mu.shape)} mean={mu.mean().item():.4f} std={mu.std().item():.4f} "
          f"min={mu.min().item():.3f} max={mu.max().item():.3f}", flush=True)
    # 4 张图 μ 两两余弦(flatten 后)
    mf = mu.flatten(1)  # [4, 1*256*64]
    cm = F.cosine_similarity(mf[:,None,:], mf[None,:,:], dim=-1)  # [4,4]
    off = cm[torch.triu_indices(4,4,offset=1).unbind()]
    print(f"[μ 塌缩检查] 4 图 μ 两两余弦: mean={off.mean().item():.4f} "
          f"min={off.min().item():.4f} max={off.max().item():.4f}  (>>0.9=塌缩)", flush=True)
    # z(=μ) 解码
    xh = tok.decode(z, (256,256))   # [4,3,1,256,256]
    print(f"[x_hat] shape={tuple(xh.shape)} mean={xh.mean().item():.4f} std={xh.std().item():.4f} "
          f"min={xh.min().item():.3f} max={xh.max().item():.3f}", flush=True)
    print(f"[x]     mean={x.mean().item():.4f} std={x.std().item():.4f}  (x∈[0,1])", flush=True)
    # 4 张 x_hat 两两余弦(都一样=解码器对 z 无响应)
    xf = xh.flatten(1)
    cx = F.cosine_similarity(xf[:,None,:], xf[None,:,:], dim=-1)
    offx = cx[torch.triu_indices(4,4,offset=1).unbind()]
    print(f"[x_hat 多样性] 4 图重建两两余弦: mean={offx.mean().item():.4f}  (>>0.99=重建几乎相同)", flush=True)
    l1 = (x - xh).abs().mean().item()
    print(f"[L1(x,x_hat)] = {l1:.4f}  (随机初始化预期 ~1+)", flush=True)

# 2) 梯度到达检查:L1 反传后看各子模块梯度范数
tok.zero_grad()
enc = tok._forward_core(x, sample=False)
l1 = (x - enc["x_hat"]).abs().mean()
l1.backward()
print(f"\n[梯度范数] L1={l1.item():.4f} 反传后:", flush=True)
for name in ["encoder", "gsb", "decoder", "sem_vit", "decompressor"]:
    m = getattr(tok, name, None)
    if m is None: continue
    g = sum((p.grad.detach().abs().sum().item() for p in m.parameters() if p.grad is not None))
    n = sum(p.numel() for p in m.parameters() if p.grad is not None)
    print(f"  {name:12s}: grad_L1_sum={g:.4e}  (有梯度参数 {n/1e6:.1f}M)", flush=True)
# 解码器像素头专项
for pname, p in tok.decoder.named_parameters():
    if "pixel" in pname.lower() or "to_pix" in pname.lower() or "head" in pname.lower():
        print(f"  decoder.{pname}: grad_norm={p.grad.norm().item() if p.grad is not None else 'None'}", flush=True)
