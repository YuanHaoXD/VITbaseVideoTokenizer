"""切分边界规范化验收测试（第 15 号 bug 修复，docs/08 §6.5 / docs/06 §6.8）。

背景（2026-07-12 NPU 服务器 P1-smoke overfit 诊断）：SigLIP2 残差流含巨激活
（massive activations，O(10³)、近图像无关的通道），而三处新造接口裸接了它：
  ① GenViT 出口 → GSB.proj（μ/ρ 被巨激活主导 → μ 跨图余弦 0.997、ρ 钉死 clamp 上界
     → σ=e^10 噪声 + clamp 外零梯度死区，sample=True 训练原理上不可收敛）；
  ② decoder 末 block 残差流 → 随机初始化像素头（x_hat 值域 ±44，目标却是 [0,1]）；
  ③ 学生编码器输入未做 SigLIP 归一化（[0,1] 直入 embeddings，教师侧却是 processor 归一化）。
本文件把三处修复钉成契约：GSB 瓶颈入口 LN、decoder 像素头前 final_ln + 校准初始化、
GenViT 输入按 processor mean/std 归一化（buffer 承载）。

集群运行：pytest tests/test_boundary_norms.py -v
"""
import torch

from models.uvt.gsb import GSB
from models.uvt.uvt_tokenizer import UVTConfig, UVTTokenizer


def _tiny_model():
    torch.manual_seed(0)
    # unfold_positions 取 tiny 6 层合法值（同 test_tokenizer._tiny_cfg 的已上报约定）。
    return UVTTokenizer(UVTConfig(tiny=True, unfold_positions=(3, 6)))


def test_gsb_bottleneck_norm_tames_massive_activations():
    """GSB 入口 LN：模拟巨激活通道的 h，μ 与 KL 必须保持 O(1) 量级。

    修复前（无 LN）：|μ| 冲到 ~30（w·1000），ρ 同源钉死 clamp 上界 20 → e^ρ=e^20
    → KL 天文数字；修复后 LN 把逐 token 尺度拉回，μ/ρ/KL 全部有界。
    """
    torch.manual_seed(0)
    gsb = GSB(d_model=1152, c_latent=64)
    h = torch.randn(2, 1, 16, 1152)
    h[..., 0] = 1000.0                      # 模拟 SigLIP2 残差流的图像无关巨激活通道
    z, mu, kl = gsb.compress(h)
    assert torch.isfinite(z).all() and torch.isfinite(kl)
    assert mu.abs().max().item() < 5.0, f"μ 未被瓶颈入口 LN 驯服：max|μ|={mu.abs().max().item():.1f}"
    assert kl.item() < 100.0, f"KL 爆炸（ρ 饱和的症状）：kl={kl.item():.3e}"


def test_gsb_compress_deterministic_path():
    """compress(sample=False) 返回 z=μ（确定性路径收口进 GSB，替代调用方直捅 gsb.proj）。"""
    torch.manual_seed(0)
    gsb = GSB(d_model=64, c_latent=16)
    h = torch.randn(2, 1, 4, 64)
    z, mu, _ = gsb.compress(h, sample=False)
    assert torch.equal(z, mu)


def test_decoder_final_ln_and_calibrated_head():
    """decoder 末端 LN + 像素头校准初始化：随机初始化模型的 x_hat 应落在 [0,1] 邻域（≈灰图）。

    修复前：末 block 残差流（真权重下尺度 O(10²)）直入默认初始化 Linear → x_hat ±44。
    修复后：final_ln 收尺度 + 头 weight std=0.02 / bias=0.5 → 初始输出以 0.5 为中心小扰动。
    """
    model = _tiny_model()
    assert hasattr(model.decoder, "final_ln"), "decoder 缺 final_ln（像素头前规范化）"
    model.eval()
    x = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
    x_hat = out["x_hat"]
    assert 0.2 < x_hat.mean().item() < 0.8, f"初始 x_hat 未校准到 [0,1] 邻域：mean={x_hat.mean().item():.2f}"
    assert x_hat.std().item() < 1.0, f"初始 x_hat 尺度失控：std={x_hat.std().item():.2f}"


def test_encoder_applies_siglip_input_normalization():
    """GenViT 输入归一化：px_mean/px_std buffer 存在且在 patchify 前生效。

    验证方式：先用 buffer 正常前向；再把 buffer 置恒等（mean=0,std=1）并手工喂
    预归一化输入，两者输出必须逐位一致——证明归一化恰好发生在 embeddings 之前。
    """
    model = _tiny_model()
    enc = model.encoder
    assert hasattr(enc, "px_mean") and hasattr(enc, "px_std"), "GenViT 缺 px_mean/px_std buffer"
    assert tuple(enc.px_mean.shape) == (1, 3, 1, 1)
    enc.eval()
    x = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        h1 = enc(x)
        x_pre = (x - enc.px_mean) / enc.px_std       # [1,3,1,1] 对 4D 广播
        enc.px_mean.zero_()
        enc.px_std.fill_(1.0)
        h2 = enc(x_pre)
    assert torch.allclose(h1, h2, atol=1e-6), "输入归一化未在 embeddings 前施加（或位置不对）"
