"""L-1 · 重建损失（像素三兄弟的 L1/LPIPS + latent 侧 KL + 瓶颈一致性 L_cos）。

对应 01 篇 §2.9 Stage-1 基础损失里的 L1 + LPIPS + KL + L_cos 四项，语义详见「损失详情.md」§1/§2：
  - L1（结构/颜色/位置）：逐像素绝对差，温和不糊化（不用 L2）。
  - LPIPS（感知）：冻结 VGG 特征差，专治 L1 的糊。
  - KL（latent 规整，权重 1e-6 起，D14）：把 GSB 输出拢向 N(0,1)，KL 过大是「重建发糊」头号嫌疑。
  - L_cos（HYDRA 式 4，01 §2.6 / 损失详情 §2.2）：压缩-展开后特征方向不迷路。

【接口冻结】recon_loss(x, out, cfg) -> dict{l1, lpips, kl, cos_consistency, total}。

【L_cos 的模块解耦裁决】L_cos 需要「SemViT.in_proj 投影后的 μ」与「瓶颈前 h」。为了不让损失模块反向
依赖模型模块（GSB.expand 已按 M-5 删除、反投影下放到 SemViT.in_proj / decoder.in_proj），本函数**不**自己
调用任何投影层，而是要求调用方（trainer / UVTTokenizer.forward_train）预先把投影结果写进 out：
    out["mu_proj"] = sem_vit.in_proj(gsb.to_canonical(mu))   # [B,T1,N,D]
    out["h"]       = 瓶颈前 Gen-ViT 输出 h                     # [B,T1,N,D]
缺失任一键或 cfg.use_cos_consistency=False 时，cos_consistency 记 0（不产生梯度）。
"""
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange

# LPIPS 网络单例缓存：LPIPS 内含冻结 VGG，反复构造代价高，按设备缓存一份。
_LPIPS_CACHE = {}


def _get_lpips(device: torch.device, net: str = "vgg"):
    """惰性加载 LPIPS（import 放函数内：环境无 lpips 库时给出清晰报错，而非污染模块导入）。"""
    key = (str(device), net)
    if key not in _LPIPS_CACHE:
        try:
            import lpips  # noqa: PLC0415  —— 故意函数内导入，见上
        except ImportError as e:  # pragma: no cover - 环境相关
            raise ImportError(
                "losses.recon 需要 `lpips` 库计算感知损失（LARP 同款 net='vgg'）。"
                "请 `pip install lpips`，或在 cfg 里把 lpips_weight 置 0 关闭该项。"
            ) from e
        model = lpips.LPIPS(net=net).to(device)
        model.eval()
        for p in model.parameters():  # 冻结：LPIPS 只作度量，不接收梯度
            p.requires_grad_(False)
        _LPIPS_CACHE[key] = model
    return _LPIPS_CACHE[key]


def _to_5d(x: torch.Tensor) -> torch.Tensor:
    """像素张量统一成 [B,3,F,H,W]（图像 [B,3,H,W] 在时间维升 1，与 §0 约定一致）。"""
    if x.dim() == 4:
        return x.unsqueeze(2)
    assert x.dim() == 5, f"像素张量需为 4D/5D，收到 {x.dim()}D"
    return x


def _cfg(cfg, name: str, default):
    """从全局 cfg 读损失权重（硬约束：不在计算处硬编码；cfg 暂缺该字段时回退到 §2.9 文档默认值）。

    contract note → 建议 M-10 的 UVTConfig 补入 l1_weight / lpips_weight / cos_weight 字段，
    届时本 getattr 回退即自然失效。kl_weight 已在 UVTConfig 中（默认 1e-6）。
    """
    return getattr(cfg, name, default)


def recon_loss(x: torch.Tensor, out: dict, cfg) -> dict:
    """重建损失总入口。

    Args:
        x:   原始像素 [B,3,H,W] 或 [B,3,1+T,H,W]，取值区间与 decoder 输出一致（默认 [0,1]，LARP 惯例）。
        out: UVTTokenizer.forward_train 的输出 dict，至少含 "x_hat" 与 "kl"；
             若开 L_cos 还需 "mu_proj" 与 "h"（见模块 docstring）。
        cfg: 全局配置对象（读取 l1_weight / lpips_weight / kl_weight / cos_weight / use_cos_consistency）。

    Returns:
        dict{l1, lpips, kl, cos_consistency, total}——前四项为「已乘权重」的标量张量，total 为其和。
    """
    x = _to_5d(x)
    x_hat = _to_5d(out["x_hat"])
    assert x.shape == x_hat.shape, f"x{tuple(x.shape)} 与 x_hat{tuple(x_hat.shape)} 形状不一致"

    w_l1 = _cfg(cfg, "l1_weight", 1.0)
    w_lpips = _cfg(cfg, "lpips_weight", 1.0)
    w_kl = _cfg(cfg, "kl_weight", 1e-6)
    w_cos = _cfg(cfg, "cos_weight", 1.0)

    # —— L1：逐像素绝对差取平均（损失详情 §1.1：温和、出图更锐） ——
    l1 = (x - x_hat).abs().mean()

    # —— LPIPS：逐帧过冻结 VGG 比中间层特征（损失详情 §1.2） ——
    if w_lpips > 0:
        frames_x = rearrange(x, "b c t h w -> (b t) c h w").contiguous()
        frames_xh = rearrange(x_hat, "b c t h w -> (b t) c h w").contiguous()
        lpips_net = _get_lpips(x.device)
        # normalize=True：LARP 同款，输入按 [0,1] 语义再内部映射到 [-1,1]（cfg 可覆盖）。
        lpips_val = lpips_net(
            frames_x, frames_xh, normalize=_cfg(cfg, "lpips_normalize", True)
        ).mean()
    else:
        lpips_val = x.new_zeros(())

    # —— KL：GSB 已在 compress() 里算好（01 §2.6），此处只按权重计入 ——
    kl = out.get("kl", x.new_zeros(()))
    if not torch.is_tensor(kl):
        kl = x.new_tensor(float(kl))

    # —— L_cos：unproj(μ) 与瓶颈前 h 的方向对齐，按 token 平均（HYDRA 式 4） ——
    cos_consistency = _cos_consistency(out, cfg, x)

    l1_t = w_l1 * l1
    lpips_t = w_lpips * lpips_val
    kl_t = w_kl * kl
    cos_t = w_cos * cos_consistency
    total = l1_t + lpips_t + kl_t + cos_t
    return {"l1": l1_t, "lpips": lpips_t, "kl": kl_t, "cos_consistency": cos_t, "total": total}


def _cos_consistency(out: dict, cfg, ref: torch.Tensor) -> torch.Tensor:
    """L_cos = mean_token(1 − cos(mu_proj, h))；开关关或缺键时返回常量 0（无梯度）。"""
    if not _cfg(cfg, "use_cos_consistency", True):
        return ref.new_zeros(())
    mu_proj: Optional[torch.Tensor] = out.get("mu_proj")
    h: Optional[torch.Tensor] = out.get("h")
    if mu_proj is None or h is None:
        # 架构方契约修订 2026-07-08（任务书 v1.2）：开关开着却缺键 = 契约违反，必须炸而不是静默记 0。
        # 静默失效正是本项目防线要消灭的 bug 类别（同 GSB.normalize 持久化问题）；
        # M-10 的 forward_train 已被契约要求输出 mu_proj 与 h。
        raise ValueError(
            "use_cos_consistency=True 但 out 缺 'mu_proj'/'h'——调用方违反 L-1 契约"
            "（M-10 forward_train 须输出 mu_proj = sem_vit.in_proj(gsb.to_canonical(mu)) 与 h）；"
            "若本阶段不需要 L_cos，请在 cfg 显式置 use_cos_consistency=False。"
        )
    # 沿特征维 D 求余弦，[B,T1,N,D] -> [B,T1,N]，再对全部 token 取平均。
    cos = F.cosine_similarity(mu_proj, h, dim=-1)
    return (1.0 - cos).mean()
