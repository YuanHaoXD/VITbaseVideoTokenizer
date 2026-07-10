"""GSB · 连续瓶颈（M-5，参见 01 §2.6）。

职责：把 h[B,T1,N,D] 压成唯一对外 latent z[B,T1,N,c_latent]（生成/重建/语义共同接口），
并提供规范化往返（ADR-5：Stage 3 起 ẑ=(z-m)/s）。

与 01 §2.6 参考实现的差异（任务书 §2 M-5 裁决）：
  **删去 expand/unproj**——反投影（64→D）按 HYDRA 的 W_unproj^und 上标拆到各消费方：
  Sem-ViT 与 decoder 各自持有独立 `in_proj: Linear(c_latent→D)`。否则 Stage-3
  "冻 decoder 训 Sem 侧" 会在共享投影上打架。故本类只有前向 proj，无反投影。
"""
import torch
import torch.nn as nn


class GSB(nn.Module):
    def __init__(self, d_model: int = 1152, c_latent: int = 64):
        super().__init__()
        # HYDRA 原文式 2：proj 输出 2*c_latent，chunk 成 (μ, ρ)。
        self.proj = nn.Linear(d_model, 2 * c_latent)
        # 通道级统计缓冲（Stage 3 由 estimate_latent_stats 写入），随 checkpoint 持久化。
        self.register_buffer("z_mean", torch.zeros(c_latent))
        self.register_buffer("z_std", torch.ones(c_latent))
        # 规范化开关：Stage 3 置 True（ADR-5）。默认关，前两阶段消费物理 latent。
        # 架构方契约修订 2026-07-07（契约问题#4）：开关必须随 checkpoint 持久化——
        # 否则 Stage-3 训完重载后 normalize 静默回到 False，规范化接口失效（静默错误）。
        # 用 0/1 标量 buffer 承载，property 保持 `.normalize` 的 bool 读写接口不变。
        self.register_buffer("_normalize_flag", torch.zeros((), dtype=torch.uint8))

    @property
    def normalize(self) -> bool:
        return bool(self._normalize_flag.item())

    @normalize.setter
    def normalize(self, value: bool) -> None:
        self._normalize_flag.fill_(1 if value else 0)

    def compress(self, h: torch.Tensor):
        """h:[B,T1,N,D] -> (z, mu, kl)。z 为物理（未规范化）latent。"""
        mu, rho = self.proj(h).chunk(2, dim=-1)
        # ρ clamp(-30, 20)：LeanVAE 同款数值保护，防 exp 溢出/下溢。
        rho = rho.clamp(-30.0, 20.0)
        # 重参数化：z = μ + ε·exp(0.5ρ)。
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * rho)
        # KL = -0.5·mean(1 + ρ - μ² - e^ρ)（标准 VAE 项，按元素平均）。
        kl = -0.5 * (1 + rho - mu.pow(2) - rho.exp()).mean()
        return z, mu, kl

    def to_canonical(self, z: torch.Tensor) -> torch.Tensor:
        """物理 latent -> 规范化 ẑ（normalize=False 时直通）。"""
        return (z - self.z_mean) / self.z_std if self.normalize else z

    def from_canonical(self, z_can: torch.Tensor) -> torch.Tensor:
        """规范化 ẑ -> 物理 latent（to_canonical 的逆，normalize=False 时直通）。"""
        return z_can * self.z_std + self.z_mean if self.normalize else z_can
