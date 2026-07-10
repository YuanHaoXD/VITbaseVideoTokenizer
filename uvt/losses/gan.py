"""L-3 · GAN 对抗损失（从主仓 models/loss.py 的 `lpips_disc_loss` 移植，仅 Stage 2 启用）。

语义详见「损失详情.md」§1.3：GAN 是唯一「活的」损失——判别器 D 学找茬，decoder 学消除「假的痕迹」，
逼重建落到「像真照片」的流形上，磨掉 LPIPS 满意但仍蜡质/过平滑的质感。

【移植范围】TransformerDiscriminator + ns_smooth（单边平滑）+ LeCam 正则 + d_update_freq + r1(可选)。
【剥离项（对应 05 §3 L-3 ②）】
  - VQ 相关：measure_perplexity、codebook/commitment 项——离散路线专用，UVT 连续瓶颈不需要（损失详情 §6）。
  - pixel-L1 与 LPIPS 感知项：已由 L-1 losses/recon.py 承担；此处只保留**纯对抗项**，避免在 Stage-2
    总损失里与 recon 重复计入（这是与 LARP 原 forward 的关键差异：原 forward 把 rec/perc 也捆在一起）。
【协议改动（对应 05 §3 L-3 ①）】判别器 patchify 改为我方协议：
    temporal_patch_size=4, patch_size=16, 输入 17×256² = 1 锚帧 + 16 视频帧。
  ⚠ 契约张力：17 不能被时间 patch 4 整除。裁决=**锚帧隔离**（ADR-4'，与全模型一致）：锚帧走独立空间
    patchify（时间 patch=1），非锚 16 帧走 4× 时间 patchify（16%4==0，恰好整除），拼成 1+4=5 个时间位。
    另留 `disc_anchor_isolated=False` 回退臂（要求总帧数被 temporal_patch_size 整除）。
【优化器语义（05 §3 L-3 ③）】本模块**不建优化器**（交给 trainer）；D 侧按 01 §2.9 用
    Adam(betas=(0.5, 0.9))，lr = G_lr × dis_lr_multiplier(默认 1.0)——(0.5,0.9) 是 GAN 社区经验 betas，
    抑制判别器早期过快收敛（损失详情 §1.3）。相关字段见 GANLossConfig。

损失权重全部来自 GANLossConfig（硬约束：不在计算处硬编码）。
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================================
# 1. GAN 损失原语（从 models/loss.py 原样移植；measure_perplexity 等 VQ 项已剥离）
# ======================================================================================
def lecam_reg(real_pred, fake_pred, ema_real_pred, ema_fake_pred):
    """LeCam 正则（https://arxiv.org/abs/2104.03310）：惩罚判别器过度自信，稳定/数据高效 GAN 训练。

    D 太强时 decoder 收不到有用梯度（损失详情 §1.3）；用 EMA 把 real/fake 打分互相拉住。
    """
    assert real_pred.ndim == 0 and ema_fake_pred.ndim == 0
    lecam_loss = torch.mean(torch.pow(torch.relu(real_pred - ema_fake_pred), 2))
    lecam_loss = lecam_loss + torch.mean(torch.pow(torch.relu(ema_real_pred - fake_pred), 2))
    return lecam_loss


def r1_gradient_penalty(discriminator, real_video, penalty_cost=1.0):
    """R1 梯度惩罚（可选正则，默认关）。对真样本输入求梯度并惩罚其 L2 范数。"""
    real_video = real_video.detach().clone().requires_grad_(True)
    out = discriminator(real_video)
    out = out.float()
    gradients = autograd.grad(
        outputs=out,
        inputs=real_video,
        grad_outputs=torch.ones(out.size(), device=real_video.device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(real_video.size(0), -1)
    gradient_penalty = torch.mean(torch.sum(gradients ** 2, dim=1)) * penalty_cost
    return out, gradient_penalty


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    return 0.5 * (loss_real + loss_fake)


def hinge_g_loss(logits_fake):
    return -torch.mean(logits_fake)


def ns_d_loss(logits_real, logits_fake):
    real_loss = F.binary_cross_entropy_with_logits(logits_real, torch.ones_like(logits_real))
    fake_loss = F.binary_cross_entropy_with_logits(logits_fake, torch.zeros_like(logits_fake))
    return real_loss + fake_loss


def ns_d_loss_single_side_smooth(logits_real, logits_fake):
    """ns_smooth：单边标签平滑（真标签抖到 ~0.7~1.0、假标签抖到 0~0.3），梯度性质更稳（损失详情 §1.3）。"""
    real_target = torch.ones_like(logits_real) - torch.randn_like(logits_real).abs() * 0.15
    real_target.clamp_min_(0.7)
    fake_target = torch.randn_like(logits_fake).abs() * 0.15
    fake_target.clamp_max_(0.3)
    real_loss = F.binary_cross_entropy_with_logits(logits_real, real_target)
    fake_loss = F.binary_cross_entropy_with_logits(logits_fake, fake_target)
    return real_loss + fake_loss


def ns_g_loss(logits_fake):
    return -torch.mean(F.logsigmoid(logits_fake))


def adopt_weight(weight, global_step, threshold=0, value=0.0):
    """Stage-2 起点门控：global_step < threshold 时权重置 value(0)，即判别器尚未接入。"""
    if global_step < threshold:
        weight = value
    return weight


# ======================================================================================
# 2. 正弦-余弦 3D 位置编码（从 models/embed.py 移植的纯 numpy 版，避免耦合 legacy models 包）
# ======================================================================================
def _get_1d_sincos(embed_dim, pos, scale_factor=10000):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / scale_factor ** omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def _get_2d_sincos(embed_dim, grid_size):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.stack(np.meshgrid(grid_w, grid_h), axis=0).reshape([2, 1, grid_size, grid_size])
    emb_h = _get_1d_sincos(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def _get_3d_sincos(embed_dim, grid_size, frame_num):
    emb_2d = _get_2d_sincos(embed_dim, grid_size).reshape([1, grid_size, grid_size, embed_dim])
    emb_1d = _get_1d_sincos(embed_dim, np.arange(frame_num, dtype=np.float32))
    emb_1d = emb_1d.reshape([frame_num, 1, 1, embed_dim])
    return (emb_2d + emb_1d).reshape([-1, embed_dim])


# ======================================================================================
# 3. 判别器骨架（自持 pre-LN transformer + SDPA，等价 LARP 的 timm-based TransformerEncoderFused）
# ======================================================================================
class _MHSA(nn.Module):
    """多头自注意力（SDPA 路径，qkv_bias=False，对齐 LARP TransformerEncoderFused 的 timm Block 设置）。"""

    def __init__(self, dim, n_heads, qkv_bias=False):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, s, d = x.shape
        qkv = self.qkv(x).reshape(b, s, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        out = F.scaled_dot_product_attention(q, k, v)  # 判别器全双向，无 mask
        out = out.transpose(1, 2).reshape(b, s, d)
        return self.proj(out)


class _EncoderBlock(nn.Module):
    """pre-LN transformer block：ln1→attn→残差, ln2→mlp→残差（mlp_ratio=4, GELU）。"""

    def __init__(self, dim, n_heads, mlp_ratio=4, qkv_bias=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = _MHSA(dim, n_heads, qkv_bias=qkv_bias)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerDiscriminator(nn.Module):
    """时空 transformer 判别器（移植自 LARP，patchify 换我方协议 + 锚帧隔离）。

    输入 x: [B, C, T, H, W]（我方协议 T=17=1锚+16帧）→ 输出 logits [B, 1]。
    """

    def __init__(self, hidden_size=384, n_heads=8, n_layers=8, input_size=256,
                 temporal_patch_size=4, patch_size=16, in_channels=3,
                 frame_num=16, anchor_isolated=True):
        super().__init__()
        self.anchor_isolated = anchor_isolated
        self.temporal_patch_size = temporal_patch_size
        self.patch_size = patch_size
        self.frame_num = frame_num  # 非锚帧数（anchor_isolated 时）/ 总帧数（否则）

        assert input_size % patch_size == 0, "空间边长须被 patch_size 整除"
        assert frame_num % temporal_patch_size == 0, (
            f"帧数 {frame_num} 须被 temporal_patch_size {temporal_patch_size} 整除"
            "（锚帧隔离方案下这是 16%4，天然成立）"
        )
        self.grid = input_size // patch_size            # 16
        self.num_spatial = self.grid * self.grid        # 256
        n_vid_tpos = frame_num // temporal_patch_size    # 4
        self.token_t = (1 + n_vid_tpos) if anchor_isolated else n_vid_tpos  # 5 / 4

        # 非锚帧：Conv3d 做 4× 时间 + 16×16 空间 patchify（等价 PatchEmbed3D）
        self.video_embed = nn.Conv3d(
            in_channels, hidden_size,
            kernel_size=(temporal_patch_size, patch_size, patch_size),
            stride=(temporal_patch_size, patch_size, patch_size), bias=True,
        )
        if anchor_isolated:
            # 锚帧：独立空间 patchify（时间核=1，永不折叠，镜像 OmniTokenizer first-frame 分头）
            self.anchor_embed = nn.Conv3d(
                in_channels, hidden_size,
                kernel_size=(1, patch_size, patch_size),
                stride=(1, patch_size, patch_size), bias=True,
            )

        video_token_num = self.token_t * self.num_spatial
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        # 冻结的 sincos 3D 位置编码（frozen 语义与 LARP 一致；随机学习式 PE 是替代方案）
        pos = _get_3d_sincos(hidden_size, self.grid, self.token_t)
        self.register_buffer(
            "pos_embed", torch.from_numpy(pos).float().reshape(1, video_token_num, hidden_size)
        )

        self.blocks = nn.ModuleList(
            [_EncoderBlock(hidden_size, n_heads) for _ in range(n_layers)]
        )
        self.norm_final = nn.LayerNorm(hidden_size, eps=1e-6)
        self.fc = nn.Linear(hidden_size, 1)
        self._init_weights()

    def _init_weights(self):
        def basic(m):
            if isinstance(m, (nn.Linear, nn.Conv3d)):
                w = m.weight.data
                nn.init.xavier_uniform_(w.view(w.shape[0], -1))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(basic)
        nn.init.xavier_uniform_(self.cls_token)

    def _patchify(self, x):
        if self.anchor_isolated:
            anchor, frames = x[:, :, :1], x[:, :, 1:]
            a = self.anchor_embed(anchor).flatten(2).transpose(1, 2)   # [B, hw, D]
            f = self.video_embed(frames).flatten(2).transpose(1, 2)    # [B, t*hw, D]
            return torch.cat([a, f], dim=1)                            # [B, (1+t)*hw, D]
        return self.video_embed(x).flatten(2).transpose(1, 2)

    def forward(self, x):
        """x: [B,C,T,H,W] → logits [B,1]。"""
        tokens = self._patchify(x) + self.pos_embed
        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        z = torch.cat([cls, tokens], dim=1)
        for blk in self.blocks:
            z = blk(z)
        z_cls = self.norm_final(z[:, 0])
        return self.fc(z_cls)


# ======================================================================================
# 4. GAN 损失配置 + 主模块
# ======================================================================================
@dataclass
class GANLossConfig:
    """Stage-2 GAN 全部超参（01 §2.9 LARP 配方）。硬约束：损失权重从此读取，不在计算处硬编码。

    contract note → 这些字段与 M-10 的 UVTConfig 相互独立（GAN 仅 Stage-2 用）；如需统一，
    可由 M-10 把本 dataclass 内嵌进 UVTConfig，本模块只依赖属性名不依赖来源。
    """
    # —— 判别器结构（transformer-D 384×8L） ——
    disc_hidden_size: int = 384
    disc_n_heads: int = 8
    disc_n_layers: int = 8
    disc_input_size: int = 256
    disc_tran_temporal_patch_size: int = 4     # 我方协议（vs LARP 默认 1）
    disc_tran_patch_size: int = 16
    disc_in_channels: int = 3
    disc_frame_num: int = 16                    # 非锚帧数；总输入 = 1 锚 + 16 = 17 帧
    disc_anchor_isolated: bool = True
    # —— 损失/正则 ——
    disc_loss: str = "ns_smooth"                # {hinge, ns, ns_smooth}
    disc_weight: float = 0.3                    # §2.9 disc_weight 0.3
    disc_factor: float = 1.0
    lecam_weight: float = 1e-3                  # §2.9 LeCam 1e-3
    lecam_ema_decay: float = 0.999
    r1_gp_weight: float = 0.0                   # r1 可选，默认关
    d_update_freq: int = 5                      # §2.9 D 每 5 步更新一次
    disc_start: int = 0                         # 判别器接入步（Stage-2 内的相对 global_step）
    # —— 优化器语义（本模块不建优化器；trainer 按此建 D 侧 Adam(0.5,0.9)） ——
    dis_lr_multiplier: float = 1.0             # D lr = G lr × 此值（§2.9：×1）
    dis_adam_betas: tuple = (0.5, 0.9)         # GAN 社区经验 betas


class UVTGANLoss(nn.Module):
    """Stage-2 对抗损失（生成器项 + 判别器项分开调用，交替优化）。

    用法（trainer）：
        gan = UVTGANLoss(GANLossConfig())
        # —— G 步（decoder 更新，每步都算） ——
        g_loss, g_info = gan.generator_loss(x_hat, global_step)   # 加进 recon.total 一起 backward
        # —— D 步（判别器更新，d_update_freq 步一次） ——
        if gan.should_update_d(global_step):
            d_loss, d_info = gan.discriminator_loss(x_real, x_hat.detach(), global_step)
    """

    def __init__(self, cfg: GANLossConfig = None):
        super().__init__()
        self.cfg = cfg = cfg or GANLossConfig()
        assert cfg.disc_loss in ("hinge", "ns", "ns_smooth"), f"未知 disc_loss: {cfg.disc_loss}"

        self.discriminator = TransformerDiscriminator(
            hidden_size=cfg.disc_hidden_size,
            n_heads=cfg.disc_n_heads,
            n_layers=cfg.disc_n_layers,
            input_size=cfg.disc_input_size,
            temporal_patch_size=cfg.disc_tran_temporal_patch_size,
            patch_size=cfg.disc_tran_patch_size,
            in_channels=cfg.disc_in_channels,
            frame_num=cfg.disc_frame_num,
            anchor_isolated=cfg.disc_anchor_isolated,
        )

        if cfg.disc_loss == "hinge":
            self.d_loss_fn, self.g_loss_fn = hinge_d_loss, hinge_g_loss
        elif cfg.disc_loss == "ns":
            self.d_loss_fn, self.g_loss_fn = ns_d_loss, ns_g_loss
        else:  # ns_smooth
            self.d_loss_fn, self.g_loss_fn = ns_d_loss_single_side_smooth, ns_g_loss

        if cfg.lecam_weight > 0.0:
            self.register_buffer("lecam_ema_real", torch.tensor(0.0))
            self.register_buffer("lecam_ema_fake", torch.tensor(0.0))

    # —— d_update_freq 机制：D 每 cfg.d_update_freq 步才更新一次（损失详情 §1.3 稳定器） ——
    def should_update_d(self, global_step: int) -> bool:
        return (global_step % self.cfg.d_update_freq) == 0

    @torch.no_grad()
    def _update_lecam_ema(self, real, fake):
        decay = self.cfg.lecam_ema_decay
        real, fake = real.float().mean(), fake.float().mean()
        self.lecam_ema_real.mul_(decay).add_(real, alpha=1 - decay)
        self.lecam_ema_fake.mul_(decay).add_(fake, alpha=1 - decay)

    def generator_loss(self, reconstructions: torch.Tensor, global_step: int):
        """生成器侧**纯对抗**损失（不含 L1/LPIPS——那在 recon.py）。返回 (loss, info)。"""
        cfg = self.cfg
        disc_factor = adopt_weight(cfg.disc_factor, global_step, threshold=cfg.disc_start)
        if disc_factor <= 0.0:
            zero = reconstructions.new_zeros(())
            return zero, {"g_loss": 0.0, "g_loss_weight": 0.0}
        logits_fake = self.discriminator(reconstructions)
        g_loss = self.g_loss_fn(logits_fake)
        weight = cfg.disc_weight * disc_factor
        loss = weight * g_loss
        return loss, {"g_loss": g_loss.item(), "g_loss_weight": weight}

    def discriminator_loss(self, inputs: torch.Tensor, reconstructions: torch.Tensor,
                           global_step: int):
        """判别器侧损失 = d_loss + LeCam(+r1)。reconstructions 需已 detach。返回 (loss, info)。"""
        cfg = self.cfg
        disc_factor = adopt_weight(cfg.disc_factor, global_step, threshold=cfg.disc_start)
        if disc_factor <= 0.0:
            zero = inputs.new_zeros(())
            return zero, {"d_loss": 0.0, "d_lecam_loss": 0.0, "logits_real": 0.0, "logits_fake": 0.0}

        if self.training and cfg.r1_gp_weight > 0.0:
            logits_real, r1_gp = r1_gradient_penalty(
                self.discriminator, inputs.contiguous(), penalty_cost=cfg.r1_gp_weight
            )
        else:
            logits_real = self.discriminator(inputs.contiguous())
            r1_gp = inputs.new_zeros(())
        logits_fake = self.discriminator(reconstructions.contiguous().detach())

        if cfg.lecam_weight > 0.0:
            # 注：LARP 原实现把 lecam_weight 施加了两次（lecam_loss 内一次、total 里又一次），
            # 此处按正确语义**只施加一次**（§2.9 明确 LeCam=1e-3 为我方目标值）。
            lecam = lecam_reg(
                real_pred=logits_real.mean(), fake_pred=logits_fake.mean(),
                ema_real_pred=self.lecam_ema_real, ema_fake_pred=self.lecam_ema_fake,
            )
            self._update_lecam_ema(logits_real, logits_fake)
        else:
            lecam = inputs.new_zeros(())

        d_loss = self.d_loss_fn(logits_real, logits_fake)
        total = d_loss + cfg.lecam_weight * lecam + r1_gp
        info = {
            "d_loss": d_loss.item(),
            "d_lecam_loss": float(lecam.item()),
            "logits_real": logits_real.mean().item(),
            "logits_fake": logits_fake.mean().item(),
        }
        if cfg.r1_gp_weight > 0.0:
            info["r1_gp"] = float(r1_gp.item())
        return total, info
