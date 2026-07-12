"""UVTTokenizer · 总装（M-10，张量流见 01 §1 全图）。

组装 GenViT(M-6) / GSB(M-5) / PixelDecoder(M-8) / SemViT(M-7) / Decompressor(M-9)，
提供训练前向（forward_train）、推理前向（_inference，经 forward 训练态分派）、
阶段冻结（set_stage，01 §2.9 表）、Stage-3 latent 统计（estimate_latent_stats，ADR-5）、
HF 发布接口（PyTorchModelHubMixin，照抄 LARP 用法）。

【normalize 三路分流（契约②，Stage-3 命门）】
  - h = genvit(video)；z, mu, kl = gsb.compress(h)
  - decoder 吃**物理** z：`x_hat = decoder(z, hw)`            （非规范化）
  - sem_vit 吃**规范** z：`s, s_pool = sem_vit(gsb.to_canonical(z))`
  - L_cos 的 mu_proj = `sem_vit.in_proj(gsb.to_canonical(mu))` （canonical μ 再 in_proj）
  - out['h'] = gsb.norm(h)（瓶颈入口规范化后特征，第 15 号修复；L_cos 对齐目标）
  normalize=False（Stage1/2）时 to_canonical 恒等，全自洽。

【DDP 契约③】forward 训练态必须分派到 forward_train——DDP 只 hook forward，
  TR-2 依赖此分派才走训练图（绕过则梯度不同步）。

【gsb 无 expand/unproj（契约④）】反投影归各消费方自持 in_proj（sem_vit.in_proj / decoder.in_proj）。

【decomp_out 仅视频（契约⑥）】forward_train 只在输入是视频（F>1）时调 Decompressor(s[:,1:])；
  图像 batch（F=1）的 decomp_out=None（L-2 vid 项据此屏蔽）。

【state_dict 钩子（契约⑤）】post-hook 剔除 Decompressor（训练期专用，不导出）。
  注：gsb.normalize 自架构方 v1.1 修订起已由持久化 buffer `_normalize_flag` 承载，
  随 checkpoint 自动恢复——无需额外的加载钩子（与早期 stale 文字不同，以 gsb.py 实际代码为准）。
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from .decoder import PixelDecoder
from .decompressor import Decompressor
from .encoder import GenViT
from .gsb import GSB
from .sem_vit import SemViT
from .siglip_backbone import load_siglip_parts

# 训练框架注册：models.make 从 yaml 的 model.args 实例化（同 larp_tokenizer 的 @register 模式）。
# 置于相对导入之后；models/__init__ 第 1 行已定义 register，此处 from models import 不构成循环。
from models import register  # noqa: E402


@dataclass
class UVTConfig:
    """全局配置（任务书 §0.5：贯穿所有模块，字段含默认值与 ADR 出处）。"""

    model_name: str = "google/siglip2-so400m-patch16-256"   # §2.1：固定分辨率 256 权重（非 naflex）
    gen_depth: int = 13                  # ADR：HYDRA 均衡切分（27 层取 13+14）
    c_latent: int = 64                   # HYDRA C=64 甜点（§1 压缩率 1152→64）
    fold_positions: tuple = (0, 6)       # ADR-3 默认 {输入处, Gen-ViT 第6层后}；P2-pre 后冻结
    unfold_positions: tuple = (21, 27)   # 镜像 fold（§2.7；27=末块之后）
    attn_mode: str = "tubelet"           # ADR-2 {full,causal,tubelet}；P2 消融
    rope_dims: int = 32                  # ADR-8 时间 RoPE（每头前 32 维）；0=关（消融臂）
    decoder_init: str = "siglip"         # §2.7 {siglip,random}；P1 消融
    use_cos_consistency: bool = True     # HYDRA L_cos（§2.6 式4）；P1 消融（可关）
    kl_weight: float = 1e-6              # D14 / §2.9（KL 起 1e-6 扫）
    tiny: bool = False                   # 离线小模型（测试用，禁下载权重，§0）

    # —— 损失权重字段（L-1/L-2 用 getattr 读；定义它们消除回退，任务书 v1.2 契约问题#①）——
    l1_weight: float = 1.0               # §2.9 Stage-1
    lpips_weight: float = 1.0            # §2.9 Stage-1（扫 {0.5,1,4}）
    cos_weight: float = 1.0              # §2.9 L_cos 权重
    distill_img_patch_weight: float = 1.0  # §2.8 式3 隐含权重
    distill_img_pool_weight: float = 1.0   # §2.8 / ADR-9 池化项
    distill_vid_weight: float = 1.0        # §2.8 视频项
    lambda_dist: float = 0.5             # §2.8 默认 0.5（P3 扫 {0.25,0.5,1.0}），trainer 合并时施加


def _strip_decompressor_hook(module: nn.Module, destination: dict, prefix: str,
                             local_metadata: dict) -> None:
    """state_dict post-hook：剔除 Decompressor.* 键（训练期专用，checkpoint 不导出）。

    在本模块（UVTTokenizer）子树全部写入 destination 之后触发，按 "decompressor." 子串
    匹配删除（不依赖 prefix 细节，跨 DDP/module 包装均稳健）。PyTorch≥2.1 的 post-hook API。
    """
    for k in [key for key in destination if "decompressor." in key]:
        del destination[k]


@register('uvt_tokenizer')
class UVTTokenizer(nn.Module, PyTorchModelHubMixin):
    """UVT 总装模型。发布方式学 LARP（plain 继承 PyTorchModelHubMixin）。"""

    def __init__(self, cfg: "UVTConfig | None" = None, **kwargs):
        super().__init__()
        # 兼容两条构造路径：
        #   ① 代码侧传 UVTConfig 实例（单测 / 直接 new）；
        #   ② 训练框架 models.make 从 yaml 的 model.args 以 **kwargs 传入（config-driven；
        #      models/models.py:make 见签名含 **kwargs 时不过滤参数、全部透传）。
        if cfg is None:
            cfg = UVTConfig(**kwargs) if kwargs else UVTConfig()
        self.cfg = cfg
        parts = load_siglip_parts(self.cfg.model_name, self.cfg.gen_depth,
                                  self.cfg.tiny, self.cfg.rope_dims)

        # tiny 健壮性（交叉验证发现）：默认 fold/unfold 位置 ((0,6)/(21,27)) 是为生产 27 层
        # 设的；tiny 骨干层数远少（如 6 层），位置越界会触发 encoder/decoder 的 range 断言。
        # 仅当位置**实际越界**时按层数自动派生（生产 tiny=False 永不触发，零影响）。
        n_gen = len(parts.gen_blocks)   # tiny 下 gen_depth 被夹到实际层数
        n_dec = len(parts.dec_blocks)
        if n_gen > 0 and max(self.cfg.fold_positions) > n_gen:
            self.cfg.fold_positions = (0, max(1, n_gen // 2))
        if n_dec > 0 and max(self.cfg.unfold_positions) > n_dec:
            self.cfg.unfold_positions = (max(1, n_dec // 2), n_dec)

        # 各部件注册：parts 各字段与消费方一一对应，无重复注册。
        self.encoder = GenViT(parts, self.cfg)
        dim = parts.gen_blocks[0].attn.dim                 # D（1152 或 tiny 的 64）
        self.gsb = GSB(dim, self.cfg.c_latent)
        self.decoder = PixelDecoder(parts, self.cfg)
        self.sem_vit = SemViT(parts, self.cfg)

        # Decompressor：num_heads 取骨干头数（蒸馏专用，checkpoint 不导出）。
        num_heads = parts.gen_blocks[0].attn.num_heads
        self.decompressor = Decompressor(dim, num_heads)

        # state_dict post-hook：剔除 Decompressor（训练期附件）。
        self.register_state_dict_post_hook(_strip_decompressor_hook)

        # 默认 Stage 1（全参可训）；trainer 会在 DDP 构造前调 set_stage 覆盖。
        self.set_stage(1)

    # ------------------------------------------------------------------ 属性
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    # ------------------------------------------------------------------ 前向分派（DDP 契约③）
    def forward(self, video: torch.Tensor) -> dict:
        """训练态 → forward_train；推理态 → _inference。DDP 只 hook forward，故必须经此分派。"""
        return self.forward_train(video) if self.training else self._inference(video)

    # ------------------------------------------------------------------ 核心
    def _forward_core(self, video: torch.Tensor, sample: bool) -> dict:
        """forward_train / _inference 共用核心。返回 L-1/L-2/TR-2 全部所需键。

        normalize 三路分流（契约②）：decoder 吃物理 z；sem_vit 吃规范 z；mu_proj = in_proj(canonical μ)。
        """
        attn_mode = self.cfg.attn_mode
        if video.dim() == 4:                                   # 图像升维 [B,3,1,H,W]
            video = video.unsqueeze(2)
        B, _, F, H, W = video.shape
        hw = (H, W)

        # ① GenViT → h（瓶颈前，物理）
        h = self.encoder(video, attn_mode)                     # [B, 1+(F-1)//4, N, D]

        # ② GSB 压缩 → z(物理), mu, kl（确定性路径已收口进 compress(sample=False)，
        #    第 15 号修复：此前直捅 gsb.proj 会绕过瓶颈入口 LN）
        z, mu, kl = self.gsb.compress(h, sample=sample)

        # ③ decoder 吃**物理** z（ADR-5：decoder 物理上永远消费反归一化的 z）
        x_hat = self.decoder(z, hw, attn_mode)                 # [B,3,1+T,H,W]

        # ④ sem_vit 吃**规范** z（Stage-3 起规范化，之前 to_canonical 恒等）
        s, s_pool = self.sem_vit(self.gsb.to_canonical(z), attn_mode)  # s[B,T1,N,D], s_pool[B,D]

        # ⑤ L_cos 的 mu_proj = sem_vit.in_proj(canonical μ)（契约②；L-1 直接读此键）
        mu_proj = self.sem_vit.in_proj(self.gsb.to_canonical(mu))      # [B,T1,N,D]

        out = {
            "x_hat": x_hat,
            # out["h"] = 瓶颈入口规范化后的特征（gsb.norm(h)，第 15 号修复）：L_cos 的对齐
            # 目标应是瓶颈实际消费的特征——裸 h 被巨激活主导，余弦对齐会被共享分量平凡满足。
            "z": z, "mu": mu, "kl": kl, "h": self.gsb.norm(h),
            "s": s, "s_pool": s_pool, "mu_proj": mu_proj,
        }

        # ⑥ decomp_out 仅视频（契约⑥）：F>1 才升采样 s[:,1:]；图像 batch 置 None（L-2 vid 据此屏蔽）。
        if F > 1:
            out["decomp_out"] = self.decompressor(s[:, 1:])    # [B, 4*(T1-1), N, D]
        else:
            out["decomp_out"] = None
        return out

    def forward_train(self, video: torch.Tensor) -> dict:
        """训练前向：{x_hat, z, mu, kl, h, s, s_pool, mu_proj, decomp_out}。"""
        return self._forward_core(video, sample=True)

    @torch.no_grad()
    def _inference(self, video: torch.Tensor) -> dict:
        """推理前向（确定性 z=μ）：键集与 forward_train 一致，供 eval/FVD/评测读出。"""
        return self._forward_core(video, sample=False)

    # ------------------------------------------------------------------ 分项接口
    def encode(self, video: torch.Tensor, sample: bool = True) -> dict:
        """{z, mu, kl, h}。z 为物理 latent。sample=False 时 z=μ（确定性）。
        h 为瓶颈入口规范化后的特征（gsb.norm，第 15 号修复，与 forward_train 的 out['h'] 一致）。"""
        if video.dim() == 4:
            video = video.unsqueeze(2)
        h = self.encoder(video, self.cfg.attn_mode)
        z, mu, kl = self.gsb.compress(h, sample=sample)
        return {"z": z, "mu": mu, "kl": kl, "h": self.gsb.norm(h)}

    def decode(self, z_phys: torch.Tensor, hw) -> torch.Tensor:
        """物理 latent → 像素（消费物理 z，非规范化）。"""
        return self.decoder(z_phys, hw, self.cfg.attn_mode)

    @torch.no_grad()
    def semantic(self, video: torch.Tensor):
        """推理用：(s, s_pool)。内部走规范化 z（契约 M-10）。确定性 z=μ。"""
        enc = self.encode(video, sample=False)
        return self.sem_vit(self.gsb.to_canonical(enc["z"]), self.cfg.attn_mode)

    # ------------------------------------------------------------------ Stage-3 入口（ADR-5）
    @torch.no_grad()
    def estimate_latent_stats(self, loader, n: int = 10_000) -> None:
        """统计 z 通道 mean/std 写入 GSB buffer 并置 normalize=True（ADR-5）。

        n 为累计 token 上限（64 通道统计足够稳定）；loader 产出 dict 含 'video' 或 'gt' 键。
        normalize 由持久化 buffer 承载（gsb.py），随 checkpoint 自动恢复。
        """
        c = self.cfg.c_latent
        sum_ = torch.zeros(c, device=self.device)
        sumsq = torch.zeros(c, device=self.device)
        count = 0
        for batch in loader:
            x = batch.get("video", batch.get("gt")) if isinstance(batch, dict) else None
            if x is None:
                continue
            x = x.to(self.device)
            if x.dim() == 4:
                x = x.unsqueeze(2)
            h = self.encoder(x, self.cfg.attn_mode)
            z, _, _ = self.gsb.compress(h)                     # 物理 z
            zf = z.reshape(-1, c).float()                      # [M, c]
            sum_ += zf.sum(dim=0)
            sumsq += zf.pow(2).sum(dim=0)
            count += zf.shape[0]
            if count >= n:
                break
        assert count > 0, "estimate_latent_stats: loader 产出 0 个样本"
        mean = sum_ / count
        var = (sumsq / count) - mean.pow(2)
        std = var.clamp(min=1e-6).sqrt()
        self.gsb.z_mean.copy_(mean.to(self.gsb.z_mean.dtype))
        self.gsb.z_std.copy_(std.to(self.gsb.z_std.dtype))
        self.gsb.normalize = True

    # ------------------------------------------------------------------ 阶段冻结（01 §2.9 表）
    def set_stage(self, stage: int) -> None:
        """1/2/3 冻结策略（01 §2.9 损失总表）。

        Stage 1 基础：全部可训。
        Stage 2 精修：仅 decoder 可训（encoder/GSB 冻结；§2.9「仅 decoder」）。
        Stage 3 调和：仅 Sem-ViT（含 map_head）可训；其余冻结（decomp_out 梯度仍经冻结的
                      Decompressor 流回 Sem-ViT，不影响 Sem-ViT 训练）。
        """
        assert stage in (1, 2, 3), f"stage 必须为 1/2/3，实得 {stage}"
        self.stage = stage

        def _set(module: nn.Module, rg: bool) -> None:
            for p in module.parameters():
                p.requires_grad_(rg)

        if stage == 1:
            _set(self.encoder, True)
            _set(self.gsb, True)
            _set(self.decoder, True)
            _set(self.sem_vit, True)
            _set(self.decompressor, True)
        elif stage == 2:
            _set(self.encoder, False)
            _set(self.gsb, False)
            _set(self.decoder, True)
            _set(self.sem_vit, False)
            _set(self.decompressor, False)
        else:  # stage == 3
            _set(self.encoder, False)
            _set(self.gsb, False)
            _set(self.decoder, False)
            _set(self.sem_vit, True)        # 含 map_head（SemViT.map_head 为其子模块）
            _set(self.decompressor, False)
