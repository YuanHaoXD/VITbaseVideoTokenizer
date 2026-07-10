"""L-2 · 双教师蒸馏损失（三项余弦距离，ADR-7/9）。

对应 01 篇 §2.8 式 3 的忠实实现 + 池化项扩展，语义详见「损失详情.md」§3：
  ┌ img_patch : SemViT 锚帧 256 patch  →  SigLIP2 教师 patch 特征   （逐块图像语义）
  ┤ img_pool  : MAP 头池化整图向量 s_pool → SigLIP2 教师整图嵌入   （图像级语义；zero-shot 前提，ADR-9）
  └ vid       : Decompressor 升回 16 帧特征 → InternVideo 教师 16 帧（运动/时序；纯图像 batch 关闭）

余弦距离 d_cos(a,b)=1−cos(a,b)：只比方向不比长度（特征模长跨网络不可比，方向携带语义，蒸馏行规）。

【接口冻结】DistillLoss(nn.Module).forward(s, s_pool, decomp_out, t_img_patch, t_img_pool, t_vid, is_video)
            -> dict{img_patch, img_pool, vid, total}。

【维度对齐裁决（05 §3 L-2）】学生→教师维度不一致时，每项挂一个**可学习 Linear 头**（属训练期附件，
  同 Decompressor 一样不导出 checkpoint）；维度一致时退化为 nn.Identity 免参数。故本类须持有这些头 → 做成 nn.Module。

【空间网格对齐（ADR-7）】vid 项里 Decompressor 输出的空间网格与视频教师不一致时，对 decomp_out 做
  F.interpolate 双线性对齐到教师网格（教师侧只管原样输出，对齐在学生侧做）。

【视频项屏蔽】is_video[B] 全 False（纯图像 batch）时 vid 项返回常量 0 且**不触碰 head_vid**——
  保证 head_vid 参数梯度为 None（test_distill_masking 断言）；mixed batch 时按样本掩码只在视频样本上回传。
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def d_cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """余弦距离 1−cos，沿最后一维（特征维）计算，形状退化最后一维。"""
    return 1.0 - F.cosine_similarity(a, b, dim=-1)


def _head(student_dim: int, teacher_dim: int) -> nn.Module:
    """维度不一致→可学习 Linear（无 bias，纯方向对齐）；一致→Identity。"""
    if student_dim == teacher_dim:
        return nn.Identity()
    return nn.Linear(student_dim, teacher_dim, bias=False)


def _align_grid(feat: torch.Tensor, n_target: int) -> torch.Tensor:
    """把 [B,T,N,C] 的空间 token 数 N 双线性对齐到 n_target（ADR-7）。假设方形网格。"""
    b, t, n, c = feat.shape
    if n == n_target:
        return feat
    hw = int(round(n ** 0.5))
    hw_t = int(round(n_target ** 0.5))
    assert hw * hw == n, f"空间 token 数 {n} 非完全平方，无法推方形网格做双线性对齐"
    # [B,T,N,C] -> [B*T, C, H, W] -> interpolate -> [B,T,N',C]
    grid = feat.reshape(b * t, hw, hw, c).permute(0, 3, 1, 2)
    grid = F.interpolate(grid, size=(hw_t, hw_t), mode="bilinear", align_corners=False)
    return grid.permute(0, 2, 3, 1).reshape(b, t, hw_t * hw_t, c)


class DistillLoss(nn.Module):
    """蒸馏三项损失（持有三个学生→教师对齐头）。

    Args:
        student_dim:     SemViT / Decompressor 输出特征维 D（默认 1152，SigLIP2-So400M 宽度）。
        teacher_img_dim: 图像教师（SigLIP2）特征维 D_t_img。
        teacher_vid_dim: 视频教师（InternVideo）特征维 D_t_vid。
        cfg:             全局配置对象（读三项权重，硬约束：不在计算处硬编码）。

    说明：decomp_out 按 M-9 约定「末端 Linear 已投到教师维度」，故通常 teacher_vid_dim==student_dim 时
    head_vid 退化为 Identity；此处仍保留头位以兼容 Decompressor 输出维=D 的情形。
    """

    def __init__(self, student_dim: int = 1152, teacher_img_dim: int = 1152,
                 teacher_vid_dim: int = 1152, cfg=None):
        super().__init__()
        self.head_img_patch = _head(student_dim, teacher_img_dim)
        self.head_img_pool = _head(student_dim, teacher_img_dim)
        self.head_vid = _head(student_dim, teacher_vid_dim)
        # 权重从 cfg 读（缺字段回退到式 3 的隐含权重 1.0；λ_dist=0.5 由 trainer 在合并 recon+distill 时施加）。
        self.w_img_patch = getattr(cfg, "distill_img_patch_weight", 1.0)
        self.w_img_pool = getattr(cfg, "distill_img_pool_weight", 1.0)
        self.w_vid = getattr(cfg, "distill_vid_weight", 1.0)

    def forward(self, s: torch.Tensor, s_pool: torch.Tensor,
                decomp_out: Optional[torch.Tensor],
                t_img_patch: torch.Tensor, t_img_pool: torch.Tensor,
                t_vid: Optional[torch.Tensor], is_video: torch.Tensor) -> dict:
        """
        Args:
            s:           SemViT patch 特征 [B,T1,N,D]。
            s_pool:      MAP 池化整图向量 [B,D]。
            decomp_out:  Decompressor 输出 [B,16,N_dec,D]（纯图像时可为 None）。
            t_img_patch: SigLIP2 patch 特征 [B,N,D_t_img]（对齐锚帧 s[:,0]）。
            t_img_pool:  SigLIP2 整图嵌入 [B,D_t_img]。
            t_vid:       InternVideo 特征 [B,16,N_t,D_t_vid]（纯图像时可为 None）。
            is_video:    [B] bool/0-1，标记该样本是否为视频。
        Returns:
            dict{img_patch, img_pool, vid, total}——各项为已乘权重的标量张量。
        """
        # —— 锚帧 patch 项：s[:,0] 与 SigLIP2 的 256 patch 特征逐 token 余弦 —— (损失详情 §3 表 第1行)
        s_anchor = self.head_img_patch(s[:, 0])          # [B,N,D_t_img]
        img_patch = d_cos(s_anchor, t_img_patch).mean()  # 逐 token cos → 平均

        # —— 池化项：整图语义方向对齐（ADR-9，继承文本塔的关键） ——
        s_pool_h = self.head_img_pool(s_pool)            # [B,D_t_img]
        img_pool = d_cos(s_pool_h, t_img_pool).mean()

        # —— 视频项：Decompressor(s[:,1:]) 与 InternVideo 16 帧特征（纯图像 batch 屏蔽） ——
        vid = self._vid_term(decomp_out, t_vid, is_video, ref=s)

        img_patch_t = self.w_img_patch * img_patch
        img_pool_t = self.w_img_pool * img_pool
        vid_t = self.w_vid * vid
        total = img_patch_t + img_pool_t + vid_t
        return {"img_patch": img_patch_t, "img_pool": img_pool_t, "vid": vid_t, "total": total}

    def _vid_term(self, decomp_out: Optional[torch.Tensor], t_vid: Optional[torch.Tensor],
                  is_video: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        # 纯图像 batch / 无 decompressor 输出：返回常量 0，**绝不调用 head_vid** → 其参数梯度保持 None。
        if decomp_out is None or t_vid is None or not bool(is_video.any()):
            return ref.new_zeros(())

        dv = self.head_vid(decomp_out)                   # [B,16,N_dec,D_t_vid]
        dv = _align_grid(dv, t_vid.shape[2])             # 空间网格双线性对齐到教师（ADR-7）
        per_token = d_cos(dv, t_vid)                     # [B,16,N_t]
        per_sample = per_token.flatten(1).mean(dim=1)    # [B]，每样本逐 token 平均

        mask = is_video.to(per_sample.dtype)             # [B]，图像样本置 0
        denom = mask.sum().clamp(min=1.0)
        return (per_sample * mask).sum() / denom         # 只在视频样本上取平均，图像样本零贡献零梯度
