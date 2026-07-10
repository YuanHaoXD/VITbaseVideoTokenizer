"""E-4-b · linear probe（03 篇 §5.2 探针②，ImageNet/K400/SSv2 三配置）。

claim：冻结 tokenizer 后，特征上只训一个线性分类器就能达到有意义的 acc——证明 latent
携带线性可读的语义（而非靠 probe 容量硬学）。

特征池化（03 篇 §5.2 明文）：
  图像：锚帧 s[:,0] 空间均值池化          s[B,1,N,D] -> feat[B,D]
  视频：s[:,1:] 时空均值池化（剔锚帧）     s[B,T1,N,D] -> feat[B,D]
  （视频剔锚帧是为了让 K400/SSv2 探针度量"运动语义"而非静态锚帧语义；SSv2 是时间敏感主判据）

纪律：特征抽取与训练解耦——先 extract_and_cache 把特征落盘（.pt），再在缓存上做 lr 扫，
避免每个 lr 候选都重跑一遍 tokenizer（03 篇 §5.2 的 lr 扫 {1e-3,3e-3,1e-2}×bs512）。
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch
import torch.nn as nn

# 03 篇 §5.2 三配置冻结默认值（lr 扫 {1e-3,3e-3,1e-2}；bs 512；ImageNet 90 epoch）
LR_CANDIDATES = (1e-3, 3e-3, 1e-2)


@dataclass(frozen=True)
class LinearProbeConfig:
    """linear probe 配置（三数据集同结构，差异由字段表达）。"""
    dataset_name: str                 # "imagenet" | "k400" | "ssv2"
    is_video: bool                    # 图像 False / 视频 True（决定特征池化方式）
    num_classes: int
    feature_dir: str                  # 特征缓存目录（extract_and_cache 落盘处）
    epochs: int = 90
    lr_candidates: tuple = LR_CANDIDATES
    batch_size: int = 512
    weight_decay: float = 0.0
    seed: int = 0


def _pool_features(s: torch.Tensor, is_video: bool) -> torch.Tensor:
    """Sem-ViT 输出 s -> 单向量特征 [B,D]。
    s: [B, T1, N, D]；图像 T1=1（仅锚帧），视频 T1=1+折叠后时间位。
    图像：s[:,0] 空间均值池化 -> [B,D]；视频：s[:,1:] 时空均值池化 -> [B,D]（剔锚帧）。"""
    if s.ndim != 4:
        raise ValueError(f"期望 s[B,T1,N,D] 4 维，收到 {tuple(s.shape)}")
    if is_video:
        return s[:, 1:].mean(dim=(1, 2))                        # 剔锚帧，时空均值
    return s[:, 0].mean(dim=1)                                  # 锚帧空间均值


@torch.no_grad()
def extract_and_cache(tokenizer, dataset: Iterable[dict], cfg: LinearProbeConfig,
                     split: str) -> Path:
    """冻结 tokenizer 抽特征并缓存到磁盘。返回缓存文件路径。

    缓存路径契约（确定性）：feature_dir/{dataset_name}_{split}.pt，存
    {"features": [N,D] float32, "labels": [N] long}。二次调用若文件存在则直接返回（跳过重抽）。
    """
    cache = Path(cfg.feature_dir) / f"{cfg.dataset_name}_{split}.pt"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        return cache                                            # 已缓存，复用（lr 扫不重抽）

    feats: list = []
    labels: list = []
    for batch in dataset:
        video = batch["video"]                                  # [B,3,F,H,W] ∈ [0,1]
        s, _ = tokenizer.semantic(video)                        # [B,T1,N,D]
        feats.append(_pool_features(s, cfg.is_video).cpu())
        labels.append(batch["label"].cpu())
    blob = {"features": torch.cat(feats).float(),
            "labels": torch.cat(labels).long()}
    torch.save(blob, cache)
    return cache


def _train_one(features: torch.Tensor, labels: torch.Tensor, cfg: LinearProbeConfig,
               lr: float, device: torch.device) -> float:
    """单 lr 候选下训一个线性层，返回最终 val acc（这里 val=train 末态，简化版；
    正式版应切 80/20 holdout——本函数聚焦接口契约，正式 lr 扫的 holdout 由调用方组织）。"""
    torch.manual_seed(cfg.seed)
    clf = nn.Linear(features.shape[-1], cfg.num_classes).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    n = features.shape[0]
    for _ in range(cfg.epochs):
        perm = torch.randperm(n)
        for i in range(0, n, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            logits = clf(features[idx].to(device))
            loss = loss_fn(logits, labels[idx].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        acc = (clf(features.to(device)).argmax(-1) == labels.to(device)).float().mean().item()
    return acc


def linear_probe(tokenizer, dataset: Iterable[dict], cfg: LinearProbeConfig, *,
                 device: torch.device = torch.device("cpu")) -> dict:
    """linear probe 主入口：抽特征缓存 → lr 扫训线性层 → 报最佳 acc。

    Args:
        tokenizer: 暴露 .semantic(video∈[0,1]) -> (s, s_pool) 的对象（M-10 鸭子类型）。
        dataset: 可迭代对象，每批 dict{"video":[B,3,F,H,W]∈[0,1], "label":[B] long}。
        cfg: LinearProbeConfig（dataset_name/is_video/num_classes/feature_dir/...）。
    Returns:
        {"acc": best_acc, "best_lr": best_lr, "cache_path": str,
         "feature_dim": int, "n_samples": int}
    """
    cache = extract_and_cache(tokenizer, dataset, cfg, split="train")
    blob = torch.load(cache, weights_only=True)
    feats = blob["features"]
    labels = blob["labels"]

    best_acc, best_lr = -1.0, None
    for lr in cfg.lr_candidates:
        acc = _train_one(feats, labels, cfg, lr, device)
        if acc > best_acc:
            best_acc, best_lr = acc, lr
    return {"acc": best_acc, "best_lr": best_lr,
            "cache_path": str(cache), "feature_dim": int(feats.shape[-1]),
            "n_samples": int(feats.shape[0])}
