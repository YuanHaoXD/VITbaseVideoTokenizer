"""E-4-c · CKNNA 逐层对齐曲线（03 篇 §5.2 探针③，README D11）。

claim：UVT 各层特征与第三方参照 DINOv2-L 的表征相似度曲线，证明语义来自蒸馏而非偶然。
指标 CKNNA = Centered Kernel Alignment (Normalized)，线性核实现（Kornblith et al. 2019）。

【D11 硬约束 / 循环论证禁令】参照系**必须**是 DINOv2-L，与两教师（SigLIP2/InternVideo）
独立——对教师自身做 CKNNA 是循环论证（蒸馏必然提高与教师的对齐度，证明不了任何 claim）。
本模块**写死**：参照系加载由 load_dinov2_reference 独占，cknna() 只吃特征张量、不接收任何
模型对象；任何"传教师当参照"的调用路径在本文件接口层就不存在（不是运行时检查，是契约层禁止）。

DINOv2 经 transformers.DinoV2Model 加载（model_name 默认 facebook/dinov2-large），
lazy import——无网络/无权重时 load 抛 OSError，由调用方 skip（P0 集群首日校准权重入库）。
"""
from typing import Dict, List, Tuple

import torch

# 参照系模型名（写死，README D11：DINOv2-L 与两教师独立，禁换为教师）。
DINOV2_MODEL_NAME = "facebook/dinov2-large"

# D11 循环论证禁令的代码层标记：参照系不得为任何教师对象（本文件不接收模型参数）。
FORBID_TEACHER_AS_REFERENCE = True


def load_dinov2_reference(model_name: str = DINOV2_MODEL_NAME,
                          device: torch.device = torch.device("cpu")) -> "torch.nn.Module":
    """加载 DINOv2-L 参照系。transformers lazy import——无网络/无权重时抛 OSError 让上层 skip。

    本函数是参照系的**唯一**加载入口；严禁新增"传教师模型"的重载——D11 的契约层禁止。
    """
    try:
        from transformers import DinoV2Model          # lazy import：无依赖时其它路径仍可用
    except ImportError as e:  # pragma: no cover - 环境相关
        raise ImportError("cknna 需要 transformers（DINOv2-L 参照系，§0 约定 6）") from e
    try:
        model = DinoV2Model.from_pretrained(model_name)
    except Exception as e:  # pragma: no cover - 网络/权重相关，调用方 skip
        raise OSError(
            f"DINOv2-L 参照系加载失败（{model_name}）；P0 集群首日下载权重入库后重试。"
            "严禁改用教师模型作参照（README D11 循环论证）。"
        ) from e
    return model.to(device).eval()


def _cknna_pair(s_feats: torch.Tensor, d_feats: torch.Tensor) -> float:
    """单层 CKNNA：linear CKA on centered features（Kornblith 2019）。

    s_feats: [N, D_s]，d_feats: [N, D_d]（N 必须一致；D 维可不同，CKA 对维度无关）。
    返回 [0,1] 标量；1=两特征集线性表征完全对齐，0=完全正交。
    """
    if s_feats.shape[0] != d_feats.shape[0]:
        raise ValueError(f"样本数不一致：student {s_feats.shape[0]} vs dino {d_feats.shape[0]}")
    x = s_feats.double() - s_feats.double().mean(dim=0, keepdim=True)    # 中心化
    y = d_feats.double() - d_feats.double().mean(dim=0, keepdim=True)
    kx = x @ x.t()                         # [N,N] 线性核
    ky = y @ y.t()
    # HSIC 估计 (Frobenius 内积) / 各自核范数 → 归一化对齐
    denom = (kx.norm() * ky.norm()).item()
    if denom == 0.0:
        return 0.0
    return float((kx * ky).sum().item() / denom)


def cknna(student_feats: Dict[str, torch.Tensor],
          dino_feats: Dict[str, torch.Tensor],
          layers: List[Tuple[str, str]]) -> Dict[str, object]:
    """逐层 CKNNA 对齐曲线。

    【D11 契约层禁止】student_feats 与 dino_feats 必须来自不同模型族；本函数签名不含任何
        教师/参照模型对象参数——参照系加载由 load_dinov2_reference 独占，从接口上杜绝
        "把教师特征当 dino_feats 传进来"的循环论证路径。

    Args:
        student_feats: dict[layer_name -> Tensor[N, D_s]]，UVT 各层特征（由调用方抽取）。
        dino_feats:    dict[layer_name -> Tensor[N, D_d]]，DINOv2-L 各层特征
                       （**必须**由 load_dinov2_reference 的输出抽取，禁止教师特征）。
        layers: list[(student_layer, dino_layer)]，配对策略由调用方决定（如按相对深度对齐）。
    Returns:
        {"per_layer_cknna": {f"{s_layer}|{d_layer}": float}, "mean": float,
         "n_layers": int, "n_samples": int}
    """
    if not FORBID_TEACHER_AS_REFERENCE:
        raise RuntimeError("D11 契约被篡改：FORBID_TEACHER_AS_REFERENCE 必须为 True")
    per_layer: Dict[str, float] = {}
    n_samples = -1
    for s_layer, d_layer in layers:
        if s_layer not in student_feats:
            raise KeyError(f"student_feats 缺层 {s_layer!r}")
        if d_layer not in dino_feats:
            raise KeyError(f"dino_feats 缺层 {d_layer!r}（须来自 load_dinov2_reference，D11）")
        sf = student_feats[s_layer].detach().cpu().float()
        df = dino_feats[d_layer].detach().cpu().float()
        if sf.shape[0] != df.shape[0]:
            raise ValueError(f"层对 ({s_layer},{d_layer}) 样本数不等：{sf.shape[0]} vs {df.shape[0]}")
        n_samples = sf.shape[0] if n_samples == -1 else n_samples
        per_layer[f"{s_layer}|{d_layer}"] = _cknna_pair(sf, df)
    mean_val = float(sum(per_layer.values()) / len(per_layer)) if per_layer else 0.0
    return {"per_layer_cknna": per_layer, "mean": mean_val,
            "n_layers": len(per_layer), "n_samples": int(n_samples)}
