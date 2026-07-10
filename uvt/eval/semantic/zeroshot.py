"""E-4-a · zero-shot ImageNet 分类（03 篇 §5.2 探针①，ADR-9 的直接回报）。

claim：蒸馏把 UVT 的 s_pool 对齐到 SigLIP2 图文空间后，直接用教师文本塔算图文相似度
做分类——零额外训练。这是"语义蒸馏继承了文本塔"的硬证据（卖点）。

流程：
  ① class_names → prompt 模板 "a photo of a {}" → teacher.encode_text → 文本嵌入 [K,D_t]；
  ② 数据集每批 video → tokenizer.semantic → s_pool [B,D_t]（D_t 对齐文本塔, ADR-9）；
  ③ 分类 = argmax_k cos(s_pool, text_embed[k])；统计 top-1（top-5 可选）。

【s_pool 语义】s_pool 是 Sem-ViT 的 MAP 池化头输出（M-7），与 SigLIP2 的 pooler_output
同空间（共享 MAP 权重初始化 + L-2 img_pool 蒸馏对齐），故可直接与文本嵌入比 cos。
"""
from typing import Callable, Iterable, Sequence

import torch
import torch.nn.functional as F

# SigLIP 系通用 prompt 模板（SigLIP 论文/官方示例同款；CLIP 用 "a photo of a {}"）。
# P0-golden 校准时验证 SigLIP2 的最佳 prompt（可能 "a photo of {}." 略优）。
PROMPT_TEMPLATE = "a photo of a {}"


@torch.no_grad()
def zeroshot_classify(tokenizer, dataset: Iterable[dict],
                      class_names: Sequence[str], teacher, *,
                      device: torch.device = torch.device("cpu"),
                      prompt_template: str = PROMPT_TEMPLATE,
                      top5: bool = False) -> dict:
    """zero-shot 分类，报 ImageNet top-1（top-5 可选）。

    Args:
        tokenizer: 暴露 .semantic(video∈[0,1]) -> (s, s_pool) 的对象（M-10 鸭子类型）。
            s_pool: [B, D_t]，须与 teacher.encode_text 输出同维同空间（ADR-9 对齐）。
        dataset: 可迭代对象，每批产出 dict{"video": [B,3,F,H,W]∈[0,1]（图像 F=1）,
            "label": [B] long}。**像素值域 [0,1]**（数据集通用输出；tokenizer.semantic
            内部处理其归一化）。
        class_names: K 个类名（如 ImageNet 1000 类 WordNet name）。
        teacher: T-1 SigLIP2Teacher；用其 .encode_text(prompts)->[K,D_t]。
        prompt_template: 含一个 "{}" 占位符；默认 SigLIP 系 "a photo of a {}"。
        top5: True 时额外报 top-5（K≥5 才有意义）。
    Returns:
        {"top1": float, "n": int, "top5": float (top5=True 时)}。
    """
    if "{}" not in prompt_template:
        raise ValueError(f"prompt_template 必须含 '{{}}' 占位符，收到 {prompt_template!r}")
    prompts = [prompt_template.format(c) for c in class_names]
    text_embed = teacher.encode_text(prompts).to(device)        # [K, D_t]
    text_embed = F.normalize(text_embed, dim=-1)
    K = text_embed.shape[0]

    n_total = 0
    n_top1 = 0
    n_top5 = 0
    for batch in dataset:
        video = batch["video"].to(device)                       # [B,3,F,H,W] ∈ [0,1]
        labels = batch["label"].to(device)                      # [B]
        _, s_pool = tokenizer.semantic(video)                   # [B, D_t]
        if s_pool.shape[-1] != text_embed.shape[-1]:
            raise ValueError(
                f"s_pool 维度 {s_pool.shape[-1]} ≠ 文本嵌入维度 {text_embed.shape[-1]}；"
                "ADR-9 要求 s_pool 与教师文本塔同空间——检查 img_pool 蒸馏是否启用")
        sims = F.normalize(s_pool, dim=-1) @ text_embed.t()     # [B, K]  cos 相似度
        pred_top1 = sims.argmax(dim=-1)                         # [B]
        n_total += labels.numel()
        n_top1 += (pred_top1 == labels).sum().item()
        if top5 and K >= 5:
            pred_top5 = sims.topk(5, dim=-1).indices            # [B, 5]
            n_top5 += sum(labels[b].item() in pred_top5[b].tolist() for b in range(labels.numel()))

    out = {"top1": n_top1 / max(n_total, 1), "n": n_total}
    if top5:
        out["top5"] = n_top5 / max(n_total, 1)
    return out
