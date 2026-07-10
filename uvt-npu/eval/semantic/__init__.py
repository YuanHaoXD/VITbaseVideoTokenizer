"""E-4 · 语义线评测包（03 篇 §5.2 三探针协议）。

三个探针各证明一个 claim，全部消费 UVT 的语义输出 (s, s_pool)（M-10 tokenizer.semantic）：

  zeroshot.py     —— zero-shot ImageNet（ADR-9 的直接回报：继承 SigLIP2 文本塔，零训练）
  linear_probe.py —— ImageNet/K400/SSv2 linear probe（冻结 tokenizer，只训一个线性头）
  cknna.py        —— 与 DINOv2-L 的逐层对齐曲线（第三方参照，禁教师自身参照，README D11）

【接口纪律】三个模块对 tokenizer 走鸭子类型（不 import M-10），约定其
  tokenizer.semantic(video∈[0,1]) -> (s[B,T1,N,D], s_pool[B,D])
与 T-1 教师走 SigLIP2Teacher.encode_text(prompts)->[K,D_t]（s_pool 已对齐文本塔空间, ADR-9）。
"""
from .zeroshot import PROMPT_TEMPLATE, zeroshot_classify
from .linear_probe import LinearProbeConfig, linear_probe
from .cknna import DINOV2_MODEL_NAME, cknna, load_dinov2_reference

__all__ = [
    "PROMPT_TEMPLATE",
    "zeroshot_classify",
    "LinearProbeConfig",
    "linear_probe",
    "DINOV2_MODEL_NAME",
    "cknna",
    "load_dinov2_reference",
]
