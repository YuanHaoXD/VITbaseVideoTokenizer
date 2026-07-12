"""SigLIP2 骨干加载与切分（M-4）。

一次加载，切成模型各部件的初始化来源（01 §2.1 "一模三用"）：
  - embeddings：patchify + 学习式空间 PE（ADR-8 默认保留）；
  - gen_blocks：第 1..gen_depth 层  → Gen-ViT；
  - sem_blocks：其余层             → Sem-ViT；
  - dec_blocks：**全部 27 层的另一份独立深拷贝** → decoder 初始化（01 §2.7）；
  - post_ln / map_head：末端 LN 与 MAP 注意力池化头（ADR-9 zero-shot 的关键）。

实现级偏离（任务书 §2 M-4）：用固定分辨率 256 权重而非 naflex——网格与训练分辨率 1:1，
naflex 的变分辨率打包接口徒增复杂度，P4 混合分辨率阶段再评估。

`tiny=True`：随机初始化小模型，所有单测离线可跑（禁下载权重，见 §0）。
transformers 的导入放在函数内部（避免顶层 import 失败波及其他模块，§0 硬约束）。
"""
from dataclasses import dataclass
from typing import List

import torch.nn as nn

from .blocks import UVTBlock


@dataclass
class BackboneParts:
    embeddings: nn.Module          # SiglipVisionEmbeddings 深拷贝（patchify + 空间 PE）
    gen_blocks: List[UVTBlock]     # 第 1..gen_depth 层
    sem_blocks: List[UVTBlock]     # 其余层
    dec_blocks: List[UVTBlock]     # 全部 27 层的另一份独立深拷贝（decoder 初始化）
    post_ln: nn.Module             # 末端 LayerNorm
    map_head: nn.Module            # MAP 注意力池化头（ADR-9）
    # 学生输入归一化统计（第 15 号修复，docs/06 §6.8）：来自官方 processor 配置，
    # 供 GenViT 在 patchify 前施加（教师 T-1 同源，绝不手写常数——tiny 除外，见 load）。
    image_mean: tuple = (0.5, 0.5, 0.5)
    image_std: tuple = (0.5, 0.5, 0.5)


def load_siglip_parts(model_name: str = "google/siglip2-so400m-patch16-256",
                      gen_depth: int = 13, tiny: bool = False,
                      rope_dims: int = 32) -> BackboneParts:
    """加载并切分 SigLIP2 视觉塔为 BackboneParts。

    gen_depth：Gen-ViT 层数（ADR：HYDRA 均衡切分，27 层取 13+14）。
    tiny：离线随机初始化小模型（测试用）；tiny 下 gen_depth 允许夹断到实际层数，
          非 tiny 下 gen_depth 必须严格小于总层数（静默夹断会掩盖配置错误）。
    rope_dims：架构方契约修订 2026-07-07（任务书 M-4 卡 v1.1）——下传给 UVTBlock.from_siglip，
          使 ADR-8 的 rope 消融臂可经本入口构造。
    """
    import copy

    from transformers import SiglipVisionConfig, SiglipVisionModel

    # 前向兼容：transformers ≤5.12 的 SiglipVisionModel 包一层 .vision_model（内层
    # SiglipVisionTransformer）；5.13+ 移除了该包装，embeddings/encoder/post_layernorm/head
    # 直接挂在 SiglipVisionModel 上。两者择一，使本模块在集群可能装的任意版本下都能加载
    # （避免「代码对 4.47 正确、对最新版崩」的隐性风险）。
    def _tower(model):
        return model.vision_model if hasattr(model, "vision_model") else model

    if tiny:
        # 离线可跑的小模型：参数与任务书 §2 M-4 冻结一致，随机初始化。
        config = SiglipVisionConfig(
            hidden_size=64,
            num_hidden_layers=6,
            num_attention_heads=2,
            intermediate_size=128,
            image_size=64,
            patch_size=16,
        )
        vision = _tower(SiglipVisionModel(config))
        # tiny 无 processor 文件（离线随机权重，归一化统计无意义），用 SigLIP 惯例默认
        # (0.5,0.5,0.5)——这是 dataclass 默认值，非"手写生产常数"（生产路径严格读 processor）。
        image_mean, image_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    else:
        # 固定分辨率 256 权重（非 naflex，见模块 docstring）。
        vision = _tower(SiglipVisionModel.from_pretrained(model_name))
        n_layers = len(vision.encoder.layers)
        assert n_layers == 27, f"SigLIP2-So400m 应为 27 层，实得 {n_layers}"
        assert vision.config.hidden_size == 1152, f"应为 1152 宽，实得 {vision.config.hidden_size}"
        # 学生输入归一化统计（第 15 号修复）：与教师 T-1 同源读官方 processor 配置，
        # 绝不手写常数（T-1 卡纪律同款）。生产路径严格要求可加载（缺文件即炸，不静默回退）。
        from transformers import AutoImageProcessor
        ip = AutoImageProcessor.from_pretrained(model_name)
        image_mean, image_std = tuple(ip.image_mean), tuple(ip.image_std)

    layers = vision.encoder.layers
    n = len(layers)

    # 架构方裁决（契约问题#2）：tiny 允许夹断（测试便利）；非 tiny 严格校验，
    # 静默夹断会掩盖 gen_depth 配置错误（Sem-ViT 变空却不报错）。
    if not tiny:
        assert 0 < gen_depth < n, f"gen_depth={gen_depth} 必须在 (0, {n}) 内"
    split = min(gen_depth, n)

    # 三份 UVTBlock：gen / sem 切分共用同一切点；dec 是**全部层的另一份独立深拷贝**。
    # from_siglip 内部对每层各做 deepcopy，故 gen_blocks[i] 与 dec_blocks[i] 权重相等但存储独立。
    gen_blocks = [UVTBlock.from_siglip(layers[i], rope_dims) for i in range(split)]
    sem_blocks = [UVTBlock.from_siglip(layers[i], rope_dims) for i in range(split, n)]
    dec_blocks = [UVTBlock.from_siglip(layers[i], rope_dims) for i in range(n)]

    embeddings = copy.deepcopy(vision.embeddings)
    post_ln = copy.deepcopy(vision.post_layernorm)
    # vision_use_head 默认 True，故 head 存在；None 兜底以防未来关闭池化头的配置。
    map_head = copy.deepcopy(vision.head) if getattr(vision, "head", None) is not None else None

    return BackboneParts(
        embeddings=embeddings,
        gen_blocks=gen_blocks,
        sem_blocks=sem_blocks,
        dec_blocks=dec_blocks,
        post_ln=post_ln,
        map_head=map_head,
        image_mean=image_mean,
        image_std=image_std,
    )
