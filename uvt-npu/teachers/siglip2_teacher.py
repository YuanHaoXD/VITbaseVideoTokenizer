"""T-1 · SigLIP2 冻结图像教师（蒸馏三项里 img_patch / img_pool 两项的教师侧，ADR-9）。

职责（05 §3 T-1 / 01 §2.8）：
  - forward(images_01 ∈ [0,1]) -> (patch_feats [B,N,D_t], pooled [B,D_t])
      patch_feats = vision tower 的 last_hidden_state（对齐学生锚帧 s[:,0] 的 256 token）；
      pooled      = MAP 注意力池化后的整图嵌入——SigLIP2 的图文对齐**住在这里**而非 patch 特征里，
                    只蒸 patch 无法继承文本塔，zero-shot 评测就不成立（ADR-9 的全部动机）。
  - encode_text(prompts) -> [K,D_t]：文本塔，仅评测期 zero-shot 分类用（E-4 zeroshot.py 消费）。

【归一化硬约束】内部必须走 transformers AutoProcessor 的官方归一化——本类在加载期从
  AutoProcessor.image_processor 读取 image_mean/image_std 存成 buffer，在设备上等价施加；
  **绝不手写 (0.5,0.5,0.5) 之类常数**（test_siglip2_golden 用官方 pipeline fixture 抓偏差）。
  官方 pipeline = resize → rescale(1/255) → normalize(mean,std)；本类输入已是 [0,1] float，
  rescale 已隐含，resize 归 E-1 protocols / dataset 管（全仓唯一评测预处理），故此处只做 normalize
  并**断言**空间尺寸与 processor 期望一致（防止插值方式偷偷分叉）。

【tiny 模式】随机小 SiglipModel + SiglipImageProcessor 官方默认配置（离线可构建，单测用）；
  文本侧用确定性伪 tokenize（无 tokenizer 文件也能测形状/冻结语义）。
"""
from typing import List, Tuple

import torch
import torch.nn as nn


class SigLIP2Teacher(nn.Module):
    """冻结的 SigLIP2 教师。所有输出均无梯度（教师参数 requires_grad=False + forward no_grad）。"""

    def __init__(self, model_id: str = "google/siglip2-so400m-patch16-256", tiny: bool = False):
        super().__init__()
        self.model_id = model_id
        self.tiny = tiny
        # transformers 导入放构造期（模块导入不应强依赖，报错信息更聚焦）。
        try:
            import transformers  # noqa: F401, PLC0415
        except ImportError as e:  # pragma: no cover - 环境相关
            raise ImportError(
                "teachers.siglip2_teacher 需要 `transformers>=4.47`（§0 全局约定 6）。"
            ) from e

        if tiny:
            self._build_tiny()
        else:
            self._build_real()

        # 冻结教师：参数不可训 + 常驻 eval（train() 由 _frozen_train 覆盖，防 trainer 误切）。
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    # ---------------------------------------------------------------- 构建
    def _build_real(self):
        from transformers import AutoModel, AutoProcessor

        # AutoModel 自动解析 checkpoint 对应类（固定分辨率 siglip2 与 SiglipModel 架构兼容，
        # 与 M-4 用 SiglipVisionModel 加载同一 checkpoint 的选型一致）。
        self.model = AutoModel.from_pretrained(self.model_id)
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        ip = self.processor.image_processor
        self._register_norm(ip.image_mean, ip.image_std)
        size = ip.size
        # transformers 各版本 size 类型不一：int / {"height":..,"width":..} dict / SizeDict 对象
        # （5.13+ processor.size 为 SizeDict，非 int 也非普通 dict，int() 直接报错）。统一兼容。
        if isinstance(size, int):
            self.expected_hw = (size, size)
        elif hasattr(size, 'get'):          # dict / SizeDict 等 dict-like
            h = size.get('height') or size.get('shortest_edge') or 256
            w = size.get('width') or size.get('shortest_edge') or 256
            self.expected_hw = (int(h), int(w))
        else:
            self.expected_hw = (int(size), int(size))

    def _build_tiny(self):
        """随机 tiny 模型（离线单测）。归一化仍走 SiglipImageProcessor **官方默认配置**——
        即便 tiny 也不手写 mean/std，与硬约束保持同一代码路径语义。"""
        from transformers import SiglipConfig, SiglipImageProcessor, SiglipModel

        cfg = SiglipConfig(
            vision_config=dict(hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                               intermediate_size=128, image_size=64, patch_size=16),
            text_config=dict(hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                             intermediate_size=128, vocab_size=64, max_position_embeddings=16),
        )
        self.model = SiglipModel(cfg)
        self.processor = None  # tiny 无完整 processor（无 tokenizer 文件）
        ip = SiglipImageProcessor()  # 官方默认 mean/std，离线可实例化
        self._register_norm(ip.image_mean, ip.image_std)
        self.expected_hw = (64, 64)

    def _register_norm(self, mean, std):
        # [1,3,1,1] buffer：随 .to(device) 迁移；值来自官方 processor 配置（非手写）。
        self.register_buffer("_mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)

    # ---------------------------------------------------------------- 属性
    @property
    def dim(self) -> int:
        """教师特征维 D_t（So400M=1152），供 L-2 DistillLoss 建对齐头用。"""
        return self.model.config.vision_config.hidden_size

    def train(self, mode: bool = True):
        """教师永远 eval（覆盖 nn.Module.train，防 trainer 的 model.train() 波及）。"""
        return super().train(False)

    # ---------------------------------------------------------------- 前向
    @torch.no_grad()
    def forward(self, images_01: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            images_01: [B,3,H,W]，取值 [0,1]（rescale 已隐含，见模块 docstring）。
        Returns:
            (patch_feats [B,N,D_t], pooled [B,D_t])。
        """
        assert images_01.dim() == 4 and images_01.shape[1] == 3, \
            f"期望 [B,3,H,W]，收到 {tuple(images_01.shape)}"
        h, w = images_01.shape[-2:]
        assert (h, w) == self.expected_hw, (
            f"输入 {h}x{w} 与教师 processor 期望 {self.expected_hw} 不符——"
            "resize 必须由 eval/protocols.py（E-1，全仓唯一评测预处理）或 dataset 完成，"
            "教师侧不做插值以防插值方式分叉。"
        )
        pixel_values = (images_01.to(self._mean.dtype) - self._mean) / self._std  # 官方归一化
        out = self.model.vision_model(pixel_values=pixel_values)
        # last_hidden_state: [B,N,D]（无 CLS，SigLIP 视觉塔本就无 CLS token）
        # pooler_output:     [B,D]  MAP 注意力池化头输出 = 图文对齐嵌入（ADR-9）
        return out.last_hidden_state, out.pooler_output

    # ---------------------------------------------------------------- 文本塔
    @staticmethod
    def _text_features_as_tensor(out) -> torch.Tensor:
        """前向兼容：transformers ≤5.12 的 get_text_features 直接返回 [K,D_t] 张量；
        5.13+ 改为返回 BaseModelOutputWithPooling（pooler_output 为文本嵌入）。
        统一抽成 [K,D_t] 张量，使 zero-shot 评测在集群任意版本下行为一致。"""
        if torch.is_tensor(out):
            return out
        for k in ("text_embeds", "text_features", "pooler_output"):
            v = getattr(out, k, None)
            if torch.is_tensor(v) and v.dim() == 2:
                return v
        raise TypeError(f"encode_text: 无法从 {type(out).__name__} 抽取 [K,D_t] 文本嵌入")

    @torch.no_grad()
    def encode_text(self, prompts: List[str]) -> torch.Tensor:
        """zero-shot 评测用文本嵌入 [K,D_t]（E-4 消费；prompt 模板由评测脚本负责）。"""
        device = self._mean.device
        if self.tiny:
            # 确定性伪 tokenize：无 tokenizer 文件时仍可测形状/无梯度语义。
            vocab = self.model.config.text_config.vocab_size
            seq_len = self.model.config.text_config.max_position_embeddings
            ids = torch.stack([
                torch.tensor([(hash((p, i)) % vocab) for i in range(seq_len)], dtype=torch.long)
                for p in prompts
            ]).to(device)
            return self._text_features_as_tensor(self.model.get_text_features(input_ids=ids))
        # SigLIP 系惯例 padding="max_length", max_length=64（官方示例同款）。
        # ⚠ P0-golden 校准时验证：siglip2 tokenizer 的 padding/max_length 约定是否与 siglip1 一致。
        enc = self.processor(text=list(prompts), padding="max_length", max_length=64,
                             return_tensors="pt")
        return self._text_features_as_tensor(
            self.model.get_text_features(input_ids=enc["input_ids"].to(device)))
