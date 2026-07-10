"""T-2 · InternVideo 冻结视频教师（蒸馏 vid 项的教师侧，ADR-7）。

职责（05 §3 T-2 / 01 §2.8）：
  forward(video_01 [B,3,16,H,W]) -> feats [B,16,N_t,D_t]
  - 丢弃 CLS/聚合 token（ADR-7：只留时空 patch token，聚合 token 与学生 token 网格无对应关系）；
  - 空间网格与学生不一致时**教师侧只管原样输出**，双线性对齐在 L-2 losses/distill.py 做（ADR-7）；
  - config 字段 teacher_id 支持 InternVideo-Next-L 与备胎 OpenGVLab/InternVideo2-Stage2_1B-224p-f4
   （01 §2.8：换教师只影响绝对值不影响消融趋势）。

【适配器模式（本卡的 API 不确定性对策）】InternVideo 系没有稳定的 transformers 官方集成，加载/预处理/
  特征抽取三步的真实 API 均需对着官方 repo 校准。因此拆成三个可替换方法：
      _load_model()   —— 怎么加载权重
      _preprocess()   —— 输入协议（分辨率/归一化/帧组织）
      _extract()      —— 前向 + token 布局解析 → [B,16,N_t,D_t]
  每个不确定点用「⚠ P0-golden 校准时验证」注释标出；golden fixture 首日生成入库后按 fixture 修正。
  单测一律用 MockTeacher（文件末尾），不碰真权重。

【不确定点清单（P0-golden 必须逐条核销）】见各方法内 ⚠ 注释与模块末尾 UNCERTAINTIES 汇总。
"""
from dataclasses import dataclass, field
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class InternVideoTeacherConfig:
    """视频教师配置。字段默认值多为「文献值/合理猜测」，⚠ 标注者以 P0-golden 校准为准。"""
    # ⚠ P0-golden 校准时验证：InternVideo-Next-L 的确切 HF repo id（论文刚发布，命名未联网核实）。
    teacher_id: str = "OpenGVLab/InternVideo-Next-L"
    # 备胎（01 §2.8 ADR-7）：InternVideo2 Stage2 1B，224p，单 clip 4 帧。
    fallback_id: str = "OpenGVLab/InternVideo2-Stage2_1B-224p-f4"
    # ⚠ P0-golden 校准时验证：教师特征维（InternVideo2-1B 为 ViT-g/14 系，猜 1408；Next-L 未知）。
    teacher_dim: int = 1408
    # ⚠ P0-golden 校准时验证：教师输入分辨率（Stage2 checkpoint 名带 224p）。
    input_size: int = 224
    # ⚠ P0-golden 校准时验证：单次前向的帧数。"-f4" 后缀提示 4 帧/clip；16 帧输入将按 4 帧
    #   分 4 个 clip 依次前向再在时间维拼接（_extract 内实现，是否合法需对官方 demo 核实）。
    clip_frames: int = 4
    # ⚠ P0-golden 校准时验证：序列前缀 token 数（CLS=1？是否还有额外聚合 token？）。
    num_prefix_tokens: int = 1
    # ⚠ P0-golden 校准时验证：归一化常数。InternVideo2 官方 demo 用 ImageNet mean/std（待核）；
    #   注意 T-2 无 AutoProcessor 可依（区别于 T-1 的硬约束），故此处允许配置化常数，
    #   但最终值必须与官方预处理逐位对齐并锁进 golden fixture（ADR-7）。
    image_mean: tuple = (0.485, 0.456, 0.406)
    image_std: tuple = (0.229, 0.224, 0.225)
    # 我方协议帧数（1 锚帧之外的 16 帧，Decompressor 输出与此对齐）。
    frames: int = 16
    # trust_remote_code：InternVideo 系模型代码不在 transformers 主库内。
    trust_remote_code: bool = True
    extra: dict = field(default_factory=dict)


class InternVideoTeacher(nn.Module):
    """冻结的 InternVideo 视频教师（适配器骨架）。

    真实权重路径未经离线验证——所有 ⚠ 点在 P0-golden 首日对官方 repo/demo 校准后修正；
    单测请用 MockTeacher。
    """

    def __init__(self, cfg: InternVideoTeacherConfig = None):
        super().__init__()
        self.cfg = cfg or InternVideoTeacherConfig()
        self.model = self._load_model()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()
        mean = torch.tensor(self.cfg.image_mean).view(1, 3, 1, 1, 1)
        std = torch.tensor(self.cfg.image_std).view(1, 3, 1, 1, 1)
        self.register_buffer("_mean", mean, persistent=False)
        self.register_buffer("_std", std, persistent=False)

    # ---------------------------------------------------------------- 适配点 1：加载
    def _load_model(self) -> nn.Module:
        """加载教师权重。子类/后续校准可整体替换本方法。"""
        try:
            from transformers import AutoModel  # noqa: PLC0415
        except ImportError as e:  # pragma: no cover
            raise ImportError("teachers.internvideo_teacher 需要 `transformers`。") from e
        # ⚠ P0-golden 校准时验证：两个候选 id 是否都能走 AutoModel+trust_remote_code 路径。
        #   InternVideo2-Stage2 的 HF 卡片历史上要求 trust_remote_code=True 并可能暴露
        #   非标准入口（如 model.get_vid_feat / model.encode_vision）；InternVideo-Next 未知，
        #   若其只发 github 权重则本方法需改为「本地 checkpoint + 官方建模代码」的加载方式。
        return AutoModel.from_pretrained(
            self.cfg.teacher_id, trust_remote_code=self.cfg.trust_remote_code
        )

    # ---------------------------------------------------------------- 适配点 2：预处理
    def _preprocess(self, video_01: torch.Tensor) -> torch.Tensor:
        """[B,3,16,H,W] ∈[0,1] → 教师输入协议（分辨率 + 归一化）。

        教师输入分辨率是教师自身协议的一部分（此处 resize 合法）；与**学生 token 网格**的
        空间对齐则是 L-2 的职责（ADR-7），两者不要混淆。
        """
        b, c, t, h, w = video_01.shape
        assert c == 3 and t == self.cfg.frames, \
            f"期望 [B,3,{self.cfg.frames},H,W]，收到 {tuple(video_01.shape)}"
        s = self.cfg.input_size
        if (h, w) != (s, s):
            # ⚠ P0-golden 校准时验证：官方预处理的 resize 方式（bilinear? bicubic? 是否 center-crop）。
            frames = video_01.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            frames = F.interpolate(frames, size=(s, s), mode="bilinear", align_corners=False)
            video_01 = frames.reshape(b, t, c, s, s).permute(0, 2, 1, 3, 4)
        return (video_01.to(self._mean.dtype) - self._mean) / self._std

    # ---------------------------------------------------------------- 适配点 3：抽特征
    def _extract(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """前向教师并把输出整理成 [B,16,N_t,D_t]（丢 CLS/聚合 token，ADR-7）。"""
        b, c, t, h, w = pixel_values.shape
        f = self.cfg.clip_frames
        assert t % f == 0, f"帧数 {t} 须被 clip_frames {f} 整除"
        chunks = []
        for i in range(t // f):  # 16 帧按 4 帧/clip 分段前向（⚠ 见 cfg.clip_frames 注释）
            clip = pixel_values[:, :, i * f:(i + 1) * f]
            # ⚠ P0-golden 校准时验证：真实前向入口。候选（按官方 demo 优先级）：
            #   ① model.encode_vision(clip) / model.get_vid_feat(clip)（InternVideo2 remote code 惯例）
            #   ② model(pixel_values=clip).last_hidden_state（标准 transformers 语义）
            out = self.model(clip) if not hasattr(self.model, "encode_vision") \
                else self.model.encode_vision(clip)
            hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
            if isinstance(hidden, (tuple, list)):  # remote code 可能返回 (feat, aux...) 元组
                hidden = hidden[0]
            chunks.append(self._layout(hidden, frames_in_clip=f))
        feats = torch.cat(chunks, dim=1)  # [B, 16, N_t, D_t]
        assert feats.shape[1] == self.cfg.frames, \
            f"时间维应为 {self.cfg.frames}，实得 {feats.shape[1]}（检查 _layout 的时间上采样）"
        return feats

    def _layout(self, hidden: torch.Tensor, frames_in_clip: int) -> torch.Tensor:
        """[B,S,D] → [B,frames_in_clip,N_t,D]：剥前缀 token + 解析时空布局。"""
        assert hidden.dim() == 3, f"期望 [B,S,D]，实得 {hidden.dim()}D"
        # 丢弃 CLS/聚合 token（ADR-7）——只留与空间网格对应的 patch token。
        tokens = hidden[:, self.cfg.num_prefix_tokens:]
        b, s, d = tokens.shape
        # ⚠ P0-golden 校准时验证：token 排布假设为 [t0 的 N_t 个, t1 的 N_t 个, ...]（时间外层、
        #   空间内层，ViT 视频模型主流约定）；若教师做了时间下采样（tubelet>1），
        #   下面 repeat_interleave 把 t' 复制回逐帧对齐——是否合理同样待 golden 校准。
        if s % frames_in_clip == 0:
            t_out = frames_in_clip
        else:
            t_out = 1  # 兜底：整段 clip 只输出一组空间 token（时间被完全池化）
            assert s % t_out == 0
        n_t = s // t_out
        feats = tokens.reshape(b, t_out, n_t, d)
        if t_out < frames_in_clip:
            feats = feats.repeat_interleave(frames_in_clip // t_out, dim=1)
        return feats

    # ---------------------------------------------------------------- 对外接口（冻结）
    @property
    def dim(self) -> int:
        """教师特征维 D_t，供 L-2 DistillLoss 建对齐头。⚠ 真实值以加载后模型 config 为准。"""
        return self.cfg.teacher_dim

    def train(self, mode: bool = True):
        """教师永远 eval（覆盖 train，防 trainer 的 model.train() 波及）。"""
        return super().train(False)

    @torch.no_grad()
    def forward(self, video_01: torch.Tensor) -> torch.Tensor:
        """[B,3,16,H,W] ∈[0,1] → feats [B,16,N_t,D_t]（无梯度）。"""
        return self._extract(self._preprocess(video_01))


class MockTeacher(nn.Module):
    """离线测试替身：接口与 InternVideoTeacher.forward 完全一致，输出确定性、输入相关、无梯度。

    实现 = 空间平均池化到 g×g 网格 + 固定种子的冻结 Linear(3→dim)：
      - 确定性：同输入同输出（golden 骨架与 L-2 单测都依赖）；
      - 输入相关：能测「对齐头真的在学东西」而不是对常数回归；
      - 零下载、零网络。
    """

    def __init__(self, dim: int = 64, spatial_tokens: int = 16, frames: int = 16, seed: int = 0):
        super().__init__()
        g = int(round(spatial_tokens ** 0.5))
        assert g * g == spatial_tokens, f"spatial_tokens={spatial_tokens} 须为完全平方数"
        self.grid = g
        self.frames = frames
        self._dim = dim
        gen = torch.Generator().manual_seed(seed)
        proj = nn.Linear(3, dim)
        with torch.no_grad():
            proj.weight.copy_(torch.randn(dim, 3, generator=gen))
            proj.bias.zero_()
        for p in proj.parameters():
            p.requires_grad_(False)
        self.proj = proj

    @property
    def dim(self) -> int:
        return self._dim

    def train(self, mode: bool = True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, video_01: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = video_01.shape
        assert c == 3 and t == self.frames, \
            f"期望 [B,3,{self.frames},H,W]，收到 {tuple(video_01.shape)}"
        frames = video_01.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        pooled = F.adaptive_avg_pool2d(frames, (self.grid, self.grid))       # [B*T,3,g,g]
        tokens = pooled.flatten(2).transpose(1, 2)                            # [B*T,N_t,3]
        feats = self.proj(tokens).reshape(b, t, self.grid * self.grid, -1)    # [B,T,N_t,D]
        return feats


# ======================================================================================
# 不确定点汇总（P0-golden 首日逐条核销；核销后删除对应 ⚠ 注释并锁 golden fixture）
# ======================================================================================
UNCERTAINTIES = [
    "U1 teacher_id: InternVideo-Next-L 的确切 HF repo id / 是否只发 github 权重",
    "U2 加载路径: AutoModel+trust_remote_code 是否适用于两个候选 checkpoint",
    "U3 前向入口: model() vs encode_vision() vs get_vid_feat()，返回结构（对象/元组/张量）",
    "U4 输入协议: 分辨率 224?、resize 插值方式、归一化 mean/std、帧是否要求特定 dtype/排布",
    "U5 clip 协议: f4 变体是否必须 4 帧/前向；16 帧拆 4 clip 拼接是否与官方特征等价",
    "U6 token 布局: 前缀 token 数、时间外层/空间内层假设、tubelet 时间下采样倍率",
    "U7 teacher_dim: 1408(ViT-g 猜测) 是否正确；Next-L 的实际宽度",
]
