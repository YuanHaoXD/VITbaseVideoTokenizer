"""评测协议唯一实现（E-1，03 篇 §3）。

全仓库所有重建/语义评测的预处理**只允许**经过本文件（Hydra-X 表 9 的 identical-scripts
纪律：预处理不统一是重建指标不可比的头号原因）。其他模块只能 import 本文件，
禁止复制其中任何逻辑。

规格（03 篇 §3，逐条冻结）：
  图像:  短边 resize 到 256（bicubic, antialias=True）→ center crop 256 → [0,1] → [-1,1]
  视频:  DAVIS 取前 17 帧（起点=第一可用帧）；UCF 随机固定起点（seed=0，全表固定）；
         逐帧做与图像相同的空间处理；输出 [3, 17, 256, 256]
  张量:  单视频 [3,F,H,W]、单图像 [3,H,W]（§0 全局约定的像素轴序，不带 batch 维）

纯函数纪律：本文件所有函数确定性、无全局状态、无 RNG 全局副作用。
UCF 的"随机固定起点"用哈希派生（sha256(seed:video_id)），与数据遍历顺序无关——
同一视频在任何机器/任何 run 得到同一起点（比"全局 RNG 按顺序抽"更稳，见实现注释）。
"""
import hashlib

import torch
import torch.nn.functional as F

# ---- 协议常量（冻结，改动=评测协议变更，必须走任务书修订） ----
EVAL_SIZE = 256      # 评测分辨率（短边 resize 与 center crop 的目标边长）
EVAL_FRAMES = 17     # 评测 clip 帧数（1 锚帧 + 16，见 §0 张量约定）
UCF_CLIP_SEED = 0    # UCF 起点采样种子（03 篇 §3：seed=0 全表固定）


def _as_float01(x: torch.Tensor) -> torch.Tensor:
    """统一到 float32 [0,1]。uint8 输入（decord/PIL 常见）自动 /255；float 输入校验范围。"""
    if x.dtype == torch.uint8:
        return x.float() / 255.0
    if not torch.is_floating_point(x):
        raise TypeError(f"输入 dtype 必须是 uint8 或浮点，收到 {x.dtype}")
    # 宽松校验：抓"喂了 [-1,1] 或 [0,255] float"这类协议违规（评测尺子的常见事故源）
    if float(x.min()) < -1e-3 or float(x.max()) > 1.0 + 1e-3:
        raise ValueError(
            f"float 输入必须在 [0,1]（收到 min={float(x.min()):.4f}, max={float(x.max()):.4f}）；"
            "若已是 [-1,1] 请勿重复归一化，若是 [0,255] 请先转 uint8 或 /255"
        )
    return x.float()


def resize_short_side(frames: torch.Tensor, size: int = EVAL_SIZE) -> torch.Tensor:
    """短边 resize（bicubic, antialias=True）。frames: [B,3,H,W]。

    长边尺寸取 int 截断（与 torchvision `Resize(size)` 的
    `_compute_resized_output_size` 约定一致：new_long = int(size * long / short)），
    在此写死以防上游库换版本时协议漂移。
    """
    _, _, h, w = frames.shape
    if h <= w:
        new_h, new_w = size, int(size * w / h)
    else:
        new_h, new_w = int(size * h / w), size
    return F.interpolate(
        frames, size=(new_h, new_w), mode="bicubic", antialias=True, align_corners=False
    )


def center_crop(frames: torch.Tensor, size: int = EVAL_SIZE) -> torch.Tensor:
    """中心裁剪。frames: [B,3,H,W]，H/W 均须 >= size。"""
    _, _, h, w = frames.shape
    if h < size or w < size:
        raise ValueError(f"center_crop 前尺寸 ({h},{w}) 小于目标 {size}——请先 resize_short_side")
    top = (h - size) // 2
    left = (w - size) // 2
    return frames[:, :, top:top + size, left:left + size]


def to_eval_range(x01: torch.Tensor) -> torch.Tensor:
    """[0,1] -> [-1,1]（协议输出值域；重建指标 data_range=2.0 与此对应）。"""
    return x01 * 2.0 - 1.0


def from_eval_range(x: torch.Tensor) -> torch.Tensor:
    """[-1,1] -> [0,1]（喂 Inception/I3D 等期望 [0,1] 输入的下游网络时用，并夹紧数值溢出）。"""
    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)


def _spatial_pipeline(frames: torch.Tensor) -> torch.Tensor:
    """图像/视频帧共用的空间处理：[B,3,H,W](uint8 或 float[0,1]) -> [B,3,256,256] ∈ [-1,1]。"""
    frames = _as_float01(frames)
    frames = resize_short_side(frames, EVAL_SIZE)
    frames = center_crop(frames, EVAL_SIZE)
    return to_eval_range(frames)


def preprocess_image(img: torch.Tensor) -> torch.Tensor:
    """图像协议入口：[3,H,W] -> [3,256,256] ∈ [-1,1]。"""
    if img.ndim != 3 or img.shape[0] != 3:
        raise ValueError(f"preprocess_image 期望 [3,H,W]，收到 {tuple(img.shape)}")
    return _spatial_pipeline(img.unsqueeze(0)).squeeze(0)


# ---- clip 起点规则（03 篇 §3） ----

def clip_start_davis(num_frames: int, clip_len: int = EVAL_FRAMES) -> int:
    """DAVIS：起点=第一可用帧（即 0）。"""
    if num_frames < clip_len:
        raise ValueError(f"视频仅 {num_frames} 帧，不足评测 clip 长度 {clip_len}（应在预筛阶段剔除）")
    return 0


def clip_start_ucf(num_frames: int, video_id: str, clip_len: int = EVAL_FRAMES,
                   seed: int = UCF_CLIP_SEED) -> int:
    """UCF：随机固定起点，seed=0 全表固定。

    实现裁决（写入交付报告）：起点 = sha256(f"{seed}:{video_id}") % 可用起点数。
    不用"全局 RNG 按遍历顺序抽"——那样起点依赖 DataLoader 顺序/worker 数，
    跨机器跨 run 不可复现；哈希派生对 (seed, video_id) 纯确定，与顺序无关。
    video_id 用数据集内稳定字符串（建议：datalist 中的相对路径）。
    """
    if num_frames < clip_len:
        raise ValueError(f"视频仅 {num_frames} 帧，不足评测 clip 长度 {clip_len}（应在预筛阶段剔除）")
    n_starts = num_frames - clip_len + 1
    digest = hashlib.sha256(f"{seed}:{video_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") % n_starts


def preprocess_video(video: torch.Tensor, dataset: str, video_id: str = None,
                     seed: int = UCF_CLIP_SEED) -> torch.Tensor:
    """视频协议入口：[3,F,H,W] -> [3,17,256,256] ∈ [-1,1]。

    dataset: "davis"（起点=0）或 "ucf"（哈希固定起点，必须传 video_id）。
    先按起点取 17 连续帧（不跳帧），再逐帧做与图像相同的空间处理。
    """
    if video.ndim != 4 or video.shape[0] != 3:
        raise ValueError(f"preprocess_video 期望 [3,F,H,W]，收到 {tuple(video.shape)}")
    num_frames = video.shape[1]

    dataset = dataset.lower()
    if dataset == "davis":
        start = clip_start_davis(num_frames)
    elif dataset == "ucf":
        if video_id is None:
            raise ValueError("UCF 协议必须传 video_id（datalist 相对路径）以派生固定起点")
        start = clip_start_ucf(num_frames, video_id, seed=seed)
    else:
        raise ValueError(f"未知评测数据集协议: {dataset!r}（可选 'davis' / 'ucf'）")

    clip = video[:, start:start + EVAL_FRAMES]          # [3,17,H,W]
    frames = clip.permute(1, 0, 2, 3)                   # [17,3,H,W]：帧折进 batch 走空间管线
    frames = _spatial_pipeline(frames)                  # [17,3,256,256]
    return frames.permute(1, 0, 2, 3).contiguous()      # [3,17,256,256]
