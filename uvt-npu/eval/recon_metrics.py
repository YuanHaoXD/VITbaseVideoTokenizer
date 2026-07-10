"""重建线指标（E-2，03 篇 §3 / §5.1）。

指标与实现来源（全部对齐 03 篇规格，勿私改）：
  PSNR   : 逐帧计算后对帧取均值；峰值=协议值域宽度 data_range=2.0（[-1,1]）
  SSIM   : pytorch-msssim 的 ssim，data_range=2.0；逐帧算后对帧平均
  LPIPS  : vendored AlexNet 版（lpips 包 net="alex"；权重随包 vendored，首次构建缓存）
  rFID   : vendored pytorch-fid（eval/fid/，Inception-v3 pool3 2048 维）
  rFVD   : styleganv 版为主选（eval/fvd/styleganv/），videogpt 版并行输出仅作交叉核对，
           论文只报主选（03 篇 §3 的裁决规则：P0 校准中哪版能复现公开数字排序即定为主选）

输入约定：一律吃 protocols.py 的输出——
  图像 [B,3,256,256] ∈ [-1,1]；视频 [B,3,17,256,256] ∈ [-1,1]。
预处理必须来自 eval/protocols.py（全仓唯一实现），本文件不做任何 resize/crop。

17 帧 clip 对 I3D 16 帧窗口的处理（03 篇 §3，须写进论文评测细节）：
  I3D 特征提取前**取 clip 的前 16 帧**（video[:, :, :16]）。锚帧在位 0，
  前 16 帧 = 锚帧 + 前 15 个后续帧；真/假两侧同样截取，保持配对。
"""
import math

import numpy as np
import torch

from . import protocols

# I3D 特征窗口帧数：17 帧协议 clip 送 FVD 前截取的前缀长度（见模块 docstring）
FVD_FRAMES = 16

# 协议值域宽度：[-1,1] 的 data_range=2.0（03 篇 §3 明文）
DATA_RANGE = 2.0


def _flatten_video_to_frames(x: torch.Tensor) -> torch.Tensor:
    """[B,3,H,W] 或 [B,3,F,H,W] -> [B*F,3,H,W] 与帧数 F（图像视作 F=1）。"""
    if x.ndim == 4:
        return x, 1
    if x.ndim == 5:
        b, c, f, h, w = x.shape
        return x.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w), f
    raise ValueError(f"期望 [B,3,H,W] 或 [B,3,F,H,W]，收到 {tuple(x.shape)}")


# ---------------------------------------------------------------------------
# PSNR / SSIM / LPIPS（逐帧 → 对帧平均，返回 per-sample 值便于上层聚合与报方差）
# ---------------------------------------------------------------------------

def psnr(x: torch.Tensor, y: torch.Tensor, data_range: float = DATA_RANGE) -> torch.Tensor:
    """逐帧 PSNR 后对帧取均值。x/y: [B,3,H,W] 或 [B,3,F,H,W] ∈ [-1,1]。返回 [B]。

    完全重合的帧（mse=0）返回 +inf（test_metrics_sanity 的恒等断言依据）。
    """
    if x.shape != y.shape:
        raise ValueError(f"形状不一致: {tuple(x.shape)} vs {tuple(y.shape)}")
    xf, f = _flatten_video_to_frames(x)
    yf, _ = _flatten_video_to_frames(y)
    mse = (xf.float() - yf.float()).pow(2).mean(dim=(1, 2, 3))          # [B*F] 逐帧 MSE
    peak2 = data_range * data_range
    frame_psnr = torch.where(
        mse > 0,
        10.0 * torch.log10(peak2 / mse),
        torch.full_like(mse, float("inf")),
    )
    return frame_psnr.view(-1, f).mean(dim=1)                            # 对帧平均 -> [B]


def ssim(x: torch.Tensor, y: torch.Tensor, data_range: float = DATA_RANGE) -> torch.Tensor:
    """逐帧 SSIM（pytorch-msssim, data_range=2.0）后对帧取均值。返回 [B]。"""
    from pytorch_msssim import ssim as _msssim_ssim   # 延迟导入：无该依赖时其余指标仍可用
    if x.shape != y.shape:
        raise ValueError(f"形状不一致: {tuple(x.shape)} vs {tuple(y.shape)}")
    xf, f = _flatten_video_to_frames(x)
    yf, _ = _flatten_video_to_frames(y)
    # size_average=False -> 逐帧标量 [B*F]；帧折叠在 batch 维，等价"逐帧算后平均"
    frame_ssim = _msssim_ssim(xf.float(), yf.float(), data_range=data_range, size_average=False)
    return frame_ssim.view(-1, f).mean(dim=1)


class LPIPSMetric:
    """vendored AlexNet 版 LPIPS（lpips 包）。输入 [-1,1]（LPIPS 官方约定，与协议值域一致）。

    做成类以缓存网络（calculate_lpips.py 源码在模块级构建网络，全局状态——此处收进实例）。
    """

    def __init__(self, device: torch.device = torch.device("cpu")):
        import lpips as _lpips   # 延迟导入
        self.net = _lpips.LPIPS(net="alex", spatial=False).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """逐帧 LPIPS 后对帧取均值。x/y: [B,3,H,W] 或 [B,3,F,H,W] ∈ [-1,1]。返回 [B]。"""
        if x.shape != y.shape:
            raise ValueError(f"形状不一致: {tuple(x.shape)} vs {tuple(y.shape)}")
        xf, f = _flatten_video_to_frames(x)
        yf, _ = _flatten_video_to_frames(y)
        d = self.net(xf.to(self.device).float(), yf.to(self.device).float())  # [B*F,1,1,1]
        return d.view(-1, f).mean(dim=1).cpu()


# ---------------------------------------------------------------------------
# rFID（vendored pytorch-fid：Inception-v3 pool3 + 官方 Frechet 距离）
# ---------------------------------------------------------------------------

class RFIDEvaluator:
    """流式 rFID：add_batch 逐批喂真/重建图像，compute 出分。

    与 pytorch-fid 官方数值路径一致：激活按 float64 存满整表后 np.mean/np.cov
    （不用 running-sum 近似——校准尺子以官方实现为准）。ImageNet val 50k 的激活
    表约 50k×2048×8B ≈ 0.8 GB/侧，单机内存可承受。
    """

    def __init__(self, device: torch.device = torch.device("cpu")):
        from .fid.inception import InceptionV3
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]     # pool3（03 篇 §3 指定）
        self.model = InceptionV3([block_idx]).to(device).eval()
        self.device = device
        self._real_acts: list = []
        self._fake_acts: list = []

    @torch.no_grad()
    def _activations(self, images: torch.Tensor) -> np.ndarray:
        """协议值域 [-1,1] -> [0,1]（vendored InceptionV3 期望 [0,1]，内部自己乘回 [-1,1]）。"""
        x = protocols.from_eval_range(images).to(self.device).float()
        pred = self.model(x)[0]                              # [B,2048,1,1]
        return pred.squeeze(3).squeeze(2).cpu().numpy().astype(np.float64)

    def add_batch(self, real: torch.Tensor, fake: torch.Tensor) -> None:
        """real/fake: [B,3,256,256] ∈ [-1,1]（protocols.preprocess_image 的输出与其重建）。"""
        self._real_acts.append(self._activations(real))
        self._fake_acts.append(self._activations(fake))

    def compute(self) -> float:
        from .fid.fid_score import calculate_frechet_distance
        real = np.concatenate(self._real_acts, axis=0)
        fake = np.concatenate(self._fake_acts, axis=0)
        mu_r, sigma_r = np.mean(real, axis=0), np.cov(real, rowvar=False)
        mu_f, sigma_f = np.mean(fake, axis=0), np.cov(fake, rowvar=False)
        return float(calculate_frechet_distance(mu_f, sigma_f, mu_r, sigma_r))


# ---------------------------------------------------------------------------
# rFVD（styleganv 主选 + videogpt 交叉核对，两版并行输出）
# ---------------------------------------------------------------------------

class RFVDEvaluator:
    """流式 rFVD，method ∈ {"styleganv", "videogpt"}。

    输入协议 clip [B,3,17,256,256] ∈ [-1,1]：
      ① 取前 16 帧适配 I3D 窗口（见模块 docstring，论文评测细节须注明）；
      ② 转回 [0,1]（两版 vendored 实现的入参约定均为 [0,1] BCTHW/BTCHW）；
      ③ I3D 的 224 双线性 resize 属于 I3D 自带预处理，保留在 vendored 代码内，不算协议预处理。
    """

    def __init__(self, device: torch.device = torch.device("cpu"),
                 method: str = "styleganv", batch_size: int = 16):
        if method == "styleganv":
            from .fvd.styleganv import fvd as _impl
        elif method == "videogpt":
            from .fvd.videogpt import fvd as _impl
        else:
            raise ValueError(f"未知 FVD 实现: {method!r}（可选 'styleganv' / 'videogpt'）")
        self.method = method
        self._impl = _impl
        self.device = device
        self.batch_size = batch_size
        self._i3d = None                     # 惰性加载（权重文件首次调用时下载/读取）
        self._real_feats: list = []
        self._fake_feats: list = []

    def _detector(self):
        if self._i3d is None:
            self._i3d = self._impl.load_i3d_pretrained(device=self.device)
        return self._i3d

    def _features(self, clips: torch.Tensor) -> np.ndarray:
        if clips.ndim != 5 or clips.shape[1] != 3:
            raise ValueError(f"期望 [B,3,F,256,256]，收到 {tuple(clips.shape)}")
        clips = clips[:, :, :FVD_FRAMES]                       # 17 帧 -> 前 16 帧（I3D 窗口）
        clips01 = protocols.from_eval_range(clips).cpu().float()
        if self.method == "styleganv":
            # styleganv get_fvd_feats: [0,1] BCTHW -> np [N,400]
            feats = self._impl.get_fvd_feats(clips01, i3d=self._detector(),
                                             device=self.device, bs=self.batch_size)
            return np.asarray(feats, dtype=np.float64)
        # videogpt get_fvd_logits: [0,1] BCTHW -> torch [N,400]（其内部自转 uint8 与 [-1,1]）
        logits = self._impl.get_fvd_logits(clips01, i3d=self._detector(),
                                           device=self.device, bs=self.batch_size)
        return logits.cpu().numpy().astype(np.float64)

    def add_batch(self, real: torch.Tensor, fake: torch.Tensor) -> None:
        """real/fake: [B,3,17,256,256] ∈ [-1,1]（protocols.preprocess_video 的输出与其重建）。"""
        self._real_feats.append(self._features(real))
        self._fake_feats.append(self._features(fake))

    def compute(self) -> float:
        real = np.concatenate(self._real_feats, axis=0)
        fake = np.concatenate(self._fake_feats, axis=0)
        if self.method == "styleganv":
            return float(self._impl.frechet_distance(fake, real))
        return float(self._impl.frechet_distance(torch.from_numpy(fake), torch.from_numpy(real)))


def compute_rfvd_both(real_clips: torch.Tensor, fake_clips: torch.Tensor,
                      device: torch.device = torch.device("cpu"),
                      batch_size: int = 16) -> dict:
    """一次性小规模评测入口：两版 rFVD 并行输出（主选 styleganv，videogpt 仅交叉核对）。"""
    out = {}
    for method in ("styleganv", "videogpt"):
        ev = RFVDEvaluator(device=device, method=method, batch_size=batch_size)
        ev.add_batch(real_clips, fake_clips)
        out[f"rfvd_{method}"] = ev.compute()
    return out


# ---------------------------------------------------------------------------
# 汇总入口：一段数据（图像或视频）的全套重建指标
# ---------------------------------------------------------------------------

class ReconMetricsSuite:
    """Gate 快照/正式对比表共用的流式聚合器。

    用法：
        suite = ReconMetricsSuite(device, is_video=True)
        for real, fake in pairs:           # 均为 protocols 输出值域
            suite.add_batch(real, fake)
        report = suite.compute()           # dict：psnr/ssim/lpips (+ rfid 或 rfvd 两版)
    """

    def __init__(self, device: torch.device = torch.device("cpu"), is_video: bool = False,
                 with_lpips: bool = True, with_frechet: bool = True):
        self.is_video = is_video
        self._psnr_vals: list = []
        self._ssim_vals: list = []
        self._lpips_vals: list = []
        self._lpips = LPIPSMetric(device) if with_lpips else None
        if with_frechet:
            if is_video:
                self._frechet = {m: RFVDEvaluator(device, method=m)
                                 for m in ("styleganv", "videogpt")}
            else:
                self._frechet = {"fid": RFIDEvaluator(device)}
        else:
            self._frechet = {}

    def add_batch(self, real: torch.Tensor, fake: torch.Tensor) -> None:
        self._psnr_vals.append(psnr(real, fake))
        self._ssim_vals.append(ssim(real, fake))
        if self._lpips is not None:
            self._lpips_vals.append(self._lpips(real, fake))
        for ev in self._frechet.values():
            ev.add_batch(real, fake)

    def compute(self) -> dict:
        report = {
            "psnr": float(torch.cat(self._psnr_vals).mean()),
            "ssim": float(torch.cat(self._ssim_vals).mean()),
        }
        if self._lpips_vals:
            report["lpips"] = float(torch.cat(self._lpips_vals).mean())
        for name, ev in self._frechet.items():
            key = "rfid" if name == "fid" else f"rfvd_{name}"
            report[key] = ev.compute()
        return report
