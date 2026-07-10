"""E-3 · 五锚点校准 runner（03 篇 §4 五锚点校准流程 + 02 篇 §4 锚点表）。

职责（唯一）：在五家公开 tokenizer checkpoint 上跑**我们的 protocols.py + recon_metrics.py**，
得到"我们尺子下"的重建数字，与各锚点公开参照值对照，产出《校准报告》markdown。
排序一致性（rank-order 复现）是主判据（README D9：相对排序复现），绝对值偏差记录在案
（预期 PSNR ±0.1dB 内、rFID/rFVD ±10~20% 属正常实现差异，03 篇 §4）。

流程（03 篇 §4，每个锚点重复）：
  ① 官方权重 + 官方脚本 → 复现其 README/论文数字（验证环境与权重完好）；
  ② 官方权重 + 我们的 protocols.py → "我们尺子下"的数字；
  ③ 两套数字写入《校准报告》：五家在我们尺子下的**排序**须与公开数字排序一致。

本 runner 的核心是 ②（跑我们的尺子在各锚点 checkpoint 上）。① 各锚点官方脚本各异
（OmniTokenizer 的 vqgan_eval.py、LARP 的 eval_larp_tokenizer.py、LeanVAE 的
eval_recon_videos.py、AToken 的 atoken_inference、Wan2.2 的 vid_recon.py），无法统一调用，
故做成**可选 hook**：official_repro_hook(anchor)->dict 由 P0 首日人工补，留接口不实现。

【尺子纪律】所有真值/重建一律经 eval.protocols 预处理（全仓唯一评测预处理，E-1）；
重建指标一律经 eval.recon_metrics（E-2）。本文件不做任何 resize/crop/插值，禁止复制
protocols/recon_metrics 的任何逻辑。
"""
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Tuple

import torch

from . import protocols as _P       # noqa: F401  契约：本模块只经此入口拿值域常量
from .recon_metrics import ReconMetricsSuite

# ---- 02 篇 §4 五锚点冻结表（公开参照值；改动=协议变更，须走任务书修订） ----


@dataclass(frozen=True)
class CalibrationAnchor:
    """单个校准锚点的元数据。expected 的键名与 recon_metrics.compute() 对齐
    （psnr/ssim/lpips/rfid/rfvd_styleganv/rfvd_videogpt），便于排序一致性比对。"""
    name: str
    ckpt_ref: str            # 权重来源（HF id / 路径），文档用途
    official_script: str     # 官方评测脚本（文档用途；本 runner 不调用）
    expected: dict = field(default_factory=dict)   # 公开参照值（02 篇 §4 表）
    notes: str = ""


ANCHORS: Tuple[CalibrationAnchor, ...] = (
    CalibrationAnchor(
        name="OmniTokenizer",
        ckpt_ref="Daniel0724/OmniTokenizer (imagenet_ucf.ckpt + VAE 版)",
        official_script="vqgan_eval.py（Phase B docker 内）",
        expected={"rfid": 1.11, "rfid_vae": 0.69,
                  "rfvd_styleganv": 42.35, "rfvd_styleganv_vae": 23.44},
        notes="主锚点（Phase B 改造基座，02 篇 §3）",
    ),
    CalibrationAnchor(
        name="LARP",
        ckpt_ref="hywang66/LARP-L-Long-tokenizer (173M)",
        official_script="eval/eval_larp_tokenizer.py",
        expected={"rfvd_styleganv": 20.0},
        notes="工程骨架来源（D4），UCF-101 rFVD 参照",
    ),
    CalibrationAnchor(
        name="LeanVAE",
        ckpt_ref="LeanVAE HF 16ch checkpoint",
        official_script="evaluation/eval_recon_videos.py",
        expected={"psnr": 30.15, "lpips": 0.046},
        notes="效率基线（16ch），同学现成数据路径",
    ),
    CalibrationAnchor(
        name="AToken",
        ckpt_ref="apple/ml-atoken (download_checkpoints.sh)",
        official_script="atoken_inference + 我们的指标脚本",
        expected={"psnr": 29.7, "rfid": 0.21, "psnr_davis": 26.6, "rfvd_styleganv_davis": 29.2},
        notes="Apple Sample Code License——只读校准，不入仓（02 篇 §6）",
    ),
    CalibrationAnchor(
        name="Wan2.2",
        ckpt_ref="lightx2v/Autoencoders (Wan2.2-VAE)",
        official_script="vid_recon.py + 我们的指标脚本",
        expected={"psnr": 31.25, "rfid": 0.749, "psnr_davis": 27.64, "rfvd_styleganv_davis": 14.78},
        notes="重建对标主线（D3 / 01 篇重建 claim）",
    ),
)

# 编/解码 callable 契约：值域 [-1,1]，与 protocols 输出一致；hw 用于 decoder 还原空间尺寸
EncodeDecode = Tuple[Callable[[torch.Tensor], torch.Tensor],
                     Callable[[torch.Tensor, tuple], torch.Tensor]]


def calibrate_track(anchor: CalibrationAnchor,
                    encode_decode: EncodeDecode,
                    real_iter: Iterable[torch.Tensor], *,
                    is_video: bool,
                    device: torch.device = torch.device("cpu"),
                    with_lpips: bool = True,
                    with_frechet: bool = True) -> dict:
    """单锚点单 track 在我们尺子下的重建数字（流程 ②）。

    real_iter: 产出**已过 protocols** 的真值 batch（[-1,1]；图像 [B,3,256,256]，
        视频 [B,3,17,256,256]）。调用方负责 protocols 预处理，本函数不插值。
    encode_decode: (encode(x_eval)->z, decode(z,hw)->x_hat)；x_hat 与 real 同值域同形状。
    返回 recon_metrics.compute() 的 dict（psnr/ssim/lpips + rfid 或 rfvd_*）。
    """
    encode, decode = encode_decode
    suite = ReconMetricsSuite(device=device, is_video=is_video,
                              with_lpips=with_lpips, with_frechet=with_frechet)
    with torch.no_grad():
        for real in real_iter:
            hw = (real.shape[-2], real.shape[-1])
            fake = decode(encode(real), hw)
            if fake.shape != real.shape:
                raise ValueError(
                    f"[{anchor.name}] 重建形状 {tuple(fake.shape)} ≠ 真值 {tuple(real.shape)}；"
                    "锚点 decode 须输出与 protocols 真值逐位配对的张量")
            suite.add_batch(real, fake)
    return suite.compute()


def _ordinal(seq) -> list:
    """返回 seq 的秩序（名次列表，0=最小）；用于排序一致性比对（README D9 主判据）。"""
    order = sorted(range(len(seq)), key=lambda i: seq[i])
    rank = [0] * len(seq)
    for r, i in enumerate(order):
        rank[i] = r
    return rank


def _rank_consistency(records: list) -> list:
    """对每个 (track, metric) 计算锚点排序一致性。返回 [{track, metric, consistent, ...}]。

    仅当某 (track, metric) 下 ≥2 个锚点同时有"我们"与"公开"两值时才比（n<2 无统计意义）。
    FID/FVD/LPIPS 越小越好、PSNR/SSIM 越大越好——按"小为优"统一比较方向：
    小为优指标直接比；大为优指标取负后再比（等价反转方向）。
    """
    HIGHER_BETTER = {"psnr", "ssim"}
    by_tm: dict = {}
    for r in records:
        for m, ours in r["ours"].items():
            exp = r["expected"].get(m, r["expected"].get(f"{m}_davis"))
            if exp is None:
                continue
            by_tm.setdefault((r["track"], m), []).append((r["name"], ours, float(exp)))
    out = []
    for (track, metric), triple in by_tm.items():
        if len(triple) < 2:
            continue
        names = [t[0] for t in triple]
        sign = -1.0 if metric in HIGHER_BETTER else 1.0
        rank_ours = _ordinal([sign * t[1] for t in triple])
        rank_exp = _ordinal([sign * t[2] for t in triple])
        out.append({
            "track": track, "metric": metric,
            "anchors": names, "consistent": rank_ours == rank_exp,
            "our_rank": rank_ours, "expected_rank": rank_exp,
        })
    return out


def run_calibration(anchors: Iterable[CalibrationAnchor],
                    tracks: dict,
                    load_fn: Callable[[CalibrationAnchor, str], Optional[EncodeDecode]],
                    output_md: str, *,
                    device: torch.device = torch.device("cpu"),
                    official_repro_hook: Optional[Callable[[CalibrationAnchor], dict]] = None,
                    with_lpips: bool = True,
                    with_frechet: bool = True,
                    ) -> list:
    """五锚点校准主入口，产出 markdown《校准报告》到 output_md。

    tracks: dict[track_name -> (is_video: bool, real_iter)]，例如
        {"imagenet": (False, imagenet_iter), "davis": (True, davis_iter)}。
        real_iter 产出**已过 protocols** 的真值 batch。
    load_fn: (anchor, track_name) -> EncodeDecode | None；返回 None 表示该锚点不支持该
        track（跳过）。锚点权重加载逻辑各异（五家跨 LARP/OmniTokenizer/LeanVAE/AToken/Wan2.2），
        由调用方负责，本 runner 不绑定任何框架。
    official_repro_hook: (anchor) -> dict | None；流程 ① 官方脚本复现（P0 首日人工补）；
        None = 仅跑流程 ②，报告中标注"待 P0-golden 首日补"。
    with_lpips / with_frechet: 透传给 calibrate_track（控制是否计 LPIPS / rFID·rFVD），
        离线/冒烟可关以省时（测试用 with_lpips=False）。
    返回 records 列表（每条 = 一个锚点×track 的 {name, track, ours, expected, official}）。
    """
    records: list = []
    for anchor in anchors:
        official = official_repro_hook(anchor) if official_repro_hook else None
        for track_name, (is_video, real_iter) in tracks.items():
            ed = load_fn(anchor, track_name)
            if ed is None:
                continue                       # 该锚点不支持此 track（如 LARP 无 imagenet 行）
            ours = calibrate_track(anchor, ed, real_iter, is_video=is_video, device=device,
                                   with_lpips=with_lpips, with_frechet=with_frechet)
            records.append({"name": anchor.name, "track": track_name,
                            "ours": ours, "expected": dict(anchor.expected),
                            "official": official})
    _write_report(records, output_md, with_official=official_repro_hook is not None)
    return records


def _write_report(records: list, output_md: str, *, with_official: bool) -> None:
    """《校准报告》markdown 生成器。结构：概览 → 我们尺子数字表 → 排序一致性 → 官方脚本复现。"""
    lines = ["# UVT 校准报告（P0）", ""]
    lines.append(f"> 锚点×track 记录数：{len(records)}；"
                 f"排序一致性为主判据（README D9）。绝对值偏差见 §3。")
    lines += ["", "## 1. 我们尺子下的数字（流程 ②，核心）", ""]
    lines.append("| 锚点 | track | PSNR | SSIM | LPIPS | rFID | rFVD(styleganv) | rFVD(videogpt) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in records:
        o = r["ours"]
        def f(k):
            v = o.get(k)
            return f"{v:.3f}" if isinstance(v, float) else "—"
        lines.append(f"| {r['name']} | {r['track']} | {f('psnr')} | {f('ssim')} | "
                     f"{f('lpips')} | {f('rfid')} | {f('rfvd_styleganv')} | {f('rfvd_videogpt')} |")

    lines += ["", "## 2. 排序一致性（主判据，README D9）", ""]
    consist = _rank_consistency(records)
    if not consist:
        lines.append("> 无 (track,metric) 同时具备 ≥2 锚点的我们/公开两值——首日 P0 跑齐后复算。")
    else:
        lines.append("| track | metric | 锚点 | 一致? | 我们秩序 | 公开秩序 |")
        lines.append("|---|---|---|---|---|---|")
        for c in consist:
            flag = "✅" if c["consistent"] else "❌"
            lines.append(f"| {c['track']} | {c['metric']} | {', '.join(c['anchors'])} | "
                         f"{flag} | {c['our_rank']} | {c['expected_rank']} |")
        n_ok = sum(1 for c in consist if c["consistent"])
        lines.append("")
        lines.append(f"> 一致 {n_ok}/{len(consist)} 项；任何 ❌ 须在 P0-golden 首日定位"
                     "（实现混杂 / 协议偏差 / 权重错误）。")

    lines += ["", "## 3. 官方脚本复现（流程 ①，可选 hook）", ""]
    if with_official and any(r["official"] is not None for r in records):
        lines.append("| 锚点 | 官方复现数字 |")
        lines.append("|---|---|")
        seen = set()
        for r in records:
            if r["official"] is not None and r["name"] not in seen:
                seen.add(r["name"])
                kv = ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in r["official"].items())
                lines.append(f"| {r['name']} | {kv} |")
    else:
        lines.append("> official_repro_hook 未提供或未产出——待 P0-golden 首日人工补"
                     "（各锚点官方脚本各异，无法统一调用，03 篇 §4）。")

    with open(output_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
