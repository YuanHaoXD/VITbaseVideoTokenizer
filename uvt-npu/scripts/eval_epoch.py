"""eval_epoch.py · 单个 checkpoint 的重建 eval + 可视化 + 存档(全量训练的阶段性检查)。

用途:训练跑到某 epoch(暂停或对 milestone/epoch-last)后,拿权重在【留出集】(ImageNet
  test-*.parquet,非训练用的 train-*)上做**确定性**(sample=False)重建评测,并存 GT|重建对比图供肉眼看。
  —— 回答"训练集 PSNR 之外,泛化到底如何 + 人眼看糊不糊"。

产出(--outdir/<tag>/):
  · metrics.json          —— 结构化指标(epoch, psnr, ssim, [rfid], n, sample)
  · compare_epochNN.png   —— [上排 GT | 下排 重建] 对比图
  · 追加一行到 --logmd(默认 docs/TRAINING_LOG.md 的 eval 表)

用法(训练暂停后,单卡):
  source scripts/env_npu.sh
  ASCEND_RT_VISIBLE_DEVICES=0 $PYBIN scripts/eval_epoch.py \
      --ckpt /home/ma-user/work/dataset/yh222/uvt_runs/full05/uvt_stage1_imagenet_full_npu/stage1_b8/epoch-5.pth \
      --n 256 --bs 16 --tag full05_ep5

冒烟(无需真权重,验证脚本本身,tiny 随机模型):
  $PYBIN scripts/eval_epoch.py --tiny --n 4 --bs 2 --cpu
"""
import argparse
import io
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import accel  # noqa: E402  (triton shim + import torch_npu，必须在 torch 前)
import torch  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SIG = os.path.join(REPO, os.pardir, "models", "siglip2-so400m-patch16-256")
DEFAULT_PARQUET_DIR = "/home/ma-user/work/dataset/yh222/Datasets/imagenet-1k/data"


def parse_args():
    p = argparse.ArgumentParser(description="UVT 单 checkpoint 重建 eval + 可视化")
    p.add_argument("--ckpt", type=str, default=None, help="checkpoint 路径(epoch-N.pth / epoch-last.pth)")
    p.add_argument("--n", type=int, default=256, help="留出集评测图数")
    p.add_argument("--bs", type=int, default=16, help="前向 batch")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--sample", type=int, default=0, help="0=确定性 z=μ(eval 标准);1=重参数采样")
    p.add_argument("--img_k", type=int, default=8, help="对比图放几张")
    p.add_argument("--parquet_dir", type=str, default=os.environ.get("IMAGENET_PARQUET_DIR", DEFAULT_PARQUET_DIR))
    p.add_argument("--val_glob", type=str, default="test-*.parquet", help="留出集分片(默认 test-*,非训练 train-*)")
    p.add_argument("--outdir", type=str, default=os.path.join(REPO, "logs", "eval"))
    p.add_argument("--logmd", type=str, default=os.path.join(REPO, "docs", "TRAINING_LOG.md"))
    p.add_argument("--tag", type=str, default="eval", help="run 名(如 full05_ep5)")
    p.add_argument("--rfid", action="store_true", help="尝试算 rFID(需 inception 权重,best-effort)")
    p.add_argument("--tiny", action="store_true", help="冒烟:tiny 随机模型,不加载 ckpt")
    p.add_argument("--cpu", action="store_true", help="强制 CPU(冒烟用)")
    p.add_argument("--seed", type=int, default=1234, help="留出集抽样种子(固定=每次同一批图,可比)")
    return p.parse_args()


def load_val_images(parquet_dir, glob_pat, n, size, seed):
    """从【留出集】parquet 抽 n 张 → [n,3,1,size,size] ∈ [0,1](短边 resize + center crop,同 overfit_probe)。"""
    import glob
    import pyarrow.parquet as pq

    files = sorted(glob.glob(os.path.join(parquet_dir, glob_pat)))
    if not files:
        raise FileNotFoundError(f"无 {glob_pat} 于 {parquet_dir}")
    # 跨多个分片凑够 n（每片先读第 0 个 row group 足够）
    pool = []
    for f in files:
        pool.extend(pq.ParquetFile(f).read_row_group(0).to_pylist())
        if len(pool) >= max(n * 4, n + 16):
            break
    rng = np.random.RandomState(seed)
    idxs = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
    imgs = []
    for i in idxs:
        rec = pool[int(i)]
        im = Image.open(io.BytesIO(rec["image"]["bytes"])).convert("RGB")
        w, h = im.size
        s = size / min(w, h)
        im = im.resize((round(w * s), round(h * s)), Image.BICUBIC)
        w, h = im.size
        l, t = (w - size) // 2, (h - size) // 2
        im = im.crop((l, t, l + size, t + size))
        arr = torch.from_numpy(np.array(im, dtype=np.uint8)).permute(2, 0, 1).float() / 255.0
        imgs.append(arr)
    return torch.stack(imgs).unsqueeze(2)  # [n,3,1,H,W]


def save_compare_grid(x, x_hat, path, k):
    """[上排 GT | 下排 重建] 对比 PNG。x,x_hat: [B,3,1,H,W] ∈ [0,1]。"""
    k = min(k, x.shape[0])
    gt = np.transpose((x[:k, :, 0].clamp(0, 1) * 255).byte().cpu().numpy(), (0, 2, 3, 1))
    rc = np.transpose((x_hat[:k, :, 0].clamp(0, 1) * 255).byte().cpu().numpy(), (0, 2, 3, 1))
    grid = np.concatenate([np.concatenate(list(gt), axis=1),
                           np.concatenate(list(rc), axis=1)], axis=0)
    Image.fromarray(grid).save(path)


def build_model(args, dev):
    """从 checkpoint 重建**完全一致**的模型(用 ckpt 自带 model spec);Decompressor 已被存盘钩子剔除→strict=False。"""
    from models.uvt.uvt_tokenizer import UVTTokenizer, UVTConfig
    import models as models_pkg

    if args.tiny:
        cfg = UVTConfig(model_name=DEFAULT_SIG, tiny=True)
        tok = UVTTokenizer(cfg).to(dev).eval()
        return tok, -1

    ckt = torch.load(args.ckpt, map_location="cpu", weights_only=False)  # torch2.6:checkpoint 含 EasyDict,需关 weights_only
    epoch = int(ckt.get("epoch", -1))
    spec = ckt["model"]                       # {name, args, sd}
    sd = spec["sd"]
    sd = { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }  # 去 DDP 前缀
    tok = models_pkg.make(spec, load_sd=False).to(dev).eval()  # 用 ckpt 自带 args 建型
    missing, unexpected = tok.load_state_dict(sd, strict=False)
    # 预期 missing 只含 decompressor.*(存盘剔除,eval 不需要);其余 missing/unexpected 应为空
    bad_missing = [m for m in missing if "decompressor." not in m]
    assert not bad_missing, f"非预期缺失权重: {bad_missing[:5]}"
    assert not unexpected, f"非预期多余权重: {unexpected[:5]}"
    return tok, epoch


@torch.no_grad()
def main():
    args = parse_args()
    if args.tiny:
        args.size = 64
    dev = torch.device("cpu") if args.cpu else accel.device()
    os.makedirs(os.path.join(args.outdir, args.tag), exist_ok=True)
    run_dir = os.path.join(args.outdir, args.tag)

    print(f"[eval] dev={dev} ckpt={args.ckpt} n={args.n} sample={args.sample}", flush=True)
    tok, epoch = build_model(args, dev)
    n_params = sum(p.numel() for p in tok.parameters()) / 1e6
    print(f"[eval] model #params={n_params:.1f}M epoch={epoch}", flush=True)

    x = load_val_images(args.parquet_dir, args.val_glob, args.n, args.size, args.seed) if not args.tiny \
        else torch.rand(args.n, 3, 1, args.size, args.size)
    x = x.to(dev)

    # 分 batch 前向重建（确定性 sample=False）
    from eval.recon_metrics import psnr as psnr_fn, ssim as ssim_fn
    recons = []
    for i in range(0, x.shape[0], args.bs):
        xb = x[i:i + args.bs]
        out = tok._forward_core(xb, sample=bool(args.sample))
        recons.append(out["x_hat"].clamp(0, 1).float())
    x_hat = torch.cat(recons, dim=0)

    # 指标（在 [0,1] 上,data_range=1）
    psnr_val = float(psnr_fn(x_hat, x).mean().item())
    ssim_val = float(ssim_fn(x_hat, x).mean().item())
    rfid_val = None
    if args.rfid and not args.tiny:
        try:
            from eval.recon_metrics import RFIDEvaluator
            rfid_val = float(RFIDEvaluator().compute(x, x_hat))
        except Exception as e:  # noqa: BLE001 — rFID best-effort,inception 权重可能缺
            print(f"[eval] rFID 跳过(best-effort 失败): {e}", flush=True)

    # 存对比图
    stamp = time.strftime("%Y%m%d_%H%M%S")
    img_path = os.path.join(run_dir, f"compare_ep{epoch:02d}_{stamp}.png")
    save_compare_grid(x, x_hat, img_path, args.img_k)

    # 存 metrics.json
    rec = {"tag": args.tag, "epoch": epoch, "psnr": round(psnr_val, 3),
           "ssim": round(ssim_val, 4), "rfid": (round(rfid_val, 4) if rfid_val is not None else None),
           "n": int(x.shape[0]), "sample": args.sample, "ckpt": args.ckpt,
           "img": img_path, "time": stamp}
    with open(os.path.join(run_dir, "metrics.json"), "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 追加到 TRAINING_LOG 的 eval 表(存在则 append 一行)
    row = (f"| {args.tag} | {epoch} | {psnr_val:.3f} | {ssim_val:.4f} | "
           f"{'—' if rfid_val is None else f'{rfid_val:.4f}'} | {x.shape[0]} | "
           f"sample={args.sample} | `{os.path.relpath(img_path, REPO)}` |")
    try:
        with open(args.logmd, "a") as f:
            f.write("\n" + row)
    except Exception as e:  # noqa: BLE001
        print(f"[eval] 追加 TRAINING_LOG 失败(非致命): {e}", flush=True)

    print(f"[eval] ✅ epoch={epoch} PSNR={psnr_val:.3f} SSIM={ssim_val:.4f} "
          f"rFID={'—' if rfid_val is None else f'{rfid_val:.4f}'} n={x.shape[0]}", flush=True)
    print(f"[eval] 对比图: {img_path}", flush=True)
    print(f"[eval] TRAINING_LOG 行:\n{row}", flush=True)


if __name__ == "__main__":
    main()
