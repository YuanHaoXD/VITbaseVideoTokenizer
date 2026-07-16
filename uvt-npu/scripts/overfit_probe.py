"""过拟合探针(overfit probe)—— 全量开跑前的"小数据集 sanity gate"。

方法论(那位有名科研大佬的老配方):先在一个**很小的固定数据集**上把模型往死里 overfit,
若 PSNR 能爬到很高(记忆住)⇒ 模型容量/损失组装/梯度回流/数据管线整条链路无病;
若爬不上去 ⇒ 立刻知道有 bug,而不必等几天的全量跑完才发现。时间成本从"天"降到"分钟"。

与 p1_smoke_overfit.py 的区别:这是**可复用的实验台**——
  · 真·小数据集(从 ImageNet parquet 抽 N 张,固定,minibatch 循环)
  · **对齐 run-full-03 的真实训练目标**(lpips0.5 + cos0.5 + distill0.5 + kl1e-6),
    不是简化 loss —— 这样这里过拟合能暴露的问题 == 全量跑会遇到的问题
  · wandb 监控(离线模式,本机无外网;跑完 `wandb sync <dir>` 可上传)
  · 每隔若干步 dump 一张 [GT | 重建] 对比图,肉眼看重建质量演进
  · 全部落盘到 uvt-npu/logs/<run>/(overfit.log + images/ + wandb/)

用法(单卡,生产模型 tiny=false + 真 SigLIP2):
  source scripts/env_npu.sh
  ASCEND_RT_VISIBLE_DEVICES=0 $PYBIN scripts/overfit_probe.py --n 16 --bs 4 --steps 2000

冒烟自检(无需 NPU/真权重,验证管线本身):
  $PYBIN scripts/overfit_probe.py --tiny --steps 5 --n 4 --bs 2

关键环境变量 / 参数见 argparse。所有开关都有合理默认。
"""
import argparse
import io
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# accel 首个 import：内含 triton shim + import torch_npu，必须在 import torch 前。
from utils import accel  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SIG = os.path.join(REPO, os.pardir, "models", "siglip2-so400m-patch16-256")
DEFAULT_PARQUET_DIR = "/home/ma-user/work/dataset/yh222/Datasets/imagenet-1k/data"


def parse_args():
    p = argparse.ArgumentParser(description="UVT overfit probe (small-dataset sanity gate)")
    p.add_argument("--n", type=int, default=16, help="固定小数据集大小(抽 N 张真图)")
    p.add_argument("--bs", type=int, default=4, help="minibatch 大小(N 内循环)")
    p.add_argument("--steps", type=int, default=2000, help="优化步数")
    p.add_argument("--lr", type=float, default=5e-4, help="学习率(overfit 用较高 lr,无 warmup)")
    p.add_argument("--size", type=int, default=256, help="输入分辨率(so400m 原生 256,tiny 时自动降 64)")
    # 损失配比：默认对齐 run-full-03(cfgs/uvt_stage1_imagenet_full_npu.yaml)
    p.add_argument("--lpips_w", type=float, default=0.5)
    p.add_argument("--cos_w", type=float, default=0.5)
    p.add_argument("--kl_w", type=float, default=1e-6)
    p.add_argument("--lambda_dist", type=float, default=0.5, help="distill 总权重(0=纯重建,排除蒸馏拉扯)")
    p.add_argument("--sample", type=int, default=1, help="1=重参数采样 z(真实训练路径);0=确定性 z=μ")
    p.add_argument("--log_every", type=int, default=10, help="标量日志间隔(步)")
    p.add_argument("--img_every", type=int, default=100, help="dump 对比图间隔(步)")
    p.add_argument("--img_k", type=int, default=6, help="对比图里放几张样本")
    p.add_argument("--siglip", type=str, default=os.environ.get("UVT_SIGLIP", DEFAULT_SIG))
    p.add_argument("--parquet_dir", type=str, default=os.environ.get("IMAGENET_PARQUET_DIR", DEFAULT_PARQUET_DIR))
    p.add_argument("--logdir", type=str, default=os.path.join(REPO, "logs"))
    p.add_argument("--tag", type=str, default="", help="run 名后缀")
    p.add_argument("--tiny", action="store_true", help="冒烟自检:tiny 模型 + tiny 教师,可 CPU 跑,验证管线")
    p.add_argument("--cpu", action="store_true", help="强制 CPU(冒烟自检用;避免误占正在训练的 NPU 卡)")
    p.add_argument("--no_wandb", action="store_true", help="关闭 wandb")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


class Tee:
    """把 stdout 同时写到终端和日志文件。"""
    def __init__(self, path):
        self.f = open(path, "w")
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.f.write(s)
        self.f.flush()

    def flush(self):
        self.stdout.flush()
        self.f.flush()


def load_real_images(parquet_dir, n, size, seed):
    """从 ImageNet parquet 抽 n 张真图 → [n,3,1,size,size] ∈ [0,1](短边 resize + center crop)。"""
    import glob
    import pyarrow.parquet as pq

    files = sorted(glob.glob(os.path.join(parquet_dir, "train-*.parquet")))
    if not files:
        raise FileNotFoundError(f"无 parquet 于 {parquet_dir}")
    tbl = pq.ParquetFile(files[0]).read_row_group(0).to_pylist()
    rng = np.random.RandomState(seed)
    idxs = rng.choice(len(tbl), size=min(n, len(tbl)), replace=False)
    imgs = []
    for i in idxs:
        im = Image.open(io.BytesIO(tbl[int(i)]["image"]["bytes"])).convert("RGB")
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
    """存 [上排 GT | 下排 重建] 对比 PNG。x,x_hat: [B,3,1,H,W] ∈ [0,1]。"""
    k = min(k, x.shape[0])
    gt = (x[:k, :, 0].clamp(0, 1) * 255).byte().cpu().numpy()       # [k,3,H,W]
    rc = (x_hat[:k, :, 0].clamp(0, 1) * 255).byte().cpu().numpy()
    gt = np.transpose(gt, (0, 2, 3, 1))  # [k,H,W,3]
    rc = np.transpose(rc, (0, 2, 3, 1))
    top = np.concatenate(list(gt), axis=1)   # 横向拼 → [H, k*W, 3]
    bot = np.concatenate(list(rc), axis=1)
    grid = np.concatenate([top, bot], axis=0)  # 纵向拼 GT / recon
    Image.fromarray(grid).save(path)
    return grid


def main():
    args = parse_args()
    if args.tiny:
        args.size = 64  # tiny 位置编码 4×4 → 输入 64

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- run 目录 ----
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"overfit_{stamp}" + (f"_{args.tag}" if args.tag else "") + ("_tiny" if args.tiny else "")
    run_dir = os.path.join(args.logdir, run_name)
    img_dir = os.path.join(run_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    sys.stdout = Tee(os.path.join(run_dir, "overfit.log"))
    print(f"[run] {run_name}  ->  {run_dir}", flush=True)
    print(f"[args] {vars(args)}", flush=True)

    dev = torch.device("cpu") if args.cpu else accel.device()
    amp_device = dev.type  # 'npu' / 'cuda' / 'cpu'
    amp_enabled = dev.type != "cpu"
    print(f"[device] {dev}  backend={accel.BACKEND}  amp={amp_enabled}", flush=True)

    # ---- wandb(离线;本机无外网)----
    wb = None
    if not args.no_wandb:
        try:
            os.environ.setdefault("WANDB_MODE", "offline")
            os.environ["WANDB_DIR"] = run_dir
            import wandb
            wb = wandb.init(project="uvt-overfit", name=run_name, dir=run_dir,
                            config=vars(args), mode="offline")
            print(f"[wandb] offline @ {run_dir}/wandb  (跑完可 `wandb sync` 上传)", flush=True)
        except Exception as e:
            print(f"[wandb] 初始化失败,继续无 wandb:{e}", flush=True)
            wb = None

    # ---- 数据:固定小数据集 ----
    if args.tiny:
        x = torch.rand(args.n, 3, 1, args.size, args.size)  # 冒烟用随机数即可验证管线
        print(f"[data] tiny 冒烟:随机 {args.n} 张 {args.size}²", flush=True)
    else:
        x = load_real_images(args.parquet_dir, args.n, args.size, args.seed)
        print(f"[data] 真图 {x.shape[0]} 张 {args.size}²  range=[{x.min():.2f},{x.max():.2f}]", flush=True)
    x = x.to(dev)

    # ---- 模型 + 教师(对齐 run-full-03 目标)----
    from models.uvt.uvt_tokenizer import UVTTokenizer, UVTConfig
    from losses.recon import recon_loss
    from losses.distill import DistillLoss
    from teachers.siglip2_teacher import SigLIP2Teacher

    cfg = UVTConfig(model_name=args.siglip, tiny=args.tiny,
                    lpips_weight=args.lpips_w, cos_weight=args.cos_w,
                    kl_weight=args.kl_w, use_cos_consistency=True)
    tok = UVTTokenizer(cfg).to(dev)
    tok.set_stage(1)
    tok.train()
    dim = getattr(cfg, "student_dim", None) or (64 if args.tiny else 1152)
    n_params = sum(p.numel() for p in tok.parameters()) / 1e6
    print(f"[model] UVTTokenizer #params={n_params:.1f}M  dim={dim}  stage=1", flush=True)

    use_dist = args.lambda_dist > 0
    if use_dist:
        t_img = SigLIP2Teacher(model_id=args.siglip, tiny=args.tiny).to(dev).eval()
        for p in t_img.parameters():
            p.requires_grad_(False)
        distill = DistillLoss(student_dim=dim, teacher_img_dim=dim, teacher_vid_dim=dim, cfg=cfg).to(dev)
        print(f"[teacher] SigLIP2 image teacher on;lambda_dist={args.lambda_dist}", flush=True)

    opt = torch.optim.AdamW([p for p in tok.parameters() if p.requires_grad],
                            lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)

    is_video = torch.zeros(args.bs, dtype=torch.bool, device=dev)

    # ---- μ 塌缩自检(真图应分化)----
    if not args.tiny:
        with torch.no_grad():
            mu = tok.encode(x, sample=False)["mu"].flatten(1)
            cm = F.cosine_similarity(mu[:, None, :], mu[None, :, :], dim=-1)
            off = cm[torch.triu_indices(x.shape[0], x.shape[0], offset=1).unbind()]
        print(f"[μ 塌缩检查] 两两余弦 mean={off.mean():.4f} min={off.min():.4f} "
              f"max={off.max():.4f}  (塌缩→接近1;健康→更低/分化)", flush=True)

    # ---- overfit 主循环 ----
    print("\n[train] 开始过拟合小数据集...\n", flush=True)
    best_psnr = 0.0
    for step in range(args.steps):
        sel = torch.randint(0, x.shape[0], (args.bs,), device=dev)
        xb = x[sel]
        opt.zero_grad()
        with torch.autocast(device_type=amp_device, dtype=torch.bfloat16, enabled=amp_enabled):
            out = tok._forward_core(xb, sample=bool(args.sample))
            rec = recon_loss(xb, out, cfg)
            loss = rec["total"]
            d_total = torch.zeros((), device=dev)
            if use_dist:
                t_patch, t_pool = t_img(xb[:, :, 0])
                d = distill(out["s"], out["s_pool"], out.get("decomp_out"),
                            t_patch, t_pool, None, is_video)
                d_total = d["total"]
                loss = loss + args.lambda_dist * d_total
        loss.backward()
        opt.step()

        if step % args.log_every == 0 or step == args.steps - 1:
            with torch.no_grad():
                xh = out["x_hat"].float().clamp(0, 1)
                mse = ((xh - xb.float()) ** 2).mean()
                psnr = (-10 * torch.log10(mse.clamp_min(1e-10))).item()
            best_psnr = max(best_psnr, psnr)
            print(f"step {step:4d}: loss={loss.item():.4f} psnr={psnr:5.2f} "
                  f"(l1={rec['l1'].item():.3f} lpips={rec['lpips'].item():.3f} "
                  f"cos={rec['cos_consistency'].item():.3f} kl={rec['kl'].item():.2e} "
                  f"dist={d_total.item():.3f})", flush=True)
            if wb is not None:
                wb.log({"loss": loss.item(), "psnr": psnr, "l1": rec["l1"].item(),
                        "lpips": rec["lpips"].item(), "cos": rec["cos_consistency"].item(),
                        "kl": rec["kl"].item(), "dist": d_total.item(), "lr": args.lr}, step=step)

        if step % args.img_every == 0 or step == args.steps - 1:
            with torch.no_grad():
                # 用固定的前 img_k 张(而非当前 minibatch)看整体演进
                xv = x[: args.img_k]
                ov = tok._forward_core(xv, sample=False)
                grid = save_compare_grid(xv, ov["x_hat"].float(),
                                         os.path.join(img_dir, f"step_{step:05d}.png"), args.img_k)
            if wb is not None:
                import wandb
                wb.log({"recon(上GT/下recon)": wandb.Image(grid)}, step=step)

    print(f"\n[done] best_psnr={best_psnr:.2f}  images→{img_dir}", flush=True)
    if not args.tiny:
        verdict = "PASS(重建管线健康✓)" if best_psnr > 30 else "注意:未过 30,查管线/步数/lr"
        print(f"[verdict] {verdict}", flush=True)
    if wb is not None:
        wb.finish()


if __name__ == "__main__":
    main()
