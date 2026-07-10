# UVT NPU 移植说明（uvt-npu/）

> 本目录是主仓 `uvt/`（CUDA/LARP）的 **NPU 移植版**，原 `uvt/` 零改动。移植已验证：单卡冒烟 ✓、8 卡 HCCL DDP 冒烟 ✓。

## 1. 环境

- **硬件**：8× 华为昇腾 Ascend 910B2（每卡 64G HBM），Kunpeng-920 aarch64，CANN 8.2.RC1（`npu-smi 25.5.1`）
- **Python 环境**：conda env `PyTorch-2.6.0`（python 3.11.10，torch 2.6.0，**torch_npu 2.6.0.post5**，torchvision 0.21.0，transformers 4.53.1，timm 1.0.9，numpy 1.26.4，scipy 1.15.3）
- **已补装到该 env**：`mergedeep`、`pytorch-msssim`、`lpips`、`moviepy`、`wandb`（走华为云镜像 `repo.myhuaweicloud.com`）
- **decord 无 aarch64 轮子**：已把 `datasets/video_dataset.py` 的 `import decord` 改成惰性导入（`_decord()`），null128 假数据路径不受影响；真实视频解码需后端（opencv/pyav），待真实数据阶段补。

激活环境（本机 conda init 异常，直接用绝对路径 python）：
```bash
PYBIN=/home/ma-user/anaconda3/envs/PyTorch-2.6.0/bin/python
```

## 2. 移植改动清单（相对原 uvt/）

| 文件 | 改动 | 原因 |
|---|---|---|
| `utils/accel.py`（新增） | triton shim + `import torch_npu` + 统一设备 facade（`is_available/set_device/current_device/empty_cache/manual_seed_all/rng/GradScaler/autocast/init_process_group backend/map_location/pin_memory_device`） | 把散落的 `torch.cuda.*` 收口，支持 npu/cuda/cpu 三态 |
| `train.py` | 顶部内联 triton shim；`torch.cuda→accel`；`nccl→hccl` | shim 必须早于 `import torch`；分布式后端随加速器走 |
| `trainers/base_trainer.py` | `torch.cuda→accel`；`fvd_calculator` 构造 try/except 容错；DDP 加 `find_unused_parameters`（cfg 可调，默认 True）；DataLoader 加 `pin_memory_device` | I3D 权重缺失不挡训练；图像 batch 不用 Decompressor 需 find_unused；NPU pin_memory |
| `trainers/uvt_tokenizer_trainer.py` | `torch.cuda→accel`；autocast `device_type→accel.BACKEND`；DDP `find_unused_parameters` | NPU autocast 设备类型；同上 DDP |
| `datasets/video_dataset.py` | decord 改惰性导入 | aarch64 无 decord 轮子 |
| `cfgs/uvt_stage1_npu.yaml`（新增） | `compile: false`（其余同 `uvt_stage1.yaml`） | NPU 上 `torch.compile` 经 inductor→triton 链路脆弱，先稳后快 |
| `scripts/env_npu.sh`（新增） | 固化 `TORCH_DEVICE_BACKEND_AUTOLOAD=0` + `HF_ENDPOINT=https://hf-mirror.com` | 见下面两个关键坑 |
| `conftest.py`（新增） | pytest 收集前应用 triton shim | 让测试的 `import torch` 不崩 |

**模型代码（`models/uvt/*`、`losses/*`、`teachers/*`）一行未动** —— 它们本就设备无关（用 `.to(device)`）。改动全在框架层。

## 3. 两个关键坑（必须知道）

### 坑①：triton 3.6 与 torch 2.6.0 不兼容
本机装的是 triton 3.6，但 torch 2.6.0 要的是 3.2。新版把 `triton.compiler.compiler.AttrsDescriptor` 改名了，而 `import torch`（自动加载 torch_npu）会链式触发 `torch._inductor.runtime.hints → from triton...import AttrsDescriptor → ImportError`。aarch64 镜像没有 3.2 的轮子，降级无解。
**解法**：triton 在 NPU 上根本不用（昇腾走 CANN 编译），在 `import torch_npu` 前给 `triton.compiler.compiler` 补一个占位 `AttrsDescriptor` 即可骗过这条 import（见 `utils/accel.py` 顶部、`train.py` 顶部、`conftest.py`）。

### 坑②：torchrun 启动器自身先 import torch
`torchrun`/`torch.distributed.run` 启动器在跑 `train.py` 之前自己就 `import torch`，发生在 train.py 顶部 shim 之前 → 启动器直接崩。
**解法**：环境变量 `TORCH_DEVICE_BACKEND_AUTOLOAD=0` 关掉 torch 的后端自动加载（启动器 import torch 就干净了），torch_npu 改由 `utils/accel.py` 在 shim 之后**显式** import（已就位）。`scripts/env_npu.sh` 已固化。

## 4. 运行命令

```bash
cd /cache/VITbaseVideoTokenizer/uvt-npu
source scripts/env_npu.sh

# —— 单卡 tiny 冒烟（无权重下载，验全图）——
$PYBIN train.py --cfg cfgs/uvt_stage1_npu.yaml \
    --csv_file null128 --batch_size 2 --frame_num 17 --input_size 64 --num_workers 0 \
    --out_path /cache/_uvt_npu_smoke --replace \
    --opts compile false model.args.tiny true teachers.tiny true teachers.vid_mock true \
           teachers.vid_mock_args.dim 64 teachers.vid_mock_args.spatial_tokens 16 \
           distill.student_dim 64 distill.teacher_img_dim 64 distill.teacher_vid_dim 64 \
           max_epoch 1 grad_accumulates 1 model.args.lpips_weight 0.0

# —— 8 卡 HCCL tiny 冒烟 ——
$PYBIN -m torch.distributed.run --nproc_per_node=8 train.py --cfg cfgs/uvt_stage1_npu.yaml \
    --csv_file null128 --batch_size 8 --frame_num 17 --input_size 64 --num_workers 8 \
    --out_path /cache/_uvt_npu_smoke8 --replace \
    --opts compile false model.args.tiny true teachers.tiny true teachers.vid_mock true \
           teachers.vid_mock_args.dim 64 teachers.vid_mock_args.spatial_tokens 16 \
           distill.student_dim 64 distill.teacher_img_dim 64 distill.teacher_vid_dim 64 \
           max_epoch 1 grad_accumulates 1 model.args.lpips_weight 0.0
```
> tiny 冒烟用 `--input_size 64`（tiny 骨干 position embedding 只有 16 位=4×4），并把 distill/mock 维度全降到 64。生产训练（`tiny:false`）用 `--input_size 256`、删掉那些 tiny 维度 override。

### 已验证结果（2026-07-10）
- 单卡：`Epoch 1 done 31.6s`，loss 4.41→2.67（84 步），fps 12，所有损失项有限。
- 8 卡：`Epoch 1 done 24.6s`，loss 3.68，fps 19.5，HCCL 梯度同步正常，exit 0。

## 5. 待下载（真实训练 `tiny:false` 用）

> 本机 `huggingface.co` **不通**，但 `hf-mirror.com` 通 → `scripts/env_npu.sh` 已设 `HF_ENDPOINT=https://hf-mirror.com`，`from_pretrained` 会自动走镜像。

| 优先级 | 用途 | 权重/数据 | 获取方式 | 放置/备注 |
|---|---|---|---|---|
| **必须** | 主干初始化 + 图像教师 + 文本塔 | SigLIP2-So400M-**patch16-256** | HF `google/siglip2-so400m-patch16-256`（走 hf-mirror） | `UVTConfig.model_name` 默认即此；首次 `from_pretrained` 自动下载到 `~/.cache/huggingface`。**任何非 tiny 训练的硬前提** |
| 需要(视频蒸馏) | 视频教师 | InternVideo2-Stage2-1B | HF `OpenGVLab/InternVideo2-Stage2_1B`（走 hf-mirror） | `cfg.teachers.vid.teacher_id`；设 `teachers.vid_mock false` 才用。加载方式读其官方 repo（P0 先验证可加载） |
| eval 用 | FVD 的 I3D | `i3d_torchscript.pt` / `i3d_pretrained_400.pt` | OmniTokenizer release（HF `Daniel0724/OmniTokenizer`） | 放 `utils/fvd/i3d_torchscript.pt`；**训练不需要**（FVDCalculator 已容错），FVD eval 才需要 |
| eval 用 | CKNNA 参照 | DINOv2-L | HF `facebook/dinov2-large`（走 hf-mirror） | `eval/semantic/cknna.py` 惰性导入，无则 skip；**训练不需要** |
| 已就绪 | lpips 感知损失 | VGG16 | torchvision（冒烟时已自动下载缓存） | 已缓存在 `~/.cache/torch/hub`，无需再动 |
| 训练数据 | 主线训练 | ImageNet-1k / UCF-101 / OpenVid-1M / DAVIS-2017 | 见 `docs/03_数据与评测.md` | LARP 式 CSV 元数据放 `data/metadata/`；转码/去重见 03 §2 |

**最小可跑真实训练（仅图像 Stage 1，免 InternVideo）**：下载 SigLIP2 即可，`teachers.vid_mock true` 先顶着。
**完整训练**：再下 InternVideo + 数据集。

## 6. 已知遗留（真实数据阶段处理）

- **decord 后端**：真实视频解码需 decord 替代（opencv/pyav）——改 `datasets/video_dataset.py` 的 `_decord()` 内部。
- **torch.compile**：当前关。NPU ge 后端稳定后可试开（`--opts compile true`）。
- **Stage 2/3**：冒烟只验了 Stage 1。Stage 2（GAN，D 侧 Adam + LeCam）、Stage 3（estimate_latent_stats）逻辑同源、DDP 已验，但真实权重下未跑过，建议真训练时单测一次。
