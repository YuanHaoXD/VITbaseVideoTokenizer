#!/usr/bin/env bash
# launch_full05.sh · Stage-1 全量 ImageNet 训练(8卡 HCCL DDP)
# 断连保护:本脚本用 setsid 调用,完全脱离终端;日志重定向到持久 NFS。
# 用法: bash scripts/launch_full05.sh   (或经 setsid 挂起,见文末注释)
set -euo pipefail

REPO=/home/ma-user/work/dataset/yh222/VITbaseVideoTokenizer/uvt-npu
RUNROOT=/home/ma-user/work/dataset/yh222/uvt_runs
OUT=$RUNROOT/full05
LOG=$RUNROOT/full05.log

cd "$REPO"
# NPU 环境(TORCH_DEVICE_BACKEND_AUTOLOAD=0 / HF_ENDPOINT / PYBIN)——必需,见 NPU_NOTES.md
source scripts/env_npu.sh

{
  echo "==================================================================="
  echo "[launch_full05] start $(date '+%Y-%m-%d %H:%M:%S')"
  echo "[launch_full05] host=$(hostname) PYBIN=$PYBIN"
  echo "[launch_full05] out=$OUT  cfg=cfgs/uvt_stage1_imagenet_full_npu.yaml"
  echo "==================================================================="
} >> "$LOG" 2>&1

# 8 卡 HCCL DDP。--replace: 覆盖 out 目录(full05 是新目录,首跑无妨)。
# --csv_file null128 仅占位(真实数据走 joint_dataset/parquet);vid_mock 无需设(纯图像 batch vid 项自屏蔽)。
exec "$PYBIN" -m torch.distributed.run --nproc_per_node=8 train.py \
    --cfg cfgs/uvt_stage1_imagenet_full_npu.yaml \
    --csv_file null128 --batch_size 8 --frame_num 17 --input_size 256 \
    --num_workers 8 --out_path "$OUT" --replace \
    >> "$LOG" 2>&1

# —— 断连挂起方式(启动时用):
#   setsid bash scripts/launch_full05.sh < /dev/null > /dev/null 2>&1 &
#   → setsid 新建会话,脱离控制终端;SSH 断开不发 SIGHUP 给它。日志仍写 full05.log。
