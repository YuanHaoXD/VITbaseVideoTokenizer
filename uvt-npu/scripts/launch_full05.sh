#!/usr/bin/env bash
# launch_full05.sh · Stage-1 全量 ImageNet 训练(8卡 HCCL DDP)· 续跑安全 + wandb
# 断连保护:本脚本用 setsid 调用,完全脱离终端;日志重定向到持久 NFS。
# ⚠️ 无 --replace:save_dir 存在 epoch-last.pth 时 base_trainer 自动【续跑】(不删旧盘);
#    首次(save_dir 不存在)则从头开始。wandb 走 offline(本机无外网),跑完 `wandb sync` 可上传。
set -euo pipefail

REPO=/home/ma-user/work/dataset/yh222/VITbaseVideoTokenizer/uvt-npu
RUNROOT=/home/ma-user/work/dataset/yh222/uvt_runs
OUT=$RUNROOT/full05
LOG=$RUNROOT/full05.log

cd "$REPO"
source scripts/env_npu.sh
export WANDB_MODE=offline          # 本机无外网:强制离线;wandb 写 $OUT/wandb,后续 `wandb sync` 上传

{
  echo "==================================================================="
  echo "[launch_full05] start/resume $(date '+%Y-%m-%d %H:%M:%S')  WANDB_MODE=$WANDB_MODE"
  echo "[launch_full05] out=$OUT  cfg=cfgs/uvt_stage1_imagenet_full_npu.yaml"
  echo "==================================================================="
} >> "$LOG" 2>&1

# 8 卡 HCCL DDP。无 --replace → epoch-last.pth 在则自动续跑。
# -w 开 wandb;无 wandb.yaml 故必须同时给 --wandb_project 与 --wandn_entity(否则 base_trainer KeyError)。
exec "$PYBIN" -m torch.distributed.run --nproc_per_node=8 train.py \
    --cfg cfgs/uvt_stage1_imagenet_full_npu.yaml \
    --csv_file null128 --batch_size 8 --frame_num 17 --input_size 256 \
    --num_workers 8 --out_path "$OUT" \
    -w --wandb_project uvt-full05 --wandn_entity yuanhao \
    >> "$LOG" 2>&1

# —— 断连挂起(启动时用):
#   setsid bash scripts/launch_full05.sh < /dev/null > /dev/null 2>&1 &

