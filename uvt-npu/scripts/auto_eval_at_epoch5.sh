#!/usr/bin/env bash
# auto_eval_at_epoch5.sh · 等 epoch-5.pth 落盘 → 优雅暂停训练 → 留出集 eval+可视化+存档
# 自主运行(setsid 挂起),即使管理会话断开也会完成。用户指令:"跑完epoch5然后暂停训练测试一下"。
set -uo pipefail

REPO=/home/ma-user/work/dataset/yh222/VITbaseVideoTokenizer/uvt-npu
RUNROOT=/home/ma-user/work/dataset/yh222/uvt_runs
CKPTDIR=$RUNROOT/full05/uvt_stage1_imagenet_full_npu/stage1_b8
CKPT=$CKPTDIR/epoch-5.pth
STATUS=$RUNROOT/full05.epoch5_eval.log

cd "$REPO"
source scripts/env_npu.sh >/dev/null 2>&1

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$STATUS"; }

log "=== auto_eval_at_epoch5 启动,等待 $CKPT ==="

# 1) 等 epoch-5.pth 出现且大小稳定(约16GB,写盘~50s)
while [ ! -f "$CKPT" ]; do
  # 训练若整体死亡(0进程)则退出,避免空等
  if [ "$(ps aux | grep 'train.py' | grep -v grep | wc -l)" -eq 0 ]; then
    log "训练进程已全部消失且无 epoch-5.pth,放弃等待并退出。"; exit 2
  fi
  sleep 120
done
log "检测到 epoch-5.pth,等待写盘稳定..."
s1=$(stat -c%s "$CKPT"); sleep 40; s2=$(stat -c%s "$CKPT")
while [ "$s1" != "$s2" ]; do s1=$s2; sleep 30; s2=$(stat -c%s "$CKPT"); done
log "epoch-5.pth 写盘完成 ($((s2/1073741824))GB)。"

# 2) 优雅停训练(SIGTERM 主进程 → 各 rank 干净退出)
PID=$(pgrep -f "torch.distributed.run.*out_path.*full05" | head -1)
if [ -n "$PID" ]; then
  log "SIGTERM 训练主进程 PID=$PID(暂停训练)..."
  kill -TERM "$PID"
  for i in $(seq 1 30); do
    [ "$(ps aux | grep 'train.py' | grep -v grep | wc -l)" -eq 0 ] && break
    sleep 5
  done
fi
log "训练已停,进程数=$(ps aux | grep 'train.py' | grep -v grep | wc -l)。等 NPU 释放..."
sleep 20

# 3) 满卡跑留出集 eval(确定性 sample=0)+ rFID + 可视化,追加 TRAINING_LOG
log "开始 epoch-5 留出集 eval(ImageNet test-*,sample=False,n=512,+rFID)..."
"$PYBIN" scripts/eval_epoch.py --ckpt "$CKPT" --n 512 --bs 16 --img_k 8 \
    --sample 0 --rfid --tag full05_ep5 2>&1 | tee -a "$STATUS"
log "=== epoch-5 eval 完成 ==="
log "对比图: $REPO/logs/eval/full05_ep5/   指标已追加 docs/TRAINING_LOG.md"
log "⏸ 训练已暂停在 epoch-5。要续跑: cd $REPO && setsid bash scripts/launch_full05.sh < /dev/null > /dev/null 2>&1 &"
