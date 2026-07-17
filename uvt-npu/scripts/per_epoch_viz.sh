#!/usr/bin/env bash
# per_epoch_viz.sh · 每 epoch 完成(epoch-last.pth 更新)→ CPU 侧留出集 eval + 重建可视化
# 【零占卡】走 --cpu:不碰 8 卡训练,零 OOM 风险(主机 1.5TB 内存)。honors #2 每epoch分析 + #3 每epoch可视化。
# 轻量:n=48、PSNR/SSIM + GT|重建对比图(rFID 慢,留给 epoch-5 里程碑的满卡评测)。
set -uo pipefail

REPO=/home/ma-user/work/dataset/yh222/VITbaseVideoTokenizer/uvt-npu
CKPTDIR=/home/ma-user/work/dataset/yh222/uvt_runs/full05/uvt_stage1_imagenet_full_npu/stage1_b8
CKPT=$CKPTDIR/epoch-last.pth
STATUS=/home/ma-user/work/dataset/yh222/uvt_runs/full05.perepoch_viz.log

cd "$REPO"; source scripts/env_npu.sh >/dev/null 2>&1
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$STATUS"; }
log "=== per_epoch_viz 启动(CPU 侧,监视 $CKPT) ==="

last=""
while true; do
  # 训练全死 → 退出(不空转)
  if [ "$(ps aux | grep 'train.py' | grep -v grep | wc -l)" -eq 0 ]; then
    log "训练进程已结束,per_epoch_viz 退出。"; exit 0
  fi
  if [ -f "$CKPT" ]; then
    m=$(stat -c%Y "$CKPT")
    if [ "$m" != "$last" ]; then
      # 等写盘稳定(避免读到半个)
      s1=$(stat -c%s "$CKPT"); sleep 30; s2=$(stat -c%s "$CKPT")
      if [ "$s1" = "$s2" ]; then
        last=$m
        log "检测到新 epoch-last(mtime=$m),CPU 评测+可视化中..."
        "$PYBIN" scripts/eval_epoch.py --cpu --ckpt "$CKPT" --n 48 --bs 4 --img_k 8 \
            --sample 0 --tag full05_epochviz >> "$STATUS" 2>&1 \
          && log "✓ 本 epoch 可视化完成(图 logs/eval/full05_epochviz/,指标入 TRAINING_LOG)" \
          || log "✗ 本 epoch CPU 评测失败(见上),训练不受影响,等下个 epoch 重试。"
      fi
    fi
  fi
  sleep 180
done
