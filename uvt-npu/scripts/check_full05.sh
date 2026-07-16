#!/usr/bin/env bash
# check_full05.sh · 一眼看 full05 训练状态(断连回来自查用)
# 用法: bash scripts/check_full05.sh
RUN=/home/ma-user/work/dataset/yh222/uvt_runs
LOG=$RUN/full05.log

echo "======== full05 状态 $(date '+%m-%d %H:%M:%S') ========"
# 1. 进程
N=$(ps aux | grep 'train.py' | grep -v grep | wc -l)
echo "[进程] train.py 进程数 = $N  (健康应为 17: 主+8worker+8dataloader; 0=已停)"

# 2. 最新进度(进度条最后一行)
echo "[进度] $(tail -c 800 "$LOG" | tr '\r' '\n' | grep -aE 'train:.*it/s' | tail -1)"

# 3. checkpoint 落盘情况(首要验证点)
echo "[存盘] $(find "$RUN/full05" -name '*.pth' -o -name '*.pt' 2>/dev/null | wc -l) 个 checkpoint:"
find "$RUN/full05" -name '*.pth' -o -name '*.pt' 2>/dev/null | while read f; do
  printf "        %s  (%s)\n" "$(basename "$f")" "$(date -r "$f" '+%m-%d %H:%M')"
done

# 4. 报错扫描
ERR=$(grep -aE 'Traceback|RuntimeError|SignalException|CUDA|HCCL.*error|out of memory' "$LOG" 2>/dev/null | grep -viE 'collect_env|owner does not match' | tail -3)
if [ -n "$ERR" ]; then echo "[!! 报错]"; echo "$ERR"; else echo "[报错] 无"; fi
echo "==================================================="
echo "完整日志: tail -f $LOG"
