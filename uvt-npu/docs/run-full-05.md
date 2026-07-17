# run-full-05 · Stage-1 全量 ImageNet 训练记录

> 开始: 2026-07-16 晚
> 决策: 用户拍板「续跑全量训练(full05)」——不动论文偏离,按当前 cfg 从头训,这次能存 checkpoint。
> 前置: 论文对照调研完成(见 `PAPER_CODE_DIFF.md`);checkpoint 保存 bug 已修(commit 22a7d1f)。

## 为什么是 full05(接 full04)
- full03: epoch-1 边界存盘崩溃(common.py 裸 cuda)→ 已修。
- full04: 用修复后代码重启,跑到 2662/19609 步被手动停(SIGTERM),为做论文调研 + overfit 验证让路。**未存 checkpoint**。
- full05: 8 卡全空闲,从头正式跑。**目标: 跑过 epoch-1 边界,确认 checkpoint 落盘成功**(这是 full03 崩的地方,首要验证点)。

## 配置(cfgs/uvt_stage1_imagenet_full_npu.yaml)
| 项 | 值 | 备注 |
|---|---|---|
| stage | 1 | 基础训练(全参可训) |
| 数据 | ImageNet-1k parquet 全量 1.28M | joint_dataset/parquet_image_dataset |
| max_epoch | 60 | |
| global batch | 256 | 8卡 × bs8 × grad_accum4 |
| steps/epoch | 19609 | |
| lr | 2e-4 peak | |
| lpips_weight | **0.5** | full-03 下调(HYDRA λ_perc=0.1,原1.0偏高压PSNR;为PSNR让路) |
| cos_weight | **0.5** | full-03 下调(减轻共享z语义扰动) |
| kl_weight | 1e-6 | |
| lambda_dist | 0.5 | |
| latest_interval | 1 | 每 epoch 存 latest |
| save_epoch | 5 | 每 5 epoch 存里程碑 |
| teachers.vid_mock | (真训练用真教师) | 纯图像 batch,vid 项本就屏蔽 |

## 运行方式(断连保护)
- **无 tmux/screen** → 用 `setsid` 完全脱离终端 + 控制进程组,断 SSH/关终端不受影响。
- 启动脚本: `scripts/launch_full05.sh`
- 主日志: `/home/ma-user/work/dataset/yh222/uvt_runs/full05.log`
- output(ckpt/tensorboard): `/home/ma-user/work/dataset/yh222/uvt_runs/full05/`
- 8 卡 HCCL DDP: `torch.distributed.run --nproc_per_node=8`

## 首要验证点(按时间)
1. [~5min] 8 worker 起来、数据加载、第一个 step 出 loss/psnr → 训练循环健康。
2. [~3.5h] **epoch-1 边界: checkpoint 落盘成功**(full03 崩点,最关键)。
3. [持续] PSNR 爬升轨迹(full03/04 经验: epoch-1 内到 18-19)。

## 进度日志(每次巡检追加)
- **2026-07-16 22:30 启动成功**。setsid 挂起(主进程 PPID=1,已脱离终端,断连不停)。
  - 数据集: `Joint source: parquet_image_dataset, len=161236`(真实 ImageNet,非假数据)✓
  - 8 卡 HCCL DDP 全部在算(HBM 4-8GB)✓ 无报错 ✓
  - 训练循环启动: `train: psnr=2.76 loss=3.145 step3/19609`(warmup 起点,正常)✓
  - 主日志: `full05.log` · output: `full05/` · launcher PID 存于 `full05.launcher.pid`
  - 自查命令: `bash scripts/check_full05.sh`
  - 待观察: [ ] epoch-1 边界存盘(~3.5h 后,full03 崩点) [ ] PSNR 爬升(经验 epoch-1 内到 18-19)

- **2026-07-17 02:12 ✅ epoch-1 边界存盘成功 —— full03 崩溃点最终验收通过。**
  - `Latest checkpoint saved. Time: 50.64s` → `epoch-last.pth` 15.8GB 落盘 ✓
  - **存盘后训练未崩,继续 epoch-2**(存盘那刻正是 full03 崩点;修复根除确认)✓
  - PSNR: warmup 2.76 → **epoch-1 达 24.2**(优于 full03/04 的 18-19 经验)✓
  - lr_m=1.0(warmup 结束进满 lr)。吞吐 ~1.47 it/s。
  - 注:日志有 torch_npu `_use_new_zipfile_serialization` **告警**(非错误),存盘正常完成。
  - checkpoint 保存 bug(commit 22a7d1f)**在真实 8 卡全量训练上验收通过**。

- **2026-07-17 10:03 巡检 · PSNR 爬升健康**。
  - epoch-1=24.2 → epoch-2/3 存盘正常(09:35 epoch-last 滚动)→ **epoch-4 进行中 PSNR≈29.4**。
  - 稳定 ~1.55 it/s,进程 81,零报错,已连续跑 ~11.5h。
  - **坐标校准**:论文 ImageNet 重建 PSNR 表1≈31.1~31.7 / 表9(S3)32.04;当前仅 Stage-1、
    epoch-4/60、且训练态 `sample=True`(压低 PSNR)——已逼近论文门口,真实上限预计更高。
  - **caveat**:此为训练集即时 PSNR(非 eval 协议 val);cfg 的 lpips/cos=0.5 是 full-03 为 PSNR
    主动下调,偏向 PSNR 好看,rFID/感知需 Stage-2 GAN 补;质量须 PSNR+rFID 合看。

## 为什么 60 epoch(依据)
- 论文 Stage-1 = **300k iterations**(附录 A.2);Table-1 那个 31.73 的重建消融用 ~150k–300k iter。
- 本 run 算账:19609 micro-batch/epoch ÷ grad_accum 4 = **4902 优化器step/epoch**;
  **60 epoch × 4902 ≈ 294k 优化器step ≈ 论文 300k**(精确对齐要 61.2 epoch,60 是取整)。
- ⚠️ **时间尺度:~0.39 opt-step/s → 60 epoch(294k)≈ 9 天**(多天级长跑,TRAINING_LOG §58 已记)。
- 推论:60 是"跑满论文预算"的上限;**不一定要跑满**——Table-1 在 150k(≈30 epoch)就出好结果,
  应靠周期 eval 看何时 plateau 提前停,而非盲跑 60。

## ⚠️ eval 现状:当前【没有】留出集 eval(训练在"盲跑")
- cfg `eval_epoch: 100000000` = **周期 eval 关闭**;原因:test_dataset 无真实 csv(ucf101_val 空)+
  FVD i3d 模型缺失(FVDCalculator init failed)。
- trainer **有** eval 机制(uvt_tokenizer_trainer.evaluate_step:412 + PSNR/SSIM/FVD),只是没接 val 集。
- 现在只有进度条的**训练集即时 PSNR**,没有 held-out 的 PSNR/SSIM/rFID → 无法判断泛化/何时停。
- 可用素材:ImageNet `test-*.parquet`(28分片,可作留出集)+ `eval/recon_metrics.py`/`protocols.py`。
- **待决策(见下)**:补周期 eval 的方式(改配重启接 val / 离线评 checkpoint),及 cadence(每 epoch / 每 5 epoch)。




