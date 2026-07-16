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

