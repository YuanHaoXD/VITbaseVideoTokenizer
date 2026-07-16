# UVT 训练日志(TRAINING_LOG)· 冲击 Hydra-X 指标

> 本文件是**全量训练攻坚的指标账 + 问题/方案记录**,与 `WORKLOG.md`(总进度)分工:这里只记训练指标曲线、观察到的问题、尝试的解决方案、消融结论。每次 /loop 唤醒必更新。

## 🎯 目标(Hydra-X 论文 "Ours" 行,越接近越好)

| 数据集 | PSNR↑ | SSIM↑ | rFID↓ / rFVD↓ |
|---|---|---|---|
| **ImageNet** | **31.73** | **0.8936** | rFID **0.329** |
| **DAVIS** | **27.97** | **0.8307** | rFVD **11.19** |

对照(同论文其它注意力):Full 31.10/0.8890/0.367;Causal 31.38/0.8901/0.352;Tubelet 31.42/0.8907/0.347。
我们的模型用 tubelet(`attn_mode: tubelet`),对标 Tubelet→Ours 区间。

### 📌 关键情报(研究agent 2026-07-15,据 docs/background/Hydra-X)——决定攻坚路线
- **31.73 是 Table-1 架构消融最优行(Tubelet+层级patchify),训练协议=ImageNet-1k、batch 256、~150k–300k iter,损失=L1+LPIPS+KL。不是三阶段旗舰、不是 GAN 堆出来的。**
- 推论:**PSNR 31 档靠"架构+L1+LPIPS+KL"Stage-1 就能到**(四变体 PSNR 全 31.1–31.7);**rFID 0.329 几乎必须 Stage-2 GAN**(纯 L1+LPIPS 的 rFID 约 0.5–0.8 档)。架构主要买 rFID/rFVD 不是 PSNR。
- 我方 **C=64 与 Hydra-X 一致**,通道非瓶颈;**latent 宽度是禁区**(C≥256 毁 DiT 生成)。重建杠杆在 decoder 容量,不在 bottleneck。
- **诊断出的头号配置错配**:全局 batch。旧 run(full-01/02)= 8×8×accum1 = **64**,而 Hydra-X = **256** → lr 2e-4 是为 256 定的,**旧 run 有效 lr 偏高 4×**(warmup 期噪声大)。

## 📊 指标进展(最新在上)

| 日期 | run | 步/epoch | 数据 | ImageNet PSNR | SSIM | rFID | 备注 |
|---|---|---|---|---|---|---|---|
| 2026-07-16 | **full-04** | 19609(accum4) | 全1.28M | 训练中(从头) | 待评测 | 待评测 | ★主线;=full-03同配置,修 ckpt 崩溃后重启 |
| 2026-07-15 | full-03 | 19609(accum4) | 全1.28M | 19.1@epoch1末 → **崩** | — | — | epoch-末 save ckpt 裸cuda崩溃,0 ckpt(已修) |
| 2026-07-15 | full-02 | 19609 | 全1.28M | ~9.5(step411,warmup) | 待评测 | 待评测 | 已停(batch错配) |
| 2026-07-15 | val8(单epoch) | 272 | 2 shards | ~8.5(500步早期) | — | — | 仅打通验证,非收敛值 |

> 说明:上面是"训练路径验证"的早期值,远未收敛。真正的攻坚从 run-full-02 开始。**PSNR/SSIM/rFID 的真实评测由子agent B 搭的 `eval_metrics.py` 在 checkpoint 上跑**(训练中的 psnr 是训练时估计,非正式指标)。

## 🧪 运行记录

### run-full-04(2026-07-16 启动)· ★当前主线 · = full-03 同配置,修 ckpt 崩溃后重启
- **为何重启**:full-03 跑完 epoch-1(PSNR ~19.1，单调无平台)在 epoch 边界 `save_checkpoint('epoch-last.pth')` **全 8 卡崩溃**——`utils/common.py:gather_object_from_all` 两处裸 cuda(NPU 端口遗漏），**0 checkpoint 落盘**。已修（commit `22a7d1f`，真机 2 卡 HCCL 验证崩溃路径闭合），WORKLOG 2026-07-16（晚）有全貌。
- **配置 identical**:`cfgs/uvt_stage1_imagenet_full_npu.yaml`，accum4(全局batch256)/lpips0.5/cos0.5/lr2e-4/max_epoch60。**paired-baseline 纪律，不趁重启改 bs/lr**（避免多变量混淆）。out_path `.../full04`（保留 full03 log/tensorboard）。
- **从头训**（无 ckpt 可 resume，SigLIP2 init）。**首个验收点**:epoch-1 末能否干净存出 `epoch-last.pth`（本 fix 的验收）；落盘后即可跑 `scripts/eval_metrics.py --ckpt` 出真实 6 指标。
- **预期同 full-03**:Stage-1 收敛 PSNR 29.5–31.5 / rFID ~0.5–0.8；+S2 GAN 压 rFID ~0.33。~0.35 opt-step/s → 60 epoch ≈ 多天级长跑。

### run-full-03(2026-07-15 启动)· 已崩(epoch-末 ckpt 保存 bug)· 见 full-04
- **依据**:研究agent 分析(见上"关键情报")。相对 full-02 三处协调改动:
  - `grad_accumulates: 1→4` → **全局 batch 64→256=Hydra-X**;一举对齐 lr(2e-4 变合理)、warmup(≈5k opt-step)、总量(60ep≈294k opt-step≈Hydra-X S1)。**最高优先级、决定成败。**
  - `lpips_weight: 1.0→0.5`(HYDRA 参考 λ_perc=0.1,原 1.0 是 10×,压 PSNR)——单点 ROI 最高的 PSNR 杠杆,预期 +0.3~0.8dB(rFID 略升,靠 S2 GAN 补)。
  - `cos_weight: 1.0→0.5`(降共享 z 语义扰动,S2 本关,低风险)。
  - **不动**:lambda_dist=0.5(已是 Hydra-X 一半,勿砍语义卖点)、kl=1e-6、distill=1.0、C=64、lr=2e-4、max_epoch=60、bs=8。
- **预期(诚实区间)**:Stage-1 收敛 PSNR **29.5–31.5** / rFID ~0.5–0.8;+ Stage-2 GAN 守 PSNR、rFID 压到 ~0.33。
- **状态**:刚启动。⚠️ **accum4 下进度条 19609 是 micro-batch,优化步=其 1/4(~4902)**;同 micro-step PSNR 比 full-02 低是正常的(权重更新少),按 opt-step 看。step135(≈34 opt-step)psnr=6.1。
- **早期 opt-step 轨迹**(2026-07-15,warmup 内):opt25→600 = psnr 5.45→7.16→8.02→8.33→8.78→8.87(opt175)→9.01(200)→9.60(400)→9.88(600),单调上行。**吞吐 1.56→1.43 it/s ≈3.5h/epoch**。
- **⏱️ 时间尺度**:~0.35 opt-step/s → Hydra-X 规模 150k opt-step ≈ **5 天**,60 epoch(~294k)≈ **9 天**。这是多天级长跑。
- **加速爬升**(2026-07-15 ~1h):opt375→1500 = psnr 9.54→10.07→11.89→**13.33**(warmup 内,lr_m 0.341)。**曲线加速无平台**,健康。
- **续爬**(~3h,epoch-1 将完成):opt2000→4000 = psnr 14.72→15.83→16.75→17.53→**18.22**(lr_m 0.825,warmup 尽)。**单调无平台**,epoch-1 内已到 18+。首个 checkpoint 出后跑真实评测校准。
- **🔭 监控重构(2026-07-15)**:20min 定时巡检对多天长跑浪费 → 改为 **事件驱动看门狗(epoch完成/崩溃立即通知)+ 2h 稀疏定时巡检(cron fc52b07d)**。首个真实指标在 epoch-1 checkpoint(~3.5h)后由 eval_metrics 出。

### run-full-02(2026-07-15,已停)· 全量加载器打通
- 真·全量 1.28M、rank分片+pre_sharded、steps/epoch=19609 验证通过。因全局 batch=64 配置错配(有效 lr 4×),启动 ~15min 即停,切 run-full-03。

### run-full-01(2026-07-15,已停)· 40 片子集 bootstrap
- 174320 图子集,psnr 2.87→~9.0(warmup),仅为先跑起来。

## ⚠️ 问题 & 尝试方案(最新在上)

### P0(开局)· 全量数据加载的跨片洗牌抖动
- **问题**:parquet 直读 + DistributedSampler 全排列洗牌 → 单分片 LRU 缓存被反复换出,全 294 分片时 IO 抖动严重(见 `parquet_image_dataset.py` docstring)。
- **✅ 已解决(run-full-02)**:rank 分片(每 rank 只 in_memory 自己 `files[rank::world]` ~37 片)+ JointLoader `pre_sharded`(跳过 DistributedSampler)。全 1.28M 无冗余(~128GB)无抖动。子agent A 实现,我复核修了下面两个真机 bug。

### P0b · 全量加载器的两个"隔离测试测不出"的真机 bug(已修)
- **bug1 · all_reduce CPU tensor**:JointLoader 步数对齐用 `dist.all_reduce` 但 tensor 在 CPU → **HCCL 不支持 CPU tensor,真 8 卡会挂**。子agent只 CPU 模拟测过。**修**:tensor 放 `accel.device()`(uvt 无 accel 退 cuda),双仓字节一致。
- **bug2 · pre_sharded 未转发**:`trainers/uvt_tokenizer_trainer.make_datasets` 的 `sources.append` 没把 cfg 的 `pre_sharded` 传给 JointLoader → DistributedSampler 仍叠加 → **二次分片(steps/epoch 掉到 2519=161236/64,每卡只训 1/8 数据)**。运行时用 steps/epoch 抓出。**修**:append 补 `pre_sharded`,双仓。修后 steps/epoch=19609 ✓。
- **教训**:子agent 的隔离单测(CPU 模拟 rank、直接构造 JointLoader)测不出①设备后端 ②真实 trainer 装配链路 的 bug。**新数据/DDP 路径必须真机 8 卡冒烟 + 核对 steps/epoch 数值**才算数。

### P0(开局)· DAVIS/rFVD/rFID 评测缺口
- **问题**:无 DAVIS 数据集、无 I3D 权重;ImageNet rFID 与 DAVIS/rFVD 尚不能算。
- **方案(子agent B 开发中)**:下载 DAVIS-2017 + I3D;把 `eval/recon_metrics.py` 串成可跑的 PSNR/SSIM/rFID(ImageNet val)+ PSNR/SSIM/rFVD(DAVIS)脚本 `scripts/eval_metrics.py`。

### P1(开局)· 吞吐偏低(~1.24 it/s / 8卡79 img/s)
- **问题**:8×910B2 上 bs=8 仅 ~79 img/s,~4.4 小时/epoch(全量 19609 步),冲高 PSNR 的迭代周期长。疑因:教师 so400m 额外前向 + 未开 torch.compile(NPU inductor→triton 脆弱,当前关) + bs 偏小。
- **实测 HBM**:bs=8 每卡 **41.7/65.5GB(~64%)**,余量 ~24GB → **bs 可加大(~12)**。AICore 利用率波动(部分时刻不饱和,疑数据/同步间隙)。
- **待协调改动(研究agent分析中)**:bs↑ 需 lr 同步 scale;连同损失配比一起在 run-full-03 一次性协调改,避免多变量混淆。

## 🔬 进行中的研究/实验(子agent)
- **子agent B**:评测台 `eval_metrics.py`(273行)+ I3D 权重已下、DAVIS 下载中。就绪后可在 checkpoint 上出真实 6 指标。
- **研究agent(通向31)**:✅ 已交付。结论落地为 run-full-03(见上)。

## 🗺️ 消融/实验 roadmap(研究agent建议,每项 paired 同 seed/data-order)
1. **run-full-03 基线**(accum4+lpips0.5+cos0.5)——先确认能否进 30+ 档。★进行中
2. LPIPS 扫 {0.25, 0.5, 1.0} paired —— 定 PSNR-vs-rFID 折中点。
3. distill {off, 0.25, 0.5} paired —— **量化 Q1**(语义拖累几 dB,用数据替代猜测)。
4. KL {1e-6, 1e-5} —— 确认 1e-6 最优。
5. Stage-1 达 ~31 后 → **Stage-2 GAN**(decoder-only,守 PSNR、攻 rFID 0.33)。
> 算力约束:8 卡=同时仅 1 个全量 run,收敛 ~数天/run → 消融只能少数几个 + 或用早期轨迹对比(不够定论但有信息)。优先把 run-full-03 跑到收敛出真实指标,再决定下一步。
