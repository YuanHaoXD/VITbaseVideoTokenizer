# UVT 训练日志(TRAINING_LOG)· 冲击 Hydra-X 指标

> 本文件是**全量训练攻坚的指标账 + 问题/方案记录**,与 `WORKLOG.md`(总进度)分工:这里只记训练指标曲线、观察到的问题、尝试的解决方案、消融结论。每次 /loop 唤醒必更新。

## 🎯 目标(Hydra-X 论文 "Ours" 行,越接近越好)

| 数据集 | PSNR↑ | SSIM↑ | rFID↓ / rFVD↓ |
|---|---|---|---|
| **ImageNet** | **31.73** | **0.8936** | rFID **0.329** |
| **DAVIS** | **27.97** | **0.8307** | rFVD **11.19** |

对照(同论文其它注意力):Full 31.10/0.8890/0.367;Causal 31.38/0.8901/0.352;Tubelet 31.42/0.8907/0.347。
我们的模型用 tubelet(`attn_mode: tubelet`),对标 Tubelet→Ours 区间。

## 📊 指标进展(最新在上)

| 日期 | run | 步/epoch | 数据 | ImageNet PSNR | SSIM | rFID | 备注 |
|---|---|---|---|---|---|---|---|
| 2026-07-15 | full-02 | 19609 | 全1.28M | ~9.5(step411,warmup) | 待评测 | 待评测 | 主线;epoch未完 |
| 2026-07-15 | val8(单epoch) | 272 | 2 shards | ~8.5(500步早期) | — | — | 仅打通验证,非收敛值 |

> 说明:上面是"训练路径验证"的早期值,远未收敛。真正的攻坚从 run-full-02 开始。**PSNR/SSIM/rFID 的真实评测由子agent B 搭的 `eval_metrics.py` 在 checkpoint 上跑**(训练中的 psnr 是训练时估计,非正式指标)。

## 🧪 运行记录

### run-full-02(2026-07-15 启动)· 真·全量 1.28M Stage 1 ★当前主线
- **配置**:`cfgs/uvt_stage1_imagenet_full_npu.yaml`,8 卡,真 so400m(889M),bf16,tubelet,64 维 latent。
- **数据**:ImageNet parquet **全 294 片≈1.28M**,rank 分片(每 rank ~16 万图 in_memory,rank0 len=161236)+ pre_sharded。内存 238G/1509G。
- **规模**:**19609 步/epoch**(=跨rank MIN 对齐;每 rank 训满自己 ~16万、8卡并集=全1.28M/epoch),bs8,max_epoch=60,~4.4h/epoch。
- **状态**:进行中。step10 psnr=2.42,warmup 起步。**这是冲指标的主线 run。**

### run-full-01(2026-07-15,已停)· 40 片子集 bootstrap
- 174320 图子集,psnr 2.87→~9.0(step310,warmup平台),仅为先跑起来;全量加载器就绪后停,切 run-full-02。

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
- **研究agent(通向31)**:分析语义损失是否拖累 PSNR / 64维latent天花板 / 是否必须上Stage2 / lr-bs-warmup,产出 run-full-03 协调配置。
