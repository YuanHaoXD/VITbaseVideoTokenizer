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
| 2026-07-15 | val8(单epoch) | 272 | 2 shards | ~8.5(500步早期) | — | — | 仅打通验证,非收敛值 |

> 说明:上面是"训练路径验证"的早期值,远未收敛。真正的攻坚从下面 run-full-01 开始。

## 🧪 运行记录

### run-full-01(2026-07-15 启动)· 大规模 Stage 1 起步
- **配置**:`cfgs/uvt_stage1_imagenet_full_npu.yaml`,8 卡,真 so400m(889M),bf16,tubelet,64 维 latent。
- **数据**:ImageNet parquet,`max_shards=40`(**len=174320**,in_memory,内存占用 261G/1509G)。⚠️ 非全 1.28M——真·全量加载器(rank 分片)由子agent A 并行开发中,就绪后 run-full-02 切全量。
- **规模**:2723 步/epoch(174320/8卡/bs8),max_epoch=60,存盘到持久盘 `yh222/uvt_runs/full01`(latest 每 epoch、full 每 5 epoch)。
- **吞吐**:~**1.24 it/s**(8卡~79 img/s),≈37 分钟/epoch。⚠️ 偏低——见问题 P1。
- **早期曲线**:step1 psnr=2.87 → step144 psnr=8.50(与验证 run 一致);仍在 warmup(lr_m 刚到 0.1)。**远未收敛,曲线随 epoch 滚动更新。**
- **状态**:进行中(PID 首轮启动)。下个 /loop 唤醒读日志记 epoch-1 收敛 PSNR。

### run-08-val(2026-07-15)· 8 卡打通验证(历史,非攻坚)
- 272 步、PSNR 2.60→8.54(单epoch早期)、HCCL 与单卡一致。仅证路径健康。

## ⚠️ 问题 & 尝试方案(最新在上)

### P0(开局)· 全量数据加载的跨片洗牌抖动
- **问题**:parquet 直读 + DistributedSampler 全排列洗牌 → 单分片 LRU 缓存被反复换出,全 294 分片时 IO 抖动严重(见 `parquet_image_dataset.py` docstring)。
- **现状方案(临时)**:run-full-01 用 `max_shards=80` + `in_memory=true`(80×~436MB=35GB/rank×8≈280GB,内存 1282GB 富余),规避抖动、先跑起来。
- **根治方案(子agent A 开发中)**:rank 分片——每 rank 只持有 `files[rank::world]` 并跳过 DistributedSampler,in_memory 全量 1.28M 无冗余(~128GB 总),无抖动。

### P0(开局)· DAVIS/rFVD/rFID 评测缺口
- **问题**:无 DAVIS 数据集、无 I3D 权重;ImageNet rFID 与 DAVIS/rFVD 尚不能算。
- **方案(子agent B 开发中)**:下载 DAVIS-2017 + I3D;把 `eval/recon_metrics.py` 串成可跑的 PSNR/SSIM/rFID(ImageNet val)+ PSNR/SSIM/rFVD(DAVIS)脚本 `scripts/eval_metrics.py`。

### P1(开局)· 吞吐偏低(~1.24 it/s / 8卡79 img/s)
- **问题**:8×910B2 上 bs=8 仅 ~79 img/s,~37 分钟/epoch,冲高 PSNR 的迭代周期长。疑因:教师 so400m 额外前向 + 未开 torch.compile(NPU inductor→triton 脆弱,当前关) + bs 偏小(64G HBM 未吃满)。
- **待尝试**:①增大 bs(试 16/卡,观察 HBM 与吞吐);②教师前向 `@torch.no_grad` + 是否可缓存(教师对同图确定,但增广随机,难缓存);③评估 NPU 上开 compile 的可行性(风险高,单独试)。留待后续 fire / 子agent。
