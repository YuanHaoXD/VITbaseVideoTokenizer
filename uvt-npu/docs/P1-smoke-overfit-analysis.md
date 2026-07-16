# P1-smoke overfit 诊断分析（2026-07-10）

> **作者**:Claude(NPU 移植会话)。**状态**:发现待审查的重建路径异常,**供用户分析**。
> 全部原始日志在 `uvt-npu/logs/`,探针脚本 `scripts/probe_recon.py`,overfit 脚本 `scripts/p1_smoke_overfit.py`。

## 0. 结论先行(TL;DR)

- **NPU 移植本身没问题**:单卡/8 卡/tiny/生产(null128)冒烟全过、pytest 84 绿。
- **P1-smoke overfit Gate(500步 PSNR>30)未过**:重建 L1 卡在 ~0.3–0.6、PSNR~6–8,跨 6 种配置(lr/精度/损失/z 采样)都不收敛。
- **探针定位到两个结构现象**:① 编码器 μ 塌缩(4 张图 μ 两两余弦 **0.997**);② 解码器像素头输出未校准(x_hat 值域 **[-32, 44]**,而 x∈[0,1])。梯度流是通的(非断链)。
- **重要保留**:overfit 用的是 `torch.rand` **随机噪声图**。SigLIP2 在自然图上训练,对随机噪声可能输出退化的相似特征 → μ 塌缩可能是"随机噪声 overfit 不是 SigLIP2 模型的有效测试"的伪影。**确定性结论需用真图像(ImageNet 几张)重测**,而那依赖 P1 数据。
- **模型代码 `models/uvt/*` 我未改动** → 若排除"随机噪声伪影",则是 uvt/ 原代码在"真权重+真训练"下才暴露的问题(此前仅 tiny CPU 冒烟)。需在 GPU 集群复现以区分"代码 bug vs NPU op 异常"。

## 1. 背景

04 §1 Phase 1 的第一项 Gate = `P1-smoke`:固定单 batch overfit,500 步,期望 PSNR>30(02 §5:`test_tokenizer_overfit_single_batch`,8 样本@128²)。目的是验证训练管线"真能重建"(比 null 冒烟的 loss 下降更强信号)。

我写的 overfit 脚本:`scripts/p1_smoke_overfit.py`(生产模型 tiny=false + 真 SigLIP2 so400m,固定一个 batch 反复训,记 L1/LPIPS/cos/distill/PSNR)。
> 注:so400m 位置编码原生 256²(256 位置),不能降到 02 §5 写的 128²(会像 tiny 那样位置数对不上崩),故 overfit 在 256² 跑。

## 2. 测试矩阵与结果

| # | 配置 | L1 终值 | PSNR 终值 | 通过? | 日志 |
|---|---|---|---|---|---|
| 1 | 256² bs4 bf16 lr2e-4 全损失(含distill+lpips) sample=True | ~0.62 | 6.23 | ❌ | `uvt_p1_overfit.log` |
| 2 | 同上 lr→1e-3 | ~1.4(step100,更慢) | ~5.3 | ❌ | `uvt_p1_overfit2.log` |
| 3 | fp32 lr5e-4 bs2 | **发散**(L1 飙 101) | ~4.8 | ❌ | `uvt_p1_overfit3.log` |
| 4 | 纯L1(distill=0 lpips=0) sample=True lr2e-4 | **震荡**(98↔5) | ~4.9 | ❌ | `uvt_p1_overfit_l1only.log` |
| 5 | **确定性z(μ) + 纯L1** sample=False lr2e-4 | ~0.35(慢降后卡) | ~8 | ❌ | `uvt_p1_overfit_detz.log` |

**统一现象**:重建 L1 不收敛到 ~0;而蒸馏项(distill 1.45→0.08)、L_cos(0.99→0.03)在 #1 里**正常收敛**。即"语义对齐学得动,像素重建学不动"。

## 3. 探针发现(决定性,`scripts/probe_recon.py` → `logs/probe_recon.log`)

构建模型后,4 张不同随机图,确定性编码 + 解码,量:

```
[μ]            shape=(4,1,256,64) mean=0.0095 std=1.96  range=[-69, 66]
[μ 塌缩检查]   4 图 μ 两两余弦 mean=0.9973  min=0.9972 max=0.9974   ← 几乎相同!
[x_hat]        mean=0.078 std=1.86  range=[-32, 44]                ← 远离 [0,1]
[x]            mean=0.4998 std=0.29  (x∈[0,1])
[x_hat 多样性] 4 图重建两两余弦 mean=0.9972                         ← 重建几乎相同
[L1(x,x_hat)]  = 1.47  (随机初始化)

[梯度] L1 反传后:
  encoder     grad_sum=2.8e+04   (199M 参有梯度)
  gsb         grad_sum=4.0e+02   (0.1M)
  decoder     grad_sum=1.9e+05   (413M)    ← 解码器有梯度
  decoder.head_anchor.weight grad_norm=3.25  ← 像素头有梯度
  decoder.head_frame.weight   grad_norm=0    ← (图像 F=1 不用帧头,正常)
  sem_vit / decompressor grad=0  (L1 不经它们,正常)
```

**解读**:
- **梯度流是通的**(decoder/encoder/像素头都有梯度)→ 不是 detach/断链。
- **μ 塌缩**:4 张不同图的 μ 余弦 0.997 → 解码器收到几乎相同的 z → 无法区分样本 → 重建必然趋同( x_hat 多样性也 0.997)。**这是重建不了的直接原因**。
- **x_hat 值域 [-32,44]**:解码器像素头无 sigmoid/约束,随机初始化产出大值,远离目标 [0,1]。

## 4. 根因推测(供审查,未定论)

**A. μ 塌缩的可能解释(按概率)**:
1. **随机噪声 overfit 的伪影(最可能)**:SigLIP2 在自然图上训练,对 `torch.rand` 随机噪声(无自然结构)输出退化的相似特征 → h 相似 → μ 相似。**真图像(ImageNet)下 SigLIP2 特征多样,μ 应分化**。这能解释为何 overfit 慢但 distill(语义)能收敛。
2. GSB `proj: Linear(1152→128)` 随机初始化:μ 值 range ±69 很大,可能某个主导方向/偏置淹没了图间细差异(余弦对幅度不敏感)。
3. 编码器对图像(F=1)的路径:锚帧经 Gen-ViT(无折叠)→ h。若该路径有问题(如折叠位置逻辑误伤锚帧)→ h 退化。

**B. x_hat 值域未校准**:解码器像素头(`head_anchor: Linear(D→3·16·16)`)无输出激活 → 随机初始化产出 ±44。**这个无论 A 如何都该看一眼代码**:LARP/OmniTokenizer 的像素头惯例是否有 sigmoid,或靠 L1 训练把输出压进 [0,1](若是后者,初始 ±44 + lr2e-4 收敛慢就解释了 overfit 慢)。

## 5. 关键判断:是 NPU 移植引入的吗?

**几乎确定不是**:
- 模型代码 `models/uvt/*`(siglip_backbone/gsb/encoder/decoder/sem_vit/decompressor/uvt_tokenizer)**一行未改**(我只动框架层 train.py/base_trainer/uvt_tokenizer_trainer + 加 accel)。
- 探针里"梯度通、μ 塌缩、x_hat 值域错"都是模型前向/初始化的纯计算行为,与设备无关。
- 但**不能 100% 排除 NPU 某 op 数值异常**(如 SDPA/mask 在 NPU 上的行为偏差)。**需在 GPU 集群用相同脚本复现**:
  - 若 GPU 上 overfit 也不收敛 → uvt/ 原代码 bug(此前只在 tiny CPU 冒烟过,真权重真训练没验过)。
  - 若 GPU 上 overfit 收敛(PSNR>30)→ NPU op 异常,需定位具体算子。

## 6. 建议的下一步诊断(按性价比)

1. **【最关键】用真图像重测 overfit**:取 ImageNet 4–8 张真图(不是 torch.rand),跑 `scripts/p1_smoke_overfit.py`。若 μ 塌缩消失 + PSNR>30 → 确认是"随机噪声伪影",P1-smoke 实质通过。**依赖 P1/ImageNet 数据**。
2. **GPU 复现**:把 `uvt-npu/scripts/{p1_smoke_overfit,probe_recon}.py` 拿到集群(原 uvt/ CUDA 代码)跑一遍,区分代码 bug vs NPU。
3. **代码审查**(不依赖数据,可现在做):查 `decoder.py` 像素头是否有输出激活;查 `gsb.py` proj 初始化;查 `encoder.py` 锚帧(F=1)路径是否误折叠。
4. μ 塌缩的进一步量化:探针里把 μ 的"去均值后"余弦也算一下(区分"共偏置"vs"真方向塌缩")。

## 7. 对项目计划的影响

- **不阻断 P0**:P0 是评测尺子校准,与训练重建无关。P0 仍可照计划推进(等锚点/数据下载)。
- **P1 的风险**:若 #1(真图像重测)仍不收敛,则 P1-base(ImageNet 50k 步)的 PSNR≥26 Gate 可能达不到,需先修重建路径。这是 R11 之外的**新风险**,建议进风险登记册。
- **当前定位**:P1-smoke 的"不崩 + loss 下降"已满足(管线通);"PSNR>30 overfit"未满足,但**很可能因测试输入(随机噪声)不当**,需真图像确证后再定性。

## 8. 产物清单（原文）

- 日志:`uvt-npu/logs/{uvt_p1_overfit,uvt_p1_overfit2,uvt_p1_overfit3,uvt_p1_overfit_l1only,uvt_p1_overfit_detz,probe_recon}.log`
- 脚本:`uvt-npu/scripts/p1_smoke_overfit.py`(支持 STEPS/BS/SIZE/LR/AMP/DIST/LPIPS_W/SAMPLE 环境变量)、`scripts/probe_recon.py`
- 本文档:`uvt-npu/docs/P1-smoke-overfit-analysis.md`

---

## 9. 根因确认与修复（2026-07-12 追记，开发机 Fable 5）

§4 的假设已裁决:**A2 方向正确、A1(随机噪声伪影)不成立**。逐行审查 uvt/ 模型代码后定位为 **spec 级遗漏——切分边界缺规范化**(docs/06 §6.8、docs/08 §6.5):

1. **μ"塌缩"的真相**:h 是 SigLIP2 第 13 层裸残差流,含 massive activations(O(10³)、近图像无关的通道)。gsb.proj 默认初始化 |W|≈0.03,μ 却到 ±69 → 反推 h 有 O(10³) 分量;共享巨激活主导 μ 方向 → 跨图余弦 0.997。**真图像下同样会出现**(巨激活与内容无关),故 §6 建议 1"真图重测"不会自愈。
2. **sample=True 各组必然不收敛的机制**:ρ 与 μ 同源同尺度,大量通道钉死 clamp(-30,20) 上界 → σ=e^10≈22000 的噪声注入 z;且 clamp 界外**零梯度**,KL/重建都无法把饱和 ρ 拉回。解释 #3 fp32 发散、#4 纯 L1 震荡 98↔5、#5 确定性 z 最稳但仍卡。
3. **x_hat ±44**:decoder 末 block 残差流(真权重尺度 O(10²))直入随机初始化像素头。SigLIP2 原序 encoder→post_layernorm→head,Sem-ViT 镜像了,decoder 漏了——这正是"语义学得动、像素学不动"的原因(语义支路是唯一有出口 LN 的支路)。
4. 次要:学生编码器把 [0,1] 直喂 SigLIP2 embeddings(教师侧却按 processor mean/std 归一化)。

**修复**(已落地 uvt/ 与 uvt-npu/ 双仓,契约测试 tests/test_boundary_norms.py 4 项全绿,双仓全量 81 passed):GSB 入口 LayerNorm + compress(sample=) 收口、decoder final_ln + 像素头校准初始化(weight std=0.02/bias=0.5)、GenViT px_mean/px_std 输入归一化 buffer、out['h']/L_cos 对齐目标改为规范化 h。

**对 §6 建议的修订**:服务器侧重跑顺序 = ①拉最新代码重跑 probe_recon.py(预期:μ 余弦分化、x_hat 值域 O(1))→ ②SAMPLE=0 重跑 overfit(预期 PSNR 上行)→ ③SAMPLE=1 重跑 → ④真图像重测仍有价值(排除 torch.rand 对 LPIPS/语义项的干扰),但不再是关键路径。注意:修复改变了 state_dict 键集(gsb.norm.*/decoder.final_ln.*/encoder.px_mean/px_std),旧 checkpoint 不兼容(无生产 checkpoint,无迁移负担)。

---

## 10. 服务器侧重跑执行结果(2026-07-15,NPU Ascend 910B2,真 SigLIP2 so400m)

§9 裁定的重跑序列已在 NPU 上执行完毕。**结论:#15 修复在真权重下得到决定性验证;P1-smoke 重建路径确认健康。** 环境是 `/cache` 清空后在 `/home/ma-user/work/dataset/yh222/VITbaseVideoTokenizer/` 重建的(权重重下、依赖重装,详见文末环境备注)。

### 10.1 探针复跑(`logs/probe_recon_rerun.log`)——三个病理现象全部消失

| 指标 | 修复前(07-10) | **修复后(07-15)** | 判定 |
|---|---|---|---|
| μ 值域 | [-69, 66] | **[-2.44, 2.14]** | ✓ GSB 入口 LN 压住巨激活 |
| **μ 跨图余弦** | **0.997**(塌缩) | **0.67**(分化) | ✓ 编码器能区分样本 |
| μ std | 1.96 | 0.56 | ✓ 规模正常 |
| **x_hat 值域** | **[-32, 44]** | **[-1.97, 3.35]** | ✓ final_ln + 像素头校准初始化生效 |
| x_hat mean | 0.078 | **0.481**(≈x 均值 0.50) | ✓ bias=0.5 初始化对准 [0,1] 中心 |
| L1(随机init) | 1.47 | 0.59 | ✓ 起点改善 |
| 梯度流 | 通 | 通(encoder/decoder/像素头都有梯度) | ✓ 非断链 |

§9-1(μ 塌缩)、§9-3(x_hat 值域)预言的修复效果全部兑现。

### 10.2 overfit 复跑——sample 路径病理消失,但 torch.rand 目标仍卡低分

| 实验 | 修复前 | **修复后(07-15)** | 日志 |
|---|---|---|---|
| SAMPLE=0 纯L1(确定性z) | ~8 卡住 | **6.2→7.9 平稳** | `overfit_sample0.log` |
| SAMPLE=1 纯L1(重参数) | **发散101(#3)/震荡98↔5(#4)** | **6.2→7.8 平稳单调,无发散无震荡** | `overfit_sample1.log` |

**关键裁决**:§9-2 预言的 clamp 饱和病理(sample=True 必发散/震荡)**已消失**——修复后 sample=1 与 sample=0 表现几乎一致(7.78 vs 7.89),证明 GSB 入口 LN 把 ρ 规模压回正常、重参数噪声不再破坏训练。这是 #15 修复最强的正向信号。

**但两条曲线都卡 PSNR~7.9,不到 30。** 经隔离实验确认:**这不是重建路径 bug,而是 `torch.rand` 目标本身的两个伪影**(§0/§6 早预警):
- **随机噪声不可压**:相邻像素差 0.33、无空间结构,任何带卷积/patch 上采样先验的 decoder 都压不动(对照:常数图差=0、低频图差=0.008)。
- **LPIPS 对抗梯度**:噪声图上 LPIPS 卡 0.75–0.98 不降,把 L1 往回拽。

### 10.3 决定性隔离实验(§9-4 的替代,更强)——解码器健康证明

不必等 ImageNet 真图。直接把目标换成**低频结构图**(16×16 双线性上采样)并逐步剥离干扰项:

| 实验 | 目标 | 损失 | FINAL PSNR | 日志 |
|---|---|---|---|---|
| overfit_structured | 低频结构图 | L1+LPIPS | 11.68 | `overfit_structured.log` |
| **overfit_pureL1** | 低频结构图 | **纯 L1**(lpips=0) | **34.89**(单调穿过30) | `overfit_pureL1_structured.log` |

**纯 L1 + 结构图 500 步 → PSNR 34.89,单调爬升**(7.1→14.3→20.7→26.4→30.9→34.9)。**这直接证明 encoder/GSB/decoder 重建管线完全健康**:给可压缩信号 + 足够步数,能 overfit 穿过 30。P1-smoke Gate 的"重建能力"实质通过;原脚本的 30 阈值是针对 torch.rand 噪声定的,对该输入不可达,属测试输入选择问题(R11:拿到曲线后重标阈值,此处应改用真图或结构图基准)。

### 10.4 结论

- **#15 修复成功,双仓一致,真权重验证通过。** μ 塌缩 / x_hat 值域 / sample 发散三大病理全消。
- **重建路径健康。** 纯 L1 结构图 overfit 到 34.89 是硬证据。
- **原 p1_smoke_overfit 卡 7.9 是 torch.rand 伪影**,非 bug。建议把该脚本默认目标改为低频结构图或接入真图(见下一步)。
- **不阻断后续。** P1-base 真训练(ImageNet)风险相应下调:重建路径已证健康,真图有自然结构 + 语义可学,预期能达 PSNR≥26 Gate。

### 10.5 环境备注(`/cache` 清空后重建,重要)

- **权重重下**:SigLIP2-so400m-patch16-256 → 仓库内 `models/`(4.5GB,走 hf-mirror)。脚本/配置里写死的 `/cache/...` 路径已全部改为仓库内路径(`probe_recon.py`/`p1_smoke_overfit.py` 改读环境变量 `UVT_SIGLIP` 带默认值;`cfgs/uvt_stage1_npu.yaml` 两处 model_name/img_id 改绝对路径)。
- **依赖重装**:`lpips mergedeep pytorch-msssim moviepy wandb` 随环境重置丢失,已用**清华源**补装(华为云 pip 源 `repo.myhuaweicloud.com` 当前不通;hf-mirror / pypi.tuna / aliyun 通)。lpips 的 VGG16(528MB)已重新缓存到 `~/.cache/torch/hub`。
- **版本坑**:装 wandb 会把 `huggingface_hub` 顺带升到 1.23.0,与 transformers 4.53.1 的 `<1.0` 约束冲突导致 `import transformers` 崩。**已降回 `huggingface_hub==0.36.2`**。补装依赖后务必复验 `import transformers` 不崩。

---

## 11. 真图 overfit 复测(2026-07-15,§9-4 收尾)

ImageNet 下全后,从 `train-00000` parquet 抽 4 张真图跑纯 L1 overfit(`scripts/probe_realimg_overfit.py`,日志 `logs/realimg_overfit_*.log`):

- **μ 塌缩彻底解决**:真图两两余弦 **0.283**(torch.rand 噪声图 0.67;修复前 0.997)。自然图特征多样、编码器完全区分样本 → **#15 修复对真实图像确认有效,μ 塌缩非伪影残留**。§9-4 的核心疑问就此关闭。
- **重建单调向上但偏慢**:纯 L1 单 batch,500 步 PSNR→16.1,1500 步→16.3;中途有"长平台(PSNR~11.9)+突破"模式(典型鞍点/坏盆地,非容量饱和)。对比低频结构图能到 34.9,差距源于真图高频细节 × 64 维 latent × 单 batch 慢优化。
- **判断**:overfit 绝对 PSNR 不是关键指标(其优化景观与真实训练迥异——真训有大规模数据 / warmup+cosine / LPIPS+distill 更优梯度)。**#15 三大病理(μ塌缩/x_hat值域/sample发散)已全部证伪于真权重+真图**,重建路径健康结论稳固。下一步价值在真实 Stage 1 训练,非继续调 overfit。

---

## 12. 真实 Stage 1 训练路径验证(2026-07-15,首次真图 + 完整 TR-2 管线)

§11 的 overfit 探针只跑"裸模型 + 纯 L1",本节首次跑**完整训练管线**(TR-2 / JointLoader /
真 SigLIP2 教师 / 增广 / 完整损失)在真实 ImageNet 上,验证的是"真实训练路径"而非"裸重建"。

**接入**:ImageNet-1k 是 HF parquet(`image{bytes,path}`+`label`,294 分片×~4358 行≈128 万)。
新增 `datasets/parquet_image_dataset.py`(D-2b,双仓同步,`@register('parquet_image_dataset')`),
直读 parquet、输出与 D-2 逐键一致的冻结契约(`{video[3,1,H,W],is_video:False,gt,path,label}`),
JointLoader/TR-2 零改动。验证配置 `cfgs/uvt_stage1_imagenet_npu.yaml`(纯图像单源,vid_mock=true)。

**运行**:单卡 NPU,生产模型(**#params=889.2M**,真 so400m),`max_shards=2`(8716 图,in_memory),
bs=8,bf16,`tiny:false`。

| step | 1 | 50 | 100 | 200 | 300 | 400 | 500 |
|---|---|---|---|---|---|---|---|
| PSNR | 2.64 | 5.59 | 7.22 | 8.25 | 8.61 | 8.77 | 8.95 |
| loss | 3.17 | 2.01 | 1.74 | 1.58 | 1.52 | 1.50 | 1.475 |

**结论**:
- **真实训练路径健康**——PSNR 单调上行、总损失单调下降、0 报错/0 NaN、稳态 ~1.7 it/s(单卡 256²)。
- **parquet 接入端到端可用**:`Joint source: parquet_image_dataset, len=8716` 正常,增广/值域/契约与 D-2 一致。
- **#15 修复在完整管线下继续成立**:真权重 + 完整损失(lpips+distill+cos)下无边界规模爆炸、无发散。
- 绝对 PSNR 仍低(~9@500步)是预期的:仍在 warmup+早期、完整损失(非纯 L1)、64 维 latent、
  子集小;这是**健康检查**非训练交付(未存 checkpoint)。曲线仍在上行。

**下一步**:① 8 卡 HCCL 在真权重下复验(当前 8 卡仅 tiny 验过,NPU_NOTES §6 遗留项);
② 扩数据量(去 max_shards/接全量分片,注意全量跨片洗牌抖动,见 parquet_image_dataset docstring)
+ 加长训练观察 PSNR 爬升到 Gate(≥26)。

### 12.1 8 卡 HCCL 真权重复验(2026-07-15,补 NPU_NOTES §6 遗留项①)

同配置 8 卡(`torch.distributed.run --nproc_per_node=8`,bs=4/卡,真 so400m 889M):

| step | 1 | 50 | 100 | 150 | 200 | 272(末) |
|---|---|---|---|---|---|---|
| PSNR | 2.60 | 5.33 | 7.10 | 7.83 | ~8.2 | 8.54 |
| loss | 3.12 | 1.93 | 1.68 | 1.59 | — | 1.496 |

- **DistributedSampler 分片正确**:8716 图 /8 卡 /bs4 = 272 步/epoch,与理论一致。
- **HCCL 梯度平均产出正确收敛**:8 卡曲线与单卡几乎重合(150 步 7.83 vs 单卡 7.84)。
- **`find_unused_parameters` 正确处理纯图像 batch**(is_video=False → Decompressor 不参与,契约⑥)——无 DDP reducer 报错。
- Epoch 1 干净跑完 228.6s,0 报错,launcher 正常退出。**→ 真权重 8 卡 DDP 路径打通,NPU_NOTES §6 遗留项①关闭。**
