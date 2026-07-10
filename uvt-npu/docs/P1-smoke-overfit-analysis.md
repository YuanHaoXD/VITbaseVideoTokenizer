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

## 8. 产物清单

- 日志:`uvt-npu/logs/{uvt_p1_overfit,uvt_p1_overfit2,uvt_p1_overfit3,uvt_p1_overfit_l1only,uvt_p1_overfit_detz,probe_recon}.log`
- 脚本:`uvt-npu/scripts/p1_smoke_overfit.py`(支持 STEPS/BS/SIZE/LR/AMP/DIST/LPIPS_W/SAMPLE 环境变量)、`scripts/probe_recon.py`
- 本文档:`uvt-npu/docs/P1-smoke-overfit-analysis.md`
