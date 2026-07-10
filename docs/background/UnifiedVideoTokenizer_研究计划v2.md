# Unified Video Tokenizer 研究计划 v2（tokenizer-only）

> 前置文档：《Hydra-X_阅读大纲与复现方案.md》《Hydra-X_复现代码架构与实施计划.md》（下称 v1）
> v2 变更：目标从"复现 Hydra-X 全系统"收窄为"**借鉴 Hydra-X 的 tokenizer 经验，做我们自己的统一视频 tokenizer**"。砍掉 v1 的 Phase 5~7（UMM 部分），随之消灭 v1 自我批判中的头部风险（ADR-1、长视频协议、UMM 数据工程）；新增探路消融、tokenizer 级语义评测协议、latent 可生成性验证。
> 日期：2026-07-07
> **v2.1 修订（同日，LARP/LeanVAE 代码实读后，详见《LARP_LeanVAE_代码架构分析.md》）**：①主仓训练骨架由"OmniTokenizer 移植 + accelerate 重写"改为 **LARP base_trainer 改造**（唯一无 Lightning 的现代自研框架：裸 DDP+compile+AMP+多 decay EMA，需补梯度累积与 torchrun；且团队成员有实操经验）；②Stage 2 判别器方案由 taming PatchGAN 改为 **LARP 的 transformer 判别器 + LeCam + d_update_freq 配方**；③评测校准锚点扩为五家（AToken/Wan2.2/OmniTokenizer/LARP rFVD20/LeanVAE）；④LeanVAE 进效率基线，其 DAVIS 数据类与双 FVD 实现直接复用；⑤OmniTokenizer 角色收缩为 Phase B 实验台 + 联合 loader 设计参考。前提确认：项目走连续 latent 路线（若组内改走 LARP 离散 AR 路线，本计划需大改）。

---

## 0. 研究目标与交付物

**做什么**：一个统一视频 tokenizer——单个模型同时满足：

1. **图像+视频统一**：一套权重、一个表征空间处理两种输入（1 锚帧 + T 帧 clip 协议）；
2. **重建+语义统一**：latent 既能高保真重建像素（服务生成），又携带可被线性读出的语义（服务理解）；
3. **latent 可生成性**：在 latent 空间上训练生成模型（DiT）收敛良好——这是 tokenizer 被下游采用的关键属性。

**交付物**：encoder/decoder 权重 + 训练代码 + 论文级实验矩阵（消融+基线对比+探针评测）。**不交付**对话/生成系统——那是 UMM 的事，是本 tokenizer 的潜在下游。

**成功标准**：

- 重建：ImageNet/DAVIS/UCF 全面超过 OmniTokenizer（26.74 PSNR / 113.56 rFVD，差距巨大易达成），逼近或超过 AToken（29.72 / 29.19）；挑战 Wan2.2（31.25 / 14.78）；
- 语义：ImageNet zero-shot（经蒸馏继承 SigLIP2 文本塔）≥ 75%，视频 linear probe（SSv2/K400）显著高于纯重建基线；
- 可生成性：ImageNet 256 class-conditional DiT-B 在我们 latent 上的 gFID 优于在 OmniTokenizer-KL latent 上的同配置训练；
- 科学贡献：给出 tubelet/层级 patchify 两条结论在"有/无图像预训练先验"两种条件下的**边界条件研究**（Hydra-X 未做，是我们的增量）。

---

## 1. 底座选型（v1 调研的结论落地）

| 用途            | 选择                                                                                  | 理由                                                                    |
| ------------- | ----------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| 训练框架/损失/数据管线  | **OmniTokenizer**（FoundationVision，NeurIPS 2024，完整训练代码+权重，两阶段图像→图像视频联合训练）           | 唯一带全套训练代码的图像+视频联合 tokenizer；同时是 Hydra-X 表 1/9 的对照基线                   |
| 主架构初始化        | **SigLIP2-So400M-patch16**（路线 A，主线）                                                 | Hydra-X"少注意力更好"结论明确依赖图像预训练先验，路线 A 条件对齐                                |
| 快速探路（路线 B）    | 直接在 OmniTokenizer 架构上改 tubelet/层级 patchify                                          | 改动小、两周出结果；无论结论正反都是论文素材（边界条件）                                          |
| 架构参考+评测校准锚点   | **AToken**（apple/ml-atoken，有权重无训练代码）                                                | 校准我们的 rFID/rFVD 脚本；其语义评测协议直接借用                                        |
| 语义教师          | SigLIP2-So400M（图像）+ InternVideo-Next-L（视频，HF OpenGVLab collection）                  | 同 v1；注意 InternVideo-Next 无文本塔（其卖点是无视频文本监督），视频侧语义评测用 linear probe 而非检索 |
| 重建基线          | Wan2.2-VAE、VidTok、Cosmos-Tokenizer、OmniTokenizer、AToken                             | 全部开源可跑，同协议对比                                                          |
| latent 可生成性协议 | **VA-VAE / LightningDiT 的评测范式**：冻结 tokenizer，latent 上训 class-conditional DiT，报 gFID | tokenizer 论文的标准下游验证，成本 ≈ UMM 的 1/100                                  |

---

## 2. 方法设计：从 Hydra-X 借什么、自己加什么

### 2.1 借（经 v1 调研确认可行的四件事）

1. **tubelet 因果注意力**（2 帧时间窗）——条件性借鉴，见 Phase B 探路；
2. **层级时间 patchify**（2×2 两阶段，锚帧 zero-pad）——同上；插入位置按 v1 ADR-3 做预消融；
3. **GSB 瓶颈**（线性投影→[μ,ρ]→重参数化，C=64，λ_KL=1e-4 起步）+ 三阶段训练配方（基础→冻 encoder 精修 decoder+GAN→归一化调和语义侧）——直接采用；
4. **Decompressor 双教师蒸馏**（压缩轴升维后蒸馏，训完丢弃）——直接采用，这是把"重建 tokenizer"升级为"统一 tokenizer"的核心增值，OmniTokenizer/VidTok/Cosmos 都没有。

### 2.2 自己加（Hydra-X 没做的创新空间，按投入产出排序）

1. **边界条件研究**：tubelet/层级 patchify 在从头训练（路线 B）vs 预训练初始化（路线 A）下的对照——直接回应"这些结论是否只是预训练先验的伴生现象"；
2. **表征级 STI**：把"编辑对当长度 2 clip 联合编码"下沉为纯 tokenizer 能力——验证联合编码 latent 的源图保真（重建探针即可，无需 LLM），为下游编辑应用提供表征基础；可推广到"参考帧条件 tokenization"；
3. **无 GAN 训练变体**：用 AToken 的感知+Gram matrix 损失替代 GAN，对比训练稳定性与 rFID（工程价值高，社区关注）；
4. （可选）**离散变体**：FSQ（VidTok 已验证优于 VQ）加在 GSB 之后，覆盖离散 token 的下游需求。

### 2.3 仓库架构

沿用 v1 §2.2 的树，删除 `src/hydrax/models/umm/`、`heads/`、`train/umm_trainer.py`、`eval/{lmms_adapter,geneval,wise,vbench,edit_runner}`；新增：

```
src/uvt/eval/
├── semantic/
│   ├── zeroshot_imagenet.py     # 蒸馏对齐SigLIP2空间 → 直接借SigLIP2文本塔做zero-shot
│   ├── linear_probe.py          # ImageNet / K400 / SSv2 冻结特征线性探针
│   └── cknna.py                 # 与第三方参照(DINOv2)的对齐度分析, 避免v1指出的循环
└── generability/
    ├── train_dit.py             # DiT-B @ ImageNet256, latent空间, LightningDiT协议
    └── gfid_eval.py
src/uvt/models/tokenizer/fsq.py  # (可选)离散变体
```

v1 §3 的六个核心模块参考实现中，前五个（tubelet mask、层级 patchify、GSB、Decompressor、双教师蒸馏）原样适用；第六个（STI 路由）去掉 UMM 依赖，简化为 tokenizer 内的联合编码接口。

---

## 3. 分阶段实施计划

### Phase 0：评测校准（1 周，1~4 卡）——同 v1，缩减

- 用 AToken-So/C 与 Wan2.2-VAE 公开权重校准 PSNR/SSIM/rFID/rFVD 脚本；
- **Gate 修正（吸收 v1 自我批判 #11/12）**：主判据 = 两个公开 checkpoint 的**相对排序与差距比例**复现；绝对值 ±10% 为尽力目标，不作为阻断条件；
- 语义评测校准：SigLIP2 官方权重跑 zeroshot_imagenet.py，复现其公开 zero-shot 数字（±0.5%）——这校准了语义评测这把新尺子。

### Phase B：路线 B 探路消融（2~3 周，8 卡）★新增，回答"结论是否迁移"

> **可行性已经代码实读验证**，文件级改造点见《OmniTokenizer_代码架构分析.md》（2026-07-07）。要点：tubelet 改造落点 `modules/attention.py` Attention.forward（带状 mask 替换 is_causal，~40 行）；层级 patchify 落点 `omnitokenizer.py` to_patch_emb + encode() 中段插入（~80 行）；锚帧在该代码库天然隔离于时间折叠 → v1 的 ADR-4 作废；模型仅 ~80M，预算成立。

- 在 OmniTokenizer 官方代码上最小改动：其时间层因果注意力 → tubelet 2 帧窗；其时间 patchify → 层级 2×2；
- 消融矩阵扩为 3 mask × {单步 4×, 学习式 2×+2×, 2×+AvgPool2×（利用其自带 defer_temporal_pool，无参数退化版）} 取 5~6 组合——第三臂能区分"层级好在多阶段"还是"好在有学习参数"，信息量超 Hydra-X 原表；
- 数据用 ImageNet+UCF101（官方有对应 ckpt/datalist，校准闭环完整），官方配方缩短到 150k 迭代，2 seed 配对比较；三个 mask 变体必须走同一条注意力实现路径（官方已承认 SDPA 与手写路径数值不等价）；环境用 docker 锁死 README 组合（torch2.2.1+cu118+PL1.5.4），在 Linux 上跑；
- **产出（无论正反都有效）**：若增益保持 → 结论普适，路线 B 可作为低成本主线备份；若消失/翻转 → 证实"预训练先验依赖"假说，写进论文的分析章节，主线坚定走路线 A。
- 附带修订：Phase 5 可生成性协议改用该仓库自带的 Diffusion/DiT+Latte 管线（预算 5,000→~3,500）；KL 权重按其 stage3 实证值 1e-6 起步；路线 A 移植清单（联合 loader、双判别器、FVD 双实现、FSQ 全家桶）见分析文档 §6。

### Phase 1：SigLIP2 骨架 + 图像 tokenizer（2 周，8 卡）——同 v1 P1

- 27 层加载断言、13+14 切分、GSB、对称 decoder、L1+LPIPS+KL 闭环、单 batch overfit 冒烟；
- Gate：overfit PSNR>35；50k 迭代 val PSNR≥26（阈值在拿到第一条真实曲线后重新标定——吸收 v1 批判 #10）。

### Phase 2：视频扩展 + 注意力/patchify 消融（3 周，8~16 卡）——同 v1 P2

- ADR-3/4 预消融（patchify 位置×pad 方向）→ 表 1 型消融（4 变体 × 2 seed，配对比较报 delta 符号一致性——吸收 v1 批判 #9）；
- Gate：rFVD 排序在配对比较下方向一致；tubelet 延迟 ≤ full 的 40%。

### Phase 3：GSB + Decompressor 双教师蒸馏 + 语义评测（3 周，16 卡）

- v1 P3 的探针体系按批判 #4 修正：**废弃对教师的 CKNNA（循环论证）**，改为
  ① ImageNet zero-shot（借 SigLIP2 文本塔——蒸馏对齐其空间后文本塔免费继承，AToken 同款做法）；
  ② SSv2/K400 linear probe（时间敏感，第三方参照）；
  ③ CKNNA 参照系改为 DINOv2（与两个教师都无关）；
- 消融矩阵：{无蒸馏, img-only, img+Decomp(vid), img+Decomp(img), Sem 侧双向注意力} 五组；
- Gate：img+Decomp(vid) 在视频探针最优且图像探针不降；蒸馏使 zero-shot 从 ~0（纯重建）升到 >70%。

### Phase 4：完整三阶段训练 + 全基线对标（4 周，16~32 卡）——同 v1 P4

- Stage1 基础（ImageNet→混合分辨率+OpenVid）→ Stage2 冻 encoder+GAN 精修（平行跑无 GAN 的 Gram loss 变体）→ Stage3 归一化调和；
- 平行小任务：纯 ImageNet 版（对齐 Hydra-XTok† 协议，数值级可比）；
- Gate：成功标准表（§0）中的重建与语义两行。

### Phase 5：latent 可生成性 + 表征级 STI（3 周，16 卡）★替代 v1 的 UMM 阶段

- P5-1：冻结 tokenizer，DiT-B @ ImageNet256 class-conditional，LightningDiT 协议，对照组=同配置跑 OmniTokenizer-KL latent 与 SD-VAE latent；报 gFID-50k；
- P5-2：表征级 STI——ImgEdit 图像对经"长度 2 clip 联合编码 vs 独立编码"，比较：源图 latent 重建 PSNR、联合编码下 target latent 对源结构的探针可读性；
- Gate：我们 latent 上的 DiT gFID ≤ OmniTokenizer latent 对照组；STI 联合编码源图重建不劣于独立编码（验证联合编码无损）。

### 算力预算（A100-80G 等效）

| Phase  | GPU·h                                           | 备注                           |
| ------ | ----------------------------------------------- | ---------------------------- |
| P0     | ~150                                            |                              |
| PB     | ~2,500                                          | 探路消融                         |
| P1     | ~1,500                                          |                              |
| P2     | ~12,000                                         | 可砍 seed 至 ~7,000             |
| P3     | ~10,000                                         | 五组消融                         |
| P4     | ~15,000                                         | 含无 GAN 变体与 † 版               |
| P5     | ~5,000                                          | DiT-B ~400k 迭代 + STI         |
| **合计** | **~46,000（完整）/ ~28,000（紧凑：砍 seed、降分辨率、P4 单变体）** | 对比 v1 的 62,000+（且 v1 不含数据工程） |

紧凑档在 16 卡上约 10~11 周，8 卡约 5 个月。若只有单机 8 卡以下，需再降一档（ViT-B 规模 + 128²/256² 主打消融故事，放弃对标 Wan2.2 的数值目标）——届时告知实际卡数我再出微缩配置。

---

## 4. 风险登记册 v3（相对 v1 的增删）

| 风险                                 | 等级          | 说明与缓解                                             |
| ---------------------------------- | ----------- | ------------------------------------------------- |
| ~~ADR-1 LLM 输入路径~~                 | **已消灭**     | 无 LLM                                             |
| ~~长视频理解协议~~                        | **已消灭**     | 无理解任务，clip 协议只服务重建/蒸馏                             |
| ~~UMM 数据工程~~                       | **已消灭**     | 只需 ImageNet+OpenVid+ImgEdit（全开源直下）                |
| tubelet/层级结论不迁移                    | 中→**转化为机会** | Phase B 把它变成边界条件研究，正反结论皆可发表                       |
| 蒸馏与重建目标冲突（λ_dist 平衡）               | 中           | 三阶段解耦天然缓解；P3 对 λ_dist∈{0.25,0.5,1.0} 小网格          |
| zero-shot 继承文本塔的前提是蒸馏空间严格对齐        | 中           | P3 Gate 显式检验；若对齐不足退化为 linear probe（结论仍成立，卖点弱化）    |
| gFID 对照实验的公平性（latent 维度/token 数不同） | 中           | 固定 DiT 配置与训练预算，报多个 checkpoint 的 gFID 曲线而非单点       |
| KL 权重量纲（v1 文档矛盾处）                  | 低           | P1 显式扫 {1e-4, 1e-5, 1e-6}，以重建-蒸馏 Pareto 选择，结果回填文档 |
| GAN 不稳定                            | 低           | 仅 Stage2 且 encoder 冻结；平行的 Gram loss 变体本身是卖点也是保险   |

---

## 5. 与 v1 的对应关系（防止信息丢失）

- v1 §1（开源调研矩阵、HYDRA 超参表、ADR-2~7）：**继续有效**，ADR-1 作废，ADR-6 简化为表征级；
- v1 §3（六模块参考代码+单测）：前五个原样用，STI 简化；
- v1 §5（误差防控五道防线、实验卡规范、复盘协议）：**全部保留**，是本计划的执行纪律；
- v1 的 Phase 5~7、Show-o2 移植、lmms-eval 适配：归档，若未来升级 UMM 再启用。
