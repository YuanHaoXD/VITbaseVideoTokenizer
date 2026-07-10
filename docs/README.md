# UVT（Unified Video Tokenizer）最终方案 · 总纲

> 版本：Final v1.0（2026-07-07）。**本文件夹五篇文档是唯一权威版本**，取代此前所有草稿（Hydra-X 阅读大纲、v1/v2 计划、三份代码分析——它们降级为背景资料）。
> 文档地图：
> - `README.md`（本篇）：项目定义、全部关键决策及理由、术语表、分工、预算、时间线
> - `01_架构与方法.md`：概念、模型完整规格、张量流、全部核心模块参考实现、架构裁决记录（ADR）
> - `02_代码实施.md`：仓库与环境、LARP 骨架改造清单、OmniTokenizer Phase B 改造点（文件:行级）、单元测试
> - `03_数据与评测.md`：每个数据集的获取/处理/去重管线、评测协议唯一实现、五家校准锚点、语义与可生成性评测
> - `04_实验与验收.md`：逐实验枚举（run ID/配置/seed）、Gate 验收、统计纪律、实验卡、复盘协议、风险登记册
> - `05_代码总体架构与实现任务书.md`：顶层代码架构 + 逐文件任务卡（职责/接口契约/来源/验收），实现可直接分派给其他模型或同学
> - `code/`：样板实现（`attention_mask.py`、`temporal_fold.py`）——任务卡的代码风格与质量标尺
> - `损失详情.md`：全部损失函数的教学版详解（是什么/为什么/失控症状速查）

---

## 1. 项目定义

**做什么**：一个统一视频 tokenizer（工作代号 UVT）——单一 ViT 模型、单一表征空间，同时满足：
1. **图像+视频统一**：一套权重处理两种输入（1 锚帧 + 16 帧 clip 协议）；
2. **重建+语义统一**：latent 既能高保真重建像素（PSNR/rFID/rFVD 对标 Wan2.2），又携带可线性读出的语义（zero-shot/linear probe 对标 AToken）；
3. **latent 可生成性**：冻结 tokenizer 后在 latent 上训 DiT 收敛良好（gFID 对照实验取胜）。

**不做什么**：不做 UMM（不接 LLM 做端到端理解/生成系统）；不做离散主线（FSQ 仅作可选附赠分支）；不做长视频生成/编辑系统（表征级 STI 除外）。

**交付物**：encoder/decoder 权重（HF from_pretrained 可加载）+ 训练代码 + 论文级实验矩阵。

## 2. 关键决策记录（每一条：决策 + 理由 + 出处）

| # | 决策 | 理由 | 依据文档/证据 |
|---|---|---|---|
| D1 | 只做 tokenizer，不做 UMM | 论文知识集中在 tokenizer 消融（成本 10%），系统工程占成本 90%；UMM 侧存在论文含糊点（LLM 输入路径）与巨额数据工程 | Hydra-X 表 1/2/3 vs 表 4-7 的成本分布；自我批判复盘 |
| D2 | **连续 latent 主线**，FSQ 离散作可选附赠 | ① HYDRA 路线硬前提（GSB/蒸馏/decoder 全在连续空间）；② tokenizer 层连续比 VQ 更易训（无码本坍缩/STE 梯度，LARP 配置里 6+ 个量化补丁旋钮是反面证据）；③ 下游 diffusion 吃连续；④ FSQ 分支保离散兼容 | LARP/OmniTokenizer 配置实读；Hydra-X 表 4（VQ 系理解分系统性低） |
| D3 | 最终架构**自建标准 ViT**（SigLIP2-So400M 初始化），不基于任何现成 tokenizer 架构 | 核心卖点（预训练先验、语义蒸馏、继承文本塔 zero-shot）只有标准 ViT 承载；OmniTokenizer/LARP 架构均无法加载 SigLIP2 权重 | SigLIP2=27 层标准 ViT；OmniTokenizer=分解式+Swin；LARP=holistic query |
| D4 | 工程骨架 = **LARP base_trainer 改造** | 三仓库中唯一无 Lightning 的现代框架（裸 DDP+compile+AMP+多 decay EMA+null 冒烟数据集）；OmniTokenizer/LeanVAE 均锁死 PL 1.5.4；团队成员有 LARP 实操经验 | 《LARP_LeanVAE_代码架构分析》§1.2 |
| D5 | GAN 方案 = LARP 的 **transformer 判别器 + LeCam + ns_smooth + D 每 5 步更新** | 比 taming PatchGAN 现代（MAGVIT-v2 系），与纯 transformer 叙事一致；仅在 Stage 2（encoder 冻结）引入，天然稳 | LARP loss.py 实读 |
| D6 | Phase B 探路消融在 **OmniTokenizer** 上做，不能换 LARP | 只有它有"时间注意力轴 + 锚帧协议"，是 Hydra-X 消融的"改造前状态"；LARP 的 holistic query 无时间轴，两个消融问题在它身上不存在 | 《OmniTokenizer_代码架构分析》§3 |
| D7 | 锚帧协议采用 **OmniTokenizer 式锚帧隔离**（首帧独立、永不参与时间折叠），**主动偏离** Hydra-X 的 zero-pad 方案 | 隔离式更干净、是行业标准（Wan-VAE/LeanVAE 同款）、消掉 zero-pad 方向歧义（原 ADR-4）；保留 zero-pad 对照臂验证无损 | 三份代码分析交叉验证 |
| D8 | 教师 = SigLIP2-So400M（图像，兼初始化+文本塔）+ InternVideo-Next-L（视频，备胎 InternVideo2-Stage2-1B） | 论文同款；SigLIP2 一模三用（初始化/教师/zero-shot 文本塔） | Hydra-X §6 Implementation |
| D9 | 评测先行校准：**五家公开 checkpoint 锚点**（AToken、Wan2.2-VAE、OmniTokenizer、LARP、LeanVAE），主判据=相对排序复现 | 消除"评测实现错误"这一最大不可归因误差源；绝对值对齐不可强求（FVD 实现敏感、Hydra-X 协议未公开脚本） | 自我批判 #11/12 的修正 |
| D10 | 数据主力 = ImageNet-1k + UCF-101（Phase B/协议层）+ OpenVid-1M/OpenVidHD（规模层）；VideoUFO 可选补充；**禁用 WebVid** | tokenizer 不需要 caption → 选型标准=画质/多样性/许可/可下载；1M 级足够（300k it×bs256≈75 epoch）；WebVid 有版权纠纷 | 数据调研（2026-07-07 核实） |
| D11 | 语义评测 = zero-shot（继承 SigLIP2 文本塔）+ K400/SSv2 linear probe + CKNNA（参照 DINOv2）；**禁用对教师自身的 CKNNA** | 后者是循环论证（对视频教师蒸馏必然提高与视频教师的对齐度）；DINOv2 与两教师独立 | 自我批判 #4 的修正 |
| D12 | 可生成性验证 = 冻结 tokenizer + DiT（ImageNet 256 类条件，gFID）+ Latte（UCF-101，FVD），管线用 OmniTokenizer 仓库自带 | tokenizer 论文标准做法（VA-VAE/MAETok/AToken 同款），成本≈UMM 的 1/100；管线现成 | OmniTokenizer Diffusion/ 目录实读 |
| D13 | 所有消融 ≥2 seed、同 seed 同数据序**配对比较**、报 delta 符号一致性；主判据用 rFVD/rFID（效应量大），PSNR 辅助 | Hydra-X 表 1 的 PSNR 效应量仅 0.04dB，n=2 估不出 σ，配对设计是小样本下唯一可靠方案 | 自我批判 #9 的修正 |
| D14 | KL 权重从 **1e-6** 起步向上扫到 1e-4 | 同体系实证参照：OmniTokenizer VAE 阶段用 1e-6；HYDRA 用 1e-4 但其损失量纲不同（flow decoder） | 两处代码/论文交叉 |
| D15 | 无 GAN 变体（Gram loss / DINOv3 感知损失）作为 Stage 2 的平行对照臂 | AToken 证明无 GAN 可行，ViTok-v2（2026-05）用 DINOv3 感知损失进一步验证；既是保险也是卖点 | AToken 论文 + ViTok-v2 调研 |

## 3. 术语表（本项目语境下的精确含义）

| 术语 | 含义 |
|---|---|
| tokenizer（视觉） | 自编码器意义的 encoder（像素→latent）+ decoder（latent→像素），非 NLP 分词器 |
| token | 一个向量。像素 patch 经"拉直+线性投影"得到（无词表）；离散 token = 向量在码本中的最近邻 id |
| 标准 ViT | patchify + 清一色全局自注意力 pre-LN block + 常规位置编码；判据=能否加载 SigLIP/CLIP 系预训练权重 |
| 锚帧（anchor） | clip 的第 1 帧，独立编码、永不参与时间折叠（D7）；纯图像=只有锚帧 |
| tubelet 因果注意力 | 时间位 t 只注意 {t−1, t} 两个时间位的全部空间 token（折叠后时间位为单位） |
| 层级时间 patchify | 两个 2× 折叠阶段替代单步 4×；折叠=相邻 2 时间位在通道维拼接+线性投影 |
| GSB | Generation-Semantic Bottleneck：Linear→(μ,ρ)→重参数化采样出 64 维连续 latent z + KL 正则 |
| Gen-ViT / Sem-ViT | SigLIP2 的 27 层从第 13/14 层间切开的前后两段；前段供重建结构，后段把 z 展开为语义特征 s |
| Decompressor | 训练期专用的 4× 时间上采样小模块，把压缩 latent 升回原帧率以接受视频教师蒸馏，训完丢弃 |
| 语义蒸馏 | Sem-ViT 输出与冻结教师特征做 cosine 距离对齐 |
| 表征调和（Stage 3） | 统计 z 的逐通道均值/方差，定义规范化 latent ẑ 接口，只训 Sem-ViT 消除两头特征异质性 |
| STI（表征级） | 编辑图像对 (源,目标) 当长度 2 clip 联合过 Sem-ViT（tubelet 因果 mask），源侧不受目标污染 |
| rFID / rFVD | 重建的 Fréchet 距离（Inception-v3 / I3D 特征），r=reconstruction |
| gFID | 在冻结 latent 上训 DiT 后的生成 FID——度量"latent 可生成性" |
| zero-shot（我们的用法） | 蒸馏对齐 SigLIP2 嵌入空间后，直接用其文本塔算图文相似度做分类，零额外训练 |
| linear probe | 冻结 tokenizer，特征上只训一个线性分类器 |
| CKNNA | 表征相似度指标；参照系必须用第三方模型（DINOv2） |
| LeCam | GAN 判别器正则项，抑制判别器过强 |
| 配对比较 | 两个变体用相同 seed/初始化/数据顺序训练，比较逐对差值而非绝对值 |

## 4. 团队分工（三人，几乎无阻塞依赖）

| 成员 | 职责 | 对应文档章节 |
|---|---|---|
| 熟 LARP 的同学 | ① 主仓 trainer 改造（补梯度累积+torchrun）；② transformer 判别器损失移植；③ Phase 5 可生成性（DiT/Latte） | 02 §2、§3；04 Phase 5 |
| 熟 LeanVAE 的同学 | ① 评测协议统一与五锚点校准；② 数据管线（下载/转码/去重/csv）；③ 图像视频联合 loader；④ LeanVAE 效率基线入表 | 03 全篇；02 §2.3 |
| 你 | ① 模型核心（SigLIP2 加载切分/GSB/tubelet/层级 patchify/Decompressor/STI）；② Phase B 在 OmniTokenizer 上的消融；③ 实验设计与复盘主持 | 01 全篇；02 §4；04 全篇 |

## 5. 时间线与算力预算（A100/H800-80G 等效）

| 阶段 | 内容 | GPU·h | 周数（16 卡） | 负责人 |
|---|---|---|---|---|
| P0 | 环境+数据落盘+五锚点校准 | ~150 | 1 | LeanVAE 同学主导，全员环境 |
| PB | OmniTokenizer 探路消融（边界条件） | ~2,500 | 2–3（与 P1 并行） | 你 |
| P1 | 主仓骨架+图像 tokenizer 闭环 | ~1,500 | 2 | 你+LARP 同学 |
| P2 | 视频扩展+表 1 型消融 | ~12,000（紧凑 7,000） | 3 | 你 |
| P3 | 蒸馏+Decompressor+语义评测 | ~10,000 | 3 | 你+LeanVAE 同学 |
| P4 | 三阶段完整训练+全基线对标 | ~15,000 | 4 | 全员 |
| P5 | DiT/Latte 可生成性 + 表征级 STI | ~3,500 | 2–3 | LARP 同学 |
| **合计** | | **~44,700（紧凑 ~27,000）** | **13–16 周** | |

**档位落定（2026-07-07）：用户确认公司集群算力充裕 → 执行完整档，04 §6 降档规则封存备用。** 同时启用以下增强包（原为省算力妥协的项目，按优先级）：

| 增强项 | 内容 | 增量成本 | 回报 |
|---|---|---|---|
| E1 | 关键消融 seed 2→3（P2 主消融、P3 五臂、Phase B） | +~8,000 | 统计结论从"符号一致"升级为可报均值±std |
| E2 | P4 GAN/noGAN 双臂都跑满 100k（原计划 noGAN 只做保险） | +~4,000 | 无 GAN 训练成为独立卖点章节 |
| E3 | 补 C=16 GSB 变体做 gFID 通道混杂对照（风险 R6 的完整解法） | +~5,000 | 可生成性对比的公平性无懈可击 |
| E4 | P4 增加 512² 视频精修阶段（原只 256²+混合分辨率图像） | +~6,000 | 高分辨率对标 Wan2.2 的底气 |
| E5 | 消融矩阵补全：Phase B 与 P2 的 mask×fold 全交叉（原取 5~6 组合） | +~5,000 | 交互效应可分析（tubelet 与层级折叠是否协同） |
| 合计 | | **+~28,000 → 总预算 ~73,000 GPU·h** | |

**算力充裕时的三条纪律（比预算更重要）**：
1. **瓶颈随之转移到"人的分析吞吐"**——并行跑的实验再多，没人看结果就是浪费。规则：任何时刻在跑的消融组数 ≤ 团队一周能复盘完的数量；实验卡纪律（04 §4）在多实验并行下是生命线；
2. **不要因为算力多就把 UMM 加回来**——UMM 被砍的原因是数据工程与论文含糊点，不是 GPU（README D1 的理由一条都没变）；
3. **日历时间压缩**：Phase 内消融臂全并行后，墙钟瓶颈只剩依赖链 P1→P2→P3→P4→P5 与 P4 的串行三阶段，预计 **13–16 周 → 9–11 周**；Phase B 与 P1/P2 照旧并行。

数据合规提醒（公司集群语境）：ImageNet/SSv2 的许可为非商业研究用途，OpenVid/VideoUFO 为 CC-BY-4.0——发表学术论文通常没问题，但建议向公司确认"集群上跑学术项目"的政策与成果归属，避免投稿时的利益声明麻烦。

## 6. 与既往文档的关系

背景资料（不再维护）：《Hydra-X_阅读大纲与复现方案》《Hydra-X_复现代码架构与实施计划 v1》《UnifiedVideoTokenizer_研究计划 v2》《OmniTokenizer/LARP_LeanVAE 代码架构分析》《VTok精读与LARP_LeanVAE评估》。本方案已吸收其全部有效内容；若发现冲突，以本文件夹为准。
