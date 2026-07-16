# UVT 代码 vs Hydra-X / HYDRA 论文 — 对照工作笔记

> 目的:回答"当前模型与两篇论文里的方法出入是什么"。
> 论文: Hydra-X (arXiv 2606.13289) · HYDRA 前作 (arXiv 2603.15228)
> 代码锚点: uvt/models/uvt/*.py + uvt/losses/*.py + UVTConfig
> 状态: 论文 PDF 下载/解析中;本笔记先记录**从代码 + 项目背景文档**已确立的锚点,待论文原文核验。

## A. 当前代码的既定事实(已从代码读出,确定)

### UVTConfig 默认值 (uvt_tokenizer.py:46-69)
| 字段 | 默认 | 注释出处 |
|---|---|---|
| model_name | siglip2-so400m-patch16-**256** (非 naflex) | §2.1 固定分辨率 |
| gen_depth | 13 | HYDRA 均衡切分 27→13+14 |
| c_latent | 64 | HYDRA C=64 |
| fold_positions | **(0, 6)** | ADR-3: {输入处, GenViT 第6层后} |
| unfold_positions | (21, 27) | 镜像 |
| attn_mode | **tubelet** | ADR-2 {full,causal,tubelet} |
| rope_dims | 32 | ADR-8 时间 RoPE 每头前32维 |
| use_cos_consistency | True | HYDRA L_cos(§2.6 式4) |
| kl_weight | **1e-6** | D14/§2.9 |
| l1_weight | 1.0 | |
| lpips_weight | 1.0 (默认;overfit/训练用 0.5) | |
| cos_weight | 1.0 | |
| lambda_dist | **0.5** | §2.8 默认 |

### 三路 normalize 分流 (uvt_tokenizer.py:137-177) — Stage3 命门
- decoder 吃**物理** z
- sem_vit 吃**规范** z = gsb.to_canonical(z)
- L_cos 的 mu_proj = sem_vit.in_proj(to_canonical(mu))
- out["h"] = gsb.norm(h) (第15号修复:巨激活规范化)
- Stage1/2 时 to_canonical 恒等

### 三阶段冻结 (set_stage, :244-276)
- S1 全可训 / S2 仅 decoder / S3 仅 sem_vit(含 map_head)

## B. 项目【自己声明】的偏离点(来自 docs/background,权威)

### B1. 两处已确认 HYDRA→Hydra-X 差异 (复现文档 §1.2 第60-63行)
1. **解码器**: HYDRA=pixel flow decoder(流匹配式,λ_FM=1.0/λ_perc=0.1/λ_gan=0.075);
   Hydra-X=AToken式 **27层对称ViT decoder + L1直接回归**(式7=L1+LPIPS+GAN+KL)。
   → 本项目实现 Hydra-X 版。
2. **初始化/教师**: HYDRA=InternViT-2.5; Hydra-X=SigLIP2(初始化+图像教师)+InternVideo-Next-L(视频教师)。

### B1.5 【代码直读发现】第三处主动偏离:锚帧折叠方式 (temporal_fold.py:4)
- **Hydra-X 论文**: 每个时间 patchify 阶段,锚帧做 **zero-pad**,使其经历与其余帧相同的折叠算子。
- **本项目代码**: ADR-4' — 锚帧(时间位0)**永不参与折叠,直通隔离**;主动偏离 Hydra-X 的
  zero-pad 方案,理由=与 OmniTokenizer/Wan-VAE/LeanVAE 行业标准一致;zero-pad 作为对照臂(P2-pre 消融)。
  证据行: temporal_fold.py:26 `anchor, frames = x[:, :1], x[:, 1:]` 只折叠 frames。
- 折叠算子: Linear(2D→D) + rearrange,近"两帧平均"初始化 (temporal_fold.py:18-23),等价 Conv3d(kt=2,st=2)✓与论文一致。

### B1.6 【代码直读发现】Decompressor 实现细节 (decompressor.py)
- 结构 ✓与论文一致: 两级 2× 上采样 (4→8→16), full 双向注意力。
- 上采样算子: **Linear(D→2D)+channel-to-time rearrange** (decompressor.py:39-46),
  论文说"1×1 conv 通道翻倍 C→2C" → 数学等价(论文在 latent C=64 上,代码在 student_dim D=1152 上)。
- **偏离(实现层)**: 代码 Decompressor **不投影到教师维度**(decompressor.py:8-10),
  投影职责交给 loss 的 head_vid。非方法偏离,是职责重组。
- rope_dims=0 (训练期附件无需时间RoPE)。

### B2. 本项目相对 Hydra-X 的收窄(CLAUDE.md + docs/08)
- **只做 tokenizer, 不做 UMM** — 砍掉 LLM/flow-head/编辑(STI)/五任务。
  → Hydra-X 核心贡献二(STI 编辑,§5.3,表3 +6.9dB)**完全不在本项目范围**。
  → 只复现贡献一(Hydra-XTok tokenizer 的三大设计)。

### B3. ADR 裁决(论文歧义处的本地选择,复现文档 §1.3)
- ADR-2: tubelet "帧"=折叠后时间位(位t见{t-1,t},锚帧位0只见自己) [置信度高]
- ADR-3: 两个2×patchify 插入位置={patch embed后, encoder深度1/2} 默认→**代码定为(0,6)** [置信度低,全文最大歧义]
- ADR-4: 锚帧 zero-pad 方向=[零帧,锚帧] [置信度低]
- ADR-5: Stage3 归一化 ẑ=(z-mean)/std, decoder 路径含反归一化 [中高]
- ADR-8: 时间 RoPE 每头前32维

### B4. 蒸馏方式差异(复现文档 §1.2 第55行)
- HYDRA: 多层特征蒸馏(Gen-ViT/Sem-ViT 各选若干层对齐教师对应层, Eq.6)
- Hydra-X 式(3): 只在 **Sem-ViT 输出处**蒸馏
- 本项目: 按 Hydra-X 简化版实现,多层蒸馏作为可选增强(未实现)

### B5. 从 HYDRA 补齐、但两文数值不一致处
- λ_KL: HYDRA=**1e-4**; 本项目代码默认 **1e-6**(≠ HYDRA!) ← 待核验论文是否给出
- λ_dist: HYDRA=1.0; 本项目默认 0.5
- L_cos: HYDRA Eq.4 有; Hydra-X 式(2)/(7) 未提及 → 本项目默认开启(配置开关)

## C. 待论文原文核验的问题(TODO — 解析完 PDF 逐条回答)
1. [Hydra-X 式7] recon loss 各 λ 论文到底给没给数值? 代码 l1=1/lpips=1/kl=1e-6 是否对齐?
2. [Hydra-X 式3] distill: d_cos(s_0,T_img) + d_cos(D(s_1:),T_vid) — 代码是否精确一致?图像池化项(distill_img_pool)是论文有的还是本地加的(ADR-9)?
3. [Hydra-X §4.1] 层级 patchify 到底插在哪(论文有没有明说)? 代码(0,6) 是否合理?
4. [Hydra-X 图2] tubelet 窗口=2(本帧+前一帧)? 与代码 attention_mask.py 是否一致?
5. [Hydra-X §4.2] Decompressor 结构:4×上采样=两阶段2×,每阶段"1×1conv通道翻倍+channel-to-time"? decoder block 双向? 与 decompressor.py 是否一致?
6. [HYDRA Eq.2] GSB 结构 W_proj∈R^{D×2C}→[μ,ρ]→重参数化;反投影 W_unproj — 代码 gsb.py 是否一致? (注:CLAUDE.md 说"gsb无expand/unproj,反投影归各消费方in_proj" ← 这是相对 HYDRA 的**结构改动**,要确认)
7. [HYDRA] Gen/Sem 切分 12+12 最优 → 本项目 13+14, 差异影响?
8. KL 权重 1e-6 vs HYDRA 1e-4 — 谁对?论文 Hydra-X 有没有给?
9. rope_dims/3D RoPE: Hydra-X 说"跟随 AToken 做 3D RoPE" — 代码只对时间轴加32维,空间轴? 与论文一致?
10. cos_consistency(L_cos): Hydra-X 论文里到底有没有? 若没有则是本项目从 HYDRA 引入的附加项。
