# Hydra-X: Native Unified Multimodal Models with Holistic Visual Tokenizers

## 阅读大纲 + 完整复现方案

> arXiv: 2606.13289v1 (2026-06-11) · Nanjing University / CASIA / Tencent Hunyuan
> 作者：Guozhen Zhang, Xuerui Qiu, Yutao Cui 等；通讯：Limin Wang
> **注意：截至撰写本文档（2026-07-07），论文未提供官方代码仓库，复现需从零实现。**

---

# 第一部分：阅读大纲（含方法完整拆解）

## 0. 一句话总结

Hydra-X 是第一个用**单个 ViT** 同时完成图像与视频 tokenization 的原生统一多模态模型（UMM）。其 tokenizer（Hydra-XTok）通过三个反直觉的设计——**tubelet 因果注意力、层级式时间 patchify、Decompressor 双教师蒸馏**——把一个图像 tokenizer 高效改造成图像+视频统一 tokenizer；并把视频用的时间因果机制复用到图像编辑上（把 source/target 当作长度为 2 的视频片段联合编码），在 7B 规模上统一了五个任务：图像/视频理解、图像/视频生成、指令图像编辑。

## 1. 问题背景与动机（对应 §1, §2）

### 1.1 UMM 的视觉编码路线之争

- **解耦编码器**（BAGEL、Janus-Pro 等）：理解走语义 ViT，生成走独立 VAE → 两条视觉通路表征不匹配，LLM 需自行对齐，且互相竞争注意力。
- **统一 tokenizer**（Show-o2、TUNA、HYDRA、Harmon、UniTok 等）：单一表征空间服务理解+生成 → 消除表征失配，理解与生成可互相促进。
- 图像统一 tokenizer 已被广泛研究，但**图像+视频的整体（holistic）tokenizer 几乎空白**。

### 1.2 现有视频 UMM 的两种权宜方案及缺陷

1. **逐帧 tokenizer**（如 TransNext）：图像编码器独立处理每帧 → tokenizer 内部无时间交互，无法编码运动/短时因果，LLM 只能拿到互相割裂的逐帧特征。
2. **级联设计**（Show-o2、TUNA）：3D causal VAE + 语义编码器堆叠 → VAE 独立训练、无语义约束，可能丢弃对理解关键的信息。
3. AToken 虽统一了图像/视频 tokenization，但对重建和理解**输出两套任务专用特征**，并非统一表征。

### 1.3 两个核心挑战

- **(a)** 如何高效地给一个原生（图像预训练的）ViT 注入时空重建能力；
- **(b)** 如何把图像级与视频级语义同时嵌入压缩后的 latent 空间。

## 2. 前置知识：HYDRA 框架（对应 §3，必读前作）

Hydra-X 完全继承 HYDRA（Qiu et al., 2026，图像-only UMM）的"表征调和 tokenization"骨架：

```
x ──Gen-ViT──> h ──Bottleneck──> z ──Sem-ViT──> s <──align── T(x)
                                 │
                                 └──(仅tokenizer训练期)──> Pixel Decoder ──> x̂
```

- 单个 ViT 拆成 **Gen-ViT**（提取结构特征 h）与 **Sem-ViT**（从 latent 恢复语义特征 s），中间是 **Generation–Semantic Bottleneck**（把 h 压成紧凑 latent z ∈ R^{N×C}）。
- Sem-ViT 输出 s 与预训练语义教师 T(x) 做蒸馏对齐。
- **LLM 只消费 Sem-ViT 输出 s**（理解和生成都是）；像素解码器只在 tokenizer 训练期使用（推理生成时 LLM 生成的 latent 经 decoder 还原像素）。
- 阅读建议：先读 HYDRA 原文搞清 Bottleneck 的具体形态与 compress-then-restore 蒸馏思想，再读本文。

## 3. Hydra-XTok：单 ViT 内的整体视觉 tokenization（对应 §4，核心方法一）

### 3.1 总体设定

- Gen-ViT / Sem-ViT 均由 **SigLIP 2** 初始化；encoder/decoder 为对称 ViT 对（decoder 27 层），加 **3D RoPE** 做时空联合建模（跟随 AToken 的做法）。
- 视频片段 x ∈ R^{3×(1+T)×H×W}：编码为 1 个锚帧（anchor）latent + 其余 T 帧按 4× 时间压缩，得
  **z ∈ R^{C×(1+T/4)×(H/16)×(W/16)}，C = 64**（空间 16×，时间 4× 压缩）。
- 总损失：**L = L_rec + λ·L_dist**（重建项 + 语义蒸馏项，见 §5.1/附录 A.1）。

### 3.2 发现一：时间注意力——"少即是多"（§4.1）

对比三种注意力掩码（图 2）：
| 掩码 | 定义 | 延迟(17×512² clip) | ImageNet PSNR/rFID | DAVIS PSNR/rFVD |
|---|---|---|---|---|
| Full | 全时空双向注意力（AToken/OmniTokenizer 的标准选择） | 0.49s | 31.10 / 0.367 | 27.40 / 16.20 |
| Causal | 对所有历史帧的因果注意力 | 0.45s | 31.38 / 0.352 | 27.62 / 14.05 |
| **Tubelet** | 因果且限制在 2 帧窗口：只看本帧 + 紧邻前一帧 | **0.17s** | 31.42 / 0.347 | 27.69 / 13.69 |

**结论**：时间感受野越大重建越差——全局注意力破坏了图像预训练学到的局部性结构先验。Tubelet（最小时间感受野）既最快又最好。

### 3.3 发现二：层级式时间 patchify 优于单步（§4.1）

- 基线（AToken/OmniTokenizer）：输入处一次性 4× 时间 patchify。
- 本文：**两个连续的 2× patchify 阶段**（渐进多尺度折叠时间轴）。
- 每个时间 patchify 阶段，**锚帧做零填充（zero-pad）**，使其经历与其余帧相同的操作。
- 表 1 最后一行（Tubelet + 层级 2×2）：ImageNet PSNR 31.73 / rFID 0.329；DAVIS PSNR 27.97 / rFVD 11.19 —— 全面最优，延迟 0.25s。

### 3.4 发现三：Decompressor 解决视频语义监督的不对称性（§4.2）

**问题**：图像的 Sem-ViT 输出与帧同分辨率，可与图像教师逐 token 对齐；但视频 latent 被时间压到 1+T/4 个时间位，而现有视频编码器都工作在原始帧率 → 视频流没有天然的语义监督源。

**方案**：加一个轻量 **Decompressor D**（小 ViT）：

- 结构：4× 时间上采样器 = 两个连续的（时间上采样 → transformer block）阶段；每次时间上采样 = 1×1 卷积把通道翻倍（C→2C）+ channel-to-time 重排 —— 正好逆转 encoder 的层级 2×2 时间 patchify。
- **只在 tokenizer 训练期使用，训练完丢弃**；LLM 仍然只看紧凑的 Sem-ViT 输出 s。

**蒸馏损失（式 3）**：

```
L_dist = d_cos(s_0, T_img(x)) + d_cos(D(s_1:), T_vid(x))
```

- s_0：开头未压缩的图像（锚帧）token → 直接对齐图像教师；
- s_1:：压缩的视频 latent → 经 D 升回原始时间长度后对齐视频教师；
- 纯图像 batch 时视频项 mask 掉；
- d_cos(a,b) = 1 − cos(a,b)。

**教师**：T_img = SigLIP-SO400M-patch16-naflex（SigLIP 2）；T_vid = InternVideo-Next-L。

**表 2 消融结论**（1.5B UMM 上评测）：

1. 语义蒸馏不可或缺：去掉后图像/视频理解全面崩溃（MVBench 29.8 vs 45.4；MME 989 vs 1501）；
2. **img distill + Decompressor w/ video teacher 是最优组合**：视频理解最强（MVBench 45.4, VideoMME 45.0），同时图像生成（GenEval 72.0）和编辑（ImgEdit 3.20）也最好 → 语义更丰富的 latent 加速 LLM 在生成/编辑上的收敛；
3. Sem-ViT 改双向注意力反而全面变差 → "少即是多"在理解侧同样成立。

## 4. Hydra-X：UMM 整体架构与编辑创新（对应 §5，核心方法二）

### 4.1 整体架构（§5.1）

- 标准原生 UMM 模板：文本 token 与 Hydra-XTok 产出的视觉 token 交错成单序列，进共享 LLM（Qwen2.5-7B-Instruct；消融用 1.5B）。
- 两个头：**自回归语言头**（next-token prediction）+ **视觉头**（rectified flow matching）。
- 复合损失（式 4）：L = λ1·L_NTP + λ2·L_FM，λ1 = λ2 = 1。
- 五个任务共享同一 tokenizer：T2I、I2T、T2V、V2T、图像编辑。

### 4.2 独立编码绕过了 latent（问题诊断，§5.2）

- 传统编辑管线（HYDRA/Show-o2/TUNA/BAGEL）：source x_c 与 target x_t **各自独立**过 tokenizer，s_c ⊥ s_t——两者在 tokenizer 内部零交互；
- LLM 只能在两条已被压缩的语义流之上从头学习跨图对齐 → 高层语义编辑还行，**细节保真编辑一致性差**（latent 级的细粒度结构信息在进 LLM 之前就丢了）。

### 4.3 Tokenizer 级 source-target 交互（STI，§5.3）

- 核心洞察：Sem-ViT 为视频已具备 tubelet 因果注意力 → **把 (x_c, x_t) 当作长度 2 的 clip 直接喂给 Hydra-XTok**，零新增参数。
- 非对称复用（刻意设计）：
  - **Gen-ViT 仍独立编码两图**（编辑对不是时间相邻帧，禁用 Gen-ViT 的跨帧 tubelet 注意力）→ latent z_c, z_t 保持重建保真；
  - **Sem-ViT 联合处理 [z_c; z_t]**，用与视频完全相同的 tubelet 因果 mask：s_c 只看 z_c；s_t 看 [z_c; z_t]（式 6）。
- **表 3 结果**：Recon-PSNR（ImgEdit 上源图重建 PSNR，直接度量编辑一致性）**20.74 → 27.65（+6.9 dB）**；ImgEdit 2.80 → 3.20；GenEval +1.46；其余基准持平或微升。
- 结论：编辑一致性失败源于 tokenizer 内部的 latent 级隔离，而非 LLM 容量或监督不足（附录 D 有定性证据：Indep 变体把车"重新想象"成另一辆车，STI 近乎像素级还原）。

## 5. 实现细节与训练配方（对应 §6 Implementation + 附录 A，复现最关键部分）

### 5.1 Tokenizer 损失（附录 A.1，式 7）

```
L_rec = λ1·‖x−x̂‖₁ + λ_perc·L_LPIPS + λ_gan·L_GAN − λ_KL·Σ_j (1 + ρ_j − μ_j² − exp(ρ_j))
```

L1 像素重建 + LPIPS 感知 + GAN 对抗（提升纹理真实感）+ KL 正则（latent 后验对齐标准正态；μ/ρ 为压缩 latent 的均值/对数方差 → Bottleneck 是 VAE 式重参数化）。
**注意：各 λ 具体数值论文未给出**（复现风险点，见第二部分 §9）。

### 5.2 Tokenizer 三阶段预训练（附录 A.2）

| 阶段       | 内容                                                                                                                   | 关键配置                                                   |
| -------- | -------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| S1 基础训练  | SigLIP-2 初始化 → 先 ImageNet-1.2M @256²，再混合分辨率（256² 视频 + 256~2048 图像）                                                   | 300k iters，AdamW，峰值 lr 2e-4，SigLIP-2/InternVideo 双教师蒸馏 |
| S2 解码器精修 | 冻结 encoder，只微调 27 层 ViT decoder                                                                                      | 引入 GAN 损失                                              |
| S3 表征调和  | 统计 Gen-ViT latent 的逐通道 mean/std → 冻结 Gen-ViT + decoder，解冻 Sem-ViT；Gen-ViT 特征**归一化后**再喂 Sem-ViT 和 decoder，只更新 Sem-ViT | 消除两头之间的特征异质性                                           |

Tokenizer 预训练总成本：**约 24h × 256 GPUs（≈6,100 GPU·h）**。

### 5.3 UMM 三阶段训练（附录 A.3，表 8）

| 设置           | Stage 1 对齐                                            | Stage 2 综合预训练                            | Stage 3 高质量 SFT                                                                          |
| ------------ | ----------------------------------------------------- | ---------------------------------------- | ---------------------------------------------------------------------------------------- |
| 可训参数         | 冻结 LLM，只训视觉组件（projector、timestep embedding、flow head） | 全参数解锁                                    | 全参数                                                                                      |
| LR           | Vision Head 1e-4；Sem-ViT 5e-5                         | Vision Head 5e-5；LLM & Sem-ViT 2e-5      | 同 Stage 2                                                                                |
| 基础分辨率        | 256                                                   | 512                                      | 1024                                                                                     |
| Batch Size   | 1024                                                  | 1024                                     | 1024                                                                                     |
| 任务           | 图像理解+生成                                               | +视频 & 编辑                                 | +纯文本理解                                                                                   |
| 数据比例†        | 0:1:3:0:0:0:0                                         | 0:2:6:1:1:0:0                            | 1:0:3:3:0:1:3                                                                            |
| 数据量          | 100M 图文对                                              | 30M 理解 + 30M 生成（从S1筛选）+ ~2M 编辑 + ~10M 视频 | 6M MMU 指令（LLaVA-OneVision + Pixmo）+ 1.2M 视频指令（LLaVA-Video）+ 10M 美学过滤图 + 6M 高保真合成图 + 编辑数据 |
| 硬件 / 步数 / 时长 | 256 GPU / 50K / ~10h                                  | 512 GPU / 200K / ~96h                    | 512 GPU / 20K / ~24h                                                                     |

† 数据比例 = Text : Image Caption : Image Generation : Video Caption : Video SFT : Image SFT : Edit

**UMM 总算力 ≈ 2,560 + 49,152 + 12,288 ≈ 64,000 GPU·h；加 tokenizer 共 ≈ 70,000 GPU·h。**

### 5.4 消融实验设定（附录 A.4 —— 小算力复现的模板）

- 理解消融：LLaVA-1.5 数据 + LLaVA-Video SFT 数据；
- 生成消融：Qwen2.5-**1.5B** 基座，20M 图文对训练，再用 ImgEdit 数据集微调编辑；
- 重建消融：ImageNet-1k（1.2M）训 **150k iters**，rFID 评测。

## 6. 实验结果要点（对应 §6 + 附录 B/G）

- **重建（表 9）**：Hydra-XTok (Stage 3, 16×空间/4×时间)：ImageNet PSNR 32.04 / rFID 0.465；DAVIS 28.19 / rFVD 11.61；UCF 36.88 / rFVD 3.11 —— 全面超过专用视频 VAE Wan2.2 和统一 tokenizer AToken。仅用 ImageNet 训练的变体（Hydra-XTok†，16×/1×）PSNR 32.96 / rFID 0.154，在 2 倍压缩率下超过 FLUX.1 的 8× VAE → 增益来自架构而非数据。
- **图像理解（表 4）**：AI2D 86.5 / MME 2350 / OCRBench 84.5 / ChartQA 86.5 —— 7B UMM 中最强，逼近 Qwen2.5-VL。
- **视频理解（表 5）**：MVBench 59.1 / Video-MME 60.0 / LongVideoBench 59.5，超 Show-o2-7B。
- **生成（表 6）**：GenEval 0.88（与 BAGEL-14B 持平）、WISE 0.56、VBench Total 83.49（17 帧 640×384）。
- **编辑（表 7）**：ImgEdit 4.34 / GEdit 7.17，在 Extract/Remove 等需要源图保真的维度领先最多——直接验证 STI。

## 7. 批判性阅读清单（读时带着这些问题）

1. 层级 2×2 patchify 的**两个阶段插在 encoder 的哪两层之间**？正文与附录均未明说（复现最大歧义点）。
2. Bottleneck 的具体结构（几层？attention 还是 MLP？）沿用 HYDRA，需回读 HYDRA 原文。
3. 各损失权重 λ1/λ_perc/λ_gan/λ_KL/λ_dist 未给出；GAN 判别器结构未说明。
4. 视频生成推理时 LLM 如何自回归产出多帧 latent、flow head 如何条件化——沿用 Show-o2 类模板但细节未展开。
5. 理解路径喂给 LLM 的是 s（Sem-ViT 输出），生成路径 flow head 预测的目标是什么（z 还是 s）？从 HYDRA 范式推断 LLM 在 s 空间操作、decoder 从 z 重建，生成时 flow 目标应为 z（经归一化）——需以 HYDRA 论文佐证。
6. VideoMME 60.0 仍低于专用视频 LMM（Qwen2.5-VL 等）；DD（Dynamic Degree）只有 35.42，视频动态性偏弱（表 13）——统一化的代价。
7. 数据配方大量依赖内部数据（100M 图文对、10M 视频、6M 合成图），社区无法完全对齐。

---

# 第二部分：完整复现方案

## 0. 总体策略：四级复现路线

论文全量复现需 ~70,000 GPU·h（256~512 张 GPU）+ 大量内部数据，个人/小团队不现实。方案按验证价值分四级，**每一级都独立可交付、可验证论文的部分核心 claim**：

| 级别  | 目标               | 验证的 claim                                   | 硬件          | 时间         |
| --- | ---------------- | ------------------------------------------- | ----------- | ---------- |
| T0  | 小规模 tokenizer 消融 | 表 1（tubelet > causal > full；层级 > 单步）、表 2 趋势 | 8×A100/H800 | 2~4 周      |
| T1  | 完整 Hydra-XTok 复现 | 表 9（重建对标 Wan2.2/AToken）                     | 8~32×A100   | 4~8 周      |
| T2  | 1.5B UMM（论文消融规模） | 表 2 全部、表 3（STI +7dB）                        | 32~64×A100  | 6~10 周     |
| T3  | 7B 全量            | 表 4~7 主结果                                   | 256~512 GPU | 数月，需数据工程团队 |

**建议执行顺序：T0 → T1 → T2；T3 视资源决定。** 下文按模块给出实现细节。

## 1. 代码底座与依赖

```
基础框架   : PyTorch 2.x + FSDP（或 DeepSpeed ZeRO-2/3）+ flash-attention 2
             （tubelet mask 用 flex_attention 或 block-diagonal varlen 实现最高效）
tokenizer  : 自研（无开源可直接用）；GAN/LPIPS 部分借鉴
             CompVis/taming-transformers 与 Stability latent-diffusion 的
             VQGAN 判别器 + LPIPS 实现
UMM        : 底座参考 Show-o2（github.com/showlab/Show-o，Apache-2.0，
             含 flow-matching 视觉头 + Qwen 系 LLM 的完整训练代码）；
             或 BAGEL 的开源代码做交错序列模板
评测       : lmms-eval（理解类基准）、GenEval 官方、WISE 官方、
             VBench 官方、ImgEdit-Bench / GEdit-Bench 官方脚本、
             pytorch-fid / common_metrics_on_video_quality（rFID/rFVD）
```

**预训练权重（HuggingFace）**

- `google/siglip2-so400m-patch16-naflex` —— Gen-ViT/Sem-ViT 初始化 + 图像教师（同一权重两用）
- 视频教师 InternVideo-Next-L（OpenGVLab）；**若未开源，降级方案**：`OpenGVLab/InternVideo2-Stage2_1B`（或 6B 蒸馏版）、`OpenGVLab/VideoMAEv2-Large`、UMT-L。教师换了绝对分数会变，但表 2 的**趋势**（video teacher > img teacher > 无）应当保持——这是复现要验证的东西。
- `Qwen/Qwen2.5-1.5B` / `Qwen/Qwen2.5-7B-Instruct`

## 2. Hydra-XTok 模型实现（核心工程）

### 2.1 张量流与形状（以 T=16, H=W=256, patch=16 为例）

```
输入 clip x: [3, 1+16, 256, 256]        # 1 锚帧 + 16 帧
├─ 空间 patchify 16×16 → 每帧 16×16=256 token
├─ 时间 patchify 阶段①(2×): 锚帧 zero-pad 配对; 16 帧→8 时间位
│    → token 数 (1+8)×256
├─ Gen-ViT blocks（前半，tubelet causal attn + 3D RoPE）
├─ 时间 patchify 阶段②(2×): 8→4 时间位 → (1+4)×256 token
├─ Gen-ViT blocks（后半）→ h
├─ Bottleneck: 线性投影到 2C=128 → 拆成 (μ, ρ) → 重参数化采样
│    → z: [64, 1+4, 16, 16]             # C=64, 空间16×, 时间4×
├─ Sem-ViT（tubelet causal attn）→ s: [(1+4)×256, D_sem]
│    ├─ s_0 (锚帧 token) ──d_cos──> SigLIP2(x_0)          # 图像蒸馏
│    └─ s_1: ──Decompressor(4×时间上采样)──d_cos──> InternVideo(x_1:)  # 视频蒸馏
└─ Decoder（27 层 ViT，对称结构，含两个 2× 时间 un-patchify）→ x̂
```

纯图像输入即 T=0：只有锚帧，无时间 patchify 生效，z: [64,1,16,16]，蒸馏只有图像项。

### 2.2 关键组件实现要点

**(a) Tubelet 因果注意力 mask**

- 定义：时间位 t 的 token 可注意 {t−1, t} 两个时间位的全部空间 token（帧内全双向，跨帧只看紧邻前一位，锚帧只看自己）。
- 实现：flex_attention 的 `mask_mod`，或按时间位构造 block mask；训练时图像 batch 与视频 batch 分开组（避免 mask 混杂）。
- **歧义点**：patchify 之后"帧"变成"时间位"（2/4 帧折叠成 1 位），mask 以折叠后的时间位为单位——从图 2 与延迟数据（tubelet 0.17s 远低于 full 0.49s）推断如此。

**(b) 层级时间 patchify**

- 每阶段：沿时间维把相邻 2 个时间位在通道维拼接 + 线性投影（time-to-channel，等价 Conv3d(kernel_t=2, stride_t=2)）。
- 锚帧 zero-pad：给锚帧配一个全零"影子帧"一起折叠，保证与视频帧经历相同算子。
- **插入位置论文未明说**。默认方案：阶段①在空间 patch embed 之后立刻做，阶段②在 encoder 深度 1/2 处。T0 阶段把插入位置作为附加消融（两阶段位置 {0, L/3, L/2}）跑一遍，选重建最优的。
- SigLIP2 权重加载：时间 patchify 是新增层（随机初始化），空间 patch embed 与 transformer blocks 直接继承；3D RoPE 替换原位置编码（空间两轴继承插值，时间轴新增）。

**(c) Bottleneck（VAE 头）**

- 由 KL 项形式（式 7 含 μ、log-var ρ）确定为对角高斯重参数化：`Linear(D_vit → 128)` → split → `z = μ + exp(ρ/2)·ε`；un-projection `Linear(64 → D_vit)` 进 Sem-ViT/decoder。细节可回读 HYDRA 原文校准。

**(d) Decompressor（≈2 层，训练后丢弃）**

```python
# 单阶段 ×2：
z: [B, T', N_s, C_d]
→ Conv1x1(C_d → 2*C_d)            # 通道翻倍
→ reshape: [B, T', N_s, 2, C_d] → [B, 2*T', N_s, C_d]   # channel-to-time
→ TransformerBlock(bidirectional, 3D RoPE)
```

输出经线性头对齐视频教师特征维度后算 cosine 蒸馏损失。

**(e) 判别器（Stage 2 GAN）**

- 论文未说明结构。默认：taming-transformers 的 PatchGAN（NLayerDiscriminator），视频帧逐帧判别 + hinge loss；GAN 权重用 LDM 的 adaptive weight 技巧稳定训练。

### 2.3 训练配置（对齐附录 A.2）

```yaml
optimizer: AdamW (betas 0.9/0.95, wd 0.05)   # betas/wd 论文未给，取 ViT 常规值
lr: 2e-4 peak, cosine decay, warmup 5k
loss_weights:                                 # 论文未给，从 LDM/VQGAN 惯例出发
  l1: 1.0
  lpips: 1.0
  kl: 1e-6 ~ 1e-8    # 低压缩 VAE 惯例，扫 3 个量级
  gan: 0.1~0.5 (仅 Stage 2, adaptive)
  dist: 0.5~1.0      # 扫 {0.25, 0.5, 1.0}，以理解基准选优
stage1: ImageNet 256² (纯图) → 混合 256² 视频 + 256~2048 图像, 共 300k it
stage2: 冻结 encoder, 只训 decoder, +GAN, ~50k it
stage3: 统计 z 的逐通道 mean/std → 归一化 → 只训 Sem-ViT, ~50k it
```

### 2.4 数据（tokenizer 阶段，全开源可得）

- 图像：ImageNet-1k（1.28M）；高分辨率补充：SA-1B 子集 / LAION-HR 子集（256~2048 混合分辨率）。
- 视频：论文未指明训练视频集。可用 Panda-70M 子集（~1M clip）或 OpenVid-1M / K710；采样 17 帧（1+16）、256²。
- 评测：ImageNet val 50k（rFID/PSNR/SSIM）、DAVIS-2017（17×256²，rFVD）、UCF-101。

## 3. T0：核心消融复现（第一个里程碑，8 卡 2~4 周）

目的：**用最小代价验证论文三个反直觉发现**。按附录 A.4 的消融协议：ImageNet-1k 训 150k it（重建），视频侧加 DAVIS/UCF 评测。

跑 6 个 tokenizer 变体（可缩到 ViT-B 规模 + 128²/256² 分辨率以省算力）：

1. Full attention + 单步 4× patchify（AToken 基线）
2. Causal attention + 单步 4×
3. Tubelet attention + 单步 4×
4. **Tubelet + 层级 2×2（Ours）**
5. Ours + 无蒸馏 vs + img 蒸馏 vs + img+Decompressor(video) 蒸馏（表 2 的 3 个关键行）
6. Sem-ViT 双向 vs tubelet（表 2 最后一行）

**验收标准（对齐趋势而非绝对值）**：

- PSNR/rFVD 排序：Ours > Tubelet > Causal > Full；层级 > 单步（同压缩率下）；
- 延迟：tubelet 显著低于 full（约 3×）；
- 蒸馏消融：video-teacher 版在小型视频理解探针（如线性 probe MVBench 子集或小 LLM 接管）上最优。

## 4. T1：完整 Hydra-XTok（表 9 复现）

- 全 So400m 规模 + 三阶段完整训练（§2.3 配置），256~2048 混合分辨率。
- 验收：ImageNet rFID < 0.6 / PSNR > 31；DAVIS PSNR ≥ 27.5、rFVD ≤ 15（允许因视频数据不同有差距，但应稳定超过 AToken 公开权重的 26.60/29.19，逼近或超过 Wan2.2 的 27.64/14.78）。
- 同时产出 Hydra-XTok†（纯 ImageNet 版）对照，验证"架构而非数据"的 claim（PSNR ≈ 32.9 @16×）。

## 5. T2：1.5B UMM 与 STI 消融（论文第二核心贡献验证）

### 5.1 架构组装

```
[text tokens | <img_start> s_1..s_N <img_end> | ...]  → Qwen2.5-1.5B →
  ├─ language head: 标准 NTP loss（文本位置）
  └─ vision head:   rectified flow matching（视觉位置）
```

- 视觉头照 Show-o2 模板：timestep embedding + 若干 transformer 层 + 线性输出，对（加噪的）视觉 latent 预测 velocity；训练目标 v = z₁ − z₀（rectified flow），噪声注入在 latent 序列上、LLM 提供上下文条件。
- 理解路径：s 经 2 层 MLP projector 进 LLM（LLaVA 惯例）。
- 生成推理：flow head 迭代去噪（~25~50 步 Euler）得 z，过冻结 decoder 出像素；视频 = 17 帧 latent（1+4 个时间位）一次生成。

### 5.2 训练（按附录 A.4 消融协议，非全量三阶段）

1. 理解：LLaVA-1.5（558k pretrain + 665k SFT）+ LLaVA-Video-178K；
2. 生成：20M 图文对（开源替代：CC12M + LAION-aesthetic 子集 + JourneyDB，配 recaption）；
3. 编辑：ImgEdit 训练集微调。

### 5.3 STI 实现与验收（表 3）

- 编辑样本路由：(x_c, x_t) 拼成长度 2 clip → Gen-ViT **关闭跨帧注意力**独立编码 → Sem-ViT 用 tubelet 因果 mask 联合编码；训练时 target 侧 latent 加噪走 FM 损失，source 侧干净。
  - **实现细节推断**：训练时 x_t 是 ground-truth 目标图；推理时没有 x_t，用纯噪声 latent 占位进 Sem-ViT（causal mask 保证 s_c 不受污染）。这是论文未明说的部分，需按此默认实现并做健全性检查（源图重建 PSNR 是否复现 +7dB）。
- 对照组 Hydra-X-Indep：同参数、同数据，仅独立编码。
- **验收**：Recon-PSNR 提升 ≥ 5dB（论文 6.9dB）；ImgEdit 分数提升 ~0.3+；GenEval 不降。

## 6. T3：7B 全量（可选，需集群）

按附录 A.3 表 8 三阶段执行。数据是主要瓶颈，开源替代配方：

- S1 100M 图文对 → LAION-2B 过滤子集 + COYO-700M 子集 + recaption（Qwen2.5-VL 重写 caption，对齐论文用 LLM rewriter 的做法）；
- S2 视频 10M → Panda-70M 过滤子集；编辑 2M → ImgEdit + UltraEdit + SEED-Data-Edit 混合；
- S3 → LLaVA-OneVision（论文同款）+ Pixmo（论文同款）+ LLaVA-Video-178K（论文同款）+ 美学过滤图 + 合成图（FLUX 蒸馏生成）。
- 训练系统：FSDP + gradient checkpointing；512² 阶段 packing 序列长度 ~8k；1024² 阶段启用变分辨率 bucketing。
- 验收（允许数据差异打折）：GenEval ≥ 0.80、MME ≥ 2100、MVBench ≥ 55、ImgEdit ≥ 3.8。

## 7. 评测基础设施（与论文完全对齐）

| 任务   | 基准（split）                                                                                                                      | 工具                               |
| ---- | ------------------------------------------------------------------------------------------------------------------------------ | -------------------------------- |
| 图像理解 | AI2D(test), MME(test), MMMU(val), OCRBench(test), MMBench(dev_en), RealWorldQA(test), ChartQA(test), DocVQA(val), InfoVQA(val) | lmms-eval                        |
| 视频理解 | MVBench, Video-MME(w/o sub), LongVideoBench(val), LVBench                                                                      | lmms-eval                        |
| 图像生成 | GenEval, WISE                                                                                                                  | 官方脚本                             |
| 视频生成 | VBench（17 帧 640×384）                                                                                                           | 官方脚本                             |
| 编辑   | ImgEdit-Bench（9 维度）, GEdit-Bench（G-SC/G-PQ/G-Over）                                                                             | 官方脚本（GPT judge，注意 judge 版本影响绝对分） |
| 重建   | ImageNet/DAVIS/UCF, 256² center-crop 统一协议                                                                                      | pytorch-fid, rFVD(I3D)           |

## 8. 里程碑时间表（8~32 卡团队）

| 周     | 交付                                                                                             |
| ----- | ---------------------------------------------------------------------------------------------- |
| 1~2   | 代码骨架：ViT+3D RoPE+tubelet mask+层级 patchify+VAE bottleneck；SigLIP2 加载；重建训练闭环（L1+LPIPS+KL）跑通 128² |
| 3~4   | T0 消融 6 变体 @ ImageNet/DAVIS → **验证发现一、二**                                                      |
| 5~6   | Decompressor + 双教师蒸馏；T0 蒸馏消融 → **验证发现三趋势**                                                     |
| 7~10  | T1：三阶段完整 tokenizer（含 GAN 精修、表征调和）→ 对标表 9                                                       |
| 11~14 | T2：1.5B UMM 训练闭环（NTP+FM 双头），LLaVA 协议理解 + 20M 生成                                                |
| 15~16 | STI vs Indep 对照 → **验证表 3（+7dB）**                                                              |
| 17+   | （可选）T3 数据工程与 7B 训练                                                                             |

## 9. 风险清单与缓解

| #   | 风险/歧义                                         | 缓解                                                       |
| --- | --------------------------------------------- | -------------------------------------------------------- |
| 1   | 两个 2× patchify 的插入层未说明                        | 作为超参消融（§2.2b）；邮件联系作者 zgzaacm@gmail.com                   |
| 2   | 损失权重 λ 全部未给                                   | 从 VQGAN/LDM 惯例出发小网格扫描；以 rFID + 理解 probe 双指标选择            |
| 3   | InternVideo-Next-L 可能未开源                      | 降级 InternVideo2/VideoMAEv2/UMT-L，验证趋势而非绝对值               |
| 4   | Bottleneck/flow head 细节在前作 HYDRA 中            | 复现前精读 HYDRA（Qiu et al. 2026）与 Show-o2，按其公开实现对齐           |
| 5   | 编辑推理时 target 占位方式未说明                          | 按 §5.3 推断实现；以源图重建 PSNR 做健全性检查                            |
| 6   | 内部数据（100M 图文/10M 视频/6M 合成）不可得                 | §6 开源替代配方；预期绝对分数下降 1~3 点，趋势结论不受影响                        |
| 7   | GAN 训练不稳定                                     | adaptive weight + R1 正则；GAN 只在 decoder 精修阶段引入（论文同款，天然更稳） |
| 8   | GPT-judge 类基准（GEdit/ImgEdit/WISE）分数随 judge 漂移 | 固定 judge 模型与版本，同 judge 下跑基线模型对照                          |

## 10. 参考实现资源

- SigLIP2: `google/siglip2-so400m-patch16-naflex`（HF）
- Show-o2: github.com/showlab/Show-o（flow-matching 头 + UMM 训练模板）
- BAGEL: github.com/ByteDance-Seed/Bagel（交错序列 + 编辑数据处理参考）
- taming-transformers / latent-diffusion：GAN 判别器、LPIPS、adaptive weight
- AToken（Lu et al. 2025）、OmniTokenizer：全时空注意力基线对照
- lmms-eval: github.com/EvolvingLMMs-Lab/lmms-eval
- VBench: github.com/Vchitect/VBench；GenEval: github.com/djghosh13/geneval
- 前作 HYDRA（Qiu et al., 2026）与 TUNA、AToken 论文——三篇必读上下文
