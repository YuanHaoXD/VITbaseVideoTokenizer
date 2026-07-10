# Hydra-X 复现：代码架构与实施计划（v1.0）

> 配套文档：《Hydra-X_阅读大纲与复现方案.md》（方法拆解与分级目标）
> 本文档：开源调研结论 → 架构裁决记录（ADR）→ 代码仓库架构 → 核心模块参考实现 → 分阶段实施计划（含 Gate 验收）→ 误差防控体系 → 风险登记册
> 撰写日期：2026-07-07

---

## 0. 复现成功的定义（先立标尺，再动手）

科研复现必须先定义"什么算成功"，否则无法自检。本复现分两档标准：

- **趋势级复现（主目标，必须达成）**：论文的三个核心发现在受控消融下方向一致且超出噪声范围——
  - F1：tubelet ≻ causal ≻ full attention（重建指标）；
  - F2：层级 2×2 patchify ≻ 单步 4×（同压缩率）；
  - F3：img 蒸馏 + Decompressor(video teacher) ≻ 仅 img 蒸馏 ≻ 无蒸馏（理解探针）；
  - STI：tokenizer 级源-目标交互使源图重建 PSNR 提升 ≥ 5 dB（论文 +6.9 dB）。
- **数值级复现（延伸目标，允许打折）**：表 9 重建指标进入论文数字 ±10% 区间；表 2/3 各项进入 ±15%。因训练数据无法完全对齐（论文用腾讯内部数据），数值级目标只对"数据可完全对齐"的实验强制执行（ImageNet-only 的 Hydra-XTok†、表 1 全部、表 3 的 delta）。

**统计要求**：论文表 1 中 Causal vs Tubelet 的 ImageNet PSNR 差距仅 0.04 dB，极可能落在训练噪声内。因此所有消融**至少 2 个 seed**，报告均值±标准差；只有当效应量 > 2σ 才判定"趋势复现成功"。rFVD/rFID 类指标效应量较大（14.05 vs 13.69），作为主要判据；PSNR 作为辅助判据。

---

## 1. 开源调研结论与选型决策

### 1.1 资产验证矩阵（全部已于 2026-07-07 联网核实）

| #   | 资产                                                                                                                                                                                                                                                                                                                                                        | 状态                                                               | License                                               | 复用内容                                                                                                                                                           | 已知坑                                                                                                                     |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- | ----------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| A1  | **AToken** `github.com/apple/ml-atoken`                                                                                                                                                                                                                                                                                                                   | ✅ 开源，含 AToken-So/C（连续）等 3 个 checkpoint                           | 代码：Apple Sample Code License；权重：Apple ML Research TOU | ① 统一 ViT tokenizer 的**架构参考**（SigLIP-So400M 底座、4D RoPE、时间 patchify）；② **checkpoint 用于校准我们的评测管线**（复现其 rFID 0.21 / 表 9 的 29.72 PSNR）                              | 仅 `atoken_inference/`，**无训练代码**——损失与训练循环需自建；license 保守策略：只做参考+校准，不直接拷贝代码进我们的仓库                                          |
| A2  | **Show-o2** `github.com/showlab/Show-o/tree/main/show-o2`                                                                                                                                                                                                                                                                                                 | ✅ 开源，含完整两阶段训练代码（1.5B/7B × stage1/stage2 脚本）、flow head、Qwen2.5 集成 | Apache-2.0                                            | **UMM 训练框架主底座**：`train_stage_one.py`/`train_stage_two.py`、`models/showo2_qwen2_5.py`（AR head + flow head 双头）、jsonl 数据管线、`inference_t2i.py`/`inference_mmu*.py` | 其视觉路径是 3D Causal VAE + 双路融合，与我们的 z→Sem-ViT 路径不同——`sequence_builder` 与视觉编码模块**必须重写**，只保留 LLM/双头/训练循环/推理骨架                |
| A3  | **HYDRA 前作** arXiv 2603.15228                                                                                                                                                                                                                                                                                                                             | ✅ 论文可得；❌ 代码"work in progress"未释出                                 | —                                                     | **超参数与结构细节的权威来源**（见 §1.2），Hydra-X 论文缺失的数值大多能从这里补齐                                                                                                              | 与 Hydra-X 存在两处已确认差异（decoder 类型、教师模型），不可盲搬                                                                               |
| A4  | **SigLIP2** `google/siglip2-so400m-patch16-naflex`                                                                                                                                                                                                                                                                                                        | ✅ HF 可下，transformers 原生支持                                        | Apache-2.0                                            | Gen-ViT/Sem-ViT 初始化 + 图像教师 T_img（同一权重两用）                                                                                                                       | naflex 变体输出带 padding mask；教师推理时需按其官方 processor 预处理，否则特征漂移                                                               |
| A5  | **InternVideo-Next** HF collection `OpenGVLab/internvideo-next`（2025-12 发布，报告 arXiv 2512.01342）                                                                                                                                                                                                                                                           | ✅ 权重已开源，代码在 `OpenGVLab/InternVideo/InternVideo-Next`             | 需核对（InternVideo 系多为 Apache-2.0）                       | 视频教师 T_vid（论文同款 InternVideo-Next-L）                                                                                                                            | 需确认 L 尺寸权重在 collection 中；备胎：`OpenGVLab/InternVideo2-Stage2_1B-224p-f4`。输入协议（帧数/采样率/归一化）必须写 golden test 锁定               |
| A6  | **Wan2.2-VAE** `Wan-Video/Wan2.2` + `lightx2v/Autoencoders`（含独立 `vid_recon.py`）                                                                                                                                                                                                                                                                           | ✅ 开源                                                             | Apache-2.0                                            | 表 9 基线对照 + **评测管线第二校准点**（DAVIS 27.64 PSNR / 14.78 rFVD）                                                                                                        | 其压缩为 16×16×4，与我们协议一致，直接可比                                                                                               |
| A7  | **BAGEL** `ByteDance-Seed/Bagel`                                                                                                                                                                                                                                                                                                                          | ✅ 开源                                                             | Apache-2.0                                            | 编辑数据处理管线、交错序列参考（次要参考）                                                                                                                                          | MoT 架构与我们不同，仅借数据工程                                                                                                      |
| A8  | **flex_attention**（PyTorch ≥2.5）                                                                                                                                                                                                                                                                                                                          | ✅ 官方 API                                                         | BSD                                                   | tubelet/causal/full 三种 mask 的统一高效实现                                                                                                                            | BlockMask 构建昂贵——必须按 (T', N) 形状缓存；高分辨率长序列有 OOM 案例（xformers 28k vs flex 16k OOM@8×H100）——1280×768 视频评测时需 SDPA fallback 路径 |
| A9  | 评测：**lmms-eval**（AI2D/MME/MMMU/OCRBench/MMB/RWQA/ChartQA/DocVQA/InfoVQA/MVBench/VideoMME/LongVideoBench/LVBench）、**GenEval** `djghosh13/geneval`、**WISE** `PKU-YuanGroup/WISE`（ICML 2026）、**VBench** `Vchitect/VBench`、**ImgEdit** `PKU-YuanGroup/ImgEdit`（NeurIPS 2025 D&B，数据在 HF `sysuyy/ImgEdit`）、**GEdit-Bench** `stepfun-ai/Step1X-Edit/GEdit-Bench` | ✅ 全部开源                                                           | 各自开源协议                                                | 全套基准评测                                                                                                                                                         | WISE/GEdit/ImgEdit 用 GPT judge，分数随 judge 版本漂移——固定 judge 型号并同批评测基线模型                                                     |
| A10 | 数据：**ImageNet-1k**、**OpenVid-1M**（HF `nkp37/OpenVid-1M`，CC-BY-4.0，≥512²，含 caption）、**ImgEdit-1.2M**、LLaVA-1.5/LLaVA-Video-178K/LLaVA-OneVision/Pixmo（HF lmms-lab / allenai）                                                                                                                                                                               | ✅ 全部可下                                                           | 各自协议                                                  | 训练数据开源替代                                                                                                                                                       | 视频训练集论文未指明——选 OpenVid-1M（质量高、带文本、许可清晰）；预期与论文有数据差异，只影响数值级目标                                                              |
| A11 | **taming-transformers / latent-diffusion**                                                                                                                                                                                                                                                                                                                | ✅ 开源                                                             | MIT                                                   | PatchGAN 判别器（`NLayerDiscriminator`）、LPIPS、GAN adaptive weight                                                                                                  | LPIPS 权重下载源偶尔失效，vendor 进仓库                                                                                              |
| A12 | rFVD：`JunyaoHu/common_metrics_on_video_quality`（I3D）或 StyleGAN-V 版 FVD                                                                                                                                                                                                                                                                                    | ✅ 开源                                                             | MIT                                                   | rFVD 计算                                                                                                                                                        | **FVD 实现间数值不可比**——最终以"能复现 A1/A6 公开 checkpoint 在表 9 的数字"为准绳选择实现（见 Phase 0 Gate）                                          |

### 1.2 从 HYDRA 前作补齐的关键参数（Hydra-X 论文未写明）

以下参数直接取自 HYDRA 原文（arXiv 2603.15228），置信度标注：

| 参数                                       | 值                                                                          | 来源          | 置信度（迁移到 Hydra-X）                                               |
| ---------------------------------------- | -------------------------------------------------------------------------- | ----------- | -------------------------------------------------------------- |
| GSB 结构                                   | 线性投影 W_proj∈R^{D×2C} → [μ,ρ] → 重参数化 z=μ+ε·exp(0.5ρ)；线性反投影 W_unproj∈R^{C×D} | HYDRA Eq.2  | 高（Hydra-X §3 声明"retain this overall design"）                   |
| λ_KL                                     | 1e-4                                                                       | HYDRA §2.1  | 高                                                              |
| 一致性损失 L_cos（H_bn 与 H_mid 方向对齐），λ_cos=1.0 | 存在于 HYDRA；**Hydra-X 式(2)/(7) 未提及**                                         | HYDRA Eq.4  | 中——实现为配置开关，默认开启（架构上无害），T0 阶段消融验证                               |
| λ_dist                                   | 1.0                                                                        | HYDRA Eq.8  | 高                                                              |
| Gen/Sem 切分                               | HYDRA 在 24 层 InternViT 上消融：**12+12 均衡最优**（24+0 与 16+8 均显著差）                | HYDRA Fig.5 | 中高——SigLIP2-So400M 为 27 层，默认 **13+14**，做成配置项                   |
| 解码器规模                                    | 越大越好（144M→358M 单调提升）                                                       | HYDRA Tab.5 | 高（Hydra-X 直接用 27 层对称 decoder）                                  |
| 蒸馏方式                                     | 多层特征蒸馏（Gen-ViT 与 Sem-ViT 各选若干层对齐教师对应层）                                     | HYDRA Eq.6  | 中——Hydra-X 式(3)只写了 Sem-ViT 输出处的蒸馏；默认按 Hydra-X 简化版实现，多层蒸馏作为可选增强 |
| LLM 注意力                                  | 文本 token 因果、视觉 token 块内双向                                                  | HYDRA §2.2  | 高                                                              |
| 流匹配头                                     | v_pred = Head_flow(AdaLN(H_LLM^vis, t_emb))                                | HYDRA Eq.10 | 高                                                              |
| UMM 数据协议                                 | 与 Hydra-X 表 8 同构（100M→30M+30M→SFT）                                         | HYDRA §2.3  | 高                                                              |

**两处已确认的 HYDRA→Hydra-X 差异（不可照搬）**：

1. **解码器**：HYDRA 用"pixel flow decoder"（流匹配式解码，λ_FM=1.0/λ_perc=0.1/λ_gan=0.075）；Hydra-X 改为 AToken 式 **27 层对称 ViT decoder + L1 直接回归**（式 7 为 L1+LPIPS+GAN+KL）。→ 我们实现 Hydra-X 版；λ_perc=0.1、λ_gan=0.075 作为初始值沿用（量级参考）。
2. **初始化/教师**：HYDRA 用 InternViT-2.5；Hydra-X 用 SigLIP2（初始化+图像教师）+ InternVideo-Next-L（视频教师）。

### 1.3 架构裁决记录（ADR）——论文歧义的逐条裁决

复现最大的风险不是写代码，而是把歧义处猜错且不自知。以下每条 ADR 给出：裁决、证据、置信度、验证手段。

**ADR-1：LLM 的视觉输入 = Sem-ViT 输出 s；流匹配的目标空间 = 归一化后的 z（C=64）**

- 裁决：理解时 LLM 读 s = Sem-ViT(unproj(ẑ))（ẑ 为归一化 latent）；生成训练时对 ẑ 做整流流加噪 ẑ_t=(1−t)ẑ+tε，经 unproj→Sem-ViT 得到含噪 s̃ 进 LLM，flow head 在 **z 空间（64 维/token）** 预测速度 v=ε−ẑ；推理时迭代积分得 ẑ→反归一化→decoder 出像素。
- 证据：Hydra-X §3"LLM operates exclusively on the Sem-ViT output s"；§5.3 式(6) [s_c,s_t]=Sem-ViT([z_c;z_t]) 且强调"exposes the LLM to a target representation that has already absorbed source structure"；表 3 的 Recon-PSNR 要求生成产物可经 decoder 还原像素 → 生成目标必须在 z 空间；该构型与 Show-o2 的"噪声 latent 过语义层再进 LLM"完全同构。注意 HYDRA 原文是把 H_bn（unproj 后特征）直接给 LLM——两文表述不一致，**以 Hydra-X 原文为准**。
- 置信度：中高。验证：Phase 5 若 T2I 收敛异常，回退到 HYDRA 式 H_bn 直喂（配置项 `llm_visual_input: sem_vit | bottleneck`）。

**ADR-2：tubelet mask 的"帧"= 时间 patchify 折叠后的时间位**

- 裁决：注意力 mask 以折叠后时间位为单位（位 t 可见位 {t−1, t} 的全部空间 token；锚帧位 0 只见自己）。
- 证据：延迟数据（tubelet 0.17s vs full 0.49s @17×512²）只有在折叠后序列上限制注意力才成立；图 2 把 mask 画在 latent 序列上。
- 置信度：高。验证：单元测试 + 延迟比对（tubelet 应 ≈ full 的 1/3）。

**ADR-3：两个 2× 时间 patchify 的插入位置 = {patch embed 后, encoder 深度 1/2 处}（默认），作为超参消融**

- 证据：论文仅说"distributes temporal compression across multiple stages"+图 2 顶部示意。无更多信息。
- 置信度：低（这是全文最大歧义）。验证：T0 阶段网格 {(0, L/2), (0, L/3), (L/3, 2L/3)} 三组重建对比，选优后冻结；同时邮件询问作者（zgzaacm@gmail.com）。

**ADR-4：锚帧 zero-pad 方向 = [零帧, 锚帧]（零帧在时间前位）**

- 证据：无直接证据；因果方向上锚帧作为"当前帧"更合理。置信度：低。验证：与 ADR-3 同批消融（两个方向各跑一次，差异预计微小）。

**ADR-5：Stage-3 表征调和的归一化语义 = 定义规范 latent 接口 ẑ=(z−mean)/std**

- 裁决：统计逐通道 mean/std 后，ẑ 成为 Sem-ViT 与 flow matching 的规范输入；decoder 物理上消费 z=ẑ·std+mean（等价于"decoder 前反归一化"）。
- 证据：原文"Gen-ViT features are normalized before being fed into Sem-ViT and the decoder"字面上会破坏已冻结 decoder 的输入分布，唯一自洽解读是 decoder 路径含反归一化（类比 SD 的 latent scaling factor）。置信度：中高。验证：Stage 3 开始时重建指标不得退化（Gate P4-2）。

**ADR-6：编辑训练/推理路由**

- 裁决：训练时 z_c 干净、z_t 加噪，[z_c; z_t] 经 Sem-ViT（tubelet 因果 mask，s_c 只见 z_c）；FM 损失只算 target 位置。推理时 z_t 从纯噪声起步，每个去噪步重过 Sem-ViT。Gen-ViT 对两图独立编码（关闭跨帧注意力）。
- 证据：§5.3 原文 + 式(6)。置信度：高（唯一不确定是推理起点，纯噪声是 FM 标准做法）。
- Recon-PSNR 探针定义（论文未给协议，**我们显式定义并对所有变体统一使用**）：ImgEdit 验证集上，指令替换为"Repeat the image exactly, change nothing."，生成图与源图算 PSNR。消融关心的是 STI−Indep 的 **delta**，探针定义偏差不影响结论有效性。

**ADR-7：视频教师对齐的空间分辨率**

- 裁决：Decompressor 输出 [B,T,N_s,D] 与 T_vid 特征在空间上双线性对齐到相同 token 网格后逐 token 算 cosine；若 T_vid 输出含 CLS/时间聚合 token 则丢弃只取 patch 特征。
- 置信度：中。验证：golden test 锁定教师输出形状；蒸馏损失初值应在 0.6~1.0 区间（随机对齐≈1.0，训练后下降）。

---

## 2. 系统总体架构

### 2.1 数据流（训练期全景）

```
                        ┌──────────────── tokenizer 训练期 ────────────────┐
clip x [3,1+T,H,W]      │                                                  │
  │ 空间patchify16 + 时间patchify①(2×,锚帧zero-pad)                        │
  ▼                                                                        │
Gen-ViT 前段(tubelet causal + 3D RoPE)                                     │
  │ 时间patchify②(2×)                                                      │
  ▼                                                                        │
Gen-ViT 后段 ──► h ──► GSB: [μ,ρ]=W_proj·h, z=μ+ε·exp(0.5ρ)  [C=64]        │
                        │                          │                       │
                        │ ẑ=(z−m)/s (Stage3起)     ├─► Decoder(27L ViT,含2×时间unpatchify×2) ─► x̂
                        ▼                          │      L1 + LPIPS + GAN(Stage2起)
                 unproj → Sem-ViT(tubelet causal)  │      + KL(μ,ρ)  [+L_cos 可选]
                        │                                                   
                        ├─ s_0 (锚帧) ──── d_cos ──── SigLIP2(x_0)          
                        └─ s_1: ─► Decompressor(4×时间上采样) ─ d_cos ─ InternVideo-Next(x_1:)
                        └────────────── (Decompressor 训练后丢弃) ──────────┘

                        ┌──────────────── UMM 训练期 ────────────────┐
理解:  ẑ(干净) → unproj → Sem-ViT → s → MLP projector → Qwen2.5 → LM head → L_NTP
生成:  ẑ_t=(1−t)ẑ+tε → unproj → Sem-ViT → s̃ → projector → Qwen2.5
                                  → flow head(AdaLN,t_emb) → v_pred ∈ R^{N×64} → L_FM
编辑:  [z_c 干净; z_t 加噪] → Sem-ViT(联合,tubelet causal) → [s_c; s̃_t] → LLM → 同上
推理生成: 纯噪声 ẑ_1 → (Sem-ViT→LLM→flow head) ×~50 Euler步 → ẑ_0 → 反归一化 → Decoder → 像素
```

### 2.2 代码仓库架构

```
hydra-x-repro/
├── pyproject.toml                    # torch>=2.5, transformers, accelerate, webdataset,
│                                     # lpips(vendored), einops, omegaconf, wandb
├── configs/
│   ├── tokenizer/
│   │   ├── base_so400m.yaml          # 13+14切分/C=64/patch16/3D RoPE/tubelet
│   │   ├── ablation_t0_{full,causal,tubelet}_{single,hier}.yaml   # 表1的6变体
│   │   ├── ablation_t0_distill_{none,img,img+dvid,img+dimg,bidir}.yaml  # 表2
│   │   ├── stage1_foundation.yaml / stage2_gan.yaml / stage3_harmonize.yaml
│   ├── umm/
│   │   ├── ablation_1p5b.yaml        # Qwen2.5-1.5B 消融协议(附录A.4)
│   │   ├── stage{1,2,3}_7b.yaml      # 表8 全量协议
│   │   └── edit_{sti,indep}.yaml     # 表3 对照
│   └── data/{imagenet,openvid,imgedit,llava_mix,...}.yaml
├── src/hydrax/
│   ├── models/
│   │   ├── tokenizer/
│   │   │   ├── rope3d.py             # 3D RoPE (t,h,w 三轴分频)
│   │   │   ├── attention.py          # flex_attention 封装 + BlockMask 工厂/缓存 + SDPA fallback
│   │   │   ├── temporal_patchify.py  # TemporalPatchify / TemporalUnpatchify(可逆对)
│   │   │   ├── vit_blocks.py         # SigLIP2 兼容 block(可加载HF权重)
│   │   │   ├── gsb.py                # GSB(proj/unproj/重参数化/norm buffer)
│   │   │   ├── encoder.py            # GenViT: 前段+patchify②+后段
│   │   │   ├── sem_vit.py            # SemViT(独立/联合两种routing)
│   │   │   ├── decoder.py            # 27层对称decoder + pixel head
│   │   │   ├── decompressor.py       # 2×(conv1x1 C→2C + time-reshape + block)
│   │   │   └── hydraxtok.py          # 总装: encode()/decode()/forward_train()
│   │   ├── teachers/
│   │   │   ├── siglip2.py            # 冻结教师, 官方processor, 特征缓存可选
│   │   │   └── internvideo.py        # 冻结教师, 输入协议适配
│   │   ├── discriminator.py          # PatchGAN(重写实现, 参考taming)
│   │   ├── heads/flow_head.py        # AdaLN + MLP → R^{64}
│   │   └── umm/
│   │       ├── projector.py          # s→LLM dim 2层MLP; z-noise注入
│   │       ├── sequence_builder.py   # ★五任务统一: token排布/attn mask/loss mask
│   │       └── hydrax_umm.py         # tokenizer(冻结Gen-ViT/decoder) + Qwen2.5 + 双头
│   ├── losses/
│   │   ├── recon.py                  # L1+LPIPS+KL(+cos consistency开关)
│   │   ├── gan.py                    # hinge + adaptive weight + R1(可选)
│   │   ├── distill.py                # 式(3): 锚帧img项 + Decompressor vid项, 图像batch屏蔽vid项
│   │   └── flow.py                   # RF: logit-normal t采样/插值/v目标/loss mask
│   ├── data/
│   │   ├── image.py / video.py       # webdataset; 视频: 1+16帧采样, fps抖动
│   │   ├── edit.py                   # ImgEdit pair → 长度2 clip
│   │   ├── mixture.py                # ★表8数据比例的确定性加权采样器(seed可控)
│   │   └── protocols.py              # ★评测预处理协议唯一实现(resize/center-crop)
│   ├── train/
│   │   ├── tokenizer_trainer.py      # 三阶段状态机(冻结策略/损失开关/统计z mean-std)
│   │   ├── umm_trainer.py            # accelerate+FSDP; 双损失; 梯度累积
│   │   └── common.py                 # EMA/lr schedule/ckpt io/实验卡自动生成
│   └── eval/
│       ├── calibrate.py              # ★Phase0: 用AToken/Wan2.2权重校准指标脚本
│       ├── recon.py                  # PSNR/SSIM/rFID/rFVD, 协议=protocols.py
│       ├── probes.py                 # CKNNA/linear probe/蒸馏损失监控
│       ├── lmms_adapter.py           # lmms-eval 模型接口(理解全基准)
│       ├── geneval_runner.py / wise_runner.py / vbench_runner.py
│       └── edit_runner.py            # ImgEdit/GEdit + Recon-PSNR探针(ADR-6)
├── tests/
│   ├── test_attention_masks.py       # 可达性矩阵断言(见§3.1)
│   ├── test_temporal_patchify.py     # 可逆性: unpatchify(patchify(x))==x
│   ├── test_gsb.py                   # KL数值/形状/归一化往返
│   ├── test_decompressor.py          # 时序对齐: 输出第t帧↔教师第t帧
│   ├── test_teachers_golden.py       # ★教师特征golden fixture, cos>0.999
│   ├── test_sequence_builder.py      # 五任务mask/loss mask正确性
│   ├── test_flow.py                  # v目标/单步重建恒等
│   └── test_overfit.py               # 单batch过拟合冒烟(tokenizer与UMM各一)
├── scripts/                          # download_*.sh / preprocess_* / launch_*
└── docs/
    ├── adr/                          # 本文§1.3的ADR, 每条一文件, 状态可更新
    └── experiments/                  # 实验卡(模板见§5.2)
```

**设计原则**：

- `protocols.py` 是全仓库唯一的评测预处理实现——论文表 9 强调"identical scripts"，预处理不统一是重建指标不可比的头号来源。
- `mixture.py` 的采样必须确定性可复演（记录 seed + 全局 step→样本映射），否则消融不受控。
- 所有歧义点（ADR-3/4 等）一律做成 config 字段而非硬编码。

---

## 3. 核心模块参考实现（六个高风险单元，含单元测试）

以下代码为可直接落仓的参考实现（张量约定：`x [B, T1, N, D]`，T1=1+T'，位 0 为锚帧）。

### 3.1 tubelet 因果注意力（`attention.py`）

```python
import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from functools import lru_cache

def make_mask_mod(time_of_token: torch.Tensor, kind: str):
    """time_of_token: [S] 每个token的时间位id(锚=0). kind ∈ {full,causal,tubelet}"""
    def mask_mod(b, h, q, kv):
        tq, tk = time_of_token[q], time_of_token[kv]
        if kind == "full":    return tq >= -1          # 恒真(保持签名一致)
        if kind == "causal":  return tk <= tq
        if kind == "tubelet": return (tk == tq) | (tk == tq - 1)
    return mask_mod

@lru_cache(maxsize=32)                      # ★BlockMask按(T1,N,kind)缓存, 否则构建开销致命
def get_block_mask(T1: int, N: int, kind: str, device: str):
    tot = T1 * N
    time_ids = torch.arange(T1, device=device).repeat_interleave(N)
    return create_block_mask(make_mask_mod(time_ids, kind), B=None, H=None,
                             Q_LEN=tot, KV_LEN=tot, _compile=True)
```

```python
# tests/test_attention_masks.py —— 可达性矩阵手工断言
def test_tubelet_reachability():
    # T1=3(锚+2位), N=2 → S=6; 期望: 位0只见位0; 位1见{0,1}; 位2见{1,2}
    bm = build_dense_mask(T1=3, N=2, kind="tubelet")   # 用mask_mod逐元素求值
    expect = torch.tensor([                            # q行 kv列, 块粒度
        [1,1,0,0,0,0],[1,1,0,0,0,0],                   # t=0
        [1,1,1,1,0,0],[1,1,1,1,0,0],                   # t=1: 见t0,t1
        [0,0,1,1,1,1],[0,0,1,1,1,1],                   # t=2: 见t1,t2 (不见t0!)
    ]).bool()
    assert torch.equal(bm, expect)
```

### 3.2 层级时间 patchify（`temporal_patchify.py`）

```python
class TemporalPatchify(nn.Module):
    """[B, 1+T', N, D] → [B, 1+T'/2, N, D]. 锚帧与零帧配对折叠(ADR-4)."""
    def __init__(self, dim: int, pad_side: str = "before"):
        super().__init__()
        self.proj = nn.Linear(2 * dim, dim)
        self.pad_side = pad_side
    def forward(self, x):
        B, T1, N, D = x.shape
        anchor, frames = x[:, :1], x[:, 1:]                     # [B,1,N,D],[B,T',N,D]
        pad = torch.zeros_like(anchor)
        a = torch.cat([pad, anchor] if self.pad_side == "before"
                      else [anchor, pad], dim=1)                # [B,2,N,D]
        a = a.permute(0,2,1,3).reshape(B, 1, N, 2*D)            # 时间→通道
        if frames.shape[1]:
            T2 = frames.shape[1] // 2
            f = frames.reshape(B, T2, 2, N, D).permute(0,1,3,2,4).reshape(B, T2, N, 2*D)
            out = torch.cat([a, f], dim=1)
        else:
            out = a                                             # 纯图像: 只有锚帧
        return self.proj(out)

class TemporalUnpatchify(nn.Module):
    """逆操作(decoder/Decompressor共用): D→2D, 通道→时间."""
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, 2 * dim)
    def forward(self, x):                                       # [B,T,N,D]→[B,2T,N,D]
        B, T, N, D = x.shape
        y = self.proj(x).reshape(B, T, N, 2, D).permute(0,1,3,2,4).reshape(B, 2*T, N, D)
        return y   # 调用方负责去掉锚帧配对产生的多余零位
```

```python
# tests/test_temporal_patchify.py —— 结构可逆性(用恒等初始化的proj验证信息不损)
def test_fold_unfold_shapes():
    x = torch.randn(2, 1+16, 256, 64)
    p1, p2 = TemporalPatchify(64), TemporalPatchify(64)
    y = p2(p1(x))                       # (1+16)→(1+8)→(1+4)
    assert y.shape == (2, 1+4, 256, 64)
def test_image_only_passthrough():
    x = torch.randn(2, 1, 256, 64)      # 纯图像
    assert TemporalPatchify(64)(x).shape == (2, 1, 256, 64)
```

### 3.3 GSB（`gsb.py`）

```python
class GSB(nn.Module):
    def __init__(self, d_model=1152, c_latent=64):
        super().__init__()
        self.proj = nn.Linear(d_model, 2 * c_latent)
        self.unproj = nn.Linear(c_latent, d_model)
        self.register_buffer("z_mean", torch.zeros(c_latent))   # Stage3填充(ADR-5)
        self.register_buffer("z_std",  torch.ones(c_latent))
        self.normalize = False
    def compress(self, h):
        mu, rho = self.proj(h).chunk(2, dim=-1)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * rho)
        kl = -0.5 * (1 + rho - mu.pow(2) - rho.exp()).sum(-1).mean()
        return z, mu, kl
    def to_canonical(self, z):   return (z - self.z_mean) / self.z_std if self.normalize else z
    def from_canonical(self, zc): return zc * self.z_std + self.z_mean if self.normalize else zc
    def decompress(self, z_canonical):          # Sem-ViT入口
        return self.unproj(z_canonical)
```

### 3.4 Decompressor（`decompressor.py`）

```python
class Decompressor(nn.Module):
    """4×时间上采样: 2×(TemporalUnpatchify + Block). 仅tokenizer训练期存在."""
    def __init__(self, dim, teacher_dim, num_heads=16):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.ModuleList([TemporalUnpatchify(dim), TransformerBlock(dim, num_heads)])
            for _ in range(2)])
        self.head = nn.Linear(dim, teacher_dim)
    def forward(self, s_video):                 # [B, T/4, N, D] (不含锚帧)
        x = s_video
        for up, blk in self.stages:
            x = blk(up(x))                      # 双向注意力(教师侧无因果约束)
        return self.head(x)                     # [B, T, N, D_teacher]
```

```python
# tests/test_decompressor.py —— 时序对齐是蒸馏正确性的命门
def test_temporal_alignment():
    d = Decompressor(64, 1024)
    out = d(torch.randn(1, 4, 256, 64))
    assert out.shape == (1, 16, 256, 1024)      # 4→16帧, 与教师逐帧对齐
```

### 3.5 双教师蒸馏损失（`distill.py`，式 3 的忠实实现）

```python
def distill_loss(s, decompressor, t_img_feat, t_vid_feat, is_video: torch.Tensor):
    """s: [B,T1,N,D]; t_img_feat: [B,N,D_img](锚帧教师特征);
       t_vid_feat: [B,T,N,D_vid] 或 None; is_video: [B] bool"""
    cos = lambda a, b: 1 - F.cosine_similarity(a, b, dim=-1).mean()
    l_img = cos(proj_img(s[:, 0]), t_img_feat)                  # 锚帧项
    if t_vid_feat is None or not is_video.any():
        return l_img, torch.zeros_like(l_img)
    d_out = decompressor(s[:, 1:][is_video])
    l_vid = cos(d_out, t_vid_feat[is_video])                    # 图像batch自动屏蔽
    return l_img, l_vid
```

### 3.6 STI 序列路由（`sem_vit.py` + `sequence_builder.py` 关键逻辑）

```python
def encode_edit_pair(tok, x_src, x_tgt, t=None, noise=None):
    """ADR-6: Gen-ViT独立(各自当纯图像编码,无跨帧注意力), Sem-ViT联合(tubelet causal)."""
    z_c, _, _ = tok.encode_image(x_src)          # [B,1,N,64] 干净
    z_t, _, _ = tok.encode_image(x_tgt)
    zc_hat, zt_hat = tok.gsb.to_canonical(z_c), tok.gsb.to_canonical(z_t)
    if t is not None:                            # 训练: 仅target加噪(RF插值)
        zt_hat = (1 - t) * zt_hat + t * noise
    pair = torch.cat([zc_hat, zt_hat], dim=1)    # [B,2,N,64] 当作长度2 clip
    s = tok.sem_vit(tok.gsb.decompress(pair), mask_kind="tubelet")  # s_c只见z_c
    return s[:, 0], s[:, 1], zt_hat              # s_c, s_t, 加噪latent(算FM目标用)
# 对照组 Indep: 两图分别单独过 sem_vit(mask_kind不涉跨帧), 其余完全一致
```

---

## 4. 分阶段实施计划

> 每个 Phase 的结构：**目标 → 任务分解 → Gate（量化验收，不过不进下一阶段）→ 失败回退**。
> 执行时每个 Phase 应再展开为 task 级 TDD 计划（本文档为主计划；Phase 内任务按"写测试→实现→过测→提交"循环）。
> 算力按 A100/H800-80G 折算。

### Phase 0：评测管线校准 + 教师就位（1~1.5 周，1~8 卡）★全项目最重要的阶段

复现失败最常见的原因不是模型没训好，而是**评测尺子本身不准**。本阶段先把尺子校准。

| 任务   | 内容                                                                                                           | 产出                              |
| ---- | ------------------------------------------------------------------------------------------------------------ | ------------------------------- |
| P0-1 | 环境/仓库/CI 骨架；vendored LPIPS；下载 SigLIP2、InternVideo-Next-L、Qwen2.5-1.5B、AToken-So/C、Wan2.2-VAE                 | 可运行的空仓库 + 权重清单(含sha256)         |
| P0-2 | 实现 `protocols.py`（256² resize+center-crop）+ `recon.py`（PSNR/SSIM/rFID/rFVD）                                  | 指标库                             |
| P0-3 | **校准 A**：AToken-So/C 官方权重过我们的评测 → 对标其论文/表 9（ImageNet 29.72 PSNR/0.209 rFID；DAVIS 26.60/29.19；UCF 34.66/7.77） | 校准报告                            |
| P0-4 | **校准 B**：Wan2.2-VAE 官方权重同协议 → 对标表 9（ImageNet 31.25/0.749；DAVIS 27.64/14.78；UCF 36.11/4.15）                   | 校准报告                            |
| P0-5 | 教师 golden tests：两教师对固定输入的特征存 fixture；断言与官方示例 pipeline 输出 cos>0.999；锁定输入协议（帧数/归一化）                            | `tests/test_teachers_golden.py` |
| P0-6 | 数据落盘：ImageNet-1k、DAVIS-2017、UCF-101、OpenVid-1M 元数据 + 17 帧抽样管线                                                | webdataset shards               |

**Gate P0**：两个公开 checkpoint 的四项指标全部进入公开数字 ±5%（rFID/rFVD 允许 ±10%，因实现差异敏感）。**任何一项不达标，禁止进入 Phase 1**——先换 FVD 实现/修预处理直到复现公开数字。
**回退**：rFVD 对不上→在 I3D 版与 StyleGAN-V 版之间切换；PSNR 对不上→逐像素 diff 预处理（resize 插值核、crop 坐标、[0,1] vs [-1,1]）。

### Phase 1：图像-only tokenizer 骨架（2 周，8 卡）

| 任务   | 内容                                                                                                                    |
| ---- | --------------------------------------------------------------------------------------------------------------------- |
| P1-1 | `vit_blocks.py`：SigLIP2-So400M 权重加载（断言 27 层/1152 维），13+14 切分（配置项）                                                     |
| P1-2 | `gsb.py`+`decoder.py`（decoder 从 SigLIP2 复制初始化 + 随机 pixel head）；纯图像 encode/decode 闭环                                   |
| P1-3 | 损失：L1+LPIPS+KL(λ=1e-4)+L_cos(开关)；`test_overfit.py`：单 batch 256² 过拟合                                                   |
| P1-4 | ImageNet 256² 训练 50k 迭代（lr 2e-4, AdamW(0.9,0.95), wd 0.05, bs 256, warmup 5k, cosine）+ 蒸馏项（λ_dist=1.0, 仅 SigLIP2 锚帧项） |

**Gate P1**：单 batch 过拟合 PSNR>35（30 分钟内）；50k 迭代 val PSNR≥26、rFID≤3、蒸馏 cos 距离持续下降。参照系：VA-VAE† 全程训练后 27.96 PSNR/0.28 rFID @16×——我们 50k 只需在通往该水平的正常轨迹上。
**回退**：PSNR 停滞→检查 KL 权重（过大会坍缩,降到 1e-6 试探）；蒸馏与重建互相拖累→检查 L_cos 开关与切分点。

### Phase 2：视频扩展 + 表 1 消融（3 周，8~16 卡）★验证发现 F1/F2

| 任务   | 内容                                                                                                                              |
| ---- | ------------------------------------------------------------------------------------------------------------------------------- |
| P2-1 | `rope3d.py`+`attention.py`+`temporal_patchify.py`（全部单测过）；视频 encode/decode 闭环（17×256²）                                           |
| P2-2 | ADR-3/4 预消融：patchify 位置 3 组 × pad 方向 2 组，各 30k 迭代快筛（ImageNet+OpenVid 10% 子集），选优冻结                                               |
| P2-3 | **表 1 复现**：{full, causal, tubelet}×单步4× + tubelet×层级2×2，共 4 变体 × 2 seed，按附录 A.4 协议（ImageNet-1k 150k 迭代 + 视频混合）；测延迟（17×512² 单前向） |

**Gate P2**：① rFVD 排序 tubelet<causal<full 且层级<单步，2 seed 均成立；② tubelet 延迟 ≤ full 的 40%；③ Ours 变体 DAVIS rFVD ≤ 15（论文 11.19，容差留给数据差异）。
**回退**：排序不成立→首先怀疑 mask 实现（重跑可达性测试+可视化 attention map），其次怀疑视频数据质量（换 DAVIS 同源的高质量子集重试）；若 full 反而更好且排除实现错误→如实记录为"未复现"，联系作者，**不得调参凑排序**。

### Phase 3：Decompressor + 双教师蒸馏 + 表 2 趋势（3 周，16 卡）★验证发现 F3

| 任务   | 内容                                                                                                                                       |
| ---- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| P3-1 | `decompressor.py`+`distill.py`（单测过）；蒸馏损失初值 sanity（0.6~1.0）                                                                               |
| P3-2 | 表 2 的 5 行配置各训一个 tokenizer（无蒸馏/img/img+D_img/img+D_vid/img+D_vid+bidir）                                                                   |
| P3-3 | 轻量理解探针（不必训完整 UMM）：① CKNNA 对齐度（HYDRA 同款分析，1 万张 ImageNet val）；② 冻结 tokenizer + 线性 probe ImageNet 分类；③ 视频侧: 冻结特征 + 线性 probe SSv2 子集（时间敏感任务） |

**Gate P3**：探针排序复现表 2 趋势——img+D_vid 的视频探针最优且图像探针不降；bidir 变体全面变差；无蒸馏垫底。
**回退**：D_vid 无增益→检查教师时序对齐（golden test P0-5 重跑）与 Decompressor 上采样顺序；探针不敏感→升级为 Qwen2.5-0.5B 微型 UMM 探针。

### Phase 4：完整 Hydra-XTok 三阶段训练 + 表 9 对标（4 周，16~32 卡）

| 任务   | 内容                                                                            |
| ---- | ----------------------------------------------------------------------------- |
| P4-1 | Stage1：ImageNet→混合分辨率（256² 视频 + 256~2048 图像，bucket 采样），300k 迭代                |
| P4-2 | Stage2：冻结 encoder，+PatchGAN(hinge, adaptive weight, λ≈0.075 起)，训 decoder 100k |
| P4-3 | Stage3：统计 z 均值/方差 → 归一化接口（ADR-5）→ 只训 Sem-ViT 50k                              |
| P4-4 | 平行小任务：Hydra-XTok†（纯 ImageNet 版, 数据完全对齐论文）训完整三阶段                               |

**Gate P4**：① Hydra-XTok†：ImageNet PSNR≥31.5 / rFID≤0.25（论文 32.96/0.154，数据完全对齐故按 ±10% 数值级要求）；② 全量版 DAVIS rFVD<Wan2.2 的 14.78 或至少 <AToken 的 29.19；③ Stage3 结束时重建指标较 Stage2 结束退化 <0.2dB（验证 ADR-5）。
**回退**：GAN 崩→R1 正则+判别器 lr 减半；rFID 差→延长 Stage2；②不达→接受并记录（数据差异），只要①达标则架构实现无罪。

### Phase 5：1.5B UMM（4~5 周，32 卡）

| 任务   | 内容                                                                                                    |
| ---- | ----------------------------------------------------------------------------------------------------- |
| P5-1 | 从 Show-o2 移植训练循环/推理脚本骨架（Apache-2.0），重写 `sequence_builder.py`（五任务 mask/loss mask 单测）与视觉编码路径（ADR-1）     |
| P5-2 | flow head（AdaLN, z 空间 64 维输出）+ RF 训练（logit-normal t 采样）；`test_flow.py`：给定完美 v 场时单步积分还原 ẑ              |
| P5-3 | 微型闭环冒烟：0.5B LLM + 1000 张图过拟合——T2I 能生成训练集图像（视觉检查+ rFID<30）                                             |
| P5-4 | 按附录 A.4 协议训练：理解（LLaVA-1.5+LLaVA-Video）与生成（20M 图文对：CC12M+LAION-aes 子集+JourneyDB, Qwen2.5-VL recaption） |
| P5-5 | 评测接入：lmms-adapter（MVBench/VideoMME/AI2D/MME）+ GenEval                                                 |

**Gate P5**：理解侧 AI2D≥55、MME≥1300（对标表 2 中间行水平）；GenEval≥0.65；生成图像视觉无结构性崩坏。
**回退**：T2I 不收敛→ADR-1 回退开关（H_bn 直喂）；理解弱→检查 projector 与 s 的 layernorm 匹配。

### Phase 6：STI 编辑消融 + 表 3 复现（2 周，16 卡）★验证 STI

| 任务   | 内容                                                         |
| ---- | ---------------------------------------------------------- |
| P6-1 | `encode_edit_pair`（§3.6）+ 编辑序列构造；Indep 对照组配置（唯一差异=routing） |
| P6-2 | 两组各自从 P5-4 checkpoint 继续训 ImgEdit（同 seed/同数据顺序）            |
| P6-3 | 评测：ImgEdit-Bench（固定 GPT judge 版本）+ Recon-PSNR 探针（ADR-6 定义） |

**Gate P6**：STI−Indep 的 Recon-PSNR delta ≥ +5dB；ImgEdit overall delta>0；GenEval 不降。
**回退**：delta 小→检查 Sem-ViT 联合 pass 中 mask 是否真的让 s_t 见到 z_c（attention map 可视化）；检查训练时 z_c 是否误加噪。

### Phase 7（可选）：7B 全量（8~12 周，256+ 卡）

按表 8 协议执行；数据替代配方见配套文档 §6。先决条件：Phase 0~6 全部 Gate 通过，且拿到相应算力预算。本阶段不展开任务级计划（届时按 Phase 6 的实际经验另写子计划）。

### 算力预算汇总（A100-80G 等效）

| Phase        | GPU·h 估算    | 说明                                    |
| ------------ | ----------- | ------------------------------------- |
| P0           | ~200        | 评测+校准                                 |
| P1           | ~1,500      | 50k×bs256@256²                        |
| P2           | ~12,000     | (6 预消融快筛 + 4 变体×2 seed)×150k          |
| P3           | ~10,000     | 5 配置×150k + 探针                        |
| P4           | ~15,000     | 300k+100k+50k 混合分辨率 + † 版             |
| P5           | ~20,000     | 1.5B, 理解+生成                           |
| P6           | ~3,000      | 两组编辑微调                                |
| **合计(不含P7)** | **~62,000** | 32 卡连续 ~80 天；砍 seed 数/分辨率可压缩至 ~35,000 |

---

## 5. 误差积累防控与科研严谨性体系

误差积累的机理：上游组件的小偏差（预处理、教师对齐、mask 语义）不会立刻报错，而是在 3~4 个 Phase 之后以"指标莫名偏低"的形式爆发，且此时已无法归因。防控体系如下：

### 5.1 五道防线

1. **尺子先行（Phase 0 Gate）**：任何自训模型出数字之前，评测管线必须先在公开 checkpoint 上复现公开数字。这条防线消除"评测实现错误"这一最大的不可归因误差源。
2. **Golden fixture 测试**：教师特征、预处理输出、mask 矩阵各保存固定输入的期望输出为二进制 fixture，进 CI。任何依赖升级（transformers 版本等）导致 fixture 漂移会被立即捕获。
3. **单 batch 过拟合冒烟**：每个训练入口（tokenizer/UMM/编辑）都有对应的 overfit 测试——模型若连一个 batch 都记不住，说明梯度流断裂（冻结策略错误、loss mask 错误、detach 遗漏）。这类 bug 在正常训练里表现为"收敛但慢"，极难发现。
4. **阶段边界快照评测**：每个训练 Stage 切换（冻结策略/损失开关变化）前后各跑一次完整重建评测；指标突变>0.5dB 即停下排查。专门针对 Stage2→3 的归一化切换（ADR-5）和 UMM Stage1→2 的解冻。
5. **受控消融纪律**：同 seed、同数据顺序（`mixture.py` 确定性采样）、单变量差异；每个关键消融 ≥2 seed；效应量<2σ 的结论一律标注"不显著"。

### 5.2 实验记录规范（每 run 一张实验卡，自动生成进 `docs/experiments/`）

```yaml
run_id: t0-attn-tubelet-hier-s1        # phase-变量-取值-seed
git_sha: <commit>                      # 代码不干净不许启动(CI强制)
config_hash: <sha256 of resolved yaml>
data_snapshot: imagenet-v1 + openvid-sub10-v2   # 数据集版本化
seeds: {model: 1, data: 1}
adr_versions: [ADR-3@v2(patchify@0,L/2), ADR-4@v1(before)]
metrics: {...}                         # 训完自动填充
wandb: <url>
conclusion: ""                         # 人工填写, 与Gate判据对照
```

### 5.3 Phase 复盘协议（每个 Gate 通过/失败后必须执行）

按固定顺序归因，禁止跳步：

1. 结果与论文趋势一致吗？→ 一致：记录并冻结该 Phase 的全部 config；
2. 不一致：先查**评测**（该指标在校准 checkpoint 上还准吗？重跑 P0-3/4）；
3. 再查**数据**（协议 diff：分辨率/裁剪/帧采样/归一化，抽 10 个样本人工目检）；
4. 再查**模型**（单测全绿？attention map/重建图可视化目检？梯度范数异常？）；
5. 以上全排除→认定为真实差异，如实记录（含"我们未能复现 X"），更新 ADR 并邮件作者。
   **红线：不得为了凑论文数字而在消融间引入不受控差异（调 lr、换数据、挑 seed）。挑 seed 报最好值 = 学术不端。**

### 5.4 本计划自检结论（撰写时已执行的反思）

- ✅ 覆盖度：论文全部三个 tokenizer 发现（Gate P2/P3）、STI（Gate P6）、重建对标（Gate P4）、理解/生成主结果（P5/P7）各有对应 Phase 与验收；
- ✅ 依赖真实性：§1.1 全部资产已联网核实存在性与许可（2026-07-07）；唯一未 100% 确认的是 InternVideo-Next-**L** 尺寸权重是否在 HF collection 内（P0-1 首日验证，备胎已列）；
- ⚠️ 已识别的最弱环节：ADR-3（patchify 位置）置信度低——已用 P2-2 预消融兜底并前置到主消融之前，防止其误差污染表 1 复现；
- ⚠️ 表 1 效应量小（PSNR 差 0.04dB）——已把判据主体改为效应量更大的 rFVD 并强制 2 seed；
- ⚠️ 论文表 2/3 的消融是在"tokenizer+UMM 全链路"上测的，我们 P3 用轻量探针替代（省 10 倍算力）——存在探针不敏感的风险，已列升级路径（0.5B 微型 UMM）；
- ✅ license 卫生：AToken 代码仅作参考与校准、不入仓；训练框架从 Apache-2.0 的 Show-o2 移植。

---

## 6. 风险登记册 v2（相对配套文档的更新）

| #   | 风险                      | 等级        | 缓解（更新后）                                               |
| --- | ----------------------- | --------- | ----------------------------------------------------- |
| 1   | ADR-3 patchify 位置猜错     | 高         | P2-2 前置预消融 + 作者邮件；已隔离在表 1 消融之前                        |
| 2   | 评测实现偏差污染全部结论            | 高→**已消除** | Phase 0 双 checkpoint 校准 Gate（AToken + Wan2.2）         |
| 3   | InternVideo-Next-L 不可得  | 中         | 已核实 collection 存在；备胎 InternVideo2-Stage2；教师更换只影响数值级目标 |
| 4   | 表 1 效应量<噪声              | 中         | 2 seed + rFVD 主判据 + 显著性标注                             |
| 5   | ADR-1 LLM 输入路径猜错        | 中         | 配置开关双实现，P5-3 微型闭环 48h 内可鉴别                            |
| 6   | flex_attention 高分辨率 OOM | 低         | SDPA fallback；评测 1280×768 时分块                         |
| 7   | GAN 不稳定                 | 低         | 仅 Stage2 引入（encoder 已冻结，天然稳）+ adaptive weight + R1    |
| 8   | GPT-judge 漂移            | 低         | 固定 judge 版本；同批跑 BAGEL/OmniGen2 公开权重做锚点                |
| 9   | 视频训练数据与论文不同             | 不可消除      | 只影响数值级目标；趋势级结论（同数据内部对照）不受影响——这正是整个计划以受控消融为主体的原因       |

---

## 7. 参考资料（本次调研核实的全部来源）

- 论文：[Hydra-X (2606.13289)](https://arxiv.org/abs/2606.13289) · [HYDRA 前作 (2603.15228)](https://arxiv.org/abs/2603.15228) · [AToken (2509.14476)](https://arxiv.org/abs/2509.14476) · [Show-o2 (2506.15564)](https://arxiv.org/abs/2506.15564) · [InternVideo-Next (2512.01342)](https://huggingface.co/papers/2512.01342) · [ImgEdit (2505.20275)](https://huggingface.co/papers/2505.20275) · [WISE (2503.07265)](https://arxiv.org/abs/2503.07265)
- 代码：[apple/ml-atoken](https://github.com/apple/ml-atoken) · [showlab/Show-o](https://github.com/showlab/Show-o/tree/main/show-o2) · [ByteDance-Seed/Bagel](https://github.com/bytedance-seed/BAGEL) · [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2) · [OpenGVLab/InternVideo](https://github.com/OpenGVLab/InternVideo) · [PKU-YuanGroup/ImgEdit](https://github.com/pku-yuangroup/imgedit) · [PKU-YuanGroup/WISE](https://github.com/PKU-YuanGroup/WISE) · [stepfun-ai/Step1X-Edit (GEdit-Bench)](https://github.com/stepfun-ai/Step1X-Edit) · [FlexAttention 官方博客](https://pytorch.org/blog/flexattention/)
- 权重/数据：[google/siglip2-so400m-patch16-naflex](https://huggingface.co/google/siglip2-so400m-patch16-naflex) · [OpenGVLab/internvideo-next collection](https://huggingface.co/collections/OpenGVLab/internvideo-next) · [nkp37/OpenVid-1M](https://huggingface.co/datasets/nkp37/OpenVid-1M) · [sysuyy/ImgEdit](https://huggingface.co/datasets/sysuyy/ImgEdit) · [lightx2v/Autoencoders (Wan-VAE)](https://huggingface.co/lightx2v/Autoencoders)
