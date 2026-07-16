# UVT 当前模型 vs Hydra-X / HYDRA 论文 — 逐点出入对照

> 调研日期: 2026-07-16
> 论文原文(已用 PaddleOCR 精读):
> - **Hydra-X**: Zhang et al. 2026, arXiv 2606.13289 (27 页) — 核心来源
> - **HYDRA 前作**: Qiu et al. 2026, arXiv 2603.15228 (24 页) — 补齐 Hydra-X 未给的数值
> 代码锚点: `uvt/models/uvt/*.py` + `uvt/losses/*.py` + `UVTConfig`(与 uvt-npu 逐字节相同)
> 结论一句话: **三大设计的"骨架"忠实实现;出入集中在 (1) 项目范围收窄 (2) 三处主动偏离 (3) 论文未给数值处的本地选择。**

---

## 0. 大前提:项目范围 = Hydra-X 的【贡献一】,砍掉【贡献二】

| Hydra-X 论文 | 本项目 |
|---|---|
| **贡献一** Hydra-XTok:图像+视频统一 tokenizer(三大设计) | ✅ **全部实现**(本项目 = 这一半) |
| **贡献二** Hydra-X UMM:LLM + flow-head + 五任务 + STI 编辑(§5,表3 +6.9dB) | ❌ **完全不做**(CLAUDE.md「只做 tokenizer, 不做 UMM」) |

→ 所以论文 §5(STI 编辑,source-target 交互)、§5.1(双头 UMM)、§6.1-6.5(理解/生成/编辑评测)
   **均在范围外**,不构成"出入",是刻意收窄。下面只对照 tokenizer 部分。

---

## 1. 设计一:Tubelet 因果注意力 —— ✅ 完全一致

| 维度 | 论文 (Hydra-X §4.1, 图2) | 代码 (attention_mask.py) | 判定 |
|---|---|---|---|
| tubelet 定义 | "each token attends only to its own frame and the immediately preceding one"(2帧窗口) | `(tk==tq) \| (tk==tq-1)` (line 32) | ✅ 一致 |
| mask 单位 | latent 序列(折叠后时间位) | 折叠后时间位(ADR-2, line 4) | ✅ 一致 |
| 帧内 | 全双向 | 帧内全双向(同一 tq 全通) | ✅ 一致 |
| 三模式消融 | full/causal/tubelet(表1) | `VALID_KINDS=("full","causal","tubelet")` | ✅ 一致 |
| 实现方式 | (论文未指定) | SDPA 加性 bias, 无 flex_attention(红线) | ✅ 合规 |
| Sem-ViT 也用 tubelet | 是(表2 末行:改双向变差) | SemViT 同样吃 attn_mode | ✅ 一致 |

**结论:设计一零偏离。**

---

## 2. 设计二:层级时间 patchify —— ⚠️ 一处主动偏离(锚帧)

| 维度 | 论文 (Hydra-X §4.1) | 代码 (temporal_fold.py) | 判定 |
|---|---|---|---|
| 两步 2×2 vs 单步 4× | 两个连续 2× 阶段 | `TemporalFold2x` ×2, fold_positions=(0,6) | ✅ 一致 |
| 折叠算子 | (等价 time-to-channel) | Linear(2D→D)+rearrange, 近两帧平均初始化 | ✅ 等价 |
| **锚帧处理** | **"the anchor frame is zero-padded so that it goes through the same operation as the remaining frames"** (p4:13) | **ADR-4': 锚帧永不参与折叠,直通隔离**(line 4, 26) | ❌ **主动偏离** |
| 插入位置 | **论文未明说**("distributes across multiple stages")—— 全文最大歧义 | 代码定为 (0, 6) = {输入处, GenViT 第6层后} (ADR-3) | ⚠️ 本地选择,论文无对照 |

**偏离①(锚帧 zero-pad → 锚帧隔离):**
- 论文:锚帧补零帧,和视频帧走同一个 fold 算子。
- 代码:锚帧完全不折叠,直接旁路(`x[:,:1]` 切出来不动)。
- 项目理由(temporal_fold.py:4):与 OmniTokenizer/Wan-VAE/LeanVAE 行业标准一致;
  zero-pad 作为对照臂留给 P2-pre 消融。
- **影响**:这是方法层的真实差异,但项目已把它登记为待消融项(非疏漏)。

**歧义点(插入位置):** 论文确实没写两个 2× 插在哪两层。代码选 (0,6),背景文档 ADR-3
标注置信度**低**,列为 T0 阶段网格消融 {(0,L/2),(0,L/3),(L/3,2L/3)}。→ 这不算"错",
是论文留白处的合理默认,但需记住它未经论文背书。

---

## 3. 设计三:Decompressor 双教师蒸馏 —— ⚠️ 一处本地扩展 + 实现层重组

### 3a. Decompressor 结构 —— ✅ 一致
| 维度 | 论文 (Hydra-X §4.2, 图3) | 代码 (decompressor.py) | 判定 |
|---|---|---|---|
| 4× 时间上采样 | 两级 2×(4→8→16) | `up1→block1→up2→block2`, 4→8→16 (line 54) | ✅ 一致 |
| 上采样算子 | 1×1 conv 通道翻倍 C→2C + channel-to-time | Linear(D→2D)+rearrange (line 39-46) | ✅ 数学等价* |
| 注意力 | (训练期附件) | full 双向, rope_dims=0 (line 19-20) | ✅ 合理 |
| 训练后丢弃 | 是 | state_dict post-hook 剔除 (uvt_tokenizer.py:72) | ✅ 一致 |

\* 论文在 latent C=64 维上做 conv;代码在 student_dim D=1152 上做 Linear。等价,只是作用维度不同
  (代码 Decompressor 吃的是 Sem-ViT 输出 s,本就是 D 维,不是 latent z)。

### 3b. 蒸馏损失 L_dist —— ⚠️ 代码比论文式(3) 多一项
| 论文式(3) (Hydra-X p4:27) | 代码 (distill.py) | 判定 |
|---|---|---|
| `d_cos(s_0, T_img(x))` — 锚帧 patch → 图像教师 | `img_patch`: s[:,0] → SigLIP2 patch (line 97) | ✅ 对应 |
| `d_cos(D(s_1:), T_vid(x))` — 解压 → 视频教师 | `vid`: Decompressor(s[:,1:]) → InternVideo (line 104) | ✅ 对应 |
| **(无第三项)** | **`img_pool`: MAP池化 s_pool → SigLIP2 整图嵌入** (line 101) | ❌ **本地新增(ADR-9)** |
| d_cos = 1−cos | `d_cos = 1.0 - cosine_similarity` (line 31) | ✅ 一致 |
| 纯图像 batch 视频项 mask | `_vid_term` is_video 掩码 (line 112-125) | ✅ 一致 |

**偏离②(多一个 img_pool 池化项):**
- 论文式(3) 只对齐 patch 级 s_0 和视频级 D(s_1:)。
- 代码多了一个"整图池化向量 → 教师整图嵌入"的对齐(为 zero-shot 文本塔对齐留接口)。
- 项目理由(distill.py:5, ADR-9):继承 SigLIP2 文本塔需要 pool 向量对齐。
- **影响**:附加监督,理论上无害(权重可调/可关),但确是论文式(3) 外的项。

**实现层重组(非方法偏离):** 代码 Decompressor **不投影到教师维度**(decompressor.py:8),
投影交给 loss 的 `head_vid`。论文图3 把投影画在 Decompressor 里。→ 只是职责归属不同,数学等价。

---

## 4. Bottleneck (GSB) —— ⚠️ 一处结构重构

| 维度 | 论文 (HYDRA 式2, Hydra-X §3) | 代码 (gsb.py) | 判定 |
|---|---|---|---|
| 压缩投影 | `[μ,ρ]=W_proj·H_mid`, C=64 | `proj=Linear(D,2*64)`.chunk → (μ,ρ) (line 24,48) | ✅ 一致 |
| 重参数化 | `z=μ+ε⊙exp(0.5ρ)` | `z=mu+randn*exp(0.5*rho)` (line 52) | ✅ 一致 |
| KL 项 | `-½Σ(1+ρ−μ²−exp(ρ))` | `-0.5*(1+rho-mu²-rho.exp()).mean()` (line 54) | ✅ 一致(sum→mean 归一化差异) |
| **反投影 unproj** | **HYDRA 有 `W_unproj`**: `H_bn=μ·W_unproj` (式, p2:37) | **删除! 反投影下放到各消费方 in_proj** (gsb.py:6-9) | ❌ **结构重构** |
| 瓶颈入口 LN | (论文未提) | `self.norm=LayerNorm` 第15号修复 (line 22) | ➕ 本地加固 |

**偏离③(删除共享 W_unproj):**
- HYDRA/Hydra-X:GSB 自带反投影 W_unproj,产出 H_bn 喂给 Sem-ViT 和 decoder(共享)。
- 代码:GSB 只有前向 proj;反投影拆成 `sem_vit.in_proj` 和 `decoder.in_proj` 两个独立层。
- 项目理由(gsb.py:6-9):Stage-3「冻 decoder 训 Sem 侧」时,共享反投影层会冻结归属打架。
- **影响**:这是为了三阶段训练可分离而做的**合理工程重构**,数学上把 1 个共享层拆成 2 个独立层。
  副作用:decoder 和 Sem-ViT 的反投影不再绑定(论文里它们共享 H_bn)。

**本地加固(瓶颈入口 LN):** 第15号修复。SigLIP2 残差流有 O(10³) 巨激活,不加 LN 则 μ/ρ 被主导。
论文未提(因为它们从头训,不是切 SigLIP2 中段),这是本项目"切预训练 slice"特有的必需加固。

---

## 5. 损失权重 λ —— 论文大多未给,代码是本地选择

| 权重 | Hydra-X | HYDRA 前作 | 代码默认 | 判定 |
|---|---|---|---|---|
| λ1 (L1) | **未给** | (含在 rec) | 1.0 | 本地 |
| λ_perc (LPIPS) | **未给** | 0.1 | 1.0(overfit/训练常用 0.5) | ⚠️ 与 HYDRA 0.1 不同 |
| λ_gan | **未给** | 0.075 | (gan.py, S2) | 待核 |
| **λ_KL** | **未给** | **1e-4** | **1e-6** | ⚠️ **比 HYDRA 小 100×** |
| λ_cos | 未提及 L_cos | **1.0** | 1.0 | ✅ 一致 |
| **λ_dist** | **未给** | **1.0** | **0.5** | ⚠️ 比 HYDRA 小一半 |

**关键观察:**
- Hydra-X 论文(式7)确实**一个 λ 数值都没给**(我通读附录 A.1 确认)。所有权重要么从 HYDRA 前作借,
  要么本地扫。→ 代码里的数值**不构成"违背论文"**,因为论文没给标准。
- **实际全量训练 cfg(`uvt_stage1_imagenet_full_npu.yaml`)已由 run-full-03 主动调过权重:**
  - `lpips_weight: 0.5`(UVTConfig 默认 1.0),注释明写"HYDRA 参考 λ_perc=0.1,原1.0偏高压PSNR;为PSNR让路(rFID靠S2 GAN补)"
    → **训练者已意识到与 HYDRA 0.1 的差距并往下调**,是有意识的取舍(0.5 是 PSNR/rFID 折中,非疏漏)。
  - `cos_weight: 0.5`(默认 1.0),注释"降权减轻共享z语义扰动"。
  - `lambda_dist: 0.5`, `kl_weight: 1e-6` 保持。
- 仍存在的、与自称来源(HYDRA)不同处:
  - **λ_KL: 代码 1e-6 vs HYDRA 1e-4**(小 100 倍)。背景文档 D14 说"KL 从 1e-6 起扫",
    刻意选低值(高 KL = 重建糊头号嫌疑)。合理,但偏离 HYDRA 基准。
  - **λ_dist: 代码 0.5 vs HYDRA 1.0**(小一半)。P3 扫 {0.25,0.5,1.0},0.5 是中间默认。
  - **λ_perc: 实跑 0.5 vs HYDRA 0.1**(仍大 5 倍,但已从默认 1.0 下调)。

---

## 6. 其它对照点

| 点 | 论文 | 代码 | 判定 |
|---|---|---|---|
| 初始化/图像教师 | SigLIP2 | siglip2-so400m-patch16-**256**(非 naflex) | ✅ 一致(分辨率变体) |
| 视频教师 | InternVideo-Next-L | InternVideo(teachers/) | ✅ 一致 |
| Gen/Sem 切分 | HYDRA:12+12最优(24层); Hydra-X 用 SigLIP2 27层 | 13+14 (gen_depth=13) | ✅ 合理(27层无法均分,13/14 最接近) |
| decoder | Hydra-X:27层对称ViT+L1回归(改自HYDRA的flow decoder) | PixelDecoder 27层 (decoder.py) | ✅ 实现 Hydra-X 版 |
| **位置编码** | **"follows AToken 做 3D RoPE"(时空联合:空间2轴+时间1轴)** | **只有时间轴 1D RoPE(rope_dims=32,每头前32维);空间轴沿用 SigLIP2 原生 pos-emb** (blocks.py:61-68) | ❌ **偏离④** |
| 三阶段训练 | S1全参/S2冻enc训dec+GAN/S3统计z-stat冻dec训Sem | set_stage 1/2/3 完全对应 | ✅ 一致 |
| L_cos 存在性 | **Hydra-X 未提及**;HYDRA 式4 有 | use_cos_consistency=True(默认开) | ⚠️ 从 HYDRA 引入,Hydra-X 无 |

**偏离④(3D RoPE → 时间轴 1D RoPE):**
- 论文:跟随 AToken 用完整 3D RoPE,空间两轴 + 时间轴都用 rotary 联合建模时空位置。
- 代码:`_rope_tables` 只吃 `time_ids`(blocks.py:61),仅对时间轴、每头前 32 维做 1D rotary;
  空间位置沿用 SigLIP2 骨干自带的可学习/插值 pos-emb(未改成 rotary)。
- **影响**:位置编码方案不同。代码更保守(保护 SigLIP2 预训练的空间位置先验,只给时间轴补 RoPE),
  与"锚帧隔离""切 slice 加 LN"是同一思路——尽量少动预训练结构。但确与论文"3D RoPE"不符。
- rope_dims 本身是 ADR-8 消融开关(可设 0 关闭)。

---

## 7. 出入总清单(供决策)

### 真实方法偏离(4 处,均已在项目 ADR 登记为待消融/有意取舍)
1. **锚帧折叠**: 论文 zero-pad 走同算子 → 代码锚帧隔离直通 (ADR-4', temporal_fold.py:4)
2. **蒸馏多一项**: 论文式(3) 两项 → 代码多 img_pool 池化项 (ADR-9, distill.py:101)
3. **GSB 反投影**: 论文共享 W_unproj → 代码拆成各消费方独立 in_proj (gsb.py:6-9)
4. **位置编码**: 论文 3D RoPE(时空联合) → 代码仅时间轴 1D RoPE,空间轴留 SigLIP2 原生 (blocks.py:61)

> 共同主题:偏离 1/3/4 都是**"尽量少动 SigLIP2 预训练结构"**的同一保守思路
> (锚帧不折、空间位置不换 RoPE、反投影分离以便冻结)。这是"切预训练 slice 做 tokenizer"
> 相对 Hydra-X"从头训"的必然工程取舍,不是实现错误。偏离 2 是附加监督(zero-shot 文本塔)。

### 论文留白处的本地选择(非偏离,但无论文背书)
5. **层级 patchify 插入位置** (0,6) — 论文未明说(全文最大歧义, ADR-3 置信度低)
6. **λ_KL=1e-6**(HYDRA 1e-4)、**λ_dist=0.5**(HYDRA 1.0)、**λ_perc 实跑 0.5**(HYDRA 0.1,已从默认1.0下调)
7. **L_cos 默认开** — Hydra-X 未提及此损失(继承自 HYDRA)

### 本地必需加固(切 SigLIP2 slice 特有,论文无对应因为它们从头训)
7. 瓶颈入口 LN、GenViT 输入 px 归一化、decoder final_ln(第15号修复系列)

### 完全一致(忠实实现)
- tubelet mask、Decompressor 结构、VAE 重参数化、KL 公式、三阶段冻结、教师选择、decoder 类型

---

## 8. 待办 / 值得决策(不改代码,先记录)
- [x] ~~λ_perc 实际值~~ → 已查:全量 cfg lpips_weight=0.5(注释已注明对齐 HYDRA 方向下调),非默认 1.0。
- [x] ~~3D RoPE 空间轴~~ → 已查:代码只有时间轴 1D RoPE,空间轴用 SigLIP2 原生 pos-emb(偏离④)。
- [ ] **决策点:4 处方法偏离要不要在正式全量训练前先跑对照臂?** 还是先按当前保守默认全量训、
      把消融留到之后? (锚帧 zero-pad 臂=P2-pre;img_pool 关闭臂;3D RoPE 全轴臂;GSB 共享 unproj 臂)
- [ ] **决策点:λ_KL=1e-6 vs HYDRA 1e-4 差 100 倍** —— 是否至少扫一个 1e-4 点做对照,确认低 KL 无副作用?
