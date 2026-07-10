# VTok 精读 + LARP / LeanVAE 底座评估

> 日期：2026-07-07。VTok 为 arXiv HTML 全文精读；LARP/LeanVAE 为论文检索 + 官方仓库核实。
> 结论先行：① **VTok 与我们不撞车**——它不是像素级 tokenizer，而是 MLLM 的视频接口工程，处在栈的上一层；有四点可借鉴。② **LARP 和 LeanVAE 都是真材实料的顶会开源工作（ICLR 2025 Oral / ICCV 2025，均 MIT）**，作为底座"可以"，但两者都是纯重建路线、与我们的语义统一路线正交——实验室内三条线互补不撞车。

---

## 1. VTok 精读（arXiv 2602.04202，2026-02，JHU/字节系作者）

### 1.1 它到底做了什么（方法拆解）

**一句话：给 MLLM 设计了一种省 token 的视频输入格式，所有视觉组件全部冻结，只训 MLLM。**

```
关键帧（取第1帧）→ 冻结 CLIP-L/336 编码 → 16 个空间 token（4×4 网格）
后续帧（每6帧一组）→ CLIP 特征差 g_φ(F(x_t) − F(x_1)) → 池化成 1 个"残差运动 token"
5秒24fps视频 = 16 + 30 ≈ 46 个 token（对比逐帧采样的 64~128+）
token 预算：O(T×S) → O(S+T)
```

- **理解分支**：46 个 token 过 projector 进 MLLM（LLaVA-Next/LLaMA3-8B）出文本；
- **生成分支**：MLLM 自回归采样出同格式的视觉 token → **冻结的 HunyuanVideo-13B DiT** 把 token 当条件信号解码成视频；
- **训练**：只训 MLLM（lr 1e-5，batch 16，5M 视频文本对，λ 全为 1，作者自述"未做充分调参"）。

### 1.2 为什么不撞车（关键定性）

论文自己在 Related Work 里把 tokenization 分成两条线并明确站队："**Unlike VAE-style tokenization, this strategy does not emphasize precise pixel-level reconstruction**... This work mainly focuses on the latter（语义条件路线）"。判据逐条对照：

|        | VTok                                    | 我们（Hydra-XTok 系）           |
| ------ | --------------------------------------- | -------------------------- |
| 能否重建像素 | ❌ 无重建目标、无 decoder（借 Hunyuan DiT 当条件生成器） | ✅ 核心目标                     |
| 训练什么   | 只训 MLLM，视觉组件全冻结                         | 训练整个 encoder/decoder       |
| 评测     | TV-Align/VBench/理解基准                    | PSNR/rFID/rFVD + 探针 + gFID |
| 栈层     | **MLLM 的视频接口层**                         | **像素级统一 tokenizer 层**      |

它是我们 tokenizer 的**潜在下游用户**（这类工作将来完全可以拿我们的 latent 换掉它的 CLIP 特征），而非竞争者。v2 计划的差异化担忧解除。

### 1.3 可借鉴的四点

1. **残差式时间 token（最有价值）**：相对关键帧编码时间变化，而不是编码每帧的绝对内容。与我们的锚帧协议天然共鸣——我们的时间 latent 目前是"折叠的绝对表征"，可以探索一个**相对锚帧的残差参数化变体**（时间位 latent = 相对锚帧的 delta），有望进一步压缩时间冗余。已加入 v2 §2.2 创新备选清单。
2. **非对称 token 预算的量化证据**（表 5 消融）：语义任务上关键帧 16 token 是最优平衡（1 个最差、25 个收益递减）；时间粒度 6 帧/token 最优。对未来 UMM 接口设计有参考价值；但注意这是**语义任务**的结论，对重建级保真完全不适用。
3. **TV-Align 基准**：1000 条（prompt, 问题, 答案）三元组测生成的指令跟随（计数/方向/相对位置/大小/颜色/状态/运动），用 Qwen2.5-VL 当裁判。若未来做生成下游可采用。
4. **一个支持我们设计的反证**：他们发现"关键帧只用 1 个 token 时最差"→ 视频表征中空间与时间必须都显式保留——支持我们保留 2D 空间网格 latent 的设计，反对极端 1D 化。

### 1.4 弱点（评审视角）

- 主要提升叙事建立在**自建基准**上（TV-Align 自己定义、自己选裁判）；
- 理解侧提升不大（Qwen2.5-VL 平均 +1.8、LLaVA-Next +2.4；不微调的直接替换在多个基准上 ≤0.5，接近噪声）；
- CLIP 特征差当运动表征相当粗糙，16 个空间 token 无法承载细节（OCR/细粒度必然受限，论文回避了这类基准）；
- 格式为 ICML 投稿模板，应处于在投状态，结论未经同行评审。

---

## 2. LARP 评估（同学底座之一）

**事实核**（[github.com/hywang66/LARP](https://github.com/hywang66/LARP)，官方仓库已核实）：ICLR 2025 **Oral**（UMD 等）；MIT license；训练代码**全套**（tokenizer + AR prior + 帧预测）；依赖干净（PyTorch 2.4）；HF checkpoints：tokenizer 173M（UCF-101 **rFVD 20**）、AR 模型 632M（**gFVD 57**，发表时 SOTA）、K600 帧预测 FVD 5.1。

**方法**：抛弃 patch 网格，用一组**可学习的整体查询（holistic queries）**从视频中聚取信息生成 1D 离散 token（TiTok 思路的视频版）+ VQ 码本，并在训练时**联合一个轻量 AR transformer prior**——用下一 token 预测损失反向塑造 latent 空间，使 token 序列天然适合自回归生成。

**评估**：

- ✅ **质量过硬**：Oral + SOTA 数字 + 完整干净的代码，作为科研底座可靠；
- 定位：**离散 / AR 生成路线**——按我们的三轴分类是"ViT 骨干 × 离散 VQ（1D holistic）× 重建+AR prior 损失"，正好是我们（连续 × diffusion）在设计空间的另一极；
- 局限：无语义轴（不做理解/蒸馏）；实验协议分辨率低（UCF 128² 系），向高分辨率长视频扩展时 holistic 查询数固定、泛化受限；video-only，不做图像视频统一；
- **一个值得注意的深层联系**：LARP 的"用 AR prior 在训练时塑造 latent"与我们的"latent 可生成性"（用 DiT 事后验证）是**同一个哲学的两种实现**——"tokenizer 的 latent 应该为下游生成模型的建模难度负责"。这是与同学工作天然的对话点，甚至可以合作出一个"生成友好性塑造方法对比"的分析。

---

## 3. LeanVAE 评估（同学底座之二）

**事实核**（[github.com/westlake-repl/LeanVAE](https://github.com/westlake-repl/LeanVAE)，已核实）：ICCV 2025（西湖大学）；MIT；训练/推理/评测脚本全；**仅 40M 参数**；压缩 4×8×8；两个 checkpoint（4ch：PSNR 26.04；16ch：PSNR 30.15/LPIPS 0.046）；17 帧 1080p 编码仅 0.9s；已与 Latte 集成验证（SkyTimelapse/UCF101）；支持长视频时间分块推理。

**方法**：小波变换把每帧分解为 LL/LH/HL/HH 四个子带 + NAF（邻域感知前馈）模块 + 非重叠 patch 操作 + 压缩感知式通道瓶颈 → 相对主流视频 VAE 最多 50× FLOPs 削减。

**评估**：

- ✅ 质量可以（ICCV + 完整开源），定位是**效率前沿的卷积视频 VAE**——三轴分类："CNN × 连续 KL × 重建损失"，主打省算力；
- 局限：conv 架构 → **没有预训练权重通路、没有语义蒸馏的自然接口**（这两条都是 ViT 专属红利）；纯重建；保真上限不追顶（16ch PSNR 30.15，低于 Wan2.2 的 31+，这是效率换来的合理取舍）；
- 风险提醒：高效视频 VAE 赛道内卷严重（VidTok、Cosmos、WF-VAE、Wan-VAE、LTX-VAE 都在卷），做增量工作需要想清楚差异化；小波+压缩感知组合是它的特色，也意味着改进空间集中在别人不熟的数学工具上。

---

## 4. 综合判断与实验室内协同

**"这工作是否可以"——可以。** 两个都是发表扎实（Oral/ICCV）、开源完整、license 干净的工作，作为底座不会踩"烂代码/复现不了/许可有毒"的坑。真正要判断的是**课题定位匹配**：

| 路线   | 底座                            | 三轴定位                   | 适合的课题                            |
| ---- | ----------------------------- | ---------------------- | -------------------------------- |
| 同学 A | LARP                          | ViT × 离散 1D × AR prior | 自回归视频生成、token 效率、离散表征            |
| 同学 B | LeanVAE                       | CNN × 连续 × 效率          | 高效视频 VAE、diffusion 的轻量 latent 供给 |
| 我们   | OmniTokenizer(工程)+SigLIP2(权重) | 标准 ViT × 连续 × 语义蒸馏     | 统一（重建+语义）视频 tokenizer            |

三条线在设计空间正交，**实验室内部不撞车，反而有三处现成的协同**：

1. **共享评测基础设施**：rFVD/FVD 协议、UCF/DAVIS 数据管线、Latte/DiT 下游验证——三个项目都需要，统一实现一次（正好落在我们 Phase 0 的校准工作里，LeanVAE 和 LARP 的公开 checkpoint 还能给我们的评测脚本多加两个校准锚点）；
2. LeanVAE 可以作为我们表 9 型对比里的**效率基线**，LARP 的 rFVD 20（UCF）是离散路线的参照数字；
3. LARP 的 AR prior 思想 ↔ 我们的可生成性验证 ↔ LeanVAE 的效率约束，可以合写一个"tokenizer latent 该为下游负什么责"的分析视角。

**顺带的待读项**：检索中发现 DeRA（arXiv 2512.04483，"Decoupled Representation Alignment for Video Tokenization"，2025-12）——视频 tokenization + 表征对齐，与我们的语义蒸馏轴直接相关，列入必读清单（优先级在 VTok 之上，因为它在同一栈层）。
