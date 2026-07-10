# OmniTokenizer 代码架构分析（Phase B 改造依据）

> 分析对象：`github.com/FoundationVision/OmniTokenizer`（NeurIPS 2024，Fudan+ByteDance），2026-07-07 克隆实读，非网页转述。
> 用途：① 验证 v2 计划 Phase B（探路消融）的可行性并给出文件级改造点；② 评估路线 A 可移植的组件；③ 提供评测校准锚点。
> 结论先行：**Phase B 完全可行且比预想更顺**——OmniTokenizer 的 1+16 锚帧协议、首帧独立 patchify、单步时间折叠与 Hydra-X 协议同源（Hydra-X 的 clip 协议就是这一系的直系后代），三处改造点都有清晰的接缝；主要风险是 2021 年的 Lightning 依赖。

---

## 1. 仓库地图

```
OmniTokenizer/
├── vqgan_train.py / vqgan_eval.py        # 训练/评测入口(argparse, 164行)
├── scripts/recons/train.sh               # ★三阶段官方配方(超参全在这里, 见§4)
├── scripts/recons/eval_video.sh          # 官方评测命令(K600/UCF, VQ与VAE版)
├── OmniTokenizer/
│   ├── omnitokenizer.py (1118行)         # ★核心: VQGAN(LightningModule) + Encoder/Decoder
│   ├── modules/
│   │   ├── attention.py (689行)          # ★Transformer/Attention/WindowAttention/PEG
│   │   ├── codebook.py                   # VQ codebook(8192×8, l2归一化, 重启机制)
│   │   ├── vae.py                        # DiagonalGaussianDistribution(μ,logvar,KL)
│   │   ├── discriminator.py + base.py    # NLayerDiscriminator(2D图像) + 3D(视频)
│   │   └── lpips.py                      # LPIPS(vendored)
│   ├── quantizer/                        # FSQ/LFQ/residual-VQ全家桶(v2计划的离散变体现成!)
│   ├── data.py (948行)                   # ImageDataset/DecordVideoDataset/VideoData(联合loader)
│   └── trainer.py (1597行)               # 训练辅助(EMA/损失函数/hinge)
├── evaluation/
│   ├── common_metrics_on_video_quality/  # FVD双实现(styleganv版+videogpt版)+PSNR/SSIM/LPIPS
│   └── pytorch-fid/                      # vendored FID
├── Diffusion/
│   ├── DiT/                              # ★latent上训DiT(ImageNet class-cond)——v2 Phase 5直接用
│   └── Latte/                            # ★latent上训视频DiT(UCF101配置含omnitokenizer版yaml)
└── annotations.zip(HF下载)               # ImageNet/UCF/K600等数据列表
```

---

## 2. 模型架构精读（张量流）

### 2.1 总体（`omnitokenizer.py:63` `class VQGAN(pl.LightningModule)`）

```
视频 x [B,3,1+T,H,W]  (sequence_length=17 → 1锚帧+16帧; 断言 (f-1)%pt==0, 与Hydra-X同协议!)
│
├─ 锚帧: to_patch_emb_first_frame (omnitokenizer.py:806)   # 独立线性patchify, 8×8空间
├─ 其余帧: to_patch_emb (:814)                              # ★单步时间折叠: Rearrange pt=4 + Linear
▼  tokens [B, 1+T/4, h, w, 512]
Encoder.encode (:881)
├─ 空间transformer (:894): rearrange '(b t) (h w) d', enc_block="ttww"
│     't'=全局自注意力(+PEG位置编码)  'w'=Swin窗口注意力(window=8, 相对位置偏置)
├─ 时间transformer (:903): rearrange '(b h w) t d', 4层全't'
│     ★causal_in_temporal_transformer=True → SDPA is_causal(全历史因果)
│     ★注意: 时间注意力按空间位置分解(TimeSformer式), 非联合3D注意力
└─ 可选 defer_temporal_pool/defer_spatial_pool (:792-804): AvgPool再压2×(默认关)
▼  h [B, 512, 1+T/4, H/8, W/8]
pre_vq_conv (:144): Linear 512→8(VQ) 或 512→16(VAE的μ,ρ)      # ★瓶颈接缝=GSB的天然落点
├─ VQ: Codebook(8192, dim8, l2) (:142)
└─ VAE: DiagonalGaussianDistribution → sample (vae.py)        # latent C=8 (!对比GSB C=64)
▼  z [B, 8, 1+T/4, H/8, W/8]
post_vq_conv: Linear 8→512 → Decoder (:950)
├─ 时间transformer(:1072) → 空间transformer(:1084, dec_block="tttt"全为全局注意力)
└─ 锚帧/其余帧分别 to_pixels (:1006/:1012)                    # ★单步时间展开(逆patchify)
▼  x̂ [B,3,1+T,H,W]
```

规模感：dim=512、空间 4 层+时间 4 层、8× 空间/4× 时间压缩——encoder+decoder 合计约 8000 万参数，**比我们路线 A 的 So400M 小一个量级**，Phase B 消融很便宜。

### 2.2 损失系统（VQGAN 类内，manual optimization，G/D 交替）

| 项                  | 实现位置                                                                                     | 官方权重                                         |
| ------------------ | ---------------------------------------------------------------------------------------- | -------------------------------------------- |
| L1 重建              | `recon_loss_type/l1_weight`                                                              | 1.0                                          |
| LPIPS 感知           | `modules/lpips.py`（vendored）                                                             | 4.0                                          |
| logits-laplace     | `omnitokenizer.py:23`                                                                    | 0.4（stage1）                                  |
| 图像 GAN             | `NLayerDiscriminator`(2D, hinge)                                                         | 0.01(s1)→0(s2/3)                             |
| 视频 GAN             | `NLayerDiscriminator3D`(hinge)                                                           | 1(s1)→0.01(s2/3)                             |
| GAN 特征匹配           | `gan_feat_weight`                                                                        | 4.0                                          |
| VQ commitment / KL | codebook / `kl_weight`                                                                   | 1.0 / **1e-6**（stage3，解决 v1 文档 KL 量纲矛盾的实证参照） |
| 判别器稳定              | `dis_lr_multiplier 0.1`、`disloss_check_thres 0.001`（D 损失过低时跳过 D 步）、diffaug/noise/blur 增广 | —                                            |

### 2.3 数据管线（`data.py`）

- `ImageDataset` + `DecordVideoDataset`（decord 解码，txt datalist 注释文件，官方提供 annotations.zip）；
- `VideoData(pl.LightningDataModule)`：**联合 loader**——`--loader_type joint --batch_size 4 8 --sample_ratio 1 1` 实现图像/视频源交替采样（我们 v2 `mixture.py` 想要的东西现成）；
- 多分辨率：`--resolution_scale 0.5 0.75 1.0 1.25`（stage2 起）。

---

## 3. Phase B 三个改造点（文件:行级）

### 改造 1：tubelet 时间注意力（预计 ~40 行改动）

- **落点**：`modules/attention.py` `Attention.forward`（SDPA 路径 :451，手写路径 :473-480）。现状：`causal=True` 走 `is_causal` 全历史因果。
- **改法**：新增 `temporal_attn_mode ∈ {full, causal, tubelet}` 配置，从 `omnitokenizer.py:857-861`（时间 transformer 构造处）下传。tubelet = 带状因果 mask（位 t 只见 {t-1, t}），时间序列折叠后仅 1+4=5 位，直接构造 5×5 bool mask 传 `attn_mask` 即可，无需 flex_attention：

```python
# tubelet带状mask: allowed[i,j] = (j==i) | (j==i-1)
idx = torch.arange(T1, device=dev)
allowed = (idx[None,:] == idx[:,None]) | (idx[None,:] == idx[:,None]-1)
out = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)  # 替换 is_causal
```

- **消融映射表**（写论文时必须交代的等价性声明）：

| Hydra-X 表 1（联合 3D 注意力 ViT） | OmniTokenizer 移植版（分解式时空注意力）  |
| -------------------------- | ---------------------------- |
| Full 全时空双向                 | 时间 transformer 双向（causal 关闭） |
| Causal 全历史因果               | 现状（is_causal）                |
| Tubelet 2 帧窗因果             | 带状 mask（上述改法）                |

  空间注意力在三组中都保持逐帧分解——因此 Phase B 检验的是**时间感受野假说**在分解式架构 + 从头训练下是否成立，这正是我们要的边界条件实验。

- **已知混杂（must document）**：PEG 位置编码是时间因果的 depthwise 卷积（`attention.py:298`，kernel 3、causal padding (2,0)）→ 时间感受野 3，略超 tubelet 的 2。第一版保持不动（最小改动原则），结果出来后若差异微妙再加 `--peg_temporal_window` 消融。

### 改造 2：层级时间 patchify（预计 ~80 行）

- **落点**：
  - encoder 输入折叠 `omnitokenizer.py:814-822`（`to_patch_emb` 的 `pt=4` → `pt=2`）；
  - 在 `encode()` 的空间与时间 transformer 之间（:894-903 之间）插入学习式 `TemporalFold2x`（v1 §3.2 的 `TemporalPatchify` 直接搬来，Linear(2d→d)）；
  - decoder 对称：`decode()` 时间与空间 transformer 之间插 `TemporalUnfold2x`，`to_pixels` 的 `pt=4`→`pt=2`（:1012-1017）。
- **意外之喜**：锚帧在此代码库中**从不参与时间折叠**（首帧独立 patchify、pool 只作用于其余帧 :910-914）→ v1 的 ADR-4（锚帧 zero-pad 方向）在路线 B **不存在**，比 Hydra-X 原设计更干净。
- **免费的第三消融臂**：代码自带 `defer_temporal_pool`（:792，输入 2× 折叠 + transformer 后 AvgPool 再 2×）——这是"层级压缩"的无参数退化版。三臂对比 {单步 4×，学习式 2×+2×，2×+AvgPool2×} 能区分"层级好在多阶段"还是"好在有学习参数"，超出 Hydra-X 原表的信息量。

### 改造 3（Phase B 可选，路线 A 主用）：GSB 语义瓶颈

- **落点**：`pre_vq_conv`（:144-160）就是瓶颈接缝——把 Linear(512→16) 换成 GSB（Linear(512→128) 的 μ,ρ + C=64），`post_vq_conv` 对应改。语义分支（Sem 侧 + Decompressor 蒸馏）挂在 z 之后，作为新模块加入，不动重建路径。
- Phase B 不做此项（保持最小变量）；若路线 B 消融结果好，此项使路线 B 升级为完整候选架构。

---

## 4. 官方三阶段配方解码（`scripts/recons/train.sh` 实读）

|             | Stage 1                           | Stage 2                                | Stage 3                    |
| ----------- | --------------------------------- | -------------------------------------- | -------------------------- |
| 目标          | 图像-only 固定 256²                   | 图像+视频联合、多分辨率                           | +KL 微调成 VAE                |
| 时间 patchify | pt=2                              | **pt=4**                               | pt=4                       |
| lr          | **1e-3**(warmup 50k, cosine→5e-5) | 5e-5                                   | 5e-5                       |
| 位置编码        | 相对偏置("rel")                       | **切换 RoPE**("rope")                    | rope                       |
| GAN         | image 0.01 / video 1              | image 0 / video 0.01                   | 同左                         |
| 步数          | 500k                              | 500k                                   | （较短）                       |
| 特殊          | logitslaplace 0.4, initialize_vit | force_alternation, init_vgen "average" | --use_vae --kl_weight 1e-6 |

注意与 Hydra-X 配方的结构同源性（图像先行→联合→再加最后一个目标），Phase B 只需跑 s1 缩短版（150k）+ s2 缩短版（150k），按 v2 预算 ~2,500 GPU·h 成立（模型仅 80M）。

---

## 5. 评测资产与校准锚点（直接解决 v1 批判 #11/12 的一部分）

- 仓库自带 **FVD 双实现**（styleganv 版 + videogpt 版，`evaluation/common_metrics_on_video_quality/`）与 vendored pytorch-fid，且 `vqgan_eval.py` 就是他们论文数字的产生器；
- **校准锚点**（官方 checkpoint + 官方脚本，README 公示数字）：
  - `imagenet_ucf.ckpt`（VQ）：FID 1.11 / UCF FVD 42.35；
  - `imagenet_ucf_vae.ckpt`：FID 0.69 / FVD 23.44；
  - `imagenet_k600.ckpt`：FID 1.23 / K600 FVD 25.97；
- **校准流程**：先用他们的 ckpt + 他们的 eval 脚本复现上述数字（验证环境正确）→ 再用**我们的** `protocols.py`/rFVD 实现跑同一 ckpt（校准我们的尺子）→ 两套数字的映射关系记录在案。这比 v1 只靠 AToken/Wan2.2 对表 9 的校准多了一个"实现与数字同源"的锚点，循环论证风险显著降低；
- README 脚注警告：ImageNet-only 1.28 FID 的复现需注释掉 SDPA 路径（`attention.py:446-460`）——**官方承认 SDPA 与手写注意力数值不等价**。校准时两条路径都跑，差异记录在案（这正是 v1 防线 2 要抓的那类漂移）。

## 6. 路线 A 可移植组件清单

| 组件                                                           | 移植建议                                             |
| ------------------------------------------------------------ | ------------------------------------------------ |
| 联合 loader（`VideoData`，多源 batch_size/sample_ratio/多分辨率）       | ✅ 直接移植，改造成 accelerate 版                          |
| 双判别器 + hinge + 特征匹配 + disloss 阈值跳步 + diffaug                 | ✅ 直接移植（v1 计划里"重写 PatchGAN"可取消，用现成的）              |
| LPIPS、FVD 双实现、pytorch-fid                                    | ✅ vendored，直接用                                   |
| `quantizer/` FSQ/LFQ 全家桶                                     | ✅ v2 §2.2 离散变体现成                                 |
| Diffusion/DiT + Latte（latent 上训生成模型，含 OmniTokenizer 专用 yaml） | ✅ **v2 Phase 5 的可生成性协议直接用此管线**，省自建               |
| 锚帧协议（首帧独立 patchify/pool 隔离）                                  | ✅ 设计模式移植进路线 A（替代 Hydra-X 的 zero-pad 方案，消掉 ADR-4） |
| LightningModule 训练循环                                         | ⚠️ 见 §7 依赖风险，路线 A 建议只借逻辑、落在 accelerate 上         |

## 7. 风险与坑（实读发现）

1. **依赖年代**：`requirements.txt` 锁 `pytorch-lightning==1.5.4`（2021-12）却同时列 `lightning`、`tensorflow`；README 要求 torch 2.2.1+cu118。这套组合脆弱——**Phase B 用 docker 按 README 锁死环境（python3.10 + torch2.2.1 + cu118 + PL1.5.4），不要试图升级 PL**；升级 = 未知工作量的移植工程，不值得在探路阶段做。
2. `DecordVideoDataset` 依赖 decord，但 requirements 未列且 decord 在 Windows 上基本不可用——**Phase B 必须在 Linux 上跑**（训练本来也应如此）。
3. SDPA/手写注意力数值不等价（§5 脚注）——消融的三个 mask 变体**必须走同一条注意力实现路径**，否则引入实现混杂。
4. VAE latent C=8 vs 我们 GSB C=64：对比可生成性实验时（Phase 5），latent 维度是显著混杂变量，报告时必须注明并考虑加一组 C 对齐的对照。
5. `batch norm`（`--norm_type batch` + sync_batchnorm）遍布判别器与 patchify——多卡数值行为与 LN 不同，改动 batch 布局（如联合 loader 比例）会隐性影响 BN 统计，消融时保持 batch 配置完全一致。
6. K600/MiT/Sthv2 数据不易全量获取——Phase B 用 ImageNet+UCF101 组合（官方有对应 ckpt 与 datalist，校准闭环完整）。

## 8. 对 v2 计划的修订建议（已核实代码后）

1. Phase B 预算维持 ~2,500 GPU·h，**可行性从"假设"升级为"已验证"**：三个改造点均有清晰接缝，模型只有 ~80M；
2. Phase B 消融矩阵扩为：3 mask × {单步, 学习式层级, AvgPool 层级} 中取 5~6 个组合（全 3×3 不必要）；
3. Phase 5 改用仓库自带 DiT/Latte 管线，预算可从 5,000 降到 ~3,500 GPU·h；
4. v1 的 ADR-4（锚帧 pad 方向）在路线 B 作废；路线 A 改用 OmniTokenizer 的"锚帧隔离"模式，同样作废——从风险登记册移除；
5. KL 量纲问题落定：同为线性瓶颈+L1/LPIPS 损失体系的 OmniTokenizer 用 **1e-6**，v1 文档中"1e-4（HYDRA）"与"1e-6~1e-8（VAE 惯例）"的矛盾按 1e-6 起步、向上扫到 1e-4 解决。
