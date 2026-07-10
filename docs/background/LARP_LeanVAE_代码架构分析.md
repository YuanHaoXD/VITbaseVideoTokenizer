# LARP / LeanVAE 代码架构分析（主仓骨架选型依据）

> 2026-07-07 克隆官方仓库实读（hywang66/LARP @ ICLR25 Oral；westlake-repl/LeanVAE @ ICCV25），规格同《OmniTokenizer_代码架构分析.md》。
> **结论先行**：① LARP 的自研 trainer 是三个仓库中唯一的现代无 Lightning 框架，**具备做主仓工程骨架的条件**（需补两处：梯度累积、torchrun 多机启动）；它的 transformer 判别器 + LeCam 方案比 OmniTokenizer 的 PatchGAN 更现代，直接搬给我们的 Stage 2。② **LeanVAE 的依赖同样锁死 PL 1.5.4**（"现代工程"优势只属于 LARP），不做骨架，但其评测套件、DAVIS 数据类、效率基线价值很高。③ OmniTokenizer 的角色收缩为：Phase B 实验台 + 图像视频联合 loader 的设计参考。

---

## 1. LARP 代码架构精读

### 1.1 仓库地图

```
LARP/
├── train.py (171行)                  # 入口: yaml+$var$配置系统 → mp.spawn(main_worker)
├── trainers/
│   ├── base_trainer.py (894行)       # ★自研trainer: 裸PyTorch DDP + torch.compile
│   │                                 #   + AMP(fp16/bf16) + 多decay EMA字典 + wandb/tensorboard
│   └── larp_tokenizer_trainer.py     # tokenizer任务子类(G/D交替等)
├── models/
│   ├── larp_tokenizer.py (570行)     # ★LARPTokenizer(nn.Module, PyTorchModelHubMixin)
│   ├── bottleneck.py (383行)         # VQ正则器(l2归一化/随机量化/entropy可选)
│   ├── gptc.py                       # AR prior(GPT-C, mix self-sampling)
│   └── loss.py (452行)               # ★L1+LPIPS+transformer判别器+LeCam
├── datasets/video_dataset.py (353行) # decord+csv列表; 支持多csv拼接; ★null假数据集(冒烟测试)
├── eval/ (rfvd_evaluator等)          # rFVD评测
├── cfgs/larp_tokenizer.yaml          # 全部超参数(见1.3)
└── requirements.txt                  # ★无Lightning! pandas/einops/timm/wandb/decord/lpips
```

### 1.2 训练框架细节（承载力评估的核心）

| 能力 | 状态 | 备注 |
|---|---|---|
| 分布式 | 裸 DDP（`mp.spawn` + `init_process_group(dist_url)`，base_trainer.py:130,388） | 单机多卡风格；多机需改 torchrun 入口（半天工作量） |
| 混合精度 | ✅ AMP fp16/bf16 开关 | |
| torch.compile | ✅ 内置（:383，三种 mode） | |
| EMA | ✅ **多 decay 并行 EMA 字典**（:395-410，可同时维护多个 decay 的影子模型） | 比常见单 EMA 更好 |
| 实验管理 | ✅ yaml + `$var$` 命令行替换的配置系统；wandb + tensorboard 双日志 | 与我们"实验卡"纪律兼容 |
| 冒烟测试 | ✅ **自带 null 假数据集**（video_dataset.py:90，`csv_file=null128` 即生成假数据跑通全链路） | 正好落实 v1 防线 3 |
| 梯度累积 | ❌ 无 | 需补（~50 行） |
| 断点续训/checkpoint | ✅（含 EMA 状态） | |
| HF Hub 集成 | ✅ 模型继承 PyTorchModelHubMixin，`from_pretrained` 直接可用 | |

**判定：具备主仓骨架条件**。三仓对比——LARP（无 Lightning、torch 2.4 系、干净）≫ OmniTokenizer / LeanVAE（双双锁死 pytorch_lightning==1.5.4）。

### 1.3 模型与损失（对我们的可搬性）

- **架构**（cfgs/larp_tokenizer.yaml + larp_tokenizer.py）：标准 ViT-B 规格 encoder/decoder（768 宽 ×12 层 ×12 头）；视频 patchify = 空间 8×8 + **单步时间 4×**；encoder 里拼接 **1024 个可学习 holistic query**（`transformer_encoder_parallel`：patch token 与 query token 联合注意力，query 聚取全局信息后进瓶颈）；可分解 t/h/w 学习式位置嵌入；瓶颈 dim 16 + VQ 8192（l2 归一化 + 随机量化 τ=0.03）。
  - **不可搬**：holistic query 机制与我们的空间网格 latent 冲突；VQ 与我们的 GSB 冲突；无锚帧协议（整段视频统一 patchify，图像支持缺失）。
  - **确认此前判断**：它没有"时间注意力轴"（query 对全部时空 patch 全局注意），tubelet/层级 patchify 消融在它身上问不出来——Phase B 只能在 OmniTokenizer 上做。
- **损失（loss.py，高价值可搬件）**：L1 + LPIPS + **transformer 判别器**（384 宽 ×8 层，对时空 patch 做判别）+ `ns_smooth` 生成损失 + **LeCam 正则 0.001** + disc_weight 0.3 + **D 每 5 步更新一次**（d_update_freq）+ 可选 R1；Adam betas (0.5, 0.9)。
  - 这套是 MAGVIT-v2 系的现代 GAN 配方，比 OmniTokenizer 的 PatchGAN + BatchNorm 更稳、更贴我们的"纯 transformer"叙事。**建议替换 v2 计划 Stage 2 的判别器方案**（原计划用 taming 系 PatchGAN）。
- **AR prior（gptc.py）**：不搬进主线（我们用 DiT 验证可生成性），但它是"训练时塑造 latent 可生成性"的参照实现，Phase 5 讨论时引用。
- **数据（video_dataset.py）**：decord + csv 列表，video-only，支持多 csv `'+'` 拼接与类别/数量筛选。**缺图像数据集与图像视频混合采样**——这是从 OmniTokenizer 移植联合 loader 思路的落点。

### 1.4 协议注意点

官方配方是 UCF-101 **128²×16 帧**、epoch 制（400 epoch）、Adam(0.5,0.9)。我们主线是 256²×(1+16) 锚帧协议 + 步数制——搬骨架时训练日程模块要按我们的协议重写，不能沿用它的 epoch/低分辨率默认值。

---

## 2. LeanVAE 代码架构精读

### 2.1 仓库地图与依赖

```
LeanVAE/
├── leanvae_train.py                  # PL 1.5.4 Trainer 入口(ModelCheckpoint/WandbLogger/VideoLogger)
├── LeanVAE/
│   ├── models/autoencoder.py         # LeanVAE(nn.Module): encode/decode, 时间分块推理
│   ├── models/autoencoder_pl.py      # AutoEncoderEngine(pl.LightningModule)
│   └── modules/backbones.py          # ★DWT小波双路(低频/高频子带) + ResNAF + ISTA展开(压缩感知瓶颈) + PEG3D
├── evaluation/                       # ★PSNR/SSIM/LPIPS/FVD(styleganv+videogpt双实现) + DAVIS数据类
├── generation/Latte/                 # Latte集成(vendored, 含LeanVAE专用配置)
└── requirements.txt                  # ❌ pytorch_lightning==1.5.4; torch 2.3.0+cu118
                                      #   且 torchvision==0.14.1 与 torch 2.3 不匹配(requirements本身有坑)
```

### 2.2 架构要点

- **数据流**：输入先做 DWT 小波分解（低频 LL + 高频子带两路）→ 双路 ResNAF（深度可分离卷积式前馈）→ 融合 → **ISTA 展开网络**做通道瓶颈（压缩感知的迭代软阈值算法展开成网络层）→ 因果 latent。
- **图像+视频统一**：✅ 有（autoencoder.py:61-88：`ndim==4` 自动升为 t=1；视频压缩 4×8×8、图像 1×8×8）——**又一个"首帧因果隔离"协议的实例**，与 Wan-VAE/OmniTokenizer/Hydra-X 一脉相承，进一步确认锚帧协议是行业共识。
- **长视频**：时间分块推理（tile_inference + first_chunk 因果处理），工程完成度高。
- **判定**：conv 专用架构 + PL 1.5.4，**不适合做我们的骨架**；但作为 40M 的效率极点基线必须进对比表，且其评测套件质量好。

---

## 3. 三仓库合并后的最终零件表（v2 §1 选型的落地修订）

| 需求 | 来源 | 说明 |
|---|---|---|
| **主仓训练骨架**（DDP/AMP/compile/EMA/配置系统/日志） | **LARP base_trainer 改造** | 补梯度累积 + torchrun 入口；同学有实操经验 |
| GAN 方案（Stage 2） | **LARP loss.py**（transformer 判别器 + LeCam + ns_smooth + d_update_freq） | 替换原计划的 taming PatchGAN |
| 冒烟测试机制 | LARP null 假数据集模式 | 落实 v1 防线 3 |
| 图像+视频联合 loader | OmniTokenizer `VideoData` 设计思路移植进 LARP 骨架 | 多源 batch_size/sample_ratio/多分辨率 |
| Phase B 实验台 | OmniTokenizer（docker 锁 PL 1.5.4） | 科学原因不变：唯一有时间注意力轴+锚帧协议的"Hydra-X 改造前状态" |
| rFVD/FID 评测 | OmniTokenizer 或 LeanVAE 的 styleganv+videogpt 双实现（两家同源） | 校准锚点现有四家：AToken、Wan2.2、OmniTokenizer ckpt、LARP ckpt（UCF rFVD 20）、LeanVAE ckpt |
| DAVIS 评测数据类 | LeanVAE `evaluation/dataset/davis.py` | 现成 |
| 可生成性验证（Phase 5） | OmniTokenizer 的 DiT/Latte 或 LeanVAE 的 Latte 集成 | 两家都 vendored 了 Latte，选一即可 |
| 效率基线 | LeanVAE checkpoint（40M/4ch/16ch） | 进表 9 型对比 |
| 模型核心（ViT 切分/GSB/tubelet/层级 patchify/Decompressor） | **自写**（v1 §3 参考实现） | 谁的架构都不搬 |
| 模型分发 | LARP 的 PyTorchModelHubMixin 模式 | 发布时直接 HF from_pretrained |

## 4. 风险与坑（实读发现）

1. LARP trainer 无梯度累积、入口为单机 mp.spawn——上多机/大 batch 前必须补齐并用 null 数据集回归测试；
2. LARP 官方协议 128²/epoch 制/Adam(0.5,0.9)——那套超参为 GAN 调的，我们主线优化器仍按 v2 的 AdamW 方案，只搬框架不搬超参；
3. LeanVAE requirements 里 torch 2.3 配 torchvision 0.14.1 是不成立的组合，同学如果踩过环境坑，这就是根源；跑它的基线时以其 README 实际验证过的组合为准；
4. 三家 FVD 实现虽同源（styleganv/videogpt 两版并存），但**必须统一选定一版**并在四个公开 ckpt 上交叉校准后写死进 `protocols.py`。

## 5. 分工落地（更新版）

- **熟 LARP 的同学**：把 base_trainer 改造成主仓 trainer（补梯度累积/torchrun）→ 移植 transformer 判别器损失 → Phase 5 可生成性（其 AR prior 经验直接相关）；
- **熟 LeanVAE 的同学**：评测协议统一（双 FVD 校准 + DAVIS/UCF 管线 + 四 ckpt 锚点）→ 图像视频联合 loader（参照 OmniTokenizer 设计）→ LeanVAE 效率基线入表；
- **你**：模型核心（SigLIP2 加载与切分、GSB、tubelet mask、层级 patchify、Decompressor 蒸馏）+ Phase B 在 OmniTokenizer 上的探路消融。
