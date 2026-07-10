# VITbaseVideoTokenizer (UVT)

> **Unified Video Tokenizer** — 单个 ViT（SigLIP2-So400M 初始化）做图像+视频统一、重建+语义统一、latent 可生成的视频 tokenizer。借鉴 Hydra-X（arXiv:2606.13289），收窄为"只做 tokenizer，不做 UMM"。

## 这是什么

一个统一视频 tokenizer（工作代号 UVT）：一套权重同时处理图像与视频（1 锚帧 + 16 帧 clip 协议），latent 既能高保真重建像素，又携带可线性读出的语义，且冻结后可在其上训 DiT 收敛。交付 encoder/decoder 权重 + 训练代码 + 论文级实验矩阵。详见 [`docs/README.md`](docs/README.md)（项目总纲）。

## 仓库结构

```
docs/                       # 全部设计文档（权威）
├── README.md               # 项目定义 + D1–D15 决策 + 术语 + 分工 + 预算
├── 01_架构与方法.md         # 模型完整规格 + 张量流 + ADR
├── 02_代码实施.md           # 仓库/环境/逐文件改造清单/单元测试
├── 03_数据与评测.md         # 数据管线/评测协议/五锚点校准
├── 04_实验与验收.md         # 逐实验枚举/Gate/统计纪律/风险登记册
├── 05_代码总体架构与实现任务书.md  # 逐文件任务卡（接口冻结到签名级）
├── 06_实施编排与下一步计划.md      # 编排 + 契约修订(§6) + 实测验证(§9)
├── 07_服务器运行交接.md     # 环境/权重/数据/运行命令（给服务器端）
├── 08_给服务器GLM5_2的项目认知与交接.md  # 心智模型+历史教训（接手第一篇读）
├── 损失详情.md             # 全部损失函数教学版详解
├── code/                   # 样板实现（attention_mask.py / temporal_fold.py）
└── background/             # 降级为背景资料的早期调研文档
uvt/                        # 主仓代码（fork 自 LARP 改造）—— 真实训练在这
├── models/uvt/             # M-1~M-10 模型全实现
├── losses/ teachers/ datasets/ trainers/ eval/ tests/
├── cfgs/uvt_stage1.yaml    # 训练配置（已就绪）
└── train.py                # torchrun 入口
phase-b-omnitokenizer/      # Phase B 边界条件实验仓（fork 自 OmniTokenizer，独立 docker）
```

## 快速开始

1. **读懂项目**：先读 [`docs/08_给服务器GLM5_2的项目认知与交接.md`](docs/08_给服务器GLM5_2的项目认知与交接.md)（心智模型 + 已踩坑），再按需读 01–05。
2. **运行**：按 [`docs/07_服务器运行交接.md`](docs/07_服务器运行交接.md) 装环境、下权重、跑 null 冒烟 → P1-smoke Gate。

## 当前状态

- **25 张任务卡全部实现**（模型 M-1~M-10 / 损失 L-1~3 / 教师 T-1~2 / 数据 D-1~3 / 训练 TR-1~2 / 评测 E-1~4 / Phase B B-1~2）
- **84 测试绿**（CPU 实证）
- **全 trainer null 冒烟跑通完整训练 epoch**（loss 下降，梯度经全链路回流）
- **14 个真 bug 全修**（前向兼容 / 配置交接 / 模型契约类，详见 docs/06 §9、docs/08 §6）

## 权重下载（gitignore 的二进制）

仓库不含预训练权重（`.gitignore` 排除 `*.pt/*.ckpt/*.pth`）。按 `docs/07` §2 下载：
- **SigLIP2-So400M-patch16-256**（`google/siglip2-so400m-patch16-256`）—— 一模三用（初始化+图像教师+文本塔），P1 硬前提
- **InternVideo 视频教师** + **DINOv2-L**（CKNNA 参照）+ **五锚点校准权重**
- **FVD 的 I3D 权重**（`i3d_torchscript.pt` / `i3d_pretrained_400.pt`）—— 从 OmniTokenizer release 下载到 `phase-b-omnitokenizer/evaluation/`

## License

主仓 uvt 基于 LARP (MIT) 改造；phase-b 基于 OmniTokenizer (MIT)。AToken (Apple Sample Code) 仅只读参考，不入仓。
