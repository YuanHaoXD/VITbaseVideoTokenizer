# Phase B 改造清单（fork 自 OmniTokenizer，docker 锁死，一次性）

> 契约权威：`UVT-Final/02_代码实施.md` §3.1 / §3.2 / §3.3、`UVT-Final/05_代码总体架构与实现任务书.md` §5 B-1/B-2 卡。
> 这是 Hydra-X 三设计"是否依赖预训练先验"的科学增量（README D6）。
> 文件:行号基于 2026-07-07 主分支；本仓改完打 tag `phase-b-v1`。

---

## 0. 行号漂移报告（EXPECTED_OUTCOME 要求项）

**结论：02 §3.1 / §3.2 标注的所有行号与当前（2026-07-09 检出）主分支完全一致，零漂移。** 以下逐条核对（均为改动前的原始行号）：

| 02 §3 标注 | 实际主分支行 | 内容 | 一致? |
|---|---|---|---|
| attention.py `:451` | 451 | SDPA 调用 `F.scaled_dot_product_attention(...)` | ✓ |
| omnitokenizer.py `:857-861` | 857-861 | `enc_temporal_transformer` 构造 | ✓ |
| omnitokenizer.py `:814-822` | 814-822 | `to_patch_emb` (linear 分支) | ✓ |
| omnitokenizer.py `:894-903` | 894-903 | `encode()` 空间→时间 transformer 区间 | ✓ |
| omnitokenizer.py `:910-914` | 910-914 | 锚帧隔离 pool（只作用 `tokens[:,:,1:]`） | ✓ |
| omnitokenizer.py `:792-795` | 792-795 | `defer_temporal_pool` 现成机制 | ✓ |
| omnitokenizer.py `:1012-1017` | 1012-1017 | decoder `to_pixels` | ✓ |
| omnitokenizer.py `:1071-1084` | 1071-1084 | decoder `decode()` 时序→空间区间 | ✓ |

下文"改动清单"同时给出**改动前原始行号**（对齐 02 §3）与**改动后当前行号**（便于复核本仓 HEAD）。

---

## 1. 逐改动清单

### B-1：时间注意力三模式（`OmniTokenizer/modules/attention.py`）

| 改动后行 | 改了什么 | 为何（对齐 02 §3.1） |
|---|---|---|
| 355 | `Attention.__init__` 增参 `temporal_attn_mode="causal"` | 三模式统一开关入口 |
| 361-364 | `assert temporal_attn_mode in (...)` + `self.temporal_attn_mode = ...` | 早失败：非法值立即报错而非静默走 full |
| 446-459 | `forward` SDPA 分支新增 `if not is_spatial:` 三模式派发 | **纪律核心**：tubelet/causal/full 三模式全部走 SDPA（`F.scaled_dot_product_attention`），不混入手写路径；tubelet 构造 `allowed` bool mask 传 `attn_mask`，causal 用 `is_causal=True`，full 无 mask。门控 `not is_spatial` 保证空间注意力路径（`is_spatial=True`）行为字节级不变 |
| 629 | `Transformer.__init__` 增参 `temporal_attn_mode="causal"` | 透传链路中段 |
| 640 | `'t'` block 的 `Attention(...)` 构造增 `temporal_attn_mode=temporal_attn_mode` | 透传链路末端 |

**SDPA 纪律复核**：原 SDPA 分支（行 444 起）被拆为 `not is_spatial`（三模式，全 SDPA）与 `else`（空间，原 SDPA 逻辑）两条。原非 SDPA 手写分支（行 472+）**未触动**——它在 docker 环境（torch 2.2.1）永不触发，保留仅为代码完整性。三模式无人走手写路径，满足 02 §3.1 "三模式必须统一走 SDPA 路径"。

### B-1 透传：`OmniTokenizer/omnitokenizer.py`

| 改动后行 | 改了什么 | 为何 |
|---|---|---|
| 886-887 | `enc_temporal_transformer` 构造前 `transformer_kwargs["temporal_attn_mode"] = temporal_attn_mode` | 仅时间 transformer 透传（空间 transformer 保持默认，但 `is_spatial=True` 时本就忽略此参）。镜像原 `causal_in_temporal_transformer` 的透传写法（原 :857-858） |
| 1028-1029 | `dec_temporal_transformer` 构造前同上 | decoder 对称 |

### B-2：层级时间 patchify 三臂

#### B-2-a 新文件 `OmniTokenizer/modules/temporal_fold.py`（learned 臂专用）

- 从主仓 `uvt/models/uvt/temporal_fold.py` 拷贝；`dim` 已参数化（learned 臂构造期传 `dim=512`，对应内部 `Linear(2*512=1024 → 512)`）。
- `TemporalFold2x`：`[B,1+T,N,D] → [B,1+T/2,N,D]`，锚帧（位 0）隔离（`tokens[:,1:]` 才参与），近恒等初始化（权重=两帧平均）。
- `TemporalUnfold2x`：`[B,1+T,N,D] → [B,1+2T,N,D]`，镜像逆算子（权重=复制两份）。

#### B-2-b `OmniTokenizer/omnitokenizer.py`

| 改动后行 | 改了什么 | 为何（对齐 02 §3.2） |
|---|---|---|
| 19 | `from .modules.temporal_fold import TemporalFold2x, TemporalUnfold2x` | 引入 learned 臂模块 |
| 105-116 | `VQGAN.__init__` 新增三臂派发块：默认值兜底 + `assert` + **avgpool 臂强制 `args.defer_temporal_pool=True`**（复用现成机制 :792-795，零额外改动）+ 存 `self.temporal_{attn,fold}_mode` | 三臂统一开关入口；avgpool 臂"零改动"靠复用 defer 机制实现 |
| 125 / 134 | `OmniTokenizer_Encoder(...)` 与 `OmniTokenizer_Decoder(...)` 构造增 `temporal_attn_mode=..., temporal_fold_mode=...` | 透传到 Encoder/Decoder |
| 764-768 | `add_model_specific_args` 增 `--temporal_attn_mode {tubelet,causal,full}` 与 `--temporal_fold_mode {single,learned,avgpool}` | CLI 入口。`vqgan_train.py` 已通过既有 `OmniTokenizer_VQGAN.add_model_specific_args` 调用自动获得这两个参数，故 vqgan_train.py 对此无需改动 |
| 799, 807-808 | `OmniTokenizer_Encoder.__init__` 增参 + `self.temporal_fold = TemporalFold2x(dim) if temporal_fold_mode == "learned" else None` | learned 臂才创建 Fold 模块（single/avgpool 臂参数量零增量） |
| 929-935 | `encode()` 空间 transformer 之后、时间 transformer 之前插 `if self.temporal_fold is not None: Fold2x`（rearrange 到 `[B,T,N,D]` → Fold → 还原） | 02 §3.2 learned 臂核心。锚帧隔离由 Fold 内部保证（`tokens[:,1:]` 参与），与 :910-914 pool 隔离一致 |
| 992, 999-1000 | `OmniTokenizer_Decoder.__init__` 增参 + `self.temporal_unfold = TemporalUnfold2x(dim) if temporal_fold_mode == "learned" else None` | decoder 对称 |
| 1120-1125 | `decode()` 时间 transformer 之后、空间 transformer 之前插 `if self.temporal_unfold is not None: Unfold2x` | encoder Fold 的镜像。Unfold 后 `t` 增大，后续 rearrange（行 1132）按总数推断新 `t`，无需手算 |

**to_patch_emb / to_pixels "改 pt=2" 说明**：02 §3.2 写的"to_patch_emb（:814-822）改 pt=2 / to_pixels（:1012-1017）改 pt=2"是**运行时参数**而非代码改动——这两个 Sequential 已由 `temporal_patch_size` 参数化，learned 臂运行命令传 `--temporal_patch_size 2` 即生效，零代码改动。本仓未触碰 to_patch_emb / to_pixels 的代码体。

### vqgan_train.py

| 改动后行 | 改了什么 | 为何 |
|---|---|---|
| 24-25 | 增 `--seed` 参数（默认 1234，保持原硬编码值） | 02 §3.3 命令示例用 `--seed 1`，base 原硬编码 1234 不可配；字段化以支持配对消融 D13 |
| 32 | `pl.seed_everything(args.seed)`（原行 13 的 `pl.seed_everything(1234)` 已删，移到 parse_args 之后） | 用字段化 seed |

---

## 2. 三模式可达性说明（B-1 tubelet 核心）

tubelet 模式的 `allowed` mask（attention.py 行 452-454）：
```python
idx = torch.arange(T1, device=q.device)
allowed = (idx[None] == idx[:, None]) | (idx[None] == idx[:, None] - 1)
```
语义：位 `i` 可见位 `j` 当且仅当 `j == i` 或 `j == i - 1`，即**位 t 只见 {t-1, t}**。

以 T1=4 为例，`allowed` 矩阵（行=查询位 i，列=键位 j，True=允许）：
```
i=0: [T, F, F, F]   ← 锚位只见自己
i=1: [T, T, F, F]   ← 位1见 {0,1}
i=2: [F, T, T, F]   ← 位2见 {1,2}，不见位0 ✓
i=3: [F, F, T, T]   ← 位3见 {2,3}，不见位0/1 ✓
```
**关键性质：位 2 不见位 0**（与 05 §5 B-1 验收"5×5 mask 可达性断言"及 02 §5 `test_tubelet_reachability` 一致）。causal 模式位 2 见 {0,1,2}，full 模式见全部——三模式形成严格递增的信息流入梯度，正是 Hydra-X"预训练先验依赖度"探路的对照轴。

SDPA bool mask 约定：True=允许参与注意力（与本仓既有 SDPA 调用 `mask` 的用法一致——非 SDPA 分支 `masked_fill(~mask, -inf)` 同义），故 `allowed` 直接传入 `attn_mask`。

---

## 3. 三臂参数量差异预期（B-2）

| 臂 | 开关 | temporal_patch_size | 额外机制 | 相对 single 臂的参数增量 |
|---|---|---|---|---|
| single（基线） | `--temporal_fold_mode single` | 4 | 无 | 0（基线） |
| learned | `--temporal_fold_mode learned` | 2 | Fold2x + Unfold2x（仅 learned 臂创建） | **+2×Linear(1024,512)**：Fold `Linear(2·512→512)` 与 Unfold `Linear(512→2·512)`，各一连一解。dim=512 下 Fold=524800 参、Unfold=525312 参，合计 ≈1.05M |
| avgpool | `--temporal_fold_mode avgpool` | 2 或 4 | 强制 `defer_temporal_pool=True`（AvgPool3d 无参） | **0**（AvgPool3d 无可训参数） |

符合 05 §5 B-2 验收"learned 多 2×Linear(1024,512)"。

**压缩比算术提示（供架构方确认）**：17 帧、dim=512 时各臂的时间位总数（含锚）：
- single pt=4：(17-1)/4 + 1 = **5 位**（4× 总压缩）
- learned pt=2 + Fold2x：(17-1)/2 + 1 = 9 → Fold → **5 位**（4×，与 single 一致 ✓）
- avgpool pt=2 + defer：defer 使 patchify pt=1（16→16）+ AvgPool3d(2,1,1)（16→8）= **9 位**（仅 2× 总压缩）

> **注意**：02 §3.2 文字写 avgpool 臂用 `--temporal_patch_size 2 --defer_temporal_pool`，按此字面执行得到 2× 压缩（9 位），与 single/learned 的 4×（5 位）不等。若架构方意图是 learned-vs-avgpool **等压缩对照**，avgpool 臂应改用 `--temporal_patch_size 4 --defer_temporal_pool`（pt=2 patchify + 2× AvgPool = 4× → 5 位，与 learned 对齐）。本仓开关逻辑对两种 temporal_patch_size 都成立，命令见下文 §4，由架构方裁决实际跑哪个。

---

## 4. 三模式 × 三臂 run 命令（对齐 02 §3.3，照抄并补全）

公共骨干（02 §3.3 给定）：`--patch_embed linear --patch_size 8 --spatial_depth 4 --temporal_depth 4 --embedding_dim 512 --enc_block ttww --dec_block tttt --twod_window_size 8 --causal_in_peg --dim_head 64 --heads 8 --n_codes 8192 --codebook_dim 8 --l2_code --commitment_weight 1.0 --no_random_restart --gpus 8 --batch_size 8 --lr 1e-3 --warmup_steps 20000 --max_steps 150000 --loader_type joint --resolution 256 --sequence_length 17 --perceptual_weight 4 --image_gan_weight 0.01 --video_gan_weight 1 --gan_feat_weight 4 --train_datalist annotations/imagenet_train.txt annotations/ucf_train.txt`。

下面用 `<MODE>` ∈ {tubelet, causal, full}、`<FOLD>` ∈ {single, learned, avgpool} 占位。每个组合一条命令，seed 1。

```bash
# 臂1 single（基线，4×）：temporal_patch_size 4
python vqgan_train.py --tokenizer omnitokenizer --patch_embed linear --patch_size 8 \
  --temporal_attn_mode <MODE> --temporal_fold_mode single --temporal_patch_size 4 \
  --spatial_depth 4 --temporal_depth 4 --embedding_dim 512 --enc_block ttww --dec_block tttt \
  --twod_window_size 8 --causal_in_peg --dim_head 64 --heads 8 \
  --n_codes 8192 --codebook_dim 8 --l2_code --commitment_weight 1.0 --no_random_restart \
  --gpus 8 --batch_size 8 --lr 1e-3 --warmup_steps 20000 --max_steps 150000 \
  --loader_type joint --resolution 256 --sequence_length 17 \
  --perceptual_weight 4 --image_gan_weight 0.01 --video_gan_weight 1 --gan_feat_weight 4 \
  --train_datalist annotations/imagenet_train.txt annotations/ucf_train.txt \
  --seed 1

# 臂2 learned（4×：pt=2 patchify + Fold2x 2×）
python vqgan_train.py --tokenizer omnitokenizer --patch_embed linear --patch_size 8 \
  --temporal_attn_mode <MODE> --temporal_fold_mode learned --temporal_patch_size 2 \
  --spatial_depth 4 --temporal_depth 4 --embedding_dim 512 --enc_block ttww --dec_block tttt \
  --twod_window_size 8 --causal_in_peg --dim_head 64 --heads 8 \
  --n_codes 8192 --codebook_dim 8 --l2_code --commitment_weight 1.0 --no_random_restart \
  --gpus 8 --batch_size 8 --lr 1e-3 --warmup_steps 20000 --max_steps 150000 \
  --loader_type joint --resolution 256 --sequence_length 17 \
  --perceptual_weight 4 --image_gan_weight 0.01 --video_gan_weight 1 --gan_feat_weight 4 \
  --train_datalist annotations/imagenet_train.txt annotations/ucf_train.txt \
  --seed 1

# 臂3 avgpool（按 02 §3.2 字面：pt=2 + defer → 2× 总压缩）
python vqgan_train.py --tokenizer omnitokenizer --patch_embed linear --patch_size 8 \
  --temporal_attn_mode <MODE> --temporal_fold_mode avgpool --temporal_patch_size 2 \
  --spatial_depth 4 --temporal_depth 4 --embedding_dim 512 --enc_block ttww --dec_block tttt \
  --twod_window_size 8 --causal_in_peg --dim_head 64 --heads 8 \
  --n_codes 8192 --codebook_dim 8 --l2_code --commitment_weight 1.0 --no_random_restart \
  --gpus 8 --batch_size 8 --lr 1e-3 --warmup_steps 20000 --max_steps 150000 \
  --loader_type joint --resolution 256 --sequence_length 17 \
  --perceptual_weight 4 --image_gan_weight 0.01 --video_gan_weight 1 --gan_feat_weight 4 \
  --train_datalist annotations/imagenet_train.txt annotations/ucf_train.txt \
  --seed 1
#  （等压缩对照替代：把上行 --temporal_patch_size 2 改为 4，得 4× 总压缩，与 learned 对齐）

# 评测：复用官方 vqgan_eval.py + evaluation/ 的 FVD
```

> `--temporal_fold_mode avgpool` 会自动强制 `defer_temporal_pool=True`（VQGAN.__init__ 行 113-114），无需在命令里再加 `--defer_temporal_pool`。

---

## 5. 改动规模与约束符合性

- 改动总量远低于 02 §3 的 ≤200 行预算（净增 ~70 行代码 + 1 个 ~60 行的拷贝文件）。
- **三臂互斥**：`--temporal_fold_mode` 单选；avgpool 臂在 VQGAN.__init__ 强制 defer，learned 臂才创建 Fold/Unfold，single 臂原样——三者不叠加。
- **锚帧隔离**：learned 臂的 Fold/Unfold 内部只作用 `tokens[:,1:]`（位 0 锚帧直通），与原 `:910-914` pool 隔离语义一致；single/avgpool 臂继承原隔离逻辑未动。
- **未碰主仓**：所有改动限于 `phase-b-omnitokenizer/` 仓内，`UVT/uvt` 未触动。
- **本机不验证运行**（无 docker / 无 PL 1.5.4 环境）：仅做 `python -m py_compile` 语法验证（见交付报告）。
