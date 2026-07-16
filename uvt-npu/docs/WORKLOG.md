# UVT 工作日志（WORKLOG）

> **这是本项目的持久工作记录。每个工作助手（Claude/GLM/…）必须在推进工作时实时更新本文件,不要等收尾才写。**
> 服务器会重启并清空 Claude 对话历史(见 `NPU_NOTES.md` 与记忆 `modelarts-notebook-env`),conda 依赖也非持久 —— 本文件 + git 提交是唯一跨重启的进度载体。
>
> **约定**:最新在最上;每条含【日期】【会话目标】【做了什么】【结论/产出】【下一步】;重要结论同时回填对应 `docs/*` 并 git 提交。

---

## 项目状态快照（截至 2026-07-15）

- **代码**:UVT 主仓(`uvt/`)与 NPU 工作副本(`uvt-npu/`)模型/损失/教师/数据代码一致;`#15 切分边界规范化`修复已落地双仓,契约测试 `test_boundary_norms.py` 4 项绿,双仓全量 ~81 passed。
- **硬件/环境**:8× Ascend 910B2 全部 `OK`,torch 2.6.0 + torch_npu 2.6.0.post5 正常。⚠️ **conda 依赖非持久**:重启后 `lpips/mergedeep/pytorch_msssim` 需重装(清华源)。
- **权重**(仓库内 `models/`,持久 NFS,已 gitignore):SigLIP2-so400m-patch16-256(4.5GB)✅。外部 `yh222/models/`:dinov2-large、InternVideo2-Stage2_1B-224p-f4、JoyAI-VL ✅。
- **数据**(`yh222/Datasets/`):imagenet-1k(HF parquet,train 294 分片 + test 28 分片)、ucf101-subset ✅。⚠️ **parquet 尚未接入训练数据管线**(当前 `datasets/` 走 LARP CSV/JPEG 路径)。
- **验收进度**:P1-smoke Gate 的重建能力已实证通过(见下 2026-07-15 条);**尚未做真实 Stage 1 训练**。Stage 2/3 逻辑同源、DDP 已验,真权重下未跑过。

---

## 2026-07-15 · 会话:进度保存机制 + 恢复上一轮成果 + 推进真实训练

**背景**:服务器重启,上一轮对话丢失,且上一轮 `#15 真权重验证`成果全部**未提交**,差点二次丢失。用户要求:建立持久工作记录 + 勤 git + 接着工作。

**做了什么**（进行中,实时更新）:
1. ✅ 侦察全貌:git 有 6 个未提交改动 + 2 个未跟踪文件;环境体检 8×NPU 健在但 3 个 pip 包丢失;ImageNet 是 parquet 格式未接管线。
2. ✅ 建立本 WORKLOG.md + 记忆 `worklog-and-git-discipline`(流程要求持久化)。
3. ⏳ 提交上一轮 #15 验证成果(先 gitignore 顶层 `models/`)。
4. ⏳ 恢复环境依赖(清华源装 lpips/mergedeep/pytorch_msssim)。
5. ⏳ 接入 ImageNet parquet → 真实 Stage 1 短程验证。

**下一步**:见上 3/4/5。

---

## 2026-07-15（上一会话,已丢对话但成果在盘)· #15 真权重验证 + 环境重建

> 完整诊断见 `docs/P1-smoke-overfit-analysis.md` §10–§11。摘要:

- **`/cache` 清空后重建**:SigLIP2 重下到仓库内 `models/`(4.5GB);脚本/配置里写死的 `/cache/...` 路径改为仓库内路径(`probe_recon.py`/`p1_smoke_overfit.py` 读环境变量 `UVT_SIGLIP` 带默认;`cfgs/uvt_stage1_npu.yaml` 两处绝对路径);依赖用清华源重装;`huggingface_hub` 降回 0.36.2(wandb 会顺带升到 ≥1.0 与 transformers 4.53.1 冲突)。
- **#15 修复真权重验证通过**:三大病理全消 —— μ 跨图余弦 0.997→0.67(真图 0.283)、x_hat 值域 ±44→±2、sample=True 发散/震荡消失(与 sample=0 一致)。
- **重建路径健康硬证据**:纯 L1 + 低频结构图 overfit 500 步 → PSNR **34.89** 单调穿过 30。原 `p1_smoke_overfit` 卡 ~7.9 是 `torch.rand` 噪声目标伪影(不可压 + LPIPS 对抗),非 bug。
- **结论**:重建路径健康,不阻断后续;**下一步价值在真实 Stage 1 训练,而非继续调 overfit**。

---

## 更早（2026-07-10 及之前)

- NPU 移植完成:`utils/accel.py` 设备 facade + 两处 triton shim + `TORCH_DEVICE_BACKEND_AUTOLOAD=0`;单卡/8卡 HCCL tiny 冒烟绿(见 `NPU_NOTES.md`)。
- 主仓实现:M-1~M-10 全部落盘,84 CPU 测试绿(见 `docs/06`、`docs/08`)。
