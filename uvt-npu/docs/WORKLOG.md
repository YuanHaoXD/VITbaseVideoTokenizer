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
3. ✅ 提交上一轮 #15 验证成果(2 个 commit:`fc158bc` 路径可移植化+gitignore models/;`21fd4ff` 验证文档+WORKLOG)。工作树干净,仓在持久 NFS,提交可跨重启。
4. ✅ 恢复环境依赖:清华源装 `lpips/mergedeep/pytorch-msssim/wandb`,`huggingface_hub` 降回 0.36.2。全导入栈复验绿(torch_npu 2.6.0.post5 / transformers 4.53.1 / 8×NPU 可用)。一键恢复命令:`$PYBIN -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple lpips mergedeep pytorch-msssim wandb && $PYBIN -m pip install "huggingface-hub>=0.30.0,<1.0"`(wandb 是 base_trainer 顶部无条件 import,必装)。
5. ✅ 单卡 tiny 冒烟复验(恢复后环境):Epoch 1 干净通过,loss 2.94→2.08 单调降,各损失有限,稳态 ~5 it/s → 训练入口在恢复后的环境端到端可跑。
6. ✅ **接入 ImageNet parquet + 真实 Stage 1 训练路径验证通过**。方案 B(用户选定):新增 `datasets/parquet_image_dataset.py`(D-2b,双仓同步,`@register`,直读 HF parquet、输出 D-2 冻结契约、复用 ImageTransform),`datasets/__init__.py` 注册,`cfgs/uvt_stage1_imagenet_npu.yaml`(纯图像单源验证配置),`tests/test_parquet_dataset.py`(3 项契约测试,双仓,数据/pyarrow 缺失自动 skip)。单卡真权重(#params=889.2M)短程验证:PSNR 2.64→8.95(500 步单调上行)、loss 3.17→1.475(单调降)、0 报错 → **真实训练路径健康、parquet 接入端到端可用**。详见 `docs/P1-smoke-overfit-analysis.md` §12。测试:95 项收集通过,dataset 邻近 7 passed。

**本会话结论**:进度保存机制(WORKLOG+记忆)已建;上一轮 #15 成果已安全入库;环境已恢复;真实图像 Stage 1 训练路径已在真权重下打通验证(单卡 + 8 卡 HCCL)。3 个提交已 push 到 origin/main;8 卡复验后再补 1 个提交。

7. ✅ **push 到 origin/main**(`6c7e8a0..1e0947e`)——GitHub 多一层备份。
8. ✅ **8 卡 HCCL 真权重复验**(补 NPU_NOTES §6 遗留项①):272 步/epoch(8716/8卡/bs4,分片正确)、PSNR 2.60→8.54、曲线与单卡几乎重合(HCCL 梯度平均正确)、`find_unused_parameters` 正确处理纯图像 batch、228.6s 干净退出 0 报错。详见 `P1-smoke-overfit-analysis.md` §12.1。

**下一步(下个助手接手)**:
- ~~① 8 卡 HCCL 真权重复验~~ ✅ 已完成(§12.1)。
- ② 扩数据(去 `max_shards`/接全量 294 分片,注意全量跨片洗牌抖动 → 见 `parquet_image_dataset.py` docstring 末的方案)+ 加长训练,观察 PSNR 爬到 Gate(≥26)。
- ③ 视频教师上线(`teachers.vid_mock false` + InternVideo2,已在 `yh222/models/`)与真实视频源(decord 后端遗留,NPU_NOTES §6)。

**下一步**:见上 3/4/5。

---

## 2026-07-16（晚·续）· 建过拟合探针实验台 `scripts/overfit_probe.py`(小数据集 sanity gate)

**背景/动机**（用户）:全量跑几天才发现问题时间成本太大 → 先在**小数据集上 overfit**证明模型/损失/梯度/数据管线整条链路无病（那位有名大佬的老配方:overfit one batch first）。要求:小数据集 + 日志存 `uvt-npu/logs/` + wandb 监控 + 每隔若干步 dump 重建图。

**做了什么**:
1. ✅ 新增 `scripts/overfit_probe.py`（可复用实验台，非一次性 probe）:
   - 从 ImageNet parquet 抽 N 张固定真图，minibatch 循环 overfit。
   - **对齐 run-full-03 真实训练目标**（lpips0.5+cos0.5+distill0.5+kl1e-6，真 SigLIP2 教师）——不是简化 loss，故这里能暴露的 == 全量会遇到的。
   - wandb **离线**（本机无外网，`WANDB_MODE=offline`，跑完 `wandb sync` 上传）。
   - 每 `--img_every` 步 dump `[上排GT/下排重建]` 对比 PNG → `logs/<run>/images/`，同时进 wandb。
   - 全落盘 `logs/<run>/`（overfit.log tee + images/ + wandb/）。μ 塌缩自检、best_psnr>30 verdict。
   - `--tiny --cpu` 冒烟自检模式（不占 NPU）。
2. ✅ **CPU tiny 冒烟全绿**:管线端到端通——data/tiny模型(0.8M)/tiny教师/recon+distill 组装/loss 单调降(1.97→1.65)/backward+step/psnr/wandb离线/PNG dump(192×128=[3图×2排] 正确)。**未占用任何 NPU 卡**（`--cpu`，torch_npu AdamW patch 需卡可见但只读设备名，不分配显存）。
3. ⏳ **真跑待定（资源冲突）**:生产 overfit（tiny=false+真SigLIP2，单卡 ~20-25GB）与 full04 每卡 ~20GB 余量顶格 → 同卡跑有 OOM 风险会**连累 full04**（无 ckpt）。真跑需专卡决策（暂停 full04 / 等空卡）——已交用户定。

**下一步**:用户定资源后跑真 overfit（PSNR 应爬到 >30 甚至 >35 证明能记忆）；此台以后作为**每次全量前的前置 gate**。

---

## 2026-07-16（晚）· 修 epoch-末 checkpoint 保存崩溃（裸 cuda 端口遗漏）+ 重启主线 full04

**背景**:run-full-03 跑完 epoch-1（PSNR 爬到 ~19.1，单调无平台，健康），在 epoch 边界 `save_checkpoint('epoch-last.pth')` 处**全 8 卡崩溃**，**0 checkpoint 落盘**（~3.5h epoch 白跑）。用户要求接着工作。

**根因**:`utils/common.py` 的 `gather_object_from_all`（多卡 RNG-state gather）里两处**裸 cuda**是 NPU 端口的遗漏——
- L153 `torch.ByteTensor(...).to('cuda')`
- L182 `torch.tensor(size).cuda()`

`uvt/`（CUDA 源）用 `.cuda()` 是对的；`uvt-npu/` 必须走设备门面。这是"端口漏改"类 bug，与 CLAUDE.md 红线"禁止在 uvt-npu 里裸写 torch.cuda"同源。之所以此前 8 卡冒烟没暴露：save 路径要 `tot_gpus>1` 才进 gather 分支，且 epoch-末才触发。

**做了什么**:
1. ✅ **修 `utils/common.py`**：两处 `.to('cuda')`/`.cuda()` → `.to(accel.device())`，顶部 `from utils import accel`（accel 不 import common，无循环依赖）。**仅 uvt-npu**，uvt/ 不动（故意的设备层分叉）。
2. ✅ **全仓复扫裸 cuda**：余下命中都在**非当前路径**（LARP baseline trainer/sampler、FID/FVD eval 台）；`base_trainer` save 里的 `isinstance(scaler, torch.cuda.amp.GradScaler)` 已被 `accel.GradScaler` 返回 disabled cuda scaler 兼容（端口作者原注释），非 bug。
3. ✅ **真机 2 卡 HCCL 验证**：脚本复现崩溃函数 `gather_object_from_all`（喂与 save_checkpoint 同构的 rng-state dict）→ world=2 gather 正确、keys=[0,1]、rank_marker 对齐。崩溃路径已闭合。
4. ✅ **重启主线 = full04**（= full-03 **同配置** `cfgs/uvt_stage1_imagenet_full_npu.yaml`，identical，遵守 paired-baseline 纪律不改 bs/lr）。无 ckpt 只能从头（SigLIP2 init）。out_path `.../full04`，保留 full03 log/tensorboard 供参考。

**结论/产出**:checkpoint 保存崩溃已修并真机验证；主线 full04 已在 8 卡后台重启（`full04.log`）。**首个真实价值点**：epoch-1 末（~3.5h）能否干净存出 `epoch-last.pth` —— 即本 fix 的验收。

**下一步**:① 盯 full04 epoch-1 边界，确认 `epoch-last.pth` 落盘（fix 验收）；② 落盘后跑 `scripts/eval_metrics.py --ckpt` 出真实 6 指标校准；③ 长跑盯 PSNR 冲 Gate(≥26→29.5-31.5)。

---

## 2026-07-16 · 重建评测搭台（eval_metrics.py + I3D/Inception/DAVIS 下载）

**会话目标**:搭可跑的重建评测 —— ImageNet PSNR/SSIM/rFID + DAVIS PSNR/SSIM/rFVD；只做搭台+下载+CPU sanity,不占训练用的 8 卡。

**做了什么**:
1. **新增 `scripts/eval_metrics.py`**(双仓同步 `uvt/`+`uvt-npu/`,device resolver 在 uvt/ 无 accel 时降级 cuda)。预处理只走 `eval/protocols.py`,指标只用 `eval/recon_metrics.ReconMetricsSuite`。checkpoint 走 `models.make(ckpt['model'], load_sd=True)`,支持 `--ema`。值域链:protocols[-1,1]→`from_eval_range`喂模型[0,1]→x_hat[0,1]→`to_eval_range`回[-1,1]喂指标。
2. **下载权重**(任务假设的 `Daniel0724/OmniTokenizer` 无 i3d,实际源见记忆 `uvt-eval-weight-sources`):
   - `eval/fvd/styleganv/i3d_torchscript.pt` ← HF `flateon/FVD-I3D-torchscript`(主选 rFVD)
   - `eval/fvd/videogpt/i3d_pretrained_400.pt` ← HF `Xiaodong/FVD_I3D`(交叉核对)
   - Inception rFID `pt_inception-2015-12-05-6726825d.pth`(95.6MB)← **ghfast.top 镜像**(github 经代理仅 12KB/s)→ `~/.cache/torch/hub/checkpoints/`
   - DAVIS-2017 val 480p ← ethz 官方 zip(~100-180KB/s,后台下载+自动解压到 `Datasets/DAVIS`)
3. **CPU sanity 全绿**:6 指标全验证 —— ImageNet PSNR/SSIM/LPIPS/rFID(真 parquet)、rFVD 两版(合成 clip,styleganv 22.57/videogpt 22.80);真实 889M 模型单张 CPU 前向(img 16s/vid 27s),`reconstruct()` 形状值域正确(img[1,3,256,256]/vid[1,3,17,256,256]∈[-1,1])。

**结论/产出**:评测台已就绪。5/6 指标+真实模型前向全在 CPU 验证;DAVIS 真实数据 run 待 zip 下完(后台自动解压后补验)。**依赖非持久**:eval 用 `lpips`+`pytorch_msssim`,重启即丢需重装。

**下一步**:① 训练存出 checkpoint 后跑真评测(`--ckpt <run>/epoch-N.pth`);② ImageNet 全量 100k 图单卡估 ~20-40min,用空闲卡或 `--limit`;③ DAVIS 解压后确认 val.txt 30 序列可读。

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
