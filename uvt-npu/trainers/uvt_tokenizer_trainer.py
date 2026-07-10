"""TR-2 · UVT tokenizer 训练器：三阶段状态机（02 篇 §2.2-c，01 篇 §2.9 损失总表）。

阶段语义（stage 由 config 决定，冻结策略由 M-10 的 model.set_stage 执行）：
  Stage 1 基础：L1+LPIPS+KL+L_cos（L-1 recon）+ λ_dist·蒸馏三项（L-2 distill）；全参可训；300k 步。
  Stage 2 精修：recon（L_cos 显式关）+ GAN（L-3，LARP 配方）；仅 decoder 可训（encoder/GSB 冻结）；
               G/D 交替，d_update_freq 以「优化器步」计（梯度累积下窗口整体同开同关）。
  Stage 3 调和：仅蒸馏项；仅 Sem-ViT+池化头可训；入口先跑 model.estimate_latent_stats
               统计 z 通道 mean/std 写入 GSB buffer 并置 normalize=True（ADR-5）。

优化器（01 §2.9）：G 侧 AdamW(0.9,0.95) wd 0.05 峰值 lr 2e-4；D 侧 Adam(0.5,0.9)，
lr = G_lr × dis_lr_multiplier（来自 GANLossConfig，本文件不硬编码）。

模型接口按任务书 M-10 卡冻结签名调用（forward_train/set_stage/estimate_latent_stats）；
契约要求：UVTTokenizer.forward 在训练态须分派到 forward_train（DDP 只 hook forward——LARP 同款约定）。

训练期附件（DistillLoss 对齐头 + GAN 判别器）统一挂在 self.loss（nn.ModuleDict）：
  - 进「训练 checkpoint」的 loss 段（可断点续训），
  - 不进 model 段 → 发布用 model_sd_only checkpoint 天然不含它们（满足"不导出"要求）；
  - 不包 DDP（UVTGANLoss 是多入口模块，DDP 只同步 forward），梯度在窗口末手动 all-reduce。

实验卡：每 run 启动时自动生成 docs/experiments/<run_id>.yaml（04 篇 §4 模板字段）。
"""
import hashlib
import os
import os.path as osp
import subprocess
import time
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.optim import Adam, AdamW

import datasets
from datasets import JointLoader
from losses.distill import DistillLoss
from losses.gan import GANLossConfig, UVTGANLoss
from losses.recon import recon_loss
from trainers import register
from utils import accel  # === NPU modification: 统一加速器抽象（npu/cuda/cpu）===

from .base_trainer import BaseTrainer


@register('uvt_tokenizer_trainer')
class UVTTokenizerTrainer(BaseTrainer):

    def __init__(self, rank, cfg):
        super().__init__(rank, cfg)
        self.stage = int(cfg.get('stage', 1))
        assert self.stage in (1, 2, 3), f'stage 必须为 1/2/3，得到 {self.stage}'
        self.lambda_dist = float(cfg.get('lambda_dist', 0.5))   # 01 §2.8 默认 0.5，P3 扫
        self.clip_grad_max_norm = float(cfg.get('clip_grad_max_norm', 0.0))
        self._g_opt_steps = 0        # G 侧优化器步计数（d_update_freq 以优化器步为单位）
        self.gan = None
        self.distill_loss = None
        self.teacher_img = None
        self.teacher_vid = None

        if self.is_master:
            self._write_experiment_card()

    # ------------------------------------------------------------------ 命名
    @staticmethod
    def get_exp_name(base_exp_name, cfg, args):
        exp_name = f'{base_exp_name}/stage{cfg.get("stage", 1)}_b{args.batch_size}'
        lr = cfg.get('optimizer', {}).get('args', {}).get('lr', 2e-4)
        if float(lr) != 2e-4:
            exp_name += f'_lr{lr}'
        if args.tag:
            exp_name += f'_{args.tag}'
        return exp_name

    # ------------------------------------------------------------------ 实验卡（04 §4）
    def _write_experiment_card(self):
        """docs/experiments/<run_id>.yaml——字段按 04 篇 §4 模板；metrics 由评测回填。"""
        cfg = self.cfg
        repo_root = osp.dirname(osp.dirname(osp.abspath(__file__)))

        def _git(*args):
            try:
                out = subprocess.run(['git'] + list(args), cwd=repo_root,
                                     capture_output=True, text=True, timeout=10)
                return out.stdout.strip() if out.returncode == 0 else ''
            except Exception:
                return ''

        git_sha = _git('rev-parse', 'HEAD')
        if _git('status', '--porcelain'):
            git_sha += '-dirty'   # CI 层面禁止 dirty 启动；此处仅如实记录
        resolved = yaml.dump(_plain(cfg), sort_keys=True)
        card = {
            'run_id': cfg.get('run_id', cfg['env'].get('exp_name', 'unnamed')),
            'git_sha': git_sha,
            'config_hash': hashlib.sha256(resolved.encode()).hexdigest(),
            'data_version': cfg.get('data_version', ''),
            'seeds': {
                'model': cfg.get('manualSeed', -1),
                'data': cfg.get('joint_dataset', {}).get('seed', cfg.get('manualSeed', -1)),
            },
            'adr_versions': list(cfg.get('adr_versions', [])),
            'paired_with': list(cfg.get('paired_with', [])),
            'stage': self.stage,
            'metrics': {},
            'conclusion': '',
        }
        card_dir = osp.join(repo_root, 'docs', 'experiments')
        os.makedirs(card_dir, exist_ok=True)
        card_path = osp.join(card_dir, f"{card['run_id']}.yaml".replace('/', '_'))
        with open(card_path, 'w') as f:
            yaml.dump(card, f, sort_keys=False, allow_unicode=True)
        self.log(f'Experiment card written: {card_path}')

    # ------------------------------------------------------------------ 数据（D-3 接入）
    def make_datasets(self):
        """有 joint_dataset 配置时用 JointLoader（图像/视频多源交替）；否则回落 LARP 原路径。"""
        super().make_datasets()   # 处理 test_dataset（train_dataset 缺省时自动跳过）
        jd = self.cfg.get('joint_dataset')
        if jd is None:
            return
        sources = []
        for s in jd['sources']:
            ds = datasets.make(s['dataset'])
            self.log(f"Joint source: {s['dataset']['name']}, len={len(ds)}, "
                     f"bs={s['batch_size']}, ratio={s['ratio']}")
            sources.append({'dataset': ds, 'batch_size': int(s['batch_size']),
                            'ratio': int(s['ratio']), 'name': s['dataset']['name']})
        trace_path = osp.join(self.cfg['env']['save_dir'], 'sampling_trace.json')
        self.train_loader = JointLoader(
            sources, seed=int(jd.get('seed', 0)),
            num_workers=int(jd.get('num_workers', 0)), trace_path=trace_path)
        # JointLoader 实现了 set_epoch（鸭子类型），挂进 dist_samplers 复用 base.train() 的逐 epoch 调用
        self.dist_samplers.append(self.train_loader)

    # ------------------------------------------------------------------ 模型（阶段冻结）
    def modify_model_before_compile_ddp(self, model):
        # 冻结必须发生在 DDP 构造之前（DDP 只为构造时 requires_grad=True 的参数建 reducer）
        model.set_stage(self.stage)
        return model

    def make_model(self, model_spec=None, load_sd=False):
        super().make_model(model_spec, load_sd)
        if load_sd:
            # 断点续训路径 base 会跳过 modify_model_before_compile_ddp：补冻结并重建 DDP 包装，
            # 避免 reducer 等待冻结参数的梯度（DDP 的 "expected to finish reduction" 错误）。
            self.orig_model.set_stage(self.stage)
            if self.distributed:
                from torch.nn.parallel import DistributedDataParallel
                self.model_ddp = DistributedDataParallel(
                    self.model, device_ids=[accel.current_device()],
                    find_unused_parameters=self.cfg.get('find_unused_parameters', True))

    # ------------------------------------------------------------------ 损失与教师
    def make_loss(self, loss_spec=None, load_sd=False):
        cfg = self.cfg
        modules = {}
        if self.stage in (1, 3):
            dcfg = cfg.get('distill', {})
            self.distill_loss = DistillLoss(
                student_dim=int(dcfg.get('student_dim', 1152)),
                teacher_img_dim=int(dcfg.get('teacher_img_dim', 1152)),
                teacher_vid_dim=int(dcfg.get('teacher_vid_dim', 1152)),
                cfg=cfg).to(self.device)
            modules['distill'] = self.distill_loss
            self._build_teachers()
        if self.stage == 2:
            gan_over = dict(cfg.get('gan', {}))
            if 'dis_adam_betas' in gan_over:
                gan_over['dis_adam_betas'] = tuple(gan_over['dis_adam_betas'])
            self.gan = UVTGANLoss(GANLossConfig(**gan_over)).to(self.device)
            modules['gan'] = self.gan

        self.loss = nn.ModuleDict(modules)   # 训练期附件容器：进 loss 段 checkpoint，不进 model 段
        # base.save_checkpoint 只在 cfg 含 'loss' 键时保存附件 → 确保键存在（可断点续训）
        if 'loss' not in self.cfg:
            self.cfg['loss'] = {'name': 'uvt_attachments', 'args': {}}
        if load_sd and loss_spec is not None and 'sd' in loss_spec:
            self.loss.load_state_dict(loss_spec['sd'])
            self.log('Loaded loss attachments (distill heads / GAN) from checkpoint.')

    def _build_teachers(self):
        """冻结教师（T-1/T-2）。惰性导入：仅蒸馏阶段才碰 transformers 等重依赖。"""
        tcfg = self.cfg.get('teachers', {})
        from teachers.siglip2_teacher import SigLIP2Teacher
        self.teacher_img = SigLIP2Teacher(
            model_id=tcfg.get('img_id', 'google/siglip2-so400m-patch16-256'),
            tiny=bool(tcfg.get('tiny', False))).to(self.device).eval()
        self.teacher_img.requires_grad_(False)
        if bool(tcfg.get('vid_mock', False)):   # 冒烟/单测：无网络 mock 视频教师
            from teachers.internvideo_teacher import MockTeacher
            self.teacher_vid = MockTeacher(**dict(tcfg.get('vid_mock_args', {})))
        else:
            from teachers.internvideo_teacher import (InternVideoTeacher,
                                                      InternVideoTeacherConfig)
            vid_args = dict(tcfg.get('vid', {}))
            vcfg = InternVideoTeacherConfig(**vid_args) if vid_args else None
            self.teacher_vid = InternVideoTeacher(vcfg) if vcfg is not None else InternVideoTeacher()
        self.teacher_vid = self.teacher_vid.to(self.device).eval()
        self.teacher_vid.requires_grad_(False)

    # ------------------------------------------------------------------ 优化器 / scaler
    def configure_optimizers(self, config, load_sd=False):
        # G 侧：模型可训参数（set_stage 已冻结无关部分）+ 蒸馏对齐头（训练期附件）
        g_params = [p for p in self.orig_model.parameters() if p.requires_grad]
        if self.distill_loss is not None:
            g_params += [p for p in self.distill_loss.parameters() if p.requires_grad]
        g_args = dict(config.get('args', {}))
        g_args.setdefault('lr', 2e-4)                     # 01 §2.9：峰值 lr 2e-4
        g_args.setdefault('weight_decay', 0.05)           # 01 §2.9：wd 0.05
        g_args['betas'] = tuple(g_args.get('betas', (0.9, 0.95)))  # AdamW(0.9,0.95)
        g_opt = AdamW(g_params, **g_args)

        if self.stage == 2:
            gcfg = self.gan.cfg
            d_lr = float(g_args['lr']) * float(gcfg.dis_lr_multiplier)
            d_opt = Adam(self.gan.discriminator.parameters(),
                         lr=d_lr, betas=tuple(gcfg.dis_adam_betas))
            # base.apply_lr_multiplier 的双优化器路径读 optimizer.loss_args.lr：同步写回 cfg
            config['loss_args'] = dict(config.get('loss_args', {}))
            config['loss_args']['lr'] = d_lr
            self.cfg['optimizer']['loss_args'] = config['loss_args']
            self.optimizer = [g_opt, d_opt]
            if load_sd:
                sd = config['sd']
                g_opt.load_state_dict(sd[0])
                d_opt.load_state_dict(sd[1])
        else:
            self.optimizer = g_opt
            if load_sd:
                sd = config['sd']
                g_opt.load_state_dict(sd[0] if isinstance(sd, (list, tuple)) else sd)

    def configure_scalers(self, sd=None, load_sd=False):
        if self.stage != 2:
            return super().configure_scalers(sd, load_sd)
        enabled = self.use_amp and self.amp_dtype == torch.float16
        g_scaler = accel.GradScaler(enabled=enabled)
        d_scaler = accel.GradScaler(enabled=enabled)
        if load_sd and enabled:
            assert sd is not None, 'GradScaler state_dict not found in checkpoint'
            g_scaler.load_state_dict(sd[0])
            d_scaler.load_state_dict(sd[1])
        self.scaler = [g_scaler, d_scaler]

    # ------------------------------------------------------------------ Stage-3 入口
    def train(self):
        if self.starting_epoch > 1:
            # 断点续训：优化器步计数按「每 epoch ceil(n/k) 步」重建（近似值，仅影响
            # d_update_freq/disc_start 的相位，不影响梯度正确性）
            k = max(1, self.grad_accumulates)
            n = len(self.train_loader)
            self._g_opt_steps = (self.starting_epoch - 1) * ((n + k - 1) // k)
        if self.stage == 3 and self.starting_epoch == 1:
            # ADR-5：先在 10k 样本上统计 z 通道 mean/std → 写 GSB buffer 并置 normalize=True。
            # 断点续训（starting_epoch>1）跳过：stats 已随 checkpoint 的 buffer 恢复。
            n = int(self.cfg.get('latent_stats_n', 10000))
            self.log(f'Stage 3 entry: estimate_latent_stats on {n} samples (ADR-5)')
            self.orig_model.estimate_latent_stats(self.train_loader, n=n)
            self._sync_stats_to_ema()
        super().train()

    @torch.no_grad()
    def _sync_stats_to_ema(self):
        """EMA 副本在统计前深拷贝，buffer（z_mean/z_std）与 normalize 开关需手动同步。"""
        src_buffers = dict(self.orig_model.named_buffers())
        for ema_model in self.ema_model_dict.values():
            for name, buf in ema_model.named_buffers():
                if name in src_buffers and buf.shape == src_buffers[name].shape:
                    buf.copy_(src_buffers[name])
            # 架构方注（v1.2 校对）：GSB.normalize 自 v1.1 起已由持久化 buffer `_normalize_flag` 承载，
            # 上面的 named_buffers 拷贝已覆盖它——下面的 property 写入是冗余保险（无害，保留），
            # 原注释"不在 state_dict 里"已过时。本方法整体仍必要：LARP 的 EMA 更新只触碰参数不触碰 buffer。
            for m in ema_model.modules():
                if hasattr(m, 'z_mean') and hasattr(m, 'normalize'):
                    m.normalize = True

    # ------------------------------------------------------------------ 单步
    def _recon_cfg(self):
        """recon_loss 的 cfg 视图：Stage-2 契约要求显式关 L_cos（L-1 缺键即炸的防线）。"""
        cfg = deepcopy(dict(self.cfg.get('loss_weights', {})))
        base = self.cfg
        view = _AttrView(cfg, base)
        if self.stage == 2:
            view.use_cos_consistency = False
        return view

    def _teacher_targets(self, x, is_video):
        """冻结教师前向（no_grad）。x:[B,3,T,H,W]∈[0,1]；返回 (t_patch, t_pool, t_vid)。"""
        with torch.no_grad():
            anchor = x[:, :, 0]                             # 锚帧 [B,3,H,W]
            t_patch, t_pool = self.teacher_img(anchor)
            t_vid = None
            if x.shape[2] > 1 and bool(is_video.any()):
                t_vid = self.teacher_vid(x[:, :, 1:])       # 非锚 16 帧 → [B,16,N_t,D_t]
        return t_patch, t_pool, t_vid

    def _iter_step(self, data, is_train):
        start = time.time()
        x = data['gt'].to(self.device, non_blocking=True)   # 图像源亦带 'gt' 别名（D-2）
        B = x.shape[0]
        if x.dim() == 4:
            x = x.unsqueeze(2)                              # [B,3,H,W] → [B,3,1,H,W]（§0 约定）
        if 'is_video' in data:
            is_video = data['is_video'].to(self.device).reshape(B)
        else:
            is_video = x.new_full((B,), x.shape[2] > 1, dtype=torch.bool)

        info = {}
        # d_update_freq 以「优化器步」为单位：梯度累积窗口内同开同关
        opt_index = self._g_opt_steps

        with accel.autocast(dtype=self.amp_dtype, enabled=self.use_amp):
            out = self.model_ddp(x)   # 契约：训练态 forward → forward_train（M-10）
            assert isinstance(out, dict) and 'x_hat' in out, \
                'M-10 契约：训练态 forward 须返回 forward_train 的 dict（含 x_hat）'
            x_hat = out['x_hat']

        # ---------- D 步（仅 Stage 2 训练态；freq 命中时先于 G 更新，LARP 交替次序） ----------
        # 评测态跳过：discriminator_loss 会更新 LeCam EMA buffer，eval 不应有副作用
        if self.stage == 2 and is_train and self.gan.should_update_d(opt_index):
            self.gan.discriminator.requires_grad_(True)
            with accel.autocast(dtype=self.amp_dtype, enabled=self.use_amp):
                d_loss, d_info = self.gan.discriminator_loss(x, x_hat.detach(), opt_index)
            info.update(d_info)
            if is_train and d_loss.requires_grad:
                # 判别器不在任何 DDP 包装内 → 无 no_sync 语义，只需窗口末手动 all-reduce
                self.scaler[1].scale(self.scale_loss_for_grad_accum(d_loss)).backward()
                if self.should_optim_step:
                    if self.clip_grad_max_norm > 0.0:
                        self.scaler[1].unscale_(self.optimizer[1])
                        torch.nn.utils.clip_grad_norm_(
                            self.gan.discriminator.parameters(), self.clip_grad_max_norm)
                    self._allreduce_grads(self.gan.discriminator.parameters())
                    self.scaler[1].step(self.optimizer[1])
                    self.scaler[1].update()
                    self.optimizer[1].zero_grad(set_to_none=True)

        # ---------- G 侧损失组装（阶段开关，01 §2.9） ----------
        with accel.autocast(dtype=self.amp_dtype, enabled=self.use_amp):
            loss = x.new_zeros(())
            if self.stage in (1, 2):
                rec = recon_loss(x, out, self._recon_cfg())
                loss = loss + rec['total']
                for k in ('l1', 'lpips', 'kl', 'cos_consistency'):
                    info[f'loss_{k}'] = float(rec[k].item())
            if self.stage in (1, 3):
                t_patch, t_pool, t_vid = self._teacher_targets(x, is_video)
                d_out = self.distill_loss(
                    out['s'], out['s_pool'], out.get('decomp_out'),
                    t_patch, t_pool, t_vid, is_video)
                loss = loss + self.lambda_dist * d_out['total']
                for k in ('img_patch', 'img_pool', 'vid'):
                    info[f'distill_{k}'] = float(d_out[k].item())
            if self.stage == 2:
                # G 对抗项：冻结 D（前向图不再进 D 参数），梯度只回 decoder
                self.gan.discriminator.requires_grad_(False)
                g_adv, g_info = self.gan.generator_loss(x_hat, opt_index)
                loss = loss + g_adv
                info.update(g_info)

            with torch.no_grad():
                mse = ((x_hat.float() - x.float()) ** 2).reshape(B, -1).mean(dim=-1)
                info['psnr'] = float((-10 * torch.log10(mse.clamp_min(1e-10))).mean().item())
            info['loss'] = float(loss.item())

        # ---------- G 反传（梯度累积 + DDP no_sync，TR-1b 基座接口） ----------
        if is_train:
            g_scaler = self.scaler[0] if isinstance(self.scaler, list) else self.scaler
            g_opt = self.optimizer[0] if isinstance(self.optimizer, list) else self.optimizer
            with self.grad_accum_ctx(self.model_ddp):
                g_scaler.scale(self.scale_loss_for_grad_accum(loss)).backward()
            if self.should_optim_step:
                if self.clip_grad_max_norm > 0.0:
                    g_scaler.unscale_(g_opt)
                    torch.nn.utils.clip_grad_norm_(
                        [p for group in g_opt.param_groups for p in group['params']],
                        self.clip_grad_max_norm)
                if self.distill_loss is not None:
                    # 蒸馏对齐头在 model 的 DDP 之外，窗口末手动 all-reduce
                    self._allreduce_grads(self.distill_loss.parameters())
                g_scaler.step(g_opt)
                g_scaler.update()
                g_opt.zero_grad(set_to_none=True)
                self._g_opt_steps += 1
                for ema_decay, ema_model in self.ema_model_dict.items():
                    self.update_ema(ema_model, decay=ema_decay)

        elif x_hat.shape[2] >= 10:   # 评测期收集 FVD 统计（LARP 同款）
            self.fake_stats = self.fvd_calculator.get_feature_stats_for_batch(
                x_hat.float().clamp(0., 1.), self.fake_stats)
            self.running_real_stats = self.fvd_calculator.get_feature_stats_for_batch(
                x, self.running_real_stats)

        info['fps'] = B / (time.time() - start)
        return info

    def _allreduce_grads(self, params):
        """DDP 外参数的梯度同步（等价 DDP 的 mean all-reduce；窗口末调用一次）。"""
        if not self.distributed:
            return
        for p in params:
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad.div_(self.tot_gpus)

    def train_step(self, data):
        return self._iter_step(data, is_train=True)

    def evaluate_step(self, data):
        with torch.no_grad():
            return self._iter_step(data, is_train=False)


# ---------------------------------------------------------------------- 工具
def _plain(obj):
    """edict/嵌套结构 → 纯 python 类型（yaml 稳定序列化用，剔除不可序列化项）。"""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items() if k != 'sd'}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


class _AttrView:
    """recon_loss 的 getattr 型 cfg 视图：优先 loss_weights 子表 → 顶层 cfg → 调用方 setattr 覆盖。"""

    def __init__(self, primary: dict, fallback):
        self._primary = primary
        self._fallback = fallback

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        if name in self._primary:
            return self._primary[name]
        try:
            return self._fallback[name]     # edict：键访问
        except Exception:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if name.startswith('_'):
            object.__setattr__(self, name, value)
        else:
            self._primary[name] = value
