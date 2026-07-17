"""
    A basic trainer.

    The general procedure in run() is:
        make_datasets()
            create . train_loader, test_loader, dist_samplers
        make_model()
            create . model_ddp, model
        train()
            create . optimizer, epoch, log_buffer
            for epoch = 1 ... max_epoch:
                adjust_learning_rate()
                train_epoch()
                    train_step()
                evaluate_epoch()
                    evaluate_step()
                visualize_epoch()
                save_checkpoint()
"""

import contextlib  # === UVT modification (TR-1b): 梯度累积的 no_sync 需要空上下文兜底 ===
import json
import os
import os.path as osp
import random
import shutil
import time
from collections import OrderedDict
from copy import deepcopy
from math import cos, pi

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed
import torch.distributed as dist
import torch.nn as nn
import wandb
import yaml
from filelock import FileLock
from pandas import DataFrame
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import datasets
import models
import utils
from trainers import register
from utils import FeatureStats, FIDCalculator, FVDCalculator
from utils import accel  # === NPU modification: 统一加速器抽象（npu/cuda/cpu），替代散落的 torch.cuda.* ===



def print_grad(grad, name=None):
    if torch.distributed.get_rank() == 0:
        print(f"{name=}Grad size: {grad.size()}, strides: {grad.stride()}")


def get_orig_module(module):
    if hasattr(module, 'module'):
        module = module.module
    if hasattr(module, '_orig_mod'):
        module = module._orig_mod
    return module


def map_location_fn(storage, loc):
    # NPU: 把 checkpoint tensor 落到当前加速器（npu/cuda/cpu），原 torch.cuda 写法已收口到 accel。
    return accel.map_location_fn(storage, loc)

@register('base_trainer')
class BaseTrainer():
    def __init__(self, rank, cfg):
        self.rank = rank
        self.cfg = cfg
        self.is_master = (rank == 0)

        env = cfg['env']
        self.tot_gpus = env['tot_gpus']
        self.distributed = (env['tot_gpus'] > 1)
        self.use_amp = cfg['use_amp']
        if self.use_amp:
            if 'amp_dtype' in cfg:
                if cfg['amp_dtype'] == 'float16':
                    self.amp_dtype = torch.float16
                elif cfg['amp_dtype'] == 'bfloat16':
                    self.amp_dtype = torch.bfloat16
                else:
                    raise ValueError(f'Unknown AMP dtype: {cfg["amp_dtype"]}')
            else:
                self.amp_dtype = torch.bfloat16
        else:
            self.amp_dtype = torch.float32

        self.compile = cfg['compile']
        self.compile_mode = cfg['compile_mode']
        self.stepwise_logging = cfg['stepwise_logging']
        self.n_steps_per_epoch = -1
        self.lr_mult_epochwise = 0.0
        self.init_checkpoint = cfg.get('init_checkpoint', None)
        self.save_best = cfg.get('save_best', False)
        self.ema_decay_list = [float(x) for x in cfg['ema_decay'].split('_') if x != '']
        self.epoch = None

        # === UVT modification (TR-1b): 梯度累积（02 篇 §2.2-b）===
        # grad_accumulates: 每 k 个 micro-step 才执行一次 optimizer.step()，默认 1=关闭。
        # 理由：Stage 1 全局 batch 256，8 卡下每卡 32（视频 17 帧）会 OOM，需累积。
        # 窗口状态（micro_step_in_window / should_optim_step）由 train_epoch 维护，
        # 子类 train_step 通过 grad_accum_ctx()/scale_loss_for_grad_accum() 使用（G/D 双优化器同构）。
        self.grad_accumulates = int(cfg.get('grad_accumulates', 1))
        assert self.grad_accumulates >= 1, f'grad_accumulates 必须 >= 1，得到 {self.grad_accumulates}'
        self.micro_step_in_window = 0     # 当前累积窗口内的 micro-step 序号（0-based）
        self.should_optim_step = True     # 当前 micro-step 是否应执行 optimizer.step()
        # === UVT modification end ===

        # Setup distributed devices
        # === UVT modification (TR-1a): torchrun 下用 LOCAL_RANK 绑卡 ===
        # rank 是全局 rank（多机下可能 >= 单机卡数），仅用于 is_master/日志/采样分片；
        # 物理选卡必须用 LOCAL_RANK。无该环境变量（mp 单机遗留/单卡调试）时退回 rank。
        local_rank = int(os.environ.get('LOCAL_RANK', rank))
        # === NPU modification: 统一用 accel 绑卡 + 选设备（npu/cuda/cpu），替代 torch.cuda ===
        if accel.is_available():
            accel.set_device(local_rank)
        self.device = accel.device(accel.current_device() if accel.is_available() else 0)
        # === NPU modification end ===
        self.enable_wandb = False

        if self.is_master:
            logger, writer, self.time_str = utils.set_save_dir(env['save_dir'], replace=False)
            with open(osp.join(env['save_dir'], 'cfg.yaml'), 'w') as f:
                yaml.dump(cfg, f, sort_keys=False)

            self.log = logger.info
            self.train_psnr, self.val_psnr, self.val_ssim = [], {}, {}
            self.train_loss, self.val_loss = [], {}

            self.enable_tb = True
            self.writer = writer

        else:
            self.log = lambda *args, **kwargs: None
            self.enable_tb = False

        if self.distributed:
            # === UVT modification (TR-1a): torchrun 入口已在 train.py 完成 init_process_group ===
            # 保留 tcp 初始化路径仅作 mp.spawn 遗留兼容（进程组未初始化时才走）。
            if not dist.is_initialized():
                dist_url = f"tcp://localhost:{env['port']}"
                dist.init_process_group(backend=accel.DIST_BACKEND, init_method=dist_url,
                                        world_size=self.tot_gpus, rank=self.rank)
            # === UVT modification end ===
            self.log(f'Distributed training enabled.')

        cudnn.benchmark = env['cudnn']

        # FVDCalculator 需要 I3D 权重（utils/fvd/i3d_torchscript.pt，gitignore 需下载）。
        # 权重缺失时不挡训练启动——置 None，eval 阶段的 FVD 计算本就在 try/except 内降级为 99999.99。
        try:
            self.fvd_calculator = FVDCalculator(device=self.device)
        except Exception as e:
            self.log(f'FVDCalculator init failed (FVD eval disabled): {e}')
            self.fvd_calculator = None
        # self.fid_calculator = FIDCalculator(device=self.device)
        self.fid_calculator = None

        if cfg.get('fvd_real_stats_path') is not None:
            if cfg['fvd_real_stats_path'].lower() in ['none', 'null', 'no', '']:
                self.fvd_real_stats = None
            else:
                assert os.path.exists(cfg['fvd_real_stats_path']), f"Real stats file not found: {cfg['fvd_real_stats_path']}"
                self.fvd_real_stats = FeatureStats.load(cfg['fvd_real_stats_path'])
        else:
            self.fvd_real_stats = None

        if cfg.get('fid_real_stats_path') is not None:
            if cfg['fid_real_stats_path'].lower() in ['none', 'null', 'no', '']:
                self.fid_real_stats = None
            else:
                assert os.path.exists(cfg['fid_real_stats_path']), f"Real stats file not found: {cfg['fid_real_stats_path']}"
                self.fid_real_stats = FeatureStats.load(cfg['fid_real_stats_path'])
        else:
            self.fid_real_stats = None

        
        self.log(f'Environment setup done.')


    @staticmethod
    def get_exp_name(base_exp_name, cfg, args):
        raise NotImplementedError

    def enable_wandb_if_needed(self, wandb_run_id=None):
        cfg = self.cfg
        env = cfg['env']
        # enable wandb
        if self.is_master and env['wandb_upload']:
            self.enable_wandb = True
            if os.path.exists('wandb.yaml'):
                with open('wandb.yaml', 'r') as f:
                    wandb_cfg = yaml.load(f, Loader=yaml.FullLoader)
            else:
                wandb_cfg = {
                    'project': env['wandb_project'],
                    'entity': env['wandb_entity'],
                }

            slurm_array_job_id = os.environ.get('SLURM_ARRAY_JOB_ID')
            slurm_array_task_id = os.environ.get('SLURM_ARRAY_TASK_ID')
            slurm_array_job_id = slurm_array_job_id if slurm_array_job_id is not None else ''
            slurm_array_task_id = slurm_array_task_id if slurm_array_task_id is not None else ''
            slurm_id = f"{slurm_array_job_id}_{slurm_array_task_id}"

            os.environ['WANDB_DIR'] = env['save_dir']
            os.environ['WANDB_NAME'] = env['exp_name'] + f'__{slurm_id}'
            if 'api_key' in wandb_cfg:
                os.environ['WANDB_API_KEY'] = wandb_cfg['api_key']

            if wandb_run_id is not None:
                init_kwargs = {'id': wandb_run_id, 'resume': 'must'}
            else:
                init_kwargs = {'resume': 'allow'}

            if not utils.check_website_access_bool('https://wandb.ai'):
                self.log('Wandb website not accessible. Running in offline mode.')
                init_kwargs['mode'] = 'offline'
                this_run_metadata = {
                    'project': wandb_cfg['project'],
                    'entity': wandb_cfg['entity'],
                    'dir': os.path.abspath(os.path.join(env['save_dir'], 'wandb'))
                }
                run_metadata_json_path = './wandb_run_metadata.json'
                # get slurm array task id as int
                slurm_array_task_id = os.environ.get('SLURM_ARRAY_TASK_ID')
                slurm_array_task_id = int(slurm_array_task_id) if slurm_array_task_id is not None else None
                if slurm_array_task_id is not None:
                    assert slurm_array_task_id >= 1, 'SLURM_ARRAY_TASK_ID should be >= 1'
                    time.sleep((slurm_array_task_id - 1) * 5)

                with FileLock(run_metadata_json_path + '.lock'):
                    if os.path.exists(run_metadata_json_path):
                        with open(run_metadata_json_path, 'r') as f:
                            run_metadata_all = json.load(f)
                    else:
                        run_metadata_all = []

                    run_metadata_all.append(this_run_metadata)
                    with open(run_metadata_json_path, 'w') as f:
                        json.dump(run_metadata_all, f, indent=4)

            wandb.init(
                project=wandb_cfg['project'], 
                entity=wandb_cfg['entity'],
                config=cfg,
                **init_kwargs
            )

    def run(self):
        self.make_datasets()
        self.starting_epoch = 1
        self.global_step = 0
        wandb_run_id = None
        resume_ckt_path = os.path.join(self.cfg['env']['save_dir'], 'epoch-last.pth') 
        if 'ckt' in self.cfg:
            resume_ckt_path = self.cfg['ckt']

        if os.path.exists(resume_ckt_path):
            # resume training from a checkpoint
            self.log(f'Resuming training from {resume_ckt_path}')
            # weights_only=False:torch 2.6 默认 True 会挡 checkpoint 里的 EasyDict(config)→UnpicklingError。
            # 我们自己存的可信 checkpoint,显式关掉安全限制(续跑 load 路径,torch2.6 潜伏 bug)。
            latest_ckt = torch.load(resume_ckt_path, map_location=map_location_fn, weights_only=False)
            if 'wandb_run_id' in latest_ckt:
                wandb_run_id = latest_ckt['wandb_run_id']
            self.enable_wandb_if_needed(wandb_run_id)
            self.make_model(latest_ckt['model'], load_sd=True)

            if 'loss' in latest_ckt:
                self.make_loss(latest_ckt['loss'], load_sd=True)

            self.configure_optimizers(latest_ckt['optimizer'], load_sd=True)
            self.configure_scalers(latest_ckt.get('scaler_sd', None), load_sd=True)

            self.starting_epoch = latest_ckt['epoch'] + 1
            if self.enable_wandb:
                wandb.run._step = self.starting_epoch

            if 'rng_states_per_rank' in latest_ckt and self.rank in latest_ckt['rng_states_per_rank']:
                local_rng_state_dict = latest_ckt['rng_states_per_rank'][self.rank]
                torch.set_rng_state(local_rng_state_dict['torch_rng_state'])
                if accel.is_available() and local_rng_state_dict['torch_cuda_rng_state'] is not None:
                    accel.set_rng_state(local_rng_state_dict['torch_cuda_rng_state'])
                np.random.set_state(local_rng_state_dict['numpy_rng_state'])
                random.setstate(local_rng_state_dict['python_rng_state'])

            del latest_ckt

        else:
            # start a new training, optionally initialize parameters from a checkpoint init_checkpoint
            self.enable_wandb_if_needed()
            if self.init_checkpoint is not None and os.path.exists(self.init_checkpoint):
                self.log(f'Initializing model from {self.init_checkpoint}')
                init_ckt = torch.load(self.init_checkpoint, weights_only=False)  # torch2.6:同上,可信 checkpoint
                model_spec = deepcopy(self.cfg['model'])
                model_spec['sd'] = init_ckt['model']['sd']
                self.init_checkpoint = ''
                self.cfg['init_checkpoint'] = ''

                self.make_model(model_spec, load_sd=True)
                self.make_loss()
                del init_ckt
            else:
                self.make_model()
                self.make_loss()

            self.configure_optimizers(self.cfg['optimizer'], load_sd=False)
            self.configure_scalers(load_sd=False)

        if accel.is_available():
            accel.empty_cache()
        self.train()

        if self.enable_tb:
            self.writer.close()
        if self.enable_wandb:
            wandb.finish()

    def make_datasets(self):
        """
            By default, train dataset performs shuffle and drop_last.
            Distributed sampler will extend the dataset with a prefix to make the length divisible by tot_gpus, samplers should be stored in .dist_samplers.

            Cfg example:

            train/test_dataset:
                name:
                args:
                loader: {batch_size: , num_workers: }
        """
        cfg = self.cfg
        self.dist_samplers = []

        def make_distributed_loader(dataset, batch_size, num_workers, shuffle=False, drop_last=False):
            sampler = DistributedSampler(dataset, shuffle=shuffle) if self.distributed else None
            loader = DataLoader(
                dataset,
                batch_size // self.tot_gpus,
                drop_last=drop_last,
                sampler=sampler,
                shuffle=(shuffle and (sampler is None)),
                num_workers=num_workers // self.tot_gpus,
                persistent_workers = True if num_workers > 0 else False,
                pin_memory=True,
                pin_memory_device=accel.pin_memory_device())
            return loader, sampler

        # NPU/UVT 修正：有 joint_dataset 时（UVT 路径），train_dataset 仅是 LARP make_cfg 预处理
        # 所需的占位节，真正训练 loader 由子类用 JointLoader 重建。此处不再构建它——否则
        # batch_size//tot_gpus 在小 batch（bs<world_size）下得 0 而崩，且纯属冗余。
        if cfg.get('train_dataset') is not None and cfg.get('joint_dataset') is None:
            train_dataset = datasets.make(cfg['train_dataset'])
            self.log(f'Train dataset: len={len(train_dataset)}')
            self.cfg.update({'TrainSize': len(train_dataset),})
            l = cfg['train_dataset']['loader']
            self.train_loader, train_sampler = make_distributed_loader(
                train_dataset, l['batch_size'], l['num_workers'], shuffle=True, drop_last=True)
            self.dist_samplers.append(train_sampler)

        if cfg.get('test_dataset') is not None:
            l = cfg['test_dataset']['loader']
            self.test_loader_dict = {}
            for dataset_name, dataset_csv in cfg['test_dataset']['csv_paths'].items():
                if dataset_csv:
                    test_dataset = datasets.make(cfg['test_dataset'], args={'csv_file': dataset_csv})
                    self.log(f'Test dataset: {dataset_name}, len={len(test_dataset)}')
                    self.cfg.update({f'TestSize_{dataset_name}': len(test_dataset),})
                    test_loader, test_sampler = make_distributed_loader(
                        test_dataset, l['batch_size'], l['num_workers'], shuffle=False, drop_last=True)
                    self.test_loader_dict.update({dataset_name: test_loader})
                    self.dist_samplers.append(test_sampler)

    def update_model_spec(self, model_spec):
        """
            Update model_spec with cfg.
        """
        return model_spec

    def make_model(self, model_spec=None, load_sd=False):
        if model_spec is None:
            model_spec = self.cfg['model']

        model_spec = self.update_model_spec(model_spec)
        model = models.make(model_spec, load_sd=load_sd).to(self.device)
        self.log(model)
        model_size = utils.compute_num_params(model, text=False)
        model_size_str = utils.text2str(model_size)

        self.log(f'Model: #params={model_size_str}')
        self.log(f'SLURM_JOB_ID: {os.environ.get("SLURM_JOB_ID")}')
        self.log(f'SLUMR_ARRAY_JOB_ID: {os.environ.get("SLURM_ARRAY_JOB_ID")}')
        self.log(f'SLURM_ARRAY_TASK_ID: {os.environ.get("SLURM_ARRAY_TASK_ID")}')
        self.log(f'wandb_name: {os.environ.get("WANDB_NAME")}')

        if self.enable_wandb:
            wandb.run.summary['Model #params'] = model_size_str
            wandb.run.summary['SLURM_JOB_ID'] = os.environ.get('SLURM_JOB_ID')
            wandb.run.summary['SLURM_ARRAY_JOB_ID'] = os.environ.get('SLURM_ARRAY_JOB_ID')
            wandb.run.summary['SLURM_ARRAY_TASK_ID'] = os.environ.get('SLURM_ARRAY_TASK_ID')

        if not load_sd:
            model = self.modify_model_before_compile_ddp(model)

        if self.distributed:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        if self.compile:
            self.log(f'Compiling model with mode: {self.compile_mode}')
            model_compiled = torch.compile(model, mode=self.compile_mode)
        else:
            model_compiled = model

        if self.distributed:
            # === UVT modification (TR-1a): 多机下 device_ids 必须是本地设备而非全局 rank ===
            # === NPU modification: device_ids 取当前加速器（npu/cuda）===
            # find_unused_parameters：JointLoader 会产出纯图像 batch（F=1），按契约⑥ Decompressor
            # 该步不参与（decomp_out=None），故模型含「条件性未用参数」。开 True 让 DDP 正确处理，
            # 否则 "Expected to have finished reduction" 报错。cfg 可关（纯视频 batch 时省一点通信）。
            model_ddp = DistributedDataParallel(
                model_compiled, device_ids=[accel.current_device()],
                find_unused_parameters=self.cfg.get('find_unused_parameters', True))
            # === UVT modification end ===
        else:
            model_ddp = model_compiled

        self.model = model_compiled
        self.model_ddp = model_ddp

        # EMA models
        self.ema_model_dict = {}
        for ema_decay in self.ema_decay_list:
            ema_model = deepcopy(get_orig_module(self.model))
            for p in ema_model.parameters():
                p.requires_grad_(False)
            self.update_ema(ema_model, decay=0)
            ema_model.eval()

            if load_sd:
                if 'ema_sd' in model_spec and ema_decay in model_spec['ema_sd']:
                    ema_model.load_state_dict(model_spec['ema_sd'][ema_decay])
                    self.log(f'Loaded EMA model state dict with decay {ema_decay} from checkpoint.')
                else:
                    self.log(f'No EMA model state dict found with decay {ema_decay} in checkpoint.')
                    raise ValueError('No EMA model state dict found with decay {ema_decay} in checkpoint.')
            self.ema_model_dict[ema_decay] = ema_model

    def modify_model_before_compile_ddp(self, model):
        return model

    @property
    def orig_model(self):
        if not hasattr(self, 'model'):
            raise ValueError('Model not made yet.')
        return get_orig_module(self.model)


    def make_loss(self, loss_spec=None, load_sd=False):
        return

    def configure_optimizers(self, config, load_sd=False):
        self.optimizer = utils.make_optimizer(self.model_ddp.parameters(), config, load_sd=load_sd)

    def configure_scalers(self, sd=None, load_sd=False):
        scaler = accel.GradScaler(enabled=self.use_amp and accel.is_available() and self.amp_dtype == torch.float16)
        # if the amp_dtype is bfloat16, the scaler should be disabled
        if self.amp_dtype == torch.bfloat16:
            assert not scaler.is_enabled(), 'GradScaler should be disabled when using bfloat16'

        if load_sd and self.use_amp and self.amp_dtype == torch.float16:
            assert sd is not None, 'GradScaler state_dict not found in checkpoint'
            scaler.load_state_dict(sd)
        self.scaler = scaler

    def dump_csv(self, cfg):
        if self.is_master:
            def dump_cfg(cfg, kv_dict={}, prefix=''):
                for k,v in cfg.items():
                    if k != 'sd':
                        if isinstance(v, dict):
                            kv_dict = dump_cfg(v, kv_dict, f'{prefix}{k}_')
                        else:
                            kv_dict.update({f'{prefix}{k}': v})
                return kv_dict
            csv_dict = {}
            csv_dict = dump_cfg(cfg, csv_dict)

            def psnr_str(psnr_list, precision=2):
                return '_'.join([str(round(x,precision)) for x in psnr_list])

            def best_psnr(psnr_list, precision=2):
                return round(max(psnr_list), precision) if len(psnr_list) else 0

            if self.train_psnr:
                csv_dict.update({'train_psnr_list': psnr_str(self.train_psnr),
                    'train_psnr': best_psnr(self.train_psnr), })
            for v_dict, v_name in zip([self.val_psnr, self.val_ssim], ['psnr', 'ssim']):
                for dataset_name, val_v in v_dict.items():
                    v_str = psnr_str(val_v, 2 if 'psnr' in v_name else 4)
                    best_v_str = best_psnr(val_v, 2 if 'psnr' in v_name else 4)
                    csv_dict.update({f'{dataset_name}_val_{v_name}_list': v_str, 
                        f'{dataset_name}_val_{v_name}': best_v_str,})

            csv_path = os.path.join(cfg['env']['save_dir'], f'results_{self.time_str}.csv')
            csv_df = DataFrame.from_dict(csv_dict, orient='index').T
            csv_df.to_csv(csv_path)

    def train(self):
        """
            For epochs perform training, evaluation, and visualization.
            Note that ave_scalars update ignores the actual current batch_size.
        """
        cfg = self.cfg
        max_epoch = cfg['max_epoch']
        eval_epoch = cfg.get('eval_epoch', max_epoch + 1)
        vis_epoch = cfg.get('vis_epoch', max_epoch + 1)
        save_epoch = cfg.get('save_epoch', max_epoch + 1)
        latest_interval = cfg.get('latest_interval', 1)
        epoch_timer = utils.EpochTimer(max_epoch)
        self.n_steps_per_epoch = len(self.train_loader)
        self.max_steps = self.n_steps_per_epoch * max_epoch
        self.current_fvd = 99999.99
        self.current_fid = 99999.99

        for epoch in range(self.starting_epoch, max_epoch + 1):
            self.epoch = epoch
            self.global_step = (epoch - 1) * self.n_steps_per_epoch
            self.log_buffer = [f'Epoch {epoch}']

            if self.distributed:
                for sampler in self.dist_samplers:
                    sampler.set_epoch(epoch)

            self.t_data, self.t_model = 0, 0
            self.log(f'Epoch {epoch} started.')
            st = time.time()
            self.train_epoch()
            self.log(f'Epoch {epoch} training done. Time: {time.time()-st:.2f}s')

            if epoch % eval_epoch == 0:
                self.evaluate_epoch()

            if epoch % vis_epoch == 0:
                self.visualize_epoch()

            if epoch % save_epoch == 0:
                self.save_checkpoint(f'epoch-{epoch}.pth')

            if epoch % latest_interval == 0:
                # log time of this saving checkpoint
                st = time.time()
                self.save_checkpoint('epoch-last.pth', save_best=self.save_best)
                self.log_buffer.append(f'\nLatest checkpoint saved. Time: {time.time()-st:.2f}s\n')

            epoch_time, tot_time, est_time = epoch_timer.epoch_done()
            t_data_ratio = self.t_data / (self.t_data + self.t_model + 1e-6)
            self.log_buffer.append(f'{epoch_time} (d {t_data_ratio:.2f}) {tot_time}/{est_time}')
            self.log(', '.join(self.log_buffer))

        self.dump_csv(cfg)

    def apply_lr_multiplier(self, lr_mult):
        if isinstance(self.optimizer, torch.optim.Optimizer): # single optimizer
            model_base_lr = self.cfg['optimizer']['args']['lr']
            for param_group in self.optimizer.param_groups: # update the optimizer
                param_group['lr'] = model_base_lr * lr_mult
        else: # multiple optimizers
            assert (
                isinstance(self.optimizer, list)
                and len(self.optimizer) >= 1
                and isinstance(self.optimizer[0], torch.optim.Optimizer)
                and isinstance(self.optimizer[1], torch.optim.Optimizer)
            ), 'optimizer should be a list of optimizers'
            model_base_lr = self.cfg['optimizer']['args']['lr']
            loss_base_lr = self.cfg['optimizer']['loss_args']['lr']
            for param_group in self.optimizer[0].param_groups: # update the model optimizer
                param_group['lr'] = model_base_lr * lr_mult
            for param_group in self.optimizer[1].param_groups: # update the loss optimizer
                param_group['lr'] = loss_base_lr * lr_mult

    def adjust_learning_rate_stepwise(self):
        lr_type = self.cfg['optimizer']['lr_type']
        max_epoch = self.cfg['max_epoch']
        max_step = self.n_steps_per_epoch * max_epoch
        current_step = self.global_step
        lr_mult = self.lr_mult_epochwise

        assert 'warmup_epoch' in self.cfg['optimizer'], 'warmup_epoch not found in optimizer'
        warmup_step = self.cfg['optimizer']['warmup_epoch'] * self.n_steps_per_epoch

        if lr_type == 'cosine':
            if 'min_lr_mult' in self.cfg['optimizer']:
                min_lr_mult = self.cfg['optimizer']['min_lr_mult']
            else:
                self.log('min_lr_mult not found in optimizer, using 0.1')
                min_lr_mult = 0.1
            if current_step <= warmup_step:
                lr_mult = min_lr_mult + (1. - min_lr_mult) * current_step / warmup_step
            else:
                lr_mult = min_lr_mult + (1. - min_lr_mult) * 0.5 * (cos(pi*(current_step - warmup_step) / (max_step - warmup_step)) + 1)
        elif lr_type == 'step':
            return lr_mult # do not need to adjust learning rate stepwise
        else:
            raise NotImplementedError(f'lr_type {lr_type} not implemented')

        self.apply_lr_multiplier(lr_mult)
        return lr_mult

    def log_temp_scalar(self, k, v, t=None):
        if t is None:
            t = self.epoch
        if self.enable_tb:
            self.writer.add_scalar(k, v, global_step=t)
        if self.enable_wandb:
            wandb.log({k: v}, step=t)

    def dist_all_reduce_mean_(self, x):
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        x.div_(self.tot_gpus)

    def sync_ave_scalars_(self, ave_scalars):
        for k in ave_scalars.keys():
            x = torch.tensor(ave_scalars[k].item(), dtype=torch.float32, device=self.device)
            self.dist_all_reduce_mean_(x)
            ave_scalars[k].v = x.item()
            ave_scalars[k].n *= self.tot_gpus

    # === UVT modification (TR-1b): 梯度累积辅助接口（G/D 双优化器共用）===
    def grad_accum_ctx(self, ddp_module):
        """DDP 下前 k-1 个 micro-step 返回 no_sync() 省 all-reduce 通信；
        窗口末步 / 非 DDP / 未开累积时返回空上下文。
        用法（子类 train_step 内，G 侧与 D 侧同构）：
            with self.grad_accum_ctx(self.model_ddp):
                scaler.scale(self.scale_loss_for_grad_accum(loss)).backward()
            if self.should_optim_step:
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
        """
        if (self.grad_accumulates > 1 and not self.should_optim_step
                and hasattr(ddp_module, 'no_sync')):
            return ddp_module.no_sync()
        return contextlib.nullcontext()

    def scale_loss_for_grad_accum(self, loss):
        """累积窗口内每个 micro-step 的 loss 除以 k，使梯度总和等价于大 batch 均值。"""
        if self.grad_accumulates > 1:
            return loss / self.grad_accumulates
        return loss
    # === UVT modification end ===

    def train_step(self, data):
        raise NotImplementedError('train_step not implemented in base trainer')

    def train_epoch(self):
        self.model_ddp.train()
        ave_scalars = dict()

        pbar = self.train_loader
        if self.is_master:
            pbar = tqdm(pbar, desc='train', leave=True)

        t1 = time.time()
        for i, data in enumerate(pbar):
            self.global_step += 1
            t0 = time.time()
            self.t_data += t0 - t1
            # === UVT modification (TR-1b): 维护梯度累积窗口状态 ===
            # 注意 global_step 语义保持 LARP 原状 = micro-step 计数（LR 调度覆盖范围不变）。
            # 窗口末步或 epoch 末批（尾窗不满 k 也要 step，避免残留梯度跨 epoch）置 should_optim_step。
            self.micro_step_in_window = i % self.grad_accumulates
            self.should_optim_step = (
                (i + 1) % self.grad_accumulates == 0
                or (i + 1) == self.n_steps_per_epoch
            )
            # === UVT modification end ===
            lr_mult = self.adjust_learning_rate_stepwise()
            ret = self.train_step(data)
            self.t_model += time.time() - t0

            B = len(next(iter(data.values())))
            for k, v in ret.items():
                if ave_scalars.get(k) is None:
                    ave_scalars[k] = utils.Averager()
                ave_scalars[k].add(v, n=B)
                if self.stepwise_logging:
                    self.log_temp_scalar(f'train/{k}', v, self.global_step)

            if self.is_master:
                if 'psnr' in ave_scalars and 'loss' in ave_scalars:
                    info = f'train: psnr={ave_scalars["psnr"].v:5.2f}, loss={ave_scalars["loss"].v:.4f}'
                    info += f', iu={ret["index_usage"]:.4f}' if 'index_usage' in ret else ''
                    info += f', iub={ret["index_usage_batch"]:.4f}' if 'index_usage_batch' in ret else ''
                    info += f', lr_m={lr_mult:.3f}'
                    pbar.set_description(desc=info, refresh=False)
                elif 'loss' in ave_scalars:
                    pbar.set_description(desc=f'train: loss={ave_scalars["loss"].v:.4f}, lr_m={lr_mult:.3f}', refresh=False)

            t1 = time.time()

        if self.distributed:
            self.sync_ave_scalars_(ave_scalars)

        logtext = 'train:'
        if not self.stepwise_logging:
            for k, v in ave_scalars.items():
                # pdb.set_trace()
                logtext += f' {k}={v.item():.4f}'
                self.log_temp_scalar('train/' + k, v.item())
        self.log_buffer.append(logtext)
        if self.is_master:
            if 'psnr' in ave_scalars:
                self.train_psnr.append(ave_scalars['psnr'].v)
            if 'loss' in ave_scalars:
                self.train_loss.append(ave_scalars['loss'].v)

    def evaluate_step(self, data):
        data = {k: v.to(self.device) for k, v in data.items()}
        with torch.no_grad():
            loss = self.model_ddp(data)
        return {'loss': loss.item()}

    def evaluate_epoch(self):
        self.model_ddp.eval()
        ave_scalars = dict()

        for dataset_name, test_loader in self.test_loader_dict.items():
            pbar = test_loader
            if self.is_master:
                pbar = tqdm(pbar, desc=f'eval {dataset_name}', leave=True)

            self.fake_stats = None
            self.running_real_stats = None
            self.img_fake_stats = None
            self.img_running_real_stats = None

            t1 = time.time()
            for data in pbar:
                t0 = time.time()
                self.t_data += t0 - t1
                ret = self.evaluate_step(data)
                self.t_model += time.time() - t0

                B = len(next(iter(data.values())))
                for k, v in ret.items():
                    if ave_scalars.get(k) is None:
                        ave_scalars[k] = utils.Averager()
                    ave_scalars[k].add(v, n=B)

                if self.is_master:
                    if 'psnr' in ave_scalars and 'ssim' in ave_scalars and 'fps' in ave_scalars:
                        pbar.set_description(desc=f'Eval: FPS={ave_scalars["fps"].v:.2f}, psnr={ave_scalars["psnr"].v:.2f}, ssim={ave_scalars["ssim"].v:.4f}' )
                    if 'loss' in ave_scalars:
                        pbar.set_description(desc=f'Eval: loss={ave_scalars["loss"].v:.4f}')
                t1 = time.time()

            if self.distributed:
                self.sync_ave_scalars_(ave_scalars)

            logtext = '\n eval:'
            for k, v in ave_scalars.items():
                logtext += f' {dataset_name}_{k}={v.item():.4f}'
                self.log_temp_scalar(f'test/{dataset_name}_' + k, v.item())

            if self.fake_stats is not None and self.is_master:
                try:
                    if self.running_real_stats is not None:
                        self.log('Calculating FVD with running real stats')
                        fvd = self.fvd_calculator.calculate_fvd(
                            self.fake_stats,
                            self.running_real_stats,
                        )
                        self.running_real_stats = None

                    elif self.fvd_real_stats is not None:
                        self.log('Calculating FVD with loaded real stats')
                        fvd = self.fvd_calculator.calculate_fvd(
                            self.fake_stats,
                            self.fvd_real_stats,
                        )
                    else:
                        self.log('Calculating FVD with entire dataset')
                        fvd = self.fvd_calculator.calculate_fvd_with_dataset(
                            self.fake_stats, 
                            test_loader.dataset,
                            bs=32, 
                            cache_stats=(self.rank == 0)
                        )
                except Exception as e:
                    self.log(f'FVD calculation failed: {e}')
                    fvd = 99999.99

                logtext += f' {dataset_name}_fvd={fvd:.4f}'
                self.log_temp_scalar(f'test/{dataset_name}_fvd', fvd)
                self.current_fvd = fvd


            if self.img_fake_stats is not None and self.is_master and self.fid_calculator is not None:
                try:
                    if self.img_running_real_stats is not None:
                        self.log('Calculating FID with running real stats')
                        real = self.img_running_real_stats
                    elif self.fid_real_stats is not None:
                        self.log('Calculating FID with loaded real stats')
                        real = self.fid_real_stats
                    else:
                        self.log('Calculating FID with entire dataset')
                        real = test_loader.dataset
                    fid = self.fid_calculator.calculate_fid_smart(
                        self.img_fake_stats, real, bs=32, cache_stats=(self.rank == 0)
                    )
                except Exception as e:
                    self.log(f'FID calculation failed: {e}')
                    fid = 99999.99

                logtext += f' {dataset_name}_fid={fid:.4f}'
                self.log_temp_scalar(f'test/{dataset_name}_fid', fid)
                self.current_fid = fid


            if self.distributed:
                dist.barrier()

            self.log_buffer.append(logtext)
            if self.is_master:
                if dataset_name not in self.val_psnr:
                    self.val_psnr[dataset_name] = []
                    self.val_loss[dataset_name] = []
                if 'psnr' in ave_scalars:
                    self.val_psnr[dataset_name].append(ave_scalars['psnr'].v)
                if 'loss' in ave_scalars:
                    self.val_loss[dataset_name].append(ave_scalars['loss'].v)


    def visualize_epoch(self):
        pass


    @torch.no_grad()
    def update_ema(self, ema_model: nn.Module, decay: float):
        """
        Step the EMA model towards the current model.
        """
        ema_params = OrderedDict(ema_model.named_parameters())
        model_params = OrderedDict(get_orig_module(self.model).named_parameters())

        for name, param in model_params.items():
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)
            

    def save_checkpoint(self, filename, save_best=False, model_sd_only=False):
        # also save the rng states for each rank
        self.log('Preparing to save rng states...')
        rng_states_per_rank = {}

        local_rng_state_dict = {
            'torch_rng_state': torch.get_rng_state(),
            'torch_cuda_rng_state': accel.get_rng_state() if accel.is_available() else None,
            'numpy_rng_state': np.random.get_state(),
            'python_rng_state': random.getstate(),
        }
        if self.tot_gpus > 1:
            rng_states_per_rank.update(utils.gather_object_from_all(local_rng_state_dict))
        else:
            rng_states_per_rank[0] = local_rng_state_dict
        if not self.is_master:
            return
        
        self.log('Saving checkpoint...')
        model_spec = deepcopy(dict(self.cfg['model']))
        model_spec['sd'] = get_orig_module(self.model).state_dict()
        optimizer_spec = deepcopy(dict(self.cfg['optimizer']))

        if hasattr(self, 'optimizer') and self.optimizer is not None:
            if isinstance(self.optimizer, torch.optim.Optimizer): # single optimizer
                optimizer_spec['sd'] = self.optimizer.state_dict()
            else: # multiple optimizers
                assert isinstance(self.optimizer, list) and len(self.optimizer) >= 1 and isinstance(self.optimizer[0], torch.optim.Optimizer)
                optimizer_spec['sd'] = [opt.state_dict() for opt in self.optimizer]

        if hasattr(self, 'scaler') and self.scaler is not None:
            if isinstance(self.scaler, torch.cuda.amp.GradScaler): # single scaler
                scaler_sd = self.scaler.state_dict()
            else: # multiple scalers
                assert isinstance(self.scaler, list) and len(self.scaler) >= 1 and isinstance(self.scaler[0], torch.cuda.amp.GradScaler)
                scaler_sd = [scaler.state_dict() for scaler in self.scaler]
        else:
            scaler_sd = None

        checkpoint = {
            'model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': self.epoch,
            'cfg': self.cfg,
            'wandb_run_id': wandb.run.id if self.enable_wandb else None,
            'rng_states_per_rank': rng_states_per_rank,
        }

        if self.epoch == self.cfg['max_epoch']:
            del checkpoint['optimizer']
            del checkpoint['rng_states_per_rank']

        if 'loss' in dict(self.cfg):
            loss_spec = deepcopy(dict(self.cfg['loss']))
            if hasattr(self, 'loss') and isinstance(self.loss, nn.Module):
                loss_spec['sd'] = get_orig_module(self.loss).state_dict()
            checkpoint['loss'] = loss_spec

        if self.use_amp and scaler_sd is not None:
            checkpoint['scaler_sd'] = scaler_sd
            if self.epoch == self.cfg['max_epoch']:
                del checkpoint['scaler_sd']

        # save EMA model state_dict
        for ema_decay, ema_model in self.ema_model_dict.items():
            ema_sd = ema_model.state_dict()
            if 'ema_sd' not in checkpoint['model']:
                checkpoint['model']['ema_sd'] = {}
            checkpoint['model']['ema_sd'][ema_decay] = ema_sd

        checkpoint = self.before_save_checkpoint(checkpoint)

        if model_sd_only:
            checkpoint = {
                'model': model_spec,
                'cfg': self.cfg,
            }

        torch.save(checkpoint, osp.join(self.cfg['env']['save_dir'], filename))

        if save_best:
            # automically choose fvd or fid. If both are available, choose fvd
            if self.current_fid < 99999:
                best_prefix = 'best_fid_'
                current_metric = self.current_fid
            elif self.current_fvd < 99999:
                best_prefix = 'best_fvd_'
                current_metric = self.current_fvd
            else:
                self.log('No metric available for best checkpoint')
                return
            
            # find the existing best checkpoint
            all_files = os.listdir(self.cfg['env']['save_dir'])
            best_ckpt_files = [f for f in all_files if f.startswith(best_prefix)]
            get_metric_from_filename = lambda x: float(x[len(best_prefix):-4])
            previous_metrics = [get_metric_from_filename(f) for f in best_ckpt_files]
            if len(previous_metrics) == 0:
                previous_best_metric = 99999.99
            else:
                previous_best_metric = min(previous_metrics)

            if current_metric < previous_best_metric:
                best_filename = f'{best_prefix}{current_metric:.2f}.pth'
                shutil.copyfile(osp.join(self.cfg['env']['save_dir'], filename), osp.join(self.cfg['env']['save_dir'], best_filename))
                self.log(f'New best checkpoint saved: {best_filename}')
                # remove the previous best checkpoint
                for f in best_ckpt_files:
                    os.remove(osp.join(self.cfg['env']['save_dir'], f))

    def before_save_checkpoint(self, checkpoint):
        return checkpoint
