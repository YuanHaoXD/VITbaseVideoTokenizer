"""
    Generate a cfg object according to a cfg file and args, then spawn Trainer(rank, cfg).
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.distributed as dist  # === UVT modification (TR-1a): torchrun 入口需要显式管理进程组 ===
import yaml
from easydict import EasyDict as edict
from mergedeep import merge

import trainers
import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg')
    parser.add_argument('--data_path', default='data/k400')
    parser.add_argument('--csv_file', default='k400_train.js')
    parser.add_argument('--eval_frames', type=str, default='none')
    parser.add_argument('--frame_num', type=int, default=4)
    parser.add_argument('--input_size', type=int, default=128)
    parser.add_argument('--batch_size', '-b', type=int, default=16)
    parser.add_argument('--num_workers', '-j', type=int, default=16)
    parser.add_argument('--out_path', type=str, default='default')
    parser.add_argument('--name', '-n', default=None)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--cudnn', action='store_true')
    parser.add_argument('--replace', action='store_true')
    parser.add_argument('--wandb-upload', '-w', action='store_true')
    parser.add_argument('--wandn_entity', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument(
        '--opts', type=str, nargs='*', default=[], help='cfg args to update'
    )
    parser.add_argument('--manualSeed', type=int, default=-1, help='manual seed')
    parser.add_argument('--comment', type=str, default='')
    parser.add_argument('--debug', action='store_true')


    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    return args


def make_cfg(args):
    if args.debug:
        args.name = 'debug'
        if args.wandb_upload:
            print('!!!wandb upload is disabled in debug mode')
            args.wandb_upload = False
        args.replace = True

    with open(args.cfg, 'r', encoding='utf-8') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    def translate_cfg_(d):
        for k, v in d.items():
            if isinstance(v, dict):
                translate_cfg_(v)
            elif isinstance(v, list):
                # UVT fix: 原版只递归 dict 不递归 list → joint_dataset.sources(列表)里的
                # $var$ 永不被替换（crop_size 保持 '$input_size$' 字符串）。补 list 递归。
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        translate_cfg_(item)
                    elif isinstance(item, str) and item.startswith('$') and item.endswith('$'):
                        v[i] = getattr(args, item.replace('$', ''))
            elif isinstance(v, str):
                if v.startswith('$') and v.endswith('$'):
                    v = getattr(args, v.replace('$', ''))
                d[k] = v

    translate_cfg_(cfg)

    if args.name is None:
        exp_name = os.path.basename(args.cfg).split('.')[0]
    else:
        exp_name = args.name

    env = edict()
    # === UVT modification (TR-1a): torchrun 入口 ===
    # 世界大小改由 torchrun 注入的 WORLD_SIZE 决定（多机/容器调度友好）；
    # 无分布式环境变量时退化为 1（单进程单卡直跑，方便调试）。
    env['tot_gpus'] = int(os.environ.get('WORLD_SIZE', 1))
    # === UVT modification end ===
    env['cudnn'] = args.cudnn
    env['wandb_upload'] = args.wandb_upload
    if args.wandn_entity is not None:
        env['wandb_entity'] = args.wandn_entity
    if args.wandb_project is not None:
        env['wandb_project'] = args.wandb_project 
    cfg['env'] = env

    def build_tree(tree_list):
        if len(tree_list) >= 2:
            return {
                tree_list[0]: (
                    build_tree(tree_list[1:]) if len(tree_list) > 2 else tree_list[-1]
                )
            }

    def nested_v(dict, keys):
        for key in keys:
            dict = dict[key]
        return dict

    def convert(type, x):
        if type == bool and isinstance(x, str):
            if x.lower() == 'true':
                return True
            elif x.lower() == 'false':
                return False
            else:
                raise ValueError('Cannot convert {} to bool'.format(x))
        elif (type == list or type == tuple) and isinstance(x, str):
            x = x.split('_')
            return [eval(x0) for x0 in x]
        else:
            return type(x)

    assert len(args.opts) % 2 == 0
    for cur_cfg_key, v in zip(args.opts[::2], args.opts[1::2]):
        keys = cur_cfg_key.split('.')
        v = convert(type(nested_v(cfg, keys)), v)
        cfg = merge(cfg, build_tree(keys + [v]))

    cfg = edict(cfg)
    cfg.comment = args.comment
    cfg.train_dataset.args.cls_vid_num = cfg.train_dataset.args.cls_vid_num.strip(
        "'"
    ).strip('"')

    env.exp_name = trainers.trainers_dict[cfg['trainer']].get_exp_name(
        exp_name, cfg, args
    )
    env.save_dir = os.path.join(args.out_path, env.exp_name)
    env.port = str(2960 + utils.hash_string_to_int(env.save_dir) % 10000)
    cfg.manualSeed = args.manualSeed
    return cfg


def main_worker(rank, cfg):
    manualSeed = cfg['manualSeed']
    if manualSeed != -1:
        manualSeed += rank
        torch.manual_seed(manualSeed)
        np.random.seed(manualSeed)
        random.seed(manualSeed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(manualSeed)

    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.allow_tf32 = True

    if cfg['compile']:
        from einops._torch_specific import \
            allow_ops_in_compiled_graph  # requires einops>=0.6.1
        allow_ops_in_compiled_graph()

    trainer = trainers.trainers_dict[cfg['trainer']](rank, cfg)
    trainer.run()


# === UVT modification (TR-1a): torchrun 入口，替代 mp.spawn（02 篇 §2.2-a）===
# 启动方式统一为：
#   torchrun --nproc_per_node=8 --nnodes=N train.py --cfg ...
# torchrun 为每个进程注入 RANK / LOCAL_RANK / WORLD_SIZE 环境变量；
# 本入口直接读取它们并用默认 env:// 方式 init_process_group("nccl")。
# 无这些环境变量（如 Windows/单卡调试 `python train.py`）时退化为单进程直跑，
# 不初始化任何分布式状态。理由：mp.spawn 是单机风格，多机与容器调度不友好。
def main():
    args = parse_args()

    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    cfg = make_cfg(args)

    if world_size > 1:
        # torchrun 分布式路径：先绑卡再建组（NCCL 要求每进程独占一卡）
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')

    # save_dir 只由全局 rank 0 创建/清理，其余 rank 通过 barrier 等待其就绪
    if rank == 0:
        utils.ensure_path(cfg['env']['save_dir'], args.replace)
    if world_size > 1:
        dist.barrier()

    try:
        main_worker(rank, cfg)
    finally:
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()
# === UVT modification end ===


if __name__ == '__main__':
    main()