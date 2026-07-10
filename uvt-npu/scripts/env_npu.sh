#!/bin/bash
# === UVT NPU 运行环境（source 它，或被其它脚本 source）===
# 用法：source scripts/env_npu.sh
#
# 固化三个必需/建议的环境变量：
#   1. TORCH_DEVICE_BACKEND_AUTOLOAD=0  ——【必需】torchrun 启动器（torch.distributed.run）自身
#      会 import torch，而本机 import torch 会自动加载 torch_npu→inductor→triton（AttrsDescriptor
#      不兼容）→ 崩。关掉自动加载，torch_npu 改由 utils/accel.py 在 shim 后显式 import。
#      （单卡 `python train.py` 因 train.py 顶部内联了 shim 不需要它，但开着也无害，统一开。）
#   2. HF_ENDPOINT=https://hf-mirror.com  ——【建议】本机 huggingface.co 不通，走 hf-mirror 拉
#      SigLIP2/InternVideo 等权重。已能直连 HF 的话可覆盖：HF_ENDPOINT=https://huggingface.co。
#   3. PYBIN  —— PyTorch-2.6.0 conda 环境的 python（含 torch 2.6.0 + torch_npu 2.6.0.post5）。
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export PYBIN=${PYBIN:-/home/ma-user/anaconda3/envs/PyTorch-2.6.0/bin/python}
export PYTHONPATH=$(dirname $(dirname $(realpath "$BASH_SOURCE"))):${PYTHONPATH}
echo "[env_npu] TORCH_DEVICE_BACKEND_AUTOLOAD=$TORCH_DEVICE_BACKEND_AUTOLOAD  HF_ENDPOINT=$HF_ENDPOINT"
