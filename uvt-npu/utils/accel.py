"""utils/accel.py — 统一加速器抽象（NPU / CUDA / CPU）。

为什么存在：原 LARP/UVT 代码硬编码 ``torch.cuda.*`` 与 ``backend='nccl'``。
本机是华为昇腾 NPU（Ascend 910B2 + CANN 8.2.RC1 + torch_npu），需用
``torch.npu.*`` 与 ``hccl`` 后端。本模块把所有设备相关调用收口到一个 facade，
框架层（base_trainer / train / uvt_tokenizer_trainer）只需 ``from utils import accel``
然后 ``accel.X``，不再直接碰 ``torch.cuda``。模型/损失/教师/数据代码本身设备无关，不动。

**导入本模块即完成两件必须在 import torch_npu 之前做的事：**

1. **triton 兼容 shim**：本机 triton 是 3.6（torch 2.6.0 要的是 3.2），新版把
   ``triton.compiler.compiler.AttrsDescriptor`` 改名了；而 ``import torch_npu`` 会链式触发
   ``torch._inductor.runtime.hints`` 去硬 import 这个名字 → ImportError。这里在导入前给它补回去
   （triton 在 NPU 上根本不用，纯为骗过这条 import）。详见 docs/08 §6「前向兼容类」同类问题。
2. **import torch_npu**：注册 NPU 后端。导入后 ``torch.npu`` 可用，``torch.cuda.is_available()``
   在这台 NPU-only 机器上返回 False。

设备选择优先级：**npu > cuda > cpu**。``accel.BACKEND`` ∈ {'npu','cuda','cpu'}；
``accel.DIST_BACKEND`` ∈ {'hccl','nccl','gloo'}。
"""
# ───────────────────────────── 1. triton shim（必须在 import torch 之前！）─────────
# 本机 triton 3.6 改名了 AttrsDescriptor，而 import torch 会自动加载 torch_npu→inductor→
# 去 from triton...import AttrsDescriptor → 崩。所以这段必须比 ``import torch`` 先跑。
# triton.compiler.compiler 自身不 import torch，故放最前安全。
try:
    import triton.compiler.compiler as _tcc  # noqa: F401
    if not hasattr(_tcc, "AttrsDescriptor"):
        # 新版 triton 改名了；补一个占位类即可让 ``from ... import AttrsDescriptor`` 通过。
        _tcc.AttrsDescriptor = type("AttrsDescriptor", (), {})
except Exception:
    # 没有 triton 也不会用到它（NPU 走 CANN 编译）——忽略。
    pass

import torch

# ───────────────────────────── 2. import torch_npu ────────────────────────
_HAS_NPU = False
try:
    import torch_npu  # noqa: F401  —— 注册 NPU 后端
    if torch.npu.is_available():
        _HAS_NPU = True
except Exception:
    pass

_HAS_CUDA = torch.cuda.is_available()

if _HAS_NPU:
    BACKEND = "npu"
elif _HAS_CUDA:
    BACKEND = "cuda"
else:
    BACKEND = "cpu"

# 分布式后端：NPU→hccl，CUDA→nccl，CPU→gloo
DIST_BACKEND = {"npu": "hccl", "cuda": "nccl"}.get(BACKEND, "gloo")


# ───────────────────────────── 基本设备 API ──────────────────────────────
def is_available():
    """有加速器（npu 或 cuda）可用。"""
    return _HAS_NPU or _HAS_CUDA


def device_count():
    if _HAS_NPU:
        return torch.npu.device_count()
    if _HAS_CUDA:
        return torch.cuda.device_count()
    return 0


def set_device(i):
    if _HAS_NPU:
        torch.npu.set_device(i)
    elif _HAS_CUDA:
        torch.cuda.set_device(i)
    # cpu: no-op


def current_device():
    if _HAS_NPU:
        return torch.npu.current_device()
    if _HAS_CUDA:
        return torch.cuda.current_device()
    return 0


def device(index=None):
    """返回当前加速器的 torch.device（无加速器则 cpu）。"""
    if not is_available():
        return torch.device("cpu")
    if index is None:
        index = current_device()
    return torch.device(BACKEND, index)


def empty_cache():
    if _HAS_NPU:
        torch.npu.empty_cache()
    elif _HAS_CUDA:
        torch.cuda.empty_cache()


# ───────────────────────────── 随机数状态 ────────────────────────────────
def manual_seed_all(seed):
    if _HAS_NPU:
        torch.npu.manual_seed_all(seed)
    elif _HAS_CUDA:
        torch.cuda.manual_seed_all(seed)


def get_rng_state():
    if _HAS_NPU:
        return torch.npu.get_rng_state()
    if _HAS_CUDA:
        return torch.cuda.get_rng_state()
    return None


def set_rng_state(state):
    if state is None:
        return
    if _HAS_NPU:
        torch.npu.set_rng_state(state)
    elif _HAS_CUDA:
        torch.cuda.set_rng_state(state)


# ───────────────────────────── AMP / autocast ────────────────────────────
def autocast(dtype=None, enabled=True):
    """设备类型随 BACKEND 走（npu/cuda/cpu）的 autocast 上下文。"""
    return torch.autocast(device_type=BACKEND, dtype=dtype, enabled=enabled)


def GradScaler(enabled=True):
    """GradScaler。

    本项目配置用 bf16（cfg amp_dtype=bfloat16），GradScaler 恒为 **disabled**（bf16 无需
    loss scaling）。disabled 的 ``torch.cuda.amp.GradScaler`` 是纯透传（scale 返回原 loss、
    step 调 opt.step、update 空操作），即便在 NPU-only 机器上也安全；且与 base_trainer 里
    ``isinstance(scaler, torch.cuda.amp.GradScaler)`` 的判定天然兼容（无需改那两处）。

    若将来在 NPU 上启用 fp16 需要真正生效的 scaler，应改用 ``torch.npu.amp.GradScaler``，
    并同步改 base_trainer 的 isinstance 判定。当前 bf16 路径无需如此。
    """
    return torch.cuda.amp.GradScaler(enabled=enabled)


# ───────────────────────────── DataLoader pin_memory ─────────────────────
def pin_memory_device():
    """DataLoader 的 pin_memory_device：'npu'/'cuda' 或 None（cpu）。"""
    return BACKEND if BACKEND in ("npu", "cuda") else None


# ───────────────────────────── checkpoint map_location ───────────────────
def map_location_fn(storage, loc):
    """torch.load 的 map_location：把 tensor 落到当前加速器。"""
    if not is_available():
        return storage
    if loc.startswith(BACKEND):
        # torch 2.6:_StorageBase.to(dev) 位置参数失效(TypeError),须关键字 device=(实测 npu/cuda 均可)。
        return storage.to(device=device(current_device()))
    # 其它来源（如 cpu 保存的 ckpt）原样返回，由后续 .to(self.device) 搬运
    return storage
