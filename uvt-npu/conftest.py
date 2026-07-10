"""pytest 配置：在收集任何 test 模块之前应用 triton 兼容 shim。

原因：本机 triton 3.6 与 torch 2.6.0 不兼容（AttrsDescriptor 改名），而 test 模块顶部
``import torch`` 会触发 torch_npu 自动加载→inductor→triton→ImportError。pytest 在收集阶段
会 import 本文件（早于 test 模块），故在此补回 AttrsDescriptor，使后续 import torch 不崩。

（与 train.py 顶部、utils/accel.py 里的 shim 等价，三处冗余但幂等，保各种入口都稳。）
"""
try:
    import triton.compiler.compiler as _tcc
    if not hasattr(_tcc, "AttrsDescriptor"):
        _tcc.AttrsDescriptor = type("AttrsDescriptor", (), {})
except Exception:
    pass
