"""M-1 验收测试：models/uvt/attention_mask.py。

契约（02_代码实施.md §5 / 05 §2 M-1）：
- make_time_ids / make_bool_mask / attn_bias 三件套，kind ∈ {full, causal, tubelet}。
- tubelet 可达性是全模型的命门：时间位 t 只见 {t-1, t}，锚位(位0)只见自己。
- full 模式 attn_bias 返回 None（走 SDPA 无 mask 快速路径）。

纯离线，只依赖 torch。
"""
import pytest
import torch

from models.uvt.attention_mask import attn_bias, make_bool_mask, make_time_ids


def test_tubelet_reachability():
    """T1=3,N=2 tubelet：位 t 只见 {t-1,t}；锚位(位0)只见自己（不见位2←→位0）。

    期望矩阵（02§5）：
      [[1,1,0,0,0,0],[1,1,0,0,0,0],
       [1,1,1,1,0,0],[1,1,1,1,0,0],
       [0,0,1,1,1,1],[0,0,1,1,1,1]]
    """
    torch.manual_seed(0)
    time_ids = make_time_ids(T1=3, N=2, device=torch.device("cpu"))
    assert time_ids.tolist() == [0, 0, 1, 1, 2, 2], \
        f"time_ids 排布错误: {time_ids.tolist()}"

    mask = make_bool_mask(time_ids, kind="tubelet")
    expected = torch.tensor(
        [
            [1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    assert mask.dtype == torch.bool
    assert mask.shape == (6, 6)
    assert torch.equal(mask, expected), f"tubelet 可达性矩阵错误:\n{mask.int()}"

    # 命门断言：位2 不见位0（否则信息会从未来泄漏回锚帧所在的因果链）。
    assert not mask[4, 0] and not mask[5, 0], "位2 必须不见位0"


def test_causal_reachability():
    """causal: tk <= tq。位0只见位0；位2见{0,1,2}。"""
    torch.manual_seed(0)
    time_ids = make_time_ids(3, 2, torch.device("cpu"))
    mask = make_bool_mask(time_ids, kind="causal")
    expected = torch.tensor(
        [
            [1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    assert mask.dtype == torch.bool
    assert mask.shape == (6, 6)
    assert torch.equal(mask, expected), f"causal 可达性矩阵错误:\n{mask.int()}"


def test_full_all_ones():
    """full: 全时空双向，所有位置互见。"""
    torch.manual_seed(0)
    time_ids = make_time_ids(3, 2, torch.device("cpu"))
    mask = make_bool_mask(time_ids, kind="full")
    assert mask.dtype == torch.bool
    assert mask.shape == (6, 6)
    assert mask.all(), "full mask 必须全为 True"


def test_attn_bias_full_returns_none():
    """full 模式 attn_bias 返回 None：调用方走 SDPA 无 mask 快速路径。"""
    torch.manual_seed(0)
    bias = attn_bias(T1=4, N=3, kind="full",
                     device=torch.device("cpu"), dtype=torch.float32)
    assert bias is None


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_attn_bias_cached_dtype(dtype):
    """causal bias: 形状 [1,1,S,S]；dtype 匹配入参；允许项=0、禁止项=finfo.min。"""
    torch.manual_seed(0)
    T1, N = 3, 2
    S = T1 * N
    bias = attn_bias(T1=T1, N=N, kind="causal",
                     device=torch.device("cpu"), dtype=dtype)
    assert bias is not None
    assert bias.shape == (1, 1, S, S)
    assert bias.dtype == dtype

    time_ids = make_time_ids(T1, N, torch.device("cpu"))
    forbidden = make_bool_mask(time_ids, "causal").logical_not()  # True=禁止位置
    permitted = forbidden.logical_not()
    neg = torch.finfo(dtype).min

    assert torch.all(bias[0, 0][forbidden] == neg), \
        f"禁止项应等于 finfo({dtype}).min={neg}"
    assert torch.all(bias[0, 0][permitted] == 0.0), "允许项应为 0"

    # lru_cache 契约：同 (T1,N,kind,device,dtype) 再次调用返回同一 tensor 对象。
    bias2 = attn_bias(T1=T1, N=N, kind="causal",
                      device=torch.device("cpu"), dtype=dtype)
    assert bias2 is bias, "同键应命中 lru_cache 返回同一 tensor"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
