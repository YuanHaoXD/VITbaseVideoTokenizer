"""M-2 验收测试：models/uvt/temporal_fold.py。

契约（02_代码实施.md / 05 §2 M-2）：
- TemporalFold2x(dim):   [B,1+T,N,D] -> [B,1+T/2,N,D]，相邻两帧通道拼接 + Linear(2D→D)。
- TemporalUnfold2x(dim): [B,1+T,N,D] -> [B,1+2T,N,D]，Fold 的逆算子。
- 锚帧(位0)永不参与折叠（ADR-4' 锚帧隔离）；纯图像(只有锚位)直通。
- Fold 初始化为"两帧平均"，Unfold 初始化为"复制两份"，二者在恒等输入下互逆。

纯离线，只依赖 torch。
"""
import pytest
import torch

from models.uvt.temporal_fold import TemporalFold2x, TemporalUnfold2x


def test_fold_shapes():
    """[2,1+16,256,1152] 经两次 TemporalFold2x -> [2,1+4,256,1152]（16→8→4）。"""
    torch.manual_seed(0)
    B, T1, N, D = 2, 1 + 16, 256, 1152
    x = torch.randn(B, T1, N, D)
    fold_a = TemporalFold2x(D)
    fold_b = TemporalFold2x(D)

    y = fold_a(x)
    assert y.shape == (2, 1 + 8, 256, 1152), f"第一次 fold 后形状错误: {y.shape}"
    z = fold_b(y)
    assert z.shape == (2, 1 + 4, 256, 1152), f"第二次 fold 后形状错误: {z.shape}"


def test_image_passthrough():
    """纯图像(只有锚位)直通：[2,1,256,1152] -> 形状与数值均不变（锚帧隔离）。"""
    torch.manual_seed(0)
    fold = TemporalFold2x(1152)
    x = torch.randn(2, 1, 256, 1152)
    y = fold(x)
    assert y.shape == x.shape, f"纯图像应直通保持形状: {y.shape}"
    assert torch.equal(y, x), "纯图像直通应返回原 tensor（锚帧隔离，frames 为空触发直通分支）"


def test_unfold_shapes():
    """[2,1+4,256,1152] 经两次 TemporalUnfold2x -> [2,1+16,256,1152]（4→8→16）。"""
    torch.manual_seed(0)
    B, T1, N, D = 2, 1 + 4, 256, 1152
    x = torch.randn(B, T1, N, D)
    unf_a = TemporalUnfold2x(D)
    unf_b = TemporalUnfold2x(D)

    y = unf_a(x)
    assert y.shape == (2, 1 + 8, 256, 1152), f"第一次 unfold 后形状错误: {y.shape}"
    z = unf_b(y)
    assert z.shape == (2, 1 + 16, 256, 1152), f"第二次 unfold 后形状错误: {z.shape}"


def test_fold_unfold_init_inverse():
    """Fold(两帧平均) 与 Unfold(复制两份) 的初始化在恒等输入下互逆。

    构造每对相邻帧相等的输入：Fold 平均两相等帧=原值，Unfold 复制两份=还原；
    故 fold 再 unfold 应近似还原原输入。
    """
    torch.manual_seed(0)
    D = 32
    B, N = 2, 4
    anchor = torch.randn(B, 1, N, D)
    pair0 = torch.randn(B, 1, N, D)
    pair1 = torch.randn(B, 1, N, D)
    # 4 帧 = [pair0, pair0, pair1, pair1]：每对相邻帧相等（恒等输入）。
    frames = torch.cat([pair0, pair0, pair1, pair1], dim=1)
    x = torch.cat([anchor, frames], dim=1)  # [B, 1+4, N, D]

    fold = TemporalFold2x(D)
    unfold = TemporalUnfold2x(D)
    with torch.no_grad():
        y = fold(x)       # [B, 1+2, N, D]
        z = unfold(y)     # [B, 1+4, N, D]

    assert y.shape == (B, 1 + 2, N, D), f"fold 后形状错误: {y.shape}"
    assert z.shape == x.shape, f"unfold 后形状错误: {z.shape}"
    assert torch.allclose(z, x, atol=1e-6), \
        f"fold→unfold 未还原恒等输入: max|Δ|={(z - x).abs().max().item()}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
