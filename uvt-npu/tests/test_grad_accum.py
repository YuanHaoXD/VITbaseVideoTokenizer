"""TR-1b 验收测试：梯度累积（grad_accumulates=4 与 =1 同 seed 数值一致性）。

分两层：
  1. test_grad_accum_math_equivalence —— 累积语义的数学等价性（CPU 即可跑）：
     「micro-loss/k 累积 backward + 窗口末 step」 ≡ 「整批一次 backward + step」。
     这正是 base_trainer.scale_loss_for_grad_accum / should_optim_step 实现的语义。
  2. test_grad_accum_trainer_loss_consistency —— 训练器级一致性（骨架，需 GPU）：
     BaseTrainer 强制 cuda 设备 + AMP + DDP（no_sync 路径），CPU 无法覆盖。
     完整验收命令（8 卡）：
       torchrun --nproc_per_node=8 train.py --cfg cfgs/uvt_stage1.yaml \
           --csv_file null128 --opts grad_accumulates 4
     与 grad_accumulates=1（micro-bs ×4）对照，同 seed 前 50 步 loss 曲线应逐点近似。
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


def _train_once(grad_accumulates, steps=5, batch=16):
    """模拟 base_trainer 的累积语义：k 个 micro-step 各 backward(loss/k)，窗口末 step。"""
    assert batch % grad_accumulates == 0
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 16), nn.Tanh(), nn.Linear(16, 1))
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    data_gen = torch.Generator().manual_seed(42)   # 数据序与模型初始化 seed 解耦

    losses = []
    for _ in range(steps):
        x = torch.randn(batch, 8, generator=data_gen)
        y = torch.randn(batch, 1, generator=data_gen)
        window_loss = 0.0
        for xb, yb in zip(x.chunk(grad_accumulates), y.chunk(grad_accumulates)):
            # 等价 base_trainer：scale_loss_for_grad_accum(loss).backward()
            loss = F.mse_loss(model(xb), yb) / grad_accumulates
            loss.backward()
            window_loss += loss.item()
        # 等价 base_trainer：should_optim_step → step + zero_grad
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(window_loss)
    return losses, [p.detach().clone() for p in model.parameters()]


def test_grad_accum_math_equivalence():
    """grad_accum=4（micro-bs=4）与 =1（bs=16）同 seed：loss 与最终权重逐点一致。

    注意等价性前提：micro-batch 等大小（chunk 均分），损失为 batch 均值——
    sum_i mean(loss_i)/k == mean(loss_全批)。BaseTrainer 的 drop_last=True 保证批形状恒定。
    """
    l1, p1 = _train_once(grad_accumulates=1)
    l4, p4 = _train_once(grad_accumulates=4)
    for a, b in zip(l1, l4):
        assert abs(a - b) < 1e-6, f'loss 不一致: {a} vs {b}'
    for a, b in zip(p1, p4):
        assert torch.allclose(a, b, atol=1e-6), '累积后权重与整批更新不一致'


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason='需 GPU：BaseTrainer 强制 cuda 设备 / AMP / DDP no_sync 路径')
def test_grad_accum_trainer_loss_consistency():
    """【骨架，待 M-10 就位后补全 —— 需 GPU 跑】

    计划步骤：
      1. tiny UVTConfig + csv_file=null128 假数据集，构造 uvt_tokenizer_trainer（stage 1）；
      2. 同 manualSeed 跑两组各 50 步：
         A 组 grad_accumulates=1, batch_size=32；B 组 grad_accumulates=4, batch_size=8；
      3. 断言两组按「优化器步」对齐的 loss 曲线逐点近似（bf16 下 rtol≈1e-2）；
      4. 多卡（torchrun --nproc_per_node=8）重复 2-3，额外覆盖 DDP no_sync 通信路径，
         并断言 no_sync 开/关的最终梯度一致（通信优化不得改变数值）。
    """
    pytest.skip('骨架：依赖 M-10 UVTTokenizer 与 GPU 集群环境（TR-1 验收项，见 docstring）')
