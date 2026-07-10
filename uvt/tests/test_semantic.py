"""E-4 语义线验收测试：eval/semantic/{zeroshot,linear_probe,cknna}.py。

契约断言（03 篇 §5.2 + README D11）：
- zeroshot：s_pool[ B,D_t] × teacher.encode_text[K,D_t] → top-1；prompt 模板校验。
- linear_probe：特征缓存路径契约（{dataset}_{split}.pt）；图像锚帧均值池化 / 视频剔锚帧时空池化。
- cknna：**D11 负向断言**——cknna() 接口层不接收任何教师/参照模型对象（ast 源码级检查，
  无需 torch 即可验证）；参照系加载由 load_dinov2_reference 独占；DINOv2 lazy import skip。

纪律：Windows 只做 py_compile（§0 全局约定），故纯契约测试用 ast 离线验证；
需 torch 的测试用 pytest.importorskip 并注明集群命令：
  集群：cd uvt && python -m pytest tests/test_semantic.py -v
"""
import ast
from pathlib import Path

import pytest

SEM_DIR = Path(__file__).resolve().parent.parent / "eval" / "semantic"


def _src(name: str) -> str:
    return (SEM_DIR / name).read_text(encoding="utf-8")


# ======================================================================================
# D11 负向断言（离线，ast 源码级——无需 torch）
# ======================================================================================
def test_cknna_signature_has_no_teacher_param():
    """【README D11 / 循环论证禁令】cknna() 接口层不得接收任何教师/参照模型对象。

    断言 cknna 的形参只允许 (student_feats, dino_feats, layers)——从签名层杜绝
    "把教师特征当 dino_feats 传进来"或"传教师模型当参照"的调用路径。
    这是契约层禁止（非运行时检查），ast 检查保证未来重构不会偷偷加回 teacher 参数。
    """
    tree = ast.parse(_src("cknna.py"))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "cknna")
    # 扫所有形参位置（posonly / 普通 / kwonly），确保任何位置都不混入教师参数
    params = ([a.arg for a in getattr(fn.args, "posonlyargs", [])]
              + [a.arg for a in fn.args.args]
              + [a.arg for a in getattr(fn.args, "kwonlyargs", [])])

    forbidden = {"teacher", "reference", "ref_model", "reference_model", "siglip",
                 "internvideo", "model"}
    leaked = forbidden & set(params)
    assert not leaked, f"cknna() 禁止接收教师/参照模型参数（D11），发现: {leaked}"
    assert set(params) == {"student_feats", "dino_feats", "layers"}, \
        f"cknna() 形参应为 (student_feats, dino_feats, layers)，实际: {params}"


def test_cknna_forbid_constant_is_true():
    """FORBID_TEACHER_AS_REFERENCE 必须为 True 且被 cknna 运行时校验（D11 双保险）。"""
    tree = ast.parse(_src("cknna.py"))
    consts = {n.targets[0].id: n.value
              for n in ast.walk(tree)
              if isinstance(n, ast.Assign)
              and isinstance(n.targets[0], ast.Name)
              and n.targets[0].id == "FORBID_TEACHER_AS_REFERENCE"}
    assert "FORBID_TEACHER_AS_REFERENCE" in consts
    assert consts["FORBID_TEACHER_AS_REFERENCE"].value is True
    # cknna 函数体内必须有对该常量的运行时断言（防被人改成 False 绕过）
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "cknna")
    assert any(isinstance(n, ast.Raise) for n in ast.walk(fn)), \
        "cknna() 必须对 FORBID_TEACHER_AS_REFERENCE 做运行时校验（raise）"


def test_cknna_reference_load_is_dinov2_only():
    """load_dinov2_reference 是参照系的唯一加载入口，且默认 model_name=DINOv2-L。"""
    tree = ast.parse(_src("cknna.py"))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "load_dinov2_reference")
    # 默认值 DINOv2_MODEL_NAME 引用，且模块顶层该常量 = "facebook/dinov2-large"
    consts = {n.targets[0].id: n.value.value
              for n in ast.walk(tree)
              if isinstance(n, ast.Assign)
              and isinstance(n.targets[0], ast.Name)
              and n.targets[0].id == "DINOV2_MODEL_NAME"}
    assert consts.get("DINOV2_MODEL_NAME") == "facebook/dinov2-large"
    # 不允许出现 load_siglip / load_internvideo 之类参照加载函数
    other_loaders = [n.name for n in ast.walk(tree)
                     if isinstance(n, ast.FunctionDef)
                     and n.name.startswith("load_") and n.name != "load_dinov2_reference"]
    assert not other_loaders, f"cknna.py 禁止其它参照加载入口（D11）: {other_loaders}"


def test_prompt_template_has_placeholder():
    """zeroshot.PROMPT_TEMPLATE 含 {} 占位符（离线 ast）。"""
    tree = ast.parse(_src("zeroshot.py"))
    consts = {n.targets[0].id: n.value.value
              for n in ast.walk(tree)
              if isinstance(n, ast.Assign)
              and isinstance(n.targets[0], ast.Name)
              and n.targets[0].id == "PROMPT_TEMPLATE"}
    assert "{}" in consts["PROMPT_TEMPLATE"], "PROMPT_TEMPLATE 必须含 {} 占位符"


def test_linear_probe_config_fields():
    """LinearProbeConfig 三数据集差异字段齐全（dataset_name/is_video/num_classes/feature_dir）。"""
    tree = ast.parse(_src("linear_probe.py"))
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == "LinearProbeConfig")
    fields = {f.target.id for f in cls.body
              if isinstance(f, ast.AnnAssign) and isinstance(f.target, ast.Name)}
    required = {"dataset_name", "is_video", "num_classes", "feature_dir"}
    assert required <= fields, f"LinearProbeConfig 缺字段: {required - fields}"


# ======================================================================================
# 需 torch 的测试（集群运行；Windows 经 importorskip 跳过）
# ======================================================================================
def test_cknna_pair_identity_and_low_correlation():
    """_cknna_pair：相同特征→1.0；两独立随机特征→接近 0（CKA 对独立特征近正交）。"""
    torch = pytest.importorskip("torch")
    from eval.semantic.cknna import _cknna_pair

    torch.manual_seed(0)
    x = torch.randn(32, 8)
    assert abs(_cknna_pair(x, x) - 1.0) < 1e-9, "相同特征 CKNNA 应为 1.0"
    # 两独立随机特征的线性 CKA 期望≈0（不同维度也行——CKA 对维度无关）
    y = torch.randn(32, 12)
    low = _cknna_pair(x, y)
    assert low < 0.3, f"独立随机特征 CKNNA 应近 0，得到 {low}"


def test_cknna_output_structure():
    torch = pytest.importorskip("torch")
    from eval.semantic.cknna import cknna

    sf = {"l3": torch.randn(20, 10), "l6": torch.randn(20, 10)}
    df = {"l13": torch.randn(20, 12), "l27": torch.randn(20, 12)}
    out = cknna(sf, df, layers=[("l3", "l13"), ("l6", "l27")])
    assert set(out["per_layer_cknna"].keys()) == {"l3|l13", "l6|l27"}
    assert 0.0 <= out["mean"] <= 1.0
    assert out["n_layers"] == 2 and out["n_samples"] == 20


def test_zeroshot_output_top1_and_n():
    """mock tokenizer + mock teacher：top1∈[0,1]，n 正确。"""
    torch = pytest.importorskip("torch")

    class _MockTeacher:
        def __init__(self, d):
            self._d = d
        def encode_text(self, prompts):
            torch.manual_seed(42)
            return torch.randn(len(prompts), self._d)

    class _MockTokenizer:
        def __init__(self, d):
            self._d = d
        def semantic(self, video):
            b = video.shape[0]
            s = torch.randn(b, 1, 4, self._d)
            return s, torch.randn(b, self._d)

    from eval.semantic.zeroshot import zeroshot_classify

    d = 16
    dataset = [{"video": torch.rand(4, 3, 1, 64, 64),
                "label": torch.randint(0, 5, (4,))} for _ in range(3)]
    out = zeroshot_classify(_MockTokenizer(d), dataset, [f"c{i}" for i in range(5)],
                            _MockTeacher(d))
    assert 0.0 <= out["top1"] <= 1.0
    assert out["n"] == 12
    assert "top5" not in out                       # top5=False 默认不报


def test_zeroshot_rejects_bad_prompt_template():
    torch = pytest.importorskip("torch")
    from eval.semantic.zeroshot import zeroshot_classify

    class _T:
        def encode_text(self, p):
            return torch.randn(len(p), 4)
    class _M:
        def semantic(self, v):
            return None, torch.randn(v.shape[0], 4)

    with pytest.raises(ValueError):
        zeroshot_classify(_M(), iter([{"video": torch.rand(1,3,1,8,8),
                                       "label": torch.tensor([0])}]),
                          ["a"], _T(), prompt_template="no placeholder")


def test_zeroshot_dim_mismatch_raises():
    """s_pool 维度 ≠ 文本嵌入维度须报错（ADR-9 对齐失败的早判）。"""
    torch = pytest.importorskip("torch")
    from eval.semantic.zeroshot import zeroshot_classify

    class _T:
        def encode_text(self, p):
            return torch.randn(len(p), 32)         # 文本 D=32
    class _M:
        def semantic(self, v):
            return None, torch.randn(v.shape[0], 16)   # s_pool D=16（不一致）

    with pytest.raises(ValueError, match="维度"):
        zeroshot_classify(_M(), iter([{"video": torch.rand(1,3,1,8,8),
                                       "label": torch.tensor([0])}]),
                          ["a", "b"], _T())


def test_linear_probe_cache_path_contract(tmp_path):
    """extract_and_cache 缓存路径 = feature_dir/{dataset_name}_{split}.pt；二次调用复用。"""
    torch = pytest.importorskip("torch")
    from eval.semantic.linear_probe import LinearProbeConfig, extract_and_cache

    class _M:
        def semantic(self, v):
            b = v.shape[0]
            return torch.randn(b, 1, 4, 8), torch.randn(b, 8)

    cfg = LinearProbeConfig(dataset_name="toyset", is_video=False, num_classes=3,
                            feature_dir=str(tmp_path), epochs=1)
    ds = [{"video": torch.rand(4, 3, 1, 16, 16),
           "label": torch.tensor([0, 1, 2, 0])}]
    p1 = extract_and_cache(_M(), ds, cfg, split="train")
    assert p1 == tmp_path / "toyset_train.pt", f"缓存路径契约违反: {p1}"
    assert p1.exists()
    blob = torch.load(p1, weights_only=True)
    assert blob["features"].shape == (4, 8)
    assert blob["labels"].shape == (4,)
    # 二次调用复用（lr 扫不重抽——03 §5.2）
    p2 = extract_and_cache(_M(), ds, cfg, split="train")
    assert p2 == p1


def test_linear_probe_feature_pooling_image_vs_video():
    """图像：s[:,0] 空间均值；视频：s[:,1:] 时空均值（剔锚帧）。"""
    torch = pytest.importorskip("torch")
    from eval.semantic.linear_probe import _pool_features

    s_img = torch.randn(2, 1, 4, 8)            # [B,T1=1,N=4,D=8]
    f_img = _pool_features(s_img, is_video=False)
    assert f_img.shape == (2, 8)
    expected = s_img[:, 0].mean(dim=1)
    assert torch.allclose(f_img, expected)

    s_vid = torch.randn(2, 5, 4, 8)            # [B,T1=5,N=4,D=8] 视频剔锚帧
    f_vid = _pool_features(s_vid, is_video=True)
    assert f_vid.shape == (2, 8)
    expected_v = s_vid[:, 1:].mean(dim=(1, 2))
    assert torch.allclose(f_vid, expected_v)


def test_linear_probe_end_to_end_mock(tmp_path):
    """linear_probe 主入口：缓存→lr 扫→报 acc（mock tokenizer + tiny 数据）。"""
    torch = pytest.importorskip("torch")
    from eval.semantic.linear_probe import LinearProbeConfig, linear_probe

    class _M:
        def semantic(self, v):
            b = v.shape[0]
            return torch.randn(b, 1, 4, 8), torch.randn(b, 8)

    cfg = LinearProbeConfig(dataset_name="toy", is_video=False, num_classes=3,
                            feature_dir=str(tmp_path), epochs=2,
                            lr_candidates=(1e-2,))
    ds = [{"video": torch.rand(8, 3, 1, 16, 16),
           "label": torch.randint(0, 3, (8,))} for _ in range(2)]
    out = linear_probe(_M(), ds, cfg)
    assert 0.0 <= out["acc"] <= 1.0
    assert out["best_lr"] == 1e-2
    assert out["feature_dim"] == 8
    assert out["n_samples"] == 16
    assert out["cache_path"].endswith("toy_train.pt")


def test_cknna_dinov2_load_raises_without_network(monkeypatch):
    """load_dinov2_reference 在无 transformers/无权重时抛可被上层 skip 的异常（D11 严禁改教师参照）。"""
    pytest.importorskip("torch")
    import sys
    # 模拟 transformers 缺失（lazy import 路径）
    monkeypatch.setitem(sys.modules, "transformers", None)
    from eval.semantic.cknna import load_dinov2_reference
    with pytest.raises((ImportError, OSError)):
        load_dinov2_reference()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
