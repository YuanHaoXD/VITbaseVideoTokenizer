"""E-3 校准线验收测试：eval/calibrate.py。

契约断言（03 篇 §4 五锚点校准流程 + 02 篇 §4 锚点表 + README D9 排序复现）：
- ANCHORS 冻结表为五家（OmniTokenizer/LARP/LeanVAE/AToken/Wan2.2），期望参照值键名对齐 recon_metrics。
- calibrate_track：跑我们的 protocols + recon_metrics 在 (encode,decode) 上，返回指标 dict。
- run_calibration：产出《校准报告》markdown，含"我们尺子"/"排序一致性"/"官方脚本复现"三节。
- 排序一致性主判据（README D9）：_rank_consistency 对 toy 序列正确判一致/不一致。

纪律：Windows 只做 py_compile；需 torch 的测试 importorskip 并注明集群命令：
  集群：cd uvt && python -m pytest tests/test_calibrate.py -v
"""
from pathlib import Path

import pytest

CALIBRATE_SRC = (Path(__file__).resolve().parent.parent / "eval" / "calibrate.py").read_text("utf-8")


# ======================================================================================
# 离线契约测试（ast 级——无需 torch）
# ======================================================================================
def test_anchors_table_five_entries_via_ast():
    """ANCHORS 冻结表必须有五家锚点（02 篇 §4 表）。ast 检查避免 import torch。"""
    import ast
    tree = ast.parse(CALIBRATE_SRC)
    # 找 CalibrationAnchor 调用的 name= 字面量
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            fn_name = func.id if isinstance(func, ast.Name) else None
            if fn_name == "CalibrationAnchor":
                for kw in node.keywords:
                    if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                        names.append(kw.value.value)
    assert names == ["OmniTokenizer", "LARP", "LeanVAE", "AToken", "Wan2.2"], \
        f"五锚点顺序/名称不符 02 篇 §4 表：{names}"


def test_calibrate_signature_freeze_via_ast():
    """核心接口签名冻结（任务书契约）：calibrate_track / run_calibration 形参稳定。"""
    import ast
    tree = ast.parse(CALIBRATE_SRC)
    sigs = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name in ("calibrate_track", "run_calibration"):
            sigs[n.name] = ([a.arg for a in n.args.args],
                            [a.arg for a in getattr(n.args, "kwonlyargs", [])])
    ct_args, ct_kw = sigs["calibrate_track"]
    assert "anchor" in ct_args and "encode_decode" in ct_args and "real_iter" in ct_args
    assert {"is_video", "device"} <= set(ct_kw), \
        f"calibrate_track kwonly 须含 is_video/device，实际: {ct_kw}"
    rc_args, rc_kw = sigs["run_calibration"]
    assert {"anchors", "tracks", "load_fn", "output_md"} <= set(rc_args)
    assert "official_repro_hook" in rc_kw, "run_calibration 必须留 official_repro_hook 接口（流程①）"


def test_expected_metric_keys_align_with_recon_metrics():
    """anchor.expected 的键名必须落在 recon_metrics.compute() 输出键集内（排序比对前提）。"""
    import ast
    # recon_metrics.compute 输出键：psnr/ssim/lpips/rfid/rfvd_styleganv/rfvd_videogpt
    valid = {"psnr", "ssim", "lpips", "rfid", "rfvd_styleganv", "rfvd_videogpt"}
    tree = ast.parse(CALIBRATE_SRC)
    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "CalibrationAnchor":
            for kw in node.keywords:
                if kw.arg == "expected" and isinstance(kw.value, ast.Dict):
                    for k in kw.value.keys:
                        if isinstance(k, ast.Constant):
                            used.add(k.value)
    # 允许带 _davis / _vae 后缀的变体键（02 §4 表里区分数据集/版本），核心 metric 须合法
    for k in used:
        base = k.replace("_davis", "").replace("_vae", "").replace("_imagenet", "")
        assert base in valid, f"anchor.expected 键 {k!r} 不在 recon_metrics 输出键集 {valid}"


# ======================================================================================
# 排序一致性逻辑（纯函数，但模块 import torch——importorskip 后测）
# ======================================================================================
def test_rank_consistency_toy():
    """_rank_consistency：(track,metric) 下两排序同序→consistent=True；反序→False。"""
    torch = pytest.importorskip("torch")
    from eval.calibrate import _rank_consistency

    # ours 与 expected 同序（rfid 越小越好）
    records_same = [
        {"name": "A", "track": "imagenet", "ours": {"rfid": 1.0}, "expected": {"rfid": 1.0}},
        {"name": "B", "track": "imagenet", "ours": {"rfid": 2.0}, "expected": {"rfid": 2.0}},
    ]
    out = _rank_consistency(records_same)
    assert len(out) == 1 and out[0]["consistent"] is True

    # 反序 → 不一致
    records_inv = [
        {"name": "A", "track": "imagenet", "ours": {"rfid": 2.0}, "expected": {"rfid": 1.0}},
        {"name": "B", "track": "imagenet", "ours": {"rfid": 1.0}, "expected": {"rfid": 2.0}},
    ]
    out = _rank_consistency(records_inv)
    assert out[0]["consistent"] is False


def test_rank_consistency_higher_better_psnr():
    """PSNR 越大越好：方向须正确反转（同序仍判一致）。"""
    torch = pytest.importorskip("torch")
    from eval.calibrate import _rank_consistency

    records = [
        {"name": "A", "track": "davis", "ours": {"psnr": 30.0}, "expected": {"psnr": 30.0}},
        {"name": "B", "track": "davis", "ours": {"psnr": 28.0}, "expected": {"psnr": 28.0}},
    ]
    out = _rank_consistency(records)
    assert out[0]["consistent"] is True and out[0]["metric"] == "psnr"


# ======================================================================================
# calibrate_track / run_calibration（mock encode/decode + mock real_iter）
# ======================================================================================
def test_calibrate_track_identity_mock():
    """mock encode/decode（恒等，值域[-1,1]）→ psnr 极高 / ssim≈1。不开 LPIPS/Frechet 避重依赖。"""
    torch = pytest.importorskip("torch")
    from eval.calibrate import CalibrationAnchor, calibrate_track

    def encode(x):
        return x                     # mock：z=x
    def decode(z, hw):
        return z                     # mock：x_hat=z（恒等重建）

    anchor = CalibrationAnchor("mock", "mock", "mock", {"psnr": 999.0})
    real_iter = [torch.rand(2, 3, 256, 256) * 2 - 1]   # [-1,1] 图像 batch（已过 protocols）
    out = calibrate_track(anchor, (encode, decode), real_iter,
                          is_video=False, with_lpips=False, with_frechet=False)
    assert "psnr" in out and "ssim" in out
    assert out["psnr"] > 80.0, f"恒等重建 PSNR 应极高，得到 {out['psnr']}"
    assert out["ssim"] > 0.999


def test_calibrate_track_shape_mismatch_raises():
    """重建形状 ≠ 真值形状须报错（防锚点 decode 输出未配对）。"""
    torch = pytest.importorskip("torch")
    from eval.calibrate import CalibrationAnchor, calibrate_track

    def encode(x):
        return x
    def decode(z, hw):
        return z[..., :128, :]      # 故意错尺寸

    anchor = CalibrationAnchor("bad", "x", "x")
    real_iter = [torch.rand(1, 3, 256, 256) * 2 - 1]
    with pytest.raises(ValueError, match="重建形状"):
        calibrate_track(anchor, (encode, decode), real_iter,
                        is_video=False, with_lpips=False, with_frechet=False)


def test_run_calibration_writes_report(tmp_path):
    """run_calibration：mock load_fn → 跑两锚点×一 track → markdown 含三节标题。"""
    torch = pytest.importorskip("torch")
    from eval.calibrate import CalibrationAnchor, run_calibration

    anchors = [
        CalibrationAnchor("A", "a", "a", {"rfid": 1.0}),
        CalibrationAnchor("B", "b", "b", {"rfid": 2.0}),
    ]
    real_iter = [torch.rand(2, 3, 256, 256) * 2 - 1]
    tracks = {"imagenet": (False, real_iter)}

    def load_fn(anchor, track):
        def enc(x):
            return x
        def dec(z, hw):
            return z
        return (enc, dec)

    out_md = tmp_path / "report.md"
    records = run_calibration(anchors, tracks, load_fn, str(out_md),
                              with_lpips=False, with_frechet=False)   # 关重指标：本测验证报告结构非 FID 值
    assert len(records) == 2
    text = out_md.read_text(encoding="utf-8")
    assert "# UVT 校准报告（P0）" in text
    assert "我们尺子下的数字" in text          # §1
    assert "排序一致性" in text                 # §2
    assert "官方脚本复现" in text               # §3
    assert "待 P0-golden 首日人工补" in text    # official_repro_hook 未提供的标注


def test_run_calibration_with_official_hook(tmp_path):
    """official_repro_hook 提供时，报告 §3 填表而非标"待补"。"""
    torch = pytest.importorskip("torch")
    from eval.calibrate import CalibrationAnchor, run_calibration

    anchors = [CalibrationAnchor("A", "a", "a", {"rfid": 1.0})]
    tracks = {"imagenet": (False, [torch.rand(1, 3, 256, 256) * 2 - 1])}

    def load_fn(anchor, track):
        return (lambda x: x, lambda z, hw: z)
    def hook(anchor):
        return {"rfid_official": 1.05}

    out_md = tmp_path / "r.md"
    run_calibration(anchors, tracks, load_fn, str(out_md),
                    official_repro_hook=hook, with_lpips=False, with_frechet=False)
    text = out_md.read_text(encoding="utf-8")
    assert "rfid_official=1.050" in text        # hook 数字进了表


def test_load_fn_none_skips_track(tmp_path):
    """load_fn 返回 None（锚点不支持该 track）→ 跳过，records 不含该锚点。"""
    torch = pytest.importorskip("torch")
    from eval.calibrate import CalibrationAnchor, run_calibration

    anchors = [CalibrationAnchor("A", "a", "a")]
    tracks = {"imagenet": (False, [torch.rand(1, 3, 256, 256) * 2 - 1])}
    records = run_calibration(anchors, tracks,
                              lambda a, t: None,   # 全不支持
                              str(tmp_path / "r.md"), with_lpips=False)
    assert records == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
