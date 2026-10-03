"""Bounded CPU-only tests for the isolated, non-authorizing parity diagnostic."""
from __future__ import annotations

import ast
import copy
import importlib.util
import inspect
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/diagnose_relational_part_sporc_parity.py"
_SPEC = importlib.util.spec_from_file_location("sporc_parity_diagnostic_under_test", _SCRIPT)
D = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(D)


def test_difference_reports_absolute_relative_and_summary_errors():
    reference = np.array([0., 100., -2.])
    actual = np.array([2e-5, 100.001, -1.9])
    result = D.difference(actual, reference, atol=5e-5, rtol=5e-5)
    delta = np.abs(actual - reference)
    assert result["passed"] is False
    assert result["shape_matches"] is True and result["finite"] is True
    assert result["elements"] == 3
    assert result["outside_tolerance_count"] == 1
    assert result["maximum_absolute_error"] == pytest.approx(delta.max())
    assert result["mean_absolute_error"] == pytest.approx(delta.mean())
    assert result["root_mean_square_error"] == pytest.approx(np.sqrt(np.mean(delta**2)))
    assert D.difference(actual[:2], reference[:2], atol=5e-5, rtol=5e-5)["passed"] is True


def test_difference_does_not_broadcast_shape_mismatches():
    result = D.difference(np.zeros((1, 10)), np.zeros((2, 10)), atol=0, rtol=0)
    assert result == {"passed": False, "shape_matches": False,
                      "actual_shape": [1, 10], "reference_shape": [2, 10]}


@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_difference_fails_closed_for_nonfinite_arrays(side, invalid):
    values = [np.ones(3), np.ones(3)]
    values[side][1] = invalid
    result = D.difference(*values, atol=1e-5, rtol=1e-5)
    assert result["passed"] is False and result["finite"] is False
    assert "maximum_absolute_error" not in result


def test_difference_empty_pair_selection_has_zero_error():
    result = D.difference(np.array([]), np.array([]), atol=0, rtol=0)
    assert result["passed"] is True and result["elements"] == 0
    assert result["maximum_absolute_error"] == result["mean_absolute_error"] == 0


def test_logit_difference_reports_batch_boundaries_and_class_changes():
    reference = np.zeros((130, 10), dtype=np.float32)
    reference[:, 0] = 1
    actual = reference.copy()
    actual[65, 1] = 2
    actual[129, 0] += 1e-6
    result = D.logit_difference(actual, reference, atol=5e-5, rtol=5e-5)
    assert result["argmax_disagreements"] == 1
    assert result["outside_tolerance_count"] == 1
    assert [(b["entry_start"], b["entry_stop"]) for b in result["per_batch"]] == [(0, 64), (64, 128), (128, 130)]
    assert [b["passed"] for b in result["per_batch"]] == [True, False, True]
    assert result["per_batch"][1]["maximum_absolute_error"] == 2


def test_logit_difference_shape_or_nonfinite_results_do_not_attempt_argmax():
    result = D.logit_difference(np.zeros((64, 10)), np.zeros((32, 10)), atol=0, rtol=0)
    assert result["passed"] is False and "argmax_disagreements" not in result
    actual = np.zeros((64, 10))
    actual[0, 0] = np.nan
    result = D.logit_difference(actual, np.zeros_like(actual), atol=0, rtol=0)
    assert result["finite"] is False and "per_batch" not in result


def test_logit_difference_accepts_array_like_inputs():
    result = D.logit_difference([[1., 0.]], [[1., 0.]], atol=0, rtol=0)
    assert result["passed"] is True and result["argmax_disagreements"] == 0


@pytest.mark.parametrize("shape", [(10,), (2, 3, 10)])
def test_logit_difference_rejects_nonmatrix_logits(shape):
    with pytest.raises(ValueError):
        D.logit_difference(np.zeros(shape), np.zeros(shape), atol=0, rtol=0)


@pytest.mark.parametrize("mask_has_channel", [False, True])
def test_pair_difference_separates_valid_diagonal_and_off_diagonal(mask_has_channel):
    reference = np.zeros((2, 4, 3, 3), dtype=np.float32)
    actual = reference.copy()
    valid = np.array([[True, True, False], [True, False, False]])
    actual[:, :, 2, 2] = np.nan  # Masked positions are intentionally excluded.
    actual[0, 3, 0, 0] = 0.5
    actual[0, 3, 0, 1] = 0.25
    mask = valid[:, None, :] if mask_has_channel else valid
    result = D.pair_difference(actual, reference, mask, atol=1e-5, rtol=1e-5)
    assert set(result) == {"lnkt", "lnz", "lndelta", "lnm2"}
    for channel in result.values():
        assert channel["valid_diagonal"]["elements"] == 3
        assert channel["valid_off_diagonal"]["elements"] == 2
        assert all(group["finite"] for group in channel.values())
    assert result["lnm2"]["valid_diagonal"]["maximum_absolute_error"] == 0.5
    assert result["lnm2"]["valid_off_diagonal"]["maximum_absolute_error"] == 0.25
    assert result["lnm2"]["valid_diagonal"]["outside_tolerance_count"] == 1
    assert result["lnm2"]["valid_off_diagonal"]["outside_tolerance_count"] == 1
    assert all(result[name]["valid_diagonal"]["passed"] for name in ("lnkt", "lnz", "lndelta"))


def test_pair_difference_reports_nonfinite_valid_pairs():
    reference = np.zeros((1, 4, 2, 2))
    actual = reference.copy()
    actual[0, 0, 0, 0] = np.inf
    result = D.pair_difference(actual, reference, np.ones((1, 2), dtype=bool), atol=0, rtol=0)
    assert result["lnkt"]["valid_diagonal"]["passed"] is False
    assert result["lnkt"]["valid_diagonal"]["finite"] is False
    assert result["lnkt"]["valid_off_diagonal"]["passed"] is True


@pytest.mark.parametrize("actual_shape,reference_shape,mask_shape", [
    ((1, 4, 3, 3), (1, 4, 2, 2), (1, 3)),
    ((1, 3, 3, 3), (1, 3, 3, 3), (1, 3)),
    ((1, 4, 3, 2), (1, 4, 3, 2), (1, 3)),
    ((1, 4, 3, 3), (1, 4, 3, 3), (1, 2, 3)),
    ((1, 4, 3, 3), (1, 4, 3, 3), (2, 3)),
    ((1, 4, 3, 3), (1, 4, 3, 3), (3,)),
])
def test_pair_difference_rejects_misaligned_shapes(actual_shape, reference_shape, mask_shape):
    with pytest.raises(ValueError):
        D.pair_difference(np.zeros(actual_shape), np.zeros(reference_shape),
                          np.ones(mask_shape, dtype=bool), atol=0, rtol=0)


def _fake_math_backend(monkeypatch):
    flags = {"flash": True, "mem_efficient": True, "math": False, "cudnn": True, "fastpath": True}
    cuda = SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True))
    for name in ("flash", "mem_efficient", "math", "cudnn"):
        setattr(cuda, name + "_sdp_enabled", lambda name=name: flags[name])
        setattr(cuda, "enable_" + name + "_sdp", lambda value, name=name: flags.__setitem__(name, value))
    mha = SimpleNamespace(get_fastpath_enabled=lambda: flags["fastpath"],
                          set_fastpath_enabled=lambda value: flags.__setitem__("fastpath", value))
    fake = SimpleNamespace(backends=SimpleNamespace(cuda=cuda, cudnn=SimpleNamespace(allow_tf32=True), mha=mha))
    monkeypatch.setitem(sys.modules, "torch", fake)
    return fake, flags


@pytest.mark.parametrize("tf32_disabled", [False, True])
@pytest.mark.parametrize("math_attention", [False, True])
@pytest.mark.parametrize("raise_inside", [False, True])
def test_diagnostic_math_restores_every_flag_even_on_failure(monkeypatch, tf32_disabled, math_attention, raise_inside):
    fake, flags = _fake_math_backend(monkeypatch)
    before = dict(flags)
    try:
        with D.diagnostic_math(tf32_disabled=tf32_disabled, math_attention=math_attention):
            assert fake.backends.cuda.matmul.allow_tf32 is (not tf32_disabled)
            assert fake.backends.cudnn.allow_tf32 is (not tf32_disabled)
            assert flags == ({"flash": False, "mem_efficient": False, "math": True,
                              "cudnn": False, "fastpath": False} if math_attention else before)
            if raise_inside:
                raise RuntimeError("probe failure")
    except RuntimeError as error:
        assert raise_inside and str(error) == "probe failure"
    assert flags == before
    assert fake.backends.cuda.matmul.allow_tf32 is True
    assert fake.backends.cudnn.allow_tf32 is True


def test_diagnostic_math_handles_missing_optional_attention_controls(monkeypatch):
    fake = SimpleNamespace(backends=SimpleNamespace(
        cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
        cudnn=SimpleNamespace(allow_tf32=True)))
    monkeypatch.setitem(sys.modules, "torch", fake)
    with D.diagnostic_math(tf32_disabled=True, math_attention=True):
        assert fake.backends.cudnn.allow_tf32 is False
    assert fake.backends.cuda.matmul.allow_tf32 is False
    assert fake.backends.cudnn.allow_tf32 is True


def test_baseline_forward_resets_once_and_replays_cloned_inputs_with_cpu_pairs():
    import torch

    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.counter = 99
            self.seen_counters = []

        def forward(self, features, *, v, mask, uu):
            assert not torch.is_grad_enabled()
            assert not self.training
            torch.testing.assert_close(uu, v.unsqueeze(-1) + v.unsqueeze(-2))
            self.seen_counters.append(self.counter)
            result = (features.sum((1, 2)) + uu.sum((1, 2, 3)) + self.counter)[:, None].expand(-1, 10).clone()
            self.counter += 1
            # Model-side mutation must not contaminate later variants or the
            # serialized diagnostic input snapshot.
            features.fill_(-999)
            v.fill_(0)
            mask.fill_(False)
            uu.fill_(999)
            return result

    class Baseline(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mod = Network()
            self.pair_devices = []

        def explicit_standard_four(self, vectors, mask):
            assert vectors.device == mask.device
            self.pair_devices.append(vectors.device.type)
            return vectors.unsqueeze(-1) + vectors.unsqueeze(-2)

    model = Baseline().eval()
    batches = [{"features": np.full((2, 17, 3), i, dtype=np.float32),
                "points": np.zeros((2, 2, 3), dtype=np.float32),
                "lorentz_vectors": np.full((2, 4, 3), i, dtype=np.float32),
                "mask": np.ones((2, 1, 3), dtype=bool)} for i in (1, 2)]
    before = copy.deepcopy(batches)
    resets = []

    def reset(models):
        assert models == [model]
        resets.append(True)
        model.mod.counter = 0

    direct = D.baseline_forward(model, batches, torch.device("cpu"), reset)
    cpu_pairs = D.baseline_forward(model, batches, torch.device("cpu"), reset, cpu_pairs=True)
    np.testing.assert_array_equal(direct, cpu_pairs)
    np.testing.assert_array_equal(direct[:, 0], [123, 123, 247, 247])
    assert direct.shape == (4, 10) and direct.dtype == np.float32
    assert len(resets) == 2
    assert model.mod.seen_counters == [0, 1, 0, 1]
    assert model.pair_devices == ["cpu"] * 4
    for original, unchanged in zip(before, batches):
        for key in original:
            np.testing.assert_array_equal(original[key], unchanged[key])


def test_main_only_publishes_diagnostic_artifacts_and_never_authorizes_inference():
    tree = ast.parse(inspect.getsource(D.main))
    writes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {"atomic_file", "write_json"}:
            target = node.args[0]
            assert isinstance(target, ast.BinOp) and isinstance(target.op, ast.Div)
            assert isinstance(target.left, ast.Name) and target.left.id == "output"
            assert isinstance(target.right, ast.Constant)
            writes.append(target.right.value)
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value in {"authorizes_inference", "production_settings_changed", "used_for_model_selection"}:
                    assert isinstance(value, ast.Constant) and value.value is False
    assert sorted(writes) == ["diagnostic_logits.npz", "diagnostic_report.json", "first_batch_inputs_and_pairs.npz"]
    assert "portability_validation.json" not in inspect.getsource(D.main)


def test_cli_help_needs_no_frozen_runtime_or_gpu():
    result = subprocess.run([sys.executable, str(_SCRIPT), "--help"], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "--campaign" in result.stdout and "--output" in result.stdout
