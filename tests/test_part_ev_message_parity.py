"""Exercise the benchmark fix, including failures that must NOT be warmed away."""
import copy
import random
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from part_ev_message_study import parity as Q
from scripts import run_part_ev_message_study as M
from tests.test_part_fullscale_model import bounded_cpu_threads
from tests.test_part_ev_screen_training import batch
from tests.test_part_ev_message_protocol import parity_fixture


def setup():
    task = M.P.TASKS[0]
    return task, M.D.make_model(task, torch.device("cpu")), batch()


def assert_rng_equal(before, after):
    assert Q.same_rng(before, after)
    assert before["python"] == after["python"]
    assert before["numpy"][0] == after["numpy"][0]
    np.testing.assert_array_equal(before["numpy"][1], after["numpy"][1])
    assert before["numpy"][2:] == after["numpy"][2:]
    assert before["trimmers"] == after["trimmers"]


def test_fixed_warmup_records_transient_error_then_checks_both_orders(monkeypatch):
    task, model, sample = setup()
    native = model.forward
    calls, sequence = [], []
    reference_factory = M.reference_model

    def new_forward(*values):
        calls.append(True)
        sequence.append("new")
        logits = native(*values)
        # A simulated first-call-only perturbation, NOT a claim about the GPU cause.
        return logits + .001 if len(calls) == 1 else logits

    def factory(*args, **kwargs):
        ref = reference_factory(*args, **kwargs)
        forward = ref.forward
        def tracked(*values):
            sequence.append("reference")
            return forward(*values)
        ref.forward = tracked
        return ref

    monkeypatch.setattr(model, "forward", new_forward)
    monkeypatch.setattr(M, "reference_model", factory)
    before, rng = copy.deepcopy(model.state_dict()), M.D.capture_rank_state(model)
    report = M.initial_parity(model, task, sample, torch.device("cpu"))
    assert report["passed"] and report["policy"] == Q.POLICY
    assert not report["modes"]["eval_fp32"]["warmup"][0]["comparison"]["passed"]
    assert sequence == ["reference", "new", "reference", "new", "reference", "new", "new", "reference"] * 3
    for mode in Q.POLICY["modes"]:
        row = report["modes"][mode]
        assert [r["order"] for r in row["measured"]] == Q.POLICY["measured_orders"]
        assert all(r["comparison"]["exact"] and r["reference_repeat"]["exact"]
                   and r["new_repeat"]["exact"] and r["rng_matches"] for r in row["measured"])
    assert len(calls) == 12  # Bounded, not retry-until-it-passes.
    assert_rng_equal(rng, M.D.capture_rank_state(model))
    assert all(torch.equal(t, model.state_dict()[k]) for k, t in before.items())


@pytest.mark.parametrize("kind", ["persistent", "measured_only", "reverse_order", "nonfinite", "rng", "exception"])
def test_bad_probe_fails_and_restores_state(monkeypatch, kind):
    task, model, sample = setup()
    # Restore heterogeneous training flags, trimmer counters and branch switches, too.
    model.mod.blocks[0].eval()
    model.mod.trimmer._counter = 9
    model.mod.blocks[0].attn.disable_new = True
    before = copy.deepcopy(model.state_dict())
    flags = [m.training for m in model.modules()]
    rng = M.D.capture_rank_state(model)
    calls = []
    native = model.forward

    def broken(*values):
        calls.append(True)
        assert not model.mod.blocks[0].attn.disable_new
        logits = native(*values)
        if kind == "exception":
            random.random()
            np.random.random()
            torch.rand(1)
            raise RuntimeError("fixture failure")
        if kind == "rng":
            torch.rand(1)
        if kind == "nonfinite":
            return logits * float("nan")
        if kind == "persistent" or kind == "measured_only" and len(calls) == 3 or kind == "reverse_order" and len(calls) == 4:
            return logits + .001
        return logits

    monkeypatch.setattr(model, "forward", broken)
    with pytest.raises(Q.ParityError) as error:
        M.initial_parity(model, task, sample, torch.device("cpu"))
    report = error.value.report
    assert report["passed"] is False and report["optimizer_steps"] == 0
    assert "error" in report
    if kind == "measured_only":
        assert "reference_first comparison" in str(error.value)
    if kind == "reverse_order":
        assert "new_first comparison" in str(error.value)
    assert len(calls) <= 4
    assert all(torch.equal(t, model.state_dict()[k]) for k, t in before.items())
    assert flags == [m.training for m in model.modules()]
    assert model.mod.trimmer.enabled and model.mod.blocks[0].attn.disable_new
    assert_rng_equal(rng, M.D.capture_rank_state(model))


def test_backbone_mismatch_is_not_repaired_or_ignored():
    task, model, sample = setup()
    with torch.no_grad():
        next(model.parameters()).add_(.01)
    before = copy.deepcopy(model.state_dict())
    rng = M.D.capture_rank_state(model)
    with pytest.raises(Q.ParityError, match="Changed initial EV/backbone") as error:
        M.initial_parity(model, task, sample, torch.device("cpu"))
    assert error.value.report["modes"] == {}
    assert all(torch.equal(t, model.state_dict()[k]) for k, t in before.items())
    assert_rng_equal(rng, M.D.capture_rank_state(model))


def test_nonzero_new_branch_still_fails_strict_comparison():
    task, model, sample = setup()
    with torch.no_grad():
        model.mod.blocks[0].attn.state.output.fill_(.01)
    with pytest.raises(Q.ParityError, match="eval_fp32 reference_first comparison"):
        M.initial_parity(model, task, sample, torch.device("cpu"))


def test_repeatability_is_checked_even_if_models_match_each_other(monkeypatch):
    task, model, sample = setup()
    original = Q.forward_once
    counts = {}
    def drifting(engine, target, *args, **kwargs):
        logits, rng = original(engine, target, *args, **kwargs)
        counts[id(target)] = counts.get(id(target), 0) + 1
        if counts[id(target)] == 4:
            logits += .001  # Same drift in both models: cross-model parity alone misses it.
        return logits, rng
    monkeypatch.setattr(Q, "forward_once", drifting)
    with pytest.raises(Q.ParityError, match="new_first reference_repeat"):
        M.initial_parity(model, task, sample, torch.device("cpu"))


def test_fp16_tolerance_stays_strict_and_cuda_rng_lengths_are_checked():
    x = torch.ones(2, 10, dtype=torch.float16)
    y = x.clone()
    y[0, 0] += 2**-10
    assert not Q.compare(y, x)["passed"]
    assert Q.POLICY["rtol"] == 1e-5 and Q.POLICY["atol"] == 1e-6
    assert not Q.same_rng(dict(torch=x, cuda=[]), dict(torch=x, cuda=[x]))


def test_half_forward_path_with_cpu_emulation():
    _, model, sample = setup()
    @contextmanager
    def precision(_model, _device, *, training):
        with torch.autocast("cpu", dtype=torch.float16, enabled=training):
            yield
    engine = SimpleNamespace(capture_rank_state=M.D.capture_rank_state, restore_rank_state=M.D.restore_rank_state,
                             seed_everything=M.D.seed_everything, model_precision=precision)
    values, _ = M.D.tensors(sample, torch.device("cpu"))
    # Only emulate the dtype control locally; this is not a CUDA runtime test.
    logits, _ = Q.forward_once(engine, model, copy.deepcopy(model.state_dict()), values,
                              torch.device("cuda"), mode="train_fp16", seed=872)
    assert logits.dtype == torch.float16


def test_admission_retains_finite_warmup_mismatch_but_rejects_missing_measured_evidence():
    report = parity_fixture()
    report["modes"]["train_fp16"]["warmup"][0]["comparison"] = dict(
        passed=False, exact=False, maximum_absolute_error=.0009765625)
    Q.validate_report(report)
    del report["modes"]["train_fp16"]["measured"][0]["new_repeat"]
    with pytest.raises(ValueError):
        Q.validate_report(report)
