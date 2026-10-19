import copy
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from scripts import diagnose_part_ev_message_parity as Q
from scripts import run_part_ev_message_study as M
from tests.test_part_fullscale_model import inputs, bounded_cpu_threads
from tests.test_part_ev_screen_training import batch


def models():
    M.D.seed_everything(101)
    ref = M.reference_model("part_ev", trim=False)
    M.D.seed_everything(101)
    model = M.build_model("ev_state", trim=True)
    snapshots = dict(reference=copy.deepcopy(ref.state_dict()), new=copy.deepcopy(model.state_dict()))
    return ref, model, snapshots


def test_original_tolerance_and_half_dtype_remain_strict():
    expected = torch.ones(2, 10, dtype=torch.float16)
    actual = expected.clone()
    actual[0, 0] += 2**-10
    row = Q.compare(actual, expected)
    assert not row["passed"] and row["outside_tolerance_count"] == 1
    assert row["maximum_absolute_error"] == 0.0009765625
    assert row["actual_dtype"] == "torch.float16"
    assert Q.compare(expected, expected)["exact"]
    assert not Q.compare(expected.float(), expected)["passed"]
    assert not Q.compare(torch.full_like(expected, torch.nan), expected)["finite"]
    assert not Q.compare(expected[:1], expected)["shape_matches"]


def test_trace_hooks_and_functional_patch_are_restored_on_error():
    ref, model, _ = models()
    linear = F.linear
    counts = [(m, len(m._forward_hooks), len(m._forward_pre_hooks)) for m in model.modules()]
    with pytest.raises(RuntimeError, match="fixture"):
        with Q.capture_trace(model):
            assert F.linear is not linear
            raise RuntimeError("fixture")
    assert F.linear is linear
    assert all(len(m._forward_hooks) == post and len(m._forward_pre_hooks) == pre for m, post, pre in counts)


@pytest.mark.parametrize("mode", ["eval_fp32", "train_fp32", "train_fp16"])
def test_modes_trace_zero_branch_and_preserve_all_state(mode):
    ref, model, snapshots = models()
    before = M.D.capture_rank_state(model)
    # CPU dtype-path test only, not a substitute for the actual CUDA diagnostic.
    @contextmanager
    def precision(_model, device, *, training):
        with torch.autocast(device.type, dtype=torch.float16, enabled=training):
            yield
    engine = SimpleNamespace(capture_rank_state=M.D.capture_rank_state, restore_rank_state=M.D.restore_rank_state,
                             seed_everything=M.D.seed_everything, model_precision=precision)
    row, logits = Q.analyze_mode(engine, model, ref, snapshots, inputs(batch=2, particles=7),
                                torch.device("cpu"), mode, 872)
    for key in ("reference_repeat", "new_repeat", "enabled_vs_reference", "disabled_vs_reference",
                "enabled_vs_disabled", "reference_instrumentation_effect",
                "enabled_instrumentation_effect", "disabled_instrumentation_effect"):
        assert row[key]["exact"], key
    assert row["enabled_trace"]["first_outside_tolerance"] is None
    assert row["rng_enabled_vs_reference"]["torch_cpu_exact"]
    assert all(v["nonzero_elements"] == 0 for v in row["branch_evidence"].values())
    assert len(row["branch_evidence"]) == 8
    expected_dtype = torch.float16 if mode == "train_fp16" else torch.float32
    assert logits["enabled"].dtype == expected_dtype
    assert torch.equal(M.D.capture_rank_state(model)["torch"], before["torch"])
    assert M.D.capture_rank_state(model)["trimmers"] == before["trimmers"]
    assert model.training and model.mod.trimmer.enabled
    assert all(not block.attn.disable_new for block in model.mod.blocks)
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, snapshots["new"][name])


def test_first_divergence_is_localized_not_just_final_logits():
    ref, model, snapshots = models()
    with torch.no_grad():
        model.mod.blocks[2].attn.edge_projection.mul_(1.1)
    snapshots["new"] = copy.deepcopy(model.state_dict())
    row, _ = Q.analyze_mode(M.D, model, ref, snapshots, inputs(batch=2, particles=7),
                           torch.device("cpu"), "eval_fp32", 872)
    assert row["enabled_trace"]["first_outside_tolerance"] == "particle.2.ev_projection_input"
    assert row["disabled_trace"]["first_outside_tolerance"] == "particle.2.ev_projection_input"
    assert not row["enabled_vs_reference"]["passed"]
    assert row["enabled_vs_disabled"]["exact"]


def test_nonzero_extension_is_reported_and_disabling_is_informative():
    ref, model, snapshots = models()
    with torch.no_grad():
        model.mod.blocks[0].attn.state.output.fill_(.01)
    snapshots["new"] = copy.deepcopy(model.state_dict())
    row, _ = Q.analyze_mode(M.D, model, ref, snapshots, inputs(batch=2, particles=7),
                           torch.device("cpu"), "eval_fp32", 872)
    assert row["branch_evidence"]["particle.0.extension.state"]["nonzero_elements"] > 0
    assert not row["enabled_vs_reference"]["passed"]
    assert row["disabled_vs_reference"]["exact"]


def test_forward_failure_restores_flags_buffers_rng_and_hooks(monkeypatch):
    _, model, snapshots = models()
    rng = M.D.capture_rank_state(model)
    def broken(*_):
        torch.rand(10)
        raise RuntimeError("broken forward")
    monkeypatch.setattr(model, "forward", broken)
    linear = F.linear
    with pytest.raises(RuntimeError, match="broken forward"):
        Q.forward_once(M.D, model, snapshots["new"], inputs(), torch.device("cpu"),
                       mode="eval_fp32", seed=872, disabled=True, trace=True)
    assert F.linear is linear and model.training and model.mod.trimmer.enabled
    assert all(not b.attn.disable_new for b in model.mod.blocks)
    assert torch.equal(rng["torch"], M.D.capture_rank_state(model)["torch"])


def test_plan_binding_detects_campaign_and_diagnostic_tampering(tmp_path, monkeypatch):
    campaign = tmp_path / "campaign"
    output = campaign / "diagnostics/parity.fixture"
    output.mkdir(parents=True)
    for name in ("campaign.json", "launch.env", "launch.json", "launch.sha256", "source.sha256"):
        (campaign / name).write_text("{}\n")
    (output / "launch.json").write_text("{}\n")
    (output / "launch.env").write_text("# immutable\n")
    spec = dict(launch={}, content_hash="fixture", tasks=M.P.TASKS)
    fake = SimpleNamespace(P=SimpleNamespace(load_campaign=lambda _: spec, task_directory=M.P.task_directory))
    monkeypatch.setattr(Q, "frozen_driver", lambda _: fake)
    Q.prepare(campaign, output, 0)
    plan = Q.load_plan(output)
    assert plan["policy"]["authorizes_training"] is False
    (campaign / "campaign.json").write_text('{"changed":true}\n')
    with pytest.raises(ValueError, match="artifact changed"):
        Q.load_plan(output)
    (campaign / "campaign.json").write_text("{}\n")
    (output / "launch.env").write_text("# modified\n")
    with pytest.raises(ValueError, match="resources changed"):
        Q.load_plan(output)


def test_runner_collects_original_failure_without_authorizing_or_training(tmp_path, monkeypatch):
    output = tmp_path / "diagnostic"
    output.mkdir()
    donor = tmp_path / "donor"
    (donor / "benchmarks").mkdir(parents=True)
    sample = batch()
    task = M.P.TASKS[0]
    spec = dict(content_hash="fixture", tasks=M.P.TASKS, inventory={}, donor=dict(path=str(donor)))
    Q.create_json(donor / "benchmarks/EV_RANGER.json", dict(input_fingerprints={"0": M.D.batch_fingerprint(sample)}))
    plan = dict(campaign=str(tmp_path), task=task, task_index=0, campaign_sha256="fixture", content_hash="plan")
    monkeypatch.setattr(Q, "load_plan", lambda _: plan)
    monkeypatch.setattr(Q, "verify_artifacts", lambda _: None)
    engine = M.create_engine()
    monkeypatch.setattr(engine, "build_loader", lambda *a, **kw: ([sample], None))
    stopped = []
    monkeypatch.setattr(engine, "shutdown_loader", lambda _: stopped.append(True))
    monkeypatch.setattr(engine, "training_step", lambda *a, **kw: pytest.fail("No training allowed"))
    def failed_probe(*args):
        raise AssertionError("original parity mismatch")
    fake = SimpleNamespace(D=engine, P=SimpleNamespace(load_campaign=lambda _: spec, verify_inventory_metadata=lambda _: None),
                           gpu_setup=lambda _: (torch.device("cpu"), {}), reference_model=M.reference_model,
                           initial_parity=failed_probe)
    monkeypatch.setattr(Q, "frozen_driver", lambda _: fake)
    Q.run_diagnostic(output)
    result = Q.read(output / "parity_report.json")
    assert result["completed"] and not result["authorizes_training"]
    assert result["optimizer_steps"] == 0 and not result["original_parity_probe"]["passed"]
    assert set(result["modes"]) == set(Q.POLICY["modes"])
    assert result["logits_sha256"] == Q.sha(output / "logits.npz")
    assert stopped == [True]
    assert not (tmp_path / "production_approval.json").exists()


def test_report_explicitly_distinguishes_completion_from_admission(tmp_path):
    Q.write_report(tmp_path, dict(completed=False, authorizes_training=False, modes={}, error="fixture failure"))
    text = (tmp_path / "parity_report.md").read_text()
    assert "No training authorized" in text and "fixture failure" in text
    with pytest.raises(FileExistsError):
        Q.write_report(tmp_path, {})


@pytest.mark.parametrize("architecture", M.P.COUNTS)
def test_trace_supports_all_five_variants(architecture):
    M.D.seed_everything(101)
    model = M.build_model(architecture, trim=False).eval()
    with torch.no_grad(), Q.capture_trace(model) as trace:
        model(*inputs(batch=2, particles=7))
    assert "particle.0.ev_projection_input" in trace and "class.1.block_output" in trace
    evidence = Q.branch_evidence(trace)
    if architecture == "ev_separate":
        assert evidence == {}
    elif architecture == "ev_routing":
        assert all(row["versus_native_weights"]["exact"] for row in evidence.values())
    else:
        assert all(row["nonzero_elements"] == 0 for row in evidence.values())


def test_existing_diagnostic_results_are_not_overwritten(tmp_path, monkeypatch):
    (tmp_path / "parity_report.json").write_text('{"keep":true}')
    monkeypatch.setattr(Q, "load_plan", lambda _: pytest.fail("Must refuse before loading campaign"))
    with pytest.raises(FileExistsError, match="already exists"):
        Q.run_diagnostic(tmp_path)
    assert Q.read(tmp_path / "parity_report.json") == {"keep": True}
