from types import SimpleNamespace
import pytest
import torch
from part_ev_state_fullscale import protocol as P
from scripts import run_part_ev_state_fullscale as M
from tests.test_part_ev_screen_training import batch


@pytest.mark.parametrize("failure",[None,"confirmation_skip","validation_memory","nonfinite_state","parity"])
def test_new_benchmark_records_calibration_and_never_writes_production(tmp_path,monkeypatch,capsys,failure):
    from tests.test_part_ev_screen_protocol import precision_record
    limits = dict(P.BENCHMARK,warmup_steps=2,measured_steps=4,window_steps=2,
                  calibration_attempts=2,confirmation_attempts=3)
    task = P.TASKS[3]
    spec = dict(tasks=P.TASKS,recipes=P.recipes(),content_hash="probe",inventory={},benchmark=limits)
    constructed = []
    def load(_):
        # Optimizer lazy imports can change importlib's visible vendor metadata.
        # Never reload the environment fingerprint in the middle of the probe.
        assert not constructed
        return spec
    monkeypatch.setattr(P,"load_campaign",load)
    optimizer_factory = M.D.build_optimizer
    def optimizer(*args):
        result = optimizer_factory(*args)
        constructed.append(True)
        return result
    monkeypatch.setattr(M.D,"build_optimizer",optimizer)
    monkeypatch.setattr(P,"verify_inventory_metadata",lambda _:None)
    monkeypatch.setattr(M,"gpu_setup",lambda _: (torch.device("cpu"),dict(gpu_memory_bytes=1000)))
    monkeypatch.setattr(M,"initial_parity",lambda *a:dict(passed=True))
    if failure == "parity":
        def rejected(*_):
            raise M.ParityError("train_fp16 new_first mismatch", dict(passed=False, error="fixture"))
        monkeypatch.setattr(M,"initial_parity",rejected)
    monkeypatch.setattr(M.D,"make_model",lambda *a:torch.nn.Linear(2,2))
    monkeypatch.setattr(M.D,"build_loader",lambda *a,**kw:([batch()]*20,None))
    monkeypatch.setattr(M.D,"validate",lambda *a,**kw:dict(accuracy=.6,seconds=2,events=128000,
                                                       precision=precision_record("torch.float32")))
    full = []
    def step(*args,**kwargs):
        assert failure != "parity", "No optimizer updates after failed parity"
        if kwargs.get("full_length"):
            full.append(True)
        skipped = kwargs.get("full_length") and (len(full) == 1 or failure == "confirmation_skip")
        return dict(loss=1.,amp_skipped=bool(skipped),gradient_norm=None if skipped else 1.,
                    precision=precision_record("torch.float16"))
    monkeypatch.setattr(M.D,"training_step",step)
    monkeypatch.setattr(M,"finite_state",lambda *a:failure != "nonfinite_state")
    monkeypatch.setattr(torch.cuda,"reset_peak_memory_stats",lambda:None)
    monkeypatch.setattr(torch.cuda,"synchronize",lambda:None)
    peaks = iter([50,100,950 if failure == "validation_memory" else 200])
    monkeypatch.setattr(torch.cuda,"max_memory_allocated",lambda:next(peaks))
    args = SimpleNamespace(output=tmp_path,task_index=3)
    if failure == "parity":
        with pytest.raises(M.ParityError,match="train_fp16 new_first"):
            M.benchmark(args)
        assert "initial_parity_failed" in capsys.readouterr().out
        assert not list((tmp_path/"benchmarks").glob("*.json"))
        assert not list((tmp_path/"benchmarks").glob("*.pt"))
        assert not (tmp_path/"runs").exists()
        return
    if failure:
        with pytest.raises(RuntimeError,match="admission"):
            M.benchmark(args)
    else:
        M.benchmark(args)
    row = P.read_json(tmp_path/"benchmarks/EV_STATE_LATE4_RANGER.json")
    assert row["passed"] is (failure is None)
    assert len(row["full_length_attempts"]) == 5
    assert row["full_length_attempts"][0]["amp_skipped"] is True
    assert row["measured_steps"] == 4 and len(row["windows"]) == 2
    assert not (tmp_path/"runs").exists()
    assert (tmp_path/"benchmarks/EV_STATE_LATE4_RANGER_probe.pt").exists()


def test_review_keeps_full_schedule_walltimes_and_three_models(tmp_path):
    from tests.test_part_ev_state_fullscale_protocol import fixture
    spec,_ = fixture(tmp_path)
    summary = P.benchmark_summary(tmp_path,spec)
    for row in summary['models']:
        row['jets_per_second'] = 1000
    view = M.review_view(summary)
    assert len(view['models']) == 3
    assert view['recommended_train_hours'] == summary['recommended_train_hours']
    assert view['recommended_evaluate_hours'] >= 5
    assert all(row['seed'] == 101 for row in view['models'])


