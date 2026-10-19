import copy
from types import SimpleNamespace
import pytest
import torch
from part_ev_message_study import protocol as P
from part_ev_message_study.engine import create_engine
from scripts import run_part_ev_message_study as M, run_part_ev_screen as original
from part_ev_fullscale.checkpoint import seed_everything, restore_rank_state, capture_rank_state
from tests.test_part_ev_screen_training import batch
from tests.test_part_fullscale_model import bounded_cpu_threads


def test_adapter_has_private_globals_and_preserves_original_trainer():
    old_protocol,old_builder = original.P,original.build_model
    engine = create_engine()
    assert engine is not original and engine.P is P
    assert engine.train.__globals__ is not original.train.__globals__
    assert original.P is old_protocol and original.build_model is old_builder
    assert len(original.P.TASKS) == 7 and len(engine.P.TASKS) == 5


@pytest.mark.parametrize("index",range(5))
def test_new_models_resume_exact_optimizer_rng_state(tmp_path,index):
    engine = create_engine()
    task = P.TASKS[index]
    recipe = P.recipes()[task["recipe"]]
    device = torch.device("cpu")
    model = engine.make_model(task,device)
    opt = engine.build_optimizer(model,recipe)
    scaler = torch.amp.GradScaler("cpu",enabled=False)
    seed_everything(121)
    for step in range(2):
        engine.training_step(model,opt,scaler,batch(),device,recipe,step)
    spec,runtime = dict(content_hash="fixture"),dict(cpu=True)
    engine.save_checkpoint(tmp_path/"last.pt",model,opt,scaler,dict(task=task,step=2),spec,runtime)
    for step in range(2,7):
        engine.training_step(model,opt,scaler,batch(),device,recipe,step)
    expected = copy.deepcopy(model.state_dict())
    another = engine.make_model(task,device)
    opt2 = engine.build_optimizer(another,recipe)
    scaler2 = torch.amp.GradScaler("cpu",enabled=False)
    state,rng = engine.load_checkpoint(tmp_path/"last.pt",another,opt2,scaler2,spec,runtime,task)
    restore_rank_state(another,rng)
    for step in range(state["step"],7):
        engine.training_step(another,opt2,scaler2,batch(),device,recipe,step)
    for name,value in expected.items():
        torch.testing.assert_close(another.state_dict()[name],value,rtol=0,atol=0)
    assert opt.step_counter == opt2.step_counter
    for a,b in zip(opt.slow,opt2.slow):
        torch.testing.assert_close(a,b,rtol=0,atol=0)


@pytest.mark.parametrize("index",range(5))
def test_parity_probe_and_diagnostics_restore_state(index):
    engine = create_engine()
    task = P.TASKS[index]
    model = engine.make_model(task,torch.device("cpu")).train()
    before = copy.deepcopy(model.state_dict())
    rng = capture_rank_state(model)
    assert M.initial_parity(model,task,batch(),torch.device("cpu"))["passed"]
    for name,value in before.items():
        torch.testing.assert_close(value,model.state_dict()[name],rtol=0,atol=0)
    record = engine.context_diagnostic(model,batch(),torch.device("cpu"))
    assert len(record["layers"]) >= 8
    assert torch.equal(torch.get_rng_state(),rng["torch"])
    assert capture_rank_state(model)["trimmers"] == rng["trimmers"]
    assert model.training


def test_training_loop_first_tie_and_completion_unchanged(tmp_path,monkeypatch):
    engine = create_engine()
    task = P.TASKS[0]
    recipe = dict(P.recipes()[task["recipe"]],total_steps=4,steps_per_epoch=2,validation_events_per_epoch=512)
    spec = dict(tasks=P.TASKS,recipes={task["recipe"]:recipe},content_hash="fixture",inventory={})
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(10))
        def forward(self,*values):
            return self.weight.expand(len(values[0]),-1)
    monkeypatch.setattr(P,"load_campaign",lambda _:spec)
    monkeypatch.setattr(P,"benchmark_summary",lambda *a:{})
    monkeypatch.setattr(engine,"gpu_setup",lambda _: (torch.device("cpu"),{}))
    monkeypatch.setattr(engine,"make_model",lambda *a:Tiny())
    monkeypatch.setattr(engine,"build_loader",lambda *a,**kw:([batch()]*20,None))
    monkeypatch.setattr(engine,"validate",lambda *a:dict(accuracy=.7,seconds=1.,macro_auc_ovo=.8,macro_auc_ovr=.8,loss=.2))
    args = SimpleNamespace(output=tmp_path,task_index=0)
    engine.train(args)
    run = P.task_directory(tmp_path,0)
    done = P.read_json(run/"training_complete.json")
    assert (done["step"],done["validated_checks"],done["best_check"]) == (4,2,1)
    checksum = P.file_sha256(run/"best_model_val.pt")
    engine.train(args)
    assert checksum == P.file_sha256(run/"best_model_val.pt")


@pytest.mark.parametrize("failure",[None,"confirmation_skip","validation_memory","nonfinite_state","parity"])
def test_new_benchmark_records_calibration_and_never_writes_production(tmp_path,monkeypatch,capsys,failure):
    from tests.test_part_ev_screen_protocol import precision_record
    limits = dict(P.BENCHMARK,warmup_steps=2,measured_steps=4,window_steps=2,
                  calibration_attempts=2,confirmation_attempts=3)
    task = P.TASKS[0]
    spec = dict(tasks=P.TASKS,recipes=P.recipes(),content_hash="probe",inventory={},benchmark=limits)
    monkeypatch.setattr(P,"load_campaign",lambda _:spec)
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
    args = SimpleNamespace(output=tmp_path,task_index=0)
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
    row = P.read_json(tmp_path/"benchmarks/EV_STATE_RANGER.json")
    assert row["passed"] is (failure is None)
    assert len(row["full_length_attempts"]) == 5
    assert row["full_length_attempts"][0]["amp_skipped"] is True
    assert row["measured_steps"] == 4 and len(row["windows"]) == 2
    assert not (tmp_path/"runs").exists()
    assert (tmp_path/"benchmarks/EV_STATE_RANGER_probe.pt").exists()
