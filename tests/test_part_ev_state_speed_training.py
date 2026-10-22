import copy
from types import SimpleNamespace
import pytest
import torch
from part_ev_state_speed import protocol as P, benchmark as B
from part_ev_state_speed.engine import create_engine, benchmark_driver
from part_ev_fullscale.checkpoint import seed_everything, restore_rank_state, capture_rank_state
from scripts import run_part_ev_screen as original
from tests.test_part_ev_screen_training import batch
from tests.test_part_fullscale_model import bounded_cpu_threads


def test_private_engine_and_benchmark_do_not_modify_shared_drivers():
    from scripts import run_part_ev_message_study as old_driver
    engine = create_engine()
    probe = benchmark_driver(engine, lambda *a: {})
    assert engine.P is P and probe.P is P
    assert engine.train.__globals__ is not original.train.__globals__
    assert probe.benchmark.__globals__ is not old_driver.benchmark.__globals__
    assert len(old_driver.P.TASKS) == 5 and len(original.P.TASKS) == 7


@pytest.mark.parametrize("index",range(3))
def test_complete_ranger_resume_matches_uninterrupted(tmp_path,index):
    engine = create_engine()
    task = P.TASKS[index]
    recipe = P.recipes()[task["recipe"]]
    device = torch.device("cpu")
    model = engine.make_model(task,device)
    optimizer = engine.build_optimizer(model,recipe)
    scaler = torch.amp.GradScaler("cpu",enabled=False)
    seed_everything(88)
    for step in range(2):
        engine.training_step(model,optimizer,scaler,batch(),device,recipe,step)
    spec,runtime = dict(content_hash="fixture"),dict(cpu=True)
    engine.save_checkpoint(tmp_path/"last.pt",model,optimizer,scaler,dict(task=task,step=2),spec,runtime)
    for step in range(2,7):
        engine.training_step(model,optimizer,scaler,batch(),device,recipe,step)
    expected = copy.deepcopy(model.state_dict())
    other = engine.make_model(task,device)
    optimizer2 = engine.build_optimizer(other,recipe)
    scaler2 = torch.amp.GradScaler("cpu",enabled=False)
    state,rng = engine.load_checkpoint(tmp_path/"last.pt",other,optimizer2,scaler2,spec,runtime,task)
    restore_rank_state(other,rng)
    for step in range(state["step"],7):
        engine.training_step(other,optimizer2,scaler2,batch(),device,recipe,step)
    for name,value in expected.items():
        torch.testing.assert_close(other.state_dict()[name],value,rtol=0,atol=0)
    assert optimizer.step_counter == optimizer2.step_counter
    for a,b in zip(optimizer.slow,optimizer2.slow):
        torch.testing.assert_close(a,b,rtol=0,atol=0)


@pytest.mark.parametrize("index",range(3))
def test_diagnostics_restore_training_state(index):
    engine = create_engine()
    model = engine.make_model(P.TASKS[index],torch.device("cpu")).train()
    before = copy.deepcopy(model.state_dict())
    rng = capture_rank_state(model)
    result = engine.context_diagnostic(model,batch(),torch.device("cpu"))
    assert len(result["layers"]) >= 8 and model.training
    assert torch.equal(torch.get_rng_state(),rng["torch"])
    for key,value in before.items():
        torch.testing.assert_close(model.state_dict()[key],value,rtol=0,atol=0)
    assert model._state_pair_cache.stem is None


def test_trained_checkpoint_probe_keeps_actual_nonzero_state_weights(tmp_path):
    from part_ev_message_study.model import build_model
    from tests.test_part_fullscale_model import inputs
    engine = create_engine()
    model = build_model("ev_state", trim=False)
    with torch.no_grad():
        for block in model.mod.blocks:
            block.attn.state.output.normal_(std=.02)
    checkpoint = tmp_path/"best.pt"
    torch.save(dict(model=model.state_dict()),checkpoint)
    spec = dict(state_reference=dict(checkpoint=str(checkpoint),checkpoint_sha256=P.file_sha256(checkpoint)))
    values = inputs(batch=3,particles=71)
    data = ({k:v for k,v in zip(("pf_points","pf_features","pf_vectors","pf_mask"),values)},
            {"_label_":torch.zeros(3,dtype=torch.long)})
    evidence = B.trained_equivalence(engine,spec,data,torch.device("cpu"))
    assert evidence["passed"] and evidence["argmax_disagreements"] == 0
    assert P.file_sha256(checkpoint) == spec["state_reference"]["checkpoint_sha256"]


def test_failed_extended_parity_restores_rng_and_preserves_evidence(monkeypatch):
    engine = create_engine()
    task = P.TASKS[0]
    model = engine.make_model(task,torch.device("cpu")).train()
    before = capture_rank_state(model)
    monkeypatch.setattr(B,"ev_parity",lambda *a:dict(passed=True))
    def fail(*a):
        torch.rand(17)
        raise AssertionError("nonzero gradient mismatch")
    monkeypatch.setattr(B,"gradient_equivalence",fail)
    with pytest.raises(B.ParityError,match="gradient mismatch") as error:
        B.initial_parity(engine,{},model,task,batch(),torch.device("cpu"))
    assert error.value.report["passed"] is False
    assert torch.equal(torch.get_rng_state(),before["torch"])


def test_training_loop_and_first_tie_selection(tmp_path,monkeypatch):
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
    engine.train(SimpleNamespace(output=tmp_path,task_index=0))
    done = P.read_json(P.task_directory(tmp_path,0)/"training_complete.json")
    assert (done["step"],done["validated_checks"],done["best_check"]) == (4,2,1)


@pytest.mark.skipif(not torch.cuda.is_available(),reason="Cluster CUDA required")
@pytest.mark.parametrize("index",range(3))
def test_cuda_nonzero_forward_backward_gate(index):
    torch.backends.cuda.matmul.allow_tf32 = False
    assert B.gradient_equivalence(P.TASKS[index]["architecture"],torch.device("cuda"))["passed"]
