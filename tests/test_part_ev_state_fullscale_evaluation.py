from types import SimpleNamespace
import pytest
import torch
from part_ev_state_fullscale import protocol as P
from part_ev_state_fullscale.engine import create_engine


def test_all_seven_finalists_and_resumable_file_evaluation(tmp_path,monkeypatch):
    engine = create_engine()
    recipe = dict(P.recipes()['ranger_full_v1'],evaluation_population=20)
    spec = dict(tasks=P.TASKS,content_hash='fixture',recipes={'ranger_full_v1':recipe},
        inventory=dict(splits=dict(val=dict(files=[dict(name=f'file_{i}.root',num_entries=10) for i in range(2)]))))
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(10))
        def forward(self,*values): return self.weight.expand(len(values[0]),-1)
    monkeypatch.setattr(P,'load_campaign',lambda _:spec)
    monkeypatch.setattr(P,'benchmark_summary',lambda *a:{})
    monkeypatch.setattr(engine,'gpu_setup',lambda _: (torch.device('cpu'),{}))
    monkeypatch.setattr(engine,'make_model',lambda *a:Tiny())
    monkeypatch.setattr(engine,'classification_metrics',lambda pred,y,**kw:dict(events=len(y),accuracy=.1,macro_auc_ovo=.5))
    for index,task in enumerate(P.TASKS):
        run = P.task_directory(tmp_path,index)
        checkpoint = run/'best_model_val.pt'
        checkpoint.parent.mkdir(parents=True)
        torch.save(dict(task=task,campaign_sha256='fixture',runtime={},model=Tiny().state_dict()),checkpoint)
        P.write_json(run/'training_complete.json',dict(task=task,step=1000000,validated_checks=50,
            campaign_sha256='fixture',best_checkpoint_sha256=P.file_sha256(checkpoint)))
    count = []
    def loader(*args):
        count.append(True)
        return [(dict((k,torch.zeros(10,1,1)) for k in ('pf_points','pf_features','pf_vectors','pf_mask')),
                 dict(_label_=torch.arange(10)))]
    monkeypatch.setattr(engine,'finite_validation_loader',loader)
    args = SimpleNamespace(output=tmp_path,task_index=3)
    engine.evaluate(args)
    assert len(P.read_json(tmp_path/'selection/locked_checkpoints.json')['checkpoints']) == 7
    assert len(count) == 2
    engine.evaluate(args)
    assert len(count) == 2  # Reuse committed logits; no repeated model inference.
    sidecar = P.task_directory(tmp_path,3)/'full_validation/file_000.json'
    row = P.read_json(sidecar)
    row['contract']['entry_start'] = 1
    P.write_json(sidecar,row)
    with pytest.raises(ValueError,match='prediction changed'): engine.evaluate(args)


def test_ten_check_completion_cannot_unlock_full_validation(tmp_path):
    engine = create_engine()
    task = P.TASKS[0]
    spec = dict(tasks=[task],content_hash='fixture',recipes=P.recipes())
    run = P.task_directory(tmp_path,0)
    P.write_json(run/'training_complete.json',dict(task=task,step=200000,validated_checks=10,
        campaign_sha256='fixture',best_checkpoint_sha256='short'))
    with pytest.raises(ValueError,match='Every training'): engine.lock_finalists(tmp_path,spec)
    assert not (tmp_path/'selection/locked_checkpoints.json').exists()
