from types import SimpleNamespace
import numpy as np
import pytest
import torch
from part_ev_screen import protocol as P
from part_ev_screen import reporting
from scripts import run_part_ev_screen as D
from tests.test_part_fullscale_data import write_root, fixture_inventory
from tests.test_part_fullscale_model import bounded_cpu_threads


def test_all_seven_evaluate_resume_and_report_real_root(tmp_path, monkeypatch):
    path = tmp_path/'data/HToBB_120.root'
    write_root(path,events=30)
    inventory = fixture_inventory('val',[path])
    inventory['splits']['val']['files'][0]['num_entries'] = 30
    spec = dict(tasks=P.TASKS,recipes=P.recipes(),content_hash='small-integration-fixture',
                inventory=inventory,parameters=P.COUNTS)
    for recipe in spec['recipes'].values(): recipe['evaluation_population'] = 30
    monkeypatch.setattr(P,'load_campaign',lambda _:spec)
    monkeypatch.setattr(P,'benchmark_summary',lambda *args:{})
    monkeypatch.setattr(D,'gpu_setup',lambda _: (torch.device('cpu'),{}))
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.bias = torch.nn.Parameter(torch.arange(10,dtype=torch.float32))
        def forward(self,*values): return self.bias.expand(len(values[0]),-1)
    monkeypatch.setattr(D,'make_model',lambda *args:Tiny())
    for index,task in enumerate(P.TASKS):
        run = P.task_directory(tmp_path,index)
        checkpoint = run/'best_model_val.pt'
        D.atomic_torch_save(dict(task=task,runtime={},campaign_sha256=spec['content_hash'],model=Tiny().state_dict()),checkpoint)
        P.write_json(run/'training_complete.json',dict(task=task,step=200000,validated_checks=10,
            campaign_sha256=spec['content_hash'],best_checkpoint_sha256=P.file_sha256(checkpoint),
            best_check=8,train_seconds=800,validation_seconds=70,amp_skipped_steps=1))
        for check in range(1,11):
            P.write_json(run/'validation'/f'check_{check:02d}.json',
                dict(accuracy=.1,macro_auc_ovo=.5,macro_auc_ovr=.5,loss=2.,check=check,step=check*20000))
        P.write_json(tmp_path/'benchmarks'/f"{task['run_id']}.json",
                     dict(jets_per_second=1000,full_length_peak_bytes=1024**3))
    for index in range(7): D.evaluate(SimpleNamespace(output=tmp_path,task_index=index))
    # Reuse committed predictions rather than executing model inference again.
    def fail(*args): raise AssertionError('Should reuse committed file')
    monkeypatch.setattr(D,'finite_validation_loader',fail)
    D.evaluate(SimpleNamespace(output=tmp_path,task_index=0))
    reporting.report(tmp_path)
    result = P.read_json(tmp_path/'reports/screen_report.json')
    assert len(result['models']) == 7 and result['single_seed_exploratory'] is True
    assert result['contrasts']['accuracy']['context_gate_interaction'] == 0
    text = (tmp_path/'reports/screen_report.md').read_text()
    assert 'not test results' in text and 'not an independent holdout' in text
    assert '99.5%' in text and '99%' in text
    assert (tmp_path/'reports/learning_curves.png').stat().st_size > 1000
    # Missing runs cannot silently become zeros.
    metric_path = P.task_directory(tmp_path,6)/'full_validation/metrics.json'
    metric_path.unlink()
    with pytest.raises(FileNotFoundError): reporting.report(tmp_path)
    # Corrupt committed predictions must stop resumption, not be overwritten.
    prediction = P.task_directory(tmp_path,0)/'full_validation/file_000.npz'
    prediction.write_bytes(b'corrupt fixture')
    with pytest.raises(ValueError,match='changed'):
        D.evaluate(SimpleNamespace(output=tmp_path,task_index=0))
