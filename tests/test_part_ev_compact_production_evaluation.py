from types import SimpleNamespace
import numpy as np
import pytest
import torch
from part_ev_compact_production import protocol as P,engine as E,reporting as R
from part_ev_compact import protocol as C
from part_ev_fullscale.checkpoint import atomic_torch_save
from tests.test_part_fullscale_model import bounded_cpu_threads


def test_filewise_evaluation_resumes_and_verifies_shards(tmp_path,monkeypatch):
    from part_ev_screen import data
    task=dict(C.TASKS[0],recipe='compact'); runtime={'gpu':'cpu'}
    recipe=C.recipe('compact'); recipe['evaluation_population']=20
    files=[dict(name=f'f{i}.root',num_entries=10) for i in range(2)]
    plan=dict(content_hash='prod',science=dict(tasks=[task],recipes={'compact':recipe},
        inventory=dict(splits=dict(val=dict(files=files)))))
    class Model(torch.nn.Module):
        def __init__(self): super().__init__(); self.scale=torch.nn.Parameter(torch.tensor(1.))
        def forward(self,*args): return args[0]*self.scale
    model=Model(); run=P.run_directory(tmp_path,task); checkpoint=run/'best_model_val.pt'
    atomic_torch_save(dict(model=model.state_dict(),task=task,runtime=runtime,campaign_sha256='prod'),checkpoint)
    monkeypatch.setattr(E,'gpu_setup',lambda *a:(torch.device('cpu'),runtime))
    monkeypatch.setattr(E.signal,'signal',lambda *a:None)
    monkeypatch.setattr(E,'audit',lambda *a:dict(checkpoint=str(checkpoint),checkpoint_sha256=P.sha(checkpoint)))
    calls=[]; interrupt=[True]
    def finite(inventory,file):
        calls.append(file['name'])
        if interrupt[0] and file['name']=='f1.root': raise SystemExit(75)
        x=torch.eye(10)
        return [(dict(pf_points=x,pf_features=x,pf_vectors=x,pf_mask=x),{'_label_':torch.arange(10)},{})]
    monkeypatch.setattr(data,'finite_validation_loader',finite)
    worker=E.create_engine(tmp_path,plan,0); worker.make_model=lambda *a:Model()
    args=SimpleNamespace(output=tmp_path,task_index=0)
    with pytest.raises(SystemExit): worker.evaluate(args)
    assert (run/'full_validation/file_000.json').exists() and not (run/'full_validation/metrics.json').exists()
    before=P.sha(run/'full_validation/file_000.npz')
    interrupt[0]=False; calls.clear(); worker.evaluate(args)
    assert calls==['f1.root'] and P.sha(run/'full_validation/file_000.npz')==before
    metric=P.read(run/'full_validation/metrics.json')['metrics']
    assert metric['accuracy']==1 and metric['events']==20 and metric['qcd_rejection']
    assert len(list((tmp_path/'runs').iterdir()))==1  # Doesn't require the other training to finish.
    (run/'full_validation/file_000.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='predictions changed'): worker.evaluate(args)


def test_report_recomputes_metrics_checks_labels_and_shows_realized_compute(tmp_path,monkeypatch):
    from part_screen_common import controls
    from scripts.run_part_ev_screen import classification_metrics,atomic_npz
    tasks=[dict(t,recipe=t['architecture']) for t in C.TASKS]
    plan=dict(content_hash='prod',review={'budget':{'exposure_ratio':1.2438}},
        science=dict(tasks=tasks,parameters=C.COUNTS,inventory={'splits':{'val':{'files':[]}}}))
    logits=np.tile(np.eye(10,dtype=np.float32),(2,1)); labels=np.tile(np.arange(10),2)
    path=tmp_path/'predictions.npz'; atomic_npz(path,logits=logits,labels=labels)
    metrics=classification_metrics(logits,labels,rejection=True)
    def audit(output,plan,i):
        return dict(checkpoint_sha256='best',best_check=50,
            checks=[dict(check=j+1,step=(20000 if i==0 else 24876)*(j+1),accuracy=1.) for j in range(50)],
            complete=dict(train_seconds=1.,validation_seconds=1.,
                compute=dict(attempted_updates=1000000 if i==0 else 1243800,dense_training_flops=100+i)))
    monkeypatch.setattr(R,'audit',audit)
    monkeypatch.setattr(controls,'validate_predictions',lambda *a,**kw:dict(metrics=metrics,metrics_sha256='metric',
        predictions=[dict(path=str(path))],labels_by_file={'f':'hash'}))
    result=R.report(tmp_path,plan)
    assert result['complete'] and result['independent_test'] is False
    assert result['observed_matched_to_compact_dense_flop_ratio']==1.01
    assert result['contrast']['accuracy_pp']==0.
    assert '24,876' in (tmp_path/'reports/production_report.md').read_text()
    metrics['macro_auc_ovo']-=.001
    with pytest.raises(ValueError,match='Metrics differ'): R.report(tmp_path,plan)


def test_ready_does_not_trust_a_completion_marker(tmp_path,monkeypatch):
    task=dict(C.TASKS[0],recipe='compact')
    plan={'science':{'tasks':[task]}}
    assert R.ready(tmp_path,plan,[0])==[[0,True,True]]
    run=P.run_directory(tmp_path,task); P.write(run/'training_complete.json',{})
    def invalid(*a): raise ValueError('Invalid training budget')
    monkeypatch.setattr(R,'audit',invalid)
    with pytest.raises(ValueError,match='budget'): R.ready(tmp_path,plan,[0])


def test_status_reports_checks_without_claiming_integrity(tmp_path):
    task=dict(C.TASKS[0],recipe='compact'); plan={'science':{'tasks':[task]}}
    P.write(P.run_directory(tmp_path,task)/'validation/check_01.json',dict(check=1,step=20000,accuracy=.8))
    result=R.status(tmp_path,plan)
    assert result['models'][0]['best_check']==1
    assert 'not completion approval' in result['note']
