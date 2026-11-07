import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from part_selected_production import protocol as P, engine as E
from part_selected_fullscale import protocol as science
from part_screen_common.engine import ScreenEngine
from part_ev_fullscale.checkpoint import seed_everything, trusted_load
from scripts.diagnose_part_selected_restart import compare_tree
from tests.test_part_selected_restart import Toy, step as toy_step
from tests.test_part_fullscale_model import bounded_cpu_threads


def review():
    return dict(complete=True, benchmarks_healthy=True, models=[dict(task=t, status='measured',
        projection=dict(recommended_train_hours=P.TRAIN_HOURS[i]-1,
                        recommended_evaluate_hours=P.EVALUATE_HOURS[i])) for i,t in enumerate(science.TASKS)])


@pytest.mark.parametrize('index', range(7))
def test_measured_walltime_gate(index):
    row = review(); P.validate_resources(row)
    row['models'][index]['projection']['recommended_train_hours'] = P.TRAIN_HOURS[index]+1
    with pytest.raises(ValueError, match='walltime'): P.validate_resources(row)
    row = review(); row['models'][index]['projection']['recommended_evaluate_hours'] = 7
    with pytest.raises(ValueError, match='walltime'): P.validate_resources(row)


def test_plan_identity_source_evidence_and_policy(tmp_path, monkeypatch):
    root=tmp_path/'production'; root.mkdir()
    for n in P.FILES:
        p=root/'source'/n; p.parent.mkdir(parents=True,exist_ok=True); p.write_text('# frozen\n')
    evidence=tmp_path/'benchmark.json'; evidence.write_text('{}')
    spec=dict(tasks=science.TASKS)
    plan=P.hashed(dict(schema=P.SCHEMA,authorizes_training=True,output=str(root),benchmark=str(tmp_path),retry=str(tmp_path),
        policy=P.POLICY,train_hours=list(P.TRAIN_HOURS),evaluate_hours=list(P.EVALUATE_HOURS),
        source={n:P.sha(root/'source'/n) for n in P.FILES},evidence={str(evidence):P.sha(evidence)},science=spec,review=review()))
    P.write(root/'production.json',plan)
    monkeypatch.setattr(P,'activate',lambda *a:(None,SimpleNamespace(load_campaign=lambda *a,**k:spec)))
    assert P.load(root)==plan
    evidence.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='evidence changed'): P.load(root)
    evidence.write_text('{}')
    (root/'source'/P.FILES[0]).write_text('# changed')
    with pytest.raises(ValueError,match='source changed'): P.load(root)
    (root/'source'/P.FILES[0]).write_text('# frozen\n')
    bad=copy.deepcopy(plan); bad['policy']=dict(P.POLICY,total_steps=200000)
    P.write(root/'production.json',P.hashed({k:v for k,v in bad.items() if k!='content_hash'}))
    with pytest.raises(ValueError,match='policy'): P.load(root)


@pytest.fixture
def training(tmp_path, monkeypatch):
    from part_ev_screen import data
    monkeypatch.setattr(data,'shutdown_loader',lambda _:None)
    monkeypatch.setattr(E.shutil,'disk_usage',lambda _:SimpleNamespace(free=18*2**30))
    task=science.TASKS[0]
    recipe=copy.deepcopy(science.recipes()['ranger_full_v1'])
    recipe.update(total_steps=12,epochs=3,steps_per_epoch=4,validation_events_per_epoch=512)
    runtime=dict(gpu='cpu-test',campaign_sha256='parent')
    evidence=tmp_path/'evidence.json'; P.write(evidence,dict(runtime=runtime))
    plan=dict(content_hash='production',science=dict(tasks=[task],recipes={task['recipe']:recipe}),
              review=dict(models=[dict(evidence_path=str(evidence))]))
    from part_ev_fullscale.protocol import file_sha256, write_json, read_json
    adapter=SimpleNamespace(SCHEMA=P.SCHEMA,write_json=write_json,read_json=read_json,file_sha256=file_sha256)

    class Engine(ScreenEngine):
        def __init__(self, stop_after=None, validation_stop=False, crash_commit=False):
            super().__init__(adapter,None,None,None)
            self.spec=dict(content_hash='production'); self.task=task; self.recipe=recipe
            self.runtime=runtime; self.device=torch.device('cpu'); self.torch=torch
            self.steps=[]; self.stop_after=stop_after; self.validation_stop=validation_stop
            self.crash_commit=crash_commit; self.validation_calls=0

        def make_model(self,*_):
            seed_everything(101); self.model=Toy(); return self.model

        def loader(self,split):
            # Independent deterministic stream generator, like seeded workers.
            def batches():
                g=torch.Generator().manual_seed(121 if split=='train' else 122)
                while True: yield ({'x':torch.randn(8,4,generator=g)}, {}, {})
            return batches()

        def check_stop(self):
            if self.stop_after is not None and len(self.steps)>=self.stop_after: raise SystemExit(75)

        def training_step(self,model,opt,scaler,batch,device,recipe,i):
            row=toy_step(model,opt,scaler,batch,device,recipe,i)
            self.steps.append(i); row['clipped']=False
            return row

        def validate(self,model,iterator,device,events):
            next(iterator)
            if self.validation_stop: raise SystemExit(75)
            model.eval()
            self.validation_calls+=1
            return dict(accuracy=.5+.1*min(self.validation_calls,2),events=events,seconds=1.,macro_auc_ovo=.9)

        def save(self,path,model,opt,scaler,state,spec,runtime):
            if self.crash_commit and state['validated_checks']==1: raise RuntimeError('crash after candidate')
            super().save(path,model,opt,scaler,state,spec,runtime)
    return plan,Engine


def test_full_state_resume_preserves_update_sequence_rng_and_optimizer(tmp_path,training):
    plan,Engine=training
    full=tmp_path/'full'; resumed=tmp_path/'resumed'
    E.train_body(full,plan,0,Engine())
    partial=Engine(stop_after=5)
    with pytest.raises(SystemExit) as exc: E.train_body(resumed,plan,0,partial)
    assert exc.value.code==75
    run=P.run_directory(resumed,plan['science']['tasks'][0])
    saved=trusted_load(run/'last.pt')
    assert saved['state']['step']==5 and saved['state']['validated_checks']==1
    assert set(saved)>= {'model','optimizer','scaler','rng','state'}
    assert 'slow' in saved['optimizer'] and saved['rng']['trimmers']['']==5
    following=Engine(); following.validation_calls=1
    E.train_body(resumed,plan,0,following)
    assert partial.steps+following.steps==list(range(12))
    expected=trusted_load(P.run_directory(full,plan['science']['tasks'][0])/'last.pt')
    actual=trusted_load(run/'last.pt')
    for key in ('model','optimizer','scaler','rng'):
        assert compare_tree(expected[key],actual[key])['exact'],key
    assert E.audit(resumed,plan,0)['best_check']==2  # First tie, not last check.
    # Completion marker crash: finish directly from last without replay/training.
    (run/'training_complete.json').unlink()
    recovery=Engine(); recovery.loader=lambda _:pytest.fail('Should not replay completed schedule')
    E.train_body(resumed,plan,0,recovery)
    assert E.audit(resumed,plan,0)['complete']['step']==12


@pytest.mark.parametrize('failure', ['validation_stop','crash_commit'])
def test_validation_transaction_does_not_advance_on_interruption(tmp_path,training,failure):
    plan,Engine=training
    with pytest.raises((SystemExit,RuntimeError)):
        E.train_body(tmp_path,plan,0,Engine(**{failure:True}))
    run=P.run_directory(tmp_path,plan['science']['tasks'][0])
    last=trusted_load(run/'last.pt')
    assert last['state']['step']==4 and last['state']['validated_checks']==0
    assert last['state']['best_file'] is None
    assert not (run/'training_complete.json').exists()
    recovered=Engine(); E.train_body(tmp_path,plan,0,recovered)
    assert recovered.steps==list(range(4,12))
    assert E.audit(tmp_path,plan,0)['best_check']==2


def test_resume_rejects_tampered_selected_checkpoint(tmp_path,training):
    plan,Engine=training
    with pytest.raises(SystemExit): E.train_body(tmp_path,plan,0,Engine(stop_after=5))
    run=P.run_directory(tmp_path,plan['science']['tasks'][0])
    (run/'selected/check_01.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='checkpoint referenced'): E.train_body(tmp_path,plan,0,Engine())


def test_stream_cursor_invalid_or_incomplete_finish_rejected():
    t=science.TASKS[0]; recipe=science.recipes()['ranger_full_v1']
    state=E.initial_state(t); E.validate_state(state,t,recipe)
    for step,checks in ((-1,0),(20001,0),(40000,0),(1,1),(1000001,50)):
        with pytest.raises(ValueError): E.validate_state(dict(state,step=step,validated_checks=checks),t,recipe)


@pytest.mark.parametrize('i',range(7))
def test_production_composition_uses_original_factory_and_full_step(tmp_path,monkeypatch,i):
    from part_selected_fullscale import benchmark as B, steps
    from tests.test_part_selected_fullscale_model import factory
    task=science.TASKS[i]
    spec=dict(content_hash='parent',tasks=science.TASKS,recipes=science.recipes(),inventory={})
    runtime=dict(campaign_sha256='parent',gpu='cpu-test')
    def init(self,output,index,check_stop):
        self.output=output; self.index=index; self.spec=spec; self.task=task
        self.runtime=runtime; self.device=torch.device('cpu'); self.check_stop=check_stop
    monkeypatch.setattr(B.Bench,'__init__',init)
    monkeypatch.setattr(B.Bench,'model',lambda self:factory(task['architecture'],fast=i in (2,5,6)))
    row=tmp_path/'evidence.json'; P.write(row,dict(runtime=runtime))
    plan=dict(benchmark=str(tmp_path),science=spec,content_hash='production',
              review=dict(models=[dict(evidence_path=str(row))]*7))
    worker=E.create_engine(tmp_path,plan,i)
    model=worker.make_model(task,torch.device('cpu'))
    assert sum(p.numel() for p in model.parameters())==science.COUNTS[task['architecture']]
    assert worker.training_step is steps.training_step
    assert worker.recipe['total_steps']==1000000 and worker.recipe['epochs']==50
    assert worker.validate.__globals__['check_stop']==worker.check_stop
    assert worker.P.load_campaign(tmp_path)['content_hash']=='production'


def test_evaluation_is_per_task_resumable_and_rejects_modified_shards(tmp_path,monkeypatch):
    import numpy as np
    from part_selected_fullscale import benchmark as B
    from part_ev_fullscale.checkpoint import atomic_torch_save
    from part_ev_fullscale.protocol import CLASS_NAMES
    from part_ev_screen import data
    t=science.TASKS[0]; runtime=dict(gpu='cpu',campaign_sha256='parent')
    recipe=copy.deepcopy(science.recipes()['ranger_full_v1']); recipe['evaluation_population']=20
    files=[dict(name=f'f{i}.root',num_entries=10) for i in range(2)]
    spec=dict(tasks=science.TASKS,recipes={t['recipe']:recipe},content_hash='parent',
              inventory=dict(splits=dict(val=dict(files=files))))
    class Model(torch.nn.Module):
        def __init__(self): super().__init__(); self.scale=torch.nn.Parameter(torch.tensor(1.))
        def forward(self,*args): return args[0]*self.scale
    model=Model()
    def init(self,output,index,check_stop):
        self.output=output; self.task=t; self.spec=spec; self.runtime=runtime
        self.device=torch.device('cpu'); self.check_stop=check_stop
    monkeypatch.setattr(B.Bench,'__init__',init)
    monkeypatch.setattr(B.Bench,'model',lambda self:Model())
    ev=tmp_path/'benchmark.json'; P.write(ev,dict(runtime=runtime))
    plan=dict(benchmark=str(tmp_path),science=spec,content_hash='production',
              review=dict(models=[dict(evidence_path=str(ev))]*7))
    run=P.run_directory(tmp_path,t)
    checkpoint=run/'best_model_val.pt'
    atomic_torch_save(dict(model=model.state_dict(),task=t,runtime=runtime,campaign_sha256='production'),checkpoint)
    monkeypatch.setattr(E,'audit',lambda *a:dict(checkpoint=str(checkpoint),checkpoint_sha256=P.sha(checkpoint)))
    calls=[]; interrupt=[True]
    def finite(inventory,file):
        calls.append(file['name'])
        if interrupt[0] and file['name']=='f1.root': raise SystemExit(75)
        x=torch.eye(10)
        return [(dict(pf_points=x,pf_features=x,pf_vectors=x,pf_mask=x),{'_label_':torch.arange(10)},{})]
    monkeypatch.setattr(data,'finite_validation_loader',finite)
    worker=E.create_engine(tmp_path,plan,0)
    args=SimpleNamespace(output=tmp_path,task_index=0)
    with pytest.raises(SystemExit): worker.evaluate(args)
    assert (run/'full_validation/file_000.json').exists()
    assert not (run/'full_validation/metrics.json').exists()
    before=P.sha(run/'full_validation/file_000.npz')
    interrupt[0]=False; calls.clear()
    worker.evaluate(args)
    assert calls==['f1.root'] and P.sha(run/'full_validation/file_000.npz')==before
    metrics=P.read(run/'full_validation/metrics.json')['metrics']
    assert metrics['events']==20 and metrics['accuracy']==1.
    assert metrics['class_counts']==dict.fromkeys(CLASS_NAMES,2)
    assert len(metrics['qcd_rejection'])>0
    # Only task 0 has any completion/selection information; other tasks were not awaited.
    assert len(list((tmp_path/'runs').iterdir()))==1
    (run/'full_validation/file_000.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='predictions changed'): worker.evaluate(args)


def test_inspection_requires_seven_healthy_benchmarks(tmp_path,monkeypatch):
    spec=dict(tasks=science.TASKS,launch=dict(python='frozen',account='account'))
    result=review(); result['benchmarks_healthy']=False
    driver=SimpleNamespace(execute=lambda _:result)
    monkeypatch.setattr(P,'activate',lambda *a:(driver,SimpleNamespace(load_campaign=lambda _:spec)))
    with pytest.raises(ValueError,match='healthy'): P.inspect(tmp_path,tmp_path)


def test_report_recomputes_metrics_and_does_not_claim_external_controls(tmp_path,monkeypatch):
    import numpy as np
    from part_selected_production import reporting as R
    from part_screen_common import controls
    from scripts.run_part_ev_screen import classification_metrics, atomic_npz
    plan=dict(content_hash='production',benchmark=str(tmp_path/'benchmark'),
        science=dict(tasks=science.TASKS,parameters=science.COUNTS,
                     inventory=dict(splits=dict(val=dict(files=[])))))
    registry=Path(plan['benchmark'])/'controls/registry.json'
    P.write(registry,dict(note='metadata only'))
    before=P.sha(registry)
    logits=np.tile(np.eye(10,dtype=np.float32),(2,1)); labels=np.tile(np.arange(10),2)
    path=tmp_path/'predictions.npz'; atomic_npz(path,logits=logits,labels=labels)
    metrics=classification_metrics(logits,labels,rejection=True)
    monkeypatch.setattr(R,'audit',lambda *a:dict(checkpoint_sha256='best',best_check=1,
        checks=[dict(check=i+1,accuracy=1.) for i in range(50)],complete=dict(train_seconds=1.,validation_seconds=1.)))
    monkeypatch.setattr(controls,'validate_predictions',lambda *a,**k:dict(metrics=metrics,metrics_sha256='metric',
        predictions=[dict(path=str(path))]))
    result=R.report(tmp_path,plan)
    assert result['complete'] and len(result['models'])==7
    assert result['external_controls']['status']=='pending_separate_compatibility_audit'
    assert result['independent_test'] is False
    assert P.sha(registry)==before
    assert (tmp_path/'reports/production_report.md').is_file()
    metrics['macro_auc_ovo']-=.01
    with pytest.raises(ValueError,match='metrics differ'): R.report(tmp_path,plan)
