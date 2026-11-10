import copy
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from part_ev_compact_production import protocol as P,engine as E
from part_ev_compact import protocol as C
from part_screen_common.engine import ScreenEngine
from part_ev_fullscale.checkpoint import seed_everything,trusted_load
from tests.test_part_selected_restart import Toy,step as toy_step
from scripts.diagnose_part_selected_restart import compare_tree
from tests.test_part_fullscale_model import bounded_cpu_threads


def review():
    return dict(passed=True,authorizes_training=False,budget=dict(matched_steps=1243800),
        models=[dict(architecture=a,steps=n,projected_hours=h,recommended_hours=limit,fits_tigris_20_days=True)
                for a,n,h,limit in [('compact',1000000,114.868,150),('matched',1243800,56.975,75)]],
        recipes={'compact':C.recipe('compact'),'matched':C.recipe('matched',1243800)})


@pytest.mark.parametrize('index',range(2))
def test_walltime_and_frozen_budget_gate(index):
    row=review(); P.validate_review(row)
    row['models'][index]['recommended_hours']-=1
    with pytest.raises(ValueError,match='walltime'): P.validate_review(row)
    row=review(); row['models'][index]['steps']-=50
    with pytest.raises(ValueError,match='schedule'): P.validate_review(row)


def test_equal_selection_opportunities_lr_and_storage():
    from part_ev_fullscale.protocol import learning_rate
    row=review(); a,b=row['recipes'].values()
    assert a['steps_per_epoch']==20000 and b['steps_per_epoch']==24876
    for i in range(50):
        assert learning_rate(i*20000,a)==learning_rate(i*24876,b)
    storage=P.storage_budget([{'checkpoint_bytes':80*2**20}]*2,C.COUNTS)
    assert 6*2**30 < storage['required_free_bytes'] < 9*2**30
    assert storage['reserve_bytes']==4*2**30


def test_admission_recomputes_success_and_authenticates_report(tmp_path,monkeypatch):
    from tests.test_part_ev_compact_review import reports
    root=reports(tmp_path,monkeypatch)
    spec=C.load(root)
    spec.update(tasks=C.TASKS,parameters=C.COUNTS,inventory={},environment={'architecture':'aarch64'},
                launch=dict(python='frozen',account='reu-aisocial'))
    for name in ('campaign.json','snapshot.json','launch.json'): P.write(root/name,{})
    for arch in C.COUNTS:
        path=root/'benchmarks'/f'{arch}.json'; r=P.read(path); r.pop('content_hash')
        r['checkpoint_bytes']=60*2**20; P.write(path,C.hashed(r))
    summary=C.summarize(root); P.write(root/'reports/benchmark_summary.json',C.hashed(summary))
    monkeypatch.setattr(P,'activate',lambda _:C)
    evidence=P.inspect(root)
    assert evidence['science']['recipes']['matched']['total_steps']==1362850
    assert evidence['evaluate_hours']==[5,5]
    assert evidence['science']['recipes']['compact']['evaluation_population']==5000000
    # Forged success / modified summary cannot be used for production.
    summary['budget']['matched_steps']+=50
    P.write(root/'reports/benchmark_summary.json',C.hashed(summary))
    with pytest.raises(ValueError,match='summary differs'): P.inspect(root)


def test_production_metadata_source_and_evidence_gate(tmp_path,monkeypatch):
    output=tmp_path/'production'; output.mkdir()
    for n in P.FILES:
        path=output/'source'/n; path.parent.mkdir(parents=True,exist_ok=True); path.write_text('# frozen')
    evidence=tmp_path/'benchmark.json'; evidence.write_text('{}')
    admission=dict(benchmark=str(tmp_path),evaluate_hours=P.EVALUATE_HOURS,evidence={str(evidence):P.sha(evidence)})
    plan=P.hashed(dict(schema=P.SCHEMA,authorizes_training=True,output=str(output),policy=P.POLICY,
        source=P.inventory(output/'source'),**admission))
    P.write(output/'production.json',plan)
    monkeypatch.setattr(P,'inspect',lambda _:admission)
    assert P.load(output)==plan
    evidence.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='evidence changed'): P.load(output)
    evidence.write_text('{}'); (output/'source/extra.py').write_text('# sneaked in')
    with pytest.raises(ValueError,match='source changed'): P.load(output)


@pytest.fixture
def training(tmp_path,monkeypatch):
    from part_ev_screen import data
    monkeypatch.setattr(data,'shutdown_loader',lambda _:None)
    monkeypatch.setattr(E.shutil,'disk_usage',lambda _:SimpleNamespace(free=18*2**30))
    task=dict(C.TASKS[0],recipe='compact')
    recipe=C.recipe('compact'); recipe.update(total_steps=12,epochs=3,steps_per_epoch=4,validation_events_per_epoch=512)
    runtime=dict(gpu='cpu-test')
    plan=dict(content_hash='production',science=dict(tasks=[task],recipes={'compact':recipe},
        parameters={'compact':sum(p.numel() for p in Toy().parameters())}),runtime_by_architecture={'compact':runtime},
        benchmark_spec=dict(input_reference=dict(fingerprints={})),
        flop_curves={'compact':dict(convention='toy',coefficients={'training':['1','2','3']},
                                   samples={'1':{'training':{'per_jet':6}}})})
    adapter=SimpleNamespace(SCHEMA=P.SCHEMA,write_json=P.write,read_json=P.read,file_sha256=P.sha)
    class Engine(ScreenEngine):
        def __init__(self,stop_after=None,validation_stop=False,crash_commit=False,replay_stop=False):
            super().__init__(adapter,None,None,None)
            self.spec=dict(content_hash='production'); self.task=task; self.recipe=recipe
            self.runtime=runtime; self.device=torch.device('cpu'); self.torch=torch
            self.steps=[]; self.stop_after=stop_after; self.validation_stop=validation_stop
            self.crash_commit=crash_commit; self.validation_calls=0; self.replay_stop=replay_stop
        def make_model(self,*_): seed_everything(101); return Toy()
        def observe(self,model):
            return model.register_forward_hook(lambda *_:setattr(self,'last_length',4))
        def loader(self,split):
            def batches():
                g=torch.Generator().manual_seed(121 if split=='train' else 122)
                while True: yield ({'x':torch.randn(8,4,generator=g)},{},{})
            return batches()
        def check_stop(self):
            if self.stop_after is not None and len(self.steps)>=self.stop_after: raise SystemExit(75)
        def training_step(self,model,opt,scaler,batch,device,recipe,i,**kw):
            r=toy_step(model,opt,scaler,batch,device,recipe,i); self.steps.append(i)
            return dict(r,clipped=False,precision={'passed':True})
        def validate(self,model,iterator,device,events,**kw):
            next(iterator)
            if self.validation_stop: raise SystemExit(75)
            model.eval(); self.validation_calls+=1
            return dict(accuracy=.5+.1*min(self.validation_calls,2),events=events,seconds=1.,
                        macro_auc_ovo=.9,precision={'passed':True})
        def save(self,path,model,opt,scaler,state,spec,runtime):
            if self.crash_commit and state['validated_checks']==1: raise RuntimeError('crash after candidate')
            super().save(path,model,opt,scaler,state,spec,runtime)
        def replay(self,iterator,count):
            if self.replay_stop: next(iterator); raise SystemExit(75)
            super().replay(iterator,count)
    return plan,Engine


def test_full_state_restart_matches_continuation_and_first_tie(tmp_path,training):
    plan,Engine=training
    E.train_body(tmp_path/'full',plan,0,Engine())
    run=P.run_directory(tmp_path/'resumed',plan['science']['tasks'][0])
    partial=Engine(stop_after=5)
    with pytest.raises(SystemExit) as exc: E.train_body(tmp_path/'resumed',plan,0,partial)
    assert exc.value.code==75
    saved=trusted_load(run/'last.pt')
    assert saved['state']['length_histogram']=={'4':5}
    assert 'slow' in saved['optimizer'] and saved['rng']['trimmers']['']==5
    following=Engine(); following.validation_calls=1
    E.train_body(tmp_path/'resumed',plan,0,following)
    assert partial.steps+following.steps==list(range(12))
    actual=trusted_load(run/'last.pt')
    expected=trusted_load(P.run_directory(tmp_path/'full',plan['science']['tasks'][0])/'last.pt')
    for key in ('model','optimizer','scaler','rng'): assert compare_tree(expected[key],actual[key])['exact'],key
    assert E.audit(tmp_path/'resumed',plan,0)['best_check']==2
    # A missing final marker can be recovered without repeating any updates.
    (run/'training_complete.json').unlink()
    recovery=Engine(); recovery.loader=lambda _:pytest.fail('No replay needed')
    E.train_body(tmp_path/'resumed',plan,0,recovery)
    assert E.audit(tmp_path/'resumed',plan,0)['complete']['compute']['attempted_updates']==12


@pytest.mark.parametrize('failure',['validation_stop','crash_commit'])
def test_validation_transaction_preserves_prevalidation_cursor(tmp_path,training,failure):
    plan,Engine=training
    with pytest.raises((SystemExit,RuntimeError)): E.train_body(tmp_path,plan,0,Engine(**{failure:True}))
    run=P.run_directory(tmp_path,plan['science']['tasks'][0]); state=trusted_load(run/'last.pt')['state']
    assert state['step']==4 and state['validated_checks']==0 and state['best_file'] is None
    recovery=Engine(); E.train_body(tmp_path,plan,0,recovery)
    assert recovery.steps==list(range(4,12))
    assert E.audit(tmp_path,plan,0)['best_check']==2


def test_stop_during_replay_never_commits_wrong_rng(tmp_path,training):
    plan,Engine=training
    with pytest.raises(SystemExit): E.train_body(tmp_path,plan,0,Engine(stop_after=5))
    run=P.run_directory(tmp_path,plan['science']['tasks'][0]); before=P.sha(run/'last.pt')
    with pytest.raises(SystemExit): E.train_body(tmp_path,plan,0,Engine(replay_stop=True))
    assert P.sha(run/'last.pt')==before


def test_selected_tamper_and_histogram_corruption_fail(tmp_path,training):
    plan,Engine=training
    with pytest.raises(SystemExit): E.train_body(tmp_path,plan,0,Engine(stop_after=5))
    run=P.run_directory(tmp_path,plan['science']['tasks'][0]); (run/'selected/check_01.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='referenced'): E.train_body(tmp_path,plan,0,Engine())
    state=E.initial_state(Engine().task); state['length_histogram']={'128':1}
    with pytest.raises(ValueError,match='histogram'): E.validate_state(state,Engine().task,Engine().recipe)


@pytest.mark.parametrize('index',range(2))
def test_actual_models_full_step_and_per_arm_schedule(tmp_path,monkeypatch,index):
    from part_selected_fullscale.steps import training_step
    from part_ev_screen import data
    runtime={'gpu':'cpu'}
    monkeypatch.setattr(E,'gpu_setup',lambda *a:(torch.device('cpu'),runtime))
    monkeypatch.setattr(E.signal,'signal',lambda *a:None)
    tasks=[dict(t,recipe=t['architecture']) for t in C.TASKS]
    plan=dict(content_hash='prod',science=dict(tasks=tasks,recipes=review()['recipes'],parameters=C.COUNTS))
    engine=E.create_engine(tmp_path,plan,index)
    model=engine.make_model(engine.task,engine.device)
    assert sum(p.numel() for p in model.parameters())==C.COUNTS[engine.task['architecture']]
    assert engine.training_step is training_step
    assert engine.recipe['total_steps']==[1000000,1243800][index]
    assert engine.validate.__globals__['check_stop']==engine.check_stop
    hook=engine.observe(model)
    from part_ev_compact.flops import synthetic
    model.eval()
    with torch.no_grad(): model(*synthetic(2,17,torch.device('cpu')))
    assert engine.last_length==17
    hook.remove()


def test_compute_accounts_for_all_attempts_without_retargeting():
    curve=dict(convention='test',coefficients={'training':['1','2','3']},samples={'1':{'training':{'per_jet':6}}})
    state=dict(step=3,amp_skipped_steps=1,length_histogram={'1':1,'2':2})
    r=E.compute_accounting(state,curve)
    assert r['dense_training_flops']==512*(6+2*17)
    assert r['attempted_updates']==3 and r['successful_updates']==2
    assert not r['includes_validation']
