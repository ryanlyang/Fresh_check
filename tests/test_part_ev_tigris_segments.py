import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from part_ev_tigris_segments import protocol as P, engine as E
from part_ev_fullscale.checkpoint import trusted_load
from tests.test_part_ev_ddp_segments import assert_nested_equal
from tests.test_part_ev_state_tigris import result_fixture
from tests.test_part_fullscale_model import bounded_cpu_threads


def test_existing_campaign_sources_unchanged_and_full_recipe():
    assert not any('tigris_segments' in name for name in P.B.source_manifest())
    assert P.B.source_manifest().items()<=P.source_manifest().items()
    recipe=P.B.recipes()[P.TASK['recipe']]
    assert (recipe['total_steps'],recipe['epochs'],recipe['global_batch_size'])==(1000000,50,512)
    D=E.create_engine()
    assert D.P is P and D.training_step.__globals__ is vars(D)
    assert P.B.TASKS==[P.TASK] and P.TASK['architecture']=='state_layerwise'
    assert P.B.TRANSFER['authorizes_training'] is False


def test_allocation_deadline_counts_startup_and_requires_72h():
    env=dict(SLURM_JOB_START_TIME='1000',SLURM_JOB_END_TIME=str(1000+259200))
    assert E.allocation_deadline(env,wall_clock=lambda:2800,monotonic=lambda:10)==10+70.5*3600
    for bad in ({},dict(env,SLURM_JOB_END_TIME='nan'),dict(env,SLURM_JOB_END_TIME='2000')):
        with pytest.raises(ValueError): E.allocation_deadline(bad)
    with pytest.raises(ValueError): E.allocation_deadline(env,wall_clock=lambda:300000)
    D=SimpleNamespace(STOP=False)
    with pytest.raises(E.PlannedStop): E.Deadline(D,10,clock=lambda:11).check()
    D.STOP=True
    with pytest.raises(SystemExit): E.Deadline(D,10,clock=lambda:11).check()


@pytest.fixture
def requested(tmp_path,monkeypatch):
    bench=tmp_path/'benchmark';bench.mkdir()
    _,result_path,result=result_fixture(bench)
    parent=dict(content_hash='diagnostic',donor={'path':str(tmp_path/'sporc')},
                recipes=P.B.recipes(),model=P.B.CONFIG,parameters={'state_layerwise':2327462},
                inventory={'same':'files'},environment={'same':'ARM'})
    P.write_json(bench/'campaign.json',parent)
    (bench/'submitted_job_ids.tsv').write_text('prepare\t723340\nbenchmark\t723341\n')
    monkeypatch.setattr(P.B,'load_campaign',lambda root,**kw:parent)
    monkeypatch.setattr(P.B,'verify_inventory_metadata',lambda _:None)
    monkeypatch.setattr(P,'source_manifest',lambda:dict(source='frozen'))
    output=tmp_path/'production'
    P.write_json(output/'launch.json',dict(python='ARM'))
    req=P.prepare(output,bench,'723341')
    chain=output/'segment_chains'/'first'
    plan=P.prepare_chain(chain,output,4)
    return output,chain,plan,bench,result_path,result


def test_submission_can_precede_result_but_admission_cannot(requested):
    output,chain,plan,bench,path,row=requested
    original={p:p.read_bytes() for p in bench.rglob('*') if p.is_file()}
    path.unlink()  # Emulate a still-pending benchmark.
    assert P.load_chain(chain)==plan and not (output/'campaign.json').exists()
    with pytest.raises(FileNotFoundError): P.admit(chain,plan)
    P.write_json(path,P.hashed(dict(row,passed=False)))
    with pytest.raises(ValueError): P.admit(chain,plan)
    assert not (chain/'admitted.json').exists() and not (output/'campaign.json').exists()
    P.write_json(path,P.hashed(row))
    assert P.admit(chain,plan)['passed']
    spec=P.load_campaign(output)
    assert spec['recipes']==P.B.recipes() and spec['tasks']==[P.TASK]
    assert P.expected_runtime(spec)['campaign_sha256']==spec['content_hash']!='diagnostic'
    assert original=={p:p.read_bytes() for p in bench.rglob('*') if p.is_file()}
    P.write_json(path,P.hashed(dict(row,jets_per_second=123)))
    with pytest.raises(ValueError): P.load_campaign(output)


def test_wrong_job_id_and_bad_slot_budget_fail_closed(requested,monkeypatch):
    output,chain,plan,bench,_,_=requested
    with pytest.raises(ValueError,match='ledger'): P.reference(bench,'999')
    original=P.expected_spec
    monkeypatch.setattr(P,'expected_spec',lambda *a,**k:P.hashed(dict(original(*a,**k),recommended_train_hours=500)))
    with pytest.raises(ValueError,match='exceeds'): P.admit(chain,plan)
    assert not (output/'campaign.json').exists()


def test_chain_and_gate_require_committed_completion(requested,monkeypatch):
    output,chain,plan,_,_,_=requested
    P.admit(chain,plan);spec=P.load_campaign(output)
    assert P.previous_snapshot(chain,plan,0)==plan['initial']
    with pytest.raises(FileNotFoundError): P.previous_snapshot(chain,plan,1)
    row=P.hashed(dict(chain_sha256=plan['content_hash'],index=0,status='paused',after=plan['initial']))
    P.write_json(P.receipt_path(chain,0),row)
    assert P.previous_snapshot(chain,plan,1)==plan['initial']
    with pytest.raises(FileExistsError): P.previous_snapshot(chain,plan,0)
    cursor=dict(step=900000,validated_checks=45,checkpoint_sha256='last',complete=False)
    monkeypatch.setattr(P,'snapshot',lambda *a:cursor)
    P.write_json(P.receipt_path(chain,3),P.hashed(dict(row,index=3,after=cursor)))
    with pytest.raises(ValueError,match='incomplete'): P.completion_gate(chain,plan,spec)
    cursor.update(step=1000000,validated_checks=50,complete=True)
    P.write_json(P.receipt_path(chain,3),P.hashed(dict(row,index=3,status='complete',after=cursor)))
    assert P.completion_gate(chain,plan,spec)==cursor
    cursor['checkpoint_sha256']='other'
    with pytest.raises(ValueError): P.completion_gate(chain,plan,spec)


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=torch.nn.Linear(3,10)
        self.drop=torch.nn.Dropout(.2)
        self._counter=0

    def forward(self,points,*rest):
        if self.training:
            self._counter+=1
            # Exercise Python/NumPy state as well as torch dropout.
            import random
            points=points*(.8+.1*random.random()+.1*np.random.random())
        return self.linear(self.drop(points[:,:,0]))


class Loader:
    def __iter__(self):
        gen=torch.Generator().manual_seed(1901)
        while True:
            x=torch.randn(512,3,1,generator=gen)
            yield ({k:x for k in ('pf_points','pf_features','pf_vectors','pf_mask')},
                   {'_label_':torch.arange(512)%10},{})


@pytest.fixture
def tiny(tmp_path,monkeypatch):
    recipe=dict(P.B.recipes()[P.TASK['recipe']],total_steps=12,epochs=3,steps_per_epoch=4,
                validation_events_per_epoch=1536,lr_decay_epochs=[1,2])
    spec=dict(content_hash='test',recipes={P.TASK['recipe']:recipe},inventory={},
              parameters={'state_layerwise':40},tasks=[P.TASK],benchmark_runtime=dict(cpu=True))
    runtime=P.expected_runtime(spec)
    monkeypatch.setattr(P,'require_admission',lambda *a:None)
    monkeypatch.setattr(E,'restart_check',lambda *a:None)  # Tested separately with real optimizer below.
    monkeypatch.setitem(P.POLICY,'checkpoint_interval',2)
    def driver():
        D=E.create_engine()
        def model(*a):
            D.seed_everything(101)
            return Toy()
        D.make_model=model
        D.build_loader=lambda *a,**kw:(Loader(),None)
        D.shutdown_loader=lambda _:None
        return D
    def campaign(name):
        root=tmp_path/name
        P.write_json(root/'campaign.json',spec)
        chain=root/'segment_chains'/'first'
        plan=dict(campaign=str(root),content_hash=name,segments=4,initial=P.snapshot(root,spec))
        return root,chain,plan
    return spec,runtime,driver,campaign


def test_native_training_exact_resume_including_interrupted_validation_and_replay(tiny):
    spec,runtime,driver,campaign=tiny
    device=torch.device('cpu')
    continuous,cchain,cplan=campaign('continuous')
    E.train_segment(cchain,cplan,spec,0,device,runtime,float('inf'),engine=driver())
    output,chain,plan=campaign('segmented')
    D=driver();clock=[0.]
    step=D.training_step
    def stop_after_three(*a,**kw):
        row=step(*a,**kw)
        if a[6]==2: clock[0]=2
        return row
    D.training_step=stop_after_three
    first=E.train_segment(chain,plan,spec,0,device,runtime,1,engine=D,clock=lambda:clock[0])
    assert first['status']=='paused' and first['after']['step']==3
    D=driver();clock[0]=0
    validate=D.validate
    def interrupted(model,iterator,*a,**kw):
        def batches():
            yield next(iterator)
            clock[0]=2
            yield next(iterator)
        return validate(model,batches(),*a,**kw)
    D.validate=interrupted
    # Raise while validation is in progress, preserving its pre-validation snapshot.
    # The validation generator expires after its first batch; the inherited validator
    # next checks the deadline on the subsequent iteration (use 3 batches here).
    second=E.train_segment(chain,plan,spec,1,device,runtime,1,engine=D,clock=lambda:clock[0])
    assert second['status']=='paused' and second['after']['step']==4 and second['after']['validated_checks']==0
    third=E.train_segment(chain,plan,spec,2,device,runtime,float('inf'),engine=driver())
    assert third['status']=='complete'
    expected=trusted_load(P.task_directory(continuous)/'last.pt')
    actual=trusted_load(P.task_directory(output)/'last.pt')
    for key in ('model','optimizer','scaler','rng'):
        assert_nested_equal(actual[key],expected[key])
    for key in ('step','validated_checks','best_check','best_accuracy','amp_skipped_steps'):
        assert actual['state'][key]==expected['state'][key]
    assert len(list((P.task_directory(output)/'validation').glob('check_*.json')))==3
    before=P.file_sha256(P.task_directory(output)/'last.pt')
    assert E.train_segment(chain,plan,spec,3,device,runtime,float('inf'),engine=driver())['status']=='already_complete'
    assert P.file_sha256(P.task_directory(output)/'last.pt')==before


def test_external_signal_and_replay_without_progress_do_not_release_successor(tiny):
    spec,runtime,driver,campaign=tiny
    output,chain,plan=campaign('signal')
    D=driver();step=D.training_step
    def signal(*a,**kw):
        result=step(*a,**kw);D.STOP=True;return result
    D.training_step=signal
    with pytest.raises(SystemExit): E.train_segment(chain,plan,spec,0,torch.device('cpu'),runtime,float('inf'),engine=D)
    assert not P.receipt_path(chain,0).exists()
    assert P.snapshot(output,spec)['step']==1
    # Adopt the saved state into a fresh chain, then use up the allocation during replay.
    plan=dict(plan,content_hash='recovery',initial=P.snapshot(output,spec))
    D=driver()
    with pytest.raises(RuntimeError,match='No committed progress'):
        E.train_segment(chain,plan,spec,0,torch.device('cpu'),runtime,0,engine=D,clock=lambda:1)
    assert not P.receipt_path(chain,0).exists()


def test_validation_candidate_commit_failure_does_not_overwrite_committed_best(tiny,monkeypatch):
    spec,runtime,driver,campaign=tiny
    output,chain,plan=campaign('commit_failure')
    D=driver()
    D.validate=lambda *a,**kw:dict(accuracy=.7 if not (P.task_directory(output)/'selected/check_01.pt').exists() else .8,
                                 seconds=1.,events=1536,macro_auc_ovo=.9,macro_auc_ovr=.9,loss=.1)
    save=D.save_checkpoint
    def fail(path,model,opt,scaler,state,*a):
        if state['validated_checks']==2: raise OSError('write failure')
        save(path,model,opt,scaler,state,*a)
    D.save_checkpoint=fail
    with pytest.raises(OSError): E.train_segment(chain,plan,spec,0,torch.device('cpu'),runtime,float('inf'),engine=D)
    saved=trusted_load(P.task_directory(output)/'last.pt')
    assert saved['state']['validated_checks']==1 and saved['state']['best_check']==1
    assert P.snapshot(output,spec)['step']==8
    assert P.file_sha256(P.task_directory(output)/'selected/check_01.pt')==saved['state']['best_sha256']
    assert (P.task_directory(output)/'selected/check_02.pt').exists()
    assert not P.receipt_path(chain,0).exists()


def test_restart_check_with_real_ranger_and_native_dropout(tmp_path,monkeypatch):
    D=E.create_engine()
    D.make_model=lambda *a:Toy()
    D.build_loader=lambda *a,**kw:(Loader(),None)
    D.shutdown_loader=lambda _:None
    spec=dict(content_hash='restart',inventory={},recipes=P.B.recipes())
    E.restart_check(tmp_path,spec,{'cpu':True},torch.device('cpu'),D)
    assert P.read_json(tmp_path/'restart_check/result.json')['passed']
    before=P.file_sha256(tmp_path/'restart_check/probe.pt')
    E.restart_check(tmp_path,spec,{'cpu':True},torch.device('cpu'),D)
    assert P.file_sha256(tmp_path/'restart_check/probe.pt')==before


def test_recover_final_checkpoint_without_replaying_finished_stream(tiny):
    spec,runtime,driver,campaign=tiny
    output,chain,plan=campaign('final_marker')
    E.train_segment(chain,plan,spec,0,torch.device('cpu'),runtime,float('inf'),engine=driver())
    run=P.task_directory(output)
    checksum=P.file_sha256(run/'last.pt')
    (run/'training_complete.json').unlink()  # Fixture: crash before completion marker.
    (run/'best_model_val.pt').unlink()
    recovery=output/'segment_chains'/'recovery'
    new_plan=dict(plan,content_hash='recover',segments=1,initial=P.snapshot(output,spec))
    D=driver()
    def no_loader(*a,**kw): raise AssertionError('Finished schedule must not replay its data')
    D.build_loader=no_loader
    assert E.train_segment(recovery,new_plan,spec,0,torch.device('cpu'),runtime,float('inf'),engine=D)['status']=='complete'
    assert P.file_sha256(run/'last.pt')==checksum
    assert P.completion_gate(recovery,new_plan,spec)['complete']


def test_all_fifty_checks_and_first_best_tie_are_required(tiny):
    spec,runtime,driver,campaign=tiny
    spec['recipes'][P.TASK['recipe']].update(total_steps=100,steps_per_epoch=2,epochs=50)
    output,chain,plan=campaign('fifty')
    D=driver();checks=[]
    def validation(*a,**kw):
        checks.append(len(checks)+1)
        return dict(accuracy=.8 if len(checks) in (40,50) else .7,seconds=1.,events=1536,
                    macro_auc_ovo=.9,macro_auc_ovr=.9,loss=.1)
    D.validate=validation
    E.train_segment(chain,plan,spec,0,torch.device('cpu'),runtime,float('inf'),engine=D)
    done=P.read_json(P.task_directory(output)/'training_complete.json')
    assert (done['step'],done['validated_checks'],done['best_check'])==(100,50,40)
    assert len(list((P.task_directory(output)/'validation').glob('check_*.json')))==50
    done['validated_checks']=49
    P.write_json(P.task_directory(output)/'training_complete.json',done)
    with pytest.raises(ValueError): P.snapshot(output,spec)


def test_checkpoint_runtime_and_required_restart_components_checked(tiny):
    spec,runtime,driver,campaign=tiny
    output,chain,plan=campaign('invalid_checkpoint')
    E.train_segment(chain,plan,spec,0,torch.device('cpu'),runtime,float('inf'),engine=driver())
    last=P.task_directory(output)/'last.pt'
    saved=trusted_load(last)
    for change in (lambda r:r.update(schema='sporc'),lambda r:r['runtime'].update(world_size=2),
                   lambda r:r.pop('optimizer'),lambda r:r['state'].update(step=13),
                   lambda r:r['state'].update(best_file='../outside.pt')):
        bad=copy.deepcopy(saved);change(bad)
        torch.save(bad,last)
        with pytest.raises(ValueError): P.snapshot(output,spec)
