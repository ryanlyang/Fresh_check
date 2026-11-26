from pathlib import Path
import pytest
import torch
from part_ev_masked_predictive import runtime as R, restart as K, engine as OLD
from part_ev_masked_speed.step import attempt as benchmarked
from part_ev_masked_fast_production import execution as X, verification as V, protocol as P

torch.set_num_threads(1);CPU=torch.device('cpu')


@pytest.mark.parametrize('label',[20000,40000,900000])
def test_production_wrapper_preserves_exact_measured_step_with_latent_logging(label):
    s=R.fresh('masked',CPU,1,trim=False);initial=K.capture(s);x,y=R.tensors(R.fixed_batch(16,17),CPU)
    old=benchmarked(s,x,y,microbatch=256,schedule_label=label,diagnostic=True,audit=True,amp=False)
    expected=K.capture(s);K.restore(s,initial)
    actual=X.attempt(s,x,y,inspect_latents=True,schedule_label=label,diagnostic=True,audit=True,amp=False)
    K.compare(K.capture(s),expected,exact=True)
    assert all(old[k]==actual[k] for k in ('loss','ce','auxiliary','mask_sha256','eligible','targets','scale'))
    assert (actual['latents'] is not None)==(label==40000)
    assert not s.base._forward_hooks and not s.base._forward_pre_hooks


def test_latent_hooks_removed_after_exception(monkeypatch):
    s=R.fresh('masked',CPU,1,trim=False)
    def fail(*a,**k):raise RuntimeError('fixture')
    monkeypatch.setattr(X,'fast_attempt',fail)
    with pytest.raises(RuntimeError):X.attempt(s,None,None,inspect_latents=True)
    assert not s.base._forward_hooks and not s.base._forward_pre_hooks


def test_private_engine_does_not_mutate_historical_runner():
    old_step,old_schema,old_task=OLD.attempt,OLD.P.SCHEMA,OLD.P.task('masked')
    E=X.engine()
    assert E.attempt is X.attempt and E.P.SCHEMA==P.SCHEMA
    assert E.train.__globals__['save'] is E.save
    assert E.save.__globals__['P'] is E.P
    assert E.P.task('masked')['run_id']!=old_task['run_id']
    assert OLD.attempt is old_step and OLD.P.SCHEMA==old_schema and OLD.P.task('masked')==old_task
    with pytest.raises(ValueError):E.P.task('control')


def test_actual_production_serializer_and_wrong_identity(tmp_path):
    spec=P.hashed(dict(runtime=dict(gpu='fixture'),diagnostic=True))
    assert V.production_checkpoint(tmp_path,spec,CPU)['passed']
    E=X.engine();s=V.fresh(CPU)
    with pytest.raises(ValueError):E.restore(tmp_path/'probes/production_restart/last.pt',s,dict(spec,content_hash='wrong'),'masked')


def test_named_production_gradient_gate_and_transition_restart(tmp_path):
    result=V.parity(CPU,amp=False,batch=80,particles=17)
    assert result['passed'] and result['microbatch']==256
    result=V.restart_probe(V.fresh(CPU),tmp_path/'probe',transitions=True)
    from part_ev_masked_predictive.evidence import restart
    restart(result,'masked',True)


def test_real_stream_resume_with_deterministic_fixture_loaders(tmp_path,monkeypatch):
    # Local exercise of iterator/replay plumbing; actual ROOT workers are checked on GH200.
    calls=[]
    class Loader:
        def __init__(self,split):self.split=split
        def __iter__(self):
            for i in range(30):
                batch=R.fixed_batch(8,9+i%3)
                batch[1]['_label_']=(batch[1]['_label_']+i)%10
                yield batch
    def build(split,*a,**k):calls.append(split);return Loader(split),None
    monkeypatch.setattr(V,'build_loader',build);monkeypatch.setattr(V,'shutdown_loader',lambda *a:None)
    class E:
        def validate(self,base,iterator,device,rank,world,events,audit=False):
            for _ in range(2):next(iterator)
            return dict(accuracy=.8,macro_auc_ovo=.98)
    monkeypatch.setattr(V,'engine',E)
    result=V.real_stream_resume(tmp_path,dict(inventory={}),dict(workers=2),CPU)
    assert result['passed'] and result['train_replayed']==3 and result['val_replayed']==2
    assert calls==['train','val','train','val'] and len(result['input_sha256'])==7


def test_private_training_loop_stops_with_complete_fast256_state(tmp_path,monkeypatch):
    E=X.engine();spec=P.hashed(dict(runtime={},resources=dict(local_batch=8,workers=0),inventory={}))
    batches=[R.fixed_batch(8,9) for _ in range(3)]
    monkeypatch.setattr(E,'build_loader',lambda *a,**k:(batches,None))
    monkeypatch.setattr(E,'shutdown_loader',lambda *a:None)
    original=E.attempt;calls=[]
    def step(*a,**k):
        result=original(*a,**k);calls.append(result);return result
    E.attempt=step
    monkeypatch.setattr(R,'stopping',lambda *a:len(calls)>=2)
    with pytest.raises(SystemExit) as error:E.train(tmp_path,spec,'masked',CPU,0,1)
    assert error.value.code==75 and len(calls)==2
    state,_=E.restore(P.run_directory(tmp_path)/'last.pt',R.fresh('masked',CPU,1),spec,'masked')
    assert state['step']==2 and state['auxiliary_attempts']==0


def test_fifty_checks_selected_student_and_first_tie_with_new_identity(tmp_path):
    from part_ev_masked_predictive.config import recipe
    from part_ev_fullscale.checkpoint import atomic_torch_save
    from part_ev_fullscale.optimizer import optimizer_metadata
    E=X.engine();s=R.fresh('masked',CPU,1,trim=False)
    spec=dict(content_hash='new_fixture',runtime={},recipes={'ranger_full_v1':recipe()},parameters={'masked':2327462})
    root=tmp_path/'production';run=P.run_directory(tmp_path);P.write(root/'campaign.json',spec)
    P.write(run/'optimizer.json',optimizer_metadata(s.base,s.opt));checks=[]
    for i in range(1,51):
        row=dict(check=i,step=i*20000,events=1280000,accuracy=.9 if i in (12,49) else .8,
                 task=P.task('masked'),campaign_sha256=spec['content_hash'])
        P.write(run/'validation'/f'check_{i:02d}.json',row);checks.append(row)
    atomic_torch_save(dict(model=s.base.student.state_dict(),task=P.task('masked'),runtime=E.expected_runtime(spec),
        campaign_sha256=spec['content_hash'],validation=checks[11]),run/'best_model_val.pt')
    state=E.initial_state('masked');state.update(step=1000000,validated_checks=50,successful_updates=1000000,
        auxiliary_attempts=879999,ema_successes=1000000,best_check=12,best_accuracy=.9,campaign_sha256=spec['content_hash'],
        runtime=E.expected_runtime(spec),best_checkpoint_sha256=P.sha(run/'best_model_val.pt'))
    P.write(run/'training_complete.json',state)
    assert E.audit_training(tmp_path,spec,'masked')['best_check']==12
    state['best_check']=49;P.write(run/'training_complete.json',state)
    with pytest.raises(ValueError,match='first highest'):E.audit_training(tmp_path,spec,'masked')
