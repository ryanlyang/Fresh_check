"""Real CPU math/restart tests; these do not stand in for GPU/DDP admission."""
import copy
import math
import pytest
import torch
from part_ev_multiscale import verification as V, runtime as R, restart as S, engine as E
from part_ev_multiscale import diagnostics as D
from part_ev_fullscale.checkpoint import capture_rank_state, restore_rank_state, trusted_load, atomic_torch_save
from part_ev_multiscale import protocol as P
from part_ev_multiscale.config import COUNTS,recipe

torch.set_num_threads(1)
DEVICE=torch.device('cpu')


def test_full_independent_stage_and_attention_oracles():
    proof=V.oracle(DEVICE)
    assert proof['full_stage']['gradients'] and proof['full_stage']['padding']


def test_native_pipeline_identity_and_rng():
    row=V.initial_identity(DEVICE,0,1)
    assert row['native_pipeline']==dict(trimmer=1,embedding=1,pairs=1,classifier=1,
        class_attention_0=1,class_attention_1=1)


@pytest.mark.parametrize('arm',['control','multiscale'])
def test_active_graph_distributed_oracle_and_coordinated_failures(arm):
    assert V.active_parity(arm,DEVICE,0,1)['passed']
    assert V.distributed(arm,DEVICE,0,1)['passed']
    assert V.rejection(arm,DEVICE,0,1)['passed']


def test_real_ranger_seven_update_repeat_disk_and_resume(tmp_path):
    base,model,opt,scaler,learning=V.learning('multiscale',DEVICE,0,1)
    proof=S.probe(base,model,opt,scaler,DEVICE,0,1,tmp_path)
    assert learning['successful_updates']==8 and proof['passed']
    assert all(len(t['updates'])==7 for t in proof['trials'])
    assert all(t['max_state_error']<=1e-4 for t in proof['trials'])


def test_full_training_checkpoint_roundtrip_and_cursor_rejections(tmp_path):
    base,model,opt,scaler=R.fresh('multiscale',DEVICE,1,trim=True)
    values,y=R.tensors(R.fixed_batch(8,7),DEVICE)
    for i in range(2):R.step(base,model,opt,scaler,values,y,i,1,amp=False)
    state=E.initial_state('multiscale');state['step']=2
    spec=dict(content_hash='cpu_fixture',runtime=dict(platform='cpu_test'))
    path=tmp_path/'last.pt'
    before=S.snapshot(base,opt,scaler)
    E.save(path,base,opt,scaler,state,spec,0,1)
    R.step(base,model,opt,scaler,values,y,2,1,amp=False)
    restored,rng=E.restore(path,base,opt,scaler,spec,'multiscale',0,1)
    restore_rank_state(base,rng)
    assert restored==state
    S.compare(S.snapshot(base,opt,scaler),before,exact=True)
    row=trusted_load(path)
    cases=[lambda r:r.update(runtime={}),
        lambda r:r['state'].update(step=40001,validated_checks=0),
        lambda r:r['state'].update(best_file='../../foreign.pt',best_check=1),
        lambda r:r['state'].update(amp_skipped_steps=3),
        lambda r:r['state'].update(train_seconds=float('nan')),
        lambda r:r['optimizer'].pop('slow')]
    for mutate in cases:
        bad=copy.deepcopy(row);mutate(bad);atomic_torch_save(bad,tmp_path/'bad.pt')
        with pytest.raises((ValueError,KeyError)):
            E.restore(tmp_path/'bad.pt',base,opt,scaler,spec,'multiscale',0,1)


def test_bounded_diagnostics_restore_parameters_rng_and_modes():
    base,_,_,_=R.fresh('multiscale',DEVICE,1,trim=True);V.activate(base)
    before={n:p.detach().clone() for n,p in base.named_parameters()}
    rng=capture_rank_state(base);base.train()
    row=D.sample(base,DEVICE)
    assert len(row['layers'])==4 and len(row['attention_entropy'])==22
    assert all(math.isfinite(x) for x in row['disabled_logit_changes'].values())
    assert base.training and base.mod.trimmer.enabled
    S.compare(capture_rank_state(base),rng,exact=True)
    for n,p in base.named_parameters():torch.testing.assert_close(p,before[n],rtol=0,atol=0)


def test_bounded_profile_attribution_cpu():
    for arm in ('control','multiscale'):
        row=D.profile(arm,DEVICE)
        assert len(row)==2
        if arm=='multiscale':
            assert all(len(r['stages'])==46 for r in row)
            assert all(s['cpu_us']>0 for r in row for s in r['stages'].values())


def test_fifty_check_selection_audit_and_first_tie(tmp_path):
    from part_ev_fullscale.optimizer import optimizer_metadata
    base,_,opt,_=R.fresh('multiscale',DEVICE,1)
    spec=dict(content_hash='cpu_fixture',runtime=dict(platform='cpu_test'),
        recipes={'ranger_full_v1':recipe()},parameters=COUNTS)
    root=tmp_path/'production';run=P.run_directory(tmp_path,'multiscale')
    P.write(root/'campaign.json',spec)
    P.write(run/'optimizer.json',optimizer_metadata(base,opt))
    rows=[]
    for check in range(1,51):
        row=dict(check=check,step=20000*check,events=1280000,
            accuracy=.9 if check in (12,49) else .8,task=P.task('multiscale'),campaign_sha256=spec['content_hash'])
        P.write(run/'validation'/f'check_{check:02d}.json',row);rows.append(row)
    atomic_torch_save(dict(model=base.state_dict(),task=P.task('multiscale'),runtime=E.expected_runtime(spec),
        campaign_sha256=spec['content_hash'],validation=rows[11]),run/'best_model_val.pt')
    state=E.initial_state('multiscale')
    state['geometry_diagnostics'].update(sampled_attempts=10000,last_ranks=[dict(schema='bounded_geometry_v1')])
    state.update(step=1000000,validated_checks=50,best_check=12,best_accuracy=.9,
        campaign_sha256=spec['content_hash'],runtime=E.expected_runtime(spec),
        best_checkpoint_sha256=P.sha(run/'best_model_val.pt'))
    P.write(run/'training_complete.json',state)
    assert E.audit_training(tmp_path,spec,'multiscale')['best_check']==12
    state['best_check']=49;P.write(run/'training_complete.json',state)
    with pytest.raises(ValueError,match='first highest'):E.audit_training(tmp_path,spec,'multiscale')


def test_enabled_scaler_state_can_be_authenticated_on_cpu(tmp_path):
    base,_,opt,_=R.fresh('multiscale',DEVICE,1)
    scaler=torch.amp.GradScaler('cpu',enabled=True)
    spec=dict(content_hash='cpu_fixture',runtime={})
    E.save(tmp_path/'last.pt',base,opt,scaler,E.initial_state('multiscale'),spec,0,1)
    check=torch.amp.GradScaler('cpu',enabled=True)
    E.restore(tmp_path/'last.pt',base,opt,check,spec,'multiscale',0,1)
    assert check.state_dict()==scaler.state_dict()


def test_root_only_work_exchanges_heartbeats_and_propagates_errors(monkeypatch):
    import time
    packets=[]
    monkeypatch.setattr(R.dist,'broadcast_object_list',lambda rows,src:packets.append(rows[0]))
    assert R.root_call(lambda:(time.sleep(.01),42)[1],0,2)==42
    assert None in packets and packets[-1]==(True,42)
    with pytest.raises(RuntimeError,match='fixture'):
        R.root_call(lambda:(_ for _ in ()).throw(OSError('fixture')),0,2)


def test_multiscale_amp_and_fp32_validation_contexts(emulate_cuda_autocast):
    from part_ev_screen.precision import model_precision,audit_precision
    base,_,opt,_=R.fresh('multiscale',DEVICE,1,trim=False);V.activate(base)
    values,y=R.tensors(R.fixed_batch(8,7),DEVICE)
    observed=[]
    handles=[s.fine_ffn_norm.register_forward_hook(lambda m,a,o:observed.append(o.dtype))
             for s in base.mod.multiscale.stages]
    try:
        with model_precision(base,torch.device('cuda'),training=True),audit_precision(base,expected_dtype=torch.float16) as audit:
            loss=torch.nn.functional.cross_entropy(base(*values),y)
        loss.backward()
        assert audit['passed'] and all(d==torch.float32 for d in observed)
        assert all(p.dtype==torch.float32 and p.grad is not None and torch.isfinite(p.grad).all() for p in base.parameters())
        with torch.no_grad(),model_precision(base,torch.device('cuda'),training=False),audit_precision(base,expected_dtype=torch.float32) as validation:
            base.eval();base(*values)
        assert validation['passed'] and not base.mod.use_amp
    finally:
        for h in handles:h.remove()


# Reuse CPU emulation of CUDA autocast only; actual target-GPU gates still run
# in the queued paired benchmark and cannot be satisfied by this fixture.
from tests.test_part_ev_screen_precision import emulate_cuda_autocast

