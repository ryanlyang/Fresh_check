import copy
import math
import pytest
import torch
from part_ev_masked_predictive import runtime as R,verification as V,restart as S,engine as E,protocol as P,diagnostics as D,benchmark as B
from part_ev_masked_predictive.step import attempt
from part_ev_masked_predictive.config import recipe,COUNTS,TRAINABLE,PLATFORMS
from part_ev_fullscale.checkpoint import trusted_load,atomic_torch_save,restore_rank_state,capture_rank_state
from part_ev_fullscale.optimizer import optimizer_metadata

torch.set_num_threads(1);DEVICE=torch.device('cpu')


@pytest.mark.parametrize('transitions',[False,True])
def test_real_ranger_full_state_repeat_disk_and_resume_with_masks_ema_and_boundaries(tmp_path,transitions):
    s,_=V.learning(DEVICE,0,1);proof=S.probe(s,tmp_path,transitions=transitions)
    assert proof['passed'] and all(len(t['updates'])==7 for t in proof['trials'])
    assert all(t['max_state_error']<=1e-4 for t in proof['trials'])
    if transitions:assert proof['labels']==[19999,20000,20001,39999,40000,899999,900000]


@pytest.mark.parametrize('arm',COUNTS)
def test_production_complete_restart_roundtrip_and_rejects_partial_cursors(tmp_path,arm):
    s=R.fresh(arm,DEVICE,1,trim=False);x,y=R.tensors(R.fixed_batch(8,9),DEVICE)
    for _ in range(2):attempt(s,x,y,amp=False)
    before=S.capture(s);state=E.initial_state(arm)
    state.update(step=2,successful_updates=2,ema_successes=2 if arm=='masked' else 0)
    spec=dict(content_hash='fixture',runtime={});path=tmp_path/'last.pt'
    E.save(path,s,state,spec);attempt(s,x,y,amp=False)
    restored,rng=E.restore(path,s,spec,arm);restore_rank_state(s.base,rng)
    assert state==restored;S.compare(S.capture(s),before,exact=True)
    row=trusted_load(path)
    mutations=[lambda q:q.update(runtime={}),lambda q:q['session'].update(last_schedule_label=40000),
        lambda q:q['state'].update(step=40001),lambda q:q['state'].update(successful_updates=0),
        lambda q:q['session']['model'].pop(next(iter(q['session']['model']))),
        lambda q:q['session']['optimizer'].pop('slow')]
    if arm=='masked':mutations.extend([lambda q:q['session'].pop('teacher'),
        lambda q:q['rank_states'][0].update(generators=None),lambda q:q['session']['teacher'].update(successes=999)])
    for modify in mutations:
        bad=copy.deepcopy(row);modify(bad);atomic_torch_save(bad,tmp_path/'bad.pt')
        with pytest.raises((ValueError,KeyError,TypeError,RuntimeError)):
            E.restore(tmp_path/'bad.pt',s,spec,arm)


def test_selected_student_export_and_fifty_check_first_tie_audit(tmp_path):
    s=R.fresh('masked',DEVICE,1,trim=False)
    spec=dict(content_hash='fixture',runtime={},recipes={'ranger_full_v1':recipe()},parameters=COUNTS)
    root=tmp_path/'production';run=P.run_directory(tmp_path,'masked');P.write(root/'campaign.json',spec)
    P.write(run/'optimizer.json',optimizer_metadata(s.base,s.opt));checks=[]
    for i in range(1,51):
        r=dict(check=i,step=i*20000,events=1280000,accuracy=.9 if i in (12,49) else .8,
            task=P.task('masked'),campaign_sha256=spec['content_hash'])
        P.write(run/'validation'/f'check_{i:02d}.json',r);checks.append(r)
    atomic_torch_save(dict(model=s.base.student.state_dict(),task=P.task('masked'),runtime=E.expected_runtime(spec),
        campaign_sha256=spec['content_hash'],validation=checks[11]),run/'best_model_val.pt')
    state=E.initial_state('masked');state.update(step=1000000,validated_checks=50,successful_updates=1000000,
        auxiliary_attempts=879999,ema_successes=1000000,best_check=12,best_accuracy=.9,campaign_sha256=spec['content_hash'],
        runtime=E.expected_runtime(spec),best_checkpoint_sha256=P.sha(run/'best_model_val.pt'))
    P.write(run/'training_complete.json',state);assert E.audit_training(tmp_path,spec,'masked')['best_check']==12
    state['best_check']=49;P.write(run/'training_complete.json',state)
    with pytest.raises(ValueError,match='first highest'):E.audit_training(tmp_path,spec,'masked')


def test_bounded_profiles_latent_diagnostics_preserve_training_state():
    s=R.fresh('masked',DEVICE,1,trim=True);before=S.capture(s)
    row=D.sample(s);assert row['target_variance']>=0 and row['zero_kv_max_change']>=0
    S.compare(S.capture(s),before,exact=True)
    for arm in COUNTS:
        p=D.profile(arm,DEVICE);assert len(p)==2
        if arm=='masked':assert D.TRAIN_SCOPES.issubset(p[0]['stages'])


def test_disposable_full_state_and_student_prediction_io(tmp_path):
    s=R.fresh('masked',DEVICE,1,trim=False);row=B.io(s,tmp_path)
    assert row['student_only'] and row['prediction_events']==100000
    assert row['full_state_includes_transfer'] and row['restart_seconds']>row['transfer_seconds']>0
    assert all(v>0 for k,v in row.items() if k.endswith('_seconds'))
    selected=trusted_load(tmp_path/'probes/masked/io_student.pt')['model']
    assert set(selected)==set(s.base.student.state_dict()) and not any(k.startswith('predictor.') for k in selected)


def test_root_replay_wiring_does_not_draw_aux_rng_or_advance_models(monkeypatch):
    created=[];closed=[]
    def loader(split,*a,**k):created.append(split);return [R.fixed_batch(8,9+i) for i in range(3)],None
    monkeypatch.setattr(B,'build_loader',loader);monkeypatch.setattr(B,'shutdown_loader',lambda x:closed.append(1))
    rng=capture_rank_state(torch.nn.Identity());row=B.replay(dict(inventory={}),dict(resources=PLATFORMS['tigris']),0,1,DEVICE)
    assert row['passed'] and row['no_aux_draws_during_replay'] and created==['train','train','val','val'] and len(closed)==4
    assert all(x['next_target_verified'] and x['next_target_events']==1 for x in row['streams'].values())
    S.compare(capture_rank_state(torch.nn.Identity()),rng,exact=True)


def test_training_latent_logging_does_not_change_update_or_rng():
    s=R.fresh('masked',DEVICE,1,trim=False);x,y=R.tensors(R.fixed_batch(8,9),DEVICE);before=S.capture(s)
    plain=attempt(s,x,y,schedule_label=40000,diagnostic=True,amp=False);expected=S.capture(s)
    S.restore(s,before)
    logged=attempt(s,x,y,schedule_label=40000,diagnostic=True,amp=False,inspect_latents=True)
    S.compare(S.capture(s),expected,exact=True)
    assert plain['latents'] is None and all(math.isfinite(x) for x in logged['latents'].values())
    for key in ('loss','ce','auxiliary','eligible','targets','modes','skipped'):assert plain[key]==logged[key]


def test_coherent_rejection_and_stop():assert V.rejection(DEVICE,0,1,'masked')['rank_local_io']


def test_schedule_override_requires_explicit_diagnostic_identity():
    s=R.fresh('masked',DEVICE,1,trim=False);x,y=R.tensors(R.fixed_batch(8,9),DEVICE)
    with pytest.raises(ValueError,match='diagnostic'):attempt(s,x,y,schedule_label=40000,amp=False)


@pytest.mark.parametrize('arm',COUNTS)
def test_production_loop_cooperative_stop_commits_complete_real_attempts(tmp_path,monkeypatch,arm):
    s=R.fresh(arm,DEVICE,1,trim=False);closed=[]
    monkeypatch.setattr(E.R,'fresh',lambda *a,**k:s)
    monkeypatch.setattr(E.R,'stopping',lambda *a:s.attempts>=2)
    monkeypatch.setattr(E,'build_loader',lambda *a,**k:([R.fixed_batch(8,9) for _ in range(3)],None))
    monkeypatch.setattr(E,'shutdown_loader',lambda loader:closed.append(loader))
    spec=dict(content_hash='fixture',runtime={},resources=PLATFORMS['tigris'],inventory={})
    with pytest.raises(SystemExit) as stopped:E.train(tmp_path,spec,arm,DEVICE,0,1)
    assert stopped.value.code==75 and len(closed)==2
    run=P.run_directory(tmp_path,arm);saved=trusted_load(run/'last.pt')
    assert saved['state']['step']==saved['state']['successful_updates']==saved['session']['attempts']==2
    assert saved['session']['last_schedule_label']==2 and saved['state']['validated_checks']==0
    assert not (run/'training_complete.json').exists() and not (run/'best_model_val.pt').exists()
    if arm=='masked':
        assert saved['session']['teacher']['buffer_copies']==2 and saved['session']['teacher']['successes']==2
        assert saved['rank_states'][0]['generators']['draws']==0


def test_cpu_emulated_fp16_teacher_source_predictor_precision(emulate_cuda_autocast):
    # Emulate only CUDA autocast on CPU; this is never target-device evidence.
    from part_ev_screen.precision import model_precision,audit_precision
    s=R.fresh('masked',DEVICE,1,trim=False);x,y=R.tensors(R.fixed_batch(8,9),DEVICE)
    with model_precision(s.base.student,torch.device('cuda'),training=True),audit_precision(s.base.student,expected_dtype=torch.float16) as p:
        out=s.base.student(*x)
    out.square().mean().backward();assert p['passed']


from tests.test_part_ev_screen_precision import emulate_cuda_autocast
