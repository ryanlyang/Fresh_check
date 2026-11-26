import copy
from types import SimpleNamespace
import pytest
from tests.test_part_ev_masked_predictive_protocol import evidence
from part_ev_masked_fast_production import protocol as P
from part_ev_masked_fast_production.config import SCHEMA,POLICY,SCIENCE,RESOURCES,TASK


def fixture():
    _,old,_=evidence('tigris');m=old['models'][1]
    launch=dict(content_hash='launch',gpu=old['runtime']['gpu'],choice=dict(provisional_train_hours=255),gradient_count=2)
    row=dict(schema=SCHEMA,launch_sha256='launch',complete=True,authorizes_training=False,policy=POLICY,
        gpu=launch['gpu'],parity=[dict(passed=True,repeat=i,amp=a,microbatch=256,batch=512,particles=97,gradients=2,
            max_state_absolute=0.,max_gradient_absolute=0.,exact_masks_rng_scaler=True,exact_bn_buffers=True,
            full_optimizer_teacher=True,varied_lengths=True) for i in range(2) for a in (False,True)],
        legacy_fast32=dict(collected=True,trials=[dict(amp=a,passed=False,error='historical diagnostic only') for a in (False,True)]),
        restart=m['correctness'][0]['restart'],transitions=m['correctness'][0]['transitions'],
        real_stream_resume=dict(passed=True,real_stream=True,full_state=True,exact_roundtrip=True,next_masks_equal=True,
            next_inputs_equal=True,optimizer_teacher_equal=True,rng_restored=True,updates=7,train_replayed=3,val_replayed=2,
            validation_boundaries=2,input_sha256=['a'*64]*7),
        production_checkpoint=dict(passed=True,complete_attempts=3,exact_roundtrip=True,production_serializer=True),
        peak_allocated=35*2**30,finite_state=True)
    for key in ('stream','clean','fixed','checkpoint','validation','inference','metric_5m_seconds','cold_fetch_seconds'):
        row[key]=copy.deepcopy(m[key])
    for key in ('stream','clean','fixed'):row[key]['microbatch']=256
    return launch,row


def test_candidate_admission_uses_measured_timing_and_prior_floor():
    launch,row=fixture();result=P.validate_evidence(launch,row)
    assert result['passed'] and not result['authorizes_training'] and result['train_hours']>=255
    assert result['evaluate_hours']>=2 and result['model']==TASK['run_id']
    launch['choice']['provisional_train_hours']=400
    assert P.validate_evidence(launch,row)['train_hours']==400
    assert RESOURCES['gpus']==1 and RESOURCES['microbatch']==256 and SCIENCE['auxiliary_microbatch']==256


@pytest.mark.parametrize('change',[
    lambda r:r.update(complete=False),
    lambda r:r.update(authorizes_training=True),
    lambda r:r['parity'].pop(),
    lambda r:r['parity'][0].update(exact_bn_buffers=False),
    lambda r:r['parity'][1].update(gradients=1),
    lambda r:r['parity'][2].update(microbatch=32),
    lambda r:r['parity'][3].update(max_state_absolute=.01),
    lambda r:r['legacy_fast32'].update(collected=False),
    lambda r:r['real_stream_resume'].update(next_masks_equal=False),
    lambda r:r['real_stream_resume'].update(val_replayed=0),
    lambda r:r['production_checkpoint'].update(production_serializer=False),
    lambda r:r['stream'].update(microbatch=128),
    lambda r:r['stream'].update(measured_steps=300),
    lambda r:r['stream'].update(auxiliary_selected=128),
    lambda r:r['stream']['ranks'][0].update(teacher_precision=None),
    lambda r:r['stream']['ranks'][0].update(skips=1000),
    lambda r:r['clean']['ranks'][0].update(mean_auxiliary=.1),
    lambda r:r['restart']['trials'][1]['updates'][0].update(mask_sha256='x'*64),
    lambda r:r['validation']['precision'].update(passed=False),
    lambda r:r.update(peak_allocated=96*2**30),
    lambda r:r.update(finite_state=False),
    lambda r:r.update(finite_state='true'),
    lambda r:r['checkpoint'].update(full_state_includes_transfer=False),
    lambda r:r['inference']['latency'].update(particles=17),
])
def test_incomplete_or_weakened_production_evidence_rejected(change):
    launch,row=fixture();change(row)
    with pytest.raises((ValueError,KeyError)):P.validate_evidence(launch,row)


def test_oversized_or_failed_preflight_cannot_make_production_plan(tmp_path,monkeypatch):
    launch=dict(max_train_hours=480,max_evaluate_hours=24)
    monkeypatch.setattr(P,'load',lambda *a,**k:launch)
    for result in (dict(passed=False),dict(passed=True,train_hours=481,evaluate_hours=2),
                   dict(passed=True,train_hours=255,evaluate_hours=25)):
        monkeypatch.setattr(P,'review',lambda *a,r=result:r)
        with pytest.raises(ValueError,match='no training submitted'):P.production_spec(tmp_path)
    assert not list(tmp_path.iterdir())


def test_storage_own_budget_and_reserve_only_no_deletion(tmp_path,monkeypatch):
    target=tmp_path/'new';keep=tmp_path/'keep';keep.write_text('keep')
    monkeypatch.setattr(P.shutil,'disk_usage',lambda *a:SimpleNamespace(free=1))
    with pytest.raises(OSError):P.check_storage(target)
    assert keep.read_text()=='keep' and not target.exists()
    monkeypatch.setattr(P.shutil,'disk_usage',lambda *a:SimpleNamespace(free=POLICY['reserve_bytes']+POLICY['budget_bytes']+1))
    assert P.check_storage(target)==POLICY['reserve_bytes']+POLICY['budget_bytes']
