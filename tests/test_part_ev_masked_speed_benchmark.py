import copy
from types import SimpleNamespace
import pytest
import torch
from part_ev_masked_speed import benchmark as B, protocol as D, verification as V
from part_ev_masked_speed.config import SCHEMA, VARIANTS, POLICY

torch.set_num_threads(1)


def test_real_variable_length_gradients_and_restart(tmp_path):
    variant=dict(name='test64',optimized=True,microbatch=64)
    result=V.parity(variant,torch.device('cpu'),amp=False,batch=80,particles=17)
    assert result['passed'] and result['exact_bn_buffers'] and result['varied_lengths']
    result=V.restart(variant,torch.device('cpu'),tmp_path/'restart.pt')
    assert result['passed'] and result['lookahead_crossed']
    assert (tmp_path/'restart.pt').is_file()


def test_measure_uses_warmup_and_worst_window_including_supplier(monkeypatch):
    ticks=iter([0.,3.,7.,12.]); seen=[]
    monkeypatch.setattr(B.time,'monotonic',lambda:next(ticks))
    monkeypatch.setattr(B.R,'synchronize',lambda *a:None)
    monkeypatch.setattr(B.R,'stopping',lambda *a:False)
    def execute(s,x,y,v,**kw):
        seen.append(kw)
        return dict(skipped=0,eligible=1,precision={},auxiliary_precision={},rng_isolated=kw['audit'])
    monkeypatch.setattr(B,'execute',execute)
    row=B.measure(SimpleNamespace(device='cpu'),lambda i:(None,None,str(i)),VARIANTS[1],
                  phase='active',steps=6,windows=3,warmup=2)
    assert len(seen)==8 and [r['audit'] for r in seen]==[True]+[False]*7
    assert row['seconds_per_attempt']==2.5 and row['total_seconds']==12
    assert [w['seconds'] for w in row['windows']]==[3,4,5]
    assert seen[-1]['schedule_label']==100007


def valid_item(variant):
    timing=lambda n,w:dict(steps=n,warmup=POLICY['warmup'],windows=[dict(steps=n//w,seconds=2*n/w) for _ in range(w)],
        amp_skips=0,seconds_per_attempt=2.,total_seconds=2*n,stream_sha256='a'*64,rng_isolated=True)
    return dict(**variant,passed=True,restart=dict(passed=True,exact_roundtrip=True,full_state=True,
        lookahead_crossed=True,updates=7,schedule_boundaries=[19999,20000,20001,39999,40000,899999,900000]),
        parity=[] if not variant['optimized'] else [dict(passed=True,amp=a,batch=POLICY['parity_batch'],
            particles=POLICY['parity_particles'],gradients=500,max_state_absolute=0.,max_gradient_absolute=0.,
            exact_masks_rng_scaler=True,exact_bn_buffers=True,varied_lengths=True,full_optimizer_teacher=True)
            for a in (False,True)],finite_state=True,peak_gib=35.,
        stages={s:dict(calls=1,cpu_us=1.,device_us=1.) for s in ('masked.clean','masked.teacher',
            'masked.visible_predictor','masked.auxiliary_backward','masked.optimizer','masked.ema')},
        active=timing(POLICY['measured'],POLICY['windows']),clean=timing(POLICY['clean_steps'],1),
        fixed128=timing(POLICY['fixed_steps'],1))


GPU=dict(total_memory=95*2**30,name='NVIDIA GH200',driver='fixture')


@pytest.mark.parametrize('change',[
    lambda r:r.update(microbatch=7),
    lambda r:r.update(finite_state=False),
    lambda r:r.update(peak_gib=90),
    lambda r:r.update(parity=r['parity'][:1]),
    lambda r:r['parity'][0].update(exact_bn_buffers=False),
    lambda r:r['parity'][0].update(max_state_absolute=1.),
    lambda r:r['restart'].update(full_state=False),
    lambda r:r['restart'].update(updates=1),
    lambda r:r['active'].update(steps=1),
    lambda r:r['active'].update(amp_skips=300),
    lambda r:r['active'].update(seconds_per_attempt=float('nan')),
    lambda r:r['active'].update(seconds_per_attempt=1.),
    lambda r:r['active'].update(total_seconds=1.),
    lambda r:r['active'].update(windows=[]),
    lambda r:r['active'].update(rng_isolated=False),
    lambda r:r['stages'].pop('masked.teacher'),
])
def test_invalid_raw_evidence_is_not_admitted(change):
    row=valid_item(VARIANTS[2]);D.validate_variant(row,GPU)
    change(row)
    with pytest.raises(ValueError):D.validate_variant(row,GPU)


@pytest.fixture
def summary_fixture(tmp_path,monkeypatch):
    reference=tmp_path/'reference';reference.mkdir();(reference/'benchmark.json').write_text('donor')
    launch=dict(content_hash='launch',reference=str(reference),donor_files={'benchmark.json':D.P.sha(reference/'benchmark.json')},
        reference_runtime_gpu=GPU,reference_masked=dict(validation_hours=18.,write_hours=.1))
    monkeypatch.setattr(D,'load',lambda *a,**k:launch)
    row=dict(schema=SCHEMA,launch_sha256='launch',donor_files=launch['donor_files'],complete=True,
        authorizes_training=False,runtime_gpu=GPU,reference_masked=launch['reference_masked'],
        variants=[valid_item(v) for v in VARIANTS])
    def save():D.P.write(tmp_path/'results.json',D.P.hashed(row))
    save()
    return tmp_path,row,save


def test_summary_does_not_authorize_training_or_relax_480_limit(summary_fixture):
    root,row,save=summary_fixture
    result=D.summarize(root)
    assert result['passed'] and result['complete'] and not result['authorizes_training']
    assert result['fastest_validated_candidate'] is None
    assert not result['models'][1]['within_480_hours']
    item=row['variants'][2]
    for phase in ('active','clean','fixed128'):
        t=item[phase];t['seconds_per_attempt']/=2;t['total_seconds']/=2
        for b in t['windows']:b['seconds']/=2
    save();result=D.summarize(root)
    assert result['fastest_validated_candidate']=='fast64'
    assert result['models'][2]['active_speedup_vs_baseline']==2
    assert result['models'][2]['within_480_hours']


@pytest.mark.parametrize('change',[
    lambda r:r['variants'][1]['active'].update(stream_sha256='b'*64),
    lambda r:r['variants'][1]['clean'].update(stream_sha256='b'*64),
    lambda r:r['variants'].append(copy.deepcopy(r['variants'][1])),
    lambda r:r.update(reference_masked=dict(validation_hours=1.,write_hours=0)),
    lambda r:r.update(authorizes_training=True),
])
def test_summary_rejects_unmatched_or_forged_comparisons(summary_fixture,change):
    root,row,save=summary_fixture;change(row);save()
    with pytest.raises(ValueError):D.summarize(root)


def test_partial_and_rejected_candidates_are_visible(summary_fixture):
    root,row,save=summary_fixture
    row['variants'][2]=dict(**VARIANTS[2],passed=False,error='OOM candidate rejected');save()
    result=D.summarize(root)
    assert result['passed'] and not result['models'][2]['passed']
    row['variants']=row['variants'][:2];row['complete']=False;save()
    assert not D.summarize(root)['passed']


def test_donor_changes_invalidate_review(summary_fixture):
    root,_,_=summary_fixture;(root/'reference/benchmark.json').write_text('changed')
    with pytest.raises(ValueError,match='Donor evidence'):D.summarize(root)
