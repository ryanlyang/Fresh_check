import copy
from pathlib import Path
from types import SimpleNamespace
import pytest
from part_ev_workspace import protocol as P
from part_ev_workspace.config import *
from part_ev_workspace.evidence import validate


def precision(dtype):
    return dict(passed=True,parameter_dtypes=['torch.float32'],expected_activation_dtype=dtype,
                observed_activations={'linear':dtype})


def evidence(platform='sporc'):
    resources=PLATFORMS[platform];world=resources['gpus']
    spec=dict(content_hash='campaign',model_inventories={a:{'x':{},'y':{}} for a in COUNTS})
    runtime=dict(platform=platform,resources=resources,global_batch=512,tf32=False,cudnn_benchmark=False,
        nccl_environment={'TORCH_NCCL_ASYNC_ERROR_HANDLING':'1'},gpu=dict(name='NVIDIA '+('A100' if world==2 else 'GH200'),
            total_memory=100*2**30,driver='fixture',capability=[8,0] if world==2 else [9,0]))
    def timing(steps,windows):
        ranks=[dict(rank=r,windows=[dict(steps=n,seconds=float(n)) for n in windows],seconds=float(steps),skips=0,
            precision=precision('torch.float16'),parameters_sha256='same',scale=8192.,lookahead=3) for r in range(world)]
        return dict(measured_steps=steps,global_batch=512,seconds=float(steps),jets_per_second=512.,ranks=ranks,skips=0)
    models=[]
    for arm in COUNTS:
        checks=dict(initial_identity=dict(passed=True,native_calls={str(i):1 for i in range(8)},
            native_pipeline=dict(trimmer=1,embedding=1,pairs=1,classifier=1,class_attention_0=1,class_attention_1=1),
            nonzero_state=True,rng_exact=True),
            active_fp32=dict(passed=True,gradients=2,checkpoint_backward=True),
            active_amp=dict(passed=True,gradients=2,checkpoint_backward=True),
            distributed=dict(passed=True,world_size=world,global_events=8,one_particle=True,unequal_lengths=True),
            rejection=dict(passed=True,empty_input=True,nonfinite=True,rank_local_io=True,stop=True),
            learning=dict(passed=True,successful_updates=8,attempts=10))
        if arm=='workspace':checks['attention_oracle']=dict(passed=True,precision='fp64',independent_head_loop=True,gradients=True,
            full_stage=dict(passed=True,precision='fp64',independent_stage=True,gradients=True,
                permutation=True,padding=True,event_isolation=True))
        restart=dict(passed=True,updates=7,full_state=True,policy=PARITY,validation_boundaries=1,
            trials=[dict(trial=t,exact_initial_restore=True,max_state_error=0.,updates=[
                dict(update=i+1,skipped=0,scale=8192.,loss=1.,lookahead=i%6) for i in range(7)]) for t in ('reference','repeat','disk','resumed')])
        keys={f'workspace.stages.{i}.{s}' for i in range(4) for s in ('read','mix','write','ffn')}
        keys.add('workspace.class_read.attention')
        profiles=[dict(rank=r,profiles=[dict(training=b,events=2,particles=17,
            stages={k:dict(calls=1,cpu_us=1.,device_us=0.) for k in keys} if arm=='workspace' else {}) for b in (True,False)]) for r in range(world)]
        models.append(dict(arm=arm,parameters=COUNTS[arm],finite_state=True,stream=timing(4200,POLICY['windows']),
            fixed=timing(40,[40]),first_batch_sha256=['f'*64]*world,
            correctness=[dict(rank=r,checks=copy.deepcopy(checks),restart=copy.deepcopy(restart)) for r in range(world)],
            gpus=[dict(rank=r,peak_allocated=10*2**30,peak_reserved=11*2**30) for r in range(world)],
            validation=dict(events=128000,seconds=100.,precision=precision('torch.float32')),
            inference={n:dict(batch=b,particles=128,steps=s,seconds=1.,precision=precision('torch.float32'))
                for n,b,s in [('throughput',256,20),('latency',1,100),('full_validation',512,3)]},
            profiles=profiles,diagnostics=dict(events=8,particles=17,layers=[{}]*4,attention_entropy={str(i):1 for i in range(13)}),
            checkpoint=dict(restart_seconds=1.,load_seconds=1.,weights_seconds=1.),
            metric_5m_seconds=100.,cold_fetch_seconds=1.))
    row=dict(schema=SCHEMA,complete=True,campaign_sha256='campaign',policy=POLICY,parity=PARITY,
        runtime=runtime,models=models,stream_replay=[dict(rank=r,passed=True,real_stream=True,model_updates=0,rng_restored=True,
            streams={s:dict(consumed_batches=2,passed=True,next_sha256='a'*64) for s in ('train','val')}) for r in range(world)])
    return spec,row,dict(platform=platform,resources=resources)


@pytest.mark.parametrize('platform',PLATFORMS)
def test_complete_paired_evidence_recomputes_projection(platform):
    spec,row,launch=evidence(platform)
    result=validate(spec,row,launch)
    assert result['passed'] and not result['authorizes_training']
    assert [m['parameters'] for m in result['models']]==list(COUNTS.values())
    assert result['models'][0]['train_hours']>=362


@pytest.mark.parametrize('mutation',[
    lambda r:r.update(complete=False),
    lambda r:r['models'][1]['stream']['ranks'][0]['windows'].pop(),
    lambda r:r['models'][0]['stream']['ranks'][0].update(skips=210),
    lambda r:r['models'][0]['gpus'][0].update(peak_allocated=99*2**30),
    lambda r:r['models'][1]['correctness'][0]['checks'].pop('active_amp'),
    lambda r:r['models'][1]['correctness'][0]['restart']['trials'].pop(),
    lambda r:r['models'][0]['validation']['precision'].update(passed=False),
    lambda r:r['models'][1]['first_batch_sha256'].__setitem__(0,'different'),
    lambda r:r['models'][1]['profiles'][0]['profiles'][0]['stages'].clear(),
    lambda r:r['stream_replay'].clear(),
    lambda r:r['models'][0]['stream']['ranks'][0].update(seconds=1.),
    lambda r:r['models'][1]['correctness'][0]['restart']['trials'][1]['updates'][0].update(loss=1.01),
    lambda r:r['models'][1]['correctness'][0]['restart']['trials'][2]['updates'][0].update(lookahead=5),
    lambda r:r['models'][0]['correctness'][0]['checks']['initial_identity']['native_pipeline'].update(pairs=2),
    lambda r:r['stream_replay'][0].update(rng_restored=False),
])
def test_incomplete_or_forged_evidence_blocks(mutation):
    spec,row,launch=evidence();mutation(row)
    with pytest.raises((ValueError,KeyError)):validate(spec,row,launch)


def test_aggregate_storage_shared_platforms_without_double_count(tmp_path,monkeypatch,capsys):
    repo=tmp_path;registry=repo/'checkpoints/part_ev_workspace';registry.mkdir(parents=True)
    output=registry/'sporc';other=registry/'tigris';other.mkdir()
    P.write(other/'launch.json',P.hashed(dict(reservation_bytes=campaign_budget())))
    needed=P.required_storage(repo,output,campaign_budget())
    expected=POLICY['reserve_bytes']+2*campaign_budget()-P.used_bytes(other)
    assert needed==expected
    assert 18.9<needed/2**30<19.
    monkeypatch.setattr('shutil.disk_usage',lambda p:SimpleNamespace(free=needed-1))
    actual=P.check_storage(dict(repo=str(repo),output=str(output),auto_production=True))
    assert actual==POLICY['reserve_bytes']+campaign_budget()
    assert 'Storage advisory' in capsys.readouterr().err


@pytest.mark.parametrize('platform',PLATFORMS)
@pytest.mark.parametrize('stage',['benchmark','pipeline','production'])
def test_other_campaign_reservations_do_not_block_48gib(tmp_path,monkeypatch,capsys,platform,stage):
    output=tmp_path/platform
    launch=dict(repo=str(tmp_path),output=str(output),platform=platform,
                auto_production=stage=='pipeline',storage_policy=STORAGE_POLICY)
    monkeypatch.setattr(P,'required_storage',lambda *a:round(48.87*2**30))
    monkeypatch.setattr('shutil.disk_usage',lambda p:SimpleNamespace(free=48*2**30))
    actual=P.check_storage(launch,production=stage=='production')
    budget=POLICY['benchmark_bytes'] if stage=='benchmark' else campaign_budget()
    assert actual==POLICY['reserve_bytes']+budget
    warning=capsys.readouterr().err
    assert '48.87 GiB' in warning and 'Proceeding as requested' in warning


@pytest.mark.parametrize('auto,production',[(False,False),(True,False),(False,True)])
def test_this_campaign_still_requires_real_local_headroom(tmp_path,monkeypatch,auto,production):
    launch=dict(repo=str(tmp_path),output=str(tmp_path/'new'),auto_production=auto)
    budget=campaign_budget() if auto or production else POLICY['benchmark_bytes']
    floor=POLICY['reserve_bytes']+budget
    monkeypatch.setattr('shutil.disk_usage',lambda p:SimpleNamespace(free=floor-1))
    monkeypatch.setattr(P,'required_storage',lambda *a:(_ for _ in ()).throw(AssertionError('should fail locally first')))
    with pytest.raises(OSError,match="this campaign's remaining output budget"):
        P.check_storage(launch,production=production)


@pytest.mark.parametrize('error',[ValueError('stale foreign metadata'),OSError('foreign path disappeared'),KeyError('old schema')])
def test_unavailable_foreign_accounting_is_advisory(tmp_path,monkeypatch,capsys,error):
    launch=dict(repo=str(tmp_path),output=str(tmp_path/'new'),auto_production=True)
    monkeypatch.setattr('shutil.disk_usage',lambda p:SimpleNamespace(free=48*2**30))
    monkeypatch.setattr(P,'required_storage',lambda *a:(_ for _ in ()).throw(error))
    assert P.check_storage(launch)==POLICY['reserve_bytes']+campaign_budget()
    assert 'could not be calculated' in capsys.readouterr().err


def test_current_output_reduces_only_this_campaign_budget(tmp_path,monkeypatch):
    output=tmp_path/'current';output.mkdir();(output/'already-written.bin').write_bytes(b'abc')
    launch=dict(repo=str(tmp_path),output=str(output),auto_production=True)
    expected=POLICY['reserve_bytes']+campaign_budget()-3
    monkeypatch.setattr('shutil.disk_usage',lambda p:SimpleNamespace(free=expected))
    monkeypatch.setattr(P,'required_storage',lambda *a:expected)
    assert P.check_storage(launch)==expected


def test_environment_uses_stable_distribution_roots(monkeypatch):
    import sys
    a=P.runtime_environment()
    sys.path.append('/tmp/synthetic_setuptools_vendor')
    try:assert P.runtime_environment()==a
    finally:sys.path.pop()


def test_source_inventory_rejects_changed_or_new_files(tmp_path):
    source=tmp_path/'source';source.mkdir();(source/'test.py').write_text('original')
    P.write(tmp_path/'source_manifest.json',P.inventory(source));P.verify_snapshot(tmp_path)
    (source/'unexpected.pyc').write_bytes(b'x')
    with pytest.raises(ValueError):P.verify_snapshot(tmp_path)
