import copy
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from part_ev_state_tigris import protocol as P, benchmark as B
from part_ev_fullscale import protocol as F
from tests.test_part_ev_message_protocol import parity_fixture
from tests.test_part_ev_screen_protocol import precision_record


def hashed(row):
    return dict(row,content_hash=P.digest({k:v for k,v in row.items() if k!='content_hash'}))


def result_fixture(tmp_path):
    runtime=dict(gpu='NVIDIA GH200 480GB',gpu_memory_bytes=96*1024**3,
                 world_size=1,batch_size=512,workers=2,batchnorm='native',
                 tf32=False,cudnn_benchmark=False,campaign_sha256='diagnostic')
    parity=parity_fixture()
    parity.update(reference_kind='layerwise_EV_zero_state_branch',state_gradients=dict(
        passed=True,architecture='state_layerwise',full_model=True,state_tensors=32,
        checkpoint_backward=True,forward=dict(passed=True),tensors=100,max_absolute_error=1e-6))
    windows=[dict(steps=n,seconds=n*.1) for n in (1000,1000,1000,1000,200)]
    val=dict(events=128000,seconds=100,precision=precision_record('torch.float32'))
    row=dict(passed=True,task=copy.deepcopy(P.TASKS[0]),campaign_sha256='diagnostic',runtime=runtime,
        initial_parity=parity,measured_steps=4200,amp_skipped_steps=2,seconds=420.,jets_per_second=5120.,
        training_precision=precision_record('torch.float16'),validation=val,
        windows=windows,checkpoint_seconds=1.,cold_fetch_seconds=20.,
        native_peak_bytes=30*1024**3,full_length_peak_bytes=36*1024**3,validation_peak_bytes=25*1024**3,
        input_fingerprints={str(i):str(i) for i in (0,20,1020,2020,4020)},
        full_length_attempts=[dict(attempt=i+1,loss=1.,state_finite=True,amp_skipped=i==0,
                                  gradient_norm=None if i==0 else 1.) for i in range(18)])
    row['projected_training_hours']=P.projected_hours(P.recipes()['ranger_full_v1'],windows,val,1.,20.)
    path=tmp_path/'benchmarks'/f"{P.TASKS[0]['run_id']}.json"
    P.write_json(path,hashed(row))
    return dict(content_hash='diagnostic',donor=dict(original_estimate={'train_hours':216})),path,row


def test_same_model_and_full_recipe_without_mutating_historical_contracts():
    from part_ev_state_fullscale import protocol as historical
    assert len(P.TASKS)==1 and P.TASKS[0]==P.donor_protocol.TASK
    assert P.CONFIG['state_blocks']==list(range(8)) and P.CONFIG['state_width']==32
    assert P.COUNTS['state_layerwise']==2327462
    recipe=P.recipes()['ranger_full_v1'];original=F.recipe_for()
    for name in ('total_steps','epochs','steps_per_epoch','global_batch_size','validation_events_per_epoch',
                 'optimizer','lookahead_k','lookahead_alpha','weight_decay','lr_decay_epochs'):
        assert recipe[name]==original[name]
    assert recipe['total_steps']==1000000 and recipe['epochs']==50
    assert len(historical.TASKS)==7 and historical.BENCHMARK_INDICES==(0,3,6)
    assert P.TRANSFER['world_size']==1 and P.TRANSFER['authorizes_training'] is False
    assert not any('state_tigris' in name for name in P.donor_protocol.source_manifest())
    assert P.donor_protocol.source_manifest().items() <= P.source_manifest().items()


def test_projection_counts_fifty_validations_and_checkpoint_io(tmp_path):
    spec,_,row=result_fixture(tmp_path)
    view=P.review_view(tmp_path,spec)
    expected=(100000+50*10*100+1152+20)/3600
    assert view['projected_total_hours']==pytest.approx(expected)
    assert view['recommended_train_hours']==math.ceil(expected*1.3)
    assert view['projected_train_compute_hours']==pytest.approx(100000/3600)
    assert view['projected_validation_hours']==pytest.approx(50000/3600)
    assert view['authorizes_training'] is False and view['peak_gib']==36


@pytest.mark.parametrize('change',[
    lambda r:r.update(passed=False),lambda r:r['runtime'].update(gpu='A100'),
    lambda r:r['runtime'].update(world_size=2),lambda r:r['runtime'].update(batch_size=256),
    lambda r:r['runtime'].update(batchnorm='SyncBatchNorm'),lambda r:r['runtime'].update(tf32=True),
    lambda r:r.update(full_length_peak_bytes=95*1024**3),
    lambda r:r['initial_parity']['state_gradients'].update(state_tensors=16),
    lambda r:r['initial_parity']['state_gradients'].update(passed=False),
    lambda r:r['initial_parity'].update(reference_kind='ordinary_EV_zero_state_branch'),
    lambda r:r.update(jets_per_second=1),lambda r:r.update(projected_training_hours=10),
    lambda r:r['validation'].update(events=128),
    lambda r:r['full_length_attempts'][-1].update(amp_skipped=True),
    lambda r:r['training_precision'].update(passed=False),
])
def test_benchmark_failures_never_accepted(tmp_path,change):
    spec,path,row=result_fixture(tmp_path)
    change(row);P.write_json(path,hashed(row))
    with pytest.raises(ValueError): P.review_view(tmp_path,spec)


def test_arm_environment_allows_platform_change_not_weaver_change(monkeypatch):
    donor={'environment':dict(weaver_sources={'model':'frozen'})}
    environment=dict(architecture='aarch64',torch_cuda='12.6',weaver_sources={'model':'frozen'})
    monkeypatch.setattr(P,'reference_metadata',lambda:None)
    monkeypatch.setattr(F,'environment',lambda:environment)
    assert P.runtime_environment(donor)==environment
    for changes in (dict(architecture='x86_64'),dict(torch_cuda=None),dict(weaver_sources={'model':'changed'})):
        before=environment.copy();environment.update(changes)
        with pytest.raises(ValueError): P.runtime_environment(donor)
        environment.clear();environment.update(before)


def test_preparation_is_independent_and_donor_stays_read_only(tmp_path,monkeypatch):
    donor=tmp_path/'donor';donor.mkdir()
    P.write_json(donor/'campaign.json',dict(read_only=True))
    original=(donor/'campaign.json').read_bytes()
    parent=dict(inventory={'same':'population'})
    contract=dict(path=str(donor),original_estimate={'train_hours':216})
    monkeypatch.setattr(P,'donor_contract',lambda _: (parent,contract))
    monkeypatch.setattr(P,'runtime_environment',lambda _:dict(architecture='aarch64'))
    monkeypatch.setattr(P,'verify_inventory_metadata',lambda _:None)
    monkeypatch.setattr(P,'source_manifest',lambda:dict(code='frozen'))
    output=tmp_path/'probe';P.write_json(output/'launch.json',{'donor':str(donor)})
    for bad in (donor,donor/'child',tmp_path):
        with pytest.raises(ValueError): P.prepare(bad,donor)
    spec=P.prepare(output,donor)
    assert P.load_campaign(output)==spec
    assert spec['inventory']==parent['inventory'] and spec['authorizes_training'] is False
    assert (donor/'campaign.json').read_bytes()==original and len(list(donor.iterdir()))==1
    with pytest.raises(FileExistsError): P.prepare(output,donor)
    P.write_json(output/'campaign.json',hashed(dict(spec,transfer={})))
    with pytest.raises(ValueError): P.load_campaign(output)


def test_private_driver_uses_existing_benchmark_and_only_new_gpu_setup(monkeypatch):
    from scripts import run_part_ev_state_compare as original
    old_tasks=copy.deepcopy(original.P.TASKS)
    engine,probe=B.create_driver()
    assert probe.D is engine and engine.P is P and probe.P is P
    assert probe.benchmark.__code__.co_code==original.probe.benchmark.__code__.co_code
    assert original.P.TASKS==old_tasks and len(original.P.TASKS)==5
    assert engine.gpu_setup is probe.gpu_setup and engine.gpu_setup is not original.D.gpu_setup
    assert engine.build_model is original.D.build_model
    monkeypatch.setattr(B.platform,'machine',lambda:'x86_64')
    with pytest.raises(RuntimeError,match='Tigris ARM'): engine.gpu_setup({})


def test_soft_deadline_does_not_turn_partial_benchmark_into_success(monkeypatch):
    engine=SimpleNamespace(check_stop=lambda:None)
    def bench(_): engine.check_stop()
    monkeypatch.setattr(B,'create_driver',lambda:(engine,SimpleNamespace(benchmark=bench)))
    clock=iter((0, P.TRANSFER['soft_seconds']+1))
    monkeypatch.setattr(B.time,'monotonic',lambda:next(clock))
    with pytest.raises(TimeoutError): B.run(None)


def test_donor_verification_skips_only_cross_platform_environment_comparison(tmp_path,monkeypatch):
    spec=dict(task=copy.deepcopy(P.TASKS[0]),recipes=P.recipes(),content_hash='frozen')
    for name in ('campaign.json','production_approval.json','preflight/result.json'):
        P.write_json(tmp_path/name,{'identity':name})
    before={p:p.read_bytes() for p in tmp_path.rglob('*.json')}
    calls=[]
    def load(path,*,check_environment):
        assert path==tmp_path and check_environment is False
        calls.append('verify_frozen_source_recipe_data')
        return spec
    def approval(path,actual):
        assert path==tmp_path and actual is spec
        calls.append('verify_existing_approval')
        return {'train_hours':216}
    monkeypatch.setattr(P.donor_protocol,'load',load)
    monkeypatch.setattr(P.donor_protocol,'require_approval',approval)
    actual,contract=P.donor_contract(tmp_path)
    assert actual is spec and contract['original_estimate']=={'train_hours':216}
    assert calls==['verify_frozen_source_recipe_data','verify_existing_approval']
    assert before=={p:p.read_bytes() for p in tmp_path.rglob('*.json')}


def test_gpu_setup_requires_one_gh200_and_freezes_precision(monkeypatch):
    engine,_=B.create_driver()
    torch=engine.torch
    monkeypatch.setattr(B.platform,'machine',lambda:'aarch64')
    monkeypatch.setenv('WORLD_SIZE','1')
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'device_count',lambda:1)
    monkeypatch.setattr(torch.cuda,'set_device',lambda _:None)
    monkeypatch.setattr(torch,'set_num_threads',lambda _:None)
    monkeypatch.setattr(B.signal,'signal',lambda *a:None)
    monkeypatch.setattr(B.signal,'SIGUSR1',10,raising=False)  # Linux worker, Windows test host.
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda _:'NVIDIA GH200 480GB')
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda _:SimpleNamespace(total_memory=96*1024**3))
    _,runtime=engine.gpu_setup(dict(content_hash='fixture'))
    assert runtime['batch_size']==512 and runtime['batchnorm']=='native'
    assert runtime['precision']==engine.PRECISION_POLICY and runtime['tf32'] is False
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda _:'NVIDIA H100')
    with pytest.raises(ValueError,match='GH200'): engine.gpu_setup(dict(content_hash='fixture'))
    monkeypatch.setattr(torch.cuda,'device_count',lambda:2)
    with pytest.raises(RuntimeError,match='exactly one'): engine.gpu_setup(dict(content_hash='fixture'))
