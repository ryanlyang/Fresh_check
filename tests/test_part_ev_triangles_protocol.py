import copy
import math
from types import SimpleNamespace
import pytest
from part_ev_triangles import protocol as P
from scripts.run_part_ev_triangles import main


def precision(dtype):
    return dict(passed=True,expected_activation_dtype=dtype,parameter_dtypes=['torch.float32'],
                observed_activations={'mod.pair_readout.key':dtype})


def evidence():
    def measure(count):
        windows=[dict(steps=min(1000,count-i),seconds=min(1000,count-i)/2)
                 for i in range(0,count,1000)]
        rank=dict(windows=windows,precision=precision('torch.float16'),parameters_sha256='same',
                  scaler=32768,skips=0,lookahead_counter=2)
        return dict(measured_steps=count,healthy=True,seconds=count/2,jets_per_second=1024,
                    skips=0,ranks=[copy.deepcopy(rank),copy.deepcopy(rank)])
    diag=dict(healthy=True,events=8,particles=17,disabled_logit_rms_change=.01,
        layers=[dict(layer=i,left_preactivation_rms=.2,right_preactivation_rms=.2,gate_preactivation_rms=.2,
            left_saturation=.01,right_saturation=.01,gate_saturation=.01,triangle_to_state_norm=.1,
            triangle_to_ev_norm=.1,zero_denominator_fraction=0.) for i in (6,7)])
    row=dict(passed=True,complete=True,authorizes_training=False,campaign_sha256='spec',
        parameters=P.PARAMETERS,baseline_parameters=P.BASE_PARAMETERS,
        stream=measure(4200),fixed=measure(40),baseline_stream=measure(4200),baseline_fixed=measure(40),
        correctness=[dict(passed=True,rank=r,tolerances=copy.deepcopy(P.PARITY),
            checks=[dict(kind=k,passed=True) for k in sorted(P.CHECKS)],
            delayed_activation=dict(passed=True,successful_updates=8,attempts=10,skips=2,
                initial_upstream_zero=True,initial_output_signal=True,upstream_updated=True,
                all_triangle_gradients=True)) for r in (0,1)],
        restart=copy.deepcopy(P.RESTART),finite_state=True,baseline_finite_state=True,
        full128_validation_passed=True,checkpoint_seconds=1.,
        gpus=[dict(rank=r,name='NVIDIA A100',total_memory=40*2**30,peak_allocated=27*2**30,
                   peak_reserved=31*2**30,first_input_sha256='rank'+str(r)) for r in (0,1)],
        validation=dict(events=128000,seconds=30,precision=precision('torch.float32')),
        inference=dict(batch_size=256,particles=128,batches=20,seconds=4.,precision='fp32',
                       batch1_iterations=100,batch1_seconds=2.),
        diagnostics=[dict(copy.deepcopy(diag),rank=r) for r in (0,1)],
        triangle_gradient_norms=[{str(i):.1 for i in range(10)} for r in (0,1)],
        runtime=dict(world_size=2,local_batch=256,global_batch=512,workers_per_rank=1,
            batchnorm='SyncBatchNorm',tf32=False,cudnn_benchmark=False,nccl_environment={}))
    hist=[0]*128;hist[127]=100
    neighbors=dict(histogram=hist,receivers=100,tie_fraction=1.,mean_weight_mass=8.,maximum_support=127)
    stages={k:dict(cpu_us=1.,device_us=1.,calls=2) for k in P.PROFILE_STAGES}
    def profile(training):
        memory=dict(pid=101,scope='process_only_excludes_workers',rss_bytes=2**30,peak_rss_bytes=2**30)
        return dict(stages=copy.deepcopy(stages),policy=copy.deepcopy(P.PROFILE_POLICY),
            sample=dict(source_events=256,source_particles=128,events=2,particles=32,training=training),
            host_memory=dict(before=copy.deepcopy(memory),after=copy.deepcopy(memory)))
    row.update(backend='sparse_tiled',adversarial_ties=measure(3),full128_tied_validation_passed=True,
        neighbor_statistics=[dict(rank=r,train=copy.deepcopy(neighbors),adversarial=copy.deepcopy(neighbors),
                                  validation=copy.deepcopy(neighbors)) for r in (0,1)],
        profiles=[dict(rank=r,real_train=profile(True),adversarial_train=profile(True),
                       real_validation=profile(False)) for r in (0,1)],
        host_peak_rss_bytes=[2**30,2**30])
    row['baseline_gpus']=copy.deepcopy(row['gpus'])
    row['baseline_inference']=copy.deepcopy(row['inference'])
    return row


def setup_review(tmp_path,monkeypatch,row):
    P.write_json(tmp_path/'campaign.json',{})
    spec=dict(content_hash='spec',launch=dict(nccl_environment={}))
    monkeypatch.setattr(P,'load',lambda *a,**kw:spec)
    row['content_hash']=P.digest(row)
    P.write_json(tmp_path/'benchmark.json',row)


def test_full_projection_and_ratios_never_production(tmp_path,monkeypatch):
    setup_review(tmp_path,monkeypatch,evidence())
    row=P.review(tmp_path)
    assert row['passed'] and not row['authorizes_training'] and row['parameters']==2334662
    assert row['projected_train_hours']==pytest.approx(500000/3600)
    assert row['projected_validation_hours']==pytest.approx(30*50*10/3600)
    assert row['projected_checkpoint_hours']==pytest.approx(1152/3600)
    assert row['recommended_train_hours']==math.ceil(1.3*(500000+15000+1152)/3600)
    assert row['training_time_vs_state_ev']==row['inference_time_vs_state_ev']==1
    assert row['batch1_latency_ms']==20


@pytest.mark.parametrize('path,value',[
    (('authorizes_training',),True),(('parameters',),2327462),(('campaign_sha256',),'other'),
    (('baseline_stream','ranks',1,'parameters_sha256'),'different'),
    (('stream','ranks',0,'precision','passed'),False),(('fixed','skips'),2),
    (('stream','ranks',0,'windows',0,'steps'),9),(('stream','jets_per_second'),float('inf')),
    (('correctness',1,'rank'),0),(('correctness',0,'checks',0,'passed'),False),
    (('correctness',0,'delayed_activation','upstream_updated'),False),
    (('restart','fixture_cursor_replay'),False),(('baseline_finite_state',),False),
    (('full128_validation_passed',),False),(('runtime','tf32'),True),
    (('gpus',1,'peak_allocated'),39*2**30),(('baseline_gpus',0,'name'),'GH200'),
    (('validation','events'),127999),(('validation','precision','passed'),False),
    (('inference','batch_size'),128),(('baseline_inference','seconds'),0),
    (('diagnostics',0,'layers',0,'left_saturation'),2.),(('diagnostics',1,'layers',1,'right_preactivation_rms'),float('nan')),
    (('backend',),'dense_oracle'),(('full128_tied_validation_passed',),False),
    (('neighbor_statistics',0,'adversarial','maximum_support'),8),
    (('neighbor_statistics',1,'train','receivers'),0),
    (('profiles',0,'real_train','stages','triangles.tie_mix','calls'),0),
    (('profiles',0,'real_train','policy','max_events'),256),
    (('profiles',0,'real_train','policy','diagnostic_only'),False),
    (('profiles',0,'real_train','sample','events'),256),
    (('profiles',0,'real_train','sample','particles'),128),
    (('profiles',0,'real_train','sample','training'),False),
    (('profiles',0,'adversarial_train','sample','source_particles'),32),
    (('profiles',0,'real_validation','sample','training'),True),
    (('profiles',1,'real_train','host_memory','before','rss_bytes'),None),
    (('profiles',0,'real_train','host_memory','after','scope'),'whole_job'),
    (('host_peak_rss_bytes',0),0),
    (('triangle_gradient_norms',0,'0'),0.),(('checkpoint_seconds',),0),
])
def test_fail_closed(tmp_path,monkeypatch,path,value):
    row=evidence();target=row
    for key in path[:-1]:target=target[key]
    target[path[-1]]=value
    # In-memory admission still rejects nonfinite values before serialization.
    if type(value) is float and not math.isfinite(value):
        with pytest.raises(ValueError):P.summarize(dict(content_hash='spec',launch=dict(nccl_environment={})),row)
    else:
        setup_review(tmp_path,monkeypatch,row)
        with pytest.raises(ValueError):P.review(tmp_path)


def test_missing_or_forged_report(tmp_path,monkeypatch):
    assert P.review(tmp_path)['missing']==['campaign.json','benchmark.json']
    assert main(['--phase','summary','--output',str(tmp_path)])==1
    with pytest.raises(SystemExit):main(['--phase','train','--output',str(tmp_path)])
    row=evidence();setup_review(tmp_path,monkeypatch,row)
    row['complete']=False;P.write_json(tmp_path/'benchmark.json',row)
    with pytest.raises(ValueError):P.review(tmp_path)


def fixture_launch(output,donor,repo):
    (output/'source').mkdir(parents=True)
    (output/'source/a.py').write_text('# snapshot')
    manifest={'a.py':P.file_sha256(output/'source/a.py')}
    P.write_json(output/'source_manifest.json',manifest)
    launch=dict(schema=P.SCHEMA,output=str(output),repo=str(repo),reference=str(donor),python='unused',
        minutes=P.MINUTES,host_memory_gib=P.HOST_MEMORY_GIB,source_sha256=P.digest(manifest),authorizes_training=False,
        output_budget_bytes=P.BUDGET_BYTES,nccl_environment={})
    launch['content_hash']=P.digest(launch);P.write_json(output/'launch.json',launch)
    return launch


def test_prepare_load_and_inventory_independent(tmp_path,monkeypatch):
    donor=tmp_path/'donor';donor.mkdir();P.write_json(donor/'campaign.json',{'old':'immutable'})
    output=tmp_path/'new';fixture_launch(output,donor,tmp_path)
    old=dict(inventory={'files':'verified'},environment={'runtime':'unchanged'})
    monkeypatch.setattr(P,'reference',lambda *a,**kw:old)
    monkeypatch.setattr(P,'runtime_environment',lambda:old['environment'])
    monkeypatch.setattr(P,'verify_inventory_metadata',lambda *a:None)
    monkeypatch.setattr(P,'verify_data_contents',lambda *a:{'file':'bytes'})
    monkeypatch.setattr(P,'check_storage',lambda *a:8*2**30)
    before=(donor/'campaign.json').read_bytes()
    spec=P.prepare(output)
    assert P.load(output)==spec and spec['recipe']['total_steps']==1000000
    assert not spec['historical_control_reuse_authorized'] and not spec['authorizes_training']
    assert (donor/'campaign.json').read_bytes()==before and len(list(donor.iterdir()))==1
    with pytest.raises(FileExistsError):P.prepare(output)
    (output/'source/unexpected.py').write_text('extra')
    with pytest.raises(ValueError,match='source'):P.load(output)


def test_old_launch_still_readable_for_frozen_review(tmp_path):
    donor=tmp_path/'donor';donor.mkdir()
    output=tmp_path/'old'
    launch=fixture_launch(output,donor,tmp_path)
    launch.pop('host_memory_gib');launch.pop('content_hash')
    launch['content_hash']=P.digest(launch);P.write_json(output/'launch.json',launch)
    assert P.launch_config(output)==launch


def test_recorded_host_ram_mismatch_rejected(tmp_path):
    donor=tmp_path/'donor';donor.mkdir()
    output=tmp_path/'new'
    launch=fixture_launch(output,donor,tmp_path)
    launch['host_memory_gib']=dict(P.HOST_MEMORY_GIB,benchmark=64)
    launch.pop('content_hash');launch['content_hash']=P.digest(launch)
    P.write_json(output/'launch.json',launch)
    with pytest.raises(ValueError,match='launch identity'):P.launch_config(output)


def test_storage_aggregate_and_read_only(tmp_path,monkeypatch):
    from part_ev_extensions_tigris import protocol as G
    output=tmp_path/'checkpoints/part_ev_triangles/new'
    other=output.parent/'old';other.mkdir(parents=True)
    row=dict(output_budget_bytes=2*2**30);row['content_hash']=P.digest(row)
    P.write_json(other/'launch.json',row)
    monkeypatch.setattr(G,'required_storage',lambda *a:8*2**30)
    monkeypatch.setattr(G,'used_bytes',lambda *a:0)
    monkeypatch.setattr(P.shutil,'disk_usage',lambda *a:SimpleNamespace(free=11*2**30))
    assert P.check_storage(tmp_path,output)==11*2**30
    assert not output.exists()
    monkeypatch.setattr(P.shutil,'disk_usage',lambda *a:SimpleNamespace(free=11*2**30-1))
    with pytest.raises(OSError):P.check_storage(tmp_path,output)
