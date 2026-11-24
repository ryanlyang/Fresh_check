"""Synthetic admission records are test fixtures, never GPU evidence."""
import copy
from types import SimpleNamespace
import pytest
from part_ev_ffn_experts import protocol as P
from part_ev_ffn_experts.config import *
from part_ev_ffn_experts.evidence import validate


def counts(tokens=8):
    return dict(tokens=[tokens]*6,requested=[[tokens,tokens,0,0] for _ in range(6)],
        executed=[[tokens,tokens,0,0] for _ in range(6)],primary=[[tokens,0,0,0] for _ in range(6)],
        secondary=[[0,tokens,0,0] for _ in range(6)])


def precision(dtype,paths):
    expected={n:'torch.float32' if f=='router' else dtype for n,f in paths.items()}
    return dict(passed=True,parameter_dtypes=['torch.float32'],expected_activation_dtype=dtype,
        expected_by_path=expected,observed_activations=copy.deepcopy(expected),
        router_paths=sorted(n for n,f in paths.items() if f=='router'))


def evidence(platform='sporc'):
    r=PLATFORMS[platform];world=r['gpus']
    paths={a:dict(native='native_expert',**({'router':'router'} if a=='experts' else {})) for a in COUNTS}
    spec=dict(content_hash='fixture',model_inventories={a:dict(x={},y={}) for a in COUNTS},precision_paths=paths)
    runtime=dict(platform=platform,resources=r,global_batch=512,tf32=False,cudnn_benchmark=False,
        nccl_environment={'TORCH_NCCL_ASYNC_ERROR_HANDLING':'1'},
        gpu=dict(name='NVIDIA '+('A100' if world==2 else 'GH200'),capability=[8,0] if world==2 else [9,0],
            driver='fixture',total_memory=100*2**30))
    models=[]
    for arm in COUNTS:
        candidate=arm=='experts'
        def timing(steps,windows):
            ranks=[dict(rank=i,seconds=float(steps),windows=[dict(steps=n,seconds=float(n)) for n in windows],
                precision=precision('torch.float16',paths[arm]),parameters_sha256='a'*64,scale=8192.,skips=0,lookahead=1,
                loss_sums=dict(ce=2.*steps,balance=1.1*steps if candidate else 0.,total=(2.+.001*1.1 if candidate else 2.)*steps),
                routing_totals=counts(512*steps) if candidate else None) for i in range(world)]
            return dict(measured_steps=steps,global_batch=512,seconds=float(steps),jets_per_second=512.,skips=0,ranks=ranks)
        names={'trimmer','embedding','pairs','classifier'}|{f'{p}_{i}' for p in ('attention','fc1','fc2') for i in range(8)}
        names|={f'class_{i}' for i in range(2)}|{f'bank_{i}' for i in range(6)}
        checks=dict(initial_identity=dict(passed=True,native_calls={n:1 for n in names},rng_exact=True,
            trimmed_inputs_exact=True,nonzero_state=True,shared_ce_gradient=True),
            active_fp32=dict(passed=True,amp=False,gradients=2,input_gradients=True,checkpoint_backward=True,exact_routes=True),
            active_amp=dict(passed=True,amp=True,gradients=2,input_gradients=True,checkpoint_backward=True,exact_routes=True),
            distributed=dict(passed=True,world_size=world,global_events=8,fp32=True,dropout=False,find_unused_parameters=False,
                one_particle=True,unequal_lengths=True,unequal_tokens=True,input_gradient_world_factor=True),
            rejection=dict(passed=True,empty_input=True,nonfinite=True,bad_stats=candidate,rank_local_io=True,stop=True),
            learning=dict(passed=True,successful_updates=8,attempts=10,skips=2,alpha=.001 if candidate else 0.,auxiliary_included=candidate))
        if candidate:
            checks['experts_oracle']=dict.fromkeys(('passed','fp64','independent_particle_loop','gradients','exact_routes',
                'unequal_count_proxy','shared_parameter_gradients','input_gradient_world_factor','naive_average_rejected',
                'stable_ties','device_empty_gradients'),True)
            checks['bank_ddp']=dict(passed=True,world_size=world,actual_data_parallel=world==2,unequal_tokens=True,
                local_empty=True,global_empty=True,all_gradients=True,shared_parameter_gradients=True,input_gradient_world_factor=True,
                phases=[dict(phase=p,global_tokens=16,counts=c,zero_drops=True) for p,c in
                    [('empty_local',[2,2,14,14]),('empty_global',[16,16,0,0])]])
            checks['coverage']=dict(passed=True,fixture='controlled_postnorm_two_sign_coverage_ce_only_v1',
                banks=[dict(block=i,all_four_learned=True,successful_updates=8,gradient_norms=dict(ce=.1,auxiliary=.2)) for i in SCIENCE['blocks']])
        restart=dict(passed=True,rank=0,updates=7,full_state=True,validation_boundaries=1,policy=copy.deepcopy(PARITY),
            fixture='trained_outputs_separated_router_biases_rolling_native_trim_dropout_v1',trials=[dict(trial=t,
                max_state_error=0.,worst_tensor_path=None,exact_initial_restore=True,updates=[dict(update=i+1,
                    loss=2.0011 if candidate else 2.,ce=2.,balance=1.1 if candidate else None,skipped=0,
                    scale=8192.,lookahead=i%6,lr=.001,input_sha256='f'*64,
                    routes=dict(sha256='d'*64,minimum_margin=4. if candidate else None),counts=counts() if candidate else None)
                    for i in range(7)]) for t in ('reference','repeat','disk','resumed')])
        scopes={'experts.context_router','experts.stable_selection','experts.packing','experts.mlp','experts.combination'}
        diagnostics=dict(events=8,particles=17,experts=candidate)
        if candidate:diagnostics.update(alpha=.001,banks=[dict(block=i,requested=[8,8,0,0],executed=[8,8,0,0],tokens=8,
            healthy=True,full_entropy=1.38,selected_entropy=.69,minimum_margin=.1,correction_dense_rms_ratio=.1,normalized_input_rms=1.)
            for i in SCIENCE['blocks']],gradient_norms={g:dict(ce=.1,auxiliary=.1) for g in ('router','experts')})
        models.append(dict(arm=arm,parameters=COUNTS[arm],finite_state=True,stream=timing(4200,POLICY['windows']),fixed=timing(40,[40]),
            first_batch_sha256=['f'*64]*world,cold_fetch_seconds=1.,correctness=[dict(rank=i,checks=copy.deepcopy(checks),
                restart=copy.deepcopy(restart)) for i in range(world)],
            gpus=[dict(rank=i,peak_allocated=10*2**30,peak_reserved=11*2**30) for i in range(world)],
            validation=dict(events=128000,seconds=100.,precision=precision('torch.float32',paths[arm])),
            inference={name:dict(batch=b,particles=128,steps=n,seconds=1.,precision=precision('torch.float32',paths[arm]))
                for name,b,n in [('throughput',256,20),('latency',1,100),('full_validation',512,3)]},
            profiles=[dict(rank=i,profiles=[dict(training=training,events=2,particles=17,
                stages={s:dict(calls=1,cpu_us=1.,device_us=0.) for s in scopes} if candidate else {})
                for training in (True,False)]) for i in range(world)],diagnostics=diagnostics,
            checkpoint=dict(restart_seconds=1.,load_seconds=1.,weights_seconds=1.),metric_5m_seconds=100.))
    row=dict(schema=SCHEMA,complete=True,campaign_sha256='fixture',runtime=runtime,models=models,
        policy=copy.deepcopy(POLICY),parity=copy.deepcopy(PARITY),stream_replay=[dict(rank=i,passed=True,real_stream=True,
            model_updates=0,rng_restored=True,streams={s:dict(passed=True,consumed_batches=2,next_sha256='a'*64)
                for s in ('train','val')}) for i in range(world)])
    return spec,row,dict(platform=platform,resources=r)


@pytest.mark.parametrize('platform',PLATFORMS)
def test_raw_paired_admission_and_measured_recommendations(platform):
    s,r,l=evidence(platform);result=validate(s,r,l)
    assert result['passed'] and not result['authorizes_training']
    assert all(360<=m['train_hours']<=480 for m in result['models'])


@pytest.mark.parametrize('mutation',[
    lambda r:r['models'][1]['stream']['ranks'][0]['precision']['observed_activations'].update(router='torch.float16'),
    lambda r:r['models'][1]['stream']['ranks'][0]['routing_totals']['executed'][0].__setitem__(0,0),
    lambda r:r['models'][1]['stream']['ranks'][0]['loss_sums'].update(balance=0.),
    lambda r:r['models'][0]['stream']['ranks'][0]['loss_sums'].update(balance=1.),
    lambda r:r['models'][1]['correctness'][0]['checks']['bank_ddp'].update(local_empty=False),
    lambda r:r['models'][1]['correctness'][0]['checks']['coverage']['banks'].pop(),
    lambda r:r['models'][1]['correctness'][0]['checks']['experts_oracle'].update(naive_average_rejected=False),
    lambda r:r['models'][1]['correctness'][0]['restart']['trials'][1]['updates'][0]['routes'].update(sha256='b'*64),
    lambda r:r['models'][1]['correctness'][0]['restart']['trials'][1]['updates'][0]['counts']['requested'][0].__setitem__(0,1),
    lambda r:r['models'][1]['correctness'][0]['restart']['trials'][1]['updates'][0].update(lr=.01),
    lambda r:r['models'][1]['correctness'][0]['restart']['trials'][1]['updates'][0].update(balance=2.),
    lambda r:r['models'][1]['gpus'][0].update(peak_allocated=99*2**30),
    lambda r:r['models'][1]['profiles'][0]['profiles'][0]['stages'].pop('experts.mlp'),
    lambda r:r['models'][1]['validation']['precision'].update(passed=False),
    lambda r:r['models'][0]['stream']['ranks'][0]['windows'].pop(),
    lambda r:r['models'][0]['stream']['ranks'][0].update(skips=210),
    lambda r:r['stream_replay'].clear(),
])
def test_missing_or_forged_router_loss_restart_and_fit_evidence_blocks(mutation):
    s,r,l=evidence();mutation(r)
    with pytest.raises((ValueError,KeyError,TypeError)):validate(s,r,l)


def test_aggregate_is_advisory_and_local_floor_remains(tmp_path,monkeypatch,capsys):
    launch=dict(repo=str(tmp_path),output=str(tmp_path/'new'),auto_production=True)
    monkeypatch.setattr(P,'required_storage',lambda *a:100*2**30)
    monkeypatch.setattr('shutil.disk_usage',lambda p:SimpleNamespace(free=48*2**30))
    floor=P.check_storage(launch);assert floor==campaign_budget()+POLICY['reserve_bytes']
    assert 'Proceeding as requested' in capsys.readouterr().err
    monkeypatch.setattr('shutil.disk_usage',lambda p:SimpleNamespace(free=floor-1))
    with pytest.raises(OSError):P.check_storage(launch)


@pytest.mark.parametrize('recommendation,field',[(481,'train_hours'),(25,'evaluate_hours')])
def test_measured_ceiling_cannot_be_capped_down(tmp_path,monkeypatch,recommendation,field):
    launch=dict(content_hash='launch',max_train_hours=480,max_evaluate_hours=24)
    spec=dict(content_hash='spec')
    summary=dict(passed=True,models=[dict(arm=a,train_hours=100,evaluate_hours=2) for a in COUNTS])
    summary['models'][1][field]=recommendation
    monkeypatch.setattr(P,'load_launch',lambda *a,**k:launch)
    monkeypatch.setattr(P,'load',lambda *a,**k:spec)
    monkeypatch.setattr(P,'review',lambda *a,**k:summary)
    with pytest.raises(ValueError,match='ceiling'):P.production_spec(tmp_path)
