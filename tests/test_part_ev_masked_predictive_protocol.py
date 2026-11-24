import copy
from types import SimpleNamespace
import pytest
from part_ev_masked_predictive import protocol as P
from part_ev_masked_predictive.config import *
from part_ev_masked_predictive.evidence import validate
from part_ev_masked_predictive.diagnostics import TRAIN_SCOPES


def precision(dtype='torch.float16'):
    return dict(passed=True,expected_activation_dtype=dtype,parameter_dtypes=['torch.float32'],observed_activations={'linear':dtype})


def evidence(platform='sporc'):
    resources=PLATFORMS[platform];world=resources['gpus']
    spec=dict(content_hash='campaign',training_inventories={'masked':{'a':{},'b':{}}})
    def timing(arm,steps,phase,windows):
        active=arm=='masked' and phase=='active'
        ranks=[dict(rank=r,windows=[dict(steps=n,seconds=float(n)) for n in windows],seconds=float(steps),skips=0,
            parameters_sha256='p',teacher_sha256='t' if arm=='masked' else None,scale=8192.,lookahead=3,
            precision=precision(),auxiliary_precision=precision() if active else None,
            teacher_precision=precision() if active else None,visible_precision=precision() if active else None,
            rng_isolated=True,eligible=steps*256 if active else 0,targets=steps*256 if active else 0,
            positive_aux_attempts=steps if active else 0,mean_ce=1.,mean_auxiliary=.5 if active else 0.) for r in range(world)]
        return dict(phase=phase,diagnostic_schedule=POLICY['diagnostic_schedule'],measured_steps=steps,global_batch=512,
            warmup=10 if steps==40 else 20,label_start=100000 if phase=='active' else 900000,
            microbatch=32,observed_local_batch=512//world,seconds=float(steps),jets_per_second=512.,skips=0,
            ranks=ranks,auxiliary_selected=256 if active else 0)
    def restart(arm,transitions):
        labels=[19999,20000,20001,39999,40000,899999,900000] if transitions else list(range(100000,100007))
        return dict(passed=True,updates=7,full_state=True,validation_boundaries=1,policy=PARITY,labels=labels,
            schedule='transitions' if transitions else 'active',trials=[dict(trial=name,exact_initial_restore=True,
                max_state_error=0.,updates=[dict(update=i+1,schedule_label=u,skipped=0,scale=8192.,bn_isolated=True,
                    coefficient=coefficient(u) if arm=='masked' else 0.,loss=1.,ce=.9,auxiliary=.5,
                    mask_sha256='a'*64 if arm=='masked' and coefficient(u)>0 else None) for i,u in enumerate(labels)])
                for name in ('reference','repeat','disk','resumed')])
    models=[]
    for arm in COUNTS:
        checks=dict(native=dict(passed=True,native_calls={str(i):1 for i in range(13)},export_keys=True,
                gradients=True,nonzero_state=True,deployed_parameters=COUNTS[arm]),
            failure=dict(passed=True,empty_input=True,nonfinite=True,rank_local_io=True,stop=True),
            learning=dict(passed=True,successful_updates=8,attempts=10,positive_loss=True))
        if arm=='masked':checks.update(
            predictor=dict(passed=True,fp64=True,head_loop=True,gradients=True,jet_mean=True),
            views=dict(passed=True,mask_counts=True,canonical_permutation=True,visible_rebase=True,alignment=True,rng_exact=True,yaml_authenticated=True),
            leakage=dict(passed=True,cases=['hidden_p4','hidden_independent','full_relative','old_padding','postclean_bn','clean_pair_cache'],
                fixed_masks=True,fixed_queries=True,b0_functional=True),
            sequential_fp32=dict(passed=True,simultaneous_oracle=True,preclean_bn=True,gradients=2),
            sequential_amp=dict(passed=True,simultaneous_oracle=True,preclean_bn=True,gradients=2),
            checkpoint_fp32=dict(passed=True,checkpoint_backward=True,input_gradients=True,cache_clear=True,particles=97,visible_particles=73,gradients=2),
            checkpoint_amp=dict(passed=True,checkpoint_backward=True,input_gradients=True,cache_clear=True,particles=97,visible_particles=73,gradients=2),
            auxiliary_gradient=dict(passed=True,backbone_gradient_norm=.1,teacher_detached=True),
            taps=dict(passed=True,taps=[6,8],normalizations=3,parameters=TEACHER,detached=True),
            distributed=dict(passed=True,world=world,global_oracle=True,cpu_health_only=False,public_no_sync=True,
                broadcast_buffers=False,find_unused_parameters=False,cases=[dict(case=n) for n in ('zero','one_rank','empty_last')]),
            skip=dict(passed=True,parameter_ema_skipped=True,bn_copied=True,skipped=1,mask_draws=1))
        profiles=[]
        for rank in range(world):
            rows=[]
            for training in (True,False):
                keys=TRAIN_SCOPES if training and arm=='masked' else ({'masked.bn_snapshot','masked.clean','masked.clean_backward','masked.optimizer','masked.ema'} if training else {'masked.export'})
                rows.append(dict(training=training,events=2,particles=17,stages={k:dict(calls=1,cpu_us=1.,device_us=0.) for k in keys}))
            profiles.append(dict(rank=rank,profiles=rows))
        models.append(dict(arm=arm,parameters=COUNTS[arm],trainable_parameters=TRAINABLE[arm],teacher_parameters=TEACHER if arm=='masked' else 0,
            finite_state=True,stream=timing(arm,4200,'active',POLICY['windows']),clean=timing(arm,200,'clean',[200]) if arm=='masked' else None,
            fixed=timing(arm,40,'active',[40]),first_batch_sha256=['f'*64]*world,
            correctness=[dict(rank=r,checks=copy.deepcopy(checks),restart=restart(arm,False),transitions=restart(arm,True)) for r in range(world)],
            gpus=[dict(rank=r,peak_allocated=10*2**30,peak_reserved=11*2**30,
                host=dict(VmRSS_bytes=1024,VmHWM_bytes=2048)) for r in range(world)],
            validation=dict(events=128000,seconds=100.,precision=precision('torch.float32')),
            inference={n:dict(batch=b,particles=128,steps=s,seconds=1.,precision=precision('torch.float32')) for n,b,s in
                [('throughput',256,20),('latency',1,100),('full_validation',512,3)]},
            profiles=profiles,diagnostics=dict(events=8,particles=17,target_variance=.1,prediction_variance=.1,
                target_norm=1.,prediction_norm=1.,cosine=.1,zero_kv_max_change=.1),
            checkpoint=dict(student_only=True,full_state_includes_transfer=True,transfer_seconds=.1,
                prediction_events=100000,restart_seconds=1.,load_seconds=1.,weights_seconds=1.,
                prediction_write_seconds=1.,prediction_load_seconds=1.),cold_fetch_seconds=1.,metric_5m_seconds=100.))
    row=dict(schema=SCHEMA,complete=True,campaign_sha256='campaign',policy=POLICY,parity=PARITY,
        runtime=dict(platform=platform,resources=resources,global_batch=512,broadcast_buffers=False,tf32=False,cudnn_benchmark=False,
            nccl_environment={'TORCH_NCCL_ASYNC_ERROR_HANDLING':'1'},gpu=dict(name='NVIDIA '+('A100' if world==2 else 'GH200'),
            total_memory=100*2**30,driver='fixture',capability=[8,0] if world==2 else [9,0])),models=models,
        stream_replay=[dict(rank=r,passed=True,real_stream=True,no_aux_draws_during_replay=True,rng_restored=True,
            streams={s:dict(passed=True,consumed_batches=2,next_sha256='a'*64,next_mask_sha256='b'*64,
                next_target_verified=True,next_target_events=1,next_target_sha256='c'*64,
                target_scope='fresh_frozen_teacher_next_batch') for s in ('train','val')}) for r in range(world)])
    return spec,row,dict(platform=platform,resources=resources)


@pytest.mark.parametrize('platform',PLATFORMS)
def test_complete_paired_evidence_and_phase_weighted_projection(platform):
    spec,row,launch=evidence(platform);result=validate(spec,row,launch)
    assert result['passed'] and not result['authorizes_training'] and len(result['models'])==2
    assert result['models'][1]['parameters']==COUNTS['masked'] and result['models'][1]['trainable_parameters']==TRAINABLE['masked']


@pytest.mark.parametrize('mutate',[
    lambda r:r.update(complete=False),
    lambda r:r['models'][1]['stream']['ranks'][0].update(positive_aux_attempts=0),
    lambda r:r['models'][1]['stream']['ranks'][0].update(mean_auxiliary=0.),
    lambda r:r['models'][1]['stream'].update(auxiliary_selected=128),
    lambda r:r['models'][1]['stream'].update(microbatch=16),
    lambda r:r['models'][1]['stream'].update(warmup=0),
    lambda r:r['models'][1]['stream'].update(label_start=1),
    lambda r:r['models'][1]['stream']['ranks'][0].update(teacher_precision=None),
    lambda r:r['models'][1]['stream']['ranks'][0].update(skips=210),
    lambda r:r['models'][1]['gpus'][0].update(peak_allocated=99*2**30),
    lambda r:r['models'][1]['gpus'][0]['host'].clear(),
    lambda r:r['models'][1]['correctness'][0]['checks'].pop('leakage'),
    lambda r:r['models'][1]['correctness'][0]['checks']['distributed'].update(cpu_health_only=True),
    lambda r:r['models'][1]['correctness'][0]['checks']['skip'].update(bn_copied=False),
    lambda r:r['models'][1]['correctness'][0]['checks']['taps'].update(taps=[8]),
    lambda r:r['models'][1]['correctness'][0]['checks']['views'].update(visible_rebase=False),
    lambda r:r['models'][1]['correctness'][0]['transitions']['labels'].__setitem__(0,1),
    lambda r:r['models'][1]['correctness'][0]['restart']['trials'][1]['updates'][0].update(loss=1.1),
    lambda r:r['models'][1]['correctness'][0]['restart']['trials'][1]['updates'][0].update(mask_sha256='z'*64),
    lambda r:r['models'][1]['first_batch_sha256'].__setitem__(0,'different'),
    lambda r:r['models'][1]['validation']['precision'].update(passed=False),
    lambda r:r['models'][1]['profiles'][0]['profiles'][0]['stages'].clear(),
    lambda r:r['models'][1]['checkpoint'].update(student_only=False),
    lambda r:r['models'][1]['checkpoint'].update(full_state_includes_transfer=False),
    lambda r:r['runtime'].update(broadcast_buffers=True),
    lambda r:r['stream_replay'].clear(),
    lambda r:r['stream_replay'][0]['streams']['train'].update(next_target_verified=False),
])
def test_incomplete_forged_or_inactive_evidence_blocks(mutate):
    spec,row,launch=evidence();mutate(row)
    with pytest.raises((ValueError,KeyError,TypeError)):validate(spec,row,launch)


def test_storage_own_floor_only_aggregate_is_advisory(tmp_path,monkeypatch,capsys):
    proposal=dict(repo=str(tmp_path),output=str(tmp_path/'new'),auto_production=True)
    floor=campaign_budget()+POLICY['reserve_bytes']
    monkeypatch.setattr('shutil.disk_usage',lambda p:SimpleNamespace(free=floor+1))
    monkeypatch.setattr(P,'required_storage',lambda *a:48.87*2**30)
    assert P.check_storage(proposal)==floor and 'advisory' in capsys.readouterr().err
    monkeypatch.setattr('shutil.disk_usage',lambda p:SimpleNamespace(free=floor-1))
    with pytest.raises(OSError):P.check_storage(proposal)


def test_frozen_source_inventory_rejects_import_drift(tmp_path):
    root=tmp_path/'source';root.mkdir();(root/'module.py').write_text('original')
    P.write(tmp_path/'source_manifest.json',P.inventory(root));P.verify_snapshot(tmp_path)
    (root/'cache.pyc').write_bytes(b'new')
    with pytest.raises(ValueError):P.verify_snapshot(tmp_path)


def test_full_schedule_counts_and_ema_endpoints():
    assert sum(coefficient(u)>0 for u in range(1,1000001))==879999
    assert coefficient(20000)==coefficient(900000)==0 and coefficient(20001)>0 and coefficient(899999)>0
    assert momentum(1)==pytest.approx(.996) and momentum(1000000)==pytest.approx(.9999)
