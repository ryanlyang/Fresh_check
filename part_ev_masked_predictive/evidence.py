"""Recompute phase-weighted resource admission from complete raw evidence."""
import math
from .config import SCHEMA,COUNTS,TRAINABLE,TEACHER,POLICY,PARITY
from part_ev_message_study.protocol import precision_ok as _precision_ok


def precision_ok(row,dtype):return isinstance(row,dict) and _precision_ok(row,dtype)


def require(ok,message):
    if not ok:raise ValueError('Masked predictive admission: '+message)


def positive(x):return type(x) in (int,float) and math.isfinite(x) and x>0


def timing(row,world,arm,phase,steps,windows):
    require(row['phase']==phase and row['diagnostic_schedule']==POLICY['diagnostic_schedule']
        and row['measured_steps']==steps and row['global_batch']==512 and row['microbatch']==32
        and row['warmup']==(10 if steps==40 else 20)
        and row['label_start']==(100000 if phase=='active' else 900000)
        and row['observed_local_batch']==512//world and positive(row['seconds']),'phase/coverage/batch')
    require(row['auxiliary_selected']==(256 if arm=='masked' and phase=='active' else 0),'auxiliary population')
    require(len(row['ranks'])==world and {r['rank'] for r in row['ranks']}==set(range(world)),'rank timing coverage')
    require(math.isclose(row['seconds'],max(r['seconds'] for r in row['ranks']),rel_tol=1e-10)
        and math.isclose(row['jets_per_second'],512*steps/row['seconds'],rel_tol=1e-10),'throughput arithmetic')
    ids=set()
    for r in row['ranks']:
        require([w['steps'] for w in r['windows']]==windows and all(positive(w['seconds']) for w in r['windows']),'timing windows')
        require(sum(w['seconds'] for w in r['windows'])<=r['seconds']*1.001,'window arithmetic')
        require(type(r['skips']) is int and 0<=r['skips']<.05*steps,'AMP recovery fraction')
        require(precision_ok(r['precision'],'torch.float16') and r['rng_isolated'] is True,'clean precision/native RNG')
        ids.add((r['parameters_sha256'],r['teacher_sha256'],r['scale'],r['lookahead'],r['skips']))
        require(positive(r['mean_ce']) and math.isfinite(r['mean_auxiliary']),'loss health')
        if arm=='masked' and phase=='active':
            require(all(precision_ok(r[k],'torch.float16') for k in ('auxiliary_precision','teacher_precision','visible_precision')),
                    'teacher/source/predictor FP16 observations')
            require(r['eligible']>0 and r['targets']>=r['eligible'] and r['positive_aux_attempts']>0
                and positive(r['mean_auxiliary']),'active task never executed')
        else:require(r['eligible']==r['targets']==r['positive_aux_attempts']==0 and r['mean_auxiliary']==0,'inactive/control acquired auxiliary work')
    require(len(ids)==1 and row['skips']==row['ranks'][0]['skips'],'replica/scaler/teacher disagreement')
    return max(w['seconds']/w['steps'] for r in row['ranks'] for w in r['windows'])


def restart(record,arm,transitions):
    labels=[19999,20000,20001,39999,40000,899999,900000] if transitions else list(range(100000,100007))
    require(record['passed'] is True and record['updates']==7 and record['full_state'] is True
        and record['policy']==PARITY and record['validation_boundaries']==1
        and record['labels']==labels and record['schedule']==('transitions' if transitions else 'active'),'restart coverage/policy')
    require([r['trial'] for r in record['trials']]==['reference','repeat','disk','resumed'],'restart trials')
    first=record['trials'][0]['updates']
    require(len(first)==7,'restart length')
    for trial in record['trials']:
        require(trial['exact_initial_restore'] is True and 0<=trial['max_state_error']<=PARITY['restart_absolute']
            and len(trial['updates'])==7,'restart state restoration')
        for i,(actual,expected) in enumerate(zip(trial['updates'],first)):
            require(actual['update']==i+1 and actual['schedule_label']==labels[i] and actual['skipped']==0
                and positive(actual['scale']) and actual['bn_isolated'] is True,'restart schedule/BN')
            for key in expected:
                if key in ('loss','ce','auxiliary'):
                    limit=min(PARITY['loss_absolute'],PARITY['loss_atol']+PARITY['loss_rtol']*abs(expected[key]))
                    require(math.isfinite(actual[key]) and abs(actual[key]-expected[key])<=limit,'restart float loss bound')
                else:require(actual[key]==expected[key],'restart masks/RNG/EMA/cursor schedule')
            if arm=='masked' and actual['coefficient']>0:
                require(isinstance(actual['mask_sha256'],str) and len(actual['mask_sha256'])==64,'restart masks missing')


def validate(spec,row,launch):
    require(row.get('schema')==SCHEMA and row.get('complete') is True and row.get('campaign_sha256')==spec['content_hash'],'incomplete/wrong campaign')
    require(row['policy']==POLICY and row['parity']==PARITY,'changed policy')
    runtime=row['runtime'];world=launch['resources']['gpus']
    require(runtime['resources']==launch['resources'] and runtime['platform']==launch['platform']
        and runtime['global_batch']==512 and runtime['broadcast_buffers'] is False
        and runtime['tf32'] is False and runtime['cudnn_benchmark'] is False,'runtime/BN policy')
    require(runtime['nccl_environment']=={'TORCH_NCCL_ASYNC_ERROR_HANDLING':'1'},'NCCL overrides')
    gpu=runtime['gpu'];token='A100' if world==2 else 'GH200'
    require(token in gpu['name'] and positive(gpu['total_memory']) and isinstance(gpu['driver'],str) and bool(gpu['driver'])
        and gpu['capability']==([8,0] if world==2 else [9,0]),'actual GPU identity')
    if token=='GH200':require(gpu['total_memory']>=80*2**30,'GH200 device memory')
    require(len(row['stream_replay'])==world and {r['rank'] for r in row['stream_replay']}==set(range(world)),'ROOT replay rank coverage')
    for r in row['stream_replay']:
        require(all(r[k] is True for k in ('passed','real_stream','no_aux_draws_during_replay','rng_restored')),'ROOT replay/aux RNG')
        require(set(r['streams'])=={'train','val'} and all(x['passed'] is True and x['consumed_batches']==2
            and len(x['next_sha256'])==len(x['next_mask_sha256'])==len(x['next_target_sha256'])==64
            and x['next_target_verified'] is True and x['next_target_events']==1
            and x['target_scope']=='fresh_frozen_teacher_next_batch' for x in r['streams'].values()),'ROOT continuation')
    require([m['arm'] for m in row['models']]==list(COUNTS),'paired arm coverage')
    require(row['models'][0]['first_batch_sha256']==row['models'][1]['first_batch_sha256'],'unmatched first rank batches')
    recommendations=[]
    for m in row['models']:
        arm=m['arm'];require(m['parameters']==COUNTS[arm] and m['trainable_parameters']==TRAINABLE[arm]
            and m['teacher_parameters']==(TEACHER if arm=='masked' else 0) and m['finite_state'] is True,'capacity/finite state')
        sa=timing(m['stream'],world,arm,'active',4200,POLICY['windows'])
        timing(m['fixed'],world,arm,'active',40,[40])
        sc=timing(m['clean'],world,arm,'clean',200,[200]) if arm=='masked' else sa
        require(len(m['correctness'])==world and {r['rank'] for r in m['correctness']}==set(range(world)),'correctness ranks')
        for r in m['correctness']:
            checks=r['checks'];required={'native','failure','learning'}
            if arm=='masked':required|={'predictor','views','leakage','sequential_fp32','sequential_amp','checkpoint_fp32','checkpoint_amp','auxiliary_gradient','taps','distributed','skip'}
            require(set(checks)==required and all(c['passed'] is True for c in checks.values()),'missing correctness gate')
            native=checks['native'];require(len(native['native_calls'])==13 and all(v==1 for v in native['native_calls'].values())
                and native['export_keys'] is True and native['gradients'] is True and native['nonzero_state'] is True
                and native['deployed_parameters']==COUNTS[arm],'native identity/export')
            require(checks['learning']['successful_updates']==8 and 8<=checks['learning']['attempts']<=20
                and checks['learning']['positive_loss'] is True,'actual learning')
            require(all(checks['failure'][k] is True for k in ('empty_input','nonfinite','rank_local_io','stop')),'coherent failures')
            if arm=='masked':
                require(all(checks['predictor'][k] is True for k in ('fp64','head_loop','gradients','jet_mean')),'predictor oracle')
                require(all(checks['views'][k] is True for k in ('mask_counts','canonical_permutation','visible_rebase','alignment','rng_exact','yaml_authenticated')),
                    'visible preprocessing/mask oracle')
                require(set(checks['leakage']['cases'])=={'hidden_p4','hidden_independent','full_relative','old_padding','postclean_bn','clean_pair_cache'}
                    and checks['leakage']['fixed_masks'] is True and checks['leakage']['fixed_queries'] is True
                    and checks['leakage']['b0_functional'] is True,'leakage/B0 gate')
                for key in ('sequential_fp32','sequential_amp'):
                    require(checks[key]['simultaneous_oracle'] is True and checks[key]['preclean_bn'] is True
                        and checks[key]['gradients']==len(spec['training_inventories']['masked']),'sequential gradients')
                for key in ('checkpoint_fp32','checkpoint_amp'):
                    require(checks[key]['checkpoint_backward'] is True and checks[key]['input_gradients'] is True
                        and checks[key]['cache_clear'] is True and checks[key]['particles']==97 and checks[key]['visible_particles']>64
                        and checks[key]['gradients']==len(spec['training_inventories']['masked']),'functional-buffer chunk recomputation')
                require(positive(checks['auxiliary_gradient']['backbone_gradient_norm'])
                    and checks['auxiliary_gradient']['teacher_detached'] is True,'auxiliary backbone gradient')
                require(checks['taps']['taps']==[6,8] and checks['taps']['normalizations']==3
                    and checks['taps']['parameters']==TEACHER and checks['taps']['detached'] is True,'teacher taps')
                d=checks['distributed'];require(d['world']==world and d['global_oracle'] is True and d['cpu_health_only'] is False
                    and d['public_no_sync'] is True and d['broadcast_buffers'] is False and d['find_unused_parameters'] is False
                    and [x['case'] for x in d['cases']]==['zero','one_rank','empty_last'],'actual-device DDP phases')
                require(all(checks['skip'][k] is True for k in ('parameter_ema_skipped','bn_copied'))
                    and checks['skip']['skipped']==checks['skip']['mask_draws']==1,'skip/EMA/BN transaction')
            restart(r['restart'],arm,False);restart(r['transitions'],arm,True)
        require(len(m['gpus'])==world and {x['rank'] for x in m['gpus']}==set(range(world))
            and all(0<x['peak_allocated']<.9*gpu['total_memory'] and x['peak_reserved']>=x['peak_allocated'] for x in m['gpus']),'device headroom')
        require(all(positive(x['host'].get('VmRSS_bytes')) and positive(x['host'].get('VmHWM_bytes'))
            and x['host']['VmHWM_bytes']>=x['host']['VmRSS_bytes'] for x in m['gpus']),'host peak evidence')
        val=m['validation'];require(val['events']==128000 and positive(val['seconds']) and precision_ok(val['precision'],'torch.float32'),'FP32 validation')
        for name,batch,steps in [('throughput',256,20),('latency',1,100),('full_validation',512,3)]:
            x=m['inference'][name];require(x['batch']==batch and x['particles']==128 and x['steps']==steps
                and positive(x['seconds']) and precision_ok(x['precision'],'torch.float32'),'exported FP32 inference')
        require(len(m['profiles'])==world and {x['rank'] for x in m['profiles']}==set(range(world)),'profile ranks')
        from .diagnostics import TRAIN_SCOPES
        for r in m['profiles']:
            require(len(r['profiles'])==2 and {p['training'] for p in r['profiles']}=={True,False},'train/export profiles')
            for p in r['profiles']:
                required=TRAIN_SCOPES if p['training'] and arm=='masked' else (
                    {'masked.bn_snapshot','masked.clean','masked.clean_backward','masked.optimizer','masked.ema'} if p['training'] else {'masked.export'})
                require(p['events']==2 and p['particles']==17 and required.issubset(p['stages']),'bounded profiling coverage')
                require(all(e['calls']>0 and e['cpu_us']>=0 and e['device_us']>=0 and math.isfinite(e['cpu_us']+e['device_us'])
                    and e['cpu_us']+e['device_us']>0 for e in p['stages'].values()),'CPU/device profile attribution')
        require(m['diagnostics']['events']==8 and m['diagnostics']['particles']==17,'bounded latent sample')
        if arm=='masked':require(all(math.isfinite(m['diagnostics'][k]) for k in ('target_variance','prediction_variance','target_norm','prediction_norm','cosine','zero_kv_max_change')),'latent diagnostic health')
        io=m['checkpoint'];require(io['student_only'] is True and io['full_state_includes_transfer'] is True
            and positive(io['transfer_seconds']) and io['restart_seconds']>io['transfer_seconds']
            and io['prediction_events']==100000 and all(positive(io[k])
            for k in ('restart_seconds','load_seconds','weights_seconds','prediction_write_seconds','prediction_load_seconds')),'full state/student/prediction I/O')
        require(positive(m['cold_fetch_seconds']) and positive(m['metric_5m_seconds']),'cold/merge metric measurements')
        compute=(879999*sa+120001*sc) if arm=='masked' else 1e6*sa
        validation=500*val['seconds'];writes=1102*io['restart_seconds']+51*io['weights_seconds']
        train=math.ceil(1.3*(compute+validation+writes+m['cold_fetch_seconds'])/3600)
        evaluation=math.ceil(1.3*(5e6/128000*val['seconds']+m['metric_5m_seconds']+io['load_seconds']+
            50*(io['prediction_write_seconds']+io['prediction_load_seconds']))/3600)
        recommendations.append(dict(arm=arm,parameters=COUNTS[arm],trainable_parameters=TRAINABLE[arm],
            train_hours=max(1,train),evaluate_hours=max(1,evaluation),compute_hours=compute/3600,
            all_active_compute_hours=1e6*sa/3600,validation_hours=validation/3600,write_hours=writes/3600,
            active_seconds_per_attempt=sa,clean_seconds_per_attempt=sc,train_jets_per_second=m['stream']['jets_per_second'],
            peak_gib=max(g['peak_allocated'] for g in m['gpus'])/2**30))
    return dict(passed=True,complete=True,authorizes_training=False,models=recommendations,runtime=runtime,
        note='Independent platform, matched clean exposure; phase-weighted estimate +30%, excludes queue/replay. Reviewed dispatch required.')
