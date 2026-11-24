"""Recompute admission and resource recommendations from raw paired evidence."""
import math
from .config import SCHEMA, POLICY, PARITY, COUNTS
from part_ev_message_study.protocol import precision_ok


def positive(x):return type(x) in (float,int) and math.isfinite(x) and x>0


def require(condition,message):
    if not condition:raise ValueError('Workspace admission: '+message)


def validate(spec,row,launch):
    require(row.get('schema')==SCHEMA and row.get('complete') is True,'incomplete benchmark')
    require(row.get('campaign_sha256')==spec['content_hash'],'wrong campaign')
    require(row.get('policy')==POLICY and row.get('parity')==PARITY,'changed policy/tolerances')
    runtime=row['runtime'];resources=launch['resources'];world=resources['gpus']
    require(runtime['resources']==resources and runtime['platform']==launch['platform']
        and runtime['global_batch']==512 and runtime['tf32'] is False and runtime['cudnn_benchmark'] is False,'runtime mismatch')
    require(runtime['nccl_environment']=={'TORCH_NCCL_ASYNC_ERROR_HANDLING':'1'},'unapproved NCCL environment')
    gpu=runtime['gpu'];token='A100' if launch['platform']=='sporc' else 'GH200'
    require(token in gpu['name'] and positive(gpu['total_memory'])
        and isinstance(gpu.get('driver'),str) and bool(gpu['driver'])
        and gpu.get('capability')==([8,0] if token=='A100' else [9,0]),'wrong GPU/driver identity')
    if token=='GH200':require(gpu['total_memory']>=80*2**30,'GH200 memory')
    require(len(row['stream_replay'])==world and {r['rank'] for r in row['stream_replay']}==set(range(world)),
            'missing real-stream replay')
    for r in row['stream_replay']:
        require(r['passed'] is True and r['real_stream'] is True and r['model_updates']==0
            and r['rng_restored'] is True,'invalid replay')
        require(set(r['streams'])=={'train','val'} and all(v['consumed_batches']==2 and v['passed'] is True
            and len(v['next_sha256'])==64 for v in r['streams'].values()),'stream continuation missing')
    require([m['arm'] for m in row['models']]==list(COUNTS),'missing/duplicate arms')
    require(row['models'][0]['first_batch_sha256']==row['models'][1]['first_batch_sha256'],'unmatched input streams')
    models=[]
    for m in row['models']:
        arm=m['arm'];require(m['parameters']==COUNTS[arm] and m['finite_state'] is True,'capacity/finite state')
        require(len(m['first_batch_sha256'])==world,'rank fingerprints')
        for part,steps,windows in [('stream',4200,POLICY['windows']),('fixed',40,[40])]:
            data=m[part]
            require(data['measured_steps']==steps and data['global_batch']==512 and positive(data['seconds']),'timing coverage')
            require(len(data['ranks'])==world and {r['rank'] for r in data['ranks']}==set(range(world)),'rank timings')
            require(all(positive(r['seconds']) for r in data['ranks']) and math.isclose(
                data['seconds'],max(r['seconds'] for r in data['ranks']),rel_tol=1e-10),'worst-rank time')
            require(math.isclose(data['jets_per_second'],512*steps/data['seconds'],rel_tol=1e-10),'throughput arithmetic')
            ids=set()
            for rank in data['ranks']:
                require([w['steps'] for w in rank['windows']]==windows and all(positive(w['seconds']) for w in rank['windows']),'window coverage')
                require(sum(w['seconds'] for w in rank['windows'])<=rank['seconds']*1.001,'window time arithmetic')
                require(precision_ok(rank['precision'],'torch.float16'),'FP16 observed activation audit')
                require(type(rank['skips']) is int and 0<=rank['skips']<.05*steps,'AMP skips')
                ids.add((rank['parameters_sha256'],rank['scale'],rank['skips'],rank['lookahead']))
            require(len(ids)==1,'replica parameters/scaler/Lookahead differ')
            require(data['skips']==data['ranks'][0]['skips'],'skip arithmetic')
        require(len(m['correctness'])==world and {r['rank'] for r in m['correctness']}==set(range(world)),'rank correctness')
        for r in m['correctness']:
            checks=r['checks'];required={'initial_identity','active_fp32','active_amp','distributed','rejection','learning'}
            if arm=='workspace':required.add('attention_oracle')
            require(set(checks)==required and all(c['passed'] is True for c in checks.values()),'missing correctness gate')
            initial=checks['initial_identity']
            require({str(k):v for k,v in initial['native_calls'].items()}=={str(i):1 for i in range(8)}
                and initial['native_pipeline']==dict(trimmer=1,embedding=1,pairs=1,classifier=1,
                    class_attention_0=1,class_attention_1=1)
                and initial['nonzero_state'] is True and initial['rng_exact'] is True,'native identity/calls')
            if arm=='workspace':
                oracle=checks['attention_oracle']
                require(oracle['precision']=='fp64' and oracle['independent_head_loop'] is True
                    and oracle['gradients'] is True,'independent attention oracle')
                stage=oracle['full_stage']
                require(stage['passed'] is True and stage['precision']=='fp64'
                    and all(stage[k] is True for k in ('independent_stage','gradients','permutation','padding','event_isolation')),
                    'independent full-stage oracle')
            for name in ('active_fp32','active_amp'):
                require(checks[name]['gradients']==len(spec['model_inventories'][arm]) and checks[name]['checkpoint_backward'] is True,'active gradients coverage')
            distributed=checks['distributed']
            require(distributed['world_size']==world and distributed['global_events']==8
                and distributed['one_particle'] is True and distributed['unequal_lengths'] is True,'actual GPU/DDP boundary proof')
            learning=checks['learning']
            require(learning['successful_updates']==8 and 8<=learning['attempts']<=20,'Ranger activation')
            require(all(checks['rejection'][k] is True for k in ('empty_input','nonfinite','rank_local_io','stop')),'failure coordination')
            restart=r['restart']
            require(restart['passed'] is True and restart['updates']==7 and restart['full_state'] is True
                and restart['policy']==PARITY and restart['validation_boundaries']==1,'restart policy')
            require([t['trial'] for t in restart['trials']]==['reference','repeat','disk','resumed'],'restart trial coverage')
            for trial in restart['trials']:
                require(trial['exact_initial_restore'] is True and len(trial['updates'])==7
                    and 0<=trial['max_state_error']<=PARITY['restart_absolute'],'restart state bounds')
                require([u['update'] for u in trial['updates']]==list(range(1,8))
                    and all(u['skipped']==0 and positive(u['scale']) and math.isfinite(u['loss']) for u in trial['updates']),'restart schedule/finite loss')
                reference=restart['trials'][0]['updates']
                for actual,expected in zip(trial['updates'],reference):
                    require(actual['scale']==expected['scale'] and actual['lookahead']==expected['lookahead']
                        and type(actual['lookahead']) is int and 0<=actual['lookahead']<6,'restart scaler/Lookahead schedule')
                    limit=min(PARITY['loss_absolute'],PARITY['loss_atol']+PARITY['loss_rtol']*abs(expected['loss']))
                    require(abs(actual['loss']-expected['loss'])<=limit,'restart loss bound')
        require(len(m['gpus'])==world and {g['rank'] for g in m['gpus']}==set(range(world)),'GPU peaks')
        require(all(0<g['peak_allocated']<.9*gpu['total_memory'] and g['peak_reserved']>=g['peak_allocated'] for g in m['gpus']),'10% allocated-memory headroom')
        val=m['validation']
        require(val['events']==128000 and positive(val['seconds']) and precision_ok(val['precision'],'torch.float32'),'FP32 streaming validation')
        for name,batch,steps in [('throughput',256,20),('latency',1,100),('full_validation',512,3)]:
            i=m['inference'][name]
            require(i['batch']==batch and i['particles']==128 and i['steps']==steps and positive(i['seconds'])
                and precision_ok(i['precision'],'torch.float32'),'FP32 inference/stress coverage')
        require(len(m['profiles'])==world and {r['rank'] for r in m['profiles']}==set(range(world)),'small profiles')
        for r in m['profiles']:
            require(len(r['profiles'])==2 and {p['training'] for p in r['profiles']}=={True,False},'train/eval profile')
            for p in r['profiles']:
                require(p['events']==2 and p['particles']==17,'bounded profile size')
                if arm=='workspace':
                    keys={f'workspace.stages.{i}.{s}' for i in range(4) for s in ('read','mix','write','ffn')}
                    keys.add('workspace.class_read.attention')
                    require(set(p['stages'])==keys,'profile stages')
                    require(all(e['calls']>0 and e['cpu_us']>=0 and e['device_us']>=0
                        and math.isfinite(e['cpu_us']+e['device_us']) and e['cpu_us']+e['device_us']>0
                        for e in p['stages'].values()),'CPU/device profile attribution')
        require(m['diagnostics']['events']==8 and m['diagnostics']['particles']==17,'bounded diagnostics')
        if arm=='workspace':require(len(m['diagnostics']['layers'])==4 and len(m['diagnostics']['attention_entropy'])==13,'workspace diagnostics paths')
        io=m['checkpoint'];require(all(positive(io[k]) for k in ('restart_seconds','load_seconds','weights_seconds')),'checkpoint I/O timing')
        require(positive(m['cold_fetch_seconds']),'cold fetch timing')
        require(positive(m['metric_5m_seconds']),'full-population CPU metric cost')
        s=max(w['seconds']/w['steps'] for r in m['stream']['ranks'] for w in r['windows'])
        compute=1e6*s;validation=50*10*val['seconds']
        writes=1102*io['restart_seconds']+51*io['weights_seconds']
        train=math.ceil(1.3*(compute+validation+writes+m['cold_fetch_seconds'])/3600)
        # Streaming timing includes I/O and metrics; add measured synthetic 5M
        # metric cost and checkpoint load rather than claiming kernel-only ETA.
        evaluation=math.ceil(1.3*(5e6/128000*val['seconds']+m['metric_5m_seconds']+io['load_seconds'])/3600)
        models.append(dict(arm=arm,parameters=COUNTS[arm],train_hours=max(1,train),evaluate_hours=max(1,evaluation),
            compute_hours=compute/3600,validation_hours=validation/3600,write_hours=writes/3600,
            train_jets_per_second=m['stream']['jets_per_second'],peak_gib=max(g['peak_allocated'] for g in m['gpus'])/2**30))
    return dict(passed=True,complete=True,authorizes_training=False,models=models,runtime=runtime,
        note='Matched single-seed full schedule; 30% margin excludes queue/replay. Dispatch requires explicit authority.')
