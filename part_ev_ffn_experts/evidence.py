"""Fail-closed recomputation of experts-specific raw admission and projections."""
import math
from .config import SCHEMA,SCIENCE,COUNTS,POLICY,PARITY
from .precision import precision_ok


def require(value,message):
    if not value:raise ValueError('Experts admission: '+message)


def positive(x):return type(x) in (int,float) and math.isfinite(x) and x>0


def routes_ok(counts):
    try:
        if set(counts)!={'requested','executed','tokens','primary','secondary'}:return False
        if len(counts['tokens'])!=6:return False
        for b in range(6):
            t=counts['tokens'][b]
            if type(t) is not int or t<=0:return False
            for key in ('requested','executed','primary','secondary'):
                row=counts[key][b]
                if len(row)!=4 or any(type(x) is not int or x<0 for x in row):return False
            r,c,a,z=(counts[k][b] for k in ('requested','executed','primary','secondary'))
            if r!=c or sum(r)!=2*t or sum(a)!=t or sum(z)!=t or [x+y for x,y in zip(a,z)]!=r:return False
    except (KeyError,TypeError,IndexError):return False
    return True


def restart_ok(row,arm):
    require(row['passed'] is True and row['updates']==7 and row['full_state'] is True
        and row['validation_boundaries']==1 and row['policy']==PARITY
        and row['fixture']=='trained_outputs_separated_router_biases_rolling_native_trim_dropout_v1','restart identity')
    require([t['trial'] for t in row['trials']]==['reference','repeat','disk','resumed'],'restart trials')
    reference=row['trials'][0]['updates']
    for trial in row['trials']:
        require(trial['exact_initial_restore'] is True and 0<=trial['max_state_error']<=PARITY['restart_absolute']
            and (trial['max_state_error']==0 or isinstance(trial['worst_tensor_path'],str)),'restart state bounds')
        require([u['update'] for u in trial['updates']]==list(range(1,8)),'restart update coverage')
        for actual,expected in zip(trial['updates'],reference):
            require(actual['skipped']==0 and positive(actual['scale']) and actual['scale']==expected['scale']
                and actual['lookahead']==expected['lookahead'] and type(actual['lookahead']) is int
                and 0<=actual['lookahead']<6 and actual['lr']==expected['lr']
                and len(actual['input_sha256'])==64 and actual['input_sha256']==expected['input_sha256'],
                'exact restart schedules/input')
            require(len(actual['routes']['sha256'])==64 and actual['routes']['sha256']==expected['routes']['sha256'],
                'exact discrete routes')
            if arm=='experts':
                require(routes_ok(actual['counts']) and actual['counts']==expected['counts']
                    and positive(actual['routes']['minimum_margin']),'exact requested/executed counts')
            else:require(actual['counts'] is None and actual['balance'] is None,'control auxiliary contamination')
            for field in ('loss','ce','balance'):
                a,b=actual[field],expected[field]
                if b is None:require(a is None,'control auxiliary value')
                else:
                    limit=min(PARITY['loss_absolute'],PARITY['loss_atol']+PARITY['loss_rtol']*abs(b))
                    require(math.isfinite(a) and abs(a-b)<=limit,'restart computed loss bound')


def correctness(spec,row,arm,world):
    require(len(row)==world and {r['rank'] for r in row}==set(range(world)),'rank correctness coverage')
    for r in row:
        c=r['checks'];keys={'initial_identity','active_fp32','active_amp','distributed','rejection','learning'}
        if arm=='experts':keys|={'experts_oracle','bank_ddp','coverage'}
        require(set(c)==keys and all(e['passed'] is True for e in c.values()),'correctness gates')
        initial=c['initial_identity'];names={'trimmer','embedding','pairs','classifier'}
        names|={f'{p}_{i}' for p in ('attention','fc1','fc2') for i in range(8)}
        names|={f'class_{i}' for i in range(2)}|{f'bank_{i}' for i in range(6)}
        require(initial['native_calls']=={n:1 for n in names} and all(initial[k] is True for k in
            ('rng_exact','trimmed_inputs_exact','nonzero_state','shared_ce_gradient')),'native initial equivalence')
        for key,amp in [('active_fp32',False),('active_amp',True)]:
            v=c[key]
            require(v['gradients']==len(spec['model_inventories'][arm]) and v['amp'] is amp
                and all(v[k] is True for k in ('input_gradients','checkpoint_backward','exact_routes')),'active gradient/recompute coverage')
        d=c['distributed']
        require(d['world_size']==world and d['global_events']==8 and d['fp32'] is True and d['dropout'] is False
            and d['find_unused_parameters'] is False and all(d[k] is True for k in
                ('one_particle','unequal_lengths','unequal_tokens','input_gradient_world_factor')),'actual global model parity')
        learning=c['learning']
        require(learning['successful_updates']==8 and 8<=learning['attempts']<=20
            and learning['skips']==learning['attempts']-8
            and learning['alpha']==(.001 if arm=='experts' else 0.)
            and learning['auxiliary_included'] is (arm=='experts'),'real Ranger learning/loss')
        require(all(c['rejection'][k] is True for k in ('empty_input','nonfinite','rank_local_io','stop')),'coordinated failure gates')
        if arm=='experts':
            require(c['rejection']['bad_stats'] is True,'malformed routing counts rejection')
            oracle=c['experts_oracle']
            require(all(oracle[k] is True for k in ('fp64','independent_particle_loop','gradients','exact_routes',
                'unequal_count_proxy','shared_parameter_gradients','input_gradient_world_factor','naive_average_rejected',
                'stable_ties','device_empty_gradients')),'independent math/loss proof')
            d=c['bank_ddp']
            require(d['world_size']==world and d['actual_data_parallel'] is (world==2) and all(d[k] is True for k in
                ('unequal_tokens','local_empty','global_empty','all_gradients','shared_parameter_gradients','input_gradient_world_factor')),
                'empty-population actual data-parallel proof')
            require([p['phase'] for p in d['phases']]==['empty_local','empty_global']
                and all(p['zero_drops'] is True and sum(p['counts'])==2*p['global_tokens'] for p in d['phases']),
                'empty-population route count coverage')
            coverage=c['coverage']
            require(coverage['fixture']=='controlled_postnorm_two_sign_coverage_ce_only_v1'
                and [b['block'] for b in coverage['banks']]==SCIENCE['blocks']
                and all(b['all_four_learned'] is True and b['successful_updates']==8
                    and all(positive(v) for v in b['gradient_norms'].values()) for b in coverage['banks']),
                'all bank/expert CE learning coverage')
        restart_ok(r['restart'],arm)


def timing(m,part,steps,windows,world,paths):
    data=m[part]
    require(data['measured_steps']==steps and data['global_batch']==512 and positive(data['seconds']),'timing coverage')
    require(len(data['ranks'])==world and {r['rank'] for r in data['ranks']}==set(range(world)),'rank timings')
    require(math.isclose(data['seconds'],max(r['seconds'] for r in data['ranks']),rel_tol=1e-10)
        and math.isclose(data['jets_per_second'],512*steps/data['seconds'],rel_tol=1e-10),'throughput arithmetic')
    replicas=set()
    for r in data['ranks']:
        require([w['steps'] for w in r['windows']]==windows and all(positive(w['seconds']) for w in r['windows'])
            and positive(r['seconds']) and sum(w['seconds'] for w in r['windows'])<=r['seconds']*1.001,'window coverage/arithmetic')
        require(precision_ok(r['precision'],'torch.float16',paths),'path-aware training precision')
        require(type(r['skips']) is int and 0<=r['skips']<.05*steps,'sustained AMP skips')
        replicas.add((r['parameters_sha256'],r['scale'],r['skips'],r['lookahead']))
        losses=r['loss_sums']
        require(all(math.isfinite(v) and v>=0 for v in losses.values()) and losses['ce']>0,'loss coverage')
        expected=losses['ce']+.001*losses['balance'] if m['arm']=='experts' else losses['ce']
        require(math.isclose(losses['total'],expected,rel_tol=1e-5,abs_tol=1e-4),'actual auxiliary loss arithmetic')
        counts=r['routing_totals']
        if m['arm']=='experts':
            require(losses['balance']>0 and routes_ok(counts),
                'real timing dropless routing/loss coverage')
        else:require(counts is None and losses['balance']==0,'control has routing work')
    require(len(replicas)==1 and data['skips']==data['ranks'][0]['skips'],'replica optimizer/scaler mismatch')
    require(all(r['routing_totals']==data['ranks'][0]['routing_totals'] for r in data['ranks']),'global counters differ across ranks')


def validate(spec,row,launch):
    require(row.get('schema')==SCHEMA and row.get('complete') is True and row['campaign_sha256']==spec['content_hash'],
        'benchmark campaign identity')
    require(row['policy']==POLICY and row['parity']==PARITY,'changed policy/tolerances')
    runtime=row['runtime'];resources=launch['resources'];world=resources['gpus'];gpu=runtime['gpu']
    token='A100' if launch['platform']=='sporc' else 'GH200'
    require(runtime['platform']==launch['platform'] and runtime['resources']==resources and runtime['global_batch']==512
        and runtime['tf32'] is False and runtime['cudnn_benchmark'] is False
        and runtime['nccl_environment']=={'TORCH_NCCL_ASYNC_ERROR_HANDLING':'1'},'runtime contract')
    require(token in gpu['name'] and positive(gpu['total_memory']) and bool(gpu['driver'])
        and gpu['capability']==([8,0] if token=='A100' else [9,0]),'GPU identity')
    if token=='GH200':require(gpu['total_memory']>=80*2**30,'GH200 memory')
    require(len(row['stream_replay'])==world and {r['rank'] for r in row['stream_replay']}==set(range(world)),'real-stream ranks')
    for r in row['stream_replay']:
        require(r['passed'] is True and r['real_stream'] is True and r['model_updates']==0 and r['rng_restored'] is True
            and set(r['streams'])=={'train','val'} and all(s['passed'] is True and s['consumed_batches']==2
                and len(s['next_sha256'])==64 for s in r['streams'].values()),'ROOT replay/RNG evidence')
    require([m['arm'] for m in row['models']]==list(COUNTS),'paired arms')
    require(row['models'][0]['first_batch_sha256']==row['models'][1]['first_batch_sha256'],'unmatched stream starts')
    models=[]
    for m in row['models']:
        arm=m['arm'];paths=spec['precision_paths'][arm]
        require(m['parameters']==COUNTS[arm] and m['finite_state'] is True and len(m['first_batch_sha256'])==world,'capacity/finite state')
        timing(m,'stream',4200,POLICY['windows'],world,paths);timing(m,'fixed',40,[40],world,paths)
        correctness(spec,m['correctness'],arm,world)
        require(len(m['gpus'])==world and {g['rank'] for g in m['gpus']}==set(range(world))
            and all(0<g['peak_allocated']<.9*gpu['total_memory'] and g['peak_reserved']>=g['peak_allocated'] for g in m['gpus']),
            '10% allocated GPU-memory headroom')
        v=m['validation']
        require(v['events']==128000 and positive(v['seconds']) and precision_ok(v['precision'],'torch.float32',paths),'FP32 streamed validation')
        for name,batch,steps in [('throughput',256,20),('latency',1,100),('full_validation',512,3)]:
            i=m['inference'][name]
            require(i['batch']==batch and i['particles']==128 and i['steps']==steps and positive(i['seconds'])
                and precision_ok(i['precision'],'torch.float32',paths),'FP32 inference/stress')
        require(len(m['profiles'])==world and {p['rank'] for p in m['profiles']}==set(range(world)),'profile ranks')
        for p in m['profiles']:
            require(len(p['profiles'])==2 and {a['training'] for a in p['profiles']}=={True,False},'train/eval profile')
            for a in p['profiles']:
                require(a['events']==2 and 0<a['particles']<=32,'bounded profiles')
                if arm=='experts':
                    require(set(a['stages'])=={'experts.context_router','experts.stable_selection','experts.packing','experts.mlp','experts.combination'}
                        and all(s['calls']>0 and s['cpu_us']>=0 and s['device_us']>=0
                            and math.isfinite(s['cpu_us']+s['device_us']) and s['cpu_us']+s['device_us']>0 for s in a['stages'].values()),
                            'scope attribution incl. CPU-only routing work')
        diagnostic=m['diagnostics'];require(diagnostic['events']==8 and diagnostic['particles']<=32,'bounded diagnostics')
        if arm=='experts':
            require([b['block'] for b in diagnostic['banks']]==SCIENCE['blocks']
                and all(math.isfinite(b[k]) for b in diagnostic['banks'] for k in
                    ('full_entropy','selected_entropy','minimum_margin','correction_dense_rms_ratio','normalized_input_rms')), 'routing diagnostics')
            require(diagnostic['alpha']==.001 and all(b['requested']==b['executed'] and sum(b['requested'])==2*b['tokens']
                and b['healthy'] is True for b in diagnostic['banks'])
                and set(diagnostic['gradient_norms'])=={'router','experts'}
                and all(set(v)=={'ce','auxiliary'} and all(math.isfinite(n) and n>=0 for n in v.values())
                    for v in diagnostic['gradient_norms'].values()),'routing/gradient diagnostics')
        io=m['checkpoint']
        require(all(positive(io[k]) for k in ('restart_seconds','load_seconds','weights_seconds'))
            and positive(m['metric_5m_seconds']) and positive(m['cold_fetch_seconds']),'I/O/metric cost')
        compute=1e6*max(w['seconds']/w['steps'] for r in m['stream']['ranks'] for w in r['windows'])
        val=500*v['seconds'];writes=1102*io['restart_seconds']+51*io['weights_seconds']
        train=math.ceil(1.3*(compute+val+writes+m['cold_fetch_seconds'])/3600)
        evaluate=math.ceil(1.3*(5e6/128000*v['seconds']+m['metric_5m_seconds']+io['load_seconds'])/3600)
        models.append(dict(arm=arm,parameters=COUNTS[arm],train_hours=max(1,train),evaluate_hours=max(1,evaluate),
            compute_hours=compute/3600,validation_hours=val/3600,write_hours=writes/3600,
            train_jets_per_second=m['stream']['jets_per_second'],peak_gib=max(g['peak_allocated'] for g in m['gpus'])/2**30))
    return dict(passed=True,complete=True,authorizes_training=False,models=models,runtime=runtime,
        note='Paired full schedule, experts-aware global balancing. 30% margin excludes queue/replay; explicit dispatch authority required.')
