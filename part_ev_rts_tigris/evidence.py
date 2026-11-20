"""Native-one-GH200 gates; model-specific checks retain their original tolerances.

The per-model admission checks below deliberately operate on one real rank.
They never duplicate a rank or reinterpret GH200 evidence as an A100 benchmark.
"""
import importlib
import math
from . import protocol as P

RENAMES = {'active_SyncBN_DDP':'active_native_global_batch', 'no_pairs_DDP':'no_pairs_native',
           'unequal_lengths_DDP':'native_trimming', 'tie_only_rank_DDP':'tie_mixture_native'}
EXECUTION = dict(world_size=1, batchnorm='native', distributed=False, device_type='cuda')


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


from part_ev_message_study.protocol import precision_ok

def check_measurement(row, steps):
    if (row['measured_steps'] != steps or row['healthy'] is not True
            or not positive(row['seconds']) or not 0 <= row['skips'] < steps*.05
            or not math.isclose(row['jets_per_second'],512*steps/row['seconds'],rel_tol=1e-10)
            or len(row['ranks']) != 1): raise ValueError('Invalid sustained timing/health evidence')
    identities = set()
    windows = [min(1000, steps-i) for i in range(0,steps,1000)]
    for rank in row['ranks']:
        if (not precision_ok(rank['precision'], 'torch.float16')
                or [w['steps'] for w in rank['windows']] != windows
                or rank['skips'] != row['skips']
                or any(not positive(w['seconds']) for w in rank['windows'])):
            raise ValueError('Missing windows/precision')
        identities.add((rank['parameters_sha256'],rank['scaler'],rank['skips'],rank['lookahead_counter']))
    if len(identities) != 1: raise ValueError('Inconsistent native state')


def summarize_readout(spec, row):
    science = importlib.import_module(P.PACKAGES['readout']+'.protocol')
    PARAMETERS, BASE_PARAMETERS, PARITY = science.PARAMETERS, science.BASE_PARAMETERS, science.PARITY
    CHECKS = {RENAMES.get(k,k) for k in science.CHECKS}
    RESTART = dict(science.RESTART, ranks=1)

    if row.get('campaign_sha256') != spec['content_hash']: raise ValueError('Wrong benchmark campaign')
    if row.get('authorizes_training') is not False: raise ValueError('Benchmark never authorizes training')
    if row.get('passed') is not True or row.get('complete') is not True:
        return dict(passed=False, complete=False, authorizes_training=False, error=row.get('error'))
    if row.get('parameters') != PARAMETERS or row.get('baseline_parameters') != BASE_PARAMETERS:
        raise ValueError('Measured model count differs')
    for part, steps in (('stream',4200), ('fixed',40), ('baseline_stream',4200), ('baseline_fixed',40)):
        check_measurement(row[part], steps)
    checks = row['correctness']
    if (len(checks) != 1 or {r['rank'] for r in checks} != {0}
            or any(r['passed'] is not True or r['tolerances'] != PARITY
                   or {c['kind'] for c in r['checks']} != CHECKS
                   or any(c['passed'] is not True for c in r['checks']) for r in checks)):
        raise ValueError('Incomplete correctness evidence')
    for rank in checks:
        path=next(c for c in rank['checks'] if c['kind']=='unchanged_particle_path')
        if (path.get('attention_calls')!=8 or path.get('trimmer_calls')!=1
                or path.get('exact_inputs_masks_trimming_rng') is not True
                or path.get('exact_base_parameters') is not True
                or path.get('forward_rtol')!=PARITY['forward_rtol']
                or path.get('forward_atol')!=PARITY['forward_atol']
                or len(path.get('block_max_abs_difference',[]))!=8
                or any(type(v) not in (int,float) or not math.isfinite(v) or v<0
                       for v in path['block_max_abs_difference'])):
            raise ValueError('Missing audited native particle-path parity')
        delayed = rank['delayed_activation']
        if (delayed.get('passed') is not True or delayed['successful_updates'] != 3
                or not 3 <= delayed['attempts'] <= 20 or delayed['skips'] != delayed['attempts']-3
                or not all(delayed[k] is True for k in ('initial_upstream_zero','initial_output_signal',
                                                       'upstream_updated','all_readout_gradients'))):
            raise ValueError('Readout learning path not established')
    if (row['restart'] != RESTART or row.get('full128_validation_passed') is not True
            or row['finite_state'] is not True or row['baseline_finite_state'] is not True
            or not positive(row['checkpoint_seconds'])):
        raise ValueError('Restart/finite-state evidence failed')
    runtime = P.runtime()
    if row['runtime'] != runtime: raise ValueError('Unexpected benchmark runtime')
    for name in ('gpus','baseline_gpus'):
        gpus = row[name]
        if (len(gpus) != 1 or {g['rank'] for g in gpus} != {0}
                or len({(g['name'],g['total_memory']) for g in gpus}) != 1
                or any('GH200' not in g['name'] or not 0 < g['peak_allocated'] < .9*g['total_memory']
                       or g['peak_reserved'] < g['peak_allocated'] for g in gpus)):
            raise ValueError('Need one GH200 and 10% allocation headroom')
    by_rank=lambda items:{g['rank']:(g['name'],g['total_memory'],g['first_input_sha256']) for g in items}
    if by_rank(row['gpus'])!=by_rank(row['baseline_gpus']):
        raise ValueError('Baseline/candidate hardware or initial rank data differ')
    val = row['validation']
    if (val['events'] != 128000 or not positive(val['seconds'])
            or not precision_ok(val['precision'],'torch.float32')):
        raise ValueError('Incomplete FP32 streaming validation')
    for name in ('inference','baseline_inference'):
        inf = row[name]
        if (inf['batch_size'] != 256 or inf['particles'] != 128 or inf['batches'] != 20
                or inf['precision'] != 'fp32' or inf['batch1_iterations'] != 100
                or not positive(inf['seconds']) or not positive(inf['batch1_seconds'])):
            raise ValueError('Incomplete inference/latency evidence')
    diagnostic_fields = {'summary_rms','summary_similarity','query0_norm','query1_norm','delta_to_class_norm',
        'key_rms','value_rms','pair_weight_similarity','mean_pair_count',
        'disabled_logit_rms_change','swapped_summary_logit_rms_change','zero_summary_logit_rms_change'}
    diagnostic_fields |= {f'query{a}_{kind}' for a in range(2)
                         for kind in ('normalized_entropy','max_weight','effective_support')}
    if (len(row['diagnostics']) != 1
            or any(d['healthy'] is not True or d['events'] != 8 or d['particles'] != 17
                   or d['logits_finite'] is not True or d['eligible_jets'] != 8
                   or any(type(d.get(k)) is not float or not math.isfinite(d[k]) for k in diagnostic_fields)
                   for d in row['diagnostics'])): raise ValueError('Missing fixed-sample diagnostics')
    for d in row['diagnostics']:
        if any(not 0 <= d[f'query{a}_normalized_entropy'] <= 1.00001
               or not 0 < d[f'query{a}_max_weight'] <= 1
               or not 1-1e-5 <= d[f'query{a}_effective_support'] <= d['mean_pair_count']+1e-3
               for a in range(2)): raise ValueError('Invalid pooling diagnostic')
    gradients=row['readout_gradient_norms']
    if len(gradients)!=1 or any(len(g)!=7 or any(not positive(v) for v in g.values()) for g in gradients):
        raise ValueError('Missing finite active readout gradient norms')
    train = max(w['seconds']/w['steps'] for r in row['stream']['ranks'] for w in r['windows'])*1e6
    validation = val['seconds'] * 50 * 1280000/val['events']
    checkpoint = (1000+150+2)*row['checkpoint_seconds']
    return dict(passed=True,complete=True,authorizes_training=False,parameters=PARAMETERS,
        train_jets_per_second=row['stream']['jets_per_second'],
        training_time_vs_state_ev=row['baseline_stream']['jets_per_second']/row['stream']['jets_per_second'],
        fixed128_inference_jets_per_second=5120/row['inference']['seconds'],
        inference_time_vs_state_ev=row['inference']['seconds']/row['baseline_inference']['seconds'],
        batch1_latency_ms=10*row['inference']['batch1_seconds'],
        peak_allocated_gib=max(g['peak_allocated'] for g in row['gpus'])/2**30,
        projected_train_hours=train/3600,projected_validation_hours=validation/3600,
        projected_checkpoint_hours=checkpoint/3600,
        recommended_train_hours=math.ceil(1.3*(train+validation+checkpoint)/3600),
        recommended_evaluate_hours=max(1,math.ceil(1.3*val['seconds']*5e6/val['events']/3600)),
        note='Full-schedule projection, +30% margin; excludes queue/replay. No accuracy or production approval.')



def summarize_triangles(spec, row):
    science = importlib.import_module(P.PACKAGES['triangles']+'.protocol')
    PARAMETERS, BASE_PARAMETERS, PARITY = science.PARAMETERS, science.BASE_PARAMETERS, science.PARITY
    CHECKS = {RENAMES.get(k,k) for k in science.CHECKS}
    RESTART = dict(science.RESTART, ranks=1)
    CONFIG, PROFILE_POLICY, PROFILE_STAGES = science.CONFIG, science.PROFILE_POLICY, science.PROFILE_STAGES
    validate_histogram = science.validate_histogram

    if row.get('campaign_sha256') != spec['content_hash']: raise ValueError('Wrong benchmark campaign')
    if row.get('authorizes_training') is not False: raise ValueError('Benchmark never authorizes training')
    if row.get('passed') is not True or row.get('complete') is not True:
        return dict(passed=False, complete=False, authorizes_training=False, error=row.get('error'))
    if row.get('parameters') != PARAMETERS or row.get('baseline_parameters') != BASE_PARAMETERS:
        raise ValueError('Measured model count differs')
    for part, steps in (('stream',4200), ('fixed',40), ('baseline_stream',4200), ('baseline_fixed',40)):
        check_measurement(row[part], steps)
    checks = row['correctness']
    if (len(checks) != 1 or {r['rank'] for r in checks} != {0}
            or any(r['passed'] is not True or r['tolerances'] != PARITY
                   or {c['kind'] for c in r['checks']} != CHECKS
                   or any(c['passed'] is not True for c in r['checks']) for r in checks)):
        raise ValueError('Incomplete correctness evidence')
    for rank in checks:
        delayed = rank['delayed_activation']
        if (delayed.get('passed') is not True or delayed['successful_updates'] != 8
                or not 8 <= delayed['attempts'] <= 20 or delayed['skips'] != delayed['attempts']-8
                or not all(delayed[k] is True for k in ('initial_upstream_zero','initial_output_signal',
                                                       'upstream_updated','all_triangle_gradients'))):
            raise ValueError('Triangle learning path not established')
    if (row['restart'] != RESTART or row.get('full128_validation_passed') is not True
            or row['finite_state'] is not True or row['baseline_finite_state'] is not True
            or not positive(row['checkpoint_seconds'])):
        raise ValueError('Restart/finite-state evidence failed')
    runtime = P.runtime()
    if row['runtime'] != runtime: raise ValueError('Unexpected benchmark runtime')
    for name in ('gpus','baseline_gpus'):
        gpus = row[name]
        if (len(gpus) != 1 or {g['rank'] for g in gpus} != {0}
                or len({(g['name'],g['total_memory']) for g in gpus}) != 1
                or any('GH200' not in g['name'] or not 0 < g['peak_allocated'] < .9*g['total_memory']
                       or g['peak_reserved'] < g['peak_allocated'] for g in gpus)):
            raise ValueError('Need one GH200 and 10% allocation headroom')
    by_rank=lambda items:{g['rank']:(g['name'],g['total_memory'],g['first_input_sha256']) for g in items}
    if by_rank(row['gpus'])!=by_rank(row['baseline_gpus']):
        raise ValueError('Baseline/candidate hardware or initial rank data differ')
    val = row['validation']
    if (val['events'] != 128000 or not positive(val['seconds'])
            or not precision_ok(val['precision'],'torch.float32')):
        raise ValueError('Incomplete FP32 streaming validation')
    for name in ('inference','baseline_inference'):
        inf = row[name]
        if (inf['batch_size'] != 256 or inf['particles'] != 128 or inf['batches'] != 20
                or inf['precision'] != 'fp32' or inf['batch1_iterations'] != 100
                or not positive(inf['seconds']) or not positive(inf['batch1_seconds'])):
            raise ValueError('Incomplete inference/latency evidence')
    if row.get('backend') != CONFIG['backend']:
        raise ValueError('Backend must be the admitted bounded sparse implementation')
    fields=('left_preactivation_rms','right_preactivation_rms','gate_preactivation_rms',
        'left_saturation','right_saturation','gate_saturation','triangle_to_state_norm','triangle_to_ev_norm',
        'zero_denominator_fraction')
    diagnostics=row['diagnostics']
    if len(diagnostics)!=1 or {d['rank'] for d in diagnostics}!={0}:
        raise ValueError('Missing rank diagnostics')
    for d in diagnostics:
        if (d['healthy'] is not True or d['events']!=8 or d['particles']!=17
                or not positive(d['disabled_logit_rms_change']) or [x['layer'] for x in d['layers']]!=[6,7]):
            raise ValueError('Missing active triangle sensitivity evidence')
        for layer in d['layers']:
            if any(type(layer.get(k)) is not float or not math.isfinite(layer[k]) or layer[k]<0 for k in fields):
                raise ValueError('Nonfinite triangle diagnostics')
            if any(layer[k]>1 for k in fields if k.endswith('saturation') or k=='zero_denominator_fraction'):
                raise ValueError('Invalid triangle diagnostic fraction')
    gradients=row['triangle_gradient_norms']
    if len(gradients)!=1 or any(len(g)!=10 or any(not positive(v) for v in g.values()) for g in gradients):
        raise ValueError('Missing active triangle gradients')
    check_measurement(row['adversarial_ties'],3)
    if row.get('full128_tied_validation_passed') is not True:
        raise ValueError('Adversarial full-length FP32 validation missing')
    histograms=row['neighbor_statistics']
    if len(histograms)!=1 or {h['rank'] for h in histograms}!={0}:
        raise ValueError('Missing sparse/tie-path population evidence')
    for h in histograms:
        for kind in ('train','adversarial'):
            validate_histogram(h[kind])
        if h['rank']==0:validate_histogram(h['validation'])
        adversarial=h['adversarial']
        if (adversarial['maximum_support']!=127 or adversarial['tie_fraction']!=1
                or not math.isclose(adversarial['mean_weight_mass'],8,rel_tol=1e-5)):
            raise ValueError('Adversarial all-tied fixture not exercised')
    profiles=row['profiles']
    if len(profiles)!=1 or {r['rank'] for r in profiles}!={0}:
        raise ValueError('Missing phase timing evidence')
    for profile in profiles:
        for kind in ('real_train','adversarial_train',*(['real_validation'] if profile['rank']==0 else [])):
            entry=profile[kind]
            if entry.get('policy')!=PROFILE_POLICY:
                raise ValueError('Missing bounded diagnostic profiling policy')
            sample=entry.get('sample',{})
            if (any(type(sample.get(k)) is not int or sample[k]<1
                    for k in ('events','particles','source_events','source_particles'))
                    or sample['events']!=min(sample['source_events'],PROFILE_POLICY['max_events'])
                    or sample['particles']!=min(sample['source_particles'],PROFILE_POLICY['max_particles'])
                    or sample.get('training') is not (kind!='real_validation')
                    or (kind=='adversarial_train' and (sample['source_events'],sample['source_particles'])!=(512,128))):
                raise ValueError('Invalid bounded profiling sample')
            memory=entry.get('host_memory',{})
            for phase in ('before','after'):
                m=memory.get(phase,{})
                if (m.get('scope')!='process_only_excludes_workers'
                        or type(m.get('pid')) is not int or m['pid']<1
                        or not positive(m.get('rss_bytes')) or not positive(m.get('peak_rss_bytes'))):
                    raise ValueError('Missing profiling process memory evidence')
            p=entry['stages']
            required=PROFILE_STAGES if kind=='adversarial_train' else PROFILE_STAGES-{'triangles.tied_tile'}
            if (not required.issubset(p) or not set(p).issubset(PROFILE_STAGES)
                    or any(not positive(v['cpu_us']) or v['calls']<1
                           or not math.isfinite(v['device_us']) or v['device_us']<0 for v in p.values())):
                raise ValueError('Missing triangle stage profiling')
    if len(row['host_peak_rss_bytes'])!=1 or any(not positive(v) for v in row['host_peak_rss_bytes']):
        raise ValueError('Missing rank host-memory measurement')
    train = max(w['seconds']/w['steps'] for r in row['stream']['ranks'] for w in r['windows'])*1e6
    validation = val['seconds'] * 50 * 1280000/val['events']
    checkpoint = (1000+150+2)*row['checkpoint_seconds']
    return dict(passed=True,complete=True,authorizes_training=False,parameters=PARAMETERS,
        train_jets_per_second=row['stream']['jets_per_second'],
        training_time_vs_state_ev=row['baseline_stream']['jets_per_second']/row['stream']['jets_per_second'],
        fixed128_inference_jets_per_second=5120/row['inference']['seconds'],
        inference_time_vs_state_ev=row['inference']['seconds']/row['baseline_inference']['seconds'],
        batch1_latency_ms=10*row['inference']['batch1_seconds'],
        peak_allocated_gib=max(g['peak_allocated'] for g in row['gpus'])/2**30,
        projected_train_hours=train/3600,projected_validation_hours=validation/3600,
        projected_checkpoint_hours=checkpoint/3600,
        recommended_train_hours=math.ceil(1.3*(train+validation+checkpoint)/3600),
        recommended_evaluate_hours=max(1,math.ceil(1.3*val['seconds']*5e6/val['events']/3600)),
        note='Full-schedule projection, +30% margin; excludes queue/replay. No accuracy or production approval.')



def summarize_support(spec, row):
    science = importlib.import_module(P.PACKAGES['support']+'.protocol')
    PARAMETERS, BASE_PARAMETERS, PARITY = science.PARAMETERS, science.BASE_PARAMETERS, science.PARITY
    CHECKS = {RENAMES.get(k,k) for k in science.CHECKS}
    RESTART = dict(science.RESTART, ranks=1)

    if row.get('campaign_sha256') != spec['content_hash']: raise ValueError('Wrong benchmark campaign')
    if row.get('authorizes_training') is not False: raise ValueError('Benchmark never authorizes training')
    if row.get('passed') is not True or row.get('complete') is not True:
        return dict(passed=False, complete=False, authorizes_training=False, error=row.get('error'))
    if row.get('parameters') != PARAMETERS or row.get('baseline_parameters') != BASE_PARAMETERS:
        raise ValueError('Measured model count differs')
    for part, steps in (('stream',4200), ('fixed',40), ('baseline_stream',4200), ('baseline_fixed',40)):
        check_measurement(row[part], steps)
    checks = row['correctness']
    if (len(checks) != 1 or {r['rank'] for r in checks} != {0}
            or any(r['passed'] is not True or r['tolerances'] != PARITY
                   or {c['kind'] for c in r['checks']} != CHECKS
                   or any(c['passed'] is not True for c in r['checks']) for r in checks)):
        raise ValueError('Incomplete correctness evidence')
    for rank in checks:
        delayed = rank['delayed_activation']
        if (delayed.get('passed') is not True or delayed['successful_updates'] != 8
                or not 8 <= delayed['attempts'] <= 20 or delayed['skips'] != delayed['attempts']-8
                or not all(delayed[k] is True for k in ('initial_upstream_zero','initial_output_signal',
                                                       'upstream_updated','all_support_gradients'))):
            raise ValueError('Support learning path not established')
    if (row['restart'] != RESTART or row.get('full128_validation_passed') is not True
            or row['finite_state'] is not True or row['baseline_finite_state'] is not True
            or not positive(row['checkpoint_seconds'])):
        raise ValueError('Restart/finite-state evidence failed')
    runtime = P.runtime()
    if row['runtime'] != runtime: raise ValueError('Unexpected benchmark runtime')
    for name in ('gpus','baseline_gpus'):
        gpus = row[name]
        if (len(gpus) != 1 or {g['rank'] for g in gpus} != {0}
                or len({(g['name'],g['total_memory']) for g in gpus}) != 1
                or any('GH200' not in g['name'] or not 0 < g['peak_allocated'] < .9*g['total_memory']
                       or g['peak_reserved'] < g['peak_allocated'] for g in gpus)):
            raise ValueError('Need one GH200 and 10% allocation headroom')
    by_rank=lambda items:{g['rank']:(g['name'],g['total_memory'],g['first_input_sha256']) for g in items}
    if by_rank(row['gpus'])!=by_rank(row['baseline_gpus']):
        raise ValueError('Baseline/candidate hardware or initial rank data differ')
    val = row['validation']
    if (val['events'] != 128000 or not positive(val['seconds'])
            or not precision_ok(val['precision'],'torch.float32')):
        raise ValueError('Incomplete FP32 streaming validation')
    for name in ('inference','baseline_inference'):
        inf = row[name]
        if (inf['batch_size'] != 256 or inf['particles'] != 128 or inf['batches'] != 20
                or inf['precision'] != 'fp32' or inf['batch1_iterations'] != 100
                or not positive(inf['seconds']) or not positive(inf['batch1_seconds'])):
            raise ValueError('Incomplete inference/latency evidence')
    from part_ev_support.diagnostics import validate_diagnostic
    diagnostics=row['diagnostics']
    if (len(diagnostics)!=1 or {d['rank'] for d in diagnostics}!={0}
            or len({d['sample_sha256'] for d in diagnostics})!=1):
        raise ValueError('Missing rank diagnostics')
    for diagnostic in diagnostics:validate_diagnostic(diagnostic)
    gradients=row['support_gradient_norms']
    if len(gradients)!=1 or any(len(g)!=48 or any(not positive(v) for v in g.values()) for g in gradients):
        raise ValueError('Missing finite active support gradients')
    if len(row['host_peak_rss_bytes'])!=1 or any(not positive(v) for v in row['host_peak_rss_bytes']):
        raise ValueError('Missing per-rank host RSS')
    if row['parameter_inventory_sha256']!=spec['parameter_inventory_sha256']:
        raise ValueError('Measured parameter inventory changed')
    train = max(w['seconds']/w['steps'] for r in row['stream']['ranks'] for w in r['windows'])*1e6
    validation = val['seconds'] * 50 * 1280000/val['events']
    checkpoint = (1000+150+2)*row['checkpoint_seconds']
    return dict(passed=True,complete=True,authorizes_training=False,parameters=PARAMETERS,
        train_jets_per_second=row['stream']['jets_per_second'],
        training_time_vs_state_ev=row['baseline_stream']['jets_per_second']/row['stream']['jets_per_second'],
        fixed128_inference_jets_per_second=5120/row['inference']['seconds'],
        inference_time_vs_state_ev=row['inference']['seconds']/row['baseline_inference']['seconds'],
        batch1_latency_ms=10*row['inference']['batch1_seconds'],
        peak_allocated_gib=max(g['peak_allocated'] for g in row['gpus'])/2**30,
        projected_train_hours=train/3600,projected_validation_hours=validation/3600,
        projected_checkpoint_hours=checkpoint/3600,
        recommended_train_hours=math.ceil(1.3*(train+validation+checkpoint)/3600),
        recommended_evaluate_hours=max(1,math.ceil(1.3*val['seconds']*5e6/val['events']/3600)),
        note='Full-schedule projection, +30% margin; excludes queue/replay. No accuracy or production approval.')



def summarize(spec, row):
    if row.get('passed') is not True or row.get('complete') is not True:
        return dict(passed=False, complete=False, authorizes_training=False, error=row.get('error'))
    rows = row['correctness']
    if len(rows) != 1 or rows[0].get('execution') != EXECUTION:
        raise ValueError('Correctness must actually run on one native CUDA rank')
    science = importlib.import_module(P.PACKAGES[spec['model_kind']]+'.protocol')
    if len(rows[0]['checks']) != len(science.CHECKS):
        raise ValueError('Duplicate/missing correctness checks')
    for group in ('gpus','baseline_gpus'):
        for gpu in row[group]:
            if (gpu['total_memory'] < P.POLICY['minimum_vram_bytes']
                    or not gpu['peak_allocated'] <= gpu['peak_reserved'] <= gpu['total_memory']
                    or not math.isfinite(gpu['cold_fetch_seconds']) or gpu['cold_fetch_seconds'] < 0):
                raise ValueError('Invalid native GPU memory/cold-fetch evidence')
    if len(row['host_peak_rss_bytes']) != 1 or not positive(row['host_peak_rss_bytes'][0]):
        raise ValueError('Missing process host-memory evidence')
    result = {'readout':summarize_readout, 'triangles':summarize_triangles,
              'support':summarize_support}[spec['model_kind']](spec, row)
    # Include cold-start cost; the per-model projections already include all 50
    # validations and 1,152 checkpoint writes. Queue/replay remain excluded.
    cold = row['gpus'][0]['cold_fetch_seconds']
    total = sum(result[k] for k in ('projected_train_hours','projected_validation_hours',
                                    'projected_checkpoint_hours')) + cold/3600
    result.update(cold_fetch_seconds=cold, recommended_train_hours=math.ceil(1.3*total),
        gpu=row['gpus'][0]['name'], runtime=row['runtime'],
        note='One GH200/native BatchNorm; +30% margin, excludes queue/replay/contention. '
             'No accuracy claim. Automatic dispatch separately verifies opt-in, storage and walltime limits.')
    return result

