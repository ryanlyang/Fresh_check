"""Immutable benchmark-only identity and evidence admission for pair readout."""
import copy
import math
from pathlib import Path
import shutil
from part_ev_fullscale.protocol import digest, file_sha256, read_json, write_json
from part_ev_state_fullscale.protocol import recipes
from part_ev_message_study.controls import checked_hash, verify_data_contents
from part_ev_message_study.protocol import verify_inventory_metadata, precision_ok
from part_ev_state_bias.protocol import reference, runtime_environment, verify_snapshot, positive
from .model import CONFIG, PARAMETERS, BASE_PARAMETERS

SCHEMA = 'part_ev_pair_readout_benchmark_v1'
TASK = dict(run_id='PAIR_READOUT_STATE_EV_LAYERWISE', architecture='state_pair_readout_layerwise',
            recipe='ranger_full_v1', seed=101)
MINUTES = dict(prepare=120, benchmark=330, summary=15)
POLICY = dict(world_size=2, global_batch=512, local_batch=256, workers_per_rank=1,
    validation_workers=2, validation_batch=512, validation_events=128000,
    warmup=20, measured=4200, window=1000, fixed_warmup=10, fixed_steps=40,
    restart_steps=7, delayed_successful_updates=3, delayed_max_attempts=20,
    max_seconds=18000, memory_fraction=.9, max_skip_fraction=.05, planning_margin=1.3,
    checkpoint_interval=1000, training_precision='fp16', validation_precision='fp32',
    batchnorm='SyncBatchNorm', authorizes_training=False)
PARITY = dict(forward_rtol=1e-5, forward_atol=1e-6, gradient_rtol=1e-4, gradient_atol=1e-5,
              ddp_forward_rtol=5e-4, ddp_gradient_rtol=2e-3, ddp_atol=2e-5,
              amp_forward_rtol=2e-3, amp_forward_atol=2e-3,
              amp_gradient_rtol=1e-2, amp_gradient_atol=1e-3)
CHECKS = {'zero_identity', 'active_chunk', 'active_SyncBN_DDP', 'no_pairs_DDP',
          'unequal_lengths_DDP', 'active_amp_chunk',
          'pooling_oracle', 'boundaries_masks_lifetime', 'unchanged_particle_path'}
RESTART = dict(passed=True,updates=7,ranks=2,full_state=True,validation_boundaries=1,
    repeatability=True,fixture_cursor_replay=True,real_stream_resume_verified=False)
BUDGET_BYTES = 2*2**30


def check_storage(repo, output):
    # Reuse read-only accounting for registered SPORC production and GH200
    # diagnostics; replace its prospective 1-GiB slot with our 2-GiB slot.
    from part_ev_extensions_tigris.protocol import required_storage, used_bytes, BENCHMARK_BYTES
    output, repo = Path(output), Path(repo)
    used = used_bytes(output)
    needed = required_storage(repo, output) - max(0, BENCHMARK_BYTES-used) + max(0, BUDGET_BYTES-used)
    for path in (repo/'checkpoints/part_ev_pair_readout').glob('*/launch.json'):
        if path.parent == output: continue
        other = checked_hash(read_json(path))
        report = path.parent/'summary.json'
        if not report.exists() or read_json(report).get('passed') is not True:
            needed += max(0, other['output_budget_bytes']-used_bytes(path.parent))
    existing = output
    while not existing.exists(): existing = existing.parent
    if shutil.disk_usage(existing).free < needed:
        raise OSError(f'Need {needed/2**30:.2f} GiB aggregate headroom; nothing deleted')
    return needed


def launch_config(output):
    output = Path(output).resolve(strict=True)
    launch = checked_hash(read_json(output/'launch.json'))
    if (launch.get('schema') != SCHEMA or launch.get('output') != str(output)
            or launch.get('authorizes_training') is not False
            or launch.get('output_budget_bytes') != BUDGET_BYTES
            or launch.get('minutes') != MINUTES):
        raise ValueError('Frozen launch identity changed')
    source_hash = verify_snapshot(output)
    if launch['source_sha256'] != source_hash: raise ValueError('Launch/source mismatch')
    donor = Path(launch['reference']).resolve(strict=True)
    if output == donor or output.is_relative_to(donor) or donor.is_relative_to(output):
        raise ValueError('Use an independent output, disjoint from reference')
    return launch


def prepare(output):
    output = Path(output).resolve()
    launch = launch_config(output)
    if (output/'campaign.json').exists(): raise FileExistsError('Already prepared; use a fresh campaign')
    donor = reference(launch['reference'])
    verify_inventory_metadata(donor)
    check_storage(launch['repo'], output)
    contents = verify_data_contents(donor['inventory'])
    spec = dict(schema=SCHEMA, task=TASK, recipe=copy.deepcopy(recipes()['ranger_full_v1']),
        policy=POLICY, parity=PARITY, model=CONFIG, parameters=PARAMETERS,
        launch=launch, source_sha256=launch['source_sha256'], environment=runtime_environment(),
        reference_sha256=file_sha256(Path(launch['reference'])/'campaign.json'),
        inventory=donor['inventory'], data_contents_sha256=digest(contents),
        authorizes_training=False, historical_control_reuse_authorized=False)
    spec['content_hash'] = digest(spec)
    write_json(output/'data_contents.json', contents)
    write_json(output/'campaign.json', spec)
    return spec


def load(output, *, check_environment=True):
    output = Path(output)
    launch = launch_config(output)
    spec = checked_hash(read_json(output/'campaign.json'))
    expected = dict(schema=SCHEMA, task=TASK, recipe=recipes()['ranger_full_v1'],
        policy=POLICY, parity=PARITY, model=CONFIG, parameters=PARAMETERS,
        launch=launch, source_sha256=launch['source_sha256'], authorizes_training=False,
        historical_control_reuse_authorized=False)
    if any(spec.get(k) != v for k, v in expected.items()): raise ValueError('Frozen science changed')
    donor = reference(launch['reference'], check_environment=check_environment)
    if (file_sha256(Path(launch['reference'])/'campaign.json') != spec['reference_sha256']
            or donor['inventory'] != spec['inventory'] or donor['environment'] != spec['environment']
            or digest(read_json(output/'data_contents.json')) != spec['data_contents_sha256']):
        raise ValueError('Reference/data/environment identity changed')
    return spec


def check_measurement(row, steps):
    if (row['measured_steps'] != steps or row['healthy'] is not True
            or not positive(row['seconds']) or not 0 <= row['skips'] < steps*.05
            or not math.isclose(row['jets_per_second'],512*steps/row['seconds'],rel_tol=1e-10)
            or len(row['ranks']) != 2): raise ValueError('Invalid sustained timing/health evidence')
    identities = set()
    windows = [min(1000, steps-i) for i in range(0,steps,1000)]
    for rank in row['ranks']:
        if (not precision_ok(rank['precision'], 'torch.float16')
                or [w['steps'] for w in rank['windows']] != windows
                or rank['skips'] != row['skips']
                or any(not positive(w['seconds']) for w in rank['windows'])):
            raise ValueError('Missing windows/precision')
        identities.add((rank['parameters_sha256'],rank['scaler'],rank['skips'],rank['lookahead_counter']))
    if len(identities) != 1: raise ValueError('Distributed replicas differ')


def summarize(spec, row):
    if row.get('campaign_sha256') != spec['content_hash']: raise ValueError('Wrong benchmark campaign')
    if row.get('authorizes_training') is not False: raise ValueError('Benchmark never authorizes training')
    if row.get('passed') is not True or row.get('complete') is not True:
        return dict(passed=False, complete=False, authorizes_training=False, error=row.get('error'))
    if row.get('parameters') != PARAMETERS or row.get('baseline_parameters') != BASE_PARAMETERS:
        raise ValueError('Measured model count differs')
    for part, steps in (('stream',4200), ('fixed',40), ('baseline_stream',4200), ('baseline_fixed',40)):
        check_measurement(row[part], steps)
    checks = row['correctness']
    if (len(checks) != 2 or {r['rank'] for r in checks} != {0,1}
            or any(r['passed'] is not True or r['tolerances'] != PARITY
                   or {c['kind'] for c in r['checks']} != CHECKS
                   or any(c['passed'] is not True for c in r['checks']) for r in checks)):
        raise ValueError('Incomplete correctness evidence')
    for rank in checks:
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
    runtime = dict(world_size=2,local_batch=256,global_batch=512,workers_per_rank=1,
        batchnorm='SyncBatchNorm',tf32=False,cudnn_benchmark=False,
        nccl_environment=spec['launch']['nccl_environment'])
    if row['runtime'] != runtime: raise ValueError('Unexpected benchmark runtime')
    for name in ('gpus','baseline_gpus'):
        gpus = row[name]
        if (len(gpus) != 2 or {g['rank'] for g in gpus} != {0,1}
                or len({(g['name'],g['total_memory']) for g in gpus}) != 1
                or any('A100' not in g['name'] or not 0 < g['peak_allocated'] < .9*g['total_memory']
                       or g['peak_reserved'] < g['peak_allocated'] for g in gpus)):
            raise ValueError('Need two matching A100s and 10% allocation headroom')
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
    if (len(row['diagnostics']) != 2
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
    if len(gradients)!=2 or any(len(g)!=7 or any(not positive(v) for v in g.values()) for g in gradients):
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


def review(output):
    output = Path(output)
    missing = [f for f in ('campaign.json','benchmark.json') if not (output/f).is_file()]
    if missing: return dict(passed=False,complete=False,authorizes_training=False,missing=missing)
    return summarize(load(output,check_environment=False), checked_hash(read_json(output/'benchmark.json')))
