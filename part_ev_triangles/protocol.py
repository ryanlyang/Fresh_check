"""Immutable benchmark-only identity and evidence admission for triangle updates."""
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

SCHEMA = 'part_ev_triangles_benchmark_v1'
TASK = dict(run_id='TRIANGLE_STATE_EV_LAYERWISE', architecture='state_triangles_layerwise',
            recipe='ranger_full_v1', seed=101)
MINUTES = dict(prepare=120, benchmark=330, summary=15)
POLICY = dict(world_size=2, global_batch=512, local_batch=256, workers_per_rank=1,
    validation_workers=2, validation_batch=512, validation_events=128000,
    warmup=20, measured=4200, window=1000, fixed_warmup=10, fixed_steps=40,
    restart_steps=7, delayed_successful_updates=8, delayed_max_attempts=20,
    max_seconds=18000, memory_fraction=.9, max_skip_fraction=.05, planning_margin=1.3,
    checkpoint_interval=1000, training_precision='fp16', validation_precision='fp32',
    batchnorm='SyncBatchNorm', authorizes_training=False)
PARITY = dict(forward_rtol=1e-5, forward_atol=1e-6, gradient_rtol=1e-4, gradient_atol=1e-5,
              ddp_forward_rtol=5e-4, ddp_gradient_rtol=2e-3, ddp_atol=2e-5,
              amp_forward_rtol=2e-3, amp_forward_atol=2e-3,
              amp_gradient_rtol=1e-2, amp_gradient_atol=1e-3)
CHECKS = {'zero_identity', 'active_chunk', 'active_SyncBN_DDP', 'no_pairs_DDP',
          'unequal_lengths_DDP', 'active_amp_chunk',
          'triangle_oracle', 'boundaries_masks_lifetime', 'native_path_and_geometry', 'tie_only_rank_DDP'}
RESTART = dict(passed=True,updates=7,ranks=2,full_state=True,validation_boundaries=1,
    repeatability=True,fixture_cursor_replay=True,real_stream_resume_verified=False)
BUDGET_BYTES = 2*2**30
PROFILE_STAGES={'triangles.geometry_and_packing','triangles.state_and_projections',
    'triangles.ordinary_mix','triangles.tie_mix','triangles.output_projection',
    'triangles.factor_projections','triangles.ordinary_tile','triangles.tied_tile'}


def check_storage(repo, output):
    from part_ev_pair_readout.protocol import check_storage as existing_budget
    from part_ev_extensions_tigris.protocol import used_bytes
    output, repo = Path(output), Path(repo)
    needed = existing_budget(repo, output)
    for path in (repo/'checkpoints/part_ev_triangles').glob('*/launch.json'):
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
        if (delayed.get('passed') is not True or delayed['successful_updates'] != 8
                or not 8 <= delayed['attempts'] <= 20 or delayed['skips'] != delayed['attempts']-8
                or not all(delayed[k] is True for k in ('initial_upstream_zero','initial_output_signal',
                                                       'upstream_updated','all_triangle_gradients'))):
            raise ValueError('Triangle learning path not established')
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
    if row.get('backend') != CONFIG['backend']:
        raise ValueError('Backend must be the admitted bounded sparse implementation')
    fields=('left_preactivation_rms','right_preactivation_rms','gate_preactivation_rms',
        'left_saturation','right_saturation','gate_saturation','triangle_to_state_norm','triangle_to_ev_norm',
        'zero_denominator_fraction')
    diagnostics=row['diagnostics']
    if len(diagnostics)!=2 or {d['rank'] for d in diagnostics}!={0,1}:
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
    if len(gradients)!=2 or any(len(g)!=10 or any(not positive(v) for v in g.values()) for g in gradients):
        raise ValueError('Missing active triangle gradients')
    check_measurement(row['adversarial_ties'],3)
    if row.get('full128_tied_validation_passed') is not True:
        raise ValueError('Adversarial full-length FP32 validation missing')
    histograms=row['neighbor_statistics']
    if len(histograms)!=2 or {h['rank'] for h in histograms}!={0,1}:
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
    if len(profiles)!=2 or {r['rank'] for r in profiles}!={0,1}:
        raise ValueError('Missing phase timing evidence')
    for profile in profiles:
        for kind in ('real_train','adversarial_train',*(['real_validation'] if profile['rank']==0 else [])):
            p=profile[kind]
            required=PROFILE_STAGES if kind=='adversarial_train' else PROFILE_STAGES-{'triangles.tied_tile'}
            if (not required.issubset(p) or not set(p).issubset(PROFILE_STAGES)
                    or any(not positive(v['cpu_us']) or v['calls']<1
                           or not math.isfinite(v['device_us']) or v['device_us']<0 for v in p.values())):
                raise ValueError('Missing triangle stage profiling')
    if len(row['host_peak_rss_bytes'])!=2 or any(not positive(v) for v in row['host_peak_rss_bytes']):
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


def validate_histogram(row):
    histogram=row['histogram']
    if (len(histogram)!=128 or any(type(n) is not int or n<0 for n in histogram)
            or sum(histogram)<=0 or row['receivers']!=sum(histogram)):
        raise ValueError('Invalid support histogram')
    maximum=max(i for i,n in enumerate(histogram) if n)
    if (row['maximum_support']!=maximum or not 0<=row['mean_weight_mass']<=8.0001
            or not math.isclose(row['tie_fraction'],sum(histogram[9:])/sum(histogram),abs_tol=1e-12)):
        raise ValueError('Invalid sparse/tie statistics')


def review(output):
    output = Path(output)
    missing = [f for f in ('campaign.json','benchmark.json') if not (output/f).is_file()]
    if missing: return dict(passed=False,complete=False,authorizes_training=False,missing=missing)
    return summarize(load(output,check_environment=False), checked_hash(read_json(output/'benchmark.json')))
