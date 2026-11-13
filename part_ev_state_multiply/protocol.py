"""Independent, immutable benchmark contract; never approves production."""
import copy
import importlib.metadata
import math
from pathlib import Path
import shutil
import sysconfig

from part_ev_fullscale.protocol import digest, file_sha256, read_json, write_json, environment
from part_ev_fullscale.data import DATA_CONFIG
from part_ev_state_fullscale.protocol import recipes
from part_ev_message_study.controls import checked_hash, verify_data_contents
from part_ev_message_study.protocol import verify_inventory_metadata, precision_ok
from .model import CONFIG, PARAMETERS, DIAGNOSTIC_FIELDS

SCHEMA = 'part_ev_state_multiply_benchmark_v1'
TASK = dict(run_id='STATE_MULTIPLY_EV_LAYERWISE', architecture='state_multiply_layerwise',
            recipe='ranger_full_v1', seed=101)
POLICY = dict(world_size=2, global_batch=512, local_batch=256, workers_per_rank=1,
              validation_workers=2, validation_batch=512, validation_events=128000,
              warmup=20, measured=4200, window=1000, fixed_warmup=10, fixed_steps=40,
              restart_steps=7, delayed_successful_updates=3, delayed_max_attempts=20,
              max_seconds=18000, memory_fraction=.90,
              max_skip_fraction=.05, planning_margin=1.30, checkpoint_interval=1000,
              training_precision='fp16', validation_precision='fp32',
              batchnorm='SyncBatchNorm', authorizes_training=False)
PARITY = dict(forward_rtol=1e-5, forward_atol=1e-6,
              gradient_rtol=1e-4, gradient_atol=1e-5,
              ddp_forward_rtol=5e-4, ddp_gradient_rtol=2e-3, ddp_atol=2e-5)
REFERENCE_SCHEMA = 'part_ev_layerwise_ddp_v1'


def runtime_environment():
    row = environment()
    # Stable interpreter install roots, NOT lazy-import additions to sys.path.
    roots = sorted({sysconfig.get_path(k) for k in ('purelib', 'platlib')})
    row['installed_distributions'] = dict(sorted((d.metadata['Name'].lower(), d.version)
        for d in importlib.metadata.distributions(path=roots) if d.metadata['Name']))
    return row


def verify_snapshot(output):
    output = Path(output)
    source = output/'source'
    expected = read_json(output/'source_manifest.json')
    actual = {}
    for path in source.rglob('*'):
        if path.is_symlink():
            raise ValueError('Snapshot symlink forbidden')
        if path.is_file():
            actual[path.relative_to(source).as_posix()] = file_sha256(path)
    if not actual or actual != expected:
        raise ValueError('Frozen source inventory/content changed; use a fresh campaign')
    return digest(expected)


def reference(path, *, check_environment=True):
    root = Path(path).resolve(strict=True)
    spec = checked_hash(read_json(root/'campaign.json'))
    if (spec.get('schema') != REFERENCE_SCHEMA or spec['task']['architecture'] != 'state_layerwise'
            or spec['task']['seed'] != 101 or spec['policy']['world_size'] != 2
            or spec['recipes'] != recipes() or spec['parameters'] != {'state_layerwise':2327462}):
        raise ValueError('Expected original seed-101 two-A100 full layerwise State-EV reference')
    runtime = dict(local_batch=256,global_batch=512,train_workers_per_rank=1,
                   validation_workers=2,validation_batch=512,batchnorm='SyncBatchNorm')
    if any(spec['policy'].get(k)!=v for k,v in runtime.items()):
        raise ValueError('Reference data-stream/runtime differs')
    checked_hash(spec['inventory'], 'inventory_sha256')
    if set(spec['inventory']['splits']) != {'train','val'}:
        raise ValueError('Only official train and validation inventories are allowed')
    for split, count in (('train',1000),('val',50)):
        rows = spec['inventory']['splits'][split]['files']
        if len(rows) != count or any(r['num_entries'] != 100000 for r in rows):
            raise ValueError('Expected the complete 100M/5M inventory')
    # Inspect donor snapshot, not equality with today's evolving repository.
    for name, checksum in spec['source'].items():
        file = (root/'source'/name).resolve(strict=True)
        if not file.is_relative_to(root/'source') or file_sha256(file) != checksum:
            raise ValueError('Historical source evidence changed')
    config_name = 'particle_transformer/data/JetClass/JetClass_full.yaml'
    if spec['source'].get(config_name) != file_sha256(DATA_CONFIG):
        raise ValueError('Preprocessing configuration differs from reference')
    if check_environment and spec['environment'] != runtime_environment():
        current = runtime_environment()
        changed = [k for k in set(current)|set(spec['environment']) if current.get(k)!=spec['environment'].get(k)]
        raise ValueError(f'Use unchanged reference environment; differing fields: {sorted(changed)}')
    return spec


def prepare(output):
    output = Path(output).resolve()
    launch = read_json(output/'launch.json')
    donor_root = Path(launch['reference']).resolve(strict=True)
    if (output == donor_root or output.is_relative_to(donor_root) or donor_root.is_relative_to(output)):
        raise ValueError('Use an independent output directory')
    if (output/'campaign.json').exists():
        raise FileExistsError('Already prepared; never overwrite benchmark evidence')
    source_hash = verify_snapshot(output)
    donor = reference(donor_root)
    verify_inventory_metadata(donor)
    if shutil.disk_usage(output).free < 6*1024**3:
        raise OSError('Keep 6 GiB free for this benchmark and checkpoint headroom; nothing deleted')
    contents = verify_data_contents(donor['inventory'])
    spec = dict(schema=SCHEMA, task=TASK, recipe=copy.deepcopy(recipes()['ranger_full_v1']),
                policy=POLICY, parity=PARITY, model=CONFIG, parameters=PARAMETERS,
                launch=launch, source_sha256=source_hash, environment=runtime_environment(),
                reference_sha256=file_sha256(donor_root/'campaign.json'),
                inventory=donor['inventory'], data_contents_sha256=digest(contents),
                authorizes_training=False, historical_control_reuse_authorized=False)
    spec['content_hash'] = digest(spec)
    write_json(output/'data_contents.json', contents)
    write_json(output/'campaign.json', spec)
    return spec


def load(output, *, check_environment=True):
    output = Path(output)
    spec = checked_hash(read_json(output/'campaign.json'))
    fixed = dict(schema=SCHEMA,task=TASK,recipe=recipes()['ranger_full_v1'],policy=POLICY,
                 parity=PARITY,model=CONFIG,parameters=PARAMETERS,authorizes_training=False)
    if any(spec.get(k) != v for k,v in fixed.items()):
        raise ValueError('Frozen model/recipe/benchmark contract changed')
    if (spec['source_sha256'] != verify_snapshot(output)
            or spec['launch'] != read_json(output/'launch.json')
            or spec['data_contents_sha256'] != digest(read_json(output/'data_contents.json'))):
        raise ValueError('Source, launch or verified-data evidence changed')
    donor_root = Path(spec['launch']['reference'])
    donor = reference(donor_root, check_environment=check_environment)
    if (file_sha256(donor_root/'campaign.json') != spec['reference_sha256']
            or donor['inventory'] != spec['inventory'] or donor['environment'] != spec['environment']):
        raise ValueError('Reference contract changed')
    return spec


def positive(value):
    return type(value) in (int,float) and math.isfinite(value) and value > 0


def review(output):
    output = Path(output)
    missing = [p for p in ('campaign.json','benchmark.json') if not (output/p).is_file()]
    if missing:
        return dict(passed=False, complete=False, missing=missing, authorizes_training=False)
    spec = load(output, check_environment=False)
    row = checked_hash(read_json(output/'benchmark.json'))
    if row['campaign_sha256'] != spec['content_hash']:
        raise ValueError('Benchmark belongs to another campaign')
    if row.get('passed') is not True or row.get('complete') is not True:
        return dict(passed=False,complete=False,error=row.get('error'),authorizes_training=False)
    for part, steps in (('stream',POLICY['measured']),('fixed',POLICY['fixed_steps'])):
        result = row[part]
        if (result['measured_steps'] != steps or result['healthy'] is not True
                or not positive(result['seconds']) or not 0 <= result['skips'] < steps*POLICY['max_skip_fraction']
                or not math.isclose(result['jets_per_second'],512*steps/result['seconds'],rel_tol=1e-10)
                or len(result['ranks']) != 2):
            raise ValueError('Incomplete/invalid throughput or AMP evidence')
        identities = set()
        window_counts = [min(POLICY['window'],steps-start) for start in range(0,steps,POLICY['window'])]
        for rank in result['ranks']:
            if (not precision_ok(rank['precision'],'torch.float16')
                    or [w['steps'] for w in rank['windows']] != window_counts
                    or rank['skips'] != result['skips']
                    or any(not positive(w['steps']) or not positive(w['seconds']) for w in rank['windows'])):
                raise ValueError('Missing measured windows/precision')
            identities.add((rank['parameters_sha256'],rank['scaler'],rank['skips'],rank['lookahead_counter']))
        if len(identities)!=1: raise ValueError('Distributed replicas differ')
    required_checks={'zero_identity','active_chunk','active_SyncBN_DDP','unchanged_attention_and_ev'}
    if (len(row['correctness']) != 2 or {r['rank'] for r in row['correctness']}!={0,1}
            or any(r['passed'] is not True or r['tolerances']!=PARITY
                   or {c['kind'] for c in r['checks']}!=required_checks
                   or any(c['passed'] is not True for c in r['checks']) for r in row['correctness'])
            or row['restart'] != dict(passed=True,updates=7,ranks=2,full_state=True)
            or row['finite_state'] is not True):
        raise ValueError('Correctness, full-state restart and finite-state gates required')
    from part_ev_message_study.parity import validate_report
    for rank in row['correctness']:
        validate_report(rank['initialization'])
        delayed=rank['delayed_activation']
        if (delayed['passed'] is not True or delayed['first_success_gate_gradients_zero'] is not True
                or delayed['first_success_output_gradients_nonzero'] is not True
                or delayed['all_gate_matrices_active'] is not True or delayed['all_gate_gradients_nonzero'] is not True
                or delayed['successful_updates']!=POLICY['delayed_successful_updates']
                or not 3<=delayed['attempts']<=POLICY['delayed_max_attempts']
                or delayed['skips']!=delayed['attempts']-delayed['successful_updates']):
            raise ValueError('Delayed modulation learning path did not pass')
    if len(row['diagnostics'])!=2: raise ValueError('Missing per-rank modulation diagnostics')
    for sample in row['diagnostics']:
        if (sample['events']!=8 or sample['particles']!=17 or sample['precision']!='fp32'
                or sample['dropout'] is not False or sample['optimizer_steps']!=0 or len(sample['layers'])!=8):
            raise ValueError('Incomplete fixed-sample diagnostics')
        for layer in sample['layers']:
            for key in DIAGNOSTIC_FIELDS:
                value=layer[key]
                if type(value) not in (int,float) or not math.isfinite(value) or (key!='gate_mean' and value<0):
                    raise ValueError('Missing/nonfinite modulation diagnostics')
            if (abs(layer['gate_mean'])>1 or layer['gate_abs_max']>1 or layer['gate_saturation_fraction']>1):
                raise ValueError('Modulation diagnostics violate the bounded gate')
    required_runtime=dict(world_size=2,local_batch=256,global_batch=512,workers_per_rank=1,
                          batchnorm='SyncBatchNorm',tf32=False,cudnn_benchmark=False,
                          nccl_environment=spec['launch']['nccl_environment'])
    if row['runtime'] != required_runtime: raise ValueError('Frozen runtime differs')
    gpus = row['gpus']
    if (len(gpus)!=2 or {g['rank'] for g in gpus}!={0,1}
            or len({(g['name'],g['total_memory']) for g in gpus})!=1
            or any('A100' not in g['name'] or not 0<g['peak_allocated']<.9*g['total_memory'] for g in gpus)):
        raise ValueError('Two matching A100s with 10% allocation headroom required')
    val = row['validation']
    if (val['events'] != POLICY['validation_events'] or not positive(val['seconds'])
            or not precision_ok(val['precision'],'torch.float32') or not positive(row['checkpoint_seconds'])):
        raise ValueError('FP32 validation/checkpoint timing missing')
    seconds_per_step = max(w['seconds']/w['steps'] for r in row['stream']['ranks'] for w in r['windows'])
    recipe = spec['recipe']
    train = seconds_per_step*recipe['total_steps']
    validation = val['seconds']*50*recipe['validation_events_per_epoch']/val['events']
    checkpoint = (recipe['total_steps']//1000+3*50+2)*row['checkpoint_seconds']
    inference=row['inference']
    if (inference['batch_size']!=256 or inference['particles']!=128 or inference['batches']!=20
            or not positive(inference['seconds']) or inference['precision']!='fp32'):
        raise ValueError('Fixed FP32 inference timing missing')
    return dict(passed=True,complete=True,authorizes_training=False,parameters=PARAMETERS,
                train_jets_per_second=row['stream']['jets_per_second'],
                validation_end_to_end_jets_per_second=val['events']/val['seconds'],
                fixed128_inference_jets_per_second=256*20/inference['seconds'],
                peak_allocated_gib=max(g['peak_allocated'] for g in gpus)/1024**3,
                projected_train_hours=train/3600,projected_validation_hours=validation/3600,
                projected_checkpoint_hours=checkpoint/3600,
                recommended_train_hours=math.ceil(1.30*(train+validation+checkpoint)/3600),
                recommended_evaluate_hours=max(1,math.ceil(1.30*val['seconds']*5e6/val['events']/3600)),
                note='One long allocation intended; +30% margin, excludes queue/replay. No production authorization.')
