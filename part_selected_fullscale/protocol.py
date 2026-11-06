"""Frozen seven-run science and fail-closed, benchmark-only resource review."""
import copy
import math
from pathlib import Path
from part_ev_fullscale.protocol import file_sha256, file_lock, read_json, write_json
from part_screen_common.contracts import checked_json, hashed
from part_selected_fast import protocol as F
from part_ev_state_fullscale.protocol import recipes as full_recipes
from .model import CONFIGS, COUNTS

SCHEMA = 'selected_fullscale_benchmarks_v1'
ROOT = Path(__file__).resolve().parents[1]
IDS = ('PART_MATCH_STATE_SHARED', 'PART_MATCH_STATE_LAYERWISE', 'SELECTED_PART_MATCH_STATE_LAYERWISE',
       'STATE_EV_SHARED', 'EV_LAYERWISE', 'SELECTED_PART_SHARED', 'SELECTED_STATE_EV_LAYERWISE')
TASKS = [dict(run_id=name, architecture=arch, seed=101, recipe='ranger_full_v1')
         for name, arch in zip(IDS, CONFIGS)]
SELECTED = (2, 5, 6)
BENCHMARK = dict(warmup_steps=20, selected_steps=5000, standard_steps=1000, window_steps=1000,
    validation_events=131072, calibration_attempts=8, confirmation_attempts=10,
    max_skip_fraction=.01, memory_fraction_limit=.90, planning_margin=1.30,
    phase_seconds=5*3600, shutdown_margin_seconds=600, paired_trials=3, paired_steps=200)
EXECUTION = dict(tree_threads=4, cache_bytes=0, compile_region=False, state_blocks=list(range(8)),
                 state_width=32, state_query_chunk=64, checkpoint_chunks=True,
                 batch_size=512, loader_workers=2, cpus=12, memory_gib=128)


def recipes():
    result=full_recipes()
    # Architecture belongs to the explicit per-task CONFIGS, not the baseline's
    # shared-bias/no-selected-relation defaults carried by the original recipe.
    for key in ('pair_bias','selected_relations'):
        result['ranger_full_v1'].pop(key)
    return result


def task(index):
    if type(index) is not int or not 0 <= index < 7:
        raise ValueError('Task index must be 0..6')
    return copy.deepcopy(TASKS[index])


def source_manifest(root=ROOT):
    root = Path(root)
    result = F.source_manifest(root)
    for package in ('part_selected_fullscale', 'part_ev_message_study', 'part_ev_state_speed', 'part_ev_state_fullscale'):
        result.update({p.relative_to(root).as_posix(): file_sha256(p) for p in (root/package).glob('*.py')})
    for name in ('scripts/run_part_selected_fullscale.py', 'scripts/submit_part_selected_fullscale.py',
                 'sbatch/run_part_selected_fullscale_tigris.sh', 'sbatch/submit_part_selected_fullscale_tigris.sh'):
        result[name] = file_sha256(root/name)
    return dict(sorted(result.items()))


def storage_projection(checkpoint_bytes=None, weight_bytes=None):
    # Conservative FP32 model + RAdam moments + Lookahead; actual files replace
    # the estimates after benchmarking. Never cache the training population.
    weights = weight_bytes or {a: n*4+1024**2 for a, n in COUNTS.items()}
    restarts = checkpoint_bytes or {a: 4*n*4+4*1024**2 for a, n in COUNTS.items()}
    core = 2*sum(restarts.values()) + 51*sum(weights.values()) + 7*5000000*48
    allowance = core + 3*2**30  # frozen source, build, bounded diagnostics/logs
    return dict(core_bytes=core, campaign_allowance_bytes=allowance, reserve_bytes=4*2**30,
                required_free_bytes=allowance+4*2**30, measured=checkpoint_bytes is not None,
                note='Full 5M outputs, not future 20M test; free space is not reserved.')


def create_spec(output):
    output = Path(output)
    launch = read_json(output/'launch.json')
    old = F.reference(launch['reference'])
    return hashed(dict(schema=SCHEMA, tasks=TASKS, parameters=COUNTS, model=CONFIGS, recipes=recipes(),
        benchmark=BENCHMARK, execution=EXECUTION, precision=old['precision'], environment=old['environment'],
        inventory=old['inventory'], input_reference=old['input_reference'], launch=launch,
        reference=dict(path=launch['reference'], spec_sha256=file_sha256(Path(launch['reference'])/'experiment_spec.json')),
        source=source_manifest(), inputs={name: file_sha256(output/name) for name in
            ('inputs/selected.yaml','inputs/normalization.json','inputs/native_stream.json',
             'backend/manifest.json','backend/parity.json','controls/registry.json')},
        storage=storage_projection(), authorizes_training=False))


def load_campaign(output, *, check_environment=True):
    output = Path(output)
    spec = checked_json(output/'experiment_spec.json', SCHEMA)
    for key, expected in dict(tasks=TASKS, parameters=COUNTS, model=CONFIGS, recipes=recipes(),
                             benchmark=BENCHMARK, execution=EXECUTION, source=source_manifest(),
                             authorizes_training=False).items():
        if spec.get(key) != expected:
            raise ValueError('Frozen full-scale contract changed: '+key)
    if spec['launch'] != read_json(output/'launch.json'):
        raise ValueError('Launch settings changed')
    old = F.reference(spec['reference']['path'], environment=check_environment)
    if file_sha256(Path(spec['reference']['path'])/'experiment_spec.json') != spec['reference']['spec_sha256']:
        raise ValueError('Selected reference changed')
    for key in ('inventory','input_reference','environment','precision'):
        if spec[key] != old[key]:
            raise ValueError('Selected reference contract changed: '+key)
    for name, sha in spec['inputs'].items():
        if file_sha256(output/name) != sha:
            raise ValueError('Prepared artifact changed: '+name)
    for name in ('inputs/selected.yaml','inputs/normalization.json','inputs/native_stream.json'):
        if spec['inputs'][name] != old['inputs'][name]:
            raise ValueError('Selected normalization/input mismatch')
    backend = read_json(output/'backend/parity.json')
    if backend.get('passed') is not True or backend.get('exact') is not True:
        raise ValueError('Native/fast tree parity missing')
    return spec


def project(row):
    recipe = recipes()['ranger_full_v1']
    seconds_per_step = max(w['seconds']/w['steps'] for w in row['windows'])
    validation = row['validation']
    for value in (row['startup_seconds'],row['checkpoint_seconds'],row['weight_save_seconds'],validation['seconds']):
        if not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
            raise ValueError('Invalid timing measurement')
    if validation['seconds']<=0 or row['checkpoint_seconds']<=0 or row['weight_save_seconds']<=0:
        raise ValueError('Missing validation/checkpoint timing')
    val_seconds = validation['seconds']*recipe['validation_events_per_epoch']/validation['events']
    saves = recipe['total_steps']//1000 + 3*recipe['epochs'] + 2
    seconds = (row['startup_seconds'] + recipe['total_steps']*seconds_per_step + recipe['epochs']*val_seconds
               + saves*row['checkpoint_seconds'] + recipe['epochs']*row['weight_save_seconds'])
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('Invalid runtime projection')
    return dict(projected_training_hours=seconds/3600, recommended_train_hours=math.ceil(1.3*seconds/3600),
        recommended_evaluate_hours=math.ceil(1.3*validation['seconds']*5000000/validation['events']/3600),
        checkpoint_writes=saves, validation_checks=recipe['epochs'],
        note='Projection includes validation AUC, I/O and startup; excludes queue/replay. Margin is not a guarantee.')


def validate_benchmark(row, spec, index):
    t = task(index)
    F.R.validate_runtime(row.get('runtime', {}))
    steps = BENCHMARK['selected_steps' if index in SELECTED else 'standard_steps']
    if (row.get('task') != t or row.get('campaign_sha256') != spec['content_hash']
            or row['runtime'].get('campaign_sha256') != spec['content_hash']
            or row['runtime'].get('execution') != EXECUTION
            or row.get('passed') is not True or row.get('complete') is not True
            or row.get('measured_steps') != steps or row.get('authorizes_training') is not False
            or row.get('execution') != EXECUTION or row.get('parity', {}).get('passed') is not True):
        raise ValueError('Incomplete or mismatched benchmark')
    windows = row.get('windows', [])
    if (len(windows) != steps//1000 or any(w.get('steps') != 1000 or not math.isfinite(w['seconds'])
            or w['seconds'] <= 0 for w in windows)):
        raise ValueError('Incomplete sustained windows')
    expected = {k:v for k,v in spec['input_reference']['fingerprints'].items() if int(k)<steps+20}
    if row.get('input_fingerprints') != expected:
        raise ValueError('Input stream differs from native Tigris')
    if (type(row.get('amp_skipped_steps')) is not int or not 0<=row['amp_skipped_steps']/steps<.01
            or row.get('full_length_amp_skips') != 0
            or row.get('full_length_confirmed') != 10 or row.get('training_precision',{}).get('passed') is not True
            or row.get('validation',{}).get('precision',{}).get('passed') is not True
            or row['validation'].get('events') != BENCHMARK['validation_events']):
        raise ValueError('Precision/full-length/validation requirements failed')
    for key, limit in (('gpu_peak_allocated_bytes', row['runtime']['gpu_memory_bytes']),
                       ('gpu_peak_reserved_bytes', row['runtime']['gpu_memory_bytes']),
                       ('host_peak_bytes', EXECUTION['memory_gib']*2**30)):
        value = row.get(key, float('inf'))
        if not math.isfinite(value) or not 0 < value < .9*limit:
            raise ValueError('Insufficient memory headroom: '+key)
    return project(row)


def validate_profile(row,spec,index):
    if index not in SELECTED: raise ValueError('Not a selected profile task')
    if (row.get('task')!=task(index) or row.get('campaign_sha256')!=spec['content_hash']
            or row.get('execution')!=EXECUTION or row.get('complete') is not True
            or row.get('passed') is not True or row.get('authorizes_training') is not False
            or row.get('parity',{}).get('passed') is not True):
        raise ValueError('Incomplete or mismatched selected profile')
    F.R.validate_runtime(row['runtime'])
    if row['runtime'].get('execution')!=EXECUTION or row['runtime'].get('campaign_sha256')!=spec['content_hash']:
        raise ValueError('Profile runtime differs')
    expected={k:spec['input_reference']['fingerprints'][k] for k in ('0','20')}
    for variant in ('reference','candidate'):
        measurements=row.get('measurements',{}).get(variant,[])
        if len(measurements)!=3: raise ValueError('Need all three paired trials')
        for trial,m in enumerate(measurements):
            if (m.get('trial')!=trial or m.get('steps')!=200 or m.get('input_fingerprints')!=expected
                    or type(m.get('amp_skips')) is not int or not 0<=m['amp_skips']/200<.01):
                raise ValueError('Paired trial input/count/AMP mismatch')
    from scripts.profile_part_selected_fast import summarize
    summary=summarize(row['measurements']['reference'],row['measurements']['candidate'])
    if any(row.get(k)!=v for k,v in summary.items()):
        raise ValueError('Paired speedup not supported by measurements')
    return summary
