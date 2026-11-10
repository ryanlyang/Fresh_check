"""Stdlib admission boundary: new orchestration, original benchmark science."""
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile

SCHEMA = 'compact_state_production_v1'
FILES = (
    'part_ev_compact_production/__init__.py', 'part_ev_compact_production/protocol.py',
    'part_ev_compact_production/engine.py', 'part_ev_compact_production/reporting.py',
    'scripts/submit_part_ev_compact_production.py', 'scripts/run_part_ev_compact_production.py',
    'sbatch/submit_part_ev_compact_production_tigris.sh', 'sbatch/run_part_ev_compact_production_tigris.sh',
)
POLICY = dict(checkpoint_interval=1000, shutdown_margin_seconds=900, initial_amp_scale=32768,
    selection='highest_accuracy_first_tie', validation_checks=50, evaluation_population=5000000,
    restart='latest_full_state_exact_stream_replay', fresh_seed=101, persistent_pair_cache=False,
    test_data=False, compute='observed_trim_lengths_dense_fwd_bwd_recompute_excludes_validation',
    retention='latest_full_restart_and_at_most_50_improving_weight_checkpoints')
EVALUATE_HOURS = [5, 5]
REPORT_HOURS = 1


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def hashed(row):
    return dict(row, content_hash=hashlib.sha256(json.dumps(
        row, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest())


def checked(path):
    row = read(path)
    if row != hashed({k:v for k,v in row.items() if k != 'content_hash'}):
        raise ValueError('Metadata hash changed: '+str(path))
    return row


def write(path, row):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            json.dump(row, handle, indent=2, allow_nan=False)
            handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def inventory(source):
    result = {}
    for path in sorted(Path(source).rglob('*')):
        if path.is_symlink(): raise ValueError('Symlink in frozen source')
        if path.is_file() and '__pycache__' not in path.parts and path.suffix not in ('.pyc', '.pyo'):
            result[path.relative_to(source).as_posix()] = sha(path)
    return result


def activate(benchmark):
    """Authenticate before importing, and reject live-checkout scientific imports."""
    benchmark = Path(benchmark).resolve(strict=True)
    source = benchmark/'source'
    manifest = read(benchmark/'snapshot.json')
    if not manifest or inventory(source) != manifest:
        raise ValueError('Benchmark frozen source changed')
    settings = read(benchmark/'launch.json')
    if Path(sys.executable).resolve() != Path(settings['python']).resolve():
        raise ValueError('Use the original benchmark interpreter')
    sys.path.insert(0, str(source))
    from part_ev_compact import protocol as science
    for name, module in list(sys.modules.items()):
        scientific = (name.startswith(('part_ev_', 'part_selected_', 'part_screen_', 'part_layerwise_'))
                      or name.startswith('scripts.run_part_'))
        if scientific and not name.startswith(('part_ev_compact_production','scripts.run_part_ev_compact_production')):
            path = getattr(module, '__file__', None)
            if path and not Path(path).resolve().is_relative_to(source):
                raise ValueError('Scientific import outside benchmark snapshot: '+name)
    return science


def validate_review(review):
    if review.get('passed') is not True or review.get('authorizes_training') is not False:
        raise ValueError('Require passed benchmark evidence, not training artifacts')
    models = review['models']
    if [r['architecture'] for r in models] != ['compact', 'matched']:
        raise ValueError('Require both compact and matched benchmarks')
    for row in models:
        hours = row['recommended_hours']
        if (type(hours) is not int or not 0 < hours <= 480 or row['fits_tigris_20_days'] is not True
                or not math.isfinite(row['projected_hours']) or row['projected_hours'] <= 0
                or hours < math.ceil(1.3*row['projected_hours'])):
            raise ValueError('Invalid/insufficient measured training walltime')
        recipe = review['recipes'][row['architecture']]
        if (row['steps'] != recipe['total_steps'] or recipe['epochs'] != 50
                or recipe['steps_per_epoch']*50 != recipe['total_steps']
                or recipe['global_batch_size'] != 512):
            raise ValueError('Frozen schedule differs from compute budget')
    if (models[0]['steps'] != 1000000 or
            models[1]['steps'] != review['budget']['matched_steps']):
        raise ValueError('Wrong training budget')


def storage_budget(rows, parameters):
    # Bound all 50 possible improving candidates, best copy, two simultaneous
    # atomic latest-checkpoint writes per arm, predictions, snapshots and logs.
    if len(rows) != 2 or any(type(r.get('checkpoint_bytes')) is not int or r['checkpoint_bytes'] <= 0 for r in rows):
        raise ValueError('Missing/invalid measured checkpoint storage')
    weights = sum(51*(8*n+2**20) for n in parameters.values())
    restarts = sum(3*int(r['checkpoint_bytes']) for r in rows)
    predictions = 2*5000000*48 + 64*2**20
    return dict(required_free_bytes=4*2**30+weights+restarts+predictions+2**30,
        reserve_bytes=4*2**30, weights_bytes=weights, restart_bytes=restarts,
        predictions_bytes=predictions, miscellaneous_bytes=2**30,
        note='Conservative remaining-output budget; no dataset or pair-cache copies; not a quota reservation.')


def inspect(benchmark):
    benchmark = Path(benchmark).resolve(strict=True)
    science = activate(benchmark)
    spec = science.load(benchmark)
    review = science.summarize(benchmark)  # Recompute, do not trust Slurm COMPLETED.
    saved = science.checked(benchmark/'reports/benchmark_summary.json')
    if saved != science.hashed(review): raise ValueError('Saved benchmark summary differs from evidence')
    validate_review(review)
    rows = [science.checked(benchmark/'benchmarks'/f'{a}.json') for a in ('compact','matched')]
    for i, row in enumerate(rows):
        hours = 1.3*row['validation']['seconds']*5000000/row['validation']['events']/3600
        if not math.isfinite(hours) or hours > EVALUATE_HOURS[i]:
            raise ValueError('Five-hour evaluation request below measured estimate')
    recipes = {a:dict(r, evaluation_population=5000000) for a,r in review['recipes'].items()}
    tasks = [dict(t, recipe=t['architecture']) for t in spec['tasks']]
    names = ('campaign.json','snapshot.json','launch.json','benchmarks/compact.json',
             'benchmarks/matched.json','reports/benchmark_summary.json')
    return dict(benchmark=str(benchmark), benchmark_spec=spec, review=review,
        science=dict(tasks=tasks, recipes=recipes, parameters=spec['parameters'], inventory=spec['inventory']),
        evidence={str(benchmark/n):sha(benchmark/n) for n in names},
        python=spec['launch']['python'], account=spec['launch']['account'],
        train_hours=[r['recommended_hours'] for r in review['models']],
        evaluate_hours=EVALUATE_HOURS, storage=storage_budget(rows, spec['parameters']),
        runtime_by_architecture={r['task']['architecture']:dict(gpu=r['gpu'],gpu_memory_bytes=r['gpu_memory_bytes'],
            world_size=1,batch_size=512,workers=2,batchnorm='native',tf32=False,cudnn_benchmark=False,
            environment_sha256=hashed(spec['environment'])['content_hash']) for r in rows},
        flop_curves={r['task']['architecture']:r['flops'] for r in rows})


def load(output):
    output = Path(output).resolve(strict=True)
    plan = checked(output/'production.json')
    if (plan.get('schema') != SCHEMA or plan.get('authorizes_training') is not True
            or plan['policy'] != POLICY or plan['output'] != str(output)
            or plan['evaluate_hours'] != EVALUATE_HOURS or set(plan['source']) != set(FILES)):
        raise ValueError('Production policy/location changed')
    if inventory(output/'source') != plan['source']:
        raise ValueError('Production frozen source changed')
    for name, checksum in plan['evidence'].items():
        if sha(name) != checksum: raise ValueError('Benchmark evidence changed: '+name)
    current = inspect(plan['benchmark'])
    if any(plan.get(k) != v for k,v in current.items()):
        raise ValueError('Production admission no longer matches measured evidence')
    return plan


def run_directory(output, task):
    return Path(output)/'runs'/task['run_id']/f"seed_{task['seed']}"
