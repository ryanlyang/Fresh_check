"""Stdlib identity boundary between benchmark evidence and production outputs."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys

SCHEMA = 'selected_seven_production_v1'
TRAIN_HOURS = (48, 48, 192, 144, 72, 192, 336)
EVALUATE_HOURS = (2, 2, 3, 2, 2, 3, 4)
FILES = (
    'part_selected_production/__init__.py', 'part_selected_production/protocol.py',
    'part_selected_production/engine.py', 'part_selected_production/reporting.py',
    'scripts/submit_part_selected_production.py', 'scripts/run_part_selected_production.py',
    'sbatch/run_part_selected_production_tigris.sh', 'sbatch/submit_part_selected_production_tigris.sh',
)
POLICY = dict(total_steps=1000000, checks=50, global_batch_size=512,
              checkpoint_interval=1000, shutdown_margin_seconds=900,
              restart='latest_full_state_with_exact_stream_replay',
              selection='highest_accuracy_first_tie', evaluation_population=5000000,
              persistent_pair_cache=False, test_data=False)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def hashed(value):
    return dict(value, content_hash=hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest())


def checked(path):
    row = read(path)
    if row != hashed({k: v for k, v in row.items() if k != 'content_hash'}):
        raise ValueError('Metadata hash changed: ' + str(path))
    return row


def write(path, value):
    """Atomic metadata commit; callers serialize writers with a file lock."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def activate(benchmark, retry):
    """Only the original snapshot supplies models, streams, optimizer and math."""
    benchmark, retry = Path(benchmark).resolve(strict=True), Path(retry).resolve(strict=True)
    settings = read(benchmark / 'launch.json')
    if Path(sys.executable).resolve() != Path(settings['python']).resolve():
        raise ValueError('Use the frozen benchmark Python environment')
    source = benchmark / 'source'
    sys.path.insert(0, str(source))
    retry_driver = module('_production_retry_evidence', retry / 'retry_part_selected_fullscale.py')
    plan = retry_driver.load_plan(retry)
    if Path(plan['campaign']) != benchmark or plan['python'] != settings['python']:
        raise ValueError('Retry belongs to a different benchmark/environment')
    import part_selected_fullscale.protocol as science
    if not Path(science.__file__).resolve().is_relative_to(source):
        raise ValueError('Scientific imports must come from the benchmark snapshot')
    return retry_driver, science


def inspect(benchmark, retry):
    """Recompute the complete mixed-evidence review, without changing artifacts."""
    benchmark, retry = Path(benchmark), Path(retry)
    driver, science = activate(benchmark, retry)
    spec = science.load_campaign(benchmark)  # Before optimizer/JIT imports.
    from types import SimpleNamespace
    with contextlib.redirect_stdout(io.StringIO()):
        review = driver.execute(SimpleNamespace(mode='review', output=retry))
    if not (review['complete'] and review['benchmarks_healthy']):
        raise ValueError('All seven authenticated benchmarks must be healthy')
    validate_resources(review)
    evidence = {}
    for i, task in enumerate(spec['tasks']):
        root = benchmark if i in science.SELECTED else retry
        path = root / 'benchmarks' / (task['run_id'] + '.json')
        evidence[str(path)] = sha(path)
    # load_plan authenticates the diagnostic, overlays, selected profiles and
    # original source. Pin its manifest as well as all seven measurement files.
    for path in (benchmark/'experiment_spec.json', benchmark/'snapshot.json',
                 retry/'retry.sha256', retry/'retry_plan.json'):
        evidence[str(path)] = sha(path)
    return dict(science=spec, review=review, evidence=evidence,
                python=spec['launch']['python'], account=spec['launch']['account'])


def validate_resources(review):
    if len(review.get('models', [])) != 7:
        raise ValueError('Need exactly seven resource measurements')
    for i, row in enumerate(review['models']):
        p = row.get('projection', {})
        if (row.get('status') != 'measured' or
                not 0 < p.get('recommended_train_hours', float('inf')) <= TRAIN_HOURS[i] or
                not 0 < p.get('recommended_evaluate_hours', float('inf')) <= EVALUATE_HOURS[i]):
            raise ValueError('Agreed walltime is below measured recommendation for task ' + str(i))


def load(output, *, environment=True):
    output = Path(output).resolve(strict=True)
    plan = checked(output / 'production.json')
    if (plan['schema'] != SCHEMA or plan.get('authorizes_training') is not True or plan['policy'] != POLICY or
            plan['train_hours'] != list(TRAIN_HOURS) or
            plan['evaluate_hours'] != list(EVALUATE_HOURS) or plan['output'] != str(output)):
        raise ValueError('Production policy/location changed')
    if set(plan['source']) != set(FILES):
        raise ValueError('Incomplete production source inventory')
    for name, checksum in plan['source'].items():
        path = output / 'source' / name
        if path.is_symlink() or sha(path) != checksum:
            raise ValueError('Production source changed: ' + name)
    for path, checksum in plan['evidence'].items():
        if sha(path) != checksum:
            raise ValueError('Benchmark evidence changed: ' + path)
    _, science = activate(plan['benchmark'], plan['retry'])
    spec = science.load_campaign(Path(plan['benchmark']), check_environment=environment)
    if spec != plan['science']:
        raise ValueError('Frozen scientific contract changed')
    validate_resources(plan['review'])
    return plan


def run_directory(output, task):
    return Path(output) / 'runs' / task['run_id'] / ('seed_' + str(task['seed']))
