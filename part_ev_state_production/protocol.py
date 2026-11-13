"""Independent production admission. Import benchmark science only after authentication."""
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile

SCHEMA = 'state_extensions_production_v1'
PACKAGES = {'bias':'part_ev_state_bias', 'multiply':'part_ev_state_multiply'}
FILES = (
    'part_ev_state_production/__init__.py', 'part_ev_state_production/protocol.py',
    'part_ev_state_production/engine.py', 'scripts/run_part_ev_state_production.py',
    'scripts/submit_part_ev_state_production.py', 'sbatch/run_part_ev_state_production.sh',
    'sbatch/submit_part_ev_state_production.sh',
)
POLICY = dict(world_size=2, global_batch=512, local_batch=256, workers_per_rank=1,
    validation_workers=2, validation_batch=512, checkpoint_interval=1000,
    heartbeat_batches=100, shutdown_margin_seconds=900, max_skip_fraction=.05,
    restart='latest_full_state_exact_stream_replay', validation_checks=50,
    selection='highest_accuracy_first_tie', test_data=False, persistent_pair_cache=False,
    retention='latest_restart_and_at_most_50_improving_weights_plus_best')
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
        if path.is_file():
            result[path.relative_to(source).as_posix()] = sha(path)
    return result



def activate(benchmark, model):
    if model not in PACKAGES: raise ValueError('Unknown model')
    benchmark = Path(benchmark).resolve(strict=True)
    source = benchmark/'source'
    manifest = read(benchmark/'source_manifest.json')
    if not manifest or inventory(source) != manifest:
        raise ValueError('Benchmark frozen source changed')
    if Path(sys.executable).resolve() != Path(read(benchmark/'launch.json')['python']).resolve():
        raise ValueError('Use the original benchmark interpreter')
    sys.path.insert(0, str(source))
    science = importlib.import_module(PACKAGES[model]+'.protocol')
    for name, module in list(sys.modules.items()):
        scientific = name.startswith(('part_ev_', 'part_selected_', 'part_screen_', 'part_layerwise_',
                                      'scripts.run_part_', 'scripts.check_part_'))
        if scientific and not name.startswith(('part_ev_state_production','scripts.run_part_ev_state_production',
                                               'scripts.submit_part_ev_state_production')):
            path = getattr(module, '__file__', None)
            if path and not Path(path).resolve().is_relative_to(source):
                raise ValueError('Scientific import outside benchmark snapshot: '+name)
    return science


def validate_review(review):
    if any(review.get(k) is not True for k in ('passed','complete')) or review.get('authorizes_training') is not False:
        raise ValueError('Require complete passed benchmark evidence')
    projections = [review[k] for k in ('projected_train_hours','projected_validation_hours',
                                      'projected_checkpoint_hours')]
    if any(type(x) not in (float,int) or not math.isfinite(x) or x <= 0 for x in projections):
        raise ValueError('Invalid measured walltime')
    if (type(review['recommended_train_hours']) is not int
            or review['recommended_train_hours'] != math.ceil(1.3*sum(projections))
            or type(review['recommended_evaluate_hours']) is not int
            or review['recommended_evaluate_hours'] < 1):
        raise ValueError('Require unchanged benchmark walltime recommendations')


def storage_budget(parameters):
    # Conservative buffers; both old probes and all old campaigns are left alone.
    weights = 51*(8*parameters+2**20)
    restarts = 3*(24*parameters+16*2**20)
    predictions = 5000000*48+64*2**20
    return dict(required_free_bytes=4*2**30+weights+restarts+predictions+2**30,
        reserve_bytes=4*2**30, weights_bytes=weights, restart_bytes=restarts,
        predictions_bytes=predictions, miscellaneous_bytes=2**30,
        note='No dataset/cache copies. Includes atomic writes, 50 candidates and 4 GiB reserve; not a quota reservation.')


def inspect(benchmark, model):
    benchmark = Path(benchmark).resolve(strict=True)
    science = activate(benchmark,model)
    spec = science.load(benchmark)
    science.verify_inventory_metadata(spec)
    review = science.review(benchmark)
    validate_review(review)
    if read(benchmark/'summary.json') != review:
        raise ValueError('Saved summary differs from recomputed benchmark evidence')
    row = checked(benchmark/'benchmark.json')
    gpu = row['gpus'][0]
    names = ('campaign.json','launch.json','source_manifest.json','data_contents.json',
             'benchmark.json','summary.json')
    return dict(benchmark=str(benchmark), model_kind=model, benchmark_sha256=spec['content_hash'],
        task=spec['task'], tasks=[spec['task']], recipes={spec['task']['recipe']:spec['recipe']},
        parameters={spec['task']['architecture']:spec['parameters']}, inventory=spec['inventory'],
        model_config=spec['model'], environment=spec['environment'],
        evidence={str(benchmark/n):sha(benchmark/n) for n in names}, review=review,
        python=spec['launch']['python'], account=spec['launch']['account'], partition='tier3',
        train_hours=review['recommended_train_hours'], evaluate_hours=review['recommended_evaluate_hours'],
        expected_gpu=dict(name=gpu['name'], total_memory=gpu['total_memory'],
                          nccl_environment=spec['launch']['nccl_environment']),
        storage=storage_budget(spec['parameters']))


def load(output):
    output = Path(output).resolve(strict=True)
    spec = checked(output/'campaign.json')
    if (spec.get('schema') != SCHEMA or spec.get('authorizes_training') is not True
            or spec['policy'] != POLICY or spec['output'] != str(output)
            or set(spec['source']) != set(FILES) or inventory(output/'source') != spec['source']):
        raise ValueError('Production frozen policy/source/location changed')
    for name, checksum in spec['evidence'].items():
        if sha(name) != checksum: raise ValueError('Benchmark evidence changed: '+name)
    current = inspect(spec['benchmark'],spec['model_kind'])
    if any(spec.get(k) != v for k,v in current.items()):
        raise ValueError('Production admission no longer matches benchmark evidence')
    return spec


def expected_runtime(spec):
    return dict(campaign_sha256=spec['content_hash'], world_size=2, local_batch_size=256,
        workers_per_rank=1, batchnorm='SyncBatchNorm', gpu=spec['expected_gpu']['name'],
        gpu_memory_bytes=spec['expected_gpu']['total_memory'], tf32=False, cudnn_benchmark=False,
        nccl_environment=spec['expected_gpu']['nccl_environment'])


def run_directory(output, spec):
    task = spec['task']
    return Path(output)/'runs'/task['run_id']/f"seed_{task['seed']}"


# Lazy imports: stdlib-only admission must run before importing scientific code.
def file_lock(*args, **kwargs):
    from part_ev_fullscale.protocol import file_lock as lock
    return lock(*args, **kwargs)


read_json = read
write_json = write
file_sha256 = sha
def digest(row):
    return hashed(row)['content_hash']

