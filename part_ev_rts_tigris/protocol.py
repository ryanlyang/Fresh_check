"""Authenticate SPORC science read-only; freeze a separate ARM/GH200 identity."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

SCHEMA = 'state_readout_triangles_support_tigris_v1'
PACKAGES = {'readout': 'part_ev_pair_readout', 'triangles': 'part_ev_triangles',
            'support': 'part_ev_support'}
COUNTS = {'readout': 2349702, 'triangles': 2334662, 'support': 2384070}
FILES = (
    'part_ev_rts_tigris/__init__.py', 'part_ev_rts_tigris/protocol.py',
    'part_ev_rts_tigris/evidence.py', 'part_ev_rts_tigris/benchmark.py',
    'part_ev_rts_tigris/production.py', 'part_ev_rts_tigris/submission.py',
    'scripts/run_part_ev_rts_tigris.py', 'scripts/submit_part_ev_rts_tigris.py',
    'sbatch/run_part_ev_rts_tigris.sh', 'sbatch/submit_part_ev_rts_tigris.sh',
)
POLICY = dict(world_size=1, global_batch=512, local_batch=512, workers_per_rank=2,
    torch_threads=4, validation_workers=2, validation_batch=512, validation_events=128000,
    warmup=20, measured=4200, window=1000, fixed_warmup=10, fixed_steps=40,
    restart_steps=7, memory_fraction=.90, max_skip_fraction=.05, planning_margin=1.30,
    checkpoint_interval=1000, max_seconds=18000, minimum_vram_bytes=80*2**30,
    training_precision='fp16', validation_precision='fp32', batchnorm='native',
    inference_batch=256, inference_batches=20, latency_iterations=100,
    tf32=False, cudnn_benchmark=False, test_data=False, authorizes_training=False)
MINUTES = {'prepare': 120, 'benchmark': 330, 'summary': 15, 'dispatch': 30}
RESERVE_BYTES = 4*2**30
BENCHMARK_BYTES = 2**30


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def hashed(row):
    return dict(row, content_hash=digest(row))


def checked(path):
    row = read(path)
    if row != hashed({k: v for k, v in row.items() if k != 'content_hash'}):
        raise ValueError('Metadata hash changed: '+str(path))
    return row


def write(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(row, stream, indent=2, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def inventory(source):
    result = {}
    for path in sorted(Path(source).rglob('*')):
        if path.is_symlink(): raise ValueError('Symlink in frozen source')
        if path.is_file(): result[path.relative_to(source).as_posix()] = sha(path)
    return result


def verify_imports(source):
    """No science may leak from the evolving live checkout into a donor run."""
    for name, module in list(sys.modules.items()):
        if name.startswith(('part_ev_rts_tigris', 'scripts.run_part_ev_rts_tigris',
                            'scripts.submit_part_ev_rts_tigris')):
            continue
        if name.startswith(('part_ev_', 'part_selected_', 'part_screen_', 'part_layerwise_',
                            'scripts.run_part_', 'scripts.check_part_')):
            filename = getattr(module, '__file__', None)
            if filename and not Path(filename).resolve().is_relative_to(source):
                raise ValueError('Scientific import outside frozen SPORC benchmark: '+name)


def activate(benchmark, kind):
    if kind not in PACKAGES: raise ValueError('Unknown model')
    source = Path(benchmark).resolve(strict=True)/'source'
    manifest = read(Path(benchmark)/'source_manifest.json')
    if not manifest or inventory(source) != manifest:
        raise ValueError('SPORC benchmark frozen source changed')
    sys.path.insert(0, str(source))
    science = importlib.import_module(PACKAGES[kind]+'.protocol')
    verify_imports(source)
    return science


def donor_contract(benchmark, kind):
    """No checkpoint loading, data rehash, environment mutation or donor writes."""
    benchmark = Path(benchmark).resolve(strict=True)
    science = activate(benchmark, kind)
    if kind == 'support' and getattr(science, 'NATIVE_PATH_VERSION', None) != 2:
        raise ValueError('Support donor predates the native-path parity fix; prepare a fresh '
                         'SPORC support benchmark from corrected source and use its new path')
    parent = science.load(benchmark, check_environment=False)
    science.verify_inventory_metadata(parent)
    if (parent['task'] != science.TASK or parent['task']['seed'] != 101
            or parent['parameters'] != COUNTS[kind]
            or parent['recipe'] != science.recipes()['ranger_full_v1']):
        raise ValueError('Donor model/full training recipe differs')
    evidence = {name: sha(benchmark/name) for name in (
        'campaign.json', 'launch.json', 'source_manifest.json', 'data_contents.json')}
    return science, parent, dict(path=str(benchmark), content_hash=parent['content_hash'],
        evidence=evidence)


def runtime_environment(science, parent):
    from part_ev_fullscale.model import reference_metadata
    reference_metadata()
    actual = science.runtime_environment()  # Stable install roots, not lazy sys.path vendors.
    if actual['architecture'] != 'aarch64' or actual['torch_cuda'] is None:
        raise ValueError('Use a CUDA-enabled ARM Python on Tigris; never the SPORC x86 environment')
    if actual['weaver_sources'] != parent['environment']['weaver_sources']:
        raise ValueError('Weaver model/data/optimizer sources differ from the SPORC experiment')
    return actual


def inspect(benchmark, kind):
    science, parent, donor = donor_contract(benchmark, kind)
    return dict(donor=donor, environment=runtime_environment(science, parent),
                task=parent['task'], parameters=parent['parameters'])


def launch_config(output, *, interpreter=True):
    output = Path(output).resolve(strict=True)
    row = checked(output/'launch.json')
    if (row.get('schema') != SCHEMA or row.get('policy') != POLICY
            or row.get('authorizes_training') is not False or row['model_kind'] not in PACKAGES
            or row.get('auto_production') is not True
            or type(row.get('max_train_hours')) is not int or not 1 <= row['max_train_hours'] <= 480
            or type(row.get('max_evaluate_hours')) is not int or not 1 <= row['max_evaluate_hours'] <= 24
            or row['output'] != str(output) or row['partition'] != 'tigris'
            or set(row['source']) != set(FILES) or inventory(output/'source') != row['source']):
        raise ValueError('Frozen Tigris launch/source/location changed')
    if output.parent != Path(row['repo'])/'checkpoints/part_ev_rts_tigris':
        raise ValueError('Tigris output is outside its independent registry')
    donor = Path(row['benchmark']).resolve(strict=True)
    if output == donor or output.is_relative_to(donor) or donor.is_relative_to(output):
        raise ValueError('Tigris output must be disjoint from SPORC')
    if interpreter and Path(sys.executable).resolve() != Path(row['python']).resolve():
        raise ValueError('Use the frozen ARM interpreter')
    return row


def runtime():
    return {k: POLICY[k] for k in ('world_size', 'global_batch', 'local_batch',
        'workers_per_rank', 'torch_threads', 'batchnorm', 'tf32', 'cudnn_benchmark')}


def used_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file() and not p.is_symlink())


def required_storage(repo, output, *, production=False):
    """Aggregate outstanding output budgets, not a filesystem quota reservation."""
    repo, output = Path(repo), Path(output)
    required = RESERVE_BYTES + max(0, BENCHMARK_BYTES-used_bytes(output))
    for family in ('part_ev_rts_tigris', 'part_ev_extensions_tigris'):
        for launch in (repo/'checkpoints'/family).glob('*/launch.json'):
            if launch.parent == output: continue
            other = checked(launch)
            summary_file = launch.parent/'summary.json'
            if not summary_file.is_file() or read(summary_file).get('passed') is not True:
                required += max(0, other['output_budget_bytes']-used_bytes(launch.parent))
    # Count benchmark reservations from the other three SPORC diagnostic families too.
    for family in PACKAGES.values():
        for launch in (repo/'checkpoints'/family).glob('*/launch.json'):
            other = checked(launch)
            if not (launch.parent/'summary.json').is_file() or not read(launch.parent/'summary.json').get('passed'):
                required += max(0, other.get('output_budget_bytes', 2*2**30)-used_bytes(launch.parent))
    for plan_file in (repo/'checkpoints/part_ev_rts_tigris').glob('*/production.json'):
        if plan_file.parent == output and production: continue
        plan = checked(plan_file)
        if not (plan_file.parent/'production/reports/production_report.json').is_file():
            required += max(0, plan['storage']['required_free_bytes']-RESERVE_BYTES-used_bytes(plan_file.parent/'production'))
    sporc = repo/'checkpoints/part_ev_state_production'
    for owner in sporc.glob('*.owner.json'):
        other = Path(read(owner)['output'])
        if other != other.resolve() or not other.is_relative_to(sporc):
            raise ValueError('Unexpected SPORC owner path')
        plan = checked(other/'campaign.json')
        if not (other/'reports/production_report.json').is_file():
            budget = plan['storage']
            required += max(0, budget['required_free_bytes']-budget['reserve_bytes']-used_bytes(other))
    if production:
        launch = checked(output/'launch.json')
        budget = storage_budget(COUNTS[launch['model_kind']])
        required += max(0, budget['required_free_bytes']-RESERVE_BYTES-used_bytes(output/'production'))
    return required


def check_storage(launch, *, production=False):
    output = Path(launch['output'])
    needed = required_storage(launch['repo'], output, production=production)
    if shutil.disk_usage(output if output.exists() else output.parent).free < needed:
        raise OSError(f'Need {needed/2**30:.2f} GiB aggregate output headroom; nothing deleted')
    return needed


def prepare(output):
    output = Path(output)
    launch = launch_config(output)
    if (output/'campaign.json').exists(): raise FileExistsError('Already prepared; use a fresh output')
    science, parent, donor = donor_contract(launch['benchmark'], launch['model_kind'])
    env = runtime_environment(science, parent)
    if launch['admission'] != dict(donor=donor, environment=env,
                                  task=parent['task'], parameters=parent['parameters']):
        raise ValueError('Donor/ARM environment changed after submission')
    check_storage(launch)
    spec = hashed(dict(schema=SCHEMA, launch=launch, policy=POLICY, donor=donor,
        model_kind=launch['model_kind'], task=parent['task'], recipe=parent['recipe'],
        model=parent['model'], parameters=parent['parameters'], parity=science.PARITY,
        inventory=parent['inventory'], environment=env, authorizes_training=False,
        cross_platform_restart=False, cross_hardware_numerical_equivalence=False))
    if launch['model_kind'] == 'support':
        spec = hashed(dict({k:v for k,v in spec.items() if k != 'content_hash'},
                           parameter_inventory_sha256=parent['parameter_inventory_sha256']))
    write(output/'campaign.json', spec)
    return spec


def load(output, *, check_environment=True):
    output = Path(output)
    launch = launch_config(output)
    spec = checked(output/'campaign.json')
    science, parent, donor = donor_contract(launch['benchmark'], launch['model_kind'])
    fixed = dict(schema=SCHEMA, launch=launch, policy=POLICY, donor=donor,
        model_kind=launch['model_kind'], task=parent['task'], recipe=parent['recipe'],
        model=parent['model'], parameters=parent['parameters'], parity=science.PARITY,
        inventory=parent['inventory'], authorizes_training=False,
        cross_platform_restart=False, cross_hardware_numerical_equivalence=False)
    if launch['model_kind'] == 'support':
        fixed['parameter_inventory_sha256'] = parent['parameter_inventory_sha256']
    if any(spec.get(k) != v for k, v in fixed.items()):
        raise ValueError('Tigris campaign or read-only SPORC evidence changed')
    if launch['admission'] != dict(donor=donor, environment=spec['environment'],
                                  task=parent['task'], parameters=parent['parameters']):
        raise ValueError('Campaign differs from submission admission')
    if check_environment and runtime_environment(science, parent) != spec['environment']:
        raise ValueError('Prepared ARM environment changed')
    return spec


def summary(output):
    output = Path(output)
    missing = [name for name in ('campaign.json', 'benchmark.json') if not (output/name).is_file()]
    if missing:
        return dict(passed=False, complete=False, authorizes_training=False, missing=missing)
    spec = load(output, check_environment=False)
    from .evidence import summarize
    return summarize(spec, checked(output/'benchmark.json'))



# Independent native-one-GPU training. No changes to the SPORC engine.
PRODUCTION_POLICY = dict(world_size=1, global_batch=512, local_batch=512, workers_per_rank=2,
    validation_workers=2, validation_batch=512, checkpoint_interval=1000,
    heartbeat_batches=100, shutdown_margin_seconds=900, max_skip_fraction=.05,
    restart='latest_full_state_exact_stream_replay', validation_checks=50,
    selection='highest_accuracy_first_tie', test_data=False, persistent_pair_cache=False,
    retention='latest_restart_and_at_most_50_improving_weights_plus_best')

write_json = write
read_json = read
file_sha256 = sha

def storage_budget(parameters):
    # Conservative buffers; both old probes and all old campaigns are left alone.
    weights = 51*(8*parameters+2**20)
    restarts = 3*(24*parameters+16*2**20)
    predictions = 5000000*48+64*2**20
    return dict(required_free_bytes=4*2**30+weights+restarts+predictions+2**30,
        reserve_bytes=4*2**30, weights_bytes=weights, restart_bytes=restarts,
        predictions_bytes=predictions, miscellaneous_bytes=2**30,
        note='No dataset/cache copies. Includes atomic writes, 50 candidates and 4 GiB reserve; not a quota reservation.')


def expected_runtime(spec):
    return dict(campaign_sha256=spec['content_hash'], world_size=1, local_batch_size=512,
        workers_per_rank=2, batchnorm='native', gpu=spec['expected_gpu']['name'],
        gpu_memory_bytes=spec['expected_gpu']['total_memory'], nccl_environment={},
        tf32=False, cudnn_benchmark=False)


def run_directory(output, spec):
    return Path(output)/'runs'/spec['task']['run_id']/f"seed_{spec['task']['seed']}"


from contextlib import contextmanager

@contextmanager
def file_lock(path, *, blocking=False):
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield


def production_spec(output):
    """Recompute admission, pin all native benchmark evidence and derive the full budget."""
    output = Path(output)
    spec = load(output)
    result = summary(output)
    launch = spec['launch']
    if (result.get('passed') is not True or result.get('complete') is not True
            or read(output/'summary.json') != result or launch['auto_production'] is not True):
        raise ValueError('Automatic production needs opt-in and unchanged passed GH200 evidence')
    train, evaluate = result['recommended_train_hours'], result['recommended_evaluate_hours']
    if (type(train) is not int or type(evaluate) is not int or train < 1 or evaluate < 1
            or train > launch['max_train_hours'] or evaluate > launch['max_evaluate_hours']):
        raise ValueError('Measured walltime exceeds pre-approved budget; no training submitted')
    gpu = checked(output/'benchmark.json')['gpus'][0]
    return hashed(dict(schema=SCHEMA, benchmark_sha256=spec['content_hash'],
        model_kind=spec['model_kind'], task=spec['task'], tasks=[spec['task']],
        parameters={spec['task']['architecture']:spec['parameters']},
        recipes={spec['task']['recipe']:spec['recipe']}, inventory=spec['inventory'],
        model_config=spec['model'], environment=spec['environment'], policy=PRODUCTION_POLICY,
        expected_gpu=dict(name=gpu['name'], total_memory=gpu['total_memory'], nccl_environment={}),
        evidence={name:sha(output/name) for name in ('launch.json','campaign.json','benchmark.json','summary.json')},
        storage=storage_budget(spec['parameters']), train_hours=train, evaluate_hours=evaluate,
        report_hours=1, seed=101, authorizes_training=True, fresh_training=True))


def load_production(output):
    output = Path(output)
    saved = checked(output/'production.json')
    if saved != production_spec(output) or checked(output/'production/campaign.json') != saved:
        raise ValueError('Production admission, evidence or environment changed')
    return saved

