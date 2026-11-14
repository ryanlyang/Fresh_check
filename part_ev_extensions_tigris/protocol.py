"""Authenticate SPORC science read-only; freeze a separate ARM/GH200 identity."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

SCHEMA = 'state_extensions_tigris_benchmark_v1'
PACKAGES = {'bias': 'part_ev_state_bias', 'multiply': 'part_ev_state_multiply',
            'memory': 'part_ev_pair_memory'}
COUNTS = {'bias': 2329510, 'multiply': 2343846, 'memory': 2364294}
FILES = (
    'part_ev_extensions_tigris/__init__.py', 'part_ev_extensions_tigris/protocol.py',
    'part_ev_extensions_tigris/evidence.py', 'part_ev_extensions_tigris/benchmark.py',
    'scripts/run_part_ev_extensions_tigris.py', 'scripts/submit_part_ev_extensions_tigris.py',
    'sbatch/run_part_ev_extensions_tigris.sh', 'sbatch/submit_part_ev_extensions_tigris.sh',
)
POLICY = dict(world_size=1, global_batch=512, local_batch=512, workers_per_rank=2,
    torch_threads=4, validation_workers=2, validation_batch=512, validation_events=128000,
    warmup=20, measured=4200, window=1000, fixed_warmup=10, fixed_steps=40,
    restart_steps=7, memory_fraction=.90, max_skip_fraction=.05, planning_margin=1.30,
    checkpoint_interval=1000, max_seconds=18000, minimum_vram_bytes=80*2**30,
    training_precision='fp16', validation_precision='fp32', batchnorm='native',
    inference_batch=256, inference_batches=20, latency_iterations=100,
    tf32=False, cudnn_benchmark=False, test_data=False, authorizes_training=False)
MINUTES = {'prepare': 120, 'benchmark': 330, 'summary': 15}
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
        if name.startswith(('part_ev_extensions_tigris', 'scripts.run_part_ev_extensions_tigris',
                            'scripts.submit_part_ev_extensions_tigris')):
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
    parent = science.load(benchmark, check_environment=False)
    science.verify_inventory_metadata(parent)
    review = science.review(benchmark)
    if (review.get('passed') is not True or review.get('complete') is not True
            or review.get('authorizes_training') is not False
            or read(benchmark/'summary.json') != review):
        raise ValueError('Require unchanged passed SPORC benchmark and summary')
    if (parent['task'] != science.TASK or parent['task']['seed'] != 101
            or parent['parameters'] != COUNTS[kind]
            or parent['recipe'] != science.recipes()['ranger_full_v1']):
        raise ValueError('Donor model/full training recipe differs')
    evidence = {name: sha(benchmark/name) for name in (
        'campaign.json', 'launch.json', 'source_manifest.json', 'data_contents.json',
        'benchmark.json', 'summary.json')}
    return science, parent, dict(path=str(benchmark), content_hash=parent['content_hash'],
        evidence=evidence, sporc_summary=review)


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
            or row['output'] != str(output) or row['partition'] != 'tigris'
            or set(row['source']) != set(FILES) or inventory(output/'source') != row['source']):
        raise ValueError('Frozen Tigris launch/source/location changed')
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


def required_storage(repo, output):
    """One shared reserve plus outstanding budgets; read SPORC registry only."""
    repo, output = Path(repo), Path(output)
    required = RESERVE_BYTES+max(0, BENCHMARK_BYTES-used_bytes(output))
    registry = repo/'checkpoints/part_ev_extensions_tigris'
    for launch in registry.glob('*/launch.json'):
        if launch.parent == output: continue
        other = checked(launch)
        summary = launch.parent/'summary.json'
        # Failed/missing reports retain their budget until an explicit cleanup.
        if not summary.is_file() or read(summary).get('passed') is not True:
            required += max(0, other['output_budget_bytes']-used_bytes(launch.parent))
    sporc = repo/'checkpoints/part_ev_state_production'
    for owner in sporc.glob('*.owner.json'):
        other = Path(read(owner)['output'])
        if other != other.resolve() or not other.is_relative_to(sporc):
            raise ValueError('Unexpected SPORC owner path')
        plan = checked(other/'campaign.json')
        if not (other/'reports/production_report.json').is_file():
            budget = plan['storage']
            required += max(0, budget['required_free_bytes']-budget['reserve_bytes']-used_bytes(other))
    return required


def check_storage(launch):
    output = Path(launch['output'])
    needed = required_storage(launch['repo'], output)
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
