"""Six fresh trainings plus four read-only ongoing full-schedule controls."""
import copy
import importlib.util
import math
from pathlib import Path
from part_ev_state_fullscale import protocol as old
from part_ev_message_study.controls import runtime_identity
from .model import CONFIG, COUNTS

_import = importlib.util.spec_from_file_location('part_ev_state_fullscale._capacity_protocol', old.__file__)
_base = importlib.util.module_from_spec(_import)
_import.loader.exec_module(_base)
ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'part_ev_state_capacity_v1'
SEEDS = old.SEEDS
DEFAULT_CONTROLS = '/home/ryreu/atlas/Fresh_check/checkpoints/part_ev_state_fullscale/state_full_20261024T185049Z'
CONTROL_INDICES = (0, 1, 2, 6)
TASKS = [dict(run_id=name, architecture=arch, seed=seed, recipe='ranger_full_v1')
         for name, arch in (('EV_STATE_FAST_RANGER', 'state_fast'),
                            ('PART_STATE_MATCHED_RANGER', 'part_state_matched')) for seed in SEEDS]
TASKS += [copy.deepcopy(old.TASKS[i]) for i in CONTROL_INDICES]
TRAIN_INDICES = tuple(range(6))
BENCHMARK_INDICES = (0, 3)
BENCHMARK = copy.deepcopy(old.BENCHMARK)


def task_directory(output, index):
    if type(index) is not int or not 0 <= index < len(TASKS):
        raise ValueError('Task index must be 0..9')
    task = TASKS[index]
    return Path(output) / 'runs' / task['run_id'] / f"seed_{task['seed']}"


def source_manifest(root):
    root = Path(root)
    names = set(old.source_manifest(root))
    names.update(p.relative_to(root).as_posix() for p in (root / 'part_ev_state_capacity').glob('*.py'))
    names.update(('scripts/run_part_ev_state_capacity.py',
                  'scripts/switch_part_ev_state_capacity.py',
                  'sbatch/submit_part_ev_state_capacity.sh', 'sbatch/run_part_ev_state_capacity.sh'))
    return {name: old.file_sha256(root / name) for name in sorted(names)}


def control_contract(path, *, check_environment=True):
    root = Path(path).resolve(strict=True)
    spec = old.load_campaign(root, check_environment=check_environment)
    # Audit frozen source as well as compatibility with the new checkout.
    for name, checksum in spec['source'].items():
        file = (root / 'source' / name).resolve(strict=True)
        if not file.is_relative_to(root / 'source') or old.file_sha256(file) != checksum:
            raise ValueError('Frozen control source changed')
    summary = old.require_approval(root, spec)
    ledger = [line.split() for line in (root / 'submitted_job_ids.tsv').read_text().splitlines() if line.strip()]
    jobs = {}
    for phase in ('train', 'evaluate', 'report'):
        ids = [row[1] for row in ledger if len(row) == 2 and row[0] == phase]
        if len(ids) != 1 or not ids[0].isdigit():
            raise ValueError('Expected one original submission chain; inspect retries manually')
        jobs[phase] = ids[0]
    contract = dict(path=str(root), campaign_sha256=spec['content_hash'],
        campaign_file_sha256=old.file_sha256(root / 'campaign.json'),
        approval_file_sha256=old.file_sha256(root / 'production_approval.json'),
        jobs=jobs, indices=list(CONTROL_INDICES), runtime=runtime_identity(summary['runtime']))
    return spec, contract


def create_spec(inventory, launch, data_contents):
    source, controls = control_contract(launch['controls'])
    if inventory != source['inventory'] or old.digest(data_contents) != source['data_contents_sha256']:
        raise ValueError('New study and controls must use identical verified data')
    for key in ('python', 'data_root', 'gpu', 'account', 'partition', 'cpus', 'memory'):
        if launch[key] != source['launch'][key]:
            raise ValueError(f'Control resource differs: {key}')
    spec = _base.create_spec(inventory, launch, data_contents)
    spec.pop('content_hash')
    spec['controls'] = controls
    spec['content_hash'] = old.digest(spec)
    return spec


def load_campaign(output, *, check_environment=True):
    spec = _base.load_campaign(output, check_environment=check_environment)
    source, controls = control_contract(spec['launch']['controls'], check_environment=False)
    if controls != spec.get('controls') or source['inventory'] != spec['inventory']:
        raise ValueError('Frozen control provenance changed')
    return spec


for _name, _value in dict(ROOT=ROOT, SCHEMA=SCHEMA, TASKS=TASKS, CONFIG=CONFIG,
                         COUNTS=COUNTS, BENCHMARK_INDICES=BENCHMARK_INDICES,
                         BENCHMARK=BENCHMARK, source_manifest=source_manifest,
                         task_directory=task_directory).items():
    setattr(_base, _name, _value)
_core_review = _base.benchmark_summary


def benchmark_summary(output, spec, runtime=None):
    summary = _core_review(output, spec, runtime)
    if runtime_identity(summary['runtime']) != spec['controls']['runtime']:
        raise ValueError('GPU/runtime differs from the reused controls')
    for row in summary['models']:
        parity = row['initial_parity']
        if row['task']['architecture'] == 'state_fast':
            g = parity.get('state_gradients', {})
            if (parity.get('reference_kind') != 'ordinary_EV_zero_state_branch'
                    or g.get('passed') is not True or g.get('checkpoint_backward') is not True
                    or g.get('forward', {}).get('passed') is not True or g.get('tensors', 0) < 1
                    or not math.isfinite(g.get('max_absolute_error', math.inf))):
                raise ValueError('All-eight nonzero-state forward/gradient evidence missing')
        elif (parity.get('reference_kind') != 'independent_width599_ParT'
              or parity.get('capacity_control') != dict(parameters=2323750, target=2323710,
                    delta=40, particle_ffn_width=599, class_ffn_width=512)):
            raise ValueError('Capacity-control admission missing')
    # A reused model may evaluate more slowly than the two new probes.
    source = old.read_json(Path(spec['controls']['path']) / 'campaign.json')
    old_summary = old.require_approval(Path(spec['controls']['path']), source)
    fingerprints = old_summary['models'][0]['input_fingerprints']
    if any(row['input_fingerprints'] != fingerprints for row in summary['models']):
        raise ValueError('Seed-101 input stream differs from the reused control campaign')
    summary['recommended_evaluate_hours'] = max(summary['recommended_evaluate_hours'],
                                                old_summary['recommended_evaluate_hours'])
    return summary


_base.benchmark_summary = benchmark_summary


def __getattr__(name):
    return getattr(_base, name)
