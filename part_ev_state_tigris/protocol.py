"""Read-only SPORC provenance; separate native ARM benchmark identity."""
import copy
import importlib.util
import math
from pathlib import Path

from part_ev_fullscale import protocol as F
from part_ev_fullscale.model import reference_metadata
from part_ev_layerwise_ddp import protocol as donor_protocol
from part_ev_message_study.controls import checked_hash
from part_ev_state_compare.model import CONFIG, COUNTS
from part_ev_state_fullscale import protocol as full

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'part_ev_state_tigris_benchmark_v1'
TASKS = [copy.deepcopy(donor_protocol.TASK)]
BENCHMARK_INDICES = (0,)
BENCHMARK = copy.deepcopy(full.BENCHMARK)
TRANSFER = dict(architecture='aarch64', gpu='GH200', minimum_vram_bytes=80*1024**3,
                world_size=1, batch_size=512, workers=2, batchnorm='native',
                initialization='fresh_no_sporc_checkpoints', authorizes_training=False,
                cross_hardware_numerical_equivalence=False, benchmark_seconds=2*3600,
                soft_seconds=105*60)
digest, read_json, write_json = F.digest, F.read_json, F.write_json
CLASS_NAMES = F.CLASS_NAMES
file_sha256, file_lock = F.file_sha256, F.file_lock
recipes, projected_hours, learning_rate = full.recipes, full.projected_hours, full.learning_rate
verify_inventory_metadata = full.verify_inventory_metadata

# Reuse the established full-budget evidence validator in a private namespace.
_import = importlib.util.spec_from_file_location('part_ev_state_fullscale._tigris_review', full.__file__)
_review = importlib.util.module_from_spec(_import)
_import.loader.exec_module(_review)
_review.TASKS, _review.BENCHMARK_INDICES = TASKS, BENCHMARK_INDICES


def source_manifest():
    result = donor_protocol.source_manifest()
    paths = list((ROOT/'part_ev_state_tigris').glob('*.py'))
    paths += [ROOT/name for name in ('scripts/benchmark_part_ev_state_tigris.py',
              'sbatch/run_part_ev_state_tigris_benchmark.sh',
              'sbatch/submit_part_ev_state_tigris_benchmark.sh')]
    result.update({p.relative_to(ROOT).as_posix():file_sha256(p) for p in paths})
    return result


def runtime_environment(donor):
    reference_metadata()
    actual = F.environment()  # Stable relevant-library/source fingerprint, not lazy sys.path vendors.
    if actual['architecture'] != 'aarch64' or actual['torch_cuda'] is None:
        raise ValueError('Use the CUDA-enabled ARM environment on Tigris')
    if actual['weaver_sources'] != donor['environment']['weaver_sources']:
        raise ValueError('Weaver model/preprocessing/optimizer sources differ from SPORC')
    return actual


def donor_contract(path):
    path = Path(path).resolve(strict=True)
    spec = donor_protocol.load(path, check_environment=False)
    estimate = donor_protocol.require_approval(path, spec)
    if spec['task'] != TASKS[0] or spec['recipes'] != recipes():
        raise ValueError('Expected the full-schedule layerwise all-eight seed-101 campaign')
    contract = dict(path=str(path), campaign_sha256=spec['content_hash'],
                    campaign_file_sha256=file_sha256(path/'campaign.json'),
                    approval_sha256=file_sha256(path/'production_approval.json'),
                    preflight_sha256=file_sha256(path/'preflight/result.json'),
                    original_estimate=estimate)
    return spec, contract


def prepare(output, donor):
    output, donor = Path(output).resolve(), Path(donor).resolve(strict=True)
    if output == donor or output.is_relative_to(donor) or donor.is_relative_to(output):
        raise ValueError('Use a separate output, never inside or above the SPORC campaign')
    if (output/'campaign.json').exists():
        raise FileExistsError('Use a new immutable diagnostic directory')
    parent, contract = donor_contract(donor)
    verify_inventory_metadata(parent)
    environment = runtime_environment(parent)
    launch = read_json(output/'launch.json')
    if Path(launch['donor']).resolve(strict=True) != donor:
        raise ValueError('Reference must match the frozen launch configuration')
    spec = dict(schema=SCHEMA, tasks=TASKS, recipes=recipes(), model=CONFIG,
                parameters={'state_layerwise':COUNTS['state_layerwise']}, benchmark=BENCHMARK,
                transfer=TRANSFER, inventory=parent['inventory'], environment=environment,
                source=source_manifest(), donor=contract,
                launch=launch, authorizes_training=False)
    spec['content_hash'] = digest(spec)
    write_json(output/'campaign.json', spec)
    return spec


def load_campaign(output, *, check_environment=True):
    output = Path(output)
    spec = checked_hash(read_json(output/'campaign.json'))
    expected = dict(schema=SCHEMA,tasks=TASKS,recipes=recipes(),model=CONFIG,
                    parameters={'state_layerwise':COUNTS['state_layerwise']},benchmark=BENCHMARK,
                    transfer=TRANSFER,source=source_manifest(),authorizes_training=False)
    if any(spec.get(key) != value for key,value in expected.items()):
        raise ValueError('Frozen Tigris benchmark contract changed')
    parent, contract = donor_contract(spec['donor']['path'])
    if (contract != spec['donor'] or parent['inventory'] != spec['inventory']
            or read_json(output/'launch.json') != spec['launch']):
        raise ValueError('Read-only donor, dataset identity or launch configuration changed')
    if check_environment and runtime_environment(parent) != spec['environment']:
        raise ValueError('Prepared Tigris environment changed')
    return spec


def validate_runtime(runtime):
    if ('GH200' not in runtime['gpu'] or runtime['gpu_memory_bytes'] < TRANSFER['minimum_vram_bytes']
            or any(runtime.get(k) != TRANSFER[k] for k in ('world_size','batch_size','workers','batchnorm'))
            or runtime.get('tf32') is not False or runtime.get('cudnn_benchmark') is not False):
        raise ValueError('Need one GH200, batch 512, native BatchNorm and frozen precision')


def benchmark_summary(output, spec):
    summary = _review.benchmark_summary(output, spec)
    validate_runtime(summary['runtime'])
    row = summary['models'][0]
    parity = row['initial_parity']
    gradient = parity.get('state_gradients', {})
    if (parity.get('reference_kind') != 'layerwise_EV_zero_state_branch'
            or gradient.get('architecture') != 'state_layerwise'
            or gradient.get('full_model') is not True or gradient.get('state_tensors') != 32
            or gradient.get('passed') is not True or gradient.get('checkpoint_backward') is not True
            or gradient.get('forward', {}).get('passed') is not True
            or gradient.get('tensors', 0) < 1
            or not math.isfinite(gradient.get('max_absolute_error', math.inf))):
        raise ValueError('All-eight nonzero-state gradient/equivalence evidence missing')
    rate = row['jets_per_second']
    if not math.isfinite(rate) or rate <= 0 or not math.isclose(
            rate, 512*BENCHMARK['measured_steps']/row['seconds'], rel_tol=1e-10):
        raise ValueError('Invalid throughput evidence')
    return summary


def review_view(output, spec):
    summary = benchmark_summary(output, spec)
    row = summary['models'][0]
    recipe = recipes()['ranger_full_v1']
    train = max(w['seconds']/w['steps'] for w in row['windows'])*recipe['total_steps']/3600
    validation = row['validation']['seconds']*recipe['epochs']*recipe['validation_events_per_epoch']/BENCHMARK['validation_events']/3600
    return dict(passed=True, authorizes_training=False, model=TASKS[0],
                gpu=summary['runtime']['gpu'], parameters=COUNTS['state_layerwise'],
                train_jets_per_second=row['jets_per_second'],
                peak_gib=max(row[k] for k in ('native_peak_bytes','full_length_peak_bytes','validation_peak_bytes'))/1024**3,
                amp_skipped_steps=row['amp_skipped_steps'],
                projected_train_compute_hours=train, projected_validation_hours=validation,
                projected_total_hours=row['projected_training_hours'],
                recommended_train_hours=summary['recommended_train_hours'],
                recommended_evaluate_hours=summary['recommended_evaluate_hours'],
                sporc_original_estimate=spec['donor']['original_estimate'],
                note='Includes all 50 validations, checkpoint writes and 30% walltime margin. '
                     'Excludes queues, restart replay and shared-filesystem contention. '
                     'Fresh single-GPU native BatchNorm, not bitwise-equivalent to A100 DDP. '
                     'No production training submitted or authorized by this diagnostic.')
