"""Separate hardware-transfer contract, not an A100-matched runtime claim."""
import copy
import math
from pathlib import Path

from part_selected_screen import protocol as S
from part_selected_screen.memory import EXECUTION, PARITY
from part_ev_screen.protocol import runtime_environment, recipes
from part_ev_fullscale.model import reference_metadata
from part_layerwise_screen.protocol import environment_differences
from part_screen_common.contracts import (
    checked_json, hashed, donor_contract, runtime_identity, verify_dataset_metadata,
)
from part_ev_fullscale.protocol import read_json, write_json, file_sha256, file_lock

SCHEMA = 'part_selected_tigris_v1'
REPORT_SCHEMA = 'part_selected_tigris_report_v1'
TASKS, CONFIG, COUNTS = S.TASKS, S.CONFIG, S.COUNTS
SHARED, MATCHED, BENCHMARK = S.SHARED, S.MATCHED, S.BENCHMARK
LAYERWISE = S.LAYERWISE
task_directory = S.task_directory
layerwise_contract = S.layerwise_contract
TRANSFER = dict(
    policy='gh200_hardware_transfer_v1', architecture='aarch64', gpu='GH200',
    batch_size=512, workers=2, world_size=1, batchnorm='native',
    donor_hardware_matched=False, throughput_comparable_to_donors=False,
    numerical_equivalence_across_hardware_claimed=False,
    input_fingerprints='exact_match_to_frozen_native_tigris_stream',
    weaver_sources='exact_match_to_sporc_donor',
    training='fresh_initialization_no_sporc_checkpoint_resume',
)


def source_manifest(root):
    root = Path(root)
    result = S.source_manifest(root)
    paths = list((root/'part_selected_tigris').glob('*.py'))
    paths += [root/name for name in (
        'scripts/run_part_selected_tigris.py',
        'scripts/setup_part_selected_tigris_env.sh',
        'sbatch/run_part_selected_tigris.sh', 'sbatch/submit_part_selected_tigris.sh')]
    result.update({p.relative_to(root).as_posix(): file_sha256(p) for p in paths})
    return dict(sorted(result.items()))


def transfer_environment(donor):
    """Allow platform/package changes, but never a different Weaver algorithm."""
    reference_metadata()  # Version and ParticleTransformer source hash pinned.
    actual = runtime_environment()
    if actual.get('architecture') != 'aarch64':
        raise ValueError('Run this separate campaign in the Tigris ARM environment')
    if not actual.get('weaver_sources') or actual['weaver_sources'] != donor['environment'].get('weaver_sources'):
        raise ValueError('Tigris Weaver/preprocessing/optimizer sources differ from SPORC donor')
    return actual


def create_spec(output):
    output = Path(output)
    source = Path(__file__).resolve().parents[1]
    launch = read_json(output/'launch.json')
    shared = donor_contract(launch['donor'], SHARED, local_source=source)
    environment = transfer_environment(shared)
    verify_dataset_metadata(shared['inventory'])
    layerwise = layerwise_contract(launch['layerwise_donor'], shared, source)
    normal = checked_json(output/'inputs/normalization.json', 'part_selected_normalization_v1')
    inputs = checked_json(output/'inputs/native_stream.json', 'part_selected_tigris_inputs_v1')
    if (inputs['inventory_sha256'] != shared['inventory']['inventory_sha256']
            or inputs['environment'] != environment
            or inputs['donor_fingerprints'] != shared['input_fingerprints']
            or set(inputs['fingerprints']) != {'0', '20', '1020', '2020', '4020'}):
        raise ValueError('Native Tigris input reference differs')
    from part_selected_screen.normalization import SETTINGS
    if (normal['inventory_sha256'] != shared['inventory']['inventory_sha256']
            or normal['settings'] != SETTINGS or normal['population'] != 'training_only'):
        raise ValueError('Normalization training population differs')
    return hashed(dict(schema=SCHEMA, tasks=copy.deepcopy(TASKS), model=CONFIG, parameters=COUNTS,
        recipes={'ranger_short_v1': recipes()['ranger_short_v1']}, donor=shared, layerwise_donor=layerwise,
        environment=environment, environment_differences_from_donor=environment_differences(shared['environment'], environment),
        hardware_transfer=TRANSFER, input_reference=inputs,
        inventory=shared['inventory'], precision=S.POLICY,
        benchmark=BENCHMARK, launch=launch, source=source_manifest(source),
        inputs={p: file_sha256(output/p) for p in ('inputs/native_stream.json', 'inputs/selected.yaml', 'inputs/normalization.json',
                                                'backend/backend_manifest.json')}))


def load_campaign(output, *, check_environment=True):
    output = Path(output)
    spec = checked_json(output/'experiment_spec.json', SCHEMA)
    if (spec['tasks'] != TASKS or spec['model'] != CONFIG or spec['parameters'] != COUNTS
            or spec['precision'] != S.POLICY or spec['benchmark'] != BENCHMARK
            or spec['hardware_transfer'] != TRANSFER
            or spec['recipes'] != {'ranger_short_v1': recipes()['ranger_short_v1']}):
        raise ValueError('Frozen GH200 selected-relations protocol differs')
    if spec['source'] != source_manifest(Path(__file__).resolve().parents[1]):
        raise ValueError('Use the unchanged frozen Tigris source snapshot')
    required = {'inputs/native_stream.json', 'inputs/selected.yaml', 'inputs/normalization.json', 'backend/backend_manifest.json'}
    if set(spec['inputs']) != required or any(file_sha256(output/p) != sha for p, sha in spec['inputs'].items()):
        raise ValueError('Prepared input/backend artifacts changed')
    if spec['environment'].get('architecture') != 'aarch64':
        raise ValueError('Frozen environment is not ARM')
    if spec['input_reference'] != checked_json(output/'inputs/native_stream.json', 'part_selected_tigris_inputs_v1'):
        raise ValueError('Native input reference changed')
    if check_environment:
        # Equality is to THIS campaign's environment, never to its A100 donor.
        differences = environment_differences(spec['environment'], runtime_environment())
        if differences:
            raise ValueError(f'Use the unchanged Tigris preparation environment: {differences[:20]}')
    return spec


def validate_runtime(runtime):
    if ('GH200' not in runtime.get('gpu', '').upper()
            or runtime.get('gpu_memory_bytes', 0) < 80*1024**3
            or runtime.get('world_size') != 1 or runtime.get('batch_size') != 512
            or runtime.get('workers') != 2 or runtime.get('batchnorm') != 'native'
            or runtime.get('tf32') is not False or runtime.get('cudnn_benchmark') is not False
            or runtime.get('precision') != S.POLICY):
        raise ValueError('Expected one large-memory GH200 with the unchanged screen recipe')


def require_approval(output, spec, runtime=None):
    rows = []
    expected = runtime_identity(runtime) if runtime is not None else None
    for task in TASKS:
        row = read_json(Path(output)/'benchmarks'/f"{task['run_id']}.json")
        observed = row.get('runtime', {})
        validate_runtime(observed)
        if expected is None:
            expected = runtime_identity(observed)
        if (row.get('passed') is not True or row.get('task') != task
                or row.get('campaign_sha256') != spec['content_hash']
                or observed.get('campaign_sha256') != spec['content_hash']
                or runtime_identity(observed) != expected
                or row.get('input_fingerprints') != spec['input_reference']['fingerprints']
                or row.get('inputs_match_native_tigris') is not True):
            raise ValueError('All four models must pass on one GH200 runtime and the native Tigris input stream')
        parity = row.get('initial_parity', {}).get('memory_execution', {})
        if (row.get('initial_parity', {}).get('passed') is not True
                or parity.get('passed') is not True or parity.get('execution') != EXECUTION
                or parity.get('policy') != PARITY):
            raise ValueError('Missing/mismatched GH200 numerical parity evidence')
        hours = row['projected_training_hours']*1.3
        eval_hours = row['validation']['seconds']*5000000/row['validation']['events']/3600*1.3
        if (not math.isfinite(hours) or hours <= 0 or not math.isfinite(eval_hours) or eval_hours <= 0
                or hours > float(spec['launch']['train_hours']) or eval_hours > float(spec['launch']['evaluate_hours'])):
            raise ValueError('Measured projection exceeds frozen walltime; inspect before a fresh submission')
        rows.append(row)
    return rows


def training_runtime(output, spec):
    """Runtime authenticated by all four GH200 benchmarks, not donor hardware."""
    return runtime_identity(require_approval(output, spec)[0]['runtime'])
