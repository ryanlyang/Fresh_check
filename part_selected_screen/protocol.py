"""Frozen seed-101 four-cell selected-relations experiment."""
from pathlib import Path
import copy
import math
from part_ev_fullscale.protocol import read_json, write_json, file_sha256, file_lock
from part_ev_screen.protocol import recipes
from part_ev_screen.precision import POLICY
from part_layerwise_screen.protocol import require_environment
from .memory import EXECUTION, PARITY
from part_screen_common.contracts import (hashed, checked_json, donor_contract,
    verify_source, runtime_identity, verify_dataset_metadata)

SCHEMA = 'part_selected_screen_v1'
SHARED = ['PART_RANGER', 'EV_RANGER']
LAYERWISE = ['PART_LAYERWISE_RANGER', 'EV_LAYERWISE_RANGER']
COUNTS = dict(selected_shared=2153406, selected_shared_ev=2218942,
              selected_layerwise=2157158, selected_layerwise_ev=2222694)
TASKS = [dict(run_id=name, architecture=arch, recipe='ranger_short_v1', seed=101)
    for name, arch in zip(('SELECTED_PART_SHARED', 'SELECTED_EV_SHARED',
                          'SELECTED_PART_LAYERWISE', 'SELECTED_EV_LAYERWISE'), COUNTS)]
MATCHED = dict(zip((t['run_id'] for t in TASKS), SHARED + LAYERWISE))
CONFIG = dict(families=['PT', 'TRACK', 'REGION'], encoded_dimensions=[8, 12, 12],
    region_lca='correct_set_bit_binary_lifting_not_legacy_overlifting',
    pair_input_width=36, pair_stem_width=64, pair_processing='dense_directed_including_padding_as_native',
    pair_normalization_population='directed_pairs_not_native_triangle',
    particle_inputs='unchanged_native_17', extra_inputs='four_unclipped_track_measurements',
    trimming='unchanged_native_trim_then_recompute_relations_on_retained_particles',
    ev='unchanged_independent_per_layer_projections',
    initialization='native_backbone_then_rng_isolated_added_parameters',
    memory_execution=EXECUTION, memory_parity=PARITY)
BENCHMARK = dict(warmup_steps=20, measured_steps=5000, window_steps=1000,
    full_length_steps=4, validation_events=131072, max_skip_fraction=.01, memory_fraction_limit=.9)


def task_directory(output, index):
    if type(index) is not int or not 0 <= index < 4:
        raise ValueError('Only four new task indices 0..3 are valid')
    return Path(output)/'runs'/TASKS[index]['run_id']/'seed_101'


def source_manifest(root):
    root = Path(root)
    paths = []
    for name in ('part_selected_screen', 'part_layerwise_screen', 'part_screen_common',
                 'part_ev_screen', 'part_ev_fullscale', 'teacher_logit_reco/relational_part'):
        paths += list((root/name).rglob('*.py'))
        paths += list((root/name).rglob('*.cpp'))
    paths += [root/p for p in ('scripts/run_part_selected_screen.py', 'scripts/run_part_ev_screen.py',
        'scripts/build_relational_part_tree_backend.py', 'sbatch/run_part_selected_screen_sporc.sh',
        'sbatch/submit_part_selected_screen_sporc.sh', 'particle_transformer/data/JetClass/JetClass_full.yaml',
        'teacher_logit_reco/__init__.py')]
    return {p.relative_to(root).as_posix(): file_sha256(p) for p in sorted(paths)}


def layerwise_contract(root, shared, source):
    from part_layerwise_screen import protocol as L
    root = Path(root).resolve(strict=True)
    spec = checked_json(root/'experiment_spec.json', L.SCHEMA)
    if (spec['tasks'] != L.TASKS or spec['model'] != L.CONFIG or spec['parameters'] != L.COUNTS
            or spec['precision'] != POLICY or spec['donor'] != shared
            or spec['environment'] != shared['environment'] or spec['inventory'] != shared['inventory']
            or spec['recipes'] != {'ranger_short_v1': recipes()['ranger_short_v1']}
            or spec['admission'] != L.ADMISSION):
        raise ValueError('Layerwise controls do not match the shared donor recipe/data/runtime')
    if spec['source'] != L.source_manifest(source):
        raise ValueError('Layerwise numerical source differs from completed controls')
    verify_source(root/'source', spec['source'])
    return dict(path=str(root), campaign_sha256=spec['content_hash'], controls=LAYERWISE,
        tasks={t['run_id']: t for t in spec['tasks']}, recipes=spec['recipes'],
        runtime=shared['runtime'], inventory=spec['inventory'],
        parameters={t['run_id']: L.COUNTS[t['architecture']] for t in spec['tasks']},
        experiment_spec_sha256=file_sha256(root/'experiment_spec.json'))


def create_spec(output):
    output = Path(output)
    source = Path(__file__).resolve().parents[1]
    launch = read_json(output/'launch.json')
    shared = donor_contract(launch['donor'], SHARED, local_source=source)
    require_environment(shared['environment'])
    verify_dataset_metadata(shared['inventory'])
    layerwise = layerwise_contract(launch['layerwise_donor'], shared, source)
    normal = checked_json(output/'inputs/normalization.json', 'part_selected_normalization_v1')
    from .normalization import SETTINGS
    if (normal['inventory_sha256'] != shared['inventory']['inventory_sha256']
            or normal['settings'] != SETTINGS or normal['population'] != 'training_only'):
        raise ValueError('Normalization training population differs')
    return hashed(dict(schema=SCHEMA, tasks=copy.deepcopy(TASKS), model=CONFIG, parameters=COUNTS,
        recipes={'ranger_short_v1': recipes()['ranger_short_v1']}, donor=shared, layerwise_donor=layerwise,
        environment=shared['environment'], inventory=shared['inventory'], precision=POLICY,
        benchmark=BENCHMARK, launch=launch, source=source_manifest(source),
        inputs={p: file_sha256(output/p) for p in ('inputs/selected.yaml', 'inputs/normalization.json',
                                                'backend/backend_manifest.json')}))


def load_campaign(output, *, check_environment=True):
    output = Path(output)
    spec = checked_json(output/'experiment_spec.json', SCHEMA)
    if (spec['tasks'] != TASKS or spec['model'] != CONFIG or spec['parameters'] != COUNTS
            or spec['precision'] != POLICY or spec['benchmark'] != BENCHMARK
            or spec['recipes'] != {'ranger_short_v1': recipes()['ranger_short_v1']}):
        raise ValueError('Frozen selected-relations protocol differs')
    if spec['source'] != source_manifest(Path(__file__).resolve().parents[1]):
        raise ValueError('Use the unchanged frozen source snapshot')
    if any(file_sha256(output/p) != sha for p, sha in spec['inputs'].items()):
        raise ValueError('Prepared input/backend artifacts changed')
    if check_environment:
        require_environment(spec['environment'])
    return spec


def require_approval(output, spec, runtime=None):
    if runtime is not None and runtime_identity(runtime) != spec['donor']['runtime']:
        raise ValueError('Runtime differs from reused controls')
    rows = []
    for task in TASKS:
        row = read_json(Path(output)/'benchmarks'/f"{task['run_id']}.json")
        if (row.get('passed') is not True or row.get('task') != task
                or row.get('campaign_sha256') != spec['content_hash']
                or runtime_identity(row.get('runtime', {})) != spec['donor']['runtime']
                or row.get('input_fingerprints') != spec['donor']['input_fingerprints']):
            raise ValueError('All four new models must pass their matched GPU checks')
        parity = row.get('initial_parity', {}).get('memory_execution', {})
        if (parity.get('passed') is not True or parity.get('execution') != EXECUTION
                or parity.get('policy') != PARITY):
            raise ValueError('Missing/mismatched memory execution parity check')
        hours = row['projected_training_hours']*1.3
        eval_hours = row['validation']['seconds']*5000000/row['validation']['events']/3600*1.3
        if (not math.isfinite(hours) or hours <= 0 or not math.isfinite(eval_hours) or eval_hours <= 0
                or hours > float(spec['launch']['train_hours']) or eval_hours > float(spec['launch']['evaluate_hours'])):
            raise ValueError('Measured projected runtime exceeds the queued walltime; inspect benchmarks before resubmission')
        rows.append(row)
    return rows
