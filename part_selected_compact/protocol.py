"""Frozen benchmark-only admission; donors and existing campaigns are read-only."""
import math
from pathlib import Path
import sys
from part_ev_fullscale.protocol import file_lock, file_sha256, read_json, write_json
from part_ev_compact.protocol import (hashed, checked, runtime_environment, require_space,
                                     verify_inventory, recipe as native_recipe)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'selected_compact_benchmarks_v1'
TASKS = [dict(run_id='SELECTED_COMPACT_STATE_LAYERWISE', architecture='compact', seed=101),
         dict(run_id='SELECTED_PART_MATCH_COMPACT_COMPUTE', architecture='matched', seed=101)]
COUNTS = {'compact': 2272870, 'matched': 2273582}
BENCHMARK = dict(warmup_steps=20, measured_steps=4200, window_steps=1000,
                 validation_events=131072, full_length_confirmations=10,
                 phase_seconds=18000, shutdown_margin_seconds=600, planning_margin=1.3)
EXECUTION = dict(batch_size=512, workers=2, world_size=1, gpu='GH200', batchnorm='native',
                 cpus=12, memory_gib=128, tree_threads=4, tf32=False,
                 cudnn_benchmark=False, persistent_pair_cache=False, compile_region=False)
MODEL = dict(rank=16, state_blocks=list(range(8)), shared_state_pair_projection=True,
             compact_bias='layerwise', ordinary_ev_all_blocks=True, selected_relations=['PT', 'TRACK', 'REGION'],
             pair_population='dense_directed', unordered_pair_projection=False,
             merged_output_projection=True, checkpoint_chunks=64,
             matched_particle_ffn=570, matched_class_ffn=512, matched_bias='shared', matched_ev=False)
INPUTS = ('inputs/selected.yaml', 'inputs/normalization.json', 'inputs/native_stream.json')
NUMERICAL_FILES = (
    'part_ev_fullscale/model.py', 'part_ev_fullscale/data.py', 'part_ev_fullscale/optimizer.py',
    'part_ev_fullscale/checkpoint.py', 'part_ev_screen/precision.py', 'part_layerwise_screen/model.py',
    'scripts/run_part_ev_screen.py', 'part_selected_fullscale/steps.py',
    'particle_transformer/data/JetClass/JetClass_full.yaml',
    'part_selected_screen/model.py', 'part_selected_screen/data.py', 'part_selected_screen/features.py',
    'part_selected_screen/normalization.py', 'part_selected_screen/region.py',
    'part_selected_fast/model.py', 'part_selected_fast/trees.py',
    'teacher_logit_reco/relational_part/normalization.py',
    'teacher_logit_reco/relational_part/relation_pt.py',
    'teacher_logit_reco/relational_part/relation_track.py',
    'teacher_logit_reco/relational_part/relation_region.py',
    'teacher_logit_reco/relational_part/region_tree.py',
    'teacher_logit_reco/relational_part/ca_tree.py',
    'teacher_logit_reco/relational_part/contracts.py',
)


def recipe(architecture, total_steps=1000000):
    row = native_recipe(architecture, total_steps)
    row.update(selected_relations=['PT', 'TRACK', 'REGION'], pair_population='dense_directed')
    return row


def check_files(files):
    for name, expected in files.items():
        path = Path(name)
        if path.is_symlink() or not path.is_file() or file_sha256(path) != expected:
            raise ValueError('Pinned donor artifact changed: '+name)


def donor(launch):
    """Authenticate passed recovery WITHOUT activating or writing its old code."""
    from part_selected_compiler_recovery import protocol as R
    reference, recovery = Path(launch['reference']), Path(launch['recovery'])
    old = checked(reference/'experiment_spec.json')
    if old.get('schema') != 'selected_fullscale_benchmarks_v1':
        raise ValueError('Require the original seven-run selected benchmark donor')
    row = R.checked(recovery/'recovery.json')
    if (row.get('schema') != R.SCHEMA or row.get('output') != str(recovery)
            or set(row.get('source', {})) != set(R.FILES)
            or row.get('indices') != list(R.INDICES)
            or row.get('transition') != dict(old=R.OLD_BANNER, new=R.NEW_BANNER)):
        raise ValueError('Unsupported compiler recovery evidence')
    R.check_files({str(recovery/'source'/n): h for n, h in row['source'].items()})
    # Execute the same reviewed backend verifier, not a permissive new loader.
    for name in ('part_selected_compiler_recovery/backend.py', 'part_selected_compiler_recovery/protocol.py'):
        if file_sha256(ROOT/name) != row['source'][name]:
            raise ValueError('Reviewed backend verifier changed: '+name)
    approved = R.approval(recovery, row)
    plan = R.checked(Path(row['campaign'])/'production.json')
    if (plan['content_hash'] != row['production_sha256'] or plan['science'] != old
            or Path(plan['benchmark']) != reference
            or Path(plan['output']) != Path(row['campaign'])
            or plan['python'] != row['python'] or plan['python'] != launch['python']
            or Path(sys.executable).resolve() != Path(plan['python']).resolve()
            or R.evidence(plan) != row['artifacts']):
        raise ValueError('Recovery, interpreter, backend or donor identity differs')
    files = dict(row['artifacts'])
    files.update({str(recovery/n): file_sha256(recovery/n) for n in ('recovery.json', 'compatibility.json')})
    files[str(reference/'experiment_spec.json')] = file_sha256(reference/'experiment_spec.json')
    for name in NUMERICAL_FILES:
        expected = old['source'][name]
        if file_sha256(ROOT/name) != expected:
            raise ValueError('Selected numerical/data source changed: '+name)
        files[str(reference/'source'/name)] = expected
    for name in INPUTS:
        files[str(reference/name)] = old['inputs'][name]
    check_files(files)
    env = runtime_environment()
    if env != old['environment'] or env['architecture'] != 'aarch64':
        raise ValueError('Use the unchanged Tigris selected environment')
    verify_inventory(old['inventory'])
    stream = checked(reference/'inputs/native_stream.json')
    if (stream != old['input_reference'] or stream['environment'] != env
            or stream['inventory_sha256'] != old['inventory']['inventory_sha256']
            or set(stream['fingerprints']) != {'0', '20', '1020', '2020', '4020'}):
        raise ValueError('Donor stream fingerprints differ')
    context = dict(benchmark=str(reference), science=dict(reference=plan['science']['reference']))
    return old, files, context, approved['evidence']


def verify_backends(spec):
    from part_selected_compiler_recovery.backend import verify
    check_files(spec['donor_files'])
    reference, fast, evidence = verify(spec['backend_context'], expected=spec['backend_evidence'])
    return reference, fast, evidence


def prepare(output):
    output = Path(output)
    require_space(output)
    if (output/'campaign.json').exists():
        raise FileExistsError('Preparation is immutable; use a fresh output')
    launch = read_json(output/'launch.json')
    old, files, context, evidence = donor(launch)
    # All checks complete before copying small immutable input metadata.
    verify_backends(dict(donor_files=files, backend_context=context, backend_evidence=evidence))
    for name in INPUTS:
        path = output/name
        path.parent.mkdir(parents=True, exist_ok=True)
        content = (Path(launch['reference'])/name).read_bytes()
        if path.exists() and path.read_bytes() != content:
            raise ValueError('Prepared input differs: '+name)
        if not path.exists():
            path.write_bytes(content)
    from part_selected_screen.data import validate_config
    validate_config(output/'inputs/selected.yaml')
    spec = hashed(dict(schema=SCHEMA, tasks=TASKS, parameters=COUNTS, model=MODEL,
        compact_recipe=recipe('compact'), benchmark=BENCHMARK, execution=EXECUTION,
        environment=old['environment'], inventory=old['inventory'], input_reference=old['input_reference'],
        inputs={name: file_sha256(output/name) for name in INPUTS}, donor_files=files,
        backend_context=context, backend_evidence=evidence, launch=launch,
        snapshot_sha256=file_sha256(output/'snapshot.json'), authorizes_training=False))
    write_json(output/'campaign.json', spec)
    return spec


def load(output):
    output = Path(output)
    spec = checked(output/'campaign.json')
    for key, expected in dict(schema=SCHEMA, tasks=TASKS, parameters=COUNTS, model=MODEL,
                              benchmark=BENCHMARK, execution=EXECUTION,
                              compact_recipe=recipe('compact'), authorizes_training=False).items():
        if spec.get(key) != expected:
            raise ValueError('Frozen contract differs: '+key)
    if (spec['snapshot_sha256'] != file_sha256(output/'snapshot.json')
            or spec['launch'] != read_json(output/'launch.json')
            or Path(sys.executable).resolve() != Path(spec['launch']['python']).resolve()
            or spec['environment'] != runtime_environment()):
        raise ValueError('Snapshot, interpreter, launch or environment changed')
    check_files(spec['donor_files'])
    check_files({str(output/n): h for n, h in spec['inputs'].items()})
    verify_inventory(spec['inventory'])
    return spec


def positive(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def validate_benchmark(row, spec, task):
    from .flops import CONVENTION, LENGTHS, quadratic, mean_cost
    if (row.get('passed') is not True or row.get('complete') is not True
            or row.get('authorizes_training') is not False or row.get('task') != task
            or row.get('campaign_sha256') != spec['content_hash'] or row.get('execution') != EXECUTION
            or row.get('parameters') != COUNTS[task['architecture']]
            or row.get('backend_evidence') != spec['backend_evidence']
            or row.get('input_fingerprints') != spec['input_reference']['fingerprints']
            or set(row.get('selected_input_fingerprints', {})) != set(spec['input_reference']['fingerprints'])
            or row.get('measured_steps') != BENCHMARK['measured_steps']
            or row.get('parity', {}).get('passed') is not True
            or row.get('parity', {}).get('directed_pairs') is not True
            or row.get('restart', {}).get('passed') is not True
            or row.get('restart', {}).get('exact_each_update') is not True):
        raise ValueError('Incomplete/mismatched benchmark evidence')
    for key in ('full_length_confirmed',):
        if row.get(key) != BENCHMARK['full_length_confirmations']:
            raise ValueError('Missing full-length confirmation')
    if (type(row.get('amp_skips')) is not int or not 0 <= row['amp_skips'] < .01*BENCHMARK['measured_steps']
            or row.get('training_precision', {}).get('passed') is not True
            or row.get('validation', {}).get('precision', {}).get('passed') is not True
            or row['validation']['events'] != BENCHMARK['validation_events']):
        raise ValueError('AMP/precision/validation gate failed')
    for value, limit in ((row['peak_reserved_bytes'], row['gpu_memory_bytes']),
                         (row['host_peak_bytes'], EXECUTION['memory_gib']*2**30)):
        if not positive(value) or not positive(limit) or value >= .9*limit:
            raise ValueError('Insufficient memory headroom')
    if 'GH200' not in row.get('gpu', '') or row['gpu_memory_bytes'] < 80*2**30:
        raise ValueError('Unexpected GPU')
    hist, curve = row['length_histogram'], row['flops']
    if (sum(hist.values()) != BENCHMARK['measured_steps'] or curve['convention'] != CONVENTION
            or curve.get('selected_encoders_included') is not True or curve.get('directed_pairs') is not True
            or curve.get('checkpoint_recomputation_included') is not True
            or curve.get('cpu_tree_work_included') is not False
            or set(curve['samples']) != {str(n) for n in LENGTHS}):
        raise ValueError('Incomplete FLOP/length evidence')
    for kind in ('forward', 'training'):
        if any(not positive(r[kind]['per_jet']) for r in curve['samples'].values()):
            raise ValueError('Invalid measured FLOPs')
        fitted = quadratic({n: r[kind]['per_jet'] for n, r in curve['samples'].items() if n != '1'})
        if fitted != curve['coefficients'][kind]:
            raise ValueError('FLOP fit disagrees with independent probes')
    windows = row['windows']
    if [w['steps'] for w in windows] != [1000, 1000, 1000, 1000, 200]:
        raise ValueError('Incomplete sustained timing windows')
    times = [w['seconds'] for w in windows]+[row['validation']['seconds'], row['checkpoint_seconds'], row['startup_seconds']]
    if any(not positive(v) for v in times) or type(row['checkpoint_bytes']) is not int or row['checkpoint_bytes'] <= 0:
        raise ValueError('Missing or invalid timing/storage evidence')
    return mean_cost(curve['coefficients']['training'], hist,
                     singleton_cost=curve['samples']['1']['training']['per_jet'])


def summarize(output):
    from .flops import budget
    output = Path(output)
    spec = load(output)
    rows = [checked(output/'benchmarks'/f'{t["architecture"]}.json') for t in TASKS]
    costs = {t['architecture']: validate_benchmark(row, spec, t) for t, row in zip(TASKS, rows)}
    if (rows[0]['length_histogram'] != rows[1]['length_histogram']
            or rows[0]['selected_input_fingerprints'] != rows[1]['selected_input_fingerprints']
            or rows[0]['gpu'] != rows[1]['gpu'] or rows[0]['gpu_memory_bytes'] != rows[1]['gpu_memory_bytes']):
        raise ValueError('Arms differ in directed inputs, trimmed lengths or GPU type')
    matched = budget(costs['compact'], costs['matched'])
    projections = []
    for row in rows:
        arch = row['task']['architecture']
        steps = matched[arch+'_steps']
        train = steps*max(w['seconds']/w['steps'] for w in row['windows'])
        val = 50*1280000/row['validation']['events']*row['validation']['seconds']
        # Conservative allowance covers periodic restarts + validation/best saves.
        saves = (steps//1000+202)*row['checkpoint_seconds']
        hours = (train+val+saves+row['startup_seconds'])/3600
        projections.append(dict(architecture=arch, parameters=COUNTS[arch], steps=steps, checks=50,
            projected_hours=hours, recommended_hours=math.ceil(1.3*hours), fits_tigris_20_days=math.ceil(1.3*hours)<=480,
            projected_validation_hours=val/3600,
            recommended_evaluate_hours=math.ceil(1.3*row['validation']['seconds']*5000000/row['validation']['events']/3600),
            train_jets_per_second=512*BENCHMARK['measured_steps']/sum(w['seconds'] for w in row['windows']),
            peak_reserved_gib=row['peak_reserved_bytes']/2**30, host_peak_gib=row['host_peak_bytes']/2**30,
            forward_flops_per_jet_at_128=row['flops']['samples']['128']['forward']['per_jet']))
    storage = (4*2**30+2**30+sum(51*(8*n+2**20) for n in COUNTS.values())
               +sum(3*r['checkpoint_bytes'] for r in rows)+2*5000000*48)
    return dict(passed=True, authorizes_training=False, budget=matched, models=projections,
        recipes={a: recipe(a, matched[a+'_steps']) for a in COUNTS},
        mean_training_flops_per_jet={a: float(v) for a, v in costs.items()},
        required_free_gib_for_future_production=storage/2**30,
        evidence={t['architecture']: file_sha256(output/'benchmarks'/f'{t["architecture"]}.json') for t in TASKS},
        note='Benchmarks only. Dense training-compute match, not equal walltime/all operations. '
             'CPU trees, I/O and validation excluded from FLOPs but included in runtime projections. '
             'Accuracy retention unmeasured; production requires separate implementation/admission.')
