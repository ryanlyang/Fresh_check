"""Independent, benchmark-only frozen protocol; no historical writes."""
import importlib.metadata
import math
from pathlib import Path
import shutil
import sysconfig
from part_ev_fullscale.protocol import digest, file_sha256, read_json, write_json, recipe_for, file_lock

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'compact_state_benchmarks_v1'
TASKS = [dict(run_id='COMPACT_STATE_LAYERWISE', architecture='compact', seed=101),
         dict(run_id='PART_MATCH_COMPACT_COMPUTE', architecture='matched', seed=101)]
COUNTS = {'compact': 2262950, 'matched': 2263662}
BENCHMARK = dict(warmup_steps=20, measured_steps=4200, window_steps=1000,
                 validation_events=131072, full_length_confirmations=10,
                 phase_seconds=18000, shutdown_margin_seconds=600, planning_margin=1.3)
EXECUTION = dict(batch_size=512, workers=2, world_size=1, gpu='GH200', batchnorm='native',
                 cpus=12, memory_gib=128, tf32=False, cudnn_benchmark=False)
MODEL = dict(rank=16, state_blocks=list(range(8)), shared_state_pair_projection=True,
             layerwise_bias=True, ordinary_ev_all_blocks=True, standard_relations_only=True,
             unordered_pair_projection=True, merged_output_projection=True, checkpoint_chunks=64,
             matched_particle_ffn=570, matched_class_ffn=512, matched_bias='shared')


def hashed(row): return dict(row, content_hash=digest(row))


def checked(path):
    row = read_json(path)
    if row.get('content_hash') != digest({k:v for k,v in row.items() if k != 'content_hash'}):
        raise ValueError(f'Metadata hash changed: {path}')
    return row


def recipe(architecture, total_steps=1000000):
    if architecture not in COUNTS or type(total_steps) is not int or total_steps % 50 or total_steps < 1000000:
        raise ValueError('Need valid architecture and a budget divisible by 50')
    if architecture == 'compact' and total_steps != 1000000:
        raise ValueError('Compact budget is fixed')
    row = recipe_for()
    row.update(total_steps=total_steps, steps_per_epoch=total_steps//50,
               pair_bias='layerwise' if architecture == 'compact' else 'shared',
               ddp_batchnorm='not_applicable_single_GPU_native_BatchNorm')
    return row


def runtime_environment():
    from part_ev_fullscale.protocol import environment
    row = environment()
    # Importing torch optimizers can append setuptools/_vendor to sys.path.
    # Count installed distributions ONLY at the interpreter's install roots.
    paths = sorted({sysconfig.get_path(k) for k in ('purelib', 'platlib')})
    # Match the donor's exact tuple-sorting/collision convention. Some existing
    # environments expose multiple metadata records for the same distribution.
    # A bare dict comprehension keeps whichever record happens to be enumerated
    # last, and can falsely report a downgrade. This is legacy fingerprint
    # compatibility, NOT a semantic-version resolver or a claim about which
    # duplicate record describes the imported code. Keep core package/version
    # and Weaver source checks unchanged, and do not remove metadata on disk.
    row['installed_distributions'] = dict(sorted(
        (d.metadata['Name'].lower(), d.version)
        for d in importlib.metadata.distributions(path=paths) if d.metadata['Name']))
    return row


def require_space(output, minimum=6*2**30):
    existing = Path(output)
    while not existing.exists(): existing = existing.parent
    if shutil.disk_usage(existing).free < minimum:
        raise OSError(f'Need {minimum/2**30:g} GiB free; nothing deleted')


def verify_inventory(inventory):
    from part_ev_fullscale.data import CLASSES, SPLITS
    if inventory.get('inventory_sha256') != digest({k:v for k,v in inventory.items() if k != 'inventory_sha256'}):
        raise ValueError('Inventory hash changed')
    for split, size in (('train', 1000), ('val', 50)):
        rows = inventory['splits'][split]['files']
        expected = {f'{name}_{i:03d}.root' for name in CLASSES for i in SPLITS[split][1]}
        if (len(rows) != size or {r['name'] for r in rows} != expected
                or len({r['path'] for r in rows}) != size):
            raise ValueError('Incomplete or duplicate dataset inventory')
        for row in rows:
            path = Path(row['path'])
            st = path.stat()
            if (path.is_symlink() or path.name != row['name'] or row['num_entries'] != 100000
                    or st.st_size != row['size_bytes'] or st.st_mtime_ns != row['mtime_ns']):
                raise ValueError(f'Dataset metadata changed: {path}')


def prepare(output):
    output = Path(output)
    require_space(output)
    launch = read_json(output/'launch.json')
    reference = Path(launch['reference'])
    old = checked(reference/'experiment_spec.json')
    if old.get('schema') != 'selected_fullscale_benchmarks_v1':
        raise ValueError('Use the existing seven-run Tigris benchmark as metadata donor')
    env = runtime_environment()
    if env != old['environment']:
        differences = [k for k in env if env[k] != old['environment'].get(k)]
        raise ValueError(f'Use the unchanged donor environment; differences: {differences}')
    if env['architecture'] != 'aarch64': raise ValueError('This campaign is for Tigris ARM')
    files = ('part_ev_fullscale/model.py', 'part_ev_fullscale/data.py', 'part_ev_fullscale/optimizer.py',
             'part_ev_fullscale/checkpoint.py', 'part_ev_screen/precision.py', 'part_layerwise_screen/model.py',
             'scripts/run_part_ev_screen.py', 'particle_transformer/data/JetClass/JetClass_full.yaml')
    for name in files:
        expected = old['source'][name]
        if file_sha256(ROOT/name) != expected or file_sha256(reference/'source'/name) != expected:
            raise ValueError('Native numerical/data source changed: '+name)
    verify_inventory(old['inventory'])
    stream = checked(reference/'inputs/native_stream.json')
    if (stream != old['input_reference'] or file_sha256(reference/'inputs/native_stream.json') != old['inputs']['inputs/native_stream.json']
            or stream['inventory_sha256'] != old['inventory']['inventory_sha256'] or stream['environment'] != env
            or set(stream['fingerprints']) != {'0','20','1020','2020','4020'}):
        raise ValueError('Native Tigris input evidence changed')
    spec = hashed(dict(schema=SCHEMA, tasks=TASKS, parameters=COUNTS, model=MODEL,
        compact_recipe=recipe('compact'), matched_recipe_policy='same_50_checks_stretched_fractional_LR_schedule',
        benchmark=BENCHMARK, execution=EXECUTION, environment=env,
        inventory=old['inventory'], input_reference=stream,
        reference=dict(path=str(reference), spec_sha256=file_sha256(reference/'experiment_spec.json')),
        snapshot_sha256=file_sha256(output/'snapshot.json'), launch=launch, authorizes_training=False))
    if (output/'campaign.json').exists():
        if checked(output/'campaign.json') != spec: raise ValueError('Prepared campaign changed')
    else: write_json(output/'campaign.json', spec)
    return spec


def load(output):
    output = Path(output)
    spec = checked(output/'campaign.json')
    for key, value in dict(schema=SCHEMA,tasks=TASKS,parameters=COUNTS,model=MODEL,
                           compact_recipe=recipe('compact'),benchmark=BENCHMARK,
                           execution=EXECUTION,authorizes_training=False).items():
        if spec.get(key) != value: raise ValueError('Frozen contract changed: '+key)
    if (spec['snapshot_sha256'] != file_sha256(output/'snapshot.json')
            or spec['launch'] != read_json(output/'launch.json') or spec['environment'] != runtime_environment()):
        raise ValueError('Snapshot, launch or environment changed')
    verify_inventory(spec['inventory'])
    return spec


def summarize(output):
    from .flops import budget, mean_cost, quadratic, CONVENTION
    output = Path(output)
    spec = load(output)
    rows = [checked(output/'benchmarks'/f'{t["architecture"]}.json') for t in TASKS]
    costs, projections = {}, []
    for task, row in zip(TASKS, rows):
        if (row.get('passed') is not True or row.get('complete') is not True or row.get('task') != task
                or row.get('campaign_sha256') != spec['content_hash'] or row.get('authorizes_training') is not False
                or row['measured_steps'] != BENCHMARK['measured_steps']
                or row['input_fingerprints'] != spec['input_reference']['fingerprints']
                or row['parity']['passed'] is not True or row['restart']['passed'] is not True
                or row['flops']['convention'] != CONVENTION
                or sum(row['length_histogram'].values()) != BENCHMARK['measured_steps']
                or not 0 <= row['amp_skips'] < BENCHMARK['measured_steps']*.01
                or row['full_length_confirmed'] != 10
                or row['training_precision']['passed'] is not True
                or row['validation']['precision']['passed'] is not True
                or row['validation']['events'] != BENCHMARK['validation_events']):
            raise ValueError('Incomplete/mismatched benchmark: '+task['architecture'])
        if (not isinstance(row['gpu_memory_bytes'],int) or row['gpu_memory_bytes'] < 80*2**30
                or not math.isfinite(row['peak_reserved_bytes'])
                or not 0 < row['peak_reserved_bytes'] < .9*row['gpu_memory_bytes']
                or 'GH200' not in row.get('gpu','')):
            raise ValueError('Insufficient GPU memory headroom')
        curve = row['flops']
        if set(curve['samples']) != {'1','17','32','64','65','96','127','128'}:
            raise ValueError('Incomplete FLOP probes')
        for kind in ('forward','training'):
            fitted=quadratic({n:r[kind]['per_jet'] for n,r in curve['samples'].items() if n!='1'})
            if fitted != curve['coefficients'][kind]: raise ValueError('FLOP coefficients inconsistent with probes')
        costs[task['architecture']] = mean_cost(curve['coefficients']['training'], row['length_histogram'],
                                               singleton_cost=curve['samples']['1']['training']['per_jet'])
    if rows[0]['length_histogram'] != rows[1]['length_histogram']:
        raise ValueError('Trimmed-length populations differ; do not freeze a compute match')
    matched = budget(costs['compact'], costs['matched'])
    for row in rows:
        arch = row['task']['architecture']; steps = matched[arch+'_steps']
        windows = row['windows']
        if sum(w['steps'] for w in windows) != BENCHMARK['measured_steps']:
            raise ValueError('Incomplete sustained timing windows')
        values = [w['seconds']/w['steps'] for w in windows] + [row['validation']['seconds'], row['checkpoint_seconds']]
        if any(not math.isfinite(v) or v <= 0 for v in values): raise ValueError('Invalid timing')
        seconds = (steps*max(values[:len(windows)]) + 50*1280000/row['validation']['events']*row['validation']['seconds']
                   + (steps//1000+152)*row['checkpoint_seconds'] + row['startup_seconds'])
        projections.append(dict(architecture=arch, steps=steps, checks=50,
            parameters=COUNTS[arch], projected_hours=seconds/3600, recommended_hours=math.ceil(1.3*seconds/3600),
            train_jets_per_second=512*BENCHMARK['measured_steps']/sum(w['seconds'] for w in windows),
            peak_reserved_gib=row['peak_reserved_bytes']/2**30,
            forward_flops_per_jet_at_128=row['flops']['samples']['128']['forward']['per_jet'],
            fits_tigris_20_days=math.ceil(1.3*seconds/3600)<=480))
    return dict(passed=True, authorizes_training=False, budget=matched, models=projections,
                mean_training_flops_per_jet={k:float(v) for k,v in costs.items()},
                evidence={t['architecture']:file_sha256(output/'benchmarks'/f'{t["architecture"]}.json') for t in TASKS},
                recipes={a:recipe(a,matched[a+'_steps']) for a in COUNTS},
                note='Benchmark-only. Accuracy retention unmeasured; no production training is submitted. '
                     'Compute match estimates dense training work, not equal time or all operations.')
