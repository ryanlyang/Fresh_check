"""New execution identity; immutable science/reference and fail-closed admission."""
import copy
import math
from pathlib import Path
from part_selected_tigris import protocol as R
from part_selected_screen.memory import EXECUTION, PARITY
from part_screen_common.contracts import checked_json, hashed, verify_source, verify_dataset_metadata, runtime_identity
from part_ev_fullscale.protocol import file_sha256, file_lock, read_json, write_json
from part_ev_screen.protocol import runtime_environment, recipes
from .parity import POLICY
from . import trees

SCHEMA='part_selected_fast_v1'
REPORT_SCHEMA='part_selected_fast_report_v1'
TASKS,COUNTS,CONFIG,MATCHED=R.TASKS,R.COUNTS,R.CONFIG,R.MATCHED
SHARED,LAYERWISE=R.SHARED,R.LAYERWISE
S=R.S
BENCHMARK=R.BENCHMARK
task_directory=R.task_directory
MIN_SPEEDUP=1.05


def execution(launch):
    threads=int(launch['tree_threads']); cpus=int(launch['cpus']); cache=int(launch['cache_bytes'])
    if not 1<=threads<=64 or cpus<threads+4 or cache<0:
        raise ValueError('Need tree threads plus four CPU slots and a nonnegative cache budget')
    memory=int(launch['memory_gib'])*2**30
    if cache > memory//2:
        raise ValueError('Cache may use at most half requested host RAM; retain loader/backend headroom')
    if type(launch['compile_region']) is not bool:
        raise ValueError('compile_region must be boolean')
    return dict(version='selected_fast_v1',tree_backend='selected_tree_batch_v1',tree_threads=threads,
                cache_bytes=cache,cache_policy='validation_exact_order_only',compile_region=launch['compile_region'],
                pipelining=False,track_reassociation=False)


def source_manifest(root):
    root=Path(root)
    manifest=R.source_manifest(root)
    paths=[p for p in (root/'part_selected_fast').rglob('*') if p.suffix in ('.py','.cpp','.h')]
    paths += [root/p for p in ('scripts/profile_part_selected_fast.py','scripts/run_part_selected_fast.py',
                              'scripts/submit_part_selected_fast.py',
                              'sbatch/submit_part_selected_fast_tigris.sh','sbatch/run_part_selected_fast_tigris.sh')]
    manifest.update({p.relative_to(root).as_posix():file_sha256(p) for p in paths})
    return dict(sorted(manifest.items()))


def reference(root, *, environment=True):
    root=Path(root).resolve(strict=True)
    # R validates unchanged numerical source against this snapshot. New fast
    # files do not modify its manifest. Also authenticate the saved donor copy.
    spec=R.load_campaign(root,check_environment=environment)
    verify_source(root/'source',spec['source'])
    verify_dataset_metadata(spec['inventory'])
    return spec


def create_spec(output):
    output=Path(output)
    launch=read_json(output/'launch.json')
    parent=Path(launch['reference']).resolve(strict=True)
    if output==parent or output.is_relative_to(parent) or parent.is_relative_to(output):
        raise ValueError('Fast campaign must be disjoint from the original')
    old=reference(parent)
    if runtime_environment()!=old['environment']:
        raise ValueError('Use the reference Tigris environment unchanged')
    for name in ('inputs/selected.yaml','inputs/normalization.json','inputs/native_stream.json'):
        if file_sha256(output/name)!=old['inputs'][name]:
            raise ValueError('Reference inputs/normalization differ')
    return hashed(dict(schema=SCHEMA,reference=dict(path=str(parent),spec_sha256=file_sha256(parent/'experiment_spec.json'),
        campaign_sha256=old['content_hash']), tasks=copy.deepcopy(TASKS),parameters=COUNTS,model=CONFIG,
        precision=old['precision'],recipes=old['recipes'],inventory=old['inventory'],environment=old['environment'],
        donor=old['donor'],layerwise_donor=old['layerwise_donor'],input_reference=old['input_reference'],
        hardware_transfer=old['hardware_transfer'],environment_differences_from_donor=old['environment_differences_from_donor'],
        execution=execution(launch),launch=launch,benchmark=BENCHMARK,parity=POLICY,
        source=source_manifest(Path(__file__).resolve().parents[1]),
        inputs={p:file_sha256(output/p) for p in ('inputs/selected.yaml','inputs/normalization.json',
                'inputs/native_stream.json','backend/manifest.json','backend/parity.json')}))


def load_campaign(output, *, check_environment=True):
    output=Path(output)
    spec=checked_json(output/'experiment_spec.json',SCHEMA)
    if (spec['tasks']!=TASKS or spec['parameters']!=COUNTS or spec['model']!=CONFIG or spec['precision']!=S.POLICY
            or spec['benchmark']!=BENCHMARK or spec['parity']!=POLICY
            or spec['recipes']!={'ranger_short_v1':recipes()['ranger_short_v1']}
            or spec['launch']!=read_json(output/'launch.json') or spec['execution']!=execution(spec['launch'])):
        raise ValueError('Frozen fast execution/science contract differs')
    if spec['source']!=source_manifest(Path(__file__).resolve().parents[1]):
        raise ValueError('Use the frozen fast source')
    parent=Path(spec['reference']['path'])
    if file_sha256(parent/'experiment_spec.json')!=spec['reference']['spec_sha256']:
        raise ValueError('Reference campaign changed')
    old=reference(parent,environment=check_environment)
    if old['content_hash']!=spec['reference']['campaign_sha256']:
        raise ValueError('Reference identity differs')
    for key in ('inventory','environment','donor','layerwise_donor','input_reference','hardware_transfer','environment_differences_from_donor'):
        if spec[key]!=old[key]: raise ValueError(f'Reference field differs: {key}')
    required={'inputs/selected.yaml','inputs/normalization.json','inputs/native_stream.json','backend/manifest.json','backend/parity.json'}
    if set(spec['inputs'])!=required or any(file_sha256(output/p)!=sha for p,sha in spec['inputs'].items()):
        raise ValueError('Prepared artifact changed')
    for name in required-{'backend/manifest.json','backend/parity.json'}:
        if spec['inputs'][name]!=old['inputs'][name]: raise ValueError('Reference preprocessing changed')
    backend=read_json(output/'backend/parity.json')
    if (backend.get('passed') is not True or backend.get('exact') is not True
            or backend.get('invalid_input_rejected') is not True
            or [r['particles'] for r in backend.get('cases',[])]!=[1,2,11,32,64,128]):
        raise ValueError('Native compiled backend has not passed reference parity')
    return spec


def validate_runtime(runtime):
    R.validate_runtime(runtime)


def require_approval(output,spec,runtime=None):
    # Retain original stream, precision, memory-parity and walltime admission.
    rows=R.require_approval(output,spec,runtime)
    for row,task in zip(rows,TASKS):
        profile=checked_json(Path(output)/'profiles'/f"{task['run_id']}.json",'selected_fast_profile_v1')
        if (profile.get('campaign_sha256')!=spec['content_hash'] or profile.get('task')!=task
                or profile.get('execution')!=spec['execution'] or profile.get('passed') is not True
                or profile.get('runtime')!=row['runtime']):
            raise ValueError('Missing or incompatible paired profile')
        speed=profile.get('conservative_speedup',float('nan'))
        if not math.isfinite(speed) or speed<MIN_SPEEDUP:
            raise ValueError('Need a measured same-GH200 gain of at least 5% before production')
        from scripts.profile_part_selected_fast import summarize, STEPS
        measurements=profile.get('measurements',{})
        summary=summarize(measurements.get('reference',[]),measurements.get('candidate',[]))
        if summary['conservative_speedup']!=speed or profile.get('inputs_match') is not True:
            raise ValueError('Paired measurements do not support the reported gain')
        for variant in ('reference','candidate'):
            for trial,measurement in enumerate(measurements[variant]):
                if (measurement.get('trial')!=trial or measurement.get('steps')!=STEPS
                        or measurement.get('input_fingerprints')!={k:spec['input_reference']['fingerprints'][k] for k in ('0','20')}):
                    raise ValueError('Incomplete or mismatched paired timing stream')
        fast=row.get('initial_parity',{}).get('fast_execution',{})
        if fast.get('passed') is not True or fast.get('policy')!=POLICY:
            raise ValueError('Missing fast reference/gradient/optimizer parity')
        if row['runtime'].get('execution')!=spec['execution']:
            raise ValueError('Benchmark execution flags differ')
        if row['measured_steps']!=BENCHMARK['measured_steps'] or len(row['windows'])!=5:
            raise ValueError('Incomplete sustained benchmark')
        # Compare the admitted candidate's slow-window complete-job projection
        # against a conservative reference estimate. The reference validation
        # sample is small, so this is planning evidence, not a speed guarantee.
        reference_steps=min(m['seconds']/m['steps'] for m in measurements['reference'])
        cold_validation=measurements['reference'][0].get('validation_cold',{})
        v_seconds=cold_validation.get('seconds',float('nan'))
        v_events=cold_validation.get('events',0)
        if not math.isfinite(v_seconds) or v_seconds<=0 or v_events!=4096:
            raise ValueError('Missing cold reference validation timing')
        recipe=spec['recipes'][task['recipe']]
        reference_hours=(reference_steps*recipe['total_steps'] +
            v_seconds/v_events*recipe['epochs']*recipe['validation_events_per_epoch'])/3600
        if row['projected_training_hours']*MIN_SPEEDUP>reference_hours:
            raise ValueError('Sustained candidate projection including startup is not faster than reference')
        # Arena pages are demand-allocated. A short probe may not fill them;
        # conservatively add the entire cache allowance to observed process RSS.
        host=max(row.get('host_peak_bytes',float('inf')),profile.get('host_peak_bytes',float('inf')))+spec['execution']['cache_bytes']
        if not math.isfinite(host) or host<=0 or host>=int(spec['launch']['memory_gib'])*2**30*.9:
            raise ValueError('Host memory headroom not established')
        reserved=row.get('gpu_peak_reserved_bytes',float('inf'))
        if not math.isfinite(reserved) or reserved<=0 or reserved>=row['runtime']['gpu_memory_bytes']*.9:
            raise ValueError('Reserved GPU memory headroom not established')
    return rows


def training_runtime(output,spec):
    return runtime_identity(require_approval(output,spec)[0]['runtime'])
