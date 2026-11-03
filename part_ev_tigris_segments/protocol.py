"""Separate production identity; the submitted benchmark remains immutable."""
import copy
from pathlib import Path

from part_ev_state_tigris import protocol as B
from part_ev_message_study.controls import checked_hash, runtime_identity
from part_ev_fullscale.checkpoint import trusted_load
from part_ev_state_fullscale.reporting import validate_training

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'part_ev_tigris_segments_v1'
TASK = copy.deepcopy(B.TASKS[0])
TASKS = [TASK]
CLASS_NAMES = B.CLASS_NAMES
POLICY = dict(walltime_seconds=72*3600, shutdown_margin_seconds=3600,
              default_segments=4, max_segments=8, checkpoint_interval=1000,
              evaluation_hours=5, dependency='afterok', world_size=1,
              fresh_initialization=True, restart='latest_full_state_exact_stream_replay',
              startup_restart_updates=7, startup_warmup_updates=20)
read_json, write_json, digest = B.read_json, B.write_json, B.digest
file_sha256, file_lock, learning_rate = B.file_sha256, B.file_lock, B.learning_rate


def hashed(row):
    return dict(row,content_hash=digest({k:v for k,v in row.items() if k!='content_hash'}))


def source_manifest():
    rows = B.source_manifest()
    paths = list((ROOT/'part_ev_tigris_segments').glob('*.py'))
    paths += [ROOT/name for name in ('scripts/run_part_ev_tigris_segments.py',
              'sbatch/submit_part_ev_tigris_segments.sh','sbatch/run_part_ev_tigris_segments.sh')]
    rows.update({p.relative_to(ROOT).as_posix():file_sha256(p) for p in paths})
    return rows


def reference(benchmark, job, *, check_environment=True):
    benchmark = Path(benchmark).resolve(strict=True)
    spec = B.load_campaign(benchmark,check_environment=check_environment)
    rows = [line.split() for line in (benchmark/'submitted_job_ids.tsv').read_text().splitlines()]
    if rows.count(['benchmark',str(job)]) != 1 or not str(job).isdigit():
        raise ValueError('Benchmark job must match the original submission ledger')
    return spec, dict(path=str(benchmark),job=str(job),campaign_sha256=spec['content_hash'],
                     file_sha256=file_sha256(benchmark/'campaign.json'))


def prepare(output, benchmark, job):
    output, benchmark = Path(output).resolve(), Path(benchmark).resolve(strict=True)
    parent, identity = reference(benchmark,job)
    for forbidden in (benchmark, Path(parent['donor']['path']).resolve()):
        if output==forbidden or output.is_relative_to(forbidden) or forbidden.is_relative_to(output):
            raise ValueError('Production must be separate from both benchmark and SPORC')
    if (output/'request.json').exists(): raise FileExistsError('Production request already frozen')
    row = hashed(dict(schema=SCHEMA,reference=identity,policy=POLICY,source=source_manifest(),
                      launch=read_json(output/'launch.json')))
    write_json(output/'request.json',row)
    return row


def request(output, *, check_environment=True):
    output = Path(output)
    row = checked_hash(read_json(output/'request.json'))
    parent, identity = reference(row['reference']['path'],row['reference']['job'],
                                 check_environment=check_environment)
    if (row['schema']!=SCHEMA or row['reference']!=identity or row['policy']!=POLICY
            or row['source']!=source_manifest() or row['launch']!=read_json(output/'launch.json')):
        raise ValueError('Frozen production request changed')
    return row, parent


def expected_spec(output, *, check_environment=True):
    req, parent = request(output,check_environment=check_environment)
    reference_root = Path(req['reference']['path'])
    summary = B.benchmark_summary(reference_root,parent)  # Never rely on Slurm exit status alone.
    if summary['recommended_evaluate_hours'] > POLICY['evaluation_hours']:
        raise ValueError('Benchmark projects more than the queued 5-hour evaluation; review resources')
    row = dict(schema=SCHEMA,request_sha256=req['content_hash'],policy=POLICY,source=source_manifest(),
               tasks=TASKS,recipes=parent['recipes'],model=parent['model'],parameters=parent['parameters'],
               inventory=parent['inventory'],environment=parent['environment'],
               benchmark_file_sha256=file_sha256(reference_root/'benchmarks'/f"{TASK['run_id']}.json"),
               benchmark_runtime=summary['runtime'],recommended_train_hours=summary['recommended_train_hours'])
    return hashed(row)


def load_campaign(output, *, check_environment=True):
    actual = checked_hash(read_json(Path(output)/'campaign.json'))
    if actual != expected_spec(output,check_environment=check_environment):
        raise ValueError('Production campaign or admitted benchmark evidence changed')
    return actual


def expected_runtime(spec):
    return dict(spec['benchmark_runtime'],campaign_sha256=spec['content_hash'])


def benchmark_summary(output,spec,runtime=None):
    if load_campaign(output,check_environment=False)!=spec:
        raise ValueError('Production admission changed')
    if runtime is not None and runtime!=expected_runtime(spec):
        raise ValueError('Runtime differs from the admitted one-GH200 benchmark')
    return dict(runtime=expected_runtime(spec))


def task_directory(output,index=0):
    if type(index) is not int or index!=0: raise ValueError('This campaign has only seed101 layerwise all-eight State-EV')
    return Path(output)/'runs'/TASK['run_id']/f"seed_{TASK['seed']}"


def audit_training(output,spec):
    return validate_training(Path(output),spec,TASK,runtime_identity(expected_runtime(spec)))


def snapshot(output,spec):
    run = task_directory(output)
    last = run/'last.pt'
    if not last.exists():
        if (run/'training_complete.json').exists(): raise ValueError('Completion without latest checkpoint')
        return dict(step=0,validated_checks=0,checkpoint_sha256=None,complete=False)
    row = trusted_load(last)
    state, recipe = row['state'],spec['recipes'][TASK['recipe']]
    if (row.get('schema')!=SCHEMA or row['campaign_sha256']!=spec['content_hash']
            or row['runtime']!=expected_runtime(spec) or state['task']!=TASK
            or not all(k in row for k in ('model','optimizer','scaler','rng'))
            or type(state['step']) is not int or type(state['validated_checks']) is not int
            or not 0<=state['step']<=recipe['total_steps']
            or not 0<=state['validated_checks']<=state['step']//recipe['steps_per_epoch']
            or state['step']//recipe['steps_per_epoch']-state['validated_checks']>1):
        raise ValueError('Invalid latest checkpoint identity/cursor')
    if state['best_check'] is not None:
        name = f"selected/check_{state['best_check']:02d}.pt"
        if (not 1<=state['best_check']<=state['validated_checks'] or state['best_file']!=name
                or file_sha256(run/name)!=state['best_sha256']):
            raise ValueError('Committed best candidate is missing or changed')
    elif state['validated_checks']!=0:
        raise ValueError('Validated progress lacks a selected checkpoint')
    complete = (run/'training_complete.json').exists()
    if complete:
        finished = audit_training(output,spec)['training']
        if any(state[k]!=finished[k] for k in state):
            raise ValueError('Completion differs from latest checkpoint')
    return dict(step=state['step'],validated_checks=state['validated_checks'],
                checkpoint_sha256=file_sha256(last),complete=complete)


def prepare_chain(chain,output,segments):
    chain, output = Path(chain).resolve(),Path(output).resolve(strict=True)
    if type(segments) is not int or not 1<=segments<=POLICY['max_segments']:
        raise ValueError('Choose 1..8 segments')
    if chain.parent!=output/'segment_chains': raise ValueError('Chain must be directly under production/segment_chains')
    if (chain/'plan.json').exists(): raise FileExistsError('Chain already prepared')
    req,_ = request(output)
    if (output/'campaign.json').exists():
        initial = snapshot(output,load_campaign(output))
    else:
        initial = dict(step=0,validated_checks=0,checkpoint_sha256=None,complete=False)
    plan = hashed(dict(schema=SCHEMA,campaign=str(output),request_sha256=req['content_hash'],
                       source=source_manifest(),segments=segments,initial=initial))
    write_json(chain/'plan.json',plan)
    return plan


def load_chain(chain, *, check_environment=True):
    chain = Path(chain).resolve(strict=True)
    plan = checked_hash(read_json(chain/'plan.json'))
    output = Path(plan['campaign']).resolve(strict=True)
    req,_ = request(output,check_environment=check_environment)
    if (plan['schema']!=SCHEMA or plan['source']!=source_manifest()
            or plan['request_sha256']!=req['content_hash'] or chain.parent!=output/'segment_chains'
            or type(plan['segments']) is not int or not 1<=plan['segments']<=POLICY['max_segments']):
        raise ValueError('Segment chain changed')
    return plan


def admit(chain,plan):
    output = Path(plan['campaign'])
    with file_lock(output/'.admission.lock',blocking=False):
        spec = expected_spec(output)
        if (output/'campaign.json').exists() and read_json(output/'campaign.json')!=spec:
            raise ValueError('Existing production identity differs; nothing overwritten')
        cursor = snapshot(output,spec)
        if cursor!=plan['initial']: raise ValueError('Progress changed since submission')
        remaining = spec['recommended_train_hours']*(1-cursor['step']/spec['recipes'][TASK['recipe']]['total_steps'])
        if remaining>plan['segments']*71:
            raise ValueError('Measured runtime exceeds queued slots; review timing and submit a longer chain')
        B.verify_inventory_metadata(spec)
        if not (output/'campaign.json').exists(): write_json(output/'campaign.json',spec)
        ready = hashed(dict(chain_sha256=plan['content_hash'],campaign_sha256=spec['content_hash']))
        path = Path(chain)/'admitted.json'
        if path.exists() and read_json(path)!=ready: raise ValueError('Admission changed')
        if not path.exists(): write_json(path,ready)
        return dict(passed=True,recommended_train_hours=spec['recommended_train_hours'],segments=plan['segments'])


def require_admission(chain,plan,spec):
    row = checked_hash(read_json(Path(chain)/'admitted.json'))
    if row!=hashed(dict(chain_sha256=plan['content_hash'],campaign_sha256=spec['content_hash'])):
        raise ValueError('Missing/changed chain admission')


def receipt_path(chain,index):
    return Path(chain)/'receipts'/f'segment_{index:02d}.json'


def previous_snapshot(chain,plan,index):
    if type(index) is not int or not 0<=index<plan['segments']: raise ValueError('Invalid segment index')
    if receipt_path(chain,index).exists(): raise FileExistsError('Segment already committed; use a new continuation chain')
    if index==0: return plan['initial']
    row = checked_hash(read_json(receipt_path(chain,index-1)))
    if (row['chain_sha256']!=plan['content_hash'] or row['index']!=index-1
            or row['status'] not in ('paused','complete','already_complete')
            or row['after']['complete']!=(row['status']!='paused')):
        raise ValueError('Invalid previous handoff')
    return row['after']


def completion_gate(chain,plan,spec):
    require_admission(chain,plan,spec)
    row = checked_hash(read_json(receipt_path(chain,plan['segments']-1)))
    current = snapshot(plan['campaign'],spec)
    if (row['chain_sha256']!=plan['content_hash'] or row['index']!=plan['segments']-1
            or row['status'] not in ('complete','already_complete')
            or not current['complete'] or row['after']!=current):
        raise ValueError('Full schedule incomplete; retained last.pt can continue in a new chain')
    return current
