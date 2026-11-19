"""Stdlib-only intent/source authentication; no GPU work or scientific imports."""
from pathlib import Path
import re
from part_ev_state_production import protocol as P

SCHEMA='ev_benchmark_followup_v1'
MODELS={'readout':('part_ev_pair_readout','evread'),
        'triangles':('part_ev_triangles','evtri'),
        'support':('part_ev_support','evsup')}
POLICY=dict(seed=101,total_steps=1000000,validation_checks=50,global_batch=512,
    train_gpus=2,evaluate_gpus=1,partition='tier3',dependency='afterok_summary',
    full_validation_events=5000000,test_data=False,controller_minutes=120,
    controller_cpus=4,controller_memory_gib=16,automatic_retries=False)
FILES=(*P.FILES,'part_ev_benchmark_followup/__init__.py','part_ev_benchmark_followup/protocol.py',
    'scripts/submit_part_ev_benchmark_followup.py','scripts/run_part_ev_benchmark_followup.py',
    'sbatch/submit_part_ev_benchmark_followup.sh','sbatch/run_part_ev_benchmark_followup.sh')


def canonical(path):
    path=Path(path)
    if not path.is_absolute() or path.absolute()!=path.resolve():
        raise ValueError('Use absolute paths without symlinks or traversal')
    return path.resolve()


def validate_paths(repo,output,benchmark,production,kind):
    repo,output,benchmark,production=map(canonical,(repo,output,benchmark,production))
    if kind not in MODELS:raise ValueError('Unknown follow-up model')
    if output.parent!=repo/'checkpoints/part_ev_benchmark_followup':
        raise ValueError('Follow-up output must be a direct registry child')
    if production.parent!=repo/'checkpoints/part_ev_state_production':
        raise ValueError('Production must be a direct production-registry child')
    if benchmark.parent!=repo/'checkpoints'/MODELS[kind][0]:
        raise ValueError('Benchmark path does not match this model')


def benchmark_identity(benchmark,kind):
    benchmark=canonical(benchmark)
    family,_=MODELS[kind]
    launch=P.checked(benchmark/'launch.json')
    if (launch.get('schema')!=family+'_benchmark_v1' or launch.get('output')!=str(benchmark)
            or launch.get('partition')!='tier3' or launch.get('authorizes_training') is not False
            or not re.fullmatch(r'[A-Za-z0-9_-]+',launch['account'])):
        raise ValueError('Unexpected benchmark launch identity')
    manifest=P.read(benchmark/'source_manifest.json')
    if not manifest or P.inventory(benchmark/'source')!=manifest or P.digest(manifest)!=launch['source_sha256']:
        raise ValueError('Benchmark source inventory changed')
    jobs={}
    for line in (benchmark/'submitted_job_ids.tsv').read_text().splitlines():
        fields=line.split()
        if len(fields)!=2 or fields[0] in jobs or not fields[1].isdecimal():
            raise ValueError('Ambiguous benchmark submission ledger')
        jobs[fields[0]]=fields[1]
    if set(jobs)!={'prepare','benchmark','summary'} or len(set(jobs.values()))!=3:
        raise ValueError('Require the complete three-job benchmark ledger')
    return dict(launch_sha256=P.sha(benchmark/'launch.json'),summary_job=jobs['summary'],
        pinned_files={name:P.sha(benchmark/name) for name in
                      ('launch.json','source_manifest.json','submitted_job_ids.tsv')},
        python=launch['python'],account=launch['account'])


def load(output):
    output=canonical(output)
    arm=P.checked(output/'followup.json')
    if (arm.get('schema')!=SCHEMA or arm.get('policy')!=POLICY or arm.get('reviewed') is not True
            or arm.get('output')!=str(output) or set(arm['source'])!=set(FILES)
            or P.inventory(output/'source')!=arm['source']):
        raise ValueError('Frozen follow-up intent/source changed')
    validate_paths(arm['repo'],output,arm['benchmark'],arm['production'],arm['model_kind'])
    if benchmark_identity(arm['benchmark'],arm['model_kind'])!=arm['benchmark_identity']:
        raise ValueError('Benchmark binding changed after follow-up was armed')
    for field in ('max_train_hours','max_evaluate_hours'):
        if type(arm[field]) is not int or arm[field]<1:
            raise ValueError('Invalid follow-up resource ceiling')
    return arm


def admit(arm,evidence):
    """The full frozen benchmark reviewer has already recomputed all its gates."""
    P.validate_review(evidence['review'])
    task=evidence['task'];recipe=evidence['recipes'][task['recipe']]
    if (evidence['model_kind']!=arm['model_kind'] or evidence['benchmark']!=arm['benchmark']
            or evidence['python']!=arm['benchmark_identity']['python']
            or evidence['account']!=arm['benchmark_identity']['account']
            or task['seed']!=101 or recipe['total_steps']!=1000000 or recipe['epochs']!=50
            or recipe['steps_per_epoch']!=20000 or recipe['validation_events_per_epoch']!=1280000
            or recipe['global_batch_size']!=512
            or evidence['train_hours']!=evidence['review']['recommended_train_hours']
            or evidence['evaluate_hours']!=evidence['review']['recommended_evaluate_hours']):
        raise ValueError('Admission changed the approved model/seed/full schedule')
    if (evidence['train_hours']>arm['max_train_hours']
            or evidence['evaluate_hours']>arm['max_evaluate_hours']):
        raise ValueError('Measured recommendation exceeds approved walltime ceiling; nothing submitted')


def status(output):
    arm=load(output)
    output=Path(output)
    return dict(model=arm['model_kind'],summary_job=arm['benchmark_identity']['summary_job'],
        production=arm['production'],limits={k:arm[k] for k in ('max_train_hours','max_evaluate_hours')},
        submission=P.read(output/'submission_intent.json') if (output/'submission_intent.json').exists() else None,
        result=P.read(output/'result.json') if (output/'result.json').exists() else None,
        note='Read-only metadata, not a live scheduler or model/checkpoint audit.')
