#!/usr/bin/env python3
"""Plan/submit seven individual full-schedule GH200 jobs with measured walltimes."""
import argparse
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from part_selected_production import protocol as P

ROOT = Path(__file__).resolve().parents[1]


def command(argv, **kwargs):
    return subprocess.check_output(argv, text=True, **kwargs).strip()


def environment(python, source):
    env = {k:v for k,v in os.environ.items() if not k.startswith(('SBATCH_', 'SLURM_'))}
    binary = Path(python).parent
    env.update(PYTHONPATH=str(source), PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
        PYTHONUNBUFFERED='1', PATH=str(binary)+os.pathsep+env.get('PATH',''),
        LD_LIBRARY_PATH=str(binary.parent/'lib')+os.pathsep+env.get('LD_LIBRARY_PATH',''),
        OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='4', MAX_JOBS='2', WORLD_SIZE='1')
    for name in ('RANK','LOCAL_RANK','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT','CUDA_VISIBLE_DEVICES'):
        env.pop(name, None)
    return env


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('plan','submit','resume','status'), required=True)
    p.add_argument('--benchmark', type=Path)
    p.add_argument('--retry', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--concurrency', type=int, choices=range(1,8), default=3)
    p.add_argument('--indices', default='0,1,2,3,4,5,6')
    p.add_argument('--reviewed', action='store_true')
    a = p.parse_args(argv)
    try: a.indices = [int(i) for i in a.indices.split(',')]
    except ValueError: p.error('indices must be comma-separated 0..6')
    if not a.indices or len(set(a.indices)) != len(a.indices) or any(i not in range(7) for i in a.indices):
        p.error('indices must be unique, within 0..6')
    if a.mode in ('plan','submit') and (a.benchmark is None or a.retry is None):
        p.error('plan/submit require --benchmark and --retry')
    if a.mode == 'submit' and a.indices != list(range(7)):
        p.error('Fresh production contains exactly seven tasks; indices is for resume only')
    if a.mode in ('submit','resume') and not a.reviewed:
        p.error('Submission requires --reviewed after inspecting --mode plan')
    for name in ('output','benchmark','retry'):
        path = getattr(a,name)
        if path is not None:
            if not path.is_absolute(): p.error('--'+name+' must be absolute')
            setattr(a,name,path.resolve())
    for other in (a.benchmark, a.retry):
        if other is not None and (a.output == other or a.output.is_relative_to(other) or other.is_relative_to(a.output)):
            p.error('Production output must be disjoint from benchmark/retry')
    if a.output == ROOT or ROOT.is_relative_to(a.output): p.error('Unsafe output directory')
    if a.mode in ('plan','submit'):
        production_base = (ROOT/'checkpoints/part_selected_production').resolve()
        if a.output == production_base or not a.output.is_relative_to(production_base):
            p.error('Fresh output must be a new child of checkpoints/part_selected_production')
    return a


def inspect(a):
    settings = P.read(a.benchmark/'launch.json')
    return json.loads(command([settings['python'], '-B', str(ROOT/'scripts/run_part_selected_production.py'),
        '--phase','inspect','--benchmark',str(a.benchmark),'--retry',str(a.retry)],
        env=environment(settings['python'], a.benchmark/'source')))


def job_request(output, plan, phase, index=None, dependency=None):
    hours = .5 if phase == 'report' else (plan['train_hours'] if phase=='train' else plan['evaluate_hours'])[index]
    args = ['sbatch','--parsable','--job-name=sel7prod_'+phase+(f'_{index}' if index is not None else ''),
        '--account='+plan['account'],'--partition=tigris','--nodes=1','--ntasks=1',
        '--cpus-per-task=12','--mem=128G',f'--time={int(hours*60)}',
        '--output='+str(output/'logs/%x_%j.out'),'--error='+str(output/'logs/%x_%j.err'),
        '--export=ALL','--kill-on-invalid-dep=yes']
    if phase != 'report': args += ['--gres=gpu:gh200:1','--signal=B:USR1@900']
    if dependency: args += ['--dependency='+dependency]
    return args+[str(output/'source/sbatch/run_part_selected_production_tigris.sh'),str(output),phase,
                 str(index) if index is not None else '-']


def unfinished(output, plan, indices):
    result = []
    for i in indices:
        run = P.run_directory(output, plan['science']['tasks'][i])
        # Worker audits markers and hashes before trusting completion. Never
        # infer success solely from an sbatch/sacct exit code.
        train = not (run/'training_complete.json').exists()
        evaluate = not (run/'full_validation/metrics.json').exists()
        if train or evaluate: result.append((i, train, True))
    return result


def check_active(output):
    live = {line.split('_')[0] for line in command(['squeue','-h','--me','-o','%i']).splitlines()}
    ledger = output/'submitted_job_ids.tsv'
    if ledger.exists() and any(line.split()[1] in live for line in ledger.read_text().splitlines()):
        raise ValueError('This production has active/pending jobs; do not submit duplicates')
    intent = output/'submission_intent.json'
    if intent.exists() and P.read(intent).get('status') == 'unknown':
        raise ValueError('Unresolved sbatch attempt; inspect submission_intent.json and scheduler before resuming')


def remaining_storage(output, plan):
    """Budget remaining writes, not only a floor, when resuming a partial campaign."""
    storage = plan['review']['storage']
    used = sum(p.stat().st_size for p in output.rglob('*') if p.is_file() and not p.is_symlink())
    reserve = storage.get('reserve_bytes', 4*2**30)
    return reserve + max(0, storage['required_free_bytes']-reserve-used)


def dispatch(output, plan, jobs, concurrency):
    env = environment(plan['python'], Path(plan['benchmark'])/'source')
    # Slurm checks current account/QOS/partition eligibility for EVERY resource
    # shape BEFORE any job is submitted. It does not guarantee a start date.
    for index, training, _ in jobs:
        for phase in (('train','evaluate') if training else ('evaluate',)):
            request = job_request(output, plan, phase, index)
            command([request[0], '--test-only', *request[1:]], env=env, stderr=subprocess.STDOUT)
    request = job_request(output, plan, 'report')
    command([request[0], '--test-only', *request[1:]], env=env, stderr=subprocess.STDOUT)

    def submit(phase, index=None, dependency=None):
        request = job_request(output, plan, phase, index, dependency)
        P.write(output/'submission_intent.json', dict(status='unknown', request=request))
        job = command(request, env=env).split(';')[0]
        if not job.isdecimal(): raise ValueError('Ambiguous sbatch response; inspect scheduler: '+job)
        with (output/'submitted_job_ids.tsv').open('a', encoding='utf-8') as stream:
            stream.write(f'{phase}_{index}\t{job}\n'); stream.flush(); os.fsync(stream.fileno())
        P.write(output/'submission_intent.json', dict(status='recorded', job=job, request=request))
        print(f'{phase} {index if index is not None else ""} job: {job}', flush=True)
        return job

    lanes, evaluates = [], []
    for index, training, _ in jobs:
        train_id = None
        if training:
            dependency = 'afterany:'+lanes[-concurrency] if len(lanes) >= concurrency else None
            train_id = submit('train', index, dependency)
            lanes.append(train_id)
        evaluates.append(submit('evaluate', index, 'afterok:'+train_id if train_id else None))
    scheduled = {i for i, _, _ in jobs}
    absent = [i for i,t in enumerate(plan['science']['tasks']) if i not in scheduled and
              not (P.run_directory(output,t)/'full_validation/metrics.json').is_file()]
    if absent:
        print('Final report deferred; unscheduled evaluations missing for:', absent)
    else:
        submit('report', dependency='afterok:'+':'.join(evaluates) if evaluates else None)


def freeze(a, evidence):
    if a.output.exists(): raise ValueError('Output exists; use explicit --mode resume, never overwrite')
    if command(['git','-C',str(ROOT),'status','--porcelain','--',*P.FILES]):
        raise ValueError('Commit/push/pull production source before submission')
    commit = command(['git','-C',str(ROOT),'rev-parse','HEAD'])
    blobs = {n:subprocess.check_output(['git','-C',str(ROOT),'show',f'{commit}:{n}']) for n in P.FILES}
    a.output.mkdir(parents=True)
    (a.output/'logs').mkdir()
    for name, data in blobs.items():
        path = a.output/'source'/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(data)
    plan = P.hashed(dict(schema=P.SCHEMA, authorizes_training=True,
        output=str(a.output), benchmark=str(a.benchmark), retry=str(a.retry),
        policy=P.POLICY, train_hours=list(P.TRAIN_HOURS), evaluate_hours=list(P.EVALUATE_HOURS),
        source={n:P.sha(a.output/'source'/n) for n in P.FILES}, commit=commit, **evidence))
    P.write(a.output/'production.json', plan)
    (a.output/'snapshot.sha256').write_text(''.join(
        f'{P.sha(a.output/name)}  {name}\n' for name in ['production.json', *('source/'+n for n in P.FILES)]),encoding='utf-8')
    return plan


def launch(a):
    if a.mode in ('plan','submit'):
        evidence = inspect(a)
        print('Full schedule: 1,000,000 updates, 50 validations, seven seed-101 models.', flush=True)
        for i, row in enumerate(evidence['review']['models']):
            print(i,row['task']['run_id'],f'train={P.TRAIN_HOURS[i]}h evaluate={P.EVALUATE_HOURS[i]}h')
        print('One GH200, 12 CPUs, 128 GiB per GPU job; maximum simultaneous trainings:',a.concurrency)
        if a.mode == 'plan':
            print('Required free storage GiB:',round(evidence['review']['storage']['required_free_bytes']/2**30,2))
            print('Read-only plan. No files or jobs changed. Slurm eligibility is checked at submission.')
            return
    else:
        plan = P.checked(a.output/'production.json')
        if a.mode == 'status':
            print(command([plan['python'],'-B',str(a.output/'source/scripts/run_part_selected_production.py'),
                '--phase','status','--output',str(a.output)],env=environment(plan['python'],Path(plan['benchmark'])/'source')))
            return
    if platform.machine() != 'aarch64': raise ValueError('Submit on Tigris, not SPORC/local Windows')
    for name in ('git','sbatch','squeue','scontrol'):
        if shutil.which(name) is None: raise ValueError('Missing '+name)
    import fcntl
    # One production owner for this benchmark, even if --output is changed.
    registry = ROOT/'checkpoints/part_selected_production'
    registry.mkdir(parents=True, exist_ok=True)
    with (registry/'.submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = evidence['science']['content_hash'] if a.mode=='submit' else plan['science']['content_hash']
        owner = registry/(identity+'.owner.json')
        if owner.exists() and P.read(owner)['output'] != str(a.output):
            raise ValueError('Benchmark already has a production owner: '+P.read(owner)['output'])
        if a.mode == 'submit':
            existing = a.output.parent
            while not existing.exists(): existing = existing.parent
            if shutil.disk_usage(existing).free < evidence['review']['storage']['required_free_bytes']:
                raise OSError('Insufficient production headroom; nothing deleted')
            if not re.fullmatch(r'[A-Za-z0-9_-]+',evidence['account']): raise ValueError('Invalid account')
            plan = freeze(a,evidence)
            P.write(owner,dict(output=str(a.output),campaign_sha256=plan['content_hash']))
        check_active(a.output)
        # Fresh-process validation avoids lazy optimizer imports changing the
        # environment inventory; checks all source and evidence before submission.
        command([plan['python'],'-B',str(a.output/'source/scripts/run_part_selected_production.py'),
                 '--phase','verify','--output',str(a.output)],
                 env=environment(plan['python'],Path(plan['benchmark'])/'source'))
        if shutil.disk_usage(a.output).free < remaining_storage(a.output,plan):
            raise OSError('Insufficient headroom for remaining production outputs and 4 GiB reserve')
        dispatch(a.output,plan,unfinished(a.output,plan,a.indices),a.concurrency)
        print('Production:',a.output)
        print('Original benchmarks/controls/data unchanged. No test evaluation queued.')


if __name__ == '__main__': launch(parse())
