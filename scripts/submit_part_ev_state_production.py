#!/usr/bin/env python3
"""Plan/submit independent state-extension production using passed benchmarks."""
import argparse
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from part_ev_state_production import protocol as P

ROOT=Path(__file__).resolve().parents[1]


def command(argv,**kwargs):
    return subprocess.check_output([str(x) for x in argv],text=True,**kwargs).strip()


def environment(python,source):
    env={k:v for k,v in os.environ.items() if not k.startswith(('SBATCH_','SLURM_'))}
    binary=Path(python).parent
    env.update(PYTHONPATH=str(source),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
        PATH=str(binary)+os.pathsep+env.get('PATH',''),
        LD_LIBRARY_PATH=str(binary.parent/'lib')+os.pathsep+env.get('LD_LIBRARY_PATH',''),
        OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2',MAX_JOBS='2',WORLD_SIZE='1')
    for name in ('RANK','LOCAL_RANK','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT','CUDA_VISIBLE_DEVICES'):
        env.pop(name,None)
    return env


def parse(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('plan','submit','resume','status'),required=True)
    p.add_argument('--benchmark',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',choices=tuple(P.PACKAGES))
    p.add_argument('--reviewed',action='store_true')
    a=p.parse_args(argv)
    if a.mode in ('plan','submit') and (a.benchmark is None or a.model is None):
        p.error('Fresh campaign needs --model and --benchmark')
    if a.mode in ('resume','status') and (a.benchmark is not None or a.model is not None):
        p.error('Resume/status use only the frozen --output campaign')
    if a.mode in ('submit','resume') and not a.reviewed: p.error('Inspect --mode plan, then use --reviewed')
    for key in ('output','benchmark'):
        path=getattr(a,key)
        if path is not None:
            if not path.is_absolute(): p.error('Use absolute paths')
            if path.absolute()!=path.resolve(): p.error('No symlink/path traversal in campaign paths')
            setattr(a,key,path.resolve())
    base=ROOT/'checkpoints/part_ev_state_production'
    if a.output==base or not a.output.is_relative_to(base): p.error('Output must be a child of '+str(base))
    if a.benchmark and (a.output==a.benchmark or a.output.is_relative_to(a.benchmark)
                        or a.benchmark.is_relative_to(a.output)):
        p.error('Production and benchmark must be disjoint')
    return a


def inspect(a):
    python=P.read(a.benchmark/'launch.json')['python']
    return json.loads(command([python,'-B',ROOT/'scripts/run_part_ev_state_production.py',
        '--phase','inspect','--benchmark',a.benchmark,'--model',a.model],env=environment(python,a.benchmark/'source')))


def worker_query(output,plan,phase):
    return json.loads(command([plan['python'],'-B',output/'source/scripts/run_part_ev_state_production.py',
        '--phase',phase,'--output',output],env=environment(plan['python'],Path(plan['benchmark'])/'source')))


def request(output,plan,phase,dependency=None):
    hours=P.REPORT_HOURS if phase=='report' else plan['train_hours' if phase=='train' else 'evaluate_hours']
    row=['sbatch','--parsable','--job-name=evstateprod_'+plan['model_kind']+'_'+phase,
        '--partition=tier3','--account='+plan['account'],'--nodes=1','--ntasks=1',
        '--cpus-per-task=8','--mem=64G',f'--time={int(hours*60)}',
        '--output='+str(output/'logs/%x_%j.out'),'--error='+str(output/'logs/%x_%j.err'),
        '--export=ALL','--kill-on-invalid-dep=yes']
    if phase!='report':
        row+=['--gres=gpu:a100:'+('2' if phase=='train' else '1'),'--signal=B:USR1@900']
    if dependency: row+=['--dependency='+dependency]
    return row+[str(output/'source/sbatch/run_part_ev_state_production.sh'),str(output),phase]


def check_partition(plan):
    row=command(['scontrol','show','partition','tier3','-o'])
    fields=dict(item.split('=',1) for item in row.split() if '=' in item)
    if fields.get('State')!='UP': raise ValueError('tier3 is not UP')
    limit=fields.get('MaxTime','')
    if limit=='UNLIMITED': return
    match=re.fullmatch(r'(?:(\d+)-)?(\d+):(\d+):(\d+)',limit)
    if not match: raise ValueError('Cannot establish tier3 maximum walltime')
    days,hours,minutes,seconds=[int(x or 0) for x in match.groups()]
    available=days*86400+hours*3600+minutes*60+seconds
    if max(plan['train_hours'],plan['evaluate_hours'],P.REPORT_HOURS)*3600>available:
        raise ValueError('Measured walltime exceeds live tier3 maximum; no jobs submitted')


def check_active(output):
    live={line.split('_')[0].strip() for line in command(['squeue','--me','-h','-o','%i']).splitlines()}
    ledger=output/'submitted_job_ids.tsv'
    if ledger.exists() and any(line.split()[1] in live for line in ledger.read_text().splitlines()):
        raise ValueError('Production has active/pending jobs; do not submit duplicates')
    intent=output/'submission_intent.json'
    if intent.exists() and P.read(intent)['status']=='unknown':
        raise ValueError('Unresolved sbatch attempt; inspect intent and scheduler before retrying')


def remaining_storage(output,plan):
    used=sum(p.stat().st_size for p in output.rglob('*') if p.is_file() and not p.is_symlink())
    storage=plan['storage']; reserve=storage['reserve_bytes']
    return reserve+max(0,storage['required_free_bytes']-reserve-used)


def freeze(a,evidence):
    if a.output.exists(): raise ValueError('Output exists; use explicit resume, never overwrite')
    if command(['git','-C',ROOT,'status','--porcelain','--',*P.FILES]):
        raise ValueError('Commit/push/pull production files before submission')
    commit=command(['git','-C',ROOT,'rev-parse','HEAD'])
    blobs={n:subprocess.check_output(['git','-C',str(ROOT),'show',f'{commit}:{n}']) for n in P.FILES}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.mkdir(); (a.output/'logs').mkdir()
    for name,blob in blobs.items():
        path=a.output/'source'/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(blob)
    plan=P.hashed(dict(schema=P.SCHEMA,authorizes_training=True,output=str(a.output),policy=P.POLICY,
        source={n:P.sha(a.output/'source'/n) for n in P.FILES},commit=commit,**evidence))
    P.write(a.output/'campaign.json',plan)
    (a.output/'snapshot.sha256').write_text(''.join(f'{P.sha(a.output/n)}  {n}\n'
        for n in ['campaign.json',*('source/'+n for n in P.FILES)]),encoding='utf-8')
    return plan


def dispatch(output,plan,phases):
    env=environment(plan['python'],Path(plan['benchmark'])/'source')
    # All resource requests are tested before the first actual submission.
    check_partition(plan)
    for phase in phases:
        row=request(output,plan,phase)
        command([row[0],'--test-only',*row[1:]],env=env,stderr=subprocess.STDOUT)
    previous=None
    for phase in phases:
        row=request(output,plan,phase,'afterok:'+previous if previous else None)
        P.write(output/'submission_intent.json',dict(status='unknown',request=row))
        job=command(row,env=env).split(';')[0]
        if not job.isdecimal(): raise ValueError('Uncertain sbatch response; inspect scheduler and intent')
        with (output/'submitted_job_ids.tsv').open('a',encoding='utf-8') as stream:
            stream.write(f'{phase}\t{job}\n'); stream.flush(); os.fsync(stream.fileno())
        P.write(output/'submission_intent.json',dict(status='recorded',job=job,request=row))
        print(f'{phase} job: {job}',flush=True)
        previous=job


def aggregate_storage(registry, output, plan):
    # Account for remaining outputs of other campaigns created by THIS launcher.
    # This is not a reservation against unrelated jobs consuming the same filesystem.
    reserve=plan['storage']['reserve_bytes']
    required=remaining_storage(output,plan)-reserve
    for owner in registry.glob('*.owner.json'):
        other=Path(P.read(owner)['output'])
        if other==output: continue
        if not other.is_relative_to(registry):
            raise ValueError('Unexpected owner path')
        saved=P.checked(other/'campaign.json')
        if not (other/'reports/production_report.json').exists():
            required+=remaining_storage(other,saved)-saved['storage']['reserve_bytes']
    return required+reserve


def launch(a):
    if a.mode in ('plan','submit'):
        evidence=inspect(a)
        task=evidence['task'];recipe=evidence['recipes'][task['recipe']]
        print(task['run_id'],f"seed={task['seed']} updates={recipe['total_steps']:,} checks=50",
              f"train={evidence['train_hours']}h evaluate={evidence['evaluate_hours']}h report=1h")
        print('Train: two same-node A100s. Evaluate: one A100. 8 CPUs / 64 GiB per job.')
        print('Required free GiB:',round(evidence['storage']['required_free_bytes']/2**30,2))
        if a.mode=='plan':
            print('Read-only. No files/jobs changed. Submission also checks other registered output budgets and Slurm.'); return
    else:
        plan=P.checked(a.output/'campaign.json')
        if a.mode=='status': print(json.dumps(worker_query(a.output,plan,'status'),indent=2)); return
    if platform.system()!='Linux' or platform.machine()!='x86_64':
        raise ValueError('Submit on SPORC, not Tigris/local Windows')
    for name in ('git','sbatch','squeue','scontrol'):
        if shutil.which(name) is None: raise ValueError('Missing '+name)
    import fcntl
    registry=ROOT/'checkpoints/part_ev_state_production'; registry.mkdir(parents=True,exist_ok=True)
    with (registry/'.submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        identity=evidence['benchmark_sha256'] if a.mode=='submit' else plan['benchmark_sha256']
        owner=registry/(identity+'.owner.json')
        if owner.exists() and P.read(owner)['output']!=str(a.output):
            raise ValueError('Benchmark already has production owner: '+P.read(owner)['output'])
        if a.mode=='submit':
            if not re.fullmatch(r'[A-Za-z0-9_-]+',evidence['account']): raise ValueError('Invalid account')
            if shutil.disk_usage(registry).free < aggregate_storage(registry,a.output,evidence):
                raise OSError('Insufficient aggregate output headroom; nothing deleted')
            check_partition(evidence)
            plan=freeze(a,evidence)
            P.write(owner,dict(output=str(a.output),campaign_sha256=plan['content_hash']))
        elif not owner.exists() or P.read(owner)['campaign_sha256']!=plan['content_hash']:
            raise ValueError('Missing/changed production owner; reconcile before resubmitting')
        check_active(a.output)
        worker_query(a.output,plan,'verify')
        if shutil.disk_usage(a.output).free < aggregate_storage(registry,a.output,plan):
            raise OSError('Insufficient aggregate remaining-output headroom including 4 GiB reserve')
        phases=worker_query(a.output,plan,'ready')
        if phases not in ([],['report'],['evaluate','report'],['train','evaluate','report']):
            raise ValueError('Unexpected production continuation')
        if phases: dispatch(a.output,plan,phases)
        else: print('All artifacts audited complete. Nothing submitted.')
        print('Production:',a.output)
        print('Only this model chain submitted. Old campaigns, benchmark artifacts, environments and data unchanged.')


if __name__=='__main__': launch(parse())
