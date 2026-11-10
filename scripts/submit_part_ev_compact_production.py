#!/usr/bin/env python3
"""Plan/submit independent compact State-EV production using passed benchmarks."""
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
from part_ev_compact_production import protocol as P

ROOT=Path(__file__).resolve().parents[1]


def command(argv,**kwargs):
    return subprocess.check_output([str(x) for x in argv],text=True,**kwargs).strip()


def environment(python,source):
    env={k:v for k,v in os.environ.items() if not k.startswith(('SBATCH_','SLURM_'))}
    binary=Path(python).parent
    env.update(PYTHONPATH=str(source),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
        PATH=str(binary)+os.pathsep+env.get('PATH',''),
        LD_LIBRARY_PATH=str(binary.parent/'lib')+os.pathsep+env.get('LD_LIBRARY_PATH',''),
        OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4',MAX_JOBS='2',WORLD_SIZE='1')
    for name in ('RANK','LOCAL_RANK','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT','CUDA_VISIBLE_DEVICES'):
        env.pop(name,None)
    return env


def parse(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('plan','submit','resume','status'),required=True)
    p.add_argument('--benchmark',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--indices',default='0,1',help='Resume only: comma-separated task indices')
    p.add_argument('--reviewed',action='store_true')
    a=p.parse_args(argv)
    try: a.indices=[int(i) for i in a.indices.split(',')]
    except ValueError: p.error('indices must be 0,1')
    if not a.indices or len(set(a.indices))!=len(a.indices) or any(i not in (0,1) for i in a.indices):
        p.error('indices must be unique 0,1')
    if a.mode in ('plan','submit') and a.benchmark is None: p.error('benchmark required')
    if a.mode in ('plan','submit') and a.indices!=[0,1]: p.error('Fresh campaign requires both tasks')
    if a.mode in ('submit','resume') and not a.reviewed: p.error('Inspect --mode plan, then use --reviewed')
    for key in ('output','benchmark'):
        path=getattr(a,key)
        if path is not None:
            if not path.is_absolute(): p.error('Use absolute paths')
            if path.absolute()!=path.resolve(): p.error('No symlink/path traversal in campaign paths')
            setattr(a,key,path.resolve())
    base=ROOT/'checkpoints/part_ev_compact_production'
    if a.output==base or not a.output.is_relative_to(base): p.error('Output must be a child of '+str(base))
    if a.benchmark and (a.output==a.benchmark or a.output.is_relative_to(a.benchmark)
                        or a.benchmark.is_relative_to(a.output)):
        p.error('Production and benchmark must be disjoint')
    return a


def inspect(a):
    python=P.read(a.benchmark/'launch.json')['python']
    return json.loads(command([python,'-B',ROOT/'scripts/run_part_ev_compact_production.py',
        '--phase','inspect','--benchmark',a.benchmark],env=environment(python,a.benchmark/'source')))


def worker_query(output,plan,phase):
    return json.loads(command([plan['python'],'-B',output/'source/scripts/run_part_ev_compact_production.py',
        '--phase',phase,'--output',output],env=environment(plan['python'],Path(plan['benchmark'])/'source')))


def request(output,plan,phase,index=None,dependency=None):
    hours=P.REPORT_HOURS if phase=='report' else plan['train_hours' if phase=='train' else 'evaluate_hours'][index]
    row=['sbatch','--parsable','--job-name=evcompactprod_'+phase+(f'_{index}' if index is not None else ''),
        '--partition=tigris','--account='+plan['account'],'--nodes=1','--ntasks=1',
        '--cpus-per-task=12','--mem=128G',f'--time={int(hours*60)}',
        '--output='+str(output/'logs/%x_%j.out'),'--error='+str(output/'logs/%x_%j.err'),
        '--export=ALL','--kill-on-invalid-dep=yes']
    if phase!='report': row+=['--gres=gpu:gh200:1','--signal=B:USR1@900']
    if dependency: row+=['--dependency='+dependency]
    return row+[str(output/'source/sbatch/run_part_ev_compact_production_tigris.sh'),str(output),phase,
                '-' if index is None else str(index)]


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
    P.write(a.output/'production.json',plan)
    (a.output/'snapshot.sha256').write_text(''.join(f'{P.sha(a.output/n)}  {n}\n'
        for n in ['production.json',*('source/'+n for n in P.FILES)]),encoding='utf-8')
    return plan


def dispatch(output,plan,jobs,*,report_ready):
    env=environment(plan['python'],Path(plan['benchmark'])/'source')
    # Validate EVERY resource shape before accepting even the first job.
    for index,training,_ in jobs:
        for phase in (('train','evaluate') if training else ('evaluate',)):
            row=request(output,plan,phase,index)
            command([row[0],'--test-only',*row[1:]],env=env,stderr=subprocess.STDOUT)
    if report_ready:
        row=request(output,plan,'report')
        command([row[0],'--test-only',*row[1:]],env=env,stderr=subprocess.STDOUT)
    def submit(phase,index=None,dependency=None):
        row=request(output,plan,phase,index,dependency)
        P.write(output/'submission_intent.json',dict(status='unknown',request=row))
        job=command(row,env=env).split(';')[0]
        if not job.isdecimal(): raise ValueError('Uncertain sbatch response; inspect scheduler and intent')
        with (output/'submitted_job_ids.tsv').open('a',encoding='utf-8') as stream:
            stream.write(f'{phase}_{index}\t{job}\n'); stream.flush(); os.fsync(stream.fileno())
        P.write(output/'submission_intent.json',dict(status='recorded',job=job,request=row))
        print(f'{phase} {index if index is not None else ""} job: {job}',flush=True)
        return job
    evaluations=[]
    for index,training,_ in jobs:
        train=submit('train',index) if training else None  # No cross-arm dependency or concurrency cap.
        evaluations.append(submit('evaluate',index,'afterok:'+train if train else None))
    if report_ready:
        submit('report',dependency='afterok:'+':'.join(evaluations) if evaluations else None)
    else: print('Report deferred until both evaluations are available or scheduled.')


def launch(a):
    if a.mode in ('plan','submit'):
        evidence=inspect(a)
        for i,task in enumerate(evidence['science']['tasks']):
            recipe=evidence['science']['recipes'][task['recipe']]
            print(i,task['run_id'],f"updates={recipe['total_steps']:,} checks=50 steps/check={recipe['steps_per_epoch']:,}",
                f"train={evidence['train_hours'][i]}h evaluate={evidence['evaluate_hours'][i]}h")
        print('One GH200 / 12 CPUs / 128 GiB each; both training jobs independently eligible.')
        print('Required free GiB:',round(evidence['storage']['required_free_bytes']/2**30,2))
        if a.mode=='plan':
            print('Read-only. No files or jobs changed. Eligibility checked by Slurm at submission.'); return
    else:
        plan=P.checked(a.output/'production.json')
        if a.mode=='status': print(json.dumps(worker_query(a.output,plan,'status'),indent=2)); return
    if platform.machine()!='aarch64': raise ValueError('Submit on Tigris, not SPORC/local Windows')
    for name in ('git','sbatch','squeue','scontrol'):
        if shutil.which(name) is None: raise ValueError('Missing '+name)
    import fcntl
    registry=ROOT/'checkpoints/part_ev_compact_production'; registry.mkdir(parents=True,exist_ok=True)
    with (registry/'.submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        identity=evidence['benchmark_spec']['content_hash'] if a.mode=='submit' else plan['benchmark_spec']['content_hash']
        owner=registry/(identity+'.owner.json')
        if owner.exists() and P.read(owner)['output']!=str(a.output):
            raise ValueError('Benchmark already has production owner: '+P.read(owner)['output'])
        if a.mode=='submit':
            if shutil.disk_usage(registry).free < evidence['storage']['required_free_bytes']:
                raise OSError('Insufficient production headroom; nothing deleted')
            if not re.fullmatch(r'[A-Za-z0-9_-]+',evidence['account']): raise ValueError('Invalid account')
            plan=freeze(a,evidence)
            P.write(owner,dict(output=str(a.output),campaign_sha256=plan['content_hash']))
        check_active(a.output)
        worker_query(a.output,plan,'verify')
        if shutil.disk_usage(a.output).free < remaining_storage(a.output,plan):
            raise OSError('Insufficient remaining-output headroom including 4 GiB reserve')
        all_jobs=worker_query(a.output,plan,'ready')
        jobs=[row for row in all_jobs if row[0] in a.indices]
        report_ready=not any(row[0] not in a.indices for row in all_jobs)
        dispatch(a.output,plan,jobs,report_ready=report_ready)
        print('Production:',a.output)
        print('Existing campaigns, benchmark artifacts, environments and data unchanged. No test jobs queued.')


if __name__=='__main__': launch(parse())
