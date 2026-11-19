#!/usr/bin/env python3
"""Arm independent, frozen benchmark-to-production controllers on SPORC."""
import argparse
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from part_ev_benchmark_followup import protocol as F
from part_ev_state_production import protocol as P
from scripts import submit_part_ev_state_production as S

ROOT=Path(__file__).resolve().parents[1]
command=S.command


def parse(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('plan','arm','retry','status'),required=True)
    p.add_argument('--model',choices=tuple(F.MODELS))
    p.add_argument('--benchmark',type=Path)
    p.add_argument('--production',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-train-hours',type=int)
    p.add_argument('--max-evaluate-hours',type=int)
    p.add_argument('--reviewed',action='store_true')
    a=p.parse_args(argv)
    if a.mode in ('arm','retry') and not a.reviewed:
        p.error('Automatic production is opt-in: inspect plan, then pass --reviewed')
    if a.mode in ('plan','arm'):
        if not all((a.model,a.benchmark,a.production)):
            p.error('Fresh follow-up requires --model, --benchmark and --production')
        a.max_train_hours=480 if a.max_train_hours is None else a.max_train_hours
        a.max_evaluate_hours=24 if a.max_evaluate_hours is None else a.max_evaluate_hours
        if min(a.max_train_hours,a.max_evaluate_hours)<1:p.error('Positive resource ceilings required')
        F.validate_paths(ROOT,a.output,a.benchmark,a.production,a.model)
    elif any(x is not None for x in (a.model,a.benchmark,a.production,a.max_train_hours,a.max_evaluate_hours)):
        p.error('Retry/status use only frozen --output; no new resource or model choices')
    for name in ('output','benchmark','production'):
        if getattr(a,name) is not None:setattr(a,name,F.canonical(getattr(a,name)))
    return a


def proposal(a):
    return dict(schema=F.SCHEMA,repo=str(ROOT),output=str(a.output),model_kind=a.model,
        benchmark=str(a.benchmark),production=str(a.production),reviewed=True,policy=F.POLICY,
        benchmark_identity=F.benchmark_identity(a.benchmark,a.model),
        max_train_hours=a.max_train_hours,max_evaluate_hours=a.max_evaluate_hours)


def inspect_completed(arm):
    # Authenticate completed evidence in a clean interpreter, including when
    # Slurm has already aged the summary job out of its active records.
    evidence=S.inspect(SimpleNamespace(benchmark=Path(arm['benchmark']),model=arm['model_kind']))
    F.admit(arm,evidence)


def dependency(arm):
    job=arm['benchmark_identity']['summary_job']
    try:
        text=command(['scontrol','show','job',job,'-o'],stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError:
        inspect_completed(arm)
        return None
    fields=dict(token.split('=',1) for token in text.split() if '=' in token)
    prefix=F.MODELS[arm['model_kind']][1]
    user=re.fullmatch(r'[^()]+\((\d+)\)',fields.get('UserId',''))
    expected_log=str(Path(arm['benchmark'])/'logs'/f'{prefix}_summary_{job}.out')
    if (fields.get('JobId')!=job or fields.get('JobName')!=prefix+'_summary'
            or fields.get('Partition')!='tier3' or not user or int(user[1])!=os.getuid()
            or fields.get('StdOut')!=expected_log):
        raise ValueError('Summary job identity does not match benchmark ledger/current user')
    state=fields.get('JobState')
    if state=='COMPLETED' and fields.get('ExitCode')=='0:0':
        inspect_completed(arm)
        return None
    if state not in ('PENDING','RUNNING','CONFIGURING','COMPLETING','SUSPENDED'):
        raise ValueError('Summary did not succeed; do not arm automatic production')
    return 'afterok:'+job


def request(output,arm,dep):
    row=['sbatch','--parsable','--job-name=evfollow_'+arm['model_kind'],
        '--partition=tier3','--account='+arm['benchmark_identity']['account'],
        '--nodes=1','--ntasks=1','--cpus-per-task=4','--mem=16G','--time=120',
        '--output='+str(output/'logs/%x_%j.out'),'--error='+str(output/'logs/%x_%j.err'),
        '--export=ALL','--kill-on-invalid-dep=yes']
    if dep:row+=['--dependency='+dep]
    return row+[str(output/'source/sbatch/run_part_ev_benchmark_followup.sh'),str(output)]


def freeze(output,arm):
    if output.exists():raise FileExistsError('Follow-up exists; inspect it, do not overwrite')
    if command(['git','-C',ROOT,'status','--porcelain','--',*F.FILES]):
        raise ValueError('Commit/push/pull follow-up and production files before arming')
    commit=command(['git','-C',ROOT,'rev-parse','HEAD'])
    blobs={n:subprocess.check_output(['git','-C',str(ROOT),'show',f'{commit}:{n}']) for n in F.FILES}
    output.mkdir();(output/'logs').mkdir()
    for name,blob in blobs.items():
        path=output/'source'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(blob)
    arm=P.hashed(dict(arm,commit=commit,source=P.inventory(output/'source')))
    P.write(output/'followup.json',arm)
    (output/'snapshot.sha256').write_text(''.join(f'{P.sha(output/n)}  {n}\n' for n in
        ['followup.json',*('source/'+n for n in F.FILES)]),encoding='utf-8')
    F.load(output)
    return arm


def retry_allowed(output):
    intent=output/'submission_intent.json'
    if not intent.is_file() or P.read(intent).get('status')!='recorded':
        raise ValueError('Unresolved controller submission; reconcile journal and Slurm first')
    live=set(command(['squeue','--me','-h','-o','%i']).splitlines())
    if any(line.split()[1] in live for line in (output/'submitted_job_ids.tsv').read_text().splitlines()):
        raise ValueError('Controller still active/pending; do not duplicate')
    result=P.read(output/'result.json') if (output/'result.json').exists() else {}
    if result.get('status') in ('dispatch_started','needs_reconciliation','submitted'):
        raise ValueError('Controller reached production handoff; inspect production instead of retrying')


def launch(a):
    if a.mode=='status':print(json.dumps(F.status(a.output),indent=2));return
    arm=proposal(a) if a.mode in ('plan','arm') else F.load(a.output)
    if arm['repo']!=str(ROOT):raise ValueError('Use the original repository for follow-up management')
    print(arm['model_kind']+': after verified benchmark PASS, automatically submit fresh full production.')
    print('Seed 101; 1,000,000 updates / 50 validations; train 2 A100s, evaluate 1 A100; no test data.')
    print(f"GPU-job RAM: {P.host_memory_gib(arm['model_kind'],'train')} GiB. Independent train -> evaluate -> report.")
    print('Summary job:',arm['benchmark_identity']['summary_job'],'| Production:',arm['production'])
    print('Walltime ceilings (not requested times):',arm['max_train_hours'],'h train;',arm['max_evaluate_hours'],'h evaluation.')
    print('Actual requests use passed benchmark recommendations and must fit live tier3 limits.')
    if a.mode=='plan':
        print('Read-only. Arming queues one 2h / 4 CPU / 16 GiB controller; no GPU jobs yet.');return
    if platform.system()!='Linux' or platform.machine()!='x86_64':
        raise ValueError('Arm on SPORC x86_64, not Tigris/Windows')
    for name in ('git','sbatch','squeue','scontrol'):
        if shutil.which(name) is None:raise ValueError('Missing '+name)
    import fcntl
    registry=ROOT/'checkpoints/part_ev_benchmark_followup';registry.mkdir(parents=True,exist_ok=True)
    with (registry/'.submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        owner=registry/(arm['benchmark_identity']['launch_sha256']+'.owner.json')
        if owner.exists() and P.read(owner)['output']!=str(a.output):
            raise ValueError('Benchmark already has an armed controller: '+P.read(owner)['output'])
        for path in registry.glob('*.owner.json'):
            other=P.read(path)
            if other['production']==arm['production'] and other['output']!=str(a.output):
                raise ValueError('Production target already belongs to another controller')
        if Path(arm['production']).exists():raise FileExistsError('Production output already exists')
        if a.mode=='retry':
            if not owner.exists() or P.read(owner)['followup_sha256']!=arm['content_hash']:
                raise ValueError('Follow-up owner changed')
            retry_allowed(a.output)
        elif a.output.exists():raise FileExistsError('Follow-up output exists; inspect status')
        dep=dependency(arm)
        # Controller needs little space; production headroom is rechecked under
        # the shared production lock at dispatch, once measurements exist.
        if shutil.disk_usage(registry).free<4*2**30+64*2**20:
            raise OSError('Need 4 GiB reserve plus 64 MiB controller headroom; nothing deleted')
        env=S.environment(arm['benchmark_identity']['python'],a.output/'source')
        row=request(a.output,arm,dep)
        if a.mode=='arm':row[-2]=str(ROOT/'sbatch/run_part_ev_benchmark_followup.sh')
        command([row[0],'--test-only',*row[1:]],env=env,stderr=subprocess.STDOUT)
        if a.mode=='arm':
            arm=freeze(a.output,arm)
            P.write(owner,dict(output=str(a.output),production=arm['production'],followup_sha256=arm['content_hash']))
        row=request(a.output,arm,dep)
        P.write(a.output/'submission_intent.json',dict(status='unknown',request=row))
        job=command(row,env=env).split(';')[0]
        if not job.isdecimal():raise ValueError('Uncertain sbatch response; reconcile journal and Slurm')
        with (a.output/'submitted_job_ids.tsv').open('a',encoding='utf-8') as handle:
            handle.write(f'controller\t{job}\n');handle.flush();os.fsync(handle.fileno())
        P.write(a.output/'submission_intent.json',dict(status='recorded',job=job,request=row))
        print('Controller job:',job,'| Follow-up:',a.output)
        print('Existing benchmarks/jobs unchanged. Production admission will recheck evidence, storage and resources.')


if __name__=='__main__':launch(parse())
