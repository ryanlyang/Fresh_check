#!/usr/bin/env python3
"""Standard-library-only Slurm submitter; dry-run/status never submit jobs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
REFERENCE='/home/ryreu/atlas/Fresh_check/checkpoints/part_selected_tigris/selected_gh200_20261023T073044Z'
RELEVANT=['part_selected_fast','part_selected_tigris','part_selected_screen','part_layerwise_screen',
          'part_screen_common','part_ev_screen','part_ev_fullscale','teacher_logit_reco/relational_part',
          'scripts/run_part_ev_screen.py','scripts/run_part_selected_screen.py','scripts/run_part_selected_tigris.py',
          'scripts/run_part_selected_fast.py','scripts/profile_part_selected_fast.py','scripts/submit_part_selected_fast.py',
          'scripts/build_relational_part_tree_backend.py','scripts/setup_part_selected_tigris_env.sh',
          'sbatch/run_part_selected_screen_sporc.sh','sbatch/submit_part_selected_screen_sporc.sh',
          'sbatch/run_part_selected_tigris.sh','sbatch/submit_part_selected_tigris.sh','teacher_logit_reco/__init__.py',
          'sbatch/submit_part_selected_fast_tigris.sh','sbatch/run_part_selected_fast_tigris.sh',
          'particle_transformer/data/JetClass/JetClass_full.yaml']


def command(args,**kwargs):
    return subprocess.check_output(args,text=True,**kwargs).strip()


def parse(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('profile','benchmark','run','train','evaluate','report','status','review'),default='benchmark')
    p.add_argument('--reference',type=Path,default=Path(REFERENCE))
    p.add_argument('--output',type=Path)
    p.add_argument('--python',type=Path,default=Path(sys.executable))
    p.add_argument('--concurrency',type=int,choices=range(1,5),default=2)
    p.add_argument('--cpus',type=int,default=12)
    p.add_argument('--tree-threads',type=int,default=4)
    p.add_argument('--memory-gib',type=int,default=128)
    p.add_argument('--cache-gib',type=int,default=0)
    p.add_argument('--compile-region',action='store_true')
    p.add_argument('--train-hours',type=int,default=168)
    p.add_argument('--evaluate-hours',type=int,default=12)
    p.add_argument('--account',default='reu-aisocial')
    p.add_argument('--partition',default='tigris')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args(argv)
    if args.output is None:
        if args.resume or args.mode not in ('profile','benchmark','run'): p.error('--output required')
        args.output=ROOT/'checkpoints/part_selected_fast'/('selected_fast_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    if (not 1<=args.tree_threads<=64 or args.cpus<args.tree_threads+4 or not 8<=args.memory_gib<=400
            or not 0<=args.cache_gib<=args.memory_gib//2 or min(args.train_hours,args.evaluate_hours)<=0):
        p.error('Invalid resources: reserve four CPUs and at least half host RAM outside tree/cache work')
    if any(not re.fullmatch(r'[A-Za-z0-9_-]+',x) for x in (args.partition,args.account)): p.error('Invalid account/partition')
    for name in ('reference','output','python'):
        path=getattr(args,name)
        if not path.is_absolute(): p.error(f'--{name} must be absolute')
        setattr(args,name,path.resolve())
    if args.output==args.reference or args.output.is_relative_to(args.reference) or args.reference.is_relative_to(args.output):
        p.error('Output and reference must be disjoint')
    return args


def settings(a):
    return dict(reference=str(a.reference),python=str(a.python),cpus=str(a.cpus),tree_threads=str(a.tree_threads),
                memory_gib=str(a.memory_gib),cache_bytes=a.cache_gib*2**30,compile_region=a.compile_region,
                train_hours=str(a.train_hours),evaluate_hours=str(a.evaluate_hours),account=a.account,partition=a.partition)


def graph(mode,a):
    jobs=[]
    if mode in ('profile','benchmark','run'):
        jobs += [('prepare',5,False),('profile',5,True)]
        if mode=='profile': return jobs
        jobs += [('benchmark',5,True),('gate',2,False)]
        if mode=='benchmark': return jobs
    elif mode=='train': jobs += [('gate',2,False)]
    if mode in ('run','train'): jobs += [('train',int(a.train_hours),True)]
    if mode in ('run','train','evaluate'): jobs += [('lock',2,False),('evaluate',int(a.evaluate_hours),True)]
    if mode in ('run','train','evaluate','report'): jobs += [('report',2,False)]
    return jobs


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def verify_snapshot(output):
    output=Path(output).resolve()
    manifest=json.loads((output/'snapshot.json').read_text())
    actual={p.relative_to(output/'source').as_posix() for p in (output/'source').rglob('*')
            if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    if not manifest or set(manifest)!=actual:
        raise ValueError('Frozen source file inventory changed')
    for name,sha in manifest.items():
        path=(output/'source'/name).resolve()
        if not path.is_relative_to(output/'source') or digest(path)!=sha:
            raise ValueError('Frozen source changed: '+name)


def launch(a):
    configuration=settings(a)
    if (a.output/'launch.json').exists():
        saved=json.loads((a.output/'launch.json').read_text())
        if a.mode in ('status','review'):
            configuration=saved
        elif configuration!=saved:
            raise ValueError('Settings differ from frozen campaign; reuse the same flags or create a new output')
        verify_snapshot(a.output)
    if a.dry_run:
        print(f'Campaign: {a.output}\nRead-only reference: {a.reference}')
        print(' -> '.join(f'{phase}{"[0-3%"+str(a.concurrency)+"]" if array else ""} {hours}h' for phase,hours,array in graph(a.mode,a)))
        print(f'One GH200 per GPU task; CPUs={a.cpus}, RAM={a.memory_gib} GiB, tree threads={a.tree_threads}, cache={a.cache_gib} GiB.')
        print('No files changed or jobs submitted. Default benchmark stops before training.')
        return
    if a.mode in ('status','review'):
        if not (a.output/'launch.json').exists(): raise ValueError('Campaign missing')
        if not (a.output/'experiment_spec.json').exists():
            print('Preparation not complete; inspect prepare log and job ledger.'); return
        subprocess.run([configuration['python'],str(a.output/'source/scripts/run_part_selected_fast.py'),
                        '--phase',a.mode,'--output',str(a.output)],check=True)
        return
    import platform
    if platform.machine()!='aarch64': raise ValueError('Submit on Tigris ARM')
    for executable in ('git','sbatch','squeue','scontrol'):
        if shutil.which(executable) is None: raise ValueError('Missing '+executable)
    if not a.python.is_file() or not (a.reference/'experiment_spec.json').is_file(): raise ValueError('Missing Python/reference campaign')
    partition=command(['scontrol','show','partition',a.partition,'-o'])
    match=re.search(r'\bMaxTime=(\S+)',partition)
    if not match: raise ValueError('Cannot verify partition MaxTime')
    if match[1] not in ('UNLIMITED','INFINITE'):
        parts=match[1].split('-'); days=int(parts[0]) if len(parts)==2 else 0
        clock=[int(x) for x in parts[-1].split(':')]
        if len(clock)!=3: raise ValueError('Unrecognized partition time format')
        maximum=days*86400+clock[0]*3600+clock[1]*60+clock[2]
        if any(hours*3600>maximum for _,hours,_ in graph(a.mode,a)): raise ValueError('Requested phase exceeds partition MaxTime')
    if not a.output.exists():
        if a.resume or a.mode not in ('profile','benchmark','run'): raise ValueError('Fresh setup required')
        dirty=command(['git','-C',str(ROOT),'status','--porcelain','--',*RELEVANT])
        if dirty: raise ValueError('Commit/push/pull relevant source before submission')
        for name in ('scripts/submit_part_selected_fast.py','part_selected_fast/engine.py','part_selected_fast/csrc/tree_core.h'):
            command(['git','-C',str(ROOT),'cat-file','-e','HEAD:'+name])
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.mkdir()  # Exclusive creation; never overwrite an interrupted setup.
        (a.output/'source').mkdir(); (a.output/'logs').mkdir()
        with tempfile.TemporaryDirectory(prefix='selected-fast-freeze-',dir=a.output) as tmp:
            archive=Path(tmp)/'source.tar'
            subprocess.run(['git','-C',str(ROOT),'archive','--format=tar','--output',str(archive),'HEAD'],check=True)
            with tarfile.open(archive) as tar:
                # A trusted committed repository, but still reject path escape/links.
                for entry in tar.getmembers():
                    target=(a.output/'source'/entry.name).resolve()
                    if not target.is_relative_to(a.output/'source') or not (entry.isdir() or entry.isfile()):
                        raise ValueError('Unsafe source archive member')
                tar.extractall(a.output/'source')
        (a.output/'snapshot.json').write_text(json.dumps({p.relative_to(a.output/'source').as_posix():digest(p)
            for p in sorted((a.output/'source').rglob('*')) if p.is_file()},indent=2)+'\n')
        (a.output/'source_commit.txt').write_text(command(['git','-C',str(ROOT),'rev-parse','HEAD'])+'\n')
        (a.output/'launch.json').write_text(json.dumps(configuration,indent=2)+'\n')
    elif not (a.output/'launch.json').exists(): raise ValueError('Interrupted setup: inspect; refusing overwrite')
    elif not a.resume: raise ValueError('Existing campaign: use --resume after jobs terminate')
    import fcntl
    with (a.output/'.submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ledger=a.output/'submitted_job_ids.tsv'
        live={row.split('_')[0] for row in command(['squeue','-h','-u',os.environ['USER'],'-o','%i']).splitlines()}
        if ledger.exists():
            for row in ledger.read_text().splitlines():
                if row.split()[1] in live: raise ValueError('Existing campaign jobs active; no duplicate submission')
        dependency=None
        env={k:v for k,v in os.environ.items() if not k.startswith('SBATCH_')}
        env.update(PART_SELECTED_FAST_OUTPUT=str(a.output),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
        for phase,hours,array in graph(a.mode,a):
            request=['sbatch','--parsable','--job-name=selfast_'+phase,'--account='+a.account,'--partition='+a.partition,
                '--nodes=1','--ntasks=1','--cpus-per-task='+str(a.cpus),'--mem='+str(a.memory_gib)+'G',
                '--time='+str(hours)+':00:00','--output='+str(a.output/'logs/%x_%A_%a.out'),
                '--error='+str(a.output/'logs/%x_%A_%a.err'),'--export=ALL','--signal=B:USR1@180']
            if array: request+=['--array=0-3%'+str(a.concurrency),'--gpus-per-node=1']
            if dependency: request+=['--dependency=afterok:'+dependency,'--kill-on-invalid-dep=yes']
            env['PART_SELECTED_FAST_PHASE']=phase
            job=command([*request,str(a.output/'source/sbatch/run_part_selected_fast_tigris.sh')],env=env).split(';')[0]
            if not job.isdecimal(): raise ValueError('Unexpected sbatch output; inspect scheduler before retry')
            print(f'{phase}: {job}',flush=True)
            with ledger.open('a') as handle: handle.write(f'{phase}\t{job}\n')
            dependency=job
    print('Campaign:',a.output)


if __name__=='__main__': launch(parse())
