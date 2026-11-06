#!/usr/bin/env python3
"""Standard-library benchmark-only submitter, with frozen source and duplicate guards."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.submit_part_selected_fast import digest, verify_snapshot

ROOT=Path(__file__).resolve().parents[1]
CHECKPOINTS='/home/ryreu/atlas/Fresh_check/checkpoints'
REFERENCE=CHECKPOINTS+'/part_selected_tigris/selected_gh200_20261023T073044Z'
BASELINE=CHECKPOINTS+'/part_fullscale/part_ev_100m_20261013T044756Z'
STATE=CHECKPOINTS+'/part_ev_tigris_segments/benchmark_723341'


def command(args,**kwargs): return subprocess.check_output(args,text=True,**kwargs).strip()


def parse(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('benchmark','profile','review','status'),default='benchmark')
    p.add_argument('--output',type=Path)
    p.add_argument('--reference',type=Path,default=Path(REFERENCE))
    p.add_argument('--baseline-control',type=Path,default=Path(BASELINE))
    p.add_argument('--baseline-evaluation',type=Path,default=Path(BASELINE+'_validation_20261018T224308Z'))
    p.add_argument('--state-control',type=Path,default=Path(STATE))
    p.add_argument('--python',type=Path,default=Path(sys.executable))
    p.add_argument('--concurrency',type=int,choices=(1,2,3),default=1)
    p.add_argument('--account',default='reu-aisocial')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args(argv)
    if a.output is None:
        if a.resume or a.mode in ('review','status'): p.error('--output required')
        a.output=ROOT/'checkpoints/part_selected_fullscale'/('seven_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    if not re.fullmatch(r'[A-Za-z0-9_-]+',a.account): p.error('Invalid account')
    for name in ('output','reference','baseline_control','baseline_evaluation','state_control','python'):
        path=getattr(a,name)
        if not path.is_absolute(): p.error('--'+name+' must be absolute')
        setattr(a,name,path.resolve())
    for other in (a.reference,a.baseline_control,a.baseline_evaluation,a.state_control,ROOT):
        if a.output==other or other.is_relative_to(a.output): p.error('Output overlaps protected path')
        if other!=ROOT and a.output.is_relative_to(other): p.error('Output is inside a reference campaign')
    return a


def settings(a):
    return dict(reference=str(a.reference),baseline_control=str(a.baseline_control),
        baseline_evaluation=str(a.baseline_evaluation),state_control=str(a.state_control),
        python=str(a.python),account=a.account,partition='tigris',cpus=12,memory_gib=128,
        tree_threads=4,cache_bytes=0,compile_region=False,authorizes_training=False)


def graph(mode):
    if mode in ('review','status'): return []
    rows=[('prepare',5,None),('profile',5,'2,5,6')]
    if mode=='benchmark': rows.extend((('benchmark',5,'0-6'),('summary',.5,None)))
    return rows


def launch(a):
    config=settings(a)
    if (a.output/'launch.json').exists():
        saved=json.loads((a.output/'launch.json').read_text())
        if a.mode in ('review','status'): config=saved
        elif config!=saved: raise ValueError('Settings differ from frozen campaign')
        verify_snapshot(a.output)
    if a.dry_run:
        print('Campaign:',a.output)
        for phase,hours,array in graph(a.mode):
            print(phase,'hours=',hours,'array=',array,'GPUs per task=',int(array is not None))
        print('12 CPUs, 128 GiB host RAM; one GH200 per GPU job; concurrency=',a.concurrency)
        print('No files changed or jobs submitted. No production training phase exists in this benchmark launcher.')
        return
    if a.mode in ('review','status'):
        if not (a.output/'launch.json').is_file(): raise ValueError('Campaign missing')
        subprocess.run([config['python'],str(a.output/'source/scripts/run_part_selected_fullscale.py'),
                        '--phase',a.mode,'--output',str(a.output)],check=True); return
    if platform.machine()!='aarch64': raise ValueError('Submit from Tigris ARM, not SPORC')
    for name in ('git','sbatch','squeue','scontrol'):
        if shutil.which(name) is None: raise ValueError('Missing '+name)
    if not a.python.is_file() or not (a.reference/'experiment_spec.json').is_file():
        raise ValueError('Missing existing Python or selected Tigris reference')
    partition=command(['scontrol','show','partition','tigris','-o'])
    match=re.search(r'\bMaxTime=(\S+)',partition)
    if not match: raise ValueError('Cannot verify partition MaxTime')
    if match[1] not in ('UNLIMITED','INFINITE'):
        parts=match[1].split('-'); days=int(parts[0]) if len(parts)==2 else 0
        clock=[int(x) for x in parts[-1].split(':')]
        if len(clock)!=3 or days*86400+clock[0]*3600+clock[1]*60+clock[2]<5*3600:
            raise ValueError('Partition MaxTime cannot accommodate five-hour checks')
    if not a.output.exists():
        if a.resume: raise ValueError('No campaign to resume')
        # Freeze the entire committed tree, never a mixture of dirty and committed code.
        if command(['git','-C',str(ROOT),'status','--porcelain']):
            raise ValueError('Commit/push/pull or preserve unrelated work in a clean worktree before submission')
        for name in ('part_selected_fullscale/model.py','scripts/run_part_selected_fullscale.py',
                     'scripts/submit_part_selected_fullscale.py','sbatch/run_part_selected_fullscale_tigris.sh'):
            command(['git','-C',str(ROOT),'cat-file','-e','HEAD:'+name])
        # This is a pre-snapshot floor; CPU prepare calculates the full seven-run allowance.
        existing=a.output.parent
        while not existing.exists(): existing=existing.parent
        if shutil.disk_usage(existing).free<12*2**30:
            raise OSError('Need at least 12 GiB free before preparing the benchmark campaign; nothing deleted')
        a.output.parent.mkdir(parents=True,exist_ok=True); a.output.mkdir()
        (a.output/'source').mkdir(); (a.output/'logs').mkdir()
        with tempfile.TemporaryDirectory(prefix='seven-freeze-',dir=a.output) as tmp:
            archive=Path(tmp)/'source.tar'
            subprocess.run(['git','-C',str(ROOT),'archive','--format=tar','--output',str(archive),'HEAD'],check=True)
            with tarfile.open(archive) as tar:
                for entry in tar.getmembers():
                    target=(a.output/'source'/entry.name).resolve()
                    if not target.is_relative_to(a.output/'source') or not (entry.isfile() or entry.isdir()):
                        raise ValueError('Unsafe source archive entry')
                tar.extractall(a.output/'source')
        (a.output/'snapshot.json').write_text(json.dumps({p.relative_to(a.output/'source').as_posix():digest(p)
            for p in sorted((a.output/'source').rglob('*')) if p.is_file()},indent=2)+'\n')
        (a.output/'source_commit.txt').write_text(command(['git','-C',str(ROOT),'rev-parse','HEAD'])+'\n')
        (a.output/'launch.json').write_text(json.dumps(config,indent=2)+'\n')
    elif not (a.output/'launch.json').exists(): raise ValueError('Interrupted setup; inspect before retry')
    elif not a.resume: raise ValueError('Existing campaign: use --resume only after its jobs terminate')
    import fcntl
    with (a.output/'.submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ledger=a.output/'submitted_job_ids.tsv'
        live={line.split('_')[0] for line in command(['squeue','-h','-u',os.environ['USER'],'-o','%i']).splitlines()}
        if ledger.exists() and any(line.split()[1] in live for line in ledger.read_text().splitlines()):
            raise ValueError('Campaign jobs are active; refusing duplicates')
        dependency=None
        env={k:v for k,v in os.environ.items() if not k.startswith('SBATCH_')}
        env.update(SELECTED_FULL_OUTPUT=str(a.output),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
        for phase,hours,array in graph(a.mode):
            request=['sbatch','--parsable','--job-name=sel7_'+phase,'--account='+a.account,'--partition=tigris',
                '--nodes=1','--ntasks=1','--cpus-per-task=12','--mem=128G',
                '--time='+('00:30:00' if phase=='summary' else '05:00:00'),
                '--output='+str(a.output/'logs/%x_%A_%a.out'),'--error='+str(a.output/'logs/%x_%A_%a.err'),
                '--export=ALL','--signal=B:USR1@180']
            if array: request+=['--array='+array+'%'+str(a.concurrency),'--gres=gpu:gh200:1']
            if dependency:
                request+=['--dependency='+('afterany:' if phase=='summary' else 'afterok:')+dependency,
                          '--kill-on-invalid-dep=yes']
            env['SELECTED_FULL_PHASE']=phase
            job=command([*request,str(a.output/'source/sbatch/run_part_selected_fullscale_tigris.sh')],env=env).split(';')[0]
            if not job.isdecimal(): raise ValueError('Unexpected sbatch output; inspect scheduler before retry')
            with ledger.open('a') as handle:
                handle.write(phase+'\t'+job+'\n'); handle.flush(); os.fsync(handle.fileno())
            print(phase+' job: '+job,flush=True); dependency=job
    print('Campaign:',a.output)
    print('Benchmarks only. Existing campaigns and all production queues remain unchanged.')


if __name__=='__main__': launch(parse())
