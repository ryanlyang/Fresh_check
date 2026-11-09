#!/usr/bin/env python3
"""Submit ONLY the independent two-arm Tigris benchmarks, never production."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tarfile
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from part_ev_compact import protocol as P

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = '/home/ryreu/atlas/Fresh_check/checkpoints/part_selected_fullscale/seven_buildfix_20261106T005700Z'
PYTHON = '/home/ryreu/miniforge3-aarch64/envs/part_selected_tigris/bin/python'


def command(args, **kwargs):
    return subprocess.check_output([str(x) for x in args], text=True, **kwargs).strip()


def inventory(source):
    result = {}
    for p in sorted(Path(source).rglob('*')):
        if p.is_symlink(): raise ValueError('Symlink in frozen source: '+str(p))
        if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc', '.pyo'):
            result[p.relative_to(source).as_posix()] = P.file_sha256(p)
    return result


def verify_snapshot(output):
    output = Path(output)
    expected = P.read_json(output/'snapshot.json')
    if not expected or inventory(output/'source') != expected:
        raise ValueError('Frozen source contents/inventory changed')


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('benchmark', 'review'), default='benchmark')
    p.add_argument('--output', type=Path)
    p.add_argument('--reference', type=Path, default=Path(REFERENCE))
    p.add_argument('--python', type=Path, default=Path(PYTHON))
    p.add_argument('--account', default='reu-aisocial')
    p.add_argument('--concurrency', type=int, choices=(1,2), default=2)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args(argv)
    if a.output is None:
        if a.mode == 'review': p.error('--output required for review')
        a.output = ROOT/'checkpoints/part_ev_compact'/('compact_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    if not re.fullmatch('[A-Za-z0-9_-]+', a.account): p.error('Invalid account')
    for key in ('output', 'reference', 'python'):
        value = getattr(a,key)
        if not value.is_absolute(): p.error('Use absolute paths')
        # Resolve before mutation and refuse symlink traversal for outputs.
        if key == 'output' and value.absolute() != value.resolve(): p.error('Output contains symlinks')
        setattr(a,key,value.resolve())
    for other in (a.reference, ROOT):
        if a.output == other or other.is_relative_to(a.output): p.error('Output overlaps protected directory')
        if other != ROOT and a.output.is_relative_to(other): p.error('Output is inside donor')
    if not a.output.is_relative_to(ROOT/'checkpoints/part_ev_compact'):
        p.error('Use a new directory under checkpoints/part_ev_compact')
    return a


def requests(a):
    result = []
    for phase in ('prepare', 'benchmark', 'summary'):
        args = ['sbatch','--parsable','--job-name=evcompact_'+phase,'--partition=tigris',
                '--account='+a.account,'--nodes=1','--ntasks=1','--cpus-per-task=12','--mem=128G',
                '--time='+('00:30:00' if phase=='summary' else '05:00:00'),
                '--output='+str(a.output/'logs/%x_%A_%a.out'), '--error='+str(a.output/'logs/%x_%A_%a.err'),
                '--export=ALL','--signal=B:USR1@180']
        if phase == 'benchmark': args += ['--array=0-1%'+str(a.concurrency),'--gres=gpu:gh200:1']
        result.append((phase,args))
    return result


def freeze(a):
    if command(['git','-C',ROOT,'status','--porcelain']):
        raise ValueError('Commit/push/pull first; freeze requires a clean worktree')
    for name in ('part_ev_compact/model.py','part_ev_compact/benchmark.py','scripts/run_part_ev_compact.py',
                 'sbatch/run_part_ev_compact_tigris.sh'):
        command(['git','-C',ROOT,'cat-file','-e','HEAD:'+name])
    P.require_space(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    # Atomic mkdir is the duplicate submission guard. Never resume an ambiguous
    # submission automatically, overwrite evidence, or modify another campaign.
    a.output.mkdir()
    (a.output/'source').mkdir(); (a.output/'logs').mkdir()
    with tempfile.TemporaryDirectory(prefix='compact-freeze-',dir=a.output) as tmp:
        archive = Path(tmp)/'source.tar'
        subprocess.run(['git','-C',str(ROOT),'archive','--format=tar','--output',str(archive),'HEAD'],check=True)
        with tarfile.open(archive) as tar:
            members=[]
            for member in tar.getmembers():
                path=Path(member.name); target=(a.output/'source'/path).resolve()
                if not target.is_relative_to(a.output/'source') or not (member.isfile() or member.isdir()):
                    raise ValueError('Unsafe git archive entry')
                if '__pycache__' not in path.parts and path.suffix not in ('.pyc','.pyo'): members.append(member)
            tar.extractall(a.output/'source',members=members)
    P.write_json(a.output/'snapshot.json',inventory(a.output/'source'))
    P.write_json(a.output/'launch.json',dict(reference=str(a.reference),python=str(a.python),
        account=a.account,partition='tigris',concurrency=a.concurrency,source_commit=command(['git','-C',ROOT,'rev-parse','HEAD']),
        authorizes_training=False))
    verify_snapshot(a.output)


def launch(a):
    if a.dry_run:
        print('Campaign:',a.output)
        for phase, args in requests(a):
            print(phase, 'walltime='+('00:30:00' if phase=='summary' else '05:00:00'),
                  'one GH200 per array task' if phase=='benchmark' else 'CPU only')
        print('Two independent models; concurrency='+str(a.concurrency)+'. Benchmarks ONLY; no jobs or files changed.')
        return
    if a.mode == 'review':
        verify_snapshot(a.output)
        python=P.read_json(a.output/'launch.json')['python']
        subprocess.run([python,'-B',str(a.output/'source/scripts/run_part_ev_compact.py'),
                        '--phase','review','--output',str(a.output)],check=True)
        return
    if platform.machine() != 'aarch64': raise ValueError('Submit on Tigris, not SPORC or the local machine')
    if a.output.exists(): raise ValueError('Campaign already exists. Inspect its submission journal; never blindly resubmit.')
    if not a.python.is_file() or not (a.reference/'experiment_spec.json').is_file():
        raise ValueError('Missing existing interpreter or donor metadata')
    freeze(a)
    worker=str(a.output/'source/sbatch/run_part_ev_compact_tigris.sh')
    env={k:v for k,v in os.environ.items() if not k.startswith('SBATCH_')}
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
    # Validate all resource requests before accepting even the CPU prepare job.
    for phase, request in requests(a):
        subprocess.run([*request,'--test-only',worker,str(a.output),phase],env=env,check=True)
    previous=None
    for phase, request in requests(a):
        if previous:
            request += ['--dependency='+('afterany:' if phase=='summary' else 'afterok:')+previous,
                        '--kill-on-invalid-dep=yes']
        intent=a.output/f'submission_{phase}.json'
        P.write_json(intent,dict(phase=phase,status='submitting',request=request))
        job=command([*request,worker,str(a.output),phase],env=env).split(';')[0]
        if not job.isdecimal(): raise ValueError('Uncertain sbatch result; inspect scheduler and submission journal')
        P.write_json(intent,dict(phase=phase,status='accepted',job_id=job,request=request))
        with (a.output/'submitted_job_ids.tsv').open('a',encoding='utf-8') as handle:
            handle.write(phase+'\t'+job+'\n'); handle.flush(); os.fsync(handle.fileno())
        print(phase+' job: '+job,flush=True); previous=job
    print('Campaign:',a.output)
    print('No existing jobs, source snapshots, datasets or environments changed. Production is NOT queued.')


if __name__ == '__main__': launch(parse())
