#!/usr/bin/env python3
"""Independent selected compact benchmark launcher. No production mode."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tarfile
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from part_selected_compact import protocol as P
from scripts.submit_part_ev_compact import inventory, verify_snapshot

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = '/home/ryreu/atlas/Fresh_check/checkpoints/part_selected_fullscale/seven_buildfix_20261106T005700Z'
RECOVERY = '/home/ryreu/atlas/Fresh_check/checkpoints/part_selected_compiler_recovery/compilerfix_omp_20261110T232529Z'
PYTHON = '/home/ryreu/miniforge3-aarch64/envs/part_selected_tigris/bin/python'
REQUIRED = ('part_selected_compact/model.py', 'part_selected_compact/benchmark.py',
            'part_selected_compact/protocol.py', 'part_selected_compact/parity.py',
            'part_selected_compact/flops.py', 'scripts/run_part_selected_compact.py',
            'scripts/submit_part_selected_compact.py', 'sbatch/run_part_selected_compact_tigris.sh')


def command(args, **kwargs):
    return subprocess.check_output([str(x) for x in args], text=True, **kwargs).strip()


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('benchmark', 'review'), default='benchmark')
    p.add_argument('--output', type=Path)
    p.add_argument('--reference', type=Path, default=Path(REFERENCE))
    p.add_argument('--recovery', type=Path, default=Path(RECOVERY))
    p.add_argument('--python', type=Path, default=Path(PYTHON))
    p.add_argument('--account', default='reu-aisocial')
    p.add_argument('--concurrency', type=int, choices=(1, 2), default=2)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args(argv)
    if a.output is None:
        if a.mode == 'review':
            p.error('--output required for review')
        a.output = ROOT/'checkpoints/part_selected_compact'/('selected_compact_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    if not re.fullmatch('[A-Za-z0-9_-]+', a.account):
        p.error('Invalid account')
    for key in ('output', 'reference', 'recovery', 'python'):
        value = getattr(a, key)
        if not value.is_absolute():
            p.error('Use absolute paths')
        if key == 'output' and value.absolute() != value.resolve():
            p.error('Output contains symlinks or traversal')
        setattr(a, key, value.resolve())
    base = ROOT/'checkpoints/part_selected_compact'
    if a.output == base or not a.output.is_relative_to(base):
        p.error('Use a new child of checkpoints/part_selected_compact')
    for other in (a.reference, a.recovery):
        if a.output.is_relative_to(other) or other.is_relative_to(a.output):
            p.error('Output overlaps a protected donor')
    return a


def requests(a):
    result = []
    for phase in ('prepare', 'benchmark', 'summary'):
        args = ['sbatch', '--parsable', '--job-name=selcompact_'+phase, '--partition=tigris',
                '--account='+a.account, '--nodes=1', '--ntasks=1', '--cpus-per-task=12', '--mem=128G',
                '--time='+('00:30:00' if phase == 'summary' else '05:00:00'),
                '--output='+str(a.output/'logs/%x_%A_%a.out'), '--error='+str(a.output/'logs/%x_%A_%a.err'),
                '--export=ALL', '--signal=B:USR1@180']
        if phase == 'benchmark':
            args += ['--array=0-1%'+str(a.concurrency), '--gres=gpu:gh200:1']
        result.append((phase, args))
    return result


def environment(python, source):
    env = {k: v for k, v in os.environ.items() if not k.startswith(('SBATCH_', 'SLURM_'))}
    binary = Path(python).parent
    env.update(PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1',
               PATH=str(binary)+os.pathsep+env.get('PATH', ''),
               LD_LIBRARY_PATH=str(binary.parent/'lib')+os.pathsep+env.get('LD_LIBRARY_PATH', ''),
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', WORLD_SIZE='1')
    for name in ('RANK', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT', 'CUDA_VISIBLE_DEVICES'):
        env.pop(name, None)
    return env


def freeze(a):
    if command(['git', '-C', ROOT, 'status', '--porcelain']):
        raise ValueError('Commit/push/pull first; freeze requires a clean worktree')
    for name in REQUIRED:
        command(['git', '-C', ROOT, 'cat-file', '-e', 'HEAD:'+name])
    P.require_space(a.output)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.mkdir()  # Atomic duplicate guard. Never resume ambiguous submissions.
    (a.output/'source').mkdir()
    (a.output/'logs').mkdir()
    with tempfile.TemporaryDirectory(prefix='selected-compact-freeze-', dir=a.output) as tmp:
        archive = Path(tmp)/'source.tar'
        subprocess.run(['git', '-C', str(ROOT), 'archive', '--format=tar', '--output', str(archive), 'HEAD'], check=True)
        with tarfile.open(archive) as tar:
            members = []
            for member in tar.getmembers():
                path = Path(member.name)
                if not (a.output/'source'/path).resolve().is_relative_to(a.output/'source') or not (member.isfile() or member.isdir()):
                    raise ValueError('Unsafe git archive entry')
                if '__pycache__' not in path.parts and path.suffix not in ('.pyc', '.pyo'):
                    members.append(member)
            tar.extractall(a.output/'source', members=members)
    P.write_json(a.output/'snapshot.json', inventory(a.output/'source'))
    P.write_json(a.output/'launch.json', dict(output=str(a.output), reference=str(a.reference), recovery=str(a.recovery),
        python=str(a.python), account=a.account, partition='tigris', concurrency=a.concurrency,
        source_commit=command(['git', '-C', ROOT, 'rev-parse', 'HEAD']), authorizes_training=False))
    verify_snapshot(a.output)


def launch(a):
    if a.mode == 'review':
        verify_snapshot(a.output)
        launch = P.read_json(a.output/'launch.json')
        subprocess.run([launch['python'], '-B', str(a.output/'source/scripts/run_part_selected_compact.py'),
                        '--phase', 'review', '--output', str(a.output)],
                       env=environment(launch['python'], a.output/'source'), check=True)
        return
    if a.dry_run:
        print('Campaign:', a.output)
        for phase, _ in requests(a):
            print(phase, 'walltime='+('00:30:00' if phase == 'summary' else '05:00:00'),
                  'one GH200 per task, two tasks' if phase == 'benchmark' else 'CPU only')
        print('12 CPUs/128 GiB per job; concurrency='+str(a.concurrency)+'. Benchmarks ONLY; no jobs/files changed.')
        return
    if platform.machine() != 'aarch64':
        raise ValueError('Submit on Tigris, not SPORC or the local machine')
    if a.output.exists():
        raise FileExistsError('Campaign already exists; inspect its journal, never blindly resubmit')
    if not a.python.is_file() or not (a.reference/'experiment_spec.json').is_file() or not (a.recovery/'compatibility.json').is_file():
        raise ValueError('Missing interpreter, seven-run donor, or passed recovery evidence')
    freeze(a)
    worker = str(a.output/'source/sbatch/run_part_selected_compact_tigris.sh')
    env = environment(a.python, a.output/'source')
    for phase, request in requests(a):
        subprocess.run([*request, '--test-only', worker, str(a.output), phase], env=env, check=True)
    previous = None
    for phase, request in requests(a):
        if previous:
            request += ['--dependency='+('afterany:' if phase == 'summary' else 'afterok:')+previous,
                        '--kill-on-invalid-dep=yes']
        journal = a.output/f'submission_{phase}.json'
        P.write_json(journal, dict(phase=phase, status='submitting', request=request))
        job = command([*request, worker, str(a.output), phase], env=env).split(';')[0]
        if not job.isdecimal():
            raise ValueError('Uncertain sbatch result; inspect scheduler and journal')
        P.write_json(journal, dict(phase=phase, status='accepted', request=request, job_id=job))
        with (a.output/'submitted_job_ids.tsv').open('a', encoding='utf-8') as handle:
            handle.write(phase+'\t'+job+'\n')
            handle.flush()
            os.fsync(handle.fileno())
        print(phase+' job: '+job, flush=True)
        previous = job
    print('Campaign:', a.output)
    print('Benchmarks only. Existing campaigns, jobs, binaries, datasets and environments unchanged.')


if __name__ == '__main__':
    launch(parse())
