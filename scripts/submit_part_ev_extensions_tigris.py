#!/usr/bin/env python3
"""Queue independent prepare -> one-GH200 benchmark -> summary, never production."""
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
from part_ev_extensions_tigris import protocol as P

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = '/home/ryreu/miniforge3-aarch64/envs/part_ev_fullscale/bin/python'


def command(argv, **kwargs):
    return subprocess.check_output([str(x) for x in argv], text=True, **kwargs).strip()


def environment(python):
    env = {k: v for k, v in os.environ.items() if not k.startswith(('SBATCH_', 'SLURM_', 'NCCL_'))}
    binary = Path(python).parent
    env.update(PATH=str(binary)+os.pathsep+env.get('PATH', ''),
        LD_LIBRARY_PATH=str(binary.parent/'lib')+os.pathsep+env.get('LD_LIBRARY_PATH', ''),
        PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
    for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'GROUP_RANK', 'ROLE_RANK',
                'ROLE_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT', 'CUDA_VISIBLE_DEVICES', 'PYTHONPATH'):
        env.pop(key, None)
    return env


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('plan', 'benchmark', 'review'), required=True)
    p.add_argument('--model', choices=tuple(P.PACKAGES))
    p.add_argument('--benchmark', type=Path, help='Read-only passed SPORC benchmark directory')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--python', default=DEFAULT_PYTHON)
    p.add_argument('--account', default='reu-aisocial')
    a = p.parse_args(argv)
    if a.mode != 'review' and (a.model is None or a.benchmark is None):
        p.error('Fresh benchmark needs --model and --benchmark')
    if a.mode == 'review' and (a.model is not None or a.benchmark is not None):
        p.error('Review uses only the frozen --output')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', a.account): p.error('Invalid account')
    if not Path(a.python).is_absolute(): p.error('Use an absolute ARM Python path')
    for key in ('output', 'benchmark'):
        path = getattr(a, key)
        if path is not None:
            if not path.is_absolute() or path.absolute() != path.resolve():
                p.error('Use absolute paths without symlinks/traversal')
            setattr(a, key, path.resolve())
    base = ROOT/'checkpoints/part_ev_extensions_tigris'
    if a.output == base or a.output.parent != base:
        p.error('Output must be a new direct child of '+str(base))
    if a.benchmark and (a.output == a.benchmark or a.output.is_relative_to(a.benchmark)
                        or a.benchmark.is_relative_to(a.output)):
        p.error('Tigris output and SPORC benchmark must be disjoint')
    return a


def inspect(a):
    return json.loads(command([a.python, '-B', ROOT/'scripts/run_part_ev_extensions_tigris.py',
        '--phase', 'inspect', '--benchmark', a.benchmark, '--model', a.model], env=environment(a.python)))


def request(output, launch, phase, dependency=None):
    row = ['sbatch', '--parsable', '--job-name=evextgh_'+launch['model_kind']+'_'+phase,
        '--partition=tigris', '--account='+launch['account'], '--nodes=1', '--ntasks=1',
        '--cpus-per-task='+('2' if phase == 'summary' else '8'),
        '--mem='+('8G' if phase == 'summary' else '128G'), '--time='+str(P.MINUTES[phase]),
        '--output='+str(output/'logs/%x_%j.out'), '--error='+str(output/'logs/%x_%j.err'),
        '--export=ALL', '--kill-on-invalid-dep=yes']
    if phase == 'benchmark': row += ['--gres=gpu:gh200:1', '--signal=USR1@900']
    if dependency: row += ['--dependency='+dependency]
    return row+[str(output/'source/sbatch/run_part_ev_extensions_tigris.sh'), str(output), phase]


def check_partition():
    row = command(['scontrol', 'show', 'partition', 'tigris', '-o'])
    fields = dict(item.split('=', 1) for item in row.split() if '=' in item)
    if fields.get('State') != 'UP': raise ValueError('Tigris partition is not UP')
    limit = fields.get('MaxTime', '')
    if limit == 'UNLIMITED': return
    match = re.fullmatch(r'(?:(\d+)-)?(\d+):(\d+):(\d+)', limit)
    if not match: raise ValueError('Cannot establish Tigris walltime limit')
    d, h, m, s = (int(v or 0) for v in match.groups())
    if d*86400+h*3600+m*60+s < 60*max(P.MINUTES.values()):
        raise ValueError('Diagnostic exceeds Tigris walltime limit')


def freeze(a, admission):
    if a.output.exists(): raise FileExistsError('Use a fresh output; never overwrite/requeue a frozen diagnostic')
    if command(['git', '-C', ROOT, 'status', '--porcelain', '--', *P.FILES]):
        raise ValueError('Commit/push/pull the new Tigris files before submission')
    commit = command(['git', '-C', ROOT, 'rev-parse', 'HEAD'])
    blobs = {name: subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{commit}:{name}']) for name in P.FILES}
    a.output.mkdir(); (a.output/'logs').mkdir()
    for name, blob in blobs.items():
        path = a.output/'source'/name
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(blob)
    launch = P.hashed(dict(schema=P.SCHEMA, model_kind=a.model, benchmark=str(a.benchmark),
        output=str(a.output), repo=str(ROOT), python=a.python, account=a.account, partition='tigris',
        source=P.inventory(a.output/'source'), commit=commit, policy=P.POLICY, admission=admission,
        output_budget_bytes=P.BENCHMARK_BYTES, authorizes_training=False))
    P.write(a.output/'launch.json', launch)
    (a.output/'snapshot.sha256').write_text(''.join(f'{P.sha(a.output/name)}  {name}\n'
        for name in ['launch.json', *('source/'+n for n in P.FILES)]), encoding='utf-8')
    return launch


def launch(a):
    if a.mode == 'review':
        saved = P.launch_config(a.output, interpreter=False)
        print(command([saved['python'], '-B', a.output/'source/scripts/run_part_ev_extensions_tigris.py',
            '--phase', 'review', '--output', a.output], env=environment(saved['python'])))
        return
    if platform.system() != 'Linux' or platform.machine() != 'aarch64':
        raise ValueError('Run this on Tigris ARM, not SPORC or local Windows')
    if not os.access(a.python, os.X_OK): raise ValueError('Existing ARM Python not executable; no installation attempted')
    admission = inspect(a)
    print(f"{a.model}: {admission['parameters']:,} parameters; fresh seed 101; full-schedule projection")
    print('Prepare 2h CPU; benchmark 5h30m / one GH200; summary 15m CPU. No training or cross-model dependencies.')
    if a.mode == 'plan':
        print('Read-only. No files/jobs changed; submission also checks aggregate storage and live Slurm limits.')
        return
    for tool in ('git', 'sbatch', 'scontrol'):
        if shutil.which(tool) is None: raise ValueError('Missing '+tool)
    registry = ROOT/'checkpoints/part_ev_extensions_tigris'
    registry.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (registry/'.submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if a.output.exists(): raise FileExistsError('Output exists; inspect its ledger, use a fresh directory for an intentional retry')
        needed = P.check_storage(dict(output=str(a.output), repo=str(ROOT)))
        check_partition()
        proposal = dict(model_kind=a.model, account=a.account)
        env = environment(a.python)
        for phase in P.MINUTES:
            row = request(a.output, proposal, phase)
            # Existing live wrapper for --test-only; the frozen one is used for actual submissions.
            row[-3] = str(ROOT/'sbatch/run_part_ev_extensions_tigris.sh')
            command([row[0], '--test-only', *row[1:]], env=env, stderr=subprocess.STDOUT)
        frozen = freeze(a, admission)
        previous = None
        for phase in P.MINUTES:
            dependency = None if previous is None else ('afterany:' if phase == 'summary' else 'afterok:')+previous
            row = request(a.output, frozen, phase, dependency)
            P.write(a.output/'submission_intent.json', dict(status='unknown', request=row))
            job = command(row, env=env).split(';')[0]
            if not job.isdecimal(): raise ValueError('Uncertain sbatch response; inspect scheduler and intent; do not resubmit blindly')
            with (a.output/'submitted_job_ids.tsv').open('a', encoding='utf-8') as ledger:
                ledger.write(f'{phase}\t{job}\n'); ledger.flush(); os.fsync(ledger.fileno())
            P.write(a.output/'submission_intent.json', dict(status='recorded', job=job, request=row))
            print(f'{phase} job: {job}', flush=True)
            previous = job
        print('Benchmark:', a.output)
        print(f'Aggregate headroom checked: {needed/2**30:.2f} GiB. SPORC jobs/data/environments unchanged.')


if __name__ == '__main__': launch(parse())
