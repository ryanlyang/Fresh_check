#!/usr/bin/env python3
"""Queue independent benchmark-gated one-GH200 full-production pipelines."""
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
from part_ev_rts_tigris import protocol as P

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = '/home/ryreu/miniforge3-aarch64/envs/part_ev_fullscale/bin/python'


from part_ev_rts_tigris import submission as S
command, environment, request = S.command, S.environment, S.request


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('plan', 'pipeline', 'review', 'status', 'dispatch'), required=True)
    p.add_argument('--model', choices=tuple(P.PACKAGES))
    p.add_argument('--benchmark', type=Path, help='Read-only PREPARED SPORC benchmark; passing SPORC GPU results are not required')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--python', default=DEFAULT_PYTHON)
    p.add_argument('--account', default='reu-aisocial')
    p.add_argument('--reviewed', action='store_true')
    p.add_argument('--max-train-hours', type=int, default=480)
    p.add_argument('--max-evaluate-hours', type=int, default=24)
    a = p.parse_args(argv)
    if a.mode in ('pipeline','dispatch') and not a.reviewed: p.error('Inspect plan then explicitly pass --reviewed')
    if not 1 <= a.max_train_hours <= 480 or not 1 <= a.max_evaluate_hours <= 24: p.error('Invalid resource ceiling')
    if a.mode in ('plan','pipeline') and (a.model is None or a.benchmark is None):
        p.error('Fresh benchmark needs --model and --benchmark')
    if a.mode in ('review','status','dispatch') and (a.model is not None or a.benchmark is not None):
        p.error('Review uses only the frozen --output')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', a.account): p.error('Invalid account')
    if not Path(a.python).is_absolute(): p.error('Use an absolute ARM Python path')
    for key in ('output', 'benchmark'):
        path = getattr(a, key)
        if path is not None:
            if not path.is_absolute() or path.absolute() != path.resolve():
                p.error('Use absolute paths without symlinks/traversal')
            setattr(a, key, path.resolve())
    base = ROOT/'checkpoints/part_ev_rts_tigris'
    if a.output == base or a.output.parent != base:
        p.error('Output must be a new direct child of '+str(base))
    if a.benchmark and (a.output == a.benchmark or a.output.is_relative_to(a.benchmark)
                        or a.benchmark.is_relative_to(a.output)):
        p.error('Tigris output and SPORC benchmark must be disjoint')
    return a


def inspect(a):
    return json.loads(command([a.python, '-B', ROOT/'scripts/run_part_ev_rts_tigris.py',
        '--phase', 'inspect', '--benchmark', a.benchmark, '--model', a.model], env=environment(a.python)))


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
        output_budget_bytes=P.BENCHMARK_BYTES, authorizes_training=False, auto_production=True,
        max_train_hours=a.max_train_hours, max_evaluate_hours=a.max_evaluate_hours))
    P.write(a.output/'launch.json', launch)
    (a.output/'snapshot.sha256').write_text(''.join(f'{P.sha(a.output/name)}  {name}\n'
        for name in ['launch.json', *('source/'+n for n in P.FILES)]), encoding='utf-8')
    return launch


def launch(a):
    if a.mode in ('review','status','dispatch'):
        saved = P.launch_config(a.output, interpreter=False)
        print(command([saved['python'], '-B', a.output/'source/scripts/run_part_ev_rts_tigris.py',
            '--phase', a.mode, '--output', a.output], env=environment(saved['python'])))
        return
    if platform.system() != 'Linux' or platform.machine() != 'aarch64':
        raise ValueError('Run this on Tigris ARM, not SPORC or local Windows')
    if not os.access(a.python, os.X_OK): raise ValueError('Existing ARM Python not executable; no installation attempted')
    admission = inspect(a)
    print(f"{a.model}: {admission['parameters']:,} parameters; fresh seed 101; full-schedule projection")
    print('Prepare 2h CPU -> benchmark 5h30m / one GH200 -> verified summary -> automatic production dispatch.')
    print('Production: fresh seed101, 1M updates / 50 checks, one GH200/native BN; GPU phases 128 GiB host RAM.')
    print(f'Walltime ceilings: {a.max_train_hours}h train, {a.max_evaluate_hours}h evaluate; actual requests follow measurements.')
    if a.mode == 'plan':
        print('Read-only. No files/jobs changed; submission also checks aggregate storage and live Slurm limits.')
        return
    for tool in ('git', 'sbatch', 'scontrol'):
        if shutil.which(tool) is None: raise ValueError('Missing '+tool)
    registry = ROOT/'checkpoints/part_ev_rts_tigris'
    registry.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (registry/'.submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if a.output.exists(): raise FileExistsError('Output exists; inspect its ledger, use a fresh directory for an intentional retry')
        needed = P.check_storage(dict(output=str(a.output), repo=str(ROOT)))
        S.check_partition(max(P.MINUTES.values()))
        proposal = dict(model_kind=a.model, account=a.account)
        env = environment(a.python)
        for phase in P.MINUTES:
            row = request(a.output, proposal, phase)
            # Existing live wrapper for --test-only; the frozen one is used for actual submissions.
            row[-3] = str(ROOT/'sbatch/run_part_ev_rts_tigris.sh')
            command([row[0], '--test-only', *row[1:]], env=env, stderr=subprocess.STDOUT)
        frozen = freeze(a, admission)
        previous = None
        for phase in P.MINUTES:
            dependency = None if previous is None else ('afterany:' if phase == 'summary' else 'afterok:')+previous
            previous = S.submit_one(a.output, frozen, phase, dependency, 'pipeline')
        print('Benchmark:', a.output)
        print(f'Aggregate headroom checked: {needed/2**30:.2f} GiB. SPORC jobs/data/environments unchanged.')


if __name__ == '__main__': launch(parse())


