#!/usr/bin/env python3
"""Queue independent benchmark-gated one-GH200 full-production pipelines."""
import argparse
import hashlib
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
    p.add_argument('--mode', choices=('plan', 'pipeline', 'review', 'status', 'dispatch',
                                      'plan-profile-recovery', 'recover-profile'), required=True)
    p.add_argument('--model', choices=tuple(P.PACKAGES))
    p.add_argument('--benchmark', type=Path, help='Read-only PREPARED SPORC benchmark; passing SPORC GPU results are not required')
    p.add_argument('--from-campaign', type=Path, help='Original Tigris triangle campaign for profiling recovery')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--python', default=DEFAULT_PYTHON)
    p.add_argument('--account', default='reu-aisocial')
    p.add_argument('--reviewed', action='store_true')
    p.add_argument('--max-train-hours', type=int, default=480)
    p.add_argument('--max-evaluate-hours', type=int, default=24)
    a = p.parse_args(argv)
    if a.mode in ('pipeline','dispatch','recover-profile') and not a.reviewed: p.error('Inspect plan then explicitly pass --reviewed')
    if a.mode in ('plan-profile-recovery', 'recover-profile'):
        if a.from_campaign is None or a.model is not None or a.benchmark is not None:
            p.error('Profile recovery uses --from-campaign and --output, not --model/--benchmark')
    elif a.from_campaign is not None:
        p.error('--from-campaign is only for profile recovery')
    if not 1 <= a.max_train_hours <= 480 or not 1 <= a.max_evaluate_hours <= 24: p.error('Invalid resource ceiling')
    if a.mode in ('plan','pipeline') and (a.model is None or a.benchmark is None):
        p.error('Fresh benchmark needs --model and --benchmark')
    if a.mode in ('review','status','dispatch') and (a.model is not None or a.benchmark is not None):
        p.error('Review uses only the frozen --output')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', a.account): p.error('Invalid account')
    if not Path(a.python).is_absolute(): p.error('Use an absolute ARM Python path')
    for key in ('output', 'benchmark', 'from_campaign'):
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
    if a.from_campaign and (a.from_campaign.parent != base or a.output == a.from_campaign):
        p.error('Use distinct original/recovery campaigns in the same Tigris registry')
    return a


def inspect(a):
    return json.loads(command([a.python, '-B', ROOT/'scripts/run_part_ev_rts_tigris.py',
        '--phase', 'inspect', '--benchmark', a.benchmark, '--model', a.model], env=environment(a.python)))


def freeze(a, admission, *, profile_recovery=None):
    if a.output.exists(): raise FileExistsError('Use a fresh output; never overwrite/requeue a frozen diagnostic')
    if command(['git', '-C', ROOT, 'status', '--porcelain', '--', *P.FILES]):
        raise ValueError('Commit/push/pull the new Tigris files before submission')
    commit = command(['git', '-C', ROOT, 'rev-parse', 'HEAD'])
    blobs = {name: subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{commit}:{name}']) for name in P.FILES}
    a.output.mkdir(); (a.output/'logs').mkdir()
    for name, blob in blobs.items():
        path = a.output/'source'/name
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(blob)
    extra = {} if profile_recovery is None else dict(profile_recovery=profile_recovery)
    launch = P.hashed(dict(schema=P.SCHEMA, model_kind=a.model, benchmark=str(a.benchmark),
        output=str(a.output), repo=str(ROOT), python=a.python, account=a.account, partition='tigris',
        source=P.inventory(a.output/'source'), commit=commit, policy=P.POLICY, admission=admission,
        output_budget_bytes=P.BENCHMARK_BYTES, authorizes_training=False, auto_production=True,
        max_train_hours=a.max_train_hours, max_evaluate_hours=a.max_evaluate_hours, **extra))
    P.write(a.output/'launch.json', launch)
    (a.output/'snapshot.sha256').write_text(''.join(f'{P.sha(a.output/name)}  {name}\n'
        for name in ['launch.json', *('source/'+n for n in P.FILES)]), encoding='utf-8')
    return launch


def check_origin_idle(output):
    jobs = S.ledger(output, 'pipeline')
    if set(jobs) != set(P.MINUTES):
        raise ValueError('Original pipeline ledger is incomplete; reconcile before recovery')
    active = set(command(['squeue','--noheader','--me','--format=%i']).splitlines())
    if active.intersection(jobs.values()):
        raise ValueError('Original pipeline still has live jobs; no competing recovery submitted')


def profile_recovery(a):
    if platform.system() != 'Linux' or platform.machine() != 'aarch64':
        raise ValueError('Run profile recovery on Tigris ARM')
    if a.output.exists(): raise FileExistsError('Use a new recovery output; inspect prior ledger before retry')
    original = P.launch_config(a.from_campaign, interpreter=False)
    if original['model_kind'] != 'triangles' or 'profile_recovery' in original:
        raise ValueError('Only an original triangle benchmark can be recovered')
    # Inherit authority and environment; command-line defaults cannot enlarge caps.
    for key in ('python','account','max_train_hours','max_evaluate_hours'):
        setattr(a,key,original[key])
    a.model, a.benchmark = 'triangles', Path(original['benchmark'])
    checked = json.loads(command([a.python,'-B',ROOT/'scripts/run_part_ev_rts_tigris.py',
        '--phase','inspect_profile_recovery','--from-campaign',a.from_campaign],env=environment(a.python)))
    for name in ('part_ev_rts_tigris/benchmark.py', 'part_ev_rts_tigris/production.py'):
        committed = subprocess.check_output(['git','-C',str(ROOT),'show','HEAD:'+name])
        if hashlib.sha256(committed).hexdigest() != original['source'][name]:
            raise ValueError('Recovery cannot change the measured execution engine: '+name)
    result = checked['summary']
    print(f"Triangle profiling recovery: CPU-only 30-minute job; no GPU benchmark rerun. "
          f"Measured recommendation: train={result['recommended_train_hours']}h, "
          f"evaluate={result['recommended_evaluate_hours']}h.")
    print('Original files remain untouched. Production requires a separate reviewed dispatch.')
    if a.mode == 'plan-profile-recovery':
        print('Read-only. No files or jobs changed.'); return
    for tool in ('git','sbatch','scontrol','squeue'):
        if shutil.which(tool) is None: raise ValueError('Missing '+tool)
    registry = a.output.parent
    with P.file_lock(Path(original['repo'])/'checkpoints/part_ev_state_production/.submission.lock'), \
            P.file_lock(registry/'.submission.lock'):
        check_origin_idle(a.from_campaign)
        owner = registry/(P.digest(dict(model='triangles',
            donor=original['admission']['donor']['content_hash']))+'.production-owner.json')
        if owner.exists(): raise ValueError('A production owner already exists; reconcile before recovery')
        P.check_storage(dict(output=str(a.output),repo=str(ROOT)))
        S.check_partition(30)
        proposal = dict(model_kind='triangles',account=a.account)
        preview = request(a.output,proposal,'recover_profile')
        preview[-3] = str(ROOT/'sbatch/run_part_ev_rts_tigris.sh')
        command([preview[0],'--test-only',*preview[1:]],env=environment(a.python),stderr=subprocess.STDOUT)
        frozen = freeze(a,checked['admission'],profile_recovery=checked['origin'])
        job = S.submit_one(a.output,frozen,'recover_profile',None,'recovery')
    print('Recovery job:',job)
    print('Recovery output:',a.output)
    print('Inspect summary, then use --mode dispatch --reviewed on THIS new output.')


def launch(a):
    if a.mode in ('plan-profile-recovery','recover-profile'):
        return profile_recovery(a)
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


