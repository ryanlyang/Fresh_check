#!/usr/bin/env python3
"""Check/review/retry only selected production tasks 2, 5 and 6 on Tigris."""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from part_selected_compiler_recovery import protocol as P

ROOT = Path(__file__).resolve().parents[1]


def command(args, **kwargs):
    return subprocess.check_output(args, text=True, **kwargs).strip()


def environment(plan):
    env = {k: v for k, v in os.environ.items() if not k.startswith(('SBATCH_', 'SLURM_'))}
    binary = Path(plan['python']).parent
    env.update(PATH=str(binary)+os.pathsep+env.get('PATH', ''),
        LD_LIBRARY_PATH=str(binary.parent/'lib')+os.pathsep+env.get('LD_LIBRARY_PATH', ''),
        PYTHONPATH=str(Path(plan['benchmark'])/'source'), PYTHONNOUSERSITE='1',
        PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='4', MAX_JOBS='2', WORLD_SIZE='1')
    for name in ('RANK', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT', 'CUDA_VISIBLE_DEVICES'):
        env.pop(name, None)
    return env


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('plan', 'check', 'review', 'submit', 'report'), required=True)
    p.add_argument('--campaign', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reviewed', action='store_true')
    a = p.parse_args(argv)
    for name in ('output', 'campaign'):
        path = getattr(a, name)
        if path is not None:
            if not path.is_absolute():
                p.error('--'+name+' must be absolute')
            setattr(a, name, path.resolve())
    if a.mode in ('plan', 'check') and a.campaign is None:
        p.error('plan/check require --campaign')
    if a.mode in ('submit', 'report') and not a.reviewed:
        p.error('Submission requires --reviewed after a passed compatibility review')
    base = (ROOT/'checkpoints/part_selected_compiler_recovery').resolve()
    if a.output == base or not a.output.is_relative_to(base):
        p.error('output must be a child of checkpoints/part_selected_compiler_recovery')
    if a.campaign is not None and (a.output.is_relative_to(a.campaign) or a.campaign.is_relative_to(a.output)):
        p.error('Recovery output must be disjoint from production')
    return a


def validate(output, plan, phase):
    driver = output/'source/scripts/run_part_selected_compiler_recovery.py'
    return command([plan['python'], '-B', str(driver), '--phase', phase, '--output', str(output)],
                   env=environment(plan))


def freeze(a, plan, evidence):
    if a.output.exists():
        raise FileExistsError('Use a new recovery output; existing evidence is immutable')
    if command(['git', '-C', str(ROOT), 'status', '--porcelain', '--', *P.FILES]):
        raise ValueError('Commit/push/pull recovery implementation before submission')
    commit = command(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'])
    blobs = {n: subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{commit}:{n}']) for n in P.FILES}
    a.output.mkdir(parents=True)
    (a.output/'logs').mkdir()
    for name, data in blobs.items():
        path = a.output/'source'/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    row = P.hashed(dict(schema=P.SCHEMA, output=str(a.output), campaign=str(a.campaign),
        python=plan['python'], indices=list(P.INDICES), transition=dict(old=P.OLD_BANNER, new=P.NEW_BANNER),
        source={n: P.sha(a.output/'source'/n) for n in P.FILES}, commit=commit, **evidence))
    P.write(a.output/'recovery.json', row)
    (a.output/'snapshot.sha256').write_text(''.join(
        f'{P.sha(a.output/n)}  {n}\n' for n in ['recovery.json', *('source/'+n for n in P.FILES)]), encoding='utf-8')
    return row


def request(output, plan, phase, index=None, dependency=None):
    hours = 1 if phase in ('check', 'report') else plan[phase+'_hours'][index]
    args = ['sbatch', '--parsable', '--job-name=sel7recover_'+phase+(f'_{index}' if index is not None else ''),
        '--partition=tigris', '--account='+plan['account'], '--nodes=1', '--ntasks=1',
        '--cpus-per-task=12', '--mem=128G', '--time='+str(int(hours*60)),
        '--output='+str(output/'logs/%x_%j.out'), '--error='+str(output/'logs/%x_%j.err'),
        '--export=ALL', '--kill-on-invalid-dep=yes']
    if phase in ('train', 'evaluate'):
        args += ['--gres=gpu:gh200:1', '--signal=B:USR1@900']
    if dependency:
        args += ['--dependency=afterok:'+':'.join(dependency)]
    return args+[str(output/'source/sbatch/run_part_selected_compiler_recovery.sh'), str(output), phase,
                 str(index) if index is not None else '-']


def ledger(output):
    path = output/'submitted_job_ids.tsv'
    entries = []
    if path.exists():
        for line in path.read_text().splitlines():
            label, job = line.split()
            if not job.isdecimal():
                raise ValueError('Invalid ledger job ID')
            entries.append((label, job))
    return entries


def scope_check(output, campaign, live, *, report_only=False):
    # Original launcher uses the same registry lock and sees appended recovery
    # IDs, preventing its broad resume mode from racing this narrower recovery.
    for directory in (output, campaign):
        intent = directory/'submission_intent.json'
        if intent.exists() and P.read(intent).get('status') == 'unknown':
            raise ValueError('Unresolved sbatch attempt; inspect submission_intent.json before retrying')
    protected = {'report_None'}
    if not report_only:
        protected.update(f'{phase}_{i}' for i in P.INDICES for phase in ('train', 'evaluate'))
        protected.add('check_None')
    for label, job in ledger(output)+ledger(campaign):
        if label in protected and job in live:
            raise ValueError(f'Target recovery stage already active/pending: {label} {job}')


def run_directory(campaign, task):
    return campaign/'runs'/task['run_id']/f"seed_{task['seed']}"


def pending(campaign, plan):
    result = []
    for i in P.INDICES:
        run = run_directory(campaign, plan['science']['tasks'][i])
        train = not (run/'training_complete.json').is_file()
        if train or not (run/'full_validation/metrics.json').is_file():
            result.append((i, train))
    return result  # Original workers audit every marker/checkpoint before using it.


def report_dependencies(campaign, plan, scheduled, live):
    dependencies = []
    missing = []
    entries = ledger(campaign)
    for i, task in enumerate(plan['science']['tasks']):
        if i in scheduled:
            continue
        active = {job for label, job in entries if label == f'evaluate_{i}' and job in live}
        if len(active) > 1:
            raise ValueError('Duplicate live evaluations for task ' + str(i))
        if active:
            dependencies.extend(sorted(active))
        elif not (run_directory(campaign, task)/'full_validation/metrics.json').is_file():
            missing.append(i)
    return dependencies, missing


def append_ledger(directory, label, job):
    with (directory/'submitted_job_ids.tsv').open('a', encoding='utf-8') as stream:
        stream.write(f'{label}\t{job}\n'); stream.flush(); os.fsync(stream.fileno())


def submit(output, campaign, plan, phase, index=None, dependency=None):
    args = request(output, plan, phase, index, dependency)
    # Mark BOTH ledgers unresolved before sbatch: ambiguous success cannot be
    # retried by either launcher. No old records are removed.
    journals = [output] if phase == 'check' else [output, campaign]
    for directory in journals:
        P.write(directory/'submission_intent.json', dict(status='unknown', request=args, recovery=str(output)))
    job = command(args, env=environment(plan)).split(';')[0]
    if not job.isdecimal():
        raise ValueError('Ambiguous sbatch response; inspect scheduler before retrying: '+job)
    for directory in journals:
        append_ledger(directory, f'{phase}_{index}', job)
    # Record only after every ledger append succeeded.
    for directory in reversed(journals):
        P.write(directory/'submission_intent.json', dict(status='recorded', request=args, job=job, recovery=str(output)))
    print(f'{phase} {index if index is not None else ""} job: {job}', flush=True)
    return job


def dispatch(output, campaign, plan, *, report_only=False):
    live = set(command(['squeue', '-h', '--me', '-o', '%i']).splitlines())
    scope_check(output, campaign, live, report_only=report_only)
    jobs = [] if report_only else pending(campaign, plan)
    other_dependencies, missing = report_dependencies(campaign, plan, {i for i, _ in jobs}, live)
    if report_only and missing:
        raise ValueError('Report cannot run: evaluations missing without live jobs: '+str(missing))
    shapes = [(phase, i) for i, train in jobs for phase in (('train', 'evaluate') if train else ('evaluate',))]
    if not missing:
        shapes.append(('report', None))
    # Check ALL resource shapes before submitting anything.
    for phase, index in shapes:
        args = request(output, plan, phase, index)
        command([args[0], '--test-only', *args[1:]], env=environment(plan), stderr=subprocess.STDOUT)
    evaluates = []
    for i, train in jobs:
        training = submit(output, campaign, plan, 'train', i) if train else None
        evaluates.append(submit(output, campaign, plan, 'evaluate', i, [training] if training else None))
    if missing:
        print('Final report deferred; other evaluations missing without live jobs:', missing)
        print('Use --mode report --reviewed after they are queued or complete. No other jobs were changed.')
    else:
        submit(output, campaign, plan, 'report', dependency=other_dependencies+evaluates)


def launch(a):
    if a.mode in ('plan', 'check'):
        plan = P.checked(a.campaign/'production.json')
        evidence = json.loads(command([plan['python'], '-B', str(ROOT/'scripts/run_part_selected_compiler_recovery.py'),
            '--phase', 'inspect', '--campaign', str(a.campaign)], env=environment(plan)))
        print('CPU compatibility check: 1 hour, 12 CPUs, 128 GiB; no GPU or dataset read.')
        print('Recovery scope: tasks 2, 5, 6; original models, seeds, budgets and walltimes unchanged.')
        if a.mode == 'plan':
            print('Read-only plan. No files/jobs changed. No training is authorized until the check passes.')
            return
    else:
        row = P.checked(a.output/'recovery.json')
        a.campaign = Path(row['campaign'])
        plan = P.checked(a.campaign/'production.json')
        print(validate(a.output, plan, 'review'))
        if a.mode == 'review':
            return
    if platform.machine() != 'aarch64':
        raise ValueError('Submit on Tigris, not SPORC or Windows')
    for name in ('sbatch', 'squeue', 'git'):
        if shutil.which(name) is None:
            raise ValueError('Missing '+name)
    import fcntl
    registry = ROOT/'checkpoints/part_selected_compiler_recovery'
    registry.mkdir(parents=True, exist_ok=True)
    # Share the original launcher's global lock, but do not reject unrelated live
    # jobs. Both locks are submission-only; training/evaluation retain run locks.
    with (a.campaign.parent/'.submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if a.mode == 'check':
            freeze(a, plan, evidence)
            args = request(a.output, plan, 'check')
            command([args[0], '--test-only', *args[1:]], env=environment(plan), stderr=subprocess.STDOUT)
            submit(a.output, a.campaign, plan, 'check')
        else:
            owner = registry/(plan['content_hash']+'.owner.json')
            if owner.exists() and P.read(owner)['output'] != str(a.output):
                raise ValueError('Production already has a recovery owner: '+P.read(owner)['output'])
            # Keep the original campaign's full remaining-output budget; this
            # is not permission to delete data or relax storage requirements.
            used = sum(p.stat().st_size for p in a.campaign.rglob('*') if p.is_file() and not p.is_symlink())
            storage = plan['review']['storage']
            reserve = storage.get('reserve_bytes', 4*2**30)
            required = reserve+max(0, storage['required_free_bytes']-reserve-used)
            if shutil.disk_usage(a.campaign).free < required:
                raise OSError('Insufficient remaining production headroom; nothing deleted')
            P.write(owner, dict(output=str(a.output), production_sha256=plan['content_hash']))
            dispatch(a.output, a.campaign, plan, report_only=a.mode == 'report')
    print('Recovery:', a.output)
    print('No jobs cancelled. Original snapshots/backend binaries unchanged. Production retries append to its job ledger.')


if __name__ == '__main__':
    launch(parse())
