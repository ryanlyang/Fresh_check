#!/usr/bin/env python3
"""Submit two independent evaluation jobs; never train, cancel, or edit donors."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import part_ev_rejection_eval_support as S

REPO = Path(__file__).resolve().parents[1]


@contextmanager
def submission_lock(parent):
    import fcntl
    with (parent / '.submission.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def request(plan, output, phase, key=None, dependency=None):
    gpu = phase == 'worker'
    name = f'evrej_{key}' if gpu else 'evrej_report'
    args = ['sbatch', '--parsable', '--job-name=' + name,
            '--partition=' + plan['partition'], '--account=' + plan['account'],
            '--nodes=1', '--ntasks=1', '--cpus-per-task=' + ('8' if gpu else '1'),
            '--mem=' + ('64G' if gpu else '2G'),
            '--time=' + ('05:00:00' if gpu else '00:15:00'),
            '--output=' + str(output / 'logs' / '%x_%j.out'),
            '--error=' + str(output / 'logs' / '%x_%j.err'), '--export=ALL',
            '--signal=B:USR1@180', '--chdir=' + str(output)]
    if gpu:
        args += ['--gres=gpu:a100:1']
    if dependency:
        args += ['--dependency=afterok:' + dependency, '--kill-on-invalid-dep=yes']
    args += [str(output / 'source/sbatch/run_part_ev_rejection_eval.sh'),
             '--phase', phase, '--output', str(output)]
    if key:
        args += ['--model', key]
    return args


def partition_check(partition):
    info = S.command(['scontrol', 'show', 'partition', partition, '-o'])
    limit = re.search(r'\bMaxTime=(\S+)', info)
    if not limit or not re.search(r'\bState=UP\b', info):
        raise ValueError('Partition unavailable or unreadable')
    limit = limit.group(1)
    if limit in ('UNLIMITED', 'INFINITE'):
        return
    days, clock = limit.split('-', 1) if '-' in limit else ('0', limit)
    h, m, s = map(int, clock.split(':'))
    if int(days)*86400 + h*3600 + m*60 + s < 18000:
        raise ValueError('Partition must allow five-hour evaluation jobs')


def source_bytes():
    if S.command(['git', '-C', str(REPO), 'status', '--porcelain', '--', *S.SOURCES]):
        raise ValueError('Commit/push/pull the new evaluator files before submitting')
    import subprocess
    return {p: subprocess.run(['git', '-C', str(REPO), 'show', 'HEAD:' + p],
                             check=True, capture_output=True).stdout for p in S.SOURCES}


def overlapping_ledgers(parent, donors):
    result = []
    for path in parent.glob('*/plan.json'):
        other = S.checked(S.read(path))
        if {d['root'] for d in other['donors'].values()} & {d['root'] for d in donors.values()}:
            result.append(path.parent / 'submitted_job_ids.tsv')
    return result


def submit_jobs(output, plan):
    ledger = output / 'submitted_job_ids.tsv'
    ids = []
    for key in ('part', 'state'):
        # A resumed submission only allocates GPUs for unfinished models.
        result = output / key / 'metrics.json'
        if result.exists():
            row = S.checked(S.read(result))
            if row['plan_sha256'] != plan['content_hash'] or row['model'] != key:
                raise ValueError('Existing evaluation belongs to another plan/model')
            print(f'{key}: already evaluated; no GPU job submitted.')
            continue
        job = S.command(request(plan, output, 'worker', key)).split(';')[0]
        if not job.isdigit():
            raise ValueError('Unexpected sbatch response; inspect queue before retrying')
        with ledger.open('a') as f:
            f.write(f'{key}\t{job}\n')
        ids.append(job)
        print(f'{key} evaluation job: {job}', flush=True)
    job = S.command(request(plan, output, 'report', dependency=':'.join(ids) or None)).split(';')[0]
    if not job.isdigit():
        raise ValueError('Unexpected report job response')
    with ledger.open('a') as f:
        f.write(f'report\t{job}\n')
    print(f'Report job: {job}\nResults: {output}\nNo training submitted; no old jobs cancelled.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('plan', 'submit', 'resume', 'status'), required=True)
    parser.add_argument('--part', type=Path, default=S.PART)
    parser.add_argument('--state', type=Path, default=S.STATE)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--partition', default='debug')
    parser.add_argument('--account', default='reu-aisocial')
    args = parser.parse_args(argv)
    if args.mode in ('resume', 'status') and args.output is None:
        parser.error('--output required for resume/status')
    output = (args.output or S.BASE / 'part_ev_rejection_eval' /
              ('rejection_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))).resolve()
    if args.mode == 'status':
        os.execv(sys.executable, [sys.executable, str(REPO / 'scripts/run_part_ev_rejection_eval.py'),
                                 '--phase', 'status', '--output', str(output)])
    if any(not re.fullmatch(r'[A-Za-z0-9_-]+', x) for x in (args.partition, args.account)):
        parser.error('Invalid partition/account name')
    if args.mode == 'resume':
        plan = S.verify_snapshot(output)
        donors = plan['donors']
        for donor in donors.values():
            S.verify_pins(donor)
    else:
        donors = {k: S.describe(p, k) for k, p in [('part', args.part), ('state', args.state)]}
        if donors['part']['files'] != donors['state']['files']:
            raise ValueError('Donor validation inventories differ; do not mix populations')
        for d in donors.values():
            root = Path(d['root'])
            if output == root or output.is_relative_to(root) or root.is_relative_to(output):
                raise ValueError('Evaluation output must be separate from both donors')
        if output.exists():
            raise ValueError('Use a new output path, or --mode resume for an existing evaluation')
        plan = dict(schema='part_ev_rejection_eval_v1', donors=donors,
                    partition=args.partition, account=args.account,
                    population='official_validation_5M', independent_test=False,
                    targets=[0.3, 0.5], precision='FP32', batch_size=512)
    print('Two independent one-A100 jobs, 8 CPUs / 64 GiB each, five hours each.', flush=True)
    print('Full 5M validation; FP32; no training, no AUC, no automatic cancellation.', flush=True)
    for key, donor in donors.items():
        print(f"{key}: selected check {donor['best_check']}; {donor['root']}")
    if args.mode == 'plan':
        print(f'Output: {output}\nRead-only plan; no files or jobs changed.')
        return
    if os.name != 'posix':
        raise ValueError('Submit from sporcsubmit')
    # Prevent inherited sbatch options from redirecting the job/resources.
    for name in list(os.environ):
        if name.startswith('SBATCH_'):
            del os.environ[name]
    partition_check(plan['partition'])
    # Check active old jobs before making even the new output directory.
    S.refuse_active(donors, overlapping_ledgers(output.parent, donors))
    output.parent.mkdir(parents=True, exist_ok=True)
    with submission_lock(output.parent):
        S.refuse_active(donors, overlapping_ledgers(output.parent, donors))
        if shutil.disk_usage(output.parent).free < 2 * 1024**3:
            raise OSError('Keep 2 GiB free for two resumable prediction sets and headroom')
        if args.mode != 'resume':
            payload = source_bytes()
            output.mkdir()
            for name, content in payload.items():
                path = output / 'source' / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            plan['source'] = {p: S.sha(output / 'source' / p) for p in S.SOURCES}
            plan['source_commit'] = S.command(['git', '-C', str(REPO), 'rev-parse', 'HEAD'])
            plan = S.seal(plan)
            S.write(output / 'plan.json', plan)
        (output / 'logs').mkdir(exist_ok=True)
        submit_jobs(output, plan)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        raise SystemExit(f'{type(exc).__name__}: {exc}\nInspect any recorded jobs before retrying; nothing cancelled.')
