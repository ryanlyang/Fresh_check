#!/usr/bin/env python3
"""Queue one debug diagnostic from saved evidence; no benchmark/production modes."""
import argparse
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from part_ev_triangle_restart import protocol as P

ROOT = Path(__file__).resolve().parents[1]


def command(args, **kwargs):
    return subprocess.check_output([str(a) for a in args], text=True, **kwargs).strip()


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('plan', 'submit', 'review'), required=True)
    parser.add_argument('--campaign', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--account', default='reu-aisocial')
    parser.add_argument('--reviewed', action='store_true')
    args = parser.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.account):
        parser.error('Invalid account')
    args.output = P.canonical(args.output)
    registry = ROOT/'checkpoints/part_ev_triangle_restart'
    if args.output.parent != registry:
        parser.error('Use a fresh direct child of '+str(registry))
    if args.mode != 'review':
        if args.campaign is None:
            parser.error('--campaign is required')
        args.campaign = P.canonical(args.campaign)
        if args.output == args.campaign or args.output.is_relative_to(args.campaign) or args.campaign.is_relative_to(args.output):
            parser.error('Diagnostic and donor must be disjoint')
    return args


def request(output, account):
    return ['sbatch', '--parsable', '--job-name=evtri_restart', '--partition=debug',
            '--account='+account, '--nodes=1', '--ntasks=1', '--cpus-per-task=8',
            '--mem=128G', '--gres=gpu:a100:2', '--time=01:00:00',
            '--signal=USR1@300', '--no-requeue', '--export=ALL',
            '--output='+str(output/'logs/%x_%j.out'), '--error='+str(output/'logs/%x_%j.err'),
            str(output/'source/sbatch/run_part_ev_triangles_restart.sh'), str(output)]


def check_partition():
    text = command(['scontrol', 'show', 'partition', 'debug', '-o'])
    fields = dict(word.split('=', 1) for word in text.split() if '=' in word)
    if fields.get('PartitionName') != 'debug' or fields.get('State') != 'UP':
        raise ValueError('debug is not UP; no automatic fallback to tier3')
    limit = fields.get('MaxTime', '')
    if limit in ('UNLIMITED', 'INFINITE'):
        return
    match = re.fullmatch(r'(?:(\d+)-)?(\d+):(\d+):(\d+)', limit)
    if not match:
        raise ValueError('Cannot parse debug MaxTime')
    d, h, m, s = (int(v or 0) for v in match.groups())
    if d*86400+h*3600+m*60+s < P.POLICY['walltime_seconds']:
        raise ValueError('debug does not allow the requested one-hour diagnostic')


def launch(args):
    if args.mode == 'review':
        P.load_plan(args.output)
        report = args.output/'report.json'
        if report.exists():
            print(report.read_text())
        else:
            print('No complete report. Inspect logs and error_rank_*.json; no admission.')
            for error in sorted(args.output.glob('error_rank_*.json')):
                print(error.read_text())
        return
    if args.output.exists():
        raise FileExistsError('Output already exists; inspect its ledger before retrying')
    spec, settings, artifacts = P.donor(args.campaign)
    print('One debug job: 2 same-node A100s, 8 CPUs, 128 GiB RAM, 01:00:00 walltime.')
    print('Saved triangle probe; 42 synthetic updates per rank; no ROOT data or production writes.')
    if args.mode == 'plan':
        print('Read-only. Submit rechecks source/checkpoint, free space and live scheduler limits.')
        return
    if not args.reviewed:
        raise ValueError('Submission requires --reviewed')
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise ValueError('Submit on SPORC x86_64, not Tigris')
    if not os.access(settings['python'], os.X_OK):
        raise ValueError('Frozen SPORC interpreter is unavailable')
    for name in ('git', 'sbatch', 'scontrol'):
        if not shutil.which(name):
            raise ValueError('Missing '+name)
    if command(['git', '-C', ROOT, 'status', '--porcelain', '--', *P.FILES]):
        raise ValueError('Commit/push/pull diagnostic files before submission')
    commit = command(['git', '-C', ROOT, 'rev-parse', 'HEAD'])
    frozen = {name: subprocess.check_output(['git', '-C', str(ROOT), 'show', commit+':'+name]) for name in P.FILES}
    # Refuse a mismatch between the launcher being executed and the files frozen.
    if any((ROOT/name).read_bytes() != content for name, content in frozen.items()):
        raise ValueError('Working diagnostic differs from the commit being frozen')
    env = {k: v for k, v in os.environ.items() if not k.startswith(('SBATCH_', 'SLURM_'))}
    check_partition()
    test = request(args.output, args.account)
    test[-2] = str(ROOT/'sbatch/run_part_ev_triangles_restart.sh')
    command([test[0], '--test-only', *test[1:]], env=env, stderr=subprocess.STDOUT)
    registry = args.output.parent
    registry.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (registry/'.submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.output.exists():
            raise FileExistsError('Output already exists; no duplicate submission')
        if shutil.disk_usage(registry).free < 2*P.POLICY['output_budget_bytes']:
            raise OSError('Keep 2 GiB free for diagnostic scratch and safety; nothing deleted')
        args.output.mkdir()
        (args.output/'logs').mkdir()
        for name, content in frozen.items():
            path = args.output/'source'/name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as handle:
                handle.write(content)
        inventory = {name: P.sha(args.output/'source'/name) for name in P.FILES}
        plan = dict(schema=P.SCHEMA, policy=P.POLICY, output=str(args.output), campaign=str(args.campaign),
                    campaign_sha256=spec['content_hash'], artifacts=artifacts, account=args.account,
                    python=settings['python'], diagnostic_commit=commit, diagnostic_source=inventory,
                    authorizes_training=False)
        plan['content_hash'] = P.digest(plan)
        P.write_new(args.output/'plan.json', plan)
        with (args.output/'diagnostic.sha256').open('x') as handle:
            for name in ('plan.json', *('source/'+n for n in P.FILES)):
                handle.write(P.sha(args.output/name)+'  '+name+'\n')
        cmd = request(args.output, args.account)
        P.write_new(args.output/'submission_intent.json', dict(status='unknown_until_ledger_written', request=cmd))
        job = command(cmd, env=env).split(';')[0]
        if not job.isdecimal():
            raise ValueError('Ambiguous sbatch reply; inspect scheduler and intent, do not resubmit: '+job)
        with (args.output/'submitted_job_ids.tsv').open('x') as handle:
            handle.write('diagnostic\t'+job+'\n')
            handle.flush()
            os.fsync(handle.fileno())
    print('Diagnostic job:', job)
    print('Output:', args.output)
    print('Donor checkpoints, benchmark gates, production queues and environments unchanged.')


if __name__ == '__main__':
    launch(parse())
