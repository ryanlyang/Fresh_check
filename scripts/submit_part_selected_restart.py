#!/usr/bin/env python3
"""Queue one bounded restart investigation, never production or benchmark retries."""
import argparse
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.submit_part_selected_fast import verify_snapshot
from scripts.diagnose_part_selected_restart import SCHEMA, POLICY, sha, read, write_new

ROOT = Path(__file__).resolve().parents[1]
FILES = ('scripts/diagnose_part_selected_restart.py', 'scripts/submit_part_selected_restart.py',
         'sbatch/run_part_selected_restart.sh', 'sbatch/submit_part_selected_restart.sh')


def command(argv, **kwargs):
    return subprocess.check_output(argv, text=True, **kwargs).strip()


def request(output, account):
    return ['sbatch', '--parsable', '--job-name=sel7_restart', '--partition=tigris',
            '--account='+account, '--nodes=1', '--ntasks=1', '--cpus-per-task=12',
            '--mem=128G', '--gres=gpu:gh200:1', '--time=02:00:00', '--signal=B:USR1@300',
            '--output='+str(output/'restart_%j.out'), '--error='+str(output/'restart_%j.err'),
            '--export=ALL', str(output/'run_part_selected_restart.sh'), str(output)]


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', required=True, type=Path)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args(argv)
    if not args.campaign.is_absolute():
        p.error('Use an absolute existing campaign path')
    args.campaign = args.campaign.resolve(strict=True)
    return args


def launch(args):
    campaign = args.campaign
    verify_snapshot(campaign)
    settings = read(campaign/'launch.json')
    spec = read(campaign/'experiment_spec.json')
    if (spec.get('schema') != 'selected_fullscale_benchmarks_v1'
            or settings.get('partition') != 'tigris' or spec.get('launch') != settings
            or spec.get('authorizes_training') is not False):
        raise ValueError('Need the prepared seven-model Tigris benchmark campaign')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', settings['account']):
        raise ValueError('Invalid frozen account')
    if args.dry_run:
        print('Campaign:', campaign)
        print('One GH200; 12 CPUs; 128 GiB RAM; 02:00:00 walltime; tasks 0,1,3,4 sequentially.')
        print('No files/jobs changed. No training, retries, compilation, or tolerance changes.')
        return
    if platform.machine() != 'aarch64':
        raise ValueError('Submit on Tigris ARM')
    for name in ('git', 'sbatch', 'squeue'):
        if not shutil.which(name):
            raise ValueError('Missing '+name)
    if not Path(settings['python']).is_file():
        raise ValueError('Frozen Python missing')
    if command(['git', '-C', str(ROOT), 'status', '--porcelain', '--', *FILES]):
        raise ValueError('Commit/push/pull the diagnostic files before submitting')
    for name in FILES:
        command(['git', '-C', str(ROOT), 'cat-file', '-e', 'HEAD:'+name])
    if shutil.disk_usage(campaign).free < 4*2**30:
        raise OSError('Retain 4 GiB free; nothing deleted')
    directory = campaign/'diagnostics'
    if directory.is_symlink():
        raise ValueError('Unexpected diagnostics symlink')
    directory.mkdir(exist_ok=True)
    import fcntl
    with (directory/'.selected_restart.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = directory/'selected_restart_jobs.tsv'
        live = {line.split('_')[0] for line in command(['squeue', '-h', '--me', '-o', '%i']).splitlines()}
        if ledger.exists() and any(line.split()[0] in live for line in ledger.read_text().splitlines()):
            raise ValueError('A restart diagnostic is already active; refusing duplicate')
        output = Path(tempfile.mkdtemp(prefix='restart.', dir=directory))
        for name in FILES:
            data = subprocess.check_output(['git', '-C', str(ROOT), 'show', 'HEAD:'+name])
            (output/Path(name).name).write_bytes(data)
        p = dict(schema=SCHEMA, policy=POLICY, campaign=str(campaign), python=settings['python'],
                 campaign_sha256=spec['content_hash'],
                 diagnostic_commit=command(['git', '-C', str(ROOT), 'rev-parse', 'HEAD']),
                 diagnostic_source={Path(n).name: sha(output/Path(n).name) for n in FILES},
                 artifacts={n: sha(campaign/n) for n in ('experiment_spec.json', 'launch.json', 'snapshot.json')})
        write_new(output/'plan.json', p)
        names = [Path(n).name for n in FILES] + ['plan.json']
        (output/'diagnostic.sha256').write_text(''.join(f'{sha(output/n)}  {n}\n' for n in names))
        env = {k: v for k, v in os.environ.items() if not k.startswith('SBATCH_')}
        job = command(request(output, settings['account']), env=env).split(';')[0]
        if not job.isdecimal():
            raise ValueError('Unexpected sbatch output; inspect scheduler before retrying: '+job)
        with ledger.open('a') as stream:
            stream.write(f'{job}\t{output}\n')
            stream.flush()
            os.fsync(stream.fileno())
        (output/'submitted_job_id.txt').write_text(job+'\n')
    print('Diagnostic job:', job)
    print('Diagnostic directory:', output)
    print('Report:', output/'report.json')
    print('No campaign sources, successful benchmarks, checkpoints, or tolerances changed.')
    print('Diagnostic completion is NOT parity approval. Review before fixing/retrying four benchmarks.')


if __name__ == '__main__':
    launch(parse())
