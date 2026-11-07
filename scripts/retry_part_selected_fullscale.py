#!/usr/bin/env python3
"""Four-arm benchmark retry with a pinned correctness-only overlay and mixed review.

Model/data/training/validation/projection code is imported from the ORIGINAL
frozen campaign. Only the named restart, result-routing, and review files are
overlaid. Successful selected evidence stays at its original location and hash.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
INDICES = (0, 1, 3, 4)
SELECTED_IDS = ('SELECTED_PART_MATCH_STATE_LAYERWISE', 'SELECTED_PART_SHARED', 'SELECTED_STATE_EV_LAYERWISE')
FILES = ('part_selected_fullscale/parity.py', 'part_selected_fullscale/benchmark.py',
         'scripts/run_part_selected_fullscale.py', 'scripts/retry_part_selected_fullscale.py',
         'sbatch/run_part_selected_fullscale_retry.sh', 'sbatch/submit_part_selected_fullscale_retry.sh')
SCHEMA = 'selected_standard_benchmark_retry_v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def checked_metadata(path):
    row = read(path)
    digest = hashlib.sha256(json.dumps({k:v for k,v in row.items() if k != 'content_hash'},
                                      sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if row.get('content_hash') != digest: raise ValueError('Invalid metadata hash: '+str(path))
    return row


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def command(argv, **kwargs):
    return subprocess.check_output(argv, text=True, **kwargs).strip()


def verify_snapshot(campaign):
    expected = read(campaign/'snapshot.json')
    source = campaign/'source'
    actual = {p.relative_to(source).as_posix() for p in source.rglob('*')
              if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    if not expected or actual != expected.keys():
        raise ValueError('Original source inventory changed')
    for name, digest in expected.items():
        path = (source/name).resolve()
        if not path.is_relative_to(source.resolve()) or sha(path) != digest:
            raise ValueError('Original source changed: '+name)


def diagnostic_evidence(campaign, diagnostic):
    """Require the actual four completed exact instrumented pilots, not a job exit."""
    p = read(diagnostic/'plan.json')
    spec = read(campaign/'experiment_spec.json')
    if (Path(p['campaign']).resolve() != campaign or p['campaign_sha256'] != spec['content_hash']
            or p['schema'] != 'selected_restart_diagnostic_v1'):
        raise ValueError('Diagnostic belongs to a different campaign')
    for name, expected in p['artifacts'].items():
        if sha(campaign/name) != expected: raise ValueError('Diagnostic campaign artifacts changed')
    for name, expected in p['diagnostic_source'].items():
        if Path(name).name != name or sha(diagnostic/name) != expected:
            raise ValueError('Diagnostic sources changed')
    paths = ['plan.json', 'report.json'] + [f'task_{i}.json' for i in INDICES]
    if read(diagnostic/'report.json').get('complete') is not True:
        raise ValueError('Diagnostic incomplete')
    for index in INDICES:
        r = read(diagnostic/f'task_{index}.json')
        if (r.get('task') != spec['tasks'][index] or r.get('campaign_sha256') != spec['content_hash']
                or any(r.get(key) is not True for key in ('complete', 'exact_roundtrip',
                  'identical_initial_states', 'uninterrupted_repeat_exact', 'resumed_exact', 'update_sequences_equal'))):
            raise ValueError(f'Diagnostic task {index} did not establish exact controlled restart')
    return {name: sha(diagnostic/name) for name in paths}


def check_inputs(campaign, diagnostic):
    verify_snapshot(campaign)
    spec, settings = checked_metadata(campaign/'experiment_spec.json'), read(campaign/'launch.json')
    if (spec['schema'] != 'selected_fullscale_benchmarks_v1' or spec['launch'] != settings
            or settings.get('partition') != 'tigris' or spec.get('authorizes_training') is not False):
        raise ValueError('Need the existing seven-model Tigris benchmark campaign')
    artifacts = {name: sha(campaign/name) for name in ('snapshot.json', 'experiment_spec.json', 'launch.json')}
    for run_id in SELECTED_IDS:
        for kind in ('profiles', 'benchmarks'):
            name = f'{kind}/{run_id}.json'
            r = checked_metadata(campaign/name)
            if r.get('passed') is not True or r.get('complete') is not True or r.get('campaign_sha256') != spec['content_hash']:
                raise ValueError('Missing successful selected evidence: '+name)
            artifacts[name] = sha(campaign/name)
    diagnostics = diagnostic_evidence(campaign, diagnostic)
    return spec, settings, artifacts, diagnostics


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('submit', 'benchmark', 'summary', 'review'), default='submit')
    p.add_argument('--campaign', type=Path)
    p.add_argument('--diagnostic', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--task-index', type=int, choices=INDICES)
    p.add_argument('--concurrency', type=int, choices=(1, 2, 3, 4), default=1)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args(argv)
    if a.mode == 'submit':
        if a.campaign is None or a.diagnostic is None or a.output is not None:
            p.error('Submit requires --campaign and --diagnostic; output is a fresh sidecar directory')
    elif a.output is None or a.dry_run:
        p.error('Worker/review requires --output; dry-run is submit only')
    if a.mode == 'benchmark' and a.task_index is None: p.error('Benchmark requires --task-index')
    for name in ('campaign', 'diagnostic', 'output'):
        path = getattr(a, name)
        if path is not None:
            if not path.is_absolute(): p.error('--'+name+' must be absolute')
            setattr(a, name, path.resolve(strict=True))
    return a


def job_request(output, account, mode, concurrency, dependency=None):
    args = ['sbatch', '--parsable', '--job-name=sel7_retry_'+mode, '--account='+account,
            '--partition=tigris', '--nodes=1', '--ntasks=1', '--cpus-per-task=12', '--mem=128G',
            '--time='+('05:00:00' if mode == 'benchmark' else '00:30:00'),
            '--output='+str(output/'logs/%x_%A_%a.out'), '--error='+str(output/'logs/%x_%A_%a.err'),
            '--export=ALL', '--signal=B:USR1@180']
    if mode == 'benchmark': args += ['--array=0,1,3,4%'+str(concurrency), '--gres=gpu:gh200:1']
    if dependency: args += ['--dependency=afterany:'+dependency]
    return args + [str(output/'run_part_selected_fullscale_retry.sh'), str(output), mode]


def submit(a):
    spec, settings, artifacts, diagnostics = check_inputs(a.campaign, a.diagnostic)
    if not re.fullmatch(r'[A-Za-z0-9_-]+', settings['account']): raise ValueError('Invalid account')
    if a.dry_run:
        print('Retry tasks 0,1,3,4 only: five hours, one GH200, 12 CPUs, 128 GiB per task; concurrency', a.concurrency)
        print('Then a 30-minute CPU summary. No prepare/profile/train jobs. Selected evidence remains untouched.')
        print('No files or jobs changed.')
        return
    if platform.machine() != 'aarch64': raise ValueError('Submit on Tigris ARM')
    if not Path(settings['python']).is_file(): raise ValueError('Frozen Python missing')
    for name in ('git', 'sbatch', 'squeue'):
        if not shutil.which(name): raise ValueError('Missing '+name)
    if command(['git', '-C', str(ROOT), 'status', '--porcelain', '--', *FILES]):
        raise ValueError('Commit/push/pull retry source files before submission')
    for name in FILES: command(['git', '-C', str(ROOT), 'cat-file', '-e', 'HEAD:'+name])
    if shutil.disk_usage(a.campaign).free < 4*2**30: raise OSError('Keep 4 GiB free; nothing deleted')
    directory = a.campaign/'benchmark_retries'
    if directory.is_symlink(): raise ValueError('Unexpected retry-directory symlink')
    directory.mkdir(exist_ok=True)
    import fcntl
    with (directory/'.submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        live = {line.split('_')[0] for line in command(['squeue', '-h', '--me', '-o', '%i']).splitlines()}
        ledger = directory/'submitted_job_ids.tsv'
        for path in (ledger, a.campaign/'submitted_job_ids.tsv'):
            if path.exists() and any(line.split()[1] in live for line in path.read_text().splitlines()):
                raise ValueError('Original campaign/retry jobs are active; refusing duplicate submission')
        output = Path(tempfile.mkdtemp(prefix='retry.', dir=directory))
        (output/'logs').mkdir()
        for name in FILES:
            (output/Path(name).name).write_bytes(subprocess.check_output(['git', '-C', str(ROOT), 'show', 'HEAD:'+name]))
        p = dict(schema=SCHEMA, campaign=str(a.campaign), campaign_sha256=spec['content_hash'],
                 diagnostic=str(a.diagnostic), diagnostic_hashes=diagnostics, python=settings['python'],
                 parent_artifacts=artifacts, indices=list(INDICES), authorizes_training=False,
                 method='synchronized_exact_restart_v2',
                 commit=command(['git', '-C', str(ROOT), 'rev-parse', 'HEAD']),
                 overlay_source={Path(n).name: sha(output/Path(n).name) for n in FILES})
        write_new(output/'retry_plan.json', p)
        names = [Path(n).name for n in FILES]+['retry_plan.json']
        (output/'retry.sha256').write_text(''.join(f'{sha(output/n)}  {n}\n' for n in names))
        env = {k: v for k, v in os.environ.items() if not k.startswith('SBATCH_')}
        dependency = None
        for mode in ('benchmark', 'summary'):
            job = command(job_request(output, settings['account'], mode, a.concurrency, dependency), env=env).split(';')[0]
            if not job.isdecimal(): raise ValueError('Unexpected sbatch response; inspect scheduler: '+job)
            for path in (ledger, output/'submitted_job_ids.tsv'):
                with path.open('a') as stream:
                    stream.write(f'{mode}\t{job}\t{output}\n'); stream.flush(); os.fsync(stream.fileno())
            dependency = job
            print(mode+' job: '+job, flush=True)
        print('Retry:', output)
        print('Combined report:', output/'reports/benchmark_review.json')
        print('Original files unchanged. No production training submitted or authorized.')


def load_plan(output):
    manifest = {}
    for line in (output/'retry.sha256').read_text().splitlines():
        digest, name = line.split(None, 1)
        if Path(name).name != name or name in manifest: raise ValueError('Invalid retry inventory')
        manifest[name] = digest
    if set(manifest) != {Path(n).name for n in FILES} | {'retry_plan.json'}:
        raise ValueError('Incomplete retry inventory')
    for name, digest in manifest.items():
        if sha(output/name) != digest: raise ValueError('Retry snapshot changed: '+name)
    p = read(output/'retry_plan.json')
    if (p['schema'] != SCHEMA or p['indices'] != list(INDICES) or p['authorizes_training'] is not False
            or p['method'] != 'synchronized_exact_restart_v2'
            or output.parent != Path(p['campaign'])/'benchmark_retries'):
        raise ValueError('Retry contract/location changed')
    if set(p['overlay_source']) != {Path(n).name for n in FILES}: raise ValueError('Unexpected override inventory')
    for name, digest in p['overlay_source'].items():
        if sha(output/name) != digest: raise ValueError('Retry source changed: '+name)
    for name, digest in p['parent_artifacts'].items():
        if sha(Path(p['campaign'])/name) != digest: raise ValueError('Original evidence changed: '+name)
    for name, digest in p['diagnostic_hashes'].items():
        if sha(Path(p['diagnostic'])/name) != digest: raise ValueError('Restart diagnostic changed: '+name)
    verify_snapshot(Path(p['campaign']))
    return p


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def execute(a):
    p = load_plan(a.output)
    if a.mode == 'review' and Path(sys.executable).resolve() != Path(p['python']).resolve():
        env = dict(os.environ)
        binary = Path(p['python']).parent
        env.update(PYTHONPATH=str(Path(p['campaign'])/'source'), PYTHONNOUSERSITE='1',
                   PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
                   PATH=str(binary)+os.pathsep+env.get('PATH',''),
                   LD_LIBRARY_PATH=str(binary.parent/'lib')+os.pathsep+env.get('LD_LIBRARY_PATH',''))
        subprocess.run([p['python'], '-B', str(a.output/'retry_part_selected_fullscale.py'),
                        '--mode', 'review', '--output', str(a.output)], check=True, env=env)
        return
    campaign = Path(p['campaign'])
    source = campaign/'source'
    sys.path.insert(0, str(source))
    import part_selected_fullscale
    if not Path(part_selected_fullscale.__file__).resolve().is_relative_to(source):
        raise ValueError('Retry must import frozen campaign scientific code')
    parity = load_module('part_selected_fullscale._retry_parity', a.output/'parity.py')
    bench = load_module('part_selected_fullscale._retry_benchmark', a.output/'benchmark.py')
    reviewer = load_module('_selected_retry_reviewer', a.output/'run_part_selected_fullscale.py')
    provenance = dict(plan_sha256=sha(a.output/'retry_plan.json'), method=p['method'],
                      parent_campaign_sha256=p['campaign_sha256'], overlay_source=p['overlay_source'],
                      diagnostic_hashes=p['diagnostic_hashes'], authorizes_training=False)
    if a.mode in ('summary', 'review'):
        return reviewer.review(campaign, save=a.mode == 'summary', retry_root=a.output,
                               retry_provenance=provenance, retry_validator=bench.validate_retry_parity)
    start = time.monotonic()
    def stop(*_):
        if _ or time.monotonic()-start > 5*3600-600:
            raise TimeoutError('Retry benchmark incomplete; no production authorization')
    for name in ('SIGTERM', 'SIGUSR1', 'SIGALRM'):
        if hasattr(signal, name): signal.signal(getattr(signal, name), stop)
    if hasattr(signal, 'alarm'): signal.alarm(5*3600-600)
    try:
        return bench.benchmark(campaign, a.task_index, stop, retry_root=a.output,
                               restart_check=parity.full_recipe_restart, retry_provenance=provenance)
    except BaseException as error:
        directory = a.output/'failures'; directory.mkdir(exist_ok=True)
        write_new(directory/f'benchmark_{a.task_index}_{time.time_ns()}.json',
                  dict(phase='benchmark', task_index=a.task_index, complete=False, passed=False,
                       error=f'{type(error).__name__}: {error}', authorizes_training=False))
        raise
    finally:
        if hasattr(signal, 'alarm'): signal.alarm(0)


if __name__ == '__main__':
    args = parse()
    if args.mode == 'submit': submit(args)
    else: execute(args)
