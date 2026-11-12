import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest
from scripts import submit_part_selected_compact as S
from tests.test_part_fullscale_launcher import BASH, shell_path


def args(tmp, extra=()):
    return S.parse(['--output', str(S.ROOT/'checkpoints/part_selected_compact/test'),
                    '--reference', str(tmp/'reference'), '--recovery', str(tmp/'recovery'),
                    '--python', sys.executable, *extra])


@pytest.fixture
def scheduler(tmp_path, monkeypatch):
    root = tmp_path/'repo'
    root.mkdir()
    for name in (*S.REQUIRED, '__pycache__/old.pyc'):
        p = root/name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('# fixture\n')
    subprocess.run(['git', 'init', str(root)], check=True, capture_output=True)
    subprocess.run(['git', '-C', str(root), 'add', '.'], check=True, capture_output=True)
    subprocess.run(['git', '-C', str(root), '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                    'commit', '-m', 'fixture'], check=True, capture_output=True)
    for name in ('reference/experiment_spec.json', 'recovery/compatibility.json'):
        p = tmp_path/name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('{}')
    monkeypatch.setattr(S, 'ROOT', root)
    monkeypatch.setattr(S.platform, 'machine', lambda: 'aarch64')
    monkeypatch.setenv('SBATCH_GRES', 'gpu:wrong:99')
    monkeypatch.setenv('SLURM_ARRAY_TASK_ID', '88')
    state = dict(jobs=[], tests=[], fail=None, test_fail=False)
    original_command, original_run = S.command, subprocess.run
    def command(argv, **kw):
        if argv[0] == 'sbatch':
            if state['fail'] == len(state['jobs']):
                raise subprocess.CalledProcessError(1, argv)
            job = str(100+len(state['jobs']))
            state['jobs'].append((job, argv, kw['env']))
            return job+';tigris'
        return original_command(argv, **kw)
    def run(argv, **kw):
        if argv[0] == 'sbatch':
            assert '--test-only' in argv
            state['tests'].append(argv)
            if state['test_fail']:
                raise subprocess.CalledProcessError(1, argv)
            return SimpleNamespace(returncode=0)
        return original_run(argv, **kw)
    monkeypatch.setattr(S, 'command', command)
    monkeypatch.setattr(subprocess, 'run', run)
    return state


def test_dry_run_no_files_jobs_or_production(tmp_path, scheduler, capsys):
    a = args(tmp_path, ['--dry-run'])
    S.launch(a)
    assert not a.output.exists() and not scheduler['jobs']
    assert 'Benchmarks ONLY' in capsys.readouterr().out
    for mode in ('train', 'production', 'evaluate', 'resume'):
        with pytest.raises(SystemExit): args(tmp_path, ['--mode', mode])


def test_graph_frozen_source_no_cross_campaign_writes(tmp_path, scheduler):
    a = args(tmp_path)
    S.launch(a)
    assert len(scheduler['tests']) == len(scheduler['jobs']) == 3
    for i, (_, argv, env) in enumerate(scheduler['jobs']):
        assert '--partition=tigris' in argv and '--mem=128G' in argv and '--cpus-per-task=12' in argv
        assert '--time='+('00:30:00' if i == 2 else '05:00:00') in argv
        assert not any(k.startswith(('SBATCH_', 'SLURM_')) for k in env)
        if i == 1:
            assert '--array=0-1%2' in argv and '--gres=gpu:gh200:1' in argv
            assert '--dependency=afterok:100' in argv
        else:
            assert not any(s.startswith('--gres') for s in argv)
        if i == 2: assert '--dependency=afterany:101' in argv
        assert str(a.output/'source/sbatch/run_part_selected_compact_tigris.sh') in argv
    S.verify_snapshot(a.output)
    assert not (a.output/'source/__pycache__/old.pyc').exists()
    (S.ROOT/'part_selected_compact/model.py').write_text('subsequent live pull')
    S.verify_snapshot(a.output)
    assert (a.reference/'experiment_spec.json').read_text() == '{}'
    assert (a.recovery/'compatibility.json').read_text() == '{}'
    assert (a.output/'submitted_job_ids.tsv').read_text().splitlines() == ['prepare\t100', 'benchmark\t101', 'summary\t102']
    with pytest.raises(FileExistsError, match='already exists'): S.launch(a)


def test_uncertain_submission_blocks_blind_retry(tmp_path, scheduler):
    a = args(tmp_path)
    scheduler['fail'] = 1
    with pytest.raises(subprocess.CalledProcessError): S.launch(a)
    assert json.loads((a.output/'submission_prepare.json').read_text())['status'] == 'accepted'
    assert json.loads((a.output/'submission_benchmark.json').read_text())['status'] == 'submitting'
    with pytest.raises(FileExistsError): S.launch(a)


def test_resource_failure_submits_nothing(tmp_path, scheduler):
    scheduler['test_fail'] = True
    with pytest.raises(subprocess.CalledProcessError): S.launch(args(tmp_path))
    assert not scheduler['jobs']


def test_dirty_worktree_and_overlap_fail_closed(tmp_path, scheduler):
    a = args(tmp_path)
    (S.ROOT/'part_selected_compact/model.py').write_text('dirty')
    with pytest.raises(ValueError, match='clean worktree'): S.launch(a)
    assert not a.output.exists() and not scheduler['jobs']
    for path in (S.ROOT, a.reference, a.recovery, S.ROOT/'checkpoints/part_selected_compact'):
        with pytest.raises(SystemExit): args(tmp_path, ['--output', str(path)])


@pytest.mark.parametrize('kind', ['add', 'change', 'rename'])
def test_snapshot_mutation_detected(tmp_path, scheduler, kind):
    a = args(tmp_path)
    S.launch(a)
    path = a.output/'source/part_selected_compact/model.py'
    if kind == 'add': (path.parent/'unexpected.py').write_text('x')
    elif kind == 'change': path.write_text('changed')
    else: path.rename(path.with_suffix('.gone'))
    with pytest.raises(ValueError, match='Frozen source'): S.verify_snapshot(a.output)


@pytest.mark.skipif(BASH is None, reason='Bash unavailable')
def test_shell_syntax():
    root = Path(__file__).resolve().parents[1]
    for name in ('submit', 'run'):
        subprocess.run([BASH, '-n', shell_path(root/f'sbatch/{name}_part_selected_compact_tigris.sh')], check=True)
