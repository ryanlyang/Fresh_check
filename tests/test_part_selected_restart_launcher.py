import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from types import SimpleNamespace
import pytest
from scripts import submit_part_selected_restart as S
from scripts import diagnose_part_selected_restart as D
from tests.test_part_fullscale_launcher import BASH, shell_path


@pytest.fixture
def campaign(tmp_path):
    root = tmp_path/'campaign'
    (root/'source').mkdir(parents=True)
    (root/'source/model.py').write_text('# frozen original\n')
    (root/'snapshot.json').write_text(json.dumps({'model.py': D.sha(root/'source/model.py')}))
    launch = dict(partition='tigris', account='reu-aisocial', python=sys.executable)
    (root/'launch.json').write_text(json.dumps(launch))
    (root/'experiment_spec.json').write_text(json.dumps(dict(schema='selected_fullscale_benchmarks_v1',
                              launch=launch, content_hash='frozen', authorizes_training=False)))
    return root


def test_dry_run_no_writes(campaign, capsys):
    before = {p: D.sha(p) for p in campaign.rglob('*') if p.is_file()}
    S.launch(S.parse(['--campaign', str(campaign), '--dry-run']))
    assert '02:00:00' in capsys.readouterr().out
    assert {p: D.sha(p) for p in campaign.rglob('*') if p.is_file()} == before
    assert not (campaign/'diagnostics').exists()
    (campaign/'source/model.py').write_text('changed')
    with pytest.raises(ValueError, match='Frozen source'):
        S.launch(S.parse(['--campaign', str(campaign), '--dry-run']))


def test_one_two_hour_gpu_job_frozen_sources_duplicate_guard(campaign, tmp_path, monkeypatch):
    repo = tmp_path/'repo'
    repo.mkdir()
    for name in S.FILES:
        target = repo/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(S.ROOT/name, target)
    subprocess.run(['git', 'init', str(repo)], check=True, capture_output=True)
    subprocess.run(['git', '-C', str(repo), 'add', '.'], check=True, capture_output=True)
    subprocess.run(['git', '-C', str(repo), '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                    'commit', '-m', 'fixture'], check=True, capture_output=True)
    monkeypatch.setattr(S, 'ROOT', repo)
    monkeypatch.setattr(platform, 'machine', lambda: 'aarch64')
    monkeypatch.setattr(shutil, 'disk_usage', lambda _: SimpleNamespace(free=18*2**30))
    real_which = shutil.which
    monkeypatch.setattr(shutil, 'which', lambda n: '/fake/'+n if n in ('sbatch', 'squeue') else real_which(n))
    monkeypatch.setitem(sys.modules, 'fcntl', SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *a: None))
    monkeypatch.setenv('SBATCH_GRES', 'gpu:wrong:99')
    real_command = S.command
    state = dict(jobs=[], live='')
    def command(argv, **kwargs):
        if argv[0] == 'squeue':
            return state['live']
        if argv[0] == 'sbatch':
            state['jobs'].append((argv, kwargs['env']))
            return '987;cluster'
        return real_command(argv, **kwargs)
    monkeypatch.setattr(S, 'command', command)
    before = {p: D.sha(p) for p in campaign.rglob('*') if p.is_file()}
    S.launch(S.parse(['--campaign', str(campaign)]))
    assert len(state['jobs']) == 1
    argv, env = state['jobs'][0]
    assert '--time=02:00:00' in argv and '--gres=gpu:gh200:1' in argv
    assert '--mem=128G' in argv and '--cpus-per-task=12' in argv
    assert not any(a.startswith(('--array', '--dependency')) for a in argv)
    assert not any(k.startswith('SBATCH_') for k in env)
    output = Path(argv[-1])
    assert D.plan(output)['policy'] == D.POLICY
    assert all(D.sha(p) == digest for p, digest in before.items())
    report = D.review(output)
    assert not report['complete'] and not report['authorizes_benchmark_retry']
    # Collecting evidence successfully must never turn a mismatch into approval.
    for index in D.INDICES:
        D.write_new(output/f'task_{index}.json', dict(complete=True, original_gate={'passed': False},
                    exact_roundtrip=True, resumed_exact=False, uninterrupted_repeat_exact=False))
    report = D.review(output)
    assert report['complete'] and not report['authorizes_training']
    assert not report['authorizes_benchmark_retry']
    assert not report['models'][0]['original_gate_passed']
    state['live'] = '987'
    with pytest.raises(ValueError, match='already active'):
        S.launch(S.parse(['--campaign', str(campaign)]))
    assert len(state['jobs']) == 1
    (output/'diagnose_part_selected_restart.py').write_text('# modified')
    with pytest.raises(ValueError, match='Diagnostic source changed'):
        D.plan(output)


@pytest.mark.skipif(BASH is None, reason='Bash unavailable')
def test_shell_syntax():
    for name in ('submit', 'run'):
        subprocess.run([BASH, '-n', shell_path(S.ROOT/f'sbatch/{name}_part_selected_restart.sh')], check=True)
