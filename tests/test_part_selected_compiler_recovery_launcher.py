import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from types import SimpleNamespace
import pytest
from scripts import submit_part_selected_compiler_recovery as S
from part_selected_compiler_recovery import protocol as P
from part_selected_production import protocol as production
from part_selected_fullscale.protocol import TASKS
from tests.test_part_fullscale_launcher import BASH


@pytest.fixture
def setup(tmp_path, monkeypatch):
    repo = tmp_path/'repo'; repo.mkdir()
    for n in P.FILES:
        path = repo/n; path.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(S.ROOT/n, path)
    subprocess.run(['git', 'init', str(repo)], check=True, capture_output=True)
    subprocess.run(['git', '-C', str(repo), 'add', '.'], check=True, capture_output=True)
    subprocess.run(['git', '-C', str(repo), '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                    'commit', '-m', 'fixture'], check=True, capture_output=True)
    campaign = repo/'checkpoints/part_selected_production/seven'
    campaign.mkdir(parents=True)
    output = repo/'checkpoints/part_selected_compiler_recovery/fix'
    plan = P.hashed(dict(output=str(campaign), python=sys.executable, benchmark=str(tmp_path/'benchmark'),
        account='reu-aisocial', train_hours=list(production.TRAIN_HOURS), evaluate_hours=list(production.EVALUATE_HOURS),
        science=dict(tasks=TASKS), review=dict(storage=dict(required_free_bytes=12*2**30))))
    P.write(campaign/'production.json', plan)
    evidence = dict(production_sha256=plan['content_hash'], artifacts={str(campaign/'production.json'): P.sha(campaign/'production.json')})
    monkeypatch.setattr(S, 'ROOT', repo)
    monkeypatch.setattr(platform, 'machine', lambda: 'aarch64')
    monkeypatch.setattr(shutil, 'disk_usage', lambda _: SimpleNamespace(free=20*2**30))
    monkeypatch.setitem(sys.modules, 'fcntl', SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *a: None))
    real_which = shutil.which
    monkeypatch.setattr(shutil, 'which', lambda n: '/fake/'+n if n in ('sbatch', 'squeue') else real_which(n))
    monkeypatch.setenv('SBATCH_GRES', 'gpu:wrong:99')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '99')
    monkeypatch.setenv('RANK', '99')
    state = dict(jobs=[], tests=[], live='', fail=None, test_fail=False, admitted=False)
    real = S.command
    def command(args, **kwargs):
        if args[0] == 'squeue':
            return state['live']
        if '--phase' in args:
            phase = args[args.index('--phase')+1]
            if phase == 'inspect':
                import json
                return json.dumps(evidence)
            if phase == 'review':
                if not state['admitted']:
                    raise subprocess.CalledProcessError(1, args)
                return '{"passed":true}'
        if args[0] == 'sbatch':
            if '--test-only' in args:
                state['tests'].append(args)
                if state['test_fail']:
                    raise subprocess.CalledProcessError(1, args)
                return 'Valid'
            if state['fail'] == len(state['jobs']):
                raise subprocess.CalledProcessError(1, args)
            job = str(1000+len(state['jobs']))
            state['jobs'].append((args, kwargs['env']))
            return job+';tigris'
        return real(args, **kwargs)
    monkeypatch.setattr(S, 'command', command)
    args = ['--campaign', str(campaign), '--output', str(output)]
    return repo, campaign, output, plan, args, state


def ready(setup):
    repo, campaign, output, plan, args, state = setup
    S.launch(S.parse(['--mode', 'check', *args]))
    state['admitted'] = True
    return campaign, output, plan, state


def test_plan_is_read_only(setup):
    repo, campaign, output, _, args, state = setup
    before = {p: P.sha(p) for p in repo.rglob('*') if p.is_file()}
    S.launch(S.parse(['--mode', 'plan', *args]))
    assert before == {p: P.sha(p) for p in repo.rglob('*') if p.is_file()}
    assert not output.exists() and not state['jobs']


def test_check_is_bounded_cpu_only_and_does_not_touch_production(setup):
    _, campaign, output, _, args, state = setup
    before = {p: P.sha(p) for p in campaign.rglob('*') if p.is_file()}
    S.launch(S.parse(['--mode', 'check', *args]))
    assert len(state['jobs']) == len(state['tests']) == 1
    request, env = state['jobs'][0]
    assert '--time=60' in request and '--mem=128G' in request
    assert not any(a.startswith('--gres') for a in request)
    assert not any(a.startswith('--dependency') for a in request)
    assert 'SBATCH_GRES' not in env and 'CUDA_VISIBLE_DEVICES' not in env and 'RANK' not in env
    assert before == {p: P.sha(p) for p in campaign.rglob('*') if p.is_file()}
    row = P.checked(output/'recovery.json')
    assert row['source'] == {n: P.sha(output/'source'/n) for n in P.FILES}
    assert row['indices'] == [2, 5, 6]


def test_submit_requires_passed_check_and_reviewed_flag(setup):
    _, _, output, _, args, state = setup
    S.launch(S.parse(['--mode', 'check', *args]))
    with pytest.raises(SystemExit):
        S.parse(['--mode', 'submit', '--output', str(output)])
    with pytest.raises(subprocess.CalledProcessError):
        S.launch(S.parse(['--mode', 'submit', '--reviewed', '--output', str(output)]))
    assert len(state['jobs']) == 1


def test_targeted_retry_allows_other_active_jobs_and_waits_for_their_evaluations(setup):
    campaign, output, plan, state = ready(setup)
    S.append_ledger(campaign, 'train_3', '903')
    S.append_ledger(campaign, 'train_4', '904')
    S.append_ledger(campaign, 'evaluate_3', '913')
    S.append_ledger(campaign, 'evaluate_4', '914')
    state['live'] = '903\n904\n913\n914\n922343\n922345'
    for i in (0, 1):
        P.write(S.run_directory(campaign, TASKS[i])/'full_validation/metrics.json', {})
    snapshot = P.sha(campaign/'production.json')
    S.launch(S.parse(['--mode', 'submit', '--reviewed', '--output', str(output)]))
    assert len(state['jobs']) == 8  # One check, three training/evaluation pairs, final report.
    assert len(state['tests']) == 8
    for j, i in enumerate(P.INDICES):
        train, _ = state['jobs'][1+2*j]
        evaluate, _ = state['jobs'][2+2*j]
        assert f'--job-name=sel7recover_train_{i}' in train
        assert f'--time={plan["train_hours"][i]*60}' in train
        assert '--gres=gpu:gh200:1' in train
        assert not any(a.startswith('--dependency') for a in train)  # No concurrency cap.
        assert '--dependency=afterok:'+str(1001+2*j) in evaluate
        assert '--kill-on-invalid-dep=yes' in train and '--signal=B:USR1@900' in train
    final, _ = state['jobs'][-1]
    assert '--dependency=afterok:913:914:1002:1004:1006' in final
    assert P.sha(campaign/'production.json') == snapshot
    assert ('train_2', '1001') in S.ledger(campaign)
    assert ('train_3', '903') in S.ledger(campaign)


@pytest.mark.parametrize('label', ['train_2', 'evaluate_5', 'train_6', 'report_None'])
def test_duplicate_targets_or_live_report_rejected(setup, label):
    campaign, output, _, state = ready(setup)
    S.append_ledger(campaign, label, '908'); state['live'] = '908'
    with pytest.raises(ValueError, match='active/pending'):
        S.launch(S.parse(['--mode', 'submit', '--reviewed', '--output', str(output)]))
    assert len(state['jobs']) == 1


def test_resource_failure_submits_no_production_jobs(setup):
    _, output, _, state = ready(setup); state['test_fail'] = True
    with pytest.raises(subprocess.CalledProcessError):
        S.launch(S.parse(['--mode', 'submit', '--reviewed', '--output', str(output)]))
    assert len(state['jobs']) == 1


def test_partial_submission_is_recorded_and_ambiguity_blocks_both_launchers(setup):
    campaign, output, _, state = ready(setup); state['fail'] = 2
    with pytest.raises(subprocess.CalledProcessError):
        S.launch(S.parse(['--mode', 'submit', '--reviewed', '--output', str(output)]))
    assert ('train_2', '1001') in S.ledger(campaign)
    assert ('train_2', '1001') in S.ledger(output)
    for directory in (campaign, output):
        assert P.read(directory/'submission_intent.json')['status'] == 'unknown'
    with pytest.raises(ValueError, match='Unresolved sbatch'):
        S.launch(S.parse(['--mode', 'submit', '--reviewed', '--output', str(output)]))
    assert len(state['jobs']) == 2


def test_missing_other_evaluations_defer_report_not_retrain(setup, capsys):
    campaign, output, _, state = ready(setup)
    S.launch(S.parse(['--mode', 'submit', '--reviewed', '--output', str(output)]))
    assert len(state['jobs']) == 7
    assert 'Final report deferred' in capsys.readouterr().out
    with pytest.raises(ValueError, match='evaluations missing'):
        S.launch(S.parse(['--mode', 'report', '--reviewed', '--output', str(output)]))


def test_resume_skips_completed_training_and_only_schedules_missing_selected_stages(setup):
    campaign, output, _, state = ready(setup)
    for i in (2, 5):
        P.write(S.run_directory(campaign, TASKS[i])/'training_complete.json', {})
    P.write(S.run_directory(campaign, TASKS[2])/'full_validation/metrics.json', {})
    S.launch(S.parse(['--mode', 'submit', '--reviewed', '--output', str(output)]))
    names = [next(a for a in request if a.startswith('--job-name=')) for request, _ in state['jobs'][1:]]
    assert names == ['--job-name=sel7recover_evaluate_5', '--job-name=sel7recover_train_6', '--job-name=sel7recover_evaluate_6']


def test_storage_gate_is_not_bypassed(setup, monkeypatch):
    _, output, _, state = ready(setup)
    monkeypatch.setattr(shutil, 'disk_usage', lambda _: SimpleNamespace(free=1))
    with pytest.raises(OSError, match='headroom'):
        S.launch(S.parse(['--mode', 'submit', '--reviewed', '--output', str(output)]))
    assert len(state['jobs']) == 1


@pytest.mark.skipif(not BASH, reason='Bash unavailable')
def test_shell_syntax():
    for name in ('run_part_selected_compiler_recovery.sh', 'submit_part_selected_compiler_recovery.sh'):
        subprocess.run([BASH, '-n', str(S.ROOT/'sbatch'/name)], check=True, capture_output=True)


@pytest.mark.parametrize('args', [[], ['--mode', 'check', '--output', 'relative'],
    ['--mode', 'submit', '--reviewed', '--output', '/'],
    ['--mode', 'check', '--output', '/tmp/recovery']])
def test_invalid_scope_rejected(args):
    with pytest.raises(SystemExit):
        S.parse(args)
