"""Candidate-only submission against old frozen workers; no Slurm/GPU required."""
import copy
from pathlib import Path
import subprocess
import sys

import pytest

from part_ev_multiscale import protocol as P, submission as S
from part_ev_multiscale.config import PLATFORMS
from scripts import submit_part_ev_multiscale as CLI
from scripts import submit_part_ev_multiscale_candidate as C
from tests.test_part_ev_multiscale_launcher import setup


@pytest.fixture
def ready(setup, monkeypatch):
    repo, output, original, calls, state = setup
    launch = dict(original, platform='tigris', resources=PLATFORMS['tigris'])
    plan = P.hashed(dict(models=[dict(arm='control', train_hours=171, evaluate_hours=2),
                               dict(arm='multiscale', train_hours=358, evaluate_hours=3)]))
    monkeypatch.setattr(P, 'load_launch', lambda *a, **k: launch)
    monkeypatch.setattr(P, 'production_spec', lambda *a: copy.deepcopy(plan))
    monkeypatch.setattr(P, 'load_production', lambda *a, **k: copy.deepcopy(plan))
    monkeypatch.setattr(P, 'check_storage', lambda *a, **k: 12 * 2**30)
    return repo, output, launch, plan, calls, state


def test_preview_is_read_only_and_uses_existing_measured_resources(ready):
    _, output, _, _, calls, _ = ready
    before = {p.relative_to(output): p.read_bytes() for p in output.rglob('*') if p.is_file()}
    result = C.preview(output, P, S)
    assert not calls
    assert result['control_submitted'] is result['paired_report_submitted'] is False
    train, evaluate = result['requests']
    assert '--time=21480' in train and '--time=180' in evaluate
    assert '--gres=gpu:gh200:1' in train and '--gres=gpu:gh200:1' in evaluate
    assert train[-3:] == [str(output), 'train', 'multiscale']
    assert evaluate[-3:] == [str(output), 'evaluate', 'multiscale']
    after = {p.relative_to(output): p.read_bytes() for p in output.rglob('*') if p.is_file()}
    assert before == after


def test_dispatch_two_jobs_keeps_science_frozen_and_is_idempotent(ready):
    _, output, launch, plan, calls, _ = ready
    before = P.inventory(output / 'source')
    launch_bytes = (output / 'launch.json').read_bytes()
    result = C.dispatch(output, P, S)
    assert result['jobs'] == {'train_multiscale': '1001', 'evaluate_multiscale': '1002'}
    assert len(calls) == 2
    assert not any(v.startswith('--dependency=') for v in calls[0][0])
    assert '--dependency=afterok:1001' in calls[1][0]
    assert all(v[0][-1] == 'multiscale' for v in calls)
    assert all(str(output / 'source/sbatch/run_part_ev_multiscale.sh') in v[0] for v in calls)
    assert P.checked(output / 'production.json') == plan
    assert P.checked(output / 'production/campaign.json') == plan
    assert P.checked(output / 'candidate_dispatch.json')['arms'] == ['multiscale']
    assert P.inventory(output / 'source') == before
    assert (output / 'launch.json').read_bytes() == launch_bytes
    assert C.dispatch(output, P, S)['status'] == 'already_submitted'
    assert len(calls) == 2
    # A stale automatic paired dispatcher cannot add a control afterward.
    with pytest.raises(ValueError, match='Partial production chain'):
        S.dispatch(output)
    assert len(calls) == 2


@pytest.mark.parametrize('failure', ['benchmark', 'storage', 'preflight'])
def test_admission_failures_do_not_submit_or_write_production(ready, monkeypatch, failure):
    _, output, _, _, calls, state = ready
    def fail(*a, **k):
        raise ValueError('blocked admission')
    if failure == 'benchmark': monkeypatch.setattr(P, 'production_spec', fail)
    elif failure == 'storage': monkeypatch.setattr(P, 'check_storage', fail)
    else: state['fail'] = True
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        C.dispatch(output, P, S)
    assert not calls
    assert not (output / 'production.json').exists()
    assert not (output / 'candidate_dispatch.json').exists()


def test_paired_or_foreign_owner_is_not_overwritten(ready):
    _, output, launch, _, calls, _ = ready
    owner = C.owner_path(output, launch, S)
    P.write(owner, dict(output='different campaign'))
    with pytest.raises(ValueError, match='Existing production artifacts'):
        C.dispatch(output, P, S)
    assert not calls


def test_existing_paired_chain_is_rejected(ready):
    _, output, _, _, calls, _ = ready
    S.dispatch(output)
    assert len(calls) == 5
    with pytest.raises(ValueError, match='candidate-only chain'):
        C.dispatch(output, P, S)
    assert len(calls) == 5


@pytest.mark.parametrize('failure', ['ambiguous', 'partial'])
def test_failed_submission_does_not_duplicate_on_retry(ready, monkeypatch, failure):
    _, output, _, _, calls, state = ready
    if failure == 'ambiguous':
        state['ambiguous'] = True
    else:
        original = S.submit_one
        def submit(*args, **kwargs):
            if args[2] == 'evaluate': raise RuntimeError('interrupted before evaluate')
            return original(*args, **kwargs)
        monkeypatch.setattr(S, 'submit_one', submit)
    with pytest.raises((ValueError, RuntimeError)):
        C.dispatch(output, P, S)
    count = len(calls)
    with pytest.raises(ValueError): C.dispatch(output, P, S)
    assert len(calls) == count == 1


@pytest.mark.parametrize('artifact', ['production.json', 'production/runs/unexpected',
                                    'candidate_dispatch.json', 'production_job_ids.tsv'])
def test_orphan_admission_artifacts_fail_closed(ready, artifact):
    _, output, _, _, calls, _ = ready
    path = output / artifact
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{}')
    with pytest.raises(ValueError, match='Existing production artifacts'):
        C.dispatch(output, P, S)
    assert not calls


def test_resume_only_candidate_after_terminal_attempt(ready, monkeypatch):
    _, output, _, _, calls, _ = ready
    C.dispatch(output, P, S)
    monkeypatch.setattr(S, 'checkpoint_for_retry', lambda *a: False)
    monkeypatch.setattr(S, 'job_states', lambda jobs: {j: 'RUNNING' for j in jobs})
    with pytest.raises(ValueError, match='active'): C.resume(output, P, S)
    assert len(calls) == 2
    monkeypatch.setattr(S, 'job_states', lambda jobs: {j: 'CANCELLED' for j in jobs})
    result = C.resume(output, P, S)
    assert result['jobs'] == {'train_multiscale': '1003', 'evaluate_multiscale': '1004'}
    assert '--dependency=afterok:1003' in calls[3][0]
    assert all(row[0][-1] == 'multiscale' for row in calls)
    # Once training completed, retry only evaluation, still no control/report.
    monkeypatch.setattr(S, 'checkpoint_for_retry', lambda *a: True)
    result = C.resume(output, P, S)
    assert result['jobs'] == {'evaluate_multiscale': '1005'}
    assert not any(v.startswith('--dependency=') for v in calls[4][0])


def test_resume_requires_complete_accounting_and_no_finished_metrics(ready, monkeypatch):
    _, output, _, _, calls, _ = ready
    C.dispatch(output, P, S)
    monkeypatch.setattr(S, 'job_states', lambda jobs: {})
    with pytest.raises(ValueError, match='accounting incomplete'): C.resume(output, P, S)
    monkeypatch.setattr(S, 'job_states', lambda jobs: {j: 'COMPLETED' for j in jobs})
    P.write(P.run_directory(output, 'multiscale') / 'full_validation/metrics.json', {})
    with pytest.raises(ValueError, match='metrics already exist'): C.resume(output, P, S)
    assert len(calls) == 2


def test_scope_tampering_blocks_more_jobs(ready):
    _, output, _, _, calls, _ = ready
    C.dispatch(output, P, S)
    saved = P.checked(output / 'candidate_dispatch.json')
    saved.pop('content_hash'); saved['arms'] = ['control']
    P.write(output / 'candidate_dispatch.json', P.hashed(saved))
    with pytest.raises(ValueError, match='scope differs'): C.dispatch(output, P, S)
    assert len(calls) == 2


def test_cli_bootstraps_new_controller_not_old_missing_entrypoint(ready, monkeypatch):
    repo, output, launch, _, _, _ = ready
    monkeypatch.setattr(CLI, 'ROOT', repo)
    commands = []
    monkeypatch.setattr(S, 'command', lambda argv, **k: commands.append((argv, k)) or '{}')
    for mode in ('plan-candidate', 'dispatch-candidate', 'resume-candidate'):
        args = ['--mode', mode, '--output', str(output)]
        if mode != 'plan-candidate':
            with pytest.raises(SystemExit): CLI.parse(args)
            args.append('--reviewed')
        CLI.launch(CLI.parse(args))
        argv, _ = commands[-1]
        assert argv[:3] == [launch['python'], '-B', repo / 'scripts/submit_part_ev_multiscale_candidate.py']
        assert mode.removesuffix('-candidate') in argv
    with pytest.raises(SystemExit):
        CLI.parse(['--mode', 'dispatch-candidate', '--output', str(output), '--reviewed', '--arm', 'control'])
    with pytest.raises(SystemExit):
        CLI.parse(['--mode', 'plan-candidate', '--output', str(output), '--platform', 'tigris'])


def test_controller_requires_authority_and_refuses_live_imports(tmp_path):
    for mode in ('dispatch', 'resume'):
        with pytest.raises(SystemExit): C.parse(['--mode', mode, '--output', str(tmp_path)])
    (tmp_path / 'source').mkdir()
    previous = list(sys.path)
    try:
        with pytest.raises(ValueError, match='frozen campaign'): C.frozen_modules(tmp_path)
    finally:
        sys.path[:] = previous


def test_fresh_subprocess_imports_only_old_snapshot(tmp_path):
    package = tmp_path / 'source/part_ev_multiscale'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text('')
    (package / 'protocol.py').write_text('def load_launch(output, *, worker):\n    assert worker\n')
    (package / 'submission.py').write_text('')
    program = '''import importlib.util, pathlib, sys
spec = importlib.util.spec_from_file_location('candidate_controller', sys.argv[1])
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)
p, s = controller.frozen_modules(pathlib.Path(sys.argv[2]).resolve())
print(p.__file__)
print(s.__file__)
'''
    result = subprocess.run([sys.executable, '-B', '-c', program, C.__file__, str(tmp_path)],
                            capture_output=True, text=True, check=True)
    assert str(package / 'protocol.py') in result.stdout
    assert str(package / 'submission.py') in result.stdout
