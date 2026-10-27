"""The new campaign may depend on old controls, but must not mutate their jobs."""
import copy

import pytest

from scripts import part_ev_state_compare_dependencies as D
from part_ev_state_capacity import migration as historical
from part_ev_state_fullscale import reporting
from tests.test_part_ev_state_capacity_migration import setup


def test_only_two_controls_read_and_no_mutation(tmp_path):
    root, contract, scheduler = setup(tmp_path)
    # Other seeds, obsolete last-four tasks and their downstream chain are irrelevant.
    for job in ('100_1', '100_2', '100_3', '100_4', '100_5', '101', '102'):
        scheduler.rows[job]['JobState'] = 'FAILED'
    before = copy.deepcopy(scheduler.rows)
    assert D.verifier.kept_dependencies(root, {}, contract, scheduler) == ['100_0', '100_6']
    assert scheduler.rows == before and scheduler.calls == []
    assert historical.P.CONTROL_INDICES == (0, 1, 2, 6)


def test_completed_controls_audited_before_removing_dependencies(tmp_path, monkeypatch):
    root, contract, scheduler = setup(tmp_path)
    scheduler.rows['100_0']['JobState'] = 'COMPLETED'
    scheduler.rows.pop('100_6')
    monkeypatch.setattr(scheduler, 'terminal_state', lambda job: 'COMPLETED')
    calls = []
    monkeypatch.setattr(reporting, 'validate_training', lambda *args: calls.append(args))
    source = {'tasks': list(range(7))}
    assert D.verifier.kept_dependencies(root, source, contract, scheduler) == []
    assert calls == [(root, source, 0, {}), (root, source, 6, {})]
    assert scheduler.calls == []


@pytest.mark.parametrize('problem', ['failed', 'owner', 'command', 'array', 'missing_artifacts'])
def test_control_problems_fail_closed_without_job_changes(tmp_path, monkeypatch, problem):
    root, contract, scheduler = setup(tmp_path)
    row = scheduler.rows['100_0']
    if problem == 'failed': row['JobState'] = 'FAILED'
    if problem == 'owner': row['UserId'] = 'different_user(1)'
    if problem == 'command': row['Command'] = '/wrong/worker.sh'
    if problem == 'array': row['ArrayJobId'] = '999'
    if problem == 'missing_artifacts':
        row['JobState'] = 'COMPLETED'
        def missing(*args): raise FileNotFoundError('Missing selected checkpoint')
        monkeypatch.setattr(reporting, 'validate_training', missing)
    before = copy.deepcopy(scheduler.rows)
    with pytest.raises((ValueError, FileNotFoundError)):
        D.verifier.kept_dependencies(root, {'tasks': list(range(7))}, contract, scheduler)
    assert scheduler.rows == before and scheduler.calls == []


def test_cli_is_read_only_and_prints_only_dependencies(tmp_path, monkeypatch, capsys):
    root, contract, scheduler = setup(tmp_path)
    monkeypatch.setattr(D.P, 'load_campaign', lambda output: {'controls': {'path': str(root)}})
    monkeypatch.setattr(D.P, 'control_contract', lambda output: ({}, contract))
    monkeypatch.setattr(D.verifier, 'Slurm', lambda: scheduler)
    assert D.main(['--output', str(tmp_path)]) == 0
    assert capsys.readouterr().out == '100_0:100_6\n'
    with pytest.raises(SystemExit):
        D.main(['--output', str(tmp_path), '--apply'])
    assert scheduler.calls == [] and not root.exists()
