"""Walltime-only recovery: original evidence and scientific gates stay intact."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts import recover_part_selected_tigris_walltime as R
from part_selected_tigris import protocol as P
from tests.test_part_selected_tigris import benchmark_fixture


def campaign_fixture(output):
    output.mkdir(parents=True, exist_ok=True)
    spec, rows = benchmark_fixture(output)
    spec.update(schema=P.SCHEMA, tasks=copy.deepcopy(P.TASKS))
    spec.pop('content_hash')
    spec = P.hashed(spec)
    P.write_json(output/'experiment_spec.json', spec)
    P.write_json(output/'launch.json', spec['launch'])
    (output/'launch.env').write_text('# fixture\n')
    (output/'source.sha256').write_text('# fixture\n')
    for task, row, hours in zip(spec['tasks'], rows, (99.65, 99.65, 112.78, 114.66)):
        row['campaign_sha256'] = spec['content_hash']
        row['runtime']['campaign_sha256'] = spec['content_hash']
        row['projected_training_hours'] = hours
        P.write_json(output/'benchmarks'/f"{task['run_id']}.json", row)
    recovery = output/'recoveries/walltime_168h_12h'
    return spec, recovery


def byte_snapshot(output):
    return {p.relative_to(output).as_posix(): p.read_bytes() for p in output.rglob('*')
            if p.is_file() and 'recoveries' not in p.relative_to(output).parts}


def test_dry_run_and_recovery_preserve_campaign_and_benchmarks(tmp_path):
    spec, recovery = campaign_fixture(tmp_path)
    before = byte_snapshot(tmp_path)
    with pytest.raises(ValueError, match='frozen walltime'):
        P.require_approval(tmp_path, spec)
    request = R.configure(tmp_path, recovery, 168, 12, '7-00:00:00', dry_run=True)
    assert not recovery.exists()
    assert request['estimates'][-1]['train_hours_with_margin'] == pytest.approx(149.058)
    recovery.mkdir(parents=True)
    R.configure(tmp_path, recovery, 168, 12, '7-00:00:00')
    assert R.load_request(tmp_path, recovery) == request
    original = P.require_approval
    original_spec = copy.deepcopy(spec)
    with R.admission_overlay(P, tmp_path, spec, request) as approve:
        assert len(approve(tmp_path, spec)) == 4
        assert len(P.training_runtime(tmp_path, spec)) > 0
    assert P.require_approval is original
    assert spec == original_spec and byte_snapshot(tmp_path) == before
    # An unmodified caller still rejects the original 96h budget: no global bypass.
    with pytest.raises(ValueError, match='frozen walltime'):
        P.require_approval(tmp_path, spec)


@pytest.mark.parametrize('train,evaluate,max_time', [
    (96, 12, 'UNLIMITED'), (90, 13, 'UNLIMITED'), (168, 11, 'UNLIMITED'),
    (148, 12, 'UNLIMITED'), (168, 12, '6-00:00:00'), (168, 12, '(null)'),
    (True, 12, 'UNLIMITED'), (168, 0, 'UNLIMITED')])
def test_bad_or_insufficient_resource_request_writes_nothing(tmp_path, train, evaluate, max_time):
    _, recovery = campaign_fixture(tmp_path)
    before = byte_snapshot(tmp_path)
    with pytest.raises(ValueError):
        R.configure(tmp_path, recovery, train, evaluate, max_time, dry_run=True)
    assert not recovery.exists() and byte_snapshot(tmp_path) == before


def test_evaluation_budget_is_checked_too(tmp_path):
    _, recovery = campaign_fixture(tmp_path)
    path = tmp_path/'benchmarks/SELECTED_PART_SHARED.json'
    row = R.read(path)
    row['validation']['seconds'] = 1000
    P.write_json(path, row)
    with pytest.raises(ValueError, match='evaluation'):
        R.configure(tmp_path, recovery, 168, 12, 'UNLIMITED', dry_run=True)
    assert R.configure(tmp_path, recovery, 168, 16, 'UNLIMITED', dry_run=True)['walltimes']['evaluate_hours'] == 16


@pytest.mark.parametrize('change', ['failed', 'wrong_seed', 'wrong_runtime', 'wrong_stream',
    'no_parity', 'failed_parity', 'wrong_parity', 'tf32', 'nan_train', 'nan_eval'])
def test_recovery_retains_every_original_numerical_runtime_input_gate(tmp_path, change):
    spec, recovery = campaign_fixture(tmp_path)
    request = R.configure(tmp_path, recovery, 168, 12, 'UNLIMITED', dry_run=True)
    path = tmp_path/'benchmarks/SELECTED_EV_LAYERWISE.json'
    row = R.read(path)
    if change == 'failed': row['passed'] = False
    if change == 'wrong_seed': row['task']['seed'] = 202
    if change == 'wrong_runtime': row['runtime']['gpu'] = 'A100'
    if change == 'wrong_stream': row['input_fingerprints'] = {'0': 'changed'}
    if change == 'no_parity': row.pop('initial_parity')
    if change == 'failed_parity': row['initial_parity']['memory_execution']['passed'] = False
    if change == 'wrong_parity': row['initial_parity']['memory_execution']['policy'] = {}
    if change == 'tf32': row['runtime']['tf32'] = True
    if change == 'nan_train': row['projected_training_hours'] = float('nan')
    if change == 'nan_eval': row['validation']['seconds'] = float('nan')
    original_reader = P.read_json
    with R.admission_overlay(P, tmp_path, spec, request) as approve:
        with patch.object(P, 'read_json', side_effect=lambda p: row if Path(p) == path else original_reader(p)):
            with pytest.raises(ValueError): approve(tmp_path, spec)


@pytest.mark.parametrize('name', ['launch.env', 'launch.json', 'experiment_spec.json', 'source.sha256',
    'benchmarks/SELECTED_PART_SHARED.json', 'benchmarks/SELECTED_EV_SHARED.json',
    'benchmarks/SELECTED_PART_LAYERWISE.json', 'benchmarks/SELECTED_EV_LAYERWISE.json'])
def test_request_rejects_changed_original_evidence(tmp_path, name):
    _, recovery = campaign_fixture(tmp_path)
    recovery.mkdir(parents=True)
    R.configure(tmp_path, recovery, 168, 12, 'UNLIMITED')
    path = tmp_path/name
    path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError): R.load_request(tmp_path, recovery)


def test_overlay_rejects_scientific_or_campaign_changes_and_restores_on_error(tmp_path):
    spec, recovery = campaign_fixture(tmp_path)
    request = R.configure(tmp_path, recovery, 168, 12, 'UNLIMITED', dry_run=True)
    original = P.require_approval
    with pytest.raises(RuntimeError):
        with R.admission_overlay(P, tmp_path, spec, request) as approve:
            changed = copy.deepcopy(spec)
            changed['tasks'][0]['seed'] = 202
            with pytest.raises(ValueError, match='scientific'): approve(tmp_path, changed)
            with pytest.raises(ValueError, match='scientific'): approve(tmp_path/'other', spec)
            raise RuntimeError('worker failed')
    assert P.require_approval is original


def test_frozen_recovery_limits_and_path(tmp_path):
    _, recovery = campaign_fixture(tmp_path)
    recovery.mkdir(parents=True)
    R.configure(tmp_path, recovery, 168, 12, 'UNLIMITED')
    with pytest.raises(ValueError, match='frozen'):
        R.configure(tmp_path, recovery, 192, 12, 'UNLIMITED')
    with pytest.raises(ValueError, match='child'):
        R.configure(tmp_path, tmp_path/'source/recovery', 168, 12, 'UNLIMITED', dry_run=True)
    request = R.read(recovery/'request.json')
    request['walltimes']['batch_size'] = 256
    P.write_json(recovery/'request.json', R.hashed(request))
    with pytest.raises(ValueError, match='Only training/evaluation'):
        R.load_request(tmp_path, recovery)


def test_worker_uses_frozen_driver_and_requires_approval_before_training(tmp_path, monkeypatch):
    spec, recovery = campaign_fixture(tmp_path)
    recovery.mkdir(parents=True)
    R.configure(tmp_path, recovery, 168, 12, 'UNLIMITED')
    original = P.require_approval
    calls = []
    fake = SimpleNamespace(__file__=tmp_path/'source/part_selected_tigris/protocol.py',
        load_campaign=lambda output: spec, require_approval=original,
        hashed=P.hashed, write_json=P.write_json, file_lock=P.file_lock)
    monkeypatch.setattr(R.importlib, 'import_module', lambda name: fake)
    monkeypatch.setattr(R.sys, 'path', R.sys.path.copy())
    def execute(path, run_name):
        calls.append((path, run_name, list(R.sys.argv)))
        assert fake.require_approval is not original
        assert len(fake.require_approval(tmp_path, spec)) == 4
        raise SystemExit(0)
    monkeypatch.setattr(R.runpy, 'run_path', execute)
    with pytest.raises(ValueError, match='must approve'):
        R.execute(tmp_path, recovery, 'train', 0)
    assert not calls
    R.execute(tmp_path, recovery, 'gate')
    assert R.read(recovery/'approval.json')['passed'] is True
    for phase in ('train', 'lock', 'evaluate', 'report'):
        with pytest.raises(SystemExit) as error:
            R.execute(tmp_path, recovery, phase, 3 if phase in ('train', 'evaluate') else None)
        assert error.value.code == 0
        assert fake.require_approval is original
        assert calls[-1][0] == str(tmp_path/'source/scripts/run_part_selected_tigris.py')
        assert calls[-1][2][1:5] == ['--phase', phase, '--output', str(tmp_path)]
    with pytest.raises(ValueError): R.execute(tmp_path, recovery, 'benchmark')
    with pytest.raises(ValueError): R.execute(tmp_path, recovery, 'train', 4)
    approval = R.read(recovery/'approval.json')
    approval['walltimes']['train_hours'] = 999
    P.write_json(recovery/'approval.json', approval)
    with pytest.raises(ValueError, match='approval changed'): R.execute(tmp_path, recovery, 'report')
