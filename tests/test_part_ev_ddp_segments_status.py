"""Read-only viewer tests: synthetic logs only, no cluster/model dependencies."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import status_part_ev_ddp_segments as S


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')
    return path


def log(path, *rows, mtime=1000):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(json.dumps(row) for row in rows) + '\n', encoding='utf-8')
    os.utime(path, (mtime, mtime))
    return path


@pytest.fixture
def campaign(tmp_path):
    root = tmp_path / 'campaign'
    recipe = dict(total_steps=1000000, steps_per_epoch=20000, global_batch_size=512,
                  validation_events_per_epoch=1280000)
    task = dict(run_id='EV_STATE_LAYERWISE_RANGER', architecture='state_layerwise',
                seed=101, recipe='full')
    put(root / 'campaign.json', dict(task=task, recipes={'full': recipe}))
    run = root / 'runs/EV_STATE_LAYERWISE_RANGER/seed_101'
    for i in range(1, 23):
        put(run / 'validation' / f'check_{i:02d}.json',
            dict(check=i, step=i*20000, accuracy=.80+i*.001,
                 macro_auc_ovo=.98+i*.0001, seconds=1026))
    # Not valid torch files: the viewer must not attempt to deserialize them.
    (run / 'last.pt').write_bytes(b'never open or load this checkpoint')
    chain = root / 'segment_chains/chain1'
    put(chain / 'plan.json', {'initial': {'step': 0}})
    put(chain / 'receipts/segment_00.json', {'after': {'step': 459000}})
    (chain / 'submitted_job_ids.tsv').write_text('segment\t11\nsegment\t12\nsegment\t13\n')
    old = log(chain / 'logs/evseg_segment_11.out',
              dict(stage='training', step=458900, global_jets_per_second=1000), mtime=1001)
    return root, run, chain, old, recipe


def test_running_segment_wins_over_old_log_and_uses_shared_history(campaign):
    root, run, chain, old, _ = campaign
    current = log(chain / 'logs/evseg_segment_12.out',
                  dict(stage='training', step=459000, global_jets_per_second=1024), mtime=1000)
    before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    result = S.report(root, None, {'12': 'RUNNING', '13': 'PENDING'}, now=1002)
    assert f'Log: {current}' in result
    assert '459,000/1,000,000 (45.90%)' in result
    assert '19,000/20,000 (95.0%); 1,000 updates left' in result
    assert 'starts: 8.3 minutes' in result
    assert 'result: 25.4 minutes' in result
    assert 'Completed validation checks: 22/50' in result
    assert '82.2000%' in result
    assert '0.982200' not in result
    assert 'auc' not in result.lower()
    assert result.index('=== Running State-EV progress ===') > result.index('Read-only display')
    assert result.endswith('Approx. time until the result: 25.4 minutes.')
    assert {p: p.read_bytes() for p in root.rglob('*') if p.is_file()} == before


def test_replay_batches_not_training_steps_and_no_misleading_eta(campaign):
    root, _, chain, _, _ = campaign
    log(chain / 'logs/evseg_segment_12.out',
        dict(stage='resume_data_replay', stream='train', batches=120000, total=459000))
    result = S.report(root, None, {'12': 'RUNNING'}, now=1000)
    assert '459,000/1,000,000' in result
    assert '120,000/459,000 batches (26.1%)' in result
    assert 'ETA withheld during replay' in result
    assert 'Approx. time until' not in result


def test_running_job_without_log_does_not_use_previous_segment_rate(campaign):
    root, _, chain, old, _ = campaign
    result = S.report(root, None, {'12': 'RUNNING'}, now=1002)
    assert 'Job: 12 | Scheduler: RUNNING' in result
    assert 'has not created its stdout log yet' in result
    assert '459,000/1,000,000' in result
    assert 'Approx. time until' not in result
    assert f'Log: {old}' not in result


@pytest.mark.parametrize('queue,now', [({}, 1002), (None, 1002), ({'11': 'RUNNING'}, 100000)])
def test_no_live_eta_for_stale_or_not_running_logs(campaign, queue, now):
    root, _, _, _, _ = campaign
    result = S.report(root, None, queue, now=now)
    assert 'Live ETA unavailable' in result
    assert 'Approx. time until' not in result


def test_validation_boundary_and_partial_last_log_line(campaign):
    root, _, chain, _, _ = campaign
    path = log(chain / 'logs/evseg_segment_12.out',
               dict(stage='training', step=460000, global_jets_per_second=1024))
    with path.open('a') as handle:
        handle.write('{"stage": "vali')
    result = S.report(root, None, {'12': 'RUNNING'}, now=1000)
    assert '(100.0%); 0 updates left' in result
    assert 'Validation is due or underway' in result
    assert 'Approx. time until' not in result


def test_completed_run_has_no_next_check_eta(campaign):
    root, run, _, _, _ = campaign
    put(run / 'training_complete.json', {'step': 1000000})
    result = S.report(root, None, {}, now=1000)
    assert '1,000,000/1,000,000 (100.00%)' in result
    assert 'Training completion marker present' in result
    assert 'Next validation:' not in result


def test_controls_history_and_common_cutoff_ignores_later_best(campaign, tmp_path):
    root, _, _, _, recipe = campaign
    controls = tmp_path / 'controls'
    put(controls / 'campaign.json', dict(recipe=recipe, tasks=[
        dict(architecture='part', seed=101), dict(architecture='part_ev', seed=101)]))
    for name in ('part', 'part_ev'):
        for i in range(1, 24):
            put(controls / f'runs/{name}/seed_101/validation/epoch_{i:02d}.json',
                dict(step=20000*i, accuracy=.9 if i == 23 else .8))
    result = S.report(root, controls, {}, now=1002)
    assert 'ParT-EV' in result
    assert 'Best through common check 22' in result
    assert '90.0000% at check 23' in result
    common = result.split('=== Best through common check 22')[1]
    assert '90.0000%' not in common
    assert '+2.2000 pp' in common
    assert '80.0000%' in result
    assert 'auc' not in result.lower()
    assert result.index('=== Running State-EV progress ===') > result.index('Best through common check 22')


def test_missing_controls_warns_without_hiding_progress(campaign, tmp_path):
    result = S.report(campaign[0], tmp_path / 'missing', {}, now=1000)
    assert 'Control campaign missing' in result
    assert 'Completed validation checks: 22/50' in result
    assert result.index('Control campaign missing') < result.index('=== Running State-EV progress ===')


def test_each_check_has_signed_state_minus_control_deltas_not_best_so_far():
    models = [
        ('ParT', None, {1: {'accuracy': .84}, 2: {'accuracy': .81}}),
        ('ParT-EV', None, {1: {'accuracy': .82}, 2: {'accuracy': .83}}),
        ('State-EV', None, {1: {'accuracy': .83}, 2: {'accuracy': .83}}),
    ]
    # The history/summary helpers expect the check ordinal in each row too.
    for _, _, rows in models:
        for i, row in rows.items():
            row['check'] = i
    lines = S.history_lines(models, 20000)
    assert 'Delta ParT (pp)' in lines[1]
    assert 'Delta ParT-EV (pp)' in lines[1]
    assert lines[2].split()[-2:] == ['-1.0000', '+1.0000']
    assert lines[3].split()[-2:] == ['+2.0000', '+0.0000']
    assert 'same check' in lines[4]


def test_per_check_deltas_missing_values_are_dashes():
    models = [
        ('ParT', None, {1: {'check': 1, 'accuracy': .80}, 3: {'check': 3, 'accuracy': .82}}),
        ('State-EV', None, {1: {'check': 1, 'accuracy': .81}, 2: {'check': 2, 'accuracy': .82}}),
    ]
    lines = S.history_lines(models, 20000)
    assert lines[2].split()[-2:] == ['+1.0000', '--']
    assert lines[3].split()[-2:] == ['--', '--']  # No control at check 2.
    assert lines[4].split()[-2:] == ['--', '--']  # No State-EV at check 3.


def test_control_schedule_mismatch_is_rejected(campaign, tmp_path):
    root, _, _, _, recipe = campaign
    controls = tmp_path / 'controls'
    put(controls / 'campaign.json', dict(recipe=dict(recipe, total_steps=200000),
                                       tasks=[dict(architecture='part', seed=101)]))
    with pytest.raises(ValueError, match='differs in total_steps'):
        S.report(root, controls, {})


def test_validation_gap_uses_contiguous_prefix_not_count(campaign):
    root, run, _, _, _ = campaign
    (run / 'validation/check_10.json').unlink()
    result = S.report(root, None, {}, now=1000)
    assert 'Next validation: check 10' in result
    assert 'Validation history has a gap' in result


def test_cli_offline_stdlib_only(campaign):
    result = subprocess.run([sys.executable, '-B', str(Path(S.__file__)),
                             '--campaign', str(campaign[0]), '--no-controls', '--offline'],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert 'Read-only display' in result.stdout
    assert 'Approx. time until' not in result.stdout


def test_queue_query_has_timeout_and_no_mutating_command(monkeypatch):
    def query(command, **kwargs):
        assert command == ['squeue', '--me', '-h', '-o', '%i|%T|%j']
        assert kwargs['timeout'] == 10
        return subprocess.CompletedProcess(command, 0, '12|RUNNING|evseg_segment\n')
    monkeypatch.setattr(S.subprocess, 'run', query)
    assert S.queue_snapshot() == ({'12': 'RUNNING'}, None)


def test_queue_failure_is_unknown_not_running(monkeypatch):
    def query(*args, **kwargs):
        raise FileNotFoundError('no squeue')
    monkeypatch.setattr(S.subprocess, 'run', query)
    queue, warning = S.queue_snapshot()
    assert queue is None
    assert 'not assuming it is running' in warning


def test_cli_scheduler_warning_precedes_final_progress(campaign, monkeypatch, capsys):
    monkeypatch.setattr(S, 'queue_snapshot', lambda: (None, 'Scheduler unavailable'))
    assert S.main(['--campaign', str(campaign[0]), '--no-controls']) == 0
    result = capsys.readouterr().out
    assert result.index('WARNING: Scheduler unavailable') < result.index('=== Running State-EV progress ===')
    assert 'auc' not in result.lower()
