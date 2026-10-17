"""Approval verifies numerical evidence and preserves the frozen numerical recipe."""
import copy
import json
from pathlib import Path
import shutil
import sys

import pytest

from scripts import run_part_ev_screen_after_recovery as A
from scripts import check_part_ev_screen_amp_recovery as R
from part_ev_screen import protocol as P
from tests.test_part_ev_screen_amp_recovery import recovery, cpu_driver


@pytest.fixture
def ready(recovery, monkeypatch):
    campaign, diagnostic, spec, D = recovery
    monkeypatch.setattr(A, 'R', R)
    D.P.learning_rate = P.learning_rate
    D.P.task_directory = P.task_directory
    D.P.benchmark_summary = lambda *_: (_ for _ in ()).throw(ValueError('original gate still closed'))
    plan = R.read(diagnostic/'recovery.json')
    for index, task in enumerate(spec['tasks']):
        key = 'benchmarks/' + task['run_id'] + '.json'
        row = R.read(campaign/key)
        row['projected_training_hours'] = 10 + index
        row['validation'].update(events=128000, seconds=100.)
        (campaign/key).write_text(json.dumps(row))
        plan['artifacts'][key] = R.sha(campaign/key)
    plan['content_hash'] = R.digest({k: v for k, v in plan.items() if k != 'content_hash'})
    (diagnostic/'recovery.json').write_text(json.dumps(plan))
    for index in range(7):
        cpu_driver(D, spec)
        R.run(diagnostic, index)
    approval = campaign/'production_approval'
    approval.mkdir()
    shutil.copyfile(diagnostic/'launch.env', approval/'launch.env')
    shutil.copyfile(diagnostic/'launch.json', approval/'launch.json')
    return campaign, diagnostic, approval, spec, D


def write_row(diagnostic, task, row, *, steps=True):
    (diagnostic/(task['run_id']+'.json')).write_text(json.dumps(row))
    if steps:
        (diagnostic/(task['run_id']+'.steps.jsonl')).write_text(
            ''.join(json.dumps(x)+'\n' for x in row['attempts']))


def test_approval_is_separate_and_timings_come_from_original_probe(ready):
    campaign, diagnostic, approval, spec, D = ready
    originals = {p: R.sha(p) for p in (campaign/'benchmarks').iterdir()}
    originals.update({p: R.sha(p) for p in diagnostic.iterdir() if p.is_file()})
    summary = A.approve(campaign, diagnostic, approval, reviewed=True)
    assert summary['recommended_train_hours'] == 21  # ceil(16 * 1.3)
    assert summary['recommended_evaluate_hours'] == 2
    assert all(not row['passed'] for row in summary['models'])  # Never relabel old failures.
    payload, _, _, repeated = A.load_approval(approval)
    assert repeated == summary and payload['authorizes_training']
    assert payload['initialization'] == 'fresh_original_seed_model_optimizer_and_grad_scaler'
    assert originals == {p: R.sha(p) for p in originals}
    assert not (campaign/'runs').exists()
    with pytest.raises(ValueError, match='original gate'):
        D.P.benchmark_summary(campaign, spec)


def test_review_required_and_existing_production_cannot_be_reclassified(ready):
    campaign, diagnostic, approval, _, _ = ready
    with pytest.raises(ValueError, match='reviewed'):
        A.approve(campaign, diagnostic, approval, reviewed=False)
    old = campaign/'runs/PART_RANGER/seed_101/last.pt'
    old.parent.mkdir(parents=True)
    old.write_bytes(b'existing production')
    with pytest.raises(ValueError, match='no existing production'):
        A.approve(campaign, diagnostic, approval, reviewed=True)
    assert not (approval/'approval.json').exists()


@pytest.mark.parametrize('change', ['missing', 'status', 'task', 'runtime', 'checkpoint', 'input',
    'short', 'skip', 'nonfinite', 'optimizer', 'phase', 'scale', 'lr', 'memory', 'audit', 'log'])
def test_pass_label_alone_cannot_authorize_training(ready, change):
    campaign, diagnostic, approval, spec, _ = ready
    task = spec['tasks'][4]
    path = diagnostic/(task['run_id']+'.json')
    row = R.read(path)
    if change == 'missing':
        path.unlink()
    else:
        if change == 'status': row['passed'] = False
        if change == 'task': row['task']['seed'] = 202
        if change == 'runtime': row['runtime']['gpu_memory_bytes'] = 2000
        if change == 'checkpoint': row['checkpoint_state']['step'] = 1
        if change == 'input': row['input_fingerprint'] = 'other'
        if change == 'short': row['attempts'].pop()
        if change == 'skip': row['attempts'][10].update(amp_skipped=True, optimizer_steps=0, scale_after=64.)
        if change == 'nonfinite': row['attempts'][10]['gradient_norm'] = None
        if change == 'optimizer': row['attempts'][10]['optimizer_steps'] = 0
        if change == 'phase': row['attempts'][10]['phase'] = 'calibration'
        if change == 'scale': row['attempts'][10]['scale_before'] = 32.
        if change == 'lr': row['attempts'][10]['lr'] = 1e-9
        if change == 'memory': row['attempts'][10]['peak_allocated_bytes'] = 900
        if change == 'audit': row['attempts'][0]['precision']['observed_activations'] = {}
        if change == 'log': row['attempts'][10]['seconds'] = 1234.
        write_row(diagnostic, task, row, steps=change != 'log')
    with pytest.raises((ValueError, FileNotFoundError)):
        A.approve(campaign, diagnostic, approval, reviewed=True)
    assert not (approval/'approval.json').exists()


def test_valid_calibration_backoff_followed_by_ten_successes_is_eligible(ready):
    campaign, diagnostic, _, spec, D = ready
    plan = R.load_plan(diagnostic)
    task = spec['tasks'][1]
    row = R.read(diagnostic/(task['run_id']+'.json'))
    row['attempts'][0].update(amp_skipped=True, optimizer_steps=0, scale_after=64., gradient_norm=None)
    for step in row['attempts'][1:]:
        step.update(scale_before=64., scale_after=64.)
    row['final_scaler']['scale'] = 64.
    original = R.read(campaign/'benchmarks/EV_RANGER.json')
    A.validate_recovery(row, plan, spec, task, original, D.P.learning_rate)


@pytest.mark.parametrize('artifact', ['original', 'recovery', 'plan', 'shell'])
def test_every_worker_rechecks_frozen_evidence(ready, artifact):
    campaign, diagnostic, approval, _, _ = ready
    A.approve(campaign, diagnostic, approval, reviewed=True)
    if artifact == 'original':
        (campaign/'benchmarks/EV_RANGER_probe.pt').write_bytes(b'changed')
    elif artifact == 'recovery':
        task = P.TASKS[1]
        row = R.read(diagnostic/(task['run_id']+'.json'))
        row['attempts'][0]['seconds'] += 1.
        write_row(diagnostic, task, row)
    elif artifact == 'plan':
        (diagnostic/'recovery.json').write_text('{}')
    else:
        (approval/'launch.env').write_text('changed resources')
    with pytest.raises(ValueError):
        A.load_approval(approval)


def test_admission_is_scoped_and_runtime_matched_even_on_exception(ready):
    campaign, diagnostic, approval, spec, D = ready
    summary = A.approve(campaign, diagnostic, approval, reviewed=True)
    original = D.P.benchmark_summary
    with pytest.raises(RuntimeError, match='fixture'):
        with A.admission(D, campaign, spec, summary):
            assert D.P.benchmark_summary(campaign, spec, summary['runtime']) == summary
            with pytest.raises(ValueError, match='runtime'):
                D.P.benchmark_summary(campaign, spec, {'gpu_memory_bytes': 999})
            with pytest.raises(ValueError, match='another campaign'):
                D.P.benchmark_summary(campaign/'other', spec)
            raise RuntimeError('fixture')
    assert D.P.benchmark_summary is original


@pytest.mark.parametrize('phase', ['train', 'evaluate', 'report'])
def test_dispatch_uses_original_cli_and_never_probe_weights(ready, phase):
    campaign, diagnostic, approval, spec, D = ready
    A.approve(campaign, diagnostic, approval, reviewed=True)
    seen = []
    def main():
        seen.append(list(sys.argv))
        assert D.P.benchmark_summary(campaign, spec, {'gpu_memory_bytes':1000})['passed']
    D.main = main
    original_argv = sys.argv
    original_gate = D.P.benchmark_summary
    A.execute(approval, phase, None if phase == 'report' else 0)
    assert seen[0][:3] == [str(campaign/'source/scripts/run_part_ev_screen.py'), '--phase', phase]
    assert '_probe' not in ' '.join(seen[0])
    assert sys.argv is original_argv and D.P.benchmark_summary is original_gate


def test_only_explicit_resume_can_use_existing_production(ready):
    campaign, diagnostic, approval, spec, D = ready
    A.approve(campaign, diagnostic, approval, reviewed=True)
    run = P.task_directory(campaign, 1)
    run.mkdir(parents=True)
    (run/'last.pt').write_bytes(b'production state')
    seen = []
    D.main = lambda: seen.append('called original CLI')
    with pytest.raises(ValueError, match='explicitly --resume'):
        A.execute(approval, 'train', 1)
    assert not seen
    A.execute(approval, 'train', 1, resume=True)
    assert seen == ['called original CLI']
    with pytest.raises(ValueError):
        A.execute(approval, 'train', 7, resume=True)
