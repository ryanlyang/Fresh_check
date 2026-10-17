"""Bounded recovery policy, artifact isolation, and real CPU GradScaler updates."""
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from scripts import check_part_ev_screen_amp_recovery as R
from part_ev_screen import protocol as P


def audit(dtype):
    return dict(passed=True, parameter_dtypes=['torch.float32'], expected_activation_dtype=dtype,
                observed_activations={'layer': dtype})


def original(spec, task, full_skips=1):
    return dict(task=task, campaign_sha256=spec['content_hash'], passed=full_skips == 0,
                runtime=dict(gpu_memory_bytes=1000), initial_parity=dict(passed=True),
                training_precision=audit('torch.float16'), validation=dict(precision=audit('torch.float32')),
                measured_steps=4200, amp_skipped_steps=2, full_length_amp_skips=full_skips,
                native_peak_bytes=200, full_length_peak_bytes=300, validation_peak_bytes=100,
                input_fingerprints={'0': 'same'})


@pytest.mark.parametrize('skips', [0, 1, 2, 3])
def test_only_original_full_length_skip_gate_can_be_rechecked(skips):
    spec = dict(content_hash='test', benchmark=P.BENCHMARK)
    R.eligible_report(original(spec, P.TASKS[0], skips), spec, P.TASKS[0])


@pytest.mark.parametrize('change', [
    {'campaign_sha256': 'other'}, {'passed': True}, {'amp_skipped_steps': 210},
    {'full_length_peak_bytes': 900}, {'full_length_amp_skips': 4}, {'native_peak_bytes': -1},
    {'measured_steps': 4000}, {'training_precision': audit('torch.float32')},
    {'initial_parity': {'passed': False}},
])
def test_prepare_rejects_other_original_failures(change):
    spec = dict(content_hash='test', benchmark=P.BENCHMARK)
    row = original(spec, P.TASKS[0])
    row.update(change)
    with pytest.raises(ValueError):
        R.eligible_report(row, spec, P.TASKS[0])


def step_row(**changes):
    row = dict(loss=1., gradient_norm=2., state_finite=True, scale_before=128., scale_after=128.,
               optimizer_steps=1, amp_skipped=False)
    return dict(row, **changes)


def test_calibration_skip_allowed_only_with_backoff_and_no_optimizer_step():
    row = step_row(amp_skipped=True, optimizer_steps=0, scale_after=64., gradient_norm=None)
    assert R.assess_attempt(row, confirmation=False) is None
    assert R.assess_attempt(row, confirmation=True)
    assert R.assess_attempt(dict(row, optimizer_steps=1), confirmation=False)
    assert R.assess_attempt(dict(row, scale_after=128.), confirmation=False)


@pytest.mark.parametrize('changes', [dict(gradient_norm=None), dict(gradient_norm=float('inf')),
    dict(loss=float('nan')), dict(state_finite=False), dict(scale_after=0),
    dict(scale_after=float('inf')), dict(optimizer_steps=0), dict(optimizer_steps=2), dict(scale_after=64.)])
def test_bad_success_cannot_pass(changes):
    assert R.assess_attempt(step_row(**changes), confirmation=True)


def test_finite_state_covers_nested_optimizer_buffers():
    assert R.finite_state(torch, {'slow': [torch.ones(2)], 'step': 3, 'other': None})
    assert not R.finite_state(torch, {'inner': {'state': [torch.tensor(float('nan'))]}})


@pytest.fixture
def recovery(tmp_path, monkeypatch):
    campaign = tmp_path / 'campaign'
    output = campaign / 'diagnostics' / 'recovery'
    output.mkdir(parents=True)
    (campaign / 'benchmarks').mkdir()
    (output / 'launch.env').write_text('# test\n')
    R.create_json(output / 'launch.json', {'test': True})
    spec = dict(content_hash='test', tasks=copy.deepcopy(P.TASKS), benchmark=P.BENCHMARK,
                recipes=P.recipes(), inventory={}, launch={'test': True})
    for task in spec['tasks']:
        prefix = campaign / 'benchmarks' / task['run_id']
        R.create_json(prefix.with_suffix('.json'), original(spec, task))
        Path(str(prefix) + '_probe.pt').write_bytes(b'preserved checkpoint fixture')
    D = SimpleNamespace(P=SimpleNamespace(load_campaign=lambda _: spec, file_lock=P.file_lock))
    monkeypatch.setattr(R, 'frozen_driver', lambda _: D)
    R.prepare(campaign, output)
    return campaign, output, spec, D


def test_prepare_hashes_every_original_artifact_and_binds_policy(recovery):
    campaign, output, spec, _ = recovery
    plan = R.load_plan(output)
    assert len(plan['artifacts']) == 14 and plan['tasks'] == P.TASKS
    assert plan['policy']['authorizes_training'] is False
    for task in P.TASKS:
        R.verify_artifacts(campaign, plan, task)
    plan['policy']['confirmation_attempts'] = 1
    # Even a recomputed hash cannot replace this diagnostic's fixed policy.
    plan['content_hash'] = R.digest({k: v for k, v in plan.items() if k != 'content_hash'})
    (output / 'recovery.json').write_text(json.dumps(plan))
    with pytest.raises(ValueError, match='changed'):
        R.load_plan(output)


def test_tampered_checkpoint_and_config_are_rejected(recovery):
    campaign, output, _, _ = recovery
    plan = R.load_plan(output)
    (campaign / 'benchmarks/PART_RANGER_probe.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='artifact changed'):
        R.verify_artifacts(campaign, plan, P.TASKS[0])
    (output / 'launch.env').write_text('# changed\n')
    with pytest.raises(ValueError, match='configuration changed'):
        R.load_plan(output)


def cpu_driver(D, spec, *, overflow_indices=(), bad_fingerprint=False, interrupted=False):
    """Real optimizer/scaler steps; only CUDA, data loading and model are mocked."""
    calls = dict(steps=[], restore=False, shutdown=False, hooked=[])
    model = torch.nn.Linear(2, 2)
    saved_optimizer = torch.optim.SGD(model.parameters(), lr=.001, momentum=.9)
    saved_optimizer.zero_grad()
    model(torch.ones(1, 2)).sum().backward()
    saved_optimizer.step()
    saved_weights = copy.deepcopy(model.state_dict())
    saved_state = copy.deepcopy(saved_optimizer.state_dict())
    saved_scaler = torch.amp.GradScaler('cpu', init_scale=128., growth_interval=2000).state_dict()
    calls['saved_weights'], calls['saved_optimizer'] = saved_weights, saved_state
    batch = ({'pf_mask': torch.ones(512, 1, 128)}, {'_label_': torch.zeros(512, dtype=torch.long)})
    D.torch = SimpleNamespace(amp=SimpleNamespace(GradScaler=lambda _: torch.amp.GradScaler('cpu')),
        is_tensor=torch.is_tensor, isfinite=torch.isfinite,
        cuda=SimpleNamespace(reset_peak_memory_stats=lambda: None, synchronize=lambda: None,
                             max_memory_allocated=lambda: 300, max_memory_reserved=lambda: 400))
    D.gpu_setup = lambda _: (torch.device('cpu'), {'gpu_memory_bytes': 1000})
    D.make_model = lambda *_: model
    D.build_optimizer = lambda m, _: torch.optim.SGD(m.parameters(), lr=.001, momentum=.9)

    def load_checkpoint(_path, m, opt, scaler, _spec, _runtime, task):
        m.load_state_dict(saved_weights)
        opt.load_state_dict(saved_state)
        scaler.load_state_dict(saved_scaler)
        return dict(step=4220, task=task), {'captured': True}

    D.load_checkpoint = load_checkpoint
    D.build_loader = lambda *args, **kwargs: ([batch], None)
    D.seed_everything = lambda _: None
    D.batch_fingerprint = lambda _: 'wrong' if bad_fingerprint else 'same'
    D.restore_rank_state = lambda *args: calls.update(restore=True)
    D.diagnostic_mode = lambda *args: None
    D.shutdown_loader = lambda _: calls.update(shutdown=True)

    def check_stop():
        if interrupted:
            raise SystemExit(75)

    D.check_stop = check_stop

    def train(m, opt, scaler, _batch, _device, recipe, step, *, full_length, check_precision):
        assert calls['restore'] and full_length
        index = len(calls['steps'])
        calls['steps'].append(step)
        if index == 0:
            assert scaler.get_scale() == 128.  # No default 65536 reset.
            assert len(opt.state) == len(saved_optimizer.state)
            for key, value in saved_weights.items():
                assert torch.equal(m.state_dict()[key], value)
        opt.zero_grad()
        loss = m(torch.ones(2, 2)).square().mean()
        before = scaler.get_scale()
        scaler.scale(loss).backward()
        if index in overflow_indices:
            next(m.parameters()).grad.fill_(float('inf'))
        scaler.unscale_(opt)
        norm = torch.nn.utils.clip_grad_norm_(m.parameters(), float('inf'))
        scaler.step(opt)
        scaler.update()
        return dict(loss=float(loss.detach()), gradient_norm=float(norm) if torch.isfinite(norm) else None,
                    amp_skipped=scaler.get_scale() < before, lr=P.learning_rate(step, recipe),
                    clipped=False, **({'precision': audit('torch.float16')} if check_precision else {}))

    D.training_step = train
    return calls


@pytest.mark.parametrize('index', [0, 6])
def test_checkpoint_continuation_records_exact_updates_and_leaves_originals_alone(recovery, index):
    campaign, output, spec, D = recovery
    calls = cpu_driver(D, spec, overflow_indices=(0,))
    before = {p: R.sha(p) for p in (campaign / 'benchmarks').iterdir()}
    R.run(output, index)
    task = P.TASKS[index]
    result = R.read(output / (task['run_id'] + '.json'))
    assert result['passed'] and not result['authorizes_training']
    assert len(result['attempts']) == 18
    assert result['attempts'][0]['optimizer_steps'] == 0
    assert all(row['optimizer_steps'] == 1 for row in result['attempts'][1:])
    assert sum(row['phase'] == 'confirmation' for row in result['attempts']) == 10
    assert result['final_scaler']['scale'] == 64.
    assert set(calls['steps']) == ({0} if index == 0 else {9999})
    assert calls['shutdown']
    assert before == {p: R.sha(p) for p in before}
    with pytest.raises(ValueError, match='already attempted'):
        R.run(output, index)


@pytest.mark.parametrize('overflow', [(7,), (8,), (17,), tuple(range(8))])
def test_late_or_persistent_overflow_fails_with_report(recovery, overflow):
    _, output, spec, D = recovery
    calls = cpu_driver(D, spec, overflow_indices=overflow)
    with pytest.raises(FloatingPointError):
        R.run(output, 1)
    result = R.read(output / 'EV_RANGER.json')
    assert not result['passed'] and not result['authorizes_training']
    assert result['error']['type'] == 'FloatingPointError'
    assert result['attempts'][-1]['failure'] and calls['shutdown']


@pytest.mark.parametrize('failure', ['input', 'runtime', 'interrupt'])
def test_failures_write_diagnostic_not_false_success(recovery, failure):
    _, output, spec, D = recovery
    calls = cpu_driver(D, spec, bad_fingerprint=failure == 'input', interrupted=failure == 'interrupt')
    if failure == 'runtime':
        D.gpu_setup = lambda _: (torch.device('cpu'), {'gpu_memory_bytes': 2000})
    with pytest.raises((ValueError, SystemExit)):
        R.run(output, 1)
    row = R.read(output / 'EV_RANGER.json')
    assert not row['passed'] and row['error'] and not row['attempts']
    assert not calls['steps']


def test_summary_needs_all_seven_and_reports_failures(recovery, capsys):
    _, output, _, _ = recovery
    assert R.summarize(output) is False
    assert 'MISSING' in capsys.readouterr().out
    plan = R.load_plan(output)
    for task in P.TASKS:
        R.create_json(output / (task['run_id'] + '.json'), dict(
            task=task, recovery_sha256=plan['content_hash'], passed=False, authorizes_training=False,
            attempts=[], error={'message': 'fixture failure'}))
    assert not R.summarize(output)
    assert 'fixture failure' in capsys.readouterr().out


def test_frozen_import_refuses_live_modules(tmp_path):
    # Test imports above loaded the live package: the actual worker is a fresh
    # process and must import only campaign/source, not the modified checkout.
    with pytest.raises(RuntimeError, match='non-campaign'):
        R.frozen_driver(tmp_path)


def test_prepare_cannot_target_production_directory(tmp_path):
    with pytest.raises(ValueError, match='diagnostics'):
        R.prepare(tmp_path, tmp_path / 'runs')


def test_fresh_process_imports_campaign_source_not_live_checkout(tmp_path):
    source = tmp_path / 'source'
    (source / 'scripts').mkdir(parents=True)
    (source / 'part_ev_screen').mkdir()
    (source / 'part_ev_screen/__init__.py').write_text('marker = "frozen fixture"\n')
    (source / 'scripts/run_part_ev_screen.py').write_text('from part_ev_screen import marker\n')
    result = subprocess.run([sys.executable, '-c',
        'import sys; from pathlib import Path; from scripts import check_part_ev_screen_amp_recovery as R; '
        'print(R.frozen_driver(Path(sys.argv[1])).marker)', str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'frozen fixture'


@pytest.mark.parametrize('recipe_name', ['ranger_short_v1', 'adamw_short_v1'])
def test_real_screen_optimizers_only_fire_post_hook_on_unskipped_updates(recipe_name):
    from part_ev_fullscale.optimizer import build_optimizer
    model = torch.nn.Linear(2, 2)
    optimizer = build_optimizer(model, P.recipes()[recipe_name])
    scaler = torch.amp.GradScaler('cpu', init_scale=128.)
    count = []
    handle = optimizer.register_step_post_hook(lambda *_: count.append(1))
    try:
        for overflow in (False, True, False):
            optimizer.zero_grad(set_to_none=True)
            loss = model(torch.ones(2, 2)).square().mean()
            scaler.scale(loss).backward()
            if overflow:
                next(model.parameters()).grad.fill_(float('inf'))
            scaler.unscale_(optimizer)
            before = len(count)
            scaler.step(optimizer)
            scaler.update()
            assert len(count) - before == (0 if overflow else 1)
    finally:
        handle.remove()
