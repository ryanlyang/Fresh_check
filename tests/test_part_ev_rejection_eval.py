"""Local CPU/synthetic tests; never submit real jobs or open research data."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


# Driver inserts its own scripts directory, just as the frozen worker does.
L = module('submit_part_ev_rejection_eval')
R = module('run_part_ev_rejection_eval')
S = R.S


@pytest.fixture
def donor(tmp_path):
    def make(key='part'):
        root = tmp_path / key
        root.mkdir()
        python = tmp_path / 'python'
        python.touch()
        task = dict(run_id=S.IDS[key], architecture='part' if key == 'part' else 'state_layerwise',
                    recipe='ranger_full_v1', seed=101)
        recipe = dict(total_steps=1_000_000, epochs=50, steps_per_epoch=20_000,
                      global_batch_size=512, evaluation_population=5_000_000,
                      validation_events_per_epoch=1_280_000)
        files = [dict(name=f'file_{i}.root', num_entries=100_000) for i in range(50)]
        spec = S.seal(dict(tasks=[task], recipes={'ranger_full_v1': recipe},
                           inventory={'splits': {'val': {'files': files}}}))
        S.write(root / 'campaign.json', spec)
        S.write(root / 'launch.json', dict(python=str(python)))
        (root / 'config.sh').write_text(f'EVDDP_PYTHON={python.as_posix()}\n')
        (root / 'source.sha256').write_text('')
        run = root / 'runs' / task['run_id'] / 'seed_101'
        run.mkdir(parents=True)
        (run / 'best_model_val.pt').write_bytes(b'fake checkpoint')
        S.write(run / 'optimizer.json', {})
        for i in range(1, 51):
            S.write(run / 'validation' / f'check_{i:02d}.json',
                    dict(check=i, step=20_000*i, events=1_280_000, accuracy=.86 if i >= 49 else .85))
        S.write(run / 'training_complete.json', dict(task=task, step=1_000_000,
            validated_checks=50, best_check=49, best_accuracy=.86, runtime={},
            campaign_sha256=spec['content_hash'], best_checkpoint_sha256=S.sha(run / 'best_model_val.pt')))
        return root, run
    return make


def test_describe_complete_and_no_donor_writes(donor):
    root, run = donor()
    before = {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    result = S.describe(root, 'part')
    assert result['best_check'] == 49
    assert result['run'] == str(run)
    S.verify_pins(result)
    assert before == {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('change', ['budget', 'tie', 'weights', 'missing_check', 'event_count'])
def test_incomplete_or_changed_training_refused(donor, change):
    root, run = donor()
    done = S.read(run / 'training_complete.json')
    if change == 'budget':
        done['step'] = 999_999
    elif change == 'tie':
        done['best_check'] = 50
    elif change == 'weights':
        (run / 'best_model_val.pt').write_bytes(b'changed')
    elif change == 'missing_check':
        (run / 'validation/check_48.json').unlink()
    else:
        row = S.read(run / 'validation/check_48.json')
        row['events'] = 128
        S.write(run / 'validation/check_48.json', row)
    S.write(run / 'training_complete.json', done)
    with pytest.raises((ValueError, FileNotFoundError)):
        S.describe(root, 'part')


def test_later_donor_mutation_refused(donor):
    root, run = donor()
    descriptor = S.describe(root, 'part')
    (run / 'best_model_val.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        S.verify_pins(descriptor)


def test_plan_does_not_create_output(donor, tmp_path, capsys):
    part, _ = donor('part')
    state, _ = donor('state')
    out = tmp_path / 'new'
    L.main(['--mode', 'plan', '--part', str(part), '--state', str(state), '--output', str(out)])
    assert not out.exists()
    assert 'no files or jobs changed' in capsys.readouterr().out


def test_array_and_segment_duplicate_guard(tmp_path):
    root = tmp_path / 'donor'
    root.mkdir()
    (root / 'submitted_job_ids.tsv').write_text('train\t123\n')
    donors = {'part': {'root': str(root)}}
    with pytest.raises(ValueError, match='123'):
        S.refuse_active(donors, live={'123_0'})
    S.refuse_active(donors, live={'124'})
    chain = root / 'segment_chains/a'
    chain.mkdir(parents=True)
    (chain / 'submitted_job_ids.tsv').write_text('segment_3\t125\n')
    with pytest.raises(ValueError, match='125'):
        S.refuse_active(donors, live={'125'})


def test_scheduler_failure_not_treated_as_empty(monkeypatch):
    def fail(*args):
        raise RuntimeError('scheduler down')
    monkeypatch.setattr(S, 'command', fail)
    with pytest.raises(RuntimeError):
        S.refuse_active({})


def test_requests_are_independent_single_gpu_five_hour(tmp_path):
    plan = {'partition': 'debug', 'account': 'reu-aisocial'}
    for key in ('part', 'state'):
        args = L.request(plan, tmp_path, 'worker', key)
        assert '--gres=gpu:a100:1' in args
        assert '--time=05:00:00' in args
        assert '--cpus-per-task=8' in args and '--mem=64G' in args
        assert not any('dependency' in a or '--array' in a for a in args)
    args = L.request(plan, tmp_path, 'report', dependency='1:2')
    assert '--dependency=afterok:1:2' in args
    assert not any('--gres' in a for a in args)
    assert '--time=00:15:00' in args


def test_partial_submission_is_recorded(tmp_path, monkeypatch):
    calls = []
    def fake(args):
        calls.append(args)
        if len(calls) == 2:
            raise RuntimeError('sbatch failed')
        return '123'
    monkeypatch.setattr(L.S, 'command', fake)
    with pytest.raises(RuntimeError):
        L.submit_jobs(tmp_path, dict(partition='debug', account='account'))
    assert (tmp_path / 'submitted_job_ids.tsv').read_text() == 'part\t123\n'


def test_successful_submit_three_jobs_no_training(tmp_path, monkeypatch):
    calls = []
    def fake(args):
        calls.append(args)
        return str(100+len(calls))
    monkeypatch.setattr(L.S, 'command', fake)
    L.submit_jobs(tmp_path, dict(partition='debug', account='account'))
    assert len(calls) == 3
    assert '--dependency=afterok:101:102' in calls[2]
    assert '--dependency=afterok:101' not in calls[1]


def test_rejection_matches_frozen_metric():
    from part_ev_fullscale.metrics import calculate_metrics
    from part_ev_fullscale.protocol import CLASS_NAMES
    rng = np.random.default_rng(7)
    logits = rng.normal(size=(1000, 10)).astype(np.float32)
    truth = np.repeat(np.arange(10), 100)
    expected = calculate_metrics(logits, truth)
    actual = S.rejection_metrics(logits, truth, CLASS_NAMES)
    assert actual['accuracy'] == expected['accuracy']
    for name in CLASS_NAMES[1:]:
        if name not in ('Hqql', 'Tbl'):
            assert actual['qcd_rejection'][name] == expected['qcd_rejection'][name]
        assert set(actual['qcd_rejection'][name]) == {'0.3', '0.5'}


def test_ties_and_zero_acceptances():
    names = ['QCD'] + [str(i) for i in range(1, 10)]
    y = np.repeat(np.arange(10), 10)
    logits = np.zeros((100, 10), dtype=np.float32)
    tied = S.rejection_metrics(logits, y, names)['qcd_rejection']['1']['0.5']
    assert tied['achieved_signal_efficiency'] == 1 and tied['rejection'] == 1
    logits[y == 1, 1] = 100
    zero = S.rejection_metrics(logits, y, names)['qcd_rejection']['1']['0.5']
    assert zero['qcd_accepted'] == 0 and zero['rejection'] is None


@pytest.mark.parametrize('bad', ['nan', 'shape', 'labels'])
def test_bad_predictions_rejected(bad):
    logits, y = np.zeros((10, 10)), np.arange(10)
    if bad == 'nan': logits[0, 0] = np.nan
    elif bad == 'shape': logits = logits[:, :9]
    else: y = y.astype(float)
    with pytest.raises(ValueError):
        S.validate_arrays(logits, y, 10)


def test_chunk_resume_and_corruption(tmp_path):
    path = tmp_path / 'file_000.npz'
    logits, y = np.zeros((10, 10)), np.arange(10)
    np.savez(path, logits=logits, labels=y)
    contract = dict(entry_stop=10)
    S.write(path.with_suffix('.json'), dict(contract=contract, sha256=S.sha(path),
                                            precision=dict(passed=True, expected_activation_dtype='torch.float32'),
                                            labels_sha256=hashlib.sha256(y.tobytes()).hexdigest()))
    loaded, truth, _ = R.read_chunk(path, contract)
    np.testing.assert_array_equal(loaded, logits)
    np.testing.assert_array_equal(truth, y)
    with pytest.raises(ValueError):
        R.read_chunk(path, dict(entry_stop=9))
    with path.open('ab') as handle:
        handle.write(b'corrupted')
    with pytest.raises(ValueError):
        R.read_chunk(path, contract)


def test_comparison_percent_not_percentage_points():
    def row(r):
        return dict(best_check=49, metrics=dict(accuracy=.86, qcd_rejection={
            'Hbb': {'0.5': dict(rejection=r, qcd_accepted=5)}}))
    text = S.comparison_text({'part': row(100), 'state': row(120)})
    assert '+20.00' in text and 'SyncBatchNorm' in text
    text = S.comparison_text({'part': row(100), 'state': row(None)})
    assert 'ZB' in text and '| -- |' in text
    assert 'available rejection' in S.comparison_text({'part': row(100)})


def test_runtime_uses_frozen_imports_and_no_train_or_global_lock():
    import ast
    tree = ast.parse((ROOT / 'scripts/run_part_ev_rejection_eval.py').read_text())
    calls = [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert 'train' not in calls and 'lock_finalists' not in calls
    assert 'audit_training' in calls and 'load_state_dict' in calls
    assert 'convert_sync_batchnorm' in calls
    assert 'finite_validation_loader' in calls


def test_real_cpu_inference_file_roundtrip_resume_and_interruption(tmp_path):
    import torch
    from part_ev_screen.precision import model_precision, audit_precision
    from scripts.run_part_ev_screen import atomic_npz
    torch.manual_seed(71)
    model = torch.nn.Linear(4, 10).eval()
    x, y = torch.randn(20, 4), torch.arange(20) % 10
    calls = []
    def loader(*_):
        calls.append('loaded')
        yield [x[:10]], y[:10]
        yield [x[10:]], y[10:]
    D = SimpleNamespace(torch=torch, model_precision=model_precision,
        audit_precision=audit_precision, atomic_npz=atomic_npz,
        finite_validation_loader=loader, tensors=lambda b, _: b)
    device = torch.device('cpu')
    path = tmp_path / 'file_000.npz'
    record, contract = dict(num_entries=20), dict(entry_stop=20)
    kwargs = dict(D=D, spec={'inventory': {}}, model=model, device=device,
                  record=record, path=path, contract=contract, stop=lambda: None)
    logits, labels, receipt, reused = R.evaluate_file(**kwargs)
    assert not reused and len(calls) == 1 and receipt['precision']['passed']
    np.testing.assert_allclose(logits, model(x).detach().numpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(labels, y.numpy())
    before = path.read_bytes()
    assert R.evaluate_file(**kwargs)[-1] is True
    assert len(calls) == 1 and path.read_bytes() == before
    # An uncommitted prediction may be replaced, a committed changed file may not.
    path.with_suffix('.json').unlink()
    assert R.evaluate_file(**kwargs)[-1] is False and len(calls) == 2
    path.unlink()
    with pytest.raises(ValueError, match='missing'):
        R.evaluate_file(**kwargs)
    path.with_suffix('.json').unlink()
    stops = []
    def interrupted():
        stops.append(True)
        if len(stops) == 2:
            raise SystemExit(75)
    kwargs['stop'] = interrupted
    with pytest.raises(SystemExit) as caught:
        R.evaluate_file(**kwargs)
    assert caught.value.code == 75 and not path.exists() and not path.with_suffix('.json').exists()


def test_worker_bootstrap_restores_environment_without_importing_models(donor, tmp_path, monkeypatch):
    root, _ = donor('state')
    spec = S.read(root / 'campaign.json')
    spec.pop('content_hash')
    spec['expected_gpu'] = {'nccl_environment': {'NCCL_IB_DISABLE': '1'}}
    spec = S.seal(spec)
    S.write(root / 'campaign.json', spec)
    run = root / 'runs' / S.IDS['state'] / 'seed_101'
    done = S.read(run / 'training_complete.json')
    done['campaign_sha256'] = spec['content_hash']
    S.write(run / 'training_complete.json', done)
    info = S.describe(root, 'state')
    monkeypatch.setenv('WORLD_SIZE', '2')
    monkeypatch.setenv('NCCL_DEBUG', 'INFO')
    class Executed(Exception): pass
    def fake_exec(binary, args, env):
        assert Path(binary) == Path(info['python']) and '--phase' in args and 'evaluate' in args
        assert 'WORLD_SIZE' not in env and 'NCCL_DEBUG' not in env
        assert env['NCCL_IB_DISABLE'] == '1'
        assert env['PYTHONDONTWRITEBYTECODE'] == '1'
        assert env['PYTHONPATH'] == str(root / 'source')
        raise Executed
    monkeypatch.setattr(R.os, 'execve', fake_exec)
    with pytest.raises(Executed):
        R.bootstrap(tmp_path, 'state', {'donors': {'state': info}})
