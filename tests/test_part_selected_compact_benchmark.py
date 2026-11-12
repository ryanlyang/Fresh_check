from types import SimpleNamespace
import pytest
import torch
from part_selected_compact import benchmark as B, protocol as P
from scripts import run_part_selected_compact as driver
from tests.test_part_selected_compact_protocol import evidence


@pytest.fixture
def fake_gpu(tmp_path, monkeypatch):
    spec, row = evidence('compact')
    spec.update(inventory={}, environment={'fixed': True})
    state = dict(steps=0, full=0, closed=0, fail_full=False, fail_input=False, fail_backend=False)
    P.write_json(tmp_path/'inputs/normalization.json', P.hashed({'synthetic': True}))
    monkeypatch.setattr(P, 'load', lambda _: spec)
    monkeypatch.setattr(P, 'runtime_environment', lambda: spec['environment'])
    monkeypatch.setattr(P, 'require_space', lambda *_: None)
    def verify(_):
        if state['fail_backend']: raise ValueError('backend changed')
        return None, None, spec['backend_evidence']
    monkeypatch.setattr(P, 'verify_backends', verify)
    monkeypatch.setattr(B, 'host_memory_limit', lambda: 128*2**30)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda *_: 'GH200')
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda *_: SimpleNamespace(total_memory=96*2**30))
    for name in ('set_device', 'synchronize', 'empty_cache', 'reset_peak_memory_stats'):
        monkeypatch.setattr(torch.cuda, name, lambda *_: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda *_: 30*2**30)
    monkeypatch.setattr(B, 'seed_everything', lambda *_: None)
    monkeypatch.setattr(B, 'PackedTrees', lambda *_, **kw: None)
    class Trimmer(torch.nn.Module):
        enabled = True
        def forward(self, _): return None, None, torch.zeros(1,1,128), None
    class Model(torch.nn.Linear):
        def __init__(self):
            super().__init__(2,2)
            self.mod = SimpleNamespace(trimmer=Trimmer())
        def to(self, *_): return self
    monkeypatch.setattr(B, 'build_model', lambda *_, **kw: Model())
    monkeypatch.setattr(B, 'build_optimizer', lambda *_: SimpleNamespace(state_dict=lambda: {'optimizer': True}))
    monkeypatch.setattr(torch.amp, 'GradScaler', lambda *_, **kw: SimpleNamespace(state_dict=lambda: {'scaler': True}))
    monkeypatch.setattr(B, 'capture_rank_state', lambda _: {'rng': True})
    monkeypatch.setattr(B, 'execution', lambda *_, **kw: row['parity'])
    monkeypatch.setattr(B, 'restart', lambda *_: row['restart'])
    monkeypatch.setattr(B, 'curve', lambda *_: row['flops'])
    monkeypatch.setattr(B, 'validate_model', lambda _, arch: {'parameters': P.COUNTS[arch]})
    monkeypatch.setattr(B.data, 'build_loader', lambda *_, **kw: (iter(range(5000)), None))
    monkeypatch.setattr(B.data, 'shutdown_loader', lambda _: state.__setitem__('closed', state['closed']+1))
    monkeypatch.setattr(B.data, 'batch_fingerprint', lambda batch: 'wrong' if state['fail_input'] else str(batch))
    monkeypatch.setattr(B, 'all_input_fingerprint', lambda batch: 'raw'+str(batch))
    def step(model, *_, full_length=False, **kw):
        state['steps'] += 1
        state['full'] += int(full_length)
        model.mod.trimmer(None)
        return dict(amp_skipped=bool(full_length and state['fail_full']), precision={'passed': True})
    monkeypatch.setattr(B, 'training_step', step)
    monkeypatch.setattr(B, 'validate', lambda *_, **kw: row['validation'])
    class Monitor:
        peak = 32*2**30
        gpu_reserved = 30*2**30
        def __enter__(self): return self
        def __exit__(self, *_): pass
    monkeypatch.setattr(B, 'MemoryMonitor', Monitor)
    return state


@pytest.mark.parametrize('index', [0,1])
def test_full_benchmark_orchestration_is_disposable(tmp_path, fake_gpu, index):
    result = B.benchmark(tmp_path, index)
    assert result['passed'] and result['complete'] and not result['authorizes_training']
    assert fake_gpu['steps'] == 4220+10 and fake_gpu['full'] == 10 and fake_gpu['closed'] == 3
    assert result['length_histogram'] == {'128': 4200}
    assert [w['steps'] for w in result['windows']] == [1000,1000,1000,1000,200]
    assert result['selected_input_fingerprints']['4020'] == 'raw4020'
    assert result['checkpoint_bytes'] > 0 and not (tmp_path/'runs').exists()
    assert not (tmp_path/'benchmarks'/f'{P.TASKS[index]["architecture"]}.json').exists()


@pytest.mark.parametrize('fault,closed', [('fail_backend',0), ('fail_input',1), ('fail_full',1)])
def test_failure_never_writes_success(tmp_path, fake_gpu, fault, closed):
    fake_gpu[fault] = True
    with pytest.raises(ValueError): B.benchmark(tmp_path,0)
    assert fake_gpu['closed'] == closed
    assert not (tmp_path/'benchmarks/compact.json').exists()


def test_wrong_gpu_stops_before_data(tmp_path, fake_gpu, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda *_: 'A100')
    with pytest.raises(ValueError, match='GH200'): B.benchmark(tmp_path,0)
    assert fake_gpu['steps'] == 0 and fake_gpu['closed'] == 0


@pytest.fixture
def worker(tmp_path, monkeypatch):
    P.write_json(tmp_path/'launch.json', dict(output=str(tmp_path), authorizes_training=False))
    monkeypatch.setattr(driver, 'verify_snapshot', lambda _: None)
    return ['--output',str(tmp_path)]


def test_summary_incomplete_is_explicit_nonzero_not_pass(tmp_path, worker):
    assert driver.main([*worker,'--phase','summary']) == 1
    row = P.checked(tmp_path/'reports/incomplete.json')
    assert not row['passed'] and not row['authorizes_training'] and 'campaign.json' in row['missing']
    assert not (tmp_path/'reports/benchmark_summary.json').exists()


def test_review_is_read_only_when_incomplete(tmp_path, worker):
    before = {p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    assert driver.main([*worker,'--phase','review']) == 1
    assert before == {p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


def test_worker_failure_writes_diagnostic_not_benchmark(tmp_path, worker, monkeypatch):
    def fail(*_): raise TimeoutError('bounded check')
    monkeypatch.setattr(B,'benchmark',fail)
    with pytest.raises(TimeoutError): driver.main([*worker,'--phase','benchmark','--task-index','0'])
    row = P.read_json(tmp_path/'diagnostics/benchmark_0_failure.json')
    assert not row['passed'] and not row['authorizes_training']
    assert not (tmp_path/'benchmarks/compact.json').exists()
