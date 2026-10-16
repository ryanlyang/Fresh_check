"""Regression for Weaver's nested autocast overriding the screen's outer AMP.

CPU emulation exercises the actual model/operators and training entry point;
separate CUDA tests run unmodified when a GPU is available. CPU tests are not
evidence that batch 512 fits a 40-GiB A100: the Slurm benchmark must prove that.
"""
import pytest
import torch

from part_ev_screen.model import COUNTS, build_model
from part_ev_screen.precision import audit_precision, model_precision
from part_ev_screen import protocol as P
from part_ev_fullscale.optimizer import build_optimizer
from scripts import run_part_ev_screen as D
from tests.test_part_ev_screen_training import batch
from tests.test_part_fullscale_model import bounded_cpu_threads


@pytest.fixture
def emulate_cuda_autocast(monkeypatch):
    native_autocast = torch.autocast

    def outer(device_type, **kwargs):
        return native_autocast('cpu' if device_type == 'cuda' else device_type, **kwargs)

    def inner(enabled=True, **_kwargs):
        return native_autocast('cpu', dtype=torch.float16, enabled=enabled)

    monkeypatch.setattr(torch, 'autocast', outer)
    monkeypatch.setattr(torch.cuda.amp, 'autocast', inner)
    original_tensors = D.tensors
    monkeypatch.setattr(D, 'tensors', lambda data, _device: original_tensors(data, torch.device('cpu')))


@pytest.mark.parametrize('architecture', list(COUNTS))
def test_actual_training_entry_point_enables_native_amp_and_restores_flag(architecture, emulate_cuda_autocast):
    model = D.make_model(dict(architecture=architecture, seed=101), torch.device('cpu'))
    recipe = P.recipes()['adamw_short_v1']
    optimizer = build_optimizer(model, recipe)
    scaler = torch.amp.GradScaler('cpu', init_scale=16.)
    row = D.training_step(model, optimizer, scaler, batch(), torch.device('cuda'), recipe, 0,
                          check_precision=True)
    observed = row['precision']['observed_activations']
    assert row['precision']['passed'] and observed['mod.blocks.0.fc1'] == 'torch.float16'
    assert set(observed.values()) == {'torch.float16'}
    assert all(p.dtype == torch.float32 for p in model.parameters())
    assert not row['amp_skipped'] and row['gradient_norm'] > 0
    assert model.mod.use_amp is False
    assert all(not m._forward_hooks for m in model.modules() if isinstance(m, (torch.nn.Linear, torch.nn.Conv1d)))


@pytest.mark.parametrize('architecture', ['part', 'part_ev', 'part_ev_context_relation_gate'])
def test_audit_catches_original_silent_fp32_bug_and_removes_hooks(architecture, emulate_cuda_autocast):
    model = build_model(architecture, use_amp=False, trim=False).eval()
    values, _ = D.tensors(batch(), torch.device('cpu'))
    with pytest.raises(RuntimeError, match='Precision audit failed'), torch.no_grad():
        # Reproduce v1: the outer AMP flag alone is insufficient.
        with torch.autocast('cuda', dtype=torch.float16), audit_precision(model, expected_dtype=torch.float16):
            model(*values)
    assert all(not m._forward_hooks for m in model.modules() if isinstance(m, (torch.nn.Linear, torch.nn.Conv1d)))


@pytest.mark.parametrize('architecture', list(COUNTS))
def test_validation_stays_fp32_even_with_ambient_amp_and_native_flag(architecture, emulate_cuda_autocast, monkeypatch):
    model = build_model(architecture, use_amp=False, trim=False).eval()
    values, _ = D.tensors(batch(), torch.device('cpu'))
    with torch.no_grad():
        expected = model(*values)
    seen = []

    def metrics(logits, labels):
        seen.append(torch.from_numpy(logits))
        return dict(events=len(labels))

    monkeypatch.setattr(D, 'classification_metrics', metrics)
    model.mod.use_amp = True
    with torch.autocast('cuda', dtype=torch.float16):
        row = D.validate(model, iter([batch()]), torch.device('cuda'), 2, batch_size=2, check_precision=True)
    assert row['precision']['passed']
    assert set(row['precision']['observed_activations'].values()) == {'torch.float32'}
    torch.testing.assert_close(seen[0], expected, rtol=0, atol=0)
    assert model.mod.use_amp is True


def test_precision_context_restores_flags_after_exceptions_and_nesting(emulate_cuda_autocast):
    model = build_model('part_ev', trim=False, use_amp=False)
    with pytest.raises(ValueError, match='fixture'):
        with model_precision(model, torch.device('cuda'), training=True):
            assert model.mod.use_amp is True
            with model_precision(model, torch.device('cuda'), training=False):
                assert model.mod.use_amp is False
            assert model.mod.use_amp is True
            raise ValueError('fixture')
    assert model.mod.use_amp is False


def test_cpu_training_remains_fp32():
    model = build_model('part_ev_context_relation_gate', trim=False, use_amp=True)
    with model_precision(model, torch.device('cpu'), training=True):
        assert model.mod.use_amp is False
        with audit_precision(model, expected_dtype=torch.float32) as record:
            model(*D.tensors(batch(), torch.device('cpu'))[0])
    assert record['passed'] and model.mod.use_amp is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA regression requires a real GPU')
@pytest.mark.parametrize('architecture', list(COUNTS))
def test_real_cuda_amp_forward_backward_and_fp32_validation(architecture, monkeypatch):
    device = torch.device('cuda')
    model = D.make_model(dict(architecture=architecture, seed=101), device)
    recipe = P.recipes()['adamw_short_v1']
    optimizer = build_optimizer(model, recipe)
    scaler = torch.amp.GradScaler('cuda', init_scale=16.)
    row = D.training_step(model, optimizer, scaler, batch(), device, recipe, 0, check_precision=True)
    assert row['precision']['passed'] and row['precision']['expected_activation_dtype'] == 'torch.float16'
    assert not row['amp_skipped'] and row['gradient_norm'] > 0
    monkeypatch.setattr(D, 'classification_metrics', lambda logits, labels: dict(events=len(labels)))
    val = D.validate(model, iter([batch()]), device, 2, batch_size=2, check_precision=True)
    assert val['precision']['passed'] and val['precision']['expected_activation_dtype'] == 'torch.float32'
    assert model.mod.use_amp is False
