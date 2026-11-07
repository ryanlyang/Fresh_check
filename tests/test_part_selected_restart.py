import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from scripts import diagnose_part_selected_restart as D
from part_ev_fullscale.checkpoint import capture_rank_state, restore_rank_state, seed_everything
from part_ev_fullscale.optimizer import build_optimizer
from tests.test_part_fullscale_model import bounded_cpu_threads


def test_complete_tree_scan_including_buffers_optimizer_rng_and_nonfinite():
    original = dict(model={'first': torch.zeros(128), 'buffer': torch.tensor(3)},
                    optimizer={'slow': [torch.zeros(2)]}, rng={'torch': torch.tensor([1], dtype=torch.uint8)})
    changed = copy.deepcopy(original)
    changed['model']['first'][0] = 1e-7
    changed['model']['buffer'] += 1
    changed['optimizer']['slow'][0][1] = 2.
    changed['rng']['torch'][0] = 3
    r = D.compare_tree(original, changed)
    assert not r['exact'] and r['differing_leaves'] == 4
    assert r['maximum_absolute_error'] == 2.
    assert r['first_mismatches'][0]['path'] == 'model.first'
    assert r['components']['optimizer']['maximum_absolute_error'] == 2.
    for a, b in (({'x': torch.tensor(float('nan'))}, {'x': torch.tensor(float('nan'))}),
                 ({'x': torch.zeros(1)}, {'x': torch.zeros(2)}),
                 ({'x': 1}, {'y': 1}), ({'x': (1,)}, {'x': [1]})):
        assert not D.compare_tree(a, b)['exact']
    # Report size is bounded, but all leaves contribute to statistics.
    r = D.compare_tree({str(i): torch.tensor(0.) for i in range(100)},
                       {str(i): torch.tensor(float(i+1)) for i in range(100)})
    assert r['differing_leaves'] == 100
    assert len(r['first_mismatches']) == len(r['largest_errors']) == 20
    assert r['maximum_absolute_error'] == 100.
    json.dumps(r, allow_nan=False)


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.BatchNorm1d(8),
                                       torch.nn.Dropout(.2), torch.nn.Linear(8, 2))
        self._counter = 0

    def forward(self, x):
        self._counter += 1
        return self.net(x)


def step(model, optimizer, scaler, batch, device, recipe, i):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups:
        group['lr'] = .001 if i < 700000 else .0001
    loss = model(batch[0]['x']).square().mean()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return dict(lr=optimizer.param_groups[0]['lr'], amp_skipped=False)


def toy_trial(optimizer=build_optimizer, stop=lambda: None):
    seed_everything(101)
    model = Toy().eval()
    batch = ({'x': torch.randn(8, 4)}, {'_label_': torch.zeros(8, dtype=torch.long)}, {})
    before = D.cpu_copy(dict(model=model.state_dict(), rng=capture_rank_state(model)))
    result = D.trials(model, batch, torch.device('cpu'), {'optimizer': 'Weaver_RAdam_plus_Lookahead'},
                      build_optimizer=optimizer, training_step=step, capture_rank_state=capture_rank_state,
                      restore_rank_state=restore_rank_state, check_stop=stop)
    assert not model.training and model._counter == 0
    assert D.compare_tree(before, dict(model=model.state_dict(), rng=capture_rank_state(model)))['exact']
    return result


def test_real_ranger_rng_dropout_bn_and_lookahead_roundtrip():
    r = toy_trial()
    assert r['exact_roundtrip'] and r['identical_initial_states']
    assert r['uninterrupted_repeat_exact'] and r['resumed_exact'] and r['update_sequences_equal']
    assert r['first_repeat_divergence'] is r['first_resume_divergence'] is None
    assert len(r['comparisons']['resumed_after_3']) == 10
    assert not r['authorizes_training'] and not r['authorizes_benchmark_retry']


def test_real_restore_defect_is_not_mistaken_for_gpu_noise():
    def broken_optimizer(model, recipe):
        optimizer = build_optimizer(model, recipe)
        original_load = optimizer.load_state_dict
        def load(state):
            original_load(state)
            optimizer.slow[0].add_(.1)
        optimizer.load_state_dict = load
        return optimizer
    r = toy_trial(broken_optimizer)
    assert r['uninterrupted_repeat_exact']
    assert not r['exact_roundtrip'] and not r['resumed_exact']
    assert r['first_resume_divergence'] == 3
    paths = r['roundtrip_checks'][0]['restored_state']['first_mismatches']
    assert paths[0]['path'].startswith('optimizer.slow')


def test_interrupt_restores_disposable_model():
    seed_everything(101)
    model = Toy().eval()
    batch = ({'x': torch.randn(8, 4)}, {'_label_': torch.zeros(8, dtype=torch.long)}, {})
    before = D.cpu_copy(dict(model=model.state_dict(), rng=capture_rank_state(model)))
    calls = 0
    def stop():
        nonlocal calls
        calls += 1
        if calls == 4:
            raise TimeoutError('diagnostic deadline')
    with pytest.raises(TimeoutError, match='deadline'):
        D.trials(model, batch, torch.device('cpu'), {'optimizer': 'Weaver_RAdam_plus_Lookahead'},
                 build_optimizer=build_optimizer, training_step=step, capture_rank_state=capture_rank_state,
                 restore_rank_state=restore_rank_state, check_stop=stop)
    assert not model.training and model._counter == 0
    assert D.compare_tree(before, dict(model=model.state_dict(), rng=capture_rank_state(model)))['exact']


@pytest.mark.parametrize('arch', ('part_match_shared', 'part_match_layerwise', 'state_shared', 'ev_layerwise'))
def test_actual_frozen_style_models_on_cpu(arch):
    from tests.test_part_selected_fullscale_model import factory, batch_for
    from part_selected_fullscale.steps import training_step
    from part_selected_fullscale.protocol import recipes
    r = D.trials(factory(arch), batch_for(arch), torch.device('cpu'), recipes()['ranger_full_v1'],
                 build_optimizer=build_optimizer, training_step=training_step,
                 capture_rank_state=capture_rank_state, restore_rank_state=restore_rank_state)
    assert r['exact_roundtrip'] and r['uninterrupted_repeat_exact'] and r['resumed_exact']


def test_baseline_and_all_eight_layerwise_state_counts():
    from part_ev_state_compare.model import build_model as model
    assert sum(p.numel() for p in model('part').parameters()) == 2143486
    assert sum(p.numel() for p in model('state_layerwise').parameters()) == 2327462
