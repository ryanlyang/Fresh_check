import copy
import io
import json
import numpy as np
import pytest
import torch
from part_ev_triangle_restart import engine as E
from part_ev_fullscale.checkpoint import seed_everything, capture_rank_state, restore_rank_state
from part_ev_fullscale.optimizer import build_optimizer
from tests.test_part_fullscale_model import bounded_cpu_threads


def test_comparison_scans_all_leaves_and_keeps_reports_bounded():
    a = dict(model={str(i): torch.zeros(3) for i in range(40)},
             optimizer={'slow': torch.zeros(4)}, rng={'numpy': np.zeros(3, dtype=np.uint32)}, progress=3)
    b = E.cpu_copy(a)
    for i, tensor in enumerate(b['model'].values()):
        tensor[0] = .001*(i+1)
    b['optimizer']['slow'][2] = 2
    b['rng']['numpy'][0] = 9
    b['progress'] = 4
    result = E.compare_tree(a, b)
    assert not result['exact'] and result['finite'] and result['differing_leaves'] == 43
    assert result['maximum_absolute_error'] == 9
    assert set(result['components']) == {'model', 'optimizer', 'rng', 'progress'}
    assert len(result['first_mismatches']) == len(result['largest_errors']) == 12
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('a,b', [
    (torch.tensor(float('inf')), torch.tensor(float('inf'))),
    (torch.tensor(float('nan')), torch.tensor(float('nan'))),
    (float('nan'), float('nan')), (torch.zeros(2), torch.zeros(3)),
    (torch.zeros(2), torch.zeros(2, dtype=torch.float64)),
    ({'x': 1}, {'y': 1}), ([1], (1,)), (True, 1), ([1], [1, 2]),
])
def test_bad_states_are_not_excused(a, b):
    result = E.compare_tree(a, b)
    assert not result['exact']
    json.dumps(result, allow_nan=False)


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.BatchNorm1d(8),
                                       torch.nn.Dropout(.2), torch.nn.Linear(8, 2))
        self._counter = 0

    def forward(self, x):
        self._counter += 1
        return self.net(x)


def experiment(*, defect=None, stop=lambda: None, toy=None, inputs=None):
    seed_everything(101)
    base = Toy() if toy is None else toy
    values = torch.randn(8, 4) if inputs is None else inputs
    optimizer = build_optimizer(base, dict(optimizer='Weaver_RAdam_plus_Lookahead'))
    scaler = torch.amp.GradScaler('cpu', enabled=False)
    boundaries = []
    rows = []

    def forward():
        return base(values) if inputs is None else base(*values)

    def step(progress):
        base.train()
        optimizer.zero_grad(set_to_none=True)
        before = optimizer.step_counter
        loss = forward().square().mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        return dict(loss=float(loss.detach()), skipped=0, scale=scaler.get_scale(),
                    lookahead_before=before, lookahead_after=optimizer.step_counter,
                    attempt=progress['attempt'])

    # Include populated moments and slow weights, not just an empty optimizer.
    for _ in range(4):
        step(dict(attempt=0))

    def snapshot(progress):
        return dict(model=base.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                    rng=capture_rank_state(base), progress=progress)

    initial = E.cpu_copy(snapshot(dict(attempt=19997, checks=0, fixture_cursor=0)))
    stream = io.BytesIO()
    torch.save(initial, stream)

    def disk():
        stream.seek(0)
        return torch.load(stream, weights_only=False)

    def restore(state):
        base.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        restore_rank_state(base, state['rng'])
        if defect == 'always_slow':
            optimizer.slow[0].add_(.01)

    def roundtrip(state):
        buffer = io.BytesIO()
        torch.save(state, buffer)
        buffer.seek(0)
        loaded = torch.load(buffer, weights_only=False)
        if defect == 'slow':
            loaded['optimizer']['slow'][0].add_(.1)
        elif defect == 'rng':
            loaded['rng']['torch'] = initial['rng']['torch'].clone()
        elif defect == 'cursor':
            loaded['progress']['fixture_cursor'] += 1
        return loaded

    def boundary(progress):
        base.eval()
        with torch.no_grad():
            assert forward().isfinite().all()
        boundaries.append(progress['attempt'])

    result = E.trials(initial, restore=restore, snapshot=snapshot, step=step, boundary=boundary,
                      disk_initial=disk, roundtrip=roundtrip, stop=stop,
                      emit=lambda name, row: rows.append(name))
    return result, boundaries, rows


def test_real_ranger_rng_bn_dropout_all_trials_and_validation_boundary():
    result, boundaries, rows = experiment()
    assert result['complete'] and result['all_initial_restores_exact'] and result['mid_roundtrip_exact']
    assert result['trial_update_schedules_equal'] and result['total_skipped_updates'] == 0
    assert boundaries == [20000]*6
    assert len([r for r in rows if '_update_' in r]) == 28
    assert result['low_instrumentation']['disk']['final_vs_memory']['exact']
    for name in ('memory_b', 'disk_start', 'resume_after_3'):
        assert result['trials'][name]['first_divergence'] is None
    assert not result['authorizes_training'] and not result['authorizes_benchmark_retry']
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('defect,path', [('slow', 'optimizer'), ('rng', 'rng'), ('cursor', 'progress')])
def test_restore_defects_collected_without_mislabeling_as_numerical_noise(defect, path):
    result, _, _ = experiment(defect=defect)
    assert result['complete'] and result['all_initial_restores_exact']
    assert not result['mid_roundtrip_exact']
    assert result['trials']['memory_b']['first_divergence'] is None
    row = result['trials']['resume_after_3']
    assert row['first_divergence'] == 3
    assert any(r['path'].startswith(path) for r in row['roundtrips'][0]['deserialization']['first_mismatches'])


def test_initial_loading_bug_is_visible_even_when_repeated_trials_agree():
    result, _, _ = experiment(defect='always_slow')
    assert not result['all_initial_restores_exact']
    assert not result['trials']['memory_a']['initial_restore']['exact']


def test_bounded_stop():
    def stop():
        raise TimeoutError('stop now')
    with pytest.raises(TimeoutError, match='stop now'):
        experiment(stop=stop)


def test_actual_triangle_model_restart_on_cpu():
    from part_ev_triangles import benchmark as C
    base = C.M.build_model(trim=True)
    C.active(base)
    values, _ = C.B.tensors(C.B.fixed_batch(4, 9), torch.device('cpu'))
    result, _, _ = experiment(toy=base, inputs=values)
    assert result['complete'] and result['all_initial_restores_exact'] and result['mid_roundtrip_exact']
    assert result['trial_update_schedules_equal']
    # CPU exactness is observed, not a substitute for the queued A100 evidence.
    assert result['trials']['memory_b']['first_divergence'] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cpu_gpu_state_comparison():
    assert E.compare_tree({'p': torch.ones(3)}, {'p': torch.ones(3, device='cuda')})['exact']


@pytest.mark.parametrize('loss_gate_failure', [False, True])
def test_driver_wiring_and_artifact_isolation_on_cpu(tmp_path, monkeypatch, loss_gate_failure):
    """Exercise the full driver; mock only admission/hardware/collectives, not science.

    This is not distributed/CUDA evidence. Those require the queued two-A100 job.
    """
    from types import SimpleNamespace
    from scripts import diagnose_part_ev_triangles_restart as D
    from part_ev_triangles import benchmark as C, protocol as science
    from part_ev_pair_readout import benchmark as readout
    from part_ev_triangle_restart import protocol as P
    import sys
    campaign, output = tmp_path/'campaign', tmp_path/'diagnostic'
    (campaign/'probes').mkdir(parents=True)
    output.mkdir()
    recipe = science.recipes()['ranger_full_v1']
    spec = dict(content_hash='fixture', recipe=recipe)
    model = C.M.build_model(trim=True)
    opt = build_optimizer(model, recipe)
    saved = dict(campaign_sha256='fixture', model=model.state_dict(), optimizer=opt.state_dict(),
                 scaler={}, rank_states=[capture_rank_state(model)]*2,
                 progress=dict(attempt=19997, checks=0, fixture_cursor=0))
    probe = campaign/'probes/restart.pt'
    torch.save(saved, probe)
    original_bytes = probe.read_bytes()
    plan = dict(campaign=str(campaign), artifacts={'probes/restart.pt': P.sha(probe)})
    monkeypatch.setattr(P, 'load_plan', lambda root: plan)
    monkeypatch.setattr(science, 'load', lambda root: spec)
    monkeypatch.setattr(sys, 'path', list(sys.path))
    monkeypatch.setattr(C.B, 'setup', lambda expected: (0, 2, torch.device('cpu')))
    monkeypatch.setattr(C.B, 'gather', lambda value, world: [copy.deepcopy(value) for _ in range(world)])
    monkeypatch.setattr(C.B, 'agree', lambda value, device, world: bool(value))
    monkeypatch.setattr(C.B, 'STOP', False)
    monkeypatch.setattr(C, 'root_call', lambda action, rank, world: action())
    monkeypatch.setattr(C, 'fresh', lambda device, world: (model, model))
    real_fixed = C.B.fixed_batch
    monkeypatch.setattr(C.B, 'fixed_batch', lambda *a: real_fixed(4, 9))
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda _: SimpleNamespace(name='A100 mock', total_memory=40*2**30))
    if loss_gate_failure:
        def failed_comparison(*_):
            raise AssertionError('small replay loss mismatch')
        monkeypatch.setattr(readout, 'compare', failed_comparison)
    D.diagnostic(output)
    report, rank = P.read(output/'report.json'), P.read(output/'rank_0.json')
    assert report['complete'] and not report['authorizes_training']
    assert rank['all_initial_restores_exact'] and rank['mid_roundtrip_exact']
    assert rank['trial_update_schedules_equal'] and rank['ranks_agree_on_update_schedule']
    assert rank['original_loss_gate']['passed'] is not loss_gate_failure
    assert (output/'roundtrip_rank_0.pt').is_file()
    assert probe.read_bytes() == original_bytes
    assert not (campaign/'report.json').exists()
    assert (output/'events_rank_0.jsonl').is_file()


def test_driver_errors_save_incomplete_evidence_without_report(tmp_path, monkeypatch):
    from scripts import diagnose_part_ev_triangles_restart as D
    def broken(_):
        raise ValueError('probe changed')
    monkeypatch.setattr(D, 'diagnostic', broken)
    monkeypatch.setenv('RANK', '1')
    with pytest.raises(ValueError, match='probe changed'):
        D.main(['--output', str(tmp_path)])
    row = json.loads((tmp_path/'error_rank_1.json').read_text())
    assert not row['complete'] and not row['authorizes_training']
    assert not (tmp_path/'report.json').exists()
