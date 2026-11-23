"""Exact serialization, bounded trajectories, fail-closed admission; CPU is not CUDA evidence."""
import copy
import io
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from part_ev_triangles import restart as R
from part_ev_triangles.restart_policy import POLICY, METADATA, validate_evidence
from part_ev_fullscale.checkpoint import capture_rank_state, restore_rank_state, seed_everything
from part_ev_fullscale.optimizer import build_optimizer
from tests.test_part_fullscale_model import bounded_cpu_threads


def comparison():
    return dict(passed=True, finite=True, exact_discrete=True, exact=True,
                leaves=100, floating_leaves=50, max_abs_error=0., max_budget_fraction=0., worst=[])


def evidence():
    trials = {}
    updates = [dict(attempt=19997+i, fixture_cursor=i, skipped=0, loss=1., lr=.001,
                    scale_before=32768., scale_after=32768.,
                    lookahead_before=i % 6, lookahead_after=(i+1) % 6) for i in range(7)]
    for name in POLICY['trials']:
        trials[name] = dict(initial_restore=comparison(), updates=copy.deepcopy(updates),
                            progress=dict(attempt=20004, checks=1, fixture_cursor=7))
        if name != 'reference': trials[name]['comparisons'] = [comparison() for _ in range(7)]
        if name == 'resume_after_3':
            trials[name].update(mid_deserialization=comparison(), mid_restore=comparison())
    return dict(METADATA, policy=copy.deepcopy(POLICY), ranks_agree_on_update_schedule=True,
                rank_evidence=[dict(rank=r, passed=True, checkpoint_roundtrip=comparison(),
                                    trials=copy.deepcopy(trials)) for r in (0, 1)])


def state():
    return dict(model={'weight': torch.ones(4), 'num_batches_tracked': torch.tensor(2)},
                optimizer=dict(inner=dict(state={0: dict(step=torch.tensor(4.), exp_avg=torch.zeros(4),
                                                        exp_avg_sq=torch.ones(4))},
                                          param_groups=[dict(lr=.001)]),
                               slow=[torch.ones(4)], step_counter=2),
                rng=dict(numpy=np.array([1, 2], dtype=np.uint32), torch=torch.tensor([1], dtype=torch.uint8)),
                progress=dict(attempt=19997), scaler=dict(scale=32768., _growth_tracker=4))


def test_small_training_drift_is_bounded_but_restoration_remains_exact():
    a, b = state(), state()
    b['model']['weight'][0] += 7e-6
    b['optimizer']['inner']['state'][0]['exp_avg'][0] += 3e-6
    b['optimizer']['slow'][0][0] += 1e-6
    result = R.compare_state(a, b)
    assert result['passed'] and not result['exact'] and result['max_budget_fraction'] < 1
    with pytest.raises(ValueError, match='Exact restart'): R.compare_state(a, b, exact=True)


@pytest.mark.parametrize('path,value', [
    (('model', 'weight'), torch.ones(4)+.01),
    (('model', 'num_batches_tracked'), torch.tensor(3)),
    (('optimizer', 'inner', 'state', 0, 'step'), torch.tensor(4.000001)),
    (('optimizer', 'inner', 'param_groups', 0, 'lr'), .00100001),
    (('optimizer', 'step_counter'), 3), (('progress', 'attempt'), 19998),
    (('scaler', 'scale'), 32768.0001),
    (('rng', 'numpy'), np.array([1, 3], dtype=np.uint32)),
    (('rng', 'torch'), torch.tensor([2], dtype=torch.uint8)),
    (('model', 'weight'), torch.ones(4, dtype=torch.float64)),
    (('model', 'weight'), torch.ones(5)),
    (('model', 'weight'), torch.full((4,), float('nan'))),
    (('optimizer', 'inner', 'state', 0, 'exp_avg'), torch.full((4,), float('inf'))),
])
def test_corruption_discrete_metadata_nonfinite_and_large_drift_fail(path, value):
    a, b = state(), state()
    target = b
    for key in path[:-1]: target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError): R.compare_state(a, b)


def test_absolute_cap_prevents_relative_tolerance_hiding_large_change():
    with pytest.raises(ValueError, match='bounds'):
        R.compare_state({'model': {'w': torch.tensor([10000.])}},
                        {'model': {'w': torch.tensor([10000.01])}})
    for value in (float('nan'), float('inf'), np.float32('inf')):
        with pytest.raises(ValueError): R.compare_state({'x': value}, {'x': value})


@pytest.mark.parametrize('fault', ['policy', 'missing_rank', 'duplicate_rank', 'missing_trial',
    'restore', 'roundtrip', 'finite', 'rng', 'drift', 'ratio', 'missing_comparison',
    'loss', 'nan_loss', 'skip', 'cursor', 'lr', 'scale', 'counter', 'progress', 'rank_schedule', 'old_header'])
def test_admission_rechecks_every_component(fault):
    row = evidence()
    rank = row['rank_evidence'][1]
    trial = rank['trials']['resume_after_3']
    c = trial['comparisons'][-1]
    if fault == 'policy': row['policy']['loss']['atol'] *= 2
    if fault == 'missing_rank': row['rank_evidence'].pop()
    if fault == 'duplicate_rank': rank['rank'] = 0
    if fault == 'missing_trial': rank['trials'].pop('repeat')
    if fault == 'restore': trial['initial_restore']['exact'] = False
    if fault == 'roundtrip': trial['mid_deserialization']['max_abs_error'] = 1e-12
    if fault == 'finite': c['finite'] = False
    if fault == 'rng': c['exact_discrete'] = False
    if fault == 'drift': c['max_abs_error'] = 1.01e-4
    if fault == 'ratio': c['max_budget_fraction'] = 1.01
    if fault == 'missing_comparison': trial['comparisons'].pop()
    if fault == 'loss': trial['updates'][-1]['loss'] += .001
    if fault == 'nan_loss': trial['updates'][-1]['loss'] = float('nan')
    if fault == 'skip': trial['updates'][-1]['skipped'] = 1
    if fault == 'cursor': trial['updates'][-1]['fixture_cursor'] += 1
    if fault == 'lr': trial['updates'][-1]['lr'] *= 2
    if fault == 'scale': trial['updates'][-1]['scale_after'] *= 2
    if fault == 'counter': trial['updates'][-1]['lookahead_after'] = 4
    if fault == 'progress': trial['progress']['checks'] = 0
    if fault == 'rank_schedule':
        for t in rank['trials'].values():
            for update in t['updates']: update['lr'] *= 2
    if fault == 'old_header': row = dict(METADATA)
    with pytest.raises(ValueError): validate_evidence(row)


def test_no_adaptive_or_blanket_tolerance_and_small_observed_loss_variation_passes():
    row = evidence()
    for rank in row['rank_evidence']:
        for name in ('repeat', 'disk_start', 'resume_after_3'):
            rank['trials'][name]['updates'][-1]['loss'] += 6.866455078125e-5
    validate_evidence(row)
    # The control itself must meet the same fixed limit; it cannot expand admission.
    row['rank_evidence'][0]['trials']['repeat']['updates'][0]['loss'] += .01
    with pytest.raises(ValueError, match='loss'): validate_evidence(row)


def experiment(defect=None):
    seed_everything(101)
    base = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.BatchNorm1d(8),
                               torch.nn.Dropout(.2), torch.nn.Linear(8, 2))
    opt = build_optimizer(base, dict(optimizer='Weaver_RAdam_plus_Lookahead'))
    scaler = torch.amp.GradScaler('cpu', enabled=False)
    inputs = torch.randn(8, 4)
    def step(cursor):
        base.train(); opt.zero_grad(set_to_none=True)
        before = opt.step_counter
        loss = base(inputs).square().mean()
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        return dict(attempt=cursor['attempt'], fixture_cursor=cursor['fixture_cursor'], loss=float(loss.detach()),
                    skipped=0, lr=opt.param_groups[0]['lr'], scale_before=1., scale_after=1.,
                    lookahead_before=before, lookahead_after=opt.step_counter)
    for _ in range(4): step(dict(attempt=0, fixture_cursor=0))
    def snapshot(progress):
        return dict(model=base.state_dict(), optimizer=opt.state_dict(), scaler=scaler.state_dict(),
                    rng=capture_rank_state(base), progress=progress)
    initial = R.cpu_copy(snapshot(dict(attempt=19997, checks=0, fixture_cursor=0)))
    def deserialize(state):
        buffer = io.BytesIO(); torch.save(state, buffer); buffer.seek(0)
        return torch.load(buffer, weights_only=False)
    def restore(state):
        base.load_state_dict(state['model']); opt.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler']); restore_rank_state(base, state['rng'])
        if defect == 'initial': opt.slow[0][0, 0] += 1e-6
    def mid(state):
        state = deserialize(state)
        if defect == 'slow': state['optimizer']['slow'][0][0, 0] += 1e-6
        if defect == 'rng': state['rng']['torch'][0] ^= 1
        if defect == 'progress': state['progress']['fixture_cursor'] += 1
        return state
    boundaries = []
    def boundary(progress):
        base.eval()
        with torch.no_grad(): base(inputs)
        boundaries.append(progress['attempt'])
    result = R.trials(initial, restore=restore, snapshot=snapshot, step=step, boundary=boundary,
                      disk_initial=lambda: deserialize(initial), roundtrip=mid)
    return result, boundaries


def test_real_ranger_dropout_bn_and_complete_state_roundtrips():
    result, boundaries = experiment()
    assert boundaries == [20000]*4
    row = dict(METADATA, policy=copy.deepcopy(POLICY), ranks_agree_on_update_schedule=True,
               rank_evidence=[dict(copy.deepcopy(result), rank=r) for r in (0, 1)])
    validate_evidence(row)
    assert all(c['exact'] for c in result['trials']['resume_after_3']['comparisons'])


@pytest.mark.parametrize('defect', ['initial', 'slow', 'rng', 'progress'])
def test_real_roundtrip_rejects_even_small_restoration_defects(defect):
    with pytest.raises(ValueError, match='Exact restart'): experiment(defect)


def test_gh200_still_uses_legacy_probe(monkeypatch):
    from part_ev_triangles import benchmark as C
    marker = object()
    monkeypatch.setattr(C, 'legacy_restart_probe', lambda *args: marker)
    assert C.restart_probe(None, None, None, None, None, None, None, {}, 0, 1) is marker


def test_probe_requires_new_policy_and_two_ranks(tmp_path):
    with pytest.raises(ValueError, match='two-rank'):
        R.probe(tmp_path, None, None, None, None, None, None, {}, 0, 1)
    with pytest.raises(ValueError, match='Fresh campaign'):
        R.probe(tmp_path, None, None, None, None, None, None, {}, 0, 2)


class SmallDDPModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 10)
        self.dropout = torch.nn.Dropout(.2)
        self._state_pair_cache = SimpleNamespace(clear=lambda: None)

    def forward(self, points, *_):
        return self.linear(self.dropout(points.mean(-1)))


def _audit_worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2)
    try:
        from part_ev_triangles import protocol as P
        seed_everything(101+rank)
        base = SmallDDPModel()
        wrapped = torch.nn.parallel.DistributedDataParallel(base)
        opt = build_optimizer(base, P.recipes()['ranger_full_v1'])
        scaler = torch.amp.GradScaler('cpu', enabled=False)
        values, truth = R.B.tensors(R.B.split_batch(R.B.fixed_batch(8, 5), rank, 2), torch.device('cpu'))
        # Nonzero inputs and populated moments/slow weights.
        values[0].fill_(.5)
        for i in range(6): R.B.step(base, wrapped, opt, scaler, values, truth, P.recipes()['ranger_full_v1'], i, 2)
        R.B.synchronize = lambda *args: dist.barrier()
        spec = dict(content_hash='fixture', recipe=P.recipes()['ranger_full_v1'], restart_policy=POLICY)
        real_load = R.trusted_load
        def corrupt(path):
            saved = real_load(path)
            if rank == 1 and Path(path).name == 'restart_mid_rank_1.pt':
                saved['optimizer']['slow'][0][0, 0] += 1e-6
            return saved
        R.trusted_load = corrupt
        try:
            R.probe(Path(output)/'corrupt', base, wrapped, opt, scaler, values, truth, spec, rank, 2)
            raise AssertionError('One-rank corruption was admitted')
        except ValueError as exc:
            assert 'Exact restart' in str(exc) and 'optimizer.slow' in str(exc)
        finally:
            R.trusted_load = real_load
        # Only one rank requests a cooperative stop; both must stop.
        R.B.STOP = rank == 1
        with pytest.raises(TimeoutError, match='interrupted'):
            R.probe(Path(output)/'stopped', base, wrapped, opt, scaler, values, truth, spec, rank, 2)
        R.B.STOP = False
        Path(output, f'audit_rank_{rank}.ok').touch()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason='Gloo unavailable')
def test_one_rank_audit_failure_and_stop_are_propagated_to_both_workers(tmp_path):
    context = mp.spawn(_audit_worker, args=((tmp_path/'rendezvous').as_uri(), str(tmp_path)),
                       nprocs=2, join=False)
    deadline = time.monotonic()+60
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline: pytest.fail('Collective failure handling timed out')
        assert all((tmp_path/f'audit_rank_{rank}.ok').is_file() for rank in (0, 1))
    finally:
        for process in context.processes:
            if process.is_alive(): process.terminate()
