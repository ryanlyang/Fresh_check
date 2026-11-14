"""Exercise the real memory graph with the production restart/validation helpers.

CPU/Gloo uses native BatchNorm; the admitted A100 benchmark covers SyncBatchNorm
and AMP. Tiny synthetic batches/check intervals here are test fixtures only.
"""
import copy
from pathlib import Path
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from part_ev_state_production import engine as E, protocol as P
from part_ev_pair_memory import model as M, protocol as MP, benchmark as C
from tests.test_part_fullscale_model import bounded_cpu_threads


def memory_spec():
    recipe = copy.deepcopy(MP.recipes())
    recipe[MP.TASK['recipe']].update(total_steps=21, steps_per_epoch=7, epochs=3)
    return dict(content_hash='memory-production-test', model_kind='memory',
                task=MP.TASK, tasks=[MP.TASK], recipes=recipe,
                parameters={MP.TASK['architecture']: M.PARAMETERS})


def assert_exact(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_exact(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_exact(a, b)
    else:
        assert actual == expected


def _memory_worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2)
    try:
        device = torch.device('cpu')
        spec = memory_spec()
        recipe = spec['recipes'][spec['task']['recipe']]
        E.B.seed_everything(101)
        base = E.build_model(spec)
        model = torch.nn.parallel.DistributedDataParallel(
            base, find_unused_parameters=False, gradient_as_bucket_view=True)
        opt = E.build_optimizer(base, recipe)
        scale = torch.amp.GradScaler('cpu', enabled=False)
        E.B.seed_everything(101 + rank)
        values, truth = E.B.tensors(E.B.split_batch(E.B.fixed_batch(6, 9), rank, 2), device)
        for i in range(7):
            assert E.B.step(base, model, opt, scale, values, truth, recipe, i, 2)['skipped'] == 0
        memory_parameters = [*M.readers(base), *M.writers_and_gates(base)]
        assert all(p.grad is not None and p.grad.isfinite().all()
                   and p.grad.count_nonzero() > 0 for p in memory_parameters)
        assert {id(p) for p in memory_parameters} <= {
            id(p) for group in opt.param_groups for p in group['params']}
        assert len(opt.slow) == len(list(base.parameters()))

        def validate():
            loader = iter([E.B.fixed_batch(20, 9)]) if rank == 0 else None
            result = E.validate(base, loader, device, rank, 2, 20, batch_size=20, audit=True)
            assert result['events'] == 20
            return {k: v for k, v in result.items() if k not in ('seconds', 'precision')}

        first = validate()
        state = E.initial_state(spec)
        state.update(step=7, validated_checks=1, best_check=1,
                     best_accuracy=first['accuracy'], best_file='selected/check_01.pt')
        runtime = dict(world_size=2, batchnorm='BatchNorm', precision='test_cpu_fp32')
        path = Path(output) / 'last.pt'
        E.save(path, base, opt, scale, state, spec, runtime, rank, 2)
        saved = E.trusted_load(path)
        assert set(saved) == {'schema', 'campaign_sha256', 'runtime', 'model',
                              'optimizer', 'scaler', 'state', 'rank_states'}
        assert saved['state']['task'] == MP.TASK
        assert len(saved['rank_states']) == 2

        def continue_training():
            # Cross a Lookahead update after the saved checkpoint, then validate
            # on rank zero only, exactly as the production loop does.
            for i in range(7, 14):
                E.B.step(base, model, opt, scale, values, truth, recipe, i, 2)
            metric = validate()
            return C.cpu_copy((base.state_dict(), opt.state_dict(), scale.state_dict())), metric

        expected, metric = continue_training()
        expected_rng = E.capture_rank_state(base)
        restored, rng = E.restore(path, base, opt, scale, spec, runtime, rank, 2)
        assert restored == state
        E.restore_rank_state(base, rng)
        actual, resumed_metric = continue_training()
        assert_exact(actual, expected)
        assert_exact(resumed_metric, metric)
        C.assert_rng(E.capture_rank_state(base), expected_rng)
        assert len(set(E.B.gather(E.B.parameter_digest(base), 2))) == 1
        E.root_call(lambda: P.write_json(Path(output) / 'passed.json', dict(passed=True)), rank, 2)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason='Gloo unavailable')
def test_memory_production_two_rank_restart_after_validation(tmp_path):
    context = mp.spawn(_memory_worker,
                       args=((tmp_path / 'rendezvous').as_uri(), str(tmp_path)),
                       nprocs=2, join=False)
    deadline = time.monotonic() + 120
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail('Production memory recovery timed out')
        assert P.read_json(tmp_path / 'passed.json')['passed']
    finally:
        for child in context.processes:
            if child.is_alive():
                child.terminate()


def test_memory_production_eval_does_not_carry_state_between_batches():
    base = E.build_model(memory_spec()).eval()
    C.active(base)  # Nonzero readers/writers; zero initialization cannot hide leaks.
    device = torch.device('cpu')
    values, _ = E.B.tensors(E.B.fixed_batch(3, 9), device)
    unrelated, _ = E.B.tensors(E.B.fixed_batch(2, 13), device)
    weights = C.cpu_copy(base.state_dict())
    with torch.no_grad(), E.model_precision(base, device, training=False):
        first = base(*values)
        base(*unrelated)
        repeated = base(*values)
    torch.testing.assert_close(first, repeated, rtol=0, atol=0)
    assert_exact(base.state_dict(), weights)
    for state in M.states(base):
        assert state.cache.stem is None
        assert state.cache.normalized is None
