import importlib

import pytest
import torch

from part_ev_extensions_tigris import protocol as P, benchmark as B
from tests.test_part_fullscale_model import bounded_cpu_threads


@pytest.mark.parametrize('kind', list(P.PACKAGES))
def test_real_native_graph_and_full_state_resume(kind, tmp_path):
    C = importlib.import_module(P.PACKAGES[kind]+'.benchmark')
    base, model = C.fresh(torch.device('cpu'), 1)
    assert model is base and sum(p.numel() for p in base.parameters()) == P.COUNTS[kind]
    assert not any(isinstance(m, torch.nn.SyncBatchNorm) for m in base.modules())
    recipe = C.P.recipes()['ranger_full_v1']
    optimizer = C.build_optimizer(base, recipe)
    scaler = torch.amp.GradScaler('cpu', enabled=False)
    values, truth = C.B.tensors(C.B.fixed_batch(3, 9), torch.device('cpu'))
    for index in range(4): C.B.step(base, base, optimizer, scaler, values, truth, recipe, index, 1)
    heads = B.active_parameters(C, base, kind)
    assert heads and all(p.grad is not None and p.grad.isfinite().all() for p in heads)
    assert {id(p) for p in heads} <= {id(p) for group in optimizer.param_groups for p in group['params']}
    result, seconds = B.restart_probe(C, tmp_path, base, optimizer, scaler, values, truth,
                                     dict(content_hash='native-test', recipe=recipe))
    assert result == dict(passed=True, updates=7, ranks=1, full_state=True, rng_and_trimmers=True)
    assert seconds > 0 and len(optimizer.slow) == len(list(base.parameters()))
    saved = C.trusted_load(tmp_path/'probes/restart.pt')
    assert len(saved['rank_states']) == 1


@pytest.mark.parametrize('kind', list(P.PACKAGES))
def test_real_correctness_adapts_labels_not_model_or_tolerances(kind):
    C = importlib.import_module(P.PACKAGES[kind]+'.benchmark')
    old_policy = C.P.POLICY.copy()
    row = B.correctness(C, torch.device('cpu'))[0]
    assert row['passed'] and row['tolerances'] == C.P.PARITY
    assert row['execution'] == dict(world_size=1, batchnorm='native', distributed=False)
    kinds = {c['kind'] for c in row['checks']}
    assert 'active_native_global_batch' in kinds and 'active_SyncBN_DDP' not in kinds
    if kind != 'bias': assert row['delayed_activation']['successful_updates'] == 3
    if kind == 'memory': assert {'dense_recurrence_and_history', 'boundaries_masks_lifetime'} <= kinds
    assert C.P.POLICY == old_policy and C.P.POLICY['world_size'] == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Run the actual one-GH200 benchmark on Tigris')
@pytest.mark.parametrize('kind', list(P.PACKAGES))
def test_cuda_correctness(kind):
    if 'GH200' not in torch.cuda.get_device_name(): pytest.skip('GH200 required')
    C = importlib.import_module(P.PACKAGES[kind]+'.benchmark')
    assert B.correctness(C, torch.device('cuda'))[0]['passed']
