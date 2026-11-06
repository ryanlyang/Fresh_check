import copy
import pytest
import torch
from part_selected_fullscale.model import build_model, CONFIGS, COUNTS, state_branches
from part_selected_fullscale.parity import selected_parity, full_recipe_restart
from part_selected_screen.normalization import identity_normalization
from part_selected_screen.model import build_model as original
from part_selected_fast.trees import PackedTrees
from tests.test_part_selected_screen import selected_inputs
from tests.test_part_fullscale_model import bounded_cpu_threads


def factory(arch, fast=False, optimized=True):
    torch.manual_seed(101)
    provider=PackedTrees(None,oracle='reference_for_tests') if fast else None
    return build_model(arch,normalization=identity_normalization(),backend='reference_for_tests',
                       provider=provider,optimized_state=optimized)


def batch_for(arch,particles=11):
    values=list(selected_inputs(batch=3,particles=particles))
    if not CONFIGS[arch]['selected']: values[1]=values[1][:,:17]
    return ({k:v for k,v in zip(('pf_points','pf_features','pf_vectors','pf_mask'),values)},
            {'_label_':torch.tensor([0,1,2])},{})


@pytest.mark.parametrize('arch',CONFIGS)
def test_actual_counts_structure_and_backward(arch):
    model=factory(arch)
    assert sum(p.numel() for p in model.parameters())==COUNTS[arch]
    batch=batch_for(arch)
    out=model(*(batch[0][k] for k in ('pf_points','pf_features','pf_vectors','pf_mask')))
    torch.nn.functional.cross_entropy(out,batch[1]['_label_']).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(b.ffn_dim==512 for b in model.mod.cls_blocks)
    assert all(b.ffn_dim==CONFIGS[arch]['width'] for b in model.mod.blocks)
    assert sum(x is not None for x in state_branches(model))==(8 if CONFIGS[arch]['state'] else 0)
    if arch.startswith('part_match'):
        assert all(type(b.attn) is torch.nn.MultiheadAttention for b in model.mod.blocks)


@pytest.mark.parametrize('arch',('selected_match_layerwise','selected_shared','selected_state_layerwise'))
def test_full_selected_execution_and_full_recipe_restart(arch):
    candidate,reference=factory(arch,fast=True),factory(arch)
    before=copy.deepcopy(candidate.state_dict())
    ordinary=(original('selected_layerwise_ev',normalization=identity_normalization(),backend='reference_for_tests')
              if CONFIGS[arch]['state'] else None)
    row=selected_parity(candidate,reference,batch_for(arch),torch.device('cpu'),ordinary)
    assert row['passed'] and row['full_recipe_restart']['optimizer_and_rng_exact']
    assert row['full_recipe_restart']['steps'][-4:]==[699999,700000,979999,980000]
    if ordinary is not None:
        assert row['integration']['state_gradient_tensors']==32
        assert row['nonzero_execution']['passed']
    for key,v in before.items(): torch.testing.assert_close(candidate.state_dict()[key],v,rtol=0,atol=0)


def test_selected_optimized_state_matches_original_chunks_with_nonzero_branch():
    fast=factory('selected_state_layerwise',fast=True)
    slow=factory('selected_state_layerwise',optimized=False)
    with torch.no_grad():
        for branch in state_branches(fast): branch.output.fill_(.01)
    slow.load_state_dict(fast.state_dict())
    batch=batch_for('selected_state_layerwise',particles=67)
    observations=[]
    for model in (slow,fast):
        model.mod.trimmer.enabled=False
        torch.manual_seed(998)
        out=model(*(batch[0][k] for k in ('pf_points','pf_features','pf_vectors','pf_mask')))
        out.square().mean().backward()
        observations.append((out.detach(),{k:p.grad.clone() for k,p in model.named_parameters()}))
    torch.testing.assert_close(observations[0][0],observations[1][0],rtol=1e-5,atol=1e-6)
    from part_selected_screen.memory import compare, PARITY
    for key,grad in observations[0][1].items():
        compare(observations[1][1][key],grad,PARITY['fp32'],gradients=True)
    assert fast._state_pair_cache.stem is None and fast._state_pair_cache.normalized is None


@pytest.mark.parametrize('arch',('part_match_shared','part_match_layerwise','state_shared','ev_layerwise'))
def test_standard_full_recipe_restart(arch):
    assert full_recipe_restart(factory(arch),batch_for(arch),torch.device('cpu'))['passed']
