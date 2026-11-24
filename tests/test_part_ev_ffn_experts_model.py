import copy
import pytest
import torch
from part_ev_ffn_experts.model import build_model,banks
from part_ev_ffn_experts.losses import balancing
from part_ev_ffn_experts.precision import audit_precision,modules
from part_ev_state_compare.model import state_branches
from part_ev_gpu_scaling.benchmark import fixed_batch,tensors

torch.set_num_threads(1)


def active(model):
    with torch.no_grad():
        for branch in state_branches(model):branch.output.normal_(std=.02)
        for bank in banks(model):
            for e in bank.experts:e.output.weight.normal_(std=.02)


def test_counts_keys_storage_initial_logits_ce_gradients_and_rng():
    torch.manual_seed(101);control=build_model('control',trim=False);rng=torch.get_rng_state()
    torch.manual_seed(101);model=build_model(trim=False)
    assert torch.equal(torch.get_rng_state(),rng)
    assert sum(p.numel() for p in model.parameters())==3998078
    for name,value in control.state_dict().items():torch.testing.assert_close(value,model.state_dict()[name],rtol=0,atol=0)
    assert len({e.input.weight.data_ptr() for b in banks(model) for e in b.experts})==24
    with torch.no_grad():
        for branch in state_branches(control):branch.output.normal_(std=.02)
        for block in control.mod.blocks:block.c_attn.uniform_(.7,1.3);block.w_resid.uniform_(.7,1.3)
    state=model.state_dict();state.update(control.state_dict());model.load_state_dict(state)
    values,y=tensors(fixed_batch(3,7),torch.device('cpu'))
    records=[]
    for m in (control,model):
        torch.manual_seed(91);out=m(*values)
        F=torch.nn.functional.cross_entropy(out,y);F.backward();records.append((out,torch.get_rng_state()))
    torch.testing.assert_close(records[0][0],records[1][0],rtol=1e-5,atol=1e-6)
    assert torch.equal(records[0][1],records[1][1])
    for name,p in control.named_parameters():torch.testing.assert_close(p.grad,dict(model.named_parameters())[name].grad,rtol=1e-4,atol=1e-5)
    assert all(p.grad is not None for p in model.parameters())
    assert all(e.input.weight.grad.count_nonzero()==0 for b in banks(model) for e in b.experts)


def test_active_checkpoint_backward_auxiliary_and_native_call_counts():
    torch.manual_seed(102);fast=build_model(trim=False);active(fast)
    slow=copy.deepcopy(fast)
    for b in banks(slow):b.checkpointed=False
    calls={};handles=[]
    selected={'trim':fast.mod.trimmer,'embed':fast.mod.embed,'pairs':fast.mod.pair_embed}
    selected.update({f'attn_{i}':b.attn for i,b in enumerate(fast.mod.blocks)})
    selected.update({f'fc1_{i}':b.fc1 for i,b in enumerate(fast.mod.blocks)})
    selected.update({f'fc2_{i}':b.fc2 for i,b in enumerate(fast.mod.blocks)})
    selected.update({f'class_{i}':b.attn for i,b in enumerate(fast.mod.cls_blocks)})
    selected.update({f'bank_{i}':b for i,b in enumerate(banks(fast))})
    for name,m in selected.items():handles.append(m.register_forward_hook(lambda m,a,o,name=name:calls.__setitem__(name,calls.get(name,0)+1)))
    x,y=tensors(fixed_batch(3,9),torch.device('cpu'))
    for m in (fast,slow):
        torch.manual_seed(13);out=m(*x,return_router_stats=True)
        balance,_=balancing(out['banks'],1,torch.device('cpu'))
        (torch.nn.functional.cross_entropy(out['logits'],y)+.001*balance).backward()
        assert m._state_pair_cache.stem is None
    for h in handles:h.remove()
    assert calls=={n:1 for n in selected}
    for name,p in fast.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
        torch.testing.assert_close(p.grad,dict(slow.named_parameters())[name].grad,rtol=1e-4,atol=1e-5)
    fast.eval()
    with torch.no_grad():
        first=fast(*x);fast(*tensors(fixed_batch(3,3),torch.device('cpu'))[0]);again=fast(*x)
    torch.testing.assert_close(first,again,rtol=1e-5,atol=1e-6)
    assert isinstance(first,torch.Tensor) and first.shape==(3,10)


def test_fp32_precision_audit_and_missing_expert_key_rejection():
    m=build_model(trim=False).eval()
    with torch.no_grad(),audit_precision(m,expected_dtype=torch.float32) as observed:
        m(*tensors(fixed_batch(3,5),torch.device('cpu'))[0])
    assert observed['passed'] and len(observed['router_paths'])==12
    assert set(observed['observed_activations'])==set(modules(m))
    state=m.state_dict();state.pop('mod.blocks.2.specialists.experts.3.input.weight')
    with pytest.raises(RuntimeError):m.load_state_dict(state,strict=True)
