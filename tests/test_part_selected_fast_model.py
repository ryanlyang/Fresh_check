import copy
import hashlib
import numpy as np
import pytest
import torch
from part_selected_fast.cache import TreeCache
from part_selected_fast.trees import PackedTrees, widths, encode, decode, oracle_batch
from part_selected_fast.model import build_model
from part_selected_fast.parity import execution_parity
from part_selected_screen.model import build_model as reference_model
from part_selected_screen.normalization import identity_normalization
from part_selected_screen.protocol import COUNTS
from part_selected_screen.data import raw_tokens
from tests.test_part_selected_screen import selected_inputs
from tests.test_part_fullscale_model import bounded_cpu_threads


def factory(arch):
    normal=identity_normalization()
    provider=PackedTrees(None,oracle='reference_for_tests',cache=TreeCache(200000),namespace='test')
    torch.manual_seed(101)
    reference=reference_model(arch,normalization=normal,backend='reference_for_tests')
    torch.manual_seed(101)
    candidate=build_model(arch,normalization=normal,backend='reference_for_tests',provider=provider,cache_validation=True)
    return reference,candidate


@pytest.mark.parametrize('arch',COUNTS)
def test_all_modes_gradients_native_trimming_optimizer_and_state_restoration(arch):
    reference,candidate=factory(arch)
    p,f,v,m=selected_inputs()
    batch=({k:x for k,x in zip(('pf_points','pf_features','pf_vectors','pf_mask'),(p,f,v,m))},
           {'_label_':torch.tensor([0,1,2])},{})
    before=copy.deepcopy(candidate.state_dict())
    result=execution_parity(candidate,reference,batch,torch.device('cpu'))
    assert result['passed'] and result['trees']['exact']
    assert sum(p.numel() for p in candidate.parameters())==COUNTS[arch]
    for k,v in before.items(): torch.testing.assert_close(candidate.state_dict()[k],v,rtol=0,atol=0)
    assert all(p.grad is None for p in candidate.parameters())


def test_cache_bounded_eviction_exact_keys_and_immutable_records():
    cache=TreeCache(100000)
    assert cache.allocated_bytes <= cache.budget_bytes
    keys=[hashlib.sha256(str(i).encode()).digest() for i in range(20)]
    for key in keys: cache.put(key,b'record')
    assert cache.evictions>0
    assert cache.get(keys[-1])==b'record'
    assert cache.get(b'x'*32) is None
    cache.clear(); assert cache.get(keys[-1]) is None
    with pytest.raises(ValueError): cache.put(keys[0],torch.ones(3))
    with pytest.raises(ValueError): cache.put(keys[0],b'x'*(cache.SLOT_BYTES+1))
    zero=TreeCache(0); zero.put(keys[0],b'x'); assert zero.get(keys[0]) is None


def test_packed_cache_retained_particle_miss_and_empty_jet():
    p,f,v,m=selected_inputs(batch=3,particles=11)
    m[0]=0
    provider=PackedTrees(None,oracle='reference_for_tests',cache=TreeCache(1000000),namespace='test')
    raw=raw_tokens(f,v,m)
    first=provider(v,raw,m,use_cache=True)
    again=provider(v,raw,m,use_cache=True)
    assert provider.cache.hits==3
    for k in first: torch.testing.assert_close(first[k],again[k],rtol=0,atol=0)
    m[1,0,0]=False
    provider(v,raw_tokens(f,v,m),m,use_cache=True)
    assert provider.cache.misses==4
    a,b=encode(first)
    roundtrip=decode(a,b,11,'cpu')
    for k in first: assert torch.equal(first[k],roundtrip[k])


def test_reject_bad_schema_and_threads():
    for n in (0,129,True):
        with pytest.raises(ValueError): widths(n)
    with pytest.raises(ValueError): PackedTrees(None)
    with pytest.raises(ValueError): PackedTrees(None,threads=0,oracle='reference_for_tests')
    with pytest.raises(ValueError): decode(np.zeros((2,3),np.int64),np.zeros((2,4),np.float32),2,'cpu')
