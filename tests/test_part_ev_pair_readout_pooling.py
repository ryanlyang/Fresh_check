import pytest
import torch
from part_ev_pair_readout import pooling as P
from part_ev_pair_readout.oracle import verify
from tests.test_part_fullscale_model import bounded_cpu_threads


def test_independent_oracle():
    assert verify(torch.device('cpu'))['passed']


def test_gradcheck_ties_empty_and_shift():
    torch.manual_seed(11)
    scores=torch.zeros(2,2,5,dtype=torch.float64,requires_grad=True)
    values=torch.randn(2,5,3,dtype=torch.float64,requires_grad=True)
    valid=torch.tensor([[True,False,True,True,False],[False]*5])
    def pool(s,v):
        a=P.statistics(s[:,:,:2],v[:,:2],valid[:,:2])
        b=P.statistics(s[:,:,2:],v[:,2:],valid[:,2:])
        return P.finish(P.merge(a,b))[0]
    assert torch.autograd.gradcheck(pool,(scores,values))
    torch.testing.assert_close(pool(scores,values),pool(scores+1000,values))
    assert pool(scores,values)[1].count_nonzero()==0


def test_chunk_averaging_is_wrong_and_padding_does_not_leak():
    scores=torch.tensor([[[0.,0.,8.]]],requires_grad=True)
    values=torch.tensor([[[1.],[3.],[10.]]],requires_grad=True)
    valid=torch.ones(1,3,dtype=torch.bool)
    a=P.statistics(scores[:,:,:2],values[:,:2],valid[:,:2])
    b=P.statistics(scores[:,:,2:],values[:,2:],valid[:,2:])
    correct,ok=P.finish(P.merge(a,b))
    wrong=(P.finish(a)[0]+P.finish(b)[0])/2
    assert ok and not torch.allclose(correct,wrong)
    empty=P.statistics(torch.full((1,1,2),float('nan')),torch.full((1,2,1),float('nan')),torch.zeros(1,2,dtype=torch.bool))
    torch.testing.assert_close(P.finish(P.merge(a,empty))[0],P.finish(a)[0])
    bad=P.statistics(scores*float('inf'),values,valid)
    assert not bad[-1]


def test_pooling_precision_under_autocast():
    scores=torch.randn(2,2,7);values=torch.randn(2,7,32);valid=torch.ones(2,7,dtype=torch.bool)
    with torch.autocast('cpu',dtype=torch.bfloat16):
        row=P.statistics(scores,values,valid);result,ok=P.finish(row)
    assert result.dtype==row[2].dtype==row[3].dtype==torch.float32 and ok
