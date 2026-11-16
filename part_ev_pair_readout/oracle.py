"""Small differentiable pooling oracle, also executed on the benchmark GPU."""
import torch
from . import pooling


def verify(device):
    gen = torch.Generator().manual_seed(807)
    scores = torch.randn(3,2,19,generator=gen,dtype=torch.float64).to(device).requires_grad_()
    values = torch.randn(3,19,32,generator=gen,dtype=torch.float64).to(device).requires_grad_()
    valid = torch.ones(3,19,device=device,dtype=torch.bool)
    valid[1,::2] = False
    valid[2] = False
    with torch.no_grad(): scores[0,:,1:4] = scores[0,:,0,None]  # exact ties
    reference = pooling.dense_reference(scores,values,valid)
    expected = torch.autograd.grad(reference.square().sum(),(scores,values),retain_graph=True)
    for chunk in (1,4,19):
        stats = None
        for start in range(0,19,chunk):
            row = pooling.statistics(scores[:,:,start:start+chunk],values[:,start:start+chunk],valid[:,start:start+chunk])
            stats = row if stats is None else pooling.merge(stats,row)
        actual, healthy = pooling.finish(stats)
        assert healthy
        torch.testing.assert_close(actual,reference,rtol=1e-11,atol=1e-11)
        grads = torch.autograd.grad(actual.square().sum(),(scores,values),retain_graph=True)
        for a,b in zip(grads,expected): torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-10)
    # Finite extremes and empty chunks; no hidden exponential overflow.
    extreme = scores.detach()*1000
    chunks = [pooling.statistics(extreme[:,:,i:i+3],values[:,i:i+3],valid[:,i:i+3]) for i in range(0,19,3)]
    joined = chunks[0]
    for row in chunks[1:]: joined = pooling.merge(joined,row)
    actual,ok = pooling.finish(joined)
    assert ok and actual[2].count_nonzero()==0
    torch.testing.assert_close(actual,pooling.dense_reference(extreme,values,valid),rtol=1e-10,atol=1e-10)
    return dict(kind='pooling_oracle',passed=True,forward=True,backward=True,ties=True,empty=True)
