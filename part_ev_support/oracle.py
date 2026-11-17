"""Independent small dense/loop references for support equations."""
import torch
from . import aggregation as A


def dense_statistics(logits,values,valid):
    dtype=torch.float64 if logits.dtype==torch.float64 else torch.float32
    with torch.autocast(logits.device.type,enabled=False):
        l,v=logits.to(dtype),values.to(dtype)
        healthy=torch.isfinite(l).logical_or(~valid[...,None]).all() & torch.isfinite(v).logical_or(~valid[...,None]).all()
        gates=l.sigmoid().masked_fill(~valid[...,None],0)
        v=v.masked_fill(~valid[...,None],0)
        total=torch.matmul(gates.transpose(-1,-2),v)
        count=gates.sum(-2);degree=valid.sum(-1)
        return count,total,degree,healthy & torch.isfinite(total).all()


def explicit(logits,values,valid):
    counts=[];totals=[];degrees=[]
    for batch in range(len(logits)):
        c=[];s=[];m=[]
        for receiver in range(logits.shape[1]):
            # Connected, safe zero even for excluded nonfinite entries.
            count=logits[batch,receiver].masked_fill(~valid[batch,receiver,:,None],0).sum(0)*0
            content=values[batch,receiver].masked_fill(~valid[batch,receiver,:,None],0).sum(0)*0
            total=count[:,None]+content[None,:];degree=0
            for sender in range(logits.shape[2]):
                if valid[batch,receiver,sender]:
                    gate=logits[batch,receiver,sender].sigmoid()
                    count=count+gate
                    total=total+gate[:,None]*values[batch,receiver,sender][None,:]
                    degree+=1
            c.append(count);s.append(total);m.append(degree)
        counts.append(torch.stack(c));totals.append(torch.stack(s));degrees.append(m)
    return torch.stack(counts),torch.stack(totals),torch.tensor(degrees,device=logits.device),torch.tensor(True,device=logits.device)


def verify(device):
    rng=torch.Generator(device=device).manual_seed(117)
    logits=torch.randn(2,5,7,4,dtype=torch.float64,device=device,generator=rng,requires_grad=True)
    values=torch.randn(2,5,7,16,dtype=torch.float64,device=device,generator=rng,requires_grad=True)
    valid=torch.ones(2,5,7,dtype=torch.bool,device=device);valid[0,::2,1::2]=False;valid[1,3:]=False
    results=[];gradients=[]
    for call in (explicit,dense_statistics,A.statistics):
        stats=call(logits,values,valid);f,ok=A.finish(*stats)
        if not ok:raise ValueError('Oracle unhealthy')
        results.append((*stats[:3],f));gradients.append(torch.autograd.grad(f.square().sum(),(logits,values)))
    for index in (1,2):
        for x,y in zip(results[0],results[index]):torch.testing.assert_close(x,y,rtol=1e-10,atol=1e-11)
        for x,y in zip(gradients[0],gradients[index]):torch.testing.assert_close(x,y,rtol=1e-9,atol=1e-10)
    return dict(kind='aggregation_oracle',passed=True,explicit_loops=True,gradients=True)
