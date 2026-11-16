"""Independent explicit directed-triangle oracle, used only on tiny fixtures."""
import torch
from . import contraction as C
from .neighbors import weights_from_distances,pack


def explicit(a,b,gate,alpha,graph):
    batch,n,_,d=a.shape
    jets=[]
    for z in range(batch):
        rows=[]
        for i in range(n):
            targets=[]
            for j in range(n):
                # Connected zeros even when the jet has no eligible triangle.
                value=(a[z,i,j]+b[z,i,j]+gate[z,i,j])*0
                den=0.
                if graph.valid[z,i] and graph.valid[z,j] and i!=j:
                    for k in range(n):
                        if k!=i and k!=j and graph.valid[z,k]:
                            weight=graph.weights[z,i,k].to(a.dtype)
                            value=value+weight*a[z,i,k]*b[z,k,j]
                            den=den+weight
                    if den>0:value=value/den*gate[z,i,j]
                targets.append(value)
            rows.append(torch.stack(targets))
        jets.append(torch.stack(rows))
    return torch.einsum('bhij,bijd->bhid',alpha,torch.stack(jets))


def fixture(device,n=11,*,ties=True,dtype=torch.float64):
    generator=torch.Generator(device=device).manual_seed(74)
    valid=torch.ones(2,n,dtype=torch.bool,device=device)
    valid[1,1::3]=False
    distances=torch.rand(2,n,n,generator=generator,device=device)
    if ties:distances[0]=0
    nu,ok=weights_from_distances(distances,valid)
    graph=pack(nu,valid,ok)
    mask=valid[:,:,None]&valid[:,None,:]&~torch.eye(n,device=device,dtype=torch.bool)
    factors=[torch.randn(2,n,n,3,generator=generator,device=device,dtype=dtype).masked_fill(~mask[...,None],0)
             for _ in range(3)]
    alpha=torch.rand(2,2,n,n,generator=generator,device=device,dtype=dtype)
    return [x.requires_grad_() for x in (*factors,alpha)],graph


def verify(device):
    values,graph=fixture(device)
    results=[];gradients=[]
    for function in (explicit,C.dense,lambda *args:C.sparse(*args,receiver_tile=3,sender_tile=4,intermediary_tile=2,tie_tile=3)):
        out=function(*values,graph);results.append(out)
        gradients.append(torch.autograd.grad(out.square().sum(),values))
    for index in (1,2):
        torch.testing.assert_close(results[0],results[index],rtol=1e-10,atol=1e-11)
        for a,b in zip(gradients[0],gradients[index]):torch.testing.assert_close(a,b,rtol=1e-9,atol=1e-10)
    wrong=C.dense(values[0],values[1].transpose(1,2),*values[2:],graph)
    if torch.allclose(results[0],wrong):raise ValueError('Directed oracle fixture cannot detect reversed sender leg')
    return dict(kind='triangle_oracle',passed=True,directed=True,all_cutoff_ties=True,gradients=True)
