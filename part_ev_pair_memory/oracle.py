"""Small independent dense mathematical oracle, never used by production forward."""
import copy
import torch
from torch.nn import functional as F
from . import model as M


def dense(state,weights,stem,query,valid,memory):
    # Deliberately do not call model normalize/observation/write/chunk helpers.
    def ln(x): return F.layer_norm(x,(x.shape[-1],),eps=1e-5)
    p=F.linear(ln(query.transpose(0,1)),state.receiver.weight)
    q=F.linear(ln(query.transpose(0,1)),state.sender.weight)
    g=F.linear(ln(stem.permute(0,2,3,1)),state.edge.weight)
    pair=(valid[:,:,None]&valid[:,None,:])[...,None]
    r=F.gelu(ln(g+p[:,:,None]+q[:,None]+p[:,:,None]*q[:,None])).masked_fill(~pair,0)
    current=torch.einsum('hor,bhir->bhio',state.output,torch.einsum('bhij,bijr->bhir',weights,r))
    history=torch.zeros_like(current)
    if state.layer>0:
        history=torch.einsum('hor,bhir->bhio',state.memory_reader,
                             torch.einsum('bhij,bijr->bhir',weights,memory.masked_fill(~pair,0)))
    new=None
    if state.layer<7:
        c=torch.tanh(F.linear(ln(r),state.memory_writer.weight,state.memory_writer.bias)).masked_fill(~pair,0)
        if state.layer==0: new=c
        else:
            a,b,d,beta=state.memory_gate.unbind()
            u=torch.sigmoid(a*c+b*memory+d*c*memory+beta)
            new=((1-u)*memory+u*c).masked_fill(~pair,0)
    return current,history,new


def verify(device):
    """Dense double oracle for all seven memory boundaries and reverse-mode grads."""
    torch.manual_seed(927)
    model=M.build_model(trim=False,chunk=3).to(device).double()
    with torch.no_grad():
        for s in M.states(model):
            s.output.normal_(std=.02)
            if s.memory_reader is not None:s.memory_reader.normal_(std=.02)
            if s.memory_gate is not None:s.memory_gate[:3].normal_(std=.1)
    oracle=copy.deepcopy(model)
    n,b=5,2
    queries=[torch.randn(n,b,128,device=device,dtype=torch.float64,requires_grad=True) for _ in range(8)]
    stem=torch.randn(b,64,n,n,device=device,dtype=torch.float64,requires_grad=True)
    weights=[torch.randn(b,8,n,n,device=device,dtype=torch.float64).softmax(-1).requires_grad_() for _ in range(8)]
    valid=torch.ones(b,n,dtype=torch.bool,device=device);valid[1,-2:]=False
    losses=[];outputs=[]
    for target,is_dense in ((model,False),(oracle,True)):
        memory=None;results=[]
        for s,w,q in zip(M.states(target),weights,queries):
            if is_dense:c,h,memory=dense(s,w,stem,q,valid,memory)
            else:
                c,h,memory,ok=s(w,stem,q,valid,memory)
                if not ok.item():raise ValueError('Nonfinite oracle fixture')
            results.extend((c,h))
            if memory is not None:results.append(memory)
        outputs.append(results)
        losses.append(sum(x.square().sum() for x in results))
    for a,b in zip(*outputs):torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-11)
    old=[p for s in M.states(model) for p in s.parameters()]
    ref=[p for s in M.states(oracle) for p in s.parameters()]
    inputs=[stem,*queries,*weights]
    ag=torch.autograd.grad(losses[0],[*inputs,*old])
    eg=torch.autograd.grad(losses[1],[*inputs,*ref])
    for a,b in zip(ag,eg):torch.testing.assert_close(a,b,rtol=1e-9,atol=1e-10)
    # Hold later states, geometry AND attention fixed: only the first observation
    # changes. A final-block history difference must come through carried memory.
    states=M.states(model)
    def final_history(first):
        model._state_pair_cache.clear()
        memory=None
        for i,s in enumerate(states):
            _,h,memory,ok=s(weights[i],stem,first if i==0 else queries[i],valid,memory)
        return h
    final=final_history(queries[0])
    first_grad=torch.autograd.grad(final.square().sum(),states[0].memory_writer.weight,retain_graph=True)[0]
    if not first_grad.isfinite().all() or not first_grad.count_nonzero():
        raise ValueError('Final reader cannot learn from the first writer')
    altered=final_history(queries[0]+torch.randn_like(queries[0])*.2)
    if torch.allclose(final,altered,rtol=1e-8,atol=1e-10):raise ValueError('No cross-layer history dependence')
    return dict(kind='dense_recurrence_and_history',passed=True,layers=8,boundaries=7,
                final_reader_first_writer_gradient=True,earlier_observation_changes_later_read=True)
