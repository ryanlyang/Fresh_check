"""Bounded sparse directed j->k->i contraction; no all-triplet tensor."""
import torch
from torch.utils.checkpoint import checkpoint


def call(function,*args,checkpointed=True):
    if checkpointed and torch.is_grad_enabled():
        return checkpoint(function,*args,use_reentrant=False,preserve_rng_state=False)
    return function(*args)


def finish_tile(numerator,gate,nu,valid,receivers,senders):
    # Row mass is fractional at cutoff ties. k=j is REMOVED, not counted as 0.
    denominator=nu.sum(-1,keepdim=True)-nu[...,senders]
    target=valid[...,None] & (receivers[...,None]!=senders)
    # Sender validity is represented by gate=0 and also applied explicitly by callers.
    live=target & (denominator>0)
    safe=torch.where(live,denominator,torch.ones_like(denominator))
    return (gate*numerator/safe[...,None]).masked_fill(~live[...,None],0)


def ordinary_tile(a,b,gate,alpha,nu,valid,indices,candidate_weights,start,stop,jstart,jstop,ktile):
    with torch.profiler.record_function('triangles.ordinary_tile'):
        return _ordinary_tile(a,b,gate,alpha,nu,valid,indices,candidate_weights,start,stop,jstart,jstop,ktile)


def _ordinary_tile(a,b,gate,alpha,nu,valid,indices,candidate_weights,start,stop,jstart,jstop,ktile):
    batch,n,_,width=a.shape
    with torch.autocast(a.device.type,enabled=False):
        receivers=torch.arange(start,stop,device=a.device)
        senders=torch.arange(jstart,jstop,device=a.device)
        base=torch.arange(batch,device=a.device)[:,None,None]*n*n
        numerator=torch.zeros(batch,stop-start,jstop-jstart,width,dtype=a.dtype,device=a.device)
        for kstart in range(0,indices.shape[-1],ktile):
            ks=indices[:,start:stop,kstart:kstart+ktile]
            w=candidate_weights[:,start:stop,kstart:kstart+ktile]
            ai=base+receivers[None,:,None]*n+ks
            bi=base[:,:,:,None]+ks[:,:,:,None]*n+senders
            av=a.reshape(-1,width).index_select(0,ai.flatten()).reshape(*ks.shape,width)
            bv=b.reshape(-1,width).index_select(0,bi.flatten()).reshape(*ks.shape,len(senders),width)
            ew=w[:,:,:,None]*(ks[:,:,:,None]!=senders)
            numerator=numerator+torch.einsum('bqkj,bqkd,bqkjd->bqjd',ew,av,bv)
        g=gate[:,start:stop,jstart:jstop]
        t=finish_tile(numerator,g,nu[:,start:stop],valid[:,start:stop],receivers,senders)
        t=t.masked_fill(~valid[:,None,jstart:jstop,None],0)
        return torch.einsum('bhqj,bqjd->bhqd',alpha[:,:,start:stop,jstart:jstop].to(a.dtype),t)


def tied_tile(a,b,gate,alpha,nu,valid,rows,indices,candidate_weights,jstart,jstop,ktile):
    with torch.profiler.record_function('triangles.tied_tile'):
        return _tied_tile(a,b,gate,alpha,nu,valid,rows,indices,candidate_weights,jstart,jstop,ktile)


def _tied_tile(a,b,gate,alpha,nu,valid,rows,indices,candidate_weights,jstart,jstop,ktile):
    batch,n,_,width=a.shape
    with torch.autocast(a.device.type,enabled=False):
        batches=rows//n;receivers=rows%n;senders=torch.arange(jstart,jstop,device=a.device)
        numerator=torch.zeros(len(rows),len(senders),width,dtype=a.dtype,device=a.device)
        for kstart in range(0,indices.shape[-1],ktile):
            ks=indices[:,kstart:kstart+ktile];w=candidate_weights[:,kstart:kstart+ktile]
            ai=rows[:,None]*n+ks
            bi=batches[:,None,None]*n*n+ks[:,:,None]*n+senders
            av=a.reshape(-1,width).index_select(0,ai.flatten()).reshape(*ks.shape,width)
            bv=b.reshape(-1,width).index_select(0,bi.flatten()).reshape(*ks.shape,len(senders),width)
            ew=w[:,:,None]*(ks[:,:,None]!=senders)
            numerator=numerator+torch.einsum('rkj,rkd,rkjd->rjd',ew,av,bv)
        g=gate.reshape(batch*n,n,width)[rows,jstart:jstop]
        t=finish_tile(numerator,g,nu.reshape(batch*n,n)[rows],valid.flatten()[rows],receivers,senders)
        t=t.masked_fill(~valid[batches,jstart:jstop,None],0)
        weights=alpha[batches,:,receivers,jstart:jstop].to(a.dtype)
        return torch.einsum('rhj,rjd->rhd',weights,t)


def sparse(a,b,gate,alpha,graph,*,receiver_tile=16,sender_tile=64,intermediary_tile=4,tie_tile=256,checkpointed=True):
    """Caches must already mask invalid/diagonal factors. All copies are tiled."""
    if any(type(x) is not int or x<1 for x in (receiver_tile,sender_tile,intermediary_tile,tie_tile)):
        raise ValueError('Positive execution tiles required')
    nu,valid=graph.weights,graph.valid
    batch,n=valid.shape
    # Double is retained by mathematical tests; real caches are ALWAYS FP32.
    dtype=torch.float64 if a.dtype==torch.float64 else torch.float32
    a,b,gate=a.to(dtype),b.to(dtype),gate.to(dtype)
    with torch.autocast(a.device.type,enabled=False):
        outputs=[]
        with torch.profiler.record_function('triangles.ordinary_mix'):
            for start in range(0,n,receiver_tile):
                current=None
                for j in range(0,n,sender_tile):
                    row=call(ordinary_tile,a,b,gate,alpha,nu.to(dtype),valid,graph.indices,graph.candidate_weights.to(dtype),
                        start,min(n,start+receiver_tile),j,min(n,j+sender_tile),intermediary_tile,checkpointed=checkpointed)
                    current=row if current is None else current+row
                outputs.append(current)
        result=torch.cat(outputs,2)
        with torch.profiler.record_function('triangles.tie_mix'):
            for start in range(0,len(graph.tie_rows),tie_tile):
                rows=graph.tie_rows[start:start+tie_tile];current=None
                for j in range(0,n,sender_tile):
                    row=call(tied_tile,a,b,gate,alpha,nu.to(dtype),valid,rows,
                        graph.tie_indices[start:start+tie_tile],graph.tie_weights[start:start+tie_tile].to(dtype),
                        j,min(n,j+sender_tile),intermediary_tile,checkpointed=checkpointed)
                    current=row if current is None else current+row
                # Each packed receiver occurs ONCE. No duplicate-index scatter here.
                flat=result.permute(0,2,1,3).reshape(batch*n,alpha.shape[1],a.shape[-1])
                result=flat.index_add(0,rows,current).reshape(batch,n,alpha.shape[1],a.shape[-1]).permute(0,2,1,3)
        return result


def dense_features(a,b,gate,graph):
    """Small-test oracle: per-channel GEMM, never [B,N,N,N,d]."""
    nu,valid=graph.weights.to(a.dtype),graph.valid
    n=valid.shape[-1]
    with torch.autocast(a.device.type,enabled=False):
        pair=valid[:,:,None]&valid[:,None,:]&~torch.eye(n,device=a.device,dtype=torch.bool)
        left=(nu[...,None]*a).masked_fill(~pair[...,None],0)
        right=b.masked_fill(~pair[...,None],0)
        numerator=torch.einsum('bikd,bkjd->bijd',left,right)
        den=nu.sum(-1,keepdim=True)-nu
        live=pair&(den>0)
        means=(numerator/torch.where(live,den,torch.ones_like(den))[...,None]).masked_fill(~live[...,None],0)
        return means,(means*gate).masked_fill(~live[...,None],0)


def dense(a,b,gate,alpha,graph):
    with torch.autocast(a.device.type,enabled=False):
        return torch.einsum('bhij,bijd->bhid',alpha.to(a.dtype),dense_features(a,b,gate,graph)[1])
