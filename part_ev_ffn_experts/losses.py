"""Exact global top-two balancing value and DDP-averaged gradient proxy."""
import torch
import torch.distributed as dist
from part_ev_gpu_scaling.benchmark import agree


def local_health(banks,device):
    healthy=torch.ones((),device=device,dtype=torch.bool)
    try:
        if len(banks)!=6:return healthy&False
        for b in banks:
            if (b['probability_sum'].shape!=(4,) or b['requested'].shape!=(4,)
                    or b['executed'].shape!=(4,) or b['primary'].shape!=(4,) or b['secondary'].shape!=(4,)
                    or b['tokens'].shape!=() or b['health'].shape!=() or b['health'].dtype!=torch.bool
                    or b['probability_sum'].dtype not in (torch.float32,torch.float64)
                    or any(b[k].dtype!=torch.int64 for k in ('requested','executed','tokens','primary','secondary'))):
                return healthy&False
            p,c,t=b['probability_sum'],b['requested'],b['tokens']
            healthy=(healthy & b['health'] & torch.isfinite(p).all() & (p>=0).all() & (t>0)
                & (c>=0).all() & (c.sum()==2*t) & (c==b['executed']).all()
                & (b['primary']+b['secondary']==c).all()
                & torch.isclose(p.sum(),t.to(p.dtype),rtol=1e-5,atol=1e-4))
    except (KeyError,TypeError,AttributeError,RuntimeError):return healthy&False
    return healthy


def from_global(local_sums,global_sums,global_counts,global_tokens,world):
    """Pure formula, also used by independent split-population gradient tests."""
    denominator=global_tokens.to(local_sums.dtype)
    frequency=global_counts.to(local_sums.dtype)/(2*denominator[:,None])
    true_value=4*(global_sums/denominator[:,None]*frequency).sum(-1)
    proxy=world*4/denominator*(frequency*local_sums).sum(-1)
    backward=proxy+(true_value-proxy).detach()
    return backward.mean(),true_value.detach()


def balancing(banks,world,device):
    if not agree(bool(local_health(banks,device)),device,world):
        raise FloatingPointError('Invalid routing statistics on at least one rank')
    sums=torch.stack([b['probability_sum'] for b in banks])
    global_sums=sums.detach().clone()
    integers=torch.stack([torch.cat((b['requested'],b['executed'],b['primary'],b['secondary'],b['tokens'][None]))
                          for b in banks]).detach().clone()
    # Two fixed packed collectives; counts remain exact int64, not FP16/float32.
    if world>1:
        dist.all_reduce(global_sums);dist.all_reduce(integers)
    counts,executed,primary,secondary,tokens=(integers[:,:4],integers[:,4:8],integers[:,8:12],
                                           integers[:,12:16],integers[:,16])
    healthy=(torch.isfinite(global_sums).all() & (counts==executed).all()
        & (counts.sum(-1)==2*tokens).all() & (tokens>0).all())
    if not agree(bool(healthy),device,world):raise FloatingPointError('Invalid global routing reduction')
    loss,values=from_global(sums,global_sums,counts,tokens,world)
    return loss,dict(balance=values.mean(),per_bank=values,requested=counts,executed=executed,
        primary=primary,secondary=secondary,tokens=tokens,probability_sum=global_sums)
