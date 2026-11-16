"""Forward-local, order-independent geometry and fractional cutoff ties."""
from typing import NamedTuple
import torch
from torch import nn
from weaver.nn.model.ParticleTransformer import to_ptrapphim, delta_phi


class Neighborhood(NamedTuple):
    weights: torch.Tensor
    valid: torch.Tensor
    indices: torch.Tensor
    candidate_weights: torch.Tensor
    tie_rows: torch.Tensor
    tie_indices: torch.Tensor
    tie_weights: torch.Tensor
    counts: torch.Tensor
    healthy: torch.Tensor


@torch.no_grad()
def weights_from_distances(distance2, valid, budget=8):
    if type(budget) is not int or budget<1:raise ValueError('Positive neighbor budget required')
    n=valid.shape[-1]
    if n<1 or distance2.shape!=(*valid.shape,n):raise ValueError('Invalid geometry shape')
    with torch.autocast(distance2.device.type,enabled=False):
        d=distance2.float()
        eligible=valid[:,:,None]&valid[:,None,:]&~torch.eye(n,device=d.device,dtype=torch.bool)
        cutoff=d.masked_fill(~eligible,torch.inf).topk(min(budget,n),largest=False).values[:,:,-1:]
        lower=(d<cutoff)&eligible
        equal=(d==cutoff)&eligible
        fraction=(budget-lower.sum(-1,keepdim=True)).float()/equal.sum(-1,keepdim=True).clamp_min(1)
        weighted=lower.float()+equal*fraction
        nu=torch.where(eligible.sum(-1,keepdim=True)<=budget,eligible.float(),weighted)
        healthy=torch.isfinite(d).logical_or(~eligible).all() & torch.isfinite(nu).all()
        return nu,healthy


@torch.no_grad()
def pack(nu,valid,healthy,budget=8):
    """No tie-row inflation of the ordinary candidate axis. One compaction/forward."""
    batch,n=valid.shape
    counts=(nu>0).sum(-1)
    positions=torch.arange(n,device=nu.device).expand(batch,n,n)
    ordered=positions.masked_fill(nu<=0,n).sort(-1).values
    indices=ordered[:,:,:min(budget,n)].clamp_max(n-1)
    candidate_weights=nu.gather(-1,indices).masked_fill(ordered[:,:,:indices.shape[-1]]==n,0)
    candidate_weights=candidate_weights.masked_fill((counts>budget)[:,:,None],0)
    tie_rows=(counts.flatten()>budget).nonzero().flatten()
    # nonzero compaction and this bounded maximum may synchronize; report them
    # in geometry timings. They happen once, never during checkpoint replay.
    limit=int(counts.flatten()[tie_rows].max()) if tie_rows.numel() else 0
    tie_ordered=ordered.reshape(batch*n,n)[tie_rows,:limit]
    tie_indices=tie_ordered.clamp_max(n-1)
    tie_weights=nu.reshape(batch*n,n)[tie_rows].gather(-1,tie_indices).masked_fill(tie_ordered==n,0)
    return Neighborhood(nu,valid,indices,candidate_weights,tie_rows,tie_indices,tie_weights,counts,healthy)


class NeighborBuilder(nn.Module):
    """No parameters, buffers, cached indices or persistent statistics."""
    def __init__(self,budget=8):
        super().__init__();self.budget=budget

    @torch.no_grad()
    def forward(self,vectors,valid):
        with torch.profiler.record_function('triangles.geometry_and_packing'),torch.autocast(vectors.device.type,enabled=False):
            v=vectors.float().masked_fill(~valid[:,None,:],0)
            coords=to_ptrapphim(v,return_mass=False,eps=None)
            y,phi=coords[:,1],coords[:,2]
            distance=(y[:,:,None]-y[:,None,:]).square()+delta_phi(phi[:,:,None],phi[:,None,:]).square()
            nu,ok=weights_from_distances(distance,valid,self.budget)
            ok=ok & torch.isfinite(v).logical_or(~valid[:,None,:]).all()
            ok=ok & torch.isfinite(coords).logical_or(~valid[:,None,:]).all()
            return pack(nu,valid,ok,self.budget)
