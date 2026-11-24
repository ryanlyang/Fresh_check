"""Detached, fixed-capacity batched E-scheme hierarchy. No per-merge host reads.

Identical physical four-vectors form indivisible atoms. Exact distance ties use
canonical atom keys, never input indices. All production geometry is FP32.
"""
from dataclasses import dataclass
from contextlib import contextmanager
from contextvars import ContextVar
import torch
from torch.nn import functional as F

EPS = 1e-6
_OBSERVER=ContextVar('multiscale_geometry_observer',default=None)


@contextmanager
def observe():
    """Opt-in bounded diagnostic collector, never persistent model/cache state."""
    rows=[];token=_OBSERVER.set(rows)
    try:yield rows
    finally:_OBSERVER.reset(token)


@dataclass(frozen=True)
class Geometry:
    fine: torch.Tensor
    coarse: torch.Tensor
    parent: torch.Tensor
    fine_valid: torch.Tensor
    coarse_valid: torch.Tensor
    fine_ids: torch.Tensor
    coarse_ids: torch.Tensor
    fine_count: torch.Tensor
    coarse_count: torch.Tensor
    fine_vectors: torch.Tensor
    coarse_vectors: torch.Tensor
    fine_descriptors: torch.Tensor
    coarse_descriptors: torch.Tensor
    atom_ids: torch.Tensor
    atom_count: torch.Tensor
    min_margin: torch.Tensor
    exact_ties: torch.Tensor
    healthy: torch.Tensor


def coordinates(p):
    px,py,pz,e=p.unbind(-1)
    pt2=(px.square()+py.square()).clamp_min(0)
    pt=pt2.sqrt()
    y=.5*((e+pz).clamp_min(EPS).log()-(e-pz).clamp_min(EPS).log())
    phi=torch.where(pt2>EPS**2,torch.atan2(py,px),torch.zeros_like(pt))
    return pt,y,phi


def distance(a,b):
    _,ya,pa=coordinates(a);_,yb,pb=coordinates(b)
    dphi=pa-pb
    return (ya-yb).square()+torch.atan2(dphi.sin(),dphi.cos()).square()


def pairwise_sum(x):
    """Fixed padded binary reduction, not a floating-point atomic scatter."""
    while x.shape[1]>1:
        if x.shape[1]%2:x=F.pad(x,(0,0,0,1))
        x=x[:,0::2]+x[:,1::2]
    return x[:,0]


def descriptors(p,valid,members,vectors,occupancy,jet,scalar_pt):
    pt,y,phi=coordinates(p)
    gp,gy,gphi=coordinates(vectors);_,jy,jphi=coordinates(jet)
    angle=gphi-jphi[:,None]
    wrapped=torch.atan2(angle.sin(),angle.cos())
    angle=phi[:,:,None]-gphi[:,None,:]
    radius=(y[:,:,None]-gy[:,None,:]).square()+torch.atan2(angle.sin(),angle.cos()).square()
    weighted=members*pt[:,:,None]
    width=((weighted*radius).sum(1)/weighted.sum(1).clamp_min(EPS)).clamp_min(0).sqrt()
    mass2=vectors[...,3].square()-vectors[...,:3].square().sum(-1)
    result=torch.stack(((gp/scalar_pt[:,None].clamp_min(EPS)).clamp_min(EPS).log(),
        mass2.clamp_min(0).sqrt().log1p(),gy-jy[:,None],wrapped.sin(),wrapped.cos(),
        occupancy.float().log1p(),width.log1p()),-1)
    return result.masked_fill(~valid[...,None],0)


def compact(labels,vectors,active,valid,capacity):
    rank=active.long().cumsum(-1)-1
    ids=rank.gather(1,labels.clamp_min(0)).masked_fill(~valid,-1)
    membership=F.one_hot(ids.clamp_min(0),capacity).float()*valid[...,None]
    counts=membership.sum(1).long();mask=counts>0
    # Gather the surviving canonical slot in each compact region; no float scatter.
    n=active.shape[1];slots=torch.arange(n,device=labels.device)[None].expand_as(active)
    index=torch.full((labels.shape[0],capacity),n,device=labels.device,dtype=torch.long)
    index=index.scatter_reduce(1,rank.clamp_min(0),slots.masked_fill(~active,n),reduce='amin',include_self=True)
    packed=vectors.gather(1,index.clamp_max(n-1)[...,None].expand(-1,-1,4)).masked_fill(~mask[...,None],0)
    return ids,membership,counts,mask,packed


@torch.no_grad()
def build_geometry(p,valid):
    if p.ndim!=3 or p.shape[-1]!=4 or valid.shape!=p.shape[:2] or valid.dtype!=torch.bool:
        raise ValueError('Expected [B,N,4] physical vectors and bool [B,N] mask')
    if p.shape[1]<1:raise ValueError('Zero-capacity geometry')
    with torch.profiler.record_function('multiscale.geometry'),torch.autocast(p.device.type,enabled=False):
        p=p.float();b,n,_=p.shape;device=p.device
        healthy=(torch.isfinite(p).all(-1)|~valid).all() & ((p[...,3]>=0)|~valid).all() & valid.any(-1).all()
        p=torch.where(valid[...,None]&torch.isfinite(p),p,torch.zeros_like(p))
        p=torch.where(p==0,torch.zeros_like(p),p)  # canonical positive zero
        order=torch.arange(n,device=device)[None].expand(b,-1)
        with torch.profiler.record_function('multiscale.canonical_atoms'):
            for dim in (3,2,1,0):
                key=p[...,dim].gather(1,order)
                order=order.gather(1,key.argsort(dim=-1,stable=True))
            order=order.gather(1,(~valid.gather(1,order)).long().argsort(dim=-1,stable=True))
            sorted_p=p.gather(1,order[...,None].expand(-1,-1,4));sv=valid.gather(1,order)
            first=torch.ones((b,n),device=device,dtype=torch.bool)
            first[:,1:]=(sorted_p[:,1:]!=sorted_p[:,:-1]).any(-1)
            first &= sv
            ids=first.long().cumsum(-1)-1;atom_count=first.sum(-1)
            atom_ids=torch.full_like(ids,-1).scatter(1,order,ids.masked_fill(~sv,-1))
            multiplicity=torch.zeros_like(ids).scatter_add(1,ids.clamp_min(0),sv.long())
            positions=torch.arange(n,device=device)[None].expand(b,-1)
            representative=torch.full_like(ids,n).scatter_reduce(1,ids.clamp_min(0),
                positions.masked_fill(~sv,n),reduce='amin',include_self=True)
            vectors=sorted_p.gather(1,representative.clamp_max(n-1)[...,None].expand(-1,-1,4))*multiplicity[...,None]
            active=multiplicity>0;vectors=vectors.masked_fill(~active[...,None],0)
            healthy &= torch.isfinite(vectors).all()
            jet=pairwise_sum(vectors);scalar_pt=pairwise_sum((coordinates(sorted_p)[0]*sv)[...,None])[:,0]
        labels=atom_ids.clone();fine_labels=labels.clone();fine_vectors=vectors.clone();fine_active=active.clone()
        counts=atom_count.clone();slots=torch.arange(n,device=device)[None,:]
        upper=torch.triu(torch.ones((n,n),device=device,dtype=torch.bool),diagonal=1)
        dist=distance(vectors[:,:,None,:],vectors[:,None,:,:])
        min_margin=torch.full((b,),torch.inf,device=device);ties=torch.zeros(b,device=device,dtype=torch.long)
        rows=torch.arange(b,device=device)
        with torch.profiler.record_function('multiscale.merge_history'):
            for _ in range(max(0,n-4)):
                merge=counts>4
                eligible=upper[None]&active[:,:,None]&active[:,None,:]
                scores=dist.masked_fill(~eligible|~torch.isfinite(dist),torch.inf).flatten(1)
                chosen=scores.argmin(-1);i=chosen//n;j=chosen%n
                two=scores.topk(min(2,n*n),largest=False).values
                margin=two[:,1]-two[:,0]
                finite_merge=merge&torch.isfinite(two[:,0]);healthy &= (~merge|finite_merge).all()
                # Overflow is invalid evidence, not a rank-local Python failure
                # before a peer's native SyncBN. Complete a safe deterministic
                # map while propagating false health; never silently admit it.
                first_slot=slots.masked_fill(~active,n).amin(-1)
                second_slot=slots.masked_fill(~active|(slots<=first_slot[:,None]),n).amin(-1)
                fallback=first_slot.clamp_max(n-1)*n+second_slot.clamp_max(n-1)
                chosen=torch.where(finite_merge,chosen,fallback);i=chosen//n;j=chosen%n
                min_margin=torch.minimum(min_margin,torch.where(finite_merge,margin,torch.inf))
                ties+=((margin==0)&finite_merge).long()
                merged=vectors[rows,i]+vectors[rows,j]
                take_i=(slots==i[:,None])&merge[:,None]
                take_j=(slots==j[:,None])&merge[:,None]
                vectors=torch.where(take_i[...,None],merged[:,None,:],vectors).masked_fill(take_j[...,None],0)
                active &= ~take_j
                labels=torch.where((labels==j[:,None])&merge[:,None],i[:,None],labels)
                counts-=merge.long()
                row=distance(merged[:,None,:],vectors)
                dist=torch.where(take_i[:,:,None],row[:,None,:],dist)
                dist=torch.where(take_i[:,None,:],row[:,:,None],dist)
                snapshot=merge&(counts==16)
                fine_labels=torch.where(snapshot[:,None],labels,fine_labels)
                fine_vectors=torch.where(snapshot[:,None,None],vectors,fine_vectors)
                fine_active=torch.where(snapshot[:,None],active,fine_active)
        with torch.profiler.record_function('multiscale.maps'):
            fi,af,nf,fv,fp=compact(fine_labels,fine_vectors,fine_active,valid,16)
            ci,ac,nc,cv,cp=compact(labels,vectors,active,valid,4)
            parent_id=torch.full((b,16),4,device=device,dtype=torch.long).scatter_reduce(1,
                fi.clamp_min(0),ci.masked_fill(~valid,4),reduce='amin',include_self=True)
            parent=F.one_hot(parent_id.clamp_max(3),4).float()*fv[...,None]
            healthy &= ((af@parent)==ac).all() & (af.sum(-1)==valid).all() & (ac.sum(-1)==valid).all()
        with torch.profiler.record_function('multiscale.descriptors'):
            fd=descriptors(p,fv,af,fp,nf,jet,scalar_pt);cd=descriptors(p,cv,ac,cp,nc,jet,scalar_pt)
            healthy &= torch.isfinite(fd).all()&torch.isfinite(cd).all()
        result=Geometry(af,ac,parent,fv,cv,fi,ci,nf,nc,fp,cp,fd,cd,atom_ids,atom_count,
            torch.where(torch.isfinite(min_margin),min_margin,torch.zeros_like(min_margin)),ties,healthy)
        records=_OBSERVER.get()
        if records is not None:
            records.append(dict(mapping_sha256=mapping_digest(result),min_margin=float(result.min_margin.min()),
                exact_ties=int(ties.sum()),atom_count=atom_count.cpu().tolist(),
                fine_occupancy=nf.cpu().tolist(),coarse_occupancy=nc.cpu().tolist()))
        return result


def mapping_digest(g):
    """Bounded diagnostics only; never called by the production merge loop."""
    import hashlib
    h=hashlib.sha256()
    for name in ('atom_ids','fine_ids','coarse_ids','fine_count','coarse_count','parent'):
        x=getattr(g,name).detach().cpu().contiguous();h.update(x.numpy().tobytes())
    return h.hexdigest()


def aggregate(rows):
    """Constant-size monitoring summary, not per-training-event memberships."""
    if len(rows)!=1:raise ValueError('Expected exactly one hierarchy construction')
    row=rows[0]
    def stats(values):
        flat=[v for x in values for v in x if v>0]
        return dict(min=min(flat),max=max(flat),mean=sum(flat)/len(flat),active=len(flat))
    return dict(schema='bounded_geometry_v1',min_margin=row['min_margin'],exact_ties=row['exact_ties'],
        mean_atoms=sum(row['atom_count'])/len(row['atom_count']),fine=stats(row['fine_occupancy']),
        coarse=stats(row['coarse_occupancy']))
