"""Independent augmentation RNG and leakage-safe visible-defined auxiliary inputs."""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
from pathlib import Path
import torch
import yaml
from part_ev_fullscale.data import DATA_CONFIG

EPS=1e-8
INDEPENDENT=tuple(range(5,15))
NAMES=['part_pt_log','part_e_log','part_logptrel','part_logerel','part_deltaR',
       'part_charge','part_isChargedHadron','part_isNeutralHadron','part_isPhoton','part_isElectron',
       'part_isMuon','part_d0','part_d0err','part_dz','part_dzerr','part_deta','part_dphi']


@lru_cache(maxsize=1)
def transforms():
    row=yaml.safe_load(Path(DATA_CONFIG).read_text());values=row['inputs']['pf_features']['vars']
    if ([x[0] for x in values]!=NAMES or row['preprocess']['method']!='manual'
            or any(row['inputs'][k]['length']!=128 for k in ('pf_features','pf_vectors','pf_points','pf_mask'))):
        raise ValueError('Pinned native feature contract changed')
    expected={0:[1.7,.7],1:[2.,.7],2:[-4.7,.7],3:[-4.7,.7],4:[.2,4.],15:[None],16:[None]}
    if any(values[i][1:]!=value for i,value in expected.items()):raise ValueError('Native transform changed')
    for i in INDEPENDENT:
        expected_spec=[0,1,0,1] if i in (12,14) else [None]
        if values[i][1:]!=expected_spec:raise ValueError('Independent channel preprocessing changed')
    expected_raw={'part_d0':'np.tanh(part_d0val)','part_dz':'np.tanh(part_dzval)'}
    if any(row['new_variables'].get(k)!=v for k,v in expected_raw.items()):raise ValueError('Track feature formulas changed')
    if any(NAMES[i] in row['new_variables'] for i in INDEPENDENT if NAMES[i] not in expected_raw):
        raise ValueError('Independent feature gained a derived expression')
    if any('jet_' in str(row['new_variables'].get(NAMES[i],'')) for i in INDEPENDENT):
        raise ValueError('Independent feature became jet-relative')
    return tuple(tuple(x) for x in values)


def apply_transform(x,index):
    spec=transforms()[index];center=spec[1] if len(spec)>1 else None
    if center is None:return x
    scale=spec[2] if len(spec)>2 else 1.;low=spec[3] if len(spec)>3 else -5.;high=spec[4] if len(spec)>4 else 5.
    return ((x-center)*scale).clamp(low,high)


class Generators:
    names=('selection','particles','centers','ties')
    def __init__(self,rank,seed=101):
        self.rank=rank;self.generators={}
        for name in self.names:
            value=int.from_bytes(hashlib.sha256(f'masked-views-v1:{seed}:{rank}:{name}'.encode()).digest()[:8],'little')
            self.generators[name]=torch.Generator(device='cpu').manual_seed(value&((1<<63)-1))
        self.draws=0

    def draw(self,batch):
        if batch%4:raise ValueError('Clean local batch must divide evenly into two half-batch mask modes')
        k=batch//2;g=self.generators
        selection=torch.randperm(batch,generator=g['selection'])[:k]
        priorities=[torch.rand(k,128,generator=g[name]) for name in self.names[1:]]
        self.draws+=1
        return selection,*priorities

    def state_dict(self):return dict(schema='masked_cpu_rng_v1',rank=self.rank,draws=self.draws,
        generators={k:g.get_state().clone() for k,g in self.generators.items()})

    def load_state_dict(self,row):
        if row['schema']!='masked_cpu_rng_v1' or row['rank']!=self.rank or set(row['generators'])!=set(self.names):
            raise ValueError('Wrong auxiliary generator identity')
        if type(row['draws']) is not int or row['draws']<0:raise ValueError('Invalid generator cursor')
        for k,g in self.generators.items():g.set_state(row['generators'][k].cpu())
        self.draws=row['draws']


def digest(values):
    h=hashlib.sha256()
    for x in values:h.update(x.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def directions(p):
    px,py,pz,_=p.unbind(1);pt=torch.hypot(px,py)
    eta=torch.asinh(pz/pt.clamp_min(EPS));phi=torch.where(pt>EPS,torch.atan2(py,px),torch.zeros_like(pt))
    return pt,eta,phi


def canonical(features,p,valid):
    keys=torch.cat((p,features[:,5:15]),1).float().transpose(1,2)
    keys=torch.where(keys==0,torch.zeros_like(keys),keys)
    order=torch.arange(p.shape[-1],device=p.device)[None].expand(p.shape[0],-1)
    for i in range(13,-1,-1):order=order.gather(1,keys[...,i].gather(1,order).argsort(dim=-1,stable=True))
    return order.gather(1,(~valid.gather(1,order)).long().argsort(dim=-1,stable=True))


@torch.no_grad()
def rebase(features,p,valid,visible):
    with torch.autocast(p.device.type,enabled=False):
        p=p.float();safe=p.masked_fill(~valid[:,None],0)
        healthy=(torch.isfinite(safe).all()&torch.isfinite(features.masked_fill(~valid[:,None],0)).all()
                 &((p[:,3]>=0)|~valid).all())
        reference=safe.masked_fill(~visible[:,None],0).sum(-1)
        jetpt=torch.hypot(reference[:,0],reference[:,1]);energy=reference[:,3]
        eligible=(visible.sum(-1)>=2)&torch.isfinite(reference).all(-1)&(jetpt>EPS)&(energy>EPS)
        eta_vis=torch.asinh(reference[:,2]/jetpt.clamp_min(EPS));phi_vis=torch.atan2(reference[:,1],reference[:,0])
        pt,eta,phi=directions(safe);sign=torch.where(eta_vis>=0,1.,-1.)
        deta=sign[:,None]*(eta-eta_vis[:,None]);angle=phi-phi_vis[:,None]
        dphi=torch.atan2(angle.sin(),angle.cos())
        raw={0:pt.clamp_min(EPS).log(),1:safe[:,3].clamp_min(EPS).log(),
             2:(pt.clamp_min(EPS)/jetpt[:,None].clamp_min(EPS)).log(),
             3:(safe[:,3].clamp_min(EPS)/energy[:,None].clamp_min(EPS)).log(),
             4:torch.hypot(deta,dphi),15:deta,16:dphi}
        rebuilt=features.detach().float().clone()
        for i,x in raw.items():rebuilt[:,i]=apply_transform(x,i)
        rebuilt=rebuilt.masked_fill(~valid[:,None],0)
        points=torch.stack((deta,dphi),1).masked_fill(~valid[:,None],0)
        healthy &= torch.isfinite(rebuilt).all()&torch.isfinite(points).all()
        return rebuilt,points,eligible,healthy


@dataclass(frozen=True)
class Views:
    selected:torch.Tensor
    modes:torch.Tensor
    features:torch.Tensor
    points:torch.Tensor
    vectors:torch.Tensor
    valid:torch.Tensor
    visible:torch.Tensor
    target_ids:torch.Tensor
    query_valid:torch.Tensor
    locations:torch.Tensor
    eligible:torch.Tensor
    healthy:torch.Tensor

    def fingerprint(self):return digest([self.selected,self.modes,self.visible,self.target_ids,self.query_valid,self.eligible])


@torch.no_grad()
def create(retained,generators):
    features,p,mask=retained;b,_,n=p.shape;device=p.device
    draws=generators.draw(b);selected=draws[0].to(device)
    f=features[selected];p=p[selected];valid=mask[selected,0].bool();k=len(selected)
    with torch.profiler.record_function('masked.views'),torch.autocast(device.type,enabled=False):
        order=canonical(f,p,valid)
        priorities=[]
        for raw in draws[1:]:priorities.append(torch.empty(k,n,device=device).scatter(1,order,raw[:,:n].to(device)))
        uniform,centers,ties=priorities;modes=torch.arange(k,device=device)>=k//2
        count=valid.sum(-1);m=torch.minimum(count-2,(count//4).clamp_min(1)).masked_fill(count<4,0)
        _,eta,phi=directions(p.float().masked_fill(~valid[:,None],0))
        center=centers.masked_fill(~valid,torch.inf).argmin(-1)
        dy=eta-eta.gather(1,center[:,None]);angle=phi-phi.gather(1,center[:,None])
        distances=dy.square()+torch.atan2(angle.sin(),angle.cos()).square()
        distances.scatter_(1,center[:,None],-1.)
        patch=order.gather(1,ties.gather(1,order).argsort(dim=-1,stable=True))
        patch=patch.gather(1,distances.masked_fill(~valid,torch.inf).gather(1,patch).argsort(dim=-1,stable=True))
        unif=order.gather(1,uniform.masked_fill(~valid,torch.inf).gather(1,order).argsort(dim=-1,stable=True))
        ranked=torch.where(modes[:,None],patch,unif)
        target=torch.zeros_like(valid).scatter(1,ranked,torch.arange(n,device=device)[None]<m[:,None])
        visible=valid&~target
        rebuilt,points,reference_ok,healthy=rebase(f,p,valid,visible)
        eligible=(count>=4)&reference_ok
        # Original slot IDs are alignment only, never predictor features.
        target_order=order.gather(1,(~target.gather(1,order)).long().argsort(dim=-1,stable=True))
        capacity=min(32,n);ids=target_order[:,:capacity];qv=torch.arange(capacity,device=device)[None]<m[:,None]
        deta=points[:,0].gather(1,ids);dphi=points[:,1].gather(1,ids)
        locations=torch.stack((deta,dphi.sin(),dphi.cos()),-1).masked_fill(~qv[...,None],0)
        return Views(selected,modes,rebuilt,points,p,valid,visible,ids,qv,locations,eligible,healthy)


def pack(points,features,p,valid):
    """One host size decision per microbatch; never a per-jet/particle sync loop."""
    count=valid.sum(-1)
    if not bool((count>0).all()):raise ValueError('Cannot encode empty auxiliary jets')
    n=valid.shape[-1];capacity=int(count.max())
    slots=torch.arange(n,device=p.device)[None].expand_as(valid)
    ids=slots.masked_fill(~valid,n).sort(-1).values
    positions=torch.arange(capacity,device=p.device)[None].expand(len(p),-1)
    gather=ids.gather(1,positions%count[:,None]);mask=positions<count[:,None]
    values=[x.gather(2,gather[:,None].expand(-1,x.shape[1],-1)) for x in (points,features,p)]
    inverse=torch.full_like(slots,n).scatter_reduce(1,gather,positions.masked_fill(~mask,n),reduce='amin',include_self=True)
    return (*values,mask[:,None]),inverse


def slot(views,start,stop):
    chosen=torch.arange(start,stop,device=views.features.device);chosen=chosen[views.eligible[start:stop]]
    if not len(chosen):return None
    points,f,p=views.points[chosen],views.features[chosen],views.vectors[chosen]
    full,inverse=pack(points,f,p,views.valid[chosen]);visible,_=pack(points,f,p,views.visible[chosen])
    target=inverse.gather(1,views.target_ids[chosen]).masked_fill(~views.query_valid[chosen],0)
    return dict(teacher=full,visible=visible,target_ids=target,locations=views.locations[chosen],
                query_valid=views.query_valid[chosen],modes=views.modes[chosen])
