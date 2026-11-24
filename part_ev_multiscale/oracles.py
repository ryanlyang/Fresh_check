"""Slow independent loop oracles. Never used in production geometry/learned paths."""
import torch
from itertools import combinations
from torch.nn import functional as F


def pooling(hidden,g,valid):
    means=[];maxima=[]
    for b in range(len(hidden)):
        mr=[];xr=[]
        for i in range(16):
            selected=g.fine[b,:,i].bool()&valid[b]
            x=hidden[b,selected]
            mr.append(x.mean(0) if len(x) else hidden[b].sum(0)*0)
            xr.append(x.amax(0) if len(x) else hidden[b].sum(0)*0)
        means.append(torch.stack(mr));maxima.append(torch.stack(xr))
    return torch.stack(means),torch.stack(maxima)


def attention(module,query,source,valid=None,allowed=None,query_valid=None):
    b,n,_=query.shape
    if valid is None:valid=torch.ones(source.shape[:2],dtype=torch.bool,device=source.device)
    if query_valid is None:query_valid=torch.ones((b,n),dtype=torch.bool,device=query.device)
    if allowed is None:allowed=query_valid[:,:,None]&valid[:,None,:]
    rows=[]
    for event in range(b):
        q=F.linear(query[event].masked_fill(~query_valid[event,:,None],0),module.query.weight,module.query.bias)
        k=F.linear(source[event].masked_fill(~valid[event,:,None],0),module.key.weight,module.key.bias)
        v=F.linear(source[event].masked_fill(~valid[event,:,None],0),module.value.weight,module.value.bias)
        tokens=[]
        for t in range(n):
            take=allowed[event,t]&valid[event]
            if bool(query_valid[event,t]):
                heads=[]
                for h in range(8):
                    sl=slice(16*h,16*(h+1))
                    heads.append((q[t:t+1,sl]@k[take,sl].T/4).softmax(-1)@v[take,sl])
                tokens.append(F.linear(torch.cat(heads,-1),module.output.weight,module.output.bias)[0])
            else:tokens.append((q[t].sum()+k.sum()+v.sum()+module.output.weight.sum()+module.output.bias.sum())*torch.zeros_like(q[t]))
        rows.append(torch.stack(tokens))
    return torch.stack(rows)


def stage(module,h,g,valid):
    def norm(name,x):
        m=getattr(module,name);return F.layer_norm(x,(128,),m.weight,m.bias,m.eps)
    def linear(m,x):return F.linear(x,m.weight,m.bias)
    def masked(x,m):return x.masked_fill(~m[...,None],0)
    def ffn(m,x):return linear(m[2],F.gelu(linear(m[0],x)))
    fv,cv=g.fine_valid,g.coarse_valid
    u=norm('particle_norm',masked(h,valid));mean,maximum=pooling(u,g,valid)
    fine=masked(norm('fine_seed_norm',F.gelu(linear(module.fine_seed_linear,torch.cat(
        (mean,maximum,g.fine_descriptors.to(h.dtype)),-1)))),fv)
    fine=masked(fine+attention(module.particle_read,norm('fine_read_query_norm',fine),u,valid,g.fine.transpose(1,2).bool(),fv),fv)
    x=masked(norm('fine_local_attn_norm',fine),fv)
    fine=masked(fine+attention(module.fine_local,x,x,fv,(g.parent@g.parent.transpose(1,2)).bool(),fv),fv)
    fine=masked(fine+ffn(module.fine_ffn,norm('fine_ffn_norm',fine)),fv)
    rows=[]
    for b in range(len(h)):
        rows.append(torch.stack([sum((fine[b,j]*g.fine_count[b,j] for j in range(16)
            if g.parent[b,j,c]),fine[b].sum(0)*0)/g.coarse_count[b,c].clamp_min(1) for c in range(4)]))
    coarse=masked(norm('coarse_seed_norm',F.gelu(linear(module.coarse_seed_linear,torch.cat(
        (torch.stack(rows),g.coarse_descriptors.to(h.dtype)),-1)))),cv)
    coarse=masked(coarse+attention(module.child_read,norm('coarse_read_query_norm',coarse),
        masked(norm('coarse_read_kv_norm',fine),fv),fv,g.parent.transpose(1,2).bool(),cv),cv)
    x=masked(norm('coarse_global_attn_norm',coarse),cv)
    coarse=masked(coarse+attention(module.coarse_global,x,x,cv,query_valid=cv),cv)
    coarse=masked(coarse+ffn(module.coarse_ffn,norm('coarse_ffn_norm',coarse)),cv)
    fine=masked(fine+attention(module.topdown,norm('topdown_query_norm',fine),
        masked(norm('topdown_kv_norm',coarse),cv),cv,query_valid=fv),fv)
    fs=masked(norm('feedback_fine_norm',fine),fv);cs=masked(norm('feedback_coarse_norm',coarse),cv)
    own_fine=torch.stack([fs[b,g.fine_ids[b].clamp_min(0)] for b in range(len(h))])
    own_coarse=torch.stack([cs[b,g.coarse_ids[b].clamp_min(0)] for b in range(len(h))])
    delta=linear(module.feedback_output,F.gelu(linear(module.feedback_input,torch.cat((u,own_fine,own_coarse),-1))))
    return h+masked(delta,valid),fine,coarse


def tree(vectors,valid):
    """CPU dictionary/list FP32 oracle with independent pair enumeration."""
    result=[]
    for p,mask in zip(vectors.detach().cpu().float(),valid.cpu()):
        tuples=[tuple(0. if x==0 else float(x) for x in row) for row in p]
        atoms=sorted(set(tuples[i] for i in range(len(p)) if mask[i]))
        clusters={}
        for k,key in enumerate(atoms):
            members={i for i in range(len(p)) if mask[i] and tuples[i]==key}
            clusters[k]=(torch.tensor(key,dtype=torch.float32)*len(members),members)
        def snapshot():return [(k,v.clone(),set(m)) for k,(v,m) in sorted(clusters.items())]
        def coord(q):
            px,py,pz,e=q
            pt2=(px*px+py*py).clamp_min(0)
            y=((e+pz).clamp_min(1e-6).log()-(e-pz).clamp_min(1e-6).log())*.5
            phi=torch.atan2(py,px) if pt2>1e-12 else torch.tensor(0.)
            return y,phi
        fine=snapshot() if len(clusters)<=16 else None
        while len(clusters)>4:
            # Independent dictionary history and Python pair enumeration; batch
            # the scalar FP32 formula to keep the diagnostic bounded at N=128.
            keys=sorted(clusters);coords=[coord(clusters[k][0]) for k in keys]
            ys=torch.stack([c[0] for c in coords]);phis=torch.stack([c[1] for c in coords])
            pairs=list(combinations(range(len(keys)),2));left,right=torch.tensor(pairs).T
            angle=phis[left]-phis[right]
            distances=(ys[left]-ys[right]).square()+torch.atan2(angle.sin(),angle.cos()).square()
            _,i,j=min((d,keys[l],keys[r]) for d,(l,r) in zip(distances.tolist(),pairs))
            a,am=clusters[i];b,bm=clusters.pop(j)
            clusters[i]=(a+b,am|bm)
            if len(clusters)==16:fine=snapshot()
        result.append(dict(fine=fine,coarse=snapshot()))
    return result


def physical_descriptors(vectors,valid,history):
    """Independent CPU event/region formulas, retaining prescribed FP32 mass."""
    def coordinates(q):
        px,py,pz,e=q.unbind(-1);pt2=(px*px+py*py).clamp_min(0)
        return pt2.sqrt(),.5*((e+pz).clamp_min(1e-6).log()-(e-pz).clamp_min(1e-6).log()),torch.where(
            pt2>1e-12,torch.atan2(py,px),torch.zeros_like(px))
    def binary_sum(x):
        while len(x)>1:
            if len(x)%2:x=torch.cat((x,torch.zeros_like(x[:1])))
            x=x[::2]+x[1::2]
        return x[0]
    result=dict(fine=[],coarse=[])
    for p,mask,row in zip(vectors.detach().cpu().float(),valid.cpu(),history):
        source=p[mask];tuples=[tuple(0. if x==0 else float(x) for x in a) for a in source]
        atoms=sorted(set(tuples));summands=[]
        for t in atoms:summands.append(torch.tensor(t,dtype=torch.float32)*tuples.count(t))
        summands+= [torch.zeros(4) for _ in range(len(p)-len(summands))]
        jet=binary_sum(torch.stack(summands));_,jy,jphi=coordinates(jet)
        ordered=torch.tensor(sorted(tuples),dtype=torch.float32)
        pts=coordinates(ordered)[0];pts=torch.cat((pts,torch.zeros(len(p)-len(pts))))
        scalar_pt=binary_sum(pts[:,None])[0]
        for scale,capacity in (('fine',16),('coarse',4)):
            desc=torch.zeros(capacity,7)
            for j,(_,v,members) in enumerate(row[scale]):
                original=p[sorted(members)];pt,y,phi=coordinates(original);gp,gy,gphi=coordinates(v)
                angle=torch.atan2((gphi-jphi).sin(),(gphi-jphi).cos())
                da=torch.atan2((phi-gphi).sin(),(phi-gphi).cos())
                width=((pt*((y-gy).square()+da.square())).sum()/pt.sum().clamp_min(1e-6)).clamp_min(0).sqrt()
                mass=(v[3].square()-v[:3].square().sum()).clamp_min(0).sqrt()
                desc[j]=torch.stack(((gp/scalar_pt.clamp_min(1e-6)).clamp_min(1e-6).log(),mass.log1p(),
                    gy-jy,angle.sin(),angle.cos(),torch.tensor(float(len(members))).log1p(),width.log1p()))
            result[scale].append(desc)
    return {k:torch.stack(v) for k,v in result.items()}
