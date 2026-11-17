"""Isolated small eval probes; normal forwards keep no support diagnostics."""
import torch
import math
import re
from torch.nn import functional as F
from . import model as M,aggregation as A
from part_ev_fullscale.checkpoint import capture_rank_state,restore_rank_state
from part_ev_screen.precision import model_precision


def validate_diagnostic(row):
    """Admission checks numeric integrity, not a preferred scientific result."""
    def scalar(value,lo=0.,hi=float('inf'),nullable=False):
        if nullable and value is None:return
        if not isinstance(value,(float,int)) or not math.isfinite(value) or not lo<=value<=hi:
            raise ValueError('Invalid support diagnostic statistic')
    def vector(values,lo=0.,hi=float('inf'),nullable=False):
        if len(values)!=4:raise ValueError('Need four support channels')
        for v in values:scalar(v,lo,hi,nullable)
    if (row.get('healthy') is not True or row.get('population')!='first_8_validation_probe'
            or not re.fullmatch('[0-9a-f]{64}',row.get('sample_sha256',''))
            or row.get('events')!=8 or not 1<=row['particles_input']<=128
            or [r['layer'] for r in row['layers']]!=list(range(8))):
        raise ValueError('Missing isolated validation diagnostics')
    for k in ('disabled_logit_rms_change','counts_zero_logit_rms_change'):scalar(row[k])
    for layer in row['layers']:
        for k in ('gate_mean','gate_std','gate_within_receiver_std','saturation_low',
                  'saturation_high','fraction_mean','floor_fraction'):vector(layer[k],hi=1)
        vector(layer['count_mean'],hi=127);vector(layer['log_count_mean'],hi=math.log(128)+1e-6)
        vector(layer['count_degree_correlation'],-1,1,True)
        for k in ('gate_correlation','content_similarity'):
            if len(layer[k])!=4:raise ValueError('Missing channel matrix')
            for v in layer[k]:vector(v,-1,1,True)
        for k in ('support_rms','state_rms','support_to_state_norm','support_to_native_norm'):scalar(layer[k])
        scalar(layer['mean_neighbors'],hi=127)
        scalar(layer['valid_receivers'],8,1024)
        scalar(layer['no_neighbor_receivers'],hi=layer['valid_receivers'])
        degrees=[];receivers=0
        for bucket in layer['by_degree']:
            scalar(bucket['degree'],hi=127);scalar(bucket['receivers'],1,1024)
            vector(bucket['count_mean'],hi=bucket['degree']);vector(bucket['fraction_std'],hi=1)
            degrees.append(bucket['degree']);receivers+=bucket['receivers']
        if degrees!=sorted(set(degrees)) or receivers!=layer['valid_receivers']:
            raise ValueError('Invalid multiplicity-conditioned diagnostics')


def correlation(x,y):
    x=x.double().flatten();y=y.double().flatten()
    if x.numel()<2:return None
    x=x-x.mean();y=y-y.mean();den=x.norm()*y.norm()
    return None if den==0 else float((x@y/den).clamp(-1,1))


def layer_statistics(records,valid):
    """Receives detached/no-grad chunk results OUTSIDE checkpoint functions."""
    logits,values,features,degree=(torch.cat([r[k] for r in records],1) for k in range(4))
    old,heads=(torch.cat([r[k] for r in records],2) for k in (4,5))
    mask=torch.cat([r[6] for r in records],1)
    gate=logits.float().sigmoid().masked_fill(~mask[...,None],0)
    count=gate.sum(-2);live=degree>0;eligible=mask.sum().clamp_min(1)
    summary=features.float().reshape(*degree.shape,4,18)
    fraction=summary[...,17];mean=summary[...,:16]
    pair_mean=gate.sum((0,1,2))/eligible
    pair_std=(((gate-pair_mean).square()*mask[...,None]).sum((0,1,2))/eligible).sqrt()
    within=((gate-fraction[:,:,None]).square()*mask[...,None]).sum(-2)/degree.clamp_min(1)[...,None]
    num_receivers=live.sum().clamp_min(1)
    vector=lambda x:(x*live[...,None]).sum((0,1))/num_receivers
    pair_values=gate[mask]
    gate_corr=[[correlation(pair_values[:,a],pair_values[:,b]) for b in range(4)] for a in range(4)]
    similarities=[]
    for a in range(4):
        similarities.append([float(F.cosine_similarity(mean[:,:,a],mean[:,:,b],dim=-1)[live].clamp(-1,1).mean())
                             if live.any() else None for b in range(4)])
    bins=[]
    for d in sorted(set(degree[valid].flatten().tolist())):
        rows=valid&(degree==d)
        bins.append(dict(degree=int(d),receivers=int(rows.sum()),count_mean=count[rows].mean(0).tolist(),
            fraction_std=fraction[rows].std(0,unbiased=False).tolist()))
    scale=(valid.sum()*128).clamp_min(1)
    return dict(gate_mean=pair_mean.tolist(),gate_std=pair_std.tolist(),
        gate_within_receiver_std=vector(within.sqrt()).tolist(),
        saturation_low=(((gate<.01)&mask[...,None]).sum((0,1,2))/eligible).tolist(),
        saturation_high=(((gate>.99)&mask[...,None]).sum((0,1,2))/eligible).tolist(),
        count_mean=vector(count).tolist(),log_count_mean=vector(summary[...,16]).tolist(),
        fraction_mean=vector(fraction).tolist(),
        floor_fraction=vector((count<A.EPSILON).float()).tolist(),
        count_degree_correlation=[correlation(count[...,a][live],degree[live]) for a in range(4)],
        gate_correlation=gate_corr,content_similarity=similarities,by_degree=bins,
        mean_neighbors=float(degree[live].float().mean()) if live.any() else 0.,
        no_neighbor_receivers=int((valid&~live).sum()),valid_receivers=int(valid.sum()),
        support_rms=float((heads.float().square().sum()/scale).sqrt()),
        state_rms=float((old.float().square().sum()/scale).sqrt()),
        support_to_state_norm=float(heads.float().norm()/old.float().norm().clamp_min(1e-12)),
        _heads=heads.detach())


def gradient_norms(base):
    return {n:float(p.grad.float().norm()) for n,p in base.named_parameters() if M.support_key(n) and p.grad is not None}


def diagnostic_sample(base,values,device):
    rng=capture_rank_state(base);flags=[(m,m.training) for m in base.modules()]
    hooks=[];native={};records={}
    try:
        base.eval()
        for layer,block in enumerate(base.mod.blocks):
            def capture(_m,_i,out,index=layer):native[index]=out[0].detach()
            hooks.append(block.attn.attention.reference.register_forward_hook(capture))
        with torch.no_grad(),model_precision(base,device,training=False):
            with M.diagnostic_mode(base,observer=lambda layer,row:records.__setitem__(layer,row)):
                logits=base(*values)
            for h in hooks:h.remove()
            hooks=[]
            for layer,row in records.items():
                heads=row.pop('_heads');n=heads.shape[2]
                joined=heads.permute(2,0,1,3).reshape(n,len(values[0]),128)
                projected=F.linear(joined,base.mod.blocks[layer].attn.attention.reference.out_proj.weight,bias=None)
                row['support_to_native_norm']=float(projected.norm()/native[layer].norm().clamp_min(1e-12))
            with M.diagnostic_mode(base,mode='disabled'):disabled=base(*values)
            with M.diagnostic_mode(base,mode='counts_zero'):counts_zero=base(*values)
        return dict(events=len(values[0]),particles_input=values[0].shape[-1],
            healthy=bool(logits.isfinite().all() and disabled.isfinite().all() and counts_zero.isfinite().all()),
            disabled_logit_rms_change=float((disabled-logits).square().mean().sqrt()),
            counts_zero_logit_rms_change=float((counts_zero-logits).square().mean().sqrt()),
            layers=[dict(layer=i,**records[i]) for i in range(8)])
    finally:
        for h in hooks:h.remove()
        base._state_pair_cache.clear();base.mod.pair_embed.clear()
        for m,value in flags:m.training=value
        restore_rank_state(base,rng)
