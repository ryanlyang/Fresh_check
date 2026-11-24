"""Native full LW State-EV with four independent nested geometric exchanges."""
import hashlib
from contextlib import contextmanager
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from weaver.nn.model.ParticleTransformer import ParticleTransformer
from part_ev_state_compare.model import build_model as reference_model
from .attention import Attention, StableNorm, reduction_dtype
from .config import COUNTS, SCIENCE
from .geometry import build_geometry
from .pooling import pool


@contextmanager
def isolated(seed,component):
    value=int.from_bytes(hashlib.sha256(f'multiscale-v1:{seed}:{component}'.encode()).digest()[:8],'little')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(value&((1<<63)-1));yield


def initialize(module):
    for m in module.modules():
        if isinstance(m,nn.Linear):nn.init.xavier_uniform_(m.weight);nn.init.zeros_(m.bias)
        elif isinstance(m,nn.LayerNorm):nn.init.ones_(m.weight);nn.init.zeros_(m.bias)


def zero_output(m):nn.init.zeros_(m.weight);nn.init.zeros_(m.bias)


def mask(x,valid):return x.masked_fill(~valid[...,None],0)


def residual(x,delta,valid):
    with torch.autocast(x.device.type,enabled=False):
        dtype=reduction_dtype(x)
        return mask(x.to(dtype)+delta.to(dtype),valid)


class MultiscaleStage(nn.Module):
    def __init__(self):
        super().__init__()
        for name in ('particle_norm','fine_seed_norm','fine_read_query_norm','fine_local_attn_norm',
                     'fine_ffn_norm','coarse_seed_norm','coarse_read_query_norm','coarse_read_kv_norm',
                     'coarse_global_attn_norm','coarse_ffn_norm','topdown_query_norm','topdown_kv_norm',
                     'feedback_fine_norm','feedback_coarse_norm'):
            setattr(self,name,StableNorm())
        self.fine_seed_linear=nn.Linear(263,128);self.coarse_seed_linear=nn.Linear(135,128)
        for name in ('particle_read','fine_local','child_read','coarse_global','topdown'):
            setattr(self,name,Attention())
        self.fine_ffn=nn.Sequential(nn.Linear(128,512),nn.GELU(),nn.Linear(512,128))
        self.coarse_ffn=nn.Sequential(nn.Linear(128,512),nn.GELU(),nn.Linear(512,128))
        self.feedback_input=nn.Linear(384,128);self.feedback_output=nn.Linear(128,128)
        initialize(self);zero_output(self.feedback_output)

    def forward(self,hidden,g,valid):
        fv,cv=g.fine_valid,g.coarse_valid
        u=self.particle_norm(mask(hidden,valid));mean,maximum=pool(u,g,valid)
        fine=mask(self.fine_seed_norm(F.gelu(self.fine_seed_linear(torch.cat(
            (mean,maximum,g.fine_descriptors.to(mean.dtype)),-1)))),fv)
        delta,a=self.particle_read(self.fine_read_query_norm(fine),u,valid,g.fine.transpose(1,2).bool(),fv)
        fine=residual(fine,delta,fv)
        siblings=(g.parent@g.parent.transpose(1,2)).bool()
        norm=mask(self.fine_local_attn_norm(fine),fv)
        delta,b=self.fine_local(norm,norm,fv,siblings,fv)
        fine=residual(fine,delta,fv)
        fine=residual(fine,self.fine_ffn(self.fine_ffn_norm(fine)),fv)
        with torch.autocast(hidden.device.type,enabled=False):
            child_mean=g.parent.to(fine.dtype).transpose(1,2)@(fine*g.fine_count[...,None])
            child_mean=child_mean/g.coarse_count.clamp_min(1)[...,None]
        coarse=mask(self.coarse_seed_norm(F.gelu(self.coarse_seed_linear(torch.cat(
            (child_mean,g.coarse_descriptors.to(child_mean.dtype)),-1)))),cv)
        delta,c=self.child_read(self.coarse_read_query_norm(coarse),mask(self.coarse_read_kv_norm(fine),fv),
            fv,g.parent.transpose(1,2).bool(),cv)
        coarse=residual(coarse,delta,cv)
        norm=mask(self.coarse_global_attn_norm(coarse),cv)
        delta,d=self.coarse_global(norm,norm,cv,query_valid=cv)
        coarse=residual(coarse,delta,cv)
        coarse=residual(coarse,self.coarse_ffn(self.coarse_ffn_norm(coarse)),cv)
        delta,e=self.topdown(self.topdown_query_norm(fine),mask(self.topdown_kv_norm(coarse),cv),cv,query_valid=fv)
        fine=residual(fine,delta,fv)
        own_fine=mask(self.feedback_fine_norm(fine),fv).gather(1,g.fine_ids.clamp_min(0)[...,None].expand(-1,-1,128))
        own_coarse=mask(self.feedback_coarse_norm(coarse),cv).gather(1,g.coarse_ids.clamp_min(0)[...,None].expand(-1,-1,128))
        with torch.profiler.record_function('multiscale.feedback'):
            delta=self.feedback_output(F.gelu(self.feedback_input(torch.cat((u,own_fine,own_coarse),-1))))
        out=hidden+mask(delta.to(hidden.dtype),valid)
        health=g.healthy&a&b&c&d&e&torch.isfinite(fine).all()&torch.isfinite(coarse).all()&torch.isfinite(out).all()
        return out,fine,coarse,health


class FinalRead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fine_query_norm,self.fine_norm=StableNorm(),StableNorm()
        self.coarse_query_norm,self.coarse_norm=StableNorm(),StableNorm()
        self.fine_read,self.coarse_read=Attention(),Attention()
        self.fusion_input=nn.Linear(256,128);self.fusion_output=nn.Linear(128,128)
        initialize(self);zero_output(self.fusion_output)

    def forward(self,cls,fine,coarse,g):
        mf,a=self.fine_read(self.fine_query_norm(cls[:,None]),mask(self.fine_norm(fine),g.fine_valid),g.fine_valid)
        mc,b=self.coarse_read(self.coarse_query_norm(cls[:,None]),mask(self.coarse_norm(coarse),g.coarse_valid),g.coarse_valid)
        with torch.profiler.record_function('multiscale.fusion'):
            delta=self.fusion_output(F.gelu(self.fusion_input(torch.cat((mf,mc),-1))))[:,0]
        fused=cls+delta.to(cls.dtype)
        return fused,a&b&torch.isfinite(fused).all()


class Multiscale(nn.Module):
    def __init__(self,seed=101):
        super().__init__();self.stages=nn.ModuleList()
        for i in range(4):
            with isolated(seed,f'stage_{i}'):self.stages.append(MultiscaleStage())
        with isolated(seed,'final_read'):self.final_read=FinalRead()


class MultiscaleTransformer(ParticleTransformer):
    def forward(self,x,v=None,mask=None,uu=None,uu_idx=None):
        if uu is not None or uu_idx is not None or self.for_inference or self.fc is None or v is None:
            raise ValueError('Multiscale requires physical standard-relations ten-logit classification')
        try:
            with torch.no_grad():
                x,v,mask,_=self.trimmer(x,v,mask,None);valid=mask[:,0].bool();padding=~valid
                geometry=build_geometry(v.transpose(1,2),valid)
            with torch.autocast(x.device.type,enabled=self.use_amp):
                hidden=self.embed(x).masked_fill(~mask.permute(2,0,1),0)
                bias=self.pair_embed(v,None);health=geometry.healthy
                for i,block in enumerate(self.blocks):
                    hidden=block(hidden,padding_mask=padding,attn_mask=bias)
                    if (i+1) in SCIENCE['stages_after']:
                        stage=self.multiscale.stages[(i+1)//2-1];args=(hidden.transpose(0,1),geometry,valid)
                        if self.training and torch.is_grad_enabled() and self.multiscale_checkpointed:
                            h,fine,coarse,ok=checkpoint(stage,*args,use_reentrant=False,preserve_rng_state=False)
                        else:h,fine,coarse,ok=stage(*args)
                        hidden=h.transpose(0,1);health=health&ok
                cls=self.cls_token.expand(1,hidden.shape[1],-1)
                for block in self.cls_blocks:cls=block(hidden,x_cls=cls,padding_mask=padding)
                cls=self.norm(cls).squeeze(0)
                fused,ok=self.multiscale.final_read(cls,fine,coarse,geometry)
                logits=self.fc(fused)
                return torch.where(health&ok,logits,torch.full_like(logits,float('nan')))
        finally:self.pair_embed.clear()


def build_model(architecture='multiscale',*,seed=101,trim=True,use_amp=False,checkpointed=True):
    if architecture not in COUNTS:raise ValueError('Unknown multiscale arm')
    model=reference_model('state_layerwise',seed=seed,trim=trim,use_amp=use_amp)
    if architecture=='multiscale':
        if type(model.mod) is not ParticleTransformer:raise ValueError('Pinned native type changed')
        model.mod.multiscale=Multiscale(seed);model.mod.multiscale_checkpointed=checkpointed
        model.mod.__class__=MultiscaleTransformer
    if sum(p.numel() for p in model.parameters())!=COUNTS[architecture]:raise ValueError('Multiscale parameter inventory changed')
    return model


def inventory(model):
    return {name:dict(shape=list(p.shape),parameters=p.numel()) for name,p in model.named_parameters()}
