"""Full native LW State-EV plus six shared-FFN/specialist corrections."""
from contextlib import contextmanager
import hashlib
import torch
from torch import nn
from weaver.nn.model.ParticleTransformer import Block,ParticleTransformer
from part_ev_fullscale.model import ParticleTransformerWrapper
from part_ev_state_compare.model import build_model as reference
from .config import SCIENCE,COUNTS
from .routing import StableNorm,route,reduction_dtype
from .dispatch import dispatch


@contextmanager
def isolated(seed,name):
    value=int.from_bytes(hashlib.sha256(f'ffn-experts-v1:{seed}:{name}'.encode()).digest()[:8],'little')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(value&((1<<63)-1));yield


class Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.input=nn.Linear(128,256);self.norm=StableNorm(256);self.output=nn.Linear(256,128)
        nn.init.xavier_uniform_(self.input.weight);nn.init.zeros_(self.input.bias)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,x):
        with torch.profiler.record_function('experts.mlp'):
            return self.output(self.norm(torch.nn.functional.gelu(self.input(x))))


class SpecialistBank(nn.Module):
    def __init__(self,seed=101,block=2,checkpointed=True):
        super().__init__();self.checkpointed=checkpointed
        self.experts=nn.ModuleList()
        for e in range(4):
            with isolated(seed,f'block_{block}:expert_{e}'):self.experts.append(Expert())
        with isolated(seed,f'block_{block}:router'):
            self.context_norm=StableNorm(128)
            self.router=nn.Sequential(nn.Linear(384,32),nn.GELU(),nn.Linear(32,4))
            nn.init.xavier_uniform_(self.router[0].weight);nn.init.zeros_(self.router[0].bias)
            nn.init.normal_(self.router[2].weight,std=.01);nn.init.zeros_(self.router[2].bias)

    def forward(self,u,valid,*,details=False):
        values=u.transpose(0,1)
        r=route(values,valid,self.context_norm,self.router)
        correction,executed,health=dispatch(r['packed'],r['ids'],r['weights'],self.experts,
            checkpointed=self.checkpointed and self.training)
        health=(health & r['health'] & (executed==r['requested']).all()
                & (executed.sum()==2*r['packed'].shape[0]))
        with torch.autocast(u.device.type,enabled=False):
            output=torch.zeros((values.shape[0]*values.shape[1],128),device=u.device,dtype=reduction_dtype(u))
            output=output.index_copy(0,r['positions'],correction).reshape(values.shape)
            p,w=r['probabilities'],r['weights']
            stats=dict(probability_sum=p.sum(0),requested=r['requested'],executed=executed,
                tokens=valid.sum(),health=health,
                full_entropy_sum=-(p*p.clamp_min(1e-30).log()).sum().detach(),
                selected_entropy_sum=-(w*w.clamp_min(1e-30).log()).sum().detach(),
                min_margin=r['margin'].min().detach() if r['margin'].numel() else p.new_tensor(0.),
                primary=torch.bincount(r['ids'][:,0],minlength=4),
                secondary=torch.bincount(r['ids'][:,1],minlength=4))
            if details:stats['details']={k:r[k] for k in ('ids','weights','probabilities','positions','logits','margin')}
        return output.transpose(0,1),stats,health


class SpecialistBlock(Block):
    def forward_with_router_stats(self,x,valid,*,padding_mask,attn_mask,details=False):
        # Weaver 0.4.17 particle-attention path, preserving its unusual head
        # transpose, post-norm, dropout and residual arithmetic exactly.
        residual=x
        x=self.pre_attn_norm(x)
        x=self.attn(x,x,x,key_padding_mask=padding_mask,attn_mask=attn_mask)[0]
        if self.c_attn is not None:
            n=x.size(0);x=x.view(n,-1,self.num_heads,self.head_dim)
            x=torch.einsum('tbhd,h->tbdh',x,self.c_attn).reshape(n,-1,self.embed_dim)
        if self.post_attn_norm is not None:x=self.post_attn_norm(x)
        x=self.dropout(x);x+=residual
        residual=x;u=self.pre_fc_norm(x)
        dense=self.act_dropout(self.act(self.fc1(u)))
        if self.post_fc_norm is not None:dense=self.post_fc_norm(dense)
        dense=self.fc2(dense)
        correction,stats,health=self.specialists(u,valid,details=details)
        x=self.dropout(dense+correction.to(dense.dtype))
        if self.w_resid is not None:residual=torch.mul(self.w_resid,residual)
        x+=residual
        return x,stats,health

    def forward(self,x,x_cls=None,padding_mask=None,attn_mask=None,*,return_router_stats=False,details=False):
        if x_cls is not None or padding_mask is None:raise ValueError('Specialists require particle padding mask')
        output,stats,health=self.forward_with_router_stats(x,~padding_mask,padding_mask=padding_mask,
            attn_mask=attn_mask,details=details)
        return (output,stats,health) if return_router_stats else output


class ExpertsTransformer(ParticleTransformer):
    def forward(self,x,v=None,mask=None,uu=None,uu_idx=None,*,return_router_stats=False,return_route_details=False):
        if uu is not None or uu_idx is not None or self.for_inference or self.fc is None:
            raise ValueError('Experts require standard-relations ten-logit classification')
        try:
            with torch.no_grad():
                x,v,mask,_=self.trimmer(x,v,mask,None);valid=mask[:,0].bool();padding=~valid
            with torch.autocast(x.device.type,enabled=self.use_amp):
                hidden=self.embed(x).masked_fill(~mask.permute(2,0,1),0)
                bias=self.pair_embed(v,None).view(-1,v.shape[-1],v.shape[-1])
                rows=[];health=valid.any(-1).all()
                for i,block in enumerate(self.blocks):
                    if i in SCIENCE['blocks']:
                        hidden,stats,ok=block(hidden,padding_mask=padding,attn_mask=bias,
                            return_router_stats=True,details=return_route_details)
                        rows.append(stats);health=health&ok
                    else:hidden=block(hidden,padding_mask=padding,attn_mask=bias)
                cls=self.cls_token.expand(1,hidden.shape[1],-1)
                for block in self.cls_blocks:cls=block(hidden,x_cls=cls,padding_mask=padding)
                logits=self.fc(self.norm(cls).squeeze(0))
                logits=torch.where(health,logits,torch.full_like(logits,float('nan')))
            return dict(logits=logits,banks=rows,health=health) if return_router_stats else logits
        finally:self.pair_embed.clear()


class ExpertsWrapper(ParticleTransformerWrapper):
    def forward(self,points,features,vectors,mask,*,return_router_stats=False,return_route_details=False):
        if return_route_details and not return_router_stats:raise ValueError('Route details require explicit statistics')
        try:
            if self.experts_architecture=='experts':
                return self.mod(features,v=vectors,mask=mask,return_router_stats=return_router_stats,
                    return_route_details=return_route_details)
            logits=self.mod(features,v=vectors,mask=mask)
            if return_router_stats:return dict(logits=logits,banks=[],health=torch.isfinite(logits).all())
            return logits
        finally:self.mod.pair_embed.clear()


def build_model(architecture='experts',*,seed=101,trim=True,use_amp=False,checkpointed=True):
    if architecture not in COUNTS:raise ValueError('Unknown experts arm')
    base=reference('state_layerwise',seed=seed,trim=trim,use_amp=use_amp)
    if type(base) is not ParticleTransformerWrapper or type(base.mod) is not ParticleTransformer:
        raise ValueError('Pinned reference class changed')
    if architecture=='experts':
        for i in SCIENCE['blocks']:
            block=base.mod.blocks[i]
            if type(block) is not Block:raise ValueError('Pinned native block changed')
            block.specialists=SpecialistBank(seed,i,checkpointed)
            block.__class__=SpecialistBlock
        base.mod.__class__=ExpertsTransformer
    base.__class__=ExpertsWrapper;base.experts_architecture=architecture
    if sum(p.numel() for p in base.parameters())!=COUNTS[architecture]:raise ValueError('Experts parameter count changed')
    return base


def banks(base):return [block.specialists for block in base.mod.blocks if hasattr(block,'specialists')]


def inventory(base):return {n:dict(shape=list(p.shape),parameters=p.numel()) for n,p in base.named_parameters()}
