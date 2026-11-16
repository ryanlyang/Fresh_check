"""Two late triangle branches on the complete noncompact Layerwise State-EV."""
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
from weaver.nn.model.ParticleTransformer import ParticleTransformer
from part_ev_fullscale.model import reference_metadata
from part_ev_state_compare.model import build_model as reference_model,state_branches
from part_ev_message_study.model import normalize,chunk_call,mechanism_seed
from .neighbors import NeighborBuilder
from . import contraction

BASE_PARAMETERS=2327462
PARAMETERS=2334662
CONFIG=dict(base='state_layerwise',state_blocks=list(range(8)),state_width=32,
    triangle_blocks=[6,7],triangle_width=16,neighbor_budget=8,
    geometry='native_FP32_rapidity_wrapped_phi_post_trim',cutoff_ties='fractional_boundary_all_contributors',
    backend='sparse_tiled',feature_chunk=64,receiver_tile=16,sender_tile=64,intermediary_tile=4,tie_tile=256,
    checkpoint_chunks=True,precision='fp32_norm_factors_caches_mix_fp16_autocast_projections',
    initialization='isolated_per_layer_rng_zero_T',norm_epsilon=1e-5,
    relations='standard',pair_bias='layerwise',compact=False)


def triangle_key(name):return '.triangle.' in name
def branches(model):return [model.mod.blocks[l].attn.attention.triangle for l in (6,7)]
def output_parameters(model):return [t.output for t in branches(model)]
def upstream_parameters(model):return [p for n,p in model.named_parameters() if triangle_key(n) and not n.endswith('.output')]


class Triangle(nn.Module):
    def __init__(self):
        super().__init__()
        self.left=nn.Linear(32,16,bias=False)
        self.right=nn.Linear(32,16,bias=False)
        self.gate=nn.Linear(32,16)
        self.output=nn.Parameter(torch.zeros(8,16,16))
        for linear in (self.left,self.right,self.gate):nn.init.xavier_uniform_(linear.weight)
        nn.init.zeros_(self.gate.bias)

    def project(self,r,valid,diagnostic=False):
        with torch.profiler.record_function('triangles.factor_projections'):
            return self._project(r,valid,diagnostic)

    def _project(self,r,valid,diagnostic=False):
        u=normalize(r.masked_fill(~valid[...,None],0))
        raw=(self.left(u),self.right(u),self.gate(u))
        dtype=torch.float64 if u.dtype==torch.float64 else torch.float32
        with torch.autocast(r.device.type,enabled=False):
            x,y,z=(v.to(dtype) for v in raw)
            # Inspect BEFORE tanh/sigmoid can turn infinities into finite factors.
            healthy=torch.isfinite(u).logical_or(~valid[...,None]).all()
            for v in (x,y,z):healthy=healthy & torch.isfinite(v).logical_or(~valid[...,None]).all()
            a=x.tanh().masked_fill(~valid[...,None],0)
            b=y.tanh().masked_fill(~valid[...,None],0)
            gate=z.sigmoid().masked_fill(~valid[...,None],0)
            result=(a,b,gate,healthy)
            if diagnostic:
                with torch.no_grad():
                    scalars=[v.masked_fill(~valid[...,None],0).square().sum() for v in (x,y,z)]
                    scalars += [((v.abs()>.99)&valid[...,None]).sum() for v in (a,b)]
                    scalars += [(((gate<.01)|(gate>.99))&valid[...,None]).sum(),valid.sum()*16]
                    result=(*result,torch.stack(scalars).float())
            return result


def joint_features(state,triangle,weights,pairs,p,q,pair_valid,triangle_valid,*,diagnostic=False):
    g=state.edge(pairs);p,q=p[:,:,None],q[:,None]
    r=F.gelu(normalize(g+p+q+p*q)).masked_fill(~pair_valid[...,None],0)
    pooled=torch.einsum('bhij,bijr->bhir',weights,r)
    old=torch.einsum('hor,bhir->bhio',state.output,pooled)
    return (old,*triangle.project(r,triangle_valid,diagnostic))


def stage_a(state,triangle,weights,stem,query,valid,*,chunk,checkpointed,diagnostic):
    pairs=state.cache.get(stem)
    states=normalize(query.transpose(0,1));p,q=state.receiver(states),state.sender(states)
    pieces=[];n=query.shape[0];sender=torch.arange(n,device=query.device)
    function=partial(joint_features,state,triangle,diagnostic=diagnostic)
    for start in range(0,n,chunk):
        stop=min(n,start+chunk)
        pair=valid[:,start:stop,None]&valid[:,None,:]
        diagonal=torch.arange(start,stop,device=query.device)[:,None]==sender
        args=(weights[:,:,start:stop],pairs[:,start:stop],p[:,start:stop],q,pair,pair&~diagonal)
        pieces.append(chunk_call(state,function,*args) if checkpointed else function(*args))
    old=torch.cat([r[0] for r in pieces],2)
    a,b,gate=(torch.cat([r[k] for r in pieces],1) for k in (1,2,3))
    healthy=torch.stack([r[4] for r in pieces]).all()
    stats=torch.stack([r[5] for r in pieces]).sum(0) if diagnostic else None
    return old,a,b,gate,healthy,stats


def triangle_attention(block,query,padding_mask,graph,execution,*,diagnostic=False,disabled=False):
    layer=block.attn;attention=layer.attention
    stem,valid=attention._relation_stem,attention._query_valid
    if stem is None or valid is None or layer._bias is None:raise ValueError('Missing native trimmed pair context')
    ordinary,weights=attention.reference(query,query,query,key_padding_mask=padding_mask,
        need_weights=True,attn_mask=layer._bias,average_attn_weights=False,is_causal=False)
    pooled=torch.einsum('bhij,bdij->bhid',weights,stem)
    ev=torch.einsum('hod,bhid->bhio',attention.edge_projection,pooled)
    with torch.profiler.record_function('triangles.state_and_projections'):
        state,a,b,gate,healthy,stats=stage_a(attention.state,attention.triangle,weights,stem,query,valid,
            chunk=execution['feature_chunk'],checkpointed=execution['checkpointed'],diagnostic=diagnostic)
    if execution['backend']=='dense_oracle':
        if len(valid)>8 or query.shape[0]>128:raise ValueError('Dense oracle is small-test-only')
        mixed=contraction.dense(a,b,gate,weights,graph)
    else:
        mixed=contraction.sparse(a,b,gate,weights,graph,**{k:execution[k] for k in
            ('receiver_tile','sender_tile','intermediary_tile','tie_tile','checkpointed')})
    with torch.profiler.record_function('triangles.output_projection'):
        triangle=torch.einsum('hor,bhir->bhio',attention.triangle.output,mixed).to(state.dtype)
    healthy=healthy & torch.isfinite(mixed).all() & torch.isfinite(triangle).all()
    joined=((ev+state)+triangle*(0 if disabled else 1)).permute(2,0,1,3).reshape(query.shape)
    extra=F.linear(joined,attention.reference.out_proj.weight,bias=None)
    extra=extra*valid.T.unsqueeze(-1).to(extra.dtype)
    info=None
    if diagnostic:
        with torch.no_grad():
            den=(graph.weights.sum(-1,keepdim=True)-graph.weights)
            pair=valid[:,:,None]&valid[:,None,:]&~torch.eye(len(query),device=query.device,dtype=torch.bool)
            count=stats[-1].clamp_min(1)
            info=dict(left_preactivation_rms=float((stats[0]/count).sqrt()),
                right_preactivation_rms=float((stats[1]/count).sqrt()),gate_preactivation_rms=float((stats[2]/count).sqrt()),
                left_saturation=float(stats[3]/count),right_saturation=float(stats[4]/count),gate_saturation=float(stats[5]/count),
                triangle_to_state_norm=float(triangle.float().norm()/state.float().norm().clamp_min(1e-12)),
                triangle_to_ev_norm=float(triangle.float().norm()/ev.float().norm().clamp_min(1e-12)),
                zero_denominator_fraction=float(((den<=0)&pair).sum()/pair.sum().clamp_min(1)))
    return ordinary+extra,healthy,info


def triangle_block(block,x,padding_mask,graph,execution,*,diagnostic=False,disabled=False):
    """Literal pinned native residual/head-scaling/FFN order."""
    residual=x;x=block.pre_attn_norm(x)
    x,healthy,info=triangle_attention(block,x,padding_mask,graph,execution,diagnostic=diagnostic,disabled=disabled)
    if block.c_attn is not None:
        tgt_len=x.size(0);x=x.view(tgt_len,-1,block.num_heads,block.head_dim)
        x=torch.einsum('tbhd,h->tbdh',x,block.c_attn).reshape(tgt_len,-1,block.embed_dim)
    if block.post_attn_norm is not None:x=block.post_attn_norm(x)
    x=block.dropout(x);x+=residual
    residual=x;x=block.pre_fc_norm(x);x=block.act(block.fc1(x));x=block.act_dropout(x)
    if block.post_fc_norm is not None:x=block.post_fc_norm(x)
    x=block.fc2(x);x=block.dropout(x)
    if block.w_resid is not None:residual=torch.mul(block.w_resid,residual)
    x+=residual
    return x,healthy,info


class TriangleTransformer(ParticleTransformer):
    def forward_details(self,x,v=None,mask=None,uu=None,uu_idx=None,*,diagnostic=False,disabled=False):
        if uu is not None or uu_idx is not None or self.for_inference or self.fc is None:
            raise ValueError('Triangle experiment requires standard relations and classification logits')
        if (diagnostic or disabled) and (self.training or torch.is_grad_enabled() or len(x)>8 or x.shape[-1]>32):
            raise ValueError('Diagnostics require small isolated no-grad eval inputs')
        try:
            with torch.no_grad():
                x,v,mask,uu=self.trimmer(x,v,mask,uu)
                valid=mask.squeeze(1);padding_mask=~valid
                graph=self.triangle_neighbors(v,valid)
            healthy=graph.healthy;layers=[]
            with torch.autocast(x.device.type,enabled=self.use_amp):
                x=self.embed(x).masked_fill(~mask.permute(2,0,1),0)
                bias=self.pair_embed(v,uu)
                for index,block in enumerate(self.blocks):
                    if index<6:x=block(x,padding_mask=padding_mask,attn_mask=bias)
                    else:
                        x,ok,info=triangle_block(block,x,padding_mask,graph,self.triangle_execution,
                            diagnostic=diagnostic,disabled=disabled)
                        healthy=healthy & ok
                        if diagnostic:layers.append(dict(layer=index,**info))
                cls_tokens=self.cls_token.expand(1,x.size(1),-1)
                for block in self.cls_blocks:cls_tokens=block(x,x_cls=cls_tokens,padding_mask=padding_mask)
                logits=self.fc(self.norm(cls_tokens).squeeze(0))
                logits=torch.where(healthy,logits,torch.full_like(logits,float('nan')))
                return logits,dict(healthy=healthy,layers=layers)
        finally:
            self.pair_embed.clear()
            # Also cover direct diagnostic calls that bypass the outer wrapper's
            # always-call cleanup hook. Backward uses only explicit saved tensors.
            self.blocks[0].attn.attention.state.cache.clear()

    def forward(self,x,v=None,mask=None,uu=None,uu_idx=None):
        return self.forward_details(x,v,mask,uu,uu_idx)[0]


def check_inputs(_model,inputs):
    mask=inputs[3]
    if mask.ndim!=3 or mask.shape[1]!=1 or not 1<=mask.shape[-1]<=128 or not mask.bool().any(-1).all().item():
        raise ValueError('Each jet needs 1..128 positions and at least one valid particle')


def build_model(*,seed=101,trim=True,use_amp=False,chunk=64,checkpointed=True,
                receiver_tile=16,sender_tile=64,intermediary_tile=4,tie_tile=256,backend='sparse_tiled'):
    if any(type(k) is not int or k<1 for k in (chunk,receiver_tile,sender_tile,intermediary_tile,tie_tile)):
        raise ValueError('Positive chunk/tile sizes required')
    if backend not in ('sparse_tiled','dense_oracle'):raise ValueError('Unknown backend')
    reference_metadata()
    model=reference_model('state_layerwise',seed=seed,trim=trim,use_amp=use_amp)
    if type(model.mod) is not ParticleTransformer:raise ValueError('Expected pinned native transformer')
    if sum(p.numel() for p in model.parameters())!=BASE_PARAMETERS:
        raise ValueError('Reference parameter count changed before triangle construction')
    for layer in (6,7):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(mechanism_seed(seed,layer,'triangle-v1'))
            model.mod.blocks[layer].attn.attention.triangle=Triangle()
    model.mod.triangle_neighbors=NeighborBuilder()
    model.mod.triangle_execution=dict(feature_chunk=chunk,receiver_tile=receiver_tile,sender_tile=sender_tile,
        intermediary_tile=intermediary_tile,tie_tile=tie_tile,checkpointed=checkpointed,backend=backend)
    model.mod.__class__=TriangleTransformer
    model.register_forward_pre_hook(check_inputs)
    model.study_architecture='state_triangles_layerwise'
    if sum(p.numel() for p in model.parameters())!=PARAMETERS:raise ValueError('Triangle parameter contract changed')
    return model
