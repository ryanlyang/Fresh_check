"""Instance-local state adapters; native attention and transformer forwards stay intact."""
from contextlib import contextmanager
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
from part_ev_fullscale.model import reference_metadata
from part_ev_state_compare.model import build_model as reference_model,state_branches
from part_ev_state_speed.model import FastStateMessage
from part_ev_message_study.model import normalize,chunk_call,mechanism_seed
from . import aggregation as A

BASE_PARAMETERS=2327462
PARAMETERS=2384070
CONFIG=dict(base='state_layerwise',state_blocks=list(range(8)),state_width=32,
    support_blocks=list(range(8)),support_channels=4,content_width=16,summary_width=72,
    fusion_hidden=32,fusion_output=128,output_bias=False,query_chunk=64,checkpoint_chunks=True,
    pooling='independent_sigmoid_directed_offdiagonal_all_senders',
    summary_layout='channel_major_16_content_log1p_count_fraction',
    count_epsilon=A.EPSILON,norm_epsilon=1e-5,backend='einsum_fp32',
    precision='fp32_norm_sigmoid_counts_sums_summary_fp16_autocast_linears',
    initialization='zero_output_isolated_per_layer_support-v1',
    relations='standard',pair_bias='layerwise',compact=False)


def support_key(name):return '.state.support.' in name
def branches(model):return [state.support for state in state_branches(model)]
def output_parameters(model):return [branch.output.weight for branch in branches(model)]
def upstream_parameters(model):return [p for n,p in model.named_parameters() if support_key(n) and '.output.' not in n]


class Support(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate=nn.Linear(32,4)
        self.value=nn.Linear(32,16,bias=False)
        self.hidden=nn.Linear(72,32)
        self.output=nn.Linear(32,128,bias=False)
        for module in (self.gate,self.value,self.hidden):nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(self.gate.bias);nn.init.zeros_(self.hidden.bias);nn.init.zeros_(self.output.weight)

    def project(self,r,valid):
        u=normalize(r.masked_fill(~valid[...,None],0))
        logits,values=self.gate(u),self.value(u)
        healthy=torch.isfinite(u).logical_or(~valid[...,None]).all()
        return logits,values,healthy

    def fuse(self,features,degree,*,mode='normal'):
        if mode=='counts_zero':
            f=features.reshape(*features.shape[:-1],4,18)
            features=torch.cat((f[...,:16],f[...,16:]*0),-1).flatten(-2)
        hidden=F.gelu(self.hidden(features))
        result=self.output(hidden).masked_fill(~(degree>0)[...,None],0)
        healthy=torch.isfinite(hidden).all() & torch.isfinite(result).all()
        return result*(0 if mode=='disabled' else 1),healthy

    def forward(self,r,valid,*,mode='normal',oracle=False,diagnostic=False):
        logits,values,healthy=self.project(r,valid)
        if oracle:
            from .oracle import dense_statistics
            stats=dense_statistics(logits,values,valid)
        else:stats=A.statistics(logits,values,valid)
        features,ok=A.finish(*stats)
        output,fused_ok=self.fuse(features,stats[2],mode=mode)
        result=(output,healthy & ok & fused_ok)
        # Only small no-grad diagnostic forwards request these tensors; never
        # keep them on the module or append from inside checkpoint replay.
        return (*result,logits,values,features,stats[2]) if diagnostic else result


def joint_chunk(state,weights,pairs,p,q,pair_valid,support_valid,*,mode,oracle,diagnostic):
    g=state.edge(pairs);p,q=p[:,:,None],q[:,None]
    r=F.gelu(normalize(g+p+q+p*q)).masked_fill(~pair_valid[...,None],0)
    pooled=torch.einsum('bhij,bijr->bhir',weights,r)
    old=torch.einsum('hor,bhir->bhio',state.output,pooled)
    extra,healthy,*details=state.support(r,support_valid,mode=mode,oracle=oracle,diagnostic=diagnostic)
    heads=extra.reshape(*extra.shape[:2],8,16).permute(0,2,1,3).to(old.dtype)
    combined=old+heads
    healthy=healthy & torch.isfinite(combined).all()
    return (combined,healthy,*details,old,heads) if diagnostic else (combined,healthy)


class SupportState(FastStateMessage):
    def __init__(self,old,support,*,chunk,checkpointed,oracle):
        # Keep the original module objects, parameter names/order and pair cache.
        super().__init__(old,old.cache,chunk)
        self.support=support
        self.checkpointed=checkpointed;self.oracle=oracle
        self.support_mode='normal';self.observer=None

    def forward(self,weights,stem,query,valid):
        diagnostic=self.observer is not None
        if self.support_mode not in ('normal','disabled','counts_zero'):raise ValueError('Unknown support mode')
        if (diagnostic or self.support_mode!='normal') and (
                self.training or torch.is_grad_enabled() or len(valid)>8 or query.shape[0]>128):
            raise ValueError('Support interventions require small isolated no-grad eval inputs')
        if self.oracle and len(valid)>8:raise ValueError('Dense oracle is small-test-only')
        pairs=self.cache.get(stem)
        states=normalize(query.transpose(0,1));p,q=self.receiver(states),self.sender(states)
        n=len(query);senders=torch.arange(n,device=query.device);parts=[];records=[]
        function=partial(joint_chunk,self,mode=self.support_mode,oracle=self.oracle,diagnostic=diagnostic)
        for start in range(0,n,self.chunk_size):
            stop=min(n,start+self.chunk_size)
            pair=valid[:,start:stop,None]&valid[:,None,:]
            offdiag=torch.arange(start,stop,device=query.device)[:,None]!=senders
            args=(weights[:,:,start:stop],pairs[:,start:stop],p[:,start:stop],q,pair,pair&offdiag)
            row=chunk_call(self,function,*args) if self.checkpointed else function(*args)
            parts.append(row[:2])
            if diagnostic:records.append((*row[2:],pair&offdiag))
        combined=torch.cat([r[0] for r in parts],2);healthy=torch.stack([r[1] for r in parts]).all()
        if diagnostic:
            # Called outside the pure chunk. No graphs are retained on modules.
            from .diagnostics import layer_statistics
            self.observer(layer_statistics(records,valid))
        # Existing training driver coordinates finite-loss failure before backward.
        # Rank-zero eval performs the same finite-prediction check without collectives.
        return torch.where(healthy,combined,torch.full_like(combined,float('nan')))


@contextmanager
def diagnostic_mode(model,*,mode='normal',observer=None):
    if model.training or torch.is_grad_enabled():raise ValueError('Isolated eval/no-grad diagnostic only')
    states=state_branches(model);saved=[(s.support_mode,s.observer) for s in states]
    try:
        for layer,state in enumerate(states):
            state.support_mode=mode
            state.observer=None if observer is None else partial(observer,layer)
        yield
    finally:
        for state,(old,callback) in zip(states,saved):state.support_mode=old;state.observer=callback
        model._state_pair_cache.clear();model.mod.pair_embed.clear()


def check_inputs(model,inputs):
    if len(inputs)!=4:raise ValueError('Only the standard four model inputs are supported')
    mask=inputs[3]
    if mask.ndim!=3 or mask.shape[1]!=1 or not 1<=mask.shape[-1]<=128 or not mask.bool().any(-1).all().item():
        raise ValueError('Each jet requires 1..128 positions and a valid particle')
    if (model.mod.for_inference or model.mod.fc is None
            or any(block.attn.attention.disable_new for block in model.mod.blocks)):
        raise ValueError('Support experiment requires full State-EV and classification logits')


def check_native_inputs(module,args,kwargs):
    if len(args)>3 or kwargs.get('uu') is not None or kwargs.get('uu_idx') is not None:
        raise ValueError('External pairs/export are unsupported in this standard-relations experiment')


def build_model(*,seed=101,trim=True,use_amp=False,chunk=64,checkpointed=True,oracle=False):
    if type(chunk) is not int or chunk<1:raise ValueError('Positive receiver chunk required')
    reference_metadata()
    model=reference_model('state_layerwise',seed=seed,trim=trim,use_amp=use_amp)
    if sum(p.numel() for p in model.parameters())!=BASE_PARAMETERS:raise ValueError('Reference count changed')
    for layer,block in enumerate(model.mod.blocks):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(mechanism_seed(seed,layer,'support-v1'))
            support=Support()
        old=block.attn.attention.state
        block.attn.attention.state=SupportState(old,support,chunk=chunk,checkpointed=checkpointed,oracle=oracle)
    model.register_forward_pre_hook(check_inputs)
    model.mod.register_forward_pre_hook(check_native_inputs,with_kwargs=True)
    model.study_architecture='state_support_layerwise'
    if sum(p.numel() for p in model.parameters())!=PARAMETERS:raise ValueError('Support parameter contract changed')
    return model
