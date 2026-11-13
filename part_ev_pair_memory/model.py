"""Directed, forward-local pair memory for the pinned all-eight State-EV model.

Only particle-block dispatch differs from Weaver 0.4.17's audited forward.
Native modules/state keys, the one trimmer, class attention, and operation order
are retained. Memory is an explicit local tensor, NEVER a module/hook cache.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from weaver.nn.model.ParticleTransformer import ParticleTransformer

from part_ev_fullscale.model import reference_metadata
from part_ev_state_compare.model import build_model as reference_model
from part_ev_message_study.model import normalize, chunk_call, mechanism_seed

PARAMETERS = 2364294
CONFIG = dict(blocks=8, state_width=32, heads=8, query_chunk=64,
    writers=list(range(7)), gates=list(range(1,7)), readers=list(range(1,8)),
    memory='directed_forward_local_fp32_read_before_write',
    writer='tanh_linear_nonaffine_LN_current_state_features',
    update='sigmoid_a_candidate_b_memory_d_product_beta_convex_mix',
    initial_write_fractions=[.2,.5], reader='zero_bias_free_pool_then_project',
    attention='unchanged_native_MHA_and_layerwise_physics_bias',
    pair_bias='layerwise', relations='standard', compact=False,
    checkpoint='pure_nonreentrant_query_chunks', finite_check='pre_tanh_and_pre_sigmoid')
MEMORY_MARKERS = ('.memory_writer.', '.memory_reader', '.memory_gate')
DIAGNOSTIC_FIELDS = ('memory_rms','candidate_rms','update_rms','write_mean',
    'write_saturation_fraction','history_to_state_norm','writer_gradient_norm','reader_gradient_norm')


def memory_key(name):
    return any(marker in name for marker in MEMORY_MARKERS)


class MemoryStateMessage(nn.Module):
    def __init__(self, original, layer, seed, chunk=64, checkpointed=True):
        super().__init__()
        self.receiver, self.sender, self.edge = original.receiver, original.sender, original.edge
        self.output, self.cache = original.output, original.cache
        self.layer, self.chunk_size, self.checkpointed = layer, chunk, checkpointed
        self.collect_diagnostics, self.last_diagnostics = False, None
        # Fork CPU RNG: adding memory cannot move old initialization/dropout RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(mechanism_seed(seed,layer,'pair-memory-v1'))
            self.memory_writer = nn.Linear(32,32) if layer<7 else None
            if self.memory_writer is not None:
                nn.init.xavier_uniform_(self.memory_writer.weight)
                nn.init.zeros_(self.memory_writer.bias)
        self.memory_reader = nn.Parameter(torch.zeros(8,16,32)) if layer>0 else None
        self.memory_gate = nn.Parameter(torch.zeros(4,32)) if 0<layer<7 else None
        if self.memory_gate is not None:
            with torch.no_grad(): self.memory_gate[3,:16].fill_(-math.log(4))

    def observation(self, pairs, p, q, valid):
        g = self.edge(pairs)
        p, q = p[:,:,None], q[:,None]
        # EXACT original state feature, including AMP arithmetic before LN.
        return F.gelu(normalize(g+p+q+p*q)).masked_fill(~valid[...,None],0)

    def write(self, observation, memory, valid):
        """Pure bounded recurrence. Return preactivation health, not clipped health."""
        if self.memory_writer is None:
            return None, None, None, torch.ones((),device=observation.device,dtype=torch.bool)
        dtype=torch.float64 if observation.dtype==torch.float64 else torch.float32
        # Keep the plan's dense-projection AMP policy. Only the recurrence and
        # nonlinearities run in FP32; an overflowing projection is still rejected.
        pre=self.memory_writer(normalize(observation))
        with torch.autocast(observation.device.type,enabled=False):
            pre=pre.to(dtype)
            finite=torch.isfinite(pre).all()
            candidate=pre.tanh().masked_fill(~valid[...,None],0)
            if self.layer==0:
                updated=candidate
                gate=None
            else:
                a,b,d,beta=self.memory_gate.to(dtype).unbind(0)
                logits=a*candidate+b*memory+d*(candidate*memory)+beta
                finite=finite & torch.isfinite(logits).all() & torch.isfinite(memory).all()
                gate=logits.sigmoid()
                updated=((1-gate)*memory+gate*candidate).masked_fill(~valid[...,None],0)
        return updated,candidate,gate,finite

    def normalized_chunk(self, weights, pairs, p, q, valid, memory):
        observation=self.observation(pairs,p,q,valid)
        current=torch.einsum('hor,bhir->bhio',self.output,
                            torch.einsum('bhij,bijr->bhir',weights,observation))
        # Read INCOMING memory, never the candidate or the just-updated memory.
        history=(torch.zeros_like(current) if self.memory_reader is None else
                 torch.einsum('hor,bhir->bhio',self.memory_reader,
                    torch.einsum('bhij,bijr->bhir',weights,memory.masked_fill(~valid[...,None],0))))
        updated,_,_,finite=self.write(observation,memory,valid)
        return current,history,updated,finite

    def forward(self, weights, stem, query, valid, memory):
        if (memory is None)!=(self.layer==0):
            raise ValueError('Block zero starts memory; all later blocks require incoming memory')
        if memory is not None and memory.shape!=(stem.shape[0],query.shape[0],query.shape[0],32):
            raise ValueError('Pair memory must use both native trimmed particle axes')
        pairs=self.cache.get(stem)
        states=normalize(query.transpose(0,1))
        p,q=self.receiver(states),self.sender(states)
        current,history,next_chunks=[],[],[]
        healthy=torch.ones((),device=query.device,dtype=torch.bool)
        samples=[]
        if self.collect_diagnostics and (self.training or torch.is_grad_enabled()):
            raise ValueError('Memory diagnostics require isolated no-grad eval')
        for start in range(0,weights.shape[2],self.chunk_size):
            stop=start+self.chunk_size
            pair=valid[:,start:stop,None]&valid[:,None,:]
            incoming=None if memory is None else memory[:,start:stop]
            args=(weights[:,:,start:stop],pairs[:,start:stop],p[:,start:stop],q,pair,incoming)
            # Bound method, explicit tensors, no RNG/BN/hooks/mutable memory.
            c,h,n,ok=(chunk_call(self,self.normalized_chunk,*args) if self.checkpointed
                      else self.normalized_chunk(*args))
            current.append(c); history.append(h); healthy=healthy & ok
            if n is not None: next_chunks.append(n)
            if self.collect_diagnostics:
                o=self.observation(args[1],args[2],q,pair)
                updated,candidate,gate,_=self.write(o,incoming,pair)
                def squares(t):
                    return 0. if t is None else float(t.float().masked_select(pair[...,None]).square().sum())
                selected=gate.masked_select(pair[...,None]) if gate is not None else None
                samples.append((int(pair.sum())*32,squares(incoming),squares(candidate),
                    squares(updated if incoming is None else None if updated is None else updated-incoming),
                    float(selected.sum()) if selected is not None else 0.,
                    float(((selected<.01)|(selected>.99)).sum()) if selected is not None else 0.,
                    selected.numel() if selected is not None else 0))
        c,h=torch.cat(current,2),torch.cat(history,2)
        if self.collect_diagnostics:
            sums=[sum(row[k] for row in samples) for k in range(7)]
            def gradnorm(parameters):
                return sum(float(p.grad.float().square().sum()) for p in parameters if p.grad is not None)**.5
            self.last_diagnostics=dict(layer=self.layer,
                memory_rms=(sums[1]/max(1,sums[0]))**.5,candidate_rms=(sums[2]/max(1,sums[0]))**.5,
                update_rms=(sums[3]/max(1,sums[0]))**.5,write_mean=sums[4]/max(1,sums[6]),
                write_saturation_fraction=sums[5]/max(1,sums[6]),
                history_to_state_norm=float(h.float().norm()/c.float().norm().clamp_min(1e-12)),
                writer_gradient_norm=gradnorm(self.memory_writer.parameters()) if self.memory_writer else 0.,
                reader_gradient_norm=gradnorm([self.memory_reader]) if self.memory_reader is not None else 0.)
        return c,h,torch.cat(next_chunks,1) if next_chunks else None,healthy


def particle_attention(block, query, padding_mask, memory):
    """Exactly one native MHA, one shared dropout sample, one extra out projection."""
    layer=block.attn
    attention=layer.attention
    stem,valid=attention._relation_stem,attention._query_valid
    if stem is None or valid is None or layer._bias is None:
        raise ValueError('Missing native trimmed layerwise pair context')
    ordinary,weights=attention.reference(query,query,query,key_padding_mask=padding_mask,
        need_weights=True,attn_mask=layer._bias,average_attn_weights=False,is_causal=False)
    ev=torch.einsum('hod,bhid->bhio',attention.edge_projection,
                    torch.einsum('bhij,bdij->bhid',weights,stem))
    current,history,memory,healthy=attention.state(weights,stem,query,valid,memory)
    joined=((ev+current)+history).permute(2,0,1,3).reshape(query.shape)
    extra=F.linear(joined,attention.reference.out_proj.weight,bias=None)
    extra=extra*valid.T.unsqueeze(-1).to(extra.dtype)
    return ordinary+extra,memory,healthy


def particle_block(block,x,padding_mask,memory):
    """Audited native Block.forward particle path, with explicit memory result."""
    residual=x
    x=block.pre_attn_norm(x)
    x,memory,healthy=particle_attention(block,x,padding_mask,memory)
    if block.c_attn is not None:
        tgt_len=x.size(0)
        x=x.view(tgt_len,-1,block.num_heads,block.head_dim)
        x=torch.einsum('tbhd,h->tbdh',x,block.c_attn)
        x=x.reshape(tgt_len,-1,block.embed_dim)
    if block.post_attn_norm is not None: x=block.post_attn_norm(x)
    x=block.dropout(x)
    x+=residual
    residual=x
    x=block.pre_fc_norm(x)
    x=block.act(block.fc1(x))
    x=block.act_dropout(x)
    if block.post_fc_norm is not None: x=block.post_fc_norm(x)
    x=block.fc2(x)
    x=block.dropout(x)
    if block.w_resid is not None: residual=torch.mul(block.w_resid,residual)
    x+=residual
    return x,memory,healthy


class PairMemoryTransformer(ParticleTransformer):
    def forward(self,x,v=None,mask=None,uu=None,uu_idx=None):
        if uu is not None or uu_idx is not None or self.for_inference:
            raise ValueError('This experiment uses standard dense relations and logits only')
        with torch.no_grad():
            x,v,mask,uu=self.trimmer(x,v,mask,uu)
            padding_mask=~mask.squeeze(1)
        try:
            with torch.autocast(x.device.type,enabled=self.use_amp):
                x=self.embed(x).masked_fill(~mask.permute(2,0,1),0)
                self.pair_embed(v,uu)  # One native pair encoder; binds all eight biases.
                memory=None
                healthy=torch.ones((),device=x.device,dtype=torch.bool)
                for block in self.blocks:
                    x,memory,ok=particle_block(block,x,padding_mask,memory)
                    healthy=healthy & ok
                # One synchronization per forward, NOT one per chunk. Detect
                # overflow even when tanh/sigmoid would have hidden it.
                if not healthy.item(): raise FloatingPointError('Nonfinite pair-memory preactivation')
                cls_tokens=self.cls_token.expand(1,x.size(1),-1)
                for block in self.cls_blocks:
                    cls_tokens=block(x,x_cls=cls_tokens,padding_mask=padding_mask)
                x_cls=self.norm(cls_tokens).squeeze(0)
                return x_cls if self.fc is None else self.fc(x_cls)
        finally:
            self.pair_embed.clear()


def check_inputs(_model,inputs):
    mask=inputs[3]
    if mask.ndim!=3 or mask.shape[1]!=1 or mask.shape[2]<1 or not mask.bool().any(dim=-1).all().item():
        raise ValueError('Each jet must have at least one valid particle')


def build_model(*,seed=101,trim=True,use_amp=False,chunk=64,checkpointed=True):
    if type(chunk) is not int or chunk<1: raise ValueError('Positive query chunk required')
    reference_metadata()  # Reject any unaudited Weaver version or source change.
    model=reference_model('state_layerwise',seed=seed,trim=trim,use_amp=use_amp)
    if type(model.mod) is not ParticleTransformer: raise ValueError('Expected pinned native transformer')
    for index,block in enumerate(model.mod.blocks):
        block.attn.attention.state=MemoryStateMessage(block.attn.attention.state,index,seed,chunk,checkpointed)
    # Same object/modules/hooks/state names; only private particle dispatch changes.
    model.mod.__class__=PairMemoryTransformer
    model.register_forward_pre_hook(check_inputs)
    model.study_architecture='state_pair_memory_layerwise'
    if sum(p.numel() for p in model.parameters())!=PARAMETERS:
        raise ValueError('Pair-memory parameter contract changed')
    return model


def states(model):
    return [block.attn.attention.state for block in model.mod.blocks]


def readers(model):
    return [s.memory_reader for s in states(model) if s.memory_reader is not None]


def writers_and_gates(model):
    return [p for n,p in model.named_parameters() if memory_key(n) and '.memory_reader' not in n]
