"""Endpoint-conditioned channel modulation INSIDE the existing state message.

Native attention, layerwise physics bias and ordinary EV are not replaced.
No attention-bias extension, selected relations or compact path is included.
"""
import torch
from torch import nn
from torch.nn import functional as F

from part_ev_state_compare.model import build_model as reference_model
from part_ev_message_study.model import normalize, chunk_call

PARAMETERS = 2343846
CONFIG = dict(blocks=8, state_width=32, heads=8, query_chunk=64,
    gate='tanh(A_p_plus_C_q)', gate_bound=1., gate_initialization='zero',
    gate_projections='per_particle_bias_free_32x32_independent_endpoints_and_layers',
    gate_precision='fp32_projection_tanh_product_residual_after_native_h',
    feature='GELU_LN_original_h_plus_g_times_gate',
    attention='unchanged_native_MHA_and_layerwise_physics_bias',
    pair_bias='layerwise', relations='standard', compact=False)
GATE_SUFFIXES = ('.receiver_gate', '.sender_gate')
DIAGNOSTIC_FIELDS = ('gate_mean','gate_std','gate_abs_max','gate_saturation_fraction',
    'modulation_to_preactivation_norm','state_message_norm',
    'receiver_gate_norm','sender_gate_norm','receiver_gate_gradient_norm','sender_gate_gradient_norm')


class MultiplyStateMessage(nn.Module):
    def __init__(self, original, *, chunk=64, checkpointed=True):
        super().__init__()
        self.receiver, self.sender, self.edge = original.receiver, original.sender, original.edge
        self.output, self.cache = original.output, original.cache
        # Parameters (rather than randomly initialized Linear modules) preserve RNG.
        self.receiver_gate = nn.Parameter(torch.zeros(32,32))
        self.sender_gate = nn.Parameter(torch.zeros(32,32))
        self.chunk_size, self.checkpointed = chunk, checkpointed
        self.collect_diagnostics, self.last_diagnostics = False, None

    def gate_endpoints(self, p, q):
        dtype = torch.float64 if p.dtype == torch.float64 else torch.float32
        with torch.autocast(p.device.type, enabled=False):
            return (F.linear(p.to(dtype), self.receiver_gate.to(dtype)),
                    F.linear(q.to(dtype), self.sender_gate.to(dtype)))

    def preactivation(self, pairs, receiver, sender, receiver_gate, sender_gate):
        g = self.edge(pairs)
        p, q = receiver[:,:,None], sender[:,None]
        h = g + p + q + p*q  # Preserve original operation order, including AMP.
        dtype = torch.float64 if h.dtype == torch.float64 else torch.float32
        with torch.autocast(h.device.type, enabled=False):
            gamma = (receiver_gate[:,:,None] + sender_gate[:,None]).tanh()
            residual = g.to(dtype) * gamma
            updated = h.to(dtype) + residual
        return h, gamma, residual, updated

    def features(self, pairs, receiver, sender, receiver_gate, sender_gate, pair_valid):
        _, _, _, h = self.preactivation(pairs,receiver,sender,receiver_gate,sender_gate)
        return F.gelu(normalize(h)).masked_fill(~pair_valid[...,None],0)

    def normalized_chunk(self, weights, pairs, p, q, a, c, pair_valid):
        features = self.features(pairs,p,q,a,c,pair_valid)
        pooled = torch.einsum('bhij,bijr->bhir',weights,features)
        return torch.einsum('hor,bhir->bhio',self.output,pooled)

    def forward(self, weights, stem, query, valid):
        pairs = self.cache.get(stem)
        states = normalize(query.transpose(0,1))
        p, q = self.receiver(states), self.sender(states)
        a, c = self.gate_endpoints(p,q)  # Two O(N * 32^2) projections, never per pair.
        outputs = []
        for start in range(0,weights.shape[2],self.chunk_size):
            stop = start+self.chunk_size
            pair = valid[:,start:stop,None] & valid[:,None,:]
            args = (weights[:,:,start:stop],pairs[:,start:stop],p[:,start:stop],q,a[:,start:stop],c,pair)
            # Pure function: backward recomputes from explicit inputs after cache clear.
            outputs.append(chunk_call(self,self.normalized_chunk,*args) if self.checkpointed
                           else self.normalized_chunk(*args))
        message = torch.cat(outputs,2)
        if self.collect_diagnostics:
            if self.training or torch.is_grad_enabled():
                raise ValueError('Collect modulation diagnostics only in isolated no-grad eval')
            self.last_diagnostics = self.diagnose(pairs,p,q,a,c,valid,message)
        return message

    def diagnose(self,pairs,p,q,a,c,valid,message):
        # Only the small fixed diagnostic pass enters here. Retain scalar sums,
        # never full-jet gate tensors or autograd references across forwards.
        count = total = squares = saturated = maximum = h2 = residual2 = 0.
        with torch.no_grad():
            for start in range(0,valid.shape[1],self.chunk_size):
                stop=start+self.chunk_size
                pair=valid[:,start:stop,None]&valid[:,None,:]
                h,gamma,residual,_=self.preactivation(pairs[:,start:stop],p[:,start:stop],q,a[:,start:stop],c)
                gate=gamma.float().masked_select(pair[...,None])
                if not gate.numel(): continue
                count+=gate.numel();total+=float(gate.sum());squares+=float(gate.square().sum())
                saturated+=float((gate.abs()>.95).sum());maximum=max(maximum,float(gate.abs().max()))
                h2+=float(h.float().masked_select(pair[...,None]).square().sum())
                residual2+=float(residual.float().masked_select(pair[...,None]).square().sum())
            return dict(gate_mean=total/count,gate_std=max(0.,squares/count-(total/count)**2)**.5,
                gate_abs_max=maximum,gate_saturation_fraction=saturated/count,
                modulation_to_preactivation_norm=(residual2/max(h2,1e-24))**.5,
                state_message_norm=float(message.float().norm()),
                receiver_gate_norm=float(self.receiver_gate.float().norm()),
                sender_gate_norm=float(self.sender_gate.float().norm()),
                receiver_gate_gradient_norm=float(self.receiver_gate.grad.float().norm()) if self.receiver_gate.grad is not None else None,
                sender_gate_gradient_norm=float(self.sender_gate.grad.float().norm()) if self.sender_gate.grad is not None else None)


def check_inputs(_model, inputs):
    mask=inputs[3]
    if mask.ndim!=3 or mask.shape[1]!=1 or mask.shape[2]<1 or not mask.bool().any(dim=-1).all().item():
        raise ValueError('Each jet must have at least one valid particle')


def build_model(*,seed=101,trim=True,use_amp=False,chunk=64,checkpointed=True):
    if type(chunk) is not int or chunk<1: raise ValueError('Positive query chunk required')
    model=reference_model('state_layerwise',seed=seed,trim=trim,use_amp=use_amp)
    for block in model.mod.blocks:
        attention=block.attn.attention
        attention.state=MultiplyStateMessage(attention.state,chunk=chunk,checkpointed=checkpointed)
    model.register_forward_pre_hook(check_inputs)
    model.study_architecture='state_multiply_layerwise'
    if sum(p.numel() for p in model.parameters())!=PARAMETERS:
        raise ValueError('Relation-state multiplication parameter contract changed')
    return model


def states(model):
    return [block.attn.attention.state for block in model.mod.blocks]


def gate_parameters(model):
    return [p for state in states(model) for p in (state.receiver_gate,state.sender_gate)]
