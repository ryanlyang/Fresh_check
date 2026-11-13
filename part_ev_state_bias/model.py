"""Bounded state-pair bias added before the ONE native attention call.

Old models are composed, never patched. The eight new zero-initialized heads
retain their gradients; there is deliberately no zero-value bypass.
"""
import torch
from torch import nn
from torch.nn import functional as F

from part_ev_state_compare.model import build_model as reference_model
from part_ev_message_study.model import MessageAttention, normalize, chunk_call

PARAMETERS = 2329510
CONFIG = dict(blocks=8, state_width=32, heads=8, query_chunk=64,
              bias_bound=1., bias_initialization='zero', bias_projection_precision='fp32',
              state_features='unchanged_GELU_LN_g_plus_p_plus_q_plus_p_times_q',
              attention='one_native_MHA_shared_post_dropout_weights',
              pair_bias='layerwise', relations='standard', compact=False)


class StateBiasAttention(MessageAttention):
    def __init__(self, original, *, chunk=64, checkpointed=True):
        nn.Module.__init__(self)
        for name in ('reference', 'edge_projection', 'state', 'routing', 'variance',
                     'embed_dim', 'num_heads', 'head_dim'):
            setattr(self, name, getattr(original, name))
        self._relation_stem = self._query_valid = None
        self.collect_diagnostics = self.disable_new = False
        self.last_diagnostics = None
        self.bias_head = nn.Parameter(torch.zeros(8, 32))  # No RNG consumption.
        self.bias_chunk = chunk
        self.checkpointed = checkpointed

    def feature_bias(self, pairs, receiver, sender, pair_valid):
        g = self.state.edge(pairs)
        p, q = receiver[:, :, None], sender[:, None]
        r = F.gelu(normalize(g + p + q + p*q))
        r = r.masked_fill(~pair_valid[..., None], 0)
        dtype = torch.float64 if r.dtype == torch.float64 else torch.float32
        with torch.autocast(r.device.type, enabled=False):
            correction = F.linear(r.to(dtype), self.bias_head.to(dtype)).tanh()
        return correction.permute(0, 3, 1, 2).masked_fill(~pair_valid[:, None], 0)

    def correction(self, query, stem, valid):
        pairs = self.state.cache.get(stem)
        states = normalize(query.transpose(0, 1))
        p, q = self.state.receiver(states), self.state.sender(states)
        pieces = []
        for start in range(0, query.shape[0], self.bias_chunk):
            stop = start + self.bias_chunk
            pair = valid[:, start:stop, None] & valid[:, None, :]
            args = (pairs[:, start:stop], p[:, start:stop], q, pair)
            pieces.append(chunk_call(self, self.feature_bias, *args) if self.checkpointed
                          else self.feature_bias(*args))
        return torch.cat(pieces, 2)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True,
                attn_mask=None, average_attn_weights=True, is_causal=False):
        stem, valid = self._relation_stem, self._query_valid
        if (stem is None or valid is None or query is not key or key is not value
                or is_causal or attn_mask is None or attn_mask.dtype == torch.bool):
            raise ValueError('Requires bound noncausal self-attention and additive layerwise bias')
        if self.disable_new:
            raise ValueError('Disable-by-bypass is not part of this experiment')
        delta = self.correction(query, stem, valid)
        merged = attn_mask + delta.reshape_as(attn_mask).to(attn_mask.dtype)
        # The inherited implementation calls native MHA once, then uses those
        # returned weights for both EV and State-EV. No extra softmax/dropout.
        result = super().forward(query, key, value, key_padding_mask=key_padding_mask,
            need_weights=True, attn_mask=merged,
            average_attn_weights=False, is_causal=False)
        if self.collect_diagnostics:
            with torch.no_grad():
                pair = valid[:, None, :, None] & valid[:, None, None, :]
                values = delta.float().masked_fill(~pair, 0)
                count = pair.sum().clamp_min(1)
                self.last_diagnostics.update(
                    bias_rms_per_head=(values.square().sum((0,2,3))/count).sqrt().tolist(),
                    bias_abs_max_per_head=values.abs().amax((0,2,3)).tolist(),
                    bias_saturation_per_head=((values.abs()>.95).sum((0,2,3))/count).tolist(),
                    bias_gradient_norm_per_head=(self.bias_head.grad.float().norm(dim=1).tolist()
                        if self.bias_head.grad is not None else None))
                if not self.training:
                    # Eval weights have no dropout. Remove ONLY this correction
                    # at the same Q/K/states, not by a second altered model pass.
                    weights = result[1].float()
                    control = weights * (-values).exp()
                    control = control / control.sum(-1,keepdim=True).clamp_min(1e-12)
                    def per_head(rows):
                        return (rows.masked_fill(~valid[:,None],0).sum((0,2))/valid.sum()).tolist()
                    def entropy(prob):
                        return -(prob*prob.clamp_min(1e-12).log()).sum(-1)
                    self.last_diagnostics.update(
                        corrected_entropy_per_head=per_head(entropy(weights)),
                        uncorrected_entropy_per_head=per_head(entropy(control)),
                        attention_total_variation_per_head=per_head(.5*(weights-control).abs().sum(-1)))
        weights = result[1].mean(1) if average_attn_weights else result[1]
        return result[0], weights if need_weights else None


def check_inputs(_model, inputs):
    mask = inputs[3]
    if (mask.ndim != 3 or mask.shape[1] != 1 or mask.shape[2] < 1
            or not mask.bool().any(dim=-1).all().item()):
        raise ValueError('Each jet must have at least one valid particle')


def build_model(*, seed=101, trim=True, use_amp=False, chunk=64, checkpointed=True):
    if type(chunk) is not int or chunk < 1:
        raise ValueError('Positive query chunk required')
    model = reference_model('state_layerwise', seed=seed, trim=trim, use_amp=use_amp)
    for block in model.mod.blocks:
        block.attn.attention = StateBiasAttention(block.attn.attention, chunk=chunk,
                                                 checkpointed=checkpointed)
        if not checkpointed:
            # Unchunked bias oracle; existing post-attention path is retained.
            block.attn.attention.state.chunk_size = chunk
    model.register_forward_pre_hook(check_inputs)
    model.study_architecture = 'state_bias_layerwise'
    if sum(p.numel() for p in model.parameters()) != PARAMETERS:
        raise ValueError('State-bias parameter contract changed')
    return model


def attentions(model):
    return [block.attn.attention for block in model.mod.blocks]
