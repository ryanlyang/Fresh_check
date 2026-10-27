"""Layerwise all-eight State-EV, shared all-eight State-EV, and shared wide ParT.

Historical implementations are composed, never patched. Only the final bias
head is copied for the layerwise model; the pair stem remains shared.
"""
import importlib.util
import torch
from torch import nn
from part_ev_fullscale import model as native
from part_ev_state_speed.model import build_model as state_model
from part_ev_message_study.model import diagnostic_mode, diagnostics
from part_layerwise_screen.model import LayerwiseAttention, LayerwisePairs

FFN_WIDTH = 601
COUNTS = dict(part=2143486, part_ev=2209022, state_fast=2323710,
              state_layerwise=2327462, part_state_matched=2327894)
CONFIG = dict(state_blocks=list(range(8)), state_width=32, query_chunk=64,
              shared_pair_normalization=True, checkpoint_chunks=True,
              layerwise_bias='eight_independent_copied_heads_shared_native_pair_stem',
              matched_bias='shared', matched_particle_ffn_width=FFN_WIDTH,
              class_ffn_width=512, matching_target=COUNTS['state_layerwise'],
              matching_delta=432, initialization='fresh_seeded_no_donor_weights')

_spec = importlib.util.spec_from_file_location('_compare_native_model', native.__file__)
_wide = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_wide)
_wide.MATCHED_PARTICLE_FFN_HIDDEN_DIM = FFN_WIDTH


def layerwise_bias(model):
    old_pair = model.mod.pair_embed
    native_pairs = old_pair.reference
    for key, callback in list(model.mod.trimmer._forward_hooks.items()):
        if getattr(callback, '__self__', None) is old_pair:
            del model.mod.trimmer._forward_hooks[key]
    attentions = []
    for block in model.mod.blocks:
        block.attn = LayerwiseAttention(block.attn)
        attentions.append(block.attn)
    model.mod.pair_embed = LayerwisePairs(native_pairs, attentions, with_ev=True)
    model.mod.trimmer.register_forward_hook(model.mod.pair_embed.capture_mask)
    return model


def state_branches(model):
    return [getattr(block.attn, 'attention', block.attn).state for block in model.mod.blocks]


def validate_capacity_model(model):
    if (sum(p.numel() for p in model.parameters()) != COUNTS['part_state_matched']
            or any(type(b.attn) is not nn.MultiheadAttention for b in model.mod.blocks)
            or any(b.ffn_dim != FFN_WIDTH for b in model.mod.blocks)
            or any(b.ffn_dim != 512 for b in model.mod.cls_blocks)):
        raise ValueError('Expected native shared-bias ParT with eight width-601 FFNs')
    return dict(parameters=COUNTS['part_state_matched'], target=COUNTS['state_layerwise'],
                delta=432, particle_ffn_width=FFN_WIDTH, class_ffn_width=512, pair_bias='shared')


def build_model(architecture, *, seed=101, trim=True, use_amp=False, optimized=True):
    if architecture in ('state_fast', 'state_layerwise'):
        model = state_model('state_fast', seed=seed, trim=trim, use_amp=use_amp, optimized=optimized)
        if architecture == 'state_layerwise':
            layerwise_bias(model)
        model.study_architecture = architecture
    elif architecture == 'part_state_matched':
        model = _wide.build_model('part_matched', trim=trim, use_amp=use_amp)
        validate_capacity_model(model)
    elif architecture in ('part', 'part_ev'):
        model = native.build_model(architecture, trim=trim, use_amp=use_amp)
    else:
        raise ValueError('Unknown full-comparison architecture')
    if sum(p.numel() for p in model.parameters()) != COUNTS[architecture]:
        raise ValueError('Full-comparison parameter contract changed')
    return model


def nonzero_equivalence(architecture, device):
    """Full-model parity with nonzero state, dropout, masks and checkpoint backward."""
    from part_ev_state_speed.protocol import SPEED_POLICY
    from part_ev_message_study.parity import compare
    fast = build_model(architecture, trim=False).to(device).train()
    slow = build_model(architecture, trim=False, optimized=False).to(device).train()
    with torch.no_grad():
        for branch in state_branches(fast):
            branch.output.normal_(std=.02)
    slow.load_state_dict(fast.state_dict())
    n, batch = 67, 2  # Cross both the old 16-query and optimized 64-query chunks.
    valid = torch.arange(n, device=device)[None] < torch.tensor([n, n-9], device=device)[:, None]
    vectors = torch.randn(batch, 4, n, device=device)
    vectors[:, 3] = vectors[:, :3].square().sum(1).sqrt() + 1
    inputs = [torch.zeros(batch, 2, n, device=device),
              torch.randn(batch, 17, n, device=device), vectors, valid[:, None]]
    outputs, grads = [], []
    for target in (slow, fast):
        torch.manual_seed(991)
        output = target(*inputs)
        output.square().mean().backward()
        outputs.append(output.detach())
        grads.append({k: p.grad.detach().clone() for k, p in target.named_parameters() if p.grad is not None})
    forward = compare(outputs[1], outputs[0])
    if not forward['passed'] or grads[0].keys() != grads[1].keys():
        raise ValueError('Nonzero full-model state forward/gradient coverage differs')
    for key in grads[0]:
        if not torch.isfinite(grads[0][key]).all() or not torch.isfinite(grads[1][key]).all():
            raise ValueError('Nonfinite state integration gradient')
        torch.testing.assert_close(grads[1][key], grads[0][key],
            rtol=SPEED_POLICY['gradient_rtol'], atol=SPEED_POLICY['gradient_atol'])
    state_keys = [k for k in grads[1] if '.state.' in k]
    if len(state_keys) != 32 or any(not grads[1][k].count_nonzero() for k in state_keys):
        raise ValueError('All eight state branches must receive nonzero gradients')
    if architecture == 'state_layerwise':
        for i in range(8):
            keys = [k for k in grads[1] if f'.embed.heads.{i}.' in k]
            if not keys or not any(grads[1][k].count_nonzero() for k in keys):
                raise ValueError('Missing independent layerwise-bias gradients')
    return dict(passed=True, architecture=architecture, full_model=True,
                forward=forward, tensors=len(grads[0]), particles=n, state_tensors=len(state_keys),
                checkpoint_backward=True, max_absolute_error=max(
                    float((grads[1][k]-grads[0][k]).abs().max()) for k in grads[0]))
