"""Capacity control changes only particle FFN width; historical models stay untouched."""
import importlib.util
import torch
from torch import nn
from part_ev_fullscale import model as native
from part_ev_state_speed.model import build_model as state_model
from part_ev_message_study.model import diagnostic_mode, diagnostics

FFN_WIDTH = 599
COUNTS = dict(part=2143486, part_ev=2209022, state_fast=2323710,
              part_state_matched=2323750)
CONFIG = dict(state_blocks=list(range(8)), state_width=32, query_chunk=64,
              shared_pair_normalization=True, checkpoint_chunks=True,
              matched_particle_ffn_width=FFN_WIDTH, class_ffn_width=512,
              matching_target=COUNTS['state_fast'], matching_delta=40,
              pair_bias='shared', initialization='fresh_seeded_no_donor_weights')

# The established FFN construction with an isolated width constant. Do not
# mutate the historical module: pending old jobs and compatibility hashes matter.
_spec = importlib.util.spec_from_file_location('_capacity_native_model', native.__file__)
_wide = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_wide)
_wide.MATCHED_PARTICLE_FFN_HIDDEN_DIM = FFN_WIDTH


def validate_capacity_model(model):
    if (sum(p.numel() for p in model.parameters()) != COUNTS['part_state_matched']
            or any(type(b.attn) is not nn.MultiheadAttention for b in model.mod.blocks)
            or any(b.ffn_dim != FFN_WIDTH for b in model.mod.blocks)
            or any(b.ffn_dim != 512 for b in model.mod.cls_blocks)):
        raise ValueError('Expected native shared-bias ParT with eight width-599 FFNs')
    return dict(parameters=COUNTS['part_state_matched'], target=COUNTS['state_fast'],
                delta=40, particle_ffn_width=FFN_WIDTH, class_ffn_width=512)


def build_model(architecture, *, seed=101, trim=True, use_amp=False):
    if architecture == 'state_fast':
        model = state_model('state_fast', seed=seed, trim=trim, use_amp=use_amp)
    elif architecture == 'part_state_matched':
        model = _wide.build_model('part_matched', trim=trim, use_amp=use_amp)
        validate_capacity_model(model)
    elif architecture in ('part', 'part_ev'):
        model = native.build_model(architecture, trim=trim, use_amp=use_amp)
    else:
        raise ValueError('Unknown capacity-study architecture')
    if sum(p.numel() for p in model.parameters()) != COUNTS[architecture]:
        raise ValueError('Parameter contract changed')
    return model
