"""The historical PT(8), TRACK(12), REGION(12) learned descriptors."""
import torch
from torch import nn
from teacher_logit_reco.relational_part.normalization import (PT_RAW_FEATURE_NAMES,
    PT_ROBUST_FEATURE_NAMES, TRACK_NODE_CONTINUOUS_NAMES, TRACK_COMPATIBILITY_FEATURE_NAMES,
    TRACK_COMPATIBILITY_ROBUST_NAMES)
from teacher_logit_reco.relational_part.relation_pt import build_pt_raw_features, valid_pair_mask
from teacher_logit_reco.relational_part.relation_track import build_track_compatibility
from teacher_logit_reco.relational_part.relation_region import REGION_RAW_FEATURE_NAMES, REGION_ROBUST_FEATURE_NAMES
from .region import trees_for, region_features

CHANNELS = dict(PT=(PT_RAW_FEATURE_NAMES, PT_ROBUST_FEATURE_NAMES),
    TRACK_NODE=(TRACK_NODE_CONTINUOUS_NAMES, TRACK_NODE_CONTINUOUS_NAMES),
    TRACK_PAIR=(TRACK_COMPATIBILITY_FEATURE_NAMES, TRACK_COMPATIBILITY_ROBUST_NAMES),
    REGION=(REGION_RAW_FEATURE_NAMES, REGION_ROBUST_FEATURE_NAMES))


class Normalizer(nn.Module):
    def __init__(self, family, normalization):
        super().__init__()
        names, robust = CHANNELS[family]
        row = normalization['families'][family]
        if row['names'] != list(names) or row['robust'] != list(robust):
            raise ValueError(f'Feature schema differs: {family}')
        center, scale = torch.tensor(row['center']), torch.tensor(row['scale'])
        if center.shape != (len(names),) or scale.shape != center.shape or not torch.isfinite(center).all() or not torch.isfinite(scale).all() or (scale < 1e-6).any():
            raise ValueError('Invalid normalization constants')
        self.family = family
        self.register_buffer('center', center.float())
        self.register_buffer('scale', scale.float())
        self.register_buffer('robust', torch.tensor([n in robust for n in names]))

    def forward(self, raw, mask):
        raw = torch.nan_to_num(raw.float(), nan=0., posinf=0., neginf=0.)
        shape = (1, -1) + (1,) * (raw.ndim-2)
        out = torch.where(self.robust.view(shape),
            ((raw-self.center.view(shape))/self.scale.view(shape)).clamp(-8, 8), raw)
        if self.family == 'REGION':
            diagonal = torch.eye(raw.shape[-1], dtype=torch.bool, device=raw.device)[None, None]
            out[:, 4:8] = out[:, 4:8].masked_fill(diagonal, 0)
        return out.masked_fill(~mask.bool(), 0)


def mlp(a, b, c):
    return nn.Sequential(nn.Linear(a, b), nn.GELU(), nn.RMSNorm(b, eps=1e-6), nn.Linear(b, c))


def raw_features(vectors, raw, mask, floors, backend):
    with torch.no_grad(), torch.autocast(device_type=vectors.device.type, enabled=False):
        track = build_track_compatibility(raw.float(), mask,
            d0_uncertainty_floor=floors['d0'], dz_uncertainty_floor=floors['dz'])
        region = region_features(trees_for(vectors, raw, mask, backend), raw.float(), mask)
        return dict(PT=build_pt_raw_features(vectors.float(), mask),
            TRACK_NODE=track['continuous'], TRACK_PAIR=track['compatibility'], REGION=region), track


class Relations(nn.Module):
    def __init__(self, normalization, backend):
        super().__init__()
        self.normalizers = nn.ModuleDict({k: Normalizer(k, normalization) for k in CHANNELS})
        self.pt = mlp(10, 32, 8)
        self.track_node = mlp(7, 32, 16)
        self.track_pair = mlp(81, 48, 12)
        self.region = mlp(41, 32, 12)
        self.floors = dict(normalization['floors'])
        if set(self.floors) != {'d0', 'dz'} or any(not 1e-6 <= x < float('inf') for x in self.floors.values()):
            raise ValueError('Invalid training-only uncertainty floors')
        object.__setattr__(self, 'backend', backend)

    def forward(self, vectors, raw, mask):
        values, track = raw_features(vectors, raw, mask, self.floors, self.backend)
        pair_mask = valid_pair_mask(mask)
        pt = self.normalizers['PT'](values['PT'], pair_mask)
        pt = self.pt(pt.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        node = self.normalizers['TRACK_NODE'](values['TRACK_NODE'], track['track_valid'][:, None])
        node = torch.cat((node, track['track_valid'][:, None].float()), 1)
        node = self.track_node(node.transpose(1, 2)).transpose(1, 2).masked_fill(~mask.bool(), 0)
        n = node.shape[-1]
        left, right = node[:, :, :, None].expand(-1, -1, -1, n), node[:, :, None, :].expand(-1, -1, n, -1)
        compatibility = self.normalizers['TRACK_PAIR'](values['TRACK_PAIR'], track['both_tracks_valid'])
        pair = torch.cat((left, right, right-left, left*right, track['validity_one_hot'], compatibility), 1)
        pair = self.track_pair(pair.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        region = self.normalizers['REGION'](values['REGION'], pair_mask)
        region = self.region(region.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return torch.cat((pt, pair, region), 1).masked_fill(~pair_mask, 0)
