"""Native ParT plus one post-block-four grouping branch; no donor weight load."""
from __future__ import annotations
import copy
import torch
from part_ev_fullscale.model import ParticleTransformerWrapper
from .config import CONFIG, ARCHITECTURES
from .branch import GroupBranch


class GroupedParT(ParticleTransformerWrapper):
    def __init__(self, architecture, seed=101, *, trim=True, use_amp=False, config=None):
        if architecture not in ARCHITECTURES:
            raise ValueError("Unknown grouped architecture")
        super().__init__("part", trim=trim, use_amp=use_amp)
        self.group_config = copy.deepcopy(CONFIG if config is None else config)
        self.group_branch = GroupBranch(architecture, seed, self.group_config)
        self.group_architecture = architecture
        self._group_context = None
        self.mod.trimmer.register_forward_hook(self._capture)
        self.mod.blocks[self.group_config["insertion_block"] - 1].register_forward_hook(self._insert)

    def _capture(self, module, inputs, outputs):
        features, vectors, mask, _ = outputs
        self._group_context = (features.transpose(1, 2), vectors.transpose(1, 2), mask[:, 0].bool())

    def _insert(self, module, inputs, output):
        if self._group_context is None:
            raise RuntimeError("Missing current native-trimmer context")
        features, vectors, valid = self._group_context
        if output.shape[:2] != (valid.shape[1], valid.shape[0]):
            raise ValueError("Grouped hidden states and native-trimmed inputs are misaligned")
        return self.group_branch(output.transpose(0, 1), vectors, valid, features).transpose(0, 1)

    def forward(self, points, features, vectors, mask):
        try:
            return super().forward(points, features, vectors, mask)
        finally:
            self._group_context = None


def build_model(architecture, *, seed=101, trim=True, use_amp=False):
    return GroupedParT(architecture, seed, trim=trim, use_amp=use_amp)
