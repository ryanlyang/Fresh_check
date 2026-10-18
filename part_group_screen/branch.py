"""Common group content, communication and readback for every assignment arm."""
import hashlib
from contextlib import contextmanager
import torch
from torch import nn
from .geometry import descriptors
from .grouping import fixed_assignment, random_assignment, SoftRouter


@contextmanager
def isolated_seed(seed, component):
    value = int.from_bytes(hashlib.sha256(f"group-v1:{seed}:{component}".encode()).digest()[:8], "little")
    with torch.random.fork_rng(devices=[]):
        # torch.manual_seed also reseeds CUDA, which a CPU-only fork would not
        # restore. Added modules are constructed on CPU; seed only its generator.
        torch.default_generator.manual_seed(value & ((1 << 63) - 1))
        yield


class GroupBranch(nn.Module):
    def __init__(self, architecture, seed, config):
        super().__init__()
        self.architecture, self.config = architecture, dict(config)
        self.collect_diagnostics = False
        self.last_diagnostics = None
        self.disabled = False
        d, heads, eps = config["width"], config["heads"], config["norm_epsilon"]
        with isolated_seed(seed, "common_branch"):
            self.hidden_projection = nn.Linear(d, d)
            self.physical = nn.Sequential(nn.Linear(7, config["physical_hidden"]), nn.GELU(),
                                          nn.Linear(config["physical_hidden"], d))
            self.token_norm = nn.LayerNorm(d, eps=eps)
            self.attn_norm = nn.LayerNorm(d, eps=eps)
            self.self_attention = nn.MultiheadAttention(d, heads, dropout=0., batch_first=True)
            self.ffn_norm = nn.LayerNorm(d, eps=eps)
            self.ffn = nn.Sequential(nn.Linear(d, config["group_ffn"]), nn.GELU(), nn.Linear(config["group_ffn"], d))
            self.query_norm, self.group_norm = nn.LayerNorm(d, eps=eps), nn.LayerNorm(d, eps=eps)
            self.readback = nn.MultiheadAttention(d, heads, dropout=0., batch_first=True)
            self.gamma = nn.Parameter(torch.tensor(config["gamma"], dtype=torch.float32))
        self.router = None
        if architecture in ("learned", "hybrid"):
            with isolated_seed(seed, "router"):
                self.router = SoftRouter(config, locality=config["locality"] if architecture == "hybrid" else 0.)

    def forward(self, hidden, vectors, valid, features):
        if self.disabled:
            return hidden
        if not bool(valid.any(1).all()):
            raise ValueError("Grouped attention requires at least one valid particle per jet")
        if self.router is not None:
            a, active = self.router(hidden, vectors, valid)
        elif self.architecture == "fixed":
            a, active = fixed_assignment(vectors, features, valid, self.config["groups"])
        elif self.architecture == "random":
            a, active = random_assignment(vectors, features, valid, self.config["groups"], self.config["random_seed"])
        else:
            raise ValueError("Unknown assignment architecture")
        n, group_vectors, physical = descriptors(a, vectors, valid, self.config["epsilon"])
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            pooled = torch.einsum("bnk,bnd->bkd", a.float(), hidden.float()) / (n[..., None] + self.config["epsilon"])
        tokens = self.token_norm(self.hidden_projection(pooled) + self.physical(physical))
        tokens = tokens.masked_fill(~active[..., None], 0)
        norm = self.attn_norm(tokens)
        tokens = tokens + self.self_attention(norm, norm, norm, key_padding_mask=~active, need_weights=False)[0]
        tokens = tokens + self.ffn(self.ffn_norm(tokens))
        groups = self.group_norm(tokens)
        correction = self.readback(self.query_norm(hidden), groups, groups,
                                   key_padding_mask=~active, need_weights=False)[0]
        correction = correction.masked_fill(~valid[..., None], 0).to(hidden.dtype)
        scaled = self.gamma * correction
        if self.collect_diagnostics:
            with torch.no_grad():
                probabilities = a.float().clamp_min(1e-12)
                entropy = -(a * probabilities.log()).sum(-1)
                shares = n / n.sum(1, keepdim=True).clamp_min(1e-6)
                unit = torch.nn.functional.normalize(tokens.float(), dim=-1)
                similarity = unit @ unit.transpose(1, 2)
                pairs = active[:, :, None] & active[:, None, :]
                pairs &= ~torch.eye(active.shape[1], dtype=torch.bool, device=active.device)[None]
                mean_similarity = similarity.masked_select(pairs).mean() if pairs.any() else similarity.new_zeros(())
                group_pt = torch.linalg.vector_norm(group_vectors[..., :2], dim=-1)
                jet_pt_sum = (torch.linalg.vector_norm(vectors.float()[..., :2], dim=-1) * valid).sum(1, keepdim=True)
                self.last_diagnostics = dict(
                    occupancy=n.detach().cpu().tolist(),
                    pt_share=(group_pt / jet_pt_sum.clamp_min(1e-6)).cpu().tolist(),
                    mean_slot_similarity=float(mean_similarity),
                    mean_assignment_entropy=float(entropy.masked_select(valid).mean()),
                    effective_groups=(1 / shares.square().sum(1).clamp_min(1e-6)).cpu().tolist(),
                    gamma=float(self.gamma), correction_to_backbone_norm=float(scaled.float().norm() / hidden.float().norm().clamp_min(1e-12)),
                    angular_width=physical[..., 6].expm1().detach().cpu().tolist(),
                    examples=dict(assignments=a[:4].detach().cpu().tolist(),
                                  vectors=group_vectors[:4].detach().cpu().tolist(),
                                  valid=valid[:4].detach().cpu().tolist()))
        return hidden + scaled
