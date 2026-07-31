"""Registered HOSD global, particle, and pair prediction heads."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

from .contracts import TARGET_HEAD_CONTRACT, with_content_hash

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def target_head_registry() -> dict[str, Any]:
    return with_content_hash(
        {
            "contract": TARGET_HEAD_CONTRACT,
            "schema_version": 2,
            "global": {
                "queries": 4,
                "dimension": 128,
                "attention_heads": 8,
                "trunk": ["RMSNorm", "masked_cross_attention", "Linear_512_256", "GELU", "RMSNorm"],
                "heteroscedastic_log_variance_clip": [-8.0, 5.0],
                "heteroscedastic_output_scope": (
                    "registered_continuous_components_only"
                ),
            },
            "particle": ["RMSNorm", "Linear_d_d", "GELU", "Linear_d_target"],
            "pair": {
                "hidden_width": 128,
                "directional_inputs": ["h_i", "h_j", "h_i-h_j", "h_i*h_j"],
                "symmetric_inputs": ["h_i+h_j", "abs(h_i-h_j)", "h_i*h_j"],
                "diagonal_masked": True,
                "validation": "all_pairs_streamed",
            },
            "pair_sampling": {
                "domain": "hosd_pair_sample_v1",
                "binary_positive_cap": 512,
                "binary_negative_cap": 512,
                "continuous_cap": 1024,
                "rng_library_used": False,
            },
        }
    )


class GlobalTargetHead(torch.nn.Module if torch is not None else object):
    def __init__(
        self,
        target_dimension: int,
        *,
        input_dimension: int = 128,
        availability_groups: int = 1,
        heteroscedastic: bool = False,
        heteroscedastic_components: Sequence[bool] | None = None,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for HOSD heads")
        super().__init__()
        if target_dimension <= 0 or availability_groups <= 0:
            raise ValueError("target and availability dimensions must be positive")
        self.target_dimension = int(target_dimension)
        self.availability_groups = int(availability_groups)
        self.heteroscedastic = bool(heteroscedastic)
        component_mask = (
            tuple(True for _ in range(target_dimension))
            if heteroscedastic_components is None
            else tuple(bool(value) for value in heteroscedastic_components)
        )
        if len(component_mask) != target_dimension:
            raise ValueError("heteroscedastic component mask differs")
        self.register_buffer(
            "heteroscedastic_component_mask",
            torch.as_tensor(component_mask, dtype=torch.bool),
            persistent=True,
        )
        self.input_projection = (
            torch.nn.Identity()
            if input_dimension == 128
            else torch.nn.Linear(input_dimension, 128)
        )
        self.input_norm = torch.nn.RMSNorm(128)
        self.queries = torch.nn.Parameter(torch.empty(4, 128))
        torch.nn.init.normal_(self.queries, std=0.02)
        self.cross_attention = torch.nn.MultiheadAttention(
            128, 8, dropout=0.0, batch_first=True
        )
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.GELU(),
            torch.nn.RMSNorm(256),
        )
        output_dimension = target_dimension + (
            sum(component_mask) if heteroscedastic else 0
        )
        self.output = torch.nn.Linear(256, output_dimension)
        self.availability = torch.nn.Linear(256, availability_groups)

    def forward(self, particle_states: Any, particle_mask: Any) -> dict[str, Any]:
        if particle_states.ndim != 3 or particle_mask.ndim != 2:
            raise ValueError("global head expects [B,N,D] states and [B,N] mask")
        if tuple(particle_states.shape[:2]) != tuple(particle_mask.shape):
            raise ValueError("global-head state and mask shapes differ")
        if bool((particle_mask.bool().sum(dim=1) == 0).any()):
            raise ValueError("global head cannot attend to an empty event")
        x = self.input_norm(self.input_projection(particle_states))
        query = self.queries.unsqueeze(0).expand(x.shape[0], -1, -1)
        pooled, _ = self.cross_attention(
            query, x, x, key_padding_mask=~particle_mask.bool(), need_weights=False
        )
        hidden = self.trunk(pooled.reshape(x.shape[0], 512))
        raw = self.output(hidden)
        availability_logits = self.availability(hidden)
        if self.heteroscedastic:
            mean = raw[:, : self.target_dimension]
            packed_log_variance = raw[:, self.target_dimension :]
            log_variance = torch.zeros_like(mean)
            log_variance[:, self.heteroscedastic_component_mask] = (
                packed_log_variance
            )
            return {
                "value": mean,
                "mean": mean,
                "log_variance": log_variance.clamp(-8.0, 5.0),
                "heteroscedastic_component_mask": (
                    self.heteroscedastic_component_mask
                ),
                "availability_logits": availability_logits,
            }
        return {"value": raw, "availability_logits": availability_logits}


class ParticleTargetHead(torch.nn.Module if torch is not None else object):
    def __init__(self, input_dimension: int, target_dimension: int) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for HOSD heads")
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.RMSNorm(input_dimension),
            torch.nn.Linear(input_dimension, input_dimension),
            torch.nn.GELU(),
            torch.nn.Linear(input_dimension, target_dimension),
        )

    def forward(self, states: Any, mask: Any) -> Any:
        result = self.network(states)
        return result.masked_fill(~mask.bool().unsqueeze(-1), 0)


class PairTargetHead(torch.nn.Module if torch is not None else object):
    def __init__(
        self, input_dimension: int, target_dimension: int, *, symmetric: bool
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for HOSD heads")
        super().__init__()
        self.symmetric = bool(symmetric)
        self.norm = torch.nn.RMSNorm(input_dimension)
        factor = 3 if symmetric else 4
        self.network = torch.nn.Sequential(
            torch.nn.Linear(factor * input_dimension, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, target_dimension),
        )

    def forward(self, states: Any, mask: Any) -> tuple[Any, Any]:
        x = self.norm(states)
        hi = x.unsqueeze(2)
        hj = x.unsqueeze(1)
        if self.symmetric:
            pair = torch.cat((hi + hj, (hi - hj).abs(), hi * hj), dim=-1)
        else:
            pair = torch.cat((hi.expand(-1, -1, x.shape[1], -1), hj.expand(-1, x.shape[1], -1, -1), hi - hj, hi * hj), dim=-1)
        output = self.network(pair)
        if self.symmetric:
            output = 0.5 * (output + output.transpose(1, 2))
        valid = mask.bool()
        pair_mask = valid.unsqueeze(2) & valid.unsqueeze(1)
        eye = torch.eye(valid.shape[1], dtype=torch.bool, device=valid.device)
        pair_mask = pair_mask & ~eye.unsqueeze(0)
        return output.masked_fill(~pair_mask.unsqueeze(-1), 0), pair_mask

    def forward_pairs(
        self,
        states: Any,
        event_indices: Any,
        left_indices: Any,
        right_indices: Any,
    ) -> Any:
        """Evaluate only declared pairs for deterministic sampled training."""

        if not (
            event_indices.ndim == left_indices.ndim == right_indices.ndim == 1
            and len(event_indices) == len(left_indices) == len(right_indices)
        ):
            raise ValueError("sampled pair indices must be equal-length vectors")
        x = self.norm(states)
        hi = x[event_indices.long(), left_indices.long()]
        hj = x[event_indices.long(), right_indices.long()]
        if self.symmetric:
            pair = torch.cat((hi + hj, (hi - hj).abs(), hi * hj), dim=-1)
        else:
            pair = torch.cat((hi, hj, hi - hj, hi * hj), dim=-1)
        return self.network(pair)


def deterministic_pair_indices(
    *,
    epoch: int,
    identity: str,
    target_id: str,
    pair_ids: Sequence[str],
    positive: Sequence[bool] | None = None,
) -> tuple[int, ...]:
    """Return exact SHA256-smallest training indices without an RNG."""

    if int(epoch) <= 0:
        raise ValueError("pair-sampling epoch is one-based")
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("canonical pair IDs must be unique")
    if positive is not None and len(positive) != len(pair_ids):
        raise ValueError("pair strata length differs")

    def ranked(indices: Sequence[int], cap: int) -> list[int]:
        rows = []
        for index in indices:
            payload = (
                f"hosd_pair_sample_v1||{int(epoch)}||{identity}||"
                f"{target_id}||{pair_ids[index]}"
            ).encode("utf-8")
            rows.append((hashlib.sha256(payload).hexdigest(), pair_ids[index], index))
        rows.sort()
        return [index for _, _, index in rows[:cap]]

    if positive is None:
        return tuple(ranked(range(len(pair_ids)), 1024))
    positives = [index for index, value in enumerate(positive) if bool(value)]
    negatives = [index for index, value in enumerate(positive) if not bool(value)]
    return tuple(ranked(positives, 512) + ranked(negatives, 512))


def heteroscedastic_nll(
    mean: Any, log_variance: Any, target: Any, mask: Any
) -> Any:
    clipped = log_variance.clamp(-8.0, 5.0)
    value = 0.5 * (torch.exp(-clipped) * (target - mean).square() + clipped)
    selected = value.masked_select(mask.bool())
    if selected.numel() == 0:
        return value.sum() * 0
    return selected.mean()


__all__ = [
    "GlobalTargetHead",
    "PairTargetHead",
    "ParticleTargetHead",
    "deterministic_pair_indices",
    "heteroscedastic_nll",
    "target_head_registry",
]
