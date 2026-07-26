"""Directional transverse-momentum relation family."""

from __future__ import annotations

from typing import Any, Mapping

from .contracts import require_sha256, with_content_hash
from .normalization import (
    FeaturewiseNormalizer,
    GLOBAL_EPSILON,
    PT_RAW_FEATURE_NAMES,
    PT_ROBUST_FEATURE_NAMES,
)
from .pair_base import require_torch

try:
    import torch as _torch
except ImportError:  # pragma: no cover - environment dependent
    _torch = None


PT_RELATION_CONTRACT = "relational_part_pt_relation_v1"
PT_ENCODED_DIMENSION = 8


if _torch is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass
else:
    _ModuleBase = _torch.nn.Module


def valid_pair_mask(mask: Any) -> Any:
    torch = require_torch()
    if not isinstance(mask, torch.Tensor) or mask.ndim != 3 or int(mask.shape[1]) != 1:
        raise ValueError("particle mask must have shape [batch,1,particles]")
    valid = mask.bool()
    return valid.unsqueeze(-1) & valid.unsqueeze(-2)


def average_tied_descending_rank(pt: Any, mask: Any) -> Any:
    """Return permutation-equivariant zero-based average ranks in ``[0,1]``."""

    torch = require_torch()
    if not isinstance(pt, torch.Tensor) or pt.ndim != 2:
        raise ValueError("pt must have shape [batch,particles]")
    if tuple(mask.shape) != (int(pt.shape[0]), 1, int(pt.shape[1])):
        raise ValueError("mask and pt shapes disagree")
    valid = mask[:, 0].bool()
    query = pt.unsqueeze(-1)
    other = pt.unsqueeze(-2)
    both = valid.unsqueeze(-1) & valid.unsqueeze(-2)
    greater = ((other > query) & both).sum(dim=-1)
    equal = ((other == query) & both).sum(dim=-1)
    denominator = (valid.sum(dim=-1) - 1).clamp_min(1)
    rank = (
        greater.to(dtype=pt.dtype)
        + 0.5 * (equal.to(dtype=pt.dtype) - 1.0)
    ) / denominator.to(dtype=pt.dtype).unsqueeze(-1)
    rank = torch.where(
        (valid.sum(dim=-1) > 1).unsqueeze(-1) & valid,
        rank,
        torch.zeros_like(rank),
    )
    return rank


def build_pt_raw_features(lorentz_vectors: Any, mask: Any) -> Any:
    """Build the locked ``[B,10,query,context]`` PT tensor."""

    torch = require_torch()
    if (
        not isinstance(lorentz_vectors, torch.Tensor)
        or lorentz_vectors.ndim != 3
        or int(lorentz_vectors.shape[1]) != 4
    ):
        raise ValueError("lorentz_vectors must have shape [batch,4,particles]")
    if tuple(mask.shape) != (
        int(lorentz_vectors.shape[0]),
        1,
        int(lorentz_vectors.shape[2]),
    ):
        raise ValueError("mask and lorentz-vector shapes disagree")
    if not bool(torch.isfinite(lorentz_vectors).all()):
        raise FloatingPointError("lorentz_vectors contain NaN or infinity")
    valid = mask[:, 0].bool()
    work = torch.hypot(
        lorentz_vectors[:, 0].float(),
        lorentz_vectors[:, 1].float(),
    )
    work = work.masked_fill(~valid, 0.0)
    total = work.sum(dim=-1, keepdim=True)
    fraction = work / (total + GLOBAL_EPSILON)
    log_fraction = torch.log(
        (work + GLOBAL_EPSILON) / (total + GLOBAL_EPSILON)
    )
    rank = average_tied_descending_rank(work, mask)

    f_i = fraction.unsqueeze(-1)
    f_j = fraction.unsqueeze(-2)
    x_i = log_fraction.unsqueeze(-1)
    x_j = log_fraction.unsqueeze(-2)
    pt_i = work.unsqueeze(-1)
    pt_j = work.unsqueeze(-2)
    r_i = rank.unsqueeze(-1)
    r_j = rank.unsqueeze(-2)
    pair_log = torch.log(
        (pt_i + pt_j) / (total.unsqueeze(-1) + GLOBAL_EPSILON)
        + GLOBAL_EPSILON
    )
    asymmetry = (pt_j - pt_i) / (pt_j + pt_i + GLOBAL_EPSILON)
    batch, length = work.shape
    raw = torch.stack(
        (
            f_i.expand(batch, length, length),
            f_j.expand(batch, length, length),
            x_i.expand(batch, length, length),
            x_j.expand(batch, length, length),
            x_j.expand(batch, length, length)
            - x_i.expand(batch, length, length),
            pair_log,
            asymmetry,
            r_i.expand(batch, length, length),
            r_j.expand(batch, length, length),
            r_j.expand(batch, length, length)
            - r_i.expand(batch, length, length),
        ),
        dim=1,
    )
    raw = raw.to(dtype=lorentz_vectors.dtype)
    return raw.masked_fill(~valid_pair_mask(mask), 0.0)


class PTEncoder(_ModuleBase):
    raw_feature_names = PT_RAW_FEATURE_NAMES
    robust_feature_names = PT_ROBUST_FEATURE_NAMES
    encoded_dimension = PT_ENCODED_DIMENSION

    def __init__(self, normalization_artifact: Mapping[str, Any]) -> None:
        torch = require_torch()
        super().__init__()
        self.normalizer = FeaturewiseNormalizer(
            family_id="PT",
            raw_feature_names=self.raw_feature_names,
            robust_feature_names=self.robust_feature_names,
            artifact=normalization_artifact,
        )
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(len(self.raw_feature_names), 32),
            torch.nn.GELU(),
            torch.nn.RMSNorm(32, eps=GLOBAL_EPSILON),
            torch.nn.Linear(32, self.encoded_dimension),
        )

    def forward(
        self,
        lorentz_vectors: Any,
        mask: Any,
        *,
        return_details: bool = False,
    ) -> Any:
        pair_mask = valid_pair_mask(mask)
        raw = build_pt_raw_features(lorentz_vectors, mask)
        normalized = self.normalizer(raw, pair_mask)
        encoded = self.encoder(normalized.permute(0, 2, 3, 1))
        encoded = encoded.permute(0, 3, 1, 2).contiguous()
        encoded = encoded.masked_fill(~pair_mask, 0.0)
        if return_details:
            return {
                "raw": raw,
                "normalized": normalized,
                "encoded": encoded,
                "pair_mask": pair_mask,
            }
        return encoded

    def diagnostics(self, lorentz_vectors: Any, mask: Any) -> dict[str, Any]:
        torch = require_torch()
        raw = build_pt_raw_features(lorentz_vectors, mask)
        pair_mask = valid_pair_mask(mask)[:, 0]
        means = []
        for channel in range(len(self.raw_feature_names)):
            values = raw[:, channel].masked_select(pair_mask)
            means.append(
                0.0
                if int(values.numel()) == 0
                else float(values.mean().detach().cpu())
            )
        return {
            "valid_directed_pair_count": int(
                valid_pair_mask(mask).sum().detach().cpu()
            ),
            "raw_feature_names": list(self.raw_feature_names),
            "raw_feature_means": means,
            "epsilon": GLOBAL_EPSILON,
            "query_context_direction": "i_is_query_j_is_context",
            "finite": bool(torch.isfinite(raw).all()),
        }


def build_pt_relation_contract(
    *,
    relation_registry_sha256: str,
    relation_normalization_sha256: str,
) -> dict[str, Any]:
    return with_content_hash(
        {
            "contract": PT_RELATION_CONTRACT,
            "schema_version": 1,
            "relation_registry_sha256": require_sha256(
                relation_registry_sha256, name="relation_registry_sha256"
            ),
            "relation_normalization_sha256": require_sha256(
                relation_normalization_sha256,
                name="relation_normalization_sha256",
            ),
            "family_id": "PT",
            "raw_feature_names": list(PT_RAW_FEATURE_NAMES),
            "robust_feature_names": list(PT_ROBUST_FEATURE_NAMES),
            "fixed_scale_feature_names": [
                name
                for name in PT_RAW_FEATURE_NAMES
                if name not in PT_ROBUST_FEATURE_NAMES
            ],
            "raw_dimension": len(PT_RAW_FEATURE_NAMES),
            "encoded_dimension": PT_ENCODED_DIMENSION,
            "encoder": [
                "Linear(10,32)",
                "GELU",
                "RMSNorm(32,eps=1e-6)",
                "Linear(32,8)",
            ],
            "rank": "zero_based_average_tied_descending_divided_by_Nvalid_minus_1",
            "pair_domain": "all_ordered_valid_pairs_including_diagonal",
            "invalid_pair_policy": "zero_after_normalization_and_encoder",
            "dropout": 0.0,
        }
    )


__all__ = [
    "PT_ENCODED_DIMENSION",
    "PT_RELATION_CONTRACT",
    "PTEncoder",
    "average_tied_descending_rank",
    "build_pt_raw_features",
    "build_pt_relation_contract",
    "valid_pair_mask",
]
