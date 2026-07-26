"""Directional PID and charge relation families."""

from __future__ import annotations

import math
from typing import Any, Mapping

from .contracts import require_sha256, with_content_hash
from .normalization import (
    CHARGE_RAW_FEATURE_NAMES,
    CHARGE_ROBUST_FEATURE_NAMES,
    FeaturewiseNormalizer,
    GLOBAL_EPSILON,
)
from .pair_base import require_torch
from .relation_pt import valid_pair_mask

try:
    import torch as _torch
except ImportError:  # pragma: no cover - environment dependent
    _torch = None


PID_CHARGE_RELATION_CONTRACT = "relational_part_pid_charge_relations_v1"
PID_CATEGORY_NAMES = (
    "charged_hadron",
    "neutral_hadron",
    "photon",
    "electron",
    "muon",
    "unknown",
)
PID_BINARY_TOLERANCE = GLOBAL_EPSILON
PID_THRESHOLD = 0.5
PID_ENCODED_DIMENSION = 8
CHARGE_STATES = (-1, 0, 1)
CHARGE_INTEGER_TOLERANCE = GLOBAL_EPSILON
CHARGE_ENCODED_DIMENSION = 6


if _torch is None:  # pragma: no cover - environment dependent
    class _ModuleBase:
        pass
else:
    _ModuleBase = _torch.nn.Module


def audit_pid_flags(pid_flags: Any, mask: Any) -> dict[str, Any]:
    torch = require_torch()
    if (
        not isinstance(pid_flags, torch.Tensor)
        or pid_flags.ndim != 3
        or int(pid_flags.shape[1]) != 5
    ):
        raise ValueError("pid_flags must have shape [batch,5,particles]")
    if tuple(mask.shape) != (
        int(pid_flags.shape[0]),
        1,
        int(pid_flags.shape[2]),
    ):
        raise ValueError("PID flags and mask shapes disagree")
    valid = mask[:, 0].bool()
    finite = torch.isfinite(pid_flags)
    binary_distance = torch.minimum(pid_flags.abs(), (pid_flags - 1.0).abs())
    invalid_binary = valid.unsqueeze(1) & (
        (~finite) | (binary_distance > PID_BINARY_TOLERANCE)
    )
    selected_count = (pid_flags >= PID_THRESHOLD).sum(dim=1)
    zero_hot = valid & (selected_count == 0)
    multi_hot = valid & (selected_count > 1)
    return {
        "valid_particle_count": int(valid.sum().detach().cpu()),
        "zero_hot_count": int(zero_hot.sum().detach().cpu()),
        "multi_hot_count": int(multi_hot.sum().detach().cpu()),
        "invalid_binary_value_count": int(invalid_binary.sum().detach().cpu()),
    }


def pid_categories(
    pid_flags: Any,
    mask: Any,
    *,
    fail_on_multi_hot: bool = True,
) -> Any:
    torch = require_torch()
    audit = audit_pid_flags(pid_flags, mask)
    if audit["invalid_binary_value_count"]:
        raise ValueError("valid PID flags must be finite and within 1e-6 of zero or one")
    if fail_on_multi_hot and audit["multi_hot_count"]:
        raise ValueError("multi-hot PID state is forbidden")
    valid = mask[:, 0].bool()
    safe = torch.nan_to_num(pid_flags, nan=0.0, posinf=0.0, neginf=0.0)
    selected = safe >= PID_THRESHOLD
    selected_count = selected.sum(dim=1)
    category = selected.to(dtype=torch.int64).argmax(dim=1)
    category = torch.where(
        (selected_count == 1) & valid,
        category,
        torch.full_like(category, 5),
    )
    return category


class PIDEncoder(_ModuleBase):
    raw_feature_names = ("query_pid", "context_pid", "directed_pid_pair")
    encoded_dimension = PID_ENCODED_DIMENSION

    def __init__(self) -> None:
        torch = require_torch()
        super().__init__()
        self.query_embedding = torch.nn.Embedding(6, self.encoded_dimension)
        self.context_embedding = torch.nn.Embedding(6, self.encoded_dimension)
        self.pair_embedding = torch.nn.Embedding(36, self.encoded_dimension)

    def forward(
        self,
        pid_flags: Any,
        mask: Any,
        *,
        return_details: bool = False,
    ) -> Any:
        categories = pid_categories(pid_flags, mask)
        query = categories.unsqueeze(-1)
        context = categories.unsqueeze(-2)
        pair_index = query * 6 + context
        pair_mask = valid_pair_mask(mask).permute(0, 2, 3, 1)
        query_encoded = self.query_embedding(query).masked_fill(~pair_mask, 0.0)
        context_encoded = self.context_embedding(context).masked_fill(
            ~pair_mask, 0.0
        )
        pair_encoded = self.pair_embedding(pair_index).masked_fill(
            ~pair_mask, 0.0
        )
        encoded = _torch.tanh(
            (query_encoded + context_encoded + pair_encoded)
            / math.sqrt(3.0)
        )
        encoded = encoded.permute(0, 3, 1, 2).contiguous()
        encoded = encoded.masked_fill(
            ~pair_mask.permute(0, 3, 1, 2), 0.0
        )
        if return_details:
            return {
                "categories": categories,
                "pair_indices": pair_index,
                "encoded": encoded,
                "pair_mask": pair_mask.permute(0, 3, 1, 2),
            }
        return encoded

    def diagnostics(self, pid_flags: Any, mask: Any) -> dict[str, Any]:
        torch = require_torch()
        audit = audit_pid_flags(pid_flags, mask)
        categories = pid_categories(pid_flags, mask, fail_on_multi_hot=False)
        valid = mask[:, 0].bool()
        pair_mask = valid_pair_mask(mask)[:, 0]
        pair_index = categories.unsqueeze(-1) * 6 + categories.unsqueeze(-2)
        category_counts = torch.bincount(
            categories.masked_select(valid), minlength=6
        )
        pair_counts = torch.bincount(
            pair_index.masked_select(pair_mask), minlength=36
        )
        pair_count_list = [int(value) for value in pair_counts.cpu()]
        return {
            **audit,
            "category_order": list(PID_CATEGORY_NAMES),
            "category_counts": [int(value) for value in category_counts.cpu()],
            "directed_pair_counts": pair_count_list,
            "rare_state_counts": {
                "electron_query_pairs": sum(pair_count_list[18:24]),
                "electron_context_pairs": sum(
                    pair_count_list[index] for index in range(3, 36, 6)
                ),
                "muon_query_pairs": sum(pair_count_list[24:30]),
                "muon_context_pairs": sum(
                    pair_count_list[index] for index in range(4, 36, 6)
                ),
            },
            "pair_embedding_norms": [
                float(value)
                for value in self.pair_embedding.weight.detach().norm(dim=1).cpu()
            ],
        }


def quantize_charge(charge: Any, mask: Any) -> tuple[Any, Any]:
    torch = require_torch()
    if not isinstance(charge, torch.Tensor) or charge.ndim != 2:
        raise ValueError("charge must have shape [batch,particles]")
    if tuple(mask.shape) != (int(charge.shape[0]), 1, int(charge.shape[1])):
        raise ValueError("charge and mask shapes disagree")
    valid = mask[:, 0].bool()
    states = charge.new_tensor(CHARGE_STATES)
    distances = (charge.unsqueeze(-1) - states).abs()
    minimum, index = distances.min(dim=-1)
    invalid = valid & ((~torch.isfinite(charge)) | (minimum > CHARGE_INTEGER_TOLERANCE))
    if bool(invalid.any()):
        raise ValueError("valid charges must be within 1e-6 of -1, 0, or +1")
    index = torch.where(valid, index, torch.ones_like(index))
    quantized = states[index]
    return quantized, index


def build_charge_raw_features(charge: Any, mask: Any) -> tuple[Any, Any]:
    quantized, state = quantize_charge(charge, mask)
    q_i = quantized.unsqueeze(-1)
    q_j = quantized.unsqueeze(-2)
    batch, length = quantized.shape
    q_i_full = q_i.expand(batch, length, length)
    q_j_full = q_j.expand(batch, length, length)
    product = q_i_full * q_j_full
    half_difference = (q_i_full - q_j_full).abs() / 2.0
    charged_i = q_i_full != 0
    charged_j = q_j_full != 0
    raw = _torch.stack(
        (
            q_i_full,
            q_j_full,
            product,
            half_difference,
            (~charged_i & ~charged_j).to(dtype=charge.dtype),
            (charged_i ^ charged_j).to(dtype=charge.dtype),
            (charged_i & charged_j & (product > 0)).to(dtype=charge.dtype),
            (charged_i & charged_j & (product < 0)).to(dtype=charge.dtype),
        ),
        dim=1,
    )
    return raw.masked_fill(~valid_pair_mask(mask), 0.0), state


class ChargeEncoder(_ModuleBase):
    raw_feature_names = CHARGE_RAW_FEATURE_NAMES
    robust_feature_names = CHARGE_ROBUST_FEATURE_NAMES
    encoded_dimension = CHARGE_ENCODED_DIMENSION

    def __init__(self, normalization_artifact: Mapping[str, Any]) -> None:
        torch = require_torch()
        super().__init__()
        self.normalizer = FeaturewiseNormalizer(
            family_id="CHARGE",
            raw_feature_names=self.raw_feature_names,
            robust_feature_names=self.robust_feature_names,
            artifact=normalization_artifact,
        )
        self.pair_embedding = torch.nn.Embedding(9, 4)
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(12, 32),
            torch.nn.GELU(),
            torch.nn.RMSNorm(32, eps=GLOBAL_EPSILON),
            torch.nn.Linear(32, self.encoded_dimension),
        )

    def forward(
        self,
        charge: Any,
        mask: Any,
        *,
        return_details: bool = False,
    ) -> Any:
        pair_mask = valid_pair_mask(mask)
        raw, state = build_charge_raw_features(charge, mask)
        normalized = self.normalizer(raw, pair_mask)
        pair_index = state.unsqueeze(-1) * 3 + state.unsqueeze(-2)
        categorical = self.pair_embedding(pair_index).permute(0, 3, 1, 2)
        categorical = categorical.masked_fill(~pair_mask, 0.0)
        combined = _torch.cat((normalized, categorical), dim=1)
        encoded = self.encoder(combined.permute(0, 2, 3, 1))
        encoded = encoded.permute(0, 3, 1, 2).contiguous()
        encoded = encoded.masked_fill(~pair_mask, 0.0)
        if return_details:
            return {
                "raw": raw,
                "normalized": normalized,
                "charge_states": state,
                "pair_indices": pair_index,
                "family_encoder_input": combined,
                "encoded": encoded,
                "pair_mask": pair_mask,
            }
        return encoded

    def diagnostics(self, charge: Any, mask: Any) -> dict[str, Any]:
        torch = require_torch()
        _, state = quantize_charge(charge, mask)
        valid = mask[:, 0].bool()
        pair_mask = valid_pair_mask(mask)[:, 0]
        pair_index = state.unsqueeze(-1) * 3 + state.unsqueeze(-2)
        return {
            "charge_state_order": list(CHARGE_STATES),
            "charge_state_counts": [
                int(value)
                for value in torch.bincount(
                    state.masked_select(valid), minlength=3
                ).cpu()
            ],
            "directed_pair_counts": [
                int(value)
                for value in torch.bincount(
                    pair_index.masked_select(pair_mask), minlength=9
                ).cpu()
            ],
            "pair_embedding_norms": [
                float(value)
                for value in self.pair_embedding.weight.detach().norm(dim=1).cpu()
            ],
        }


def build_pid_charge_relation_contract(
    *,
    relation_registry_sha256: str,
    relation_normalization_sha256: str,
) -> dict[str, Any]:
    return with_content_hash(
        {
            "contract": PID_CHARGE_RELATION_CONTRACT,
            "schema_version": 1,
            "relation_registry_sha256": require_sha256(
                relation_registry_sha256, name="relation_registry_sha256"
            ),
            "relation_normalization_sha256": require_sha256(
                relation_normalization_sha256,
                name="relation_normalization_sha256",
            ),
            "PID": {
                "category_order": list(PID_CATEGORY_NAMES),
                "threshold": PID_THRESHOLD,
                "binary_tolerance": PID_BINARY_TOLERANCE,
                "zero_hot_policy": "unknown_category",
                "multi_hot_policy": "fail_preflight_and_forward",
                "directed_pair_index": "query_category*6+context_category",
                "query_embedding": [6, 8],
                "context_embedding": [6, 8],
                "pair_embedding": [36, 8],
                "combination": "tanh((query+context+pair)/sqrt(3))",
                "encoded_dimension": PID_ENCODED_DIMENSION,
            },
            "CHARGE": {
                "state_order": list(CHARGE_STATES),
                "integer_tolerance": CHARGE_INTEGER_TOLERANCE,
                "directed_pair_index": "query_state*3+context_state",
                "raw_feature_names": list(CHARGE_RAW_FEATURE_NAMES),
                "robust_feature_names": list(CHARGE_ROBUST_FEATURE_NAMES),
                "binary_feature_names": list(CHARGE_RAW_FEATURE_NAMES[4:]),
                "categorical_embedding": [9, 4],
                "family_encoder_input_dimension": 12,
                "encoder": [
                    "Linear(12,32)",
                    "GELU",
                    "RMSNorm(32,eps=1e-6)",
                    "Linear(32,6)",
                ],
                "encoded_dimension": CHARGE_ENCODED_DIMENSION,
            },
            "pair_domain": "all_ordered_valid_pairs_including_diagonal",
            "invalid_pair_policy": "zero_after_every_learned_encoder",
            "dropout": 0.0,
        }
    )


__all__ = [
    "CHARGE_ENCODED_DIMENSION",
    "CHARGE_INTEGER_TOLERANCE",
    "CHARGE_STATES",
    "ChargeEncoder",
    "PID_BINARY_TOLERANCE",
    "PID_CATEGORY_NAMES",
    "PID_CHARGE_RELATION_CONTRACT",
    "PID_ENCODED_DIMENSION",
    "PID_THRESHOLD",
    "PIDEncoder",
    "audit_pid_flags",
    "build_charge_raw_features",
    "build_pid_charge_relation_contract",
    "pid_categories",
    "quantize_charge",
]
