"""Deterministic frozen-expert redundancy and subset diagnostics."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import require_sha256, with_content_hash
from .registry import EXPERT_ORDER

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


COMPLEMENTARITY_CONTRACT = "retb_offline_complementarity_v2"
SUBSET_READOUT_CONTRACT = "retb_offline_subset_readouts_v1"


def subset_id(mask: int) -> str:
    if int(mask) not in range(128):
        raise ValueError("expert subset mask lies outside 0..127")
    return f"SUBSET_{int(mask):07b}"


def subset_experts(mask: int) -> list[str]:
    return [
        expert
        for index, expert in enumerate(EXPERT_ORDER)
        if int(mask) & (1 << index)
    ]


def build_subset_readout_registry(
    *,
    shape_id: str,
    pipeline_seed: int,
) -> dict[str, Any]:
    if int(pipeline_seed) not in {101, 202, 303}:
        raise ValueError("subset readout seed is not registered")
    rows = [
        {
            "subset_id": subset_id(mask),
            "bitmask": mask,
            "experts": subset_experts(mask),
            "readouts": (
                ["CLASS_PRIOR"]
                if mask == 0
                else ["SUBSET_LOGIT_LINEAR", "SUBSET_POOLED_MLP"]
            ),
            "fresh_weights": True,
            "masking_untrained_full_fusion": False,
        }
        for mask in range(128)
    ]
    return with_content_hash(
        {
            "contract": SUBSET_READOUT_CONTRACT,
            "schema_version": 1,
            "shape_id": str(shape_id),
            "pipeline_seed": int(pipeline_seed),
            "expert_order": list(EXPERT_ORDER),
            "subset_count": 128,
            "rows": rows,
        }
    )


class EmptySubsetPrior(torch.nn.Module if torch is not None else object):
    def __init__(self, class_log_prior: Sequence[float]) -> None:
        super().__init__()
        values = np.asarray(class_log_prior, dtype=np.float32)
        if values.shape != (10,) or not np.isfinite(values).all():
            raise ValueError("empty-subset class prior is invalid")
        self.register_buffer("class_log_prior", torch.from_numpy(values))

    def forward(self, *, batch_size: int, **_: Any) -> Any:
        return self.class_log_prior.view(1, 10).expand(int(batch_size), -1)


class SubsetLogitLinear(torch.nn.Module if torch is not None else object):
    def __init__(self, experts: Sequence[str]) -> None:
        super().__init__()
        self.experts = tuple(experts)
        if not self.experts or any(name not in EXPERT_ORDER for name in self.experts):
            raise ValueError("subset logit readout experts are invalid")
        self.classifier = torch.nn.Linear(10 * len(self.experts), 10)

    def forward(self, *, expert_logits: Mapping[str, Any], **_: Any) -> Any:
        return self.classifier(
            torch.cat([expert_logits[name] for name in self.experts], dim=-1)
        )


class SubsetPooledMLP(torch.nn.Module if torch is not None else object):
    def __init__(
        self,
        *,
        experts: Sequence[str],
        bank_dimensions: Mapping[str, int],
    ) -> None:
        super().__init__()
        from .fusion import BankProjection, RMSNorm

        self.experts = tuple(experts)
        if not self.experts or any(name not in EXPERT_ORDER for name in self.experts):
            raise ValueError("subset pooled readout experts are invalid")
        self.projections = torch.nn.ModuleDict(
            {
                name: BankProjection(int(bank_dimensions[name]))
                for name in self.experts
            }
        )
        width = len(self.experts) * 128
        self.classifier = torch.nn.Sequential(
            RMSNorm(width),
            torch.nn.Linear(width, 256),
            torch.nn.GELU(),
            torch.nn.Linear(256, 10),
        )

    def forward(self, *, token_banks: Mapping[str, Any], **_: Any) -> Any:
        return self.classifier(
            torch.cat(
                [
                    self.projections[name](token_banks[name]).mean(dim=1)
                    for name in self.experts
                ],
                dim=-1,
            )
        )


def build_subset_readout(
    *,
    mask: int,
    kind: str,
    bank_dimensions: Mapping[str, int],
    class_log_prior: Sequence[float] | None = None,
) -> Any:
    experts = subset_experts(mask)
    if mask == 0:
        if kind != "CLASS_PRIOR" or class_log_prior is None:
            raise ValueError("empty subset requires CLASS_PRIOR")
        return EmptySubsetPrior(class_log_prior)
    if kind == "SUBSET_LOGIT_LINEAR":
        return SubsetLogitLinear(experts)
    if kind == "SUBSET_POOLED_MLP":
        return SubsetPooledMLP(
            experts=experts, bank_dimensions=bank_dimensions
        )
    raise ValueError("subset readout kind is unregistered")


def execute_subset_readout_screen(
    registry: Mapping[str, Any],
    *,
    executor: Any,
) -> dict[str, Any]:
    if registry.get("contract") != SUBSET_READOUT_CONTRACT:
        raise ValueError("subset readout registry contract differs")
    rows = registry.get("rows", [])
    if len(rows) != 128:
        raise ValueError("subset readout registry coverage differs")
    results = []
    for row in rows:
        for kind in row["readouts"]:
            result = dict(executor(dict(row), kind))
            if result.get("status") != "completed":
                raise RuntimeError("subset readout did not complete")
            results.append(
                {
                    "subset_id": row["subset_id"],
                    "readout": kind,
                    "status": "completed",
                }
            )
    expected = 1 + 127 * 2
    if len(results) != expected:
        raise RuntimeError("subset readout execution count differs")
    return {
        "subset_count": 128,
        "readout_run_count": len(results),
        "all_completed": True,
        "results": results,
    }


def linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64).reshape(len(left), -1)
    y = np.asarray(right, dtype=np.float64).reshape(len(right), -1)
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("linear CKA inputs are incompatible")
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    cross = np.linalg.norm(x.T @ y, ord="fro") ** 2
    left_norm = np.linalg.norm(x.T @ x, ord="fro")
    right_norm = np.linalg.norm(y.T @ y, ord="fro")
    denominator = left_norm * right_norm
    return 0.0 if denominator == 0.0 else float(cross / denominator)


def jensen_shannon_divergence(
    left: np.ndarray,
    right: np.ndarray,
) -> float:
    p = np.asarray(left, dtype=np.float64)
    q = np.asarray(right, dtype=np.float64)
    if p.shape != q.shape or p.ndim < 1:
        raise ValueError("attention distributions have incompatible shapes")
    if bool((p < 0).any() or (q < 0).any()):
        raise ValueError("attention probabilities are negative")
    p = p / p.sum(axis=-1, keepdims=True).clip(min=1.0e-300)
    q = q / q.sum(axis=-1, keepdims=True).clip(min=1.0e-300)
    middle = 0.5 * (p + q)
    left_term = np.where(p > 0, p * np.log(p / middle.clip(min=1e-300)), 0.0)
    right_term = np.where(q > 0, q * np.log(q / middle.clip(min=1e-300)), 0.0)
    return float(0.5 * (left_term.sum(axis=-1) + right_term.sum(axis=-1)).mean())


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    if x.shape != y.shape:
        raise ValueError("correlation inputs are incompatible")
    if len(x) < 2:
        return None
    if x.std() == 0.0 or y.std() == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def pairwise_expert_diagnostics(
    *,
    logits_by_expert: Mapping[str, np.ndarray],
    labels: np.ndarray,
    tokens_by_expert: Mapping[str, np.ndarray],
    attention_by_expert: Mapping[str, np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    if (
        set(logits_by_expert) != set(EXPERT_ORDER)
        or set(tokens_by_expert) != set(EXPERT_ORDER)
    ):
        raise ValueError("pairwise diagnostic experts differ")
    truth = np.asarray(labels, dtype=np.int64)
    rows = []
    for left_index, left_name in enumerate(EXPERT_ORDER):
        left_logits = np.asarray(logits_by_expert[left_name], dtype=np.float64)
        left_prediction = left_logits.argmax(axis=1)
        left_correct = left_prediction == truth
        left_residual = left_logits - left_logits.mean(axis=1, keepdims=True)
        for right_name in EXPERT_ORDER[left_index + 1 :]:
            right_logits = np.asarray(
                logits_by_expert[right_name], dtype=np.float64
            )
            if left_logits.shape != right_logits.shape or left_logits.shape != (
                len(truth),
                10,
            ):
                raise ValueError("expert logits have incompatible shapes")
            right_prediction = right_logits.argmax(axis=1)
            right_correct = right_prediction == truth
            right_residual = right_logits - right_logits.mean(
                axis=1, keepdims=True
            )
            per_class = {}
            for class_index in range(10):
                selected = truth == class_index
                per_class[str(class_index)] = _safe_correlation(
                    (~left_correct[selected]).astype(float),
                    (~right_correct[selected]).astype(float),
                )
            attention_js = None
            if attention_by_expert is not None:
                attention_js = jensen_shannon_divergence(
                    attention_by_expert[left_name],
                    attention_by_expert[right_name],
                )
            rows.append(
                {
                    "left": left_name,
                    "right": right_name,
                    "prediction_disagreement": float(
                        (left_prediction != right_prediction).mean()
                    ),
                    "correct_error_contingency": [
                        [
                            int((left_correct & right_correct).sum()),
                            int((left_correct & ~right_correct).sum()),
                        ],
                        [
                            int((~left_correct & right_correct).sum()),
                            int((~left_correct & ~right_correct).sum()),
                        ],
                    ],
                    "per_class_error_correlation": per_class,
                    "centered_logit_residual_correlation": _safe_correlation(
                        left_residual, right_residual
                    ),
                    "linear_token_CKA": linear_cka(
                        tokens_by_expert[left_name],
                        tokens_by_expert[right_name],
                    ),
                    "slot_attention_JS_divergence": attention_js,
                }
            )
    return rows


def shapley_from_subset_accuracy(
    subset_accuracy: Mapping[int, float],
) -> dict[str, float]:
    if set(map(int, subset_accuracy)) != set(range(128)):
        raise ValueError("Shapley computation requires all 128 subsets")
    factorial = math.factorial
    total = factorial(7)
    output = {}
    for expert_index, expert in enumerate(EXPERT_ORDER):
        value = 0.0
        bit = 1 << expert_index
        for mask in range(128):
            if mask & bit:
                continue
            size = int(mask.bit_count())
            weight = factorial(size) * factorial(6 - size) / total
            value += weight * (
                float(subset_accuracy[mask | bit])
                - float(subset_accuracy[mask])
            )
        output[expert] = float(value)
    return output


def build_complementarity_report(
    *,
    shape_id: str,
    pipeline_seed: int,
    cache_manifest_sha256: str,
    logits_by_expert: Mapping[str, np.ndarray],
    labels: np.ndarray,
    tokens_by_expert: Mapping[str, np.ndarray],
    subset_metrics: Mapping[int, Mapping[str, float]],
    attention_by_expert: Mapping[str, np.ndarray] | None = None,
    leave_one_out_metrics: Mapping[str, Mapping[str, float]] | None = None,
    bias_zero_sensitivity: Mapping[str, float] | None = None,
    relation_shuffle_sensitivity: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    if set(map(int, subset_metrics)) != set(range(128)):
        raise ValueError("complementarity requires complete subset readouts")
    accuracy = {
        int(mask): float(metrics["accuracy"])
        for mask, metrics in subset_metrics.items()
    }
    rows = pairwise_expert_diagnostics(
        logits_by_expert=logits_by_expert,
        labels=labels,
        tokens_by_expert=tokens_by_expert,
        attention_by_expert=attention_by_expert,
    )
    return with_content_hash(
        {
            "contract": COMPLEMENTARITY_CONTRACT,
            "schema_version": 2,
            "shape_id": str(shape_id),
            "pipeline_seed": int(pipeline_seed),
            "cache_manifest_sha256": require_sha256(
                cache_manifest_sha256, name="cache_manifest_sha256"
            ),
            "expert_order": list(EXPERT_ORDER),
            "pairwise": rows,
            "correlation_policy": {
                "minimum_sample_count": 2,
                "insufficient_support_serialization": None,
                "constant_input_serialization": None,
                "mismatched_shapes": "error",
            },
            "subset_metrics": {
                subset_id(mask): dict(subset_metrics[mask])
                for mask in range(128)
            },
            "subset_coverage": 128,
            "subset_readouts_are_separately_trained": True,
            "leave_one_expert_out": dict(leave_one_out_metrics or {}),
            "shapley_accuracy_contribution": shapley_from_subset_accuracy(
                accuracy
            ),
            "bias_zero_sensitivity": dict(bias_zero_sensitivity or {}),
            "within_jet_relation_shuffle_sensitivity": dict(
                relation_shuffle_sensitivity or {}
            ),
            "diversity_regularizer_used": False,
        }
    )


__all__ = [
    "COMPLEMENTARITY_CONTRACT",
    "SUBSET_READOUT_CONTRACT",
    "build_complementarity_report",
    "build_subset_readout",
    "build_subset_readout_registry",
    "execute_subset_readout_screen",
    "jensen_shannon_divergence",
    "linear_cka",
    "pairwise_expert_diagnostics",
    "shapley_from_subset_accuracy",
    "subset_experts",
    "subset_id",
]
