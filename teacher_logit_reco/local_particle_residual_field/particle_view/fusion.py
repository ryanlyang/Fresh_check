"""Sealed stack-validation fusion and independent-seed controls."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from .contracts import (
    canonical_sha256,
    require_sha256,
    validate_content_hash,
    with_content_hash,
)
from .metrics import classification_metrics


PARTICLE_VIEW_STACK_PARTITION_CONTRACT = "particle_view_stack_partition_v1"
PARTICLE_VIEW_FUSION_RECIPE_CONTRACT = "particle_view_fusion_recipe_v1"
PARTICLE_VIEW_FUSION_REPORT_CONTRACT = "particle_view_fusion_report_v1"
STACK_FUSION_PARTITION_SEED = 82_117
A0_A0_PAIRS = ((101, 202), (202, 303), (303, 101))


def _identity_key(identity: Any, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}:{identity}".encode("utf-8")).digest()


def build_stack_fusion_partition(
    *,
    event_identities: Sequence[Any],
    stack_split_sha256: str,
    fit_count: int | None = None,
) -> dict[str, Any]:
    """Create the deterministic, immutable stack fit/evaluation partition."""

    require_sha256("stack_split_sha256", stack_split_sha256)
    identities = list(event_identities)
    if not identities or len(set(map(str, identities))) != len(identities):
        raise ValueError("stack identities must be nonempty and unique")
    if fit_count is None:
        if len(identities) % 2:
            raise ValueError("stack split must have an even event count")
        fit_count = len(identities) // 2
    if not 0 < fit_count < len(identities):
        raise ValueError("fusion fit_count must leave a held-out evaluation set")
    ordered = sorted(
        range(len(identities)),
        key=lambda index: (_identity_key(identities[index], STACK_FUSION_PARTITION_SEED), index),
    )
    fit = sorted(ordered[:fit_count])
    evaluation = sorted(ordered[fit_count:])
    identity_hash = canonical_sha256([str(value) for value in identities])
    return with_content_hash(
        {
            "contract": PARTICLE_VIEW_STACK_PARTITION_CONTRACT,
            "stack_split_sha256": stack_split_sha256,
            "stack_identity_sha256": identity_hash,
            "partition_seed": STACK_FUSION_PARTITION_SEED,
            "event_count": len(identities),
            "fit_indices": fit,
            "evaluation_indices": evaluation,
            "fit_identity_sha256": canonical_sha256(
                [str(identities[index]) for index in fit]
            ),
            "evaluation_identity_sha256": canonical_sha256(
                [str(identities[index]) for index in evaluation]
            ),
            "winner_selection_permitted": False,
        }
    )


def _check_logits(
    arrays: Sequence[np.ndarray | Sequence[Sequence[float]]],
    labels: Sequence[int] | np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray]:
    converted = [np.asarray(array, dtype=np.float64) for array in arrays]
    target = np.asarray(labels, dtype=np.int64)
    if len(converted) < 2:
        raise ValueError("fusion requires at least two logit sources")
    shape = converted[0].shape
    if (
        len(shape) != 2
        or shape[0] != target.size
        or any(array.shape != shape for array in converted)
    ):
        raise ValueError("fusion logits are not event/class aligned")
    if not all(np.isfinite(array).all() for array in converted):
        raise FloatingPointError("fusion logits contain non-finite values")
    if target.min() < 0 or target.max() >= shape[1]:
        raise ValueError("fusion labels are outside the class order")
    return converted, target


def fit_linear_logit_fusion(
    *,
    source_logits: Sequence[np.ndarray | Sequence[Sequence[float]]],
    labels: Sequence[int] | np.ndarray,
    fit_indices: Sequence[int],
    steps: int = 300,
    learning_rate: float = 0.05,
    weight_decay: float = 1.0e-4,
) -> dict[str, Any]:
    """Fit only the stack-fit half with a deterministic linear logit layer."""

    arrays, target = _check_logits(source_logits, labels)
    indices = np.asarray(fit_indices, dtype=np.int64)
    if (
        indices.ndim != 1
        or indices.size == 0
        or np.unique(indices).size != indices.size
        or indices.min() < 0
        or indices.max() >= target.size
    ):
        raise ValueError("invalid fusion fit indices")
    if steps <= 0 or learning_rate <= 0 or weight_decay < 0:
        raise ValueError("invalid fusion optimizer contract")
    features = np.concatenate(arrays, axis=1)
    x = torch.from_numpy(features[indices]).to(torch.float64)
    y = torch.from_numpy(target[indices])
    classes = arrays[0].shape[1]
    torch.manual_seed(55_019)
    weight = torch.zeros(
        classes, features.shape[1], dtype=torch.float64, requires_grad=True
    )
    for source in range(len(arrays)):
        start = source * classes
        weight.data[:, start : start + classes].add_(
            torch.eye(classes, dtype=torch.float64) / len(arrays)
        )
    bias = torch.zeros(classes, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.AdamW(
        [weight, bias], lr=learning_rate, weight_decay=weight_decay
    )
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(F.linear(x, weight, bias), y)
        if not torch.isfinite(loss):
            raise FloatingPointError("linear fusion produced a non-finite loss")
        loss.backward()
        if not torch.isfinite(weight.grad).all() or not torch.isfinite(bias.grad).all():
            raise FloatingPointError("linear fusion produced a non-finite gradient")
        optimizer.step()
    return {
        "weight": weight.detach().cpu().numpy().tolist(),
        "bias": bias.detach().cpu().numpy().tolist(),
        "optimizer": "AdamW",
        "steps": int(steps),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "fit_event_count": int(indices.size),
        "fit_indices_sha256": canonical_sha256(indices.tolist()),
    }


def apply_linear_logit_fusion(
    source_logits: Sequence[np.ndarray | Sequence[Sequence[float]]],
    parameters: Mapping[str, Any],
) -> np.ndarray:
    arrays = [np.asarray(array, dtype=np.float64) for array in source_logits]
    if not arrays or any(array.shape != arrays[0].shape for array in arrays):
        raise ValueError("fusion source shapes differ")
    features = np.concatenate(arrays, axis=1)
    weight = np.asarray(parameters["weight"], dtype=np.float64)
    bias = np.asarray(parameters["bias"], dtype=np.float64)
    if weight.shape != (arrays[0].shape[1], features.shape[1]) or bias.shape != (
        arrays[0].shape[1],
    ):
        raise ValueError("linear fusion parameters have the wrong shape")
    result = features @ weight.T + bias
    if not np.isfinite(result).all():
        raise FloatingPointError("linear fusion output is non-finite")
    return result


def build_fusion_recipe(
    *,
    fusion_id: str,
    source_bundle_sha256: Sequence[str],
    class_order: Sequence[str],
    stack_partition: Mapping[str, Any],
    method: str,
    linear_parameters: Mapping[str, Any] | None = None,
    optional_p7b: bool = False,
    p7b_hlt_only_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not fusion_id or method not in {"logit_average", "linear_logit"}:
        raise ValueError("invalid fusion recipe identity/method")
    if len(source_bundle_sha256) < 2:
        raise ValueError("fusion requires at least two bundles")
    sources = [
        require_sha256("source_bundle_sha256", value)
        for value in source_bundle_sha256
    ]
    if len(set(sources)) != len(sources):
        raise ValueError("a checkpoint cannot be fused with itself")
    if not class_order or len(set(class_order)) != len(class_order):
        raise ValueError("fusion class order is invalid")
    validate_content_hash(
        stack_partition, expected_contract=PARTICLE_VIEW_STACK_PARTITION_CONTRACT
    )
    if method == "linear_logit" and linear_parameters is None:
        raise ValueError("linear fusion requires fit parameters")
    if method == "logit_average" and linear_parameters is not None:
        raise ValueError("logit average may not contain fitted parameters")
    if optional_p7b:
        if (
            not isinstance(p7b_hlt_only_provenance, Mapping)
            or p7b_hlt_only_provenance.get("requires_oracle") is not False
            or p7b_hlt_only_provenance.get("final_test_hlt_only") is not True
        ):
            raise ValueError("optional P7b fusion lacks HLT-only provenance")
        require_sha256(
            "p7b_deployment_sha256",
            p7b_hlt_only_provenance.get("deployment_sha256"),
        )
    return with_content_hash(
        {
            "contract": PARTICLE_VIEW_FUSION_RECIPE_CONTRACT,
            "fusion_id": fusion_id,
            "method": method,
            "source_bundle_sha256": sources,
            "class_order": list(class_order),
            "stack_partition_sha256": require_sha256(
                "stack_partition_sha256", stack_partition.get("content_hash")
            ),
            "fit_identity_sha256": stack_partition["fit_identity_sha256"],
            "evaluation_identity_sha256": stack_partition[
                "evaluation_identity_sha256"
            ],
            "linear_parameters": (
                dict(linear_parameters) if linear_parameters is not None else None
            ),
            "optional_p7b": bool(optional_p7b),
            "p7b_hlt_only_provenance": (
                dict(p7b_hlt_only_provenance) if optional_p7b else None
            ),
            "hlt_only": True,
            "winner_selection_permitted": False,
        }
    )


def evaluate_fusion_recipe(
    *,
    recipe: Mapping[str, Any],
    stack_partition: Mapping[str, Any],
    source_logits: Sequence[np.ndarray | Sequence[Sequence[float]]],
    labels: Sequence[int] | np.ndarray,
    source_bundle_sha256: Sequence[str],
) -> dict[str, Any]:
    validate_content_hash(
        recipe, expected_contract=PARTICLE_VIEW_FUSION_RECIPE_CONTRACT
    )
    validate_content_hash(
        stack_partition, expected_contract=PARTICLE_VIEW_STACK_PARTITION_CONTRACT
    )
    if recipe.get("stack_partition_sha256") != stack_partition.get("content_hash"):
        raise ValueError("fusion recipe uses a different sealed partition")
    if list(source_bundle_sha256) != recipe.get("source_bundle_sha256"):
        raise ValueError("fusion source checkpoints changed")
    arrays, target = _check_logits(source_logits, labels)
    if recipe["method"] == "logit_average":
        fused = np.mean(arrays, axis=0)
    else:
        fused = apply_linear_logit_fusion(arrays, recipe["linear_parameters"])
    indices = np.asarray(stack_partition["evaluation_indices"], dtype=np.int64)
    metrics = classification_metrics(
        fused[indices],
        target[indices],
        split="stack_val_evaluation",
        class_names=recipe["class_order"],
    )
    return with_content_hash(
        {
            "contract": PARTICLE_VIEW_FUSION_REPORT_CONTRACT,
            "fusion_recipe_sha256": recipe["content_hash"],
            "stack_partition_sha256": stack_partition["content_hash"],
            "evaluation_only": True,
            "evaluation_event_count": int(indices.size),
            "metrics": metrics,
            "selection_changed": False,
        }
    )


def build_a0_a0_pair_recipes(
    *,
    checkpoints_by_seed: Mapping[int, str],
    class_order: Sequence[str],
    stack_partition: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if set(checkpoints_by_seed) != {101, 202, 303}:
        raise ValueError("A0+A0 control requires seeds 101, 202, and 303")
    recipes = []
    for left, right in A0_A0_PAIRS:
        recipes.append(
            build_fusion_recipe(
                fusion_id=f"A0_VIEW_{left}+A0_VIEW_{right}",
                source_bundle_sha256=[
                    checkpoints_by_seed[left],
                    checkpoints_by_seed[right],
                ],
                class_order=class_order,
                stack_partition=stack_partition,
                method="logit_average",
            )
        )
    return recipes


__all__ = [
    "A0_A0_PAIRS",
    "PARTICLE_VIEW_FUSION_RECIPE_CONTRACT",
    "PARTICLE_VIEW_FUSION_REPORT_CONTRACT",
    "PARTICLE_VIEW_STACK_PARTITION_CONTRACT",
    "STACK_FUSION_PARTITION_SEED",
    "apply_linear_logit_fusion",
    "build_a0_a0_pair_recipes",
    "build_fusion_recipe",
    "build_stack_fusion_partition",
    "evaluate_fusion_recipe",
    "fit_linear_logit_fusion",
]
