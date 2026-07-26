"""Deterministic Step-8 controls for privileged particle-view experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import torch

from .contracts import require_sha256, with_content_hash


PARTICLE_VIEW_CONTROL_REGISTRY_CONTRACT = "particle_view_control_registry_v1"
PARTICLE_VIEW_CONTROL_RESULT_CONTRACT = "particle_view_control_result_v1"
PARTICLE_VIEW_DIRECT_MATCH_CONTRACT = "particle_view_direct_match_v1"
PARTICLE_VIEW_STAGE_G_CONTROL_PLAN_CONTRACT = "particle_view_stage_g_control_plan_v1"

STRUCTURAL_CONTROL_IDS = (
    "ZERO_VIEW",
    "SAME_NORM_RANDOM_VIEW",
    "SIGN_REVERSED_VIEW",
    "EVENT_SHUFFLED_VIEW",
    "PARTICLE_SHUFFLED_VIEW",
    "VIEW_DIMENSION_PERMUTATION",
    "INDEPENDENTLY_SHUFFLED_DIMENSIONS",
    "JET_MEAN_BROADCAST",
    "JET_MEAN_REMOVED",
    "TRUE_VIEW_TOP_PT_25",
    "TRUE_VIEW_TOP_PT_50",
    "TRUE_VIEW_TOP_PT_75",
    "PREDICTED_VIEW_TOP_PT_25",
    "PREDICTED_VIEW_TOP_PT_50",
    "PREDICTED_VIEW_TOP_PT_75",
    "MASKED_NULL_TOKEN_ONLY",
)

TRAINED_CONTROL_IDS = (
    "IDENTICAL_CE_ONLY",
    "HLT_SELF_DISTILLATION",
    "STAGE_A_PARAMETER_MATCH",
    "STAGE_A_FLOP_MATCH",
    "SELECTED_PARAMETER_MATCH",
    "SELECTED_FLOP_MATCH",
    "DEEPER_DIRECT_HLT_PART",
    "A0_VIEW_LONG_DEPLOY",
    "A0_VIEW_TOTAL_LABEL_BUDGET",
    "RANDOM_PREDICTOR_INITIALIZATION",
    "FROZEN_RANDOM_VIEW_GENERATOR",
    "OFFLINE_GLOBAL_LOGIT_BROADCAST",
    "RAW_OFFLINE_CROSS_ATTENTION",
    "OFFLINE_CLASSIFIER_DIRECT_KD",
    "DVIEW_JOINT",
    "DVIEW_JOINT_CE_ONLY",
)

DIRECT_CONTROL_HYPERPARAMETER_GRID = {
    "learning_rate": [1.0e-4, 3.0e-4],
    "weight_decay": [1.0e-5, 1.0e-4],
    "dropout": [0.05, 0.10],
}


def build_step8_control_registry() -> dict[str, Any]:
    """Return the locked control inventory and its selection permissions."""

    rows: list[dict[str, Any]] = []
    for control_id in STRUCTURAL_CONTROL_IDS:
        rows.append(
            {
                "control_id": control_id,
                "kind": "structural_evaluation",
                "selectable": False,
                "privileged_claim_eligible": False,
                "stack_val_eligible": False,
                "final_test_eligible": False,
            }
        )
    for control_id in TRAINED_CONTROL_IDS:
        pre_stage_eligible = control_id in {
            "IDENTICAL_CE_ONLY",
            "HLT_SELF_DISTILLATION",
            "DVIEW_JOINT",
            "DVIEW_JOINT_CE_ONLY",
        }
        rows.append(
            {
                "control_id": control_id,
                "kind": "trained_control",
                "selectable": pre_stage_eligible,
                "privileged_claim_eligible": control_id == "DVIEW_JOINT",
                "stack_val_eligible": control_id
                in {
                    "IDENTICAL_CE_ONLY",
                    "SELECTED_PARAMETER_MATCH",
                    "SELECTED_FLOP_MATCH",
                    "A0_VIEW_LONG_DEPLOY",
                    "A0_VIEW_TOTAL_LABEL_BUDGET",
                },
                "final_test_eligible": False,
            }
        )
    rows.sort(key=lambda row: row["control_id"])
    return with_content_hash(
        {
            "contract": PARTICLE_VIEW_CONTROL_REGISTRY_CONTRACT,
            "rows": rows,
            "structural_control_count": len(STRUCTURAL_CONTROL_IDS),
            "trained_control_count": len(TRAINED_CONTROL_IDS),
            "scientific_quality_policy": "warn_and_continue",
            "performance_gates": False,
        }
    )


def _valid_mask(mask: torch.Tensor, view: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 3 and mask.shape[1] == 1:
        mask = mask[:, 0]
    if mask.ndim != 2 or mask.shape != view.shape[:2]:
        raise ValueError("mask must have shape [batch, particles] or [batch,1,particles]")
    return mask.to(device=view.device, dtype=torch.bool)


def _derangement(size: int, generator: torch.Generator) -> torch.Tensor:
    if size < 2:
        raise ValueError("a no-fixed-point shuffle requires at least two items")
    for _ in range(256):
        order = torch.randperm(size, generator=generator)
        if not torch.any(order == torch.arange(size)):
            return order
    # A cyclic rotation is a deterministic, guaranteed derangement.
    shift = int(torch.randint(1, size, (1,), generator=generator).item())
    return torch.roll(torch.arange(size), shifts=shift)


def _particle_shuffle(
    view: torch.Tensor, valid: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    result = torch.zeros_like(view)
    for event in range(view.shape[0]):
        indices = torch.nonzero(valid[event], as_tuple=False).flatten()
        if indices.numel() < 2:
            result[event, indices] = view[event, indices]
            continue
        order = _derangement(int(indices.numel()), generator).to(indices.device)
        result[event, indices] = view[event, indices[order]]
    return result


def _event_shuffle(
    view: torch.Tensor,
    valid: torch.Tensor,
    generator: torch.Generator,
    labels: torch.Tensor | None,
) -> tuple[torch.Tensor, list[int]]:
    batch = view.shape[0]
    result = torch.zeros_like(view)
    mapping = torch.arange(batch)
    groups = [torch.arange(batch)]
    if labels is not None:
        if labels.ndim != 1 or labels.shape[0] != batch:
            raise ValueError("labels must have shape [batch]")
        groups = [torch.nonzero(labels.cpu() == value).flatten() for value in sorted(set(labels.cpu().tolist()))]
    for group in groups:
        if group.numel() < 2:
            # Exact class preservation and a derangement are incompatible.
            # Fall back to the full-batch mapping; the result records balance.
            groups = [torch.arange(batch)]
            mapping = torch.arange(batch)
            break
        order = _derangement(int(group.numel()), generator)
        mapping[group] = group[order]
    if torch.equal(mapping, torch.arange(batch)):
        mapping = _derangement(batch, generator)
    for destination, source in enumerate(mapping.tolist()):
        count = min(int(valid[destination].sum()), int(valid[source].sum()))
        destination_indices = torch.nonzero(valid[destination], as_tuple=False).flatten()[:count]
        source_indices = torch.nonzero(valid[source], as_tuple=False).flatten()[:count]
        result[destination, destination_indices] = view[source, source_indices]
    return result, [int(value) for value in mapping.tolist()]


def _top_pt_mask(
    valid: torch.Tensor, particle_pt: torch.Tensor, fraction: float
) -> torch.Tensor:
    if particle_pt.ndim == 3 and particle_pt.shape[1] == 1:
        particle_pt = particle_pt[:, 0]
    if particle_pt.shape != valid.shape:
        raise ValueError("particle_pt must match the particle mask")
    keep = torch.zeros_like(valid)
    for event in range(valid.shape[0]):
        indices = torch.nonzero(valid[event], as_tuple=False).flatten()
        number = max(1, int(math.ceil(float(indices.numel()) * fraction)))
        order = torch.argsort(
            particle_pt[event, indices], descending=True, stable=True
        )
        keep[event, indices[order[:number]]] = True
    return keep


def apply_particle_view_control(
    view: torch.Tensor,
    mask: torch.Tensor,
    *,
    control_id: str,
    seed: int = 101,
    particle_pt: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Apply one locked structural control and return audit diagnostics.

    Invalid particles remain exactly zero for every control.  Randomness uses
    a CPU generator so identical inputs and seeds are reproducible on CPU/GPU.
    """

    if control_id not in STRUCTURAL_CONTROL_IDS:
        raise ValueError(f"unknown structural control {control_id!r}")
    if view.ndim != 3 or not torch.is_floating_point(view):
        raise ValueError("view must be a floating tensor [batch,particles,dimensions]")
    if not torch.isfinite(view).all():
        raise FloatingPointError("view contains non-finite values")
    valid = _valid_mask(mask, view)
    clean = torch.where(valid[:, :, None], view, torch.zeros_like(view))
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    mapping: list[int] | None = None

    if control_id == "ZERO_VIEW" or control_id == "MASKED_NULL_TOKEN_ONLY":
        result = torch.zeros_like(clean)
    elif control_id == "SAME_NORM_RANDOM_VIEW":
        random_cpu = torch.randn(clean.shape, generator=generator, dtype=torch.float32)
        random = random_cpu.to(device=clean.device, dtype=clean.dtype)
        random = torch.where(valid[:, :, None], random, torch.zeros_like(random))
        target_norm = clean.flatten(1).norm(dim=1)
        random_norm = random.flatten(1).norm(dim=1).clamp_min(1.0e-12)
        result = random * (target_norm / random_norm)[:, None, None]
    elif control_id == "SIGN_REVERSED_VIEW":
        result = -clean
    elif control_id == "EVENT_SHUFFLED_VIEW":
        result, mapping = _event_shuffle(clean, valid, generator, labels)
    elif control_id == "PARTICLE_SHUFFLED_VIEW":
        result = _particle_shuffle(clean, valid, generator)
    elif control_id == "VIEW_DIMENSION_PERMUTATION":
        order = torch.randperm(clean.shape[-1], generator=generator).to(clean.device)
        if clean.shape[-1] > 1 and torch.equal(order.cpu(), torch.arange(clean.shape[-1])):
            order = torch.roll(order, 1)
        result = clean[..., order]
    elif control_id == "INDEPENDENTLY_SHUFFLED_DIMENSIONS":
        components = []
        for dimension in range(clean.shape[-1]):
            shuffled, _ = _event_shuffle(
                clean[..., dimension : dimension + 1],
                valid,
                generator,
                labels=None,
            )
            components.append(shuffled)
        result = torch.cat(components, dim=-1)
    elif control_id in {"JET_MEAN_BROADCAST", "JET_MEAN_REMOVED"}:
        counts = valid.sum(dim=1, keepdim=True).clamp_min(1)
        mean = clean.sum(dim=1) / counts
        if control_id == "JET_MEAN_BROADCAST":
            result = mean[:, None].expand_as(clean)
        else:
            result = clean - mean[:, None]
    elif "_TOP_PT_" in control_id:
        if particle_pt is None:
            raise ValueError("top-pT controls require particle_pt")
        fraction = int(control_id.rsplit("_", 1)[1]) / 100.0
        keep = _top_pt_mask(valid, particle_pt.to(clean.device), fraction)
        result = torch.where(keep[:, :, None], clean, torch.zeros_like(clean))
    else:  # pragma: no cover - guarded by the inventory above
        raise AssertionError(control_id)

    result = torch.where(valid[:, :, None], result, torch.zeros_like(result))
    if not torch.isfinite(result).all():
        raise FloatingPointError("structural control produced non-finite values")
    invalid_max = (
        float(result[~valid].abs().max().item()) if torch.any(~valid) else 0.0
    )
    fixed_points = (
        sum(index == source for index, source in enumerate(mapping))
        if mapping is not None
        else None
    )
    diagnostics = {
        "control_id": control_id,
        "seed": int(seed),
        "valid_particles": int(valid.sum().item()),
        "invalid_particle_max_abs": invalid_max,
        "padding_preserved": invalid_max == 0.0,
        "event_permutation": mapping,
        "event_permutation_fixed_points": fixed_points,
        "input_norm": float(clean.norm().item()),
        "output_norm": float(result.norm().item()),
    }
    return result, diagnostics


def relative_resource_error(control: int, target: int) -> float:
    if (
        not isinstance(control, int)
        or isinstance(control, bool)
        or control < 0
        or not isinstance(target, int)
        or isinstance(target, bool)
        or target < 0
    ):
        raise ValueError("resource totals must be nonnegative integers")
    return abs(float(control) - float(target)) / max(float(target), 1.0)


@dataclass(frozen=True)
class DirectControlCandidate:
    config_id: str
    deployed_parameters: int
    forward_flops: int
    config_sha256: str

    def to_payload(self) -> dict[str, Any]:
        if not self.config_id:
            raise ValueError("direct-control config_id must be non-empty")
        if self.deployed_parameters <= 0 or self.forward_flops <= 0:
            raise ValueError("direct-control resources must be positive")
        require_sha256("config_sha256", self.config_sha256)
        return {
            "config_id": self.config_id,
            "deployed_parameters": int(self.deployed_parameters),
            "forward_flops": int(self.forward_flops),
            "config_sha256": self.config_sha256,
        }


def select_direct_resource_control(
    *,
    candidates: Sequence[DirectControlCandidate | Mapping[str, Any]],
    target_parameters: int,
    target_flops: int,
    requested_quantity: str,
    selected_bundle_sha256: str,
    flop_fixture_sha256: str,
    flop_counter_sha256: str,
) -> dict[str, Any]:
    """Select the deterministic nearest parameter- or FLOP-matched HLT model."""

    if requested_quantity not in {"parameters", "flops"}:
        raise ValueError("requested_quantity must be parameters or flops")
    if target_parameters <= 0 or target_flops <= 0:
        raise ValueError("target resources must be positive")
    for name, value in (
        ("selected_bundle_sha256", selected_bundle_sha256),
        ("flop_fixture_sha256", flop_fixture_sha256),
        ("flop_counter_sha256", flop_counter_sha256),
    ):
        require_sha256(name, value)
    normalized = [
        item.to_payload()
        if isinstance(item, DirectControlCandidate)
        else DirectControlCandidate(**dict(item)).to_payload()
        for item in candidates
    ]
    if not normalized:
        raise ValueError("direct-control candidate registry is empty")
    if len({row["config_id"] for row in normalized}) != len(normalized):
        raise ValueError("direct-control config IDs must be unique")

    def key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        parameter_error = relative_resource_error(
            int(row["deployed_parameters"]), target_parameters
        )
        flop_error = relative_resource_error(int(row["forward_flops"]), target_flops)
        primary, secondary = (
            (parameter_error, flop_error)
            if requested_quantity == "parameters"
            else (flop_error, parameter_error)
        )
        return (
            primary,
            secondary,
            int(row["deployed_parameters"]),
            int(row["forward_flops"]),
            str(row["config_id"]),
        )

    selected = min(normalized, key=key)
    parameter_error = relative_resource_error(
        selected["deployed_parameters"], target_parameters
    )
    flop_error = relative_resource_error(selected["forward_flops"], target_flops)
    matched = parameter_error <= 0.05 if requested_quantity == "parameters" else flop_error <= 0.10
    warning = None if matched else "WARN_CONTROL_MATCH_TOLERANCE"
    return with_content_hash(
        {
            "contract": PARTICLE_VIEW_DIRECT_MATCH_CONTRACT,
            "requested_quantity": requested_quantity,
            "selected_bundle_sha256": selected_bundle_sha256,
            "flop_fixture_sha256": flop_fixture_sha256,
            "flop_counter_sha256": flop_counter_sha256,
            "target": {
                "deployed_parameters": int(target_parameters),
                "forward_flops": int(target_flops),
            },
            "selected": selected,
            "relative_parameter_error": parameter_error,
            "relative_forward_flop_error": flop_error,
            "parameter_tolerance": 0.05,
            "flop_tolerance": 0.10,
            "match_within_requested_tolerance": matched,
            "quality_warning": warning,
            "warning_is_non_gating": True,
            "candidate_count": len(normalized),
        }
    )


def build_stage_g_control_plan(
    *,
    fairness_ledger: Mapping[str, Any],
    candidates: Sequence[DirectControlCandidate | Mapping[str, Any]],
    a0_checkpoint_by_seed: Mapping[int, str],
    a0_config_sha256: str,
) -> dict[str, Any]:
    """Resolve all post-selection jobs from the already-published ledger.

    This function is intentionally downstream of selection.  It never reads a
    validation metric, and a failed resource tolerance only produces a
    warning while retaining the nearest predeclared direct control.
    """

    from .selection import PARTICLE_VIEW_FAIRNESS_LEDGER_CONTRACT
    from .contracts import validate_content_hash

    validate_content_hash(
        fairness_ledger, expected_contract=PARTICLE_VIEW_FAIRNESS_LEDGER_CONTRACT
    )
    require_sha256("a0_config_sha256", a0_config_sha256)
    if set(a0_checkpoint_by_seed) != {101, 202, 303}:
        raise ValueError("Stage-G A0 controls require checkpoints for all three seeds")
    a0_checkpoints = {
        seed: require_sha256(
            f"a0_checkpoint_by_seed[{seed}]", a0_checkpoint_by_seed[seed]
        )
        for seed in (101, 202, 303)
    }
    jobs: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    for entry in fairness_ledger["entries"]:
        fairness_entry_sha256 = require_sha256(
            "fairness_entry_sha256", entry["fairness_entry_sha256"]
        )
        for replica in entry["replicas"]:
            seed = int(replica["seed"])
            common = {
                "configuration_id": entry["configuration_id"],
                "winner_bundle_sha256": entry["winner_bundle_sha256"],
                "fairness_entry_sha256": fairness_entry_sha256,
                "seed": seed,
                "train_split": "train",
                "train_identity_sha256": fairness_ledger[
                    "train_identity_sha256"
                ],
                "a0_initial_checkpoint_sha256": a0_checkpoints[seed],
                "a0_config_sha256": a0_config_sha256,
                "checkpoint_selection_split": "model_val_stop",
                "stack_val_loaded": False,
                "final_test_loaded": False,
            }
            for control_id, budget_field in (
                (
                    "A0_VIEW_LONG_DEPLOY",
                    "a0_view_long_deploy_exact_ce_updates",
                ),
                (
                    "A0_VIEW_TOTAL_LABEL_BUDGET",
                    "a0_view_total_label_budget_exact_updates",
                ),
            ):
                jobs.append(
                    {
                        **common,
                        "control_id": control_id,
                        "exact_optimizer_update_budget": int(
                            replica[budget_field]
                        ),
                        "uninterrupted_trajectory": True,
                        "registered_checkpoints": [
                            "exact_matched_update",
                            "best_model_val_stop_within_budget",
                        ],
                        "primary_comparison_checkpoint": (
                            "best_model_val_stop_within_budget"
                        ),
                    }
                )
            for requested, control_id in (
                ("parameters", "SELECTED_PARAMETER_MATCH"),
                ("flops", "SELECTED_FLOP_MATCH"),
            ):
                match = select_direct_resource_control(
                    candidates=candidates,
                    target_parameters=int(replica["deployed_parameters"]),
                    target_flops=int(replica["forward_flops_128"]),
                    requested_quantity=requested,
                    selected_bundle_sha256=entry["winner_bundle_sha256"],
                    flop_fixture_sha256=fairness_ledger[
                        "flop_fixture_sha256"
                    ],
                    flop_counter_sha256=fairness_ledger[
                        "flop_counter_sha256"
                    ],
                )
                jobs.append(
                    {
                        **common,
                        "control_id": control_id,
                        "direct_match_sha256": match["content_hash"],
                        "selected_direct_config": match["selected"],
                        "eight_trial_grid": DIRECT_CONTROL_HYPERPARAMETER_GRID,
                        "trial_count": 8,
                        "requested_tolerance_met": match[
                            "match_within_requested_tolerance"
                        ],
                    }
                )
                if match["quality_warning"] is not None:
                    warnings.append(
                        {
                            "warning_code": match["quality_warning"],
                            "severity": "warning",
                            "configuration_id": entry["configuration_id"],
                            "seed": seed,
                            "control_id": control_id,
                            "supporting_artifact_sha256": match["content_hash"],
                            "non_gating": True,
                        }
                    )
    jobs.sort(
        key=lambda row: (
            row["configuration_id"],
            row["control_id"],
            row["seed"],
        )
    )
    return with_content_hash(
        {
            "contract": PARTICLE_VIEW_STAGE_G_CONTROL_PLAN_CONTRACT,
            "fairness_ledger_sha256": fairness_ledger["content_hash"],
            "jobs": jobs,
            "job_count": len(jobs),
            "quality_warnings": warnings,
            "warnings_are_non_gating": True,
            "performance_gates": False,
            "may_start_only_after_fairness_publication": True,
        }
    )


__all__ = [
    "DirectControlCandidate",
    "PARTICLE_VIEW_CONTROL_REGISTRY_CONTRACT",
    "PARTICLE_VIEW_CONTROL_RESULT_CONTRACT",
    "PARTICLE_VIEW_DIRECT_MATCH_CONTRACT",
    "PARTICLE_VIEW_STAGE_G_CONTROL_PLAN_CONTRACT",
    "DIRECT_CONTROL_HYPERPARAMETER_GRID",
    "STRUCTURAL_CONTROL_IDS",
    "TRAINED_CONTROL_IDS",
    "apply_particle_view_control",
    "build_step8_control_registry",
    "build_stage_g_control_plan",
    "relative_resource_error",
    "select_direct_resource_control",
]
