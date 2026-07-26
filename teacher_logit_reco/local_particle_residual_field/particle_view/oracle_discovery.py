"""Oracle-view discovery objectives, co-design, ranking, and warning contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import nn

from .consumer import ParticleViewConsumerConfig
from .contracts import (
    canonical_sha256,
    require_sha256,
    sha256_file,
    validate_content_hash,
    with_content_hash,
)
from .target_generator import (
    masked_particle_mean_center,
    particle_view_rate_covariance_losses,
)
from .recovery_probe import view_huber_loss, view_relational_loss


PARTICLE_VIEW_ORACLE_OBJECTIVE_CONTRACT = "particle_view_oracle_objective_v1"
PARTICLE_VIEW_TARGET_METRICS_CONTRACT = "particle_view_target_metrics_v1"
PARTICLE_VIEW_TARGET_SELECTION_CONTRACT = "particle_view_target_selection_v1"
PARTICLE_VIEW_WARNING_CONTRACT = "particle_view_quality_warning_v1"
PARTICLE_VIEW_CODESIGN_CONTRACT = "particle_view_recoverability_codesign_v1"
PARTICLE_VIEW_CODESIGN_LEDGER_CONTRACT = (
    "particle_view_recoverability_codesign_ledger_v1"
)
PARTICLE_VIEW_TWO_PASS_CONTRACT = "particle_view_two_pass_candidate_v1"
PARTICLE_VIEW_CLEAN_CONSUMER_REGISTRATION_CONTRACT = (
    "particle_view_clean_consumer_registration_v1"
)

OFFLINE_KD_SCREEN = (0.0, 0.25, 0.5, 1.0)
_TARGET_SELECTION_STATUSES = {
    "canonical_selectable",
    "selectable",
    "performance_control",
    "diagnostic_nonselectable",
}


@dataclass(frozen=True)
class OracleObjectiveConfig:
    offline_kd_weight: float = 0.5
    variance_floor_weight: float = 0.01
    rate_weight: float = 0.02
    covariance_weight: float = 0.01
    trust_weight: float = 0.01
    temperature: float = 2.0
    rate_budget_enabled: bool = True
    contract: str = PARTICLE_VIEW_ORACLE_OBJECTIVE_CONTRACT

    def __post_init__(self) -> None:
        if self.offline_kd_weight not in OFFLINE_KD_SCREEN:
            raise ValueError("offline KD weight is outside the registered screen")
        if (
            self.variance_floor_weight != 0.01
            or self.trust_weight != 0.01
            or self.temperature != 2.0
        ):
            raise ValueError("canonical oracle objective constants changed")
        if self.rate_budget_enabled:
            if self.rate_weight != 0.02 or self.covariance_weight != 0.01:
                raise ValueError("canonical rate/covariance weights changed")
        elif self.rate_weight != 0.0 or self.covariance_weight != 0.0:
            raise ValueError("no-rate diagnostic must set both weights to zero")

    def to_payload(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "cross_entropy_weight": 1.0,
            "offline_kd_weight": self.offline_kd_weight,
            "variance_floor_weight": self.variance_floor_weight,
            "rate_weight": self.rate_weight,
            "covariance_weight": self.covariance_weight,
            "trust_weight": self.trust_weight,
            "temperature": self.temperature,
            "rate_budget_enabled": self.rate_budget_enabled,
        }


def classification_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float = 2.0,
) -> torch.Tensor:
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("offline-KD logit shapes differ")
    target = torch.softmax(teacher_logits.detach() / temperature, dim=-1)
    log_probability = torch.log_softmax(student_logits / temperature, dim=-1)
    return (
        torch.nn.functional.kl_div(
            log_probability, target, reduction="batchmean"
        )
        * temperature**2
    )


def oracle_discovery_loss(
    *,
    consumer_logits: torch.Tensor,
    labels: torch.Tensor,
    offline_logits: torch.Tensor,
    raw_centered_view: torch.Tensor,
    mask: torch.Tensor,
    trust_loss: torch.Tensor,
    config: OracleObjectiveConfig,
) -> dict[str, torch.Tensor]:
    regularizers = particle_view_rate_covariance_losses(
        raw_centered_view, mask
    )
    ce = torch.nn.functional.cross_entropy(consumer_logits, labels)
    kd = classification_kd_loss(
        consumer_logits, offline_logits, temperature=config.temperature
    )
    total = (
        ce
        + config.offline_kd_weight * kd
        + config.variance_floor_weight * regularizers["variance_floor_loss"]
        + config.rate_weight * regularizers["rate_loss"]
        + config.covariance_weight * regularizers["covariance_loss"]
        + config.trust_weight * trust_loss
    )
    if not torch.isfinite(total):
        raise FloatingPointError("oracle discovery objective is non-finite")
    return {
        "total": total,
        "cross_entropy": ce,
        "offline_kd": kd,
        "variance_floor": regularizers["variance_floor_loss"],
        "rate": regularizers["rate_loss"],
        "covariance": regularizers["covariance_loss"],
        "trust": trust_loss,
    }


def recovery_from_gains(
    oracle_gain: float, predicted_gain: float
) -> tuple[str, float | None]:
    if (
        not math.isfinite(oracle_gain)
        or not math.isfinite(predicted_gain)
        or oracle_gain <= 0.0
    ):
        return "undefined", None
    return "finite", predicted_gain / oracle_gain


@dataclass(frozen=True)
class TargetCandidateMetrics:
    run_id: str
    target_id: str
    bottleneck_width: int
    predicted_view_gain: float
    oracle_gain: float
    predicted_view_cross_entropy: float
    zero_view_accuracy: float
    predicted_view_accuracy: float
    oracle_accuracy: float
    a0_accuracy: float
    target_registration_sha256: str
    selection_status: str = "selectable"

    def to_payload(self) -> dict[str, Any]:
        for name in ("run_id", "target_id", "selection_status"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be nonempty")
        if self.bottleneck_width not in {1, 2, 4, 8}:
            raise ValueError("candidate bottleneck width is invalid")
        if self.selection_status not in _TARGET_SELECTION_STATUSES:
            raise ValueError("target candidate selection status is invalid")
        values = {
            name: float(getattr(self, name))
            for name in (
                "predicted_view_gain",
                "oracle_gain",
                "predicted_view_cross_entropy",
                "zero_view_accuracy",
                "predicted_view_accuracy",
                "oracle_accuracy",
                "a0_accuracy",
            )
        }
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("target candidate metrics must be finite")
        require_sha256(
            "target_registration_sha256", self.target_registration_sha256
        )
        status, recovery = recovery_from_gains(
            self.oracle_gain, self.predicted_view_gain
        )
        return {
            "contract": PARTICLE_VIEW_TARGET_METRICS_CONTRACT,
            "run_id": self.run_id,
            "target_id": self.target_id,
            "bottleneck_width": self.bottleneck_width,
            **values,
            "oracle_gain_over_independent_a0": (
                self.oracle_accuracy - self.a0_accuracy
            ),
            "predicted_gain_over_independent_a0": (
                self.predicted_view_accuracy - self.a0_accuracy
            ),
            "recovery_status": status,
            "recovered_fraction": recovery,
            "target_registration_sha256": self.target_registration_sha256,
            "selection_status": self.selection_status,
            "same_consumer_counterfactual": True,
        }


def build_target_metrics_artifact(
    metrics: TargetCandidateMetrics,
) -> dict[str, Any]:
    return with_content_hash(metrics.to_payload())


def build_target_metrics_from_counterfactual(
    *,
    counterfactual_metrics: Mapping[str, Any],
    run_id: str,
    target_id: str,
    bottleneck_width: int,
    a0_accuracy: float,
    target_registration_sha256: str,
    selection_status: str,
) -> dict[str, Any]:
    validate_content_hash(
        counterfactual_metrics,
        expected_contract="particle_view_counterfactual_metrics_v1",
    )
    if counterfactual_metrics.get("split") != "model_val_select":
        raise ValueError("target ranking metrics must use model_val_select")
    metrics = TargetCandidateMetrics(
        run_id=run_id,
        target_id=target_id,
        bottleneck_width=bottleneck_width,
        predicted_view_gain=float(
            counterfactual_metrics["predicted_view_gain"]
        ),
        oracle_gain=float(counterfactual_metrics["oracle_gain"]),
        predicted_view_cross_entropy=float(
            counterfactual_metrics["predicted_view"]["cross_entropy"]
        ),
        zero_view_accuracy=float(
            counterfactual_metrics["zero_view"]["accuracy"]
        ),
        predicted_view_accuracy=float(
            counterfactual_metrics["predicted_view"]["accuracy"]
        ),
        oracle_accuracy=float(
            counterfactual_metrics["true_view"]["accuracy"]
        ),
        a0_accuracy=float(a0_accuracy),
        target_registration_sha256=target_registration_sha256,
        selection_status=selection_status,
    )
    return build_target_metrics_artifact(metrics)


def _target_key(payload: Mapping[str, Any]):
    finite = payload["recovery_status"] == "finite"
    return (
        -float(payload["predicted_view_gain"]),
        0 if finite else 1,
        -float(payload["recovered_fraction"]) if finite else 0.0,
        -float(payload["oracle_gain"]),
        float(payload["predicted_view_cross_entropy"]),
        int(payload["bottleneck_width"]),
        str(payload["run_id"]),
        str(payload["target_id"]),
    )


def rank_target_candidates(
    candidates: Iterable[TargetCandidateMetrics | Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for candidate in candidates:
        if isinstance(candidate, TargetCandidateMetrics):
            rows.append(candidate.to_payload())
            continue
        row = dict(candidate)
        if "content_hash" in row:
            validate_content_hash(
                row, expected_contract=PARTICLE_VIEW_TARGET_METRICS_CONTRACT
            )
        reconstructed = TargetCandidateMetrics(
            run_id=row["run_id"],
            target_id=row["target_id"],
            bottleneck_width=int(row["bottleneck_width"]),
            predicted_view_gain=float(row["predicted_view_gain"]),
            oracle_gain=float(row["oracle_gain"]),
            predicted_view_cross_entropy=float(
                row["predicted_view_cross_entropy"]
            ),
            zero_view_accuracy=float(row["zero_view_accuracy"]),
            predicted_view_accuracy=float(row["predicted_view_accuracy"]),
            oracle_accuracy=float(row["oracle_accuracy"]),
            a0_accuracy=float(row["a0_accuracy"]),
            target_registration_sha256=str(
                row["target_registration_sha256"]
            ),
            selection_status=str(row["selection_status"]),
        ).to_payload()
        comparison = dict(row)
        comparison.pop("content_hash", None)
        if comparison != reconstructed:
            raise ValueError("target metric payload is not canonical")
        rows.append(row)
    if not rows:
        raise ValueError("target ranking requires candidates")
    ids = [row["target_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("target ranking contains duplicate target IDs")
    for row in rows:
        if row.get("contract") != PARTICLE_VIEW_TARGET_METRICS_CONTRACT:
            raise ValueError("target metric contract mismatch")
    return sorted(rows, key=_target_key)


def select_target_candidates(
    candidates: Iterable[TargetCandidateMetrics | Mapping[str, Any]],
    *,
    canonical_target_id: str,
    forward_count: int = 2,
) -> dict[str, Any]:
    ranked = rank_target_candidates(candidates)
    selectable = [
        row
        for row in ranked
        if row["selection_status"]
        in {"selectable", "canonical_selectable"}
    ]
    if len(selectable) < forward_count:
        raise ValueError("not enough structurally selectable targets")
    forwarded = selectable[:forward_count]
    canonical = next(
        (row for row in ranked if row["target_id"] == canonical_target_id),
        None,
    )
    if canonical is None:
        raise ValueError("canonical target is absent")
    if canonical["selection_status"] not in {
        "selectable",
        "canonical_selectable",
    }:
        raise ValueError("canonical target is not structurally selectable")
    if canonical["target_id"] not in {row["target_id"] for row in forwarded}:
        forwarded.append(canonical)
    controls = [
        row for row in ranked if row["selection_status"] == "performance_control"
    ]
    forwarded.extend(
        row
        for row in controls
        if row["target_id"] not in {
            selected["target_id"] for selected in forwarded
        }
    )
    artifact = with_content_hash(
        {
            "contract": PARTICLE_VIEW_TARGET_SELECTION_CONTRACT,
            "ranking_split": "model_val_select",
            "ranking_rule": [
                "highest_predicted_view_gain",
                "finite_recovery_then_highest_fraction",
                "highest_oracle_gain",
                "lowest_predicted_view_cross_entropy",
                "smaller_bottleneck",
                "lexicographic_run_id",
                "lexicographic_target_id",
            ],
            "ranked": ranked,
            "forwarded_target_ids": [
                row["target_id"] for row in forwarded
            ],
            "canonical_target_id": canonical_target_id,
            "performance_control_target_ids": [
                row["target_id"] for row in controls
            ],
            "quality_threshold_used_as_gate": False,
        }
    )
    return artifact


def build_scientific_warnings(
    metrics: TargetCandidateMetrics | Mapping[str, Any],
    *,
    graph_node: str,
    configuration_id: str,
    seed: int,
    split: str,
    supporting_metric_sha256: str,
    source_commit: str,
) -> list[dict[str, Any]]:
    row = (
        metrics.to_payload()
        if isinstance(metrics, TargetCandidateMetrics)
        else dict(metrics)
    )
    require_sha256("supporting_metric_sha256", supporting_metric_sha256)
    warnings = []
    specifications = []
    if float(row["oracle_gain"]) < 0.005:
        specifications.append(
            (
                "WARN_ORACLE_GAIN_BELOW_005",
                0.005,
                "The same-consumer privileged oracle gain is below 0.5 percentage points.",
                "Inspect target bandwidth, pair bias, tap source, and radius controls.",
            )
        )
    if float(row["oracle_gain"]) <= 0:
        specifications.append(
            (
                "WARN_ORACLE_GAIN_NONPOSITIVE",
                0.0,
                "The privileged view did not improve its own zero-view consumer endpoint.",
                "Inspect consumer reachability and shuffled/broadcast controls.",
            )
        )
    if row.get("recovery_status") == "finite" and float(
        row["recovered_fraction"]
    ) <= 0:
        specifications.append(
            (
                "WARN_RECOVERY_NONPOSITIVE",
                0.0,
                "The fixed HLT probe recovered no positive oracle gain.",
                "Inspect predictability, target rate, and recoverability co-design.",
            )
        )
    if not isinstance(source_commit, str) or not source_commit:
        raise ValueError("source_commit must be nonempty")
    timestamp = datetime.now(timezone.utc).isoformat()
    for code, threshold, interpretation, diagnostic in specifications:
        warnings.append(
            with_content_hash(
                {
                    "contract": PARTICLE_VIEW_WARNING_CONTRACT,
                    "warning_code": code,
                    "severity": "scientific_warning",
                    "graph_node": graph_node,
                    "configuration_id": configuration_id,
                    "seed": seed,
                    "split": split,
                    "observed_value": float(row["oracle_gain"])
                    if "ORACLE" in code
                    else row.get("recovered_fraction"),
                    "reference_value": threshold,
                    "declared_warning_threshold": threshold,
                    "interpretation": interpretation,
                    "suggested_diagnostic": diagnostic,
                    "supporting_metric_sha256": supporting_metric_sha256,
                    "utc_timestamp": timestamp,
                    "source_commit": source_commit,
                    "stops_execution": False,
                    "affects_selectability": False,
                }
            )
        )
    return warnings


def write_scientific_warnings(
    path: str | Path, warnings: Sequence[Mapping[str, Any]]
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        for warning in warnings:
            validate_content_hash(
                warning, expected_contract=PARTICLE_VIEW_WARNING_CONTRACT
            )
            handle.write(json.dumps(dict(warning), sort_keys=True) + "\n")


@dataclass(frozen=True)
class RecoverabilityCoDesignConfig:
    view_dim: int
    rich_dim: int = 160
    cycles: int = 12
    probe_steps_per_cycle: int = 2_000
    projection_steps_per_cycle: int = 500
    probe_learning_rate: float = 3.0e-4
    projection_learning_rate: float = 1.0e-4
    consumer_learning_rate: float = 3.0e-5
    probe_weight_decay: float = 1.0e-4
    projection_weight_decay: float = 1.0e-5
    consumer_weight_decay: float = 1.0e-4
    batch_size: int = 128
    gradient_clip: float = 1.0
    seed: int = 101
    contract: str = PARTICLE_VIEW_CODESIGN_CONTRACT

    def __post_init__(self) -> None:
        if self.view_dim not in {1, 2, 4, 8}:
            raise ValueError("co-design view width is invalid")
        locked = (
            self.rich_dim,
            self.cycles,
            self.probe_steps_per_cycle,
            self.projection_steps_per_cycle,
            self.batch_size,
        )
        if locked != (160, 12, 2_000, 500, 128):
            raise ValueError("co-design cycle/update contract changed")

    def to_payload(self) -> dict[str, Any]:
        return {
            **self.__dict__,
            "probe_projection_step_ratio": 4,
            "performance_early_termination": False,
            "rich_context_frozen": True,
            "consumer_and_probe_persist_across_cycles": True,
            "fresh_independent_probe_required_after_selection": True,
        }


class RecoverabilityCoDesignProjection(nn.Module):
    def __init__(self, config: RecoverabilityCoDesignConfig) -> None:
        super().__init__()
        self.config = config
        self.network = nn.Sequential(
            nn.LayerNorm(160),
            nn.Linear(160, 160),
            nn.GELU(),
            nn.Linear(160, config.view_dim),
        )
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, rich_context: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if rich_context.shape[:2] != mask.shape or rich_context.shape[2] != 160:
            raise ValueError("co-design rich context/mask shape mismatch")
        bounded = torch.tanh(self.network(rich_context))
        centered = masked_particle_mean_center(bounded, mask)
        return torch.where(mask[:, :, None], centered, torch.zeros_like(centered))


def co_design_schedule(config: RecoverabilityCoDesignConfig) -> list[dict[str, Any]]:
    return [
        {
            "cycle": cycle,
            "phase_order": ["probe", "projection_consumer"],
            "probe_optimizer_steps": config.probe_steps_per_cycle,
            "projection_consumer_optimizer_steps": config.projection_steps_per_cycle,
            "probe_updates": ["persistent_probe_only"],
            "projection_consumer_updates": [
                "projection",
                "persistent_discovery_consumer",
            ],
            "frozen_in_probe_phase": ["rich_context", "projection", "consumer"],
            "frozen_in_projection_phase": ["rich_context", "probe"],
        }
        for cycle in range(1, config.cycles + 1)
    ]


def co_design_projection_loss(
    *,
    consumer_logits: torch.Tensor,
    labels: torch.Tensor,
    offline_logits: torch.Tensor,
    raw_centered_view: torch.Tensor,
    probe_prediction: torch.Tensor,
    mask: torch.Tensor,
    trust_loss: torch.Tensor,
    oracle_config: OracleObjectiveConfig,
) -> dict[str, torch.Tensor]:
    """Locked projection/consumer phase loss for recoverability co-design."""

    oracle = oracle_discovery_loss(
        consumer_logits=consumer_logits,
        labels=labels,
        offline_logits=offline_logits,
        raw_centered_view=raw_centered_view,
        mask=mask,
        trust_loss=trust_loss,
        config=oracle_config,
    )
    agreement_huber = view_huber_loss(
        probe_prediction.detach(), raw_centered_view, mask
    )
    agreement_relational = view_relational_loss(
        probe_prediction.detach(), raw_centered_view, mask
    )
    total = (
        oracle["total"]
        + 0.5 * agreement_huber
        + 0.1 * agreement_relational
    )
    return {
        **oracle,
        "probe_agreement_huber": agreement_huber,
        "probe_agreement_relational": agreement_relational,
        "total": total,
    }


def build_codesign_ledger(
    *,
    config: RecoverabilityCoDesignConfig,
    rich_context_registration_sha256: str,
    provisional_head_registration_sha256: str,
    cycles: Sequence[Mapping[str, Any]],
    selected_cycle: int,
    final_projection_checkpoint_sha256: str,
) -> dict[str, Any]:
    """Authenticate every persistent co-design state and optimizer budget."""

    for name, value in (
        (
            "rich_context_registration_sha256",
            rich_context_registration_sha256,
        ),
        (
            "provisional_head_registration_sha256",
            provisional_head_registration_sha256,
        ),
        (
            "final_projection_checkpoint_sha256",
            final_projection_checkpoint_sha256,
        ),
    ):
        require_sha256(name, value)
    if len(cycles) != config.cycles:
        raise ValueError("co-design ledger requires all cycles")
    expected_cycles = set(range(1, config.cycles + 1))
    if {int(row["cycle"]) for row in cycles} != expected_cycles:
        raise ValueError("co-design ledger cycle inventory is incomplete")
    if selected_cycle not in expected_cycles:
        raise ValueError("selected co-design cycle is invalid")
    rows = []
    for raw in sorted(cycles, key=lambda row: int(row["cycle"])):
        row = dict(raw)
        for name in (
            "projection_checkpoint_sha256",
            "consumer_checkpoint_sha256",
            "probe_checkpoint_sha256",
        ):
            require_sha256(name, row[name])
        if (
            int(row["probe_optimizer_steps"])
            != config.probe_steps_per_cycle
            or int(row["projection_consumer_optimizer_steps"])
            != config.projection_steps_per_cycle
        ):
            raise ValueError("co-design optimizer-step budget changed")
        rows.append(row)
    return with_content_hash(
        {
            "contract": PARTICLE_VIEW_CODESIGN_LEDGER_CONTRACT,
            "config": config.to_payload(),
            "config_sha256": canonical_sha256(config.to_payload()),
            "rich_context_registration_sha256": rich_context_registration_sha256,
            "provisional_head_registration_sha256": provisional_head_registration_sha256,
            "cycles": rows,
            "selected_cycle": selected_cycle,
            "final_projection_checkpoint_sha256": final_projection_checkpoint_sha256,
            "performance_early_termination": False,
            "rich_context_frozen_for_all_cycles": True,
            "persistent_probe_consumer_across_cycles": True,
            "fresh_independent_probe_still_required": True,
        }
    )


def select_codesign_cycle(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != 12:
        raise ValueError("co-design selection requires all 12 cycles")
    def key(row):
        status, fraction = recovery_from_gains(
            float(row["oracle_gain"]), float(row["predicted_gain"])
        )
        return (
            -float(row["predicted_view_accuracy"]),
            0 if status == "finite" else 1,
            -float(fraction) if fraction is not None else 0.0,
            -float(row["oracle_accuracy"]),
            float(row["cross_entropy"]),
            int(row["cycle"]),
        )
    if {int(row["cycle"]) for row in rows} != set(range(1, 13)):
        raise ValueError("co-design cycle inventory is incomplete")
    return dict(min(rows, key=key))


def build_clean_consumer_registration(
    *,
    consumer_config: ParticleViewConsumerConfig,
    checkpoint_path: str | Path,
    a0_registration_sha256: str,
    coordinate_binding_sha256: str,
    selected_view_publication_sha256: str,
    normalizer_sha256: str,
    target_selection_sha256: str,
    train_identity_sha256: str,
    selected_epoch: int,
) -> dict[str, Any]:
    for name, value in (
        ("a0_registration_sha256", a0_registration_sha256),
        ("coordinate_binding_sha256", coordinate_binding_sha256),
        ("selected_view_publication_sha256", selected_view_publication_sha256),
        ("normalizer_sha256", normalizer_sha256),
        ("target_selection_sha256", target_selection_sha256),
        ("train_identity_sha256", train_identity_sha256),
    ):
        require_sha256(name, value)
    if selected_epoch <= 0:
        raise ValueError("clean consumer selected epoch must be positive")
    return with_content_hash(
        {
            "contract": PARTICLE_VIEW_CLEAN_CONSUMER_REGISTRATION_CONTRACT,
            "role": "Cview_clean",
            "consumer_config": consumer_config.to_payload(),
            "consumer_config_sha256": consumer_config.content_hash,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "a0_registration_sha256": a0_registration_sha256,
            "coordinate_binding_sha256": coordinate_binding_sha256,
            "selected_view_publication_sha256": selected_view_publication_sha256,
            "normalizer_sha256": normalizer_sha256,
            "target_selection_sha256": target_selection_sha256,
            "train_identity_sha256": train_identity_sha256,
            "selected_epoch": selected_epoch,
            "initialized_from_exact_a0": True,
            "trained_from_epoch_zero_in_final_coordinates": True,
            "raw_discovery_coordinates_seen": False,
            "live_generator_used": False,
            "maximum_epochs": 40,
            "early_stop_patience": 8,
            "checkpoint_selection_split": "model_val_stop",
        }
    )


def build_two_pass_candidate_artifact(
    *,
    target_registration_sha256: str,
    discovery_consumer_checkpoint_sha256: str,
    frozen_generator_checkpoint_sha256: str,
    provisional_normalizer_sha256: str,
    probe_consumer_checkpoint_sha256: str,
    recovery_probe_registration_sha256: str,
    model_val_select_metrics_sha256: str,
) -> dict[str, Any]:
    lineage = {
        "target_registration_sha256": target_registration_sha256,
        "discovery_consumer_checkpoint_sha256": (
            discovery_consumer_checkpoint_sha256
        ),
        "frozen_generator_checkpoint_sha256": (
            frozen_generator_checkpoint_sha256
        ),
        "provisional_normalizer_sha256": provisional_normalizer_sha256,
        "probe_consumer_checkpoint_sha256": (
            probe_consumer_checkpoint_sha256
        ),
        "recovery_probe_registration_sha256": (
            recovery_probe_registration_sha256
        ),
        "model_val_select_metrics_sha256": (
            model_val_select_metrics_sha256
        ),
    }
    for name, value in lineage.items():
        require_sha256(name, value)
    return with_content_hash(
        {
            "contract": PARTICLE_VIEW_TWO_PASS_CONTRACT,
            **lineage,
            "discovery_coordinate_system": "raw_bounded_centered",
            "probe_coordinate_system": "train_fit_standardized_clipped",
            "probe_consumer_reinitialized_from_a0": True,
            "normalizer_inserted_after_consumer_training": False,
            "recovery_probe_fixed_epochs": 8,
            "probe_consumer_fixed_epochs": 12,
            "ranking_split": "model_val_select",
        }
    )


__all__ = [
    "OFFLINE_KD_SCREEN",
    "PARTICLE_VIEW_CLEAN_CONSUMER_REGISTRATION_CONTRACT",
    "PARTICLE_VIEW_CODESIGN_LEDGER_CONTRACT",
    "PARTICLE_VIEW_CODESIGN_CONTRACT",
    "PARTICLE_VIEW_ORACLE_OBJECTIVE_CONTRACT",
    "PARTICLE_VIEW_TARGET_METRICS_CONTRACT",
    "PARTICLE_VIEW_TARGET_SELECTION_CONTRACT",
    "PARTICLE_VIEW_WARNING_CONTRACT",
    "PARTICLE_VIEW_TWO_PASS_CONTRACT",
    "OracleObjectiveConfig",
    "RecoverabilityCoDesignConfig",
    "RecoverabilityCoDesignProjection",
    "TargetCandidateMetrics",
    "build_clean_consumer_registration",
    "build_codesign_ledger",
    "build_scientific_warnings",
    "build_target_metrics_artifact",
    "build_target_metrics_from_counterfactual",
    "build_two_pass_candidate_artifact",
    "classification_kd_loss",
    "co_design_schedule",
    "co_design_projection_loss",
    "oracle_discovery_loss",
    "rank_target_candidates",
    "recovery_from_gains",
    "select_codesign_cycle",
    "select_target_candidates",
    "write_scientific_warnings",
]
