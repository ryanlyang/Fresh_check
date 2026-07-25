"""Step 8 semantic controls and adversarial-channel evidence.

The module completes the scientific boundary around the Step 7 HLG graph. It
does not introduce another primary architecture. Most runs use the exact
canonical A3 graph, with the declared B1 reliability-head addition; the
non-selectable TALT_A0 comparison deliberately reuses the simple C0 graph
against the same alternate clean consumer as TALT_A3.

Dense bridge fields and shuffled controls are allocation-local inputs.  The
only persistent distribution reference produced here is the locked 1,001 by
45 float64 quantile table.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
import hashlib
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .bridge import (
    BRIDGE_CHANNEL_ALL50,
    BRIDGE_CHANNEL_PHYSICAL45,
    BridgeScalers,
    MATCHED_WRONG_EVENT_MAP_CONTRACT,
    PREDICTION_ANCHORED_BRIDGE_SCALER_CONTRACT,
    physical_loss_groups,
)
from .bridge_campaign import (
    MEASUREMENT_MEASURED,
    MEASUREMENT_UNMEASURED,
    NORMAL_PILOT_BUDGET_BYTES,
    PAIRED3_HARD_CEILING_BYTES,
    PAIRED_SEED_IDS,
    record_registry_measurements,
    require_production_ready,
    resolve_registry_run,
    validate_campaign_registry,
)
from .bridge_consumer import STEP3_RUN_IDS
from .bridge_contracts import (
    canonical_json_bytes,
    canonical_sha256,
    sha256_file,
    validate_content_hash,
    with_content_hash,
)
from .bridge_evaluation import (
    ALL50_TEACHER_NAMESPACE,
    ALTERNATE_TEACHER_NAMESPACE,
    N3_F0_TEACHER_NAMESPACE,
    PREDICTION_ANCHORED_SELECTED_CONSUMER_CONTRACT,
    PREDICTION_ANCHORED_TEACHER_BINDING_CONTRACT,
    PRIMARY_TEACHER_NAMESPACE,
    classification_metrics,
    validate_teacher_binding,
)
from .bridge_logits import (
    PREDICTION_ANCHORED_LIVE_TEACHER_CONFIG_CONTRACT,
    PREDICTION_ANCHORED_TEACHER_LOGIT_CACHE_CONTRACT,
    verify_equal_field_zero_kd,
)
from .bridge_losses import (
    anchor_regularization,
    distillation_kl_loss,
    local_smoothness_loss,
    masked_group_balanced_huber,
)
from .bridge_reconstructor import (
    C0CorrectionConfig,
    C0CorrectionOutput,
    PHYSICAL_CHANNELS,
    PredictionAnchoredC0Correction,
    TorchBridgeScalers,
)
from .hierarchical_global_reconstructor import (
    ARCH_A3_HLG_PRIMARY,
    DIRECT_HLT,
    DIRECT_R0REP,
    BundleResourceReference,
    DirectHLGClassifier,
    DirectHLGConfig,
    HLGCorrectionOutput,
    PredictionAnchoredHLGCorrection,
    build_step7_hlg_correction_model,
    measure_step7_registry_states,
    STEP7_MEASURED_ARCHITECTURE_IDS,
)
from .hierarchical_reconstructor import (
    ARCH_A0M_CAPACITY_PARTICLE,
    build_step6_correction_model,
)


PREDICTION_ANCHORED_STEP8_RECIPE_CONTRACT = "prediction_anchored_step8_run_recipe_v1"
PREDICTION_ANCHORED_ALL50_MODEL_CONTRACT = "prediction_anchored_all50_hlg_model_v1"
PREDICTION_ANCHORED_STEP8_LINEAGE_CONTRACT = "prediction_anchored_step8_teacher_lineage_v1"
PREDICTION_ANCHORED_STEP8_CONTROL_CONTRACT = "prediction_anchored_step8_control_batch_v1"
PREDICTION_ANCHORED_PERTURBATION_CONTRACT = "prediction_anchored_small_field_perturbation_v1"
PREDICTION_ANCHORED_PERTURBATION_AUDIT_CONTRACT = "prediction_anchored_perturbation_audit_v1"
PREDICTION_ANCHORED_ALIGNMENT_CONTRACT = "prediction_anchored_bridge_alignment_v1"
PREDICTION_ANCHORED_QUANTILE_REFERENCE_CONTRACT = "prediction_anchored_bridge_quantiles_v1"
PREDICTION_ANCHORED_DISTRIBUTION_DISTANCE_CONTRACT = "prediction_anchored_bridge_distribution_distance_v1"
PREDICTION_ANCHORED_RELIABILITY_RESPONSE_CONTRACT = "prediction_anchored_reliability_response_v1"
PREDICTION_ANCHORED_STEP8_MEASUREMENT_CONTRACT = "prediction_anchored_step8_registry_measurement_v1"
PREDICTION_ANCHORED_STEP8_FIXED_STORAGE_CONTRACT = "prediction_anchored_step8_fixed_storage_v1"
PREDICTION_ANCHORED_STEP8_FIXED_STORAGE_MEASUREMENT_CONTRACT = "prediction_anchored_step8_fixed_storage_measurement_v1"
PREDICTION_ANCHORED_POST_TEACHER_RELEASE_CONTRACT = "prediction_anchored_post_teacher_release_v1"
PREDICTION_ANCHORED_STEP8_MINIATURE_CONTRACT = "prediction_anchored_step8_paired_miniature_v1"
PREDICTION_ANCHORED_STEP8_TRAIN_CONTRACT = "prediction_anchored_step8_training_v1"
PREDICTION_ANCHORED_GAIN_RECOVERY_CONTRACT = "prediction_anchored_gain_recovery_v1"
PREDICTION_ANCHORED_ADVERSARIAL_REPORT_CONTRACT = "prediction_anchored_adversarial_channel_report_v1"

A3_PRIMARY_ALIAS = "D10_XA3_full_primary"
A3_INTERACTION_RUN_IDS = (
    "D10_XA3_bridge_only",
    "D10_XA3_ce_only",
    "D10_XA3_kd_only",
    "D10_XA3_kd_bridge",
    "D10_XA3_kd_ce",
    A3_PRIMARY_ALIAS,
    "D10_XA3_full_no_warmup",
    "D10_XA3_full_no_smooth",
)
A3_ADDITIONAL_CANONICAL_RUN_IDS = tuple(
    value for value in A3_INTERACTION_RUN_IDS if value != A3_PRIMARY_ALIAS
)
ALL50_RUN_IDS = ("D10_B1_all50_fullhead", "D10_B2_all50_physical45_only")
ALTERNATE_A0_RUN_ID = "D10_TALT_A0"
ALTERNATE_A3_RUN_ID = "D10_TALT_A3"
ALTERNATE_RUN_ID = ALTERNATE_A3_RUN_ID
ALTERNATE_RUN_IDS = (ALTERNATE_A0_RUN_ID, ALTERNATE_A3_RUN_ID)
NEGATIVE_CONTROL_RUN_IDS = (
    "D10_N0_shuffled_logit_kd",
    "D10_N1_shuffled_bridge_field",
    "D10_N2_shuffled_primary",
    "D10_N3_nonprivileged_teacher_kd",
)
STEP8_SPECIAL_CANONICAL_RUN_IDS = (
    A3_ADDITIONAL_CANONICAL_RUN_IDS
    + ALL50_RUN_IDS
    + ALTERNATE_RUN_IDS
    + NEGATIVE_CONTROL_RUN_IDS
)
PERTURBATION_AUDIT_SEEDS = (9101, 9102, 9103, 9104)
PERTURBATION_MEAN_ACCURACY_LOSS_MAX = 0.002
PERTURBATION_WORST_ACCURACY_LOSS_MAX = 0.003
QUANTILE_COUNT = 1001
AUTHORIZED_DISTRIBUTION_SPLITS = (
    "model_val_stop",
    "model_val_select",
    "stack_val_consumer",
    "stack_val_deploy",
)


def _valid_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


@dataclass(frozen=True)
class Step8RunRecipe:
    run_id: str
    kd: float
    ce: float
    bridge: float
    true: float
    anchor: float
    smooth: float
    field_warmup: bool
    cache_namespace: str | None
    binding_kind: str | None
    channel_policy: str
    selectable_for_primary_deployment: bool
    architecture_id: str = ARCH_A3_HLG_PRIMARY
    control_kind: str | None = None
    all50_head: str | None = None
    conditional_parent: str | None = None

    def __post_init__(self) -> None:
        allowed = (
            set(A3_INTERACTION_RUN_IDS)
            | {ARCH_A3_HLG_PRIMARY}
            | set(ALL50_RUN_IDS)
            | set(ALTERNATE_RUN_IDS)
            | set(NEGATIVE_CONTROL_RUN_IDS)
        )
        if self.run_id not in allowed:
            raise ValueError(f"unknown Step 8 run ID {self.run_id!r}")
        for name in ("kd", "ce", "bridge", "true", "anchor", "smooth"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Step 8 coefficient {name} is invalid")
        if bool(self.cache_namespace is not None) != bool(self.kd > 0):
            raise ValueError("Step 8 target cache is present exactly when KD is nonzero")
        if self.channel_policy not in {BRIDGE_CHANNEL_PHYSICAL45, BRIDGE_CHANNEL_ALL50}:
            raise ValueError("Step 8 channel policy is invalid")
        if self.binding_kind not in {None, "primary", "all50", "alternate"}:
            raise ValueError("Step 8 binding kind is invalid")
        if self.control_kind is not None and self.run_id not in NEGATIVE_CONTROL_RUN_IDS:
            raise ValueError("only negative controls may declare a control kind")
        if self.all50_head not in {None, "full50", "physical45_only"}:
            raise ValueError("invalid all50 head semantics")
        if self.all50_head is not None and self.run_id not in ALL50_RUN_IDS:
            raise ValueError("only B1/B2 may declare all50 head semantics")
        if self.architecture_id not in {
            ARCH_A3_HLG_PRIMARY,
            "D10_A0_c0_delta",
        }:
            raise ValueError("invalid Step 8 architecture identity")
        if (
            self.architecture_id == "D10_A0_c0_delta"
            and self.run_id != ALTERNATE_A0_RUN_ID
        ):
            raise ValueError("only TALT_A0 may use the C0 architecture in Step 8")

    @property
    def canonical_run_id(self) -> str:
        return ARCH_A3_HLG_PRIMARY if self.run_id == A3_PRIMARY_ALIAS else self.run_id

    @property
    def requires_target_logit_cache(self) -> bool:
        return self.kd > 0

    def phase_coefficients(self, phase: str) -> dict[str, float]:
        if phase not in {"field_warmup", "distillation"}:
            raise ValueError("Step 8 phase must be field_warmup or distillation")
        if phase == "field_warmup":
            if not self.field_warmup:
                raise ValueError(f"{self.run_id} has no field warm-up")
            return {
                "kd": 0.0,
                "ce": 0.0,
                "bridge": float(self.bridge),
                "true": 0.0,
                "anchor": float(self.anchor),
                "smooth": float(self.smooth),
                "gate": 0.0,
            }
        return {
            "kd": float(self.kd),
            "ce": float(self.ce),
            "bridge": float(self.bridge),
            "true": float(self.true),
            "anchor": float(self.anchor),
            "smooth": float(self.smooth),
            "gate": 0.0,
        }

    def to_artifact(self) -> dict[str, Any]:
        return with_content_hash(
            {
                "contract": PREDICTION_ANCHORED_STEP8_RECIPE_CONTRACT,
                **asdict(self),
                "canonical_run_id": self.canonical_run_id,
                "architecture_id": self.architecture_id,
                "distillation_coefficients": self.phase_coefficients("distillation"),
                "warmup_coefficients": (
                    self.phase_coefficients("field_warmup") if self.field_warmup else None
                ),
                "gate_present": False,
                "paired_seed_ids": list(PAIRED_SEED_IDS),
                "teacher_parameters_frozen": True,
                "r0_parameters_frozen": True,
                "dense_control_persistence_allowed": False,
            }
        )


def _recipe(
    run_id: str,
    *,
    kd: float,
    ce: float,
    bridge: float,
    anchor: float = 0.0,
    smooth: float = 0.0,
    warmup: bool,
    namespace: str | None = None,
    kind: str | None = "primary",
    policy: str = BRIDGE_CHANNEL_PHYSICAL45,
    selectable: bool = True,
    control: str | None = None,
    all50_head: str | None = None,
    conditional_parent: str | None = None,
    architecture_id: str = ARCH_A3_HLG_PRIMARY,
) -> Step8RunRecipe:
    return Step8RunRecipe(
        run_id, kd, ce, bridge, 0.0, anchor, smooth, warmup,
        namespace if kd > 0 else None, kind, policy, selectable,
        architecture_id, control, all50_head, conditional_parent,
    )


def step8_run_recipes() -> dict[str, Step8RunRecipe]:
    primary = dict(
        kd=1.0, ce=0.5, bridge=0.2, anchor=0.02, smooth=0.01,
        warmup=True, namespace=PRIMARY_TEACHER_NAMESPACE,
    )
    recipes = [
        _recipe("D10_XA3_bridge_only", kd=0, ce=0, bridge=1, anchor=0.02, smooth=0.01, warmup=True),
        _recipe("D10_XA3_ce_only", kd=0, ce=1, bridge=0, warmup=False),
        _recipe("D10_XA3_kd_only", kd=1, ce=0, bridge=0, warmup=False, namespace=PRIMARY_TEACHER_NAMESPACE),
        _recipe("D10_XA3_kd_bridge", kd=1, ce=0, bridge=0.2, anchor=0.02, smooth=0.01, warmup=True, namespace=PRIMARY_TEACHER_NAMESPACE),
        _recipe("D10_XA3_kd_ce", kd=1, ce=0.5, bridge=0, warmup=False, namespace=PRIMARY_TEACHER_NAMESPACE),
        _recipe(A3_PRIMARY_ALIAS, **primary),
        _recipe(ARCH_A3_HLG_PRIMARY, **primary),
        _recipe("D10_XA3_full_no_warmup", **{**primary, "warmup": False}),
        _recipe("D10_XA3_full_no_smooth", **{**primary, "smooth": 0.0}),
        _recipe(
            "D10_B1_all50_fullhead", **{**primary, "namespace": ALL50_TEACHER_NAMESPACE},
            kind="all50", policy=BRIDGE_CHANNEL_ALL50, selectable=False, all50_head="full50",
        ),
        _recipe(
            "D10_B2_all50_physical45_only", **{**primary, "namespace": ALL50_TEACHER_NAMESPACE},
            kind="all50", policy=BRIDGE_CHANNEL_ALL50, selectable=False,
            all50_head="physical45_only",
        ),
        _recipe(
            ALTERNATE_A0_RUN_ID,
            **{**primary, "namespace": ALTERNATE_TEACHER_NAMESPACE},
            kind="alternate", selectable=False,
            conditional_parent="alternate_teacher_valid",
            architecture_id="D10_A0_c0_delta",
        ),
        _recipe(
            ALTERNATE_A3_RUN_ID, **{**primary, "namespace": ALTERNATE_TEACHER_NAMESPACE},
            kind="alternate", selectable=False, conditional_parent="alternate_teacher_valid",
        ),
        _recipe(
            "D10_N0_shuffled_logit_kd", kd=1, ce=0, bridge=0, warmup=False,
            namespace=PRIMARY_TEACHER_NAMESPACE, selectable=False,
            control="event_shuffled_privileged_logits",
        ),
        _recipe(
            "D10_N1_shuffled_bridge_field", kd=0, ce=0, bridge=1,
            anchor=0.02, smooth=0.01, warmup=True, selectable=False,
            control="event_shuffled_bridge_group_scale_preserved",
        ),
        _recipe(
            "D10_N2_shuffled_primary", **primary, selectable=False,
            control="same_wrong_event_map_logits_and_bridge",
        ),
        _recipe(
            "D10_N3_nonprivileged_teacher_kd", kd=1, ce=0, bridge=0,
            warmup=False, namespace=N3_F0_TEACHER_NAMESPACE, selectable=False,
            control="selected_teacher_on_f0_self_distillation",
        ),
    ]
    output = {recipe.run_id: recipe for recipe in recipes}
    if len(output) != len(recipes):
        raise AssertionError("duplicate Step 8 recipe ID")
    return output


def resolve_step8_run_recipe(run_id: str) -> Step8RunRecipe:
    try:
        return step8_run_recipes()[str(run_id)]
    except KeyError as exc:
        raise ValueError(f"unknown Step 8 run ID {run_id!r}") from exc


def validate_step8_registry_semantics(registry: Mapping[str, Any]) -> dict[str, Any]:
    validation = validate_campaign_registry(registry)
    canonical_ids = (
        (ARCH_A3_HLG_PRIMARY,)
        + A3_ADDITIONAL_CANONICAL_RUN_IDS
        + ALL50_RUN_IDS
        + ALTERNATE_RUN_IDS
        + NEGATIVE_CONTROL_RUN_IDS
    )
    if len(canonical_ids) != 16 or len(set(canonical_ids)) != len(canonical_ids):
        raise AssertionError("Step 8 canonical semantic inventory changed")
    rows = {}
    for run_id in canonical_ids:
        recipe = resolve_step8_run_recipe(run_id)
        row = resolve_registry_run(registry, run_id)
        if row["selectable_for_primary_deployment"] != recipe.selectable_for_primary_deployment:
            raise ValueError(f"Step 8 recipe/registry selectability differs for {run_id}")
        if row["conditional_parent"] != recipe.conditional_parent:
            raise ValueError(f"Step 8 recipe/registry conditional parent differs for {run_id}")
        rows[run_id] = {
            "scientific_role": row["scientific_role"],
            "selectable_for_primary_deployment": row["selectable_for_primary_deployment"],
            "execution_status": row["execution_status"],
            "recipe_sha256": recipe.to_artifact()["content_hash"],
        }
    if resolve_registry_run(registry, A3_PRIMARY_ALIAS)["content_hash"] != resolve_registry_run(
        registry, ARCH_A3_HLG_PRIMARY
    )["content_hash"]:
        raise AssertionError("Step 8 registry did not deduplicate canonical A3 alias")
    return with_content_hash(
        {
            "contract": "prediction_anchored_step8_registry_semantics_v1",
            "registry_sha256": registry["content_hash"],
            "registry_counts": {
                "configuration_count": validation["configuration_count"],
                "reconstruction_breadth_count": validation["reconstruction_breadth_count"],
                "post_teacher_configuration_count": validation["post_teacher_configuration_count"],
            },
            "canonical_semantic_rows": rows,
            "a3_alias_deduplicated": True,
            "all_semantic_rows_have_explicit_selectability": True,
        }
    )


class TorchAll50CorrectionScalers(torch.nn.Module):
    """All-50 scaler buffers; only the five diagnostic coordinates are exposed."""

    def __init__(self, artifact: Mapping[str, Any]) -> None:
        super().__init__()
        validate_content_hash(artifact)
        scaler = BridgeScalers.from_artifact(artifact)
        if scaler.channel_policy != BRIDGE_CHANNEL_ALL50:
            raise ValueError("B1/B2 require the all50 scaler artifact")
        self.artifact_sha256 = str(artifact["content_hash"])
        for name in ("sigma_delta", "trust_scale"):
            self.register_buffer(
                name,
                torch.as_tensor(getattr(scaler, name)[45:], dtype=torch.float32),
                persistent=True,
            )
        self.register_buffer(
            "active",
            torch.as_tensor(scaler.active[45:], dtype=torch.bool),
            persistent=True,
        )

    def physical_correction(
        self, standardized_raw: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if standardized_raw.shape != (*mask.shape, 5):
            raise ValueError("all50 reliability head must emit [B,P,5]")
        raw = standardized_raw * self.sigma_delta
        bounded = self.trust_scale * torch.tanh(raw / self.trust_scale)
        bounded = bounded * self.active.to(dtype=bounded.dtype)
        bounded = bounded.masked_fill(~mask.unsqueeze(-1), 0.0)
        saturated = (
            (bounded.abs() >= 0.99 * self.trust_scale)
            & mask.unsqueeze(-1)
            & self.active
        )
        return bounded, saturated


@dataclass(frozen=True)
class All50CorrectionOutput:
    f_hat: torch.Tensor
    physical_correction: torch.Tensor
    reliability_correction: torch.Tensor
    standardized_raw_correction: torch.Tensor
    standardized_raw_reliability_correction: torch.Tensor
    hidden: torch.Tensor
    mask: torch.Tensor
    saturation_mask: torch.Tensor
    reliability_saturation_mask: torch.Tensor
    diagnostics: Mapping[str, Any]
    base_output: HLGCorrectionOutput


class PredictionAnchoredAll50HLG(torch.nn.Module):
    """Canonical A3 with B1 full-head or B2 physical-only semantics."""

    def __init__(
        self,
        run_id: str,
        *,
        physical45_scaler_artifact: Mapping[str, Any],
        all50_scaler_artifact: Mapping[str, Any],
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if run_id not in ALL50_RUN_IDS:
            raise ValueError("all50 HLG run must be B1 or B2")
        self.run_id = str(run_id)
        self.base = build_step7_hlg_correction_model(
            ARCH_A3_HLG_PRIMARY,
            scaler_artifact=physical45_scaler_artifact,
            dropout=float(dropout),
        )
        self.all50_scalers = TorchAll50CorrectionScalers(all50_scaler_artifact)
        if self.run_id == ALL50_RUN_IDS[0]:
            self.reliability_head: torch.nn.Module | None = torch.nn.Sequential(
                torch.nn.Linear(160, 64),
                torch.nn.GELU(),
                torch.nn.Dropout(float(dropout)),
                torch.nn.Linear(64, 5),
            )
            final = self.reliability_head[-1]
            assert isinstance(final, torch.nn.Linear)
            torch.nn.init.zeros_(final.weight)
            torch.nn.init.zeros_(final.bias)
        else:
            self.reliability_head = None

    @property
    def full50_reachable(self) -> bool:
        return self.reliability_head is not None

    def config_artifact(self) -> dict[str, Any]:
        base = self.base.config_artifact()
        return with_content_hash(
            {
                "contract": PREDICTION_ANCHORED_ALL50_MODEL_CONTRACT,
                "run_id": self.run_id,
                "canonical_a3_model_config_sha256": base["content_hash"],
                "canonical_a3_model_config": base,
                "all50_scaler_sha256": self.all50_scalers.artifact_sha256,
                "head": [160, 64, 5] if self.full50_reachable else None,
                "head_final_layer_zero_initialized": self.full50_reachable,
                "physical_head_is_exact_canonical_a3": True,
                "channel_policy": BRIDGE_CHANNEL_ALL50,
                "full50_equal_field_reachable": self.full50_reachable,
                "reliability_coordinates_treated_as_continuous": self.full50_reachable,
                "reliability_thresholding_applied": False,
                "selectable_for_primary_deployment": False,
                "oracle_input_present": False,
            }
        )

    def forward(
        self,
        hlt_tokens: torch.Tensor,
        mask: torch.Tensor,
        f0: torch.Tensor,
        h0: torch.Tensor,
    ) -> All50CorrectionOutput:
        base = self.base(hlt_tokens, mask, f0, h0)
        if self.reliability_head is None:
            raw_s = base.f_hat.new_zeros((*base.mask.shape, 5))
            correction_s = raw_s
            saturation_s = torch.zeros_like(raw_s, dtype=torch.bool)
        else:
            raw_s = self.reliability_head(base.reasoning_state.readback)
            raw_s = raw_s.masked_fill(~base.mask.unsqueeze(-1), 0.0)
            correction_s, saturation_s = self.all50_scalers.physical_correction(
                raw_s, base.mask
            )
        f_hat = torch.cat(
            (base.f_hat[..., :45], f0.detach()[..., 45:] + correction_s), dim=-1
        ).masked_fill(~base.mask.unsqueeze(-1), 0.0)
        saturation_p = base.saturation_mask
        assert saturation_p is not None
        saturation = torch.cat((saturation_p, saturation_s), dim=-1)
        return All50CorrectionOutput(
            f_hat=f_hat,
            physical_correction=base.physical_correction,
            reliability_correction=correction_s,
            standardized_raw_correction=base.standardized_raw_correction,
            standardized_raw_reliability_correction=raw_s,
            hidden=base.hidden,
            mask=base.mask,
            saturation_mask=saturation,
            reliability_saturation_mask=saturation_s,
            diagnostics={
                **base.diagnostics,
                "contract": PREDICTION_ANCHORED_ALL50_MODEL_CONTRACT,
                "run_id": self.run_id,
                "full50_equal_field_reachable": self.full50_reachable,
                "reliability_channels_exact_pass_through": not self.full50_reachable,
                "reliability_thresholding_applied": False,
                "reliability_trust_bound_enabled": self.full50_reachable,
                "all50_scaler_sha256": self.all50_scalers.artifact_sha256,
            },
            base_output=base,
        )


def all50_group_balanced_huber(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    physical_scalers: TorchBridgeScalers,
    all50_scalers: TorchAll50CorrectionScalers,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    physical, groups = masked_group_balanced_huber(
        prediction, target, mask, physical_scalers.sigma_delta
    )
    valid = mask.to(device=prediction.device, dtype=prediction.dtype).unsqueeze(-1)
    standardized = (
        prediction[..., 45:] - target[..., 45:]
    ) / all50_scalers.sigma_delta.to(prediction)
    reliability_element = F.smooth_l1_loss(
        standardized, torch.zeros_like(standardized), reduction="none", beta=1.0
    )
    reliability = (reliability_element * valid).sum() / torch.clamp(
        valid.sum() * 5, min=1.0
    )
    all_groups = {**groups, "diagnostic.reliability5": reliability}
    # Reconstruct the exact 13-group mean rather than weighting the already
    # averaged physical term as one group.
    return torch.stack(list(all_groups.values())).mean(), all_groups


def all50_anchor_regularization(
    f_hat: torch.Tensor,
    f0: torch.Tensor,
    mask: torch.Tensor,
    physical_scalers: TorchBridgeScalers,
    all50_scalers: TorchAll50CorrectionScalers,
) -> torch.Tensor:
    physical_groups = []
    squared = (f_hat[..., :45] - f0[..., :45].detach()).square()
    squared = squared / (physical_scalers.trust_scale[:45].to(squared).square() + 1e-12)
    valid = mask.to(device=squared.device, dtype=squared.dtype).unsqueeze(-1)
    channel_mean = (squared * valid).sum((0, 1)) / torch.clamp(valid.sum((0, 1)), min=1.0)
    for indices in physical_loss_groups().values():
        physical_groups.append(channel_mean[indices].mean())
    reliability = (f_hat[..., 45:] - f0[..., 45:].detach()).square()
    reliability = reliability / (all50_scalers.trust_scale.to(reliability).square() + 1e-12)
    reliability_mean = (reliability * valid).sum() / torch.clamp(valid.sum() * 5, min=1.0)
    return torch.stack((*physical_groups, reliability_mean)).mean()


def compute_step8_objective(
    output: HLGCorrectionOutput | All50CorrectionOutput | C0CorrectionOutput,
    batch: Mapping[str, Any],
    recipe: Step8RunRecipe,
    *,
    phase: str,
    live_logits: torch.Tensor | None = None,
    target_logits: torch.Tensor | None = None,
    temperature: float = 2.0,
    all50_scalers: TorchAll50CorrectionScalers | None = None,
    physical_scalers: TorchBridgeScalers | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    coefficients = recipe.phase_coefficients(phase)
    zero = output.f_hat.sum() * 0.0
    components = {name: zero for name in coefficients}
    bridge_groups: dict[str, torch.Tensor] = {}
    smooth_groups: dict[str, torch.Tensor] = {}
    if recipe.channel_policy == BRIDGE_CHANNEL_ALL50:
        if all50_scalers is None or physical_scalers is None:
            raise ValueError("B1/B2 objective requires both physical45 and all50 scalers")
    if coefficients["kd"] > 0:
        if live_logits is None or target_logits is None:
            raise ValueError(f"{recipe.run_id} requires bound live and target logits")
        components["kd"] = distillation_kl_loss(
            live_logits, target_logits, temperature=temperature
        )
    elif target_logits is not None:
        raise ValueError(f"{recipe.run_id} has zero KD and must not load target logits")
    if coefficients["ce"] > 0:
        if live_logits is None or "labels" not in batch:
            raise ValueError(f"{recipe.run_id} requires live logits and labels")
        labels = torch.as_tensor(batch["labels"], dtype=torch.long, device=live_logits.device)
        components["ce"] = F.cross_entropy(live_logits, labels)
    if coefficients["bridge"] > 0:
        if "bridge_fields" not in batch:
            raise ValueError(f"{recipe.run_id} requires allocation-local bridge fields")
        target = torch.as_tensor(
            batch["bridge_fields"], dtype=output.f_hat.dtype, device=output.f_hat.device
        )
        if recipe.channel_policy == BRIDGE_CHANNEL_ALL50:
            assert all50_scalers is not None and physical_scalers is not None
            components["bridge"], bridge_groups = all50_group_balanced_huber(
                output.f_hat, target, output.mask, physical_scalers, all50_scalers
            )
        else:
            if physical_scalers is None:
                raise ValueError("physical45 Step 8 objective requires physical scalers")
            components["bridge"], bridge_groups = masked_group_balanced_huber(
                output.f_hat, target, output.mask, physical_scalers.sigma_delta
            )
    if coefficients["true"] != 0:
        raise AssertionError("Step 8 semantic recipes have no true-field term")
    if coefficients["anchor"] > 0:
        if physical_scalers is None:
            raise ValueError("Step 8 anchor requires physical scalers")
        f0 = torch.as_tensor(batch["f0"], dtype=output.f_hat.dtype, device=output.f_hat.device)
        if recipe.channel_policy == BRIDGE_CHANNEL_ALL50:
            assert all50_scalers is not None
            components["anchor"] = all50_anchor_regularization(
                output.f_hat, f0, output.mask, physical_scalers, all50_scalers
            )
        else:
            components["anchor"] = anchor_regularization(
                output.f_hat, f0, output.mask, physical_scalers.trust_scale
            )
    if coefficients["smooth"] > 0:
        if physical_scalers is None:
            raise ValueError("Step 8 smoothness requires physical scalers")
        components["smooth"], smooth_groups = local_smoothness_loss(
            output.physical_correction,
            torch.as_tensor(batch["hlt_tokens"], dtype=output.f_hat.dtype, device=output.f_hat.device),
            output.mask,
            physical_scalers.sigma_delta,
        )
    weighted = {name: components[name] * coefficients[name] for name in components}
    total = torch.stack(list(weighted.values())).sum()
    return total, {
        "run_id": recipe.run_id,
        "canonical_run_id": recipe.canonical_run_id,
        "phase": phase,
        "coefficients": coefficients,
        "raw_components": {name: float(value.detach().cpu()) for name, value in components.items()},
        "weighted_components": {name: float(value.detach().cpu()) for name, value in weighted.items()},
        "total": float(total.detach().cpu()),
        "bridge_groups": {name: float(value.detach().cpu()) for name, value in bridge_groups.items()},
        "smooth_groups": {name: float(value.detach().cpu()) for name, value in smooth_groups.items()},
        "channel_policy": recipe.channel_policy,
        "gate_present": False,
        "temperature": float(temperature),
    }


def verify_all50_equal_field_semantics(
    run_id: str,
    *,
    target_logits: torch.Tensor | np.ndarray,
    live_logits: torch.Tensor | np.ndarray,
    target_fields: torch.Tensor | np.ndarray,
    live_fields: torch.Tensor | np.ndarray,
    temperature: float = 2.0,
) -> dict[str, Any]:
    """Make the B1 zero-KD invariant and B2 irreducible mismatch explicit."""

    if run_id not in ALL50_RUN_IDS:
        raise ValueError("all50 invariant audit requires B1 or B2")
    target = torch.as_tensor(target_fields).float()
    live = torch.as_tensor(live_fields).float()
    if target.shape != live.shape or target.ndim != 3 or target.shape[-1] != 50:
        raise ValueError("all50 invariant fields must align as [B,P,50]")
    reliability_mismatch = float(torch.max(torch.abs(target[..., 45:] - live[..., 45:])).item())
    if run_id == ALL50_RUN_IDS[0]:
        if not torch.equal(target, live):
            raise AssertionError("B1 equal-field audit was not supplied equal full-50 fields")
        kd = verify_equal_field_zero_kd(
            target_logits, live_logits, temperature=temperature
        )
        return with_content_hash(
            {
                "contract": "prediction_anchored_all50_equal_field_semantics_v1",
                "run_id": run_id,
                "equal_field_zero_kd_applicable": True,
                "full50_fields_equal": True,
                "reliability_mismatch_max_abs": 0.0,
                "kd_audit": kd,
                "selectable_for_primary_deployment": False,
            }
        )
    if reliability_mismatch <= 0:
        raise AssertionError("B2 audit must expose its irreducible reliability mismatch")
    return with_content_hash(
        {
            "contract": "prediction_anchored_all50_equal_field_semantics_v1",
            "run_id": run_id,
            "equal_field_zero_kd_applicable": False,
            "full50_fields_equal": False,
            "reliability_mismatch_max_abs": reliability_mismatch,
            "kd_audit": None,
            "reason": "physical45 live graph cannot supply privileged reliability-five endpoint",
            "selectable_for_primary_deployment": False,
        }
    )


def validate_step8_teacher_lineage(
    recipe: Step8RunRecipe,
    *,
    binding: Mapping[str, Any] | None,
    cache_manifest: Mapping[str, Any] | None,
    live_teacher_config: Mapping[str, Any] | None,
    primary_selection: Mapping[str, Any] | None = None,
    all50_scaler_artifact: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the immutable binding -> cache -> live-run provenance chain."""

    if recipe.binding_kind is None:
        if any(value is not None for value in (binding, cache_manifest, live_teacher_config)):
            raise ValueError("teacher-free Step 8 recipe must not receive teacher artifacts")
        return with_content_hash(
            {
                "contract": PREDICTION_ANCHORED_STEP8_LINEAGE_CONTRACT,
                "run_recipe_sha256": recipe.to_artifact()["content_hash"],
                "teacher_free": True,
                "cache_loaded": False,
            }
        )
    if binding is None or live_teacher_config is None:
        raise FileNotFoundError("Step 8 teacher-bound run is missing binding/live config")
    validate_teacher_binding(
        binding,
        expected_kind=recipe.binding_kind,
        primary_selection=primary_selection if recipe.binding_kind == "primary" else None,
    )
    if recipe.binding_kind == "all50":
        if all50_scaler_artifact is None:
            raise FileNotFoundError("all50 semantic run requires its bound correction scaler")
        validate_content_hash(
            all50_scaler_artifact,
            expected_contract=PREDICTION_ANCHORED_BRIDGE_SCALER_CONTRACT,
        )
        if (
            all50_scaler_artifact.get("channel_policy") != BRIDGE_CHANNEL_ALL50
            or binding.get("all50_correction_scaler_sha256")
            != all50_scaler_artifact["content_hash"]
        ):
            raise ValueError("all50 model scaler is not the one embedded in its teacher binding")
    elif all50_scaler_artifact is not None:
        raise ValueError("non-all50 Step 8 lineage rejects an all50 scaler")
    validate_content_hash(
        live_teacher_config,
        expected_contract=PREDICTION_ANCHORED_LIVE_TEACHER_CONFIG_CONTRACT,
    )
    if live_teacher_config.get("teacher_binding_sha256") != binding["content_hash"]:
        raise ValueError("Step 8 live teacher points to another binding")
    if live_teacher_config.get("checkpoint_sha256") != binding["checkpoint_sha256"]:
        raise ValueError("Step 8 live teacher checkpoint differs from binding")
    if recipe.requires_target_logit_cache:
        if cache_manifest is None:
            raise FileNotFoundError("Step 8 KD recipe is missing its target-logit cache")
        validate_content_hash(
            cache_manifest,
            expected_contract=PREDICTION_ANCHORED_TEACHER_LOGIT_CACHE_CONTRACT,
        )
        exact = {
            "cache_namespace": recipe.cache_namespace,
            "all50_correction_scaler_sha256": (
                None if all50_scaler_artifact is None else all50_scaler_artifact["content_hash"]
            ),
            "teacher_binding_sha256": binding["content_hash"],
            "checkpoint_sha256": binding["checkpoint_sha256"],
            "live_checkpoint_sha256": binding["checkpoint_sha256"],
            "channel_policy": recipe.channel_policy,
            "same_checkpoint_target_and_live": True,
            "checkpoint_refit_forbidden": True,
            "binding_created_before_cache": True,
        }
        for name, expected in exact.items():
            if cache_manifest.get(name) != expected:
                raise ValueError(f"Step 8 lineage changed cache field {name}")
        if recipe.run_id == "D10_N3_nonprivileged_teacher_kd":
            if (
                cache_manifest.get("field_condition") != "f0"
                or float(cache_manifest.get("rho_endpoint", -1)) != 0.0
            ):
                raise ValueError("N3 requires the dedicated selected-teacher-on-f0 cache")
        elif cache_manifest.get("field_condition") != "bridge_0.100":
            raise ValueError("privileged Step 8 KD cache is not evaluated on b_0.10")
    elif cache_manifest is not None:
        raise ValueError("zero-KD Step 8 recipe must not load a target-logit cache")
    return with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_STEP8_LINEAGE_CONTRACT,
            "run_recipe_sha256": recipe.to_artifact()["content_hash"],
            "binding_kind": recipe.binding_kind,
            "teacher_binding_sha256": binding["content_hash"],
            "checkpoint_sha256": binding["checkpoint_sha256"],
            "cache_manifest_sha256": None if cache_manifest is None else cache_manifest["content_hash"],
            "cache_namespace": recipe.cache_namespace,
            "all50_correction_scaler_sha256": (
                None if all50_scaler_artifact is None else all50_scaler_artifact["content_hash"]
            ),
            "target_and_live_checkpoint_identical": True,
            "binding_to_cache_to_run_acyclic": binding.get("cache_artifact_sha256") is None,
            "checkpoint_refit_forbidden": True,
        }
    )


def _wrong_permutation(wrong_event_map: Mapping[str, Any], count: int) -> np.ndarray:
    validate_content_hash(wrong_event_map, expected_contract=MATCHED_WRONG_EVENT_MAP_CONTRACT)
    permutation = np.asarray(wrong_event_map.get("permutation"), dtype=np.int64)
    if (
        permutation.shape != (count,)
        or set(permutation.tolist()) != set(range(count))
        or np.any(permutation == np.arange(count))
    ):
        raise ValueError("Step 8 control requires a complete aligned derangement")
    return permutation


def _shuffle_bridge_group_scale_preserved(
    f0: np.ndarray,
    bridge: np.ndarray,
    mask: np.ndarray,
    permutation: np.ndarray,
) -> np.ndarray:
    anchor = np.asarray(f0, dtype=np.float32)
    target = np.asarray(bridge, dtype=np.float32)
    valid = np.asarray(mask, dtype=bool)
    if target.shape != anchor.shape or anchor.shape[-1] != 50 or valid.shape != anchor.shape[:2]:
        raise ValueError("Step 8 shuffled bridge inputs do not align")
    original = target[..., :45] - anchor[..., :45]
    shuffled = original[permutation].copy()
    shuffled *= valid[permutation, :, None]
    shuffled *= valid[:, :, None]
    for indices in physical_loss_groups().values():
        source_values = original[..., indices][valid]
        shuffled_values = shuffled[..., indices][valid]
        source_norm = float(np.linalg.norm(source_values.astype(np.float64)))
        shuffled_norm = float(np.linalg.norm(shuffled_values.astype(np.float64)))
        if source_norm == 0:
            shuffled[..., indices] = 0.0
        elif shuffled_norm == 0:
            raise ValueError("matched bridge shuffle erased a nonzero semantic group")
        else:
            shuffled[..., indices] *= np.float32(source_norm / shuffled_norm)
    output = anchor.copy()
    output[..., :45] += shuffled
    output[..., 45:] = anchor[..., 45:]
    output[~valid] = 0.0
    return output


def prepare_step8_control_batch(
    run_id: str,
    batch: Mapping[str, Any],
    *,
    target_logits: np.ndarray | torch.Tensor | None,
    wrong_event_map: Mapping[str, Any] | None,
    target_cache_namespace: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply one declared control in RAM and return no persistable dense tensor."""

    recipe = resolve_step8_run_recipe(run_id)
    if run_id not in NEGATIVE_CONTROL_RUN_IDS:
        raise ValueError("control batch preparation requires N0/N1/N2/N3")
    output = dict(batch)
    count = int(np.asarray(batch["f0"]).shape[0])
    permutation = None
    if run_id in {NEGATIVE_CONTROL_RUN_IDS[0], NEGATIVE_CONTROL_RUN_IDS[1], NEGATIVE_CONTROL_RUN_IDS[2]}:
        if wrong_event_map is None:
            raise FileNotFoundError(f"{run_id} requires its matched wrong-event map")
        permutation = _wrong_permutation(wrong_event_map, count)
    if run_id in {NEGATIVE_CONTROL_RUN_IDS[0], NEGATIVE_CONTROL_RUN_IDS[2]}:
        if target_logits is None:
            raise ValueError(f"{run_id} requires privileged target logits")
        output["target_logits"] = np.asarray(target_logits, dtype=np.float32)[permutation]
    elif run_id == NEGATIVE_CONTROL_RUN_IDS[3]:
        if target_logits is None or target_cache_namespace != N3_F0_TEACHER_NAMESPACE:
            raise ValueError("N3 requires its dedicated nonprivileged f0 cache")
        output["target_logits"] = np.asarray(target_logits, dtype=np.float32)
    elif target_logits is not None:
        raise ValueError("N1 is bridge-only and must not load target logits")
    if run_id in {NEGATIVE_CONTROL_RUN_IDS[1], NEGATIVE_CONTROL_RUN_IDS[2]}:
        output["bridge_fields"] = _shuffle_bridge_group_scale_preserved(
            np.asarray(batch["f0"]),
            np.asarray(batch["bridge_fields"]),
            np.asarray(batch["mask"]),
            permutation,
        )
    control_artifact = with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_STEP8_CONTROL_CONTRACT,
            "run_id": run_id,
            "recipe_sha256": recipe.to_artifact()["content_hash"],
            "control_kind": recipe.control_kind,
            "wrong_event_map_sha256": None if wrong_event_map is None else wrong_event_map["content_hash"],
            "target_cache_namespace": target_cache_namespace,
            "same_wrong_event_map_for_logits_and_bridge": run_id == NEGATIVE_CONTROL_RUN_IDS[2],
            "bridge_group_marginal_l2_scale_preserved": run_id in NEGATIVE_CONTROL_RUN_IDS[1:3],
            "allocation_local_dense_outputs_only": True,
            "persistent_dense_fields_written": False,
        }
    )
    return output, control_artifact


def validate_four_control_matching(
    *,
    model_config_sha256: Mapping[str, str],
    optimizer_steps: Mapping[str, int],
    paired_seed_ids: Mapping[str, Sequence[int]],
) -> dict[str, Any]:
    positives = {
        NEGATIVE_CONTROL_RUN_IDS[0]: "D10_XA3_kd_only",
        NEGATIVE_CONTROL_RUN_IDS[1]: "D10_XA3_bridge_only",
        NEGATIVE_CONTROL_RUN_IDS[2]: ARCH_A3_HLG_PRIMARY,
        NEGATIVE_CONTROL_RUN_IDS[3]: "D10_XA3_kd_only",
    }
    rows = {}
    for control, positive in positives.items():
        for mapping, label in (
            (model_config_sha256, "model config"),
            (optimizer_steps, "optimizer steps"),
            (paired_seed_ids, "paired seeds"),
        ):
            if control not in mapping or positive not in mapping:
                raise ValueError(f"four-control matching lacks {label} for {control}/{positive}")
        passed = (
            model_config_sha256[control] == model_config_sha256[positive]
            and int(optimizer_steps[control]) == int(optimizer_steps[positive])
            and tuple(paired_seed_ids[control]) == tuple(paired_seed_ids[positive]) == PAIRED_SEED_IDS
        )
        if not passed:
            raise ValueError(f"{control} is not compute/model/seed matched to {positive}")
        rows[control] = {"positive_run_id": positive, "passed": True}
    return with_content_hash(
        {
            "contract": "prediction_anchored_step8_control_matching_v1",
            "rows": rows,
            "all_four_matched": True,
        }
    )


@dataclass(frozen=True)
class Step8TrainConfig:
    run_id: str
    paired_seed_id: int
    stack_train_distill_manifest_sha256: str
    model_val_stop_manifest_sha256: str
    distillation_steps: int
    field_warmup_steps: int
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-2
    gradient_clip_norm: float = 1.0
    unique_training_jets: int = 250_000

    def __post_init__(self) -> None:
        recipe = resolve_step8_run_recipe(self.run_id)
        if int(self.paired_seed_id) not in PAIRED_SEED_IDS:
            raise ValueError("Step 8 training seed must be one of 101/202/303")
        for name in ("stack_train_distill_manifest_sha256", "model_val_stop_manifest_sha256"):
            if not _valid_sha256(getattr(self, name)):
                raise ValueError(f"{name} must be a SHA-256")
        if int(self.distillation_steps) <= 0:
            raise ValueError("Step 8 requires positive Phase 2 optimizer steps")
        if bool(int(self.field_warmup_steps) > 0) != bool(recipe.field_warmup):
            raise ValueError("field_warmup_steps must be positive exactly for warm-up recipes")
        if int(self.field_warmup_steps) < 0:
            raise ValueError("field_warmup_steps cannot be negative")
        if int(self.unique_training_jets) not in {250_000, 3_000_000}:
            raise ValueError(
                "Step 8 reconstruction runs require a locked distillation-child count"
            )
        if float(self.learning_rate) <= 0 or float(self.weight_decay) < 0:
            raise ValueError("Step 8 optimizer hyperparameters are invalid")
        if float(self.gradient_clip_norm) <= 0:
            raise ValueError("Step 8 gradient clip norm must be positive")

    @property
    def total_optimizer_steps(self) -> int:
        return int(self.field_warmup_steps + self.distillation_steps)

    def to_artifact(self) -> dict[str, Any]:
        recipe = resolve_step8_run_recipe(self.run_id)
        return with_content_hash(
            {
                "contract": PREDICTION_ANCHORED_STEP8_TRAIN_CONTRACT,
                **asdict(self),
                "total_optimizer_steps": self.total_optimizer_steps,
                "recipe_sha256": recipe.to_artifact()["content_hash"],
                "architecture_id": ARCH_A3_HLG_PRIMARY,
                "phase1_validation_may_terminate": False,
                "nonfinite_phase1_enters_phase2": False,
                "teacher_parameters_frozen": True,
                "r0_parameters_frozen": True,
                "target_logits_detached": True,
                "persistent_dense_fields_written": False,
            }
        )


def train_step8_replica(
    model: PredictionAnchoredHLGCorrection | PredictionAnchoredAll50HLG,
    batches: Sequence[Mapping[str, Any]] | Any,
    config: Step8TrainConfig,
    *,
    live_consumer: torch.nn.Module,
    consumer_forward_fn: Callable[[torch.nn.Module, Mapping[str, Any], torch.Tensor], torch.Tensor],
) -> dict[str, Any]:
    """Train one Step 8 replica while keeping R0/T10 and data products external.

    The campaign runner supplies allocation-local batches in the predeclared
    order.  The live consumer remains differentiable with respect to ``f_hat``
    but every consumer parameter is frozen here before the first forward.
    """

    recipe = resolve_step8_run_recipe(config.run_id)
    if isinstance(model, PredictionAnchoredAll50HLG):
        if model.run_id != config.run_id:
            raise ValueError("Step 8 all50 model/training run IDs disagree")
        physical_scalers = model.base.scalers
        all50_scalers: TorchAll50CorrectionScalers | None = model.all50_scalers
    else:
        if model.config.architecture_id != ARCH_A3_HLG_PRIMARY:
            raise ValueError("Step 8 training requires the exact canonical A3 architecture")
        if config.run_id in ALL50_RUN_IDS:
            raise ValueError("B1/B2 require their explicit all50 model wrapper")
        physical_scalers = model.scalers
        all50_scalers = None
    consumer_parameters = list(live_consumer.parameters())
    for parameter in consumer_parameters:
        parameter.requires_grad_(False)
    live_consumer.eval()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("Step 8 model has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    iterator = iter(batches)
    device = next(model.parameters()).device
    phase_rows = []
    losses = []
    gradient_norms = []
    completed = 0
    phases = []
    if config.field_warmup_steps:
        phases.append(("field_warmup", int(config.field_warmup_steps)))
    phases.append(("distillation", int(config.distillation_steps)))
    model.train()
    for phase, count in phases:
        if hasattr(model, "set_training_phase"):
            model.set_training_phase(phase)
        elif isinstance(model, PredictionAnchoredAll50HLG):
            model.base.set_training_phase(phase)
        phase_losses = []
        for _ in range(count):
            try:
                raw = next(iterator)
            except StopIteration as exc:
                raise RuntimeError(
                    f"Step 8 iterable ended after {completed} of {config.total_optimizer_steps} steps"
                ) from exc
            required = {"hlt_tokens", "mask", "f0", "h0"}
            missing = sorted(required - set(raw))
            if missing:
                raise ValueError(f"Step 8 training batch is missing {missing}")
            if recipe.kd == 0 and "target_logits" in raw:
                raise ValueError(f"{config.run_id} has zero KD and must not load target logits")
            typed = dict(raw)
            typed["hlt_tokens"] = torch.as_tensor(raw["hlt_tokens"], dtype=torch.float32, device=device)
            typed["mask"] = torch.as_tensor(raw["mask"], dtype=torch.bool, device=device)
            typed["f0"] = torch.as_tensor(raw["f0"], dtype=torch.float32, device=device)
            typed["h0"] = torch.as_tensor(raw["h0"], dtype=torch.float32, device=device)
            for name in ("bridge_fields", "true_fields", "target_logits"):
                if name in raw:
                    typed[name] = torch.as_tensor(raw[name], dtype=torch.float32, device=device)
            if "labels" in raw:
                typed["labels"] = torch.as_tensor(raw["labels"], dtype=torch.long, device=device)
            output = model(
                typed["hlt_tokens"], typed["mask"], typed["f0"], typed["h0"]
            )
            coefficients = recipe.phase_coefficients(phase)
            needs_live = coefficients["kd"] > 0 or coefficients["ce"] > 0
            live_logits = (
                consumer_forward_fn(live_consumer, typed, output.f_hat)
                if needs_live else None
            )
            if live_logits is not None:
                if live_logits.ndim != 2 or live_logits.shape[0] != output.f_hat.shape[0]:
                    raise ValueError("Step 8 live consumer returned invalid logits")
            target_logits = typed.get("target_logits") if coefficients["kd"] > 0 else None
            loss, objective = compute_step8_objective(
                output,
                typed,
                recipe,
                phase=phase,
                live_logits=live_logits,
                target_logits=target_logits,
                physical_scalers=physical_scalers,
                all50_scalers=all50_scalers,
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"Step 8 non-finite {phase} loss; replica cannot enter a later phase"
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(trainable, float(config.gradient_clip_norm))
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError(
                    f"Step 8 non-finite {phase} gradient; replica cannot enter a later phase"
                )
            optimizer.step()
            value = float(loss.detach().cpu())
            losses.append(value)
            phase_losses.append(value)
            gradient_norms.append(float(norm.detach().cpu()))
            completed += 1
        phase_rows.append(
            {
                "phase": phase,
                "optimizer_steps": count,
                "mean_loss": float(np.mean(phase_losses)),
                "validation_allowed_to_shorten_or_extend": False,
            }
        )
    if completed != config.total_optimizer_steps:
        raise AssertionError("Step 8 optimizer-step accounting changed")
    if any(parameter.requires_grad for parameter in consumer_parameters):
        raise AssertionError("Step 8 live consumer was not kept frozen")
    return with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_STEP8_TRAIN_CONTRACT,
            "run_id": config.run_id,
            "paired_seed_id": config.paired_seed_id,
            "recipe_sha256": recipe.to_artifact()["content_hash"],
            "train_config_sha256": config.to_artifact()["content_hash"],
            "optimizer_steps_completed": completed,
            "phase_rows": phase_rows,
            "mean_loss": float(np.mean(losses)),
            "maximum_gradient_norm_before_clip": float(max(gradient_norms)),
            "all_losses_finite": True,
            "all_gradients_finite": True,
            "teacher_parameters_frozen": True,
            "input_gradient_through_live_consumer_enabled": True,
            "target_logits_detached": True,
            "persistent_batch_or_field_tensor_written": False,
            "checkpoint_selection_split": "model_val_stop",
        }
    )


def _keyed_standard_normal(seed: int, event_id: str, particle: int, channel: int) -> float:
    digest = hashlib.sha256(
        f"{int(seed)}\0{event_id}\0{int(particle)}\0{int(channel)}".encode("utf-8")
    ).digest()
    denominator = float(2**64)
    u1 = (int.from_bytes(digest[:8], "little") + 0.5) / denominator
    u2 = (int.from_bytes(digest[8:16], "little") + 0.5) / denominator
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def apply_small_field_perturbation(
    f_hat: np.ndarray,
    mask: np.ndarray,
    event_ids: Sequence[Any],
    sigma_delta: Sequence[float],
    trust_scale: Sequence[float],
    *,
    audit_seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if int(audit_seed) not in PERTURBATION_AUDIT_SEEDS:
        raise ValueError("small-field audit seed is not one of the four locked seeds")
    fields = np.asarray(f_hat, dtype=np.float32)
    valid = np.asarray(mask, dtype=bool)
    identities = tuple(str(value) for value in event_ids)
    if fields.ndim != 3 or fields.shape[-1] != 50 or valid.shape != fields.shape[:2]:
        raise ValueError("perturbation fields/mask do not align")
    if len(identities) != fields.shape[0] or len(set(identities)) != len(identities):
        raise ValueError("perturbation audit requires unique aligned event identities")
    sigma = np.asarray(sigma_delta, dtype=np.float64)[:45]
    trust = np.asarray(trust_scale, dtype=np.float64)[:45]
    if sigma.shape != (45,) or trust.shape != (45,) or np.any(sigma <= 0) or np.any(trust <= 0):
        raise ValueError("perturbation scales must be positive physical45 vectors")
    perturbation = np.zeros(fields.shape[:2] + (45,), dtype=np.float32)
    for event, identity in enumerate(identities):
        for particle in np.flatnonzero(valid[event]).tolist():
            for channel in range(45):
                eta = _keyed_standard_normal(audit_seed, identity, particle, channel)
                raw = 0.05 * sigma[channel] * eta
                perturbation[event, particle, channel] = np.float32(
                    np.clip(raw, -0.10 * trust[channel], 0.10 * trust[channel])
                )
    output = fields.copy()
    output[..., :45] += perturbation
    output[..., 45:] = fields[..., 45:]
    output[~valid] = 0.0
    return output, with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_PERTURBATION_CONTRACT,
            "audit_seed": int(audit_seed),
            "event_order_sha256": canonical_sha256(list(identities)),
            "eta_key": ["audit_seed", "event_identity", "valid_particle_index", "physical_channel"],
            "sigma_multiplier": 0.05,
            "clip_trust_multiplier": 0.10,
            "physical_channels_perturbed": 45,
            "reliability_channels_copied": 5,
            "padding_perturbation_exact_zero": True,
            "maximum_absolute_perturbation": float(np.max(np.abs(perturbation), initial=0.0)),
        }
    )


def _mean_logit_cosine(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    numerator = np.sum(a * b, axis=1)
    cosine = np.divide(numerator, denominator, out=np.ones_like(numerator), where=denominator > 0)
    return float(np.mean(cosine))


def run_small_field_perturbation_audit(
    *,
    f_hat: np.ndarray,
    mask: np.ndarray,
    event_ids: Sequence[Any],
    labels: np.ndarray,
    sigma_delta: Sequence[float],
    trust_scale: Sequence[float],
    consumer_logits_fn: Callable[[np.ndarray], np.ndarray],
    class_order: Sequence[str],
    sample_weights: np.ndarray | None = None,
    selectable_for_primary_deployment: bool,
) -> dict[str, Any]:
    base_logits = np.asarray(consumer_logits_fn(np.asarray(f_hat, dtype=np.float32)), dtype=np.float32)
    base_metrics = classification_metrics(
        base_logits, labels, class_order=class_order, sample_weights=sample_weights
    )
    rows = []
    losses = []
    for seed in PERTURBATION_AUDIT_SEEDS:
        perturbed, perturbation = apply_small_field_perturbation(
            f_hat, mask, event_ids, sigma_delta, trust_scale, audit_seed=seed
        )
        logits = np.asarray(consumer_logits_fn(perturbed), dtype=np.float32)
        metrics = classification_metrics(
            logits, labels, class_order=class_order, sample_weights=sample_weights
        )
        accuracy_loss = float(base_metrics["accuracy"] - metrics["accuracy"])
        losses.append(accuracy_loss)
        cosine = _mean_logit_cosine(base_logits, logits)
        rows.append(
            {
                "audit_seed": seed,
                "perturbation_sha256": perturbation["content_hash"],
                "accuracy": metrics["accuracy"],
                "accuracy_change_perturbed_minus_base": float(metrics["accuracy"] - base_metrics["accuracy"]),
                "accuracy_loss_base_minus_perturbed": accuracy_loss,
                "cross_entropy": metrics["cross_entropy"],
                "cross_entropy_change_perturbed_minus_base": float(metrics["cross_entropy"] - base_metrics["cross_entropy"]),
                "mean_logit_cosine": cosine,
                "mean_logit_cosine_loss": 1.0 - cosine,
            }
        )
    mean_loss = float(np.mean(losses))
    worst_loss = float(np.max(losses))
    passed = (
        mean_loss <= PERTURBATION_MEAN_ACCURACY_LOSS_MAX
        and worst_loss <= PERTURBATION_WORST_ACCURACY_LOSS_MAX
    )
    return with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_PERTURBATION_AUDIT_CONTRACT,
            "audit_seeds": list(PERTURBATION_AUDIT_SEEDS),
            "base_metrics": base_metrics,
            "per_seed": rows,
            "mean_accuracy_loss": mean_loss,
            "worst_seed_accuracy_loss": worst_loss,
            "negative_accuracy_loss_retained": True,
            "mean_accuracy_loss_max": PERTURBATION_MEAN_ACCURACY_LOSS_MAX,
            "worst_accuracy_loss_max": PERTURBATION_WORST_ACCURACY_LOSS_MAX,
            "threshold_passed": passed,
            "selectable_for_primary_deployment": bool(selectable_for_primary_deployment),
            "automatic_selectability_gate_applies": bool(selectable_for_primary_deployment),
            "automatic_selectability_gate_passed": (
                passed if selectable_for_primary_deployment else None
            ),
        }
    )


def _finite_cosine(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=np.float64).reshape(-1)
    b = np.asarray(second, dtype=np.float64).reshape(-1)
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0 and norm_b == 0:
        return 1.0
    if norm_a == 0 or norm_b == 0:
        return 0.0
    value = float(np.dot(a, b) / (norm_a * norm_b))
    return float(np.clip(value, -1.0, 1.0))


def correction_bridge_alignment(
    f_hat: np.ndarray,
    f0: np.ndarray,
    bridge_fields: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    predicted = np.asarray(f_hat, dtype=np.float64)[..., :45] - np.asarray(f0, dtype=np.float64)[..., :45]
    target = np.asarray(bridge_fields, dtype=np.float64)[..., :45] - np.asarray(f0, dtype=np.float64)[..., :45]
    valid = np.asarray(mask, dtype=bool)
    if predicted.shape != target.shape or valid.shape != predicted.shape[:2]:
        raise ValueError("alignment fields/mask do not match")
    groups = {
        name: _finite_cosine(predicted[..., indices][valid], target[..., indices][valid])
        for name, indices in physical_loss_groups().items()
    }
    overall = _finite_cosine(predicted[valid], target[valid])
    if not math.isfinite(overall) or not all(math.isfinite(value) for value in groups.values()):
        raise FloatingPointError("bridge alignment diagnostic is non-finite")
    return with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_ALIGNMENT_CONTRACT,
            "overall_cosine": overall,
            "groupwise_cosine": groups,
            "finite": True,
            "automatic_selection_threshold": None,
            "used_for_physical_recovery_interpretation": True,
        }
    )


def build_bridge_quantile_reference(
    bridge_fields: np.ndarray,
    f0: np.ndarray,
    mask: np.ndarray,
    sigma_delta: Sequence[float],
    *,
    stack_train_distill_manifest_sha256: str,
) -> dict[str, Any]:
    if not _valid_sha256(stack_train_distill_manifest_sha256):
        raise ValueError("quantile reference requires stack_train_distill manifest SHA-256")
    bridge = np.asarray(bridge_fields, dtype=np.float64)
    anchor = np.asarray(f0, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    sigma = np.asarray(sigma_delta, dtype=np.float64)[:45]
    if bridge.shape != anchor.shape or bridge.shape[-1] != 50 or valid.shape != bridge.shape[:2]:
        raise ValueError("quantile reference fields/mask do not align")
    if sigma.shape != (45,) or np.any(sigma <= 0):
        raise ValueError("quantile reference sigma must be positive physical45")
    values = ((bridge[..., :45] - anchor[..., :45]) / sigma)[valid]
    if values.shape[0] == 0 or not np.isfinite(values).all():
        raise ValueError("quantile reference has no finite valid particles")
    levels = np.linspace(0.0, 1.0, QUANTILE_COUNT, dtype=np.float64)
    quantiles = np.quantile(values, levels, axis=0).astype(np.float64)
    return with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_QUANTILE_REFERENCE_CONTRACT,
            "fit_split": "stack_train_distill",
            "stack_train_distill_manifest_sha256": stack_train_distill_manifest_sha256,
            "quantile_count": QUANTILE_COUNT,
            "physical_channel_count": 45,
            "dtype": "float64",
            "quantile_level_convention": "linspace_0_1_inclusive_1001",
            "quantiles": quantiles.tolist(),
            "sigma_delta_sha256": canonical_sha256(sigma.tolist()),
            "valid_particle_count": int(values.shape[0]),
            "dense_training_field_persisted": False,
            "persistent_array_shape": [QUANTILE_COUNT, 45],
        }
    )


def build_bridge_quantile_reference_from_standardized_corrections(
    standardized_corrections: np.ndarray,
    sigma_delta: Sequence[float],
    *,
    stack_train_distill_manifest_sha256: str,
) -> dict[str, Any]:
    """Build the same compact reference from RAM-only valid-particle values."""

    if not _valid_sha256(stack_train_distill_manifest_sha256):
        raise ValueError("quantile reference requires stack_train_distill manifest SHA-256")
    values = np.asarray(standardized_corrections, dtype=np.float64)
    sigma = np.asarray(sigma_delta, dtype=np.float64)[:45]
    if values.ndim != 2 or values.shape[1] != 45 or values.shape[0] == 0:
        raise ValueError("standardized correction reference must have shape [N,45]")
    if not np.isfinite(values).all() or sigma.shape != (45,) or np.any(sigma <= 0):
        raise ValueError("standardized correction reference is non-finite or has invalid scales")
    levels = np.linspace(0.0, 1.0, QUANTILE_COUNT, dtype=np.float64)
    quantiles = np.quantile(values, levels, axis=0).astype(np.float64)
    return with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_QUANTILE_REFERENCE_CONTRACT,
            "fit_split": "stack_train_distill",
            "stack_train_distill_manifest_sha256": stack_train_distill_manifest_sha256,
            "quantile_count": QUANTILE_COUNT,
            "physical_channel_count": 45,
            "dtype": "float64",
            "quantile_level_convention": "linspace_0_1_inclusive_1001",
            "quantiles": quantiles.tolist(),
            "sigma_delta_sha256": canonical_sha256(sigma.tolist()),
            "valid_particle_count": int(values.shape[0]),
            "dense_training_field_persisted": False,
            "persistent_array_shape": [QUANTILE_COUNT, 45],
        }
    )


def bridge_distribution_distance(
    reference: Mapping[str, Any],
    f_hat: np.ndarray,
    f0: np.ndarray,
    mask: np.ndarray,
    sigma_delta: Sequence[float],
    *,
    validation_split: str,
) -> dict[str, Any]:
    validate_content_hash(
        reference, expected_contract=PREDICTION_ANCHORED_QUANTILE_REFERENCE_CONTRACT
    )
    if validation_split not in AUTHORIZED_DISTRIBUTION_SPLITS:
        raise PermissionError(
            "bridge distribution diagnostic is validation-only and never available on final_test"
        )
    prediction = np.asarray(f_hat, dtype=np.float64)
    anchor = np.asarray(f0, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    sigma = np.asarray(sigma_delta, dtype=np.float64)[:45]
    if prediction.shape != anchor.shape or prediction.shape[-1] != 50 or valid.shape != prediction.shape[:2]:
        raise ValueError("distribution diagnostic fields/mask do not align")
    if reference.get("sigma_delta_sha256") != canonical_sha256(sigma.tolist()):
        raise ValueError("distribution diagnostic used different correction scales")
    values = ((prediction[..., :45] - anchor[..., :45]) / sigma)[valid]
    levels = np.linspace(0.0, 1.0, QUANTILE_COUNT, dtype=np.float64)
    expected = np.asarray(reference["quantiles"], dtype=np.float64)
    observed = np.quantile(values, levels, axis=0).astype(np.float64)
    per_channel = np.mean(np.abs(observed - expected), axis=0)
    groups = {
        name: float(np.mean(per_channel[indices]))
        for name, indices in physical_loss_groups().items()
    }
    if not np.isfinite(per_channel).all() or not all(math.isfinite(value) for value in groups.values()):
        raise FloatingPointError("bridge distribution distance is non-finite")
    return with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_DISTRIBUTION_DISTANCE_CONTRACT,
            "reference_sha256": reference["content_hash"],
            "validation_split": validation_split,
            "quantile_count": QUANTILE_COUNT,
            "per_channel_mean_absolute_quantile_distance": per_channel.tolist(),
            "groupwise_w1_approximation": groups,
            "finite": True,
            "automatic_selection_threshold": None,
            "final_test_accessed": False,
        }
    )


def evaluate_reliability_only_response(
    *,
    consumer_logits_fn: Callable[[np.ndarray], np.ndarray],
    f0: np.ndarray,
    physical45_bridge: np.ndarray,
    all50_bridge: np.ndarray,
    labels: np.ndarray,
    class_order: Sequence[str],
    sample_weights: np.ndarray | None = None,
) -> dict[str, Any]:
    anchor = np.asarray(f0, dtype=np.float32)
    physical = np.asarray(physical45_bridge, dtype=np.float32)
    full = np.asarray(all50_bridge, dtype=np.float32)
    if anchor.shape != physical.shape or anchor.shape != full.shape or anchor.shape[-1] != 50:
        raise ValueError("reliability response fields do not align")
    reliability_only = anchor.copy()
    reliability_only[..., 45:] = full[..., 45:]
    fields = {
        "f0": anchor,
        "physical45_bridge": physical,
        "reliability5_only": reliability_only,
        "all50_bridge": full,
    }
    metrics = {
        name: classification_metrics(
            np.asarray(consumer_logits_fn(value), dtype=np.float32),
            labels,
            class_order=class_order,
            sample_weights=sample_weights,
        )
        for name, value in fields.items()
    }
    base = float(metrics["f0"]["accuracy"])
    gains = {name: float(value["accuracy"] - base) for name, value in metrics.items() if name != "f0"}
    return with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_RELIABILITY_RESPONSE_CONTRACT,
            "metrics": metrics,
            "accuracy_gains_over_f0": gains,
            "shortcut_risk_diagnostic_only": True,
            "selectable_for_primary_deployment": False,
        }
    )


def compute_gain_and_recovery(
    *,
    baseline_value: float,
    teacher_bridge_value: float,
    deployable_value: float,
    metric_name: str,
    metric_direction: str,
) -> dict[str, Any]:
    values = (float(baseline_value), float(teacher_bridge_value), float(deployable_value))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("gain/recovery values must be finite")
    if metric_direction == "higher_is_better":
        teacher_gain = values[1] - values[0]
        deployable_gain = values[2] - values[0]
    elif metric_direction == "lower_is_better":
        teacher_gain = values[0] - values[1]
        deployable_gain = values[0] - values[2]
    else:
        raise ValueError("metric direction must be higher_is_better or lower_is_better")
    recovery = None if teacher_gain <= 0 else float(deployable_gain / teacher_gain)
    return with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_GAIN_RECOVERY_CONTRACT,
            "metric_name": str(metric_name),
            "metric_direction": metric_direction,
            "baseline_value": values[0],
            "teacher_bridge_value": values[1],
            "deployable_value": values[2],
            "teacher_bridge_gain": float(teacher_gain),
            "deployable_gain": float(deployable_gain),
            "recovery_fraction": recovery,
            "nonpositive_teacher_gain_recorded_as_null": teacher_gain <= 0,
            "loss_improvement_sign_corrected": metric_direction == "lower_is_better",
        }
    )


def build_adversarial_channel_report(
    *,
    perturbation_audit: Mapping[str, Any],
    alignment: Mapping[str, Any],
    distribution_distance: Mapping[str, Any],
    saturation_fraction: float,
    reliability_pass_through_exact: bool,
    selectable_for_primary_deployment: bool,
) -> dict[str, Any]:
    validate_content_hash(
        perturbation_audit,
        expected_contract=PREDICTION_ANCHORED_PERTURBATION_AUDIT_CONTRACT,
    )
    validate_content_hash(alignment, expected_contract=PREDICTION_ANCHORED_ALIGNMENT_CONTRACT)
    validate_content_hash(
        distribution_distance,
        expected_contract=PREDICTION_ANCHORED_DISTRIBUTION_DISTANCE_CONTRACT,
    )
    saturation = float(saturation_fraction)
    if not math.isfinite(saturation) or not 0 <= saturation <= 1:
        raise ValueError("saturation fraction must be finite and in [0,1]")
    if not bool(alignment.get("finite")) or not bool(distribution_distance.get("finite")):
        raise ValueError("adversarial report requires finite physical diagnostics")
    perturbation_passed = bool(perturbation_audit["threshold_passed"])
    saturation_passed = saturation <= 0.01
    automatic_passed = (
        perturbation_passed and saturation_passed and bool(reliability_pass_through_exact)
    )
    return with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_ADVERSARIAL_REPORT_CONTRACT,
            "perturbation_audit_sha256": perturbation_audit["content_hash"],
            "alignment_sha256": alignment["content_hash"],
            "distribution_distance_sha256": distribution_distance["content_hash"],
            "saturation_fraction": saturation,
            "saturation_fraction_max": 0.01,
            "reliability_pass_through_exact": bool(reliability_pass_through_exact),
            "perturbation_gate_passed": perturbation_passed,
            "saturation_gate_passed": saturation_passed,
            "automatic_selectability_gates_passed": (
                automatic_passed if selectable_for_primary_deployment else None
            ),
            "selectable_for_primary_deployment": bool(selectable_for_primary_deployment),
            "alignment_automatic_threshold": None,
            "distribution_automatic_threshold": None,
            "physical_recovery_claim_requires_reported_interpretation": True,
            "hidden_alignment_or_distribution_cutoff_present": False,
        }
    )


@dataclass(frozen=True)
class Step8FixedStorage:
    child_split_manifest_bytes: int
    r0_weights_bytes: int
    target_logit_namespace_bytes: Mapping[str, int]
    recipes_bindings_reports_bytes: int
    final_deployable_bundle_bytes: int
    measurement_basis: str = "filesystem_measured"

    def __post_init__(self) -> None:
        scalar_names = (
            "child_split_manifest_bytes",
            "r0_weights_bytes",
            "recipes_bindings_reports_bytes",
            "final_deployable_bundle_bytes",
        )
        for name in scalar_names:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a measured positive byte count")
        expected = {PRIMARY_TEACHER_NAMESPACE, ALL50_TEACHER_NAMESPACE, N3_F0_TEACHER_NAMESPACE}
        observed = set(self.target_logit_namespace_bytes)
        if not expected.issubset(observed) or not observed.issubset(expected | {ALTERNATE_TEACHER_NAMESPACE}):
            raise ValueError("fixed storage target-logit namespaces are incomplete or unknown")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in self.target_logit_namespace_bytes.values()
        ):
            raise ValueError("target-logit namespace byte counts must be measured positive integers")
        if self.measurement_basis not in {
            "filesystem_measured",
            "clean_start_conservative_upper_bound",
        }:
            raise ValueError("unknown Step 8 fixed-storage measurement basis")

    @property
    def total_bytes(self) -> int:
        return int(
            self.child_split_manifest_bytes
            + self.r0_weights_bytes
            + sum(self.target_logit_namespace_bytes.values())
            + self.recipes_bindings_reports_bytes
            + self.final_deployable_bundle_bytes
        )

    def to_artifact(self) -> dict[str, Any]:
        return with_content_hash(
            {
                "contract": PREDICTION_ANCHORED_STEP8_FIXED_STORAGE_CONTRACT,
                **asdict(self),
                "target_logit_namespace_bytes": dict(self.target_logit_namespace_bytes),
                "total_bytes": self.total_bytes,
                "dense_field_cache_bytes": 0,
                "all_categories_measured": self.measurement_basis == "filesystem_measured",
                "production_safe_upper_bounds": self.measurement_basis
                == "clean_start_conservative_upper_bound",
            }
        )


def build_clean_start_step8_fixed_storage(
    child_split_manifest: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> tuple[Step8FixedStorage, dict[str, Any]]:
    """Conservatively reserve fixed bytes before campaign outputs exist."""

    validate_content_hash(
        child_split_manifest, expected_contract="prediction_anchored_child_splits_v1"
    )
    validate_campaign_registry(registry)
    event_count = int(child_split_manifest["children"]["stack_train_distill"]["count"])
    if event_count <= 0:
        raise ValueError("clean-start fixed storage requires stack_train_distill events")
    namespaces = [PRIMARY_TEACHER_NAMESPACE, ALL50_TEACHER_NAMESPACE, N3_F0_TEACHER_NAMESPACE]
    if bool(registry.get("alternate_teacher_valid")):
        namespaces.append(ALTERNATE_TEACHER_NAMESPACE)
    # Complete uncompressed NPZ row: float32[10], int64 label, and NumPy <U64 identity.
    per_event_cache_bytes = 10 * 4 + 8 + 64 * 4
    per_namespace_overhead = 2 * 1024**2
    namespace_bytes = {
        name: event_count * per_event_cache_bytes + per_namespace_overhead
        for name in namespaces
    }

    def weights_with_headroom(parameter_count: int) -> int:
        raw = int(parameter_count) * 4
        return raw + int(math.ceil(raw * 0.15))

    storage = Step8FixedStorage(
        child_split_manifest_bytes=len(canonical_json_bytes(child_split_manifest)) + 1,
        r0_weights_bytes=weights_with_headroom(16_000_000),
        target_logit_namespace_bytes=namespace_bytes,
        recipes_bindings_reports_bytes=32 * 1024**2,
        final_deployable_bundle_bytes=weights_with_headroom(52_000_000),
        measurement_basis="clean_start_conservative_upper_bound",
    )
    report = with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_STEP8_FIXED_STORAGE_MEASUREMENT_CONTRACT,
            "fixed_storage": storage.to_artifact(),
            "child_split_manifest_sha256": child_split_manifest["content_hash"],
            "registry_sha256": registry["content_hash"],
            "stack_train_distill_event_count": event_count,
            "target_cache_formula": {
                "logit_classes": 10,
                "logit_dtype_bytes": 4,
                "label_dtype_bytes": 8,
                "identity_unicode_characters": 64,
                "unicode_character_bytes": 4,
                "per_namespace_overhead_bytes": per_namespace_overhead,
            },
            "weight_dtype_bytes": 4,
            "serialization_headroom_fraction": 0.15,
            "filesystem_bytes_measured": False,
            "clean_start_production_safe_upper_bounds": True,
            "future_campaign_artifacts_required": False,
            "dense_field_artifacts_present": False,
        }
    )
    return storage, report


def _measured_path(path: str | Path, *, label: str) -> tuple[int, list[dict[str, Any]]]:
    root = Path(path)
    if root.is_symlink() or not root.exists():
        raise FileNotFoundError(f"missing/unsafe measured {label}: {root}")
    descendants = [] if root.is_file() else list(root.rglob("*"))
    symlinks = [value for value in descendants if value.is_symlink()]
    if symlinks:
        raise ValueError(f"measured {label} contains a symlink: {symlinks[0]}")
    files = [root] if root.is_file() else sorted(value for value in descendants if value.is_file())
    if not files:
        raise ValueError(f"measured {label} contains no files")
    rows = []
    total = 0
    for value in files:
        if value.is_symlink():
            raise ValueError(f"measured {label} contains a symlink: {value}")
        relative = value.name if root.is_file() else value.relative_to(root).as_posix()
        size = int(value.stat().st_size)
        total += size
        rows.append({"relative_path": relative, "size_bytes": size, "sha256": sha256_file(value)})
    return total, rows


def measure_step8_fixed_storage(
    *,
    child_split_manifest_path: str | Path,
    r0_weights_path: str | Path,
    target_logit_namespace_paths: Mapping[str, str | Path],
    recipes_bindings_reports_paths: Sequence[str | Path],
    final_deployable_bundle_path: str | Path,
) -> tuple[Step8FixedStorage, dict[str, Any]]:
    """Measure fixed persistent categories from actual immutable filesystem bytes."""

    source_paths = [
        Path(child_split_manifest_path),
        Path(r0_weights_path),
        *(Path(value) for value in target_logit_namespace_paths.values()),
        *(Path(value) for value in recipes_bindings_reports_paths),
        Path(final_deployable_bundle_path),
    ]
    resolved = [value.resolve(strict=False) for value in source_paths]
    for index, first in enumerate(resolved):
        for second in resolved[index + 1:]:
            if first == second or first in second.parents or second in first.parents:
                raise ValueError("fixed-storage source paths overlap and would be double-counted")

    child_bytes, child_rows = _measured_path(child_split_manifest_path, label="child manifest")
    r0_bytes, r0_rows = _measured_path(r0_weights_path, label="R0 weights")
    target_bytes: dict[str, int] = {}
    target_rows = {}
    for namespace, path in target_logit_namespace_paths.items():
        size, rows = _measured_path(path, label=f"target namespace {namespace}")
        names = {row["relative_path"] for row in rows}
        if names != {"teacher_logits.npz", "teacher_logits_manifest.json"}:
            raise ValueError(
                f"target namespace {namespace} does not contain only its cache and manifest"
            )
        target_bytes[str(namespace)] = size
        target_rows[str(namespace)] = rows
    if not recipes_bindings_reports_paths:
        raise ValueError("fixed storage requires measured recipes/bindings/reports")
    metadata_bytes = 0
    metadata_rows = []
    for index, path in enumerate(recipes_bindings_reports_paths):
        size, rows = _measured_path(path, label=f"metadata path {index}")
        for row in rows:
            lowered = row["relative_path"].lower()
            if any(token in lowered for token in ("dense_field", "bridge_fields", "oracle_fields", "f_true")):
                raise ValueError("metadata category contains a forbidden dense-field artifact")
        metadata_bytes += size
        metadata_rows.append({"source_index": index, "files": rows, "size_bytes": size})
    bundle_bytes, bundle_rows = _measured_path(final_deployable_bundle_path, label="final bundle")
    storage = Step8FixedStorage(
        child_split_manifest_bytes=child_bytes,
        r0_weights_bytes=r0_bytes,
        target_logit_namespace_bytes=target_bytes,
        recipes_bindings_reports_bytes=metadata_bytes,
        final_deployable_bundle_bytes=bundle_bytes,
    )
    report = with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_STEP8_FIXED_STORAGE_MEASUREMENT_CONTRACT,
            "fixed_storage": storage.to_artifact(),
            "child_split_manifest_files": child_rows,
            "r0_weight_files": r0_rows,
            "target_logit_namespace_files": target_rows,
            "recipes_bindings_reports_sources": metadata_rows,
            "final_deployable_bundle_files": bundle_rows,
            "filesystem_bytes_measured": True,
            "symlinks_allowed": False,
            "dense_field_artifacts_present": False,
        }
    )
    return storage, report


def _torch_bytes(value: Any) -> bytes:
    buffer = BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def measure_step8_registry_states(
    registry: Mapping[str, Any],
    *,
    physical45_scaler_artifact: Mapping[str, Any],
    all50_scaler_artifact: Mapping[str, Any],
    absolute_scaler_artifact: Mapping[str, Any],
    source_manifest_sha256: str,
    deployed_reference: BundleResourceReference,
    fixed_storage: Step8FixedStorage,
    selected_budget_bytes: int = NORMAL_PILOT_BUDGET_BYTES,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Measure all remaining rows and execute the final 5/6 GiB preflight."""

    semantic_registry = validate_step8_registry_semantics(registry)
    if selected_budget_bytes not in {NORMAL_PILOT_BUDGET_BYTES, PAIRED3_HARD_CEILING_BYTES}:
        raise ValueError("Step 8 budget must be the locked 5 or 6 GiB mode")
    alternate_cache_measured = (
        ALTERNATE_TEACHER_NAMESPACE in fixed_storage.target_logit_namespace_bytes
    )
    if alternate_cache_measured != bool(registry["alternate_teacher_valid"]):
        raise ValueError(
            "fixed storage alternate-cache measurement disagrees with conditional TALT state"
        )
    input_by_id = {row["canonical_run_id"]: row for row in registry["runs"]}
    upstream_unmeasured = [
        run_id for run_id in STEP3_RUN_IDS
        if input_by_id[run_id]["measurement_status"] != MEASUREMENT_MEASURED
    ]
    if upstream_unmeasured:
        raise PermissionError(
            "Step 8 requires the actual Step 3 upstream measurements: "
            + ", ".join(upstream_unmeasured)
        )
    step7_statuses = {
        input_by_id[run_id]["measurement_status"] for run_id in STEP7_MEASURED_ARCHITECTURE_IDS
    }
    if step7_statuses == {MEASUREMENT_MEASURED}:
        step7_registry = dict(registry)
        step7_measurement = None
        inherited_step7 = True
    elif step7_statuses == {MEASUREMENT_UNMEASURED}:
        step7_registry, step7_measurement = measure_step7_registry_states(
            registry,
            scaler_artifact=physical45_scaler_artifact,
            absolute_scaler_artifact=absolute_scaler_artifact,
            source_manifest_sha256=source_manifest_sha256,
            deployed_reference=deployed_reference,
        )
        inherited_step7 = False
    else:
        raise ValueError("Step 8 refuses a partially measured Step 7 registry")
    by_id = {row["canonical_run_id"]: row for row in step7_registry["runs"]}
    changed = [
        run_id for run_id in STEP8_SPECIAL_CANONICAL_RUN_IDS
        if by_id[run_id]["measurement_status"] == MEASUREMENT_MEASURED
    ]
    if changed:
        raise ValueError(f"Step 8 input already measured special rows: {changed}")
    physical_a3 = build_step7_hlg_correction_model(
        ARCH_A3_HLG_PRIMARY,
        scaler_artifact=physical45_scaler_artifact,
        dropout=0.05,
    )
    canonical_config = physical_a3.config_artifact()
    state_bytes: dict[str, int] = {}
    config_hashes: dict[str, str] = {}
    for run_id in STEP8_SPECIAL_CANONICAL_RUN_IDS:
        recipe = resolve_step8_run_recipe(run_id)
        if run_id == ALL50_RUN_IDS[0]:
            model: torch.nn.Module = PredictionAnchoredAll50HLG(
                run_id,
                physical45_scaler_artifact=physical45_scaler_artifact,
                all50_scaler_artifact=all50_scaler_artifact,
                dropout=0.05,
            )
            model_config = model.config_artifact()
        elif run_id == ALL50_RUN_IDS[1]:
            model = PredictionAnchoredAll50HLG(
                run_id,
                physical45_scaler_artifact=physical45_scaler_artifact,
                all50_scaler_artifact=all50_scaler_artifact,
                dropout=0.05,
            )
            model_config = model.config_artifact()
        elif run_id == ALTERNATE_A0_RUN_ID:
            model = PredictionAnchoredC0Correction(
                physical45_scaler_artifact,
                C0CorrectionConfig(d_model=160, dropout=0.05),
            )
            model_config = model.config_artifact()
        else:
            model = physical_a3
            model_config = canonical_config
        payload = {
            "checkpoint_contract": "prediction_anchored_step8_weights_v1",
            "run_id": run_id,
            "recipe": recipe.to_artifact(),
            "model_config": model_config,
            "model_state_dict": model.state_dict(),
            "canonical_a3_config_sha256": canonical_config["content_hash"],
            "weights_only": True,
            "optimizer_state_persisted": False,
            "generated_fields_persisted": False,
            "nonmedian_weights_persisted": False,
        }
        state_bytes[run_id] = len(_torch_bytes(payload))
        config_hashes[run_id] = model_config["content_hash"]
    updated = record_registry_measurements(step7_registry, state_bytes)
    unmeasured_runnable = [
        row["canonical_run_id"] for row in updated["runs"]
        if row["execution_status"] == "RUNNABLE"
        and row["measurement_status"] != MEASUREMENT_MEASURED
    ]
    if unmeasured_runnable:
        raise AssertionError(f"Step 8 left runnable rows UNMEASURED: {unmeasured_runnable}")
    readiness = require_production_ready(
        updated,
        fixed_persistent_bytes=fixed_storage.total_bytes,
        selected_budget_bytes=int(selected_budget_bytes),
    )
    alias_row = resolve_registry_run(updated, A3_PRIMARY_ALIAS)
    canonical_row = resolve_registry_run(updated, ARCH_A3_HLG_PRIMARY)
    if alias_row["content_hash"] != canonical_row["content_hash"]:
        raise AssertionError("canonical A3 alias was not deduplicated")
    return updated, with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_STEP8_MEASUREMENT_CONTRACT,
            "input_registry_sha256": registry["content_hash"],
            "registry_semantics_sha256": semantic_registry["content_hash"],
            "step7_measurement_sha256": (
                None if step7_measurement is None else step7_measurement["content_hash"]
            ),
            "step7_measurements_inherited": inherited_step7,
            "updated_registry_sha256": updated["content_hash"],
            "source_manifest_sha256": source_manifest_sha256,
            "measured_state_bytes": state_bytes,
            "model_config_sha256": config_hashes,
            "canonical_a3_config_sha256": canonical_config["content_hash"],
            "newly_measured_configuration_count": len(state_bytes),
            "canonical_a3_alias_deduplicated": True,
            "registry_configuration_count": updated["configuration_count"],
            "reconstruction_breadth_count": updated["reconstruction_breadth_count"],
            "post_teacher_configuration_count": updated["post_teacher_configuration_count"],
            "unmeasured_runnable_run_ids": [],
            "fixed_storage": fixed_storage.to_artifact(),
            "production_readiness": readiness,
            "selected_budget_bytes": int(selected_budget_bytes),
            "serialization_method": "torch_save_weights_only_to_verified_ram",
            "persistent_dense_fields_written": False,
        }
    )


def require_post_teacher_release(
    registry: Mapping[str, Any],
    *,
    selected_consumer: Mapping[str, Any],
    primary_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Release B6 from a confirmed, integrity-valid teacher, never from C0 success."""

    validate_campaign_registry(registry)
    validate_content_hash(
        selected_consumer,
        expected_contract=PREDICTION_ANCHORED_SELECTED_CONSUMER_CONTRACT,
    )
    if selected_consumer.get("status") != "CONFIRMED_LOCKED":
        raise PermissionError("post-teacher matrix requires a confirmed locked consumer")
    confirmation = selected_consumer.get("stack_val_consumer_confirmation", {})
    execution_authorized = bool(
        confirmation.get("execution_authorized", confirmation.get("passed"))
    )
    if not execution_authorized or not bool(confirmation.get("provenance_valid")):
        raise PermissionError(
            "post-teacher matrix requires an integrity-valid confirmed teacher"
        )
    validate_teacher_binding(
        primary_binding,
        expected_kind="primary",
        primary_selection=selected_consumer,
    )
    runnable = [
        row["canonical_run_id"] for row in registry["runs"]
        if row["post_teacher_configuration"] and row["execution_status"] == "RUNNABLE"
    ]
    return with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_POST_TEACHER_RELEASE_CONTRACT,
            "selected_consumer_sha256": selected_consumer["content_hash"],
            "primary_binding_sha256": primary_binding["content_hash"],
            "released_post_teacher_run_ids": runnable,
            "registered_post_teacher_configuration_count": registry["post_teacher_configuration_count"],
            "released_runnable_count": len(runnable),
            "teacher_gate_passed": True,
            "teacher_integrity_gate_passed": True,
            "teacher_quality_passed": bool(
                confirmation.get("quality_passed", confirmation.get("passed"))
            ),
            "teacher_quality_warning_override_applied": bool(
                confirmation.get("quality_gate_override_applied")
            ),
            "c0_success_consulted": False,
            "hlg_release_independent_of_c0": True,
        }
    )


def _mini_logits_from_fields(fields: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=fields.dtype).unsqueeze(-1)
    pooled = (fields[..., :10] * weights).sum(1) / torch.clamp(weights.sum(1), min=1.0)
    return pooled


def run_step8_paired_seed_miniature(
    *,
    physical45_scaler_artifact: Mapping[str, Any],
    batch: Mapping[str, Any],
    seed_ids: Sequence[int] = PAIRED_SEED_IDS,
) -> dict[str, Any]:
    """Three-seed tensor-path comparison of the four required model families."""

    if tuple(int(value) for value in seed_ids) != PAIRED_SEED_IDS:
        raise ValueError("Step 8 miniature seeds are locked to 101/202/303")
    required = {"hlt_tokens", "mask", "f0", "h0", "bridge_fields", "labels"}
    missing = sorted(required - set(batch))
    if missing:
        raise ValueError(f"Step 8 miniature batch is missing {missing}")
    tensors = {
        name: torch.as_tensor(
            batch[name],
            dtype=torch.bool if name == "mask" else torch.long if name == "labels" else torch.float32,
        )
        for name in required
    }
    family_ids = (ARCH_A3_HLG_PRIMARY, ARCH_A0M_CAPACITY_PARTICLE, DIRECT_HLT, DIRECT_R0REP)
    rows = []
    aggregates = {}
    for family in family_ids:
        scores = []
        for seed in seed_ids:
            torch.manual_seed(int(seed))
            if family == ARCH_A3_HLG_PRIMARY:
                model = build_step7_hlg_correction_model(
                    family, scaler_artifact=physical45_scaler_artifact, dropout=0.0
                ).eval()
                with torch.no_grad():
                    logits = _mini_logits_from_fields(
                        model(tensors["hlt_tokens"], tensors["mask"], tensors["f0"], tensors["h0"]).f_hat,
                        tensors["mask"],
                    )
            elif family == ARCH_A0M_CAPACITY_PARTICLE:
                model = build_step6_correction_model(
                    family, scaler_artifact=physical45_scaler_artifact, dropout=0.0
                ).eval()
                with torch.no_grad():
                    logits = _mini_logits_from_fields(
                        model(tensors["hlt_tokens"], tensors["mask"], tensors["f0"], tensors["h0"]).f_hat,
                        tensors["mask"],
                    )
            else:
                model = DirectHLGClassifier(
                    DirectHLGConfig(family, int(tensors["mask"].shape[1]), dropout=0.0),
                    scaler_artifact=physical45_scaler_artifact if family == DIRECT_R0REP else None,
                ).eval()
                with torch.no_grad():
                    direct = (
                        model(tensors["hlt_tokens"], tensors["mask"], tensors["f0"], tensors["h0"])
                        if family == DIRECT_R0REP
                        else model(tensors["hlt_tokens"], tensors["mask"])
                    )
                    logits = direct.logits
            ce = float(F.cross_entropy(logits, tensors["labels"]).detach())
            accuracy = float((logits.argmax(-1) == tensors["labels"]).float().mean().detach())
            if not math.isfinite(ce):
                raise FloatingPointError("Step 8 paired miniature produced non-finite CE")
            rows.append({"family_id": family, "seed_id": int(seed), "cross_entropy": ce, "accuracy": accuracy})
            scores.append((ce, int(seed), accuracy))
        ordered = sorted(scores)
        aggregates[family] = {
            "mean_cross_entropy": float(np.mean([value[0] for value in scores])),
            "sample_std_cross_entropy": float(np.std([value[0] for value in scores], ddof=1)),
            "mean_accuracy": float(np.mean([value[2] for value in scores])),
            "ordered_median_seed": ordered[1][1],
        }
    return with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_STEP8_MINIATURE_CONTRACT,
            "family_ids": list(family_ids),
            "paired_seed_ids": list(seed_ids),
            "rows": rows,
            "aggregates": aggregates,
            "complete_required_comparison": True,
            "scientific_results_allowed": False,
            "purpose": "paired_tensor_path_rehearsal_only",
        }
    )


__all__ = [
    name for name in globals()
    if name.startswith("PREDICTION_ANCHORED_")
] + [
    "A3_PRIMARY_ALIAS",
    "A3_INTERACTION_RUN_IDS",
    "A3_ADDITIONAL_CANONICAL_RUN_IDS",
    "ALL50_RUN_IDS",
    "ALTERNATE_A0_RUN_ID",
    "ALTERNATE_A3_RUN_ID",
    "ALTERNATE_RUN_ID",
    "ALTERNATE_RUN_IDS",
    "NEGATIVE_CONTROL_RUN_IDS",
    "STEP8_SPECIAL_CANONICAL_RUN_IDS",
    "PERTURBATION_AUDIT_SEEDS",
    "PERTURBATION_MEAN_ACCURACY_LOSS_MAX",
    "PERTURBATION_WORST_ACCURACY_LOSS_MAX",
    "QUANTILE_COUNT",
    "AUTHORIZED_DISTRIBUTION_SPLITS",
    "Step8RunRecipe",
    "step8_run_recipes",
    "resolve_step8_run_recipe",
    "validate_step8_registry_semantics",
    "TorchAll50CorrectionScalers",
    "All50CorrectionOutput",
    "PredictionAnchoredAll50HLG",
    "all50_group_balanced_huber",
    "all50_anchor_regularization",
    "compute_step8_objective",
    "verify_all50_equal_field_semantics",
    "validate_step8_teacher_lineage",
    "prepare_step8_control_batch",
    "validate_four_control_matching",
    "Step8TrainConfig",
    "train_step8_replica",
    "apply_small_field_perturbation",
    "run_small_field_perturbation_audit",
    "correction_bridge_alignment",
    "build_bridge_quantile_reference",
    "build_bridge_quantile_reference_from_standardized_corrections",
    "bridge_distribution_distance",
    "evaluate_reliability_only_response",
    "compute_gain_and_recovery",
    "build_adversarial_channel_report",
    "Step8FixedStorage",
    "build_clean_start_step8_fixed_storage",
    "measure_step8_fixed_storage",
    "measure_step8_registry_states",
    "require_post_teacher_release",
    "run_step8_paired_seed_miniature",
]
