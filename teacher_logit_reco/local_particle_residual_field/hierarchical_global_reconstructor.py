"""Step 7 HLT-seeded hierarchical/global bridge reconstructors.

This module is deliberately separate from :mod:`hierarchical_reconstructor`:
Step 6 remains the small, dependency-light local boundary, while this file owns
the first region/global models and their direct-classification controls.

Only raw HLT particles, their valid mask, and frozen HLT-only ``R0`` outputs
may enter a deployable forward.  The bridge/oracle field is a training target;
it is never accepted by any model in this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from io import BytesIO
import hashlib
import math
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM
from teacher_logit_reco.multiscale_subjet_part import (
    MULTISCALE_SUBJET_ASSIGNMENT_CONTRACT,
    MULTISCALE_SUBJET_PAIR_FEATURE_DIM,
    MULTISCALE_SUBJET_SEED_CONTRACT,
    MULTISCALE_SUBJET_TRANSFORMER_CONTRACT,
    MultiScaleSubjetPairBiasEncoder,
    MultiScaleSubjetPairFeatureConfig,
    MultiScaleSubjetTokenBuilderOutput,
    MultiScaleSubjetTransformerBlock,
    MultiScaleSubjetTransformerConfig,
    SoftSubjetAssignment,
    SoftSubjetAssignmentConfig,
    SoftSubjetAssignmentOutput,
    SubjetScaleSpec,
    SubjetSeedBuilderConfig,
    build_multiscale_subjet_pair_features,
    build_multiscale_subjet_seeds,
    build_prepared_subjet_inputs,
    multiscale_subjet_scale_specs_for_profile,
)

from .bridge_contracts import validate_content_hash, with_content_hash
from .bridge_losses import SEMANTIC_OFFSETS, compute_c0_objective, resolve_c0_loss_recipe
from .bridge_reconstructor import (
    NUMERICAL_SPACE_CONDITIONING,
    NUMERICAL_SPACE_CORRECTION,
    NUMERICAL_SPACE_LOSS,
    NUMERICAL_SPACE_PHYSICAL,
    PHYSICAL_CHANNELS,
    TorchBridgeScalers,
    _masked_jet_summary,
)
from .hierarchical_reconstructor import (
    ARCH_A0M_CAPACITY_PARTICLE,
    KERNEL_GAUSSIAN,
    LOCAL_RADII,
    CorrectionResourceProfile,
    LocalCorrectionConfig,
    LocalCorrectionOutput,
    ParticleReasoningState,
    PredictionAnchoredBaseFusion,
    RadiusCorrectionHead,
    TargetKernelLocalProcessor,
    build_local_graph_features,
    build_step6_correction_model,
    measure_correction_resources,
    measure_step6_registry_states,
    particle_capacity_match,
)


PREDICTION_ANCHORED_HLG_CONFIG_CONTRACT = "prediction_anchored_hlg_config_v1"
PREDICTION_ANCHORED_HLG_REGION_CONTRACT = "prediction_anchored_hlg_region_pool_v1"
PREDICTION_ANCHORED_HLG_TRANSFORMER_CONTRACT = "prediction_anchored_hlg_transformer_v1"
PREDICTION_ANCHORED_HLG_READBACK_CONTRACT = "prediction_anchored_hlg_readback_v1"
PREDICTION_ANCHORED_HLG_MODEL_CONTRACT = "prediction_anchored_hlg_correction_model_v1"
PREDICTION_ANCHORED_ABSOLUTE_SCALER_CONTRACT = "prediction_anchored_absolute_output_scaler_v1"
PREDICTION_ANCHORED_DIRECT_CONFIG_CONTRACT = "prediction_anchored_hlg_direct_config_v1"
PREDICTION_ANCHORED_DIRECT_MODEL_CONTRACT = "prediction_anchored_hlg_direct_classifier_v1"
PREDICTION_ANCHORED_DIRECT_TRAIN_CONTRACT = "prediction_anchored_hlg_direct_training_v1"
PREDICTION_ANCHORED_DEPLOYED_RESOURCE_CONTRACT = "prediction_anchored_deployed_resource_reference_v1"
PREDICTION_ANCHORED_REPRESENTATIVE_RESOURCE_CONTRACT = (
    "prediction_anchored_representative_architecture_resource_reference_v1"
)
PREDICTION_ANCHORED_STEP7_RESOURCE_CONTRACT = "prediction_anchored_step7_resource_profile_v1"
PREDICTION_ANCHORED_STEP7_MEASUREMENT_CONTRACT = "prediction_anchored_step7_registry_measurement_v1"
PREDICTION_ANCHORED_STEP7_RELOAD_CONTRACT = "prediction_anchored_step7_tiny_train_reload_v1"
PREDICTION_ANCHORED_STEP7_PAIRED_MINIATURE_CONTRACT = "prediction_anchored_step7_paired_miniature_v1"

ARCH_A2_REGIONS_NO_GLOBAL = "D10_A2_regions_no_global"
ARCH_A3_HLG_PRIMARY = "D10_A3_hlg_primary"
ARCH_A4_HLG_REFINE = "D10_A4_hlg_refine"
ARCH_A5_HLG_ABSOLUTE = "D10_A5_hlg_absolute_conditioned"
ARCH_A5S_HLG_SCRATCH = "D10_A5S_hlg_scratch_physical45"
ARCH_A6_HLG_NO_PAIR = "D10_A6_hlg_no_pair_bias"
ARCH_A7_HLG_NO_H0 = "D10_A7_hlg_no_h0"
ARCH_A7F_HLG_NO_F0 = "D10_A7F_hlg_no_f0"
ARCH_A7X_HLG_NO_RAW = "D10_A7X_hlg_no_raw_skip"
ARCH_A8_HLG_FUSED_HEAD = "D10_A8_hlg_fused_radius_heads"
ARCH_A9_HLG_GROUP_GATE = "D10_A9_hlg_group_gate"
ARCH_AS_HLG_REGIONS_2_2_1 = "D10_AS_hlg_regions_2_2_1"
ARCH_AL_HLG_REGIONS_8_8_4 = "D10_AL_hlg_regions_8_8_4"
ARCH_AFIX_HLG_FIXED_ASSIGNMENT = "D10_AFIX_hlg_fixed_assignment"
ARCH_ASAME_HLG_SAME_SCALE = "D10_ASAME_hlg_same_scale_only"
ARCH_AGLOBAL_HLG_ONE_GLOBAL = "D10_AGLOBAL_hlg_one_global_token"

STEP7_HIERARCHY_ARCHITECTURE_IDS = (
    ARCH_A2_REGIONS_NO_GLOBAL,
    ARCH_A3_HLG_PRIMARY,
    ARCH_A4_HLG_REFINE,
    ARCH_A5_HLG_ABSOLUTE,
    ARCH_A5S_HLG_SCRATCH,
    ARCH_A6_HLG_NO_PAIR,
    ARCH_A7_HLG_NO_H0,
    ARCH_A7F_HLG_NO_F0,
    ARCH_A7X_HLG_NO_RAW,
    ARCH_A8_HLG_FUSED_HEAD,
    ARCH_A9_HLG_GROUP_GATE,
    ARCH_AS_HLG_REGIONS_2_2_1,
    ARCH_AL_HLG_REGIONS_8_8_4,
    ARCH_AFIX_HLG_FIXED_ASSIGNMENT,
    ARCH_ASAME_HLG_SAME_SCALE,
    ARCH_AGLOBAL_HLG_ONE_GLOBAL,
)

DIRECT_HLT = "A0_CAP500_direct_hlt"
DIRECT_R0REP = "A0_CAP500_r0rep_direct"
STEP7_DIRECT_CONTROL_IDS = (DIRECT_HLT, DIRECT_R0REP)
STEP7_MEASURED_ARCHITECTURE_IDS = STEP7_HIERARCHY_ARCHITECTURE_IDS + STEP7_DIRECT_CONTROL_IDS

OUTPUT_CORRECTION = "bounded_correction_physical45"
OUTPUT_ABSOLUTE = "bounded_absolute_physical45"
ASSIGNMENT_SOFT = "seeded_soft"
ASSIGNMENT_FIXED = "fixed_nearest_seed"
GATE_INITIAL_PROBABILITY = 0.95
GATE_INITIAL_BIAS = 2.944438979
GATE_LOSS_COEFFICIENT = 0.005
REGION_POOL_CANONICAL_INPUT_DIM = 389
REGION_TOKEN_DIM = 160
REGION_DEAD_MASS_THRESHOLD = 1.0e-3
STEP7_PAIRED_SEEDS = (101, 202, 303)


def _valid_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _scale_specs(counts: Sequence[int]) -> tuple[SubjetScaleSpec, ...]:
    counts = tuple(int(value) for value in counts)
    if counts == (4, 4, 2):
        return tuple(multiscale_subjet_scale_specs_for_profile("few_subjets"))
    if counts not in {(2, 2, 1), (8, 8, 4)}:
        raise ValueError("Step 7 region counts must be 4/4/2, 2/2/1, or 8/8/4")
    base = tuple(multiscale_subjet_scale_specs_for_profile("few_subjets"))
    return tuple(
        SubjetScaleSpec(spec.name, count, spec.radius_min, spec.radius_max, spec.role)
        for spec, count in zip(base, counts)
    )


def _locked_seed_config(specs: tuple[SubjetScaleSpec, ...]) -> SubjetSeedBuilderConfig:
    return SubjetSeedBuilderConfig(
        scale_specs=specs,
        method_by_scale={
            "small": "leading_pt",
            "medium": "local_density",
            "large": "farthest_point",
        },
        density_pt_weight=1.0,
        include_self_in_density=False,
        eps=1.0e-8,
    )


def _locked_assignment_config(specs: tuple[SubjetScaleSpec, ...]) -> SoftSubjetAssignmentConfig:
    return SoftSubjetAssignmentConfig(
        scale_specs=specs,
        query_mode="seeded",
        embed_dim=64,
        hidden_dim=128,
        temperature=0.50,
        geometry_bias_strength=2.0,
        use_scale_embedding=True,
        radius_floor=0.03,
        dead_token_weight_threshold=REGION_DEAD_MASS_THRESHOLD,
        seed_config=_locked_seed_config(specs),
    )


def _locked_transformer_config(
    *, use_pair_bias: bool, dropout: float = 0.05
) -> MultiScaleSubjetTransformerConfig:
    return MultiScaleSubjetTransformerConfig(
        token_dim=REGION_TOKEN_DIM,
        num_layers=2,
        num_heads=4,
        ffn_dim=320,
        dropout=float(dropout),
        attention_dropout=float(dropout),
        use_pairwise_bias=bool(use_pair_bias),
        use_scale_pair_embedding=True,
        pair_bias_hidden_dim=64,
        num_scales=3,
        mask_value=-1.0e4,
        pair_feature_config=MultiScaleSubjetPairFeatureConfig(
            num_scales=3,
            eps=1.0e-6,
            delta_r_scale=5.0,
            radius_scale=0.5,
            max_log_value=14.0,
            max_log_ratio=8.0,
        ),
    )


def _hlg_architecture_options(architecture_id: str) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if architecture_id == ARCH_A2_REGIONS_NO_GLOBAL:
        options["use_global_transformer"] = False
    elif architecture_id == ARCH_A4_HLG_REFINE:
        options["refinement_passes"] = 1
    elif architecture_id == ARCH_A5_HLG_ABSOLUTE:
        options["output_mode"] = OUTPUT_ABSOLUTE
    elif architecture_id == ARCH_A5S_HLG_SCRATCH:
        options.update(output_mode=OUTPUT_ABSOLUTE, use_f0_conditioning=False, use_h0_conditioning=False)
    elif architecture_id == ARCH_A6_HLG_NO_PAIR:
        options["use_pair_bias"] = False
    elif architecture_id == ARCH_A7_HLG_NO_H0:
        options["use_h0_conditioning"] = False
    elif architecture_id == ARCH_A7F_HLG_NO_F0:
        options["use_f0_conditioning"] = False
    elif architecture_id == ARCH_A7X_HLG_NO_RAW:
        options["use_raw_conditioning"] = False
    elif architecture_id == ARCH_A8_HLG_FUSED_HEAD:
        options["fused_radius_head"] = True
    elif architecture_id == ARCH_A9_HLG_GROUP_GATE:
        options["group_gate"] = True
    elif architecture_id == ARCH_AS_HLG_REGIONS_2_2_1:
        options["region_counts"] = (2, 2, 1)
    elif architecture_id == ARCH_AL_HLG_REGIONS_8_8_4:
        options["region_counts"] = (8, 8, 4)
    elif architecture_id == ARCH_AFIX_HLG_FIXED_ASSIGNMENT:
        options["assignment_mode"] = ASSIGNMENT_FIXED
    elif architecture_id == ARCH_ASAME_HLG_SAME_SCALE:
        options["same_scale_only"] = True
    elif architecture_id == ARCH_AGLOBAL_HLG_ONE_GLOBAL:
        options["one_global_token"] = True
    elif architecture_id != ARCH_A3_HLG_PRIMARY:
        raise ValueError(f"unknown Step 7 hierarchy architecture {architecture_id!r}")
    return options


@dataclass(frozen=True)
class HLGCorrectionConfig:
    architecture_id: str = ARCH_A3_HLG_PRIMARY
    region_counts: tuple[int, int, int] = (4, 4, 2)
    use_global_transformer: bool = True
    refinement_passes: int = 0
    use_pair_bias: bool = True
    same_scale_only: bool = False
    one_global_token: bool = False
    assignment_mode: str = ASSIGNMENT_SOFT
    use_raw_conditioning: bool = True
    use_f0_conditioning: bool = True
    use_h0_conditioning: bool = True
    fused_radius_head: bool = False
    group_gate: bool = False
    output_mode: str = OUTPUT_CORRECTION
    dropout: float = 0.05
    layer_norm_epsilon: float = 1.0e-5
    local_layers: int = 2
    kernel_mode: str = KERNEL_GAUSSIAN
    width: int = 160
    graph_cap: int = 32
    graph_support: float = 0.30

    def __post_init__(self) -> None:
        if self.architecture_id not in STEP7_HIERARCHY_ARCHITECTURE_IDS:
            raise ValueError(f"unknown Step 7 hierarchy architecture {self.architecture_id!r}")
        if tuple(self.region_counts) not in {(4, 4, 2), (2, 2, 1), (8, 8, 4)}:
            raise ValueError("invalid Step 7 region count profile")
        if int(self.refinement_passes) not in {0, 1}:
            raise ValueError("Step 7 permits zero or exactly one refinement pass")
        if self.assignment_mode not in {ASSIGNMENT_SOFT, ASSIGNMENT_FIXED}:
            raise ValueError("invalid Step 7 assignment mode")
        if self.output_mode not in {OUTPUT_CORRECTION, OUTPUT_ABSOLUTE}:
            raise ValueError("invalid Step 7 output mode")
        if int(self.width) != 160 or int(self.local_layers) != 2 or self.kernel_mode != KERNEL_GAUSSIAN:
            raise ValueError("Step 7 HLG local stage is locked to Gaussian width-160 depth-2")
        if int(self.graph_cap) != 32 or float(self.graph_support) != 0.30:
            raise ValueError("Step 7 graph is locked to cap 32/support 0.30")
        if float(self.layer_norm_epsilon) != 1.0e-5:
            raise ValueError("Step 7 LayerNorm epsilon is locked to 1e-5")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0,1)")
        fields = (
            "region_counts", "use_global_transformer", "refinement_passes",
            "use_pair_bias", "same_scale_only", "one_global_token",
            "assignment_mode", "use_raw_conditioning", "use_f0_conditioning",
            "use_h0_conditioning", "fused_radius_head", "group_gate", "output_mode",
        )
        expected_options = _hlg_architecture_options(self.architecture_id)
        defaults = {
            "region_counts": (4, 4, 2), "use_global_transformer": True,
            "refinement_passes": 0, "use_pair_bias": True, "same_scale_only": False,
            "one_global_token": False, "assignment_mode": ASSIGNMENT_SOFT,
            "use_raw_conditioning": True, "use_f0_conditioning": True,
            "use_h0_conditioning": True, "fused_radius_head": False,
            "group_gate": False, "output_mode": OUTPUT_CORRECTION,
        }
        expected_values = {**defaults, **expected_options}
        differences = [name for name in fields if getattr(self, name) != expected_values[name]]
        if differences:
            raise ValueError(f"{self.architecture_id} changed locked fields: {differences}")

    @classmethod
    def for_architecture(cls, architecture_id: str, *, dropout: float = 0.05) -> "HLGCorrectionConfig":
        return cls(
            architecture_id=architecture_id,
            dropout=float(dropout),
            **_hlg_architecture_options(architecture_id),
        )

    @property
    def total_regions(self) -> int:
        return int(sum(self.region_counts))

    @property
    def region_pool_input_dim(self) -> int:
        return 160 + (50 if self.use_f0_conditioning else 0) + (160 if self.use_h0_conditioning else 0) + (
            RAW_TOKEN_DIM if self.use_raw_conditioning else 0
        ) + 5

    def to_artifact(self) -> dict[str, Any]:
        specs = _scale_specs(self.region_counts)
        assignment = _locked_assignment_config(specs)
        transformer = _locked_transformer_config(
            use_pair_bias=self.use_pair_bias, dropout=self.dropout
        )
        return with_content_hash(
            {
                "contract": PREDICTION_ANCHORED_HLG_CONFIG_CONTRACT,
                **asdict(self),
                "region_counts": list(self.region_counts),
                "region_pool_input_dim": self.region_pool_input_dim,
                "canonical_region_pool_input_dim": REGION_POOL_CANONICAL_INPUT_DIM,
                "region_scale_specs": [spec.to_dict() for spec in specs],
                "seed_config": {
                    **asdict(assignment.seed_config),
                    "scale_specs": [spec.to_dict() for spec in specs],
                },
                "assignment_config": {
                    **asdict(assignment),
                    "scale_specs": [spec.to_dict() for spec in specs],
                    "seed_config": "seed_config_above",
                },
                "transformer_config": {
                    **asdict(transformer),
                    "pair_feature_config": asdict(transformer.pair_feature_config),
                },
                "upstream_contract_versions": {
                    "seed": MULTISCALE_SUBJET_SEED_CONTRACT,
                    "assignment": MULTISCALE_SUBJET_ASSIGNMENT_CONTRACT,
                    "transformer": MULTISCALE_SUBJET_TRANSFORMER_CONTRACT,
                },
                "assignment_mass_semantics": "scale_wise_cluster_membership_sum_not_attention_sum",
                "hlt_only_region_provenance": True,
                "oracle_or_bridge_input_present": False,
                "reliability_channels": "exact_f0_pass_through",
            }
        )


def _config_for_architecture(architecture_id: str, *, dropout: float) -> HLGCorrectionConfig:
    """Validated factory kept outside the dataclass to avoid recursive construction."""

    config = HLGCorrectionConfig.for_architecture(architecture_id, dropout=dropout)
    # Validate factory output deterministically without calling __post_init__ recursively.
    if config.total_regions not in {5, 10, 20} or config.region_pool_input_dim <= 0:
        raise AssertionError("invalid factory-generated HLG config")
    return config


def fit_absolute_output_scaler(
    batches: Iterable[tuple[Any, Any]],
    *,
    source_manifest_sha256: str,
    bridge_recipe_sha256: str,
    epsilon: Sequence[float] | np.ndarray,
) -> dict[str, Any]:
    """Fit the locked q0.001/q0.999 physical-45 absolute-output bounds.

    ``batches`` contains ``(bridge_fields, particle_mask)`` pairs.  This helper
    is intended to run inside the allocation that already holds derived bridge
    values in RAM; it writes only the 45-channel statistics artifact.
    """

    if not _valid_sha256(source_manifest_sha256) or not _valid_sha256(bridge_recipe_sha256):
        raise ValueError("absolute scaler requires source-manifest and bridge-recipe SHA-256 values")
    eps = np.asarray(epsilon, dtype=np.float64)
    if eps.shape == (50,):
        eps = eps[:PHYSICAL_CHANNELS]
    if eps.shape != (PHYSICAL_CHANNELS,) or not np.isfinite(eps).all() or np.any(eps <= 0):
        raise ValueError("absolute scaler epsilon must contain 45 positive finite values")
    columns: list[list[np.ndarray]] = [[] for _ in range(PHYSICAL_CHANNELS)]
    valid_entries = 0
    batch_count = 0
    for fields_value, mask_value in batches:
        fields = np.asarray(fields_value, dtype=np.float64)
        mask = np.asarray(mask_value, dtype=bool)
        if fields.ndim != 3 or fields.shape[-1] != 50 or mask.shape != fields.shape[:2]:
            raise ValueError("absolute scaler bridge/mask shapes do not align")
        selected = fields[..., :PHYSICAL_CHANNELS][mask]
        if selected.size and not np.isfinite(selected).all():
            raise ValueError("absolute scaler input contains non-finite valid values")
        for index in range(PHYSICAL_CHANNELS):
            if selected.size:
                columns[index].append(selected[:, index])
        valid_entries += int(mask.sum())
        batch_count += 1
    if valid_entries <= 0 or batch_count <= 0:
        raise ValueError("absolute scaler requires at least one valid particle")
    lo = np.asarray(
        [np.quantile(np.concatenate(values), 0.001) for values in columns], dtype=np.float64
    )
    hi = np.asarray(
        [np.quantile(np.concatenate(values), 0.999) for values in columns], dtype=np.float64
    )
    center = (lo + hi) / 2.0
    half_range = np.maximum((hi - lo) / 2.0, eps)
    return with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_ABSOLUTE_SCALER_CONTRACT,
            "source_manifest_sha256": str(source_manifest_sha256),
            "bridge_recipe_sha256": str(bridge_recipe_sha256),
            "fit_partition": "stack_train_distill",
            "channel_policy": "physical45",
            "quantiles": [0.001, 0.999],
            "lo": lo.tolist(),
            "hi": hi.tolist(),
            "center": center.tolist(),
            "half_range": half_range.tolist(),
            "epsilon": eps.tolist(),
            "valid_particle_count": int(valid_entries),
            "batch_count": int(batch_count),
            "accumulation_dtype": "float64",
            "quantile_method": "numpy_linear_exact_from_ram_batches",
            "derived_dense_fields_persisted": False,
        }
    )


class TorchAbsoluteOutputScaler(torch.nn.Module):
    def __init__(self, artifact: Mapping[str, Any]) -> None:
        super().__init__()
        validate_content_hash(artifact, expected_contract=PREDICTION_ANCHORED_ABSOLUTE_SCALER_CONTRACT)
        self.artifact_sha256 = str(artifact["content_hash"])
        for name in ("center", "half_range"):
            value = torch.as_tensor(artifact[name], dtype=torch.float32)
            if tuple(value.shape) != (PHYSICAL_CHANNELS,) or not bool(torch.isfinite(value).all()):
                raise ValueError(f"absolute scaler {name} must contain 45 finite values")
            if name == "half_range" and bool((value <= 0).any()):
                raise ValueError("absolute scaler half_range must be positive")
            self.register_buffer(name, value, persistent=True)

    def physical(self, raw_absolute: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if raw_absolute.shape != (*mask.shape, PHYSICAL_CHANNELS):
            raise ValueError("absolute head must emit [batch, particles, 45]")
        output = self.center + self.half_range * torch.tanh(raw_absolute)
        return output.masked_fill(~mask.unsqueeze(-1), 0.0)


class SelectiveBaseFusion(torch.nn.Module):
    """Locked base fusion with physically removed branches for input ablations."""

    def __init__(self, config: HLGCorrectionConfig) -> None:
        super().__init__()
        self.config = config
        eps = float(config.layer_norm_epsilon)
        if config.use_raw_conditioning:
            self.raw_projection: torch.nn.Module | None = torch.nn.Sequential(
                torch.nn.LayerNorm(RAW_TOKEN_DIM, eps=eps),
                torch.nn.Linear(RAW_TOKEN_DIM, 64),
                torch.nn.GELU(),
            )
        else:
            self.raw_projection = None
        if config.use_f0_conditioning:
            self.f0_projection: torch.nn.Module | None = torch.nn.Sequential(
                torch.nn.LayerNorm(50, eps=eps), torch.nn.Linear(50, 64), torch.nn.GELU()
            )
        else:
            self.f0_projection = None
        if config.use_h0_conditioning:
            self.h0_projection: torch.nn.Module | None = torch.nn.Sequential(
                torch.nn.LayerNorm(160, eps=eps), torch.nn.Linear(160, 96), torch.nn.GELU()
            )
        else:
            self.h0_projection = None
        self.jet_projection = torch.nn.Sequential(torch.nn.Linear(11, 32), torch.nn.GELU())
        input_width = 32
        input_width += 64 if self.raw_projection is not None else 0
        input_width += 64 if self.f0_projection is not None else 0
        input_width += 96 if self.h0_projection is not None else 0
        self.input_width = input_width
        self.fusion = torch.nn.Sequential(
            torch.nn.LayerNorm(input_width, eps=eps),
            torch.nn.Linear(input_width, REGION_TOKEN_DIM),
            torch.nn.GELU(),
            torch.nn.LayerNorm(REGION_TOKEN_DIM, eps=eps),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        standardized_f0: torch.Tensor | None,
        h0: torch.Tensor | None,
    ) -> torch.Tensor:
        values: list[torch.Tensor] = []
        if self.raw_projection is not None:
            values.append(self.raw_projection(tokens))
        if self.f0_projection is not None:
            if standardized_f0 is None:
                raise ValueError("f0-conditioned HLG base did not receive standardized f0")
            values.append(self.f0_projection(standardized_f0))
        if self.h0_projection is not None:
            if h0 is None:
                raise ValueError("h0-conditioned HLG base did not receive h0")
            values.append(self.h0_projection(h0))
        jet = self.jet_projection(_masked_jet_summary(tokens, mask))
        values.append(jet[:, None, :].expand(-1, tokens.shape[1], -1))
        hidden = self.fusion(torch.cat(values, dim=-1))
        return hidden.masked_fill(~mask.unsqueeze(-1), 0.0)


@dataclass(frozen=True)
class HLGRegionOutput:
    region_tokens: torch.Tensor
    region_mask: torch.Tensor
    assignment_output: SoftSubjetAssignmentOutput
    pair_token_output: MultiScaleSubjetTokenBuilderOutput
    assignment_mass: torch.Tensor
    pool_inputs: torch.Tensor
    diagnostics: Mapping[str, Any]


def _fixed_nearest_seed_assignment(
    tokens: torch.Tensor,
    mask: torch.Tensor,
    *,
    specs: tuple[SubjetScaleSpec, ...],
) -> SoftSubjetAssignmentOutput:
    """Deterministic per-scale nearest valid seed assignment."""

    prepared = build_prepared_subjet_inputs(tokens, mask)
    seed_config = _locked_seed_config(specs)
    seeds = build_multiscale_subjet_seeds(tokens, mask, config=seed_config)
    batch, particles = mask.shape
    regions = int(sum(spec.num_tokens for spec in specs))
    cluster = tokens.new_zeros((batch, regions, particles), dtype=torch.float32)
    logits = tokens.new_full((batch, regions, particles), -1.0e4, dtype=torch.float32)
    scale_index = seeds.scale_index.to(device=tokens.device, dtype=torch.long)
    coordinates = prepared.coordinates.float()
    for scale_id in range(len(specs)):
        slots = torch.nonzero(scale_index == scale_id, as_tuple=False).flatten()
        centers = seeds.centers[:, slots].float()
        valid_seeds = seeds.mask[:, slots]
        delta_eta = centers[:, :, None, 0] - coordinates[:, None, :, 0]
        delta_phi = torch.atan2(
            torch.sin(centers[:, :, None, 1] - coordinates[:, None, :, 1]),
            torch.cos(centers[:, :, None, 1] - coordinates[:, None, :, 1]),
        )
        distance2 = delta_eta.square() + delta_phi.square()
        distance2 = distance2.masked_fill(~valid_seeds[:, :, None], float("inf"))
        nearest = distance2.argmin(dim=1)
        any_seed = valid_seeds.any(dim=1)
        for local_slot, global_slot in enumerate(slots.tolist()):
            member = (nearest == local_slot) & mask & any_seed[:, None]
            cluster[:, global_slot, :] = member.to(dtype=cluster.dtype)
            logits[:, global_slot, :] = torch.where(
                mask,
                -distance2[:, local_slot],
                logits[:, global_slot, :],
            )
    mass = cluster.sum(dim=-1)
    attention = cluster / torch.clamp(mass.unsqueeze(-1), min=1.0)
    region_mask = seeds.mask & (mass >= REGION_DEAD_MASS_THRESHOLD)
    attention = attention * region_mask.unsqueeze(-1).to(dtype=attention.dtype)
    cluster = cluster * region_mask.unsqueeze(-1).to(dtype=cluster.dtype)
    centers = torch.einsum("brp,bpd->brd", attention, coordinates)
    estimated_pt = torch.einsum("brp,bp->br", attention, prepared.pt_fraction.float())
    return SoftSubjetAssignmentOutput(
        assignment_weights=attention,
        cluster_weights=cluster,
        logits=logits,
        subjet_mask=region_mask,
        particle_mask=mask,
        seed_output=seeds,
        query_mode=ASSIGNMENT_FIXED,
        scale_index=scale_index,
        scale_radius=seeds.scale_radius.to(device=tokens.device, dtype=torch.float32),
        estimated_centers=centers,
        estimated_pt_fraction=estimated_pt,
        diagnostics={
            "contract": MULTISCALE_SUBJET_ASSIGNMENT_CONTRACT,
            "seed_contract": MULTISCALE_SUBJET_SEED_CONTRACT,
            "assignment_mode": ASSIGNMENT_FIXED,
            "deterministic": True,
            "tie_policy": "ascending_seed_slot_in_deterministic_seed_order",
            "hlt_only": True,
        },
    )


def _pair_token_output(
    tokens: torch.Tensor,
    mask: torch.Tensor,
    assignment: SoftSubjetAssignmentOutput,
    region_tokens: torch.Tensor,
    region_mask: torch.Tensor,
) -> MultiScaleSubjetTokenBuilderOutput:
    prepared = build_prepared_subjet_inputs(tokens, mask)
    cluster = assignment.cluster_weights.float()
    four = torch.stack((prepared.px, prepared.py, prepared.pz, prepared.energy), dim=-1).float()
    soft_four = torch.einsum("brp,bpd->brd", cluster, four)
    cluster_pt = torch.einsum("brp,bp->br", cluster, prepared.pt_fraction.float())
    zeros7 = region_tokens.new_zeros((*region_tokens.shape[:2], 7))
    zeros5 = region_tokens.new_zeros((*region_tokens.shape[:2], 5))
    return MultiScaleSubjetTokenBuilderOutput(
        subjet_tokens=region_tokens,
        subjet_mask=region_mask,
        assignment_weights=assignment.assignment_weights,
        cluster_weights=assignment.cluster_weights,
        assignment_logits=assignment.logits,
        estimated_centers=assignment.estimated_centers,
        estimated_pt_fraction=assignment.estimated_pt_fraction,
        cluster_pt_fraction=cluster_pt,
        soft_four_vectors=soft_four,
        soft_four_vector_features=zeros7,
        soft_pair_observable_summaries=zeros5,
        scale_index=assignment.scale_index,
        scale_radius=assignment.scale_radius,
        assignment_output=assignment,
        diagnostics={
            "contract": PREDICTION_ANCHORED_HLG_REGION_CONTRACT,
            "pair_geometry_from_hlt_only": True,
        },
    )


class HLGRegionPooler(torch.nn.Module):
    """Exact assignment-mass-aware region pooling and 389->320->160 projection."""

    def __init__(self, config: HLGCorrectionConfig) -> None:
        super().__init__()
        self.config = config
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(config.region_pool_input_dim, 320),
            torch.nn.GELU(),
            torch.nn.Dropout(float(config.dropout)),
            torch.nn.Linear(320, REGION_TOKEN_DIM),
        )
        self.scale_embedding = torch.nn.Embedding(3, REGION_TOKEN_DIM)
        self.output_norm = torch.nn.LayerNorm(REGION_TOKEN_DIM, eps=float(config.layer_norm_epsilon))
        torch.nn.init.normal_(self.scale_embedding.weight, mean=0.0, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        streams: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        standardized_f0: torch.Tensor | None,
        h0: torch.Tensor | None,
        assignment: SoftSubjetAssignmentOutput,
    ) -> HLGRegionOutput:
        weights = assignment.cluster_weights.float() * mask[:, None, :].to(dtype=torch.float32)
        mass = weights.sum(dim=-1)
        denominator = torch.clamp(mass, min=1.0e-12).unsqueeze(-1)
        scale_index = assignment.scale_index.to(device=tokens.device, dtype=torch.long)
        local_by_region = torch.stack(streams, dim=1).index_select(1, scale_index)

        def pool(value: torch.Tensor) -> torch.Tensor:
            return (weights.unsqueeze(-1) * value).sum(dim=2) / denominator

        pieces = [pool(local_by_region)]
        if self.config.use_f0_conditioning:
            if standardized_f0 is None:
                raise ValueError("region pool requires standardized f0")
            pieces.append(torch.einsum("brp,bpd->brd", weights, standardized_f0) / denominator)
        if self.config.use_h0_conditioning:
            if h0 is None:
                raise ValueError("region pool requires h0")
            pieces.append(torch.einsum("brp,bpd->brd", weights, h0) / denominator)
        if self.config.use_raw_conditioning:
            pieces.append(torch.einsum("brp,bpd->brd", weights, tokens.float()) / denominator)
        seed_pt = assignment.seed_output.seed_pt_fraction if assignment.seed_output is not None else assignment.estimated_pt_fraction
        scale_one_hot = F.one_hot(scale_index, num_classes=3).to(dtype=tokens.dtype)
        metadata = torch.cat(
            (
                torch.log1p(mass).unsqueeze(-1),
                seed_pt.to(device=tokens.device, dtype=tokens.dtype).unsqueeze(-1),
                scale_one_hot[None, :, :].expand(tokens.shape[0], -1, -1),
            ),
            dim=-1,
        )
        pieces.append(metadata)
        pool_inputs = torch.cat(pieces, dim=-1)
        if int(pool_inputs.shape[-1]) != int(self.config.region_pool_input_dim):
            raise RuntimeError("Step 7 region pool width disagrees with its config")
        region_mask = assignment.subjet_mask & (mass >= REGION_DEAD_MASS_THRESHOLD)
        projected = self.projection(pool_inputs)
        region_tokens = self.output_norm(projected + self.scale_embedding(scale_index)[None, :, :])
        region_tokens = region_tokens.masked_fill(~region_mask.unsqueeze(-1), 0.0)
        pool_inputs = pool_inputs.masked_fill(~region_mask.unsqueeze(-1), 0.0)
        pair_tokens = _pair_token_output(tokens, mask, assignment, region_tokens, region_mask)
        valid_count = torch.clamp(region_mask.float().sum(), min=1.0)
        entropy = -(assignment.assignment_weights * assignment.assignment_weights.clamp_min(1.0e-12).log()).sum(-1)
        diagnostics = {
            "contract": PREDICTION_ANCHORED_HLG_REGION_CONTRACT,
            "pool_input_dim": int(pool_inputs.shape[-1]),
            "canonical_pool_input_dim": REGION_POOL_CANONICAL_INPUT_DIM,
            "assignment_mass_semantics": "scale_wise_cluster_membership_sum",
            "dead_mass_threshold": REGION_DEAD_MASS_THRESHOLD,
            "empty_region_count": int((~region_mask).sum().detach().cpu()),
            "assignment_entropy_mean": float((entropy * region_mask).sum().detach().cpu() / valid_count.detach().cpu()),
            "padded_particle_mass_exact_zero": bool(torch.count_nonzero(weights * (~mask[:, None, :])).item() == 0),
            "hlt_only_provenance": True,
        }
        return HLGRegionOutput(region_tokens, region_mask, assignment, pair_tokens, mass, pool_inputs, diagnostics)


@dataclass(frozen=True)
class HLGGlobalOutput:
    region_tokens: torch.Tensor
    region_mask: torch.Tensor
    pair_features: torch.Tensor | None
    pair_mask: torch.Tensor | None
    pair_bias: torch.Tensor | None
    attention_weights: torch.Tensor | None
    diagnostics: Mapping[str, Any]


class HLGRegionTransformer(torch.nn.Module):
    """Locked two-layer/four-head region transformer with optional scale mask."""

    def __init__(self, config: HLGCorrectionConfig) -> None:
        super().__init__()
        self.hlg_config = config
        self.config = _locked_transformer_config(
            use_pair_bias=config.use_pair_bias, dropout=config.dropout
        )
        self.pair_bias_encoder = MultiScaleSubjetPairBiasEncoder(self.config)
        self.layers = torch.nn.ModuleList(
            [MultiScaleSubjetTransformerBlock(self.config) for _ in range(2)]
        )
        self.output_norm = torch.nn.LayerNorm(REGION_TOKEN_DIM, eps=1.0e-5)

    def forward(self, region: HLGRegionOutput, *, need_weights: bool = False) -> HLGGlobalOutput:
        tokens = region.region_tokens.float()
        mask = region.region_mask.bool()
        safe_mask = mask.clone()
        empty_rows = ~safe_mask.any(dim=1)
        if bool(empty_rows.any()):
            safe_mask[empty_rows, 0] = True
            tokens = tokens.clone()
            tokens[empty_rows, 0] = 0.0
        pair_output = build_multiscale_subjet_pair_features(
            region.pair_token_output, self.config.pair_feature_config
        )
        pair_bias = self.pair_bias_encoder(pair_output) if self.hlg_config.use_pair_bias else None
        cross_scale_mask = None
        if self.hlg_config.same_scale_only:
            scale = region.assignment_output.scale_index.to(device=tokens.device)
            cross_scale_mask = scale[:, None] != scale[None, :]
            blocked = cross_scale_mask[None, None, :, :]
            if pair_bias is None:
                pair_bias = tokens.new_zeros((tokens.shape[0], 4, tokens.shape[1], tokens.shape[1]))
            pair_bias = pair_bias.masked_fill(blocked, float(self.config.mask_value))
        x = tokens.masked_fill(~safe_mask.unsqueeze(-1), 0.0)
        weights_by_layer = []
        for layer in self.layers:
            x, weights = layer(x, safe_mask, pair_bias=pair_bias, need_weights=need_weights)
            x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
            if weights is not None:
                weights = weights.masked_fill(~mask[:, None, :, None], 0.0)
                weights = weights.masked_fill(~mask[:, None, None, :], 0.0)
                weights_by_layer.append(weights)
        x = self.output_norm(x).masked_fill(~mask.unsqueeze(-1), 0.0)
        attention = torch.stack(weights_by_layer, dim=0) if weights_by_layer else None
        return HLGGlobalOutput(
            region_tokens=x,
            region_mask=mask,
            pair_features=pair_output.pair_features,
            pair_mask=pair_output.pair_mask,
            pair_bias=pair_bias,
            attention_weights=attention,
            diagnostics={
                "contract": PREDICTION_ANCHORED_HLG_TRANSFORMER_CONTRACT,
                "token_dim": 160,
                "num_layers": 2,
                "num_heads": 4,
                "ffn_dim": 320,
                "dropout": float(self.config.dropout),
                "attention_dropout": float(self.config.attention_dropout),
                "mask_value": float(self.config.mask_value),
                "pair_feature_dim": MULTISCALE_SUBJET_PAIR_FEATURE_DIM,
                "pair_bias_present": pair_bias is not None,
                "same_scale_only": bool(self.hlg_config.same_scale_only),
                "cross_scale_blocked_pairs": 0 if cross_scale_mask is None else int(cross_scale_mask.sum()),
                "empty_batch_rows_safely_masked": int(empty_rows.sum().detach().cpu()),
                "hlt_only_pair_features": True,
            },
        )


class HLGParticleReadback(torch.nn.Module):
    """Exact pre-norm query cross-attention and residual readback equation."""

    def __init__(self, *, dropout: float) -> None:
        super().__init__()
        self.query_norm = torch.nn.LayerNorm(160, eps=1.0e-5)
        self.attention = torch.nn.MultiheadAttention(
            160, 4, dropout=float(dropout), batch_first=True
        )
        self.output_projection = torch.nn.Linear(160, 160)
        self.dropout = torch.nn.Dropout(float(dropout))
        self.output_norm = torch.nn.LayerNorm(160, eps=1.0e-5)

    def forward(
        self,
        particle_hidden: torch.Tensor,
        particle_mask: torch.Tensor,
        region_tokens: torch.Tensor,
        region_mask: torch.Tensor,
        *,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, Mapping[str, Any]]:
        safe_region_mask = region_mask.clone()
        safe_tokens = region_tokens
        empty = ~safe_region_mask.any(dim=1)
        if bool(empty.any()):
            safe_region_mask[empty, 0] = True
            safe_tokens = region_tokens.clone()
            safe_tokens[empty, 0] = 0.0
        context, weights = self.attention(
            self.query_norm(particle_hidden),
            safe_tokens,
            safe_tokens,
            key_padding_mask=~safe_region_mask,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        readback = self.output_norm(
            particle_hidden + self.dropout(self.output_projection(context))
        )
        readback = readback.masked_fill(~particle_mask.unsqueeze(-1), 0.0)
        if weights is not None:
            weights = weights.masked_fill(~particle_mask[:, None, :, None], 0.0)
            weights = weights.masked_fill(~region_mask[:, None, None, :], 0.0)
        return readback, weights, {
            "contract": PREDICTION_ANCHORED_HLG_READBACK_CONTRACT,
            "query_shape": list(particle_hidden.shape),
            "key_value_shape": list(region_tokens.shape),
            "num_heads": 4,
            "width": 160,
            "equation": "LN(particle_hidden + Dropout(W_o(MHA(LN(particle_hidden),regions,regions))))",
            "empty_region_rows_safely_masked": int(empty.sum().detach().cpu()),
        }


@dataclass(frozen=True)
class HLGEncoding:
    base_hidden: torch.Tensor
    radius_streams: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    readback: torch.Tensor
    particle_mask: torch.Tensor
    region_tokens: torch.Tensor
    region_mask: torch.Tensor
    assignment_output: SoftSubjetAssignmentOutput
    region_output: HLGRegionOutput
    global_output: HLGGlobalOutput | None
    readback_weights: torch.Tensor | None
    graph_edge_count: int
    diagnostics: Mapping[str, Any]


class HLGParticleEncoder(torch.nn.Module):
    """Shared local -> HLT-region -> global -> particle-readback backbone."""

    def __init__(self, config: HLGCorrectionConfig) -> None:
        super().__init__()
        self.config = config
        if config.use_raw_conditioning and config.use_f0_conditioning and config.use_h0_conditioning:
            self.base_fusion: torch.nn.Module = PredictionAnchoredBaseFusion(
                dropout=float(config.dropout), layer_norm_epsilon=float(config.layer_norm_epsilon)
            )
        else:
            self.base_fusion = SelectiveBaseFusion(config)
        local_config = LocalCorrectionConfig.for_architecture(
            "D10_A1_multiscale_local", dropout=float(config.dropout)
        )
        self.local_processor = TargetKernelLocalProcessor(local_config)
        specs = _scale_specs(config.region_counts)
        self.specs = specs
        if config.assignment_mode == ASSIGNMENT_SOFT:
            self.assignment: SoftSubjetAssignment | None = SoftSubjetAssignment(
                _locked_assignment_config(specs)
            )
        else:
            self.assignment = None
        self.region_pooler = HLGRegionPooler(config)
        self.region_transformer = HLGRegionTransformer(config) if config.use_global_transformer else None
        self.readback = HLGParticleReadback(dropout=float(config.dropout))
        if config.refinement_passes == 1:
            self.refinement_transformer: HLGRegionTransformer | None = HLGRegionTransformer(config)
            self.refinement_readback: HLGParticleReadback | None = HLGParticleReadback(
                dropout=float(config.dropout)
            )
        else:
            self.refinement_transformer = None
            self.refinement_readback = None

    def _assignment(self, tokens: torch.Tensor, mask: torch.Tensor) -> SoftSubjetAssignmentOutput:
        if self.assignment is None:
            return _fixed_nearest_seed_assignment(tokens, mask, specs=self.specs)
        return self.assignment(tokens, mask)

    @staticmethod
    def _collapse_global_token(global_output: HLGGlobalOutput) -> HLGGlobalOutput:
        weights = global_output.region_mask.to(dtype=global_output.region_tokens.dtype)
        denominator = torch.clamp(weights.sum(dim=1, keepdim=True), min=1.0)
        token = (global_output.region_tokens * weights.unsqueeze(-1)).sum(dim=1, keepdim=True) / denominator.unsqueeze(-1)
        mask = global_output.region_mask.any(dim=1, keepdim=True)
        token = token.masked_fill(~mask.unsqueeze(-1), 0.0)
        return HLGGlobalOutput(
            token,
            mask,
            global_output.pair_features,
            global_output.pair_mask,
            global_output.pair_bias,
            global_output.attention_weights,
            {**global_output.diagnostics, "one_global_token_after_region_reasoning": True},
        )

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        standardized_f0: torch.Tensor | None,
        h0: torch.Tensor | None,
        *,
        need_attention_weights: bool = False,
    ) -> HLGEncoding:
        base = self.base_fusion(tokens, mask, standardized_f0, h0)
        graph = build_local_graph_features(tokens, mask, kernel_mode=KERNEL_GAUSSIAN)
        streams = self.local_processor(base, graph, mask)
        assignment = self._assignment(tokens, mask)
        region = self.region_pooler(tokens, mask, streams, standardized_f0, h0, assignment)
        global_output = None
        if self.region_transformer is not None:
            global_output = self.region_transformer(region, need_weights=need_attention_weights)
            active_tokens = global_output.region_tokens
            active_mask = global_output.region_mask
        else:
            active_tokens = region.region_tokens
            active_mask = region.region_mask
        if self.config.one_global_token:
            if global_output is None:
                pseudo = HLGGlobalOutput(active_tokens, active_mask, None, None, None, None, {})
                collapsed = self._collapse_global_token(pseudo)
            else:
                collapsed = self._collapse_global_token(global_output)
            active_tokens, active_mask = collapsed.region_tokens, collapsed.region_mask
            global_output = collapsed
        readback, readback_weights, readback_diagnostics = self.readback(
            base, mask, active_tokens, active_mask, need_weights=need_attention_weights
        )
        refinement_count = 0
        if self.config.refinement_passes:
            assert self.refinement_transformer is not None and self.refinement_readback is not None
            # The first readback delta is the one and only particle update before repooling.
            update = readback - base
            updated_streams = tuple(
                (stream + update).masked_fill(~mask.unsqueeze(-1), 0.0) for stream in streams
            )
            refined_region = self.region_pooler(
                tokens, mask, updated_streams, standardized_f0, h0, assignment
            )
            refined_global = self.refinement_transformer(
                refined_region, need_weights=need_attention_weights
            )
            readback, readback_weights, readback_diagnostics = self.refinement_readback(
                readback,
                mask,
                refined_global.region_tokens,
                refined_global.region_mask,
                need_weights=need_attention_weights,
            )
            region = refined_region
            global_output = refined_global
            active_tokens = refined_global.region_tokens
            active_mask = refined_global.region_mask
            refinement_count = 1
        diagnostics = {
            "contract": PREDICTION_ANCHORED_HLG_MODEL_CONTRACT,
            "architecture_id": self.config.architecture_id,
            "hlt_only_region_provenance": True,
            "oracle_or_bridge_input_present": False,
            "region_count": int(active_tokens.shape[1]),
            "original_region_count": self.config.total_regions,
            "region_pool": dict(region.diagnostics),
            "global": None if global_output is None else dict(global_output.diagnostics),
            "readback": dict(readback_diagnostics),
            "refinement_pass_count": refinement_count,
            "seeds_recomputed_during_refinement": False,
            "assignments_recomputed_during_refinement": False,
            "directed_edge_count": int(graph.graph.edge_valid.sum().detach().cpu()),
        }
        return HLGEncoding(
            base,
            streams,
            readback,
            mask,
            active_tokens,
            active_mask,
            assignment,
            region,
            global_output,
            readback_weights,
            int(graph.graph.edge_valid.sum().detach().cpu()),
            diagnostics,
        )


class HLGRadiusCorrectionHead(torch.nn.Module):
    """Canonical head that also exposes its 160-value fused state for A9."""

    def __init__(self, output_dim: int, *, dropout: float, zero_init: bool = True) -> None:
        super().__init__()
        self.input_norm = torch.nn.LayerNorm(480, eps=1.0e-5)
        self.fused = torch.nn.Linear(480, 160)
        self.dropout1 = torch.nn.Dropout(float(dropout))
        self.hidden128 = torch.nn.Linear(160, 128)
        self.dropout2 = torch.nn.Dropout(float(dropout))
        self.hidden64 = torch.nn.Linear(128, 64)
        self.dropout3 = torch.nn.Dropout(float(dropout))
        self.output = torch.nn.Linear(64, int(output_dim))
        if zero_init:
            torch.nn.init.zeros_(self.output.weight)
            torch.nn.init.zeros_(self.output.bias)

    def forward(
        self, base: torch.Tensor, radius: torch.Tensor, readback: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fused = self.dropout1(F.gelu(self.fused(self.input_norm(torch.cat((base, radius, readback), dim=-1)))))
        hidden = self.dropout2(F.gelu(self.hidden128(fused)))
        hidden = self.dropout3(F.gelu(self.hidden64(hidden)))
        return self.output(hidden), fused


@dataclass(frozen=True)
class HLGCorrectionOutput(LocalCorrectionOutput):
    gate_values: torch.Tensor | None
    gate_loss: torch.Tensor
    pre_gate_physical_correction: torch.Tensor


class PredictionAnchoredHLGCorrection(torch.nn.Module):
    """All Step 7 field-producing HLG variants behind one locked API."""

    def __init__(
        self,
        scaler_artifact: Mapping[str, Any],
        config: HLGCorrectionConfig,
        *,
        absolute_scaler_artifact: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.scalers = TorchBridgeScalers(scaler_artifact)
        if config.output_mode == OUTPUT_ABSOLUTE:
            if absolute_scaler_artifact is None:
                raise ValueError(f"{config.architecture_id} requires the absolute-output scaler artifact")
            self.absolute_scaler: TorchAbsoluteOutputScaler | None = TorchAbsoluteOutputScaler(
                absolute_scaler_artifact
            )
        else:
            if absolute_scaler_artifact is not None:
                raise ValueError("correction-output HLG models reject an unused absolute scaler")
            self.absolute_scaler = None
        self.encoder = HLGParticleEncoder(config)
        if config.fused_radius_head:
            self.radius_fusion: torch.nn.Module | None = torch.nn.Sequential(
                torch.nn.LayerNorm(480, eps=1.0e-5),
                torch.nn.Linear(480, 160),
                torch.nn.GELU(),
                torch.nn.Dropout(float(config.dropout)),
            )
            self.radius_heads = torch.nn.ModuleList(
                [HLGRadiusCorrectionHead(45, dropout=float(config.dropout))]
            )
        else:
            self.radius_fusion = None
            self.radius_heads = torch.nn.ModuleList(
                [HLGRadiusCorrectionHead(15, dropout=float(config.dropout)) for _ in LOCAL_RADII]
            )
        if config.group_gate:
            self.gate_heads: torch.nn.ModuleList | None = torch.nn.ModuleList(
                [torch.nn.Linear(160, 4) for _ in LOCAL_RADII]
            )
            for gate in self.gate_heads:
                torch.nn.init.zeros_(gate.weight)
                torch.nn.init.constant_(gate.bias, GATE_INITIAL_BIAS)
                for parameter in gate.parameters():
                    parameter.requires_grad_(False)
            self._gate_phase = "field_warmup"
        else:
            self.gate_heads = None
            self._gate_phase = "not_applicable"

    @property
    def scaler_sha256(self) -> str:
        return self.scalers.artifact_sha256

    def config_artifact(self) -> dict[str, Any]:
        config = self.config.to_artifact()
        return with_content_hash(
            {
                "contract": PREDICTION_ANCHORED_HLG_MODEL_CONTRACT,
                "architecture_id": self.config.architecture_id,
                "config_sha256": config["content_hash"],
                "scaler_sha256": self.scaler_sha256,
                "absolute_scaler_sha256": None if self.absolute_scaler is None else self.absolute_scaler.artifact_sha256,
                "config": config,
                "inputs": ["raw_hlt", "valid_mask"] + (
                    ["f0_physical", "stop_gradient_h0"] if self.config.architecture_id != ARCH_A5S_HLG_SCRATCH else ["f0_reliability_passthrough_only"]
                ),
                "oracle_input_present": False,
                "deployable_without_teacher": True,
                "output_space": NUMERICAL_SPACE_PHYSICAL,
                "gate_phase": self._gate_phase,
            }
        )

    def set_training_phase(self, phase: str) -> None:
        if phase not in {"field_warmup", "distillation"}:
            raise ValueError("HLG phase must be field_warmup or distillation")
        if self.gate_heads is not None:
            trainable = phase == "distillation"
            for parameter in self.gate_heads.parameters():
                parameter.requires_grad_(trainable)
            self._gate_phase = phase

    def _validate_inputs(
        self,
        hlt_tokens: torch.Tensor,
        mask: torch.Tensor,
        f0_live: torch.Tensor,
        h0: torch.Tensor | None,
        f0_space: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = hlt_tokens.float()
        valid = mask.to(device=tokens.device, dtype=torch.bool)
        if f0_space != NUMERICAL_SPACE_PHYSICAL:
            raise ValueError("Step 7 conditioning requires physical-field f0")
        if tokens.ndim != 3 or tokens.shape[-1] != RAW_TOKEN_DIM or valid.shape != tokens.shape[:2]:
            raise ValueError("Step 7 raw-HLT token/mask shapes do not align")
        if self.config.architecture_id == ARCH_A5S_HLG_SCRATCH:
            if h0 is not None:
                raise ValueError("A5S forbids h0; pass only raw HLT and f0 reliability-five")
            reliability = f0_live.to(device=tokens.device, dtype=torch.float32).detach()
            if reliability.shape != (*valid.shape, 5):
                raise ValueError("A5S accepts only the five f0 reliability pass-through channels")
            anchor = reliability.new_zeros((*valid.shape, 50))
            anchor[..., 45:] = reliability
            frozen_h0 = reliability.new_zeros((*valid.shape, 160))
        else:
            if h0 is None:
                raise ValueError("Step 7 conditioned architectures require h0")
            anchor = f0_live.to(device=tokens.device, dtype=torch.float32).detach()
            frozen_h0 = h0.to(device=tokens.device, dtype=torch.float32).detach()
            if anchor.shape != (*valid.shape, 50) or frozen_h0.shape != (*valid.shape, 160):
                raise ValueError("Step 7 f0/h0 shapes do not align")
        if not (torch.isfinite(tokens).all() and torch.isfinite(anchor).all() and torch.isfinite(frozen_h0).all()):
            raise ValueError("Step 7 inputs contain non-finite values")
        padding = ~valid
        if bool(padding.any()) and any(
            bool(torch.count_nonzero(value[padding])) for value in (tokens, anchor, frozen_h0)
        ):
            raise ValueError("Step 7 padded raw/f0/h0 inputs must be exactly zero")
        if bool(valid.any()) and (bool((tokens[..., 0][valid] <= 0).any()) or bool((tokens[..., 3][valid] <= 0).any())):
            raise ValueError("valid HLT particles require positive pT and energy")
        return tokens, valid, anchor, frozen_h0

    @staticmethod
    def _broadcast_gates(gates: torch.Tensor) -> torch.Tensor:
        output = gates.new_zeros((*gates.shape[:2], PHYSICAL_CHANNELS))
        for radius_index in range(3):
            base = radius_index * 15
            for semantic_index, offsets in enumerate(SEMANTIC_OFFSETS.values()):
                indices = torch.as_tensor(
                    [base + int(offset) for offset in offsets], device=gates.device, dtype=torch.long
                )
                values = gates[..., radius_index, semantic_index].unsqueeze(-1).expand(*gates.shape[:2], len(offsets))
                output = output.index_copy(-1, indices, values)
        return output

    def forward(
        self,
        hlt_tokens: torch.Tensor,
        mask: torch.Tensor,
        f0_live: torch.Tensor,
        h0: torch.Tensor | None,
        *,
        f0_space: str = NUMERICAL_SPACE_PHYSICAL,
        need_attention_weights: bool = False,
    ) -> HLGCorrectionOutput:
        tokens, valid, anchor, frozen_h0 = self._validate_inputs(
            hlt_tokens, mask, f0_live, h0, f0_space
        )
        standardized_f0 = None
        if self.config.use_f0_conditioning:
            standardized_f0 = self.scalers.conditioning_standardize(
                anchor, valid, input_space=NUMERICAL_SPACE_PHYSICAL
            )
        h0_input = frozen_h0 if self.config.use_h0_conditioning else None
        encoding = self.encoder(
            tokens,
            valid,
            standardized_f0,
            h0_input,
            need_attention_weights=need_attention_weights,
        )
        fused_states: list[torch.Tensor] = []
        if self.radius_fusion is not None:
            fused_radius = self.radius_fusion(torch.cat(encoding.radius_streams, dim=-1))
            raw, fused = self.radius_heads[0](encoding.base_hidden, fused_radius, encoding.readback)
            raw_outputs = [raw]
            fused_states = [fused, fused, fused]
        else:
            raw_outputs = []
            for index, head in enumerate(self.radius_heads):
                raw, fused = head(
                    encoding.base_hidden, encoding.radius_streams[index], encoding.readback
                )
                raw_outputs.append(raw)
                fused_states.append(fused)
        standardized_raw = torch.cat(raw_outputs, dim=-1).masked_fill(~valid.unsqueeze(-1), 0.0)
        if self.config.output_mode == OUTPUT_CORRECTION:
            pre_gate, saturation = self.scalers.physical_correction(
                standardized_raw, valid, trust_bound_enabled=True
            )
            physical = pre_gate
            absolute_physical = None
        else:
            assert self.absolute_scaler is not None
            absolute_physical = self.absolute_scaler.physical(standardized_raw, valid)
            pre_gate = absolute_physical - anchor[..., :PHYSICAL_CHANNELS]
            physical = pre_gate
            saturation = None
        gate_values = None
        gate_loss = standardized_raw.new_zeros(())
        if self.gate_heads is not None:
            gate_values = torch.stack(
                [torch.sigmoid(head(fused_states[index])) for index, head in enumerate(self.gate_heads)],
                dim=2,
            )
            gate_values = gate_values.masked_fill(~valid[:, :, None, None], 0.0)
            valid_gate = valid[:, :, None, None].expand_as(gate_values)
            gate_loss = ((1.0 - gate_values[valid_gate]) ** 2).mean() if bool(valid_gate.any()) else gate_loss
            if self.config.output_mode != OUTPUT_CORRECTION:
                raise AssertionError("the gated ablation is correction-parameterized")
            physical = pre_gate * self._broadcast_gates(gate_values)
        if self.config.output_mode == OUTPUT_CORRECTION:
            physical_field = anchor[..., :PHYSICAL_CHANNELS] + physical
        else:
            assert absolute_physical is not None
            # Emit the bounded absolute tensor directly.  Reconstructing it as
            # anchor + (absolute-anchor) would introduce roundoff dependence on
            # f0[P], violating A5S's strict raw-HLT-only trainable path.
            physical_field = absolute_physical
        f_hat = torch.cat((physical_field, anchor[..., 45:]), dim=-1).masked_fill(
            ~valid.unsqueeze(-1), 0.0
        )
        reasoning = ParticleReasoningState(
            base_hidden=encoding.base_hidden,
            radius_streams=encoding.radius_streams,
            readback=encoding.readback,
            particle_mask=valid,
            region_tokens=encoding.region_tokens,
            region_mask=encoding.region_mask,
            diagnostics=encoding.diagnostics,
        )
        valid_physical = int(valid.sum().detach().cpu()) * PHYSICAL_CHANNELS
        diagnostics = {
            **encoding.diagnostics,
            "scaler_sha256": self.scaler_sha256,
            "absolute_scaler_sha256": None if self.absolute_scaler is None else self.absolute_scaler.artifact_sha256,
            "consumer_output_space": NUMERICAL_SPACE_PHYSICAL,
            "conditioning_space": NUMERICAL_SPACE_CONDITIONING,
            "correction_space": NUMERICAL_SPACE_CORRECTION,
            "loss_space": NUMERICAL_SPACE_LOSS,
            "output_parameterization": self.config.output_mode,
            "trust_bound_enabled": self.config.output_mode == OUTPUT_CORRECTION,
            "saturation_fraction": None if saturation is None else float(saturation.sum().detach().cpu()) / max(valid_physical, 1),
            "reliability_channels_exact_pass_through": bool(torch.equal(f_hat[..., 45:], anchor[..., 45:])),
            "learned_gate_present": self.gate_heads is not None,
            "gate_phase": self._gate_phase,
            "gate_loss_coefficient": GATE_LOSS_COEFFICIENT if self.gate_heads is not None else 0.0,
            "gate_mean": None if gate_values is None else float(gate_values[valid[:, :, None, None].expand_as(gate_values)].mean().detach().cpu()),
            "pre_gate_correction_norm": float(pre_gate[valid].norm().detach().cpu()) if bool(valid.any()) else 0.0,
            "post_gate_correction_norm": float(physical[valid].norm().detach().cpu()) if bool(valid.any()) else 0.0,
            "f0_conditioning_present": bool(self.config.use_f0_conditioning),
            "h0_conditioning_present": bool(self.config.use_h0_conditioning),
            "raw_conditioning_present": bool(self.config.use_raw_conditioning),
            "a5s_f0_physical_or_h0_entered_trainable_path": False if self.config.architecture_id == ARCH_A5S_HLG_SCRATCH else None,
            "directed_edge_count": encoding.graph_edge_count,
        }
        return HLGCorrectionOutput(
            f_hat=f_hat,
            physical_correction=physical,
            standardized_raw_correction=standardized_raw,
            hidden=encoding.base_hidden,
            mask=valid,
            saturation_mask=saturation,
            diagnostics=diagnostics,
            reasoning_state=reasoning,
            gate_values=gate_values,
            gate_loss=gate_loss,
            pre_gate_physical_correction=pre_gate,
        )


def build_step7_hlg_correction_model(
    architecture_id: str,
    *,
    scaler_artifact: Mapping[str, Any],
    absolute_scaler_artifact: Mapping[str, Any] | None = None,
    dropout: float = 0.05,
) -> PredictionAnchoredHLGCorrection:
    config = _config_for_architecture(architecture_id, dropout=dropout)
    return PredictionAnchoredHLGCorrection(
        scaler_artifact,
        config,
        absolute_scaler_artifact=absolute_scaler_artifact,
    )


def step7_gate_regularization(output: HLGCorrectionOutput, *, phase: str) -> tuple[torch.Tensor, float]:
    if phase not in {"field_warmup", "distillation"}:
        raise ValueError("gate phase must be field_warmup or distillation")
    coefficient = GATE_LOSS_COEFFICIENT if output.gate_values is not None and phase == "distillation" else 0.0
    return output.gate_loss, coefficient


@dataclass(frozen=True)
class RepresentativeArchitectureResourceReference:
    particle_width: int
    valid_particles: int
    r0_parameters: int
    r0_forward_flops: int
    a3_parameters: int
    a3_forward_flops: int
    t10_parameters: int
    t10_forward_flops: int
    r0_config_sha256: str
    a3_config_sha256: str
    t10_config_sha256: str
    source_manifest_sha256: str

    def __post_init__(self) -> None:
        if not 0 < int(self.valid_particles) <= int(self.particle_width):
            raise ValueError("deployed resource reference has invalid particle widths")
        for name in (
            "r0_parameters", "r0_forward_flops", "a3_parameters", "a3_forward_flops",
            "t10_parameters", "t10_forward_flops",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "r0_config_sha256", "a3_config_sha256", "t10_config_sha256",
            "source_manifest_sha256",
        ):
            if not _valid_sha256(getattr(self, name)):
                raise ValueError(f"{name} must be a SHA-256")

    @property
    def total_parameters(self) -> int:
        return int(self.r0_parameters + self.a3_parameters + self.t10_parameters)

    @property
    def total_forward_flops(self) -> int:
        return int(self.r0_forward_flops + self.a3_forward_flops + self.t10_forward_flops)

    def to_artifact(self) -> dict[str, Any]:
        return with_content_hash(
            {
                "contract": PREDICTION_ANCHORED_REPRESENTATIVE_RESOURCE_CONTRACT,
                **asdict(self),
                "total_parameters": self.total_parameters,
                "total_forward_flops": self.total_forward_flops,
                "measurement_batch_size": 1,
                "same_valid_mask_required": True,
                "reference_kind": "representative_architecture",
                "reference_locked_to_canonical_a3_not_eventual_selection": True,
                "checkpoint_hashes_present": False,
            }
        )


@dataclass(frozen=True)
class DeployedBundleResourceReference:
    particle_width: int
    valid_particles: int
    r0_parameters: int
    r0_forward_flops: int
    a3_parameters: int
    a3_forward_flops: int
    t10_parameters: int
    t10_forward_flops: int
    r0_checkpoint_sha256: str
    a3_config_sha256: str
    t10_checkpoint_sha256: str
    physical45_scaler_sha256: str
    r0_registration_sha256: str
    execution_spec_sha256: str
    child_manifest_sha256: str
    selected_consumer_sha256: str
    physical45_recipe_sha256: str
    source_manifest_sha256: str
    representative_reference_sha256: str | None = None

    def __post_init__(self) -> None:
        if not 0 < int(self.valid_particles) <= int(self.particle_width):
            raise ValueError("deployed resource reference has invalid particle widths")
        for name in (
            "r0_parameters", "r0_forward_flops", "a3_parameters", "a3_forward_flops",
            "t10_parameters", "t10_forward_flops",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "r0_checkpoint_sha256", "a3_config_sha256", "t10_checkpoint_sha256",
            "physical45_scaler_sha256",
            "r0_registration_sha256", "execution_spec_sha256",
            "child_manifest_sha256", "selected_consumer_sha256",
            "physical45_recipe_sha256",
            "source_manifest_sha256",
        ):
            if not _valid_sha256(getattr(self, name)):
                raise ValueError(f"{name} must be a SHA-256")
        if (
            self.representative_reference_sha256 is not None
            and not _valid_sha256(self.representative_reference_sha256)
        ):
            raise ValueError("representative_reference_sha256 must be a SHA-256")

    @property
    def total_parameters(self) -> int:
        return int(self.r0_parameters + self.a3_parameters + self.t10_parameters)

    @property
    def total_forward_flops(self) -> int:
        return int(self.r0_forward_flops + self.a3_forward_flops + self.t10_forward_flops)

    def to_artifact(self) -> dict[str, Any]:
        return with_content_hash(
            {
                "contract": PREDICTION_ANCHORED_DEPLOYED_RESOURCE_CONTRACT,
                **asdict(self),
                "total_parameters": self.total_parameters,
                "total_forward_flops": self.total_forward_flops,
                "measurement_batch_size": 1,
                "same_valid_mask_required": True,
                "reference_kind": "confirmed_runtime_checkpoints",
                "reference_locked_to_canonical_a3_not_eventual_selection": True,
                "checkpoint_hashes_present": True,
                "resource_values_identical_to_representative": bool(
                    self.representative_reference_sha256
                ),
            }
        )


BundleResourceReference = (
    RepresentativeArchitectureResourceReference | DeployedBundleResourceReference
)


def resource_reference_from_artifact(
    value: Mapping[str, Any], *, require_runtime: bool = False
) -> BundleResourceReference:
    contract = str(value.get("contract", ""))
    if contract == PREDICTION_ANCHORED_REPRESENTATIVE_RESOURCE_CONTRACT:
        if require_runtime:
            raise ValueError("runtime checkpoint reference is required")
        validate_content_hash(
            value, expected_contract=PREDICTION_ANCHORED_REPRESENTATIVE_RESOURCE_CONTRACT
        )
        names = (
            "particle_width", "valid_particles", "r0_parameters", "r0_forward_flops",
            "a3_parameters", "a3_forward_flops", "t10_parameters", "t10_forward_flops",
            "r0_config_sha256", "a3_config_sha256", "t10_config_sha256",
            "source_manifest_sha256",
        )
        return RepresentativeArchitectureResourceReference(
            **{name: value[name] for name in names}
        )
    validate_content_hash(
        value, expected_contract=PREDICTION_ANCHORED_DEPLOYED_RESOURCE_CONTRACT
    )
    names = (
        "particle_width", "valid_particles", "r0_parameters", "r0_forward_flops",
        "a3_parameters", "a3_forward_flops", "t10_parameters", "t10_forward_flops",
        "r0_checkpoint_sha256", "a3_config_sha256", "t10_checkpoint_sha256",
        "physical45_scaler_sha256",
        "r0_registration_sha256", "execution_spec_sha256",
        "child_manifest_sha256", "selected_consumer_sha256",
        "physical45_recipe_sha256",
        "source_manifest_sha256",
    )
    kwargs = {name: value[name] for name in names}
    kwargs["representative_reference_sha256"] = value.get(
        "representative_reference_sha256"
    )
    return DeployedBundleResourceReference(**kwargs)


@dataclass(frozen=True)
class DirectHLGConfig:
    run_id: str
    particle_width: int
    capacity_particle_hidden: int = 0
    capacity_jet_hidden: int = 0
    capacity_bank_size: int = 0
    reference_sha256: str | None = None
    num_classes: int = 10
    dropout: float = 0.05

    def __post_init__(self) -> None:
        if self.run_id not in STEP7_DIRECT_CONTROL_IDS:
            raise ValueError(f"unknown Step 7 direct control {self.run_id!r}")
        if int(self.particle_width) <= 0 or int(self.num_classes) != 10:
            raise ValueError("direct HLG requires positive particle width and ten classes")
        for name in ("capacity_particle_hidden", "capacity_jet_hidden", "capacity_bank_size"):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.reference_sha256 is not None and not _valid_sha256(self.reference_sha256):
            raise ValueError("direct HLG reference_sha256 must be a SHA-256")
        if not 0 <= float(self.dropout) < 1:
            raise ValueError("direct HLG dropout must be in [0,1)")

    @property
    def uses_r0_representation(self) -> bool:
        return self.run_id == DIRECT_R0REP

    def to_artifact(self) -> dict[str, Any]:
        return with_content_hash(
            {
                "contract": PREDICTION_ANCHORED_DIRECT_CONFIG_CONTRACT,
                **asdict(self),
                "training_partition": "stack_train_consumer_union_stack_train_distill",
                "unique_training_jets": 500_000,
                "objective": "cross_entropy_only",
                "field_output_present": False,
                "kd_present": False,
                "t10_present": False,
                "r0_present": self.uses_r0_representation,
                "raw_hlt_only": not self.uses_r0_representation,
                "capacity_locked_before_eventual_reconstructor_selection": True,
            }
        )


class _CapacityAdapter(torch.nn.Module):
    def __init__(self, hidden: int, *, dropout: float) -> None:
        super().__init__()
        self.hidden = int(hidden)
        if self.hidden:
            self.norm: torch.nn.Module | None = torch.nn.LayerNorm(160, eps=1.0e-5)
            self.network: torch.nn.Module | None = torch.nn.Sequential(
                torch.nn.Linear(160, self.hidden),
                torch.nn.GELU(),
                torch.nn.Dropout(float(dropout)),
                torch.nn.Linear(self.hidden, 160),
                torch.nn.Dropout(float(dropout)),
            )
        else:
            self.norm = None
            self.network = None

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.network is None or self.norm is None:
            return value
        return value + self.network(self.norm(value))


@dataclass(frozen=True)
class DirectHLGOutput:
    logits: torch.Tensor
    particle_hidden: torch.Tensor
    region_tokens: torch.Tensor
    mask: torch.Tensor
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class DirectHLGTrainConfig:
    run_id: str
    stack_train_consumer_manifest_sha256: str
    stack_train_distill_manifest_sha256: str
    union_manifest_sha256: str
    optimizer_steps: int
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-2
    gradient_clip_norm: float = 1.0
    unique_training_jets: int = 500_000

    def __post_init__(self) -> None:
        if self.run_id not in STEP7_DIRECT_CONTROL_IDS:
            raise ValueError("direct training run ID is invalid")
        for name in (
            "stack_train_consumer_manifest_sha256",
            "stack_train_distill_manifest_sha256",
            "union_manifest_sha256",
        ):
            if not _valid_sha256(getattr(self, name)):
                raise ValueError(f"{name} must be a SHA-256")
        if int(self.unique_training_jets) != 500_000:
            raise ValueError("both direct controls are locked to the full 500k training union")
        if int(self.optimizer_steps) <= 0 or float(self.learning_rate) <= 0:
            raise ValueError("direct training requires positive optimizer steps and learning rate")
        if float(self.weight_decay) < 0 or float(self.gradient_clip_norm) <= 0:
            raise ValueError("direct training weight decay/gradient clip are invalid")

    def to_artifact(self) -> dict[str, Any]:
        return with_content_hash(
            {
                "contract": PREDICTION_ANCHORED_DIRECT_TRAIN_CONTRACT,
                **asdict(self),
                "partition": "stack_train_consumer_union_stack_train_distill",
                "objective": "cross_entropy_only",
                "same_budget_required_for_both_direct_controls": True,
                "validation_partition": "model_val_stop",
            }
        )


class DirectHLGClassifier(torch.nn.Module):
    """Full-HLG CE classifier with no field, KD, bridge, or T10 path."""

    def __init__(self, config: DirectHLGConfig, *, scaler_artifact: Mapping[str, Any] | None = None) -> None:
        super().__init__()
        self.config = config
        if config.uses_r0_representation:
            if scaler_artifact is None:
                raise ValueError("R0-representation direct control requires the physical45 scaler")
            self.scalers: TorchBridgeScalers | None = TorchBridgeScalers(scaler_artifact)
            encoder_config = _config_for_architecture(ARCH_A3_HLG_PRIMARY, dropout=config.dropout)
        else:
            if scaler_artifact is not None:
                raise ValueError("raw-HLT direct control rejects R0/scaler inputs")
            self.scalers = None
            encoder_config = _config_for_architecture(ARCH_A5S_HLG_SCRATCH, dropout=config.dropout)
        self.encoder = HLGParticleEncoder(encoder_config)
        self.particle_capacity = _CapacityAdapter(
            config.capacity_particle_hidden, dropout=config.dropout
        )
        self.jet_capacity = _CapacityAdapter(config.capacity_jet_hidden, dropout=config.dropout)
        if int(config.capacity_bank_size):
            self.capacity_bank = torch.nn.Parameter(torch.zeros(int(config.capacity_bank_size)))
        else:
            self.register_parameter("capacity_bank", None)
        self.classifier = torch.nn.Sequential(
            torch.nn.LayerNorm(331, eps=1.0e-5),
            torch.nn.Linear(331, 160),
            torch.nn.GELU(),
            torch.nn.Dropout(float(config.dropout)),
            torch.nn.Linear(160, 10),
        )

    def config_artifact(self) -> dict[str, Any]:
        config = self.config.to_artifact()
        return with_content_hash(
            {
                "contract": PREDICTION_ANCHORED_DIRECT_MODEL_CONTRACT,
                "config": config,
                "config_sha256": config["content_hash"],
                "encoder_config": self.encoder.config.to_artifact(),
                "scaler_sha256": None if self.scalers is None else self.scalers.artifact_sha256,
                "deployable_hlt_only": True,
                "field_head_present": False,
                "teacher_present": False,
            }
        )

    @staticmethod
    def _validate_raw(tokens: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = tokens.float()
        mask = mask.to(device=tokens.device, dtype=torch.bool)
        if tokens.ndim != 3 or tokens.shape[-1] != RAW_TOKEN_DIM or mask.shape != tokens.shape[:2]:
            raise ValueError("direct HLG raw token/mask shapes do not align")
        if not bool(torch.isfinite(tokens).all()):
            raise ValueError("direct HLG tokens contain non-finite values")
        if bool((~mask).any()) and bool(torch.count_nonzero(tokens[~mask])):
            raise ValueError("direct HLG padded tokens must be zero")
        return tokens, mask

    def forward(
        self,
        hlt_tokens: torch.Tensor,
        mask: torch.Tensor,
        f0: torch.Tensor | None = None,
        h0: torch.Tensor | None = None,
        *,
        f0_space: str = NUMERICAL_SPACE_PHYSICAL,
    ) -> DirectHLGOutput:
        tokens, valid = self._validate_raw(hlt_tokens, mask)
        if self.config.uses_r0_representation:
            if f0 is None or h0 is None or self.scalers is None:
                raise ValueError("R0-representation direct control requires f0 and h0")
            anchor = f0.to(device=tokens.device, dtype=torch.float32).detach()
            frozen_h0 = h0.to(device=tokens.device, dtype=torch.float32).detach()
            if anchor.shape != (*valid.shape, 50) or frozen_h0.shape != (*valid.shape, 160):
                raise ValueError("direct R0 f0/h0 shapes do not align")
            standardized = self.scalers.conditioning_standardize(anchor, valid, input_space=f0_space)
            encoding = self.encoder(tokens, valid, standardized, frozen_h0)
        else:
            if f0 is not None or h0 is not None:
                raise ValueError("raw-HLT direct control forbids f0/h0 inputs")
            encoding = self.encoder(tokens, valid, None, None)
        particle = self.particle_capacity(encoding.readback).masked_fill(~valid.unsqueeze(-1), 0.0)
        weights = valid.to(dtype=particle.dtype)
        pooled_particle = (particle * weights.unsqueeze(-1)).sum(1) / torch.clamp(weights.sum(1, keepdim=True), min=1.0)
        pooled_particle = self.jet_capacity(pooled_particle)
        region_weight = encoding.region_mask.to(dtype=encoding.region_tokens.dtype)
        pooled_region = (encoding.region_tokens * region_weight.unsqueeze(-1)).sum(1) / torch.clamp(
            region_weight.sum(1, keepdim=True), min=1.0
        )
        if self.capacity_bank is not None:
            pooled_particle = pooled_particle.clone()
            pooled_particle[:, 0] = pooled_particle[:, 0] + self.capacity_bank.mean()
        logits = self.classifier(
            torch.cat((pooled_particle, pooled_region, _masked_jet_summary(tokens, valid)), dim=-1)
        )
        return DirectHLGOutput(
            logits,
            particle,
            encoding.region_tokens,
            valid,
            {
                **encoding.diagnostics,
                "contract": PREDICTION_ANCHORED_DIRECT_MODEL_CONTRACT,
                "run_id": self.config.run_id,
                "objective": "cross_entropy_only",
                "field_output_present": False,
                "kd_present": False,
                "t10_present": False,
                "r0_present": self.config.uses_r0_representation,
                "directed_edge_count": encoding.graph_edge_count,
            },
        )


def train_step7_direct_hlg(
    model: DirectHLGClassifier,
    batches: Iterable[Mapping[str, Any]],
    config: DirectHLGTrainConfig,
) -> dict[str, Any]:
    """Run the CE-only direct-control optimizer loop over the bound 500k union.

    The iterable is supplied by the campaign runner so Step 10 can keep source
    arrays allocation-local.  This routine deliberately has no field, KD, or
    teacher arguments and never persists batch tensors.
    """

    if model.config.run_id != config.run_id:
        raise ValueError("direct model/training run IDs disagree")
    if model.config.reference_sha256 is None:
        raise ValueError("direct training requires a capacity-locked deployed reference")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    iterator = iter(batches)
    device = next(model.parameters()).device
    losses = []
    correct = 0
    examples = 0
    gradient_norms = []
    model.train()
    for step in range(int(config.optimizer_steps)):
        try:
            raw_batch = next(iterator)
        except StopIteration as error:
            raise RuntimeError(
                f"direct training iterable ended after {step} of {config.optimizer_steps} steps"
            ) from error
        tokens = torch.as_tensor(raw_batch["hlt_tokens"], dtype=torch.float32, device=device)
        mask = torch.as_tensor(raw_batch["mask"], dtype=torch.bool, device=device)
        labels = torch.as_tensor(raw_batch["labels"], dtype=torch.long, device=device)
        if config.run_id == DIRECT_R0REP:
            if "f0" not in raw_batch or "h0" not in raw_batch:
                raise ValueError("R0-representation direct training batch lacks f0/h0")
            output = model(
                tokens,
                mask,
                torch.as_tensor(raw_batch["f0"], dtype=torch.float32, device=device),
                torch.as_tensor(raw_batch["h0"], dtype=torch.float32, device=device),
            )
        else:
            if "f0" in raw_batch or "h0" in raw_batch:
                raise ValueError("raw-HLT direct training batch must not expose f0/h0")
            output = model(tokens, mask)
        loss = F.cross_entropy(output.logits, labels)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("direct HLG CE loss is non-finite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.gradient_clip_norm))
        if not bool(torch.isfinite(norm)):
            raise FloatingPointError("direct HLG gradient norm is non-finite")
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        correct += int((output.logits.argmax(-1) == labels).sum().detach().cpu())
        examples += int(labels.numel())
        gradient_norms.append(float(norm.detach().cpu()))
    return with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_DIRECT_TRAIN_CONTRACT,
            "run_id": config.run_id,
            "train_config": config.to_artifact(),
            "model_config_sha256": model.config_artifact()["content_hash"],
            "optimizer_steps_completed": len(losses),
            "examples_seen_with_replacement_or_epochs": examples,
            "mean_train_cross_entropy": float(np.mean(losses)),
            "last_train_cross_entropy": losses[-1],
            "train_accuracy_diagnostic": correct / max(examples, 1),
            "max_preclip_gradient_norm": max(gradient_norms),
            "objective": "cross_entropy_only",
            "field_output_present": False,
            "kd_present": False,
            "teacher_present": False,
            "persistent_batch_or_field_tensor_written": False,
        }
    )


@dataclass(frozen=True)
class HLGResourceProfile:
    architecture_id: str
    trainable_parameters: int
    total_parameters: int
    forward_flops: int
    batch_size: int
    particle_width: int
    valid_particles: int
    scope: str
    method: str

    def to_artifact(self) -> dict[str, Any]:
        return with_content_hash(
            {
                "contract": PREDICTION_ANCHORED_STEP7_RESOURCE_CONTRACT,
                **asdict(self),
                "measurement_batch_size": 1,
                "flop_convention": "executed_linear_norm_gelu_mha_hooks_plus_explicit_graph_pool_pair_and_bank_ops",
            }
        )


def _profile_inputs(
    particle_width: int, valid_particles: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = torch.zeros((1, particle_width, RAW_TOKEN_DIM), dtype=torch.float32)
    mask = torch.zeros((1, particle_width), dtype=torch.bool)
    mask[:, :valid_particles] = True
    positions = torch.arange(valid_particles, dtype=torch.float32)
    tokens[0, :valid_particles, 0] = 1.0 + positions / max(valid_particles, 1)
    tokens[0, :valid_particles, 1] = 0.002 * positions
    tokens[0, :valid_particles, 2] = 0.003 * positions
    tokens[0, :valid_particles, 3] = 2.0 + positions / max(valid_particles, 1)
    return tokens, mask, torch.zeros((1, particle_width, 50)), torch.zeros((1, particle_width, 160))


def _profile_executed_forward(
    model: torch.nn.Module,
    forward_call: Callable[[], Any],
) -> tuple[Any, int]:
    """Profile one real forward with the hook convention shared by all bundle parts."""

    flops = 0

    def linear_hook(module: torch.nn.Linear, args: tuple[Any, ...], output: Any) -> None:
        nonlocal flops
        tensor = output if isinstance(output, torch.Tensor) else output[0]
        flops += int(tensor.numel()) * (
            2 * int(module.in_features) + (1 if module.bias is not None else 0)
        )

    def norm_hook(module: torch.nn.LayerNorm, args: tuple[Any, ...], output: Any) -> None:
        nonlocal flops
        flops += 5 * int(output.numel())

    def gelu_hook(module: torch.nn.GELU, args: tuple[Any, ...], output: Any) -> None:
        nonlocal flops
        flops += 8 * int(output.numel())

    def mha_hook(module: torch.nn.MultiheadAttention, args: tuple[Any, ...], output: Any) -> None:
        nonlocal flops
        query, key, _value = args[:3]
        batch, queries, dim = query.shape
        keys = key.shape[1]
        flops += batch * queries * (2 * dim * dim + dim)
        flops += batch * keys * 2 * (2 * dim * dim + dim)
        flops += batch * queries * (2 * dim * dim + dim)
        flops += 4 * batch * queries * keys * dim

    def conv_hook(
        module: torch.nn.Conv1d | torch.nn.Conv2d,
        args: tuple[Any, ...],
        output: Any,
    ) -> None:
        nonlocal flops
        kernel_ops = int(np.prod(module.kernel_size)) * int(module.in_channels) // int(
            module.groups
        )
        flops += int(output.numel()) * (
            2 * kernel_ops + (1 if module.bias is not None else 0)
        )

    def batch_norm_hook(
        module: torch.nn.BatchNorm1d | torch.nn.BatchNorm2d,
        args: tuple[Any, ...],
        output: Any,
    ) -> None:
        nonlocal flops
        flops += 5 * int(output.numel())

    hooks = []
    for module in model.modules():
        if isinstance(module, torch.nn.MultiheadAttention):
            hooks.append(module.register_forward_hook(mha_hook))
        elif isinstance(module, torch.nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))
        elif isinstance(module, torch.nn.LayerNorm):
            hooks.append(module.register_forward_hook(norm_hook))
        elif isinstance(module, torch.nn.GELU):
            hooks.append(module.register_forward_hook(gelu_hook))
        elif isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d)):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            hooks.append(module.register_forward_hook(batch_norm_hook))
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            output = forward_call()
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)
    return output, int(flops)


def measure_module_forward_resources(
    model: torch.nn.Module,
    *,
    forward_call: Callable[[], Any],
    architecture_id: str,
    scope: str,
    particle_width: int,
    valid_particles: int,
    explicit_unhooked_flops: int = 0,
) -> HLGResourceProfile:
    """Measure a non-HLG bundle component with the same executed-forward profiler."""

    if not 0 < int(valid_particles) <= int(particle_width):
        raise ValueError("resource measurement requires valid_particles within particle_width")
    _output, hooked_flops = _profile_executed_forward(model, forward_call)
    trainable = sum(int(value.numel()) for value in model.parameters() if value.requires_grad)
    total = sum(int(value.numel()) for value in model.parameters())
    return HLGResourceProfile(
        str(architecture_id),
        trainable,
        total,
        int(hooked_flops + int(explicit_unhooked_flops)),
        1,
        int(particle_width),
        int(valid_particles),
        str(scope),
        "executed_bundle_forward_resource_hooks_v2",
    )


def measure_step7_resources(
    model: PredictionAnchoredHLGCorrection | DirectHLGClassifier,
    *,
    particle_width: int,
    valid_particles: int | None = None,
) -> HLGResourceProfile:
    valid_count = int(particle_width if valid_particles is None else valid_particles)
    if not 0 < valid_count <= int(particle_width):
        raise ValueError("resource measurement requires valid_particles within particle_width")
    device = next(model.parameters()).device
    tokens, mask, f0, h0 = tuple(value.to(device) for value in _profile_inputs(int(particle_width), valid_count))
    def run_forward() -> Any:
        if isinstance(model, DirectHLGClassifier):
            return (
                model(tokens, mask, f0, h0)
                if model.config.uses_r0_representation
                else model(tokens, mask)
            )
        return (
            model(tokens, mask, f0[..., 45:], None)
            if model.config.architecture_id == ARCH_A5S_HLG_SCRATCH
            else model(tokens, mask, f0, h0)
        )

    output, flops = _profile_executed_forward(model, run_forward)
    if isinstance(model, DirectHLGClassifier):
        architecture_id = model.config.run_id
        regions = int(output.region_tokens.shape[1])
        scope = "direct_hlg_classifier"
    else:
        architecture_id = model.config.architecture_id
        regions = int(output.reasoning_state.region_tokens.shape[1])
        scope = "hlg_correction"
    edge_count = int(output.diagnostics["directed_edge_count"])
    # Operations outside hooked modules: graph kernels/weighted aggregation,
    # assignment similarity/geometry, weighted pooling, pair construction,
    # masks/residuals, and the explicitly-used capacity bank reduction.
    flops += edge_count * (46 + 3 * 160 * 2 * 3)
    original_regions = int(output.diagnostics.get("original_region_count", regions))
    flops += valid_count * original_regions * (2 * 64 + 24)
    flops += valid_count * original_regions * (
        int(getattr(model, "config").region_pool_input_dim) if isinstance(model, PredictionAnchoredHLGCorrection) else 389
    ) * 2
    flops += original_regions * original_regions * 96
    if isinstance(model, PredictionAnchoredHLGCorrection):
        flops += valid_count * PHYSICAL_CHANNELS * 8
    elif model.capacity_bank is not None:
        flops += int(model.capacity_bank.numel()) + 2
    trainable = sum(int(value.numel()) for value in model.parameters() if value.requires_grad)
    total = sum(int(value.numel()) for value in model.parameters())
    return HLGResourceProfile(
        architecture_id,
        trainable,
        total,
        int(flops),
        1,
        int(particle_width),
        valid_count,
        scope,
        "executed_bundle_forward_resource_hooks_v2",
    )


def _adapter_cost(hidden: int, applications: int) -> tuple[int, int]:
    if int(hidden) <= 0:
        return 0, 0
    # LN(160), 160->h, h->160 and the matching hook convention.
    parameters = 321 * int(hidden) + 480
    flops = int(applications) * (649 * int(hidden) + 960)
    return parameters, flops


def fit_direct_hlg_config(
    run_id: str,
    *,
    scaler_artifact: Mapping[str, Any] | None,
    reference: BundleResourceReference,
    dropout: float = 0.05,
) -> tuple[DirectHLGConfig, HLGResourceProfile, dict[str, Any]]:
    """Size a direct HLG control to the immutable canonical bundle reference."""

    base_config = DirectHLGConfig(run_id, reference.particle_width, dropout=dropout)
    base_model = DirectHLGClassifier(base_config, scaler_artifact=scaler_artifact)
    base = measure_step7_resources(
        base_model,
        particle_width=reference.particle_width,
        valid_particles=reference.valid_particles,
    )
    extra_parameters = reference.total_parameters - base.total_parameters
    if extra_parameters < 0:
        raise ValueError(
            f"{run_id} base HLG has {base.total_parameters} parameters, exceeding the "
            f"canonical deployed reference {reference.total_parameters}"
        )
    target_extra_flops = reference.total_forward_flops - base.forward_flops
    max_particle_hidden = max(0, (extra_parameters - 480) // 321)
    best: tuple[float, int, int, int, int] | None = None
    for particle_hidden in range(max_particle_hidden + 1):
        particle_parameters, particle_flops = _adapter_cost(
            particle_hidden, reference.valid_particles
        )
        remaining = extra_parameters - particle_parameters
        if remaining < 0:
            continue
        jet_candidates = {0}
        if remaining >= 801:
            desired_after_particle = target_extra_flops - particle_flops
            estimate = int(round((desired_after_particle - remaining - 480) / 328.0))
            maximum = max(0, (remaining - 480) // 321)
            for candidate in range(estimate - 2, estimate + 3):
                jet_candidates.add(min(max(candidate, 1), maximum))
            jet_candidates.add(maximum)
        for jet_hidden in jet_candidates:
            jet_parameters, jet_flops = _adapter_cost(jet_hidden, 1)
            bank = remaining - jet_parameters
            if bank < 0:
                continue
            predicted_extra_flops = particle_flops + jet_flops + (bank + 2 if bank else 0)
            error = abs(predicted_extra_flops - target_extra_flops)
            candidate_key = (float(error), particle_hidden, jet_hidden, bank, predicted_extra_flops)
            if best is None or candidate_key < best:
                best = candidate_key
    if best is None:
        raise RuntimeError("could not construct a direct HLG capacity match")
    _, particle_hidden, jet_hidden, bank, _ = best
    reference_artifact = reference.to_artifact()
    config = DirectHLGConfig(
        run_id=run_id,
        particle_width=reference.particle_width,
        capacity_particle_hidden=particle_hidden,
        capacity_jet_hidden=jet_hidden,
        capacity_bank_size=bank,
        reference_sha256=reference_artifact["content_hash"],
        dropout=dropout,
    )
    model = DirectHLGClassifier(config, scaler_artifact=scaler_artifact)
    profile = measure_step7_resources(
        model,
        particle_width=reference.particle_width,
        valid_particles=reference.valid_particles,
    )
    match = direct_capacity_match(profile, reference)
    if not match["passed"]:
        raise AssertionError(
            f"{run_id} could not satisfy direct capacity tolerance: "
            f"parameter_error={match['parameter_relative_error']:.4f}, "
            f"flop_error={match['flop_relative_error']:.4f}"
        )
    return config, profile, match


def build_capacity_matched_direct_hlg(
    run_id: str,
    *,
    scaler_artifact: Mapping[str, Any] | None,
    reference: BundleResourceReference,
    dropout: float = 0.05,
) -> tuple[DirectHLGClassifier, HLGResourceProfile, dict[str, Any]]:
    expected_scaler = scaler_artifact if run_id == DIRECT_R0REP else None
    if run_id == DIRECT_HLT and scaler_artifact is not None:
        raise ValueError("raw-HLT direct capacity builder forbids a scaler/R0 dependency")
    config, _profile, _match = fit_direct_hlg_config(
        run_id,
        scaler_artifact=expected_scaler,
        reference=reference,
        dropout=dropout,
    )
    model = DirectHLGClassifier(config, scaler_artifact=expected_scaler)
    profile = measure_step7_resources(
        model,
        particle_width=reference.particle_width,
        valid_particles=reference.valid_particles,
    )
    match = direct_capacity_match(profile, reference)
    return model, profile, match


def direct_capacity_match(
    profile: HLGResourceProfile, reference: BundleResourceReference
) -> dict[str, Any]:
    parameter_error = abs(profile.total_parameters / reference.total_parameters - 1.0)
    flop_error = abs(profile.forward_flops / reference.total_forward_flops - 1.0)
    return with_content_hash(
        {
            "contract": "prediction_anchored_direct_capacity_match_v1",
            "run_id": profile.architecture_id,
            "profile": profile.to_artifact(),
            "reference": reference.to_artifact(),
            "parameter_relative_error": parameter_error,
            "flop_relative_error": flop_error,
            "parameter_tolerance": 0.05,
            "flop_tolerance": 0.10,
            "parameter_tolerance_passed": parameter_error <= 0.05,
            "flop_tolerance_passed": flop_error <= 0.10,
            "passed": parameter_error <= 0.05 and flop_error <= 0.10,
            "matched_to_canonical_a3_bundle_not_eventual_selected_bundle": True,
        }
    )


def _torch_bytes(value: Any) -> bytes:
    buffer = BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def measure_step7_registry_states(
    registry: Mapping[str, Any],
    *,
    scaler_artifact: Mapping[str, Any],
    absolute_scaler_artifact: Mapping[str, Any],
    source_manifest_sha256: str,
    deployed_reference: BundleResourceReference,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Measure every Step 1-7 architecture row and leave only Step 8 rows open."""

    from .bridge_campaign import MEASUREMENT_UNMEASURED, record_registry_measurements
    from .hierarchical_reconstructor import STEP7_DEFERRED_ARCHITECTURE_IDS

    if not _valid_sha256(source_manifest_sha256):
        raise ValueError("Step 7 measurement requires the source-manifest SHA-256")
    if source_manifest_sha256 != deployed_reference.source_manifest_sha256:
        raise ValueError("deployed resource reference belongs to a different source manifest")
    expected_scaler_sha256 = getattr(
        deployed_reference, "physical45_scaler_sha256", None
    )
    if (
        expected_scaler_sha256 is not None
        and scaler_artifact.get("content_hash") != expected_scaler_sha256
    ):
        raise ValueError(
            "physical45 scaler differs from the confirmed runtime resource reference"
        )
    step6_registry, step6_artifact = measure_step6_registry_states(
        registry,
        scaler_artifact=scaler_artifact,
        particle_width=deployed_reference.particle_width,
        source_manifest_sha256=source_manifest_sha256,
    )
    by_id = {row["canonical_run_id"]: row for row in step6_registry["runs"]}
    changed = [
        run_id for run_id in STEP7_MEASURED_ARCHITECTURE_IDS
        if by_id[run_id]["measurement_status"] != MEASUREMENT_UNMEASURED
    ]
    if changed:
        raise ValueError(f"Step 7 input registry already measured Step 7 rows: {changed}")
    state_bytes: dict[str, int] = {}
    profiles: dict[str, Any] = {}
    config_hashes: dict[str, str] = {}
    for architecture_id in STEP7_HIERARCHY_ARCHITECTURE_IDS:
        model = build_step7_hlg_correction_model(
            architecture_id,
            scaler_artifact=scaler_artifact,
            absolute_scaler_artifact=(
                absolute_scaler_artifact
                if architecture_id in {ARCH_A5_HLG_ABSOLUTE, ARCH_A5S_HLG_SCRATCH}
                else None
            ),
            dropout=0.05,
        )
        config = model.config_artifact()
        encoded = _torch_bytes(
            {
                "checkpoint_contract": "prediction_anchored_step7_hlg_weights_v1",
                "architecture_id": architecture_id,
                "model_config": config,
                "model_state_dict": model.state_dict(),
                "scaler_sha256": model.scaler_sha256,
                "weights_only": True,
                "optimizer_state_persisted": False,
                "generated_fields_persisted": False,
            }
        )
        state_bytes[architecture_id] = len(encoded)
        config_hashes[architecture_id] = config["content_hash"]
        profiles[architecture_id] = measure_step7_resources(
            model,
            particle_width=deployed_reference.particle_width,
            valid_particles=deployed_reference.valid_particles,
        ).to_artifact()
    canonical_config_hash = HLGCorrectionConfig.for_architecture(
        ARCH_A3_HLG_PRIMARY, dropout=0.05
    ).to_artifact()["content_hash"]
    canonical_profile = profiles[ARCH_A3_HLG_PRIMARY]
    if deployed_reference.a3_config_sha256 != canonical_config_hash:
        raise ValueError(
            "deployed resource reference is bound to a different canonical A3 config"
        )
    if (
        deployed_reference.a3_parameters != int(canonical_profile["total_parameters"])
        or deployed_reference.a3_forward_flops != int(canonical_profile["forward_flops"])
    ):
        raise ValueError(
            "deployed resource reference A3 parameters/FLOPs disagree with the executed canonical A3 profile"
        )
    direct_matches: dict[str, Any] = {}
    for run_id in STEP7_DIRECT_CONTROL_IDS:
        direct, profile, match = build_capacity_matched_direct_hlg(
            run_id,
            scaler_artifact=scaler_artifact if run_id == DIRECT_R0REP else None,
            reference=deployed_reference,
            dropout=0.05,
        )
        config = direct.config_artifact()
        encoded = _torch_bytes(
            {
                "checkpoint_contract": "prediction_anchored_step7_direct_weights_v1",
                "run_id": run_id,
                "model_config": config,
                "model_state_dict": direct.state_dict(),
                "weights_only": True,
                "optimizer_state_persisted": False,
                "generated_fields_persisted": False,
            }
        )
        state_bytes[run_id] = len(encoded)
        config_hashes[run_id] = config["content_hash"]
        profiles[run_id] = profile.to_artifact()
        direct_matches[run_id] = match
    a0m = measure_correction_resources(
        build_step6_correction_model(ARCH_A0M_CAPACITY_PARTICLE, scaler_artifact=scaler_artifact),
        particle_width=deployed_reference.particle_width,
        valid_particles=deployed_reference.valid_particles,
    )
    a3_payload = profiles[ARCH_A3_HLG_PRIMARY]
    actual_a3 = CorrectionResourceProfile(
        architecture_id=ARCH_A3_HLG_PRIMARY,
        trainable_parameters=int(a3_payload["trainable_parameters"]),
        total_parameters=int(a3_payload["total_parameters"]),
        forward_flops=int(a3_payload["forward_flops"]),
        batch_size=1,
        particle_width=deployed_reference.particle_width,
        valid_particles=deployed_reference.valid_particles,
        method="executed_step7_a3_forward_profile_v1",
    )
    particle_match = particle_capacity_match(a0m, actual_a3)
    if not particle_match["passed"]:
        raise AssertionError("A0M no longer matches the executed canonical A3 profile")
    updated = record_registry_measurements(step6_registry, state_bytes)
    updated_by_id = {row["canonical_run_id"]: row for row in updated["runs"]}
    step8_deferred = [
        run_id for run_id in STEP7_DEFERRED_ARCHITECTURE_IDS
        if run_id not in STEP7_MEASURED_ARCHITECTURE_IDS
    ]
    accidentally_measured = [
        run_id for run_id in step8_deferred
        if updated_by_id[run_id]["measurement_status"] != MEASUREMENT_UNMEASURED
    ]
    if accidentally_measured:
        raise AssertionError(f"Step 7 changed Step 8 rows: {accidentally_measured}")
    artifact = with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_STEP7_MEASUREMENT_CONTRACT,
            "input_registry_sha256": registry["content_hash"],
            "step6_measurement_sha256": step6_artifact["content_hash"],
            "updated_registry_sha256": updated["content_hash"],
            "source_manifest_sha256": source_manifest_sha256,
            "source_manifest_particle_width": deployed_reference.particle_width,
            "deployed_reference": deployed_reference.to_artifact(),
            "measured_state_bytes": state_bytes,
            "model_contract_sha256": config_hashes,
            "resource_profiles": profiles,
            "particle_capacity_match_to_executed_a3": particle_match,
            "direct_capacity_matches": direct_matches,
            "newly_measured_configuration_count": len(state_bytes),
            "step8_deferred_unmeasured_run_ids": step8_deferred,
            "serialization_method": "torch_save_weights_only_to_verified_ram",
            "dense_field_cache_persisted": False,
        }
    )
    return updated, artifact


def tiny_train_reload_step7_hierarchy(
    architecture_id: str,
    *,
    scaler_artifact: Mapping[str, Any],
    absolute_scaler_artifact: Mapping[str, Any] | None,
    batch: Mapping[str, Any],
    learning_rate: float = 1.0e-3,
) -> dict[str, Any]:
    required = {"hlt_tokens", "mask", "f0", "h0", "bridge_fields"}
    missing = sorted(required - set(batch))
    if missing:
        raise ValueError(f"Step 7 tiny hierarchy batch is missing {missing}")
    tensors = {
        key: torch.as_tensor(batch[key], dtype=torch.bool if key == "mask" else torch.float32)
        for key in required
    }
    torch.manual_seed(97_000 + STEP7_HIERARCHY_ARCHITECTURE_IDS.index(architecture_id))
    absolute = absolute_scaler_artifact if architecture_id in {ARCH_A5_HLG_ABSOLUTE, ARCH_A5S_HLG_SCRATCH} else None
    model = build_step7_hlg_correction_model(
        architecture_id,
        scaler_artifact=scaler_artifact,
        absolute_scaler_artifact=absolute,
        dropout=0.0,
    )
    model.set_training_phase("field_warmup")
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=float(learning_rate)
    )
    model.train()
    model_args = (
        tensors["hlt_tokens"], tensors["mask"], tensors["f0"][..., 45:], None
    ) if architecture_id == ARCH_A5S_HLG_SCRATCH else (
        tensors["hlt_tokens"], tensors["mask"], tensors["f0"], tensors["h0"]
    )
    output = model(*model_args)
    loss, objective = compute_c0_objective(
        output,
        tensors,
        resolve_c0_loss_recipe("D10_L8_full_c0"),
        model.scalers,
        phase="field_warmup",
    )
    if not torch.isfinite(loss):
        raise FloatingPointError("Step 7 tiny hierarchy loss is non-finite")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradients = [
        parameter.grad for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not gradients or any(not bool(torch.isfinite(value).all()) for value in gradients):
        raise FloatingPointError("Step 7 tiny hierarchy gradients are invalid")
    optimizer.step()
    model.eval()
    with torch.no_grad():
        expected = model(*model_args).f_hat
    encoded = _torch_bytes(
        {
            "architecture_id": architecture_id,
            "state_dict": model.state_dict(),
            "config": model.config_artifact(),
        }
    )
    payload = torch.load(BytesIO(encoded), map_location="cpu", weights_only=False)
    reloaded = build_step7_hlg_correction_model(
        architecture_id,
        scaler_artifact=scaler_artifact,
        absolute_scaler_artifact=absolute,
        dropout=0.0,
    )
    reloaded.set_training_phase("field_warmup")
    reloaded.load_state_dict(payload["state_dict"], strict=True)
    reloaded.eval()
    reload_args = (
        tensors["hlt_tokens"], tensors["mask"], tensors["f0"][..., 45:], None
    ) if architecture_id == ARCH_A5S_HLG_SCRATCH else (
        tensors["hlt_tokens"], tensors["mask"], tensors["f0"], tensors["h0"]
    )
    with torch.no_grad():
        actual = reloaded(*reload_args).f_hat
    if not torch.equal(expected, actual):
        raise AssertionError("Step 7 hierarchy strict reload changed f_hat")
    return with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_STEP7_RELOAD_CONTRACT,
            "kind": "hierarchy_correction",
            "architecture_id": architecture_id,
            "loss": float(loss.detach().cpu()),
            "loss_coefficients": objective["coefficients"],
            "gradient_tensor_count": len(gradients),
            "all_gradients_finite": True,
            "strict_reload": True,
            "reload_exact_output": True,
            "checkpoint_sha256": hashlib.sha256(encoded).hexdigest(),
            "serialized_state_bytes": len(encoded),
            "optimizer_state_persisted": False,
            "generated_fields_persisted": False,
            "scientific_results_allowed": False,
        }
    )


def tiny_train_reload_step7_direct(
    run_id: str,
    *,
    scaler_artifact: Mapping[str, Any],
    deployed_reference: BundleResourceReference,
    batch: Mapping[str, Any],
    learning_rate: float = 1.0e-3,
) -> dict[str, Any]:
    required = {"hlt_tokens", "mask", "labels"}
    if run_id == DIRECT_R0REP:
        required.update({"f0", "h0"})
    missing = sorted(required - set(batch))
    if missing:
        raise ValueError(f"Step 7 tiny direct batch is missing {missing}")
    tensors = {
        "hlt_tokens": torch.as_tensor(batch["hlt_tokens"], dtype=torch.float32),
        "mask": torch.as_tensor(batch["mask"], dtype=torch.bool),
        "labels": torch.as_tensor(batch["labels"], dtype=torch.long),
    }
    if run_id == DIRECT_R0REP:
        tensors["f0"] = torch.as_tensor(batch["f0"], dtype=torch.float32)
        tensors["h0"] = torch.as_tensor(batch["h0"], dtype=torch.float32)
    torch.manual_seed(98_000 + STEP7_DIRECT_CONTROL_IDS.index(run_id))
    model, profile, match = build_capacity_matched_direct_hlg(
        run_id,
        scaler_artifact=scaler_artifact if run_id == DIRECT_R0REP else None,
        reference=deployed_reference,
        dropout=0.0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate))
    model.train()
    if run_id == DIRECT_R0REP:
        output = model(tensors["hlt_tokens"], tensors["mask"], tensors["f0"], tensors["h0"])
    else:
        output = model(tensors["hlt_tokens"], tensors["mask"])
    loss = F.cross_entropy(output.logits, tensors["labels"])
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradients = [value.grad for value in model.parameters() if value.grad is not None]
    if not gradients or any(not bool(torch.isfinite(value).all()) for value in gradients):
        raise FloatingPointError("Step 7 direct-control gradients are invalid")
    optimizer.step()
    model.eval()
    with torch.no_grad():
        expected = (
            model(tensors["hlt_tokens"], tensors["mask"], tensors["f0"], tensors["h0"]).logits
            if run_id == DIRECT_R0REP else model(tensors["hlt_tokens"], tensors["mask"]).logits
        )
    encoded = _torch_bytes({"config": model.config.to_artifact(), "state_dict": model.state_dict()})
    reloaded = DirectHLGClassifier(
        model.config, scaler_artifact=scaler_artifact if run_id == DIRECT_R0REP else None
    )
    reloaded.load_state_dict(torch.load(BytesIO(encoded), weights_only=False)["state_dict"], strict=True)
    reloaded.eval()
    with torch.no_grad():
        actual = (
            reloaded(tensors["hlt_tokens"], tensors["mask"], tensors["f0"], tensors["h0"]).logits
            if run_id == DIRECT_R0REP else reloaded(tensors["hlt_tokens"], tensors["mask"]).logits
        )
    if not torch.equal(expected, actual):
        raise AssertionError("Step 7 direct strict reload changed logits")
    return with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_STEP7_RELOAD_CONTRACT,
            "kind": "direct_hlg_classifier",
            "architecture_id": run_id,
            "objective": "cross_entropy_only",
            "loss": float(loss.detach().cpu()),
            "gradient_tensor_count": len(gradients),
            "strict_reload": True,
            "reload_exact_output": True,
            "resource_profile": profile.to_artifact(),
            "capacity_match": match,
            "serialized_state_bytes": len(encoded),
            "optimizer_state_persisted": False,
            "generated_fields_persisted": False,
            "scientific_results_allowed": False,
        }
    )


def run_step7_paired_seed_miniature(
    *,
    scaler_artifact: Mapping[str, Any],
    absolute_scaler_artifact: Mapping[str, Any],
    batch: Mapping[str, Any],
    architecture_ids: Sequence[str] = STEP7_HIERARCHY_ARCHITECTURE_IDS,
    seed_ids: Sequence[int] = STEP7_PAIRED_SEEDS,
) -> dict[str, Any]:
    architecture_ids = tuple(str(value) for value in architecture_ids)
    seed_ids = tuple(int(value) for value in seed_ids)
    if len(set(architecture_ids)) != len(architecture_ids) or any(
        value not in STEP7_HIERARCHY_ARCHITECTURE_IDS for value in architecture_ids
    ):
        raise ValueError("paired miniature architecture IDs are invalid or duplicated")
    if seed_ids != STEP7_PAIRED_SEEDS:
        raise ValueError("Step 7 paired miniature seeds are locked to 101/202/303")
    tensors = {
        key: torch.as_tensor(batch[key], dtype=torch.bool if key == "mask" else torch.float32)
        for key in ("hlt_tokens", "mask", "f0", "h0", "bridge_fields")
    }
    rows = []
    aggregates: dict[str, Any] = {}
    for architecture_id in architecture_ids:
        scores = []
        for seed in seed_ids:
            torch.manual_seed(seed)
            model = build_step7_hlg_correction_model(
                architecture_id,
                scaler_artifact=scaler_artifact,
                absolute_scaler_artifact=(
                    absolute_scaler_artifact
                    if architecture_id in {ARCH_A5_HLG_ABSOLUTE, ARCH_A5S_HLG_SCRATCH}
                    else None
                ),
                dropout=0.0,
            ).eval()
            with torch.no_grad():
                output = (
                    model(tensors["hlt_tokens"], tensors["mask"], tensors["f0"][..., 45:], None)
                    if architecture_id == ARCH_A5S_HLG_SCRATCH
                    else model(tensors["hlt_tokens"], tensors["mask"], tensors["f0"], tensors["h0"])
                )
            selected = output.f_hat[..., :PHYSICAL_CHANNELS][tensors["mask"]]
            target = tensors["bridge_fields"][..., :PHYSICAL_CHANNELS][tensors["mask"]]
            score = float(F.mse_loss(selected, target).detach().cpu())
            if not math.isfinite(score):
                raise FloatingPointError("Step 7 paired miniature produced a non-finite score")
            rows.append({"architecture_id": architecture_id, "seed_id": seed, "bridge_mse": score})
            scores.append((score, seed))
        ordered = sorted(scores)
        aggregates[architecture_id] = {
            "mean_bridge_mse": float(np.mean([value for value, _ in scores])),
            "sample_std_bridge_mse": float(np.std([value for value, _ in scores], ddof=1)),
            "ordered_median_seed": int(ordered[1][1]),
            "paired_seed_ids": list(seed_ids),
        }
    return with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_STEP7_PAIRED_MINIATURE_CONTRACT,
            "architecture_ids": list(architecture_ids),
            "paired_seed_ids": list(seed_ids),
            "rows": rows,
            "aggregates": aggregates,
            "complete_hierarchy_matrix": set(architecture_ids) == set(STEP7_HIERARCHY_ARCHITECTURE_IDS),
            "scientific_results_allowed": False,
            "purpose": "tensor_path_and_paired_aggregation_rehearsal_only",
        }
    )


__all__ = [
    "PREDICTION_ANCHORED_HLG_CONFIG_CONTRACT",
    "PREDICTION_ANCHORED_HLG_REGION_CONTRACT",
    "PREDICTION_ANCHORED_HLG_TRANSFORMER_CONTRACT",
    "PREDICTION_ANCHORED_HLG_READBACK_CONTRACT",
    "PREDICTION_ANCHORED_HLG_MODEL_CONTRACT",
    "PREDICTION_ANCHORED_ABSOLUTE_SCALER_CONTRACT",
    "PREDICTION_ANCHORED_DIRECT_CONFIG_CONTRACT",
    "PREDICTION_ANCHORED_DIRECT_MODEL_CONTRACT",
    "PREDICTION_ANCHORED_DIRECT_TRAIN_CONTRACT",
    "PREDICTION_ANCHORED_DEPLOYED_RESOURCE_CONTRACT",
    "PREDICTION_ANCHORED_REPRESENTATIVE_RESOURCE_CONTRACT",
    "PREDICTION_ANCHORED_STEP7_RESOURCE_CONTRACT",
    "PREDICTION_ANCHORED_STEP7_MEASUREMENT_CONTRACT",
    "PREDICTION_ANCHORED_STEP7_RELOAD_CONTRACT",
    "PREDICTION_ANCHORED_STEP7_PAIRED_MINIATURE_CONTRACT",
    "ARCH_A2_REGIONS_NO_GLOBAL",
    "ARCH_A3_HLG_PRIMARY",
    "ARCH_A4_HLG_REFINE",
    "ARCH_A5_HLG_ABSOLUTE",
    "ARCH_A5S_HLG_SCRATCH",
    "ARCH_A6_HLG_NO_PAIR",
    "ARCH_A7_HLG_NO_H0",
    "ARCH_A7F_HLG_NO_F0",
    "ARCH_A7X_HLG_NO_RAW",
    "ARCH_A8_HLG_FUSED_HEAD",
    "ARCH_A9_HLG_GROUP_GATE",
    "ARCH_AS_HLG_REGIONS_2_2_1",
    "ARCH_AL_HLG_REGIONS_8_8_4",
    "ARCH_AFIX_HLG_FIXED_ASSIGNMENT",
    "ARCH_ASAME_HLG_SAME_SCALE",
    "ARCH_AGLOBAL_HLG_ONE_GLOBAL",
    "STEP7_HIERARCHY_ARCHITECTURE_IDS",
    "DIRECT_HLT",
    "DIRECT_R0REP",
    "STEP7_DIRECT_CONTROL_IDS",
    "STEP7_MEASURED_ARCHITECTURE_IDS",
    "OUTPUT_CORRECTION",
    "OUTPUT_ABSOLUTE",
    "ASSIGNMENT_SOFT",
    "ASSIGNMENT_FIXED",
    "GATE_INITIAL_PROBABILITY",
    "GATE_INITIAL_BIAS",
    "GATE_LOSS_COEFFICIENT",
    "REGION_POOL_CANONICAL_INPUT_DIM",
    "REGION_TOKEN_DIM",
    "REGION_DEAD_MASS_THRESHOLD",
    "STEP7_PAIRED_SEEDS",
    "HLGCorrectionConfig",
    "TorchAbsoluteOutputScaler",
    "SelectiveBaseFusion",
    "HLGRegionOutput",
    "HLGRegionPooler",
    "HLGGlobalOutput",
    "HLGRegionTransformer",
    "HLGParticleReadback",
    "HLGEncoding",
    "HLGParticleEncoder",
    "HLGRadiusCorrectionHead",
    "HLGCorrectionOutput",
    "PredictionAnchoredHLGCorrection",
    "RepresentativeArchitectureResourceReference",
    "DeployedBundleResourceReference",
    "BundleResourceReference",
    "DirectHLGConfig",
    "DirectHLGTrainConfig",
    "DirectHLGOutput",
    "DirectHLGClassifier",
    "HLGResourceProfile",
    "fit_absolute_output_scaler",
    "build_step7_hlg_correction_model",
    "step7_gate_regularization",
    "measure_step7_resources",
    "measure_module_forward_resources",
    "resource_reference_from_artifact",
    "fit_direct_hlg_config",
    "build_capacity_matched_direct_hlg",
    "direct_capacity_match",
    "train_step7_direct_hlg",
    "measure_step7_registry_states",
    "tiny_train_reload_step7_hierarchy",
    "tiny_train_reload_step7_direct",
    "run_step7_paired_seed_miniature",
]
