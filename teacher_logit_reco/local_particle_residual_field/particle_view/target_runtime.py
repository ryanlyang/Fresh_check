"""Cache-backed runtime integration for Stage-B particle-view discovery.

Every declared target-generator row uses this runtime.  Frozen Stage-A
teachers are authenticated and reloaded, contextual memories are staged in
allocation-local RAM, learned tap/source adapters are checkpointed alongside
``Gview``, and every available candidate runs the same complete two-pass
normalizer/consumer/recovery-probe evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from pathlib import Path
import shutil
from typing import Any, Mapping

import numpy as np
import torch

from jetclass_fresh.part_inputs import PF_FEATURE_NAMES

from .campaign import (
    NONSELECTABLE_TARGET_SCREEN_IDS,
    PARTICLE_VIEW_LOW_DATA_CAMPAIGN_ID,
    TARGET_SCREEN_IDS,
)
from .consumer import ParticleViewConsumer, ParticleViewConsumerConfig
from .consumer_train import (
    ParticleViewConsumerTrainConfig,
    evaluate_view_counterfactuals,
    train_particle_view_consumer,
)
from .contracts import (
    canonical_sha256,
    load_hashed_json,
    sha256_file,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from .offline_teacher import (
    FrozenContextualParticleTeacher,
    FrozenTokenLayerMixture,
    ParticleTokenTapSpec,
    ParticleViewTeacherRecipe,
    build_teacher_registration,
    build_token_tap_registration,
    reload_registered_teacher,
    validate_teacher_registration,
)
from .oracle_discovery import (
    OracleObjectiveConfig,
    RecoverabilityCoDesignConfig,
    RecoverabilityCoDesignProjection,
    build_codesign_ledger,
    build_target_metrics_from_counterfactual,
    build_two_pass_candidate_artifact,
    co_design_projection_loss,
    select_codesign_cycle,
)
from .recovery_probe import (
    FixedCapacityRecoveryProbe,
    RecoveryProbeConfig,
    recovery_probe_losses,
    train_recovery_probe,
)
from .registry import validate_particle_view_registry
from .runtime_data import (
    PARTICLE_VIEW_RUNTIME_DATA_CONFIG_CONTRACT,
    AlignedLogicalJetView,
    load_aligned_logical_jet_view,
    make_logical_data_loader,
    resolve_parent_task_artifacts,
    validate_runtime_data_config,
)
from .splits import logical_split_binding
from .tap_staging import (
    StagedTeacherTap,
    StreamingTeacherTapBuilder,
    build_tap_stage_reservation,
    validate_staged_teacher_tap,
)
from .target_generator import (
    MatchingFreeParticleViewGenerator,
    ParticleViewGeneratorConfig,
)
from .target_provenance import build_target_candidate_registration
from .view_cache import (
    ParticleViewNormalizer,
    fit_particle_view_normalizer,
    normalize_particle_view,
    quantized_view_diagnostics,
    write_particle_view_normalizer,
)


PARTICLE_VIEW_TARGET_DISCOVERY_FACTORY_CONFIG_CONTRACT = (
    "particle_view_target_discovery_factory_config_v1"
)
PARTICLE_VIEW_TARGET_DISCOVERY_RECIPE_CONTRACT = (
    "particle_view_target_discovery_recipe_v1"
)
PARTICLE_VIEW_TARGET_DISCOVERY_RESULT_CONTRACT = (
    "particle_view_target_discovery_result_v1"
)
PARTICLE_VIEW_TARGET_TWO_PASS_RESULT_CONTRACT = (
    "particle_view_target_two_pass_result_v1"
)
PARTICLE_VIEW_TARGET_UNAVAILABLE_CONTRACT = (
    "particle_view_target_unavailable_v1"
)
PARTICLE_VIEW_GENERATOR_CHECKPOINT_CONTRACT = (
    "particle_view_generator_checkpoint_v1"
)

CANONICAL_TARGET_DISCOVERY_RUN_ID = "VGEN_TAP_PENULT"
TARGET_SCREEN_FORWARD_COUNT = 2


class TwoTeacherTokenMixture(torch.nn.Module):
    """Learned common-width mixture of frozen base/large teacher tokens."""

    contract = "particle_view_two_teacher_token_mixture_v1"

    def __init__(
        self,
        *,
        base_dim: int = 128,
        large_dim: int = 192,
        output_dim: int = 160,
    ) -> None:
        super().__init__()
        if (base_dim, large_dim, output_dim) != (128, 192, 160):
            raise ValueError("two-teacher mixture dimensions drifted")
        self.base_dim = base_dim
        self.large_dim = large_dim
        self.output_dim = output_dim
        self.base_projection = torch.nn.Linear(base_dim, output_dim)
        self.large_projection = torch.nn.Linear(large_dim, output_dim)
        self.source_logits = torch.nn.Parameter(torch.zeros(2))

    def config_payload(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "base_dim": self.base_dim,
            "large_dim": self.large_dim,
            "output_dim": self.output_dim,
            "mixture": "learned_softmax",
            "teachers_frozen": True,
        }

    def forward(
        self,
        base_tokens: torch.Tensor,
        large_tokens: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if (
            base_tokens.ndim != 3
            or large_tokens.ndim != 3
            or base_tokens.shape[:2] != large_tokens.shape[:2]
            or base_tokens.shape[2] != self.base_dim
            or large_tokens.shape[2] != self.large_dim
            or mask.shape != base_tokens.shape[:2]
            or mask.dtype is not torch.bool
        ):
            raise ValueError("two-teacher token mixture input changed")
        weights = torch.softmax(self.source_logits, dim=0)
        mixed = (
            weights[0] * self.base_projection(base_tokens)
            + weights[1] * self.large_projection(large_tokens)
        )
        return mixed.masked_fill(~mask.unsqueeze(-1), 0.0)

    def mix_logits(
        self,
        base_logits: torch.Tensor,
        large_logits: torch.Tensor,
    ) -> torch.Tensor:
        if base_logits.shape != large_logits.shape or base_logits.ndim != 2:
            raise ValueError("two-teacher logits differ in shape")
        weights = torch.softmax(self.source_logits, dim=0)
        return weights[0] * base_logits + weights[1] * large_logits


def _optional_positive(value: int | None, *, name: str) -> int | None:
    if value is None:
        return None
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _resolve_device(value: str | torch.device) -> torch.device:
    if isinstance(value, torch.device):
        return value
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def lorentz_vectors_to_particle_geometry(
    lorentz_vectors: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Convert repository ``(px,py,pz,E)`` tensors to ``(pt,eta,phi,m)``."""

    if (
        lorentz_vectors.ndim != 3
        or lorentz_vectors.shape[1] != 4
        or mask.ndim != 3
        or mask.shape[1] != 1
        or mask.dtype is not torch.bool
        or lorentz_vectors.shape[0] != mask.shape[0]
        or lorentz_vectors.shape[2] != mask.shape[2]
    ):
        raise ValueError(
            "lorentz vectors/mask must be [B,4,P] and boolean [B,1,P]"
        )
    px, py, pz, energy = lorentz_vectors.unbind(dim=1)
    pt = torch.sqrt(px.square() + py.square()).clamp_min(1.0e-8)
    eta = torch.asinh(pz / pt)
    phi = torch.atan2(py, px)
    mass2 = (
        energy.square() - px.square() - py.square() - pz.square()
    ).clamp_min(0.0)
    mass = torch.sqrt(mass2)
    geometry = torch.stack((pt, eta, phi, mass), dim=-1)
    valid = mask[:, 0, :, None]
    geometry = torch.where(valid, geometry, torch.zeros_like(geometry))
    if not torch.isfinite(geometry).all():
        raise ValueError("particle geometry contains nonfinite values")
    return geometry


@dataclass(frozen=True)
class TargetScreenRecipe:
    run_id: str
    query_tap_choice: str
    memory_tap_choice: str
    teacher_source: str
    generator_config: ParticleViewGeneratorConfig
    oracle_objective: OracleObjectiveConfig
    selection_status: str
    recoverability_codesign: bool = False

    def to_payload(self) -> dict[str, Any]:
        if self.run_id not in TARGET_SCREEN_IDS:
            raise ValueError("unknown target-screen run")
        if self.query_tap_choice not in {
            "raw_features",
            "raw_embed",
            "middle",
            "penultimate",
            "final",
            "mix_last3",
        }:
            raise ValueError("invalid query tap choice")
        if self.memory_tap_choice not in {
            "raw_embed",
            "middle",
            "penultimate",
            "final",
            "mix_last3",
        }:
            raise ValueError("invalid memory tap choice")
        if self.teacher_source not in {
            "base",
            "large",
            "existing",
            "base_large_mix",
            "hlt",
        }:
            raise ValueError("invalid target teacher source")
        expected_status = (
            "diagnostic_nonselectable"
            if self.run_id in NONSELECTABLE_TARGET_SCREEN_IDS
            else "performance_control"
            if self.run_id == "VGEN_MEMORY_HLT"
            else "canonical_selectable"
            if self.run_id == CANONICAL_TARGET_DISCOVERY_RUN_ID
            else "selectable"
        )
        allowed_statuses = {expected_status}
        if self.run_id in {
            "VGEN_TEACHER_EXISTING",
            "VGEN_TEACHER_MIX2",
        }:
            allowed_statuses.add("diagnostic_nonselectable")
        if self.selection_status not in allowed_statuses:
            raise ValueError("target-screen selection status drifted")
        return {
            "contract": PARTICLE_VIEW_TARGET_DISCOVERY_RECIPE_CONTRACT,
            "run_id": self.run_id,
            "query_tap_choice": self.query_tap_choice,
            "memory_tap_choice": self.memory_tap_choice,
            "teacher_source": self.teacher_source,
            "generator_config": self.generator_config.to_payload(),
            "generator_config_sha256": self.generator_config.content_hash,
            "oracle_objective": self.oracle_objective.to_payload(),
            "oracle_objective_sha256": canonical_sha256(
                self.oracle_objective.to_payload()
            ),
            "selection_status": self.selection_status,
            "recoverability_codesign": self.recoverability_codesign,
            "training_role": "Cview_discovery",
            "single_training_pool": True,
            "cross_fit": False,
        }

    @property
    def content_hash(self) -> str:
        return canonical_sha256(self.to_payload())


def build_target_screen_recipe(
    run_id: str,
    *,
    query_dim: int = 128,
    memory_dim: int = 128,
    existing_teacher_compatible: bool = False,
    teacher_mix_compatible: bool = False,
) -> TargetScreenRecipe:
    """Compile one declared screen ID into an explicit scientific recipe."""

    if run_id not in TARGET_SCREEN_IDS:
        raise KeyError(f"unknown target-screen run {run_id!r}")
    query_tap = "penultimate"
    memory_tap = "penultimate"
    teacher_source = "base"
    blocks = 2
    use_pair = True
    local_radius = None
    use_null = True
    centered = True
    width = 4
    kd = 0.5
    rate_enabled = True
    codesign = run_id == "VGEN_RECODESIGN"
    memory_source = "offline"
    self_mask = False

    memory_taps = {
        "VGEN_TAP_RAW": "raw_embed",
        "VGEN_TAP_MID": "middle",
        "VGEN_TAP_PENULT": "penultimate",
        "VGEN_TAP_FINAL": "final",
        "VGEN_TAP_MIX3": "mix_last3",
    }
    query_taps = {
        "VGEN_QUERY_RAW": "raw_features",
        "VGEN_QUERY_EMBED": "raw_embed",
        "VGEN_QUERY_MID": "middle",
        "VGEN_QUERY_PENULT": "penultimate",
        "VGEN_QUERY_MIX3": "mix_last3",
    }
    memory_tap = memory_taps.get(run_id, memory_tap)
    query_tap = query_taps.get(run_id, query_tap)
    blocks = {
        "VGEN_XATTN1": 1,
        "VGEN_XATTN2": 2,
        "VGEN_XATTN4": 4,
    }.get(run_id, blocks)
    if run_id == "VGEN_NO_PAIR":
        use_pair = False
    if run_id == "VGEN_LOCAL02":
        local_radius = 0.2
    elif run_id == "VGEN_LOCAL04":
        local_radius = 0.4
    if run_id == "VGEN_NO_NULL":
        use_null = False
    if run_id == "VGEN_UNCENTERED":
        centered = False
    width = {
        "VGEN_DIM1": 1,
        "VGEN_DIM2": 2,
        "VGEN_DIM4": 4,
        "VGEN_DIM8": 8,
    }.get(run_id, width)
    kd = {
        "VGEN_KD000": 0.0,
        "VGEN_KD025": 0.25,
        "VGEN_KD050": 0.5,
        "VGEN_KD100": 1.0,
    }.get(run_id, kd)
    if run_id == "VGEN_NO_RATE":
        rate_enabled = False
    teacher_source = {
        "VGEN_TEACHER_LARGE": "large",
        "VGEN_TEACHER_EXISTING": "existing",
        "VGEN_TEACHER_MIX2": "base_large_mix",
    }.get(run_id, teacher_source)
    if run_id == "VGEN_TEACHER_LARGE":
        memory_dim = 192
    elif run_id == "VGEN_TEACHER_MIX2":
        # The later mix runtime projects both frozen teachers into this
        # preregistered common width before the canonical generator.
        memory_dim = 160
    if run_id in {"VGEN_MEMORY_HLT", "VGEN_MEMORY_HLT_SELFMASK"}:
        teacher_source = "hlt"
        memory_source = "hlt"
        memory_dim = query_dim
        kd = 0.0
        self_mask = run_id == "VGEN_MEMORY_HLT_SELFMASK"

    actual_query_dim = (
        len(PF_FEATURE_NAMES) if query_tap == "raw_features" else query_dim
    )
    objective = (
        OracleObjectiveConfig(offline_kd_weight=kd)
        if rate_enabled
        else OracleObjectiveConfig(
            offline_kd_weight=kd,
            rate_weight=0.0,
            covariance_weight=0.0,
            rate_budget_enabled=False,
        )
    )
    status = (
        "diagnostic_nonselectable"
        if (
            run_id in NONSELECTABLE_TARGET_SCREEN_IDS
            or (
                run_id == "VGEN_TEACHER_EXISTING"
                and not existing_teacher_compatible
            )
            or (
                run_id == "VGEN_TEACHER_MIX2"
                and not teacher_mix_compatible
            )
        )
        else "performance_control"
        if run_id == "VGEN_MEMORY_HLT"
        else "canonical_selectable"
        if run_id == CANONICAL_TARGET_DISCOVERY_RUN_ID
        else "selectable"
    )
    return TargetScreenRecipe(
        run_id=run_id,
        query_tap_choice=query_tap,
        memory_tap_choice=memory_tap,
        teacher_source=teacher_source,
        generator_config=ParticleViewGeneratorConfig(
            query_dim=actual_query_dim,
            memory_dim=memory_dim,
            num_cross_attention_blocks=blocks,
            bottleneck_width=width,
            use_pair_bias=use_pair,
            use_null_token=use_null,
            center_output=centered,
            memory_source=memory_source,
            self_mask_same_particle=self_mask,
            hard_local_radius=local_radius,
        ),
        oracle_objective=objective,
        selection_status=status,
        recoverability_codesign=codesign,
    )


@dataclass
class StagedContextualMemory:
    staged_tap: StagedTeacherTap
    logits: torch.Tensor
    parent_indices: torch.Tensor

    def __post_init__(self) -> None:
        validate_staged_teacher_tap(self.staged_tap)
        if (
            self.logits.device.type != "cpu"
            or self.logits.dtype is not torch.float32
            or self.logits.ndim != 2
            or self.parent_indices.device.type != "cpu"
            or self.parent_indices.dtype is not torch.int64
            or self.parent_indices.ndim != 1
            or self.logits.shape[0] != self.parent_indices.shape[0]
            or self.staged_tap.tokens.shape[0] != self.parent_indices.shape[0]
        ):
            raise ValueError("staged contextual memory inventory is invalid")
        if torch.unique(self.parent_indices).numel() != self.parent_indices.numel():
            raise ValueError("staged contextual memory has duplicate parent rows")
        self._position_by_parent = {
            int(parent): index
            for index, parent in enumerate(self.parent_indices.tolist())
        }

    def positions(self, parent_indices: torch.Tensor) -> torch.Tensor:
        requested = parent_indices.detach().cpu().to(dtype=torch.int64)
        try:
            values = [
                self._position_by_parent[int(parent)]
                for parent in requested.tolist()
            ]
        except KeyError as exc:
            raise ValueError("batch parent row is absent from staged memory") from exc
        return torch.tensor(values, dtype=torch.int64)


def stage_contextual_memory(
    *,
    aligned: AlignedLogicalJetView,
    teacher: FrozenContextualParticleTeacher,
    teacher_checkpoint_sha256: str,
    tap_spec_sha256: str,
    source_manifest_sha256: str,
    source_role: str,
    source_view: str,
    device: str | torch.device,
    num_workers: int,
    batch_size: int = 128,
) -> StagedContextualMemory:
    """Evaluate a frozen tap once and retain only float16 tokens in CPU RAM."""

    if source_view not in {"offline", "fixed_hlt"}:
        raise ValueError("staged source_view must be offline or fixed_hlt")
    resolved_device = _resolve_device(device)
    teacher = teacher.to(resolved_device).eval()
    loader = make_logical_data_loader(
        aligned,
        mode="aligned",
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        seed=101,
    )
    total_rows = len(aligned)
    if total_rows <= 0:
        raise ValueError("cannot stage an empty logical split")
    builder: StreamingTeacherTapBuilder | None = None
    logits: torch.Tensor | None = None
    parents = torch.empty(total_rows, dtype=torch.int64)
    cursor = 0
    prefix = "" if source_view == "fixed_hlt" else "offline_"
    for raw in loader:
        batch = {
            name: (
                value.to(resolved_device, non_blocking=True)
                if isinstance(value, torch.Tensor)
                else value
            )
            for name, value in raw.items()
        }
        context = teacher(
            batch[f"{prefix}points" if prefix else "points"],
            batch[f"{prefix}features" if prefix else "features"],
            batch[
                f"{prefix}lorentz_vectors"
                if prefix
                else "lorentz_vectors"
            ],
            batch[f"{prefix}mask" if prefix else "mask"],
        )
        if context.particle_tokens.shape[1] == 1:
            staged_tokens = context.single_layer_tokens
        else:
            # Preserve every frozen layer in one authenticated rank-3 staging
            # tensor.  The learned layer mixture remains live and trainable.
            staged_tokens = context.particle_tokens.permute(
                0, 2, 1, 3
            ).reshape(
                context.particle_tokens.shape[0],
                context.particle_tokens.shape[2],
                -1,
            )
        staged_tokens = staged_tokens.float()
        batch_rows = int(staged_tokens.shape[0])
        end = cursor + batch_rows
        if end > total_rows:
            raise ValueError("staged contextual memory exceeds logical split")
        if builder is None:
            reservation = build_tap_stage_reservation(
                source_role=source_role,
                source_manifest_sha256=source_manifest_sha256,
                logical_split_sha256=aligned.logical_split_sha256,
                ordered_identity_sha256=aligned.ordered_identity_sha256,
                teacher_checkpoint_sha256=teacher_checkpoint_sha256,
                tap_spec_sha256=tap_spec_sha256,
                jets=total_rows,
                max_particles=int(staged_tokens.shape[1]),
                token_width=int(staged_tokens.shape[2]),
                identity_columns=2,
            )
            builder = StreamingTeacherTapBuilder(reservation=reservation)
            logits = torch.empty(
                (total_rows, int(context.logits.shape[1])),
                dtype=torch.float32,
            )
        elif tuple(staged_tokens.shape[1:]) != tuple(builder.tokens.shape[1:]):
            raise ValueError("teacher tap shape changed between staging batches")
        assert logits is not None
        if context.logits.shape[1] != logits.shape[1]:
            raise ValueError("teacher logit width changed between staging batches")
        batch_parents = batch["parent_indices"].detach().cpu().to(torch.int64)
        batch_labels = batch["labels"].detach().cpu().to(torch.int64)
        identities = torch.stack((batch_parents, batch_labels), dim=1)
        builder.append(staged_tokens, context.particle_mask, identities)
        logits[cursor:end].copy_(context.logits.detach().float().cpu())
        parents[cursor:end].copy_(batch_parents)
        cursor = end
    if builder is None or logits is None:
        raise ValueError("cannot stage an empty logical split")
    if cursor != total_rows:
        raise ValueError(
            "staged contextual memory row count differs from logical split"
        )
    staged = builder.finalize()
    return StagedContextualMemory(
        staged_tap=staged,
        logits=logits,
        parent_indices=parents,
    )


class StagedDiscoveryViewProvider:
    """Produce live HLT queries against authenticated RAM-staged memory."""

    def __init__(
        self,
        *,
        generator: MatchingFreeParticleViewGenerator,
        query_teacher: FrozenContextualParticleTeacher,
        staged_memory: StagedContextualMemory,
        query_tap_choice: str,
        memory_source: str = "offline",
        query_mixture: FrozenTokenLayerMixture | None = None,
        memory_mixture: FrozenTokenLayerMixture | None = None,
        memory_layer_width: int | None = None,
        secondary_staged_memory: StagedContextualMemory | None = None,
        teacher_mixture: TwoTeacherTokenMixture | None = None,
        codesign_projection: RecoverabilityCoDesignProjection | None = None,
    ) -> None:
        if memory_source not in {"offline", "hlt"}:
            raise ValueError("memory_source must be offline or hlt")
        if (memory_mixture is None) != (memory_layer_width is None):
            raise ValueError("memory layer mixture/width must be paired")
        if (secondary_staged_memory is None) != (teacher_mixture is None):
            raise ValueError("secondary memory/teacher mixture must be paired")
        if memory_mixture is not None and teacher_mixture is not None:
            raise ValueError("layer and teacher mixtures cannot be combined")
        if teacher_mixture is not None and memory_source != "offline":
            raise ValueError("two-teacher mixture requires offline memory")
        self.generator = generator
        self.query_teacher = query_teacher
        self.staged_memory = staged_memory
        self.query_tap_choice = query_tap_choice
        self.memory_source = memory_source
        self.query_mixture = query_mixture
        self.memory_mixture = memory_mixture
        self.memory_layer_width = memory_layer_width
        self.secondary_staged_memory = secondary_staged_memory
        self.teacher_mixture = teacher_mixture
        self.codesign_projection = codesign_projection

    def __call__(self, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        required = {
            "points",
            "features",
            "lorentz_vectors",
            "mask",
            "offline_lorentz_vectors",
            "offline_mask",
            "parent_indices",
        }
        if not required.issubset(batch):
            raise ValueError("aligned discovery batch is incomplete")
        query_context = self.query_teacher(
            batch["points"],
            batch["features"],
            batch["lorentz_vectors"],
            batch["mask"],
        )
        if self.query_tap_choice == "raw_features":
            query_tokens = batch["features"].transpose(1, 2)
        elif self.query_mixture is not None:
            query_tokens = self.query_mixture(
                query_context.particle_tokens,
                query_context.particle_mask,
            )
        else:
            query_tokens = query_context.single_layer_tokens
        positions = self.staged_memory.positions(batch["parent_indices"])
        device = query_tokens.device
        memory_tokens = self.staged_memory.staged_tap.tokens[
            positions
        ].to(device=device, dtype=torch.float32)
        memory_mask = self.staged_memory.staged_tap.mask[positions].to(
            device=device
        )
        expected_mask = (
            batch["mask"][:, 0]
            if self.memory_source == "hlt"
            else batch["offline_mask"][:, 0]
        )
        if not torch.equal(memory_mask, expected_mask):
            raise ValueError("RAM-staged memory mask differs from aligned cache")
        if self.memory_mixture is not None:
            width = int(self.memory_layer_width)
            if memory_tokens.shape[2] != 3 * width:
                raise ValueError("staged layer-mixture width changed")
            memory_tokens = self.memory_mixture(
                memory_tokens.reshape(
                    memory_tokens.shape[0],
                    memory_tokens.shape[1],
                    3,
                    width,
                ).permute(0, 2, 1, 3),
                memory_mask,
            )
        offline_logits = self.staged_memory.logits[positions].to(device=device)
        if self.teacher_mixture is not None:
            secondary = self.secondary_staged_memory
            secondary_positions = secondary.positions(batch["parent_indices"])
            secondary_mask = secondary.staged_tap.mask[
                secondary_positions
            ].to(device=device)
            if not torch.equal(secondary_mask, memory_mask):
                raise ValueError("two-teacher staged masks differ")
            secondary_tokens = secondary.staged_tap.tokens[
                secondary_positions
            ].to(device=device, dtype=torch.float32)
            memory_tokens = self.teacher_mixture(
                memory_tokens,
                secondary_tokens,
                memory_mask,
            )
            offline_logits = self.teacher_mixture.mix_logits(
                offline_logits,
                secondary.logits[secondary_positions].to(device=device),
            )
        if isinstance(batch, dict):
            batch["offline_logits"] = offline_logits
        else:
            raise TypeError("discovery provider requires a mutable batch mapping")
        output = self.generator(
            query_tokens,
            memory_tokens,
            query_geometry=lorentz_vectors_to_particle_geometry(
                batch["lorentz_vectors"], batch["mask"]
            ),
            memory_geometry=lorentz_vectors_to_particle_geometry(
                (
                    batch["lorentz_vectors"]
                    if self.memory_source == "hlt"
                    else batch["offline_lorentz_vectors"]
                ),
                (
                    batch["mask"]
                    if self.memory_source == "hlt"
                    else batch["offline_mask"]
                ),
            ),
            query_mask=batch["mask"][:, 0],
            memory_mask=memory_mask,
        )
        if self.codesign_projection is None:
            return {
                "view": output.view,
                "raw_centered_view": output.deterministic_centered_view,
            }
        deterministic = self.codesign_projection(
            output.rich_context,
            batch["mask"][:, 0],
        )
        view = (
            self.generator._augment(
                deterministic,
                batch["mask"][:, 0],
                generator=None,
            )
            if self.codesign_projection.training
            else deterministic
        )
        return {
            "view": view,
            "raw_centered_view": deterministic,
        }


@dataclass
class CandidateViewSet:
    """One logical split of candidate views retained only in process RAM."""

    views: torch.Tensor
    mask: torch.Tensor
    parent_indices: torch.Tensor
    logical_split_sha256: str
    ordered_identity_sha256: str

    def __post_init__(self) -> None:
        if (
            self.views.device.type != "cpu"
            or self.views.dtype is not torch.float32
            or self.views.ndim != 3
            or self.mask.device.type != "cpu"
            or self.mask.dtype is not torch.bool
            or self.mask.shape != self.views.shape[:2]
            or self.parent_indices.device.type != "cpu"
            or self.parent_indices.dtype is not torch.int64
            or self.parent_indices.shape != (self.views.shape[0],)
        ):
            raise ValueError("candidate view-set tensor inventory is invalid")
        if not torch.isfinite(self.views).all():
            raise ValueError("candidate views contain nonfinite entries")
        if torch.count_nonzero(self.views[~self.mask]):
            raise ValueError("candidate views must be exactly zero on padding")
        if torch.unique(self.parent_indices).numel() != self.parent_indices.numel():
            raise ValueError("candidate view set has duplicate parent rows")
        for name, value in (
            ("logical_split_sha256", self.logical_split_sha256),
            ("ordered_identity_sha256", self.ordered_identity_sha256),
        ):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        self._position_by_parent = {
            int(parent): index
            for index, parent in enumerate(self.parent_indices.tolist())
        }

    def positions(self, parent_indices: torch.Tensor) -> torch.Tensor:
        requested = parent_indices.detach().cpu().to(dtype=torch.int64)
        try:
            values = [
                self._position_by_parent[int(parent)]
                for parent in requested.tolist()
            ]
        except KeyError as exc:
            raise ValueError("batch parent row is absent from candidate views") from exc
        return torch.tensor(values, dtype=torch.int64)

    def normalized(
        self, normalizer: ParticleViewNormalizer
    ) -> "CandidateViewSet":
        normalized = normalize_particle_view(
            self.views,
            self.mask,
            normalizer,
        ).float().cpu().contiguous()
        return CandidateViewSet(
            views=normalized,
            mask=self.mask,
            parent_indices=self.parent_indices,
            logical_split_sha256=self.logical_split_sha256,
            ordered_identity_sha256=self.ordered_identity_sha256,
        )

    def numpy_views(self) -> np.ndarray:
        return np.ascontiguousarray(self.views.numpy(), dtype="<f4")


class IndexedCandidateViewProvider:
    """Attach one immutable RAM view set to aligned batches by parent row."""

    def __init__(self, view_set: CandidateViewSet) -> None:
        self.view_set = view_set

    def __call__(self, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        if "parent_indices" not in batch:
            raise ValueError("candidate-view batch omitted parent indices")
        positions = self.view_set.positions(batch["parent_indices"])
        device = batch["features"].device
        view = self.view_set.views[positions].to(device=device)
        mask = self.view_set.mask[positions].to(device=device)
        if not torch.equal(mask, batch["mask"][:, 0]):
            raise ValueError("candidate view mask differs from HLT batch mask")
        return {"view": view, "raw_centered_view": view}


def materialize_candidate_views_in_ram(
    *,
    aligned: AlignedLogicalJetView,
    provider: StagedDiscoveryViewProvider,
    device: str | torch.device,
    num_workers: int,
) -> CandidateViewSet:
    """Evaluate the frozen generator once without writing view payloads."""

    resolved_device = _resolve_device(device)
    provider.generator.eval()
    loader = make_logical_data_loader(
        aligned,
        mode="aligned",
        batch_size=128,
        shuffle=False,
        num_workers=num_workers,
        seed=101,
    )
    view_rows: list[torch.Tensor] = []
    mask_rows: list[torch.Tensor] = []
    parent_rows: list[torch.Tensor] = []
    with torch.no_grad():
        for raw in loader:
            batch = {
                name: (
                    value.to(resolved_device, non_blocking=True)
                    if isinstance(value, torch.Tensor)
                    else value
                )
                for name, value in raw.items()
            }
            generated = provider(batch)["raw_centered_view"]
            view_rows.append(generated.float().cpu())
            mask_rows.append(batch["mask"][:, 0].cpu())
            parent_rows.append(batch["parent_indices"].cpu())
    if not view_rows:
        raise ValueError("cannot materialize views for an empty split")
    result = CandidateViewSet(
        views=torch.cat(view_rows, dim=0).contiguous(),
        mask=torch.cat(mask_rows, dim=0).contiguous(),
        parent_indices=torch.cat(parent_rows, dim=0)
        .to(dtype=torch.int64)
        .contiguous(),
        logical_split_sha256=aligned.logical_split_sha256,
        ordered_identity_sha256=aligned.ordered_identity_sha256,
    )
    expected_parents = torch.from_numpy(
        np.asarray(aligned.parent_row_indices, dtype=np.int64)
    )
    if not torch.equal(result.parent_indices, expected_parents):
        raise ValueError("candidate view materialization changed logical row order")
    return result


class _CounterfactualLoader:
    def __init__(
        self,
        loader,
        *,
        true_views: CandidateViewSet,
        predicted_views: CandidateViewSet,
    ) -> None:
        if (
            true_views.logical_split_sha256
            != predicted_views.logical_split_sha256
            or true_views.ordered_identity_sha256
            != predicted_views.ordered_identity_sha256
            or not torch.equal(
                true_views.parent_indices, predicted_views.parent_indices
            )
            or not torch.equal(true_views.mask, predicted_views.mask)
        ):
            raise ValueError("true/predicted counterfactual view sets differ")
        self.loader = loader
        self.true_views = true_views
        self.predicted_views = predicted_views
        self.batch_size = getattr(loader, "batch_size", None)

    def __iter__(self):
        for raw in self.loader:
            batch = dict(raw)
            positions = self.true_views.positions(batch["parent_indices"])
            true_view = self.true_views.views[positions]
            predicted_view = self.predicted_views.views[positions]
            expected_mask = self.true_views.mask[positions]
            if not torch.equal(expected_mask, batch["mask"][:, 0]):
                raise ValueError("counterfactual view mask differs from HLT batch")
            batch["true_view"] = true_view
            batch["predicted_view"] = predicted_view
            yield batch

    def __len__(self) -> int:
        return len(self.loader)


class _LimitedLoader:
    def __init__(self, loader, maximum_batches: int | None) -> None:
        self.loader = loader
        self.maximum_batches = maximum_batches
        self.batch_size = getattr(loader, "batch_size", None)

    def __iter__(self):
        if self.maximum_batches is None:
            return iter(self.loader)
        return islice(iter(self.loader), self.maximum_batches)

    def __len__(self) -> int:
        base = len(self.loader)
        return base if self.maximum_batches is None else min(
            base, self.maximum_batches
        )


def _load_selected_probe_consumer(
    model: ParticleViewConsumer,
    checkpoint_path: Path,
    *,
    expected_role: str,
    device: torch.device,
) -> ParticleViewConsumer:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if (
        checkpoint.get("role") != expected_role
        or checkpoint.get("consumer_config") != model.config.to_payload()
        or checkpoint.get("consumer_config_sha256") != model.config.content_hash
        or set(checkpoint.get("joint_model_state_dicts", {}))
    ):
        raise ValueError("probe-consumer checkpoint contract differs")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model


def _load_selected_recovery_probe(
    checkpoint_path: Path,
    *,
    config: RecoveryProbeConfig,
    device: torch.device,
) -> FixedCapacityRecoveryProbe:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if (
        checkpoint.get("contract") != "particle_view_recovery_probe_v1"
        or checkpoint.get("config") != config.to_payload()
        or checkpoint.get("config_sha256") != config.content_hash
    ):
        raise ValueError("recovery-probe checkpoint contract differs")
    model = FixedCapacityRecoveryProbe(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval()


def predict_recovery_probe_views(
    *,
    model: FixedCapacityRecoveryProbe,
    aligned: AlignedLogicalJetView,
    true_views: CandidateViewSet,
    device: str | torch.device,
    num_workers: int,
) -> CandidateViewSet:
    """Predict one split without exposing labels to the fixed probe."""

    if true_views.logical_split_sha256 != aligned.logical_split_sha256:
        raise ValueError("recovery truth belongs to another logical split")
    resolved_device = _resolve_device(device)
    loader = make_logical_data_loader(
        aligned,
        mode="recovery_probe",
        true_views=true_views.numpy_views(),
        batch_size=128,
        shuffle=False,
        num_workers=num_workers,
        seed=101,
    )
    predictions: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    with torch.no_grad():
        for raw in loader:
            if set(raw) != {"features", "mask", "true_view"}:
                raise ValueError("recovery inference exposed unexpected fields")
            features = raw["features"].to(resolved_device, non_blocking=True)
            mask = raw["mask"].to(resolved_device, non_blocking=True)
            prediction = model(features, mask)
            predictions.append(prediction.float().cpu())
            masks.append(mask.cpu())
    if not predictions:
        raise ValueError("cannot predict an empty recovery split")
    predicted_mask = torch.cat(masks, dim=0).contiguous()
    if not torch.equal(predicted_mask, true_views.mask):
        raise ValueError("recovery prediction changed logical mask order")
    return CandidateViewSet(
        views=torch.cat(predictions, dim=0).contiguous(),
        mask=predicted_mask,
        parent_indices=true_views.parent_indices,
        logical_split_sha256=true_views.logical_split_sha256,
        ordered_identity_sha256=true_views.ordered_identity_sha256,
    )


def evaluate_independent_a0_accuracy(
    *,
    query_teacher: FrozenContextualParticleTeacher,
    aligned: AlignedLogicalJetView,
    device: str | torch.device,
    num_workers: int,
    maximum_batches: int | None,
) -> float:
    """Evaluate the untouched A0 checkpoint on the exact ranking rows."""

    resolved_device = _resolve_device(device)
    query_teacher = query_teacher.to(resolved_device).eval()
    loader = _LimitedLoader(
        make_logical_data_loader(
            aligned,
            mode="aligned",
            batch_size=128,
            shuffle=False,
            num_workers=num_workers,
            seed=101,
        ),
        maximum_batches,
    )
    correct = total = 0
    with torch.no_grad():
        for raw in loader:
            batch = {
                name: (
                    value.to(resolved_device, non_blocking=True)
                    if isinstance(value, torch.Tensor)
                    else value
                )
                for name, value in raw.items()
            }
            context = query_teacher(
                batch["points"],
                batch["features"],
                batch["lorentz_vectors"],
                batch["mask"],
            )
            labels = batch["labels"]
            correct += int(context.logits.argmax(dim=1).eq(labels).sum().item())
            total += int(labels.numel())
    if total == 0:
        raise ValueError("independent A0 ranking loader is empty")
    return correct / total


def _generator_checkpoint(
    *,
    consumer_checkpoint_path: Path,
    generator: MatchingFreeParticleViewGenerator,
    source_modules: Mapping[str, torch.nn.Module],
    recipe: Mapping[str, Any],
    output_path: Path,
) -> str:
    checkpoint = torch.load(
        consumer_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    states = checkpoint.get("joint_model_state_dicts", {})
    expected_states = {"Gview", *source_modules}
    if set(states) != expected_states:
        raise ValueError("discovery checkpoint joint-module inventory differs")
    generator.load_state_dict(states["Gview"], strict=True)
    for name, module in source_modules.items():
        module.load_state_dict(states[name], strict=True)
        module.eval()
    torch.save(
        {
            "contract": PARTICLE_VIEW_GENERATOR_CHECKPOINT_CONTRACT,
            "generator_config": generator.config.to_payload(),
            "generator_config_sha256": generator.config.content_hash,
            "target_discovery_recipe": dict(recipe),
            "target_discovery_recipe_sha256": recipe["content_hash"],
            "selected_consumer_epoch": int(checkpoint["epoch"]),
            "model_val_stop": dict(checkpoint["model_val_stop"]),
            "model_state_dict": generator.state_dict(),
            "source_adapter_configs": {
                name: (
                    module.config_payload()
                    if isinstance(module, TwoTeacherTokenMixture)
                    else {
                        "contract": "particle_view_token_layer_mixture_v1",
                        "layer_count": 3,
                        "mixture": "learned_softmax",
                        "teachers_frozen": True,
                    }
                )
                for name, module in source_modules.items()
            },
            "source_adapter_state_dicts": {
                name: module.state_dict()
                for name, module in source_modules.items()
            },
        },
        output_path,
    )
    return sha256_file(output_path)


class _CyclingBatches:
    def __init__(self, loader) -> None:
        self.loader = loader
        self.iterator = iter(loader)

    def next(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            try:
                return next(self.iterator)
            except StopIteration as exc:
                raise ValueError("co-design loader is empty") from exc


def _device_batch(
    raw: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    return {
        name: (
            value.to(device=device, non_blocking=True)
            if isinstance(value, torch.Tensor)
            else value
        )
        for name, value in raw.items()
    }


def _set_trainable(module: torch.nn.Module, trainable: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(trainable)
        if not trainable:
            parameter.grad = None
    module.train(trainable)


def _evaluate_codesign_cycle(
    *,
    consumer: ParticleViewConsumer,
    probe: FixedCapacityRecoveryProbe,
    provider: StagedDiscoveryViewProvider,
    loader,
    device: torch.device,
) -> dict[str, float]:
    consumer.eval()
    probe.eval()
    provider.generator.eval()
    provider.codesign_projection.eval()
    totals = {
        "zero_correct": 0,
        "true_correct": 0,
        "predicted_correct": 0,
        "predicted_ce": 0.0,
        "examples": 0,
    }
    with torch.no_grad():
        for raw in loader:
            batch = _device_batch(raw, device)
            true_view = provider(batch)["raw_centered_view"]
            predicted_view = probe(
                batch["features"], batch["mask"][:, 0]
            )
            zero_view = torch.zeros_like(true_view)
            forwards = {}
            for name, view in (
                ("zero", zero_view),
                ("true", true_view),
                ("predicted", predicted_view),
            ):
                forwards[name] = consumer(
                    batch["points"],
                    batch["features"],
                    batch["lorentz_vectors"],
                    batch["mask"],
                    view,
                    augment_clean_view=False,
                ).logits
            labels = batch["labels"]
            totals["zero_correct"] += int(
                forwards["zero"].argmax(dim=1).eq(labels).sum().item()
            )
            totals["true_correct"] += int(
                forwards["true"].argmax(dim=1).eq(labels).sum().item()
            )
            totals["predicted_correct"] += int(
                forwards["predicted"].argmax(dim=1).eq(labels).sum().item()
            )
            totals["predicted_ce"] += float(
                torch.nn.functional.cross_entropy(
                    forwards["predicted"].float(),
                    labels,
                    reduction="sum",
                ).item()
            )
            totals["examples"] += int(labels.numel())
    examples = int(totals["examples"])
    if examples == 0:
        raise ValueError("co-design validation loader is empty")
    zero_accuracy = totals["zero_correct"] / examples
    true_accuracy = totals["true_correct"] / examples
    predicted_accuracy = totals["predicted_correct"] / examples
    return {
        "zero_view_accuracy": zero_accuracy,
        "oracle_accuracy": true_accuracy,
        "predicted_view_accuracy": predicted_accuracy,
        "oracle_gain": true_accuracy - zero_accuracy,
        "predicted_gain": predicted_accuracy - zero_accuracy,
        "cross_entropy": totals["predicted_ce"] / examples,
        "examples": examples,
    }


def run_recoverability_codesign(
    *,
    generator: MatchingFreeParticleViewGenerator,
    consumer: ParticleViewConsumer,
    train_provider: StagedDiscoveryViewProvider,
    stop_provider: StagedDiscoveryViewProvider,
    train_loader,
    stop_loader,
    oracle_config: OracleObjectiveConfig,
    provisional_generator_checkpoint_sha256: str,
    provisional_consumer_registration_sha256: str,
    output_dir: str | Path,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    """Run the locked 12-cycle persistent-probe co-design procedure."""

    config = RecoverabilityCoDesignConfig(
        view_dim=generator.config.bottleneck_width,
        seed=seed,
    )
    if generator.config.width != config.rich_dim:
        raise ValueError("co-design rich-context width changed")
    torch.manual_seed(seed)
    projection = RecoverabilityCoDesignProjection(config).to(device)
    probe_config = RecoveryProbeConfig(
        view_dim=config.view_dim,
        seed=seed,
    )
    probe = FixedCapacityRecoveryProbe(probe_config).to(device)
    consumer = consumer.to(device)
    generator = generator.to(device).eval()
    _set_trainable(generator, False)
    train_provider.codesign_projection = projection
    stop_provider.codesign_projection = projection

    root = Path(output_dir).resolve() / "recoverability_codesign"
    cycle_root = root / "cycle_checkpoints"
    cycle_root.mkdir(parents=True, exist_ok=True)
    rich_registration = with_content_hash(
        {
            "contract": "particle_view_rview_registration_v1",
            "generator_config_sha256": generator.config.content_hash,
            "provisional_generator_checkpoint_sha256": (
                provisional_generator_checkpoint_sha256
            ),
            "rich_width": config.rich_dim,
            "rich_context_normalization": "LayerNorm",
            "bottleneck_parameters_excluded_from_coordinate": True,
            "frozen_for_all_codesign_cycles": True,
        }
    )
    write_immutable_json(
        root / "rich_context_registration.json",
        rich_registration,
    )
    provisional_head = with_content_hash(
        {
            "contract": "particle_view_codesign_provisional_head_v1",
            "provisional_generator_checkpoint_sha256": (
                provisional_generator_checkpoint_sha256
            ),
            "provisional_consumer_registration_sha256": (
                provisional_consumer_registration_sha256
            ),
            "head_role": "Bseed",
            "discarded_before_cycle_1": True,
        }
    )
    write_immutable_json(
        root / "provisional_head_registration.json",
        provisional_head,
    )

    probe_optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=config.probe_learning_rate,
        weight_decay=config.probe_weight_decay,
    )
    projection_optimizer = torch.optim.AdamW(
        projection.parameters(),
        lr=config.projection_learning_rate,
        weight_decay=config.projection_weight_decay,
    )
    consumer_optimizer = torch.optim.AdamW(
        consumer.parameters(),
        lr=config.consumer_learning_rate,
        weight_decay=config.consumer_weight_decay,
    )
    cycling = _CyclingBatches(train_loader)
    cycle_rows = []
    checkpoint_paths: dict[int, dict[str, Path]] = {}
    for cycle in range(1, config.cycles + 1):
        _set_trainable(probe, True)
        _set_trainable(projection, False)
        _set_trainable(consumer, False)
        for _ in range(config.probe_steps_per_cycle):
            batch = _device_batch(cycling.next(), device)
            with torch.no_grad():
                target = train_provider(batch)["raw_centered_view"]
            probe_optimizer.zero_grad(set_to_none=True)
            prediction = probe(
                batch["features"], batch["mask"][:, 0]
            )
            probe_loss = recovery_probe_losses(
                prediction,
                target,
                batch["mask"][:, 0],
            )["total"]
            if not torch.isfinite(probe_loss):
                raise FloatingPointError("co-design probe loss is non-finite")
            probe_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                probe.parameters(), config.gradient_clip
            )
            probe_optimizer.step()

        _set_trainable(probe, False)
        _set_trainable(projection, True)
        _set_trainable(consumer, True)
        for _ in range(config.projection_steps_per_cycle):
            batch = _device_batch(cycling.next(), device)
            projection_optimizer.zero_grad(set_to_none=True)
            consumer_optimizer.zero_grad(set_to_none=True)
            generated = train_provider(batch)
            deterministic = generated["raw_centered_view"]
            with torch.no_grad():
                probe_prediction = probe(
                    batch["features"], batch["mask"][:, 0]
                )
            consumer_output = consumer(
                batch["points"],
                batch["features"],
                batch["lorentz_vectors"],
                batch["mask"],
                generated["view"],
                augment_clean_view=False,
            )
            losses = co_design_projection_loss(
                consumer_logits=consumer_output.logits,
                labels=batch["labels"],
                offline_logits=batch["offline_logits"],
                raw_centered_view=deterministic,
                probe_prediction=probe_prediction,
                mask=batch["mask"][:, 0],
                trust_loss=consumer_output.trust_loss,
                oracle_config=oracle_config,
            )
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError(
                    "co-design projection loss is non-finite"
                )
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(
                projection.parameters(), config.gradient_clip
            )
            torch.nn.utils.clip_grad_norm_(
                consumer.parameters(), config.gradient_clip
            )
            projection_optimizer.step()
            consumer_optimizer.step()

        metrics = _evaluate_codesign_cycle(
            consumer=consumer,
            probe=probe,
            provider=stop_provider,
            loader=stop_loader,
            device=device,
        )
        paths = {
            "projection": cycle_root / f"cycle_{cycle:02d}_projection.pt",
            "consumer": cycle_root / f"cycle_{cycle:02d}_consumer.pt",
            "probe": cycle_root / f"cycle_{cycle:02d}_probe.pt",
        }
        torch.save(
            {
                "contract": "particle_view_codesign_projection_cycle_v1",
                "cycle": cycle,
                "config": config.to_payload(),
                "model_state_dict": projection.state_dict(),
            },
            paths["projection"],
        )
        torch.save(
            {
                "contract": "particle_view_codesign_consumer_cycle_v1",
                "cycle": cycle,
                "consumer_config": consumer.config.to_payload(),
                "model_state_dict": consumer.state_dict(),
            },
            paths["consumer"],
        )
        torch.save(
            {
                "contract": "particle_view_codesign_probe_cycle_v1",
                "cycle": cycle,
                "probe_config": probe_config.to_payload(),
                "model_state_dict": probe.state_dict(),
            },
            paths["probe"],
        )
        checkpoint_paths[cycle] = paths
        cycle_rows.append(
            {
                "cycle": cycle,
                **metrics,
                "probe_optimizer_steps": config.probe_steps_per_cycle,
                "projection_consumer_optimizer_steps": (
                    config.projection_steps_per_cycle
                ),
                "projection_checkpoint_sha256": sha256_file(
                    paths["projection"]
                ),
                "consumer_checkpoint_sha256": sha256_file(
                    paths["consumer"]
                ),
                "probe_checkpoint_sha256": sha256_file(paths["probe"]),
            }
        )

    selected = select_codesign_cycle(cycle_rows)
    selected_cycle = int(selected["cycle"])
    selected_paths = checkpoint_paths[selected_cycle]
    final_paths = {
        "projection": root / "selected_projection.pt",
        "consumer": root / "selected_consumer.pt",
        "probe": root / "selected_persistent_probe.pt",
    }
    for name, source in selected_paths.items():
        shutil.copyfile(source, final_paths[name])
    projection_payload = torch.load(
        final_paths["projection"],
        map_location="cpu",
        weights_only=False,
    )
    projection.load_state_dict(
        projection_payload["model_state_dict"], strict=True
    )
    projection.to(device).eval()
    train_provider.codesign_projection = projection
    stop_provider.codesign_projection = projection
    final_projection_sha256 = sha256_file(final_paths["projection"])
    ledger = build_codesign_ledger(
        config=config,
        rich_context_registration_sha256=rich_registration["content_hash"],
        provisional_head_registration_sha256=provisional_head[
            "content_hash"
        ],
        cycles=cycle_rows,
        selected_cycle=selected_cycle,
        final_projection_checkpoint_sha256=final_projection_sha256,
    )
    write_immutable_json(root / "codesign_ledger.json", ledger)
    write_immutable_json(
        root / "cycle_metrics.json",
        with_content_hash(
            {
                "contract": "particle_view_codesign_cycle_metrics_v1",
                "config": config.to_payload(),
                "cycles": cycle_rows,
                "selected_cycle": selected_cycle,
                "selected_metrics": selected,
                "performance_early_termination": False,
            }
        ),
    )
    codesigned_generator_path = root / "codesigned_generator.pt"
    torch.save(
        {
            "contract": "particle_view_codesigned_generator_v1",
            "generator_config": generator.config.to_payload(),
            "generator_state_dict": generator.state_dict(),
            "projection_config": config.to_payload(),
            "projection_state_dict": projection.state_dict(),
            "rich_context_registration_sha256": rich_registration[
                "content_hash"
            ],
            "codesign_ledger_sha256": ledger["content_hash"],
            "selected_cycle": selected_cycle,
        },
        codesigned_generator_path,
    )
    for paths in checkpoint_paths.values():
        for path in paths.values():
            path.unlink()
    cycle_root.rmdir()
    return {
        "projection": projection,
        "consumer": consumer,
        "persistent_probe": probe,
        "generator_checkpoint_path": codesigned_generator_path,
        "generator_checkpoint_sha256": sha256_file(
            codesigned_generator_path
        ),
        "ledger": ledger,
        "selected_cycle": selected_cycle,
        "artifact_paths": [
            root / "rich_context_registration.json",
            root / "provisional_head_registration.json",
            root / "selected_projection.pt",
            root / "selected_consumer.pt",
            root / "selected_persistent_probe.pt",
            root / "codesign_ledger.json",
            root / "cycle_metrics.json",
            codesigned_generator_path,
        ],
    }


def run_canonical_target_discovery(
    *,
    recipe: TargetScreenRecipe,
    query_teacher: FrozenContextualParticleTeacher,
    memory_teacher: FrozenContextualParticleTeacher,
    secondary_memory_teacher: FrozenContextualParticleTeacher | None,
    consumer_model: ParticleViewConsumer,
    probe_consumer_model: ParticleViewConsumer,
    codesign_consumer_model: ParticleViewConsumer | None,
    query_mixture: FrozenTokenLayerMixture | None,
    memory_mixture: FrozenTokenLayerMixture | None,
    teacher_mixture: TwoTeacherTokenMixture | None,
    train_aligned: AlignedLogicalJetView,
    stop_aligned: AlignedLogicalJetView,
    select_aligned: AlignedLogicalJetView,
    a0_registration: Mapping[str, Any],
    memory_registration: Mapping[str, Any],
    secondary_memory_registration: Mapping[str, Any] | None,
    query_tap_registration: Mapping[str, Any],
    memory_tap_registration: Mapping[str, Any],
    runtime_data_config: Mapping[str, Any],
    output_dir: str,
    device: str,
    num_workers: int,
    max_train_batches: int | None,
    max_val_batches: int | None,
    seed: int,
) -> None:
    """Run one declared discovery row plus its complete two-pass probe."""

    if recipe.run_id not in TARGET_SCREEN_IDS:
        raise ValueError("target discovery action received another screen row")
    validate_teacher_registration(a0_registration)
    validate_teacher_registration(memory_registration)
    if secondary_memory_registration is not None:
        validate_teacher_registration(secondary_memory_registration)
    if (secondary_memory_teacher is None) != (
        secondary_memory_registration is None
    ):
        raise ValueError("secondary teacher/registration must be paired")
    if (teacher_mixture is None) != (secondary_memory_teacher is None):
        raise ValueError("two-teacher adapter/secondary teacher must be paired")
    resolved_device = _resolve_device(device)
    source_manifest_sha256 = runtime_data_config["parent_manifest"][
        "manifest_sha256"
    ]
    train_memory = stage_contextual_memory(
        aligned=train_aligned,
        teacher=memory_teacher,
        teacher_checkpoint_sha256=memory_registration["checkpoint_sha256"],
        tap_spec_sha256=memory_tap_registration["tap_spec_sha256"],
        source_manifest_sha256=source_manifest_sha256,
        source_role=(
            "hlt_memory_control"
            if recipe.generator_config.memory_source == "hlt"
            else "offline_teacher"
        ),
        source_view=(
            "fixed_hlt"
            if recipe.generator_config.memory_source == "hlt"
            else "offline"
        ),
        device=resolved_device,
        num_workers=num_workers,
    )
    stop_memory = stage_contextual_memory(
        aligned=stop_aligned,
        teacher=memory_teacher,
        teacher_checkpoint_sha256=memory_registration["checkpoint_sha256"],
        tap_spec_sha256=memory_tap_registration["tap_spec_sha256"],
        source_manifest_sha256=source_manifest_sha256,
        source_role=(
            "hlt_memory_control"
            if recipe.generator_config.memory_source == "hlt"
            else "offline_teacher"
        ),
        source_view=(
            "fixed_hlt"
            if recipe.generator_config.memory_source == "hlt"
            else "offline"
        ),
        device=resolved_device,
        num_workers=num_workers,
    )
    train_secondary_memory = stop_secondary_memory = None
    if secondary_memory_teacher is not None:
        train_secondary_memory = stage_contextual_memory(
            aligned=train_aligned,
            teacher=secondary_memory_teacher,
            teacher_checkpoint_sha256=secondary_memory_registration[
                "checkpoint_sha256"
            ],
            tap_spec_sha256=memory_tap_registration[
                "secondary_tap_spec_sha256"
            ],
            source_manifest_sha256=source_manifest_sha256,
            source_role="offline_teacher_secondary",
            source_view="offline",
            device=resolved_device,
            num_workers=num_workers,
        )
        stop_secondary_memory = stage_contextual_memory(
            aligned=stop_aligned,
            teacher=secondary_memory_teacher,
            teacher_checkpoint_sha256=secondary_memory_registration[
                "checkpoint_sha256"
            ],
            tap_spec_sha256=memory_tap_registration[
                "secondary_tap_spec_sha256"
            ],
            source_manifest_sha256=source_manifest_sha256,
            source_role="offline_teacher_secondary",
            source_view="offline",
            device=resolved_device,
            num_workers=num_workers,
        )
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    recipe_payload = with_content_hash(
        {
            **recipe.to_payload(),
            "a0_registration_sha256": a0_registration["content_hash"],
            "memory_teacher_registration_sha256": memory_registration[
                "content_hash"
            ],
            "secondary_memory_teacher_registration_sha256": (
                None
                if secondary_memory_registration is None
                else secondary_memory_registration["content_hash"]
            ),
            "query_tap_registration_sha256": query_tap_registration[
                "content_hash"
            ],
            "memory_tap_registration_sha256": memory_tap_registration[
                "content_hash"
            ],
            "runtime_data_config_sha256": runtime_data_config["content_hash"],
        }
    )
    write_immutable_json(output / "target_discovery_recipe.json", recipe_payload)
    write_immutable_json(
        output / "query_tap_registration.json",
        query_tap_registration,
    )
    write_immutable_json(
        output / "memory_tap_registration.json",
        memory_tap_registration,
    )
    write_immutable_json(
        output / "train_staged_tap_manifest.json",
        train_memory.staged_tap.manifest,
    )
    write_immutable_json(
        output / "model_val_stop_staged_tap_manifest.json",
        stop_memory.staged_tap.manifest,
    )

    generator = MatchingFreeParticleViewGenerator(
        recipe.generator_config
    ).to(resolved_device)
    query_teacher = query_teacher.to(resolved_device).eval()
    train_provider = StagedDiscoveryViewProvider(
        generator=generator,
        query_teacher=query_teacher,
        staged_memory=train_memory,
        query_tap_choice=recipe.query_tap_choice,
        memory_source=recipe.generator_config.memory_source,
        query_mixture=query_mixture,
        memory_mixture=memory_mixture,
        memory_layer_width=(
            memory_registration["recipe"]["architecture"]["embed_dims"][-1]
            if memory_mixture is not None
            else None
        ),
        secondary_staged_memory=train_secondary_memory,
        teacher_mixture=teacher_mixture,
    )
    stop_provider = StagedDiscoveryViewProvider(
        generator=generator,
        query_teacher=query_teacher,
        staged_memory=stop_memory,
        query_tap_choice=recipe.query_tap_choice,
        memory_source=recipe.generator_config.memory_source,
        query_mixture=query_mixture,
        memory_mixture=memory_mixture,
        memory_layer_width=(
            memory_registration["recipe"]["architecture"]["embed_dims"][-1]
            if memory_mixture is not None
            else None
        ),
        secondary_staged_memory=stop_secondary_memory,
        teacher_mixture=teacher_mixture,
    )
    source_modules = {
        name: module
        for name, module in {
            "query_layer_mixture": query_mixture,
            "memory_layer_mixture": memory_mixture,
            "teacher_source_mixture": teacher_mixture,
        }.items()
        if module is not None
    }
    train_loader = _LimitedLoader(
        make_logical_data_loader(
            train_aligned,
            mode="aligned",
            batch_size=128,
            shuffle=True,
            num_workers=num_workers,
            seed=seed,
        ),
        max_train_batches,
    )
    stop_loader = _LimitedLoader(
        make_logical_data_loader(
            stop_aligned,
            mode="aligned",
            batch_size=128,
            shuffle=False,
            num_workers=num_workers,
            seed=seed + 1,
        ),
        max_val_batches,
    )
    _, train_split_sha256, train_identity_sha256 = logical_split_binding(
        load_hashed_json(runtime_data_config["unified_manifest"]["path"]),
        "train",
    )
    train_config = ParticleViewConsumerTrainConfig.for_role(
        "Cview_discovery", seed=seed
    )
    train_particle_view_consumer(
        model=consumer_model,
        train_loader=train_loader,
        model_val_stop_loader=stop_loader,
        config=train_config,
        output_dir=output,
        lineage={
            "a0_registration_sha256": a0_registration["content_hash"],
            "target_recipe_sha256": recipe_payload["content_hash"],
            "train_identity_sha256": train_identity_sha256,
            "model_val_stop_split_sha256": (
                stop_aligned.logical_split_sha256
            ),
        },
        view_provider=train_provider,
        validation_view_provider=stop_provider,
        oracle_config=recipe.oracle_objective,
        joint_trainable_modules={"Gview": generator, **source_modules},
        device=resolved_device,
    )
    generator_path = output / "generator_model_val_stop.pt"
    generator_sha256 = _generator_checkpoint(
        consumer_checkpoint_path=output / "best_model_val_stop.pt",
        generator=generator,
        source_modules=source_modules,
        recipe=recipe_payload,
        output_path=generator_path,
    )
    codesign_result = None
    if recipe.recoverability_codesign:
        if codesign_consumer_model is None:
            raise ValueError("co-design row omitted its fresh A0 consumer")
        codesign_result = run_recoverability_codesign(
            generator=generator,
            consumer=codesign_consumer_model,
            train_provider=train_provider,
            stop_provider=stop_provider,
            train_loader=train_loader,
            stop_loader=stop_loader,
            oracle_config=recipe.oracle_objective,
            provisional_generator_checkpoint_sha256=generator_sha256,
            provisional_consumer_registration_sha256=load_hashed_json(
                output / "consumer_registration.json"
            )["content_hash"],
            output_dir=output,
            seed=seed,
            device=resolved_device,
        )
        generator_sha256 = codesign_result[
            "generator_checkpoint_sha256"
        ]
    elif codesign_consumer_model is not None:
        raise ValueError("ordinary target received a co-design consumer")
    target = build_target_candidate_registration(
        target_id=recipe.run_id,
        campaign_id=PARTICLE_VIEW_LOW_DATA_CAMPAIGN_ID,
        selection_status=recipe.selection_status,
        seed=seed,
        generator_config=recipe.generator_config,
        source_manifest_sha256=source_manifest_sha256,
        unified_split_manifest_sha256=runtime_data_config["unified_manifest"][
            "manifest_sha256"
        ],
        train_split_sha256=train_split_sha256,
        train_identity_sha256=train_identity_sha256,
        query_tap_registration_sha256=query_tap_registration["content_hash"],
        query_checkpoint_sha256=a0_registration["checkpoint_sha256"],
        memory_tap_registration_sha256=memory_tap_registration["content_hash"],
        memory_checkpoint_sha256=memory_registration["checkpoint_sha256"],
        staged_tap_source_role=(
            "hlt_memory_control"
            if recipe.generator_config.memory_source == "hlt"
            else "offline_teacher"
        ),
        staged_tap_reservation_sha256=train_memory.staged_tap.manifest[
            "reservation_sha256"
        ],
        staged_tap_manifest_sha256=train_memory.staged_tap.manifest[
            "content_hash"
        ],
        staged_tap_logical_content_sha256=train_memory.staged_tap.manifest[
            "logical_content_sha256"
        ],
        generator_checkpoint_sha256=generator_sha256,
        offline_source_sha256=(
            None
            if recipe.generator_config.memory_source == "hlt"
            else memory_registration["source_sha256"]
        ),
        privileged_claim_eligible=(
            recipe.generator_config.memory_source == "offline"
        ),
        deployment_control_eligible=(
            recipe.run_id == "VGEN_MEMORY_HLT"
        ),
    )
    write_immutable_json(output / "target_candidate_registration.json", target)

    # Freeze the exact selected Gview, materialize raw candidate coordinates
    # only in process RAM, and fit the provisional candidate normalizer on
    # train.  No contextual tap or candidate-view payload is persisted.
    for parameter in generator.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    generator.eval()
    raw_train = materialize_candidate_views_in_ram(
        aligned=train_aligned,
        provider=train_provider,
        device=resolved_device,
        num_workers=num_workers,
    )
    raw_stop = materialize_candidate_views_in_ram(
        aligned=stop_aligned,
        provider=stop_provider,
        device=resolved_device,
        num_workers=num_workers,
    )
    normalizer = fit_particle_view_normalizer(
        raw_train.views,
        raw_train.mask,
        train_split_sha256=train_split_sha256,
        generator_checkpoint_sha256=generator_sha256,
    )
    normalizer_artifact = write_particle_view_normalizer(
        output / "provisional_normalizer.json",
        normalizer,
    )
    normalized_train = raw_train.normalized(normalizer)
    normalized_stop = raw_stop.normalized(normalizer)
    train_stage_manifest_sha256 = train_memory.staged_tap.manifest[
        "content_hash"
    ]
    stop_stage_manifest_sha256 = stop_memory.staged_tap.manifest[
        "content_hash"
    ]
    del raw_train, raw_stop, train_provider, stop_provider
    del train_memory, stop_memory
    del train_secondary_memory, stop_secondary_memory
    if resolved_device.type == "cuda":
        torch.cuda.empty_cache()

    # The probe consumer is a fresh copy of A0.  It never inherits discovery
    # consumer weights, and the normalizer is present from epoch zero.
    probe_root = output / "probe_consumer"
    probe_train_loader = _LimitedLoader(
        make_logical_data_loader(
            train_aligned,
            mode="aligned",
            batch_size=128,
            shuffle=True,
            num_workers=num_workers,
            seed=seed,
        ),
        max_train_batches,
    )
    probe_stop_loader = _LimitedLoader(
        make_logical_data_loader(
            stop_aligned,
            mode="aligned",
            batch_size=128,
            shuffle=False,
            num_workers=num_workers,
            seed=seed + 1,
        ),
        max_val_batches,
    )
    probe_consumer_registration = train_particle_view_consumer(
        model=probe_consumer_model,
        train_loader=probe_train_loader,
        model_val_stop_loader=probe_stop_loader,
        config=ParticleViewConsumerTrainConfig.for_role(
            "Cview_probe", seed=seed
        ),
        output_dir=probe_root,
        lineage={
            "a0_registration_sha256": a0_registration["content_hash"],
            "target_registration_sha256": target["content_hash"],
            "train_identity_sha256": train_identity_sha256,
            "model_val_stop_split_sha256": (
                stop_aligned.logical_split_sha256
            ),
            "normalizer_sha256": normalizer_artifact["content_hash"],
        },
        view_provider=IndexedCandidateViewProvider(normalized_train),
        validation_view_provider=IndexedCandidateViewProvider(
            normalized_stop
        ),
        device=resolved_device,
    )
    probe_consumer_model = _load_selected_probe_consumer(
        probe_consumer_model,
        probe_root / "best_model_val_stop.pt",
        expected_role="Cview_probe",
        device=resolved_device,
    )

    # The fixed-capacity recovery probe sees HLT tensors and normalized views
    # only.  The recovery_probe loader intentionally contains no labels.
    recovery_root = output / "recovery_probe"
    recovery_config = RecoveryProbeConfig(
        view_dim=recipe.generator_config.bottleneck_width,
        seed=seed,
    )
    recovery_train_loader = _LimitedLoader(
        make_logical_data_loader(
            train_aligned,
            mode="recovery_probe",
            true_views=normalized_train.numpy_views(),
            batch_size=128,
            shuffle=True,
            num_workers=num_workers,
            seed=seed,
        ),
        max_train_batches,
    )
    recovery_stop_loader = _LimitedLoader(
        make_logical_data_loader(
            stop_aligned,
            mode="recovery_probe",
            true_views=normalized_stop.numpy_views(),
            batch_size=128,
            shuffle=False,
            num_workers=num_workers,
            seed=seed + 1,
        ),
        max_val_batches,
    )
    recovery_registration = train_recovery_probe(
        config=recovery_config,
        train_loader=recovery_train_loader,
        model_val_stop_loader=recovery_stop_loader,
        output_dir=recovery_root,
        target_registration_sha256=target["content_hash"],
        normalizer_sha256=normalizer_artifact["content_hash"],
        train_identity_sha256=train_identity_sha256,
        model_val_stop_split_sha256=stop_aligned.logical_split_sha256,
        hlt_preprocessing_sha256=a0_registration[
            "preprocessing_sha256"
        ],
        device=resolved_device,
    )
    recovery_model = _load_selected_recovery_probe(
        recovery_root / "best_model_val_stop.pt",
        config=recovery_config,
        device=resolved_device,
    )

    # model_val_select is opened only after every train/stop checkpoint is
    # fixed.  It is used once for candidate ranking and never for fitting.
    select_memory = stage_contextual_memory(
        aligned=select_aligned,
        teacher=memory_teacher,
        teacher_checkpoint_sha256=memory_registration["checkpoint_sha256"],
        tap_spec_sha256=memory_tap_registration["tap_spec_sha256"],
        source_manifest_sha256=source_manifest_sha256,
        source_role=(
            "hlt_memory_control"
            if recipe.generator_config.memory_source == "hlt"
            else "offline_teacher"
        ),
        source_view=(
            "fixed_hlt"
            if recipe.generator_config.memory_source == "hlt"
            else "offline"
        ),
        device=resolved_device,
        num_workers=num_workers,
    )
    select_secondary_memory = None
    if secondary_memory_teacher is not None:
        select_secondary_memory = stage_contextual_memory(
            aligned=select_aligned,
            teacher=secondary_memory_teacher,
            teacher_checkpoint_sha256=secondary_memory_registration[
                "checkpoint_sha256"
            ],
            tap_spec_sha256=memory_tap_registration[
                "secondary_tap_spec_sha256"
            ],
            source_manifest_sha256=source_manifest_sha256,
            source_role="offline_teacher_secondary",
            source_view="offline",
            device=resolved_device,
            num_workers=num_workers,
        )
    write_immutable_json(
        output / "model_val_select_staged_tap_manifest.json",
        select_memory.staged_tap.manifest,
    )
    select_provider = StagedDiscoveryViewProvider(
        generator=generator,
        query_teacher=query_teacher,
        staged_memory=select_memory,
        query_tap_choice=recipe.query_tap_choice,
        memory_source=recipe.generator_config.memory_source,
        query_mixture=query_mixture,
        memory_mixture=memory_mixture,
        memory_layer_width=(
            memory_registration["recipe"]["architecture"]["embed_dims"][-1]
            if memory_mixture is not None
            else None
        ),
        secondary_staged_memory=select_secondary_memory,
        teacher_mixture=teacher_mixture,
        codesign_projection=(
            None
            if codesign_result is None
            else codesign_result["projection"]
        ),
    )
    raw_select = materialize_candidate_views_in_ram(
        aligned=select_aligned,
        provider=select_provider,
        device=resolved_device,
        num_workers=num_workers,
    )
    normalized_select = raw_select.normalized(normalizer)
    quantization = with_content_hash(
        {
            "contract": "particle_view_candidate_quantization_diagnostics_v1",
            "target_registration_sha256": target["content_hash"],
            "normalizer_sha256": normalizer_artifact["content_hash"],
            "split": "model_val_select",
            "split_sha256": select_aligned.logical_split_sha256,
            "four_bit": quantized_view_diagnostics(
                normalized_select.views,
                normalized_select.mask,
                normalizer,
                bits=4,
            ),
            "eight_bit": quantized_view_diagnostics(
                normalized_select.views,
                normalized_select.mask,
                normalizer,
                bits=8,
            ),
            "diagnostic_only": True,
        }
    )
    write_immutable_json(
        output / "candidate_quantization_diagnostics.json",
        quantization,
    )
    predicted_select = predict_recovery_probe_views(
        model=recovery_model,
        aligned=select_aligned,
        true_views=normalized_select,
        device=resolved_device,
        num_workers=num_workers,
    )
    ranking_loader = _LimitedLoader(
        make_logical_data_loader(
            select_aligned,
            mode="aligned",
            batch_size=128,
            shuffle=False,
            num_workers=num_workers,
            seed=seed + 2,
        ),
        max_val_batches,
    )
    counterfactual_loader = _CounterfactualLoader(
        ranking_loader,
        true_views=normalized_select,
        predicted_views=predicted_select,
    )
    counterfactual_metrics = evaluate_view_counterfactuals(
        probe_consumer_model,
        counterfactual_loader,
        device=resolved_device,
        split="model_val_select",
    )
    write_immutable_json(
        output / "model_val_select_counterfactual_metrics.json",
        counterfactual_metrics,
    )
    a0_accuracy = evaluate_independent_a0_accuracy(
        query_teacher=query_teacher,
        aligned=select_aligned,
        device=resolved_device,
        num_workers=num_workers,
        maximum_batches=max_val_batches,
    )
    target_metrics = build_target_metrics_from_counterfactual(
        counterfactual_metrics=counterfactual_metrics,
        run_id=recipe.run_id,
        target_id=recipe.run_id,
        bottleneck_width=recipe.generator_config.bottleneck_width,
        a0_accuracy=a0_accuracy,
        target_registration_sha256=target["content_hash"],
        selection_status=recipe.selection_status,
    )
    write_immutable_json(
        output / "target_candidate_metrics.json",
        target_metrics,
    )
    two_pass = build_two_pass_candidate_artifact(
        target_registration_sha256=target["content_hash"],
        discovery_consumer_checkpoint_sha256=sha256_file(
            output / "best_model_val_stop.pt"
        ),
        frozen_generator_checkpoint_sha256=generator_sha256,
        provisional_normalizer_sha256=normalizer_artifact["content_hash"],
        probe_consumer_checkpoint_sha256=probe_consumer_registration[
            "checkpoint_sha256"
        ],
        recovery_probe_registration_sha256=recovery_registration[
            "content_hash"
        ],
        model_val_select_metrics_sha256=counterfactual_metrics[
            "content_hash"
        ],
    )
    write_immutable_json(output / "two_pass_candidate.json", two_pass)
    two_pass_result = with_content_hash(
        {
            "contract": PARTICLE_VIEW_TARGET_TWO_PASS_RESULT_CONTRACT,
            "run_id": recipe.run_id,
            "seed": seed,
            "target_candidate_registration_sha256": target["content_hash"],
            "normalizer_sha256": normalizer_artifact["content_hash"],
            "probe_consumer_registration_sha256": (
                probe_consumer_registration["content_hash"]
            ),
            "recovery_probe_registration_sha256": recovery_registration[
                "content_hash"
            ],
            "counterfactual_metrics_sha256": counterfactual_metrics[
                "content_hash"
            ],
            "target_metrics_sha256": target_metrics["content_hash"],
            "two_pass_candidate_sha256": two_pass["content_hash"],
            "quantization_diagnostics_sha256": quantization["content_hash"],
            "codesign_ledger_sha256": (
                None
                if codesign_result is None
                else codesign_result["ledger"]["content_hash"]
            ),
            "model_val_select_staged_tap_manifest_sha256": (
                select_memory.staged_tap.manifest["content_hash"]
            ),
            "view_payloads_persisted": False,
            "model_val_select_used_for_fitting": False,
            "quality_gate_used": False,
            "recoverability_codesign_completed": (
                codesign_result is not None
            ),
        }
    )
    write_immutable_json(
        output / "target_two_pass_result.json",
        two_pass_result,
    )

    discovery_consumer_registration = load_hashed_json(
        output / "consumer_registration.json"
    )
    result = with_content_hash(
        {
            "contract": PARTICLE_VIEW_TARGET_DISCOVERY_RESULT_CONTRACT,
            "run_id": recipe.run_id,
            "seed": seed,
            "target_discovery_recipe_sha256": recipe_payload["content_hash"],
            "target_candidate_registration_sha256": target["content_hash"],
            "consumer_registration_sha256": discovery_consumer_registration[
                "content_hash"
            ],
            "generator_checkpoint_sha256": generator_sha256,
            "train_staged_tap_manifest_sha256": train_stage_manifest_sha256,
            "model_val_stop_staged_tap_manifest_sha256": (
                stop_stage_manifest_sha256
            ),
            "raw_discovery_coordinate_only": True,
            "two_pass_probe_pending": False,
            "two_pass_probe_completed": True,
            "target_two_pass_result_sha256": two_pass_result["content_hash"],
            "quality_gate_used": False,
        }
    )
    write_immutable_json(output / "target_discovery_result.json", result)
    del raw_select, normalized_select, predicted_select
    del select_provider, select_memory, select_secondary_memory
    del normalized_train, normalized_stop
    if resolved_device.type == "cuda":
        torch.cuda.empty_cache()


def run_target_discovery_operation(
    *,
    unavailable: Mapping[str, Any] | None = None,
    **kwargs,
) -> None:
    """Locked target operation with a non-gating unavailable-source branch."""

    if unavailable is None:
        run_canonical_target_discovery(**kwargs)
        return
    if kwargs:
        raise ValueError("unavailable target cannot receive training kwargs")
    fields = dict(unavailable)
    output_path = fields.pop("output_path")
    artifact = with_content_hash(
        {
            "contract": PARTICLE_VIEW_TARGET_UNAVAILABLE_CONTRACT,
            **fields,
            "metrics_available": False,
            "selection_status": "diagnostic_unavailable",
            "quality_gate_used": False,
            "stops_execution": False,
        }
    )
    write_immutable_json(output_path, artifact)


def build_target_discovery_factory_config(
    *,
    runtime_data_config: Mapping[str, Any],
    device: str = "auto",
    num_workers: int = 0,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    existing_teacher_compatible: bool = False,
    teacher_mix_compatible: bool = False,
    baseline_factory_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind every target-discovery workload to exact source caches."""

    validate_runtime_data_config(runtime_data_config, verify_cache_files=True)
    if not isinstance(device, str) or not device:
        raise ValueError("device must be nonempty")
    workers = int(num_workers)
    if workers < 0:
        raise ValueError("num_workers must be nonnegative")
    existing_teacher = None
    baseline_factory_config_sha256 = None
    if baseline_factory_config is not None:
        validate_content_hash(
            baseline_factory_config,
            expected_contract="particle_view_baseline_factory_config_v1",
        )
        if (
            baseline_factory_config.get("runtime_data_config_sha256")
            != runtime_data_config["content_hash"]
        ):
            raise ValueError(
                "baseline/target factories bind different runtime data"
            )
        existing_teacher = baseline_factory_config.get("existing_teacher")
        if (
            existing_teacher is not None
            and not isinstance(
                existing_teacher.get("serialized_recipe"), Mapping
            )
        ):
            existing_teacher = None
        baseline_factory_config_sha256 = baseline_factory_config[
            "content_hash"
        ]
    if existing_teacher_compatible and existing_teacher is None:
        raise ValueError(
            "compatible existing teacher requires baseline factory evidence"
        )
    if existing_teacher_compatible and (
        existing_teacher.get("recipe_reproduced_exactly") is not True
        or existing_teacher.get("observed_train_identity_sha256")
        != logical_split_binding(
            load_hashed_json(
                runtime_data_config["unified_manifest"]["path"]
            ),
            "train",
        )[2]
    ):
        # The exact identity comparison is performed again by the Stage-A
        # source registration.  Do not allow a CLI flag alone to promote it.
        raise ValueError(
            "existing teacher compatibility lacks exact provenance evidence"
        )
    existing_memory_dim = (
        128
        if existing_teacher is None
        else int(
            existing_teacher["serialized_recipe"]["architecture"][
                "embed_dims"
            ][-1]
        )
    )
    artifact = with_content_hash(
        {
            "contract": (
                PARTICLE_VIEW_TARGET_DISCOVERY_FACTORY_CONFIG_CONTRACT
            ),
            "runtime_data_config": dict(runtime_data_config),
            "runtime_data_config_sha256": runtime_data_config["content_hash"],
            "runtime": {
                "device": device,
                "num_workers": workers,
                "max_train_batches": _optional_positive(
                    max_train_batches, name="max_train_batches"
                ),
                "max_val_batches": _optional_positive(
                    max_val_batches, name="max_val_batches"
                ),
            },
            "supported_run_ids": list(TARGET_SCREEN_IDS),
            "complete_target_screen_count": len(TARGET_SCREEN_IDS),
            "screen_recipes": {
                run_id: build_target_screen_recipe(
                    run_id,
                    memory_dim=(
                        existing_memory_dim
                        if run_id == "VGEN_TEACHER_EXISTING"
                        else 128
                    ),
                    existing_teacher_compatible=existing_teacher_compatible,
                    teacher_mix_compatible=teacher_mix_compatible,
                ).to_payload()
                for run_id in TARGET_SCREEN_IDS
            },
            "compatibility": {
                "existing_teacher_compatible": bool(
                    existing_teacher_compatible
                ),
                "teacher_mix_compatible": bool(teacher_mix_compatible),
            },
            "existing_teacher": existing_teacher,
            "baseline_factory_config_sha256": (
                baseline_factory_config_sha256
            ),
            "runtime_availability": {
                run_id: (
                    "unavailable_non_gating"
                    if (
                        run_id == "VGEN_TEACHER_EXISTING"
                        and existing_teacher is None
                    )
                    else "complete_two_pass"
                )
                for run_id in TARGET_SCREEN_IDS
            },
            "production_scope": "complete_target_screen_two_pass_v1",
            "two_pass_probe_required_before_target_ranking": True,
            "quality_warnings_non_gating": True,
        }
    )
    validate_target_discovery_factory_config(artifact)
    return artifact


def validate_target_discovery_factory_config(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(
        payload,
        expected_contract=(
            PARTICLE_VIEW_TARGET_DISCOVERY_FACTORY_CONFIG_CONTRACT
        ),
    )
    expected = {
        "contract",
        "runtime_data_config",
        "runtime_data_config_sha256",
        "runtime",
        "supported_run_ids",
        "complete_target_screen_count",
        "screen_recipes",
        "compatibility",
        "existing_teacher",
        "baseline_factory_config_sha256",
        "runtime_availability",
        "production_scope",
        "two_pass_probe_required_before_target_ranking",
        "quality_warnings_non_gating",
        "content_hash",
    }
    if set(payload) != expected:
        raise ValueError("target-discovery factory field inventory mismatch")
    data = payload["runtime_data_config"]
    if data.get("content_hash") != payload["runtime_data_config_sha256"]:
        raise ValueError("target-discovery runtime-data hash mismatch")
    validate_runtime_data_config(data, verify_cache_files=False)
    runtime = payload["runtime"]
    if set(runtime) != {
        "device",
        "num_workers",
        "max_train_batches",
        "max_val_batches",
    }:
        raise ValueError("target-discovery runtime inventory mismatch")
    if (
        not isinstance(runtime["device"], str)
        or not runtime["device"]
        or not isinstance(runtime["num_workers"], int)
        or isinstance(runtime["num_workers"], bool)
        or runtime["num_workers"] < 0
    ):
        raise ValueError("target-discovery runtime settings are invalid")
    for name in ("max_train_batches", "max_val_batches"):
        _optional_positive(runtime[name], name=name)
    if (
        payload["supported_run_ids"] != list(TARGET_SCREEN_IDS)
        or payload["complete_target_screen_count"] != len(TARGET_SCREEN_IDS)
        or set(payload["screen_recipes"]) != set(TARGET_SCREEN_IDS)
        or payload["production_scope"]
        != "complete_target_screen_two_pass_v1"
        or payload["two_pass_probe_required_before_target_ranking"] is not True
        or payload["quality_warnings_non_gating"] is not True
    ):
        raise ValueError("target-discovery scope/policy changed")
    compatibility = payload["compatibility"]
    if (
        set(compatibility)
        != {"existing_teacher_compatible", "teacher_mix_compatible"}
        or any(
            not isinstance(value, bool)
            for value in compatibility.values()
        )
    ):
        raise ValueError("target-discovery compatibility inventory changed")
    existing = payload["existing_teacher"]
    baseline_hash = payload["baseline_factory_config_sha256"]
    if baseline_hash is not None:
        if (
            not isinstance(baseline_hash, str)
            or len(baseline_hash) != 64
            or any(character not in "0123456789abcdef" for character in baseline_hash)
        ):
            raise ValueError("baseline factory hash is invalid")
        if existing is not None and not isinstance(existing, Mapping):
            raise ValueError("existing-teacher evidence must be an object")
    elif existing is not None:
        raise ValueError("existing-teacher evidence lacks baseline binding")
    if compatibility["existing_teacher_compatible"] and existing is None:
        raise ValueError("selectable existing teacher has no evidence")
    expected_availability = {
        run_id: (
            "unavailable_non_gating"
            if (run_id == "VGEN_TEACHER_EXISTING" and existing is None)
            else "complete_two_pass"
        )
        for run_id in TARGET_SCREEN_IDS
    }
    if payload["runtime_availability"] != expected_availability:
        raise ValueError("target runtime availability policy changed")
    for run_id in TARGET_SCREEN_IDS:
        existing_memory_dim = (
            128
            if existing is None
            else int(
                existing["serialized_recipe"]["architecture"][
                    "embed_dims"
                ][-1]
            )
        )
        rebuilt = build_target_screen_recipe(
            run_id,
            memory_dim=(
                existing_memory_dim
                if run_id == "VGEN_TEACHER_EXISTING"
                else 128
            ),
            existing_teacher_compatible=compatibility[
                "existing_teacher_compatible"
            ],
            teacher_mix_compatible=compatibility[
                "teacher_mix_compatible"
            ],
        ).to_payload()
        if payload["screen_recipes"][run_id] != rebuilt:
            raise ValueError(f"target-screen recipe changed for {run_id}")
    return {
        "ok": True,
        "content_hash": payload["content_hash"],
        "supported_run_count": len(TARGET_SCREEN_IDS),
        "compiled_screen_count": len(TARGET_SCREEN_IDS),
    }


def _parent_artifact(
    parents: Mapping[str, Any],
    parent_id: str,
    artifact_name: str,
) -> Path:
    try:
        binding = parents[parent_id]["artifacts"][artifact_name]
    except KeyError as exc:
        raise ValueError(
            f"parent {parent_id} omitted {artifact_name}"
        ) from exc
    path = Path(binding["path"]).resolve()
    if not path.is_file() or sha256_file(path) != binding["sha256"]:
        raise ValueError(f"parent artifact changed: {parent_id}/{artifact_name}")
    return path


def _target_parent_ids(run_id: str) -> tuple[str, ...]:
    if run_id == "VGEN_TEACHER_LARGE":
        return ("A0_VIEW", "TOFF_VIEW_LARGE")
    if run_id == "VGEN_TEACHER_EXISTING":
        return ("A0_VIEW", "TOFF_VIEW_EXISTING")
    if run_id == "VGEN_TEACHER_MIX2":
        return ("A0_VIEW", "TOFF_VIEW_BASE", "TOFF_VIEW_LARGE")
    if run_id in {"VGEN_MEMORY_HLT", "VGEN_MEMORY_HLT_SELFMASK"}:
        return ("A0_VIEW",)
    return ("A0_VIEW", "TOFF_VIEW_BASE")


def _registered_parent_teacher(
    parents: Mapping[str, Any],
    parent_id: str,
):
    registration_path = _parent_artifact(
        parents, parent_id, "teacher_registration.json"
    )
    checkpoint_path = _parent_artifact(
        parents, parent_id, "best_model_val_stop.pt"
    )
    registration = load_hashed_json(registration_path)
    validate_teacher_registration(registration)
    model = reload_registered_teacher(
        registration=registration,
        checkpoint_path=checkpoint_path,
    )
    return registration, checkpoint_path, model


def build_target_discovery_factory(
    *,
    operation: str,
    config: Mapping[str, Any],
    registry: Mapping[str, Any],
    run_id: str,
    seed: int,
    task_id: str,
    output_dir: str,
) -> dict[str, Any]:
    """Prepare the canonical cache-backed target-discovery task."""

    if operation != "target_discovery":
        raise ValueError("target-discovery factory received another operation")
    validate_target_discovery_factory_config(config)
    validate_particle_view_registry(registry)
    rows = {row["run_id"]: row for row in registry["runs"]}
    if (
        run_id not in TARGET_SCREEN_IDS
        or run_id not in rows
        or int(seed) not in rows[run_id]["seed_ids"]
        or not str(rows[run_id]["scientific_role"]).startswith(
            "target_generator:"
        )
        or task_id != f"{run_id}__seed_{int(seed)}"
    ):
        raise ValueError("target-discovery run identity is invalid")
    root = Path(output_dir).resolve().parent.parent
    parents = resolve_parent_task_artifacts(
        registry=registry,
        artifact_root=root,
        run_id=run_id,
        seed=int(seed),
    )
    expected_parents = _target_parent_ids(run_id)
    if set(parents) != set(expected_parents):
        raise ValueError("target discovery parent inventory differs")
    if run_id == "VGEN_TEACHER_EXISTING" and config["existing_teacher"] is None:
        source_path = _parent_artifact(
            parents,
            "TOFF_VIEW_EXISTING",
            "existing_teacher_source_registration.json",
        )
        source = load_hashed_json(source_path)
        if source.get("selectable") is not False:
            raise ValueError(
                "unbound existing teacher unexpectedly claims selectability"
            )
        output = Path(output_dir).resolve()
        return {
            "kwargs": {
                "unavailable": {
                    "run_id": run_id,
                    "seed": int(seed),
                    "reason": (
                        "existing_teacher_not_configured_or_recipe_unbound"
                    ),
                    "source_registration_sha256": source["content_hash"],
                    "output_path": str(output / "target_unavailable.json"),
                }
            },
            "artifact_paths": [
                str(output / "target_unavailable.json"),
            ],
            "action": None,
        }
    a0_registration, a0_checkpoint_path, a0_query_model = (
        _registered_parent_teacher(parents, "A0_VIEW")
    )
    a0_consumer_model = reload_registered_teacher(
        registration=a0_registration, checkpoint_path=a0_checkpoint_path
    )
    a0_probe_consumer_model = reload_registered_teacher(
        registration=a0_registration, checkpoint_path=a0_checkpoint_path
    )
    a0_codesign_consumer_model = (
        reload_registered_teacher(
            registration=a0_registration,
            checkpoint_path=a0_checkpoint_path,
        )
        if run_id == "VGEN_RECODESIGN"
        else None
    )
    for parameter in a0_consumer_model.parameters():
        parameter.requires_grad_(True)
    for parameter in a0_probe_consumer_model.parameters():
        parameter.requires_grad_(True)
    if a0_codesign_consumer_model is not None:
        for parameter in a0_codesign_consumer_model.parameters():
            parameter.requires_grad_(True)
    compatibility = config["compatibility"]
    recipe = build_target_screen_recipe(
        run_id,
        memory_dim=(
            int(
                config["existing_teacher"]["serialized_recipe"][
                    "architecture"
                ]["embed_dims"][-1]
            )
            if (
                run_id == "VGEN_TEACHER_EXISTING"
                and config["existing_teacher"] is not None
            )
            else 128
        ),
        existing_teacher_compatible=compatibility[
            "existing_teacher_compatible"
        ],
        teacher_mix_compatible=compatibility[
            "teacher_mix_compatible"
        ],
    )
    existing_teacher = config["existing_teacher"]
    if run_id == "VGEN_TEACHER_EXISTING":
        source_registration_path = _parent_artifact(
            parents,
            "TOFF_VIEW_EXISTING",
            "existing_teacher_source_registration.json",
        )
        source_registration = load_hashed_json(source_registration_path)
        if existing_teacher is None:
            raise ValueError(
                "existing teacher is unavailable; target row must use the "
                "non-gating unavailable-target runtime"
            )
        checkpoint_path = Path(
            existing_teacher["checkpoint"]["path"]
        ).resolve()
        if (
            not checkpoint_path.is_file()
            or sha256_file(checkpoint_path)
            != existing_teacher["checkpoint"]["sha256"]
            or source_registration.get("checkpoint_sha256")
            != existing_teacher["checkpoint"]["sha256"]
        ):
            raise ValueError("existing teacher checkpoint/source binding changed")
        serialized = existing_teacher["serialized_recipe"]
        if not isinstance(serialized, Mapping):
            raise ValueError("existing teacher omitted its serialized recipe")
        existing_recipe = ParticleViewTeacherRecipe(
            role=serialized["role"],
            particle_source=serialized["particle_source"],
            architecture=serialized["architecture_name"],
            seed=int(serialized["seed"]),
            unified_split_manifest_sha256=serialized[
                "unified_split_manifest_sha256"
            ],
            train_identity_sha256=serialized["train_identity_sha256"],
            train_split_sha256=serialized["train_split_sha256"],
            model_val_stop_split_sha256=serialized[
                "model_val_stop_split_sha256"
            ],
            preprocessing_sha256=serialized["preprocessing_sha256"],
            source_sha256=serialized["source_sha256"],
            initialization_implementation_sha256=serialized[
                "initialization_implementation_sha256"
            ],
            library_versions_sha256=serialized[
                "library_versions_sha256"
            ],
        )
        if existing_recipe.to_payload() != dict(serialized):
            raise ValueError("existing teacher recipe is noncanonical")
        memory_registration = build_teacher_registration(
            recipe=existing_recipe,
            checkpoint_path=checkpoint_path,
            selected_checkpoint_kind=(
                "existing_provenance_compatible"
                if recipe.selection_status == "selectable"
                else "existing_diagnostic"
            ),
            selectable=recipe.selection_status == "selectable",
            provenance_reason=source_registration["selection_status"],
        )
        memory_checkpoint_path = checkpoint_path
        memory_model = reload_registered_teacher(
            registration=memory_registration,
            checkpoint_path=memory_checkpoint_path,
        )
    else:
        memory_registration = None
        memory_model = None
        memory_checkpoint_path = None
    memory_parent = (
        "A0_VIEW"
        if recipe.generator_config.memory_source == "hlt"
        else "TOFF_VIEW_LARGE"
        if recipe.teacher_source == "large"
        else "TOFF_VIEW_BASE"
    )
    if run_id == "VGEN_TEACHER_EXISTING":
        pass
    elif memory_parent == "A0_VIEW":
        memory_registration = a0_registration
        memory_checkpoint_path = a0_checkpoint_path
        memory_model = reload_registered_teacher(
            registration=a0_registration,
            checkpoint_path=a0_checkpoint_path,
        )
    else:
        (
            memory_registration,
            memory_checkpoint_path,
            memory_model,
        ) = _registered_parent_teacher(parents, memory_parent)
    secondary_memory_registration = None
    secondary_memory_model = None
    if recipe.teacher_source == "base_large_mix":
        (
            secondary_memory_registration,
            _,
            secondary_memory_model,
        ) = _registered_parent_teacher(parents, "TOFF_VIEW_LARGE")
    query_context_tap = (
        "raw_embed"
        if recipe.query_tap_choice == "raw_features"
        else recipe.query_tap_choice
    )
    query_spec = ParticleTokenTapSpec(
        particle_source="fixed_hlt",
        architecture=a0_registration["recipe"]["architecture_name"],
        tap_choice=query_context_tap,
    )
    memory_spec = ParticleTokenTapSpec(
        particle_source=(
            "fixed_hlt"
            if recipe.generator_config.memory_source == "hlt"
            else "offline"
        ),
        architecture=memory_registration["recipe"]["architecture_name"],
        tap_choice=recipe.memory_tap_choice,
    )
    query_tap_registration = build_token_tap_registration(
        teacher_registration=a0_registration,
        tap_spec=query_spec,
        input_normalization_sha256=a0_registration[
            "preprocessing_sha256"
        ],
    )
    memory_tap_registration = build_token_tap_registration(
        teacher_registration=memory_registration,
        tap_spec=memory_spec,
        input_normalization_sha256=memory_registration[
            "preprocessing_sha256"
        ],
    )
    if secondary_memory_registration is not None:
        secondary_spec = ParticleTokenTapSpec(
            particle_source="offline",
            architecture=secondary_memory_registration["recipe"][
                "architecture_name"
            ],
            tap_choice=recipe.memory_tap_choice,
        )
        secondary_tap = build_token_tap_registration(
            teacher_registration=secondary_memory_registration,
            tap_spec=secondary_spec,
            input_normalization_sha256=secondary_memory_registration[
                "preprocessing_sha256"
            ],
        )
        memory_tap_registration = with_content_hash(
            {
                "contract": "particle_view_two_teacher_tap_binding_v1",
                "primary_tap": memory_tap_registration,
                "tap_spec_sha256": memory_spec.content_hash,
                "primary_tap_spec_sha256": memory_spec.content_hash,
                "secondary_tap": secondary_tap,
                "secondary_tap_spec_sha256": secondary_spec.content_hash,
                "teacher_order": ["base", "large"],
            }
        )
    elif recipe.generator_config.memory_source == "hlt":
        # The HLT-memory controls must bind the exact same A0 tap hash on
        # both query and memory sides.
        memory_tap_registration = query_tap_registration
    data = config["runtime_data_config"]
    train = load_aligned_logical_jet_view(data, "train")
    stop = load_aligned_logical_jet_view(data, "model_val_stop")
    select = load_aligned_logical_jet_view(data, "model_val_select")
    hidden = int(a0_registration["recipe"]["architecture"]["embed_dims"][-1])
    heads = int(a0_registration["recipe"]["architecture"]["num_heads"])
    consumer = ParticleViewConsumer(
        a0_consumer_model,
        ParticleViewConsumerConfig(
            view_dim=recipe.generator_config.bottleneck_width,
            hidden_dim=hidden,
            num_heads=heads,
            injection_block=0,
            view_path="token_and_pair",
            learned_trust=True,
        ),
    )
    probe_consumer = ParticleViewConsumer(
        a0_probe_consumer_model,
        ParticleViewConsumerConfig(
            view_dim=recipe.generator_config.bottleneck_width,
            hidden_dim=hidden,
            num_heads=heads,
            injection_block=0,
            view_path="token_and_pair",
            learned_trust=True,
        ),
    )
    codesign_consumer = (
        ParticleViewConsumer(
            a0_codesign_consumer_model,
            ParticleViewConsumerConfig(
                view_dim=recipe.generator_config.bottleneck_width,
                hidden_dim=hidden,
                num_heads=heads,
                injection_block=0,
                view_path="token_and_pair",
                learned_trust=True,
            ),
        )
        if a0_codesign_consumer_model is not None
        else None
    )
    query_mixture = (
        FrozenTokenLayerMixture()
        if recipe.query_tap_choice == "mix_last3"
        else None
    )
    memory_mixture = (
        FrozenTokenLayerMixture()
        if (
            recipe.memory_tap_choice == "mix_last3"
            and recipe.teacher_source != "base_large_mix"
        )
        else None
    )
    teacher_mixture = (
        TwoTeacherTokenMixture()
        if recipe.teacher_source == "base_large_mix"
        else None
    )
    output = Path(output_dir).resolve()
    runtime = config["runtime"]
    return {
        "kwargs": {
            "recipe": recipe,
            "query_teacher": FrozenContextualParticleTeacher(
                a0_query_model, query_spec
            ),
            "memory_teacher": FrozenContextualParticleTeacher(
                memory_model, memory_spec
            ),
            "secondary_memory_teacher": (
                None
                if secondary_memory_model is None
                else FrozenContextualParticleTeacher(
                    secondary_memory_model,
                    secondary_spec,
                )
            ),
            "consumer_model": consumer,
            "probe_consumer_model": probe_consumer,
            "codesign_consumer_model": codesign_consumer,
            "query_mixture": query_mixture,
            "memory_mixture": memory_mixture,
            "teacher_mixture": teacher_mixture,
            "train_aligned": train,
            "stop_aligned": stop,
            "select_aligned": select,
            "a0_registration": a0_registration,
            "memory_registration": memory_registration,
            "secondary_memory_registration": (
                secondary_memory_registration
            ),
            "query_tap_registration": query_tap_registration,
            "memory_tap_registration": memory_tap_registration,
            "runtime_data_config": data,
            "output_dir": str(output),
            "device": runtime["device"],
            "num_workers": runtime["num_workers"],
            "max_train_batches": runtime["max_train_batches"],
            "max_val_batches": runtime["max_val_batches"],
            "seed": int(seed),
        },
        "artifact_paths": [
            str(output / "best_model_val_stop.pt"),
            str(output / "consumer_registration.json"),
            str(output / "training_curves.json"),
            str(output / "generator_model_val_stop.pt"),
            str(output / "target_discovery_recipe.json"),
            str(output / "query_tap_registration.json"),
            str(output / "memory_tap_registration.json"),
            str(output / "target_candidate_registration.json"),
            str(output / "target_discovery_result.json"),
            str(output / "train_staged_tap_manifest.json"),
            str(output / "model_val_stop_staged_tap_manifest.json"),
            str(output / "model_val_select_staged_tap_manifest.json"),
            str(output / "provisional_normalizer.json"),
            str(output / "probe_consumer" / "best_model_val_stop.pt"),
            str(output / "probe_consumer" / "consumer_registration.json"),
            str(output / "probe_consumer" / "training_curves.json"),
            str(output / "recovery_probe" / "best_model_val_stop.pt"),
            str(
                output
                / "recovery_probe"
                / "recovery_probe_registration.json"
            ),
            str(output / "recovery_probe" / "training_curves.json"),
            str(output / "candidate_quantization_diagnostics.json"),
            str(output / "model_val_select_counterfactual_metrics.json"),
            str(output / "target_candidate_metrics.json"),
            str(output / "two_pass_candidate.json"),
            str(output / "target_two_pass_result.json"),
            *(
                [
                    str(
                        output
                        / "recoverability_codesign"
                        / "rich_context_registration.json"
                    ),
                    str(
                        output
                        / "recoverability_codesign"
                        / "provisional_head_registration.json"
                    ),
                    str(
                        output
                        / "recoverability_codesign"
                        / "selected_projection.pt"
                    ),
                    str(
                        output
                        / "recoverability_codesign"
                        / "selected_consumer.pt"
                    ),
                    str(
                        output
                        / "recoverability_codesign"
                        / "selected_persistent_probe.pt"
                    ),
                    str(
                        output
                        / "recoverability_codesign"
                        / "codesign_ledger.json"
                    ),
                    str(
                        output
                        / "recoverability_codesign"
                        / "cycle_metrics.json"
                    ),
                    str(
                        output
                        / "recoverability_codesign"
                        / "codesigned_generator.pt"
                    ),
                ]
                if run_id == "VGEN_RECODESIGN"
                else []
            ),
        ],
        "action": None,
    }


def build_target_discovery_task_specs(
    *,
    factory_config_path: str | Path,
) -> dict[str, dict[str, str]]:
    """Return catalog fragments for all declared target-generator rows."""

    path = Path(factory_config_path).resolve()
    config = load_hashed_json(path)
    validate_target_discovery_factory_config(config)
    common = {
        "operation": "target_discovery",
        "factory": (
            "teacher_logit_reco.local_particle_residual_field.particle_view."
            "target_runtime:build_target_discovery_factory"
        ),
        "factory_config_path": str(path),
        "factory_config_sha256": sha256_file(path),
    }
    return {
        run_id: dict(common) for run_id in TARGET_SCREEN_IDS
    }


def build_canonical_target_discovery_task_specs(
    *,
    factory_config_path: str | Path,
) -> dict[str, dict[str, str]]:
    """Backward-compatible alias; now returns the complete target screen."""

    return build_target_discovery_task_specs(
        factory_config_path=factory_config_path
    )


__all__ = [
    "CANONICAL_TARGET_DISCOVERY_RUN_ID",
    "PARTICLE_VIEW_GENERATOR_CHECKPOINT_CONTRACT",
    "PARTICLE_VIEW_TARGET_DISCOVERY_FACTORY_CONFIG_CONTRACT",
    "PARTICLE_VIEW_TARGET_DISCOVERY_RECIPE_CONTRACT",
    "PARTICLE_VIEW_TARGET_DISCOVERY_RESULT_CONTRACT",
    "PARTICLE_VIEW_TARGET_TWO_PASS_RESULT_CONTRACT",
    "PARTICLE_VIEW_TARGET_UNAVAILABLE_CONTRACT",
    "CandidateViewSet",
    "IndexedCandidateViewProvider",
    "StagedContextualMemory",
    "StagedDiscoveryViewProvider",
    "TargetScreenRecipe",
    "TwoTeacherTokenMixture",
    "build_canonical_target_discovery_task_specs",
    "build_target_discovery_task_specs",
    "build_target_discovery_factory",
    "build_target_discovery_factory_config",
    "build_target_screen_recipe",
    "evaluate_independent_a0_accuracy",
    "lorentz_vectors_to_particle_geometry",
    "materialize_candidate_views_in_ram",
    "predict_recovery_probe_views",
    "run_canonical_target_discovery",
    "run_recoverability_codesign",
    "run_target_discovery_operation",
    "stage_contextual_memory",
    "validate_target_discovery_factory_config",
]
