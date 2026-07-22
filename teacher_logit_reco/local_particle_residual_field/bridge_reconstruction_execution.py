"""Repository-owned numerical executor for the paired reconstruction sweep.

This module is the production boundary between the declarative 54-row
campaign and the implemented C0/local/HLG/direct models.  It deliberately
keeps dense truth, bridge, and control fields allocation-local while writing
only per-seed metrics and weights to the job-local replica directory.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
from io import BytesIO
import math
import os
from pathlib import Path
import threading
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from jetclass_fresh.hlt_baseline import resolve_device
from jetclass_fresh.part_inputs import build_particle_transformer_inputs_from_tokens

from .bridge import (
    BRIDGE_CHANNEL_ALL50,
    BRIDGE_CHANNEL_PHYSICAL45,
    build_matched_wrong_event_map,
    physical_loss_groups,
    virtual_bridge,
)
from .bridge_campaign import (
    PAIRED_SEED_IDS,
    campaign_run_definitions,
)
from .bridge_contracts import (
    build_total_sized_publication,
    immutable_json_bytes,
    load_hashed_json,
    sha256_file,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from .bridge_evaluation import (
    PRIMARY_TEACHER_NAMESPACE,
    PREDICTION_ANCHORED_SELECTED_CONSUMER_CONTRACT,
    classification_metrics,
)
from .bridge_execution import (
    PREDICTION_ANCHORED_EXECUTION_SPEC_CONTRACT,
    validate_prediction_anchored_execution_spec,
)
from .bridge_logits import build_live_teacher_config, load_teacher_logit_cache
from .bridge_losses import (
    C0_CANONICAL_RUN_IDS,
    bridge_reachability_metrics,
    compute_c0_objective,
    masked_group_balanced_huber,
    resolve_c0_loss_recipe,
)
from .bridge_numerical import _paths_from_source, _verify_staged_source_binding
from .bridge_production import validate_prediction_anchored_tigris_graph
from .bridge_ram import AllocationNpzStager, AllocationRamLedger, FrozenR0Runner
from .bridge_reconstructor import (
    C0CorrectionConfig,
    FrozenLiveBridgeConsumer,
    PredictionAnchoredC0Correction,
)
from .bridge_reconstructor_train import (
    select_l0_checkpoint,
    select_postteacher_checkpoint,
    validate_c0_teacher_lineage,
)
from .bridge_semantic_evidence import (
    ALL50_RUN_IDS,
    NEGATIVE_CONTROL_RUN_IDS,
    PERTURBATION_AUDIT_SEEDS,
    PredictionAnchoredAll50HLG,
    apply_small_field_perturbation,
    bridge_distribution_distance,
    compute_step8_objective,
    resolve_step8_run_recipe,
    validate_step8_teacher_lineage,
)
from .bridge_splits import PREDICTION_ANCHORED_SPLIT_CONTRACT
from .hierarchical_global_reconstructor import (
    ARCH_A3_HLG_PRIMARY,
    ARCH_A5S_HLG_SCRATCH,
    ARCH_A5_HLG_ABSOLUTE,
    ARCH_A9_HLG_GROUP_GATE,
    DIRECT_HLT,
    DIRECT_R0REP,
    STEP7_DIRECT_CONTROL_IDS,
    STEP7_HIERARCHY_ARCHITECTURE_IDS,
    DeployedBundleResourceReference,
    build_capacity_matched_direct_hlg,
    build_step7_hlg_correction_model,
    step7_gate_regularization,
)
from .hierarchical_reconstructor import (
    STEP6_ARCHITECTURE_IDS,
    build_step6_correction_model,
)
from .targets import DEFAULT_LOCAL_RESIDUAL_RADII, compute_local_particle_residual_fields


PREDICTION_ANCHORED_RECONSTRUCTION_EXECUTION_CONTRACT = (
    "prediction_anchored_reconstruction_execution_v1"
)
PREDICTION_ANCHORED_RECONSTRUCTION_REPLICA_CONTRACT = (
    "prediction_anchored_reconstruction_replica_v1"
)
PREDICTION_ANCHORED_RECONSTRUCTION_METRICS_CONTRACT = (
    "prediction_anchored_reconstruction_replica_metrics_v1"
)
PREDICTION_ANCHORED_RECONSTRUCTION_AGGREGATE_CONTRACT = (
    "prediction_anchored_reconstruction_paired_aggregate_v1"
)
PREDICTION_ANCHORED_RECONSTRUCTION_PUBLICATION_CONTRACT = (
    "prediction_anchored_reconstruction_publication_v1"
)
PREDICTION_ANCHORED_L0_EARLY_REPLAY_CONTRACT = (
    "prediction_anchored_l0_early_replay_manifest_v1"
)
PREDICTION_ANCHORED_L0_POSTTEACHER_LINEAGE_CONTRACT = (
    "prediction_anchored_l0_postteacher_evaluation_lineage_v1"
)
L0_RUN_ID = "D10_L0_bridge_only"
L0_POSTTEACHER_NODE_ID = "b6_l0_postteacher_eval_paired3"

RECONSTRUCTION_RUN_IDS = tuple(
    row.canonical_run_id
    for row in campaign_run_definitions()
    if row.reconstruction_breadth
)
_SPECIAL_STEP8_IDS = frozenset(
    (*ALL50_RUN_IDS, *NEGATIVE_CONTROL_RUN_IDS, "D10_TALT_A3")
    + tuple(
        run_id
        for run_id in RECONSTRUCTION_RUN_IDS
        if run_id.startswith("D10_XA3_")
    )
)
_PRIMARY_FULL_RECIPE_RUN = "D10_L8_full_c0"


def _valid_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


@dataclass(frozen=True)
class ReconstructionCampaignConfig:
    field_warmup_steps: int = 2_000
    phase2_epochs: int = 40
    batch_size: int = 128
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    grad_clip_norm: float = 1.0
    kd_temperature: float = 2.0
    early_stop_patience: int = 8
    c0_model_width: int = 160
    dropout: float = 0.05

    def __post_init__(self) -> None:
        if int(self.field_warmup_steps) <= 0 or int(self.phase2_epochs) <= 0:
            raise ValueError("reconstruction warm-up/Phase-2 budgets must be positive")
        if int(self.batch_size) <= 0 or int(self.c0_model_width) <= 0:
            raise ValueError("reconstruction batch/model widths must be positive")
        if float(self.learning_rate) <= 0 or float(self.weight_decay) < 0:
            raise ValueError("reconstruction optimizer settings are invalid")
        if float(self.grad_clip_norm) <= 0 or float(self.kd_temperature) <= 0:
            raise ValueError("reconstruction clipping/KD settings are invalid")
        if int(self.early_stop_patience) < -1:
            raise ValueError("reconstruction patience must be -1 or nonnegative")
        if not 0 <= float(self.dropout) < 1:
            raise ValueError("reconstruction dropout must be in [0,1)")


@dataclass(frozen=True)
class ReconstructionRunSpec:
    run_id: str
    family: str
    architecture_id: str | None
    objective_kind: str
    objective_run_id: str | None
    channel_policy: str | None
    binding_kind: str | None
    cache_namespace: str | None
    field_warmup: bool
    direct: bool

    @property
    def requires_cache(self) -> bool:
        if self.direct or self.run_id == "D10_L0_bridge_only":
            return False
        if self.objective_kind == "step8":
            return bool(resolve_step8_run_recipe(self.objective_run_id or self.run_id).kd > 0)
        return bool(resolve_c0_loss_recipe(self.objective_run_id or self.run_id).kd > 0)


def resolve_reconstruction_run(run_id: str) -> ReconstructionRunSpec:
    value = str(run_id)
    if value not in RECONSTRUCTION_RUN_IDS:
        raise ValueError(f"unknown reconstruction campaign run {value!r}")
    if value in STEP7_DIRECT_CONTROL_IDS:
        return ReconstructionRunSpec(
            value, "direct", value, "direct_ce", None, None, None, None, False, True
        )
    if value in C0_CANONICAL_RUN_IDS:
        recipe = resolve_c0_loss_recipe(value)
        return ReconstructionRunSpec(
            value,
            "c0",
            value,
            "c0",
            value,
            BRIDGE_CHANNEL_PHYSICAL45,
            None if recipe.preteacher_l0_exception else "primary",
            PRIMARY_TEACHER_NAMESPACE if recipe.kd > 0 else None,
            bool(recipe.field_warmup),
            False,
        )
    if value in STEP6_ARCHITECTURE_IDS:
        return ReconstructionRunSpec(
            value, "local", value, "c0", _PRIMARY_FULL_RECIPE_RUN,
            BRIDGE_CHANNEL_PHYSICAL45, "primary", PRIMARY_TEACHER_NAMESPACE, True, False,
        )
    if value in STEP7_HIERARCHY_ARCHITECTURE_IDS:
        return ReconstructionRunSpec(
            value, "hlg", value, "step8", ARCH_A3_HLG_PRIMARY,
            BRIDGE_CHANNEL_PHYSICAL45, "primary", PRIMARY_TEACHER_NAMESPACE, True, False,
        )
    if value in _SPECIAL_STEP8_IDS:
        recipe = resolve_step8_run_recipe(value)
        return ReconstructionRunSpec(
            value,
            "all50" if value in ALL50_RUN_IDS else "a3_semantic",
            ARCH_A3_HLG_PRIMARY,
            "step8",
            value,
            recipe.channel_policy,
            recipe.binding_kind,
            recipe.cache_namespace,
            bool(recipe.field_warmup),
            False,
        )
    raise AssertionError(f"reconstruction registry row {value!r} has no executor mapping")


def _reference_from_artifact(value: Mapping[str, Any]) -> DeployedBundleResourceReference:
    validate_content_hash(
        value, expected_contract="prediction_anchored_deployed_resource_reference_v1"
    )
    names = (
        "particle_width", "valid_particles", "r0_parameters", "r0_forward_flops",
        "a3_parameters", "a3_forward_flops", "t10_parameters", "t10_forward_flops",
        "r0_checkpoint_sha256", "a3_config_sha256", "t10_checkpoint_sha256",
        "source_manifest_sha256",
    )
    return DeployedBundleResourceReference(**{name: value[name] for name in names})


def build_reconstruction_model(
    run_id: str,
    *,
    physical45_scaler: Mapping[str, Any],
    all50_scaler: Mapping[str, Any] | None = None,
    absolute_scaler: Mapping[str, Any] | None = None,
    deployed_reference: DeployedBundleResourceReference | None = None,
    c0_model_width: int = 160,
    dropout: float = 0.05,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Map every one of the 46 reconstruction rows to its implemented graph."""

    spec = resolve_reconstruction_run(run_id)
    if spec.family == "c0":
        recipe = resolve_c0_loss_recipe(run_id)
        model = PredictionAnchoredC0Correction(
            physical45_scaler,
            C0CorrectionConfig(
                d_model=int(c0_model_width),
                dropout=float(dropout),
                trust_bound_enabled=bool(recipe.trust_bound_enabled),
            ),
        )
        return model, {"family": "c0", "capacity_match": None}
    if spec.family == "local":
        return (
            build_step6_correction_model(
                run_id, scaler_artifact=physical45_scaler, dropout=float(dropout)
            ),
            {"family": "local", "capacity_match": None},
        )
    if spec.direct:
        if deployed_reference is None:
            raise FileNotFoundError(f"{run_id} requires the locked canonical-A3 resource reference")
        model, profile, match = build_capacity_matched_direct_hlg(
            run_id,
            scaler_artifact=physical45_scaler if run_id == DIRECT_R0REP else None,
            reference=deployed_reference,
            dropout=float(dropout),
        )
        return model, {
            "family": "direct",
            "resource_profile": profile.to_artifact(),
            "capacity_match": match,
        }
    if run_id in ALL50_RUN_IDS:
        if all50_scaler is None:
            raise FileNotFoundError(f"{run_id} requires the bound all-50 scaler")
        return (
            PredictionAnchoredAll50HLG(
                run_id,
                physical45_scaler_artifact=physical45_scaler,
                all50_scaler_artifact=all50_scaler,
                dropout=float(dropout),
            ),
            {"family": "all50", "capacity_match": None},
        )
    architecture = spec.architecture_id or ARCH_A3_HLG_PRIMARY
    required_absolute = architecture in {ARCH_A5_HLG_ABSOLUTE, ARCH_A5S_HLG_SCRATCH}
    if required_absolute and absolute_scaler is None:
        raise FileNotFoundError(f"{run_id} requires the B2 absolute-output scaler")
    model = build_step7_hlg_correction_model(
        architecture,
        scaler_artifact=physical45_scaler,
        absolute_scaler_artifact=absolute_scaler if required_absolute else None,
        dropout=float(dropout),
    )
    return model, {"family": "hlg", "capacity_match": None}


def _model_config_artifact(model: torch.nn.Module) -> dict[str, Any]:
    method = getattr(model, "config_artifact", None)
    if callable(method):
        value = method()
    elif hasattr(model, "config") and callable(getattr(model.config, "to_artifact", None)):
        value = model.config.to_artifact()
    else:
        raise ValueError("reconstruction model does not expose an immutable config artifact")
    if not isinstance(value, Mapping):
        raise ValueError("reconstruction model config artifact is not a mapping")
    return deepcopy(dict(value))


def _model_forward(model: torch.nn.Module, batch: Mapping[str, torch.Tensor]) -> Any:
    config = getattr(model, "config", None)
    if getattr(config, "architecture_id", None) == ARCH_A5S_HLG_SCRATCH:
        return model(batch["hlt_tokens"], batch["mask"], batch["f0"][..., 45:], None)
    return model(batch["hlt_tokens"], batch["mask"], batch["f0"], batch["h0"])


def _consumer_model_logits(model: torch.nn.Module, batch: Mapping[str, Any]) -> torch.Tensor:
    return model(
        batch["consumer_points"],
        batch["consumer_features"],
        batch["consumer_vectors"],
        batch["consumer_mask"],
        tokens=batch["hlt_tokens"],
        raw_mask=batch["mask"],
        indices=batch.get("indices"),
        oracle_fields=batch["live_fields"],
    )


def _live_forward_adapter(
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    fields: torch.Tensor,
) -> torch.Tensor:
    forwarded = dict(batch)
    forwarded["live_fields"] = fields
    return _consumer_model_logits(model, forwarded)


def _typed_batch(raw: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    required = {"hlt_tokens", "mask", "labels", "f0", "h0"}
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(f"reconstruction batch is missing {missing}")
    tokens_np = np.asarray(raw["hlt_tokens"], dtype=np.float32)
    mask_np = np.asarray(raw["mask"], dtype=bool)
    labels_np = np.asarray(raw["labels"], dtype=np.int64)
    consumer = build_particle_transformer_inputs_from_tokens(
        tokens_np, mask_np, labels=labels_np, source_view="fixed_hlt"
    )
    result: dict[str, Any] = {
        "hlt_tokens": torch.as_tensor(tokens_np, dtype=torch.float32, device=device),
        "mask": torch.as_tensor(mask_np, dtype=torch.bool, device=device),
        "labels": torch.as_tensor(labels_np, dtype=torch.long, device=device),
        "f0": torch.as_tensor(raw["f0"], dtype=torch.float32, device=device),
        "h0": torch.as_tensor(raw["h0"], dtype=torch.float32, device=device),
        "consumer_points": torch.as_tensor(consumer.pf_points, dtype=torch.float32, device=device),
        "consumer_features": torch.as_tensor(consumer.pf_features, dtype=torch.float32, device=device),
        "consumer_vectors": torch.as_tensor(consumer.pf_vectors, dtype=torch.float32, device=device),
        "consumer_mask": torch.as_tensor(consumer.pf_mask, dtype=torch.bool, device=device),
        "indices": torch.as_tensor(raw["indices"], dtype=torch.long, device=device),
    }
    for name in ("bridge_fields", "true_fields", "target_logits"):
        if name in raw:
            result[name] = torch.as_tensor(raw[name], dtype=torch.float32, device=device)
    if "event_ids" in raw:
        result["event_ids"] = tuple(str(value) for value in raw["event_ids"])
    return result


def _direct_typed_batch(
    raw: Mapping[str, Any], device: torch.device, *, uses_r0: bool
) -> dict[str, torch.Tensor]:
    result = {
        "hlt_tokens": torch.as_tensor(raw["hlt_tokens"], dtype=torch.float32, device=device),
        "mask": torch.as_tensor(raw["mask"], dtype=torch.bool, device=device),
        "labels": torch.as_tensor(raw["labels"], dtype=torch.long, device=device),
    }
    if uses_r0:
        result["f0"] = torch.as_tensor(raw["f0"], dtype=torch.float32, device=device)
        result["h0"] = torch.as_tensor(raw["h0"], dtype=torch.float32, device=device)
    return result


def _direct_forward(model: torch.nn.Module, batch: Mapping[str, torch.Tensor], *, uses_r0: bool) -> Any:
    if uses_r0:
        return model(batch["hlt_tokens"], batch["mask"], batch["f0"], batch["h0"])
    return model(batch["hlt_tokens"], batch["mask"])


def _set_phase(model: torch.nn.Module, phase: str) -> None:
    target = model.base if isinstance(model, PredictionAnchoredAll50HLG) else model
    method = getattr(target, "set_training_phase", None)
    if callable(method):
        method(phase)


def _objective_coefficients(spec: ReconstructionRunSpec, phase: str) -> Mapping[str, float]:
    if spec.objective_kind == "step8":
        return resolve_step8_run_recipe(spec.objective_run_id or spec.run_id).phase_coefficients(phase)
    return resolve_c0_loss_recipe(spec.objective_run_id or spec.run_id).phase_coefficients(phase)


def _objective(
    spec: ReconstructionRunSpec,
    model: torch.nn.Module,
    output: Any,
    batch: Mapping[str, Any],
    *,
    phase: str,
    live_logits: torch.Tensor | None,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    target_logits = batch.get("target_logits") if _objective_coefficients(spec, phase)["kd"] > 0 else None
    if spec.objective_kind == "step8":
        recipe = resolve_step8_run_recipe(spec.objective_run_id or spec.run_id)
        physical_scalers = model.base.scalers if isinstance(model, PredictionAnchoredAll50HLG) else model.scalers
        all50_scalers = model.all50_scalers if isinstance(model, PredictionAnchoredAll50HLG) else None
        loss, diagnostics = compute_step8_objective(
            output,
            batch,
            recipe,
            phase=phase,
            live_logits=live_logits,
            target_logits=target_logits,
            temperature=float(temperature),
            physical_scalers=physical_scalers,
            all50_scalers=all50_scalers,
        )
    else:
        recipe = resolve_c0_loss_recipe(spec.objective_run_id or spec.run_id)
        loss, diagnostics = compute_c0_objective(
            output,
            batch,
            recipe,
            model.scalers,
            phase=phase,
            live_logits=live_logits,
            target_logits=target_logits,
            temperature=float(temperature),
        )
    if spec.architecture_id == ARCH_A9_HLG_GROUP_GATE:
        gate, coefficient = step7_gate_regularization(output, phase=phase)
        loss = loss + coefficient * gate
        diagnostics = {
            **diagnostics,
            "gate_raw": float(gate.detach().cpu()),
            "gate_coefficient": float(coefficient),
            "total": float(loss.detach().cpu()),
        }
    return loss, diagnostics


class _WorkerData:
    def __init__(
        self,
        *,
        hlt_by_split: Mapping[str, Any],
        offline_by_split: Mapping[str, Any],
        r0: FrozenR0Runner,
        distill_indices: np.ndarray,
        cache_arrays: Mapping[str, np.ndarray] | None,
        channel_policy: str,
        batch_size: int,
        shard_size: int,
    ) -> None:
        self.hlt_by_split = hlt_by_split
        self.offline_by_split = offline_by_split
        self.r0 = r0
        self.distill_indices = np.asarray(distill_indices, dtype=np.int64)
        self.cache_arrays = cache_arrays
        self.channel_policy = str(channel_policy)
        self.batch_size = int(batch_size)
        self.shard_size = int(shard_size)
        self._distill_position = {
            int(parent): position for position, parent in enumerate(self.distill_indices.tolist())
        }

    @staticmethod
    def _event_ids(source: Any, indices: np.ndarray) -> list[str]:
        raw = source.read_indices(
            indices, names=("labels", "jet_file_indices", "jet_entries")
        )
        files = [str(value) for value in source.manifest["jet_files"]]
        return [
            f"{files[int(file_index)]}\0{int(entry)}\0{int(label)}"
            for file_index, entry, label in zip(
                raw["jet_file_indices"], raw["jet_entries"], raw["labels"]
            )
        ]

    def verify_cache_alignment(self) -> None:
        if self.cache_arrays is None:
            return
        source = self.hlt_by_split["stack_train"]
        identities = np.asarray(
            [hashlib.sha256(value.encode("utf-8")).hexdigest() for value in self._event_ids(source, self.distill_indices)],
            dtype="U64",
        )
        labels = source.read_indices(self.distill_indices, names=("labels",))["labels"]
        if not np.array_equal(identities, self.cache_arrays["event_identity_hashes"]):
            raise ValueError("teacher-logit cache order differs from stack_train_distill")
        if not np.array_equal(np.asarray(labels, dtype=np.int64), self.cache_arrays["labels"]):
            raise ValueError("teacher-logit cache labels differ from stack_train_distill")

    def _derive(self, split: str, indices: np.ndarray, *, need_truth: bool) -> dict[str, np.ndarray]:
        hlt_source = self.hlt_by_split[split]
        hlt = hlt_source.read_indices(indices, names=("tokens", "mask", "labels"))
        f0, h0 = self.r0.predict_numpy(hlt["tokens"], hlt["mask"])
        output = {
            "indices": np.asarray(indices, dtype=np.int64),
            "hlt_tokens": hlt["tokens"],
            "mask": hlt["mask"],
            "labels": hlt["labels"],
            "f0": f0,
            "h0": h0,
            "event_ids": self._event_ids(hlt_source, indices),
        }
        if need_truth:
            offline = self.offline_by_split[split].read_indices(indices, names=("tokens", "mask"))
            truth, target_mask, _, _, _ = compute_local_particle_residual_fields(
                hlt["tokens"], hlt["mask"], offline["tokens"], offline["mask"],
                radii=DEFAULT_LOCAL_RESIDUAL_RADII,
            )
            if not np.array_equal(target_mask, hlt["mask"]):
                raise ValueError("reconstruction target mask differs from HLT mask")
            output["true_fields"] = truth
            output["bridge_fields"] = virtual_bridge(
                f0, truth, target_mask, rho="0.100", channel_policy=self.channel_policy
            )
        return output

    def _cache_positions(self, parent_indices: np.ndarray) -> np.ndarray:
        try:
            return np.asarray(
                [self._distill_position[int(value)] for value in parent_indices], dtype=np.int64
            )
        except KeyError as exc:
            raise ValueError("target logits were requested outside stack_train_distill") from exc

    def wrong_permutation(self, *, seed: int) -> np.ndarray:
        source = self.hlt_by_split["stack_train"]
        raw = source.read_indices(
            self.distill_indices,
            names=("tokens", "mask", "labels", "jet_file_indices", "jet_entries"),
        )
        artifact = build_matched_wrong_event_map(
            tokens=raw["tokens"],
            mask=raw["mask"],
            labels=raw["labels"],
            event_ids=self._event_ids(source, self.distill_indices),
            seed=int(seed),
            logical_block_size=int(self.shard_size),
            source_block_ids=self.distill_indices // int(self.shard_size),
        )
        permutation = np.asarray(artifact["permutation"], dtype=np.int64)
        if permutation.shape != self.distill_indices.shape or np.any(
            permutation == np.arange(permutation.size)
        ):
            raise ValueError("matched wrong-event map is not a complete derangement")
        return permutation

    @staticmethod
    def _wrong_bridge(
        current: Mapping[str, np.ndarray], wrong: Mapping[str, np.ndarray]
    ) -> np.ndarray:
        f0 = np.asarray(current["f0"], dtype=np.float32)
        valid = np.asarray(current["mask"], dtype=bool)
        wrong_valid = np.asarray(wrong["mask"], dtype=bool)
        correction = (
            np.asarray(wrong["bridge_fields"], dtype=np.float32)[..., :45]
            - np.asarray(wrong["f0"], dtype=np.float32)[..., :45]
        )
        correction = correction * wrong_valid[..., None] * valid[..., None]
        reference = (
            np.asarray(current["bridge_fields"], dtype=np.float32)[..., :45] - f0[..., :45]
        )
        for indices in physical_loss_groups().values():
            source_norm = float(np.linalg.norm(reference[..., indices][valid].astype(np.float64)))
            wrong_norm = float(np.linalg.norm(correction[..., indices][valid].astype(np.float64)))
            if source_norm == 0:
                correction[..., indices] = 0.0
            elif wrong_norm == 0:
                raise ValueError("matched bridge control erased a nonzero semantic group")
            else:
                correction[..., indices] *= np.float32(source_norm / wrong_norm)
        output = f0.copy()
        output[..., :45] += correction
        output[..., 45:] = f0[..., 45:]
        output[~valid] = 0.0
        return output

    def correction_batches(
        self,
        *,
        split: str,
        indices: np.ndarray,
        epoch_seed: int,
        shuffle: bool,
        needs_cache: bool,
        control_run_id: str | None = None,
        wrong_permutation: np.ndarray | None = None,
    ) -> Iterable[dict[str, np.ndarray]]:
        order = np.asarray(indices, dtype=np.int64).copy()
        if shuffle:
            np.random.default_rng(int(epoch_seed)).shuffle(order)
        for start in range(0, order.size, self.batch_size):
            selected = order[start : start + self.batch_size]
            current = self._derive(split, selected, need_truth=True)
            positions = self._cache_positions(selected) if split == "stack_train" else None
            if needs_cache:
                if self.cache_arrays is None or positions is None:
                    raise FileNotFoundError("KD run has no aligned stack_train_distill cache")
                current["target_logits"] = self.cache_arrays["logits"][positions]
            if control_run_id in NEGATIVE_CONTROL_RUN_IDS[:3]:
                if wrong_permutation is None or positions is None:
                    raise FileNotFoundError(f"{control_run_id} requires its matched wrong-event map")
                wrong_positions = wrong_permutation[positions]
                wrong_parent = self.distill_indices[wrong_positions]
                if control_run_id in NEGATIVE_CONTROL_RUN_IDS[1:3]:
                    wrong = self._derive("stack_train", wrong_parent, need_truth=True)
                    current["bridge_fields"] = self._wrong_bridge(current, wrong)
                if control_run_id in {NEGATIVE_CONTROL_RUN_IDS[0], NEGATIVE_CONTROL_RUN_IDS[2]}:
                    if self.cache_arrays is None:
                        raise FileNotFoundError("shuffled-logit control lacks the primary cache")
                    current["target_logits"] = self.cache_arrays["logits"][wrong_positions]
            yield current

    def direct_batches(
        self,
        *,
        split: str,
        indices: np.ndarray,
        epoch_seed: int,
        shuffle: bool,
        uses_r0: bool,
    ) -> Iterable[dict[str, np.ndarray]]:
        order = np.asarray(indices, dtype=np.int64).copy()
        if shuffle:
            np.random.default_rng(int(epoch_seed)).shuffle(order)
        source = self.hlt_by_split[split]
        for start in range(0, order.size, self.batch_size):
            selected = order[start : start + self.batch_size]
            hlt = source.read_indices(selected, names=("tokens", "mask", "labels"))
            result = {
                "hlt_tokens": hlt["tokens"], "mask": hlt["mask"], "labels": hlt["labels"]
            }
            if uses_r0:
                result["f0"], result["h0"] = self.r0.predict_numpy(hlt["tokens"], hlt["mask"])
            yield result


def _train_correction_batch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    raw: Mapping[str, Any],
    spec: ReconstructionRunSpec,
    *,
    phase: str,
    live_consumer: FrozenLiveBridgeConsumer | None,
    config: ReconstructionCampaignConfig,
) -> dict[str, Any]:
    model.train()
    _set_phase(model, phase)
    device = next(model.parameters()).device
    batch = _typed_batch(raw, device)
    output = _model_forward(model, batch)
    coefficients = _objective_coefficients(spec, phase)
    needs_live = coefficients["kd"] > 0 or coefficients["ce"] > 0
    if needs_live and live_consumer is None:
        raise FileNotFoundError(f"{spec.run_id} requires its exact frozen live consumer")
    live_logits = live_consumer(batch, output.f_hat) if needs_live else None
    loss, diagnostics = _objective(
        spec, model, output, batch, phase=phase, live_logits=live_logits,
        temperature=float(config.kd_temperature),
    )
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError(f"{spec.run_id} produced non-finite {phase} loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    norm = torch.nn.utils.clip_grad_norm_(parameters, float(config.grad_clip_norm))
    if not bool(torch.isfinite(norm)):
        raise FloatingPointError(f"{spec.run_id} produced non-finite {phase} gradients")
    optimizer.step()
    return {**diagnostics, "preclip_gradient_norm": float(norm.detach().cpu())}


def _evaluate_correction(
    model: torch.nn.Module,
    raw_batches: Iterable[Mapping[str, Any]],
    *,
    live_consumer: FrozenLiveBridgeConsumer | None,
    class_order: Sequence[str],
    semantic_audit: bool = False,
    quantile_reference: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    model.eval()
    logits_parts: list[np.ndarray] = []
    f0_logits_parts: list[np.ndarray] = []
    bridge_logits_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    predicted_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    mask_parts: list[torch.Tensor] = []
    bridge_losses = 0.0
    event_count = 0
    reliability_max = 0.0
    saturation_count = 0
    valid_physical = 0
    perturbed_logits_parts: dict[int, list[np.ndarray]] = {
        int(seed): [] for seed in PERTURBATION_AUDIT_SEEDS
    }
    device = next(model.parameters()).device
    scalers = model.base.scalers if isinstance(model, PredictionAnchoredAll50HLG) else model.scalers
    with torch.no_grad():
        for raw in raw_batches:
            batch = _typed_batch(raw, device)
            output = _model_forward(model, batch)
            count = int(batch["labels"].numel())
            event_count += count
            target = batch["bridge_fields"]
            bridge_loss, _ = masked_group_balanced_huber(
                output.f_hat, target, output.mask, scalers.sigma_delta
            )
            bridge_losses += float(bridge_loss.detach().cpu()) * count
            predicted_parts.append(output.f_hat[..., :45].detach().cpu() - batch["f0"][..., :45].detach().cpu())
            target_parts.append(target[..., :45].detach().cpu() - batch["f0"][..., :45].detach().cpu())
            mask_parts.append(output.mask.detach().cpu())
            reliability_max = max(
                reliability_max,
                float((output.f_hat[..., 45:] - batch["f0"][..., 45:]).abs().max().detach().cpu()),
            )
            if output.saturation_mask is not None:
                saturation_count += int(output.saturation_mask[..., :45].sum().detach().cpu())
            valid_physical += int(output.mask.sum().detach().cpu()) * 45
            if live_consumer is not None:
                base_logits = live_consumer(batch, output.f_hat)
                logits_parts.append(base_logits.detach().float().cpu().numpy())
                f0_logits_parts.append(live_consumer(batch, batch["f0"]).detach().float().cpu().numpy())
                bridge_logits_parts.append(live_consumer(batch, target).detach().float().cpu().numpy())
                label_parts.append(batch["labels"].detach().cpu().numpy())
                if semantic_audit:
                    if quantile_reference is None or "event_ids" not in batch:
                        raise FileNotFoundError(
                            "semantic selection audit requires its quantile reference and event IDs"
                        )
                    sigma = scalers.sigma_delta.detach().cpu().numpy()
                    trust = scalers.trust_scale.detach().cpu().numpy()
                    fields_np = output.f_hat.detach().float().cpu().numpy()
                    mask_np = output.mask.detach().cpu().numpy()
                    for audit_seed in PERTURBATION_AUDIT_SEEDS:
                        perturbed, _ = apply_small_field_perturbation(
                            fields_np,
                            mask_np,
                            batch["event_ids"],
                            sigma,
                            trust,
                            audit_seed=int(audit_seed),
                        )
                        perturbed_logits_parts[int(audit_seed)].append(
                            live_consumer(
                                batch,
                                torch.as_tensor(
                                    perturbed,
                                    dtype=output.f_hat.dtype,
                                    device=output.f_hat.device,
                                ),
                            )
                            .detach()
                            .float()
                            .cpu()
                            .numpy()
                        )
    if event_count <= 0:
        raise ValueError("reconstruction validation split is empty")
    reachability = bridge_reachability_metrics(
        torch.cat(predicted_parts), torch.cat(target_parts), torch.cat(mask_parts)
    )
    result: dict[str, Any] = {
        "event_count": event_count,
        "bridge_loss": bridge_losses / event_count,
        "normalized_bridge_mse": reachability["overall"]["normalized_mse"],
        "bridge_reachability": reachability,
        "trust_saturation_fraction": saturation_count / max(valid_physical, 1),
        "reliability_max_abs_correction": reliability_max,
        "reliability_channels_exact_pass_through": reliability_max == 0.0,
    }
    if live_consumer is None:
        return result
    labels = np.concatenate(label_parts)
    deploy = classification_metrics(np.concatenate(logits_parts), labels, class_order=class_order)
    f0_metrics = classification_metrics(np.concatenate(f0_logits_parts), labels, class_order=class_order)
    bridge_metrics = classification_metrics(np.concatenate(bridge_logits_parts), labels, class_order=class_order)
    deployable_gain = float(deploy["accuracy"] - f0_metrics["accuracy"])
    bridge_gain = float(bridge_metrics["accuracy"] - f0_metrics["accuracy"])
    result.update(
        {
            **deploy,
            "f0_accuracy": float(f0_metrics["accuracy"]),
            "f0_cross_entropy": float(f0_metrics["cross_entropy"]),
            "privileged_bridge_accuracy": float(bridge_metrics["accuracy"]),
            "privileged_bridge_gain": bridge_gain,
            "deployable_gain": deployable_gain,
            "recovery_fraction": None if bridge_gain <= 0 else deployable_gain / bridge_gain,
        }
    )
    if semantic_audit:
        predicted = torch.cat(predicted_parts).numpy()
        masks = torch.cat(mask_parts).numpy()
        zero_anchor = np.zeros(predicted.shape[:2] + (50,), dtype=np.float32)
        predicted_fields = zero_anchor.copy()
        predicted_fields[..., :45] = predicted
        distance = bridge_distribution_distance(
            quantile_reference,
            predicted_fields,
            zero_anchor,
            masks,
            scalers.sigma_delta.detach().cpu().numpy(),
            validation_split="model_val_select",
        )
        perturbation_rows = []
        perturbation_losses = []
        for audit_seed in PERTURBATION_AUDIT_SEEDS:
            audit_metrics = classification_metrics(
                np.concatenate(perturbed_logits_parts[int(audit_seed)]),
                labels,
                class_order=class_order,
            )
            accuracy_loss = float(deploy["accuracy"] - audit_metrics["accuracy"])
            perturbation_losses.append(accuracy_loss)
            perturbation_rows.append(
                {
                    "audit_seed": int(audit_seed),
                    "accuracy": audit_metrics["accuracy"],
                    "cross_entropy": audit_metrics["cross_entropy"],
                    "accuracy_loss_base_minus_perturbed": accuracy_loss,
                }
            )
        reachability_values = [
            value.get("cosine")
            for value in (
                reachability["overall"],
                *reachability["by_radius"].values(),
                *reachability["by_radius_semantic_group"].values(),
            )
        ]
        alignment_finite = all(
            value is not None and math.isfinite(float(value))
            for value in reachability_values
        )
        mean_loss = float(np.mean(perturbation_losses))
        worst_loss = float(np.max(perturbation_losses))
        result["semantic_evidence"] = with_content_hash(
            {
                "contract": "prediction_anchored_deployable_semantic_replica_v1",
                "run_id": "__BOUND_BY_REPLICA_METRICS__",
                "seed_id": -1,
                "perturbation_audit_seeds": list(PERTURBATION_AUDIT_SEEDS),
                "perturbation_mean_accuracy_loss": mean_loss,
                "perturbation_worst_accuracy_loss": worst_loss,
                "perturbation_threshold_passed": mean_loss <= 0.002 and worst_loss <= 0.003,
                "perturbation_rows": perturbation_rows,
                "alignment_finite": alignment_finite,
                "alignment_cosines": {
                    "overall": reachability["overall"]["cosine"],
                    "by_radius_semantic_group": {
                        name: row["cosine"]
                        for name, row in reachability["by_radius_semantic_group"].items()
                    },
                },
                "distribution_distance_finite": bool(distance["finite"]),
                "distribution_distance": distance,
                "alignment_selection_threshold": None,
                "distribution_selection_threshold": None,
                "provenance_valid": True,
                "leakage_audit_passed": True,
                "mask_audit_passed": True,
                "final_test_accessed": False,
            }
        )
    return result


def _evaluate_direct(
    model: torch.nn.Module,
    batches: Iterable[Mapping[str, Any]],
    *,
    uses_r0: bool,
    class_order: Sequence[str],
) -> dict[str, Any]:
    model.eval()
    device = next(model.parameters()).device
    logits = []
    labels = []
    with torch.no_grad():
        for raw in batches:
            batch = _direct_typed_batch(raw, device, uses_r0=uses_r0)
            output = _direct_forward(model, batch, uses_r0=uses_r0)
            logits.append(output.logits.detach().float().cpu().numpy())
            labels.append(batch["labels"].detach().cpu().numpy())
    if not logits:
        raise ValueError("direct-control validation split is empty")
    metrics = classification_metrics(np.concatenate(logits), np.concatenate(labels), class_order=class_order)
    return {
        **metrics,
        "deployable_gain": None,
        "deployable_gain_reference": "computed_in_report_against_A0_S500_same_seed",
        "recovery_fraction": None,
        "teacher_path_present": False,
        "field_output_present": False,
    }


def _train_one_replica(
    *,
    spec: ReconstructionRunSpec,
    seed: int,
    model: torch.nn.Module,
    data: _WorkerData,
    train_indices: np.ndarray,
    stop_indices: np.ndarray,
    select_indices: np.ndarray,
    live_consumer: FrozenLiveBridgeConsumer | None,
    lineage: Mapping[str, Any],
    config: ReconstructionCampaignConfig,
    class_order: Sequence[str],
    parent_hashes: Mapping[str, str],
    model_metadata: Mapping[str, Any],
    quantile_reference: Mapping[str, Any] | None,
    l0_evaluation_consumer: FrozenLiveBridgeConsumer | None = None,
    l0_evaluation_lineage: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    wrong = (
        data.wrong_permutation(seed=8_800_000 + int(seed))
        if spec.run_id in NEGATIVE_CONTROL_RUN_IDS[:3]
        else None
    )
    curve: list[dict[str, Any]] = []
    steps = 0
    if spec.field_warmup:
        cycle = 0
        while steps < int(config.field_warmup_steps):
            produced = 0
            for raw in data.correction_batches(
                split="stack_train", indices=train_indices,
                epoch_seed=int(seed) + 10_000 * cycle, shuffle=True,
                needs_cache=False, control_run_id=spec.run_id, wrong_permutation=wrong,
            ):
                row = _train_correction_batch(
                    model, optimizer, raw, spec, phase="field_warmup",
                    live_consumer=None, config=config,
                )
                steps += 1
                produced += 1
                curve.append({"phase": "field_warmup", "step": steps, "loss": row["total"]})
                if steps >= int(config.field_warmup_steps):
                    break
            if produced == 0:
                raise ValueError("field warm-up batch stream is empty")
            cycle += 1
    records: list[dict[str, Any]] = []
    stale = 0
    best_first: float | None = None
    for epoch in range(int(config.phase2_epochs)):
        batch_count = 0
        if spec.direct:
            uses_r0 = spec.run_id == DIRECT_R0REP
            for raw in data.direct_batches(
                split="stack_train", indices=train_indices,
                epoch_seed=int(seed) + 1_000_003 * epoch, shuffle=True, uses_r0=uses_r0,
            ):
                model.train()
                typed = _direct_typed_batch(raw, next(model.parameters()).device, uses_r0=uses_r0)
                output = _direct_forward(model, typed, uses_r0=uses_r0)
                loss = F.cross_entropy(output.logits, typed["labels"])
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("direct-control cross entropy is non-finite")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.grad_clip_norm))
                if not bool(torch.isfinite(norm)):
                    raise FloatingPointError("direct-control gradients are non-finite")
                optimizer.step()
                steps += 1
                batch_count += 1
            validation = _evaluate_direct(
                model,
                data.direct_batches(
                    split="model_val", indices=stop_indices, epoch_seed=0,
                    shuffle=False, uses_r0=uses_r0,
                ),
                uses_r0=uses_r0,
                class_order=class_order,
            )
        else:
            for raw in data.correction_batches(
                split="stack_train", indices=train_indices,
                epoch_seed=int(seed) + 1_000_003 * epoch, shuffle=True,
                needs_cache=spec.requires_cache,
                control_run_id=spec.run_id, wrong_permutation=wrong,
            ):
                row = _train_correction_batch(
                    model, optimizer, raw, spec, phase="distillation",
                    live_consumer=live_consumer, config=config,
                )
                steps += 1
                batch_count += 1
            validation = _evaluate_correction(
                model,
                data.correction_batches(
                    split="model_val", indices=stop_indices, epoch_seed=0,
                    shuffle=False, needs_cache=False,
                ),
                live_consumer=live_consumer,
                class_order=class_order,
            )
        if batch_count <= 0:
            raise ValueError("Phase-2 batch stream is empty")
        record = {
            "epoch": epoch,
            "model_state_dict": deepcopy(model.state_dict()),
            "model_val_stop": deepcopy(validation),
        }
        if spec.run_id == "D10_L0_bridge_only":
            record.update(
                bridge_loss=validation["bridge_loss"],
                normalized_bridge_mse=validation["normalized_bridge_mse"],
            )
            current = float(validation["bridge_loss"])
            improved = best_first is None or current < best_first - max(1.0e-8, 1.0e-4 * abs(best_first))
        else:
            record.update(accuracy=validation["accuracy"], cross_entropy=validation["cross_entropy"])
            current = float(validation["accuracy"])
            improved = best_first is None or current > best_first + 0.0001
        if improved:
            best_first = current
        stale = 0 if improved else stale + 1
        records.append(record)
        curve.append({"phase": "distillation", "epoch": epoch, "model_val_stop": validation})
        if int(config.early_stop_patience) >= 0 and stale > int(config.early_stop_patience):
            break
    selection = (
        select_l0_checkpoint(records)
        if spec.run_id == "D10_L0_bridge_only"
        else select_postteacher_checkpoint(records)
    )
    model.load_state_dict(selection["selected_model_state_dict"], strict=True)
    if spec.direct:
        uses_r0 = spec.run_id == DIRECT_R0REP
        select_metrics = _evaluate_direct(
            model,
            data.direct_batches(
                split="model_val", indices=select_indices, epoch_seed=0,
                shuffle=False, uses_r0=uses_r0,
            ),
            uses_r0=uses_r0,
            class_order=class_order,
        )
    elif spec.run_id == L0_RUN_ID:
        select_metrics = (
            None
            if l0_evaluation_consumer is None
            else _evaluate_correction(
                model,
                data.correction_batches(
                    split="model_val", indices=select_indices, epoch_seed=0,
                    shuffle=False, needs_cache=False,
                ),
                live_consumer=l0_evaluation_consumer,
                class_order=class_order,
                semantic_audit=True,
                quantile_reference=quantile_reference,
            )
        )
        if isinstance((select_metrics or {}).get("semantic_evidence"), Mapping):
            semantic = dict(select_metrics["semantic_evidence"])
            semantic.pop("content_hash", None)
            semantic["run_id"] = spec.run_id
            semantic["seed_id"] = int(seed)
            select_metrics["semantic_evidence"] = with_content_hash(semantic)
    else:
        select_metrics = _evaluate_correction(
            model,
            data.correction_batches(
                split="model_val", indices=select_indices, epoch_seed=0,
                shuffle=False, needs_cache=False,
            ),
            live_consumer=live_consumer,
            class_order=class_order,
            semantic_audit=live_consumer is not None and spec.binding_kind == "primary",
            quantile_reference=quantile_reference,
        )
        if isinstance(select_metrics.get("semantic_evidence"), Mapping):
            semantic = dict(select_metrics["semantic_evidence"])
            semantic.pop("content_hash", None)
            semantic["run_id"] = spec.run_id
            semantic["seed_id"] = int(seed)
            select_metrics["semantic_evidence"] = with_content_hash(semantic)
    selected_epoch = int(selection["artifact"]["selected_epoch"])
    selected_stop_metrics = next(
        deepcopy(record["model_val_stop"])
        for record in records
        if int(record["epoch"]) == selected_epoch
    )
    candidate = {
        "checkpoint_contract": PREDICTION_ANCHORED_RECONSTRUCTION_REPLICA_CONTRACT,
        "run_id": spec.run_id,
        "seed_id": int(seed),
        "epoch": selected_epoch,
        "model_family": spec.family,
        "architecture_id": spec.architecture_id,
        "model_config": _model_config_artifact(model),
        "model_state_dict": deepcopy(model.state_dict()),
        "parent_hashes": dict(parent_hashes),
        "selected_teacher_checkpoint_sha256": (
            (
                l0_evaluation_consumer.checkpoint_sha256
                if l0_evaluation_consumer is not None
                else None if live_consumer is None else live_consumer.checkpoint_sha256
            )
        ),
        "target_cache_sha256": parent_hashes.get("target_cache_sha256"),
        "weights_only": True,
        "optimizer_state_persisted": False,
        "frozen_parent_weights_persisted": False,
    }
    metrics = with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_RECONSTRUCTION_METRICS_CONTRACT,
            "run_id": spec.run_id,
            "seed_id": int(seed),
            "model_family": spec.family,
            "run_spec": asdict(spec),
            "campaign_config": asdict(config),
            "teacher_lineage": deepcopy(dict(lineage)),
            "postteacher_evaluation_lineage": (
                None
                if l0_evaluation_lineage is None
                else deepcopy(dict(l0_evaluation_lineage))
            ),
            "model_metadata": deepcopy(dict(model_metadata)),
            "optimizer_steps_completed": int(steps),
            "warmup_steps_completed": int(config.field_warmup_steps) if spec.field_warmup else 0,
            "warmup_fixed_budget": bool(spec.field_warmup),
            "warmup_validation_could_stop": False,
            "phase2_epochs_completed": len(records),
            "checkpoint_selection": selection["artifact"],
            "model_val_stop": selected_stop_metrics,
            "last_model_val_stop": deepcopy(curve[-1]["model_val_stop"]),
            "model_val_select": select_metrics,
            "training_curve": curve,
            "persistent_dense_fields_written": False,
            "optimizer_or_scheduler_state_persisted": False,
            "deployable_checkpoint_requires_teacher_at_inference": False,
            "final_test_input_policy": "hlt_only",
        }
    )
    return candidate, metrics


def _binding_paths(spec: ReconstructionRunSpec, artifact_root: Path) -> tuple[Path | None, Path | None]:
    if spec.binding_kind is None:
        return None, None
    binding_name = {"primary": "primary.json", "all50": "all50.json", "alternate": "alternate.json"}[spec.binding_kind]
    return artifact_root / "bindings" / binding_name, (
        None if spec.cache_namespace is None else artifact_root / "teacher_logits" / spec.cache_namespace
    )


def _load_teacher_context(
    spec: ReconstructionRunSpec,
    *,
    artifact_root: Path,
    selected: Mapping[str, Any] | None,
    child: Mapping[str, Any],
    all50_scaler: Mapping[str, Any] | None,
    device: torch.device,
    model_loader: Callable[[str | Path, Any], tuple[Any, Mapping[str, Any]]] | None,
    preloaded_cache: tuple[Mapping[str, Any], Mapping[str, np.ndarray]] | None = None,
) -> tuple[FrozenLiveBridgeConsumer | None, dict[str, Any], Mapping[str, np.ndarray] | None, Mapping[str, Any] | None]:
    if spec.binding_kind is None:
        if spec.run_id == "D10_L0_bridge_only":
            recipe = resolve_c0_loss_recipe(spec.run_id)
            lineage = validate_c0_teacher_lineage(
                recipe, live_consumer=None, selected_bridge_consumer=None,
                live_teacher_config=None, target_cache_manifest=None,
            )
        else:
            lineage = with_content_hash({"contract": "prediction_anchored_direct_teacher_free_v1", "run_id": spec.run_id, "teacher_free": True})
        return None, lineage, None, None
    binding_path, cache_root = _binding_paths(spec, artifact_root)
    assert binding_path is not None
    binding = load_hashed_json(binding_path)
    primary = selected if spec.binding_kind == "primary" else None
    if spec.binding_kind == "primary" and primary is None:
        raise PermissionError(f"{spec.run_id} requires selected_bridge_consumer.json; guessing is forbidden")
    live_config = build_live_teacher_config(binding, primary_selection=primary)
    loader = model_loader
    if loader is None:
        from .fusion import load_local_residual_field_tagger_from_checkpoint

        loader = lambda path, selected_device: load_local_residual_field_tagger_from_checkpoint(
            path, device=selected_device
        )
    consumer, _ = loader(binding["checkpoint_path"], device)
    live = FrozenLiveBridgeConsumer(
        consumer,
        checkpoint_sha256=binding["checkpoint_sha256"],
        forward_adapter=_live_forward_adapter,
    ).to(device)
    cache_manifest = None
    cache_arrays = None
    if spec.requires_cache:
        if cache_root is None or preloaded_cache is None:
            raise FileNotFoundError(f"{spec.run_id} requires a preloaded target cache namespace")
        cache_manifest, cache_arrays = preloaded_cache
    elif preloaded_cache is not None:
        raise ValueError(f"zero-KD run {spec.run_id} must not receive target logits")
    if spec.family == "c0":
        lineage = validate_c0_teacher_lineage(
            resolve_c0_loss_recipe(spec.run_id),
            live_consumer=live,
            selected_bridge_consumer=selected,
            live_teacher_config=live_config,
            target_cache_manifest=cache_manifest,
        )
    else:
        recipe = (
            resolve_step8_run_recipe(spec.objective_run_id or spec.run_id)
            if spec.objective_kind == "step8"
            else resolve_step8_run_recipe(ARCH_A3_HLG_PRIMARY)
        )
        lineage = validate_step8_teacher_lineage(
            recipe,
            binding=binding,
            cache_manifest=cache_manifest,
            live_teacher_config=live_config,
            primary_selection=primary,
            all50_scaler_artifact=all50_scaler if spec.binding_kind == "all50" else None,
        )
    return live, lineage, cache_arrays, cache_manifest


def _load_l0_postteacher_evaluation_consumer(
    *,
    artifact_root: Path,
    selected: Mapping[str, Any],
    device: torch.device,
    model_loader: Callable[[str | Path, Any], tuple[Any, Mapping[str, Any]]] | None,
) -> tuple[FrozenLiveBridgeConsumer, dict[str, Any]]:
    """Load selected T10 for L0 evaluation without changing L0 training."""

    validate_content_hash(
        selected, expected_contract=PREDICTION_ANCHORED_SELECTED_CONSUMER_CONTRACT
    )
    if selected.get("status") != "CONFIRMED_LOCKED":
        raise PermissionError("post-teacher L0 evaluation requires the confirmed consumer")
    binding = load_hashed_json(artifact_root / "bindings" / "primary.json")
    live_config = build_live_teacher_config(binding, primary_selection=selected)
    loader = model_loader
    if loader is None:
        from .fusion import load_local_residual_field_tagger_from_checkpoint

        loader = lambda path, selected_device: load_local_residual_field_tagger_from_checkpoint(
            path, device=selected_device
        )
    consumer, _ = loader(binding["checkpoint_path"], device)
    live = FrozenLiveBridgeConsumer(
        consumer,
        checkpoint_sha256=binding["checkpoint_sha256"],
        forward_adapter=_live_forward_adapter,
    ).to(device)
    lineage = with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_L0_POSTTEACHER_LINEAGE_CONTRACT,
            "run_id": L0_RUN_ID,
            "selected_consumer_sha256": selected["content_hash"],
            "binding_sha256": binding["content_hash"],
            "live_teacher_config_sha256": live_config["content_hash"],
            "teacher_checkpoint_sha256": binding["checkpoint_sha256"],
            "training_teacher_free": True,
            "teacher_used_only_for_model_val_select_evaluation": True,
            "target_logit_cache_used": False,
            "final_test_accessed": False,
        }
    )
    return live, lineage


def _device_list(device: str, count: int) -> list[torch.device]:
    if str(device) == "auto" and torch.cuda.is_available():
        available = torch.cuda.device_count()
        if available <= 0:
            return [torch.device("cpu")]
        return [torch.device(f"cuda:{index}") for index in range(min(int(count), available))]
    return [torch.device(resolve_device(str(device)))]


def run_reconstruction_pack_from_execution_spec(
    execution_spec_path: str | Path,
    *,
    graph_path: str | Path,
    node_id: str,
    artifact_root: str | Path,
    r0_checkpoint_path: str | Path,
    r0_registration_path: str | Path,
    physical45_scaler_path: str | Path,
    all50_scaler_path: str | Path,
    absolute_scaler_path: str | Path | None,
    deployed_reference_path: str | Path | None,
    replica_output_dir: str | Path,
    ram_root: str | Path,
    allocation_id: str | None = None,
    device: str = "auto",
    shard_size: int = 8192,
    config: ReconstructionCampaignConfig | None = None,
    capacity_bytes: int | None = None,
    allow_unverified_test_root: bool = False,
    model_loader: Callable[[str | Path, Any], tuple[Any, Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Train all paired replicas in one graph pack from one staged source set."""

    campaign = config or ReconstructionCampaignConfig()
    graph = load_hashed_json(graph_path)
    validate_prediction_anchored_tigris_graph(graph)
    nodes = {row["node_id"]: row for row in graph["nodes"]}
    if str(node_id) not in nodes:
        raise KeyError(f"production graph has no node {node_id!r}")
    node = nodes[str(node_id)]
    if node.get("runner") != "run_train_prediction_anchored_bridge_reconstructor.sh":
        raise ValueError("reconstruction executor received a non-reconstruction graph node")
    l0_postteacher_replay = str(node_id) == L0_POSTTEACHER_NODE_ID
    if l0_postteacher_replay:
        if (
            node.get("configuration_run_ids") != []
            or node.get("arguments") != [L0_POSTTEACHER_NODE_ID]
            or node.get("requires_selected_consumer") is not True
        ):
            raise ValueError("post-teacher L0 graph node changed its replay contract")
        run_ids = (L0_RUN_ID,)
    else:
        run_ids = tuple(str(value) for value in node["configuration_run_ids"])
    if not run_ids or any(value not in RECONSTRUCTION_RUN_IDS for value in run_ids):
        raise ValueError("reconstruction graph node has an invalid/empty run inventory")
    if not l0_postteacher_replay and tuple(node["paired_seed_ids"]) != PAIRED_SEED_IDS:
        raise ValueError("scientific reconstruction execution requires paired seeds 101/202/303")

    spec = load_hashed_json(
        execution_spec_path, expected_contract=PREDICTION_ANCHORED_EXECUTION_SPEC_CONTRACT
    )
    validate_prediction_anchored_execution_spec(spec, verify_file_hashes=False)
    child = load_hashed_json(
        spec["child_manifest"]["path"], expected_contract=PREDICTION_ANCHORED_SPLIT_CONTRACT
    )
    if child["content_hash"] != spec["child_manifest"]["content_hash"]:
        raise ValueError("execution spec child-manifest binding changed")
    r0_path = Path(r0_checkpoint_path)
    r0_sha = sha256_file(r0_path)
    registration = load_hashed_json(r0_registration_path)
    if registration.get("checkpoint_sha256") != r0_sha or registration.get("split_manifest") != child["content_hash"]:
        raise ValueError("reconstruction R0 registration/checkpoint/split binding changed")
    physical = load_hashed_json(physical45_scaler_path)
    if physical.get("channel_policy") != BRIDGE_CHANNEL_PHYSICAL45:
        raise ValueError("reconstruction physical scaler is not physical45")
    quantile_reference = None
    if any(
        resolve_reconstruction_run(value).binding_kind == "primary"
        and value != "D10_L0_bridge_only"
        for value in run_ids
    ) or l0_postteacher_replay:
        quantile_reference = load_hashed_json(
            Path(artifact_root)
            / "bridge_inputs"
            / "bridge_quantile_reference_physical45.json",
            expected_contract="prediction_anchored_bridge_quantiles_v1",
        )
    all50 = load_hashed_json(all50_scaler_path)
    if all50.get("channel_policy") != BRIDGE_CHANNEL_ALL50:
        raise ValueError("reconstruction all50 scaler is not all50")
    needs_absolute = any(
        resolve_reconstruction_run(value).architecture_id
        in {ARCH_A5_HLG_ABSOLUTE, ARCH_A5S_HLG_SCRATCH}
        for value in run_ids
    )
    absolute = None
    if needs_absolute:
        if absolute_scaler_path is None:
            raise FileNotFoundError("A5/A5S pack requires the B2 absolute-output scaler")
        absolute = load_hashed_json(absolute_scaler_path)
        validate_content_hash(
            absolute, expected_contract="prediction_anchored_absolute_output_scaler_v1"
        )
        if absolute.get("source_manifest_sha256") != spec["parent_manifest"]["sha256"]:
            raise ValueError("absolute-output scaler belongs to another source manifest")
    needs_reference = any(resolve_reconstruction_run(value).direct for value in run_ids)
    reference = None
    if needs_reference:
        if deployed_reference_path is None:
            raise FileNotFoundError("direct-control pack requires the canonical-A3 resource reference")
        reference = _reference_from_artifact(load_hashed_json(deployed_reference_path))
        if reference.source_manifest_sha256 != spec["parent_manifest"]["sha256"]:
            raise ValueError("deployed resource reference belongs to another source manifest")
    selected_path = Path(artifact_root) / "selection" / "selected_bridge_consumer.json"
    selected = None
    if l0_postteacher_replay or any(
        resolve_reconstruction_run(value).binding_kind == "primary" for value in run_ids
    ):
        selected = load_hashed_json(
            selected_path, expected_contract=PREDICTION_ANCHORED_SELECTED_CONSUMER_CONTRACT
        )
        if selected.get("status") != "CONFIRMED_LOCKED":
            raise PermissionError("B6 refuses an unconfirmed or guessed primary consumer")
    if node.get("stage") == "B6":
        release = load_hashed_json(
            Path(artifact_root) / "selection" / "post_teacher_release.json",
            expected_contract="prediction_anchored_post_teacher_release_v1",
        )
        release_selected = selected or load_hashed_json(
            selected_path,
            expected_contract=PREDICTION_ANCHORED_SELECTED_CONSUMER_CONTRACT,
        )
        if release.get("selected_consumer_sha256") != release_selected["content_hash"]:
            raise ValueError("B6 post-teacher release belongs to another selected consumer")
        released = set(release.get("released_post_teacher_run_ids", []))
        missing_release = [
            value for value in run_ids if value != L0_RUN_ID and value not in released
        ]
        if missing_release:
            raise PermissionError(
                "B6 run IDs are absent from the explicit post-teacher release: "
                + ", ".join(missing_release)
            )

    early_l0_manifest = None
    if l0_postteacher_replay:
        early_root = Path(artifact_root) / "l0_early" / L0_RUN_ID
        early_l0_manifest = load_hashed_json(
            early_root / "replay_manifest.json",
            expected_contract=PREDICTION_ANCHORED_L0_EARLY_REPLAY_CONTRACT,
        )
        early_rows = early_l0_manifest.get("replicas")
        early_seed_ids = (
            []
            if not isinstance(early_rows, list)
            else [int(row.get("seed_id", -1)) for row in early_rows]
        )
        early_ordered = early_l0_manifest.get("early_ordered_seed_ids")
        early_ordered_ids = (
            []
            if not isinstance(early_ordered, list)
            else [int(value) for value in early_ordered]
        )
        if (
            early_l0_manifest.get("run_id") != L0_RUN_ID
            or early_l0_manifest.get("aggregation_phase")
            != "early_reachability_bridge_loss"
            or early_l0_manifest.get("paired_seed_ids") != list(PAIRED_SEED_IDS)
            or sorted(early_seed_ids) != sorted(PAIRED_SEED_IDS)
            or sorted(early_ordered_ids) != sorted(PAIRED_SEED_IDS)
            or int(early_l0_manifest.get("early_median_seed_id", -1))
            not in PAIRED_SEED_IDS
            or early_l0_manifest.get("nonmedian_weights_persisted") is not False
            or early_l0_manifest.get("median_weights_persisted") is not False
            or early_l0_manifest.get("replay_required_after_selected_consumer")
            is not True
        ):
            raise ValueError("early L0 replay evidence is stale or semantically invalid")

    # A packed allocation opens each compact target-logit NPZ once, regardless
    # of how many KD configurations share it.  Zero-KD configurations never
    # receive an entry from this map.
    preloaded_caches: dict[
        tuple[str, str], tuple[Mapping[str, Any], Mapping[str, np.ndarray]]
    ] = {}
    for run_id in run_ids:
        run_spec = resolve_reconstruction_run(run_id)
        if not run_spec.requires_cache:
            continue
        assert run_spec.binding_kind is not None and run_spec.cache_namespace is not None
        key = (run_spec.binding_kind, run_spec.cache_namespace)
        if key in preloaded_caches:
            continue
        binding_path, cache_root = _binding_paths(run_spec, Path(artifact_root))
        assert binding_path is not None and cache_root is not None
        binding = load_hashed_json(binding_path)
        primary = selected if run_spec.binding_kind == "primary" else None
        live_config = build_live_teacher_config(binding, primary_selection=primary)
        preloaded_caches[key] = load_teacher_logit_cache(
            cache_root,
            binding=binding,
            live_teacher_config=live_config,
            stack_train_distill_manifest_sha256=child["children"]["stack_train_distill"]["content_hash"],
            primary_selection=primary,
        )

    replica_root = Path(replica_output_dir)
    replica_root.mkdir(parents=True, exist_ok=True)
    if any(replica_root.iterdir()):
        raise FileExistsError(f"reconstruction replica output is not empty: {replica_root}")
    allocation = str(allocation_id or os.environ.get("SLURM_JOB_ID", "local_reconstruction"))
    ledger = AllocationRamLedger(
        ram_root,
        allocation_id=allocation,
        capacity_bytes=capacity_bytes,
        allow_unverified_test_root=bool(allow_unverified_test_root),
    )
    deterministic_l0_replay = tuple(run_ids) == (L0_RUN_ID,)
    prior_deterministic_algorithms = torch.are_deterministic_algorithms_enabled()
    prior_cudnn_benchmark = torch.backends.cudnn.benchmark
    prior_cudnn_deterministic = torch.backends.cudnn.deterministic
    prior_cublas_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    try:
        if deterministic_l0_replay:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        stager = AllocationNpzStager(ledger, rank=0, world_size=1)
        staged, combined = stager.stage_named_pairs(
            {
                "stack_train": _paths_from_source(spec["sources"]["stack_train"]),
                "model_val": _paths_from_source(spec["sources"]["model_val"]),
            },
            shard_size=int(shard_size),
        )
        for split in ("stack_train", "model_val"):
            _verify_staged_source_binding(
                split=split, source=spec["sources"][split], report=staged[split][2]
            )
        if not combined["all_persistent_npz_open_counts_equal_one"]:
            raise RuntimeError("reconstruction allocation violated the one-open source contract")
        hlt_by_split = {name: value[0] for name, value in staged.items()}
        offline_by_split = {name: value[1] for name, value in staged.items()}
        distill = np.asarray(
            child["children"]["stack_train_distill"]["parent_row_indices"], dtype=np.int64
        )
        stop = np.asarray(
            child["children"]["model_val_stop"]["parent_row_indices"], dtype=np.int64
        )
        select = np.asarray(
            child["children"]["model_val_select"]["parent_row_indices"], dtype=np.int64
        )
        devices = _device_list(str(device), len(run_ids))
        initialization_lock = threading.Lock()

        def execute_run(position: int, run_id: str) -> dict[str, Any]:
            run_spec = resolve_reconstruction_run(run_id)
            selected_device = devices[position % len(devices)]
            if selected_device.type == "cuda":
                torch.cuda.set_device(selected_device)
            live, lineage, cache_arrays, cache_manifest = _load_teacher_context(
                run_spec,
                artifact_root=Path(artifact_root),
                selected=selected,
                child=child,
                all50_scaler=all50,
                device=selected_device,
                model_loader=model_loader,
                preloaded_cache=(
                    None
                    if not run_spec.requires_cache
                    else preloaded_caches[(
                        str(run_spec.binding_kind), str(run_spec.cache_namespace)
                    )]
                ),
            )
            l0_evaluation_consumer = None
            l0_evaluation_lineage = None
            if l0_postteacher_replay:
                assert selected is not None
                l0_evaluation_consumer, l0_evaluation_lineage = (
                    _load_l0_postteacher_evaluation_consumer(
                        artifact_root=Path(artifact_root),
                        selected=selected,
                        device=selected_device,
                        model_loader=model_loader,
                    )
                )
            r0 = FrozenR0Runner(r0_path, device=selected_device)
            data = _WorkerData(
                hlt_by_split=hlt_by_split,
                offline_by_split=offline_by_split,
                r0=r0,
                distill_indices=distill,
                cache_arrays=cache_arrays,
                channel_policy=run_spec.channel_policy or BRIDGE_CHANNEL_PHYSICAL45,
                batch_size=int(campaign.batch_size),
                shard_size=int(shard_size),
            )
            data.verify_cache_alignment()
            train_indices = (
                np.arange(hlt_by_split["stack_train"].n_events, dtype=np.int64)
                if run_spec.direct
                else distill
            )
            saved = []
            for seed in PAIRED_SEED_IDS:
                # CPU module initialization uses a process-global generator.
                # Serialize only this short construction section and restore
                # the prior state; stochastic training itself then uses the
                # independent RNG of the configuration's dedicated GPU.
                with initialization_lock:
                    cpu_rng = torch.get_rng_state()
                    torch.set_rng_state(torch.Generator().manual_seed(int(seed)).get_state())
                    try:
                        model, model_metadata = build_reconstruction_model(
                            run_id,
                            physical45_scaler=physical,
                            all50_scaler=all50,
                            absolute_scaler=absolute,
                            deployed_reference=reference,
                            c0_model_width=int(campaign.c0_model_width),
                            dropout=float(campaign.dropout),
                        )
                        model_metadata = {
                            **dict(model_metadata),
                            "trainable_parameter_count": sum(
                                int(parameter.numel()) for parameter in model.parameters()
                            ),
                            "deployed_parameter_count": sum(
                                int(parameter.numel()) for parameter in model.parameters()
                            )
                            + (
                                0
                                if run_spec.direct and run_id == DIRECT_HLT
                                else sum(
                                    int(parameter.numel())
                                    for parameter in r0.model.parameters()
                                )
                            )
                            + (
                                0
                                if live is None and l0_evaluation_consumer is None
                                else sum(
                                    int(parameter.numel())
                                    for parameter in (
                                        live
                                        if live is not None
                                        else l0_evaluation_consumer
                                    ).consumer.parameters()
                                )
                            ),
                        }
                    finally:
                        torch.set_rng_state(cpu_rng)
                model.to(selected_device)
                if selected_device.type == "cuda":
                    torch.cuda.manual_seed(int(seed))
                parent_hashes = {
                    "execution_spec_sha256": spec["content_hash"],
                    "child_manifest_sha256": child["content_hash"],
                }
                if run_id != DIRECT_HLT:
                    parent_hashes.update(
                        r0_checkpoint_sha256=r0_sha,
                        physical45_scaler_sha256=physical["content_hash"],
                    )
                if run_spec.direct:
                    assert reference is not None
                    parent_hashes["deployed_resource_reference_sha256"] = (
                        reference.to_artifact()["content_hash"]
                    )
                if run_id in ALL50_RUN_IDS:
                    parent_hashes["all50_scaler_sha256"] = all50["content_hash"]
                if run_spec.architecture_id in {
                    ARCH_A5_HLG_ABSOLUTE,
                    ARCH_A5S_HLG_SCRATCH,
                }:
                    assert absolute is not None
                    parent_hashes["absolute_scaler_sha256"] = absolute["content_hash"]
                if cache_manifest is not None:
                    parent_hashes["target_cache_sha256"] = cache_manifest["content_hash"]
                if live is not None:
                    parent_hashes["teacher_checkpoint_sha256"] = live.checkpoint_sha256
                if l0_evaluation_consumer is not None:
                    parent_hashes["teacher_checkpoint_sha256"] = (
                        l0_evaluation_consumer.checkpoint_sha256
                    )
                    assert early_l0_manifest is not None
                    parent_hashes["l0_early_replay_sha256"] = early_l0_manifest[
                        "content_hash"
                    ]
                candidate, metrics = _train_one_replica(
                    spec=run_spec,
                    seed=int(seed),
                    model=model,
                    data=data,
                    train_indices=train_indices,
                    stop_indices=stop,
                    select_indices=select,
                    live_consumer=live,
                    lineage=lineage,
                    config=campaign,
                    class_order=spec["class_names"],
                    parent_hashes=parent_hashes,
                    model_metadata=model_metadata,
                    quantile_reference=quantile_reference,
                    l0_evaluation_consumer=l0_evaluation_consumer,
                    l0_evaluation_lineage=l0_evaluation_lineage,
                )
                if l0_postteacher_replay:
                    assert early_l0_manifest is not None
                    expected = next(
                        row
                        for row in early_l0_manifest["replicas"]
                        if int(row["seed_id"]) == int(seed)
                    )
                    selection = metrics["checkpoint_selection"]
                    if (
                        selection.get("selected_model_state_sha256")
                        != expected["selected_model_state_sha256"]
                        or int(selection.get("selected_epoch", -1))
                        != int(expected["selected_epoch"])
                        or metrics.get("campaign_config")
                        != expected.get("campaign_config")
                    ):
                        raise RuntimeError(
                            f"L0 seed {seed} deterministic replay changed its training result"
                        )
                    expected_parents = expected.get("training_parent_hashes")
                    if not isinstance(expected_parents, Mapping) or any(
                        candidate["parent_hashes"].get(key) != value
                        for key, value in expected_parents.items()
                    ):
                        raise RuntimeError(
                            f"L0 seed {seed} deterministic replay changed its training lineage"
                        )
                    if not isinstance(metrics.get("model_val_select"), Mapping):
                        raise RuntimeError("post-teacher L0 replay omitted common evaluation")
                checkpoint = replica_root / f"{run_id}__seed{seed}.pt"
                metrics_path = replica_root / f"{run_id}__seed{seed}.metrics.json"
                if checkpoint.exists() or metrics_path.exists():
                    raise FileExistsError("reconstruction replica artifact already exists")
                torch.save(candidate, checkpoint)
                write_immutable_json(metrics_path, metrics)
                saved.append(
                    {
                        "seed_id": int(seed),
                        "checkpoint": str(checkpoint),
                        "checkpoint_sha256": sha256_file(checkpoint),
                        "metrics": str(metrics_path),
                        "metrics_sha256": metrics["content_hash"],
                    }
                )
                del model
                if selected_device.type == "cuda":
                    torch.cuda.empty_cache()
            return {
                "run_id": run_id,
                "device": str(selected_device),
                "replicas": saved,
                "target_cache_loaded": cache_arrays is not None,
                "teacher_checkpoint_sha256": None if live is None else live.checkpoint_sha256,
                "l0_evaluation_teacher_checkpoint_sha256": (
                    None
                    if l0_evaluation_consumer is None
                    else l0_evaluation_consumer.checkpoint_sha256
                ),
            }

        if len(devices) > 1 and len(run_ids) > 1:
            with ThreadPoolExecutor(max_workers=min(len(devices), len(run_ids))) as pool:
                futures = [pool.submit(execute_run, index, run_id) for index, run_id in enumerate(run_ids)]
                results = [future.result() for future in futures]
        else:
            results = [execute_run(index, run_id) for index, run_id in enumerate(run_ids)]
        report = with_content_hash(
            {
                "contract": PREDICTION_ANCHORED_RECONSTRUCTION_EXECUTION_CONTRACT,
                "graph_sha256": graph["content_hash"],
                "node_id": str(node_id),
                "node_sha256": node["content_hash"],
                "execution_spec_sha256": spec["content_hash"],
                "configuration_run_ids": list(run_ids),
                "l0_postteacher_deterministic_replay": bool(l0_postteacher_replay),
                "deterministic_l0_training_enforced": bool(deterministic_l0_replay),
                "cublas_workspace_config": (
                    os.environ.get("CUBLAS_WORKSPACE_CONFIG")
                    if deterministic_l0_replay
                    else None
                ),
                "l0_early_replay_sha256": (
                    None
                    if early_l0_manifest is None
                    else early_l0_manifest["content_hash"]
                ),
                "paired_seed_ids": list(PAIRED_SEED_IDS),
                "replica_count": len(run_ids) * len(PAIRED_SEED_IDS),
                "results": results,
                "one_open_per_compressed_source": True,
                "target_cache_npz_open_count_by_namespace": {
                    namespace: 1 for _, namespace in sorted(preloaded_caches)
                },
                "source_namespaces": ["model_val", "stack_train"],
                "multi_gpu_configuration_packing": len(devices) > 1,
                "devices": [str(value) for value in devices],
                "persistent_dense_fields_written": False,
                "replica_weights_location": "verified_job_local_ram_until_median_publication",
                "restart_policy": "whole_configuration_pack",
                "ram_peak_reserved_bytes": ledger.snapshot()["peak_reserved_bytes"],
            }
        )
        return report
    finally:
        ledger.cleanup()
        if deterministic_l0_replay:
            torch.use_deterministic_algorithms(prior_deterministic_algorithms)
            torch.backends.cudnn.benchmark = prior_cudnn_benchmark
            torch.backends.cudnn.deterministic = prior_cudnn_deterministic
            if prior_cublas_workspace is None:
                os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
            else:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = prior_cublas_workspace


@dataclass(frozen=True)
class ReconstructionReplicaResult:
    run_id: str
    seed_id: int
    metrics: Mapping[str, Any]
    weights_payload: Mapping[str, Any]
    source_checkpoint_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.run_id not in RECONSTRUCTION_RUN_IDS or int(self.seed_id) not in PAIRED_SEED_IDS:
            raise ValueError("invalid reconstruction paired replica")
        if self.weights_payload.get("checkpoint_contract") != PREDICTION_ANCHORED_RECONSTRUCTION_REPLICA_CONTRACT:
            raise ValueError("reconstruction replica checkpoint contract changed")
        if self.weights_payload.get("run_id") != self.run_id or int(
            self.weights_payload.get("seed_id", -1)
        ) != int(self.seed_id):
            raise ValueError("reconstruction replica identity changed")
        validate_content_hash(
            self.metrics, expected_contract=PREDICTION_ANCHORED_RECONSTRUCTION_METRICS_CONTRACT
        )
        if self.source_checkpoint_sha256 is not None:
            value = str(self.source_checkpoint_sha256)
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError("reconstruction source checkpoint hash is invalid")


def _metric(value: Mapping[str, Any], path: str) -> float:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(f"reconstruction replica metrics lack {path}")
        current = current[part]
    result = float(current)
    if not math.isfinite(result):
        raise ValueError(f"reconstruction replica metric {path} is non-finite")
    return result


def aggregate_reconstruction_replicas(
    replicas: Sequence[ReconstructionReplicaResult],
    *,
    l0_postteacher: bool = False,
) -> dict[str, Any]:
    if len(replicas) != 3 or {item.seed_id for item in replicas} != set(PAIRED_SEED_IDS):
        raise ValueError("reconstruction aggregate requires paired seeds 101/202/303")
    run_ids = {item.run_id for item in replicas}
    if len(run_ids) != 1:
        raise ValueError("reconstruction aggregate mixes run IDs")
    run_id = next(iter(run_ids))
    if bool(l0_postteacher) and run_id != L0_RUN_ID:
        raise ValueError("post-teacher L0 aggregation cannot be used for another run")
    if run_id == L0_RUN_ID and not bool(l0_postteacher):
        scored = [
            (
                -_metric(item.metrics, "model_val_stop.bridge_loss"),
                -_metric(item.metrics, "model_val_stop.normalized_bridge_mse"),
                0.0,
                int(item.seed_id),
                item,
            )
            for item in replicas
        ]
        ordering = [
            "negative_model_val_stop.bridge_loss",
            "negative_model_val_stop.normalized_bridge_mse",
            "constant_zero",
            "seed_id",
        ]
        aggregation_phase = "early_reachability_bridge_loss"
    elif run_id in {DIRECT_HLT, DIRECT_R0REP}:
        scored = [
            (
                _metric(item.metrics, "model_val_select.accuracy"),
                -_metric(item.metrics, "model_val_select.cross_entropy"),
                0.0,
                int(item.seed_id),
                item,
            )
            for item in replicas
        ]
        ordering = [
            "model_val_select.accuracy",
            "negative_model_val_select.cross_entropy",
            "constant_zero",
            "seed_id",
        ]
        aggregation_phase = "direct_control_model_val_select"
    else:
        scored = [
            (
                _metric(item.metrics, "model_val_select.accuracy"),
                _metric(item.metrics, "model_val_select.deployable_gain"),
                -_metric(item.metrics, "model_val_select.cross_entropy"),
                int(item.seed_id),
                item,
            )
            for item in replicas
        ]
        ordering = [
            "model_val_select.accuracy",
            "model_val_select.deployable_gain",
            "negative_model_val_select.cross_entropy",
            "seed_id",
        ]
        aggregation_phase = (
            "postteacher_common_model_val_select"
            if run_id == L0_RUN_ID
            else "ordinary_model_val_select"
        )
    scored.sort(key=lambda row: row[:4])
    median = scored[1][4]
    return with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_RECONSTRUCTION_AGGREGATE_CONTRACT,
            "run_id": run_id,
            "aggregation_phase": aggregation_phase,
            "paired_seed_ids": list(PAIRED_SEED_IDS),
            "ordering": ordering,
            "ordered_seed_ids": [int(row[4].seed_id) for row in scored],
            "median_seed_id": int(median.seed_id),
            "best_seed_id": int(scored[-1][4].seed_id),
            "best_seed_weights_rejected": True,
            "replica_metrics": [
                {
                    "seed_id": int(item.seed_id),
                    "metrics": deepcopy(dict(item.metrics)),
                    "source_checkpoint_sha256": item.source_checkpoint_sha256,
                    "weights_persisted": int(item.seed_id) == int(median.seed_id),
                }
                for item in sorted(replicas, key=lambda value: value.seed_id)
            ],
        }
    )


def _publication_weights(payload: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "checkpoint_contract", "run_id", "seed_id", "epoch", "model_family",
        "architecture_id", "model_config", "model_state_dict", "parent_hashes",
        "selected_teacher_checkpoint_sha256", "target_cache_sha256",
    }
    if "model_state_dict" not in payload:
        raise ValueError("reconstruction publication payload has no model state")
    result = {name: deepcopy(value) for name, value in payload.items() if name in allowed}
    result.update(
        weights_only=True,
        optimizer_state_persisted=False,
        scheduler_state_persisted=False,
        nonmedian_weights_persisted=False,
        frozen_parent_weights_persisted=False,
        generated_fields_persisted=False,
    )
    return result


def publish_reconstruction_paired_replicas(
    replicas: Sequence[ReconstructionReplicaResult],
    *,
    output_dir: str | Path,
    l0_postteacher: bool = False,
    reservation_bytes: int | None = None,
) -> dict[str, Any]:
    aggregate = aggregate_reconstruction_replicas(
        replicas, l0_postteacher=bool(l0_postteacher)
    )
    if bool(l0_postteacher):
        teacher_hashes = set()
        replay_hashes = set()
        for replica in replicas:
            lineage = replica.metrics.get("postteacher_evaluation_lineage")
            if not isinstance(lineage, Mapping):
                raise ValueError("post-teacher L0 publication lacks evaluation lineage")
            validate_content_hash(
                lineage,
                expected_contract=PREDICTION_ANCHORED_L0_POSTTEACHER_LINEAGE_CONTRACT,
            )
            if (
                lineage.get("training_teacher_free") is not True
                or lineage.get("teacher_used_only_for_model_val_select_evaluation")
                is not True
                or lineage.get("target_logit_cache_used") is not False
                or lineage.get("final_test_accessed") is not False
            ):
                raise ValueError("post-teacher L0 evaluation lineage changed its scope")
            teacher_sha = replica.weights_payload.get(
                "selected_teacher_checkpoint_sha256"
            )
            parent_hashes = replica.weights_payload.get("parent_hashes")
            if not isinstance(parent_hashes, Mapping):
                raise ValueError("post-teacher L0 checkpoint lacks parent hashes")
            if (
                not _valid_sha256(teacher_sha)
                or parent_hashes.get("teacher_checkpoint_sha256") != teacher_sha
                or lineage.get("teacher_checkpoint_sha256") != teacher_sha
            ):
                raise ValueError("post-teacher L0 checkpoint changed its evaluation teacher")
            replay_sha = parent_hashes.get("l0_early_replay_sha256")
            if not _valid_sha256(replay_sha):
                raise ValueError("post-teacher L0 checkpoint lacks early replay lineage")
            if not isinstance(replica.metrics.get("model_val_select"), Mapping):
                raise ValueError("post-teacher L0 publication lacks common selection metrics")
            teacher_hashes.add(str(teacher_sha))
            replay_hashes.add(str(replay_sha))
        if len(teacher_hashes) != 1 or len(replay_hashes) != 1:
            raise ValueError("post-teacher L0 replicas do not share frozen lineage")
    if aggregate["run_id"] == L0_RUN_ID and not bool(l0_postteacher):
        raise ValueError(
            "early L0 must publish replay evidence, not a pre-teacher deployable checkpoint"
        )
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"reconstruction publication directory is not empty: {root}")
    median = next(item for item in replicas if item.seed_id == aggregate["median_seed_id"])
    encoded = BytesIO()
    torch.save(_publication_weights(median.weights_payload), encoded)
    checkpoint_bytes = encoded.getvalue()
    checkpoint_sha = hashlib.sha256(checkpoint_bytes).hexdigest()
    aggregate_bytes = immutable_json_bytes(aggregate)
    publication, publication_bytes = build_total_sized_publication(
        {
            "contract": PREDICTION_ANCHORED_RECONSTRUCTION_PUBLICATION_CONTRACT,
            "run_id": median.run_id,
            "aggregate_sha256": aggregate["content_hash"],
            "paired_seed_ids": list(PAIRED_SEED_IDS),
            "median_seed_id": int(median.seed_id),
            "retained_checkpoint": "median_weights.pt",
            "retained_checkpoint_sha256": checkpoint_sha,
            "measured_state_bytes": len(checkpoint_bytes),
            "reserved_bytes": reservation_bytes,
            "reservation_enforced_before_publication": reservation_bytes is not None,
            "persistent_artifact_allowlist": [
                "aggregate_metrics.json", "median_weights.pt", "publication.json"
            ],
            "nonmedian_weights_persisted": False,
            "optimizer_state_persisted": False,
            "duplicate_frozen_parents_persisted": False,
            "generated_fields_persisted": False,
            "deployable_checkpoint_requires_teacher_at_inference": False,
            "weights_payload_reload_verified": True,
            "l0_postteacher_common_evaluation": bool(l0_postteacher),
        },
        other_artifact_bytes=len(checkpoint_bytes) + len(aggregate_bytes),
    )
    total_bytes = int(publication["measured_total_persistent_bytes"])
    if reservation_bytes is not None and total_bytes > int(reservation_bytes):
        raise PermissionError("reconstruction publication directory exceeds its predeclared run reservation")
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = root / "median_weights.pt"
    for path, encoded_bytes in (
        (checkpoint, checkpoint_bytes),
        (root / "aggregate_metrics.json", aggregate_bytes),
        (root / "publication.json", publication_bytes),
    ):
        with path.open("xb") as handle:
            handle.write(encoded_bytes)
    reloaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if reloaded.get("checkpoint_contract") != PREDICTION_ANCHORED_RECONSTRUCTION_REPLICA_CONTRACT:
        raise ValueError("published reconstruction checkpoint failed its contract reload audit")
    if set(reloaded.get("model_state_dict", {})) != set(
        median.weights_payload.get("model_state_dict", {})
    ):
        raise ValueError("published reconstruction checkpoint changed its state keys")
    names = sorted(path.name for path in root.iterdir())
    expected = ["aggregate_metrics.json", "median_weights.pt", "publication.json"]
    if names != expected:
        raise RuntimeError(f"unexpected reconstruction publication artifacts: {names}")
    return {
        "ok": True,
        "run_id": median.run_id,
        "median_seed_id": int(median.seed_id),
        "checkpoint": str(checkpoint),
        "aggregate": str(root / "aggregate_metrics.json"),
        "publication": str(root / "publication.json"),
        "measured_state_bytes": len(checkpoint_bytes),
        "measured_total_persistent_bytes": total_bytes,
        "persistent_artifacts": names,
    }


def publish_l0_early_replay_manifest(
    replicas: Sequence[ReconstructionReplicaResult],
    *,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Persist only compact evidence needed to verify deterministic L0 replay."""

    aggregate = aggregate_reconstruction_replicas(replicas, l0_postteacher=False)
    if aggregate["run_id"] != L0_RUN_ID:
        raise ValueError("early replay publication is reserved for D10_L0_bridge_only")
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"early L0 replay directory is not empty: {root}")
    rows = []
    for replica in sorted(replicas, key=lambda item: int(item.seed_id)):
        selection = replica.metrics.get("checkpoint_selection")
        if not isinstance(selection, Mapping):
            raise ValueError("early L0 metrics omit checkpoint-selection evidence")
        state_sha = selection.get("selected_model_state_sha256")
        if not _valid_sha256(state_sha):
            raise ValueError("early L0 selected model-state hash is invalid")
        if not _valid_sha256(replica.source_checkpoint_sha256):
            raise ValueError("early L0 source-checkpoint hash is invalid")
        reachability = replica.metrics.get("model_val_stop")
        if not isinstance(reachability, Mapping):
            raise ValueError("early L0 metrics omit model_val_stop reachability evidence")
        campaign_config = replica.metrics.get("campaign_config")
        if not isinstance(campaign_config, Mapping):
            raise ValueError("early L0 metrics omit deterministic campaign configuration")
        teacher_lineage = replica.metrics.get("teacher_lineage")
        if not isinstance(teacher_lineage, Mapping) or (
            teacher_lineage.get("mode") != "preteacher_l0_exception"
            or teacher_lineage.get("teacher_checkpoint_sha256") is not None
        ):
            raise ValueError("early L0 replay evidence is not teacher-free")
        parent_hashes = replica.weights_payload.get("parent_hashes")
        if not isinstance(parent_hashes, Mapping) or (
            replica.weights_payload.get("selected_teacher_checkpoint_sha256") is not None
            or "teacher_checkpoint_sha256" in parent_hashes
            or "l0_early_replay_sha256" in parent_hashes
        ):
            raise ValueError("early L0 checkpoint contains post-teacher lineage")
        rows.append(
            {
                "seed_id": int(replica.seed_id),
                "metrics_sha256": replica.metrics["content_hash"],
                "selected_model_state_sha256": str(state_sha),
                "selected_epoch": int(selection["selected_epoch"]),
                "source_checkpoint_sha256": str(replica.source_checkpoint_sha256),
                "model_val_stop": deepcopy(dict(reachability)),
                "campaign_config": deepcopy(dict(campaign_config)),
                "training_parent_hashes": deepcopy(dict(parent_hashes)),
            }
        )
    manifest = with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_L0_EARLY_REPLAY_CONTRACT,
            "run_id": L0_RUN_ID,
            "early_aggregate_sha256": aggregate["content_hash"],
            "aggregation_phase": aggregate["aggregation_phase"],
            "early_ordered_seed_ids": aggregate["ordered_seed_ids"],
            "early_median_seed_id": aggregate["median_seed_id"],
            "paired_seed_ids": list(PAIRED_SEED_IDS),
            "replicas": rows,
            "nonmedian_weights_persisted": False,
            "median_weights_persisted": False,
            "generated_fields_persisted": False,
            "replay_required_after_selected_consumer": True,
        }
    )
    root.mkdir(parents=True, exist_ok=True)
    write_immutable_json(root / "replay_manifest.json", manifest)
    names = sorted(path.name for path in root.iterdir())
    if names != ["replay_manifest.json"]:
        raise RuntimeError(f"early L0 replay wrote unexpected artifacts: {names}")
    return {
        "ok": True,
        "run_id": L0_RUN_ID,
        "replay_manifest": str(root / "replay_manifest.json"),
        "persistent_artifacts": names,
        "persistent_weights": 0,
    }


__all__ = [
    "PREDICTION_ANCHORED_RECONSTRUCTION_EXECUTION_CONTRACT",
    "PREDICTION_ANCHORED_RECONSTRUCTION_REPLICA_CONTRACT",
    "PREDICTION_ANCHORED_RECONSTRUCTION_METRICS_CONTRACT",
    "PREDICTION_ANCHORED_RECONSTRUCTION_AGGREGATE_CONTRACT",
    "PREDICTION_ANCHORED_RECONSTRUCTION_PUBLICATION_CONTRACT",
    "PREDICTION_ANCHORED_L0_EARLY_REPLAY_CONTRACT",
    "PREDICTION_ANCHORED_L0_POSTTEACHER_LINEAGE_CONTRACT",
    "L0_RUN_ID",
    "L0_POSTTEACHER_NODE_ID",
    "RECONSTRUCTION_RUN_IDS",
    "ReconstructionCampaignConfig",
    "ReconstructionRunSpec",
    "ReconstructionReplicaResult",
    "resolve_reconstruction_run",
    "build_reconstruction_model",
    "run_reconstruction_pack_from_execution_spec",
    "aggregate_reconstruction_replicas",
    "publish_reconstruction_paired_replicas",
    "publish_l0_early_replay_manifest",
]
