"""Two-phase C0 training, RAM resume, selection, measurement, and publication."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from io import BytesIO
import hashlib
import math
from pathlib import Path
import random
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

import numpy as np
import torch

from .bridge_campaign import PAIRED_SEED_IDS, record_registry_measurements
from .bridge_contracts import (
    sha256_file,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from .bridge_losses import (
    C0_CANONICAL_RUN_IDS,
    C0LossRecipe,
    bridge_reachability_metrics,
    compute_c0_objective,
    resolve_c0_loss_recipe,
)
from .bridge_evaluation import (
    PREDICTION_ANCHORED_SELECTED_CONSUMER_CONTRACT,
    PRIMARY_TEACHER_NAMESPACE,
)
from .bridge_logits import (
    PREDICTION_ANCHORED_LIVE_TEACHER_CONFIG_CONTRACT,
    PREDICTION_ANCHORED_TEACHER_LOGIT_CACHE_CONTRACT,
)
from .bridge_reconstructor import (
    C0CorrectionConfig,
    FrozenLiveBridgeConsumer,
    PredictionAnchoredC0Correction,
)
from .bridge_splits import PREDICTION_ANCHORED_ACCESS_RECEIPT_CONTRACT


PREDICTION_ANCHORED_C0_TRAIN_CONFIG_CONTRACT = "prediction_anchored_c0_train_config_v1"
PREDICTION_ANCHORED_C0_RAM_SNAPSHOT_CONTRACT = "prediction_anchored_c0_ram_snapshot_v1"
PREDICTION_ANCHORED_C0_CHECKPOINT_SELECTION_CONTRACT = (
    "prediction_anchored_c0_checkpoint_selection_v1"
)
PREDICTION_ANCHORED_C0_REPLICA_CONTRACT = "prediction_anchored_c0_replica_v1"
PREDICTION_ANCHORED_C0_AGGREGATE_CONTRACT = "prediction_anchored_c0_paired_aggregate_v1"
PREDICTION_ANCHORED_C0_PUBLICATION_CONTRACT = "prediction_anchored_c0_publication_v1"
PREDICTION_ANCHORED_C0_MEASUREMENT_CONTRACT = "prediction_anchored_c0_measurement_v1"
PREDICTION_ANCHORED_C0_CPU_MINIATURE_CONTRACT = "prediction_anchored_c0_cpu_miniature_v1"
PREDICTION_ANCHORED_C0_CAMPAIGN_MANIFEST_CONTRACT = (
    "prediction_anchored_c0_campaign_manifest_v1"
)
PREDICTION_ANCHORED_C0_TEACHER_LINEAGE_CONTRACT = (
    "prediction_anchored_c0_teacher_lineage_v1"
)

BatchFactory = Callable[[str, int], Iterable[Mapping[str, Any]]]


def _torch_bytes(value: Any) -> bytes:
    buffer = BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def _torch_hash(value: Any) -> str:
    return hashlib.sha256(_torch_bytes(value)).hexdigest()


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:  # pragma: no cover
        return torch.device("cpu")


def _tensor(value: Any, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(value, device=device, dtype=dtype)


def _prepared_batch(raw: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    required = {"hlt_tokens", "mask", "f0", "h0", "labels"}
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(f"C0 batch is missing {missing}")
    output = dict(raw)
    output["hlt_tokens"] = _tensor(raw["hlt_tokens"], device=device, dtype=torch.float32)
    output["mask"] = _tensor(raw["mask"], device=device, dtype=torch.bool)
    output["f0"] = _tensor(raw["f0"], device=device, dtype=torch.float32)
    output["h0"] = _tensor(raw["h0"], device=device, dtype=torch.float32)
    if "bridge_fields" in raw:
        output["bridge_fields"] = _tensor(
            raw["bridge_fields"], device=device, dtype=torch.float32
        )
    output["labels"] = _tensor(raw["labels"], device=device, dtype=torch.long)
    if "true_fields" in raw:
        output["true_fields"] = _tensor(
            raw["true_fields"], device=device, dtype=torch.float32
        )
    if "target_logits" in raw:
        output["target_logits"] = _tensor(
            raw["target_logits"], device=device, dtype=torch.float32
        ).detach()
    mask = output["mask"]
    if output["hlt_tokens"].shape[:2] != mask.shape:
        raise ValueError("C0 token/mask shapes do not align")
    if output["f0"].shape != (*mask.shape, 50) or output["h0"].shape != (*mask.shape, 160):
        raise ValueError("C0 f0/h0 shapes do not align")
    if "bridge_fields" in output and output["bridge_fields"].shape != output["f0"].shape:
        raise ValueError("C0 bridge/f0 shapes do not align")
    return output


@dataclass(frozen=True)
class C0TrainConfig:
    run_id: str
    seed_id: int
    field_warmup_steps: int
    phase2_epochs: int
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    grad_clip_norm: float = 1.0
    kd_temperature: float = 2.0
    early_stop_patience: int = 8
    allocation_id: str = ""

    def __post_init__(self) -> None:
        recipe = resolve_c0_loss_recipe(self.run_id)
        object.__setattr__(self, "run_id", recipe.run_id)
        if int(self.seed_id) not in PAIRED_SEED_IDS:
            raise ValueError("scientific C0 training requires seed 101/202/303")
        if int(self.field_warmup_steps) < 0 or int(self.phase2_epochs) <= 0:
            raise ValueError("warm-up steps must be nonnegative and phase2_epochs positive")
        if recipe.field_warmup and int(self.field_warmup_steps) <= 0:
            raise ValueError(f"{recipe.run_id} requires a positive immutable warm-up budget")
        if not recipe.field_warmup and int(self.field_warmup_steps) != 0:
            raise ValueError(f"{recipe.run_id} must skip field warm-up exactly")
        if (
            float(self.learning_rate) <= 0
            or float(self.weight_decay) < 0
            or float(self.grad_clip_norm) < 0
        ):
            raise ValueError("invalid C0 optimizer settings")
        if float(self.kd_temperature) <= 0 or int(self.early_stop_patience) < -1:
            raise ValueError("invalid KD temperature or patience")
        if not str(self.allocation_id):
            raise ValueError("C0 RAM-local training requires an allocation ID")

    def to_artifact(self) -> dict[str, Any]:
        recipe = resolve_c0_loss_recipe(self.run_id)
        return with_content_hash(
            {
                "contract": PREDICTION_ANCHORED_C0_TRAIN_CONFIG_CONTRACT,
                **asdict(self),
                "loss_recipe_sha256": recipe.to_artifact()["content_hash"],
                "phase1_exact_fixed_budget": bool(recipe.field_warmup),
                "phase1_validation_cannot_stop": True,
                "phase2_checkpoint_split": "model_val_stop",
                "cross_allocation_resume": False,
            }
        )


@dataclass(frozen=True)
class C0CampaignConfig:
    """Immutable shared schedule for the eleven paired C0 configurations."""

    field_warmup_steps: int
    phase2_epochs: int
    batch_size: int = 128
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    grad_clip_norm: float = 1.0
    kd_temperature: float = 2.0
    early_stop_patience: int = 8
    model_width: int = 160
    particle_mlp_layers: int = 2
    head_hidden_dim: int = 64
    dropout: float = 0.05

    def __post_init__(self) -> None:
        if int(self.field_warmup_steps) <= 0 or int(self.phase2_epochs) <= 0:
            raise ValueError("C0 campaign requires positive warm-up and Phase 2 budgets")
        if int(self.batch_size) <= 0 or int(self.model_width) <= 0:
            raise ValueError("C0 campaign batch size/model width must be positive")
        if int(self.particle_mlp_layers) <= 0 or int(self.head_hidden_dim) <= 0:
            raise ValueError("C0 campaign model depth/head width must be positive")
        if float(self.learning_rate) <= 0 or float(self.weight_decay) < 0:
            raise ValueError("C0 campaign optimizer settings are invalid")
        if float(self.grad_clip_norm) < 0 or float(self.kd_temperature) <= 0:
            raise ValueError("C0 campaign clipping/KD settings are invalid")
        if int(self.early_stop_patience) < -1:
            raise ValueError("C0 campaign early-stop patience is invalid")
        if not 0 <= float(self.dropout) < 1:
            raise ValueError("C0 campaign dropout must be in [0,1)")


def build_c0_campaign_manifest(
    config: C0CampaignConfig,
    *,
    scaler_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Declare L0's early launch and the ten fail-closed post-teacher runs."""

    rows: list[dict[str, Any]] = []
    for run_id in C0_CANONICAL_RUN_IDS:
        recipe = resolve_c0_loss_recipe(run_id)
        model = PredictionAnchoredC0Correction(
            scaler_artifact,
            C0CorrectionConfig(
                d_model=int(config.model_width),
                particle_mlp_layers=int(config.particle_mlp_layers),
                head_hidden_dim=int(config.head_hidden_dim),
                dropout=float(config.dropout),
                trust_bound_enabled=bool(recipe.trust_bound_enabled),
            ),
        )
        replica_manifests = []
        for seed in PAIRED_SEED_IDS:
            replica_manifests.append(
                C0TrainConfig(
                    run_id=run_id,
                    seed_id=int(seed),
                    field_warmup_steps=(
                        int(config.field_warmup_steps) if recipe.field_warmup else 0
                    ),
                    phase2_epochs=int(config.phase2_epochs),
                    learning_rate=float(config.learning_rate),
                    weight_decay=float(config.weight_decay),
                    grad_clip_norm=float(config.grad_clip_norm),
                    kd_temperature=float(config.kd_temperature),
                    early_stop_patience=int(config.early_stop_patience),
                    allocation_id="runtime_allocation_id_required",
                ).to_artifact()
            )
        rows.append(
            {
                "run_id": run_id,
                "aliases": ["D10_A0_c0_delta"] if run_id == "D10_L8_full_c0" else [],
                "stage": "B3" if recipe.preteacher_l0_exception else "B6",
                "launch_group": (
                    "parallel_with_consumer_training"
                    if recipe.preteacher_l0_exception
                    else "after_confirmed_selected_consumer"
                ),
                "loss_recipe": recipe.to_artifact(),
                "model_contract": model.config_artifact(),
                "paired_replica_manifests": replica_manifests,
                "requires_selected_bridge_consumer_json": bool(
                    recipe.requires_selected_teacher
                ),
                "requires_primary_target_logit_cache": bool(
                    recipe.requires_target_logit_cache
                ),
                "selectable_for_primary_deployment": bool(
                    recipe.selectable_for_primary_deployment
                ),
            }
        )
    return with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_C0_CAMPAIGN_MANIFEST_CONTRACT,
            "campaign_config": asdict(config),
            "scaler_sha256": rows[0]["model_contract"]["scaler_sha256"],
            "canonical_configuration_count": len(rows),
            "paired_replica_count": len(rows) * len(PAIRED_SEED_IDS),
            "paired_seed_ids": list(PAIRED_SEED_IDS),
            "runs": rows,
            "launch_contract": {
                "B3": ["D10_L0_bridge_only", "consumer_training"],
                "B6_prerequisites": [
                    "selected_bridge_consumer.json:CONFIRMED_LOCKED",
                    "teacher_binding_primary.json",
                    "live_teacher_config.json",
                    "primary_target_logit_cache:for_nonzero_KD_only",
                ],
                "guessed_consumer_allowed": False,
            },
            "validation_splits": {
                "checkpoint_selection": "model_val_stop",
                "configuration_evaluation": "model_val_select_once_after_checkpoint",
                "stack_val_children_opened_during_training": False,
            },
            "publication": {
                "all_three_replica_metrics": True,
                "retained_weights": "ordered_median_only",
                "optimizer_or_scheduler_state": False,
                "generated_fields": False,
            },
        }
    )


@dataclass(frozen=True)
class C0RamSnapshot:
    encoded_payload: bytes = field(repr=False)
    content_hash: str
    allocation_id: str
    run_id: str
    seed_id: int
    phase: str
    completed_steps: int
    parent_hashes: Mapping[str, str]

    @property
    def resident_bytes(self) -> int:
        return len(self.encoded_payload)

    def audit_artifact(self) -> dict[str, Any]:
        return with_content_hash(
            {
                "contract": PREDICTION_ANCHORED_C0_RAM_SNAPSHOT_CONTRACT,
                "snapshot_sha256": self.content_hash,
                "allocation_id": self.allocation_id,
                "run_id": self.run_id,
                "seed_id": self.seed_id,
                "phase": self.phase,
                "completed_steps": self.completed_steps,
                "resident_bytes": self.resident_bytes,
                "parent_hashes": dict(self.parent_hashes),
                "storage": "verified_job_local_ram_only",
                "persistent_resume_checkpoint": False,
                "cross_allocation_resume": False,
            }
        )


def capture_c0_ram_snapshot(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: C0TrainConfig,
    phase: str,
    completed_steps: int,
    parent_hashes: Mapping[str, str],
    scheduler: Any | None = None,
    amp_scaler: Any | None = None,
) -> C0RamSnapshot:
    if phase not in {"zero_initialization", "post_warmup", "phase2"}:
        raise ValueError("invalid C0 RAM snapshot phase")
    if not parent_hashes or any(len(str(value)) != 64 for value in parent_hashes.values()):
        raise ValueError("C0 RAM snapshot requires immutable parent hashes")
    payload = {
        "contract": PREDICTION_ANCHORED_C0_RAM_SNAPSHOT_CONTRACT,
        "allocation_id": config.allocation_id,
        "run_id": config.run_id,
        "seed_id": int(config.seed_id),
        "phase": phase,
        "completed_steps": int(completed_steps),
        "parent_hashes": dict(parent_hashes),
        "model_state_dict": deepcopy(model.state_dict()),
        "optimizer_state_dict": deepcopy(optimizer.state_dict()),
        "scheduler_state_dict": None if scheduler is None else deepcopy(scheduler.state_dict()),
        "amp_scaler_state_dict": None if amp_scaler is None else deepcopy(amp_scaler.state_dict()),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    encoded = _torch_bytes(payload)
    return C0RamSnapshot(
        encoded_payload=encoded,
        content_hash=hashlib.sha256(encoded).hexdigest(),
        allocation_id=config.allocation_id,
        run_id=config.run_id,
        seed_id=int(config.seed_id),
        phase=phase,
        completed_steps=int(completed_steps),
        parent_hashes=dict(parent_hashes),
    )


def restore_c0_ram_snapshot(
    snapshot: C0RamSnapshot,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    allocation_id: str,
    parent_hashes: Mapping[str, str],
    scheduler: Any | None = None,
    amp_scaler: Any | None = None,
) -> dict[str, Any]:
    if str(allocation_id) != snapshot.allocation_id:
        raise PermissionError("C0 cross-allocation resume is deliberately unsupported")
    if dict(parent_hashes) != dict(snapshot.parent_hashes):
        raise ValueError("C0 RAM resume parent lineage changed")
    if hashlib.sha256(snapshot.encoded_payload).hexdigest() != snapshot.content_hash:
        raise ValueError("C0 RAM snapshot bytes changed")
    payload = torch.load(BytesIO(snapshot.encoded_payload), map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    if (scheduler is None) != (payload["scheduler_state_dict"] is None):
        raise ValueError("C0 RAM resume scheduler presence changed")
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    if (amp_scaler is None) != (payload["amp_scaler_state_dict"] is None):
        raise ValueError("C0 RAM resume AMP presence changed")
    if amp_scaler is not None:
        amp_scaler.load_state_dict(payload["amp_scaler_state_dict"])
    random.setstate(payload["python_rng_state"])
    np.random.set_state(payload["numpy_rng_state"])
    torch.set_rng_state(payload["torch_rng_state"])
    if torch.cuda.is_available() and payload["cuda_rng_states"]:
        torch.cuda.set_rng_state_all(payload["cuda_rng_states"])
    return {
        "ok": True,
        "snapshot_sha256": snapshot.content_hash,
        "allocation_id": snapshot.allocation_id,
        "phase": snapshot.phase,
        "completed_steps": snapshot.completed_steps,
        "model_state_sha256": _torch_hash(model.state_dict()),
        "optimizer_state_sha256": _torch_hash(optimizer.state_dict()),
    }


def validate_model_val_stop_access(receipt: Mapping[str, Any]) -> dict[str, Any]:
    validate_content_hash(
        receipt, expected_contract=PREDICTION_ANCHORED_ACCESS_RECEIPT_CONTRACT
    )
    expected = {
        "status": "AUTHORIZED",
        "split_name": "model_val_stop",
        "purpose": "checkpoint_selection",
        "one_shot": False,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise PermissionError(f"C0 model_val_stop receipt changed {key}")
    if receipt.get("seal_kind") is not None or receipt.get("unlock_sha256") is not None:
        raise PermissionError("model_val_stop must use its unsealed development receipt")
    return {"ok": True, "access_receipt_sha256": receipt["content_hash"]}


def validate_c0_teacher_lineage(
    recipe: C0LossRecipe,
    *,
    live_consumer: FrozenLiveBridgeConsumer | None,
    selected_bridge_consumer: Mapping[str, Any] | None,
    live_teacher_config: Mapping[str, Any] | None,
    target_cache_manifest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Fail closed unless a post-teacher C0 run uses the locked primary T10."""

    if recipe.preteacher_l0_exception:
        if any(
            value is not None
            for value in (
                live_consumer,
                selected_bridge_consumer,
                live_teacher_config,
                target_cache_manifest,
            )
        ):
            raise ValueError("early L0 must launch without a guessed selected consumer")
        return with_content_hash(
            {
                "contract": PREDICTION_ANCHORED_C0_TEACHER_LINEAGE_CONTRACT,
                "run_id": recipe.run_id,
                "mode": "preteacher_l0_exception",
                "selected_teacher_required": False,
                "target_cache_required": False,
                "teacher_checkpoint_sha256": None,
            }
        )

    if live_consumer is None or selected_bridge_consumer is None or live_teacher_config is None:
        raise ValueError(
            f"{recipe.run_id} requires selected_bridge_consumer.json and its bound live T10"
        )
    validate_content_hash(
        selected_bridge_consumer,
        expected_contract=PREDICTION_ANCHORED_SELECTED_CONSUMER_CONTRACT,
    )
    validate_content_hash(
        live_teacher_config,
        expected_contract=PREDICTION_ANCHORED_LIVE_TEACHER_CONFIG_CONTRACT,
    )
    if selected_bridge_consumer.get("status") != "CONFIRMED_LOCKED":
        raise PermissionError("post-teacher C0 requires a confirmed locked consumer")
    selected_contract = {
        "selected_rho_endpoint": 0.10,
        "bridge_channel_policy": "physical45",
        "stack_val_consumer_opened": True,
        "refit_performed": False,
    }
    for key, expected in selected_contract.items():
        if selected_bridge_consumer.get(key) != expected:
            raise ValueError(f"selected_bridge_consumer.json changed {key}")
    checkpoint = str(selected_bridge_consumer.get("checkpoint_sha256", ""))
    if live_consumer.checkpoint_sha256 != checkpoint:
        raise ValueError("live C0 consumer differs from selected_bridge_consumer.json")
    expected_live = {
        "checkpoint_sha256": checkpoint,
        "binding_kind": "primary",
        "channel_policy": "physical45",
        "parameters_frozen": True,
        "input_gradient_enabled": True,
        "checkpoint_refit_forbidden": True,
    }
    for key, expected in expected_live.items():
        if live_teacher_config.get(key) != expected:
            raise ValueError(f"C0 live teacher configuration changed {key}")

    if recipe.requires_target_logit_cache:
        if target_cache_manifest is None:
            raise ValueError(f"{recipe.run_id} requires its provenance-bound target-logit cache")
        validate_content_hash(
            target_cache_manifest,
            expected_contract=PREDICTION_ANCHORED_TEACHER_LOGIT_CACHE_CONTRACT,
        )
        expected_cache = {
            "cache_namespace": PRIMARY_TEACHER_NAMESPACE,
            "teacher_binding_kind": "primary",
            "teacher_binding_sha256": live_teacher_config.get("teacher_binding_sha256"),
            "checkpoint_sha256": checkpoint,
            "live_checkpoint_sha256": checkpoint,
            "channel_policy": "physical45",
            "field_condition": "bridge_0.100",
            "rho_endpoint": 0.10,
            "target_logits_detached": True,
            "same_checkpoint_target_and_live": True,
            "checkpoint_refit_forbidden": True,
        }
        for key, expected in expected_cache.items():
            if target_cache_manifest.get(key) != expected:
                raise ValueError(f"C0 target-logit manifest changed {key}")
    elif target_cache_manifest is not None:
        raise ValueError(f"{recipe.run_id} has zero KD and must not load target logits")

    return with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_C0_TEACHER_LINEAGE_CONTRACT,
            "run_id": recipe.run_id,
            "mode": "locked_primary_teacher",
            "selected_consumer_sha256": selected_bridge_consumer["content_hash"],
            "live_teacher_config_sha256": live_teacher_config["content_hash"],
            "teacher_binding_sha256": live_teacher_config["teacher_binding_sha256"],
            "teacher_checkpoint_sha256": checkpoint,
            "target_cache_required": bool(recipe.requires_target_logit_cache),
            "target_cache_sha256": (
                None
                if target_cache_manifest is None
                else target_cache_manifest["content_hash"]
            ),
            "same_checkpoint_selection_target_live": True,
            "checkpoint_refit_forbidden": True,
        }
    )


def select_postteacher_checkpoint(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("post-teacher checkpoint selection has no epochs")
    for record in records:
        for key in ("epoch", "accuracy", "cross_entropy", "model_state_dict"):
            if key not in record:
                raise ValueError(f"post-teacher checkpoint record lacks {key}")
        if not math.isfinite(float(record["accuracy"])) or not math.isfinite(
            float(record["cross_entropy"])
        ):
            raise ValueError("post-teacher checkpoint metrics must be finite")
    best_accuracy = max(float(record["accuracy"]) for record in records)
    accuracy_pool = [
        record for record in records if best_accuracy - float(record["accuracy"]) <= 0.0001
    ]
    best_ce = min(float(record["cross_entropy"]) for record in accuracy_pool)
    ce_pool = [
        record for record in accuracy_pool if float(record["cross_entropy"]) - best_ce <= 1.0e-6
    ]
    selected = min(ce_pool, key=lambda record: int(record["epoch"]))
    artifact = with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_C0_CHECKPOINT_SELECTION_CONTRACT,
            "family": "postteacher_accuracy_ce_earliest",
            "accuracy_tolerance": 0.0001,
            "cross_entropy_tolerance": 1.0e-6,
            "best_accuracy": best_accuracy,
            "accuracy_pool_epochs": sorted(int(record["epoch"]) for record in accuracy_pool),
            "best_cross_entropy_in_pool": best_ce,
            "cross_entropy_pool_epochs": sorted(int(record["epoch"]) for record in ce_pool),
            "selected_epoch": int(selected["epoch"]),
            "selected_accuracy": float(selected["accuracy"]),
            "selected_cross_entropy": float(selected["cross_entropy"]),
            "selected_model_state_sha256": _torch_hash(selected["model_state_dict"]),
            "kd_agreement_used_for_selection": False,
        }
    )
    return {
        "artifact": artifact,
        "selected_model_state_dict": deepcopy(selected["model_state_dict"]),
    }


def select_l0_checkpoint(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("L0 checkpoint selection has no epochs")
    for record in records:
        for key in ("epoch", "bridge_loss", "normalized_bridge_mse", "model_state_dict"):
            if key not in record:
                raise ValueError(f"L0 checkpoint record lacks {key}")
        if not math.isfinite(float(record["bridge_loss"])) or not math.isfinite(
            float(record["normalized_bridge_mse"])
        ):
            raise ValueError("L0 checkpoint metrics must be finite")
    best_loss = min(float(record["bridge_loss"]) for record in records)
    tolerance = max(1.0e-8, 1.0e-4 * abs(best_loss))
    loss_pool = [
        record for record in records if float(record["bridge_loss"]) - best_loss <= tolerance
    ]
    best_nmse = min(float(record["normalized_bridge_mse"]) for record in loss_pool)
    nmse_pool = [
        record
        for record in loss_pool
        if float(record["normalized_bridge_mse"]) == best_nmse
    ]
    selected = min(nmse_pool, key=lambda record: int(record["epoch"]))
    artifact = with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_C0_CHECKPOINT_SELECTION_CONTRACT,
            "family": "l0_bridge_loss_nmse_earliest_exception",
            "bridge_loss_tolerance": tolerance,
            "best_bridge_loss": best_loss,
            "bridge_loss_pool_epochs": sorted(int(record["epoch"]) for record in loss_pool),
            "best_normalized_bridge_mse": best_nmse,
            "normalized_mse_pool_epochs": sorted(int(record["epoch"]) for record in nmse_pool),
            "selected_epoch": int(selected["epoch"]),
            "selected_bridge_loss": float(selected["bridge_loss"]),
            "selected_normalized_bridge_mse": float(selected["normalized_bridge_mse"]),
            "selected_model_state_sha256": _torch_hash(selected["model_state_dict"]),
            "selected_teacher_required": False,
        }
    )
    return {
        "artifact": artifact,
        "selected_model_state_dict": deepcopy(selected["model_state_dict"]),
    }


def _forward_and_loss(
    model: PredictionAnchoredC0Correction,
    raw_batch: Mapping[str, Any],
    recipe: C0LossRecipe,
    *,
    phase: str,
    live_consumer: FrozenLiveBridgeConsumer | None,
    temperature: float,
) -> tuple[Any, torch.Tensor, dict[str, Any], torch.Tensor | None]:
    batch = _prepared_batch(raw_batch, _model_device(model))
    output = model(
        batch["hlt_tokens"], batch["mask"], batch["f0"], batch["h0"]
    )
    coefficients = recipe.phase_coefficients(phase)
    need_live = coefficients["kd"] > 0 or coefficients["ce"] > 0
    if need_live and live_consumer is None:
        raise ValueError(f"{recipe.run_id} requires the selected frozen live T10")
    live_logits = None
    if need_live:
        live_logits = live_consumer(batch, output.f_hat)  # type: ignore[misc]
    target_logits = batch.get("target_logits") if coefficients["kd"] > 0 else None
    loss, diagnostics = compute_c0_objective(
        output,
        batch,
        recipe,
        model.scalers,
        phase=phase,
        live_logits=live_logits,
        target_logits=target_logits,
        temperature=temperature,
    )
    return output, loss, diagnostics, live_logits


def _train_one(
    model: PredictionAnchoredC0Correction,
    optimizer: torch.optim.Optimizer,
    batch: Mapping[str, Any],
    recipe: C0LossRecipe,
    *,
    phase: str,
    live_consumer: FrozenLiveBridgeConsumer | None,
    temperature: float,
    grad_clip_norm: float,
) -> dict[str, Any]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    _, loss, diagnostics, _ = _forward_and_loss(
        model,
        batch,
        recipe,
        phase=phase,
        live_consumer=live_consumer,
        temperature=temperature,
    )
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite C0 loss; replica may not enter the next phase")
    loss.backward()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if float(grad_clip_norm) > 0:
        norm = torch.nn.utils.clip_grad_norm_(parameters, float(grad_clip_norm))
        if not torch.isfinite(norm):
            raise FloatingPointError("non-finite C0 gradients; replica may not enter the next phase")
    if any(
        parameter.grad is not None and not torch.isfinite(parameter.grad).all()
        for parameter in parameters
    ):
        raise FloatingPointError("non-finite C0 parameter gradient")
    optimizer.step()
    return diagnostics


def _evaluate_epoch(
    model: PredictionAnchoredC0Correction,
    batches: Iterable[Mapping[str, Any]],
    recipe: C0LossRecipe,
    *,
    live_consumer: FrozenLiveBridgeConsumer | None,
    temperature: float,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_events = 0
    correct = 0
    ce_total = 0.0
    predicted_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    mask_parts: list[torch.Tensor] = []
    bridge_loss_total = 0.0
    with torch.no_grad():
        for raw_batch in batches:
            output, loss, diagnostics, live_logits = _forward_and_loss(
                model,
                raw_batch,
                recipe,
                phase="distillation",
                live_consumer=live_consumer,
                temperature=temperature,
            )
            batch_size = int(output.f_hat.shape[0])
            total_events += batch_size
            total_loss += float(loss.detach().cpu().item()) * batch_size
            bridge_loss_total += float(diagnostics["raw_components"]["bridge"]) * batch_size
            prepared = _prepared_batch(raw_batch, _model_device(model))
            if "bridge_fields" in prepared:
                target_correction = (
                    prepared["bridge_fields"][..., :45] - prepared["f0"][..., :45]
                )
                predicted_parts.append(output.physical_correction.detach().cpu())
                target_parts.append(target_correction.detach().cpu())
                mask_parts.append(output.mask.detach().cpu())
            if live_logits is not None:
                labels = prepared["labels"]
                correct += int((live_logits.argmax(dim=-1) == labels).sum().detach().cpu().item())
                ce_total += float(
                    torch.nn.functional.cross_entropy(live_logits, labels, reduction="sum")
                    .detach()
                    .cpu()
                    .item()
                )
    if total_events <= 0:
        raise ValueError("C0 model_val_stop evaluation is empty")
    reachability = (
        None
        if not predicted_parts
        else bridge_reachability_metrics(
            torch.cat(predicted_parts), torch.cat(target_parts), torch.cat(mask_parts)
        )
    )
    return {
        "loss": total_loss / total_events,
        "bridge_loss": bridge_loss_total / total_events,
        "normalized_bridge_mse": (
            None if reachability is None else reachability["overall"]["normalized_mse"]
        ),
        "accuracy": None if live_consumer is None else correct / total_events,
        "cross_entropy": None if live_consumer is None else ce_total / total_events,
        "reachability": reachability,
        "event_count": total_events,
    }


def train_c0_replica(
    config: C0TrainConfig,
    *,
    model: PredictionAnchoredC0Correction,
    optimizer: torch.optim.Optimizer,
    train_batches: BatchFactory,
    model_val_stop_batches: BatchFactory,
    model_val_stop_access_receipt: Mapping[str, Any],
    parent_hashes: Mapping[str, str],
    live_consumer: FrozenLiveBridgeConsumer | None = None,
    selected_bridge_consumer: Mapping[str, Any] | None = None,
    live_teacher_config: Mapping[str, Any] | None = None,
    target_cache_manifest: Mapping[str, Any] | None = None,
    scheduler: Any | None = None,
    amp_scaler: Any | None = None,
) -> dict[str, Any]:
    """Train one paired replica without persisting resume/non-selected states."""

    recipe = resolve_c0_loss_recipe(config.run_id)
    access = validate_model_val_stop_access(model_val_stop_access_receipt)
    if bool(model.config.trust_bound_enabled) != bool(recipe.trust_bound_enabled):
        raise ValueError("C0 model trust configuration disagrees with its run recipe")
    teacher_lineage = validate_c0_teacher_lineage(
        recipe,
        live_consumer=live_consumer,
        selected_bridge_consumer=selected_bridge_consumer,
        live_teacher_config=live_teacher_config,
        target_cache_manifest=target_cache_manifest,
    )
    zero_state = deepcopy(model.state_dict())
    zero_state_hash = _torch_hash(zero_state)
    zero_snapshot = capture_c0_ram_snapshot(
        model=model,
        optimizer=optimizer,
        config=config,
        phase="zero_initialization",
        completed_steps=0,
        parent_hashes=parent_hashes,
        scheduler=scheduler,
        amp_scaler=amp_scaler,
    )
    warmup_curve: list[dict[str, Any]] = []
    completed_steps = 0
    if recipe.field_warmup:
        cycle = 0
        iterator = iter(train_batches("field_warmup", cycle))
        while completed_steps < int(config.field_warmup_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                cycle += 1
                iterator = iter(train_batches("field_warmup", cycle))
                try:
                    batch = next(iterator)
                except StopIteration as exc:
                    raise ValueError("C0 warm-up batch factory is empty") from exc
            diagnostics = _train_one(
                model,
                optimizer,
                batch,
                recipe,
                phase="field_warmup",
                live_consumer=None,
                temperature=float(config.kd_temperature),
                grad_clip_norm=float(config.grad_clip_norm),
            )
            completed_steps += 1
            warmup_curve.append({"step": completed_steps, "loss": diagnostics["total"]})
            if scheduler is not None:
                scheduler.step()
    if completed_steps != (int(config.field_warmup_steps) if recipe.field_warmup else 0):
        raise AssertionError("C0 Phase 1 did not consume its exact immutable step budget")
    post_warmup_snapshot = capture_c0_ram_snapshot(
        model=model,
        optimizer=optimizer,
        config=config,
        phase="post_warmup",
        completed_steps=completed_steps,
        parent_hashes=parent_hashes,
        scheduler=scheduler,
        amp_scaler=amp_scaler,
    )
    epoch_records: list[dict[str, Any]] = []
    phase2_curves: list[dict[str, Any]] = []
    stale_epochs = 0
    best_first_criterion: float | None = None
    for epoch in range(int(config.phase2_epochs)):
        batch_count = 0
        for batch in train_batches("distillation", epoch):
            diagnostics = _train_one(
                model,
                optimizer,
                batch,
                recipe,
                phase="distillation",
                live_consumer=live_consumer,
                temperature=float(config.kd_temperature),
                grad_clip_norm=float(config.grad_clip_norm),
            )
            completed_steps += 1
            batch_count += 1
            if scheduler is not None:
                scheduler.step()
        if batch_count == 0:
            raise ValueError("C0 Phase 2 batch factory is empty")
        validation = _evaluate_epoch(
            model,
            model_val_stop_batches("distillation", epoch),
            recipe,
            live_consumer=live_consumer,
            temperature=float(config.kd_temperature),
        )
        phase2_curves.append({"epoch": epoch, "model_val_stop": validation})
        record = {
            "epoch": epoch,
            "model_state_dict": deepcopy(model.state_dict()),
        }
        if recipe.preteacher_l0_exception:
            record.update(
                {
                    "bridge_loss": validation["bridge_loss"],
                    "normalized_bridge_mse": validation["normalized_bridge_mse"],
                }
            )
        else:
            record.update(
                {
                    "accuracy": validation["accuracy"],
                    "cross_entropy": validation["cross_entropy"],
                }
            )
        epoch_records.append(record)
        if recipe.preteacher_l0_exception:
            current = float(validation["bridge_loss"])
            improved = (
                best_first_criterion is None
                or current
                < best_first_criterion
                - max(1.0e-8, 1.0e-4 * abs(best_first_criterion))
            )
            if improved:
                best_first_criterion = current
        else:
            current = float(validation["accuracy"])
            improved = (
                best_first_criterion is None
                or current > best_first_criterion + 0.0001
            )
            if improved:
                best_first_criterion = current
        stale_epochs = 0 if improved else stale_epochs + 1
        if int(config.early_stop_patience) >= 0 and stale_epochs > int(
            config.early_stop_patience
        ):
            break
    selection_result = (
        select_l0_checkpoint(epoch_records)
        if recipe.preteacher_l0_exception
        else select_postteacher_checkpoint(epoch_records)
    )
    selected_state = selection_result["selected_model_state_dict"]
    selection = selection_result["artifact"]
    candidate = {
        "checkpoint_contract": PREDICTION_ANCHORED_C0_REPLICA_CONTRACT,
        "run_id": recipe.run_id,
        "seed_id": int(config.seed_id),
        "epoch": int(selection["selected_epoch"]),
        "model_config": model.config.to_artifact(),
        "model_state_dict": selected_state,
        "scaler_sha256": model.scaler_sha256,
        "parent_hashes": dict(parent_hashes),
        "selected_teacher_checkpoint_sha256": (
            None if live_consumer is None else live_consumer.checkpoint_sha256
        ),
        "target_cache_sha256": (
            None if target_cache_manifest is None else target_cache_manifest["content_hash"]
        ),
        "weights_only": True,
        "optimizer_state_persisted": False,
        "frozen_parent_weights_persisted": False,
    }
    audit = with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_C0_REPLICA_CONTRACT,
            "run_id": recipe.run_id,
            "seed_id": int(config.seed_id),
            "train_config": config.to_artifact(),
            "model_val_stop_access": access,
            "teacher_lineage": teacher_lineage,
            "zero_initialization_state_sha256": zero_state_hash,
            "zero_snapshot_sha256": zero_snapshot.content_hash,
            "post_warmup_snapshot_sha256": post_warmup_snapshot.content_hash,
            "warmup_steps_completed": (
                int(config.field_warmup_steps) if recipe.field_warmup else 0
            ),
            "warmup_validation_could_stop": False,
            "phase2_epochs_completed": len(epoch_records),
            "early_stop_patience": int(config.early_stop_patience),
            "early_stop_first_criterion_only": True,
            "optimizer_steps_completed": completed_steps,
            "checkpoint_selection": selection,
            "warmup_curve": warmup_curve,
            "phase2_curves": phase2_curves,
            "resume_storage": "verified_job_local_ram_only",
            "cross_allocation_resume": False,
            "persistent_optimizer_state": False,
            "persistent_generated_fields": False,
        }
    )
    return {
        "audit": audit,
        "candidate_weights": candidate,
        "post_warmup_snapshot": post_warmup_snapshot,
    }


@dataclass(frozen=True)
class C0ReplicaResult:
    run_id: str
    seed_id: int
    metrics: Mapping[str, Any]
    weights_payload: Mapping[str, Any] = field(repr=False)

    def __post_init__(self) -> None:
        if self.run_id not in C0_CANONICAL_RUN_IDS or int(self.seed_id) not in PAIRED_SEED_IDS:
            raise ValueError("invalid C0 paired replica")
        if "model_state_dict" not in self.weights_payload:
            raise ValueError("C0 replica has no trainable model state")
        if self.weights_payload.get("checkpoint_contract") != PREDICTION_ANCHORED_C0_REPLICA_CONTRACT:
            raise ValueError("C0 replica checkpoint contract changed")
        if self.weights_payload.get("run_id") != self.run_id:
            raise ValueError("C0 replica checkpoint run ID changed")
        if int(self.weights_payload.get("seed_id", -1)) != int(self.seed_id):
            raise ValueError("C0 replica checkpoint seed ID changed")


def _metric(metrics: Mapping[str, Any], path: str) -> float:
    value: Any = metrics
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise KeyError(f"C0 replica metrics lack {path}")
        value = value[part]
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"C0 replica metric {path} is non-finite")
    return result


def aggregate_c0_replicas(replicas: Sequence[C0ReplicaResult]) -> dict[str, Any]:
    if len(replicas) != 3 or {item.seed_id for item in replicas} != set(PAIRED_SEED_IDS):
        raise ValueError("C0 aggregate requires paired seeds 101/202/303")
    run_ids = {item.run_id for item in replicas}
    if len(run_ids) != 1:
        raise ValueError("C0 aggregate mixes run IDs")
    run_id = next(iter(run_ids))
    common_deployable_metrics = all(
        isinstance(item.metrics.get("model_val_select"), Mapping) for item in replicas
    )
    if common_deployable_metrics:
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
    elif run_id == "D10_L0_bridge_only":
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
    else:
        raise ValueError("post-teacher C0 publication requires model_val_select metrics")
    scored.sort(key=lambda value: value[:4])
    median = scored[1][4]
    return with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_C0_AGGREGATE_CONTRACT,
            "run_id": run_id,
            "paired_seed_ids": list(PAIRED_SEED_IDS),
            "ordering": ordering,
            "ordered_seed_ids": [int(row[4].seed_id) for row in scored],
            "median_seed_id": int(median.seed_id),
            "best_seed_id": int(scored[-1][4].seed_id),
            "best_seed_weights_rejected": True,
            "replica_metrics": [
                {"seed_id": int(item.seed_id), "metrics": deepcopy(dict(item.metrics))}
                for item in sorted(replicas, key=lambda value: value.seed_id)
            ],
        }
    )


def _weights_only(payload: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "checkpoint_contract",
        "run_id",
        "seed_id",
        "epoch",
        "model_config",
        "model_state_dict",
        "scaler_sha256",
        "parent_hashes",
        "selected_teacher_checkpoint_sha256",
        "target_cache_sha256",
    }
    if "model_state_dict" not in payload:
        raise ValueError("C0 publication payload has no model state")
    result = {key: deepcopy(value) for key, value in payload.items() if key in allowed}
    result.update(
        {
            "weights_only": True,
            "optimizer_state_persisted": False,
            "scheduler_state_persisted": False,
            "nonmedian_weights_persisted": False,
            "frozen_parent_weights_persisted": False,
            "generated_fields_persisted": False,
        }
    )
    return result


def publish_c0_paired_replicas(
    replicas: Sequence[C0ReplicaResult],
    *,
    output_dir: str | Path,
) -> dict[str, Any]:
    aggregate = aggregate_c0_replicas(replicas)
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"C0 publication directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    median = next(item for item in replicas if item.seed_id == aggregate["median_seed_id"])
    checkpoint = root / "median_weights.pt"
    torch.save(_weights_only(median.weights_payload), checkpoint)
    publication = with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_C0_PUBLICATION_CONTRACT,
            "run_id": median.run_id,
            "aggregate_sha256": aggregate["content_hash"],
            "paired_seed_ids": list(PAIRED_SEED_IDS),
            "median_seed_id": int(median.seed_id),
            "retained_checkpoint": checkpoint.name,
            "retained_checkpoint_sha256": sha256_file(checkpoint),
            "measured_state_bytes": int(checkpoint.stat().st_size),
            "persistent_artifact_allowlist": [
                "aggregate_metrics.json",
                "median_weights.pt",
                "publication.json",
            ],
            "nonmedian_weights_persisted": False,
            "optimizer_state_persisted": False,
            "duplicate_frozen_parents_persisted": False,
            "generated_fields_persisted": False,
        }
    )
    write_immutable_json(root / "aggregate_metrics.json", aggregate)
    write_immutable_json(root / "publication.json", publication)
    names = sorted(path.name for path in root.iterdir())
    if names != ["aggregate_metrics.json", "median_weights.pt", "publication.json"]:
        raise RuntimeError(f"unexpected C0 publication artifacts: {names}")
    return {
        "ok": True,
        "checkpoint": str(checkpoint),
        "aggregate": str(root / "aggregate_metrics.json"),
        "publication": str(root / "publication.json"),
        "median_seed_id": int(median.seed_id),
        "measured_state_bytes": int(checkpoint.stat().st_size),
        "persistent_artifacts": names,
    }


def measure_c0_registry_states(
    registry: Mapping[str, Any],
    *,
    scaler_artifact: Mapping[str, Any],
    model_width: int = 160,
) -> tuple[dict[str, Any], dict[str, Any]]:
    measurements: dict[str, int] = {}
    model_contracts: dict[str, str] = {}
    for run_id in C0_CANONICAL_RUN_IDS:
        recipe = resolve_c0_loss_recipe(run_id)
        model = PredictionAnchoredC0Correction(
            scaler_artifact,
            C0CorrectionConfig(
                d_model=int(model_width),
                trust_bound_enabled=recipe.trust_bound_enabled,
            ),
        )
        payload = _weights_only(
            {
                "checkpoint_contract": PREDICTION_ANCHORED_C0_REPLICA_CONTRACT,
                "run_id": run_id,
                "seed_id": 101,
                "epoch": 0,
                "model_config": model.config.to_artifact(),
                "model_state_dict": model.state_dict(),
                "scaler_sha256": model.scaler_sha256,
                "parent_hashes": {},
                "selected_teacher_checkpoint_sha256": None,
                "target_cache_sha256": None,
            }
        )
        measurements[run_id] = len(_torch_bytes(payload))
        model_contracts[run_id] = model.config_artifact()["content_hash"]
    if set(measurements) != set(C0_CANONICAL_RUN_IDS):
        raise AssertionError("C0 measurement did not cover all eleven configurations")
    updated = record_registry_measurements(registry, measurements)
    artifact = with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_C0_MEASUREMENT_CONTRACT,
            "input_registry_sha256": registry["content_hash"],
            "updated_registry_sha256": updated["content_hash"],
            "measured_state_bytes": measurements,
            "model_contract_sha256": model_contracts,
            "implemented_c0_configuration_count": 11,
            "serialization_method": "torch_save_weights_only_to_verified_ram",
        }
    )
    return updated, artifact


def run_c0_cpu_miniature(
    *,
    scaler_artifact: Mapping[str, Any],
    batch: Mapping[str, Any],
    live_consumer: FrozenLiveBridgeConsumer,
    debug_width: int = 32,
) -> dict[str, Any]:
    """One finite warm-up/distillation backward for every Step 5 recipe."""

    if int(debug_width) <= 0:
        raise ValueError("CPU miniature width must be positive")
    runs: dict[str, Any] = {}
    for run_id in C0_CANONICAL_RUN_IDS:
        recipe = resolve_c0_loss_recipe(run_id)
        torch.manual_seed(70_000 + C0_CANONICAL_RUN_IDS.index(run_id))
        model = PredictionAnchoredC0Correction(
            scaler_artifact,
            C0CorrectionConfig(
                d_model=int(debug_width),
                particle_mlp_layers=1,
                head_hidden_dim=max(8, int(debug_width)),
                dropout=0.0,
                trust_bound_enabled=recipe.trust_bound_enabled,
            ),
        )
        phases = ["field_warmup", "distillation"] if recipe.field_warmup else ["distillation"]
        phase_rows: dict[str, Any] = {}
        for phase in phases:
            model.zero_grad(set_to_none=True)
            output, loss, diagnostics, _ = _forward_and_loss(
                model,
                batch,
                recipe,
                phase=phase,
                live_consumer=(None if recipe.preteacher_l0_exception else live_consumer),
                temperature=2.0,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"CPU miniature {run_id}/{phase} is non-finite")
            loss.backward()
            gradients = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            if not gradients or any(not torch.isfinite(value).all() for value in gradients):
                raise FloatingPointError(f"CPU miniature {run_id}/{phase} has invalid gradients")
            phase_rows[phase] = {
                "loss": float(loss.detach().cpu().item()),
                "gradient_norm": float(
                    torch.sqrt(sum(value.detach().square().sum() for value in gradients)).cpu().item()
                ),
                "coefficients": diagnostics["coefficients"],
                "f0_initially_reproduced": bool(
                    torch.equal(output.f_hat.detach(), torch.as_tensor(batch["f0"]).float())
                ),
            }
        runs[run_id] = {
            "trust_bound_enabled": recipe.trust_bound_enabled,
            "field_warmup": recipe.field_warmup,
            "requires_target_logit_cache": recipe.requires_target_logit_cache,
            "phases": phase_rows,
        }
    return with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_C0_CPU_MINIATURE_CONTRACT,
            "profile": "cpu_debug_nonselectable",
            "run_ids": list(C0_CANONICAL_RUN_IDS),
            "configuration_count": len(runs),
            "runs": runs,
            "scientific_results_allowed": False,
        }
    )


__all__ = [
    "PREDICTION_ANCHORED_C0_TRAIN_CONFIG_CONTRACT",
    "PREDICTION_ANCHORED_C0_RAM_SNAPSHOT_CONTRACT",
    "PREDICTION_ANCHORED_C0_CHECKPOINT_SELECTION_CONTRACT",
    "PREDICTION_ANCHORED_C0_REPLICA_CONTRACT",
    "PREDICTION_ANCHORED_C0_AGGREGATE_CONTRACT",
    "PREDICTION_ANCHORED_C0_PUBLICATION_CONTRACT",
    "PREDICTION_ANCHORED_C0_MEASUREMENT_CONTRACT",
    "PREDICTION_ANCHORED_C0_CPU_MINIATURE_CONTRACT",
    "PREDICTION_ANCHORED_C0_CAMPAIGN_MANIFEST_CONTRACT",
    "PREDICTION_ANCHORED_C0_TEACHER_LINEAGE_CONTRACT",
    "C0TrainConfig",
    "C0CampaignConfig",
    "C0RamSnapshot",
    "C0ReplicaResult",
    "capture_c0_ram_snapshot",
    "restore_c0_ram_snapshot",
    "validate_model_val_stop_access",
    "validate_c0_teacher_lineage",
    "build_c0_campaign_manifest",
    "select_postteacher_checkpoint",
    "select_l0_checkpoint",
    "train_c0_replica",
    "aggregate_c0_replicas",
    "publish_c0_paired_replicas",
    "measure_c0_registry_states",
    "run_c0_cpu_miniature",
]
