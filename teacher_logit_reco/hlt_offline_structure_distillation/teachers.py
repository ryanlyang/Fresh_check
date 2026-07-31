"""Frozen HOSD teacher production, locking, and FP32 inference contracts."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from teacher_logit_reco.relation_expert_token_bridge.evaluation import CLASS_NAMES

from .contracts import (
    TEACHER_LOCK_CONTRACT,
    TEACHER_OUTPUT_MANIFEST_CONTRACT,
    TEACHER_TRAINING_MANIFEST_CONTRACT,
    canonical_sha256,
    require_sha256,
    validate_content_hash,
    with_content_hash,
)
from .extractors import TargetBatch


MANDATORY_TEACHERS = ("O_BASE", "O_FULLREL")
TEACHER_SEED = 101
TEACHER_EPOCHS = 40
TEACHER_LOGIT_TEMPERATURE = 2.0
POOLED_LATENT_DIMENSION = 128


def training_protocol() -> dict[str, Any]:
    return {
        "optimizer": "AdamW",
        "betas": [0.9, 0.999],
        "weight_decay": 1.0e-4,
        "gradient_clip_norm": 1.0,
        "maximum_epochs": TEACHER_EPOCHS,
        "minimum_epochs": TEACHER_EPOCHS,
        "base_learning_rate": 1.0e-3,
        "minimum_learning_rate": 1.0e-5,
        "mixed_precision": "BF16_on_GH200",
        "microbatch_size": 64,
        "gradient_accumulation_steps": 2,
        "effective_batch_size": 128,
        "dropouts": {
            "particle_attention": 0.0,
            "residual": 0.0,
            "class_attention": 0.0,
            "activation": 0.0,
        },
        "num_workers": 0,
        "warmup_updates": "min(T,max(1,ceil(0.05*T)))",
        "scheduler": "one_based_linear_warmup_then_cosine_to_1e-5",
        "checkpoint_selection": [
            "exact_maximum_val_stop_balanced_accuracy",
            "lower_val_stop_cross_entropy",
            "earlier_epoch",
        ],
        "early_stop_before_epoch_40": False,
        "resume_restores": [
            "model",
            "optimizer",
            "scheduler",
            "sampler",
            "accumulation_boundary",
            "rng_states",
        ],
    }


def build_teacher_training_manifest(
    *,
    campaign_spec_sha256: str,
    split_manifest_sha256: str,
    model_contract_hashes: Mapping[str, str],
    normalizer_hashes: Mapping[str, Mapping[str, str]],
    source: Mapping[str, Any],
    population: str = "target_500k",
) -> dict[str, Any]:
    if set(model_contract_hashes) != set(MANDATORY_TEACHERS):
        raise ValueError("both mandatory teacher model contracts are required")
    if population not in {"target_500k", "target_scale"}:
        raise ValueError("unknown teacher training population")
    return with_content_hash(
        {
            "contract": TEACHER_TRAINING_MANIFEST_CONTRACT,
            "schema_version": 1,
            "population": population,
            "train_split": (
                "model_train" if population == "target_500k" else "scale_train"
            ),
            "checkpoint_selection_split": "val_stop",
            "seed": TEACHER_SEED,
            "teacher_order": list(MANDATORY_TEACHERS),
            "teachers": [
                {
                    "teacher_id": teacher_id,
                    "architecture_role": (
                        "exact_standard_offline_base_particle_transformer"
                        if teacher_id == "O_BASE"
                        else "locked_full_relation_offline_particle_transformer"
                    ),
                    "model_contract_sha256": require_sha256(
                        model_contract_hashes[teacher_id],
                        name=f"model_contract_hashes.{teacher_id}",
                    ),
                    "normalizer_hashes": {
                        key: require_sha256(value, name=f"{teacher_id}.{key}")
                        for key, value in sorted(normalizer_hashes[teacher_id].items())
                    },
                    "required_checkpoint": (
                        f"teachers/{population}/{teacher_id}/best_val_stop.pt"
                    ),
                    "producer": "scripts/train_hosd_offline_teacher.py",
                }
                for teacher_id in MANDATORY_TEACHERS
            ],
            "training_protocol": training_protocol(),
            "campaign_spec_sha256": require_sha256(
                campaign_spec_sha256, name="campaign_spec_sha256"
            ),
            "split_manifest_sha256": require_sha256(
                split_manifest_sha256, name="split_manifest_sha256"
            ),
            "no_hosd_student_results_read": True,
            "scientific_underperformance_can_fail_or_cancel": False,
            "source": dict(source),
        }
    )


def complete_teacher_training(
    training_manifest: Mapping[str, Any],
    *,
    teacher_id: str,
    checkpoint_path: str | Path,
    selector_trace: Mapping[str, Any],
    architecture: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    validate_content_hash(
        training_manifest, expected_contract=TEACHER_TRAINING_MANIFEST_CONTRACT
    )
    if teacher_id not in MANDATORY_TEACHERS:
        raise ValueError("unknown mandatory teacher")
    checkpoint = Path(checkpoint_path)
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise FileNotFoundError(f"teacher checkpoint is absent or unsafe: {checkpoint}")
    encoded = checkpoint.read_bytes()
    return with_content_hash(
        {
            "contract": "hosd_teacher_training_completion_v1",
            "schema_version": 1,
            "teacher_id": teacher_id,
            "training_manifest_sha256": training_manifest["content_hash"],
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": hashlib.sha256(encoded).hexdigest(),
            "checkpoint_bytes": len(encoded),
            "architecture": dict(architecture),
            "architecture_sha256": canonical_sha256(architecture),
            "selector_trace": dict(selector_trace),
            "selector_trace_sha256": canonical_sha256(selector_trace),
            "seed": TEACHER_SEED,
            "train_split": training_manifest["train_split"],
            "checkpoint_selection_split": "val_stop",
            "hosd_student_results_read": False,
            "source": dict(source),
        }
    )


def build_teacher_lock(
    training_manifest: Mapping[str, Any],
    *,
    completions: Mapping[str, Mapping[str, Any]],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate existing produced checkpoints; never manufacture one."""

    validate_content_hash(
        training_manifest, expected_contract=TEACHER_TRAINING_MANIFEST_CONTRACT
    )
    if set(completions) != set(MANDATORY_TEACHERS):
        raise ValueError("teacher lock requires both mandatory completions")
    teacher_rows = []
    manifest_rows = {
        row["teacher_id"]: row for row in training_manifest["teachers"]
    }
    for teacher_id in MANDATORY_TEACHERS:
        completion = dict(completions[teacher_id])
        validate_content_hash(
            completion, expected_contract="hosd_teacher_training_completion_v1"
        )
        if (
            completion["teacher_id"] != teacher_id
            or completion["training_manifest_sha256"]
            != training_manifest["content_hash"]
            or completion["seed"] != TEACHER_SEED
            or completion["source"] != dict(source)
        ):
            raise ValueError(f"{teacher_id} training completion lineage differs")
        checkpoint = Path(completion["checkpoint_path"])
        if checkpoint.is_symlink() or not checkpoint.is_file():
            raise FileNotFoundError(f"locked teacher checkpoint is absent: {checkpoint}")
        encoded = checkpoint.read_bytes()
        if hashlib.sha256(encoded).hexdigest() != completion["checkpoint_sha256"]:
            raise ValueError(f"{teacher_id} checkpoint bytes changed before lock")
        teacher_rows.append(
            {
                "teacher_id": teacher_id,
                "training_completion_sha256": completion["content_hash"],
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_sha256": completion["checkpoint_sha256"],
                "checkpoint_bytes": completion["checkpoint_bytes"],
                "architecture_sha256": completion["architecture_sha256"],
                "selector_trace_sha256": completion["selector_trace_sha256"],
                "model_contract_sha256": manifest_rows[teacher_id][
                    "model_contract_sha256"
                ],
                "normalizer_hashes": dict(
                    manifest_rows[teacher_id]["normalizer_hashes"]
                ),
            }
        )
    return with_content_hash(
        {
            "contract": TEACHER_LOCK_CONTRACT,
            "schema_version": 1,
            "training_manifest_sha256": training_manifest["content_hash"],
            "population": training_manifest["population"],
            "teacher_order": list(MANDATORY_TEACHERS),
            "teachers": teacher_rows,
            "seed": TEACHER_SEED,
            "locked_before_hosd_student_training": True,
            "lock_is_not_a_checkpoint_producer": True,
            "hosd_results_consulted": False,
            "source": dict(source),
        }
    )


def validate_teacher_lock(
    lock: Mapping[str, Any],
    *,
    source: Mapping[str, Any] | None = None,
    verify_checkpoint_bytes: bool = True,
) -> None:
    validate_content_hash(lock, expected_contract=TEACHER_LOCK_CONTRACT)
    if lock["teacher_order"] != list(MANDATORY_TEACHERS) or lock["seed"] != 101:
        raise ValueError("teacher lock does not contain the mandatory seed-101 pair")
    if source is not None and lock["source"] != dict(source):
        raise ValueError("teacher lock source differs")
    if verify_checkpoint_bytes:
        for row in lock["teachers"]:
            checkpoint = Path(row["checkpoint_path"])
            if checkpoint.is_symlink() or not checkpoint.is_file():
                raise FileNotFoundError("locked checkpoint is absent or unsafe")
            if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != row[
                "checkpoint_sha256"
            ]:
                raise ValueError("locked checkpoint bytes drifted")


@dataclass(frozen=True)
class TeacherInferenceAdapter:
    teacher_id: str
    model: torch.nn.Module
    forward: Callable[[Mapping[str, torch.Tensor]], Any]
    pooled_latent_tap: Callable[[], torch.Tensor] | None = None
    tap_contract: str | None = None


def build_relational_teacher_adapter(
    *,
    teacher_id: str,
    model: torch.nn.Module,
) -> TeacherInferenceAdapter:
    """Expose logits and the exact final normalized class representation."""

    if teacher_id not in MANDATORY_TEACHERS:
        raise ValueError("unknown relational teacher")
    module = getattr(model, "mod", None)
    final_norm = getattr(module, "norm", None)
    if final_norm is None:
        raise TypeError("teacher model lacks Weaver's final normalization module")
    captured: dict[str, torch.Tensor] = {}

    def capture(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
        value = output.squeeze(1) if output.ndim == 3 and output.shape[1] == 1 else output
        captured["latent"] = value

    handle = final_norm.register_forward_hook(capture)
    # Keep the hook alive with the adapter/model.  Removing it would silently
    # invalidate the declared latent coordinate.
    setattr(model, "_hosd_final_norm_hook_handle", handle)

    def forward(batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        required = {"points", "features", "lorentz_vectors", "mask"}
        if not required.issubset(batch):
            raise ValueError("teacher batch lacks canonical Particle Transformer inputs")
        if teacher_id == "O_BASE":
            return model(
                batch["points"],
                batch["features"],
                batch["lorentz_vectors"],
                batch["mask"],
            )
        return model(
            batch["points"],
            batch["features"],
            batch["lorentz_vectors"],
            batch["mask"],
            batch.get("raw_tokens"),
            batch.get("region_trees"),
        )

    return TeacherInferenceAdapter(
        teacher_id=teacher_id,
        model=model,
        forward=forward,
        pooled_latent_tap=(
            (lambda: captured["latent"]) if teacher_id == "O_BASE" else None
        ),
        tap_contract=(
            "exact_normalized_preclassifier_o_base_pooled_representation"
            if teacher_id == "O_BASE"
            else None
        ),
    )


def infer_teacher_batch(
    adapter: TeacherInferenceAdapter,
    batch: Mapping[str, torch.Tensor],
) -> dict[str, TargetBatch]:
    """Run authoritative target capture in FP32 with gradients/autocast disabled."""

    if adapter.teacher_id not in MANDATORY_TEACHERS:
        raise ValueError("unlocked teacher adapter")
    was_training = adapter.model.training
    adapter.model.eval()
    try:
        device_type = next(adapter.model.parameters(), torch.empty(0)).device.type
        autocast = (
            torch.autocast(device_type=device_type, enabled=False)
            if device_type in {"cpu", "cuda"}
            else nullcontext()
        )
        with torch.no_grad(), autocast:
            raw_logits = adapter.forward(batch)
            logits = (
                raw_logits["logits"]
                if isinstance(raw_logits, Mapping)
                else raw_logits
            )
            logits = logits.detach().float().cpu()
            if logits.ndim != 2 or logits.shape[1] != len(CLASS_NAMES):
                raise ValueError("teacher logits must be [batch,10]")
            if not torch.isfinite(logits).all():
                raise ValueError("teacher logits contain non-finite values")
            mask = torch.ones_like(logits, dtype=torch.bool)
            output = {
                f"T_OFFLINE_LOGITS_{adapter.teacher_id}": TargetBatch(
                    target_id=f"T_OFFLINE_LOGITS_{adapter.teacher_id}",
                    component_names=tuple(CLASS_NAMES),
                    availability_groups=("teacher_output_available",),
                    values=logits,
                    loss_mask=mask,
                    diagnostics={
                        "teacher_id": adapter.teacher_id,
                        "dtype": "float32",
                        "class_order": list(CLASS_NAMES),
                        "temperature": TEACHER_LOGIT_TEMPERATURE,
                        "mixed_precision": False,
                    },
                )
            }
            if adapter.teacher_id == "O_BASE":
                if adapter.pooled_latent_tap is None:
                    raise ValueError("O_BASE inference requires the pooled latent tap")
                latent = adapter.pooled_latent_tap().detach().float().cpu()
                if latent.shape != (logits.shape[0], POOLED_LATENT_DIMENSION):
                    raise ValueError("O_BASE pooled latent must be [batch,128]")
                if not torch.isfinite(latent).all():
                    raise ValueError("O_BASE pooled latent contains non-finite values")
                if adapter.tap_contract != (
                    "exact_normalized_preclassifier_o_base_pooled_representation"
                ):
                    raise ValueError("O_BASE pooled latent tap contract differs")
                output["T_OFFLINE_POOLED_LATENT"] = TargetBatch(
                    target_id="T_OFFLINE_POOLED_LATENT",
                    component_names=tuple(
                        f"latent_{index:03d}"
                        for index in range(POOLED_LATENT_DIMENSION)
                    ),
                    availability_groups=("teacher_output_available",),
                    values=latent,
                    loss_mask=torch.ones_like(latent, dtype=torch.bool),
                    diagnostics={
                        "teacher_id": "O_BASE",
                        "tap_contract": adapter.tap_contract,
                        "dtype": "float32",
                    },
                )
            return output
    finally:
        adapter.model.train(was_training)


def build_teacher_output_manifest(
    *,
    teacher_lock: Mapping[str, Any],
    cache_manifest_hashes_by_split: Mapping[
        str, Mapping[str, str]
    ],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    validate_teacher_lock(teacher_lock, source=source, verify_checkpoint_bytes=False)
    required = {
        "T_OFFLINE_LOGITS_O_BASE",
        "T_OFFLINE_LOGITS_O_FULLREL",
        "T_OFFLINE_POOLED_LATENT",
    }
    expected_splits = {"model_train", "val_stop", "val_design"}
    if set(cache_manifest_hashes_by_split) != expected_splits:
        raise ValueError("teacher output manifest split coverage differs")
    checked = {}
    for split, hashes in sorted(cache_manifest_hashes_by_split.items()):
        if set(hashes) != required:
            raise ValueError("teacher output manifest target coverage differs")
        checked[split] = {
            key: require_sha256(
                value, name=f"cache_manifest_hashes_by_split.{split}.{key}"
            )
            for key, value in sorted(hashes.items())
        }
    return with_content_hash(
        {
            "contract": TEACHER_OUTPUT_MANIFEST_CONTRACT,
            "schema_version": 2,
            "teacher_lock_sha256": teacher_lock["content_hash"],
            "target_order": sorted(required),
            "split_order": ["model_train", "val_stop", "val_design"],
            "cache_manifest_hashes_by_split": checked,
            "dtype": "float32",
            "class_order": list(CLASS_NAMES),
            "temperature": TEACHER_LOGIT_TEMPERATURE,
            "pooled_latent": {
                "teacher_id": "O_BASE",
                "dimension": POOLED_LATENT_DIMENSION,
                "tap_contract": (
                    "exact_normalized_preclassifier_o_base_pooled_representation"
                ),
            },
            "label_access_for_inference": False,
            "source": dict(source),
        }
    )


__all__ = [
    "MANDATORY_TEACHERS",
    "POOLED_LATENT_DIMENSION",
    "TEACHER_EPOCHS",
    "TEACHER_LOGIT_TEMPERATURE",
    "TEACHER_SEED",
    "TeacherInferenceAdapter",
    "build_teacher_lock",
    "build_relational_teacher_adapter",
    "build_teacher_output_manifest",
    "build_teacher_training_manifest",
    "complete_teacher_training",
    "infer_teacher_batch",
    "training_protocol",
    "validate_teacher_lock",
]
