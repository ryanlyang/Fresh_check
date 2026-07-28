"""Pure-offline RETB expert objectives, initialization, and fixed-budget training."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .contracts import (
    canonical_json_bytes,
    load_hashed_json,
    require_sha256,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from .determinism import optimizer_update_counts, scheduled_learning_rate
from .evaluation import evaluate_classification

try:
    import torch
except ImportError:  # pragma: no cover - environment dependent
    torch = None


EXPERT_LOSS_REGISTRY_CONTRACT = "retb_expert_loss_registry_v1"
EXPERT_TRAINING_CONTRACT = "retb_offline_expert_training_v1"
EXPERT_CHECKPOINT_CONTRACT = "retb_offline_expert_checkpoint_v1"
EXPERT_CURVES_CONTRACT = "retb_offline_expert_training_curves_v1"
EXPERT_DIAGNOSTICS_CONTRACT = "retb_offline_expert_diagnostics_v1"
EXPERT_REGISTRATION_CONTRACT = "retb_offline_expert_registration_v1"
ATTACHMENT_PRETRAINING_CONTRACT = "retb_attachment_pretraining_record_v1"
TEACHER_LOGITS_MANIFEST_CONTRACT = "retb_teacher_logits_manifest_v1"

INITIALIZATION_MODES = (
    "INIT_SCRATCH",
    "INIT_OBASE_PARTICLE",
    "INIT_ATTACH_AFTER_PRETRAIN",
)
REGISTERED_LEARNING_RATES = (2.0e-4, 5.0e-4, 1.0e-3)
REGISTERED_PARTICLE_DROPOUTS = (0.0, 0.1)
EXPERT_LOSS_CANDIDATES: dict[str, dict[str, Any]] = {
    "ELOSS_CE": {
        "cross_entropy_weight": 1.0,
        "kd_weight": 0.0,
        "teacher": None,
    },
    "ELOSS_BASE_LOW": {
        "cross_entropy_weight": 1.0,
        "kd_weight": 0.10,
        "teacher": "O_BASE",
    },
    "ELOSS_BASE": {
        "cross_entropy_weight": 1.0,
        "kd_weight": 0.50,
        "teacher": "O_BASE",
    },
    "ELOSS_FULLREL": {
        "cross_entropy_weight": 1.0,
        "kd_weight": 0.50,
        "teacher": "O_FULLREL",
    },
    "ELOSS_ENSEMBLE": {
        "cross_entropy_weight": 1.0,
        "kd_weight": 0.50,
        "teacher": "MEAN_PROBABILITY_O_BASE_O_FULLREL",
    },
    "ELOSS_KD_DOMINANT": {
        "cross_entropy_weight": 0.25,
        "kd_weight": 1.0,
        "teacher": "SELECTED_STRONGEST",
    },
}


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is required for RETB expert training")
    return torch


def build_expert_loss_registry() -> dict[str, Any]:
    return with_content_hash(
        {
            "contract": EXPERT_LOSS_REGISTRY_CONTRACT,
            "schema_version": 1,
            "temperature": 2.0,
            "teacher_logits_detached": True,
            "kl_direction": "teacher_probability_to_student_probability",
            "teacher_ensemble": "arithmetic_mean_of_teacher_probabilities",
            "candidates": EXPERT_LOSS_CANDIDATES,
            "natural_specialization_reference": "ELOSS_CE",
            "representative_screen_experts": [
                "BASE4",
                "PT",
                "TRACK",
                "REGION",
            ],
            "selection": (
                "joint_seven_expert_beam_in_step5_not_independent_expert_choice"
            ),
            "teacher_checkpoint_hash_required_when_kd_nonzero": True,
            "teacher_selection_may_use_retb_expert_results": False,
        }
    )


def validate_expert_loss_registry(payload: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        payload, expected_contract=EXPERT_LOSS_REGISTRY_CONTRACT
    )
    actual = dict(payload)
    actual.pop("content_hash", None)
    actual.pop("source", None)
    expected = build_expert_loss_registry()
    expected.pop("content_hash")
    if actual != expected:
        raise ValueError("expert-loss registry differs from RETB v1")
    return digest


def build_teacher_logits_manifest(
    *,
    model_train_npz_sha256: str,
    val_stop_npz_sha256: str,
    teacher_checkpoint_hashes: Mapping[str, str],
    teacher_fields: Sequence[str],
) -> dict[str, Any]:
    checkpoints = {
        str(name): require_sha256(
            digest, name=f"teacher_checkpoint_hashes.{name}"
        )
        for name, digest in sorted(teacher_checkpoint_hashes.items())
    }
    fields = sorted(str(value) for value in teacher_fields)
    if not fields or len(fields) != len(set(fields)):
        raise ValueError("teacher-logit fields are empty or duplicated")
    if set(fields) != set(checkpoints):
        raise ValueError("teacher-logit fields differ from checkpoint parents")
    return with_content_hash(
        {
            "contract": TEACHER_LOGITS_MANIFEST_CONTRACT,
            "schema_version": 1,
            "model_train_npz_sha256": require_sha256(
                model_train_npz_sha256,
                name="model_train_npz_sha256",
            ),
            "val_stop_npz_sha256": require_sha256(
                val_stop_npz_sha256,
                name="val_stop_npz_sha256",
            ),
            "teacher_checkpoint_hashes": checkpoints,
            "teacher_fields": fields,
            "field_name_encoding": "teacher_logits_<teacher_field>",
            "identity_order_bound_by_containing_npz": True,
            "logits_detached_at_objective": True,
        }
    )


def validate_teacher_logits_manifest(payload: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        payload,
        expected_contract=TEACHER_LOGITS_MANIFEST_CONTRACT,
    )
    expected = build_teacher_logits_manifest(
        model_train_npz_sha256=payload.get("model_train_npz_sha256"),
        val_stop_npz_sha256=payload.get("val_stop_npz_sha256"),
        teacher_checkpoint_hashes=payload.get(
            "teacher_checkpoint_hashes", {}
        ),
        teacher_fields=payload.get("teacher_fields", []),
    )
    actual = dict(payload)
    actual.pop("content_hash", None)
    actual.pop("source", None)
    expected.pop("content_hash")
    if actual != expected:
        raise ValueError("teacher-logit manifest differs")
    return digest


@dataclass(frozen=True)
class OfflineExpertTrainingConfig:
    seed: int
    initialization: str = "INIT_SCRATCH"
    loss_id: str = "ELOSS_CE"
    learning_rate: float = 1.0e-3
    particle_dropout: float = 0.0
    maximum_epochs: int = 40
    minimum_learning_rate: float = 1.0e-5
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 1.0e-4
    gradient_clip: float = 1.0
    microbatch_size: int = 64
    gradient_accumulation_steps: int = 2
    effective_batch_size: int = 128
    accuracy_window: float = 0.0001
    num_workers: int = 0
    campaign_profile: str = "production"

    def validate(self) -> None:
        if self.campaign_profile not in {"production", "miniature_test"}:
            raise ValueError("unknown RETB expert training profile")
        if self.campaign_profile == "production" and self.seed not in {
            101,
            202,
            303,
        }:
            raise ValueError("production pipeline seed is not registered")
        if self.initialization not in INITIALIZATION_MODES:
            raise ValueError("offline expert initialization is not registered")
        if self.loss_id not in EXPERT_LOSS_CANDIDATES:
            raise ValueError("offline expert loss is not registered")
        if float(self.learning_rate) not in REGISTERED_LEARNING_RATES:
            raise ValueError("offline expert learning rate is not registered")
        if float(self.particle_dropout) not in REGISTERED_PARTICLE_DROPOUTS:
            raise ValueError("particle dropout is not registered")
        positive = (
            self.maximum_epochs,
            self.microbatch_size,
            self.gradient_accumulation_steps,
            self.effective_batch_size,
        )
        if any(int(value) <= 0 for value in positive):
            raise ValueError("training schedule integers must be positive")
        if (
            int(self.microbatch_size) * int(self.gradient_accumulation_steps)
            != int(self.effective_batch_size)
        ):
            raise ValueError(
                "microbatch and accumulation must preserve the effective batch"
            )
        if self.num_workers != 0:
            raise ValueError("RETB training locks num_workers=0")
        if not (
            self.minimum_learning_rate >= 0
            and self.minimum_learning_rate <= self.learning_rate
        ):
            raise ValueError("learning-rate endpoints are invalid")
        if self.campaign_profile == "production":
            locked = {
                "maximum_epochs": 40,
                "minimum_learning_rate": 1.0e-5,
                "beta1": 0.9,
                "beta2": 0.999,
                "weight_decay": 1.0e-4,
                "gradient_clip": 1.0,
                "effective_batch_size": 128,
                "accuracy_window": 0.0001,
                "num_workers": 0,
            }
            drift = {
                name: (getattr(self, name), expected)
                for name, expected in locked.items()
                if getattr(self, name) != expected
            }
            if drift:
                raise ValueError(f"production expert protocol drifted: {drift}")

    def artifact(
        self,
        *,
        global_determinism_sha256: str,
        expert_loss_registry_sha256: str,
    ) -> dict[str, Any]:
        self.validate()
        return with_content_hash(
            {
                "contract": EXPERT_TRAINING_CONTRACT,
                "schema_version": 1,
                "global_determinism_sha256": require_sha256(
                    global_determinism_sha256,
                    name="global_determinism_sha256",
                ),
                "expert_loss_registry_sha256": require_sha256(
                    expert_loss_registry_sha256,
                    name="expert_loss_registry_sha256",
                ),
                "config": asdict(self),
                "optimizer": "AdamW",
                "schedule": "one_based_linear_warmup_then_cosine",
                "precision": (
                    "production_GH200_BF16;"
                    "miniature_cuda_bf16_or_fp16_else_cpu_fp32"
                ),
                "epoch_selection_split": "val_stop",
                "architecture_selection_split_accessible": False,
                "fixed_epoch_budget": True,
                "early_stopping": False,
                "performance_based_termination": False,
                "checkpoint_selector": (
                    "max_accuracy_then_0p0001_window_then_min_CE_then_earliest"
                ),
                "checkpoint_retention_after_completion": [
                    "best_model_val.pt"
                ],
                "attachment_schedule": {
                    "epochs_1_through_5": (
                        "ordinary_particle_backbone_frozen"
                    ),
                    "epochs_6_through_10": (
                        "last_four_particle_blocks_trainable"
                    ),
                    "epochs_11_through_40": "complete_graph_trainable",
                    "attachment_epoch_count": 40,
                },
                "same_identity_order_at_fixed_seed": True,
            }
        )


class OfflineExpertDataset(
    torch.utils.data.Dataset if torch is not None else object
):
    """Identity-preserving offline particle view with optional teacher logits."""

    def __init__(
        self,
        *,
        tokens: np.ndarray,
        mask: np.ndarray,
        labels: np.ndarray,
        identities: Sequence[str],
        teacher_logits: Mapping[str, np.ndarray] | None = None,
        region_trees: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        _require_torch()
        self.tokens = np.asarray(tokens, dtype=np.float32)
        self.mask = np.asarray(mask, dtype=bool)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.identities = tuple(str(value) for value in identities)
        if self.tokens.ndim != 3 or self.tokens.shape[-1] != 14:
            raise ValueError("offline tokens must have shape [B,N,14]")
        if self.mask.shape != self.tokens.shape[:2]:
            raise ValueError("offline mask shape differs from tokens")
        if self.labels.shape != (len(self.tokens),):
            raise ValueError("offline label shape differs")
        if len(self.identities) != len(self.tokens):
            raise ValueError("offline identity count differs")
        if len(self.identities) != len(set(self.identities)):
            raise ValueError("offline identities are not unique")
        if any(not value for value in self.identities):
            raise ValueError("offline identity is empty")
        if bool((self.mask.sum(axis=1) <= 0).any()):
            raise ValueError("offline event has no valid particle")
        if not np.isfinite(self.tokens).all():
            raise FloatingPointError("offline tokens are nonfinite")
        if bool(((self.labels < 0) | (self.labels >= 10)).any()):
            raise ValueError("offline labels lie outside 0..9")
        self.teacher_logits = {
            str(name): np.asarray(value, dtype=np.float32)
            for name, value in (teacher_logits or {}).items()
        }
        for name, value in self.teacher_logits.items():
            if value.shape != (len(self.tokens), 10):
                raise ValueError(f"teacher {name} logits have the wrong shape")
            if not np.isfinite(value).all():
                raise FloatingPointError(f"teacher {name} logits are nonfinite")
        self.region_trees = (
            None if region_trees is None else tuple(region_trees)
        )
        if self.region_trees is not None and len(self.region_trees) != len(self):
            raise ValueError("offline REGION tree count differs")

    def __len__(self) -> int:
        return int(len(self.labels))

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "tokens": self.tokens[index],
            "mask": self.mask[index],
            "label": self.labels[index],
            "identity": self.identities[index],
            "teacher_logits": {
                name: value[index] for name, value in self.teacher_logits.items()
            },
            "region_tree": (
                None if self.region_trees is None else self.region_trees[index]
            ),
        }


class DeterministicExpertSampler(
    torch.utils.data.Sampler if torch is not None else object
):
    def __init__(self, data_source: Sequence[Any], *, seed: int) -> None:
        _require_torch()
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 1

    def set_epoch(self, epoch: int) -> None:
        if int(epoch) <= 0:
            raise ValueError("sampler epoch is one-based")
        self.epoch = int(epoch)

    def __iter__(self):
        generator = _require_torch().Generator()
        generator.manual_seed(self.seed * 1_000_003 + self.epoch)
        return iter(
            _require_torch()
            .randperm(len(self.data_source), generator=generator)
            .tolist()
        )

    def __len__(self) -> int:
        return len(self.data_source)


def collate_offline_expert_batch(
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    module = _require_torch()
    from jetclass_fresh.part_inputs import (
        build_particle_transformer_inputs_from_tokens,
    )

    if not samples:
        raise ValueError("cannot collate an empty offline batch")
    tokens = np.stack([sample["tokens"] for sample in samples]).astype(
        np.float32, copy=False
    )
    mask = np.stack([sample["mask"] for sample in samples]).astype(
        bool, copy=False
    )
    labels = np.asarray(
        [sample["label"] for sample in samples], dtype=np.int64
    )
    inputs = build_particle_transformer_inputs_from_tokens(
        tokens,
        mask,
        labels=labels,
        source_view="offline",
    )
    output: dict[str, Any] = {
        "features": module.from_numpy(inputs.pf_features).float(),
        "vectors": module.from_numpy(inputs.pf_vectors).float(),
        "mask": module.from_numpy(inputs.pf_mask).bool(),
        "raw_tokens": module.from_numpy(tokens).float(),
        "labels": module.from_numpy(labels).long(),
        "event_identities": [str(sample["identity"]) for sample in samples],
    }
    teacher_names = set(samples[0].get("teacher_logits", {}))
    if any(set(sample.get("teacher_logits", {})) != teacher_names for sample in samples):
        raise ValueError("teacher-logit fields drifted within a batch")
    if teacher_names:
        output["teacher_logits"] = {
            name: module.from_numpy(
                np.stack(
                    [sample["teacher_logits"][name] for sample in samples]
                ).astype(np.float32, copy=False)
            )
            for name in sorted(teacher_names)
        }
    trees = [sample.get("region_tree") for sample in samples]
    if any(tree is not None for tree in trees):
        if not all(tree is not None for tree in trees):
            raise ValueError("offline batch mixes present and absent REGION trees")
        output["region_trees"] = trees
    return output


def make_offline_expert_loader(
    dataset: OfflineExpertDataset,
    *,
    seed: int,
    training: bool,
    batch_size: int,
    num_workers: int = 0,
) -> Any:
    module = _require_torch()
    if int(batch_size) <= 0 or int(num_workers) != 0:
        raise ValueError("offline loader configuration is invalid")
    sampler = (
        DeterministicExpertSampler(dataset, seed=seed)
        if training
        else module.utils.data.SequentialSampler(dataset)
    )
    return module.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        sampler=sampler,
        num_workers=0,
        drop_last=False,
        collate_fn=collate_offline_expert_batch,
    )


def _teacher_probabilities(
    *,
    loss_id: str,
    teacher_logits: Mapping[str, Any] | None,
    temperature: float,
) -> Any | None:
    module = _require_torch()
    candidate = EXPERT_LOSS_CANDIDATES[str(loss_id)]
    teacher = candidate["teacher"]
    if teacher is None:
        if teacher_logits:
            raise ValueError("ELOSS_CE must not consume teacher logits")
        return None
    if not isinstance(teacher_logits, Mapping):
        raise ValueError(f"{loss_id} requires identity-bound teacher logits")
    if teacher == "MEAN_PROBABILITY_O_BASE_O_FULLREL":
        required = ("O_BASE", "O_FULLREL")
        if set(teacher_logits) != set(required):
            raise ValueError(
                "teacher ensemble fields differ from O_BASE and O_FULLREL"
            )
        probabilities = [
            module.softmax(teacher_logits[name].detach() / temperature, dim=-1)
            for name in required
        ]
        return 0.5 * (probabilities[0] + probabilities[1])
    if set(teacher_logits) != {teacher}:
        raise ValueError(f"{loss_id} requires only teacher {teacher}")
    return module.softmax(
        teacher_logits[teacher].detach() / temperature, dim=-1
    )


def offline_expert_objective(
    logits: Any,
    labels: Any,
    *,
    loss_id: str,
    teacher_logits: Mapping[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    module = _require_torch()
    if loss_id not in EXPERT_LOSS_CANDIDATES:
        raise ValueError("unknown expert-loss candidate")
    if logits.ndim != 2 or int(logits.shape[1]) != 10:
        raise ValueError("offline expert logits must have shape [B,10]")
    if tuple(labels.shape) != (int(logits.shape[0]),):
        raise ValueError("offline expert labels must have shape [B]")
    if not bool(module.isfinite(logits).all()):
        raise FloatingPointError("offline expert logits are nonfinite")
    if bool(((labels < 0) | (labels >= 10)).any()):
        raise ValueError("offline expert labels lie outside 0..9")
    temperature = 2.0
    candidate = EXPERT_LOSS_CANDIDATES[loss_id]
    ce = module.nn.functional.cross_entropy(
        logits, labels.long(), reduction="mean"
    )
    teacher_probability = _teacher_probabilities(
        loss_id=loss_id,
        teacher_logits=teacher_logits,
        temperature=temperature,
    )
    if teacher_probability is None:
        kd = logits.new_zeros(())
    else:
        if tuple(teacher_probability.shape) != tuple(logits.shape):
            raise ValueError("teacher and student logit shapes differ")
        if not bool(module.isfinite(teacher_probability).all()):
            raise FloatingPointError("teacher probability is nonfinite")
        kd = module.nn.functional.kl_div(
            module.log_softmax(logits / temperature, dim=-1),
            teacher_probability,
            reduction="batchmean",
        ) * (temperature**2)
    total = (
        float(candidate["cross_entropy_weight"]) * ce
        + float(candidate["kd_weight"]) * kd
    )
    if not bool(module.isfinite(total)):
        raise FloatingPointError("offline expert objective is nonfinite")
    return total, {
        "cross_entropy": ce.detach(),
        "knowledge_distillation": kd.detach(),
        "total": total.detach(),
    }


def copy_obase_particle_backbone(
    model: Any,
    source_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Copy only compatible ordinary embed/particle-block tensors."""

    module = _require_torch()
    target = model.state_dict()
    copied: list[str] = []
    source_names: list[str] = []
    for target_name, target_value in target.items():
        if not isinstance(target_value, module.Tensor):
            continue
        prefix = "particle_encoder.mod."
        if not target_name.startswith(prefix):
            continue
        suffix = target_name[len(prefix) :]
        if not (suffix.startswith("embed.") or suffix.startswith("blocks.")):
            continue
        candidates = (
            target_name,
            f"mod.{suffix}",
            suffix,
        )
        matches = [
            name
            for name in candidates
            if name in source_state
            and isinstance(source_state[name], module.Tensor)
            and tuple(source_state[name].shape) == tuple(target_value.shape)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"ordinary initialization cannot uniquely resolve {target_name}"
            )
        source_name = matches[0]
        target[target_name] = source_state[source_name].detach().to(
            dtype=target_value.dtype,
            device=target_value.device,
        )
        copied.append(target_name)
        source_names.append(source_name)
    if not copied:
        raise ValueError("ordinary initialization copied no particle weights")
    model.load_state_dict(target, strict=True)
    return {
        "copied_target_names": copied,
        "copied_source_names": source_names,
        "copied_tensor_count": len(copied),
        "relation_or_token_parameter_copied": False,
    }


def build_attachment_pretraining_record(
    *,
    checkpoint_sha256: str,
    epochs: int,
    label_presentations: int,
    optimizer_updates: int,
    walltime_seconds: float,
) -> dict[str, Any]:
    if min(int(epochs), int(label_presentations), int(optimizer_updates)) <= 0:
        raise ValueError("attachment pretraining counts must be positive")
    if not math.isfinite(float(walltime_seconds)) or walltime_seconds < 0:
        raise ValueError("attachment pretraining walltime is invalid")
    return with_content_hash(
        {
            "contract": ATTACHMENT_PRETRAINING_CONTRACT,
            "schema_version": 1,
            "checkpoint_sha256": require_sha256(
                checkpoint_sha256, name="checkpoint_sha256"
            ),
            "epochs": int(epochs),
            "label_presentations": int(label_presentations),
            "optimizer_updates": int(optimizer_updates),
            "walltime_seconds": float(walltime_seconds),
            "included_in_long_baseline_capacity_matching": True,
            "attachment_epochs_additional": 40,
        }
    )


def validate_attachment_pretraining_record(payload: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        payload, expected_contract=ATTACHMENT_PRETRAINING_CONTRACT
    )
    expected = build_attachment_pretraining_record(
        checkpoint_sha256=payload.get("checkpoint_sha256"),
        epochs=int(payload.get("epochs", 0)),
        label_presentations=int(payload.get("label_presentations", 0)),
        optimizer_updates=int(payload.get("optimizer_updates", 0)),
        walltime_seconds=float(payload.get("walltime_seconds", -1)),
    )
    actual = dict(payload)
    actual.pop("content_hash", None)
    actual.pop("source", None)
    expected.pop("content_hash")
    if actual != expected:
        raise ValueError("attachment pretraining record differs")
    return digest


def apply_attachment_trainability(model: Any, *, epoch: int) -> dict[str, Any]:
    if int(epoch) not in range(1, 41):
        raise ValueError("attachment epoch lies outside 1..40")
    phase = (
        "backbone_frozen"
        if epoch <= 5
        else "last_four_blocks"
        if epoch <= 10
        else "complete_graph"
    )
    trainable: list[str] = []
    frozen: list[str] = []
    for name, parameter in model.named_parameters():
        ordinary = name.startswith("particle_encoder.mod.embed.") or name.startswith(
            "particle_encoder.mod.blocks."
        )
        allowed = True
        if ordinary and phase == "backbone_frozen":
            allowed = False
        elif ordinary and phase == "last_four_blocks":
            if name.startswith("particle_encoder.mod.embed."):
                allowed = False
            elif name.startswith("particle_encoder.mod.blocks."):
                block_text = name.split("particle_encoder.mod.blocks.", 1)[1]
                block_index = int(block_text.split(".", 1)[0])
                allowed = block_index >= 4
        parameter.requires_grad_(allowed)
        (trainable if allowed else frozen).append(name)
    return {
        "phase": phase,
        "trainable_parameter_names": trainable,
        "frozen_parameter_names": frozen,
    }


def preferred_expert_epoch(
    rows: Sequence[Mapping[str, Any]],
    *,
    accuracy_window: float = 0.0001,
) -> Mapping[str, Any]:
    if not rows:
        raise ValueError("expert checkpoint selection requires completed epochs")
    checked = []
    for row in rows:
        epoch = int(row["epoch"])
        metrics = row["val_stop"]
        accuracy = float(metrics["accuracy"])
        cross_entropy = float(metrics["cross_entropy"])
        if not (math.isfinite(accuracy) and math.isfinite(cross_entropy)):
            raise FloatingPointError("checkpoint metric is nonfinite")
        checked.append((row, epoch, accuracy, cross_entropy))
    maximum = max(item[2] for item in checked)
    eligible = [
        item for item in checked if maximum - item[2] <= accuracy_window
    ]
    return min(eligible, key=lambda item: (item[3], item[1]))[0]


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    _require_torch().manual_seed(int(seed))
    if _require_torch().cuda.is_available():
        _require_torch().cuda.manual_seed_all(int(seed))


def _rng_state() -> dict[str, Any]:
    module = _require_torch()
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": module.random.get_rng_state(),
        "torch_cuda": (
            module.cuda.get_rng_state_all() if module.cuda.is_available() else None
        ),
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    module = _require_torch()
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    module.random.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") is not None and module.cuda.is_available():
        module.cuda.set_rng_state_all(state["torch_cuda"])


def _cpu_state(model: Any) -> dict[str, Any]:
    module = _require_torch()
    return {
        name: (
            value.detach().cpu().clone()
            if isinstance(value, module.Tensor)
            else copy.deepcopy(value)
        )
        for name, value in model.state_dict().items()
    }


def _state_sha256(state: Mapping[str, Any]) -> str:
    module = _require_torch()
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        if isinstance(value, module.Tensor):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(canonical_json_bytes(list(tensor.shape)))
            digest.update(
                tensor.reshape(-1).view(module.uint8).numpy().tobytes()
            )
        else:
            digest.update(canonical_json_bytes(value))
        digest.update(b"\0")
    return digest.hexdigest()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    module = _require_torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        module.save(dict(payload), temporary)
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _move_batch(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    module = _require_torch()
    return {
        name: value.to(device) if isinstance(value, module.Tensor) else value
        for name, value in batch.items()
    }


def _model_inputs(batch: Mapping[str, Any]) -> dict[str, Any]:
    aliases = {
        "features": ("features",),
        "vectors": ("vectors", "lorentz_vectors"),
        "mask": ("mask",),
        "raw_tokens": ("raw_tokens", "tokens"),
        "region_trees": ("region_trees",),
    }
    output = {}
    for target, candidates in aliases.items():
        for candidate in candidates:
            if candidate in batch:
                output[target] = batch[candidate]
                break
    return output


def _forward(model: Any, batch: Mapping[str, Any], *, details: bool) -> Any:
    inputs = _model_inputs(batch)
    if details:
        return model(return_details=True, **inputs)
    return model(**inputs)


def _resolve_teacher_batch(batch: Mapping[str, Any]) -> Mapping[str, Any] | None:
    value = batch.get("teacher_logits")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("teacher_logits batch field must be a mapping")
    return value


def _basic_validation(
    model: Any,
    loader: Iterable[Mapping[str, Any]],
    *,
    device: Any,
) -> dict[str, Any]:
    module = _require_torch()
    was_training = bool(model.training)
    model.eval()
    total = 0
    correct = 0
    ce_sum = 0.0
    try:
        with module.no_grad():
            for raw in loader:
                batch = _move_batch(raw, device)
                labels = batch.get("labels")
                if labels is None:
                    raise ValueError("val_stop batch lacks labels")
                logits = _forward(model, batch, details=False)
                if not bool(module.isfinite(logits).all()):
                    raise FloatingPointError("val_stop logits are nonfinite")
                ce_sum += float(
                    module.nn.functional.cross_entropy(
                        logits, labels.long(), reduction="sum"
                    )
                    .float()
                    .cpu()
                )
                correct += int((logits.argmax(dim=-1) == labels).sum().cpu())
                total += int(labels.numel())
    finally:
        model.train(was_training)
    if total <= 0:
        raise ValueError("val_stop loader is empty")
    return {
        "event_count": total,
        "accuracy": correct / total,
        "cross_entropy": ce_sum / total,
    }


def _add_nested(left: Any, right: Any) -> Any:
    if isinstance(right, list):
        if left is None:
            return copy.deepcopy(right)
        if len(left) != len(right):
            raise ValueError("attention statistic shape drifted")
        return [_add_nested(a, b) for a, b in zip(left, right)]
    return float(0 if left is None else left) + float(right)


def collect_expert_diagnostics(
    model: Any,
    loader: Iterable[Mapping[str, Any]],
    *,
    device: Any,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    module = _require_torch()
    tokenizer = getattr(model, "tokenizer", None)
    toggle = getattr(tokenizer, "set_collect_attention_diagnostics", None)
    attention_reader = getattr(
        tokenizer, "attention_sufficient_statistics", None
    )
    was_training = bool(model.training)
    model.eval()
    if callable(toggle):
        toggle(True)
    logits_rows = []
    label_rows = []
    identities: list[str] = []
    token_sum = None
    token_square_sum = None
    token_norm_sum = None
    token_active_count = None
    count = 0
    attention: list[dict[str, Any] | None] | None = None
    try:
        with module.no_grad():
            for raw in loader:
                batch = _move_batch(raw, device)
                labels = batch.get("labels")
                if labels is None:
                    raise ValueError("diagnostic batch lacks labels")
                output = _forward(model, batch, details=True)
                if not isinstance(output, Mapping) or "tokens" not in output:
                    raise ValueError(
                        "expert diagnostic forward must return tokens and logits"
                    )
                tokens = output["tokens"].detach().float()
                logits = output["logits"].detach().float()
                if not (
                    bool(module.isfinite(tokens).all())
                    and bool(module.isfinite(logits).all())
                ):
                    raise FloatingPointError("expert diagnostic output is nonfinite")
                current_sum = tokens.sum(dim=0).cpu()
                current_square = tokens.square().sum(dim=0).cpu()
                current_norm = tokens.norm(dim=-1).sum(dim=0).cpu()
                current_active = (tokens.norm(dim=-1) > 1.0e-8).sum(dim=0).cpu()
                token_sum = current_sum if token_sum is None else token_sum + current_sum
                token_square_sum = (
                    current_square
                    if token_square_sum is None
                    else token_square_sum + current_square
                )
                token_norm_sum = (
                    current_norm
                    if token_norm_sum is None
                    else token_norm_sum + current_norm
                )
                token_active_count = (
                    current_active
                    if token_active_count is None
                    else token_active_count + current_active
                )
                count += int(tokens.shape[0])
                logits_rows.append(logits.cpu().numpy())
                label_rows.append(labels.detach().cpu().numpy())
                batch_identities = raw.get("event_identities")
                if batch_identities is None:
                    batch_identities = [
                        f"diagnostic_row_{len(identities) + index}"
                        for index in range(int(tokens.shape[0]))
                    ]
                identities.extend(str(value) for value in batch_identities)
                if callable(attention_reader):
                    rows = attention_reader()
                    if attention is None:
                        attention = [None] * len(rows)
                    if len(rows) != len(attention):
                        raise ValueError("tokenizer attention block count drifted")
                    for index, row in enumerate(rows):
                        if row is None:
                            continue
                        if attention[index] is None:
                            attention[index] = {
                                "event_count": 0,
                                "head_count": int(row["head_count"]),
                                "slot_count": int(row["slot_count"]),
                                "entropy_sum_by_head_slot": None,
                                "maximum_sum_by_head_slot": None,
                                "probability_square_sum_by_head_slot": None,
                            }
                        target = attention[index]
                        if (
                            target["head_count"] != int(row["head_count"])
                            or target["slot_count"] != int(row["slot_count"])
                        ):
                            raise ValueError("attention diagnostic dimensions drifted")
                        target["event_count"] += int(row["event_count"])
                        for name in (
                            "entropy_sum_by_head_slot",
                            "maximum_sum_by_head_slot",
                            "probability_square_sum_by_head_slot",
                        ):
                            target[name] = _add_nested(target[name], row[name])
    finally:
        if callable(toggle):
            toggle(False)
        model.train(was_training)
    if count <= 0:
        raise ValueError("expert diagnostics received an empty loader")
    if len(identities) != len(set(identities)):
        raise ValueError("expert diagnostic identities are not unique")
    mean = token_sum / count
    variance = (token_square_sum / count - mean.square()).clamp_min(0.0)
    slot_rms = (token_square_sum.sum(dim=-1) / (count * mean.shape[-1])).sqrt()
    attention_output = []
    for row in attention or []:
        if row is None:
            attention_output.append(None)
            continue
        denominator = int(row["event_count"])
        attention_output.append(
            {
                **row,
                "mean_entropy_by_head_slot": (
                    np.asarray(row["entropy_sum_by_head_slot"], dtype=np.float64)
                    / denominator
                ).tolist(),
                "mean_maximum_by_head_slot": (
                    np.asarray(row["maximum_sum_by_head_slot"], dtype=np.float64)
                    / denominator
                ).tolist(),
                "mean_probability_square_sum_by_head_slot": (
                    np.asarray(
                        row["probability_square_sum_by_head_slot"],
                        dtype=np.float64,
                    )
                    / denominator
                ).tolist(),
            }
        )
    logits = np.concatenate(logits_rows, axis=0).astype(np.float32, copy=False)
    labels = np.concatenate(label_rows, axis=0).astype(np.int64, copy=False)
    diagnostic = with_content_hash(
        {
            "contract": EXPERT_DIAGNOSTICS_CONTRACT,
            "schema_version": 1,
            "event_count": count,
            "token_shape": [count, int(mean.shape[0]), int(mean.shape[1])],
            "token_norm": {
                "mean_l2_by_slot": (token_norm_sum / count).tolist(),
                "rms_by_slot": slot_rms.tolist(),
            },
            "token_utilization": {
                "mean_centered_channel_variance_by_slot": (
                    variance.mean(dim=-1).tolist()
                ),
                "active_fraction_by_slot": (
                    token_active_count.float() / count
                ).tolist(),
                "active_norm_threshold": 1.0e-8,
            },
            "slot_to_particle_attention_sufficient_statistics": attention_output,
            "full_attention_tensors_retained": False,
        }
    )
    metrics = evaluate_classification(logits, labels, split="val_stop")
    return diagnostic, {
        "logits": logits,
        "labels": labels,
        "identities": np.asarray(identities),
        "metrics": metrics,
    }


def _publish_predictions(
    path: Path,
    *,
    logits: np.ndarray,
    labels: np.ndarray,
    identities: np.ndarray,
) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp.npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(
            temporary,
            logits=np.asarray(logits, dtype=np.float32),
            labels=np.asarray(labels, dtype=np.int64),
            identities=np.asarray(identities),
        )
        encoded = temporary.read_bytes()
        digest = hashlib.sha256(encoded).hexdigest()
        if path.exists():
            if path.is_symlink() or path.read_bytes() != encoded:
                raise FileExistsError(
                    "refusing to overwrite different val_stop predictions"
                )
            status = "already_present"
        else:
            try:
                os.link(temporary, path)
            except FileExistsError as exc:
                raise FileExistsError(
                    "val_stop prediction artifact appeared during publication"
                ) from exc
            status = "published"
    finally:
        if temporary.exists():
            temporary.unlink()
    return {"file_sha256": digest, "status": status}


def _validate_reusable_expert_registration(
    *,
    root: Path,
    registration_path: Path,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate a completed row before allowing idempotent reuse."""

    expected_fields = dict(expected)
    maximum_epochs = int(expected_fields.pop("maximum_epochs"))
    registration = load_hashed_json(
        registration_path,
        expected_contract=EXPERT_REGISTRATION_CONTRACT,
    )
    drift = {
        name: (registration.get(name), value)
        for name, value in expected_fields.items()
        if registration.get(name) != value
    }
    if drift:
        raise ValueError(f"reusable expert registration lineage drifted: {drift}")
    if not (
        registration.get("fixed_epoch_budget_completed") is True
        and registration.get("stopped_early") is False
        and registration.get("performance_based_termination") is False
        and int(registration.get("epochs_completed", 0))
        == maximum_epochs
    ):
        raise ValueError("reusable expert row did not complete its fixed budget")
    retained = registration.get("retained_checkpoints")
    if retained != ["best_model_val.pt"]:
        raise ValueError("reusable expert checkpoint retention differs")
    artifact_contracts = {
        "training_curves.json": (
            EXPERT_CURVES_CONTRACT,
            "training_curves_sha256",
        ),
        "expert_diagnostics.json": (
            EXPERT_DIAGNOSTICS_CONTRACT,
            "diagnostics_sha256",
        ),
        "val_stop_metrics.json": (
            "retb_classification_metrics_v1",
            "val_stop_metrics_sha256",
        ),
    }
    for filename, (contract, hash_field) in artifact_contracts.items():
        artifact = load_hashed_json(
            root / filename,
            expected_contract=contract,
        )
        if artifact["content_hash"] != registration.get(hash_field):
            raise ValueError(f"reusable expert {filename} hash differs")
    checkpoint = root / "best_model_val.pt"
    predictions = root / "val_stop_predictions.npz"
    if (
        not checkpoint.is_file()
        or checkpoint.is_symlink()
        or _file_sha256(checkpoint)
        != registration.get("checkpoint_sha256")
    ):
        raise ValueError("reusable expert checkpoint bytes differ")
    if (
        not predictions.is_file()
        or predictions.is_symlink()
        or _file_sha256(predictions)
        != registration.get("val_stop_prediction_file_sha256")
    ):
        raise ValueError("reusable expert prediction bytes differ")
    if (root / "last.pt").exists() or (root / ".checkpoint_frontier").exists():
        raise ValueError("completed reusable expert row retains stale training state")
    return registration


def _resolve_precision(device: Any) -> dict[str, Any]:
    module = _require_torch()
    resolved = module.device(device)
    if resolved.type != "cuda":
        return {
            "mode": "fp32",
            "autocast": False,
            "dtype": None,
            "gradient_scaler": False,
        }
    if module.cuda.is_bf16_supported():
        return {
            "mode": "bf16",
            "autocast": True,
            "dtype": module.bfloat16,
            "gradient_scaler": False,
        }
    return {
        "mode": "fp16",
        "autocast": True,
        "dtype": module.float16,
        "gradient_scaler": True,
    }


def _validate_model_semantics(
    model: Any,
    *,
    configuration: Mapping[str, Any],
    strict: bool,
) -> None:
    encoder = getattr(model, "particle_encoder", None)
    checks = {
        "expert_id": (
            getattr(encoder, "expert_id", None),
            configuration.get("expert_id"),
        ),
        "topology": (
            getattr(encoder, "topology", None),
            configuration.get("topology"),
        ),
        "particle_dropout": (
            getattr(encoder, "particle_dropout", None),
            configuration.get("particle_dropout"),
        ),
        "measurement_embedding": (
            getattr(encoder, "measurement_embedding_enabled", None),
            configuration.get("measurement_embedding"),
        ),
        "shape_id": (
            getattr(model, "shape_id", None),
            configuration.get("shape_id"),
        ),
        "token_count": (
            getattr(model, "token_count", None),
            configuration.get("token_count"),
        ),
        "token_dimension": (
            getattr(model, "token_dimension", None),
            configuration.get("token_dimension"),
        ),
        "tokenizer_mode": (
            getattr(model, "tokenizer_mode", None),
            configuration.get("tokenizer_mode"),
        ),
    }
    drift = {
        name: (actual, expected)
        for name, (actual, expected) in checks.items()
        if (strict or actual is not None) and actual != expected
    }
    if drift:
        raise ValueError(f"instantiated expert semantics differ: {drift}")


def train_offline_expert(
    *,
    model: Any,
    train_loader: Any,
    val_stop_loader: Any,
    output_dir: str | Path,
    run_record: Mapping[str, Any],
    run_registry_sha256: str,
    step3_bundle_sha256: str,
    global_determinism_sha256: str,
    expert_loss_registry: Mapping[str, Any],
    lineage_hashes: Mapping[str, str],
    config: OfflineExpertTrainingConfig,
    device: str | Any = "cpu",
    initialization_state: Mapping[str, Any] | None = None,
    initialization_checkpoint_sha256: str | None = None,
    attachment_pretraining_record: Mapping[str, Any] | None = None,
    resource_profile: Mapping[str, Any] | None = None,
    teacher_checkpoint_hashes: Mapping[str, str] | None = None,
    teacher_logits_manifest: Mapping[str, Any] | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Train all epochs, select on val_stop, and retain one inference checkpoint."""

    module = _require_torch()
    config.validate()
    validate_expert_loss_registry(expert_loss_registry)
    if resource_profile is None and config.campaign_profile == "production":
        raise ValueError("production expert training requires a resource profile")
    resource_profile_sha256 = (
        None
        if resource_profile is None
        else validate_content_hash(resource_profile)
    )
    if run_record.get("seed") != config.seed:
        raise ValueError("run record and training seed differ")
    configuration = run_record.get("configuration", {})
    expected_configuration = {
        "initialization": config.initialization,
        "loss_id": config.loss_id,
        "learning_rate": config.learning_rate,
        "particle_dropout": config.particle_dropout,
    }
    drift = {
        name: (configuration.get(name), value)
        for name, value in expected_configuration.items()
        if configuration.get(name) != value
    }
    if drift:
        raise ValueError(f"run record differs from training configuration: {drift}")
    _validate_model_semantics(
        model,
        configuration=configuration,
        strict=config.campaign_profile == "production",
    )
    training_contract = config.artifact(
        global_determinism_sha256=global_determinism_sha256,
        expert_loss_registry_sha256=expert_loss_registry["content_hash"],
    )
    parents = {
        str(name): require_sha256(value, name=f"lineage_hashes.{name}")
        for name, value in sorted(lineage_hashes.items())
    }
    run_registry_sha = require_sha256(
        run_registry_sha256, name="run_registry_sha256"
    )
    step3_sha = require_sha256(
        step3_bundle_sha256, name="step3_bundle_sha256"
    )
    teacher = EXPERT_LOSS_CANDIDATES[config.loss_id]["teacher"]
    teacher_hashes = {
        str(name): require_sha256(
            value, name=f"teacher_checkpoint_hashes.{name}"
        )
        for name, value in sorted((teacher_checkpoint_hashes or {}).items())
    }
    required_teachers = (
        set()
        if teacher is None
        else {"O_BASE", "O_FULLREL"}
        if teacher == "MEAN_PROBABILITY_O_BASE_O_FULLREL"
        else {str(teacher)}
    )
    if set(teacher_hashes) != required_teachers:
        raise ValueError(
            "teacher checkpoint hashes differ from the expert-loss policy"
        )
    if not required_teachers and teacher_logits_manifest is not None:
        raise ValueError("ELOSS_CE cannot consume a teacher-logit manifest")
    if (
        required_teachers
        and config.campaign_profile == "production"
        and teacher_logits_manifest is None
    ):
        raise ValueError("production KD requires a teacher-logit manifest")
    if teacher_logits_manifest is not None:
        manifest_sha = validate_teacher_logits_manifest(
            teacher_logits_manifest
        )
        expected_manifest = {
            "model_train_npz_sha256": parents.get("model_train_inputs"),
            "val_stop_npz_sha256": parents.get("val_stop_inputs"),
            "teacher_checkpoint_hashes": teacher_hashes,
            "teacher_fields": sorted(required_teachers),
        }
        manifest_drift = {
            name: (teacher_logits_manifest.get(name), expected)
            for name, expected in expected_manifest.items()
            if teacher_logits_manifest.get(name) != expected
        }
        if manifest_drift:
            raise ValueError(
                f"teacher-logit manifest lineage drifted: {manifest_drift}"
            )
        if parents.get("teacher_logits_manifest") != manifest_sha:
            raise ValueError("teacher-logit manifest parent hash differs")
    for name, digest in teacher_hashes.items():
        if parents.get(f"teacher_{name}") != digest:
            raise ValueError(f"teacher {name} checkpoint lineage differs")
    if config.initialization == "INIT_SCRATCH":
        if initialization_state is not None or initialization_checkpoint_sha256:
            raise ValueError("scratch initialization cannot consume a checkpoint")
        if attachment_pretraining_record is not None:
            raise ValueError("scratch initialization cannot consume pretraining")
    else:
        if initialization_state is None:
            raise ValueError("warm initialization requires a source state")
        source_sha = require_sha256(
            initialization_checkpoint_sha256,
            name="initialization_checkpoint_sha256",
        )
        if parents.get("initialization_checkpoint") != source_sha:
            raise ValueError("initialization checkpoint lineage differs")
    if config.initialization == "INIT_ATTACH_AFTER_PRETRAIN":
        if attachment_pretraining_record is None:
            raise ValueError("attachment initialization requires pretraining")
        pretraining_sha = validate_attachment_pretraining_record(
            attachment_pretraining_record
        )
        if parents.get("attachment_pretraining") != pretraining_sha:
            raise ValueError("attachment pretraining lineage differs")
        if attachment_pretraining_record.get("checkpoint_sha256") != source_sha:
            raise ValueError(
                "attachment pretraining record names another source checkpoint"
            )
    elif attachment_pretraining_record is not None:
        raise ValueError("only attachment initialization accepts pretraining")

    root = Path(output_dir)
    if root.exists() and root.is_symlink():
        raise ValueError("expert output directory cannot be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    registration_path = root / "checkpoint_registration.json"
    if registration_path.is_file():
        return _validate_reusable_expert_registration(
            root=root,
            registration_path=registration_path,
            expected={
                "run_id": run_record["run_id"],
                "seed": config.seed,
                "training_contract_sha256": training_contract["content_hash"],
                "run_registry_sha256": run_registry_sha,
                "step3_bundle_sha256": step3_sha,
                "lineage_hashes": parents,
                "teacher_checkpoint_hashes": teacher_hashes,
                "resource_profile_sha256": resource_profile_sha256,
                "maximum_epochs": config.maximum_epochs,
            },
        )
    best_path = root / "best_model_val.pt"
    last_path = root / "last.pt"
    candidate_root = root / ".checkpoint_frontier"
    candidate_root.mkdir(exist_ok=True)
    resolved_device = module.device(device)
    if config.campaign_profile == "production":
        if resolved_device.type != "cuda":
            raise RuntimeError("production expert training requires a CUDA GH200")
        if not module.cuda.is_bf16_supported():
            raise RuntimeError("production expert training requires CUDA BF16")
        if "GH200" not in module.cuda.get_device_name(resolved_device).upper():
            raise RuntimeError("production expert training requires a GH200 GPU")
    precision = _resolve_precision(resolved_device)
    model.to(resolved_device)

    rows: list[dict[str, Any]] = []
    update_ordinal = 0
    start_epoch = 1
    initialization_report = {
        "mode": config.initialization,
        "source_checkpoint_sha256": initialization_checkpoint_sha256,
        "copy": None,
    }
    if resume and last_path.is_file():
        resume_state = module.load(last_path, map_location="cpu", weights_only=False)
        if resume_state.get("contract") != EXPERT_CHECKPOINT_CONTRACT:
            raise ValueError("resume checkpoint contract differs")
        expected_resume = {
            "run_id": run_record["run_id"],
            "training_contract_sha256": training_contract["content_hash"],
            "run_registry_sha256": run_registry_sha,
            "step3_bundle_sha256": step3_sha,
            "resource_profile_sha256": resource_profile_sha256,
        }
        resume_drift = {
            name: (resume_state.get(name), expected)
            for name, expected in expected_resume.items()
            if resume_state.get(name) != expected
        }
        if resume_drift:
            raise ValueError(f"resume checkpoint lineage drifted: {resume_drift}")
        model.load_state_dict(resume_state["model_state_dict"], strict=True)
        rows = list(resume_state["rows"])
        update_ordinal = int(resume_state["optimizer_update_ordinal"])
        start_epoch = int(resume_state["epoch_completed"]) + 1
        initialization_report = dict(resume_state["initialization_report"])
    else:
        if best_path.exists() or last_path.exists():
            raise FileExistsError("nonresumable expert checkpoint state is present")
        _set_seed(config.seed)
        if initialization_state is not None:
            initialization_report["copy"] = copy_obase_particle_backbone(
                model, initialization_state
            )

    optimizer = module.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
    )
    if resume and last_path.is_file():
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        _restore_rng_state(resume_state["rng_state"])
    try:
        training_event_count = len(train_loader.dataset)
    except (AttributeError, TypeError):
        training_event_count = len(train_loader) * config.microbatch_size
    counts = optimizer_update_counts(
        training_event_count=training_event_count,
        maximum_epochs=config.maximum_epochs,
        microbatch_size=config.microbatch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    if len(train_loader) != counts["microbatches_per_epoch"]:
        raise ValueError("train loader length differs from the locked schedule")
    if hasattr(module, "amp") and hasattr(module.amp, "GradScaler"):
        try:
            scaler = module.amp.GradScaler(
                "cuda", enabled=precision["gradient_scaler"]
            )
        except TypeError:  # pragma: no cover - older torch
            scaler = module.amp.GradScaler(
                enabled=precision["gradient_scaler"]
            )
    else:  # pragma: no cover - older torch
        scaler = module.cuda.amp.GradScaler(
            enabled=precision["gradient_scaler"]
        )
    if resume and last_path.is_file() and resume_state.get("scaler_state_dict"):
        scaler.load_state_dict(resume_state["scaler_state_dict"])

    for epoch in range(start_epoch, config.maximum_epochs + 1):
        sampler = getattr(train_loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        if config.initialization == "INIT_ATTACH_AFTER_PRETRAIN":
            trainability = apply_attachment_trainability(model, epoch=epoch)
        else:
            for parameter in model.parameters():
                parameter.requires_grad_(True)
            trainability = {
                "phase": "complete_graph",
                "trainable_parameter_names": [
                    name for name, _ in model.named_parameters()
                ],
                "frozen_parameter_names": [],
            }
        model.train()
        optimizer.zero_grad(set_to_none=True)
        event_count = 0
        accumulation_events = 0
        component_sums = {
            "cross_entropy": 0.0,
            "knowledge_distillation": 0.0,
            "total": 0.0,
        }
        for batch_index, raw in enumerate(train_loader, start=1):
            batch = _move_batch(raw, resolved_device)
            labels = batch.get("labels")
            if labels is None or int(labels.numel()) <= 0:
                raise ValueError("training batch lacks nonempty labels")
            current_events = int(labels.numel())
            autocast = module.autocast(
                device_type=resolved_device.type,
                dtype=precision["dtype"],
                enabled=precision["autocast"],
            )
            with autocast:
                logits = _forward(model, batch, details=False)
                loss, components = offline_expert_objective(
                    logits,
                    labels,
                    loss_id=config.loss_id,
                    teacher_logits=_resolve_teacher_batch(batch),
                )
                loss_sum = loss * current_events
            scaler.scale(loss_sum).backward()
            for name in component_sums:
                component_sums[name] += (
                    float(components[name].float().cpu()) * current_events
                )
            event_count += current_events
            accumulation_events += current_events
            step_now = (
                batch_index % config.gradient_accumulation_steps == 0
                or batch_index == len(train_loader)
            )
            if not step_now:
                continue
            scaler.unscale_(optimizer)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(accumulation_events)
                    if not bool(module.isfinite(parameter.grad).all()):
                        raise FloatingPointError("expert gradient is nonfinite")
            gradient_norm = module.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                config.gradient_clip,
            )
            if not bool(module.isfinite(gradient_norm)):
                raise FloatingPointError("expert gradient norm is nonfinite")
            update_ordinal += 1
            learning_rate = scheduled_learning_rate(
                update_ordinal=update_ordinal,
                total_optimizer_updates=counts["total_optimizer_updates"],
                warmup_updates=counts["warmup_updates"],
                base_learning_rate=config.learning_rate,
                minimum_learning_rate=config.minimum_learning_rate,
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            accumulation_events = 0
        if event_count != training_event_count:
            raise RuntimeError("training epoch event count drifted")
        expected_updates = epoch * counts["optimizer_updates_per_epoch"]
        if update_ordinal != expected_updates:
            raise RuntimeError("optimizer update ordinal drifted")
        val_stop = _basic_validation(
            model, val_stop_loader, device=resolved_device
        )
        row = {
            "epoch": epoch,
            "optimizer_update_ordinal": update_ordinal,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_objective": {
                name: value / event_count
                for name, value in component_sums.items()
            },
            "attachment_phase": trainability["phase"],
            "trainable_parameter_count": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            "val_stop": val_stop,
        }
        rows.append(row)
        candidate_path = candidate_root / f"epoch_{epoch:03d}.pt"
        state = _cpu_state(model)
        _atomic_torch_save(
            {
                "contract": EXPERT_CHECKPOINT_CONTRACT,
                "schema_version": 1,
                "kind": "selection_frontier",
                "run_id": run_record["run_id"],
                "epoch": epoch,
                "model_state_dict": state,
                "model_state_sha256": _state_sha256(state),
            },
            candidate_path,
        )
        maximum = max(float(value["val_stop"]["accuracy"]) for value in rows)
        retained_epochs = {
            int(value["epoch"])
            for value in rows
            if maximum - float(value["val_stop"]["accuracy"])
            <= config.accuracy_window
        }
        for stale in candidate_root.glob("epoch_*.pt"):
            stale_epoch = int(stale.stem.split("_")[-1])
            if stale_epoch not in retained_epochs:
                stale.unlink()
        selected = preferred_expert_epoch(
            rows, accuracy_window=config.accuracy_window
        )
        selected_path = candidate_root / f"epoch_{int(selected['epoch']):03d}.pt"
        if not selected_path.is_file():
            raise RuntimeError("selected expert epoch is absent from frontier")
        selected_state = module.load(
            selected_path, map_location="cpu", weights_only=False
        )
        _atomic_torch_save(
            {
                "contract": EXPERT_CHECKPOINT_CONTRACT,
                "schema_version": 1,
                "kind": "selected_inference",
                "run_id": run_record["run_id"],
                "seed": config.seed,
                "epoch": int(selected["epoch"]),
                "training_contract_sha256": training_contract["content_hash"],
                "run_registry_sha256": run_registry_sha,
                "step3_bundle_sha256": step3_sha,
                "resource_profile_sha256": resource_profile_sha256,
                "lineage_hashes": parents,
                "run_record": dict(run_record),
                "model_state_dict": selected_state["model_state_dict"],
                "model_state_sha256": selected_state["model_state_sha256"],
                "selection_metrics": dict(selected["val_stop"]),
            },
            best_path,
        )
        current_state = _cpu_state(model)
        _atomic_torch_save(
            {
                "contract": EXPERT_CHECKPOINT_CONTRACT,
                "schema_version": 1,
                "kind": "resumable_last",
                "run_id": run_record["run_id"],
                "epoch_completed": epoch,
                "training_contract_sha256": training_contract["content_hash"],
                "run_registry_sha256": run_registry_sha,
                "step3_bundle_sha256": step3_sha,
                "resource_profile_sha256": resource_profile_sha256,
                "model_state_dict": current_state,
                "model_state_sha256": _state_sha256(current_state),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": (
                    scaler.state_dict() if precision["gradient_scaler"] else None
                ),
                "optimizer_update_ordinal": update_ordinal,
                "rows": rows,
                "initialization_report": initialization_report,
                "rng_state": _rng_state(),
            },
            last_path,
        )

    selected = preferred_expert_epoch(
        rows, accuracy_window=config.accuracy_window
    )
    best = module.load(best_path, map_location="cpu", weights_only=False)
    if int(best["epoch"]) != int(selected["epoch"]):
        raise RuntimeError("retained expert checkpoint differs from selector")
    model.load_state_dict(best["model_state_dict"], strict=True)
    model.to(resolved_device)
    diagnostics, prediction = collect_expert_diagnostics(
        model, val_stop_loader, device=resolved_device
    )
    if (
        abs(
            float(prediction["metrics"]["accuracy"])
            - float(selected["val_stop"]["accuracy"])
        )
        > 1.0e-12
        or abs(
            float(prediction["metrics"]["cross_entropy"])
            - float(selected["val_stop"]["cross_entropy"])
        )
        > 2.0e-6
    ):
        raise RuntimeError("selected checkpoint metrics are not reproducible")
    curves = with_content_hash(
        {
            "contract": EXPERT_CURVES_CONTRACT,
            "schema_version": 1,
            "run_id": run_record["run_id"],
            "training_contract_sha256": training_contract["content_hash"],
            "rows": rows,
            "selected_epoch": int(selected["epoch"]),
            "epochs_completed": len(rows),
            "fixed_budget_completed": len(rows) == config.maximum_epochs,
            "stopped_early": False,
            "performance_result_affected_execution": False,
            "planned_update_counts": counts,
            "precision_mode": precision["mode"],
        }
    )
    curves_publication = write_immutable_json(root / "training_curves.json", curves)
    diagnostic_publication = write_immutable_json(
        root / "expert_diagnostics.json", diagnostics
    )
    metrics_publication = write_immutable_json(
        root / "val_stop_metrics.json", prediction["metrics"]
    )
    prediction_publication = _publish_predictions(
        root / "val_stop_predictions.npz",
        logits=prediction["logits"],
        labels=prediction["labels"],
        identities=prediction["identities"],
    )
    checkpoint_sha = _file_sha256(best_path)
    registration = with_content_hash(
        {
            "contract": EXPERT_REGISTRATION_CONTRACT,
            "schema_version": 1,
            "run_id": run_record["run_id"],
            "seed": config.seed,
            "expert_id": configuration["expert_id"],
            "shape_id": configuration["shape_id"],
            "token_count": int(configuration["token_count"]),
            "token_dimension": int(configuration["token_dimension"]),
            "relation_family": configuration.get("relation_family"),
            "normalization_sha256": parents.get("relation_normalization"),
            "region_normalization_sha256": parents.get("region_normalization"),
            "tokenizer_mode": configuration["tokenizer_mode"],
            "topology": configuration["topology"],
            "loss_id": config.loss_id,
            "teacher_checkpoint_hashes": teacher_hashes,
            "teacher_logits_manifest_sha256": parents.get(
                "teacher_logits_manifest"
            ),
            "initialization": initialization_report,
            "attachment_pretraining_sha256": parents.get(
                "attachment_pretraining"
            ),
            "training_contract_sha256": training_contract["content_hash"],
            "run_registry_sha256": run_registry_sha,
            "step3_bundle_sha256": step3_sha,
            "lineage_hashes": parents,
            "checkpoint_file": best_path.name,
            "checkpoint_sha256": checkpoint_sha,
            "model_state_sha256": best["model_state_sha256"],
            "selected_epoch": int(selected["epoch"]),
            "selected_val_stop": dict(selected["val_stop"]),
            "epochs_completed": len(rows),
            "fixed_epoch_budget_completed": len(rows) == config.maximum_epochs,
            "stopped_early": False,
            "performance_based_termination": False,
            "optimizer_updates_completed": update_ordinal,
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "resource_profile_sha256": resource_profile_sha256,
            "resource_profile": (
                None if resource_profile is None else dict(resource_profile)
            ),
            "retained_checkpoints": ["best_model_val.pt"],
            "optimizer_state_retained": False,
            "training_curves_sha256": curves["content_hash"],
            "diagnostics_sha256": diagnostics["content_hash"],
            "val_stop_metrics_sha256": prediction["metrics"]["content_hash"],
            "val_stop_prediction_file_sha256": prediction_publication[
                "file_sha256"
            ],
            "publication_status": {
                "training_curves": curves_publication["status"],
                "diagnostics": diagnostic_publication["status"],
                "metrics": metrics_publication["status"],
                "predictions": prediction_publication["status"],
            },
        }
    )
    write_immutable_json(registration_path, registration)
    if last_path.exists():
        last_path.unlink()
    for candidate in candidate_root.glob("epoch_*.pt"):
        candidate.unlink()
    candidate_root.rmdir()
    return registration


__all__ = [
    "ATTACHMENT_PRETRAINING_CONTRACT",
    "EXPERT_CHECKPOINT_CONTRACT",
    "EXPERT_DIAGNOSTICS_CONTRACT",
    "EXPERT_LOSS_CANDIDATES",
    "EXPERT_LOSS_REGISTRY_CONTRACT",
    "EXPERT_REGISTRATION_CONTRACT",
    "EXPERT_TRAINING_CONTRACT",
    "INITIALIZATION_MODES",
    "DeterministicExpertSampler",
    "OfflineExpertDataset",
    "OfflineExpertTrainingConfig",
    "REGISTERED_LEARNING_RATES",
    "REGISTERED_PARTICLE_DROPOUTS",
    "TEACHER_LOGITS_MANIFEST_CONTRACT",
    "apply_attachment_trainability",
    "build_attachment_pretraining_record",
    "build_expert_loss_registry",
    "build_teacher_logits_manifest",
    "collect_expert_diagnostics",
    "copy_obase_particle_backbone",
    "collate_offline_expert_batch",
    "make_offline_expert_loader",
    "offline_expert_objective",
    "preferred_expert_epoch",
    "train_offline_expert",
    "validate_attachment_pretraining_record",
    "validate_expert_loss_registry",
    "validate_teacher_logits_manifest",
]
