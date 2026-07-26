"""Frozen-consumer particle-view distillation and its Step-7 controls."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .contracts import (
    canonical_sha256,
    require_sha256,
    sha256_file,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from .losses import (
    heteroscedastic_huber_loss,
    uncertainty_calibration_metrics,
)
from .oracle_discovery import classification_kd_loss, recovery_from_gains
from .predictor import (
    HierarchicalParticleViewPredictor,
    ParticleViewPredictorOutput,
)
from .recovery_probe import (
    view_cosine_loss,
    view_huber_loss,
    view_relational_loss,
)


PARTICLE_VIEW_DISTILLATION_OBJECTIVE_CONTRACT = (
    "particle_view_distillation_objective_v1"
)
PARTICLE_VIEW_DISTILLATION_CONFIG_CONTRACT = (
    "particle_view_distillation_config_v1"
)
PARTICLE_VIEW_DISTILLATION_CHECKPOINT_CONTRACT = (
    "particle_view_distillation_checkpoint_v1"
)
PARTICLE_VIEW_DISTILLATION_REGISTRATION_CONTRACT = (
    "particle_view_distillation_registration_v1"
)
PARTICLE_VIEW_TARGET_LOGIT_CACHE_CONTRACT = (
    "particle_view_target_logit_cache_v1"
)
PARTICLE_VIEW_GENERALIZATION_REPORT_CONTRACT = (
    "particle_view_generalization_report_v1"
)
PARTICLE_VIEW_DISTILLATION_CAMPAIGN_CONTRACT = (
    "particle_view_distillation_campaign_v1"
)
PARTICLE_VIEW_DISTILLATION_RANKING_CONTRACT = (
    "particle_view_distillation_ranking_v1"
)
PARTICLE_VIEW_DEPLOYMENT_AUDIT_CONTRACT = (
    "particle_view_teacher_independent_audit_v1"
)

PARTICLE_VIEW_LOSS_IDS = (
    "L_VIEW",
    "L_VIEW_COS",
    "L_VIEW_REL",
    "L_VIEW_ALL",
    "L_KD",
    "L_KD_VIEW",
    "L_KD_VIEW_REL",
    "L_KD_CE",
    "L_CE",
    "L_CE_VIEW",
    "L_PRIMARY",
    "L_PRIMARY_NO_CE",
    "L_PRIMARY_CE05",
    "L_PRIMARY_CE15",
    "L_PRIMARY_CE35",
    "L_PRIMARY_NO_TRUST",
    "L_UNCERTAINTY",
)

PRIVILEGED_CLAIM_INELIGIBLE_LOSSES = frozenset({"L_CE"})

DISTILLATION_LINEAGE_FIELDS = (
    "source_manifest_sha256",
    "train_identity_sha256",
    "model_val_stop_split_sha256",
    "model_val_select_split_sha256",
    "hlt_preprocessing_sha256",
    "class_order_sha256",
    "target_selection_sha256",
    "coordinate_binding_sha256",
    "pview0_registration_sha256",
    "initial_predictor_state_sha256",
    "consumer_registration_sha256",
    "consumer_checkpoint_sha256",
    "train_target_logit_cache_sha256",
    "model_val_stop_target_logit_cache_sha256",
    "model_val_select_target_logit_cache_sha256",
)

TARGET_LOGIT_CACHE_LINEAGE_FIELDS = (
    "split_identity_sha256",
    "hlt_preprocessing_sha256",
    "class_order_sha256",
    "coordinate_binding_sha256",
    "selected_view_cache_sha256",
    "consumer_registration_sha256",
    "consumer_checkpoint_sha256",
)

JOINT_LINEAGE_FIELDS = (
    "source_manifest_sha256",
    "train_identity_sha256",
    "model_val_stop_split_sha256",
    "model_val_select_split_sha256",
    "hlt_preprocessing_sha256",
    "class_order_sha256",
    "coordinate_binding_sha256",
    "parent_distillation_registration_sha256",
    "initial_predictor_state_sha256",
    "initial_consumer_state_sha256",
    "schedule_contract_sha256",
    "oracle_consumer_checkpoint_sha256",
    "train_target_logit_cache_sha256",
    "model_val_stop_target_logit_cache_sha256",
    "model_val_select_target_logit_cache_sha256",
)

_FORBIDDEN_DEPLOYMENT_FRAGMENTS = (
    "offline",
    "teacher",
    "oracle",
    "true_view",
    "target_logit",
    "gview",
    "label",
    "view_cache",
)


def module_state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        values = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(values.dtype).encode("utf-8"))
        digest.update(str(tuple(values.shape)).encode("utf-8"))
        digest.update(values.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _validate_hash_inventory(
    values: Mapping[str, str],
    fields: Sequence[str],
    *,
    name: str,
) -> dict[str, str]:
    if set(values) != set(fields):
        raise ValueError(f"{name} hash inventory mismatch")
    result = dict(values)
    for field, value in result.items():
        require_sha256(field, value)
    return result


@dataclass(frozen=True)
class DistillationObjective:
    loss_id: str
    kd: float = 0.0
    huber: float = 0.0
    cosine: float = 0.0
    relational: float = 0.0
    ce: float = 0.0
    trust: float = 0.0
    uncertainty: float = 0.0
    temperature: float = 2.0
    privileged_claim_eligible: bool = True
    pre_stage_g_deployable_eligible: bool = True
    contract: str = PARTICLE_VIEW_DISTILLATION_OBJECTIVE_CONTRACT

    def __post_init__(self) -> None:
        if self.loss_id not in PARTICLE_VIEW_LOSS_IDS:
            raise ValueError("unknown particle-view loss ID")
        for field in (
            "kd",
            "huber",
            "cosine",
            "relational",
            "ce",
            "trust",
            "uncertainty",
        ):
            value = getattr(self, field)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid objective weight {field}")
        if self.temperature != 2.0:
            raise ValueError("frozen-consumer KD temperature must be 2.0")
        if not any(
            getattr(self, field) > 0
            for field in (
                "kd",
                "huber",
                "cosine",
                "relational",
                "ce",
                "uncertainty",
            )
        ):
            raise ValueError("distillation objective has no learning signal")
        if self.loss_id == "L_CE" and self.privileged_claim_eligible:
            raise ValueError("CE-only is privileged-claim ineligible")

    @property
    def uses_privileged_view(self) -> bool:
        return any(
            value > 0
            for value in (
                self.kd,
                self.huber,
                self.cosine,
                self.relational,
                self.uncertainty,
            )
        )

    @property
    def uses_representation_target(self) -> bool:
        return any(
            value > 0
            for value in (
                self.huber,
                self.cosine,
                self.relational,
                self.uncertainty,
            )
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            **self.__dict__,
            "architecture_balance": "predictor_registered_weight_always_on",
            "same_consumer_target_live": True,
            "target_logits_detached": True,
        }

    @property
    def content_hash(self) -> str:
        return canonical_sha256(self.to_payload())


def build_distillation_loss_screen() -> dict[str, DistillationObjective]:
    """Return the complete locked Section-15.4 loss screen."""

    rows = {
        "L_VIEW": dict(huber=1.0),
        "L_VIEW_COS": dict(huber=1.0, cosine=0.25),
        "L_VIEW_REL": dict(huber=1.0, relational=0.15),
        "L_VIEW_ALL": dict(
            huber=1.0, cosine=0.25, relational=0.15, uncertainty=0.05
        ),
        "L_KD": dict(kd=1.0),
        "L_KD_VIEW": dict(kd=1.0, huber=0.35),
        "L_KD_VIEW_REL": dict(kd=1.0, huber=0.35, relational=0.10),
        "L_KD_CE": dict(kd=1.0, ce=0.15),
        "L_CE": dict(
            ce=1.0,
            privileged_claim_eligible=False,
        ),
        "L_CE_VIEW": dict(ce=1.0, huber=0.35),
        "L_PRIMARY": dict(
            kd=1.0,
            huber=0.35,
            cosine=0.10,
            relational=0.10,
            ce=0.15,
            trust=0.01,
        ),
        "L_PRIMARY_NO_CE": dict(
            kd=1.0,
            huber=0.35,
            cosine=0.10,
            relational=0.10,
            trust=0.01,
        ),
        "L_PRIMARY_CE05": dict(
            kd=1.0,
            huber=0.35,
            cosine=0.10,
            relational=0.10,
            ce=0.05,
            trust=0.01,
        ),
        "L_PRIMARY_CE15": dict(
            kd=1.0,
            huber=0.35,
            cosine=0.10,
            relational=0.10,
            ce=0.15,
            trust=0.01,
        ),
        "L_PRIMARY_CE35": dict(
            kd=1.0,
            huber=0.35,
            cosine=0.10,
            relational=0.10,
            ce=0.35,
            trust=0.01,
        ),
        "L_PRIMARY_NO_TRUST": dict(
            kd=1.0,
            huber=0.35,
            cosine=0.10,
            relational=0.10,
            ce=0.15,
        ),
        "L_UNCERTAINTY": dict(
            kd=1.0,
            huber=0.35,
            cosine=0.10,
            relational=0.10,
            ce=0.15,
            trust=0.01,
            uncertainty=0.05,
        ),
    }
    result = {
        loss_id: DistillationObjective(loss_id=loss_id, **weights)
        for loss_id, weights in rows.items()
    }
    if tuple(result) != PARTICLE_VIEW_LOSS_IDS:
        raise RuntimeError("particle-view loss-screen inventory changed")
    return result


@dataclass(frozen=True)
class DistillationTrainConfig:
    maximum_epochs: int = 40
    early_stop_patience: int = 8
    learning_rate: float = 3.0e-4
    minimum_learning_rate: float = 3.0e-6
    warmup_updates: int = 2_000
    weight_decay: float = 1.0e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    batch_size: int = 128
    gradient_clip: float = 1.0
    accuracy_tolerance: float = 1.0e-4
    seed: int = 101
    amp: bool = True
    contract: str = PARTICLE_VIEW_DISTILLATION_CONFIG_CONTRACT

    def __post_init__(self) -> None:
        if (
            self.maximum_epochs != 40
            or self.early_stop_patience != 8
            or self.learning_rate != 3.0e-4
            or self.minimum_learning_rate != 3.0e-6
            or self.warmup_updates != 2_000
            or self.weight_decay != 1.0e-4
            or (self.adam_beta1, self.adam_beta2) != (0.9, 0.999)
            or self.batch_size != 128
            or self.gradient_clip != 1.0
            or self.accuracy_tolerance != 1.0e-4
            or self.seed not in {101, 202, 303}
        ):
            raise ValueError("Stage-E distillation recipe changed")

    def to_payload(self) -> dict[str, Any]:
        return {
            **self.__dict__,
            "optimizer": "AdamW",
            "schedule": "2000_update_linear_warmup_then_cosine",
            "checkpoint_split": "model_val_stop",
            "checkpoint_order": [
                "accuracy_within_1e-4_of_maximum",
                "lowest_cross_entropy",
                "finite_recovery_then_higher_recovery",
                "earliest_epoch",
            ],
            "model_val_select_evaluations": 1,
        }

    @property
    def content_hash(self) -> str:
        return canonical_sha256(self.to_payload())


@dataclass(frozen=True)
class JointFineTuneConfig:
    objective_id: str = "JOINT_PRIVILEGED"
    maximum_epochs: int = 8
    early_stop_patience: int = 3
    learning_rate: float = 3.0e-5
    weight_decay: float = 1.0e-4
    batch_size: int = 128
    gradient_clip: float = 1.0
    accuracy_tolerance: float = 1.0e-4
    seed: int = 101
    amp: bool = True

    def __post_init__(self) -> None:
        if self.objective_id not in {"JOINT_PRIVILEGED", "JOINT_CE_ONLY"}:
            raise ValueError("unknown joint objective")
        if (
            self.maximum_epochs != 8
            or self.early_stop_patience != 3
            or self.learning_rate != 3.0e-5
            or self.weight_decay != 1.0e-4
            or self.batch_size != 128
            or self.gradient_clip != 1.0
            or self.accuracy_tolerance != 1.0e-4
            or self.seed not in {101, 202, 303}
        ):
            raise ValueError("Stage-F joint recipe changed")

    @property
    def privileged(self) -> bool:
        return self.objective_id == "JOINT_PRIVILEGED"

    def to_payload(self) -> dict[str, Any]:
        return {
            **self.__dict__,
            "optimizer": "AdamW",
            "objective": (
                {"kd": 1.0, "huber": 0.25, "ce": 0.10, "trust": 0.01}
                if self.privileged
                else {"ce": 1.0}
            ),
            "oracle_target_network_frozen": True,
            "recovery_status": "undefined",
            "schedule_matched_control": (
                "JOINT_CE_ONLY" if self.privileged else "JOINT_PRIVILEGED"
            ),
        }

    @property
    def content_hash(self) -> str:
        return canonical_sha256(self.to_payload())


def joint_schedule_contract_sha256(config: JointFineTuneConfig) -> str:
    return canonical_sha256(
        {
            "maximum_epochs": config.maximum_epochs,
            "early_stop_patience": config.early_stop_patience,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "batch_size": config.batch_size,
            "gradient_clip": config.gradient_clip,
            "accuracy_tolerance": config.accuracy_tolerance,
            "seed": config.seed,
            "optimizer": "AdamW",
            "trainable_components": ["predictor", "consumer"],
        }
    )


def stage_e_learning_rate(
    update: int,
    total_updates: int,
    config: DistillationTrainConfig,
) -> float:
    if update <= 0 or total_updates <= 0:
        raise ValueError("learning-rate update counts must be positive")
    if update <= config.warmup_updates:
        return config.learning_rate * update / config.warmup_updates
    if total_updates <= config.warmup_updates:
        return config.learning_rate
    progress = min(
        max(
            (update - config.warmup_updates)
            / (total_updates - config.warmup_updates),
            0.0,
        ),
        1.0,
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.minimum_learning_rate + (
        config.learning_rate - config.minimum_learning_rate
    ) * cosine


def _extract_logits(output: Any) -> torch.Tensor:
    logits = output.logits if hasattr(output, "logits") else output
    if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
        raise ValueError("consumer must return rank-two logits")
    return logits


def _extract_trust_loss(output: Any, reference: torch.Tensor) -> torch.Tensor:
    value = getattr(output, "trust_loss", None)
    if value is None:
        return reference.new_zeros(())
    if not isinstance(value, torch.Tensor) or value.numel() != 1:
        raise ValueError("consumer trust loss must be scalar")
    return value.reshape(())


def _consumer_forward(
    consumer: nn.Module,
    batch: Mapping[str, torch.Tensor],
    view: torch.Tensor,
) -> Any:
    return consumer(
        batch["points"],
        batch["features"],
        batch["lorentz_vectors"],
        batch["mask"],
        view,
        augment_clean_view=False,
    )


def freeze_consumer_for_distillation(consumer: nn.Module) -> str:
    consumer.eval()
    for parameter in consumer.parameters():
        parameter.requires_grad_(False)
    return module_state_sha256(consumer)


@dataclass
class SameConsumerForward:
    target_logits: torch.Tensor
    live_logits: torch.Tensor
    live_consumer_output: Any
    predictor_output: ParticleViewPredictorOutput
    consumer_state_sha256: str


def exact_same_consumer_forward(
    *,
    predictor: nn.Module,
    frozen_consumer: nn.Module,
    batch: Mapping[str, torch.Tensor],
    true_view: torch.Tensor,
    cached_target_logits: torch.Tensor | None = None,
    cache_tolerance: float = 1.0e-6,
) -> SameConsumerForward:
    """Use one exact frozen consumer instance on both sides of the KD pair."""

    state_before = freeze_consumer_for_distillation(frozen_consumer)
    with torch.no_grad():
        target_logits = _extract_logits(
            _consumer_forward(frozen_consumer, batch, true_view)
        ).detach()
    if cached_target_logits is not None:
        cached_target_logits = cached_target_logits.to(
            device=target_logits.device, dtype=target_logits.dtype
        )
        if cached_target_logits.shape != target_logits.shape:
            raise ValueError("cached target-logit shape mismatch")
        maximum = float(
            (cached_target_logits - target_logits).abs().max().item()
        )
        if maximum > cache_tolerance:
            raise ValueError(
                "cached target logits differ from exact frozen consumer: "
                f"max_abs={maximum:.9g}"
            )
        target_logits = cached_target_logits.detach()
    predictor_output = predictor(
        batch["features"], batch["lorentz_vectors"], batch["mask"]
    )
    live_output = _consumer_forward(
        frozen_consumer, batch, predictor_output.mean
    )
    live_logits = _extract_logits(live_output)
    state_after = module_state_sha256(frozen_consumer)
    if state_after != state_before:
        raise RuntimeError("frozen consumer changed during paired forward")
    return SameConsumerForward(
        target_logits=target_logits,
        live_logits=live_logits,
        live_consumer_output=live_output,
        predictor_output=predictor_output,
        consumer_state_sha256=state_before,
    )


def distillation_losses(
    *,
    predictor_output: ParticleViewPredictorOutput,
    live_consumer_output: Any,
    target_logits: torch.Tensor | None,
    labels: torch.Tensor | None,
    true_view: torch.Tensor | None,
    mask: torch.Tensor,
    objective: DistillationObjective,
) -> dict[str, torch.Tensor]:
    live_logits = _extract_logits(live_consumer_output)
    zero = live_logits.new_zeros(())
    if mask.ndim == 3:
        mask = mask[:, 0]
    if mask.dtype != torch.bool or mask.shape != predictor_output.mean.shape[:2]:
        raise ValueError("distillation mask shape mismatch")
    invalid = ~mask[:, :, None]
    for name, values in (
        ("predictor mean", predictor_output.mean),
        ("predictor log variance", predictor_output.log_variance),
    ):
        if not torch.isfinite(values).all():
            raise FloatingPointError(f"{name} is non-finite")
        expanded = invalid.expand_as(values)
        if expanded.any() and values[expanded].abs().max().item() != 0:
            raise ValueError(f"{name} leaked onto invalid particles")
    if true_view is not None:
        if true_view.shape != predictor_output.mean.shape:
            raise ValueError("true-view shape differs from predictor output")
        if not torch.isfinite(true_view[mask]).all():
            raise FloatingPointError("true view is non-finite")
        expanded = invalid.expand_as(true_view)
        if expanded.any() and true_view[expanded].abs().max().item() != 0:
            raise ValueError("true view leaked onto invalid particles")
    if not torch.isfinite(live_logits).all():
        raise FloatingPointError("live consumer logits are non-finite")
    if target_logits is not None and not torch.isfinite(target_logits).all():
        raise FloatingPointError("target logits are non-finite")
    if objective.kd and target_logits is None:
        raise ValueError("KD objective requires frozen-consumer target logits")
    if objective.ce and labels is None:
        raise ValueError("CE objective requires labels")
    if objective.uses_representation_target and true_view is None:
        raise ValueError("representation objective requires true view")
    kd = (
        classification_kd_loss(
            live_logits, target_logits, temperature=objective.temperature
        )
        if objective.kd
        else zero
    )
    ce = F.cross_entropy(live_logits, labels) if objective.ce else zero
    huber = (
        view_huber_loss(predictor_output.mean, true_view, mask, beta=0.1)
        if objective.huber
        else zero
    )
    cosine = (
        view_cosine_loss(predictor_output.mean, true_view, mask)
        if objective.cosine
        else zero
    )
    relational = (
        view_relational_loss(predictor_output.mean, true_view, mask)
        if objective.relational
        else zero
    )
    uncertainty = (
        heteroscedastic_huber_loss(
            predictor_output.mean,
            true_view,
            predictor_output.log_variance,
            mask,
        )
        if objective.uncertainty
        else zero
    )
    trust = (
        _extract_trust_loss(live_consumer_output, live_logits)
        if objective.trust
        else zero
    )
    balance = predictor_output.balance_loss
    total = (
        objective.kd * kd
        + objective.huber * huber
        + objective.cosine * cosine
        + objective.relational * relational
        + objective.ce * ce
        + objective.trust * trust
        + objective.uncertainty * uncertainty
        + balance
    )
    if not torch.isfinite(total):
        raise FloatingPointError("distillation loss is non-finite")
    return {
        "total": total,
        "kd": kd,
        "ce": ce,
        "huber": huber,
        "cosine": cosine,
        "relational": relational,
        "uncertainty": uncertainty,
        "trust": trust,
        "balance": balance,
    }


def _target_logit_batch(
    raw: Mapping[str, Any], device: torch.device
) -> dict[str, torch.Tensor]:
    required = {
        "event_ids",
        "points",
        "features",
        "lorentz_vectors",
        "mask",
        "true_view",
    }
    if not required.issubset(raw):
        raise ValueError("target-logit batch field inventory is incomplete")
    forbidden = {"offline_tokens", "offline_logits", "teacher_logits"}
    if forbidden.intersection(raw):
        raise ValueError("target-logit cache exposed offline descendants")
    return {
        name: raw[name].to(device=device, non_blocking=True)
        for name in required
    }


def publish_target_logit_cache(
    *,
    frozen_consumer: nn.Module,
    loader,
    output_dir: str | Path,
    split: str,
    lineage: Mapping[str, str],
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Publish float32 ``z*`` bound to one consumer state and true-view cache."""

    lineage = _validate_hash_inventory(
        lineage, TARGET_LOGIT_CACHE_LINEAGE_FIELDS, name="target-logit cache"
    )
    if split not in {"train", "model_val_stop", "model_val_select"}:
        raise ValueError("target-logit cache split is not authorized")
    state_hash = freeze_consumer_for_distillation(frozen_consumer)
    device = torch.device(device)
    frozen_consumer.to(device)
    ids, logits = [], []
    with torch.no_grad():
        for raw in loader:
            batch = _target_logit_batch(raw, device)
            valid = (
                batch["mask"][:, 0]
                if batch["mask"].ndim == 3
                else batch["mask"]
            )
            if (
                valid.dtype != torch.bool
                or batch["true_view"].shape[:2] != valid.shape
                or not torch.isfinite(batch["true_view"][valid]).all()
                or (
                    (~valid).any()
                    and batch["true_view"][~valid].abs().max().item() != 0
                )
            ):
                raise ValueError(
                    "target-logit true view violates mask/finite contract"
                )
            result = _extract_logits(
                _consumer_forward(
                    frozen_consumer, batch, batch["true_view"]
                )
            )
            if not torch.isfinite(result).all():
                raise FloatingPointError("target logits are non-finite")
            ids.append(batch["event_ids"].detach().cpu().long())
            logits.append(result.detach().cpu().float())
    if not ids:
        raise ValueError("target-logit loader is empty")
    event_ids = torch.cat(ids).numpy().astype(np.int64, copy=False)
    values = torch.cat(logits).numpy().astype("<f4", copy=False)
    if len(np.unique(event_ids)) != len(event_ids):
        raise ValueError("target-logit event identities are not unique")
    order = np.argsort(event_ids, kind="stable")
    event_ids = event_ids[order]
    values = values[order]
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    data_path = root / f"{split}_frozen_consumer_logits.npz"
    manifest_path = root / f"{split}_frozen_consumer_logits.json"
    if data_path.exists() or manifest_path.exists():
        raise FileExistsError("target-logit cache output already exists")
    np.savez_compressed(data_path, event_ids=event_ids, logits=values)
    manifest = with_content_hash(
        {
            "contract": PARTICLE_VIEW_TARGET_LOGIT_CACHE_CONTRACT,
            "split": split,
            "lineage": lineage,
            "consumer_state_sha256": state_hash,
            "data_file": data_path.name,
            "data_sha256": sha256_file(data_path),
            "events": int(len(event_ids)),
            "classes": int(values.shape[1]),
            "dtype": "float32",
            "event_ids_sha256": hashlib.sha256(
                event_ids.tobytes(order="C")
            ).hexdigest(),
            "target_definition": "exact_frozen_consumer_hlt_true_view",
        }
    )
    write_immutable_json(manifest_path, manifest)
    return manifest


class FrozenTargetLogitCache:
    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path)
        self.manifest = json.loads(
            self.manifest_path.read_text(encoding="utf-8")
        )
        validate_content_hash(
            self.manifest,
            expected_contract=PARTICLE_VIEW_TARGET_LOGIT_CACHE_CONTRACT,
        )
        require_sha256(
            "consumer_state_sha256",
            self.manifest.get("consumer_state_sha256"),
        )
        _validate_hash_inventory(
            self.manifest["lineage"],
            TARGET_LOGIT_CACHE_LINEAGE_FIELDS,
            name="target-logit cache",
        )
        data_path = self.manifest_path.parent / self.manifest["data_file"]
        if sha256_file(data_path) != self.manifest["data_sha256"]:
            raise ValueError("target-logit cache data hash mismatch")
        with np.load(data_path, allow_pickle=False) as data:
            if set(data.files) != {"event_ids", "logits"}:
                raise ValueError("target-logit cache array inventory mismatch")
            self.event_ids = np.asarray(data["event_ids"], dtype=np.int64)
            self.logits = np.asarray(data["logits"], dtype=np.float32)
        if (
            self.event_ids.ndim != 1
            or self.logits.ndim != 2
            or len(self.event_ids) != len(self.logits)
            or len(self.event_ids) != self.manifest.get("events")
            or self.logits.shape[1] != self.manifest.get("classes")
            or self.manifest.get("dtype") != "float32"
            or np.any(self.event_ids[1:] <= self.event_ids[:-1])
            or not np.isfinite(self.logits).all()
        ):
            raise ValueError("target-logit cache arrays are invalid")
        if hashlib.sha256(
            self.event_ids.tobytes(order="C")
        ).hexdigest() != self.manifest["event_ids_sha256"]:
            raise ValueError("target-logit event identity hash mismatch")

    @property
    def content_hash(self) -> str:
        return self.manifest["content_hash"]

    def validate_consumer(self, consumer: nn.Module) -> None:
        if module_state_sha256(consumer) != self.manifest[
            "consumer_state_sha256"
        ]:
            raise ValueError("target-logit cache belongs to another consumer")

    def lookup(
        self, event_ids: torch.Tensor, *, device: torch.device
    ) -> torch.Tensor:
        requested = event_ids.detach().cpu().numpy().astype(np.int64, copy=False)
        indices = np.searchsorted(self.event_ids, requested)
        if (
            np.any(indices >= len(self.event_ids))
            or np.any(self.event_ids[np.minimum(indices, len(self.event_ids) - 1)] != requested)
        ):
            raise KeyError("event identity is absent from target-logit cache")
        return torch.from_numpy(self.logits[indices]).to(device=device)


def _move_distillation_batch(
    raw: Mapping[str, Any],
    device: torch.device,
    *,
    expose_privileged: bool,
) -> dict[str, torch.Tensor]:
    required = {
        "event_ids",
        "points",
        "features",
        "lorentz_vectors",
        "mask",
        "labels",
    }
    if expose_privileged:
        required.add("true_view")
    if not required.issubset(raw):
        raise ValueError("distillation batch field inventory is incomplete")
    forbidden = {"offline_tokens", "offline_logits", "teacher_logits", "oracle_logits"}
    if forbidden.intersection(raw):
        raise ValueError("distillation batch exposed offline descendants")
    if not expose_privileged and (
        "true_view" in raw or "target_logits" in raw
    ):
        raise ValueError("CE-only schedule control exposed privileged targets")
    return {
        name: raw[name].to(device=device, non_blocking=True)
        for name in required
    }


def _metrics_from_logits(
    logits: torch.Tensor, labels: torch.Tensor
) -> tuple[int, float]:
    return (
        int((logits.argmax(dim=-1) == labels).sum().item()),
        float(F.cross_entropy(logits, labels, reduction="sum").item()),
    )


def evaluate_distilled_bundle(
    predictor: nn.Module,
    live_consumer: nn.Module,
    loader,
    *,
    split: str,
    target_cache: FrozenTargetLogitCache | None,
    target_consumer: nn.Module | None = None,
    device: str | torch.device = "cpu",
    joint_consumer_changed: bool = False,
    permit_true_view_metrics: bool = True,
) -> dict[str, Any]:
    if split not in {"train", "model_val_stop", "model_val_select"}:
        raise ValueError("distillation evaluation split is unauthorized")
    device = torch.device(device)
    predictor.eval()
    live_consumer.eval()
    target_consumer = live_consumer if target_consumer is None else target_consumer
    if target_cache is not None:
        target_cache.validate_consumer(target_consumer)
    counts = {"deployable": 0, "zero": 0, "oracle_target": 0}
    cross_entropy = {"deployable": 0.0, "zero": 0.0, "oracle_target": 0.0}
    totals = {
        name: 0.0
        for name in ("kd", "huber", "cosine", "relational", "trust")
    }
    uncertainty_rows: list[tuple[torch.Tensor, ...]] = []
    gate_sum = gate_count = gate_below = gate_above = 0.0
    events = batches = 0
    with torch.no_grad():
        for raw in loader:
            batch = _move_distillation_batch(
                raw,
                device,
                expose_privileged=bool(
                    permit_true_view_metrics and "true_view" in raw
                ),
            )
            output = predictor(
                batch["features"], batch["lorentz_vectors"], batch["mask"]
            )
            valid = batch["mask"][:, 0] if batch["mask"].ndim == 3 else batch["mask"]
            if (
                not torch.isfinite(output.mean).all()
                or (
                    (~valid).any()
                    and output.mean[~valid].abs().max().item() != 0
                )
            ):
                raise ValueError(
                    "deployable predictor output violates mask/finite contract"
                )
            live_output = _consumer_forward(
                live_consumer, batch, output.mean
            )
            live_logits = _extract_logits(live_output)
            labels = batch["labels"]
            batch_events = labels.numel()
            correct, ce = _metrics_from_logits(live_logits, labels)
            counts["deployable"] += correct
            cross_entropy["deployable"] += ce
            zero_view = torch.zeros_like(output.mean)
            zero_logits = _extract_logits(
                _consumer_forward(live_consumer, batch, zero_view)
            )
            correct, ce = _metrics_from_logits(zero_logits, labels)
            counts["zero"] += correct
            cross_entropy["zero"] += ce
            if target_cache is not None:
                target_logits = target_cache.lookup(
                    batch["event_ids"], device=device
                )
                correct, ce = _metrics_from_logits(target_logits, labels)
                counts["oracle_target"] += correct
                cross_entropy["oracle_target"] += ce
                totals["kd"] += float(
                    classification_kd_loss(live_logits, target_logits).item()
                ) * batch_events
            if "true_view" in batch:
                totals["huber"] += float(
                    view_huber_loss(output.mean, batch["true_view"], valid).item()
                ) * batch_events
                totals["cosine"] += float(
                    view_cosine_loss(output.mean, batch["true_view"], valid).item()
                ) * batch_events
                totals["relational"] += float(
                    view_relational_loss(output.mean, batch["true_view"], valid).item()
                ) * batch_events
                uncertainty_rows.append(
                    (
                        output.mean.detach().cpu(),
                        batch["true_view"].detach().cpu(),
                        output.log_variance.detach().cpu(),
                        valid.detach().cpu(),
                    )
                )
            totals["trust"] += float(
                _extract_trust_loss(live_output, live_logits).item()
            ) * batch_events
            gate = getattr(live_output, "gate", None)
            if isinstance(gate, torch.Tensor):
                selected_gate = gate[..., 0][valid].detach()
                gate_sum += float(selected_gate.sum().item())
                gate_count += float(selected_gate.numel())
                gate_below += float((selected_gate < 0.01).sum().item())
                gate_above += float((selected_gate > 0.99).sum().item())
            events += batch_events
            batches += 1
    if not events:
        raise ValueError("distillation evaluation loader is empty")
    deployable_accuracy = counts["deployable"] / events
    zero_accuracy = counts["zero"] / events
    if target_cache is not None and not joint_consumer_changed:
        oracle_accuracy = counts["oracle_target"] / events
        oracle_gain = oracle_accuracy - zero_accuracy
        predicted_gain = deployable_accuracy - zero_accuracy
        recovery_status, recovery = recovery_from_gains(
            oracle_gain, predicted_gain
        )
    else:
        oracle_accuracy = (
            counts["oracle_target"] / events if target_cache is not None else None
        )
        oracle_gain = predicted_gain = recovery = None
        recovery_status = "undefined"
    uncertainty = None
    if uncertainty_rows:
        uncertainty = uncertainty_calibration_metrics(
            torch.cat([row[0] for row in uncertainty_rows]),
            torch.cat([row[1] for row in uncertainty_rows]),
            torch.cat([row[2] for row in uncertainty_rows]),
            torch.cat([row[3] for row in uncertainty_rows]),
        )
    return {
        "split": split,
        "events": events,
        "batches": batches,
        "deployable_accuracy": deployable_accuracy,
        "deployable_cross_entropy": cross_entropy["deployable"] / events,
        "zero_view_accuracy": zero_accuracy,
        "zero_view_cross_entropy": cross_entropy["zero"] / events,
        "oracle_target_accuracy": oracle_accuracy,
        "oracle_target_cross_entropy": (
            cross_entropy["oracle_target"] / events
            if target_cache is not None
            else None
        ),
        "oracle_gain": oracle_gain,
        "predicted_gain": predicted_gain,
        "recovery_status": recovery_status,
        "recovery_fraction": recovery,
        "kd": totals["kd"] / events,
        "huber": totals["huber"] / events,
        "cosine": totals["cosine"] / events,
        "relational": totals["relational"] / events,
        "trust": totals["trust"] / events,
        "trust_gate_mean": gate_sum / gate_count if gate_count else None,
        "trust_gate_fraction_below_001": (
            gate_below / gate_count if gate_count else None
        ),
        "trust_gate_fraction_above_099": (
            gate_above / gate_count if gate_count else None
        ),
        "uncertainty_calibration": uncertainty,
        "joint_consumer_changed": joint_consumer_changed,
        "model_val_select_loaded": split == "model_val_select",
        "stack_val_loaded": False,
        "final_test_loaded": False,
    }


def select_distillation_checkpoint(
    rows: Sequence[Mapping[str, Any]],
    *,
    accuracy_tolerance: float = 1.0e-4,
) -> Mapping[str, Any]:
    if not rows:
        raise ValueError("checkpoint selector received no rows")
    for row in rows:
        metrics = row.get("model_val_stop")
        if not isinstance(metrics, Mapping):
            raise ValueError("checkpoint row lacks model_val_stop metrics")
        for field in ("deployable_accuracy", "deployable_cross_entropy"):
            if not math.isfinite(float(metrics[field])):
                raise ValueError("checkpoint metric is non-finite")
    maximum = max(
        float(row["model_val_stop"]["deployable_accuracy"]) for row in rows
    )
    pool = [
        row
        for row in rows
        if float(row["model_val_stop"]["deployable_accuracy"])
        >= maximum - accuracy_tolerance
    ]

    def key(row):
        metrics = row["model_val_stop"]
        recovery = metrics.get("recovery_fraction")
        finite = (
            metrics.get("recovery_status") == "finite"
            and recovery is not None
            and math.isfinite(float(recovery))
        )
        return (
            float(metrics["deployable_cross_entropy"]),
            0 if finite else 1,
            -float(recovery) if finite else 0.0,
            int(row["epoch"]),
        )

    return min(pool, key=key)


def build_generalization_report(
    *,
    train_metrics: Mapping[str, Any],
    model_val_stop_metrics: Mapping[str, Any],
    model_val_select_metrics: Mapping[str, Any],
    configuration_id: str,
    seed: int,
) -> dict[str, Any]:
    if (
        train_metrics.get("split") != "train"
        or model_val_stop_metrics.get("split") != "model_val_stop"
        or model_val_select_metrics.get("split") != "model_val_select"
    ):
        raise ValueError("generalization report split contract failed")
    gaps = {
        "huber_train_minus_model_val_stop": float(train_metrics["huber"])
        - float(model_val_stop_metrics["huber"]),
        "kd_train_minus_model_val_stop": float(train_metrics["kd"])
        - float(model_val_stop_metrics["kd"]),
        "cosine_train_minus_model_val_stop": float(train_metrics["cosine"])
        - float(model_val_stop_metrics["cosine"]),
        "relational_train_minus_model_val_stop": float(
            train_metrics["relational"]
        )
        - float(model_val_stop_metrics["relational"]),
        "deployable_accuracy_train_minus_model_val_stop": float(
            train_metrics["deployable_accuracy"]
        )
        - float(model_val_stop_metrics["deployable_accuracy"]),
    }
    warnings = []
    accuracy_gap = gaps[
        "deployable_accuracy_train_minus_model_val_stop"
    ]
    if accuracy_gap > 0.01:
        warnings.append(
            {
                "warning_code": "WARN_LARGE_TRAIN_VALIDATION_GAP",
                "severity": "scientific",
                "observed_value": accuracy_gap,
                "reference_value": 0.0,
                "declared_warning_threshold": 0.01,
                "interpretation": (
                    "Train deployable accuracy exceeds model_val_stop by "
                    "more than one percentage point."
                ),
            }
        )
    report = with_content_hash(
        {
            "contract": PARTICLE_VIEW_GENERALIZATION_REPORT_CONTRACT,
            "configuration_id": configuration_id,
            "seed": seed,
            "train": dict(train_metrics),
            "model_val_stop": dict(model_val_stop_metrics),
            "model_val_select": dict(model_val_select_metrics),
            "gaps": gaps,
            "quality_warnings": warnings,
            "warnings_are_non_gating": True,
        }
    )
    return report


def _ranking_key(row: Mapping[str, Any]):
    recovery = row.get("recovery_fraction")
    finite = (
        row.get("recovery_status") == "finite"
        and recovery is not None
        and math.isfinite(float(recovery))
    )
    return (
        -float(row["deployable_accuracy"]),
        float(row["deployable_cross_entropy"]),
        0 if finite else 1,
        -float(recovery) if finite else 0.0,
        int(row["deployed_parameters"]),
        str(row["run_id"]),
        str(row["configuration_id"]),
    )


def rank_model_val_select_configurations(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("configuration ranking is empty")
    for row in rows:
        if row.get("split") != "model_val_select":
            raise ValueError("configuration ranking must use model_val_select")
        if not math.isfinite(float(row["deployable_accuracy"])) or not math.isfinite(
            float(row["deployable_cross_entropy"])
        ):
            raise ValueError("configuration ranking metric is non-finite")
    ordered = sorted(rows, key=_ranking_key)
    return with_content_hash(
        {
            "contract": PARTICLE_VIEW_DISTILLATION_RANKING_CONTRACT,
            "selection_split": "model_val_select",
            "ordered_configuration_ids": [
                row["configuration_id"] for row in ordered
            ],
            "winner": dict(ordered[0]),
            "rows": [dict(row) for row in ordered],
            "minimum_quality_gate": None,
        }
    )


@dataclass(frozen=True)
class DistillationCampaignRow:
    row_id: str
    target_id: str
    loss_id: str
    architecture_id: str
    consumer_id: str
    mode: str = "frozen"
    seed: int = 101
    selectable: bool = True
    privileged_claim_eligible: bool = True

    def to_payload(self) -> dict[str, Any]:
        if self.loss_id not in PARTICLE_VIEW_LOSS_IDS:
            raise ValueError("campaign row has unknown loss")
        if self.mode not in {"frozen", "joint", "joint_ce_control"}:
            raise ValueError("campaign row mode is invalid")
        if self.seed not in {101, 202, 303}:
            raise ValueError("campaign row seed is invalid")
        return dict(self.__dict__)


def build_target_loss_interaction_campaign(
    *,
    target_ids: Sequence[str],
    canonical_target_id: str,
    alternate_target_id: str,
    canonical_architecture_id: str = "P_HIER_DECODER_REFINE",
    plain_architecture_id: str = "P_PART_BASIC",
    clean_consumer_id: str = "C_CLEAN",
    robust_consumer_id: str = "C_ROBUST_MIX",
) -> dict[str, Any]:
    """Enumerate full target x loss plus mandatory focused interactions."""

    if (
        len(set(target_ids)) != len(target_ids)
        or canonical_target_id not in target_ids
        or alternate_target_id not in target_ids
    ):
        raise ValueError("target interaction inventory is invalid")
    rows: dict[str, DistillationCampaignRow] = {}

    def add(row: DistillationCampaignRow) -> None:
        payload = row.to_payload()
        existing = rows.get(row.row_id)
        if existing is not None and existing.to_payload() != payload:
            raise ValueError("campaign row ID collision")
        rows[row.row_id] = row

    objectives = build_distillation_loss_screen()
    for target_id in target_ids:
        for loss_id, objective in objectives.items():
            row_id = (
                f"target={target_id}__arch={canonical_architecture_id}"
                f"__consumer={robust_consumer_id}__loss={loss_id}"
            )
            add(
                DistillationCampaignRow(
                    row_id=row_id,
                    target_id=target_id,
                    loss_id=loss_id,
                    architecture_id=canonical_architecture_id,
                    consumer_id=robust_consumer_id,
                    privileged_claim_eligible=objective.privileged_claim_eligible,
                )
            )
    for consumer_id in (clean_consumer_id, robust_consumer_id):
        for loss_id in (
            "L_PRIMARY",
            "L_PRIMARY_NO_CE",
            "L_CE",
            "L_UNCERTAINTY",
        ):
            for architecture_id in (
                plain_architecture_id,
                canonical_architecture_id,
            ):
                target_id = canonical_target_id
                row_id = (
                    f"target={target_id}__arch={architecture_id}"
                    f"__consumer={consumer_id}__loss={loss_id}"
                )
                add(
                    DistillationCampaignRow(
                        row_id=row_id,
                        target_id=target_id,
                        loss_id=loss_id,
                        architecture_id=architecture_id,
                        consumer_id=consumer_id,
                        privileged_claim_eligible=loss_id != "L_CE",
                    )
                )
    for consumer_id in ("C_EMBED", "C_EMBED_PAIR"):
        row_id = (
            f"target={canonical_target_id}__arch={canonical_architecture_id}"
            f"__consumer={consumer_id}__loss=L_PRIMARY"
        )
        add(
            DistillationCampaignRow(
                row_id=row_id,
                target_id=canonical_target_id,
                loss_id="L_PRIMARY",
                architecture_id=canonical_architecture_id,
                consumer_id=consumer_id,
            )
        )
    for target_id in (canonical_target_id, alternate_target_id):
        for mode, loss_id in (
            ("joint", "L_PRIMARY"),
            ("joint_ce_control", "L_CE"),
        ):
            row_id = (
                f"target={target_id}__arch={canonical_architecture_id}"
                f"__consumer={robust_consumer_id}__mode={mode}"
            )
            add(
                DistillationCampaignRow(
                    row_id=row_id,
                    target_id=target_id,
                    loss_id=loss_id,
                    architecture_id=canonical_architecture_id,
                    consumer_id=robust_consumer_id,
                    mode=mode,
                    privileged_claim_eligible=mode == "joint",
                )
            )
    payload_rows = [rows[key].to_payload() for key in sorted(rows)]
    return with_content_hash(
        {
            "contract": PARTICLE_VIEW_DISTILLATION_CAMPAIGN_CONTRACT,
            "target_ids": list(target_ids),
            "canonical_target_id": canonical_target_id,
            "alternate_target_id": alternate_target_id,
            "loss_ids": list(PARTICLE_VIEW_LOSS_IDS),
            "rows": payload_rows,
            "row_count": len(payload_rows),
            "performance_gates": False,
            "warnings_are_non_gating": True,
        }
    )


class DeployableParticleViewBundle(nn.Module):
    """Inference graph whose public boundary contains HLT tensors only."""

    def __init__(self, predictor: nn.Module, consumer: nn.Module) -> None:
        super().__init__()
        self.predictor = predictor
        self.consumer = consumer

    def forward(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        view = self.predictor(features, lorentz_vectors, mask).mean
        return _extract_logits(
            self.consumer(
                points,
                features,
                lorentz_vectors,
                mask,
                view,
                augment_clean_view=False,
            )
        )


def audit_teacher_independent_deployment(
    *,
    predictor: nn.Module,
    consumer: nn.Module,
    hlt_batch: Mapping[str, torch.Tensor],
    reference_logits: torch.Tensor | None = None,
    dependency_manifest: Mapping[str, Any] | None = None,
    tolerance: float = 1.0e-6,
) -> dict[str, Any]:
    required = {"points", "features", "lorentz_vectors", "mask"}
    if set(hlt_batch) != required:
        raise ValueError("deployment audit accepts exactly four HLT inputs")
    dependency_manifest = dict(dependency_manifest or {})
    flattened = json.dumps(dependency_manifest, sort_keys=True).lower()
    forbidden = [
        fragment
        for fragment in _FORBIDDEN_DEPLOYMENT_FRAGMENTS
        if fragment in flattened
    ]
    if forbidden:
        raise ValueError(
            "deployable dependency manifest contains forbidden inputs: "
            + ",".join(forbidden)
        )
    bundle = DeployableParticleViewBundle(predictor, consumer).eval()
    state_names = [name.lower() for name in bundle.state_dict()]
    state_forbidden = sorted(
        {
            fragment
            for name in state_names
            for fragment in _FORBIDDEN_DEPLOYMENT_FRAGMENTS
            if fragment in name
        }
    )
    if state_forbidden:
        raise ValueError(
            "deployable state contains forbidden descendants: "
            + ",".join(state_forbidden)
        )
    with torch.no_grad():
        logits = bundle(**hlt_batch)
    if not torch.isfinite(logits).all():
        raise FloatingPointError("deployable logits are non-finite")
    maximum = 0.0
    if reference_logits is not None:
        if logits.shape != reference_logits.shape:
            raise ValueError("deployment reference-logit shape mismatch")
        maximum = float(
            (logits - reference_logits.to(logits)).abs().max().item()
        )
        if maximum > tolerance:
            raise ValueError("deployable reload logits differ from reference")
    return with_content_hash(
        {
            "contract": PARTICLE_VIEW_DEPLOYMENT_AUDIT_CONTRACT,
            "allowed_inputs": sorted(required),
            "forbidden_dependency_fragments": list(
                _FORBIDDEN_DEPLOYMENT_FRAGMENTS
            ),
            "dependency_manifest": dependency_manifest,
            "state_tensor_count": len(bundle.state_dict()),
            "logit_shape": list(logits.shape),
            "finite_logits": True,
            "reference_maximum_absolute_difference": maximum,
            "tolerance": tolerance,
            "teacher_independent": True,
            "final_test_hlt_only_compatible": True,
        }
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _autocast(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def _grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:  # pragma: no cover
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _state_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _candidate_better(
    candidate: Mapping[str, Any],
    selected: Mapping[str, Any] | None,
    *,
    maximum_accuracy: float,
    tolerance: float,
) -> bool:
    if selected is None:
        return True
    selected_metrics = selected["model_val_stop"]
    candidate_metrics = candidate["model_val_stop"]
    if float(selected_metrics["deployable_accuracy"]) < maximum_accuracy - tolerance:
        return True
    if float(candidate_metrics["deployable_accuracy"]) < maximum_accuracy - tolerance:
        return False
    return select_distillation_checkpoint(
        [selected, candidate], accuracy_tolerance=tolerance
    ) is candidate


def _first_hlt_batch(loader, device: torch.device) -> dict[str, torch.Tensor]:
    raw = next(iter(loader))
    return {
        name: raw[name].to(device)
        for name in ("points", "features", "lorentz_vectors", "mask")
    }


def train_frozen_consumer_distillation(
    *,
    predictor: nn.Module,
    frozen_consumer: nn.Module,
    train_loader,
    model_val_stop_loader,
    model_val_select_loader,
    train_target_cache: FrozenTargetLogitCache | None,
    model_val_stop_target_cache: FrozenTargetLogitCache | None,
    model_val_select_target_cache: FrozenTargetLogitCache | None,
    output_dir: str | Path,
    lineage: Mapping[str, str],
    objective: DistillationObjective,
    configuration_id: str,
    run_id: str,
    deployed_parameters: int,
    config: DistillationTrainConfig | None = None,
    matched_optimizer_updates: int | None = None,
    schedule_match_source_sha256: str | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Train Pview against an exact frozen clean or robust consumer."""

    config = config or DistillationTrainConfig()
    lineage = _validate_hash_inventory(
        lineage, DISTILLATION_LINEAGE_FIELDS, name="distillation"
    )
    if objective.kd and train_target_cache is None:
        raise ValueError("KD training requires train target-logit cache")
    if objective.uses_privileged_view and objective.kd == 0 and train_target_cache is None:
        # Representation-only rows do not consume logits, but the cache still
        # supplies the same-consumer oracle counterfactual used for reporting.
        raise ValueError("privileged row requires same-consumer target cache")
    caches = (
        train_target_cache,
        model_val_stop_target_cache,
        model_val_select_target_cache,
    )
    for expected, cache in zip(
        (
            lineage["train_target_logit_cache_sha256"],
            lineage["model_val_stop_target_logit_cache_sha256"],
            lineage["model_val_select_target_logit_cache_sha256"],
        ),
        caches,
    ):
        if cache is not None and cache.content_hash != expected:
            raise ValueError("distillation target-logit lineage mismatch")
    for cache, split in zip(
        caches, ("train", "model_val_stop", "model_val_select")
    ):
        if cache is None:
            continue
        if cache.manifest["split"] != split:
            raise ValueError("distillation target-logit split mismatch")
        for cache_field, distillation_field in (
            ("hlt_preprocessing_sha256", "hlt_preprocessing_sha256"),
            ("class_order_sha256", "class_order_sha256"),
            ("coordinate_binding_sha256", "coordinate_binding_sha256"),
            ("consumer_checkpoint_sha256", "consumer_checkpoint_sha256"),
        ):
            if (
                cache.manifest["lineage"][cache_field]
                != lineage[distillation_field]
            ):
                raise ValueError(
                    "distillation target-logit parent lineage mismatch"
                )
    if objective.loss_id == "L_CE":
        if (
            matched_optimizer_updates is None
            or not isinstance(matched_optimizer_updates, int)
            or isinstance(matched_optimizer_updates, bool)
            or matched_optimizer_updates <= 0
        ):
            raise ValueError(
                "schedule-matched CE control requires exact optimizer updates"
            )
        require_sha256(
            "schedule_match_source_sha256", schedule_match_source_sha256
        )
    elif (
        matched_optimizer_updates is not None
        or schedule_match_source_sha256 is not None
    ):
        raise ValueError("matched update budget is restricted to CE-only control")
    consumer_state = freeze_consumer_for_distillation(frozen_consumer)
    initial_predictor_state = module_state_sha256(predictor)
    if (
        initial_predictor_state
        != lineage["initial_predictor_state_sha256"]
    ):
        raise ValueError("distillation predictor initialization mismatch")
    for cache in caches:
        if cache is not None:
            cache.validate_consumer(frozen_consumer)
    for parameter in predictor.parameters():
        parameter.requires_grad_(True)
    _set_seed(config.seed)
    device = torch.device(device)
    predictor.to(device)
    frozen_consumer.to(device)
    loader_batch_size = getattr(train_loader, "batch_size", None)
    if loader_batch_size is not None and int(loader_batch_size) != config.batch_size:
        raise ValueError("distillation train-loader batch size changed")
    optimizer = torch.optim.AdamW(
        predictor.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(config.adam_beta1, config.adam_beta2),
    )
    amp_enabled = bool(config.amp and device.type == "cuda")
    scaler = _grad_scaler(amp_enabled)
    total_updates = max(len(train_loader) * config.maximum_epochs, 1)
    rows = []
    selected_row = None
    selected_state = None
    maximum_accuracy = -math.inf
    best_early_accuracy = -math.inf
    stale_epochs = updates = 0
    labeled_examples_processed = 0
    for epoch in range(1, config.maximum_epochs + 1):
        predictor.train()
        sums = {
            name: 0.0
            for name in (
                "total",
                "kd",
                "ce",
                "huber",
                "cosine",
                "relational",
                "uncertainty",
                "trust",
                "balance",
            )
        }
        batches = 0
        for raw in train_loader:
            if (
                matched_optimizer_updates is not None
                and updates >= matched_optimizer_updates
            ):
                break
            batch = _move_distillation_batch(
                raw,
                device,
                expose_privileged=objective.uses_representation_target,
            )
            target_logits = (
                train_target_cache.lookup(batch["event_ids"], device=device)
                if objective.kd
                else None
            )
            optimizer.zero_grad(set_to_none=True)
            updates += 1
            if objective.ce:
                labeled_examples_processed += int(batch["labels"].numel())
            learning_rate = stage_e_learning_rate(
                updates, total_updates, config
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            with _autocast(amp_enabled):
                output = predictor(
                    batch["features"],
                    batch["lorentz_vectors"],
                    batch["mask"],
                )
                live_output = _consumer_forward(
                    frozen_consumer, batch, output.mean
                )
                losses = distillation_losses(
                    predictor_output=output,
                    live_consumer_output=live_output,
                    target_logits=target_logits,
                    labels=batch["labels"],
                    true_view=batch.get("true_view"),
                    mask=batch["mask"],
                    objective=objective,
                )
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                predictor.parameters(), config.gradient_clip
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("distillation gradient is non-finite")
            scaler.step(optimizer)
            scaler.update()
            batches += 1
            for name in sums:
                sums[name] += float(losses[name].detach().item())
        if not batches:
            if (
                matched_optimizer_updates is not None
                and updates >= matched_optimizer_updates
            ):
                break
            raise ValueError("distillation train loader is empty")
        validation = evaluate_distilled_bundle(
            predictor,
            frozen_consumer,
            model_val_stop_loader,
            split="model_val_stop",
            target_cache=model_val_stop_target_cache,
            device=device,
            permit_true_view_metrics=objective.uses_privileged_view,
        )
        row = {
            "epoch": epoch,
            "optimizer_updates": updates,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_objective": {
                name: value / batches for name, value in sums.items()
            },
            "model_val_stop": validation,
        }
        rows.append(row)
        accuracy = float(validation["deployable_accuracy"])
        maximum_accuracy = max(maximum_accuracy, accuracy)
        if _candidate_better(
            row,
            selected_row,
            maximum_accuracy=maximum_accuracy,
            tolerance=config.accuracy_tolerance,
        ):
            selected_row = row
            selected_state = _state_to_cpu(predictor)
        if accuracy > best_early_accuracy:
            best_early_accuracy = accuracy
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= config.early_stop_patience:
            if matched_optimizer_updates is None:
                break
        if (
            matched_optimizer_updates is not None
            and updates >= matched_optimizer_updates
        ):
            break
    if selected_row is None or selected_state is None:
        raise RuntimeError("distillation did not select a checkpoint")
    if (
        matched_optimizer_updates is not None
        and updates != matched_optimizer_updates
    ):
        raise RuntimeError("CE control did not reach its matched update budget")
    predictor.load_state_dict(selected_state, strict=True)
    predictor.to(device).eval()
    train_metrics = evaluate_distilled_bundle(
        predictor,
        frozen_consumer,
        train_loader,
        split="train",
        target_cache=train_target_cache,
        device=device,
        permit_true_view_metrics=objective.uses_representation_target,
    )
    stop_metrics = evaluate_distilled_bundle(
        predictor,
        frozen_consumer,
        model_val_stop_loader,
        split="model_val_stop",
        target_cache=model_val_stop_target_cache,
        device=device,
        permit_true_view_metrics=objective.uses_representation_target,
    )
    select_metrics = evaluate_distilled_bundle(
        predictor,
        frozen_consumer,
        model_val_select_loader,
        split="model_val_select",
        target_cache=model_val_select_target_cache,
        device=device,
        permit_true_view_metrics=objective.uses_representation_target,
    )
    generalization = build_generalization_report(
        train_metrics=train_metrics,
        model_val_stop_metrics=stop_metrics,
        model_val_select_metrics=select_metrics,
        configuration_id=configuration_id,
        seed=config.seed,
    )
    state_after = module_state_sha256(frozen_consumer)
    if state_after != consumer_state:
        raise RuntimeError("frozen consumer changed during distillation")
    hlt_batch = _first_hlt_batch(model_val_stop_loader, device)
    with torch.no_grad():
        reference = DeployableParticleViewBundle(
            predictor, frozen_consumer
        ).eval()(**hlt_batch)
    deployment_audit = audit_teacher_independent_deployment(
        predictor=predictor,
        consumer=frozen_consumer,
        hlt_batch=hlt_batch,
        reference_logits=reference,
        dependency_manifest={
            "inputs": ["points", "features", "lorentz_vectors", "mask"],
            "coordinate_binding_sha256": lineage["coordinate_binding_sha256"],
            "hlt_preprocessing_sha256": lineage["hlt_preprocessing_sha256"],
        },
    )
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = root / "selected_distilled_predictor.pt"
    if checkpoint_path.exists():
        raise FileExistsError("distillation checkpoint already exists")
    checkpoint = {
        "contract": PARTICLE_VIEW_DISTILLATION_CHECKPOINT_CONTRACT,
        "mode": "frozen_consumer",
        "configuration_id": configuration_id,
        "run_id": run_id,
        "objective": objective.to_payload(),
        "objective_sha256": objective.content_hash,
        "training_config": config.to_payload(),
        "training_config_sha256": config.content_hash,
        "lineage": lineage,
        "selected_epoch": int(selected_row["epoch"]),
        "predictor_state_dict": _state_to_cpu(predictor),
        "consumer_state_sha256": consumer_state,
    }
    torch.save(checkpoint, checkpoint_path)
    registration = with_content_hash(
        {
            "contract": PARTICLE_VIEW_DISTILLATION_REGISTRATION_CONTRACT,
            "mode": "frozen_consumer",
            "configuration_id": configuration_id,
            "run_id": run_id,
            "seed": config.seed,
            "objective": objective.to_payload(),
            "objective_sha256": objective.content_hash,
            "training_config": config.to_payload(),
            "training_config_sha256": config.content_hash,
            "lineage": lineage,
            "checkpoint_file": checkpoint_path.name,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "selected_epoch": int(selected_row["epoch"]),
            "epochs_completed": len(rows),
            "optimizer_updates": updates,
            "matched_optimizer_updates": matched_optimizer_updates,
            "schedule_match_source_sha256": schedule_match_source_sha256,
            "exact_schedule_match_completed": (
                matched_optimizer_updates is None
                or updates == matched_optimizer_updates
            ),
            "ce_bearing_updates": updates if objective.ce else 0,
            "label_bearing_updates": updates if objective.ce else 0,
            "labeled_examples_processed": labeled_examples_processed,
            "teacher_kd_updates": updates if objective.kd else 0,
            "view_supervision_updates": (
                updates if objective.uses_representation_target else 0
            ),
            "deployed_parameters": int(deployed_parameters),
            "consumer_frozen": True,
            "consumer_state_sha256_before": consumer_state,
            "consumer_state_sha256_after": state_after,
            "initial_predictor_state_sha256": initial_predictor_state,
            "model_val_stop": stop_metrics,
            "model_val_select": select_metrics,
            "privileged_claim_eligible": objective.privileged_claim_eligible,
            "pre_stage_g_deployable_eligible": (
                objective.pre_stage_g_deployable_eligible
            ),
            "deployment_audit": deployment_audit,
            "generalization_report_sha256": generalization["content_hash"],
            "model_val_select_evaluation_count": 1,
            "stack_val_loaded": False,
            "final_test_loaded": False,
        }
    )
    write_immutable_json(
        root / "distillation_training_curves.json",
        with_content_hash(
            {
                "contract": "particle_view_distillation_training_curves_v1",
                "configuration_id": configuration_id,
                "rows": rows,
                "selected_epoch": int(selected_row["epoch"]),
            }
        ),
    )
    write_immutable_json(
        root / "distillation_generalization.json", generalization
    )
    write_immutable_json(
        root / "distillation_registration.json", registration
    )
    for parameter in predictor.parameters():
        parameter.requires_grad_(False)
    return registration


def joint_finetuning_losses(
    *,
    predictor_output: ParticleViewPredictorOutput,
    live_consumer_output: Any,
    target_logits: torch.Tensor | None,
    labels: torch.Tensor,
    true_view: torch.Tensor | None,
    mask: torch.Tensor,
    config: JointFineTuneConfig,
) -> dict[str, torch.Tensor]:
    objective = (
        DistillationObjective(
            loss_id="L_PRIMARY",
            kd=1.0,
            huber=0.25,
            ce=0.10,
            trust=0.01,
        )
        if config.privileged
        else DistillationObjective(
            loss_id="L_CE",
            ce=1.0,
            privileged_claim_eligible=False,
        )
    )
    return distillation_losses(
        predictor_output=predictor_output,
        live_consumer_output=live_consumer_output,
        target_logits=target_logits,
        labels=labels,
        true_view=true_view,
        mask=mask,
        objective=objective,
    )


def build_schedule_matched_ce_control(
    objective: DistillationObjective,
    *,
    source_registration_sha256: str,
    matched_optimizer_updates: int,
    stage_mode: str = "frozen_consumer",
) -> dict[str, Any]:
    require_sha256(
        "source_registration_sha256", source_registration_sha256
    )
    if (
        not isinstance(matched_optimizer_updates, int)
        or isinstance(matched_optimizer_updates, bool)
        or matched_optimizer_updates <= 0
    ):
        raise ValueError("matched CE update budget must be positive")
    if stage_mode not in {"frozen_consumer", "joint"}:
        raise ValueError("matched CE stage mode is invalid")
    ce = build_distillation_loss_screen()["L_CE"]
    return with_content_hash(
        {
            "contract": "particle_view_schedule_matched_ce_control_v1",
            "source_objective_sha256": objective.content_hash,
            "source_registration_sha256": source_registration_sha256,
            "matched_optimizer_updates": matched_optimizer_updates,
            "stage_mode": stage_mode,
            "control_objective": ce.to_payload(),
            "control_objective_sha256": ce.content_hash,
            "same_architecture": True,
            "same_initialization": True,
            "same_stage_boundaries": True,
            "same_freeze_unfreeze_schedule": True,
            "same_optimizer_update_budget": True,
            "privileged_targets_removed": True,
            "teacher_kd_removed": True,
            "privileged_claim_eligible": False,
        }
    )


def train_joint_finetuning(
    *,
    predictor: nn.Module,
    selected_frozen_consumer: nn.Module,
    train_loader,
    model_val_stop_loader,
    model_val_select_loader,
    train_target_cache: FrozenTargetLogitCache | None,
    model_val_stop_target_cache: FrozenTargetLogitCache | None,
    model_val_select_target_cache: FrozenTargetLogitCache | None,
    output_dir: str | Path,
    lineage: Mapping[str, str],
    configuration_id: str,
    run_id: str,
    config: JointFineTuneConfig | None = None,
    matched_optimizer_updates: int | None = None,
    schedule_match_source_sha256: str | None = None,
    device: str | torch.device = "cpu",
) -> tuple[nn.Module, nn.Module, dict[str, Any]]:
    """Run Dview_joint or its identical CE-only freeze/unfreeze control."""

    config = config or JointFineTuneConfig()
    lineage = _validate_hash_inventory(
        lineage, JOINT_LINEAGE_FIELDS, name="joint fine-tuning"
    )
    if lineage["schedule_contract_sha256"] != joint_schedule_contract_sha256(
        config
    ):
        raise ValueError("joint schedule contract mismatch")
    if config.privileged and (
        matched_optimizer_updates is not None
        or schedule_match_source_sha256 is not None
    ):
        raise ValueError("privileged joint branch defines the source budget")
    if not config.privileged and (
        matched_optimizer_updates is None
        or not isinstance(matched_optimizer_updates, int)
        or isinstance(matched_optimizer_updates, bool)
        or matched_optimizer_updates <= 0
    ):
        raise ValueError(
            "joint CE-only control requires the privileged branch update budget"
        )
    if not config.privileged:
        require_sha256(
            "schedule_match_source_sha256", schedule_match_source_sha256
        )
    _set_seed(config.seed)
    device = torch.device(device)
    observed_initial_predictor = module_state_sha256(predictor)
    observed_initial_consumer = module_state_sha256(selected_frozen_consumer)
    if (
        observed_initial_predictor
        != lineage["initial_predictor_state_sha256"]
        or observed_initial_consumer
        != lineage["initial_consumer_state_sha256"]
    ):
        raise ValueError("joint initial bundle lineage mismatch")
    oracle_consumer = deepcopy(selected_frozen_consumer).to(device)
    freeze_consumer_for_distillation(oracle_consumer)
    live_consumer = deepcopy(selected_frozen_consumer).to(device)
    predictor = deepcopy(predictor).to(device)
    for module in (predictor, live_consumer):
        module.train()
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    if config.privileged:
        for cache, split, lineage_field in zip(
            (
                train_target_cache,
                model_val_stop_target_cache,
                model_val_select_target_cache,
            ),
            ("train", "model_val_stop", "model_val_select"),
            (
                "train_target_logit_cache_sha256",
                "model_val_stop_target_logit_cache_sha256",
                "model_val_select_target_logit_cache_sha256",
            ),
        ):
            if cache is None:
                raise ValueError("joint privileged branch requires target caches")
            if (
                cache.content_hash != lineage[lineage_field]
                or cache.manifest["split"] != split
                or cache.manifest["lineage"]["consumer_checkpoint_sha256"]
                != lineage["oracle_consumer_checkpoint_sha256"]
                or cache.manifest["lineage"]["coordinate_binding_sha256"]
                != lineage["coordinate_binding_sha256"]
            ):
                raise ValueError("joint target-logit lineage mismatch")
            cache.validate_consumer(oracle_consumer)
    parameters = list(predictor.parameters()) + list(live_consumer.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    amp_enabled = bool(config.amp and device.type == "cuda")
    scaler = _grad_scaler(amp_enabled)
    rows = []
    selected_row = None
    selected_predictor_state = None
    selected_consumer_state = None
    maximum_accuracy = best_early = -math.inf
    stale = updates = 0
    labeled_examples_processed = 0
    for epoch in range(1, config.maximum_epochs + 1):
        predictor.train()
        live_consumer.train()
        batches = 0
        for raw in train_loader:
            if (
                matched_optimizer_updates is not None
                and updates >= matched_optimizer_updates
            ):
                break
            batch = _move_distillation_batch(
                raw, device, expose_privileged=config.privileged
            )
            target_logits = (
                train_target_cache.lookup(batch["event_ids"], device=device)
                if config.privileged
                else None
            )
            optimizer.zero_grad(set_to_none=True)
            with _autocast(amp_enabled):
                output = predictor(
                    batch["features"], batch["lorentz_vectors"], batch["mask"]
                )
                live_output = _consumer_forward(
                    live_consumer, batch, output.mean
                )
                losses = joint_finetuning_losses(
                    predictor_output=output,
                    live_consumer_output=live_output,
                    target_logits=target_logits,
                    labels=batch["labels"],
                    true_view=batch.get("true_view"),
                    mask=batch["mask"],
                    config=config,
                )
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                parameters, config.gradient_clip
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("joint gradient is non-finite")
            scaler.step(optimizer)
            scaler.update()
            updates += 1
            labeled_examples_processed += int(batch["labels"].numel())
            batches += 1
        if not batches:
            if (
                matched_optimizer_updates is not None
                and updates >= matched_optimizer_updates
            ):
                break
            raise ValueError("joint fine-tuning train loader is empty")
        validation = evaluate_distilled_bundle(
            predictor,
            live_consumer,
            model_val_stop_loader,
            split="model_val_stop",
            target_cache=(
                model_val_stop_target_cache if config.privileged else None
            ),
            target_consumer=oracle_consumer,
            device=device,
            joint_consumer_changed=True,
            permit_true_view_metrics=config.privileged,
        )
        row = {
            "epoch": epoch,
            "optimizer_updates": updates,
            "model_val_stop": validation,
        }
        rows.append(row)
        accuracy = float(validation["deployable_accuracy"])
        maximum_accuracy = max(maximum_accuracy, accuracy)
        if _candidate_better(
            row,
            selected_row,
            maximum_accuracy=maximum_accuracy,
            tolerance=config.accuracy_tolerance,
        ):
            selected_row = row
            selected_predictor_state = _state_to_cpu(predictor)
            selected_consumer_state = _state_to_cpu(live_consumer)
        if accuracy > best_early:
            best_early = accuracy
            stale = 0
        else:
            stale += 1
        if stale >= config.early_stop_patience:
            if matched_optimizer_updates is None:
                break
        if (
            matched_optimizer_updates is not None
            and updates >= matched_optimizer_updates
        ):
            break
    if selected_row is None:
        raise RuntimeError("joint fine-tuning selected no checkpoint")
    if (
        matched_optimizer_updates is not None
        and updates != matched_optimizer_updates
    ):
        raise RuntimeError("joint CE control did not match optimizer updates")
    predictor.load_state_dict(selected_predictor_state, strict=True)
    live_consumer.load_state_dict(selected_consumer_state, strict=True)
    predictor.eval()
    live_consumer.eval()
    evaluation_cache_train = train_target_cache if config.privileged else None
    evaluation_cache_stop = (
        model_val_stop_target_cache if config.privileged else None
    )
    evaluation_cache_select = (
        model_val_select_target_cache if config.privileged else None
    )
    train_metrics = evaluate_distilled_bundle(
        predictor,
        live_consumer,
        train_loader,
        split="train",
        target_cache=evaluation_cache_train,
        target_consumer=oracle_consumer,
        device=device,
        joint_consumer_changed=True,
        permit_true_view_metrics=config.privileged,
    )
    stop_metrics = evaluate_distilled_bundle(
        predictor,
        live_consumer,
        model_val_stop_loader,
        split="model_val_stop",
        target_cache=evaluation_cache_stop,
        target_consumer=oracle_consumer,
        device=device,
        joint_consumer_changed=True,
        permit_true_view_metrics=config.privileged,
    )
    select_metrics = evaluate_distilled_bundle(
        predictor,
        live_consumer,
        model_val_select_loader,
        split="model_val_select",
        target_cache=evaluation_cache_select,
        target_consumer=oracle_consumer,
        device=device,
        joint_consumer_changed=True,
        permit_true_view_metrics=config.privileged,
    )
    generalization = build_generalization_report(
        train_metrics=train_metrics,
        model_val_stop_metrics=stop_metrics,
        model_val_select_metrics=select_metrics,
        configuration_id=configuration_id,
        seed=config.seed,
    )
    hlt_batch = _first_hlt_batch(model_val_stop_loader, device)
    with torch.no_grad():
        reference = DeployableParticleViewBundle(
            predictor, live_consumer
        ).eval()(**hlt_batch)
    deployment_audit = audit_teacher_independent_deployment(
        predictor=predictor,
        consumer=live_consumer,
        hlt_batch=hlt_batch,
        reference_logits=reference,
        dependency_manifest={
            "inputs": ["points", "features", "lorentz_vectors", "mask"],
            "coordinate_binding_sha256": lineage["coordinate_binding_sha256"],
            "hlt_preprocessing_sha256": lineage["hlt_preprocessing_sha256"],
        },
    )
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = root / "selected_joint_bundle.pt"
    if checkpoint_path.exists():
        raise FileExistsError("joint checkpoint already exists")
    torch.save(
        {
            "contract": PARTICLE_VIEW_DISTILLATION_CHECKPOINT_CONTRACT,
            "mode": (
                "Dview_joint" if config.privileged else "Dview_joint_ce_control"
            ),
            "configuration_id": configuration_id,
            "run_id": run_id,
            "lineage": lineage,
            "config": config.to_payload(),
            "selected_epoch": int(selected_row["epoch"]),
            "predictor_state_dict": selected_predictor_state,
            "consumer_state_dict": selected_consumer_state,
            "oracle_consumer_state_sha256": module_state_sha256(
                oracle_consumer
            ),
            "initial_predictor_state_sha256": observed_initial_predictor,
            "initial_consumer_state_sha256": observed_initial_consumer,
        },
        checkpoint_path,
    )
    registration = with_content_hash(
        {
            "contract": PARTICLE_VIEW_DISTILLATION_REGISTRATION_CONTRACT,
            "mode": (
                "Dview_joint" if config.privileged else "Dview_joint_ce_control"
            ),
            "configuration_id": configuration_id,
            "run_id": run_id,
            "seed": config.seed,
            "lineage": lineage,
            "joint_config": config.to_payload(),
            "joint_config_sha256": config.content_hash,
            "checkpoint_file": checkpoint_path.name,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "selected_epoch": int(selected_row["epoch"]),
            "epochs_completed": len(rows),
            "optimizer_updates": updates,
            "matched_optimizer_updates": matched_optimizer_updates,
            "schedule_match_source_sha256": schedule_match_source_sha256,
            "exact_schedule_match_completed": (
                matched_optimizer_updates is None
                or updates == matched_optimizer_updates
            ),
            "ce_bearing_updates": updates,
            "label_bearing_updates": updates,
            "labeled_examples_processed": labeled_examples_processed,
            "teacher_kd_updates": updates if config.privileged else 0,
            "view_supervision_updates": updates if config.privileged else 0,
            "consumer_frozen": False,
            "oracle_target_network_frozen": True,
            "oracle_consumer_state_sha256": module_state_sha256(
                oracle_consumer
            ),
            "initial_predictor_state_sha256": observed_initial_predictor,
            "initial_consumer_state_sha256": observed_initial_consumer,
            "recovery_status": "undefined",
            "recovery_fraction": None,
            "model_val_select": select_metrics,
            "model_val_stop": stop_metrics,
            "generalization_report_sha256": generalization["content_hash"],
            "deployment_audit": deployment_audit,
            "privileged_claim_eligible": config.privileged,
            "schedule_matched_ce_control": (
                "required" if config.privileged else "this_artifact"
            ),
            "model_val_select_evaluation_count": 1,
            "stack_val_loaded": False,
            "final_test_loaded": False,
        }
    )
    write_immutable_json(
        root / "joint_training_curves.json",
        with_content_hash(
            {
                "contract": "particle_view_joint_training_curves_v1",
                "rows": rows,
                "selected_epoch": int(selected_row["epoch"]),
            }
        ),
    )
    write_immutable_json(
        root / "joint_generalization.json", generalization
    )
    write_immutable_json(root / "joint_registration.json", registration)
    return predictor, live_consumer, registration


def validate_distillation_registration(
    registration: Mapping[str, Any],
    *,
    root: str | Path,
    expected_lineage: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    validate_content_hash(
        registration,
        expected_contract=PARTICLE_VIEW_DISTILLATION_REGISTRATION_CONTRACT,
    )
    mode = registration.get("mode")
    if mode not in {
        "frozen_consumer",
        "Dview_joint",
        "Dview_joint_ce_control",
    }:
        raise ValueError("distillation registration mode is invalid")
    lineage_fields = (
        DISTILLATION_LINEAGE_FIELDS
        if mode == "frozen_consumer"
        else JOINT_LINEAGE_FIELDS
    )
    lineage = _validate_hash_inventory(
        registration["lineage"], lineage_fields, name="distillation registration"
    )
    if expected_lineage is not None and lineage != _validate_hash_inventory(
        expected_lineage, lineage_fields, name="expected distillation"
    ):
        raise ValueError("distillation registration lineage mismatch")
    if registration.get("model_val_select_evaluation_count") != 1:
        raise ValueError("model_val_select evaluation-count contract failed")
    if registration.get("stack_val_loaded") or registration.get(
        "final_test_loaded"
    ):
        raise ValueError("distillation registration accessed a sealed split")
    if mode == "frozen_consumer":
        if (
            not registration.get("consumer_frozen")
            or registration.get("consumer_state_sha256_before")
            != registration.get("consumer_state_sha256_after")
        ):
            raise ValueError("frozen-consumer immutability contract failed")
        if registration["objective_sha256"] != canonical_sha256(
            registration["objective"]
        ):
            raise ValueError("distillation objective hash mismatch")
        if registration["training_config_sha256"] != canonical_sha256(
            registration["training_config"]
        ):
            raise ValueError("distillation training-config hash mismatch")
        if registration["objective"]["loss_id"] == "L_CE":
            require_sha256(
                "schedule_match_source_sha256",
                registration.get("schedule_match_source_sha256"),
            )
            if (
                registration.get("matched_optimizer_updates")
                != registration.get("optimizer_updates")
                or not registration.get("exact_schedule_match_completed")
            ):
                raise ValueError("frozen CE schedule-match contract failed")
    else:
        if (
            registration.get("consumer_frozen")
            or not registration.get("oracle_target_network_frozen")
            or registration.get("recovery_status") != "undefined"
            or registration.get("recovery_fraction") is not None
        ):
            raise ValueError("joint-consumer semantic contract failed")
        if registration["joint_config_sha256"] != canonical_sha256(
            registration["joint_config"]
        ):
            raise ValueError("joint config hash mismatch")
        if mode == "Dview_joint_ce_control":
            require_sha256(
                "schedule_match_source_sha256",
                registration.get("schedule_match_source_sha256"),
            )
            if (
                registration.get("matched_optimizer_updates")
                != registration.get("optimizer_updates")
                or not registration.get("exact_schedule_match_completed")
            ):
                raise ValueError("joint CE schedule-match contract failed")
    checkpoint_path = Path(root) / registration["checkpoint_file"]
    if sha256_file(checkpoint_path) != registration["checkpoint_sha256"]:
        raise ValueError("distillation checkpoint hash mismatch")
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if (
        checkpoint.get("contract")
        != PARTICLE_VIEW_DISTILLATION_CHECKPOINT_CONTRACT
        or checkpoint.get("mode") != mode
        or checkpoint.get("configuration_id")
        != registration["configuration_id"]
        or checkpoint.get("run_id") != registration["run_id"]
        or checkpoint.get("selected_epoch") != registration["selected_epoch"]
        or checkpoint.get("lineage") != lineage
    ):
        raise ValueError("distillation checkpoint payload mismatch")
    return {
        "ok": True,
        "mode": mode,
        "checkpoint": str(checkpoint_path),
        "checkpoint_payload": checkpoint,
        "content_hash": registration["content_hash"],
    }


def load_registered_distillation_bundle(
    predictor: nn.Module,
    consumer: nn.Module,
    *,
    registration_path: str | Path,
    expected_lineage: Mapping[str, str] | None = None,
    expected_frozen_consumer_state_sha256: str | None = None,
) -> tuple[nn.Module, nn.Module]:
    path = Path(registration_path)
    registration = json.loads(path.read_text(encoding="utf-8"))
    validated = validate_distillation_registration(
        registration,
        root=path.parent,
        expected_lineage=expected_lineage,
    )
    checkpoint = validated["checkpoint_payload"]
    predictor.load_state_dict(checkpoint["predictor_state_dict"], strict=True)
    if validated["mode"] == "frozen_consumer":
        observed = module_state_sha256(consumer)
        expected = checkpoint["consumer_state_sha256"]
        if (
            observed != expected
            or (
                expected_frozen_consumer_state_sha256 is not None
                and observed != expected_frozen_consumer_state_sha256
            )
        ):
            raise ValueError("frozen consumer differs from distillation binding")
        freeze_consumer_for_distillation(consumer)
    else:
        consumer.load_state_dict(checkpoint["consumer_state_dict"], strict=True)
        consumer.eval()
        for parameter in consumer.parameters():
            parameter.requires_grad_(False)
    predictor.eval()
    for parameter in predictor.parameters():
        parameter.requires_grad_(False)
    return predictor, consumer


__all__ = [
    "DISTILLATION_LINEAGE_FIELDS",
    "JOINT_LINEAGE_FIELDS",
    "PARTICLE_VIEW_DEPLOYMENT_AUDIT_CONTRACT",
    "PARTICLE_VIEW_DISTILLATION_CAMPAIGN_CONTRACT",
    "PARTICLE_VIEW_DISTILLATION_CHECKPOINT_CONTRACT",
    "PARTICLE_VIEW_DISTILLATION_CONFIG_CONTRACT",
    "PARTICLE_VIEW_DISTILLATION_OBJECTIVE_CONTRACT",
    "PARTICLE_VIEW_DISTILLATION_RANKING_CONTRACT",
    "PARTICLE_VIEW_DISTILLATION_REGISTRATION_CONTRACT",
    "PARTICLE_VIEW_GENERALIZATION_REPORT_CONTRACT",
    "PARTICLE_VIEW_LOSS_IDS",
    "PARTICLE_VIEW_TARGET_LOGIT_CACHE_CONTRACT",
    "PRIVILEGED_CLAIM_INELIGIBLE_LOSSES",
    "TARGET_LOGIT_CACHE_LINEAGE_FIELDS",
    "DeployableParticleViewBundle",
    "DistillationCampaignRow",
    "DistillationObjective",
    "DistillationTrainConfig",
    "FrozenTargetLogitCache",
    "JointFineTuneConfig",
    "SameConsumerForward",
    "audit_teacher_independent_deployment",
    "build_distillation_loss_screen",
    "build_generalization_report",
    "build_schedule_matched_ce_control",
    "build_target_loss_interaction_campaign",
    "distillation_losses",
    "evaluate_distilled_bundle",
    "exact_same_consumer_forward",
    "freeze_consumer_for_distillation",
    "joint_finetuning_losses",
    "joint_schedule_contract_sha256",
    "load_registered_distillation_bundle",
    "module_state_sha256",
    "publish_target_logit_cache",
    "rank_model_val_select_configurations",
    "select_distillation_checkpoint",
    "stage_e_learning_rate",
    "train_frozen_consumer_distillation",
    "train_joint_finetuning",
    "validate_distillation_registration",
]
