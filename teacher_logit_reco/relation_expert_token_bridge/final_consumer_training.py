"""Fixed-budget Step-12 final-consumer training and evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from dataclasses import asdict, dataclass
import hashlib
import io
from pathlib import Path
import random
from typing import Any

import numpy as np

from .contracts import (
    bind_source,
    canonical_sha256,
    load_hashed_json,
    require_sha256,
    source_record,
    validate_content_hash,
    with_content_hash,
    write_immutable_bytes,
    write_immutable_json,
)
from .determinism import optimizer_update_counts, scheduled_learning_rate
from .evaluation import evaluate_classification
from .expert_training import DeterministicExpertSampler, preferred_expert_epoch
from .final_consumers import (
    ADAPTER_VARIANTS,
    BYPASS_CONTROLS,
    NATIVE_DROPOUT_MODES,
    REFINER_VARIANTS,
    UNRESTRICTED_EVIDENCE_VARIANTS,
    HLTResidualAdapter,
    NativeConditionedTokenRefiner,
    UnrestrictedHLTFusion,
    materialize_robust_mixture_banks,
)
from .joint_bridge import validate_common_view_metadata
from .replicas import replica_for
from .registry import EXPERT_ORDER

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


FINAL_CONSUMER_TRAINING_CONTRACT = "retb_final_consumer_training_v1"
FINAL_CONSUMER_CHECKPOINT_CONTRACT = "retb_final_consumer_checkpoint_v1"
FINAL_CONSUMER_CURVES_CONTRACT = "retb_final_consumer_curves_v1"
FINAL_CONSUMER_REGISTRATION_CONTRACT = (
    "retb_final_consumer_registration_v1"
)
FINAL_CONSUMER_INFERENCE_CONTRACT = "retb_final_consumer_inference_v1"
FINAL_CONSUMER_DATASET_CONTRACT = "retb_final_consumer_dataset_v1"
FINAL_CONSUMER_TEMPLATE_CONTRACT = "retb_final_consumer_template_v1"

CONSUMER_KINDS = (
    "PF_FROZEN",
    "OF_ROBUST",
    "TR_REFINE",
    "HF_ADAPTER",
    "HF_UNRESTRICTED",
)
FINAL_DATASET_PARENT_KEYS = frozenset(
    {
        "identity_manifest",
        "HLT_view_cache",
        "joint_prediction_cache",
        "native_HLT_cache",
        "offline_target_cache",
        "target_normalizer_set",
        "uncertainty_calibration",
    }
)


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is required for final-consumer training")
    return torch


def _array(value: Any, *, dtype: Any) -> np.ndarray:
    return np.asarray(value, dtype=dtype)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FinalConsumerDataset(
    torch.utils.data.Dataset if torch is not None else object
):
    """Identity-aligned deployable evidence plus privileged training targets."""

    def __init__(
        self,
        *,
        identities: Sequence[str],
        labels: np.ndarray,
        replica_ids: np.ndarray,
        degraded_view_hashes: Any,
        split: str,
        predicted_banks: Mapping[str, Any],
        calibrated_log_variance: Mapping[str, Any],
        native_banks: Mapping[str, Any],
        native_expert_logits: Mapping[str, Any],
        predicted_expert_logits: Mapping[str, Any],
        oracle_banks: Mapping[str, np.ndarray],
        target_normalized_banks: Mapping[str, np.ndarray],
        target_expert_logits: Mapping[str, np.ndarray],
        oracle_fusion_logits: np.ndarray,
        lineage_hashes: Mapping[str, str],
    ) -> None:
        _require_torch()
        if split not in {
            "model_train",
            "scale_train",
            "val_stop",
            "val_design",
        }:
            raise ValueError("final-consumer dataset split differs")
        ids = tuple(str(value) for value in identities)
        labels = _array(labels, dtype=np.int64)
        declared_replicas = _array(replica_ids, dtype=np.int64)
        if (
            not ids
            or len(ids) != len(set(ids))
            or labels.shape != (len(ids),)
            or bool(((labels < 0) | (labels >= 10)).any())
            or declared_replicas.shape != (len(ids),)
        ):
            raise ValueError("final-consumer identity/label population differs")
        self.identities = ids
        self.labels = labels
        self.split = split
        self.zero_based_epoch = 0
        self.replica_set = (
            (0, 1, 2, 3)
            if split in {"model_train", "scale_train"}
            else (0,)
        )
        expected = np.asarray(
            [
                replica_for(
                    policy="R_MULTI",
                    logical_role=split,
                    epoch=0,
                    canonical_identity=identity,
                )
                for identity in ids
            ],
            dtype=np.int64,
        )
        if not np.array_equal(expected, declared_replicas):
            raise ValueError("final-consumer declared replica IDs differ")

        def replicated(
            value: Any, *, dtype: Any, name: str
        ) -> dict[int, np.ndarray]:
            if isinstance(value, Mapping):
                if {int(key) for key in value} != set(self.replica_set):
                    raise ValueError(
                        f"final-consumer {name} replica coverage differs"
                    )
                result = {
                    replica: _array(
                        value.get(replica, value.get(str(replica))),
                        dtype=dtype,
                    )
                    for replica in self.replica_set
                }
            else:
                array = _array(value, dtype=dtype)
                if self.replica_set == (0,):
                    result = {0: array}
                elif array.ndim >= 2 and array.shape[:2] == (4, len(ids)):
                    result = {
                        replica: array[replica]
                        for replica in self.replica_set
                    }
                else:
                    raise ValueError(
                        f"final-consumer {name} lacks four replicas"
                    )
            if any(len(row) != len(ids) for row in result.values()):
                raise ValueError(
                    f"final-consumer {name} identity coverage differs"
                )
            return result

        if isinstance(degraded_view_hashes, Mapping):
            hashes = {
                replica: tuple(
                    str(value)
                    for value in degraded_view_hashes.get(
                        replica, degraded_view_hashes.get(str(replica))
                    )
                )
                for replica in self.replica_set
            }
        else:
            array = np.asarray(degraded_view_hashes)
            if self.replica_set == (0,):
                hashes = {0: tuple(str(value) for value in array)}
            elif array.shape == (4, len(ids)):
                hashes = {
                    replica: tuple(str(value) for value in array[replica])
                    for replica in self.replica_set
                }
            else:
                raise ValueError(
                    "final-consumer view-hash replicas differ"
                )
        for replica in self.replica_set:
            validate_common_view_metadata(
                identities=ids,
                replica_ids=[replica] * len(ids),
                degraded_view_hashes=hashes[replica],
            )
        self.degraded_view_hashes = hashes
        replica_fields = {
            "predicted_banks": predicted_banks,
            "calibrated_log_variance": calibrated_log_variance,
            "native_banks": native_banks,
            "native_expert_logits": native_expert_logits,
            "predicted_expert_logits": predicted_expert_logits,
        }
        if any(set(value) != set(EXPERT_ORDER) for value in replica_fields.values()):
            raise ValueError("final-consumer expert evidence coverage differs")
        for field, values in replica_fields.items():
            setattr(
                self,
                field,
                {
                    expert: replicated(
                        values[expert],
                        dtype=np.float32,
                        name=f"{field}.{expert}",
                    )
                    for expert in EXPERT_ORDER
                },
            )
        fixed_fields = {
            "oracle_banks": oracle_banks,
            "target_normalized_banks": target_normalized_banks,
            "target_expert_logits": target_expert_logits,
        }
        if any(set(value) != set(EXPERT_ORDER) for value in fixed_fields.values()):
            raise ValueError("final-consumer target coverage differs")
        for field, values in fixed_fields.items():
            setattr(
                self,
                field,
                {
                    expert: _array(values[expert], dtype=np.float32)
                    for expert in EXPERT_ORDER
                },
            )
        self.oracle_fusion_logits = _array(
            oracle_fusion_logits, dtype=np.float32
        )
        self.lineage_hashes = {
            name: require_sha256(value, name=f"lineage.{name}")
            for name, value in sorted(lineage_hashes.items())
        }
        arrays = [self.oracle_fusion_logits]
        for field in replica_fields:
            arrays.extend(
                row
                for values in getattr(self, field).values()
                for row in values.values()
            )
        for field in fixed_fields:
            arrays.extend(getattr(self, field).values())
        if (
            any(len(value) != len(ids) for value in arrays)
            or self.oracle_fusion_logits.shape != (len(ids), 10)
            or any(
                not np.isfinite(value).all() for value in arrays
            )
        ):
            raise ValueError("final-consumer arrays differ")
        for expert in EXPERT_ORDER:
            predicted_shape = self.predicted_banks[expert][
                self.replica_set[0]
            ].shape
            if (
                len(predicted_shape) != 3
                or self.native_banks[expert][
                    self.replica_set[0]
                ].shape
                != predicted_shape
                or self.oracle_banks[expert].shape != predicted_shape
                or self.target_normalized_banks[expert].shape
                != predicted_shape
                or any(
                    self.predicted_banks[expert][replica].shape
                    != predicted_shape
                    or self.native_banks[expert][replica].shape
                    != predicted_shape
                    or self.calibrated_log_variance[expert][replica].shape[
                        :2
                    ]
                    != predicted_shape[:2]
                    or self.native_expert_logits[expert][replica].shape
                    != (len(ids), 10)
                    or self.predicted_expert_logits[expert][replica].shape
                    != (len(ids), 10)
                    for replica in self.replica_set
                )
                or self.target_expert_logits[expert].shape
                != (len(ids), 10)
            ):
                raise ValueError(
                    f"final-consumer {expert} token/logit shapes differ"
                )

    def set_epoch(self, one_based_epoch: int) -> None:
        if int(one_based_epoch) <= 0:
            raise ValueError("final-consumer dataset epoch is one-based")
        self.zero_based_epoch = int(one_based_epoch) - 1

    def __len__(self) -> int:
        return len(self.identities)

    def __getitem__(self, index: int) -> dict[str, Any]:
        replica = replica_for(
            policy="R_MULTI",
            logical_role=self.split,
            epoch=self.zero_based_epoch,
            canonical_identity=self.identities[index],
        )
        row = {
            "identity": self.identities[index],
            "label": self.labels[index],
            "replica_id": replica,
            "degraded_view_hash": self.degraded_view_hashes[replica][index],
            "oracle_fusion_logits": self.oracle_fusion_logits[index],
        }
        for field in (
            "predicted_banks",
            "calibrated_log_variance",
            "native_banks",
            "native_expert_logits",
            "predicted_expert_logits",
        ):
            values = getattr(self, field)
            row[field] = {
                expert: values[expert][replica][index]
                for expert in EXPERT_ORDER
            }
        for field in (
            "oracle_banks",
            "target_normalized_banks",
            "target_expert_logits",
        ):
            values = getattr(self, field)
            row[field] = {
                expert: values[expert][index] for expert in EXPERT_ORDER
            }
        return row


def collate_final_consumer_batch(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    module = _require_torch()
    if not rows:
        raise ValueError("cannot collate an empty final-consumer batch")
    batch = {
        "identities": [row["identity"] for row in rows],
        "labels": module.as_tensor(
            [row["label"] for row in rows], dtype=module.int64
        ),
        "replica_ids": module.as_tensor(
            [row["replica_id"] for row in rows], dtype=module.int64
        ),
        "degraded_view_hashes": [
            row["degraded_view_hash"] for row in rows
        ],
        "oracle_fusion_logits": module.from_numpy(
            np.stack([row["oracle_fusion_logits"] for row in rows])
        ).float(),
    }
    for field in (
        "predicted_banks",
        "calibrated_log_variance",
        "native_banks",
        "native_expert_logits",
        "predicted_expert_logits",
        "oracle_banks",
        "target_normalized_banks",
        "target_expert_logits",
    ):
        batch[field] = {
            expert: module.from_numpy(
                np.stack([row[field][expert] for row in rows])
            ).float()
            for expert in EXPERT_ORDER
        }
    return batch


def make_final_consumer_loader(
    dataset: FinalConsumerDataset,
    *,
    batch_size: int,
    seed: int,
    training: bool,
) -> Any:
    module = _require_torch()
    sampler = (
        DeterministicExpertSampler(dataset, seed=int(seed))
        if training
        else module.utils.data.SequentialSampler(dataset)
    )
    return module.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        sampler=sampler,
        num_workers=0,
        drop_last=False,
        collate_fn=collate_final_consumer_batch,
    )


def _move(value: Any, device: Any) -> Any:
    if isinstance(value, Mapping):
        return {name: _move(item, device) for name, item in value.items()}
    if isinstance(value, list):
        return value
    return value.to(device) if hasattr(value, "to") else value


def _temperature_two_kl(student: Any, teacher: Any) -> Any:
    module = _require_torch()
    return module.nn.functional.kl_div(
        module.log_softmax(student.float() / 2.0, dim=-1),
        module.softmax(teacher.detach().float() / 2.0, dim=-1),
        reduction="batchmean",
    ) * 4.0


def _freeze(module: Any) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def final_consumer_forward_and_objective(
    *,
    model: Any,
    consumer_kind: str,
    batch: Mapping[str, Any],
    frozen_expert_heads: Mapping[str, Any],
    frozen_offline_fusion: Any,
    refiner: NativeConditionedTokenRefiner | None = None,
    zero_based_epoch: int = 0,
) -> tuple[Any, dict[str, Any]]:
    module = _require_torch()
    if consumer_kind not in CONSUMER_KINDS:
        raise ValueError("final-consumer kind is unregistered")
    predicted = batch["predicted_banks"]
    details: dict[str, Any] = {}
    if refiner is not None and consumer_kind != "TR_REFINE":
        with module.set_grad_enabled(any(
            parameter.requires_grad for parameter in refiner.parameters()
        )):
            predicted = refiner(
                predicted_banks=predicted,
                calibrated_log_variance=batch[
                    "calibrated_log_variance"
                ],
                native_banks=batch["native_banks"],
            )["refined_banks"]
    if consumer_kind == "PF_FROZEN":
        try:
            logits = model(predicted_banks=predicted)
        except TypeError:
            logits = model(token_banks=predicted)
        loss = module.nn.functional.cross_entropy(
            logits, batch["labels"].long()
        )
    elif consumer_kind == "OF_ROBUST":
        if model.training:
            mixed, schedule = materialize_robust_mixture_banks(
                identities=batch["identities"],
                zero_based_epoch=zero_based_epoch,
                oracle_banks=batch["oracle_banks"],
                predicted_banks=predicted,
            )
            details["mixture_schedule"] = schedule
        else:
            mixed = predicted
        logits = model(token_banks=mixed)
        loss = module.nn.functional.cross_entropy(
            logits, batch["labels"].long()
        )
    elif consumer_kind == "TR_REFINE":
        output = model(
            predicted_banks=predicted,
            calibrated_log_variance=batch["calibrated_log_variance"],
            native_banks=batch["native_banks"],
        )
        refined = output["refined_banks"]
        token_loss = module.stack(
            [
                module.nn.functional.huber_loss(
                    refined[expert],
                    batch["oracle_banks"][expert],
                    delta=0.5,
                )
                for expert in EXPERT_ORDER
            ]
        ).mean()
        predicted_head_logits = {
            expert: frozen_expert_heads[expert](refined[expert])
            for expert in EXPERT_ORDER
        }
        expert_kd = module.stack(
            [
                _temperature_two_kl(
                    predicted_head_logits[expert],
                    batch["target_expert_logits"][expert],
                )
                for expert in EXPERT_ORDER
            ]
        ).mean()
        logits = frozen_offline_fusion(token_banks=refined)
        fusion_kd = _temperature_two_kl(
            logits, batch["oracle_fusion_logits"]
        )
        ce = module.nn.functional.cross_entropy(
            logits, batch["labels"].long()
        )
        loss = token_loss + 0.25 * expert_kd + 0.5 * fusion_kd + 0.25 * ce
        details.update(
            {
                "token_loss": token_loss,
                "expert_KD": expert_kd,
                "fusion_KD": fusion_kd,
                "CE": ce,
                "refined_banks": refined,
            }
        )
    elif consumer_kind == "HF_ADAPTER":
        frozen_logits = frozen_offline_fusion(token_banks=predicted)
        output = model(
            frozen_offline_logits=frozen_logits,
            predicted_banks=predicted,
            calibrated_log_variance=batch["calibrated_log_variance"],
            native_banks=batch["native_banks"],
        )
        logits = output["combined_logits"]
        ce = module.nn.functional.cross_entropy(
            logits, batch["labels"].long()
        )
        kd = _temperature_two_kl(logits, batch["oracle_fusion_logits"])
        loss = ce + 0.5 * kd
        details.update(output)
    else:
        predicted_head_logits = {
            expert: frozen_expert_heads[expert](predicted[expert])
            for expert in EXPERT_ORDER
        }
        output = model(
            token_banks=predicted,
            calibrated_log_variance=batch["calibrated_log_variance"],
            native_banks=batch["native_banks"],
            native_expert_logits=batch["native_expert_logits"],
            predicted_expert_logits=predicted_head_logits,
        )
        logits = output["logits"]
        ce = module.nn.functional.cross_entropy(
            logits, batch["labels"].long()
        )
        kd = _temperature_two_kl(logits, batch["oracle_fusion_logits"])
        loss = ce + 0.5 * kd
        details.update(output)
    if not bool(module.isfinite(loss)) or not bool(
        module.isfinite(logits).all()
    ):
        raise FloatingPointError("final-consumer objective is nonfinite")
    details["logits"] = logits
    details["loss"] = loss
    return loss, details


def evaluate_final_consumer(
    *,
    model: Any,
    consumer_kind: str,
    loader: Any,
    frozen_expert_heads: Mapping[str, Any],
    frozen_offline_fusion: Any,
    device: Any,
    refiner: NativeConditionedTokenRefiner | None = None,
) -> dict[str, Any]:
    module = _require_torch()
    model.eval()
    _freeze(frozen_offline_fusion)
    for head in frozen_expert_heads.values():
        _freeze(head)
    if refiner is not None:
        refiner.eval()
    logits, labels, identities, replicas, hashes = [], [], [], [], []
    before_errors = {expert: [] for expert in EXPERT_ORDER}
    after_errors = {expert: [] for expert in EXPERT_ORDER}
    frozen_paths, residual_paths = [], []
    total_loss = 0.0
    with module.no_grad():
        for raw in loader:
            batch = _move(raw, device)
            loss, output = final_consumer_forward_and_objective(
                model=model,
                consumer_kind=consumer_kind,
                batch=batch,
                frozen_expert_heads=frozen_expert_heads,
                frozen_offline_fusion=frozen_offline_fusion,
                refiner=refiner,
            )
            count = len(raw["identities"])
            total_loss += float(loss.cpu()) * count
            logits.append(output["logits"].float().cpu().numpy())
            labels.append(raw["labels"].numpy())
            identities.extend(raw["identities"])
            replicas.extend(raw["replica_ids"].tolist())
            hashes.extend(raw["degraded_view_hashes"])
            if "frozen_path_logits" in output:
                frozen_paths.append(
                    output["frozen_path_logits"].float().cpu().numpy()
                )
                residual_paths.append(
                    output["residual_path_logits"].float().cpu().numpy()
                )
            refined = output.get(
                "refined_banks", batch["predicted_banks"]
            )
            for expert in EXPERT_ORDER:
                before_errors[expert].append(
                    (
                        batch["predicted_banks"][expert]
                        - batch["oracle_banks"][expert]
                    )
                    .square()
                    .float()
                    .cpu()
                    .numpy()
                )
                after_errors[expert].append(
                    (
                        refined[expert]
                        - batch["oracle_banks"][expert]
                    )
                    .square()
                    .float()
                    .cpu()
                    .numpy()
                )
    values, truth = np.concatenate(logits), np.concatenate(labels)
    metrics = evaluate_classification(
        values, truth, split=loader.dataset.split
    )
    return {
        "metrics": metrics,
        "objective": total_loss / len(truth),
        "identities": identities,
        "replica_ids": replicas,
        "degraded_view_hashes": hashes,
        "logits": values.astype(np.float32),
        "frozen_path_logits": (
            None
            if not frozen_paths
            else np.concatenate(frozen_paths).astype(np.float32)
        ),
        "residual_path_logits": (
            None
            if not residual_paths
            else np.concatenate(residual_paths).astype(np.float32)
        ),
        "token_RMSE_before": {
            expert: float(
                np.sqrt(np.concatenate(before_errors[expert]).mean())
            )
            for expert in EXPERT_ORDER
        },
        "token_RMSE_after": {
            expert: float(
                np.sqrt(np.concatenate(after_errors[expert]).mean())
            )
            for expert in EXPERT_ORDER
        },
    }


def evaluate_final_consumer_bypass_controls(
    *,
    model: Any,
    consumer_kind: str,
    loader: Any,
    frozen_expert_heads: Mapping[str, Any],
    frozen_offline_fusion: Any,
    device: Any,
    refiner: NativeConditionedTokenRefiner | None = None,
) -> dict[str, Any]:
    module = _require_torch()
    if consumer_kind not in {"HF_ADAPTER", "HF_UNRESTRICTED"}:
        raise ValueError("bypass controls require an HLT final consumer")
    model.eval()
    _freeze(frozen_offline_fusion)
    for head in frozen_expert_heads.values():
        _freeze(head)
    if refiner is not None:
        refiner.eval()
    controls = [
        control
        for control in BYPASS_CONTROLS
        if not (
            control == "RESIDUAL_GAMMA_ZERO"
            and consumer_kind != "HF_ADAPTER"
        )
    ]
    logits_by_control = {control: [] for control in controls}
    identities, labels = [], []
    with module.no_grad():
        for raw in loader:
            batch = _move(raw, device)
            predicted = batch["predicted_banks"]
            if refiner is not None:
                predicted = refiner(
                    predicted_banks=predicted,
                    calibrated_log_variance=batch[
                        "calibrated_log_variance"
                    ],
                    native_banks=batch["native_banks"],
                )["refined_banks"]
            predicted_head_logits = {
                expert: frozen_expert_heads[expert](predicted[expert])
                for expert in EXPERT_ORDER
            }
            frozen_logits = frozen_offline_fusion(
                token_banks=predicted
            )
            for control in controls:
                if consumer_kind == "HF_ADAPTER":
                    logits = model(
                        frozen_offline_logits=frozen_logits,
                        predicted_banks=predicted,
                        calibrated_log_variance=batch[
                            "calibrated_log_variance"
                        ],
                        native_banks=batch["native_banks"],
                        bypass_control=control,
                    )["combined_logits"]
                else:
                    logits = model(
                        token_banks=predicted,
                        calibrated_log_variance=batch[
                            "calibrated_log_variance"
                        ],
                        native_banks=batch["native_banks"],
                        native_expert_logits=batch[
                            "native_expert_logits"
                        ],
                        predicted_expert_logits=predicted_head_logits,
                        bypass_control=control,
                    )["logits"]
                logits_by_control[control].append(
                    logits.float().cpu().numpy()
                )
            identities.extend(raw["identities"])
            labels.append(raw["labels"].numpy())
    truth = np.concatenate(labels)
    return {
        "split": loader.dataset.split,
        "identities": identities,
        "controls": {
            control: {
                "metrics": evaluate_classification(
                    np.concatenate(logits_by_control[control]),
                    truth,
                    split=loader.dataset.split,
                ),
                "logits": np.concatenate(
                    logits_by_control[control]
                ).astype(np.float32),
            }
            for control in controls
        },
        "used_for_checkpoint_selection": False,
    }


@dataclass(frozen=True)
class FinalConsumerTrainingConfig:
    seed: int
    consumer_kind: str
    model_variant: str
    native_dropout_mode: str = "ND0_NONE"
    maximum_epochs: int = 40
    learning_rate: float = 2.0e-4
    minimum_learning_rate: float = 1.0e-5
    weight_decay: float = 1.0e-4
    gradient_clip: float = 1.0
    microbatch_size: int = 128
    gradient_accumulation_steps: int = 1
    effective_batch_size: int = 128
    campaign_profile: str = "production"

    def validate(self) -> None:
        expected_variants = {
            "OF_ROBUST": {"OF_ROBUST"},
            "TR_REFINE": set(REFINER_VARIANTS) - {"TR0_NONE"},
            "HF_ADAPTER": set(ADAPTER_VARIANTS),
            "HF_UNRESTRICTED": set(UNRESTRICTED_EVIDENCE_VARIANTS),
        }
        effective = int(self.effective_batch_size)
        if (
            self.seed not in {101, 202, 303}
            or self.consumer_kind not in CONSUMER_KINDS
            or self.model_variant
            not in expected_variants.get(self.consumer_kind, set())
            or self.native_dropout_mode not in NATIVE_DROPOUT_MODES
            or (
                self.consumer_kind
                in {"OF_ROBUST", "TR_REFINE"}
                and self.native_dropout_mode != "ND0_NONE"
            )
            or min(
                self.maximum_epochs,
                self.microbatch_size,
                self.gradient_accumulation_steps,
                effective,
            )
            <= 0
            or self.microbatch_size
            * self.gradient_accumulation_steps
            != effective
            or effective > (
                512 if self.consumer_kind == "OF_ROBUST" else 128
            )
            or effective & (effective - 1)
            or self.learning_rate != 2.0e-4
            or self.minimum_learning_rate != 1.0e-5
            or self.weight_decay != 1.0e-4
            or self.gradient_clip != 1.0
            or self.campaign_profile not in {
                "production",
                "miniature_test",
            }
            or (
                self.campaign_profile == "production"
                and self.maximum_epochs != 40
            )
        ):
            raise ValueError("final-consumer training configuration differs")

    def artifact(
        self,
        *,
        step12_bundle_sha256: str,
        run_sha256: str,
        global_determinism_sha256: str,
        lineage_hashes: Mapping[str, str],
    ) -> dict[str, Any]:
        self.validate()
        return with_content_hash(
            {
                "contract": FINAL_CONSUMER_TRAINING_CONTRACT,
                "schema_version": 1,
                "configuration": asdict(self),
                "optimizer": {
                    "name": "AdamW",
                    "betas": [0.9, 0.999],
                    "weight_decay": self.weight_decay,
                },
                "schedule": {
                    "warmup": "min_T_max_1_ceil_0.05T",
                    "post_warmup": "cosine",
                    "minimum_learning_rate": self.minimum_learning_rate,
                },
                "checkpoint_selection": "val_stop_only",
                "fixed_budget": True,
                "early_stopping": False,
                "performance_based_termination": False,
                "parents": {
                    "step12_bundle": require_sha256(
                        step12_bundle_sha256,
                        name="step12_bundle_sha256",
                    ),
                    "run": require_sha256(run_sha256, name="run_sha256"),
                    "global_determinism": require_sha256(
                        global_determinism_sha256,
                        name="global_determinism_sha256",
                    ),
                    **{
                        f"lineage.{name}": require_sha256(
                            value, name=f"lineage.{name}"
                        )
                        for name, value in sorted(lineage_hashes.items())
                    },
                },
            }
        )


def _state_sha256(state: Mapping[str, Any]) -> str:
    stream = io.BytesIO()
    _require_torch().save(dict(state), stream)
    return hashlib.sha256(stream.getvalue()).hexdigest()


def _save_torch(path: Path, payload: Mapping[str, Any]) -> None:
    stream = io.BytesIO()
    _require_torch().save(dict(payload), stream)
    write_immutable_bytes(path, stream.getvalue())


def _replace_torch(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    _require_torch().save(dict(payload), temporary)
    temporary.replace(path)


def _set_seed(seed: int) -> None:
    module = _require_torch()
    random.seed(seed)
    np.random.seed(seed)
    module.manual_seed(seed)
    if module.cuda.is_available():
        module.cuda.manual_seed_all(seed)


def _rng_state() -> dict[str, Any]:
    module = _require_torch()
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": module.get_rng_state(),
        "cuda": (
            None
            if not module.cuda.is_available()
            else module.cuda.get_rng_state_all()
        ),
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    module = _require_torch()
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    module.set_rng_state(state["torch"])
    if state["cuda"] is not None and module.cuda.is_available():
        module.cuda.set_rng_state_all(state["cuda"])


def train_final_consumer(
    *,
    model: Any,
    train_loader: Any,
    val_stop_loader: Any,
    frozen_expert_heads: Mapping[str, Any],
    frozen_offline_fusion: Any,
    output_dir: str | Path,
    run_record: Mapping[str, Any],
    step12_bundle_sha256: str,
    global_determinism_sha256: str,
    lineage_hashes: Mapping[str, str],
    source_snapshot: Mapping[str, Any],
    config: FinalConsumerTrainingConfig,
    device: Any,
    refiner: NativeConditionedTokenRefiner | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    module = _require_torch()
    config.validate()
    validate_content_hash(run_record)
    if (
        run_record.get("consumer_kind") != config.consumer_kind
        or run_record.get("model_variant") != config.model_variant
        or int(run_record.get("pipeline_seed", -1)) != config.seed
        or train_loader.dataset.split
        not in {"model_train", "scale_train"}
        or val_stop_loader.dataset.split != "val_stop"
    ):
        raise ValueError("final-consumer run/configuration lineage differs")
    contract = bind_source(
        config.artifact(
            step12_bundle_sha256=step12_bundle_sha256,
            run_sha256=run_record["content_hash"],
            global_determinism_sha256=global_determinism_sha256,
            lineage_hashes=lineage_hashes,
        ),
        source_snapshot=source_snapshot,
    )
    root = Path(output_dir)
    selected_path = root / "best_model_val.pt"
    last_path = root / "last_state.pt"
    curves_path = root / "training_curves.json"
    registration_path = root / "registration.json"
    candidates = root / "epoch_candidates"
    candidates.mkdir(parents=True, exist_ok=True)
    if (
        selected_path.is_file()
        and curves_path.is_file()
        and registration_path.is_file()
    ):
        curves = load_hashed_json(
            curves_path, expected_contract=FINAL_CONSUMER_CURVES_CONTRACT
        )
        registration = load_hashed_json(
            registration_path,
            expected_contract=FINAL_CONSUMER_REGISTRATION_CONTRACT,
        )
        expected_source_record = source_record(source_snapshot)
        if (
            curves["training_contract_sha256"] != contract["content_hash"]
            or not curves["fixed_budget_completed"]
            or curves.get("source") != expected_source_record
            or registration.get("source") != expected_source_record
            or registration.get("run_record_sha256")
            != run_record["content_hash"]
            or not registration.get("fixed_budget_completed")
            or registration.get("performance_based_termination") is not False
            or registration["checkpoint_sha256"]
            != _file_sha256(selected_path)
        ):
            raise ValueError("reusable final-consumer result differs")
        checkpoint = module.load(
            selected_path, map_location="cpu", weights_only=False
        )
        state = checkpoint.get("model_state_dict")
        if (
            checkpoint.get("contract")
            != FINAL_CONSUMER_CHECKPOINT_CONTRACT
            or checkpoint.get("kind") != "selected_inference"
            or checkpoint.get("training_contract_sha256")
            != contract["content_hash"]
            or not isinstance(state, Mapping)
            or _state_sha256(state)
            != checkpoint.get("model_state_sha256")
        ):
            raise ValueError("reusable final-consumer checkpoint differs")
        model.load_state_dict(state, strict=True)
        return registration
    _set_seed(config.seed)
    resolved = module.device(device)
    if config.campaign_profile == "production" and (
        resolved.type != "cuda"
        or not module.cuda.is_bf16_supported()
        or "GH200" not in module.cuda.get_device_name(resolved).upper()
    ):
        raise RuntimeError(
            "production final-consumer training requires GH200 BF16"
        )
    model.to(resolved)
    if refiner is not None:
        refiner.to(resolved)
        _freeze(refiner)
    frozen_offline_fusion.to(resolved)
    _freeze(frozen_offline_fusion)
    for head in frozen_expert_heads.values():
        head.to(resolved)
        _freeze(head)
    trainable = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad
    ]
    if not trainable:
        raise ValueError("final-consumer graph has no trainable parameters")
    optimizer = module.optim.AdamW(
        trainable,
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=config.weight_decay,
    )
    counts = optimizer_update_counts(
        training_event_count=len(train_loader.dataset),
        maximum_epochs=config.maximum_epochs,
        microbatch_size=config.microbatch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    rows, update_ordinal, start_epoch = [], 0, 1
    if resume and last_path.is_file():
        state = module.load(last_path, map_location="cpu", weights_only=False)
        if (
            state.get("contract") != FINAL_CONSUMER_CHECKPOINT_CONTRACT
            or state.get("kind") != "resumable_last"
            or state.get("training_contract_sha256")
            != contract["content_hash"]
            or state.get("planned_update_counts") != counts
        ):
            raise ValueError("final-consumer resume state differs")
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        rows = list(state["rows"])
        update_ordinal = int(state["optimizer_update_ordinal"])
        start_epoch = int(state["epoch_completed"]) + 1
        _restore_rng_state(state["rng_state"])
    precision_enabled = resolved.type == "cuda"
    for epoch in range(start_epoch, config.maximum_epochs + 1):
        train_loader.dataset.set_epoch(epoch)
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0
        for batch_index, raw in enumerate(train_loader, start=1):
            batch = _move(raw, resolved)
            with module.autocast(
                device_type=resolved.type,
                dtype=module.bfloat16,
                enabled=precision_enabled,
            ):
                loss, _ = final_consumer_forward_and_objective(
                    model=model,
                    consumer_kind=config.consumer_kind,
                    batch=batch,
                    frozen_expert_heads=frozen_expert_heads,
                    frozen_offline_fusion=frozen_offline_fusion,
                    refiner=refiner,
                    zero_based_epoch=epoch - 1,
                )
                scaled = loss / config.gradient_accumulation_steps
            scaled.backward()
            accumulated += 1
            if (
                accumulated == config.gradient_accumulation_steps
                or batch_index == len(train_loader)
            ):
                if accumulated < config.gradient_accumulation_steps:
                    correction = (
                        config.gradient_accumulation_steps / accumulated
                    )
                    for parameter in trainable:
                        if parameter.grad is not None:
                            parameter.grad.mul_(correction)
                update_ordinal += 1
                optimizer.param_groups[0]["lr"] = scheduled_learning_rate(
                    update_ordinal=update_ordinal,
                    total_optimizer_updates=counts[
                        "total_optimizer_updates"
                    ],
                    warmup_updates=counts["warmup_updates"],
                    base_learning_rate=config.learning_rate,
                    minimum_learning_rate=config.minimum_learning_rate,
                )
                norm = module.nn.utils.clip_grad_norm_(
                    trainable, config.gradient_clip
                )
                if not bool(module.isfinite(norm)):
                    raise FloatingPointError(
                        "final-consumer gradient norm is nonfinite"
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated = 0
        validation = evaluate_final_consumer(
            model=model,
            consumer_kind=config.consumer_kind,
            loader=val_stop_loader,
            frozen_expert_heads=frozen_expert_heads,
            frozen_offline_fusion=frozen_offline_fusion,
            device=resolved,
            refiner=refiner,
        )
        row = {
            "epoch": epoch,
            "val_stop": {
                "accuracy": validation["metrics"]["accuracy"],
                "cross_entropy": validation["metrics"]["cross_entropy"],
            },
            "objective": validation["objective"],
            "optimizer_update_ordinal": update_ordinal,
        }
        rows.append(row)
        state = {
            name: value.detach().cpu().clone()
            if hasattr(value, "detach")
            else copy.deepcopy(value)
            for name, value in model.state_dict().items()
        }
        _save_torch(
            candidates / f"epoch_{epoch:03d}.pt",
            {
                "contract": FINAL_CONSUMER_CHECKPOINT_CONTRACT,
                "schema_version": 1,
                "kind": "epoch_candidate",
                "training_contract_sha256": contract["content_hash"],
                "epoch": epoch,
                "model_state_dict": state,
                "model_state_sha256": _state_sha256(state),
            },
        )
        _replace_torch(
            last_path,
            {
                "contract": FINAL_CONSUMER_CHECKPOINT_CONTRACT,
                "schema_version": 1,
                "kind": "resumable_last",
                "training_contract_sha256": contract["content_hash"],
                "epoch_completed": epoch,
                "planned_update_counts": counts,
                "model_state_dict": state,
                "optimizer_state_dict": optimizer.state_dict(),
                "optimizer_update_ordinal": update_ordinal,
                "rows": rows,
                "rng_state": _rng_state(),
            },
        )
    if update_ordinal != counts["total_optimizer_updates"]:
        raise RuntimeError("final-consumer optimizer budget drifted")
    selected = preferred_expert_epoch(rows)
    checkpoint = module.load(
        candidates / f"epoch_{int(selected['epoch']):03d}.pt",
        map_location="cpu",
        weights_only=False,
    )
    _save_torch(
        selected_path,
        {
            "contract": FINAL_CONSUMER_CHECKPOINT_CONTRACT,
            "schema_version": 1,
            "kind": "selected_inference",
            "training_contract_sha256": contract["content_hash"],
            "epoch": int(selected["epoch"]),
            "model_state_dict": checkpoint["model_state_dict"],
            "model_state_sha256": checkpoint["model_state_sha256"],
            "selection_metrics": selected["val_stop"],
        },
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    curves = bind_source(
        with_content_hash(
            {
                "contract": FINAL_CONSUMER_CURVES_CONTRACT,
                "schema_version": 1,
                "training_contract_sha256": contract["content_hash"],
                "rows": rows,
                "selected_epoch": int(selected["epoch"]),
                "epochs_completed": len(rows),
                "fixed_budget_completed": (
                    len(rows) == config.maximum_epochs
                ),
                "stopped_early": False,
                "performance_result_affected_execution": False,
                "planned_update_counts": counts,
            }
        ),
        source_snapshot=source_snapshot,
    )
    write_immutable_json(curves_path, curves)
    registration = bind_source(
        with_content_hash(
            {
                "contract": FINAL_CONSUMER_REGISTRATION_CONTRACT,
                "schema_version": 1,
                "run_id": run_record["run_id"],
                "run_record_sha256": run_record["content_hash"],
                "pipeline_seed": config.seed,
                "consumer_kind": config.consumer_kind,
                "model_variant": config.model_variant,
                "native_dropout_mode": config.native_dropout_mode,
                "training_contract_sha256": contract["content_hash"],
                "checkpoint_sha256": _file_sha256(selected_path),
                "training_curves_sha256": curves["content_hash"],
                "fixed_budget_completed": True,
                "performance_based_termination": False,
            }
        ),
        source_snapshot=source_snapshot,
    )
    write_immutable_json(registration_path, registration)
    return registration


def load_selected_final_consumer_checkpoint(
    *,
    model: Any,
    registration_path: str | Path,
    checkpoint_path: str | Path,
    expected_run_record_sha256: str,
    expected_source: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate and load the val_stop-selected Step-12 weights."""

    registration = load_hashed_json(
        registration_path,
        expected_contract=FINAL_CONSUMER_REGISTRATION_CONTRACT,
    )
    checkpoint_path = Path(checkpoint_path)
    expected_source_record = (
        source_record(expected_source)
        if "source_commit" in expected_source
        else dict(expected_source)
    )
    if (
        registration.get("source") != expected_source_record
        or registration.get("run_record_sha256")
        != require_sha256(
            expected_run_record_sha256,
            name="expected_run_record_sha256",
        )
        or not registration.get("fixed_budget_completed")
        or registration.get("performance_based_termination") is not False
        or not checkpoint_path.is_file()
        or checkpoint_path.is_symlink()
        or _file_sha256(checkpoint_path)
        != registration.get("checkpoint_sha256")
    ):
        raise ValueError("selected final-consumer checkpoint lineage differs")
    checkpoint = _require_torch().load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    state = checkpoint.get("model_state_dict")
    if (
        checkpoint.get("contract") != FINAL_CONSUMER_CHECKPOINT_CONTRACT
        or checkpoint.get("kind") != "selected_inference"
        or checkpoint.get("training_contract_sha256")
        != registration.get("training_contract_sha256")
        or not isinstance(state, Mapping)
        or _state_sha256(state) != checkpoint.get("model_state_sha256")
    ):
        raise ValueError("selected final-consumer checkpoint differs")
    model.load_state_dict(state, strict=True)
    return registration


def publish_final_consumer_dataset(
    *,
    output_dir: str | Path,
    dataset: FinalConsumerDataset,
    parent_hashes: Mapping[str, str],
    source_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        set(parent_hashes) != set(FINAL_DATASET_PARENT_KEYS)
        or dataset.lineage_hashes != dict(sorted(parent_hashes.items()))
    ):
        raise ValueError("final-consumer dataset parent coverage differs")
    root = Path(output_dir)
    path = root / "final_consumer_dataset.pt"
    stream = io.BytesIO()
    _require_torch().save(
        {
            "contract": FINAL_CONSUMER_DATASET_CONTRACT,
            "schema_version": 1,
            "dataset": dataset,
        },
        stream,
    )
    publication = write_immutable_bytes(path, stream.getvalue())
    manifest = bind_source(
        with_content_hash(
            {
                "contract": FINAL_CONSUMER_DATASET_CONTRACT,
                "schema_version": 1,
                "split": dataset.split,
                "event_count": len(dataset),
                "replica_ids": list(dataset.replica_set),
                "identity_order_sha256": canonical_sha256(
                    list(dataset.identities)
                ),
                "dataset_filename": path.name,
                "dataset_sha256": publication["file_sha256"],
                "parent_hashes": dict(sorted(parent_hashes.items())),
                "token_coordinates": {
                    "predicted_banks": (
                        "original_offline_token_coordinates_after_inverse_"
                        "normalization"
                    ),
                    "oracle_banks": "original_offline_token_coordinates",
                    "target_normalized_banks": (
                        "model_train_normalized_predictor_coordinates"
                    ),
                    "calibrated_log_variance": (
                        "model_train_normalized_predictor_coordinates"
                    ),
                },
                "offline_constituent_matches_present": False,
                "performance_based_termination": False,
            }
        ),
        source_snapshot=source_snapshot,
    )
    write_immutable_json(root / "final_consumer_dataset.json", manifest)
    return manifest


def load_final_consumer_dataset(
    manifest_path: str | Path,
    *,
    expected_split: str,
    expected_source: Mapping[str, Any],
) -> tuple[dict[str, Any], FinalConsumerDataset]:
    manifest = load_hashed_json(
        manifest_path, expected_contract=FINAL_CONSUMER_DATASET_CONTRACT
    )
    path = Path(manifest_path).parent / manifest["dataset_filename"]
    expected_source_record = (
        source_record(expected_source)
        if "source_commit" in expected_source
        else dict(expected_source)
    )
    if (
        manifest.get("split") != expected_split
        or manifest.get("source") != expected_source_record
        or not path.is_file()
        or path.is_symlink()
        or _file_sha256(path) != manifest["dataset_sha256"]
        or manifest.get("token_coordinates")
        != {
            "predicted_banks": (
                "original_offline_token_coordinates_after_inverse_"
                "normalization"
            ),
            "oracle_banks": "original_offline_token_coordinates",
            "target_normalized_banks": (
                "model_train_normalized_predictor_coordinates"
            ),
            "calibrated_log_variance": (
                "model_train_normalized_predictor_coordinates"
            ),
        }
    ):
        raise ValueError("final-consumer dataset lineage differs")
    payload = _require_torch().load(
        path, map_location="cpu", weights_only=False
    )
    dataset = payload.get("dataset")
    if (
        payload.get("contract") != FINAL_CONSUMER_DATASET_CONTRACT
        or not isinstance(dataset, FinalConsumerDataset)
        or dataset.split != expected_split
        or len(dataset) != manifest["event_count"]
        or dataset.lineage_hashes != manifest["parent_hashes"]
        or canonical_sha256(list(dataset.identities))
        != manifest["identity_order_sha256"]
    ):
        raise ValueError("final-consumer dataset semantics differ")
    return manifest, dataset


def publish_final_consumer_template(
    *,
    output_dir: str | Path,
    model: Any,
    frozen_expert_heads: Mapping[str, Any],
    frozen_offline_fusion: Any,
    refiner: NativeConditionedTokenRefiner | None,
    run_record_sha256: str,
    component_parent_hashes: Mapping[str, str],
    source_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    if set(frozen_expert_heads) != set(EXPERT_ORDER):
        raise ValueError("final-consumer template head coverage differs")
    required = {
        "joint_prediction_checkpoint",
        "native_HLT_checkpoint_bundle",
        "uncertainty_calibration",
        "frozen_offline_fusion",
        "frozen_offline_expert_heads",
    }
    if refiner is not None:
        required.add("selected_token_refiner")
    if set(component_parent_hashes) != required:
        raise ValueError("final-consumer template parent coverage differs")
    root = Path(output_dir)
    path = root / "final_consumer_template.pt"
    stream = io.BytesIO()
    _require_torch().save(
        {
            "contract": FINAL_CONSUMER_TEMPLATE_CONTRACT,
            "schema_version": 1,
            "model": model,
            "frozen_expert_heads": dict(frozen_expert_heads),
            "frozen_offline_fusion": frozen_offline_fusion,
            "refiner": refiner,
        },
        stream,
    )
    publication = write_immutable_bytes(path, stream.getvalue())
    manifest = bind_source(
        with_content_hash(
            {
                "contract": FINAL_CONSUMER_TEMPLATE_CONTRACT,
                "schema_version": 1,
                "template_filename": path.name,
                "template_sha256": publication["file_sha256"],
                "run_record_sha256": require_sha256(
                    run_record_sha256, name="run_record_sha256"
                ),
                "component_parent_hashes": {
                    name: require_sha256(
                        value, name=f"component_parent_hashes.{name}"
                    )
                    for name, value in sorted(
                        component_parent_hashes.items()
                    )
                },
                "model_type": type(model).__name__,
                "refiner_present": refiner is not None,
                "offline_training_components_frozen": True,
                "performance_based_termination": False,
            }
        ),
        source_snapshot=source_snapshot,
    )
    write_immutable_json(root / "final_consumer_template.json", manifest)
    return manifest


def load_final_consumer_template(
    manifest_path: str | Path,
    *,
    expected_run_record_sha256: str,
    expected_source: Mapping[str, Any],
) -> tuple[dict[str, Any], Any, dict[str, Any], Any, Any]:
    manifest = load_hashed_json(
        manifest_path, expected_contract=FINAL_CONSUMER_TEMPLATE_CONTRACT
    )
    path = Path(manifest_path).parent / manifest["template_filename"]
    expected_source_record = (
        source_record(expected_source)
        if "source_commit" in expected_source
        else dict(expected_source)
    )
    if (
        manifest.get("run_record_sha256")
        != require_sha256(
            expected_run_record_sha256,
            name="expected_run_record_sha256",
        )
        or manifest.get("source") != expected_source_record
        or not path.is_file()
        or path.is_symlink()
        or _file_sha256(path) != manifest["template_sha256"]
    ):
        raise ValueError("final-consumer template lineage differs")
    payload = _require_torch().load(
        path, map_location="cpu", weights_only=False
    )
    heads = payload.get("frozen_expert_heads")
    if (
        payload.get("contract") != FINAL_CONSUMER_TEMPLATE_CONTRACT
        or set(heads or {}) != set(EXPERT_ORDER)
        or (payload.get("refiner") is not None)
        != manifest["refiner_present"]
        or type(payload.get("model")).__name__ != manifest["model_type"]
    ):
        raise ValueError("final-consumer template semantics differ")
    return (
        manifest,
        payload["model"],
        heads,
        payload["frozen_offline_fusion"],
        payload["refiner"],
    )


def publish_final_consumer_inference(
    *,
    output_dir: str | Path,
    evaluation: Mapping[str, Any],
    split: str,
    run_id: str,
    registration_sha256: str,
    identity_manifest_sha256: str,
    HLT_input_cache_sha256: str,
    source_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    if split not in {"val_stop", "val_design"}:
        raise ValueError("final-consumer inference split differs")
    logits = np.asarray(evaluation["logits"], dtype=np.float32)
    identities = tuple(str(value) for value in evaluation["identities"])
    replicas = np.asarray(evaluation["replica_ids"], dtype=np.int64)
    hashes = tuple(str(value) for value in evaluation["degraded_view_hashes"])
    validate_common_view_metadata(
        identities=identities,
        replica_ids=replicas,
        degraded_view_hashes=hashes,
    )
    if logits.shape != (len(identities), 10) or not np.isfinite(logits).all():
        raise ValueError("final-consumer inference logits differ")
    arrays = {
        "identities": np.asarray(identities, dtype=np.str_),
        "replica_ids": replicas,
        "degraded_view_hashes": np.asarray(hashes, dtype=np.str_),
        "logits": logits,
    }
    for name in ("frozen_path_logits", "residual_path_logits"):
        value = evaluation.get(name)
        if value is not None:
            arrays[name] = np.asarray(value, dtype=np.float32)
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    root = Path(output_dir)
    publication = write_immutable_bytes(
        root / "final_consumer_predictions.npz", stream.getvalue()
    )
    return bind_source(
        with_content_hash(
            {
                "contract": FINAL_CONSUMER_INFERENCE_CONTRACT,
                "schema_version": 1,
                "run_id": str(run_id),
                "split": split,
                "event_count": len(identities),
                "npz_filename": "final_consumer_predictions.npz",
                "npz_sha256": publication["file_sha256"],
                "identity_order_sha256": canonical_sha256(
                    list(identities)
                ),
                "replica_order_sha256": canonical_sha256(
                    replicas.tolist()
                ),
                "degraded_view_order_sha256": canonical_sha256(
                    list(hashes)
                ),
                "token_RMSE_before": evaluation["token_RMSE_before"],
                "token_RMSE_after": evaluation["token_RMSE_after"],
                "parents": {
                    "registration": require_sha256(
                        registration_sha256,
                        name="registration_sha256",
                    ),
                    "identity_manifest": require_sha256(
                        identity_manifest_sha256,
                        name="identity_manifest_sha256",
                    ),
                    "HLT_input_cache": require_sha256(
                        HLT_input_cache_sha256,
                        name="HLT_input_cache_sha256",
                    ),
                },
                "oracle_targets_present": False,
                "offline_inputs_present": False,
                "complete_coverage": True,
            }
        ),
        source_snapshot=source_snapshot,
    )


__all__ = [
    "CONSUMER_KINDS",
    "FINAL_CONSUMER_CHECKPOINT_CONTRACT",
    "FINAL_CONSUMER_CURVES_CONTRACT",
    "FINAL_CONSUMER_DATASET_CONTRACT",
    "FINAL_CONSUMER_INFERENCE_CONTRACT",
    "FINAL_CONSUMER_REGISTRATION_CONTRACT",
    "FINAL_CONSUMER_TEMPLATE_CONTRACT",
    "FINAL_CONSUMER_TRAINING_CONTRACT",
    "FINAL_DATASET_PARENT_KEYS",
    "FinalConsumerDataset",
    "FinalConsumerTrainingConfig",
    "collate_final_consumer_batch",
    "evaluate_final_consumer",
    "evaluate_final_consumer_bypass_controls",
    "final_consumer_forward_and_objective",
    "load_final_consumer_dataset",
    "load_selected_final_consumer_checkpoint",
    "load_final_consumer_template",
    "make_final_consumer_loader",
    "publish_final_consumer_dataset",
    "publish_final_consumer_inference",
    "publish_final_consumer_template",
    "train_final_consumer",
]
