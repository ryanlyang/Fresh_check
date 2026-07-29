"""Fixed-budget, resumable training for individual RETB token predictors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import hashlib
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import (
    load_hashed_json,
    require_sha256,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from .determinism import optimizer_update_counts, scheduled_learning_rate
from .expert_training import DeterministicExpertSampler, preferred_expert_epoch
from .predictor_losses import (
    DeterministicGradNorm,
    FIXED_WEIGHTS,
    inverse_normalize_tokens,
    predictor_objective,
    token_tail_diagnostics,
)
from .predictors import (
    ARCHITECTURES,
    CONTEXTS,
    NORMALIZATION_MODES,
    RELATION_PARTICLE_ORDER,
    UNCERTAINTY_HEADS,
)
from .registry import EXPERT_ORDER

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


PREDICTOR_TRAINING_CONTRACT = "retb_predictor_training_v1"
PREDICTOR_REGISTRATION_CONTRACT = "retb_predictor_registration_v1"
PREDICTOR_CURVES_CONTRACT = "retb_predictor_training_curves_v1"
PREDICTOR_CHECKPOINT_CONTRACT = "retb_predictor_checkpoint_v1"


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is required for RETB predictor training")
    return torch


class PredictorDataset(torch.utils.data.Dataset if torch is not None else object):
    """Identity-bound HLT evidence and offline targets with no constituent match."""

    def __init__(
        self,
        *,
        identities: Sequence[str],
        labels: np.ndarray,
        hlt_token_banks: Mapping[str, np.ndarray],
        unbiased_particle_states: np.ndarray,
        particle_mask: np.ndarray,
        target_tokens: np.ndarray,
        target_expert_logits: np.ndarray,
        target_hybrid_logits: np.ndarray,
        other_oracle_banks: Mapping[str, np.ndarray],
        target_expert_id: str,
        token_mean: np.ndarray,
        token_standard_deviation: np.ndarray,
        normalization_mode: str,
        split: str,
        lineage_hashes: Mapping[str, str],
        relation_particle_states: Mapping[str, np.ndarray] | None = None,
        relation_particle_masks: Mapping[str, np.ndarray] | None = None,
    ) -> None:
        _require_torch()
        ids = tuple(str(value) for value in identities)
        truth = np.asarray(labels, dtype=np.int64)
        if (
            not ids
            or len(ids) != len(set(ids))
            or truth.shape != (len(ids),)
            or bool(((truth < 0) | (truth >= 10)).any())
            or target_expert_id not in EXPERT_ORDER
            or normalization_mode not in NORMALIZATION_MODES
            or split not in {"model_train", "val_stop", "val_design"}
            or set(hlt_token_banks) != set(EXPERT_ORDER)
            or set(other_oracle_banks)
            != set(EXPERT_ORDER) - {target_expert_id}
        ):
            raise ValueError("predictor dataset identity/coverage differs")
        self.identities, self.labels = ids, truth
        self.target_expert_id = target_expert_id
        self.split = split
        self.normalization_mode = normalization_mode
        self.hlt_token_banks = {
            expert: np.asarray(hlt_token_banks[expert], dtype=np.float32)
            for expert in EXPERT_ORDER
        }
        self.unbiased_particle_states = np.asarray(
            unbiased_particle_states, dtype=np.float32
        )
        self.particle_mask = np.asarray(particle_mask, dtype=bool)
        self.target_tokens_original = np.asarray(
            target_tokens, dtype=np.float32
        )
        self.token_mean = np.asarray(token_mean, dtype=np.float32)
        self.token_standard_deviation = np.asarray(
            token_standard_deviation, dtype=np.float32
        )
        if (
            self.target_tokens_original.ndim != 3
            or self.token_mean.shape != self.target_tokens_original.shape[1:]
            or self.token_standard_deviation.shape
            != self.target_tokens_original.shape[1:]
            or not np.isfinite(self.token_mean).all()
            or not np.isfinite(self.token_standard_deviation).all()
            or bool((self.token_standard_deviation < 0).any())
        ):
            raise ValueError("predictor target normalizer differs")
        normalized = (
            (self.target_tokens_original - self.token_mean)
            / np.maximum(self.token_standard_deviation, 1.0e-4)
        ).astype(np.float32)
        self.target_tokens_unclipped = normalized.copy()
        if normalization_mode == "N_CLIP16":
            normalized = np.clip(normalized, -16.0, 16.0)
        elif normalization_mode == "N_CLIP8":
            normalized = np.clip(normalized, -8.0, 8.0)
        self.target_tokens = normalized
        self.target_expert_logits = np.asarray(
            target_expert_logits, dtype=np.float32
        )
        self.target_hybrid_logits = np.asarray(
            target_hybrid_logits, dtype=np.float32
        )
        self.other_oracle_banks = {
            expert: np.asarray(value, dtype=np.float32)
            for expert, value in other_oracle_banks.items()
        }
        self.relation_particle_states = (
            None
            if relation_particle_states is None
            else {
                name: np.asarray(value, dtype=np.float32)
                for name, value in relation_particle_states.items()
            }
        )
        self.relation_particle_masks = (
            None
            if relation_particle_masks is None
            else {
                name: np.asarray(value, dtype=bool)
                for name, value in relation_particle_masks.items()
            }
        )
        if (self.relation_particle_states is None) != (
            self.relation_particle_masks is None
        ):
            raise ValueError("predictor relation-particle inputs are partial")
        if self.relation_particle_states is not None and (
            set(self.relation_particle_states) != set(RELATION_PARTICLE_ORDER)
            or set(self.relation_particle_masks)
            != set(RELATION_PARTICLE_ORDER)
        ):
            raise ValueError("predictor relation-particle coverage differs")
        self.lineage_hashes = {
            name: require_sha256(value, name=f"lineage_hashes.{name}")
            for name, value in sorted(lineage_hashes.items())
        }
        count = len(ids)
        arrays = [
            *self.hlt_token_banks.values(),
            self.unbiased_particle_states,
            self.particle_mask,
            self.target_tokens,
            self.target_tokens_unclipped,
            self.target_tokens_original,
            self.target_expert_logits,
            self.target_hybrid_logits,
            *self.other_oracle_banks.values(),
        ]
        if self.relation_particle_states is not None:
            arrays.extend(self.relation_particle_states.values())
            arrays.extend(self.relation_particle_masks.values())
        if (
            any(len(value) != count for value in arrays)
            or self.particle_mask.shape
            != self.unbiased_particle_states.shape[:2]
            or self.target_expert_logits.shape != (count, 10)
            or self.target_hybrid_logits.shape != (count, 10)
            or any(
                not np.isfinite(value).all()
                for value in arrays
                if value.dtype != np.bool_
            )
        ):
            raise ValueError("predictor dataset arrays differ")
        if self.relation_particle_states is not None:
            for name in RELATION_PARTICLE_ORDER:
                if self.relation_particle_masks[name].shape != (
                    self.relation_particle_states[name].shape[:2]
                ):
                    raise ValueError("predictor relation-particle mask differs")

    def __len__(self) -> int:
        return len(self.identities)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "identity": self.identities[index],
            "label": self.labels[index],
            "hlt_token_banks": {
                expert: values[index]
                for expert, values in self.hlt_token_banks.items()
            },
            "unbiased_particle_states": self.unbiased_particle_states[index],
            "particle_mask": self.particle_mask[index],
            "relation_particle_states": (
                None
                if self.relation_particle_states is None
                else {
                    name: values[index]
                    for name, values in self.relation_particle_states.items()
                }
            ),
            "relation_particle_masks": (
                None
                if self.relation_particle_masks is None
                else {
                    name: values[index]
                    for name, values in self.relation_particle_masks.items()
                }
            ),
            "target_tokens": self.target_tokens[index],
            "target_tokens_unclipped": self.target_tokens_unclipped[index],
            "target_tokens_original": self.target_tokens_original[index],
            "target_expert_logits": self.target_expert_logits[index],
            "target_hybrid_logits": self.target_hybrid_logits[index],
            "other_oracle_banks": {
                expert: values[index]
                for expert, values in self.other_oracle_banks.items()
            },
        }


def collate_predictor_batch(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    module = _require_torch()
    if not rows:
        raise ValueError("cannot collate an empty predictor batch")
    relation_present = rows[0]["relation_particle_states"] is not None
    if any(
        (row["relation_particle_states"] is not None) != relation_present
        for row in rows
    ):
        raise ValueError("predictor relation-particle presence drifted")
    return {
        "identities": [row["identity"] for row in rows],
        "labels": module.as_tensor([row["label"] for row in rows]).long(),
        "hlt_token_banks": {
            expert: module.from_numpy(
                np.stack([row["hlt_token_banks"][expert] for row in rows])
            ).float()
            for expert in EXPERT_ORDER
        },
        "unbiased_particle_states": module.from_numpy(
            np.stack([row["unbiased_particle_states"] for row in rows])
        ).float(),
        "particle_mask": module.from_numpy(
            np.stack([row["particle_mask"] for row in rows])
        ).bool(),
        "relation_particle_states": (
            None
            if not relation_present
            else {
                name: module.from_numpy(
                    np.stack(
                        [row["relation_particle_states"][name] for row in rows]
                    )
                ).float()
                for name in RELATION_PARTICLE_ORDER
            }
        ),
        "relation_particle_masks": (
            None
            if not relation_present
            else {
                name: module.from_numpy(
                    np.stack(
                        [row["relation_particle_masks"][name] for row in rows]
                    )
                ).bool()
                for name in RELATION_PARTICLE_ORDER
            }
        ),
        "target_tokens": module.from_numpy(
            np.stack([row["target_tokens"] for row in rows])
        ).float(),
        "target_tokens_unclipped": module.from_numpy(
            np.stack([row["target_tokens_unclipped"] for row in rows])
        ).float(),
        "target_tokens_original": module.from_numpy(
            np.stack([row["target_tokens_original"] for row in rows])
        ).float(),
        "target_expert_logits": module.from_numpy(
            np.stack([row["target_expert_logits"] for row in rows])
        ).float(),
        "target_hybrid_logits": module.from_numpy(
            np.stack([row["target_hybrid_logits"] for row in rows])
        ).float(),
        "other_oracle_banks": {
            expert: module.from_numpy(
                np.stack([row["other_oracle_banks"][expert] for row in rows])
            ).float()
            for expert in rows[0]["other_oracle_banks"]
        },
    }


def make_predictor_loader(
    dataset: PredictorDataset,
    *,
    batch_size: int,
    seed: int,
    training: bool,
) -> Any:
    module = _require_torch()
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
        collate_fn=collate_predictor_batch,
    )


@dataclass(frozen=True)
class PredictorTrainingConfig:
    seed: int
    architecture: str
    context: str
    objective_id: str = "W_CANONICAL"
    uncertainty_head: str = "U_SLOT"
    normalization_mode: str = "N_UNCLIPPED"
    learning_rate: float = 5.0e-4
    dropout: float = 0.1
    maximum_epochs: int = 40
    microbatch_size: int = 256
    gradient_accumulation_steps: int = 1
    effective_batch_size: int = 256
    minimum_learning_rate: float = 1.0e-5
    weight_decay: float = 1.0e-4
    gradient_clip: float = 1.0
    campaign_profile: str = "production"

    def validate(self) -> None:
        if (
            int(self.seed) not in {101, 202, 303}
            or self.architecture not in ARCHITECTURES
            or self.context not in CONTEXTS
            or self.objective_id
            not in {*FIXED_WEIGHTS, "W_GRADNORM"}
            or self.uncertainty_head not in UNCERTAINTY_HEADS
            or self.normalization_mode not in NORMALIZATION_MODES
            or float(self.learning_rate) not in {2.0e-4, 5.0e-4, 1.0e-3}
            or float(self.dropout) not in {0.0, 0.1}
            or int(self.maximum_epochs) <= 0
            or min(
                int(self.microbatch_size),
                int(self.gradient_accumulation_steps),
                int(self.effective_batch_size),
            )
            <= 0
            or int(self.microbatch_size)
            * int(self.gradient_accumulation_steps)
            != int(self.effective_batch_size)
            or float(self.minimum_learning_rate) != 1.0e-5
            or float(self.weight_decay) != 1.0e-4
            or float(self.gradient_clip) != 1.0
            or self.campaign_profile not in {"production", "miniature_test"}
            or (
                self.campaign_profile == "production"
                and (
                    int(self.maximum_epochs) != 40
                    or int(self.effective_batch_size) != 256
                )
            )
        ):
            raise ValueError("predictor training configuration differs")

    def artifact(
        self,
        *,
        global_determinism_sha256: str,
        step9_bundle_sha256: str,
        run_record_sha256: str,
        lineage_hashes: Mapping[str, str],
    ) -> dict[str, Any]:
        self.validate()
        return with_content_hash(
            {
                "contract": PREDICTOR_TRAINING_CONTRACT,
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
                "precision": (
                    "GH200_BF16"
                    if self.campaign_profile == "production"
                    else "BF16_on_CUDA_FP32_elsewhere"
                ),
                "num_workers": 0,
                "epoch_selection": (
                    "val_stop_accuracy_window_0.0001_then_cross_entropy_"
                    "then_earliest_epoch"
                ),
                "fixed_budget": True,
                "early_stopping": False,
                "performance_based_termination": False,
                "gradnorm_validation_metrics_consumed": False,
                "parents": {
                    "global_determinism": require_sha256(
                        global_determinism_sha256,
                        name="global_determinism_sha256",
                    ),
                    "step9_bundle": require_sha256(
                        step9_bundle_sha256, name="step9_bundle_sha256"
                    ),
                    "run_record": require_sha256(
                        run_record_sha256, name="run_record_sha256"
                    ),
                    **{
                        f"lineage.{name}": require_sha256(
                            value, name=f"lineage_hashes.{name}"
                        )
                        for name, value in sorted(lineage_hashes.items())
                    },
                },
            }
        )


def _move(value: Any, device: Any) -> Any:
    if isinstance(value, Mapping):
        return {name: _move(item, device) for name, item in value.items()}
    if hasattr(value, "to"):
        return value.to(device)
    return value


def _predictor_forward(model: Any, batch: Mapping[str, Any]) -> dict[str, Any]:
    target = model.target_expert_id
    return model(
        corresponding_hlt_tokens=batch["hlt_token_banks"][target],
        hlt_token_banks=batch["hlt_token_banks"],
        unbiased_particle_states=batch["unbiased_particle_states"],
        particle_mask=batch["particle_mask"],
        relation_particle_states=batch["relation_particle_states"],
        relation_particle_masks=batch["relation_particle_masks"],
    )


def _hybrid_logits(
    *,
    frozen_fusion: Any,
    predicted_original_tokens: Any,
    other_oracle_banks: Mapping[str, Any],
    target_expert_id: str,
) -> Any:
    banks = dict(other_oracle_banks)
    banks[target_expert_id] = predicted_original_tokens
    try:
        return frozen_fusion(token_banks=banks)
    except TypeError:
        return frozen_fusion(banks)


def _batch_losses(
    *,
    model: Any,
    batch: Mapping[str, Any],
    frozen_expert_head: Any,
    frozen_fusion: Any,
    token_mean: Any,
    token_standard_deviation: Any,
    objective_id: str,
    normalization_mode: str,
    gradnorm_weights: Mapping[str, float] | None,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    output = _predictor_forward(model, batch)
    predicted_normalized = output["predicted_tokens"]
    if normalization_mode == "N_CLIP16":
        predicted_normalized = predicted_normalized.clamp(-16.0, 16.0)
    elif normalization_mode == "N_CLIP8":
        predicted_normalized = predicted_normalized.clamp(-8.0, 8.0)
    elif normalization_mode != "N_UNCLIPPED":
        raise ValueError("predictor normalization mode is unregistered")
    predicted_original = inverse_normalize_tokens(
        predicted_normalized,
        mean=token_mean,
        standard_deviation=token_standard_deviation,
    )
    predicted_expert_logits = frozen_expert_head(predicted_original)
    predicted_hybrid_logits = _hybrid_logits(
        frozen_fusion=frozen_fusion,
        predicted_original_tokens=predicted_original,
        other_oracle_banks=batch["other_oracle_banks"],
        target_expert_id=model.target_expert_id,
    )
    total, details = predictor_objective(
        weight_id=objective_id,
        uncertainty_head=model.uncertainty_head,
        predicted_tokens=predicted_normalized,
        target_tokens=batch["target_tokens"],
        log_variance=output["log_variance"],
        predicted_expert_logits=predicted_expert_logits,
        target_expert_logits=batch["target_expert_logits"],
        predicted_hybrid_logits=predicted_hybrid_logits,
        target_hybrid_logits=batch["target_hybrid_logits"],
        labels=batch["labels"],
        gradnorm_weights=gradnorm_weights,
    )
    return total, details, {
        **output,
        "raw_predicted_tokens": output["predicted_tokens"],
        "predicted_tokens": predicted_normalized,
        "predicted_original_tokens": predicted_original,
        "predicted_expert_logits": predicted_expert_logits,
        "predicted_hybrid_logits": predicted_hybrid_logits,
    }


def evaluate_predictor(
    *,
    model: Any,
    loader: Any,
    frozen_expert_head: Any,
    frozen_fusion: Any,
    token_mean: Any,
    token_standard_deviation: Any,
    objective_id: str,
    normalization_mode: str,
    device: Any,
    gradnorm_weights: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    module = _require_torch()
    model.eval()
    (
        losses,
        logits,
        expert_logits,
        labels,
        identities,
        tokens,
        original_tokens,
        raw_tokens,
        targets,
        unclipped_targets,
        log_variances,
    ) = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    with module.no_grad():
        for raw in loader:
            batch = _move(raw, device)
            total, _, outputs = _batch_losses(
                model=model,
                batch=batch,
                frozen_expert_head=frozen_expert_head,
                frozen_fusion=frozen_fusion,
                token_mean=token_mean,
                token_standard_deviation=token_standard_deviation,
                objective_id=objective_id,
                normalization_mode=normalization_mode,
                gradnorm_weights=gradnorm_weights,
            )
            losses.append(float(total.detach().cpu()) * len(raw["identities"]))
            logits.append(
                outputs["predicted_hybrid_logits"].float().cpu().numpy()
            )
            expert_logits.append(
                outputs["predicted_expert_logits"].float().cpu().numpy()
            )
            labels.append(raw["labels"].numpy())
            identities.extend(raw["identities"])
            tokens.append(outputs["predicted_tokens"].float().cpu().numpy())
            original_tokens.append(
                outputs["predicted_original_tokens"].float().cpu().numpy()
            )
            raw_tokens.append(
                outputs["raw_predicted_tokens"].float().cpu().numpy()
            )
            targets.append(raw["target_tokens"].numpy())
            unclipped_targets.append(raw["target_tokens_unclipped"].numpy())
            log_variances.append(
                outputs["log_variance"].float().cpu().numpy()
            )
    values = np.concatenate(logits)
    truth = np.concatenate(labels)
    effective_prediction = np.concatenate(tokens)
    raw_prediction = np.concatenate(raw_tokens)
    effective_target = np.concatenate(targets)
    raw_target = np.concatenate(unclipped_targets)
    shifted = values - values.max(axis=1, keepdims=True)
    cross_entropy = float(
        (
            np.log(np.exp(shifted).sum(axis=1))
            - shifted[np.arange(len(truth)), truth]
        ).mean(dtype=np.float64)
    )
    return {
        "metrics": {
            "accuracy": float((values.argmax(axis=1) == truth).mean()),
            "cross_entropy": cross_entropy,
            "objective": float(sum(losses) / len(truth)),
            "normalized_token_rmse": float(
                np.sqrt(
                    np.mean(
                        (effective_prediction - effective_target) ** 2
                    )
                )
            ),
            "normalization_tail_diagnostics": {
                "prediction_before_control_clip": token_tail_diagnostics(
                    raw_prediction, labels=truth
                ),
                "target_before_control_clip": token_tail_diagnostics(
                    raw_target, labels=truth
                ),
                "prediction_after_control_clip": token_tail_diagnostics(
                    effective_prediction, labels=truth
                ),
                "target_after_control_clip": token_tail_diagnostics(
                    effective_target, labels=truth
                ),
                "prediction_clipped_element_count": int(
                    np.count_nonzero(raw_prediction != effective_prediction)
                ),
                "target_clipped_element_count": int(
                    np.count_nonzero(raw_target != effective_target)
                ),
            },
        },
        "identities": identities,
        "labels": truth,
        "hybrid_logits": values.astype(np.float32),
        "expert_logits": np.concatenate(expert_logits).astype(np.float32),
        "predicted_tokens": effective_prediction.astype(np.float32),
        "predicted_original_tokens": np.concatenate(original_tokens).astype(
            np.float32
        ),
        "log_variance": np.concatenate(log_variances).astype(np.float32),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_sha256(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(b"\0")
        if hasattr(value, "detach"):
            array = value.detach().cpu().contiguous().numpy()
            digest.update(str(array.dtype).encode())
            digest.update(str(tuple(array.shape)).encode())
            digest.update(array.tobytes())
        else:
            digest.update(repr(value).encode())
    return digest.hexdigest()


def _atomic_torch_save(payload: Any, path: Path) -> None:
    module = _require_torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        module.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    module = _require_torch()
    module.manual_seed(int(seed))
    if module.cuda.is_available():
        module.cuda.manual_seed_all(int(seed))


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


def train_predictor(
    *,
    model: Any,
    train_loader: Any,
    val_stop_loader: Any,
    frozen_expert_head: Any,
    frozen_fusion: Any,
    token_mean: Any,
    token_standard_deviation: Any,
    output_dir: str | Path,
    run_record: Mapping[str, Any],
    step9_bundle_sha256: str,
    global_determinism_sha256: str,
    lineage_hashes: Mapping[str, str],
    config: PredictorTrainingConfig,
    device: Any = "cpu",
    resume: bool = True,
) -> dict[str, Any]:
    module = _require_torch()
    config.validate()
    validate_content_hash(run_record)
    if (
        run_record.get("architecture") != config.architecture
        or run_record.get("context") != config.context
        or run_record.get("objective_id") != config.objective_id
        or run_record.get("uncertainty_head") != config.uncertainty_head
        or run_record.get("normalization_mode") != config.normalization_mode
        or int(run_record.get("pipeline_seed", -1)) != config.seed
        or float(run_record.get("learning_rate", -1.0))
        != float(config.learning_rate)
        or float(run_record.get("dropout", -1.0)) != float(config.dropout)
        or model.architecture != config.architecture
        or model.context != config.context
        or model.uncertainty_head != config.uncertainty_head
        or model.target_expert_id != run_record.get("expert_id")
        or int(model.token_count) != int(run_record.get("token_count", -1))
        or int(model.token_dimension)
        != int(run_record.get("token_dimension", -1))
    ):
        raise ValueError("predictor run/configuration lineage differs")
    if train_loader.dataset.split != "model_train":
        raise ValueError("predictor training loader must be model_train")
    if val_stop_loader.dataset.split != "val_stop":
        raise ValueError("predictor checkpoint selection must be val_stop")
    parents = {
        name: require_sha256(value, name=f"lineage_hashes.{name}")
        for name, value in sorted(lineage_hashes.items())
    }
    required_parents = {
        "model_train_target_cache",
        "val_stop_target_cache",
        "target_normalizer",
        "slot_queries",
        "offline_target_checkpoint",
        "offline_fusion",
        "native_hlt_expert",
        "model_train_hlt_evidence_cache",
        "val_stop_hlt_evidence_cache",
        "model_train_identity_manifest",
        "val_stop_identity_manifest",
    }
    if not required_parents.issubset(parents):
        raise ValueError("predictor training lacks required lineage parents")
    contract = config.artifact(
        global_determinism_sha256=global_determinism_sha256,
        step9_bundle_sha256=step9_bundle_sha256,
        run_record_sha256=run_record["content_hash"],
        lineage_hashes=parents,
    )
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    selected_path = root / "best_model_val.pt"
    last_path = root / "last_state.pt"
    curves_path = root / "training_curves.json"
    registration_path = root / "registration.json"
    candidate_root = root / "epoch_candidates"
    candidate_root.mkdir(parents=True, exist_ok=True)
    if (
        selected_path.exists()
        and curves_path.exists()
        and registration_path.exists()
    ):
        retained = load_hashed_json(
            curves_path, expected_contract=PREDICTOR_CURVES_CONTRACT
        )
        registration = load_hashed_json(
            registration_path,
            expected_contract=PREDICTOR_REGISTRATION_CONTRACT,
        )
        if (
            retained["training_contract_sha256"] != contract["content_hash"]
            or retained["run_id"] != run_record["run_id"]
            or not retained["fixed_budget_completed"]
        ):
            raise ValueError("reusable predictor result lineage differs")
        checkpoint = module.load(
            selected_path, map_location="cpu", weights_only=False
        )
        if (
            checkpoint.get("contract") != PREDICTOR_CHECKPOINT_CONTRACT
            or checkpoint["training_contract_sha256"]
            != contract["content_hash"]
            or checkpoint["run_id"] != run_record["run_id"]
            or registration["training_contract_sha256"]
            != contract["content_hash"]
            or registration["checkpoint_sha256"]
            != _file_sha256(selected_path)
            or registration["training_curves_sha256"]
            != retained["content_hash"]
        ):
            raise ValueError("reusable predictor checkpoint differs")
        return registration
    _set_seed(config.seed)
    resolved_device = module.device(device)
    if config.campaign_profile == "production" and (
        resolved_device.type != "cuda"
        or not module.cuda.is_bf16_supported()
        or "GH200" not in module.cuda.get_device_name(
            resolved_device
        ).upper()
    ):
        raise RuntimeError("production predictor training requires GH200 BF16")
    model.to(resolved_device)
    frozen_expert_head.to(resolved_device).eval()
    frozen_fusion.to(resolved_device).eval()
    for frozen in (frozen_expert_head, frozen_fusion):
        for parameter in frozen.parameters():
            parameter.requires_grad_(False)
    mean = module.as_tensor(
        token_mean, dtype=module.float32, device=resolved_device
    )
    std = module.as_tensor(
        token_standard_deviation,
        dtype=module.float32,
        device=resolved_device,
    )
    for dataset in (train_loader.dataset, val_stop_loader.dataset):
        if (
            not np.array_equal(
                np.asarray(token_mean, dtype=np.float32), dataset.token_mean
            )
            or not np.array_equal(
                np.asarray(token_standard_deviation, dtype=np.float32),
                dataset.token_standard_deviation,
            )
            or dataset.normalization_mode != config.normalization_mode
            or dataset.target_expert_id != model.target_expert_id
        ):
            raise ValueError("predictor dataset normalizer/expert differs")
    counts = optimizer_update_counts(
        training_event_count=len(train_loader.dataset),
        maximum_epochs=config.maximum_epochs,
        microbatch_size=config.microbatch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    optimizer = module.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=config.weight_decay,
    )
    balancer = (
        DeterministicGradNorm()
        if config.objective_id == "W_GRADNORM"
        else None
    )
    rows: list[dict[str, Any]] = []
    update_ordinal = 0
    start_epoch = 1
    if resume and last_path.exists():
        state = module.load(last_path, map_location="cpu", weights_only=False)
        if (
            state.get("contract") != PREDICTOR_CHECKPOINT_CONTRACT
            or state.get("kind") != "resumable_last"
            or state["training_contract_sha256"] != contract["content_hash"]
            or state["run_id"] != run_record["run_id"]
            or state["planned_update_counts"] != counts
        ):
            raise ValueError("predictor resume state differs")
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        rows = list(state["rows"])
        update_ordinal = int(state["optimizer_update_ordinal"])
        start_epoch = int(state["epoch_completed"]) + 1
        if balancer is not None:
            balancer.load_state_dict(state["gradnorm_state"])
        _restore_rng_state(state["rng_state"])
    precision_enabled = resolved_device.type == "cuda"
    autocast_dtype = module.bfloat16
    for epoch in range(start_epoch, config.maximum_epochs + 1):
        model.train()
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0
        for batch_index, raw in enumerate(train_loader, start=1):
            batch = _move(raw, resolved_device)
            with module.autocast(
                device_type=resolved_device.type,
                dtype=autocast_dtype,
                enabled=precision_enabled,
            ):
                total, details, _ = _batch_losses(
                    model=model,
                    batch=batch,
                    frozen_expert_head=frozen_expert_head,
                    frozen_fusion=frozen_fusion,
                    token_mean=mean,
                    token_standard_deviation=std,
                    objective_id=config.objective_id,
                    normalization_mode=config.normalization_mode,
                    gradnorm_weights=(
                        None if balancer is None else balancer.current
                    ),
                )
                if balancer is not None:
                    new_weights = balancer.update(
                        terms=details["terms"],
                        shared_parameters=[
                            parameter
                            for parameter in model.parameters()
                            if parameter.requires_grad
                        ],
                        split="model_train",
                    )
                    total = sum(
                        new_weights[name] * value
                        for name, value in details["terms"].items()
                    )
                scaled = total / config.gradient_accumulation_steps
            scaled.backward()
            accumulated += 1
            final_batch = batch_index == len(train_loader)
            if (
                accumulated == config.gradient_accumulation_steps
                or final_batch
            ):
                if accumulated < config.gradient_accumulation_steps:
                    correction = (
                        config.gradient_accumulation_steps / accumulated
                    )
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(correction)
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
                norm = module.nn.utils.clip_grad_norm_(
                    model.parameters(), config.gradient_clip
                )
                if not bool(module.isfinite(norm)):
                    raise FloatingPointError(
                        "predictor gradient norm is nonfinite"
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated = 0
        validation = evaluate_predictor(
            model=model,
            loader=val_stop_loader,
            frozen_expert_head=frozen_expert_head,
            frozen_fusion=frozen_fusion,
            token_mean=mean,
            token_standard_deviation=std,
            objective_id=config.objective_id,
            normalization_mode=config.normalization_mode,
            device=resolved_device,
            gradnorm_weights=(None if balancer is None else balancer.current),
        )
        row = {
            "epoch": epoch,
            "val_stop": {
                "accuracy": validation["metrics"]["accuracy"],
                "cross_entropy": validation["metrics"]["cross_entropy"],
            },
            "objective": validation["metrics"]["objective"],
            "optimizer_update_ordinal": update_ordinal,
            "gradnorm_weights": (
                None if balancer is None else dict(balancer.current)
            ),
        }
        rows.append(row)
        epoch_state = {
            name: value.detach().cpu().clone()
            if hasattr(value, "detach")
            else copy.deepcopy(value)
            for name, value in model.state_dict().items()
        }
        _atomic_torch_save(
            {
                "contract": PREDICTOR_CHECKPOINT_CONTRACT,
                "schema_version": 1,
                "kind": "epoch_candidate",
                "training_contract_sha256": contract["content_hash"],
                "run_id": run_record["run_id"],
                "epoch": epoch,
                "model_state_dict": epoch_state,
                "model_state_sha256": _state_sha256(epoch_state),
            },
            candidate_root / f"epoch_{epoch:03d}.pt",
        )
        _atomic_torch_save(
            {
                "contract": PREDICTOR_CHECKPOINT_CONTRACT,
                "schema_version": 1,
                "kind": "resumable_last",
                "training_contract_sha256": contract["content_hash"],
                "run_id": run_record["run_id"],
                "epoch_completed": epoch,
                "planned_update_counts": counts,
                "model_state_dict": epoch_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "optimizer_update_ordinal": update_ordinal,
                "rows": rows,
                "gradnorm_state": (
                    None if balancer is None else balancer.state_dict()
                ),
                "rng_state": _rng_state(),
            },
            last_path,
        )
    if update_ordinal != counts["total_optimizer_updates"]:
        raise RuntimeError("predictor optimizer-update budget drifted")
    selected = preferred_expert_epoch(rows)
    selected_candidate = module.load(
        candidate_root / f"epoch_{int(selected['epoch']):03d}.pt",
        map_location="cpu",
        weights_only=False,
    )
    if (
        selected_candidate.get("kind") != "epoch_candidate"
        or selected_candidate["training_contract_sha256"]
        != contract["content_hash"]
        or selected_candidate["run_id"] != run_record["run_id"]
    ):
        raise ValueError("selected predictor epoch candidate differs")
    selected_state = selected_candidate["model_state_dict"]
    _atomic_torch_save(
        {
            "contract": PREDICTOR_CHECKPOINT_CONTRACT,
            "schema_version": 1,
            "kind": "selected_inference",
            "training_contract_sha256": contract["content_hash"],
            "run_id": run_record["run_id"],
            "epoch": int(selected["epoch"]),
            "model_state_dict": selected_state,
            "model_state_sha256": _state_sha256(selected_state),
            "selection_metrics": dict(selected["val_stop"]),
            "gradnorm_state": (
                None if balancer is None else balancer.state_dict()
            ),
        },
        selected_path,
    )
    curves = with_content_hash(
        {
            "contract": PREDICTOR_CURVES_CONTRACT,
            "schema_version": 1,
            "run_id": run_record["run_id"],
            "training_contract_sha256": contract["content_hash"],
            "rows": rows,
            "selected_epoch": int(selected["epoch"]),
            "epochs_completed": len(rows),
            "fixed_budget_completed": len(rows) == config.maximum_epochs,
            "stopped_early": False,
            "performance_result_affected_execution": False,
            "planned_update_counts": counts,
            "precision_mode": "BF16" if precision_enabled else "FP32",
            "gradnorm_adaptation_split": (
                None if balancer is None else "model_train"
            ),
        }
    )
    write_immutable_json(curves_path, curves)
    registration = with_content_hash(
        {
            "contract": PREDICTOR_REGISTRATION_CONTRACT,
            "schema_version": 1,
            "run_id": run_record["run_id"],
            "pipeline_seed": config.seed,
            "expert_id": model.target_expert_id,
            "architecture": config.architecture,
            "context": config.context,
            "objective_id": config.objective_id,
            "uncertainty_head": config.uncertainty_head,
            "normalization_mode": config.normalization_mode,
            "checkpoint_sha256": _file_sha256(selected_path),
            "training_curves_sha256": curves["content_hash"],
            "training_contract_sha256": contract["content_hash"],
            "lineage_hashes": parents,
            "selected_epoch": int(selected["epoch"]),
            "selected_val_stop": dict(selected["val_stop"]),
            "fixed_budget_completed": True,
            "performance_based_termination": False,
            "reused": False,
        }
    )
    write_immutable_json(registration_path, registration)
    return registration


__all__ = [
    "PREDICTOR_CHECKPOINT_CONTRACT",
    "PREDICTOR_CURVES_CONTRACT",
    "PREDICTOR_REGISTRATION_CONTRACT",
    "PREDICTOR_TRAINING_CONTRACT",
    "PredictorDataset",
    "PredictorTrainingConfig",
    "collate_predictor_batch",
    "evaluate_predictor",
    "make_predictor_loader",
    "train_predictor",
]
