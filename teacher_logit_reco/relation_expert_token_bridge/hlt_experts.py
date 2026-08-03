"""Native HLT expert datasets, initialization, objectives, and training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import hashlib
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import (
    canonical_json_bytes,
    require_sha256,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from .determinism import optimizer_update_counts, scheduled_learning_rate
from .evaluation import evaluate_classification
from .expert_training import DeterministicExpertSampler, preferred_expert_epoch
from .hlt_cache import identity_order_hash, validate_hlt_v3_cache
from .replicas import REALIZATION_POLICIES, replica_for

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


HLT_EVIDENCE_MODE_CONTRACT = "retb_native_hlt_evidence_modes_v1"
HLT_EXPERT_TRAINING_CONTRACT = "retb_native_hlt_expert_training_v2"
HLT_EXPERT_CHECKPOINT_CONTRACT = "retb_native_hlt_expert_checkpoint_v2"
HLT_EXPERT_CURVES_CONTRACT = "retb_native_hlt_expert_curves_v2"
HLT_EXPERT_REGISTRATION_CONTRACT = "retb_native_hlt_expert_registration_v2"
HLT_MODES = ("HE_SCRATCH_CE", "HE_OFFLINE_INIT", "HE_DUAL_OBJECTIVE")
DUAL_WEIGHTS = ((0.10, 0.25), (0.25, 0.25), (0.25, 0.50))
HLT_EVALUATION_REALIZATION_POLICY = "R_FIXED"


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is required for native HLT expert training")
    return torch


def _precision(device: Any) -> dict[str, Any]:
    module = _require_torch()
    resolved = module.device(device)
    enabled = resolved.type == "cuda"
    if enabled and not module.cuda.is_bf16_supported():
        raise RuntimeError("native HLT CUDA execution requires BF16 support")
    return {
        "mode": "bf16" if enabled else "fp32",
        "enabled": enabled,
        "dtype": module.bfloat16 if enabled else None,
    }


def build_hlt_evidence_mode_contract() -> dict[str, Any]:
    return with_content_hash(
        {
            "contract": HLT_EVIDENCE_MODE_CONTRACT,
            "schema_version": 1,
            "modes": {
                "HE_SCRATCH_CE": {
                    "initialization": "random",
                    "objective": "unweighted_HLT_cross_entropy",
                    "offline_targets_permitted": False,
                    "ordinary_hlt_only_baseline": True,
                    "learning_rate": 1.0e-3,
                },
                "HE_OFFLINE_INIT": {
                    "initialization": "corresponding_seed_matched_offline_expert",
                    "objective": "unweighted_HLT_cross_entropy",
                    "offline_targets_permitted": False,
                    "copied_parameter_learning_rate": 1.0e-4,
                    "new_parameter_learning_rate": 5.0e-4,
                },
                "HE_DUAL_OBJECTIVE": {
                    "initialization": "corresponding_seed_matched_offline_expert",
                    "objective": "HLT_CE_plus_token_MSE_plus_T2_expert_KL",
                    "offline_targets_required": True,
                    "privileged_training": True,
                    "candidate_weights": [list(value) for value in DUAL_WEIGHTS],
                    "copied_parameter_learning_rate": 1.0e-4,
                    "new_parameter_learning_rate": 5.0e-4,
                },
            },
            "token_alignment": {
                "definition": "mean_squared_error_over_B_K_D",
                "teacher_detached": True,
                "shape_exact": True,
            },
            "expert_logit_alignment": {
                "temperature": 2.0,
                "direction": "offline_teacher_to_HLT_student",
                "teacher_detached": True,
                "temperature_squared_multiplier": True,
            },
            "shared_hlt_normalizer_required": True,
            "expert_or_realization_specific_normalizer_forbidden": True,
            "availability_embedding_new_weights_initialize_to_zero": True,
            "fixed_epochs": 40,
            "early_stopping": False,
            "performance_based_termination": False,
        }
    )


@dataclass(frozen=True)
class NativeHLTExpertTrainingConfig:
    seed: int
    mode: str
    realization_policy: str = "R_MULTI"
    measurement_embedding: bool = False
    lambda_token: float = 0.0
    lambda_logit: float = 0.0
    maximum_epochs: int = 40
    minimum_learning_rate: float = 1.0e-5
    microbatch_size: int = 64
    gradient_accumulation_steps: int = 2
    effective_batch_size: int = 128
    weight_decay: float = 1.0e-4
    gradient_clip: float = 1.0
    accuracy_window: float = 0.0001
    num_workers: int = 0
    campaign_profile: str = "production"

    def validate(self) -> None:
        if self.mode not in HLT_MODES:
            raise ValueError("native HLT evidence mode is not registered")
        if self.realization_policy not in REALIZATION_POLICIES:
            raise ValueError("native HLT realization policy is not registered")
        if self.campaign_profile not in {"production", "miniature_test"}:
            raise ValueError("native HLT campaign profile is unknown")
        if self.campaign_profile == "production" and self.seed not in {
            101,
            202,
            303,
        }:
            raise ValueError("native HLT production seed is not registered")
        if self.mode == "HE_DUAL_OBJECTIVE":
            if (float(self.lambda_token), float(self.lambda_logit)) not in DUAL_WEIGHTS:
                raise ValueError("native HLT dual-objective weights are not registered")
        elif self.lambda_token != 0.0 or self.lambda_logit != 0.0:
            raise ValueError("non-dual native HLT mode cannot use offline losses")
        if min(
            self.maximum_epochs,
            self.microbatch_size,
            self.gradient_accumulation_steps,
            self.effective_batch_size,
        ) <= 0:
            raise ValueError("native HLT training integers must be positive")
        if (
            self.microbatch_size * self.gradient_accumulation_steps
            != self.effective_batch_size
        ):
            raise ValueError("native HLT accumulation changes effective batch")
        if self.num_workers != 0:
            raise ValueError("native HLT num_workers must remain zero")
        if self.campaign_profile == "production" and (
            self.maximum_epochs != 40 or self.effective_batch_size != 128
        ):
            raise ValueError("native HLT production schedule drifted")

    @property
    def base_learning_rate(self) -> float:
        return 1.0e-3 if self.mode == "HE_SCRATCH_CE" else 5.0e-4

    def artifact(
        self,
        *,
        global_determinism_sha256: str,
        evidence_mode_contract_sha256: str,
    ) -> dict[str, Any]:
        self.validate()
        return with_content_hash(
            {
                "contract": HLT_EXPERT_TRAINING_CONTRACT,
                "schema_version": 2,
                "config": asdict(self),
                "training_realization_policy": self.realization_policy,
                "evaluation_realization_policy": (
                    HLT_EVALUATION_REALIZATION_POLICY
                ),
                "global_determinism_sha256": require_sha256(
                    global_determinism_sha256,
                    name="global_determinism_sha256",
                ),
                "evidence_mode_contract_sha256": require_sha256(
                    evidence_mode_contract_sha256,
                    name="evidence_mode_contract_sha256",
                ),
                "optimizer": "AdamW",
                "betas": [0.9, 0.999],
                "schedule": "exact_integer_warmup_then_cosine",
                "checkpoint_selector": (
                    "val_stop_max_accuracy_0p0001_window_min_CE_earliest"
                ),
                "fixed_epoch_budget": True,
                "early_stopping": False,
                "performance_based_termination": False,
            }
        )


class NativeHLTExpertDataset(
    torch.utils.data.Dataset if torch is not None else object
):
    """Identity-aligned HLT replicas with optional privileged offline targets."""

    def __init__(
        self,
        *,
        replica_arrays: Mapping[int, Mapping[str, np.ndarray]],
        replica_metadata: Mapping[int, Mapping[str, Any]],
        labels: np.ndarray,
        identities: Sequence[str],
        logical_role: str,
        realization_policy: str,
        source_indices_by_replica: Mapping[int, Sequence[int]] | None = None,
        source_logical_role: str | None = None,
        offline_target_tokens: np.ndarray | None = None,
        offline_target_logits: np.ndarray | None = None,
        region_trees_by_replica: Mapping[
            int, Sequence[Mapping[str, Any]]
        ] | None = None,
    ) -> None:
        _require_torch()
        if realization_policy not in REALIZATION_POLICIES:
            raise ValueError("HLT dataset realization policy is unknown")
        self.logical_role = str(logical_role)
        self.replica_selection_role = str(source_logical_role or logical_role)
        self.realization_policy = str(realization_policy)
        expected_replicas = (
            {0}
            if logical_role not in {"model_train", "scale_train"}
            or realization_policy == "R_FIXED"
            else {0, 1, 2, 3}
        )
        if set(replica_arrays) != expected_replicas or set(
            replica_metadata
        ) != expected_replicas:
            raise ValueError("HLT dataset replica coverage differs")
        scale_identity_sequence = logical_role == "scale_train"
        self.identities = (
            identities
            if scale_identity_sequence
            else tuple(str(value) for value in identities)
        )
        self.labels = np.asarray(labels, dtype=np.int64)
        if (
            len(self.identities) == 0
            or self.labels.shape != (len(self.identities),)
            or (
                not scale_identity_sequence
                and len(self.identities) != len(set(self.identities))
            )
        ):
            raise ValueError("HLT dataset identity/label population differs")
        if bool(((self.labels < 0) | (self.labels >= 10)).any()):
            raise ValueError("HLT labels lie outside 0..9")
        self.replicas: dict[int, dict[str, np.ndarray]] = {}
        self.metadata = {}
        self.source_indices_by_replica = {
            replica: (
                range(len(self.identities))
                if source_indices_by_replica is None
                else source_indices_by_replica[replica]
            )
            for replica in expected_replicas
        }
        if set(self.source_indices_by_replica) != expected_replicas:
            raise ValueError("HLT source-index replica coverage differs")
        for replica in sorted(expected_replicas):
            arrays = {
                name: np.asarray(value)
                for name, value in replica_arrays[replica].items()
            }
            metadata = dict(replica_metadata[replica])
            validate_hlt_v3_cache(
                arrays,
                metadata,
                expected_logical_role=(source_logical_role or logical_role),
                expected_replica_id=replica,
            )
            indices = self.source_indices_by_replica[replica]
            if len(indices) != len(self.identities):
                raise ValueError("HLT source-index population differs")
            source_ids = arrays["identities"]
            positional = (
                isinstance(indices, range)
                and indices.start == 0
                and indices.step == 1
                and indices.stop == len(self.identities)
            )
            if positional:
                if metadata.get("identity_order_sha256") != identity_order_hash(
                    self.identities
                ):
                    raise ValueError("HLT replica identities differ")
            else:
                selected_ids = tuple(
                    str(source_ids[int(index)]) for index in indices
                )
                if selected_ids != tuple(self.identities):
                    raise ValueError("HLT replica identity subset differs")
            if metadata["realization_policy"] != realization_policy:
                raise ValueError("HLT cache realization policy differs")
            self.replicas[replica] = arrays
            self.metadata[replica] = metadata
        self.offline_target_tokens = (
            None
            if offline_target_tokens is None
            else np.asarray(offline_target_tokens, dtype=np.float32)
        )
        self.offline_target_logits = (
            None
            if offline_target_logits is None
            else np.asarray(offline_target_logits, dtype=np.float32)
        )
        if (self.offline_target_tokens is None) != (
            self.offline_target_logits is None
        ):
            raise ValueError("offline native-HLT targets must be paired")
        if self.offline_target_tokens is not None:
            if (
                self.offline_target_tokens.ndim != 3
                or self.offline_target_tokens.shape[0] != len(self)
                or self.offline_target_logits.shape != (len(self), 10)
                or not np.isfinite(self.offline_target_tokens).all()
                or not np.isfinite(self.offline_target_logits).all()
            ):
                raise ValueError("offline native-HLT target arrays differ")
        self.region_trees_by_replica = (
            None
            if region_trees_by_replica is None
            else {
                int(key): tuple(value)
                for key, value in region_trees_by_replica.items()
            }
        )
        if self.region_trees_by_replica is not None and (
            set(self.region_trees_by_replica) != expected_replicas
            or any(
                len(rows) != len(self)
                for rows in self.region_trees_by_replica.values()
            )
        ):
            raise ValueError("HLT REGION tree replica coverage differs")
        self.zero_based_epoch = 0

    def set_epoch(self, one_based_epoch: int) -> None:
        if int(one_based_epoch) <= 0:
            raise ValueError("HLT dataset epoch is one-based")
        self.zero_based_epoch = int(one_based_epoch) - 1

    def __len__(self) -> int:
        return len(self.identities)

    def replica_for_index(self, index: int) -> int:
        return int(
            replica_for(
                policy=self.realization_policy,
                logical_role=self.replica_selection_role,
                epoch=self.zero_based_epoch,
                canonical_identity=self.identities[int(index)],
            )
        )

    def locality_boundaries(self) -> tuple[int, ...]:
        if self.logical_role != "scale_train":
            return (0, len(self))
        return tuple([*range(0, len(self), 2_048), len(self)])

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.item_for_replica(index, self.replica_for_index(index))

    def item_for_replica(self, index: int, replica: int) -> dict[str, Any]:
        if int(replica) not in self.replicas:
            raise ValueError("requested HLT replica is absent")
        identity = self.identities[index]
        replica = int(replica)
        arrays = self.replicas[replica]
        source_index = int(self.source_indices_by_replica[replica][index])
        return {
            "tokens": arrays["tokens"][source_index],
            "mask": arrays["mask"][source_index],
            "measurement_states": arrays["measurement_states"][source_index],
            "label": self.labels[index],
            "identity": identity,
            "replica_id": replica,
            "offline_target_tokens": (
                None
                if self.offline_target_tokens is None
                else self.offline_target_tokens[index]
            ),
            "offline_target_logits": (
                None
                if self.offline_target_logits is None
                else self.offline_target_logits[index]
            ),
            "region_tree": (
                None
                if self.region_trees_by_replica is None
                else self.region_trees_by_replica[replica][index]
            ),
        }


def infer_native_hlt_expert_replica(
    *,
    model: Any,
    dataset: NativeHLTExpertDataset,
    replica_id: int,
    batch_size: int,
    device: str | Any = "cpu",
) -> dict[str, Any]:
    """Materialize one complete, identity-ordered native expert replica."""
    module = _require_torch()
    replica = int(replica_id)
    if replica not in dataset.replicas:
        raise ValueError("native HLT inference replica is absent")
    if int(batch_size) <= 0:
        raise ValueError("native HLT inference batch size must be positive")
    resolved = module.device(device)
    precision = _precision(resolved)
    model.to(resolved)
    model.eval()
    token_rows, logit_rows, particle_rows = [], [], []
    mask_rows, labels, identities = [], [], []
    with module.no_grad():
        for start in range(0, len(dataset), int(batch_size)):
            rows = [
                dataset.item_for_replica(index, replica)
                for index in range(start, min(start + int(batch_size), len(dataset)))
            ]
            batch = _move(collate_native_hlt_expert_batch(rows), resolved)
            with module.autocast(
                device_type=resolved.type,
                dtype=precision["dtype"],
                enabled=precision["enabled"],
            ):
                output = model(return_details=True, **_model_inputs(batch))
            particle_states = output.get("particle_states")
            if particle_states is None:
                # Lightweight test doubles may expose only tokens/logits.  The
                # production RETB expert always exposes its block-8 particle
                # tap; retain a deterministic feature-derived fallback solely
                # for miniature interface validation.
                if getattr(
                    getattr(model, "particle_encoder", None),
                    "expert_id",
                    None,
                ) is not None:
                    raise RuntimeError(
                        "native RETB expert omitted block-8 particle states"
                    )
                particle_states = batch["features"].transpose(1, 2).float()
                width = int(particle_states.shape[-1])
                particle_states = module.nn.functional.pad(
                    particle_states,
                    (0, max(0, 128 - width)),
                )[..., :128]
            if not bool(
                module.isfinite(output["tokens"]).all()
                and module.isfinite(output["logits"]).all()
                and module.isfinite(particle_states).all()
            ):
                raise FloatingPointError("native HLT inference is nonfinite")
            token_rows.append(output["tokens"].float().cpu().numpy())
            logit_rows.append(output["logits"].float().cpu().numpy())
            particle_rows.append(
                particle_states.float().cpu().numpy()
            )
            mask_rows.append(
                batch["mask"][:, 0].bool().cpu().numpy()
            )
            labels.append(batch["labels"].cpu().numpy())
            identities.extend(batch["event_identities"])
    if tuple(identities) != dataset.identities:
        raise RuntimeError("native HLT inference identity order drifted")
    return {
        "replica_id": replica,
        "identities": np.asarray(identities),
        "labels": np.concatenate(labels).astype(np.int64, copy=False),
        "tokens": np.concatenate(token_rows).astype(np.float32, copy=False),
        "logits": np.concatenate(logit_rows).astype(np.float32, copy=False),
        "particle_states": np.concatenate(particle_rows).astype(
            np.float32, copy=False
        ),
        "particle_mask": np.concatenate(mask_rows).astype(bool, copy=False),
    }


def collate_native_hlt_expert_batch(
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    module = _require_torch()
    from jetclass_fresh.part_inputs import (
        build_particle_transformer_inputs_from_tokens,
    )

    if not samples:
        raise ValueError("cannot collate an empty native HLT batch")
    tokens = np.stack([row["tokens"] for row in samples]).astype(np.float32)
    mask = np.stack([row["mask"] for row in samples]).astype(bool)
    labels = np.asarray([row["label"] for row in samples], dtype=np.int64)
    inputs = build_particle_transformer_inputs_from_tokens(
        tokens, mask, labels=labels, source_view="hlt"
    )
    output: dict[str, Any] = {
        "features": module.from_numpy(inputs.pf_features).float(),
        "vectors": module.from_numpy(inputs.pf_vectors).float(),
        "mask": module.from_numpy(inputs.pf_mask).bool(),
        "raw_tokens": module.from_numpy(tokens).float(),
        "labels": module.from_numpy(labels).long(),
        "event_identities": [row["identity"] for row in samples],
        "replica_ids": module.as_tensor(
            [row["replica_id"] for row in samples], dtype=module.int64
        ),
        "measurement_states": module.from_numpy(
            np.stack([row["measurement_states"] for row in samples])
        ).to(module.int64),
    }
    has_targets = samples[0]["offline_target_tokens"] is not None
    if any(
        (row["offline_target_tokens"] is not None) != has_targets
        for row in samples
    ):
        raise ValueError("native HLT batch mixes privileged target availability")
    if has_targets:
        output["offline_target_tokens"] = module.from_numpy(
            np.stack([row["offline_target_tokens"] for row in samples])
        ).float()
        output["offline_target_logits"] = module.from_numpy(
            np.stack([row["offline_target_logits"] for row in samples])
        ).float()
    trees = [row["region_tree"] for row in samples]
    if any(tree is not None for tree in trees):
        if not all(tree is not None for tree in trees):
            raise ValueError("native HLT batch mixes REGION tree availability")
        output["region_trees"] = trees
    return output


def make_native_hlt_expert_loader(
    dataset: NativeHLTExpertDataset,
    *,
    seed: int,
    training: bool,
    batch_size: int,
    sampler: Any | None = None,
) -> Any:
    module = _require_torch()
    if sampler is not None and not training:
        raise ValueError("custom native-HLT sampler requires training mode")
    if sampler is not None:
        selected_sampler = sampler
    elif training:
        selected_sampler = DeterministicExpertSampler(dataset, seed=seed)
    else:
        selected_sampler = module.utils.data.SequentialSampler(dataset)
    return module.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        sampler=selected_sampler,
        num_workers=0,
        drop_last=False,
        collate_fn=collate_native_hlt_expert_batch,
    )


def native_hlt_expert_objective(
    *,
    logits: Any,
    tokens: Any,
    labels: Any,
    mode: str,
    lambda_token: float = 0.0,
    lambda_logit: float = 0.0,
    offline_target_tokens: Any | None = None,
    offline_target_logits: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    module = _require_torch()
    if mode not in HLT_MODES:
        raise ValueError("native HLT objective mode is unknown")
    if logits.ndim != 2 or int(logits.shape[1]) != 10:
        raise ValueError("native HLT logits must have shape [B,10]")
    if tokens.ndim != 3 or int(tokens.shape[0]) != int(logits.shape[0]):
        raise ValueError("native HLT tokens have the wrong shape")
    if not bool(module.isfinite(logits).all() and module.isfinite(tokens).all()):
        raise FloatingPointError("native HLT predictions are nonfinite")
    ce = module.nn.functional.cross_entropy(logits, labels.long())
    if mode != "HE_DUAL_OBJECTIVE":
        if (
            offline_target_tokens is not None
            or offline_target_logits is not None
            or lambda_token != 0.0
            or lambda_logit != 0.0
        ):
            raise ValueError("non-dual native HLT objective consumed offline targets")
        token_loss = logits.new_zeros(())
        logit_kd = logits.new_zeros(())
    else:
        if (
            (float(lambda_token), float(lambda_logit)) not in DUAL_WEIGHTS
            or offline_target_tokens is None
            or offline_target_logits is None
        ):
            raise ValueError("dual native HLT objective lacks locked targets/weights")
        if tuple(tokens.shape) != tuple(offline_target_tokens.shape):
            raise ValueError("native/offline token target shapes differ")
        if tuple(logits.shape) != tuple(offline_target_logits.shape):
            raise ValueError("native/offline logit target shapes differ")
        target_tokens = offline_target_tokens.detach()
        target_logits = offline_target_logits.detach()
        if not bool(
            module.isfinite(target_tokens).all()
            and module.isfinite(target_logits).all()
        ):
            raise FloatingPointError("offline native-HLT targets are nonfinite")
        token_loss = module.nn.functional.mse_loss(tokens, target_tokens)
        temperature = 2.0
        logit_kd = module.nn.functional.kl_div(
            module.log_softmax(logits / temperature, dim=-1),
            module.softmax(target_logits / temperature, dim=-1),
            reduction="batchmean",
        ) * temperature**2
    total = ce + float(lambda_token) * token_loss + float(lambda_logit) * logit_kd
    if not bool(module.isfinite(total)):
        raise FloatingPointError("native HLT objective is nonfinite")
    return total, {
        "cross_entropy": ce.detach(),
        "token_alignment": token_loss.detach(),
        "expert_logit_alignment": logit_kd.detach(),
        "total": total.detach(),
    }


def copy_offline_expert_initialization(
    model: Any,
    offline_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Copy every compatible expert tensor; leave only availability state new."""
    module = _require_torch()
    target = model.state_dict()
    copied = []
    new = []
    parameter_names = set(dict(model.named_parameters()))
    for name, value in target.items():
        if not isinstance(value, module.Tensor):
            continue
        if name.startswith("particle_encoder.measurement_state_embedding."):
            with module.no_grad():
                value.zero_()
            target[name] = value
            new.append(name)
            continue
        source = offline_state.get(name)
        if not isinstance(source, module.Tensor) or tuple(source.shape) != tuple(
            value.shape
        ):
            raise ValueError(f"offline expert initialization lacks {name}")
        target[name] = source.detach().to(dtype=value.dtype, device=value.device)
        copied.append(name)
    if not copied:
        raise ValueError("offline expert initialization copied no tensors")
    model.load_state_dict(target, strict=True)
    return {
        "copied_tensor_names": copied,
        "copied_parameter_names": [
            name for name in copied if name in parameter_names
        ],
        "new_tensor_names": new,
        "new_parameter_names": [name for name in new if name in parameter_names],
        "copied_tensor_count": len(copied),
        "availability_embedding_zero_initialized": all(
            bool((target[name] == 0).all()) for name in new
        ),
    }


def native_hlt_parameter_groups(
    model: Any,
    *,
    mode: str,
    copied_parameter_names: Sequence[str] = (),
) -> list[dict[str, Any]]:
    if mode == "HE_SCRATCH_CE":
        return [{"params": list(model.parameters()), "lr": 1.0e-3}]
    copied = set(copied_parameter_names)
    named = dict(model.named_parameters())
    if not copied or not copied.issubset(named):
        raise ValueError("native HLT copied-parameter names differ")
    new_names = set(named) - copied
    groups = [
        {"params": [named[name] for name in sorted(copied)], "lr": 1.0e-4},
    ]
    if new_names:
        groups.append(
            {
                "params": [named[name] for name in sorted(new_names)],
                "lr": 5.0e-4,
            }
        )
    return groups


def _model_inputs(batch: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: batch[name]
        for name in ("features", "vectors", "mask", "raw_tokens", "region_trees")
        if name in batch
    }


def _move(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    module = _require_torch()
    return {
        name: value.to(device) if isinstance(value, module.Tensor) else value
        for name, value in batch.items()
    }


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


def _atomic_save(payload: Mapping[str, Any], path: Path) -> None:
    module = _require_torch()
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        module.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _evaluate(model: Any, loader: Any, device: Any) -> dict[str, Any]:
    module = _require_torch()
    precision = _precision(device)
    model.eval()
    logits, labels = [], []
    with module.no_grad():
        for raw in loader:
            batch = _move(raw, device)
            with module.autocast(
                device_type=module.device(device).type,
                dtype=precision["dtype"],
                enabled=precision["enabled"],
            ):
                output = model(return_details=True, **_model_inputs(batch))
            if not bool(module.isfinite(output["logits"]).all()):
                raise FloatingPointError("native HLT validation logits are nonfinite")
            logits.append(output["logits"].float().cpu().numpy())
            labels.append(batch["labels"].cpu().numpy())
    return evaluate_classification(
        np.concatenate(logits), np.concatenate(labels), split="val_stop"
    )


def train_native_hlt_expert(
    *,
    model: Any,
    train_loader: Any,
    val_stop_loader: Any,
    output_dir: str | Path,
    run_id: str,
    run_registry_sha256: str,
    lineage_hashes: Mapping[str, str],
    global_determinism_sha256: str,
    evidence_mode_contract_sha256: str,
    config: NativeHLTExpertTrainingConfig,
    device: str | Any = "cpu",
    offline_initialization_state: Mapping[str, Any] | None = None,
    offline_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    """Train the fixed budget; scientific underperformance never stops a row."""
    module = _require_torch()
    config.validate()
    particle_encoder = getattr(model, "particle_encoder", None)
    model_measurement_embedding = bool(
        getattr(particle_encoder, "measurement_embedding_enabled", False)
    )
    if model_measurement_embedding != config.measurement_embedding:
        raise ValueError(
            "native HLT model measurement embedding differs from its run contract"
        )
    train_dataset = getattr(train_loader, "dataset", None)
    val_dataset = getattr(val_stop_loader, "dataset", None)
    if (
        getattr(train_dataset, "logical_role", None)
        not in {"model_train", "scale_train"}
        or getattr(val_dataset, "logical_role", None) != "val_stop"
        or getattr(train_dataset, "realization_policy", None)
        != config.realization_policy
        or getattr(val_dataset, "realization_policy", None)
        != HLT_EVALUATION_REALIZATION_POLICY
    ):
        raise ValueError("native HLT loader split/realization contract differs")
    train_has_targets = (
        getattr(train_dataset, "offline_target_tokens", None) is not None
    )
    val_has_targets = (
        getattr(val_dataset, "offline_target_tokens", None) is not None
    )
    if config.mode == "HE_DUAL_OBJECTIVE":
        if not train_has_targets:
            raise ValueError("dual native HLT training lacks offline targets")
    elif train_has_targets or val_has_targets:
        raise ValueError("target-free native HLT mode received offline targets")
    contract = config.artifact(
        global_determinism_sha256=global_determinism_sha256,
        evidence_mode_contract_sha256=evidence_mode_contract_sha256,
    )
    parents = {
        str(name): require_sha256(value, name=f"lineage_hashes.{name}")
        for name, value in sorted(lineage_hashes.items())
    }
    registry_sha = require_sha256(
        run_registry_sha256, name="run_registry_sha256"
    )
    if config.mode == "HE_SCRATCH_CE":
        if offline_initialization_state is not None or offline_checkpoint_sha256:
            raise ValueError("scratch HLT expert cannot consume offline initialization")
        initialization = {
            "copied_parameter_names": [],
            "new_parameter_names": [name for name, _ in model.named_parameters()],
        }
    else:
        if offline_initialization_state is None:
            raise ValueError("initialized HLT expert requires offline state")
        offline_sha = require_sha256(
            offline_checkpoint_sha256, name="offline_checkpoint_sha256"
        )
        if parents.get("offline_expert_checkpoint") != offline_sha:
            raise ValueError("offline HLT initialization lineage differs")
        initialization = copy_offline_expert_initialization(
            model, offline_initialization_state
        )
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    registration_path = root / "checkpoint_registration.json"
    if registration_path.exists():
        from .contracts import load_hashed_json

        registration = load_hashed_json(
            registration_path,
            expected_contract=HLT_EXPERT_REGISTRATION_CONTRACT,
        )
        expected = {
            "run_id": run_id,
            "training_contract_sha256": contract["content_hash"],
            "run_registry_sha256": registry_sha,
            "lineage_hashes": parents,
        }
        if any(registration.get(key) != value for key, value in expected.items()):
            raise ValueError("reusable native HLT registration lineage differs")
        checkpoint = root / "best_model_val.pt"
        if (
            not checkpoint.is_file()
            or _file_sha256(checkpoint) != registration["checkpoint_sha256"]
        ):
            raise ValueError("reusable native HLT checkpoint bytes differ")
        curves = load_hashed_json(
            root / "training_curves.json",
            expected_contract=HLT_EXPERT_CURVES_CONTRACT,
        )
        metrics = load_hashed_json(root / "val_stop_metrics.json")
        if (
            curves["content_hash"] != registration["training_curves_sha256"]
            or metrics["content_hash"]
            != registration["val_stop_metrics_sha256"]
        ):
            raise ValueError("reusable native HLT diagnostics differ")
        return registration
    resolved = module.device(device)
    if config.campaign_profile == "production" and (
        resolved.type != "cuda"
        or not module.cuda.is_bf16_supported()
        or "GH200" not in module.cuda.get_device_name(resolved).upper()
    ):
        raise RuntimeError("production native HLT expert requires GH200 BF16")
    model.to(resolved)
    precision = _precision(resolved)
    groups = native_hlt_parameter_groups(
        model,
        mode=config.mode,
        copied_parameter_names=initialization["copied_parameter_names"],
    )
    optimizer = module.optim.AdamW(
        groups, betas=(0.9, 0.999), weight_decay=config.weight_decay
    )
    training_event_count = len(train_loader.dataset)
    counts = optimizer_update_counts(
        training_event_count=training_event_count,
        maximum_epochs=config.maximum_epochs,
        microbatch_size=config.microbatch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    if len(train_loader) != counts["microbatches_per_epoch"]:
        raise ValueError("native HLT loader differs from locked schedule")
    rows = []
    best_state = None
    update = 0
    base_group_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    for epoch in range(1, config.maximum_epochs + 1):
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        sums = {
            "cross_entropy": 0.0,
            "token_alignment": 0.0,
            "expert_logit_alignment": 0.0,
            "total": 0.0,
        }
        events = 0
        accumulated = 0
        for batch_index, raw in enumerate(train_loader, start=1):
            batch = _move(raw, resolved)
            with module.autocast(
                device_type=resolved.type,
                dtype=precision["dtype"],
                enabled=precision["enabled"],
            ):
                output = model(return_details=True, **_model_inputs(batch))
                loss, components = native_hlt_expert_objective(
                    logits=output["logits"],
                    tokens=output["tokens"],
                    labels=batch["labels"],
                    mode=config.mode,
                    lambda_token=config.lambda_token,
                    lambda_logit=config.lambda_logit,
                    offline_target_tokens=batch.get("offline_target_tokens"),
                    offline_target_logits=batch.get("offline_target_logits"),
                )
            current = int(batch["labels"].numel())
            (loss * current).backward()
            events += current
            accumulated += current
            for name in sums:
                sums[name] += float(components[name].cpu()) * current
            step_now = (
                batch_index % config.gradient_accumulation_steps == 0
                or batch_index == len(train_loader)
            )
            if not step_now:
                continue
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(accumulated)
                    if not bool(module.isfinite(parameter.grad).all()):
                        raise FloatingPointError("native HLT gradient is nonfinite")
            norm = module.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip
            )
            if not bool(module.isfinite(norm)):
                raise FloatingPointError("native HLT gradient norm is nonfinite")
            update += 1
            multiplier = scheduled_learning_rate(
                update_ordinal=update,
                total_optimizer_updates=counts["total_optimizer_updates"],
                warmup_updates=counts["warmup_updates"],
                base_learning_rate=1.0,
                minimum_learning_rate=(
                    config.minimum_learning_rate / config.base_learning_rate
                ),
            )
            for group, base_lr in zip(
                optimizer.param_groups, base_group_lrs, strict=True
            ):
                group["lr"] = base_lr * multiplier
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            accumulated = 0
        if events != training_event_count:
            raise RuntimeError("native HLT epoch event count drifted")
        metrics = _evaluate(model, val_stop_loader, resolved)
        rows.append(
            {
                "epoch": epoch,
                "optimizer_update_ordinal": update,
                "train_objective": {
                    name: value / events for name, value in sums.items()
                },
                "val_stop": {
                    "accuracy": metrics["accuracy"],
                    "cross_entropy": metrics["cross_entropy"],
                },
                "group_learning_rates": [
                    float(group["lr"]) for group in optimizer.param_groups
                ],
            }
        )
        selected = preferred_expert_epoch(
            rows, accuracy_window=config.accuracy_window
        )
        if int(selected["epoch"]) == epoch:
            best_state = _cpu_state(model)
    if best_state is None:
        raise RuntimeError("native HLT training retained no checkpoint")
    selected = preferred_expert_epoch(rows, accuracy_window=config.accuracy_window)
    model.load_state_dict(best_state, strict=True)
    final_metrics = _evaluate(model, val_stop_loader, resolved)
    checkpoint_path = root / "best_model_val.pt"
    _atomic_save(
        {
            "contract": HLT_EXPERT_CHECKPOINT_CONTRACT,
            "schema_version": 2,
            "run_id": run_id,
            "selected_epoch": int(selected["epoch"]),
            "training_contract_sha256": contract["content_hash"],
            "run_registry_sha256": registry_sha,
            "lineage_hashes": parents,
            "model_state_dict": best_state,
        },
        checkpoint_path,
    )
    curves = with_content_hash(
        {
            "contract": HLT_EXPERT_CURVES_CONTRACT,
            "schema_version": 2,
            "run_id": run_id,
            "rows": rows,
            "selected_epoch": int(selected["epoch"]),
            "optimizer_update_counts": counts,
            "fixed_budget_completed": len(rows) == config.maximum_epochs,
            "performance_based_termination": False,
        }
    )
    write_immutable_json(root / "training_curves.json", curves)
    write_immutable_json(root / "val_stop_metrics.json", final_metrics)
    registration = with_content_hash(
        {
            "contract": HLT_EXPERT_REGISTRATION_CONTRACT,
            "schema_version": 2,
            "run_id": run_id,
            "seed": config.seed,
            "mode": config.mode,
            "realization_policy": config.realization_policy,
            "training_realization_policy": config.realization_policy,
            "evaluation_realization_policy": (
                HLT_EVALUATION_REALIZATION_POLICY
            ),
            "measurement_embedding": config.measurement_embedding,
            "dual_weights": [config.lambda_token, config.lambda_logit],
            "privileged_offline_targets_consumed": (
                config.mode == "HE_DUAL_OBJECTIVE"
            ),
            "ordinary_hlt_only_baseline": config.mode == "HE_SCRATCH_CE",
            "training_contract_sha256": contract["content_hash"],
            "run_registry_sha256": registry_sha,
            "lineage_hashes": parents,
            "checkpoint_sha256": _file_sha256(checkpoint_path),
            "training_curves_sha256": curves["content_hash"],
            "val_stop_metrics_sha256": final_metrics["content_hash"],
            "selected_epoch": int(selected["epoch"]),
            "epochs_completed": len(rows),
            "fixed_epoch_budget_completed": True,
            "performance_based_termination": False,
            "precision_mode": precision["mode"],
            "initialization_report": initialization,
            "retained_checkpoints": ["best_model_val.pt"],
        }
    )
    write_immutable_json(registration_path, registration)
    return registration


__all__ = [
    "DUAL_WEIGHTS",
    "HLT_EVALUATION_REALIZATION_POLICY",
    "HLT_EVIDENCE_MODE_CONTRACT",
    "HLT_EXPERT_REGISTRATION_CONTRACT",
    "HLT_MODES",
    "NativeHLTExpertDataset",
    "NativeHLTExpertTrainingConfig",
    "build_hlt_evidence_mode_contract",
    "collate_native_hlt_expert_batch",
    "copy_offline_expert_initialization",
    "infer_native_hlt_expert_replica",
    "make_native_hlt_expert_loader",
    "native_hlt_expert_objective",
    "native_hlt_parameter_groups",
    "train_native_hlt_expert",
]
