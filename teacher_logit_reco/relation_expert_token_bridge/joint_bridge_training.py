"""Fixed-budget training and evaluation for RETB Stage-J joint graphs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from dataclasses import asdict, dataclass
import hashlib
import io
import math
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
from .joint_bridge import (
    JOINT_INPUT_POLICY,
    JOINT_VARIANTS,
    JointBridgeGraph,
    configure_joint_trainability,
    validate_common_view_metadata,
)
from .predictor_losses import FIXED_WEIGHTS, LOSS_COLUMNS, predictor_objective
from .replicas import replica_for
from .registry import EXPERT_ORDER

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


JOINT_TRAINING_CONTRACT = "retb_joint_bridge_training_v1"
JOINT_CHECKPOINT_CONTRACT = "retb_joint_bridge_checkpoint_v1"
JOINT_CURVES_CONTRACT = "retb_joint_bridge_curves_v1"
JOINT_REGISTRATION_CONTRACT = "retb_joint_bridge_registration_v1"
JOINT_INFERENCE_CONTRACT = "retb_joint_bridge_inference_v1"
STAGE_J_METRICS_CONTRACT = "retb_stage_j_val_design_metrics_v1"
JOINT_GRAPH_TEMPLATE_CONTRACT = "retb_joint_bridge_graph_template_v1"
JOINT_DATASET_CACHE_CONTRACT = "retb_joint_bridge_dataset_cache_v1"
JOINT_DATASET_PARENT_KEYS = frozenset(
    {
        "identity_manifest",
        "HLT_view_cache",
        "offline_target_cache",
        "target_normalizer_set",
    }
)
JOINT_GRAPH_COMMON_PARENT_KEYS = frozenset(
    {
        "frozen_offline_fusion",
        "frozen_offline_expert_heads",
        "offline_target_cache",
        "selected_predictor_seed_artifacts",
        "target_normalizer_set",
    }
)

VARIANT_LOSS_WEIGHTS = {
    "J0_INDEPENDENT": {
        "predictor": 1.0,
        "all_bank_fusion_KD": 0.0,
        "fused_CE": 0.0,
        "native_HLT_CE": 0.0,
    },
    "J1_SHARED_CONTEXT": {
        "predictor": 1.0,
        "all_bank_fusion_KD": 1.0,
        "fused_CE": 0.0,
        "native_HLT_CE": 0.0,
    },
    "J2_COUPLED_DECODER": {
        "predictor": 1.0,
        "all_bank_fusion_KD": 1.0,
        "fused_CE": 0.0,
        "native_HLT_CE": 0.0,
    },
    "J3_INDEPENDENT_PLUS_ADAPTER": {
        "predictor": 0.0,
        "all_bank_fusion_KD": 0.5,
        "fused_CE": 1.0,
        "native_HLT_CE": 0.0,
    },
    "J4_BRIDGE_FINETUNE": {
        "predictor": 1.0,
        "all_bank_fusion_KD": 0.5,
        "fused_CE": 1.0,
        "native_HLT_CE": 1.0,
    },
    "J5_END_TO_END": {
        "predictor": 0.5,
        "all_bank_fusion_KD": 0.5,
        "fused_CE": 1.0,
        "native_HLT_CE": 0.5,
    },
}


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is required for joint bridge training")
    return torch


def _array(value: Any, *, dtype: Any) -> np.ndarray:
    return np.asarray(value, dtype=dtype)


class JointBridgeDataset(
    torch.utils.data.Dataset if torch is not None else object
):
    """Identity-aligned joint evidence and privileged training targets."""

    def __init__(
        self,
        *,
        identities: Sequence[str],
        labels: np.ndarray,
        replica_ids: np.ndarray,
        degraded_view_hashes: Any,
        split: str,
        hlt_token_banks: Mapping[str, np.ndarray],
        unbiased_particle_states: np.ndarray,
        particle_mask: np.ndarray,
        relation_particle_states: Mapping[str, np.ndarray] | None,
        relation_particle_masks: Mapping[str, np.ndarray] | None,
        target_normalized_banks: Mapping[str, np.ndarray],
        oracle_banks: Mapping[str, np.ndarray],
        target_expert_logits: Mapping[str, np.ndarray],
        oracle_fusion_logits: np.ndarray,
        shared_raw_view: Mapping[str, Any] | None,
        lineage_hashes: Mapping[str, str],
    ) -> None:
        _require_torch()
        if split not in {
            "model_train",
            "scale_train",
            "val_stop",
            "val_design",
        }:
            raise ValueError("joint bridge dataset split differs")
        ids = tuple(str(value) for value in identities)
        truth = _array(labels, dtype=np.int64)
        replicas = _array(replica_ids, dtype=np.int64)
        if replicas.shape != (len(ids),):
            raise ValueError("joint bridge declared replica IDs differ")
        if (
            truth.shape != (len(ids),)
            or bool(((truth < 0) | (truth >= 10)).any())
            or set(hlt_token_banks) != set(EXPERT_ORDER)
            or set(target_normalized_banks) != set(EXPERT_ORDER)
            or set(oracle_banks) != set(EXPERT_ORDER)
            or set(target_expert_logits) != set(EXPERT_ORDER)
        ):
            raise ValueError("joint bridge dataset coverage differs")
        self.identities = ids
        self.labels = truth
        self.split = split
        self.realization_policy = JOINT_INPUT_POLICY
        self.zero_based_epoch = 0
        self.replica_set = (
            (0, 1, 2, 3)
            if split in {"model_train", "scale_train"}
            else (0,)
        )
        expected_at_epoch_zero = np.asarray(
            [
                replica_for(
                    policy=JOINT_INPUT_POLICY,
                    logical_role=split,
                    epoch=0,
                    canonical_identity=identity,
                )
                for identity in ids
            ],
            dtype=np.int64,
        )
        if not np.array_equal(replicas, expected_at_epoch_zero):
            raise ValueError(
                "joint bridge declared replica IDs differ from R_MULTI"
            )

        def replicas_of(value: Any, *, dtype: Any, name: str) -> dict[int, np.ndarray]:
            if isinstance(value, Mapping):
                if {int(key) for key in value} != set(self.replica_set):
                    raise ValueError(f"joint bridge {name} replica coverage differs")
                rows = {
                    replica: _array(
                        value.get(replica, value.get(str(replica))),
                        dtype=dtype,
                    )
                    for replica in self.replica_set
                }
            else:
                array = _array(value, dtype=dtype)
                if self.replica_set == (0,):
                    rows = {0: array}
                elif array.ndim >= 2 and tuple(array.shape[:2]) == (
                    4,
                    len(ids),
                ):
                    rows = {replica: array[replica] for replica in self.replica_set}
                else:
                    raise ValueError(
                        f"joint bridge {name} lacks four R_MULTI replicas"
                    )
            if any(len(row) != len(ids) for row in rows.values()):
                raise ValueError(f"joint bridge {name} identity coverage differs")
            return rows

        if isinstance(degraded_view_hashes, Mapping):
            hash_rows = {
                replica: tuple(
                    str(value)
                    for value in degraded_view_hashes.get(
                        replica, degraded_view_hashes.get(str(replica))
                    )
                )
                for replica in self.replica_set
            }
        elif self.replica_set == (0,):
            hash_rows = {0: tuple(str(value) for value in degraded_view_hashes)}
        else:
            array = np.asarray(degraded_view_hashes)
            if array.shape != (4, len(ids)):
                raise ValueError(
                    "joint bridge view hashes lack four R_MULTI replicas"
                )
            hash_rows = {
                replica: tuple(str(value) for value in array[replica])
                for replica in self.replica_set
            }
        for replica in self.replica_set:
            validate_common_view_metadata(
                identities=ids,
                replica_ids=[replica] * len(ids),
                degraded_view_hashes=hash_rows[replica],
            )
        self.degraded_view_hashes_by_replica = hash_rows
        self.hlt_token_banks = {
            name: replicas_of(
                hlt_token_banks[name],
                dtype=np.float32,
                name=f"hlt_token_banks.{name}",
            )
            for name in EXPERT_ORDER
        }
        self.unbiased_particle_states = replicas_of(
            unbiased_particle_states,
            dtype=np.float32,
            name="unbiased_particle_states",
        )
        self.particle_mask = replicas_of(
            particle_mask, dtype=bool, name="particle_mask"
        )
        self.relation_particle_states = (
            None
            if relation_particle_states is None
            else {
                name: replicas_of(
                    value,
                    dtype=np.float32,
                    name=f"relation_particle_states.{name}",
                )
                for name, value in relation_particle_states.items()
            }
        )
        self.relation_particle_masks = (
            None
            if relation_particle_masks is None
            else {
                name: replicas_of(
                    value,
                    dtype=bool,
                    name=f"relation_particle_masks.{name}",
                )
                for name, value in relation_particle_masks.items()
            }
        )
        self.target_normalized_banks = {
            name: _array(target_normalized_banks[name], dtype=np.float32)
            for name in EXPERT_ORDER
        }
        self.oracle_banks = {
            name: _array(oracle_banks[name], dtype=np.float32)
            for name in EXPERT_ORDER
        }
        self.target_expert_logits = {
            name: _array(target_expert_logits[name], dtype=np.float32)
            for name in EXPERT_ORDER
        }
        self.oracle_fusion_logits = _array(
            oracle_fusion_logits, dtype=np.float32
        )
        self.shared_raw_view = (
            None
            if shared_raw_view is None
            else {
                name: value
                for name, value in shared_raw_view.items()
            }
        )
        self.lineage_hashes = {
            name: require_sha256(value, name=f"lineage.{name}")
            for name, value in sorted(lineage_hashes.items())
        }
        count = len(ids)
        evidence_arrays = [
            *[
                row
                for values in self.hlt_token_banks.values()
                for row in values.values()
            ],
            *self.unbiased_particle_states.values(),
            *self.particle_mask.values(),
        ]
        arrays = [
            *evidence_arrays,
            *self.target_normalized_banks.values(),
            *self.oracle_banks.values(),
            *self.target_expert_logits.values(),
            self.oracle_fusion_logits,
        ]
        if any(
            self.target_normalized_banks[expert].ndim != 3
            or self.oracle_banks[expert].shape
            != self.target_normalized_banks[expert].shape
            or any(
                self.hlt_token_banks[expert][replica].shape
                != self.target_normalized_banks[expert].shape
                for replica in self.replica_set
            )
            for expert in EXPERT_ORDER
        ):
            raise ValueError("joint bridge token-bank shapes differ")
        if self.relation_particle_states is not None:
            if (
                self.relation_particle_masks is None
                or set(self.relation_particle_states)
                != {"PT", "TRACK", "REGION"}
                or set(self.relation_particle_masks)
                != {"PT", "TRACK", "REGION"}
            ):
                raise ValueError(
                    "joint bridge relation-particle coverage differs"
                )
            arrays.extend(
                row
                for values in self.relation_particle_states.values()
                for row in values.values()
            )
            arrays.extend(
                row
                for values in self.relation_particle_masks.values()
                for row in values.values()
            )
        elif self.relation_particle_masks is not None:
            raise ValueError("joint bridge relation masks lack states")
        if (
            any(len(value) != count for value in arrays)
            or any(
                tuple(self.particle_mask[replica].shape)
                != tuple(
                    self.unbiased_particle_states[replica].shape[:2]
                )
                for replica in self.replica_set
            )
            or self.oracle_fusion_logits.shape != (count, 10)
            or any(
                self.target_expert_logits[name].shape != (count, 10)
                for name in EXPERT_ORDER
            )
            or any(
                not np.isfinite(value).all()
                for value in arrays
                if value.dtype != np.bool_
            )
        ):
            raise ValueError("joint bridge dataset arrays differ")
        if self.shared_raw_view is not None:
            required = {
                "features",
                "vectors",
                "mask",
                "raw_tokens",
                "region_trees_by_expert",
            }
            if set(self.shared_raw_view) != required:
                raise ValueError("joint bridge raw-view fields differ")
            for name, dtype in (
                ("features", np.float32),
                ("vectors", np.float32),
                ("mask", bool),
                ("raw_tokens", np.float32),
            ):
                self.shared_raw_view[name] = replicas_of(
                    self.shared_raw_view[name],
                    dtype=dtype,
                    name=f"shared_raw_view.{name}",
                )
                if dtype is not bool and any(
                    not np.isfinite(value).all()
                    for value in self.shared_raw_view[name].values()
                ):
                    raise ValueError(
                        f"joint bridge raw view {name} is nonfinite"
                    )
            if set(
                self.shared_raw_view["region_trees_by_expert"]
            ) != set(EXPERT_ORDER):
                raise ValueError("joint bridge raw REGION coverage differs")
            trees = {}
            for expert, values in self.shared_raw_view[
                "region_trees_by_expert"
            ].items():
                if isinstance(values, Mapping):
                    resolved = {
                        replica: tuple(
                            values.get(replica, values.get(str(replica)))
                        )
                        for replica in self.replica_set
                    }
                elif self.replica_set == (0,):
                    resolved = {0: tuple(values)}
                else:
                    if len(values) != 4:
                        raise ValueError(
                            "joint bridge raw REGION replica coverage differs"
                        )
                    resolved = {
                        replica: tuple(values[replica])
                        for replica in self.replica_set
                    }
                if any(len(row) != count for row in resolved.values()):
                    raise ValueError(
                        "joint bridge raw REGION identity coverage differs"
                    )
                trees[expert] = resolved
            self.shared_raw_view["region_trees_by_expert"] = trees

    def set_epoch(self, one_based_epoch: int) -> None:
        if int(one_based_epoch) <= 0:
            raise ValueError("joint bridge dataset epoch is one-based")
        self.zero_based_epoch = int(one_based_epoch) - 1

    def __len__(self) -> int:
        return len(self.identities)

    def __getitem__(self, index: int) -> dict[str, Any]:
        replica = replica_for(
            policy=JOINT_INPUT_POLICY,
            logical_role=self.split,
            epoch=self.zero_based_epoch,
            canonical_identity=self.identities[index],
        )
        row = {
            "identity": self.identities[index],
            "label": self.labels[index],
            "replica_id": replica,
            "degraded_view_hash": self.degraded_view_hashes_by_replica[
                replica
            ][index],
            "hlt_token_banks": {
                name: value[replica][index]
                for name, value in self.hlt_token_banks.items()
            },
            "unbiased_particle_states": self.unbiased_particle_states[
                replica
            ][index],
            "particle_mask": self.particle_mask[replica][index],
            "relation_particle_states": (
                None
                if self.relation_particle_states is None
                else {
                    name: value[replica][index]
                    for name, value in self.relation_particle_states.items()
                }
            ),
            "relation_particle_masks": (
                None
                if self.relation_particle_masks is None
                else {
                    name: value[replica][index]
                    for name, value in self.relation_particle_masks.items()
                }
            ),
            "target_normalized_banks": {
                name: value[index]
                for name, value in self.target_normalized_banks.items()
            },
            "oracle_banks": {
                name: value[index] for name, value in self.oracle_banks.items()
            },
            "target_expert_logits": {
                name: value[index]
                for name, value in self.target_expert_logits.items()
            },
            "oracle_fusion_logits": self.oracle_fusion_logits[index],
            "shared_raw_view": None,
        }
        if self.shared_raw_view is not None:
            row["shared_raw_view"] = {
                name: (
                    {
                        expert: values[replica][index]
                        for expert, values in self.shared_raw_view[
                            "region_trees_by_expert"
                        ].items()
                    }
                    if name == "region_trees_by_expert"
                    else value[replica][index]
                )
                for name, value in self.shared_raw_view.items()
            }
        return row


def collate_joint_bridge_batch(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    module = _require_torch()
    if not rows:
        raise ValueError("cannot collate an empty joint bridge batch")
    relation_present = rows[0]["relation_particle_states"] is not None
    raw_present = rows[0]["shared_raw_view"] is not None
    if any(
        (row["relation_particle_states"] is not None) != relation_present
        or (row["shared_raw_view"] is not None) != raw_present
        for row in rows
    ):
        raise ValueError("joint bridge optional-input presence drifted")

    def stack(name: str, dtype: Any | None = None) -> Any:
        value = module.from_numpy(np.stack([row[name] for row in rows]))
        return value if dtype is None else value.to(dtype=dtype)

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
        "hlt_token_banks": {
            expert: module.from_numpy(
                np.stack([row["hlt_token_banks"][expert] for row in rows])
            ).float()
            for expert in EXPERT_ORDER
        },
        "unbiased_particle_states": stack(
            "unbiased_particle_states", module.float32
        ),
        "particle_mask": stack("particle_mask", module.bool),
        "relation_particle_states": (
            None
            if not relation_present
            else {
                source: module.from_numpy(
                    np.stack(
                        [
                            row["relation_particle_states"][source]
                            for row in rows
                        ]
                    )
                ).float()
                for source in ("PT", "TRACK", "REGION")
            }
        ),
        "relation_particle_masks": (
            None
            if not relation_present
            else {
                source: module.from_numpy(
                    np.stack(
                        [
                            row["relation_particle_masks"][source]
                            for row in rows
                        ]
                    )
                ).bool()
                for source in ("PT", "TRACK", "REGION")
            }
        ),
        "target_normalized_banks": {
            expert: module.from_numpy(
                np.stack(
                    [row["target_normalized_banks"][expert] for row in rows]
                )
            ).float()
            for expert in EXPERT_ORDER
        },
        "oracle_banks": {
            expert: module.from_numpy(
                np.stack([row["oracle_banks"][expert] for row in rows])
            ).float()
            for expert in EXPERT_ORDER
        },
        "target_expert_logits": {
            expert: module.from_numpy(
                np.stack(
                    [row["target_expert_logits"][expert] for row in rows]
                )
            ).float()
            for expert in EXPERT_ORDER
        },
        "oracle_fusion_logits": stack(
            "oracle_fusion_logits", module.float32
        ),
        "shared_raw_view": None,
    }
    if raw_present:
        batch["shared_raw_view"] = {
            "identities": batch["identities"],
            "replica_ids": batch["replica_ids"],
            "degraded_view_hashes": batch["degraded_view_hashes"],
            "features": module.from_numpy(
                np.stack(
                    [row["shared_raw_view"]["features"] for row in rows]
                )
            ).float(),
            "vectors": module.from_numpy(
                np.stack(
                    [row["shared_raw_view"]["vectors"] for row in rows]
                )
            ).float(),
            "mask": module.from_numpy(
                np.stack([row["shared_raw_view"]["mask"] for row in rows])
            ).bool(),
            "raw_tokens": module.from_numpy(
                np.stack(
                    [row["shared_raw_view"]["raw_tokens"] for row in rows]
                )
            ).float(),
            "region_trees_by_expert": {
                expert: [
                    row["shared_raw_view"]["region_trees_by_expert"][expert]
                    for row in rows
                ]
                for expert in EXPERT_ORDER
            },
        }
    return batch


def make_joint_bridge_loader(
    dataset: JointBridgeDataset,
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
        collate_fn=collate_joint_bridge_batch,
    )


def _move(value: Any, device: Any) -> Any:
    if isinstance(value, Mapping):
        return {name: _move(item, device) for name, item in value.items()}
    if isinstance(value, list):
        return value
    if hasattr(value, "to"):
        return value.to(device)
    return value


def _temperature_two_kl(predicted: Any, target: Any) -> Any:
    module = _require_torch()
    temperature = 2.0
    return module.nn.functional.kl_div(
        module.log_softmax(predicted / temperature, dim=-1),
        module.softmax(target.detach() / temperature, dim=-1),
        reduction="batchmean",
    ) * temperature**2


def joint_bridge_objective(
    *,
    graph: JointBridgeGraph,
    output: Mapping[str, Any],
    batch: Mapping[str, Any],
    objective_by_expert: Mapping[str, str],
    gradnorm_weights_by_expert: Mapping[str, Mapping[str, float] | None],
) -> tuple[Any, dict[str, Any]]:
    module = _require_torch()
    if (
        set(objective_by_expert) != set(EXPERT_ORDER)
        or set(gradnorm_weights_by_expert) != set(EXPERT_ORDER)
    ):
        raise ValueError("joint bridge objective coverage differs")
    predictor_terms = {}
    for expert in EXPERT_ORDER:
        hybrid_banks = dict(batch["oracle_banks"])
        hybrid_banks[expert] = output["predicted_tokens"][expert]
        hybrid_logits = graph.frozen_offline_fusion(
            token_banks=hybrid_banks
        )
        total, _ = predictor_objective(
            weight_id=objective_by_expert[expert],
            uncertainty_head=graph.predictors[expert].uncertainty_head,
            predicted_tokens=output["predicted_normalized_tokens"][expert],
            target_tokens=batch["target_normalized_banks"][expert],
            log_variance=output["log_variance"][expert],
            predicted_expert_logits=output["predicted_expert_logits"][expert],
            target_expert_logits=batch["target_expert_logits"][expert],
            predicted_hybrid_logits=hybrid_logits,
            target_hybrid_logits=batch["oracle_fusion_logits"],
            labels=batch["labels"],
            gradnorm_weights=gradnorm_weights_by_expert[expert],
        )
        predictor_terms[expert] = total
    predictor_mean = module.stack(list(predictor_terms.values())).mean()
    fusion_kd = _temperature_two_kl(
        output["logits"], batch["oracle_fusion_logits"]
    )
    fused_ce = module.nn.functional.cross_entropy(
        output["logits"], batch["labels"].long()
    )
    if output["native_hlt_logits"] is None:
        native_ce = fused_ce.new_zeros(())
    else:
        native_ce = module.stack(
            [
                module.nn.functional.cross_entropy(
                    output["native_hlt_logits"][expert],
                    batch["labels"].long(),
                )
                for expert in EXPERT_ORDER
            ]
        ).mean()
    weights = VARIANT_LOSS_WEIGHTS[graph.variant]
    total = (
        weights["predictor"] * predictor_mean
        + weights["all_bank_fusion_KD"] * fusion_kd
        + weights["fused_CE"] * fused_ce
        + weights["native_HLT_CE"] * native_ce
    )
    if not bool(module.isfinite(total)):
        raise FloatingPointError("joint bridge objective is nonfinite")
    return total, {
        "predictor_mean": predictor_mean.detach(),
        "all_bank_fusion_KD": fusion_kd.detach(),
        "fused_CE": fused_ce.detach(),
        "native_HLT_CE": native_ce.detach(),
        "total": total.detach(),
    }


def evaluate_joint_bridge(
    *,
    graph: JointBridgeGraph,
    loader: Any,
    objective_by_expert: Mapping[str, str],
    gradnorm_weights_by_expert: Mapping[str, Mapping[str, float] | None],
    device: Any,
) -> dict[str, Any]:
    module = _require_torch()
    graph.eval()
    logits, labels, identities, replicas, view_hashes = [], [], [], [], []
    squared_errors = {expert: [] for expert in EXPERT_ORDER}
    losses = []
    with module.no_grad():
        for raw in loader:
            batch = _move(raw, device)
            output = graph(
                shared_view=batch["shared_raw_view"]
                if graph.variant
                in {"J4_BRIDGE_FINETUNE", "J5_END_TO_END"}
                else None,
                evidence=(
                    None
                    if graph.variant
                    in {"J4_BRIDGE_FINETUNE", "J5_END_TO_END"}
                    else {
                        "hlt_token_banks": batch["hlt_token_banks"],
                        "unbiased_particle_states": batch[
                            "unbiased_particle_states"
                        ],
                        "particle_mask": batch["particle_mask"],
                        "relation_particle_states": batch[
                            "relation_particle_states"
                        ],
                        "relation_particle_masks": batch[
                            "relation_particle_masks"
                        ],
                    }
                ),
            )
            total, _ = joint_bridge_objective(
                graph=graph,
                output=output,
                batch=batch,
                objective_by_expert=objective_by_expert,
                gradnorm_weights_by_expert=gradnorm_weights_by_expert,
            )
            losses.append(float(total.cpu()) * len(raw["identities"]))
            logits.append(output["logits"].float().cpu().numpy())
            labels.append(raw["labels"].numpy())
            identities.extend(raw["identities"])
            replicas.extend(raw["replica_ids"].tolist())
            view_hashes.extend(raw["degraded_view_hashes"])
            for expert in EXPERT_ORDER:
                error = (
                    output["predicted_normalized_tokens"][expert]
                    - batch["target_normalized_banks"][expert]
                )
                squared_errors[expert].append(
                    error.square().float().cpu().numpy()
                )
    values = np.concatenate(logits)
    truth = np.concatenate(labels)
    metrics = evaluate_classification(
        values, truth, split=loader.dataset.split
    )
    return {
        "metrics": metrics,
        "objective": float(sum(losses) / len(truth)),
        "normalized_token_rmse_by_expert": {
            expert: float(
                np.sqrt(
                    np.concatenate(squared_errors[expert]).mean(
                        dtype=np.float64
                    )
                )
            )
            for expert in EXPERT_ORDER
        },
        "identities": identities,
        "replica_ids": replicas,
        "degraded_view_hashes": view_hashes,
        "logits": values.astype(np.float32),
    }


@dataclass(frozen=True)
class JointBridgeTrainingConfig:
    seed: int
    variant: str
    final_particle_blocks: int | None = None
    maximum_epochs: int = 40
    microbatch_size: int = 32
    gradient_accumulation_steps: int = 4
    effective_batch_size: int = 128
    minimum_learning_rate: float = 1.0e-5
    weight_decay: float = 1.0e-4
    gradient_clip: float = 1.0
    campaign_profile: str = "production"

    def validate(self) -> None:
        effective = int(self.effective_batch_size)
        if (
            int(self.seed) not in {101, 202, 303}
            or self.variant not in JOINT_VARIANTS
            or self.variant == "J0_INDEPENDENT"
            or (
                self.variant == "J4_BRIDGE_FINETUNE"
                and self.final_particle_blocks not in {2, 4}
            )
            or (
                self.variant != "J4_BRIDGE_FINETUNE"
                and self.final_particle_blocks is not None
            )
            or min(
                int(self.maximum_epochs),
                int(self.microbatch_size),
                int(self.gradient_accumulation_steps),
                effective,
            )
            <= 0
            or int(self.microbatch_size)
            * int(self.gradient_accumulation_steps)
            != effective
            or effective > 128
            or effective & (effective - 1)
            or float(self.minimum_learning_rate) != 1.0e-5
            or float(self.weight_decay) != 1.0e-4
            or float(self.gradient_clip) != 1.0
            or self.campaign_profile not in {"production", "miniature_test"}
            or (
                self.campaign_profile == "production"
                and int(self.maximum_epochs) != 40
            )
        ):
            raise ValueError("joint bridge training configuration differs")

    def artifact(
        self,
        *,
        step11_bundle_sha256: str,
        run_record_sha256: str,
        predictor_bundle_lock_sha256: str,
        global_determinism_sha256: str,
        lineage_hashes: Mapping[str, str],
    ) -> dict[str, Any]:
        self.validate()
        return with_content_hash(
            {
                "contract": JOINT_TRAINING_CONTRACT,
                "schema_version": 1,
                "configuration": asdict(self),
                "joint_input_policy": JOINT_INPUT_POLICY,
                "loss_weights": VARIANT_LOSS_WEIGHTS[self.variant],
                "optimizer": {
                    "name": "AdamW",
                    "betas": [0.9, 0.999],
                    "weight_decay": self.weight_decay,
                    "component_learning_rates": {
                        "HLT_particle_relation": 5.0e-5,
                        "HLT_tokenizer": 1.0e-4,
                        "predictor": 2.0e-4,
                        "adapter_or_deployable_fusion": 2.0e-4,
                    },
                },
                "schedule": {
                    "warmup": "min_T_max_1_ceil_0.05T",
                    "post_warmup": "cosine_per_parameter_group",
                    "minimum_learning_rate": self.minimum_learning_rate,
                },
                "epoch_selection": (
                    "val_stop_accuracy_window_0.0001_then_cross_entropy_"
                    "then_earliest_epoch"
                ),
                "fixed_budget": True,
                "early_stopping": False,
                "performance_based_termination": False,
                "parents": {
                    "step11_bundle": require_sha256(
                        step11_bundle_sha256, name="step11_bundle_sha256"
                    ),
                    "run_record": require_sha256(
                        run_record_sha256, name="run_record_sha256"
                    ),
                    "predictor_bundle_lock": require_sha256(
                        predictor_bundle_lock_sha256,
                        name="predictor_bundle_lock_sha256",
                    ),
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
    module = _require_torch()
    stream = io.BytesIO()
    module.save(dict(state), stream)
    return hashlib.sha256(stream.getvalue()).hexdigest()


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_torch(path: Path, payload: Mapping[str, Any]) -> None:
    module = _require_torch()
    stream = io.BytesIO()
    module.save(dict(payload), stream)
    write_immutable_bytes(path, stream.getvalue())


def _replace_torch(path: Path, payload: Mapping[str, Any]) -> None:
    module = _require_torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    module.save(dict(payload), temporary)
    temporary.replace(path)


def publish_joint_graph_template(
    *,
    output_dir: str | Path,
    graph: JointBridgeGraph,
    run_record_sha256: str,
    predictor_bundle_lock_sha256: str,
    objective_by_expert: Mapping[str, str],
    gradnorm_weights_by_expert: Mapping[str, Mapping[str, float] | None],
    component_parent_hashes: Mapping[str, str],
    source_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    required_component_parents = set(JOINT_GRAPH_COMMON_PARENT_KEYS)
    if graph.variant in {"J4_BRIDGE_FINETUNE", "J5_END_TO_END"}:
        required_component_parents.add("selected_HLT_expert_seed_artifacts")
    if graph.variant == "J5_END_TO_END":
        required_component_parents.update(
            {
                "j4_block_selection",
                "selected_J4_bridge_initialization",
            }
        )
    if (
        set(objective_by_expert) != set(EXPERT_ORDER)
        or set(gradnorm_weights_by_expert) != set(EXPERT_ORDER)
        or set(component_parent_hashes) != required_component_parents
        or any(
            objective_by_expert[expert]
            not in {*FIXED_WEIGHTS, "W_GRADNORM"}
            for expert in EXPERT_ORDER
        )
        or any(
            (
                objective_by_expert[expert] == "W_GRADNORM"
                and (
                    not isinstance(
                        gradnorm_weights_by_expert[expert], Mapping
                    )
                    or set(gradnorm_weights_by_expert[expert])
                    != set(LOSS_COLUMNS)
                )
            )
            or (
                objective_by_expert[expert] != "W_GRADNORM"
                and gradnorm_weights_by_expert[expert] is not None
            )
            for expert in EXPERT_ORDER
        )
    ):
        raise ValueError("joint graph template lineage coverage differs")
    root = Path(output_dir)
    path = root / "joint_graph_template.pt"
    stream = io.BytesIO()
    _require_torch().save(
        {
            "contract": JOINT_GRAPH_TEMPLATE_CONTRACT,
            "schema_version": 1,
            "graph": graph,
            "objective_by_expert": dict(objective_by_expert),
            "gradnorm_weights_by_expert": {
                expert: (
                    None
                    if gradnorm_weights_by_expert[expert] is None
                    else dict(gradnorm_weights_by_expert[expert])
                )
                for expert in EXPERT_ORDER
            },
        },
        stream,
    )
    publication = write_immutable_bytes(path, stream.getvalue())
    manifest = bind_source(
        with_content_hash(
            {
                "contract": JOINT_GRAPH_TEMPLATE_CONTRACT,
                "schema_version": 1,
                "variant": graph.variant,
                "allocation": graph.allocation,
                "input_policy": graph.input_policy,
                "template_filename": path.name,
                "template_sha256": publication["file_sha256"],
                "run_record_sha256": require_sha256(
                    run_record_sha256, name="run_record_sha256"
                ),
                "predictor_bundle_lock_sha256": require_sha256(
                    predictor_bundle_lock_sha256,
                    name="predictor_bundle_lock_sha256",
                ),
                "objective_by_expert": dict(objective_by_expert),
                "gradnorm_weights_by_expert": {
                    expert: (
                        None
                        if gradnorm_weights_by_expert[expert] is None
                        else dict(gradnorm_weights_by_expert[expert])
                    )
                    for expert in EXPERT_ORDER
                },
                "component_parent_hashes": {
                    name: require_sha256(
                        value, name=f"component_parent_hashes.{name}"
                    )
                    for name, value in sorted(component_parent_hashes.items())
                },
                "offline_components_training_only": True,
                "performance_based_termination": False,
            }
        ),
        source_snapshot=source_snapshot,
    )
    write_immutable_json(root / "joint_graph_template.json", manifest)
    return manifest


def load_joint_graph_template(
    manifest_path: str | Path,
    *,
    expected_variant: str,
    expected_run_record_sha256: str,
    expected_predictor_bundle_lock_sha256: str,
    expected_source: Mapping[str, Any],
) -> tuple[dict[str, Any], JointBridgeGraph, dict[str, str], dict[str, Any]]:
    manifest = load_hashed_json(
        manifest_path, expected_contract=JOINT_GRAPH_TEMPLATE_CONTRACT
    )
    path = Path(manifest_path).parent / manifest["template_filename"]
    expected_source_record = (
        source_record(expected_source)
        if "source_commit" in expected_source
        else dict(expected_source)
    )
    if (
        manifest.get("variant") != expected_variant
        or manifest.get("run_record_sha256")
        != require_sha256(
            expected_run_record_sha256, name="expected_run_record_sha256"
        )
        or manifest.get("predictor_bundle_lock_sha256")
        != require_sha256(
            expected_predictor_bundle_lock_sha256,
            name="expected_predictor_bundle_lock_sha256",
        )
        or manifest.get("source") != expected_source_record
        or not path.is_file()
        or path.is_symlink()
        or _file_sha256(path) != manifest["template_sha256"]
    ):
        raise ValueError("joint graph template lineage differs")
    payload = _require_torch().load(
        path, map_location="cpu", weights_only=False
    )
    graph = payload.get("graph")
    objectives = payload.get("objective_by_expert")
    gradnorm = payload.get("gradnorm_weights_by_expert")
    if (
        payload.get("contract") != JOINT_GRAPH_TEMPLATE_CONTRACT
        or not isinstance(graph, JointBridgeGraph)
        or graph.variant != expected_variant
        or graph.allocation != manifest["allocation"]
        or objectives != manifest["objective_by_expert"]
        or gradnorm != manifest["gradnorm_weights_by_expert"]
        or set(objectives) != set(EXPERT_ORDER)
        or set(gradnorm) != set(EXPERT_ORDER)
    ):
        raise ValueError("joint graph template semantics differ")
    return manifest, graph, objectives, gradnorm


def publish_joint_dataset_cache(
    *,
    output_dir: str | Path,
    dataset: JointBridgeDataset,
    parent_hashes: Mapping[str, str],
    source_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        set(parent_hashes) != set(JOINT_DATASET_PARENT_KEYS)
        or dataset.lineage_hashes != dict(sorted(parent_hashes.items()))
    ):
        raise ValueError("joint dataset cache parent coverage differs")
    root = Path(output_dir)
    path = root / "joint_dataset.pt"
    stream = io.BytesIO()
    _require_torch().save(
        {
            "contract": JOINT_DATASET_CACHE_CONTRACT,
            "schema_version": 1,
            "dataset": dataset,
        },
        stream,
    )
    publication = write_immutable_bytes(path, stream.getvalue())
    manifest = bind_source(
        with_content_hash(
            {
                "contract": JOINT_DATASET_CACHE_CONTRACT,
                "schema_version": 1,
                "split": dataset.split,
                "event_count": len(dataset),
                "input_policy": dataset.realization_policy,
                "replica_ids": list(dataset.replica_set),
                "identity_order_sha256": canonical_sha256(
                    list(dataset.identities)
                ),
                "dataset_filename": path.name,
                "dataset_sha256": publication["file_sha256"],
                "parent_hashes": {
                    name: require_sha256(
                        value, name=f"parent_hashes.{name}"
                    )
                    for name, value in sorted(parent_hashes.items())
                },
                "one_view_selected_per_epoch_identity": True,
                "offline_constituent_matches_present": False,
                "performance_based_termination": False,
            }
        ),
        source_snapshot=source_snapshot,
    )
    write_immutable_json(root / "joint_dataset.json", manifest)
    return manifest


def load_joint_dataset_cache(
    manifest_path: str | Path,
    *,
    expected_split: str,
    expected_source: Mapping[str, Any],
) -> tuple[dict[str, Any], JointBridgeDataset]:
    manifest = load_hashed_json(
        manifest_path, expected_contract=JOINT_DATASET_CACHE_CONTRACT
    )
    path = Path(manifest_path).parent / manifest["dataset_filename"]
    expected_source_record = (
        source_record(expected_source)
        if "source_commit" in expected_source
        else dict(expected_source)
    )
    if (
        manifest.get("split") != expected_split
        or manifest.get("input_policy") != JOINT_INPUT_POLICY
        or manifest.get("source") != expected_source_record
        or not path.is_file()
        or path.is_symlink()
        or _file_sha256(path) != manifest["dataset_sha256"]
    ):
        raise ValueError("joint dataset cache lineage differs")
    payload = _require_torch().load(
        path, map_location="cpu", weights_only=False
    )
    dataset = payload.get("dataset")
    if (
        payload.get("contract") != JOINT_DATASET_CACHE_CONTRACT
        or not isinstance(dataset, JointBridgeDataset)
        or dataset.split != expected_split
        or len(dataset) != int(manifest["event_count"])
        or canonical_sha256(list(dataset.identities))
        != manifest["identity_order_sha256"]
        or list(dataset.replica_set) != manifest["replica_ids"]
        or dataset.lineage_hashes != manifest["parent_hashes"]
    ):
        raise ValueError("joint dataset cache semantics differ")
    return manifest, dataset


def train_joint_bridge(
    *,
    graph: JointBridgeGraph,
    train_loader: Any,
    val_stop_loader: Any,
    objective_by_expert: Mapping[str, str],
    gradnorm_weights_by_expert: Mapping[str, Mapping[str, float] | None],
    output_dir: str | Path,
    run_record: Mapping[str, Any],
    step11_bundle_sha256: str,
    predictor_bundle_lock_sha256: str,
    global_determinism_sha256: str,
    lineage_hashes: Mapping[str, str],
    source_snapshot: Mapping[str, Any],
    config: JointBridgeTrainingConfig,
    device: Any = "cpu",
    resume: bool = True,
) -> dict[str, Any]:
    module = _require_torch()
    config.validate()
    validate_content_hash(run_record)
    if (
        graph.variant != config.variant
        or run_record.get("variant") != config.variant
        or int(run_record.get("pipeline_seed", -1)) != config.seed
        or train_loader.dataset.split
        not in {"model_train", "scale_train"}
        or val_stop_loader.dataset.split != "val_stop"
    ):
        raise ValueError("joint bridge run/configuration lineage differs")
    contract = bind_source(
        config.artifact(
            step11_bundle_sha256=step11_bundle_sha256,
            run_record_sha256=run_record["content_hash"],
            predictor_bundle_lock_sha256=predictor_bundle_lock_sha256,
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
    candidate_root = root / "epoch_candidates"
    candidate_root.mkdir(parents=True, exist_ok=True)
    if selected_path.is_file() and curves_path.is_file() and registration_path.is_file():
        curves = load_hashed_json(
            curves_path, expected_contract=JOINT_CURVES_CONTRACT
        )
        registration = load_hashed_json(
            registration_path, expected_contract=JOINT_REGISTRATION_CONTRACT
        )
        if (
            curves["training_contract_sha256"] != contract["content_hash"]
            or not curves["fixed_budget_completed"]
            or registration["checkpoint_sha256"]
            != _file_sha256(selected_path)
            or registration["training_curves_sha256"]
            != curves["content_hash"]
        ):
            raise ValueError("reusable joint bridge result differs")
        checkpoint = module.load(
            selected_path, map_location="cpu", weights_only=False
        )
        if (
            checkpoint.get("contract") != JOINT_CHECKPOINT_CONTRACT
            or checkpoint.get("training_contract_sha256")
            != contract["content_hash"]
        ):
            raise ValueError("reusable joint bridge checkpoint differs")
        graph.load_state_dict(checkpoint["model_state_dict"], strict=True)
        return registration
    _set_seed(config.seed)
    resolved = module.device(device)
    if config.campaign_profile == "production" and (
        resolved.type != "cuda"
        or not module.cuda.is_bf16_supported()
        or "GH200" not in module.cuda.get_device_name(resolved).upper()
    ):
        raise RuntimeError("production joint bridge training requires GH200 BF16")
    graph.to(resolved)
    trainability = configure_joint_trainability(
        graph, final_particle_blocks=config.final_particle_blocks
    )
    groups = trainability.pop("optimizer_groups")
    if not groups:
        raise ValueError("joint bridge training graph has no trainable parameters")
    base_lrs = [float(row["lr"]) for row in groups]
    optimizer = module.optim.AdamW(
        groups,
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
            state.get("contract") != JOINT_CHECKPOINT_CONTRACT
            or state.get("kind") != "resumable_last"
            or state.get("training_contract_sha256")
            != contract["content_hash"]
            or state.get("run_id") != run_record["run_id"]
            or state.get("planned_update_counts") != counts
        ):
            raise ValueError("joint bridge resume state differs")
        graph.load_state_dict(state["model_state_dict"], strict=True)
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
        graph.train()
        graph.frozen_expert_heads.eval()
        graph.frozen_offline_fusion.eval()
        if graph.hlt_experts is not None:
            for model in graph.hlt_experts.values():
                model.head.eval()
        if config.variant == "J3_INDEPENDENT_PLUS_ADAPTER":
            graph.predictors.eval()
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0
        for batch_index, raw in enumerate(train_loader, start=1):
            batch = _move(raw, resolved)
            with module.autocast(
                device_type=resolved.type,
                dtype=module.bfloat16,
                enabled=precision_enabled,
            ):
                output = graph(
                    shared_view=batch["shared_raw_view"]
                    if config.variant
                    in {"J4_BRIDGE_FINETUNE", "J5_END_TO_END"}
                    else None,
                    evidence=(
                        None
                        if config.variant
                        in {"J4_BRIDGE_FINETUNE", "J5_END_TO_END"}
                        else {
                            "hlt_token_banks": batch["hlt_token_banks"],
                            "unbiased_particle_states": batch[
                                "unbiased_particle_states"
                            ],
                            "particle_mask": batch["particle_mask"],
                            "relation_particle_states": batch[
                                "relation_particle_states"
                            ],
                            "relation_particle_masks": batch[
                                "relation_particle_masks"
                            ],
                        }
                    ),
                )
                total, _ = joint_bridge_objective(
                    graph=graph,
                    output=output,
                    batch=batch,
                    objective_by_expert=objective_by_expert,
                    gradnorm_weights_by_expert=gradnorm_weights_by_expert,
                )
                scaled = total / config.gradient_accumulation_steps
            scaled.backward()
            accumulated += 1
            if (
                accumulated == config.gradient_accumulation_steps
                or batch_index == len(train_loader)
            ):
                if accumulated < config.gradient_accumulation_steps:
                    correction = config.gradient_accumulation_steps / accumulated
                    for parameter in graph.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(correction)
                update_ordinal += 1
                for group, base_lr in zip(optimizer.param_groups, base_lrs):
                    group["lr"] = scheduled_learning_rate(
                        update_ordinal=update_ordinal,
                        total_optimizer_updates=counts[
                            "total_optimizer_updates"
                        ],
                        warmup_updates=counts["warmup_updates"],
                        base_learning_rate=base_lr,
                        minimum_learning_rate=min(
                            config.minimum_learning_rate, base_lr
                        ),
                    )
                norm = module.nn.utils.clip_grad_norm_(
                    [
                        parameter
                        for parameter in graph.parameters()
                        if parameter.requires_grad
                    ],
                    config.gradient_clip,
                )
                if not bool(module.isfinite(norm)):
                    raise FloatingPointError(
                        "joint bridge gradient norm is nonfinite"
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated = 0
        validation = evaluate_joint_bridge(
            graph=graph,
            loader=val_stop_loader,
            objective_by_expert=objective_by_expert,
            gradnorm_weights_by_expert=gradnorm_weights_by_expert,
            device=resolved,
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
            for name, value in graph.state_dict().items()
        }
        _save_torch(
            candidate_root / f"epoch_{epoch:03d}.pt",
            {
                "contract": JOINT_CHECKPOINT_CONTRACT,
                "schema_version": 1,
                "kind": "epoch_candidate",
                "training_contract_sha256": contract["content_hash"],
                "run_id": run_record["run_id"],
                "epoch": epoch,
                "model_state_dict": state,
                "model_state_sha256": _state_sha256(state),
            },
        )
        _replace_torch(
            last_path,
            {
                "contract": JOINT_CHECKPOINT_CONTRACT,
                "schema_version": 1,
                "kind": "resumable_last",
                "training_contract_sha256": contract["content_hash"],
                "run_id": run_record["run_id"],
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
        raise RuntimeError("joint bridge optimizer-update budget drifted")
    selected = preferred_expert_epoch(rows)
    candidate = module.load(
        candidate_root / f"epoch_{int(selected['epoch']):03d}.pt",
        map_location="cpu",
        weights_only=False,
    )
    _save_torch(
        selected_path,
        {
            "contract": JOINT_CHECKPOINT_CONTRACT,
            "schema_version": 1,
            "kind": "selected_inference",
            "training_contract_sha256": contract["content_hash"],
            "run_id": run_record["run_id"],
            "epoch": int(selected["epoch"]),
            "model_state_dict": candidate["model_state_dict"],
            "model_state_sha256": candidate["model_state_sha256"],
            "selection_metrics": selected["val_stop"],
        },
    )
    graph.load_state_dict(candidate["model_state_dict"], strict=True)
    curves = bind_source(
        with_content_hash(
            {
                "contract": JOINT_CURVES_CONTRACT,
                "schema_version": 1,
                "run_id": run_record["run_id"],
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
                "trainability": trainability,
                "precision_mode": "BF16" if precision_enabled else "FP32",
            }
        ),
        source_snapshot=source_snapshot,
    )
    write_immutable_json(curves_path, curves)
    registration = bind_source(
        with_content_hash(
            {
                "contract": JOINT_REGISTRATION_CONTRACT,
                "schema_version": 1,
                "run_id": run_record["run_id"],
                "pipeline_seed": config.seed,
                "variant": config.variant,
                "predictor_bundle_lock_sha256": predictor_bundle_lock_sha256,
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


def publish_joint_inference(
    *,
    output_dir: str | Path,
    evaluation: Mapping[str, Any],
    split: str,
    pipeline_seed: int,
    variant: str,
    registration_sha256: str,
    identity_manifest_sha256: str,
    hlt_input_cache_sha256: str,
    source_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        split not in {"val_stop", "val_design"}
        or int(pipeline_seed) not in {101, 202, 303}
        or variant not in JOINT_VARIANTS
    ):
        raise ValueError("joint inference identity differs")
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
        raise ValueError("joint inference logits differ")
    stream = io.BytesIO()
    np.savez_compressed(
        stream,
        identities=np.asarray(identities, dtype=np.str_),
        replica_ids=replicas,
        degraded_view_hashes=np.asarray(hashes, dtype=np.str_),
        logits=logits,
    )
    root = Path(output_dir)
    publication = write_immutable_bytes(
        root / "joint_predictions.npz", stream.getvalue()
    )
    return bind_source(
        with_content_hash(
            {
                "contract": JOINT_INFERENCE_CONTRACT,
                "schema_version": 1,
                "split": split,
                "pipeline_seed": int(pipeline_seed),
                "variant": variant,
                "event_count": len(identities),
                "npz_filename": "joint_predictions.npz",
                "npz_sha256": publication["file_sha256"],
                "identity_order_sha256": canonical_sha256(
                    list(identities)
                ),
                "replica_order_sha256": canonical_sha256(
                    replicas.tolist()
                ),
                "degraded_view_order_sha256": canonical_sha256(list(hashes)),
                "parents": {
                    "joint_registration": require_sha256(
                        registration_sha256, name="registration_sha256"
                    ),
                    "identity_manifest": require_sha256(
                        identity_manifest_sha256,
                        name="identity_manifest_sha256",
                    ),
                    "HLT_input_cache": require_sha256(
                        hlt_input_cache_sha256,
                        name="hlt_input_cache_sha256",
                    ),
                },
                "input_policy": JOINT_INPUT_POLICY,
                "offline_inputs_present": False,
                "oracle_targets_present": False,
                "complete_coverage": True,
            }
        ),
        source_snapshot=source_snapshot,
    )


__all__ = [
    "JOINT_CHECKPOINT_CONTRACT",
    "JOINT_CURVES_CONTRACT",
    "JOINT_DATASET_CACHE_CONTRACT",
    "JOINT_GRAPH_TEMPLATE_CONTRACT",
    "JOINT_DATASET_PARENT_KEYS",
    "JOINT_GRAPH_COMMON_PARENT_KEYS",
    "JOINT_INFERENCE_CONTRACT",
    "JOINT_REGISTRATION_CONTRACT",
    "JOINT_TRAINING_CONTRACT",
    "STAGE_J_METRICS_CONTRACT",
    "JointBridgeDataset",
    "JointBridgeTrainingConfig",
    "VARIANT_LOSS_WEIGHTS",
    "collate_joint_bridge_batch",
    "evaluate_joint_bridge",
    "joint_bridge_objective",
    "make_joint_bridge_loader",
    "load_joint_dataset_cache",
    "load_joint_graph_template",
    "publish_joint_dataset_cache",
    "publish_joint_graph_template",
    "publish_joint_inference",
    "train_joint_bridge",
]
