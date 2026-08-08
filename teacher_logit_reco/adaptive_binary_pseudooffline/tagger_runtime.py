"""Production E/F/G0/G1 training runtime for adaptive-binary pseudo views."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np

from jetclass_fresh.hlt_baseline import require_torch, resolve_device, set_training_seed
from jetclass_fresh.hlt_cache import jet_identity_hash, load_cached_hlt_view
from jetclass_fresh.jetclass_data import LABEL_NAMES

from .config import canonical_hash
from .prediction_cache import (
    DeployablePseudoViewBatch,
    iter_deployable_pseudo_view_shards,
    require_deployable_pseudo_view_cache,
)
from .pseudo_ram import (
    FrozenReconstructorRamSource,
    SelectedReconstructorPseudoGenerator,
)
from .pseudo_consumer import (
    ABPH_CONSUMER_PSEUDO_CONTRACT,
    ABPH_CONSUMER_PSEUDO_SCHEMA_VERSION,
    consumer_pseudo_array_names,
)
from .production import (
    AdaptiveBinaryReconstructorModel,
    AdaptiveBinaryTargetBatchSource,
    build_shared_root_dual_reconstructor,
    campaign_target_source_kwargs,
    load_selected_reconstructor,
    package_trainable_pseudo_views,
    reconstructor_step,
)
from .checkpoints import (
    ABPH_COMPACT_SELECTED_CHECKPOINT_CONTRACT,
    active_storage_profile,
    build_compact_selected_checkpoint,
    load_torch_checkpoint,
    selected_checkpoint_provenance,
    selected_model_state,
    streaming_storage_enabled,
    write_selected_checkpoint,
)
from .tagger import (
    HierarchyAwareDualStreamTagger,
    PseudoViewInputs,
    build_variant_hierarchy_aware_tagger,
    load_dual_stream_warm_starts,
)
from .tagging_objectives import compute_tagging_objective, tagging_objective_config
from .variants import variant_spec
from .training import (
    CurriculumState,
    ReconstructionLossWeights,
    ReconstructorStepContext,
    compose_reconstruction_loss,
)
from .ram_workspace import RankLocalWorkspace
from .distributed import (
    abort_distributed_runtime,
    all_gather_objects,
    all_reduce_float64_pair,
    all_reduce_min_bool,
    all_reduce_sum_int,
    barrier,
    broadcast_object,
    destroy_distributed_runtime,
    gather_error_summaries,
    initialize_distributed_runtime,
    verify_common_parameter_state,
)
from .distributed_stream import validation_range_row
from .distributed_validation import (
    TypedValidationAccumulator,
    finalize_typed_validation,
)
from .tagger_distributed import (
    TaggerTrainingModule,
    build_tagger_ddp_wrapper,
    compile_tagger_global_batch_plan,
    require_tagger_tensor_mapping,
    tagger_tensor_mapping_is_finite,
)


ABPH_TAGGER_RUNTIME_CONTRACT = "adaptive_binary_pseudooffline_tagger_runtime_v1"
ABPH_FROZEN_SINGLE_PSEUDO_SOURCES = frozenset(
    {
        "D1_kt32_mh4_particles",
        "D2_ca32_mh4_particles",
        "D7_kt8_mh4_particles_screen",
    }
)


def _uninitialized_state_names(module: Any) -> tuple[str, ...]:
    """Return lazy parameter/buffer names that cannot yet be hashed or wrapped."""

    torch = require_torch()
    lazy_types = tuple(
        value
        for value in (
            getattr(torch.nn.parameter, "UninitializedParameter", None),
            getattr(torch.nn.parameter, "UninitializedBuffer", None),
        )
        if value is not None
    )
    rows = (
        *(
            f"parameter:{name}"
            for name, value in module.named_parameters()
            if isinstance(value, lazy_types)
        ),
        *(
            f"buffer:{name}"
            for name, value in module.named_buffers()
            if isinstance(value, lazy_types)
        ),
    )
    return tuple(sorted(rows))


def _materialize_tagger_dynamic_state(
    module: Any,
    forward_once: Callable[[], Any],
    *,
    distributed_runtime: Any,
    seed: int,
) -> dict[str, Any]:
    """Materialize input-shaped tagger state before hashing, AdamW, or DDP."""

    torch = require_torch()
    before = _uninitialized_state_names(module)
    local_error: BaseException | None = None
    was_training = bool(module.training)
    try:
        if before:
            module.eval()
            with torch.no_grad():
                forward_once()
    except BaseException as exc:  # synchronized below before any DDP collective
        local_error = exc
    finally:
        module.train(was_training)

    after = _uninitialized_state_names(module)
    if local_error is None and after:
        local_error = RuntimeError(
            "tagger representative forward left uninitialized state: "
            + ", ".join(after)
        )
    summaries = gather_error_summaries(
        distributed_runtime,
        phase="tagger_dynamic_state_materialization",
        error=local_error,
        structural=True,
    )
    failures = [row for row in summaries if row.get("error_type") is not None]
    if failures:
        raise RuntimeError(
            "tagger dynamic-state materialization failed across ranks: "
            + json.dumps(failures, sort_keys=True)
        ) from local_error

    # Materialization consumes initialization/dropout RNG. Restore the declared
    # training seed so the first optimizer update keeps its reproducible stream.
    set_training_seed(int(seed))
    return {
        "required": bool(before),
        "materialized_state_names": list(before),
        "materialized_state_count": len(before),
        "remaining_uninitialized_state_names": list(after),
        "representative_forward_mode": "eval_no_grad",
        "training_seed_restored": True,
        "completed_on_all_ranks": True,
    }


def _binary_auc(scores: np.ndarray, positive: np.ndarray) -> float | None:
    values = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(positive, dtype=bool)
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.shape[0], dtype=np.float64)
    start = 0
    while start < values.shape[0]:
        stop = start + 1
        while stop < values.shape[0] and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return float(
        (ranks[labels].sum() - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def _detailed_eval_metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    values = np.asarray(logits, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int64)
    shifted = values - values.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predictions = probabilities.argmax(axis=1)
    classes = int(values.shape[1])
    confusion = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(confusion, (truth, predictions), 1)
    rows = []
    aucs = []
    for class_index in range(classes):
        support = int(confusion[class_index].sum())
        auc = _binary_auc(probabilities[:, class_index], truth == class_index)
        if auc is not None:
            aucs.append(auc)
        rows.append(
            {
                "class_index": class_index,
                "class_name": LABEL_NAMES[class_index] if class_index < len(LABEL_NAMES) else str(class_index),
                "support": support,
                "accuracy": (
                    float(confusion[class_index, class_index] / support)
                    if support
                    else None
                ),
                "ovr_auc": auc,
            }
        )
    picked = probabilities[np.arange(truth.shape[0]), truth].clip(1.0e-12, 1.0)
    return {
        "available": True,
        "accuracy": float((predictions == truth).mean()),
        "loss": float(-np.log(picked).mean()),
        "cross_entropy": float(-np.log(picked).mean()),
        "macro_ovr_auc": float(np.mean(aucs)) if aucs else None,
        "macro_per_class_accuracy": float(
            np.mean([row["accuracy"] for row in rows if row["accuracy"] is not None])
        ),
        "per_class": rows,
        "per_class_accuracy": rows,
        "confusion_matrix": confusion.tolist(),
        "n_jets": int(truth.shape[0]),
    }


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _torch_load(path: str | Path, *, device: Any = "cpu") -> Mapping[str, Any]:
    return load_torch_checkpoint(path, device=device)


def _state_dict(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if payload.get("checkpoint_contract") == ABPH_COMPACT_SELECTED_CHECKPOINT_CONTRACT:
        return selected_model_state(payload)
    for name in ("model_state_dict", "model_state", "state_dict", "model"):
        value = payload.get(name)
        if isinstance(value, Mapping):
            return value
    raise KeyError("checkpoint has no model state")


def _load_prefixed_module(module: Any, payload: Mapping[str, Any], prefix: str) -> None:
    state = _state_dict(payload)
    selected = {
        name[len(prefix) :]: value
        for name, value in state.items()
        if name.startswith(prefix)
    }
    if not selected:
        raise ValueError(f"checkpoint contains no {prefix} parameters")
    module.load_state_dict(selected, strict=True)


def _prediction_arrays(
    arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    return {
        str(name): np.asarray(arrays[name])
        for name in metadata["prediction_schema"]
    }


def _pseudo_batch(
    arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]
) -> DeployablePseudoViewBatch:
    batch = DeployablePseudoViewBatch(
        arrays=_prediction_arrays(arrays, metadata),
        view_names=tuple(metadata["view_names"]),
        hierarchy_names=tuple(metadata["hierarchy_names"]),
        frontier_depths={
            str(name): int(value)
            for name, value in metadata["frontier_depths"].items()
        },
        diagnostics={
            "offline_inputs_loaded": False,
            "teacher_logits_loaded": False,
            "offline_target_selected_hypothesis": False,
            "fixed_evaluation_hypotheses": True,
        },
    )
    batch.validate()
    return batch.to_consumer_only()


def _merge_independent_dual_batches(
    first: DeployablePseudoViewBatch,
    second: DeployablePseudoViewBatch,
) -> tuple[DeployablePseudoViewBatch, Mapping[str, np.ndarray]]:
    """Package E11's independent roots while keeping the stable input schema valid."""

    first.validate()
    second.validate()
    if first.batch_size != second.batch_size:
        raise ValueError("independent hierarchy pseudo batches differ in size")
    kt_name = first.hierarchy_names[0]
    ca_name = second.hierarchy_names[0]
    if (kt_name, ca_name) != ("exclusive_kt", "cambridge_aachen"):
        raise ValueError("independent hierarchy sources have unexpected grouping order")
    arrays = dict(first.arrays)
    arrays.update(
        {name: value for name, value in second.arrays.items() if name.startswith("particle__") or name.startswith("frontier__")}
    )
    independent = {
        kt_name: np.asarray(first.arrays["shared_root_ledger"]),
        ca_name: np.asarray(second.arrays["shared_root_ledger"]),
    }
    # PseudoViewInputs has one canonical shared-root schema. E11 passes the
    # actual independent roots separately to the hierarchy encoder.
    arrays[f"frontier__{ca_name}__depth_00__ledger"] = np.asarray(
        first.arrays[f"frontier__{kt_name}__depth_00__ledger"]
    )
    arrays[f"frontier__{ca_name}__depth_00__uncertainty"] = np.asarray(
        first.arrays[f"frontier__{kt_name}__depth_00__uncertainty"]
    )
    arrays[f"frontier__{ca_name}__depth_00__mask"] = np.asarray(
        first.arrays[f"frontier__{kt_name}__depth_00__mask"]
    )
    diagnostics = dict(first.diagnostics)
    diagnostics.pop("consumer_pseudo_schema_hash", None)
    diagnostics["consumer_only_pseudo"] = False
    merged = DeployablePseudoViewBatch(
        arrays=arrays,
        view_names=first.view_names,
        hierarchy_names=(kt_name, ca_name),
        frontier_depths={
            kt_name: int(first.frontier_depths[kt_name]),
            ca_name: int(second.frontier_depths[ca_name]),
        },
        diagnostics=diagnostics,
    )
    merged.validate()
    return merged.to_consumer_only(), independent


@dataclass(frozen=True)
class FrozenPseudoBatch:
    hlt_tokens: Any
    hlt_mask: Any
    labels: Any
    indices: Any
    pseudo: DeployablePseudoViewBatch
    independent_roots: Mapping[str, np.ndarray] | None = None


class FrozenPseudoBatchSource:
    """Shard-streamed pseudo cache joined fail-closed to one HLT split."""

    def __init__(
        self,
        *,
        hlt_cache_dir: str | Path,
        cache_dirs: Sequence[str | Path],
        split: str,
        batch_size: int,
        independent_roots: bool = False,
        maximum_batches: int | None = None,
    ) -> None:
        self.hlt = load_cached_hlt_view(hlt_cache_dir, split, verify_hash=True)
        self.cache_dirs = tuple(Path(value) for value in cache_dirs)
        self.metadata = tuple(
            require_deployable_pseudo_view_cache(value)["metadata"]
            for value in self.cache_dirs
        )
        self.split = str(split)
        self.batch_size = int(batch_size)
        self.independent_roots = bool(independent_roots)
        self.maximum_batches = maximum_batches
        self.rank = 0
        self.world_size = 1
        self.rank_start = 0
        self.rank_stop = len(self.hlt.labels)
        if self.batch_size <= 0:
            raise ValueError("tagger batch size must be positive")
        hlt_hash = self.hlt.metadata.get("hlt_content_hash")
        identity = jet_identity_hash(self.hlt.jet_ids)
        for metadata in self.metadata:
            if metadata.get("split") != self.split:
                raise ValueError("pseudo cache split differs from tagger split")
            if metadata.get("hlt_content_hash") != hlt_hash:
                raise ValueError("pseudo cache is not bound to active HLT particles")
            if metadata.get("jet_identity_hash") != identity:
                raise ValueError("pseudo cache and HLT identities differ")
            if int(metadata.get("n_jets", -1)) != len(self.hlt.labels):
                raise ValueError("pseudo cache and HLT split sizes differ")

    def iter_batches(self, *, shuffle: bool, seed: int) -> Iterator[FrozenPseudoBatch]:
        streams = [iter_deployable_pseudo_view_shards(path) for path in self.cache_dirs]
        yielded = 0
        for shard_rows in zip(*streams):
            primary_arrays, primary_ids, _ = shard_rows[0]
            for arrays, identities, _ in shard_rows[1:]:
                if identities != primary_ids:
                    raise ValueError("dual pseudo caches changed shard identity order")
                if not np.array_equal(arrays["source_indices"], primary_arrays["source_indices"]):
                    raise ValueError("dual pseudo caches changed source indices")
            source_indices = np.asarray(primary_arrays["source_indices"], dtype=np.int64)
            if shuffle:
                order = np.random.default_rng(int(seed) + int(source_indices[0])).permutation(len(source_indices))
            else:
                order = np.arange(len(source_indices), dtype=np.int64)
            for offset in range(0, len(order), self.batch_size):
                chosen = order[offset : offset + self.batch_size]
                indices = source_indices[chosen]
                expected_ids = tuple(self.hlt.jet_ids[int(index)] for index in indices)
                actual_ids = tuple(primary_ids[int(index)] for index in chosen)
                if expected_ids != actual_ids:
                    raise ValueError("tagger HLT/pseudo identity join failed")
                batches = [
                    _pseudo_batch(
                        {name: np.asarray(arrays[name])[chosen] for name in arrays},
                        metadata,
                    )
                    for (arrays, _, _), metadata in zip(shard_rows, self.metadata)
                ]
                independent = None
                pseudo = batches[0]
                if len(batches) == 2:
                    if not self.independent_roots:
                        raise ValueError("two pseudo caches are permitted only for E11")
                    pseudo, independent = _merge_independent_dual_batches(*batches)
                yield FrozenPseudoBatch(
                    hlt_tokens=np.asarray(self.hlt.tokens[indices], dtype=np.float32),
                    hlt_mask=np.asarray(self.hlt.mask[indices], dtype=bool),
                    labels=np.asarray(self.hlt.labels[indices], dtype=np.int64),
                    indices=indices,
                    pseudo=pseudo,
                    independent_roots=independent,
                )
                yielded += 1
                if self.maximum_batches is not None and yielded >= int(self.maximum_batches):
                    return


_PAIRED_TARGET_COMMON_METADATA_FIELDS = (
    "source_manifest_hash",
    "hlt_content_hash",
    "offline_content_hash",
    "jet_identity_hash",
    "label_hash",
    "label_names",
    "class_counts",
    "n_jets",
    "schema_manifest_hash",
    "root_feature_names",
    "group_feature_names",
    "particle_target_names",
    "target_semantics",
    "hlt_profile",
    "hlt_profile_version",
    "hlt_degradation_strength",
)


def _metadata_value_hash(value: Any) -> str:
    return canonical_hash({"value": value})


def _combined_target_metadata_provenance(
    branches: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    required_names = set(_PairedHierarchyTargetBatchSource.REQUIRED_HIERARCHIES)
    if set(branches) != required_names:
        raise ValueError("dual target metadata must contain exact kT and C/A branches")
    for field in _PAIRED_TARGET_COMMON_METADATA_FIELDS:
        values = {name: metadata.get(field) for name, metadata in branches.items()}
        if any(value in (None, "") for value in values.values()):
            raise ValueError(f"dual target metadata lacks common field {field}: {values}")
        if len({_metadata_value_hash(value) for value in values.values()}) != 1:
            raise ValueError(f"dual target metadata conflicts on {field}: {values}")
    branch_rows: dict[str, dict[str, Any]] = {}
    for hierarchy_name, metadata in branches.items():
        if metadata.get("grouping") != hierarchy_name:
            raise ValueError(
                f"target branch {hierarchy_name} declares {metadata.get('grouping')}"
            )
        target_hash = metadata.get("target_content_hash")
        if target_hash in (None, ""):
            raise ValueError(f"target branch {hierarchy_name} lacks aggregate hash")
        branch_rows[hierarchy_name] = {
            "grouping": hierarchy_name,
            "hierarchy_target_content_hash": target_hash,
            "grouping_algorithm_hash": canonical_hash(metadata.get("layout", {})),
            "offline_cache_content_hash": metadata.get("offline_content_hash"),
            "source_manifest_hash": metadata.get("source_manifest_hash"),
            "hlt_content_hash": metadata.get("hlt_content_hash"),
            "jet_identity_hash": metadata.get("jet_identity_hash"),
        }
    primary = branches["exclusive_kt"]
    return {
        "offline_cache_content_hash": primary["offline_content_hash"],
        "hierarchy_target_content_hash": canonical_hash(
            {
                name: row["hierarchy_target_content_hash"]
                for name, row in branch_rows.items()
            }
        ),
        "hierarchy_target_schema_hash": canonical_hash(
            {
                "root": primary.get("root_feature_names"),
                "group": primary.get("group_feature_names"),
                "particle": primary.get("particle_target_names"),
            }
        ),
        "grouping_algorithm_hash": canonical_hash(
            {
                name: row["grouping_algorithm_hash"]
                for name, row in branch_rows.items()
            }
        ),
        "root_ledger_schema_hash": canonical_hash(
            {"root_feature_names": primary.get("root_feature_names")}
        ),
        "normalization_hash": "identity_physical_units_v1",
        "dual_target_provenance": True,
        "hierarchy_branches": branch_rows,
    }


class _PairedHierarchyTargetBatchSource:
    """Lockstep kT/C-A targets for one shared-root dual reconstructor."""

    REQUIRED_HIERARCHIES = ("exclusive_kt", "cambridge_aachen")

    def __init__(
        self,
        sources: Mapping[str, AdaptiveBinaryTargetBatchSource],
    ) -> None:
        missing = [name for name in self.REQUIRED_HIERARCHIES if name not in sources]
        if missing:
            raise ValueError(f"dual target source lacks hierarchies: {missing}")
        self.sources = {
            name: sources[name] for name in self.REQUIRED_HIERARCHIES
        }
        primary = self.sources[self.REQUIRED_HIERARCHIES[0]]
        self.hlt_view = primary.hlt_view
        primary_hash = primary.hlt_view.metadata.get("hlt_content_hash")
        primary_ids = jet_identity_hash(primary.hlt_view.jet_ids)
        branch_metadata: dict[str, Mapping[str, Any]] = {}
        for name, source in self.sources.items():
            if source.split != primary.split:
                raise ValueError(f"dual target split mismatch for {name}")
            if source.hlt_view.metadata.get("hlt_content_hash") != primary_hash:
                raise ValueError(f"dual target HLT hash mismatch for {name}")
            if jet_identity_hash(source.hlt_view.jet_ids) != primary_ids:
                raise ValueError(f"dual target identity mismatch for {name}")
            branch_metadata[name] = source.metadata
        self.target_provenance = _combined_target_metadata_provenance(
            branch_metadata
        )
        self.rank = primary.rank
        self.world_size = primary.world_size

    @staticmethod
    def _merge(rows: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Any]:
        torch = require_torch()
        primary = rows[_PairedHierarchyTargetBatchSource.REQUIRED_HIERARCHIES[0]]
        for name, row in rows.items():
            for key in ("indices", "labels"):
                if not torch.equal(row[key], primary[key]):
                    raise ValueError(
                        f"dual hierarchy target batches disagree on {key} for {name}"
                    )
            if not torch.equal(row["hlt_mask"], primary["hlt_mask"]):
                raise ValueError(f"dual hierarchy HLT masks disagree for {name}")
            if not torch.equal(row["hlt_tokens"], primary["hlt_tokens"]):
                raise ValueError(f"dual hierarchy HLT particles disagree for {name}")
        return {
            **primary,
            "targets_by_hierarchy": {
                name: rows[name]["targets"]
                for name in _PairedHierarchyTargetBatchSource.REQUIRED_HIERARCHIES
            },
        }

    def next_batch(self) -> Mapping[str, Any]:
        return self._merge(
            {name: source.next_batch() for name, source in self.sources.items()}
        )

    def iter_epoch(self) -> Iterable[Mapping[str, Any]]:
        iterators = {
            name: iter(source.iter_epoch()) for name, source in self.sources.items()
        }
        while True:
            rows: dict[str, Mapping[str, Any]] = {}
            finished: list[str] = []
            for name, iterator in iterators.items():
                try:
                    rows[name] = next(iterator)
                except StopIteration:
                    finished.append(name)
            if finished:
                if len(finished) != len(iterators):
                    raise ValueError(
                        "dual hierarchy validation streams ended at different batches"
                    )
                return
            yield self._merge(rows)

    @property
    def last_validation_range(self):
        rows = tuple(
            source.last_validation_range for source in self.sources.values()
        )
        if any(row is None for row in rows):
            return None
        first = rows[0]
        if any(row.to_dict() != first.to_dict() for row in rows[1:]):
            raise ValueError("dual hierarchy validation ranges differ")
        return first


def _teacher_logits(root: Path, split: str, *, required: bool) -> np.ndarray | None:
    path = root / "teacher_logits" / "A4_offline_part_ceiling" / f"{split}.npz"
    if not required:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"required teacher logits are missing: {path}")
    with np.load(path, allow_pickle=False) as source:
        logits = np.asarray(source["logits"], dtype=np.float32)
        indices = np.asarray(source["source_indices"], dtype=np.int64)
    if logits.ndim != 2 or not np.array_equal(indices, np.arange(len(indices))):
        raise ValueError(f"teacher logit cache {path} is not contiguous")
    return logits


def _joint_state() -> CurriculumState:
    return CurriculumState(
        stage_index=7,
        stage_key="tagger_joint_reconstruction",
        phase=4,
        phase_name="tagger_joint_reconstruction",
        global_update=0,
        stage_update=0,
        stage_maximum_updates=1,
        active_capacity=32,
        stage_progress=1.0,
        teacher_forcing_probability=0.0,
        distribution_weight=0.25,
        complete=False,
    )


def _joint_reconstruction_loss(
    reconstructor: AdaptiveBinaryReconstructorModel,
    batch: Mapping[str, Any],
    *,
    split: str,
    validation: bool,
) -> Any:
    context = ReconstructorStepContext(
        curriculum=_joint_state(),
        split=split,
        mode="rollout",
        validation=validation,
        teacher_forcing_probability=0.0,
    )
    targets_by_hierarchy = batch.get("targets_by_hierarchy")
    if isinstance(targets_by_hierarchy, Mapping):
        shared_forward = batch.get("shared_reconstructor_forward")
        if shared_forward is None:
            shared_forward = reconstructor.prepare_shared_reconstruction_forward(
                batch["hlt_tokens"], batch["hlt_mask"]
            )
        shared_deployment = batch.get("shared_deployment_output")
        if shared_deployment is None:
            shared_deployment = (
                reconstructor.deploy_from_shared_reconstruction_forward(
                    shared_forward, evaluation_seed=24731
                )
            )
        losses = []
        for hierarchy_name in reconstructor.hierarchy_names:
            if hierarchy_name not in targets_by_hierarchy:
                raise ValueError(
                    f"joint dual reconstruction lacks {hierarchy_name} targets"
                )
            result = reconstructor_step(
                reconstructor,
                {
                    **batch,
                    "targets": targets_by_hierarchy[hierarchy_name],
                    "hierarchy_name": hierarchy_name,
                    "shared_reconstructor_forward": shared_forward,
                    "shared_deployment_output": shared_deployment,
                },
                context,
            )
            losses.append(
                compose_reconstruction_loss(
                    result, context, ReconstructionLossWeights()
                ).total
            )
        return require_torch().stack(losses).mean()
    result = reconstructor_step(reconstructor, batch, context)
    return compose_reconstruction_loss(
        result, context, ReconstructionLossWeights()
    ).total


def _set_joint_trainability(
    reconstructor: AdaptiveBinaryReconstructorModel,
    *,
    epoch: int,
    joint_from_start: bool,
) -> Mapping[str, bool]:
    if joint_from_start:
        active = set(reconstructor.module_groups())
    elif int(epoch) < 3:
        active = set()
    elif int(epoch) < 6:
        active = {"hierarchy_16", "hierarchy_32", "renderer", "distribution"}
    else:
        active = set(reconstructor.module_groups())
    result = {}
    for name, module in reconstructor.module_groups().items():
        trainable = name in active
        for parameter in module.parameters():
            parameter.requires_grad_(trainable)
        result[name] = trainable
    return result


def _move_frozen_batch(batch: FrozenPseudoBatch, device: Any) -> tuple[Any, Any, Any, PseudoViewInputs, Any]:
    torch = require_torch()
    tokens = torch.as_tensor(batch.hlt_tokens, device=device)
    mask = torch.as_tensor(batch.hlt_mask, device=device).bool()
    labels = torch.as_tensor(batch.labels, device=device).long()
    pseudo = PseudoViewInputs.from_deployable_batch(
        batch.pseudo, device=device, dtype=tokens.dtype
    )
    roots = None
    if batch.independent_roots is not None:
        roots = {
            name: torch.as_tensor(value, device=device, dtype=tokens.dtype)
            for name, value in batch.independent_roots.items()
        }
    return tokens, mask, labels, pseudo, roots


def _initialize_tagger(
    model: HierarchyAwareDualStreamTagger,
    *,
    root: Path,
    variant_name: str,
    initialization: str,
) -> Mapping[str, Any]:
    hlt_checkpoint = root / "runs" / "A0_hlt_part" / "best_model_val.pt"
    offline_checkpoint = root / "runs" / "A4_offline_part_ceiling" / "best_model_val.pt"
    hlt_initial = {
        name: value.detach().clone()
        for name, value in model.hlt_backbone.state_dict().items()
    }
    pseudo_initial = {
        name: value.detach().clone()
        for name, value in model.pseudo_backbone.state_dict().items()
    }
    if initialization == "all_tagger_branches_from_scratch":
        return {"kind": initialization, "hlt_loaded": False, "pseudo_loaded": False}
    warm = load_dual_stream_warm_starts(
        model,
        hlt_checkpoint=hlt_checkpoint,
        offline_checkpoint=offline_checkpoint,
        strict=True,
    )
    if initialization == "hlt_from_scratch":
        model.hlt_backbone.load_state_dict(hlt_initial, strict=True)
        warm.update({"hlt_loaded": False, "pseudo_loaded": True})
    elif initialization == "pseudo_from_scratch":
        model.pseudo_backbone.load_state_dict(pseudo_initial, strict=True)
        warm.update({"hlt_loaded": True, "pseudo_loaded": False})
    else:
        warm.update({"hlt_loaded": True, "pseudo_loaded": True})
    warm["kind"] = initialization
    return warm


def _copy_selected_tagger(model: Any, checkpoint: Path) -> None:
    payload = _torch_load(checkpoint)
    model.load_state_dict(_state_dict(payload), strict=True)


def _source_provenance(
    source: Any,
    *,
    target_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    hlt = source.hlt if hasattr(source, "hlt") else source.hlt_view
    metadata = hlt.metadata
    labels = np.asarray(hlt.labels, dtype=np.int64)
    result = {
        "source_manifest_hash": metadata.get("source_manifest_hash"),
        "jet_identity_hash": jet_identity_hash(hlt.jet_ids),
        "label_hash": hashlib.sha256(np.ascontiguousarray(labels).tobytes()).hexdigest(),
        "class_mapping_hash": canonical_hash({"label_names": list(LABEL_NAMES)}),
        "hlt_content_hash": metadata.get("hlt_content_hash"),
        "hlt_profile": metadata.get("hlt_profile"),
        "hlt_profile_version": metadata.get("hlt_profile_version"),
        "hlt_degradation_strength": metadata.get("hlt_degradation_strength"),
        "hlt_params_hash": metadata.get("hlt_params_hash"),
    }
    if hasattr(source, "cache_dirs") and hasattr(source, "metadata"):
        member_hashes: dict[str, str] = {}
        for cache_dir, pseudo_metadata in zip(source.cache_dirs, source.metadata):
            hierarchy_names = tuple(pseudo_metadata.get("hierarchy_names") or ())
            frontier_depths = dict(pseudo_metadata.get("frontier_depths") or {})
            expected_names = consumer_pseudo_array_names(
                hierarchy_names, frontier_depths
            )
            prediction_schema = dict(pseudo_metadata.get("prediction_schema") or {})
            missing = [name for name in expected_names if name not in prediction_schema]
            if missing:
                raise ValueError(
                    f"pseudo source {cache_dir} lacks consumer schema arrays: {missing}"
                )
            schema_hash = canonical_hash(
                {
                    "contract": ABPH_CONSUMER_PSEUDO_CONTRACT,
                    "schema_version": ABPH_CONSUMER_PSEUDO_SCHEMA_VERSION,
                    "arrays": {
                        name: prediction_schema[name]
                        for name in sorted(expected_names)
                    },
                }
            )
            declared = pseudo_metadata.get("consumer_pseudo_schema_hash")
            if declared not in (None, schema_hash):
                raise ValueError(f"pseudo source {cache_dir} consumer schema hash mismatch")
            member_hashes[cache_dir.name] = schema_hash
        result.update(
            {
                "consumer_pseudo_contract": ABPH_CONSUMER_PSEUDO_CONTRACT,
                "consumer_pseudo_schema_hash": canonical_hash(
                    {"members": member_hashes}
                ),
                "consumer_pseudo_member_schema_hashes": member_hashes,
                "consumer_only_pseudo_at_tagger_boundary": True,
            }
        )
    if target_provenance is not None:
        for key in (
            "offline_cache_content_hash",
            "hierarchy_target_content_hash",
            "hierarchy_target_schema_hash",
            "grouping_algorithm_hash",
            "root_ledger_schema_hash",
            "normalization_hash",
        ):
            value = target_provenance.get(key)
            if value in (None, ""):
                raise ValueError(f"selected reconstructor provenance lacks {key}")
            result[key] = value
        if target_provenance.get("dual_target_provenance") is True:
            branches = target_provenance.get("hierarchy_branches")
            if not isinstance(branches, Mapping):
                raise ValueError("dual target provenance lacks hierarchy branches")
            result["dual_target_provenance"] = True
            result["hierarchy_branches"] = {
                str(name): dict(value)
                for name, value in branches.items()
                if isinstance(value, Mapping)
            }
            if set(result["hierarchy_branches"]) != set(
                _PairedHierarchyTargetBatchSource.REQUIRED_HIERARCHIES
            ):
                raise ValueError("dual target provenance must contain exact kT/C/A branches")
    return result


def _combine_report_target_provenance(
    branches: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    required = set(_PairedHierarchyTargetBatchSource.REQUIRED_HIERARCHIES)
    if set(branches) != required:
        raise ValueError("dual report provenance must contain exact kT and C/A branches")
    common_fields = (
        "source_manifest_hash",
        "jet_identity_hash",
        "label_hash",
        "class_mapping_hash",
        "hlt_content_hash",
        "hlt_profile",
        "hlt_profile_version",
        "hlt_degradation_strength",
        "hlt_params_hash",
        "offline_cache_content_hash",
        "hierarchy_target_schema_hash",
        "root_ledger_schema_hash",
        "normalization_hash",
    )
    primary = branches["exclusive_kt"]
    result: dict[str, Any] = {}
    for field in common_fields:
        values = {name: value.get(field) for name, value in branches.items()}
        if any(value in (None, "") for value in values.values()):
            raise ValueError(
                f"dual hierarchy target provenance lacks {field}: {values}"
            )
        if len({_metadata_value_hash(value) for value in values.values()}) != 1:
            raise ValueError(
                f"dual hierarchy target provenance conflicts on {field}: {values}"
            )
        result[field] = primary[field]
    branch_rows: dict[str, dict[str, Any]] = {}
    for hierarchy_name, provenance in branches.items():
        target_hash = provenance.get("hierarchy_target_content_hash")
        grouping_hash = provenance.get("grouping_algorithm_hash")
        if target_hash in (None, "") or grouping_hash in (None, ""):
            raise ValueError(
                f"dual hierarchy branch {hierarchy_name} lacks target/grouping hash"
            )
        branch_rows[hierarchy_name] = {
            "grouping": hierarchy_name,
            "hierarchy_target_content_hash": target_hash,
            "grouping_algorithm_hash": grouping_hash,
            "offline_cache_content_hash": provenance["offline_cache_content_hash"],
            "source_manifest_hash": provenance["source_manifest_hash"],
            "hlt_content_hash": provenance["hlt_content_hash"],
            "jet_identity_hash": provenance["jet_identity_hash"],
        }
    result.update(
        {
            "hierarchy_target_content_hash": canonical_hash(
                {
                    name: row["hierarchy_target_content_hash"]
                    for name, row in branch_rows.items()
                }
            ),
            "grouping_algorithm_hash": canonical_hash(
                {
                    name: row["grouping_algorithm_hash"]
                    for name, row in branch_rows.items()
                }
            ),
            "dual_target_provenance": True,
            "hierarchy_branches": branch_rows,
        }
    )
    return result


def _selected_target_provenance(root: Path, cache_names: Sequence[str]) -> Mapping[str, Any]:
    names = tuple(str(name) for name in cache_names)
    dual = "E7_shared_root_dual" in names or {
        "D1_kt32_mh4_particles",
        "D2_ca32_mh4_particles",
    }.issubset(names)
    if dual:
        branches: dict[str, Mapping[str, Any]] = {}
        for hierarchy_name, run_name in (
            ("exclusive_kt", "D1_kt32_mh4_particles"),
            ("cambridge_aachen", "D2_ca32_mh4_particles"),
        ):
            payload = json.loads(
                (root / "runs" / run_name / "run_report.json").read_text(
                    encoding="utf-8"
                )
            )
            provenance = payload.get("provenance", {}).get("model_val")
            if not isinstance(provenance, Mapping):
                raise ValueError(f"{run_name} lacks model_val target provenance")
            branches[hierarchy_name] = provenance
        return _combine_report_target_provenance(branches)
    recognized = tuple(name for name in names if name in ABPH_FROZEN_SINGLE_PSEUDO_SOURCES)
    if len(recognized) != 1:
        raise ValueError(
            f"cannot select target provenance from pseudo caches {list(names)}"
        )
    source_name = recognized[0]
    report_path = root / "runs" / source_name / "run_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    provenance = payload.get("provenance", {}).get("model_val")
    if not isinstance(provenance, Mapping):
        raise ValueError(f"{source_name} lacks model_val target provenance")
    return provenance


def build_frozen_pseudo_generators(
    campaign_root: str | Path,
    source_names: Sequence[str],
    *,
    device: Any,
    smoke: bool,
) -> tuple[SelectedReconstructorPseudoGenerator, ...]:
    """Load the exact selected D1/D2/E7 source family in frozen eval mode."""

    root = Path(campaign_root)
    names = tuple(str(name) for name in source_names)
    if names == ("E7_shared_root_dual",):
        kt_path = root / "runs" / "D1_kt32_mh4_particles" / "best_model_val.pt"
        ca_path = root / "runs" / "D2_ca32_mh4_particles" / "best_model_val.pt"
        model = build_shared_root_dual_reconstructor(
            kt_path, ca_path, device=device, smoke=bool(smoke)
        )
        return (
            SelectedReconstructorPseudoGenerator(
                model,
                source_name=names[0],
                checkpoint_hashes={
                    "D1_kt32_mh4_particles": _sha256_file(kt_path),
                    "D2_ca32_mh4_particles": _sha256_file(ca_path),
                },
                device=device,
            ),
        )
    allowed = ABPH_FROZEN_SINGLE_PSEUDO_SOURCES
    if not names or any(name not in allowed for name in names):
        raise ValueError(f"unsupported frozen pseudo source family: {names}")
    result = []
    for name in names:
        path = root / "runs" / name / "best_model_val.pt"
        model = load_selected_reconstructor(
            path,
            variant_name=name,
            device=device,
            smoke=bool(smoke),
        )
        result.append(
            SelectedReconstructorPseudoGenerator(
                model,
                source_name=name,
                checkpoint_hashes={name: _sha256_file(path)},
                device=device,
            )
        )
    return tuple(result)


def _active_ram_workspace(rank: int) -> RankLocalWorkspace:
    root = os.environ.get("ABPH_RAM_WORKSPACE")
    if not root:
        raise RuntimeError(
            "streaming frozen pseudo sources require the verified ABPH_RAM_WORKSPACE"
        )
    return RankLocalWorkspace(
        root,
        job_id=str(os.environ.get("SLURM_JOB_ID") or os.environ.get("ABPH_JOB_ID") or "local"),
        rank=int(rank),
        create=False,
    )


def build_frozen_reconstructor_ram_source(
    campaign_root: str | Path,
    source_names: Sequence[str],
    *,
    split: str,
    batch_size: int,
    device: Any,
    smoke: bool,
    independent_roots: bool = False,
    maximum_batches: int | None = None,
    generators: Sequence[SelectedReconstructorPseudoGenerator] | None = None,
    workspace: RankLocalWorkspace | None = None,
    execution_mode: str | None = None,
) -> FrozenReconstructorRamSource:
    rank = 0 if smoke else int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    world_size = 1 if smoke else int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
    active_workspace = workspace or _active_ram_workspace(rank)
    selected_generators = tuple(generators or build_frozen_pseudo_generators(
        campaign_root, source_names, device=device, smoke=smoke
    ))
    shard_size = int(os.environ.get("ABPH_PSEUDO_RAM_SHARD_SIZE", "2048"))
    generation_batch_size = int(
        os.environ.get("ABPH_PSEUDO_GENERATION_BATCH_SIZE", str(batch_size))
    )
    lru_raw = int(os.environ.get("ABPH_PSEUDO_LRU_BYTES", "0"))
    return FrozenReconstructorRamSource(
        hlt_cache_dir=Path(campaign_root) / "inputs" / "hlt_cache",
        generators=selected_generators,
        split=split,
        batch_size=int(batch_size),
        shard_size=shard_size,
        generation_batch_size=generation_batch_size,
        execution_mode=(
            str(execution_mode)
            if execution_mode is not None
            else os.environ.get("ABPH_PSEUDO_EXECUTION_MODE", "auto")
        ),
        lru_capacity_bytes=(None if lru_raw <= 0 else lru_raw),
        workspace=active_workspace,
        rank=rank,
        world_size=world_size,
        independent_roots=independent_roots,
        maximum_batches=maximum_batches,
    )


def train_tagger_variant(args: Any, resolved: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    """Train one registry-resolved E/F/G0/G1 model with real pseudo inputs."""

    torch = require_torch()
    root = Path(args.campaign_root)
    variant_name = str(args.variant)
    run_id = str(resolved["variant"]["run_id"])
    device = resolve_device(str(args.device))
    requested_world_size = (
        1
        if bool(args.smoke)
        else int(os.environ.get("ABPH_TAGGER_DISTRIBUTED_WORLD_SIZE", "1"))
    )
    distributed_runtime = initialize_distributed_runtime(
        requested_world_size=requested_world_size, device=device
    )
    if str(getattr(device, "type", device)) == "cuda":
        device = torch.device("cuda", distributed_runtime.local_rank)
    global_batch_size = int(args.batch_size)
    if global_batch_size % distributed_runtime.world_size != 0:
        raise ValueError(
            "tagger global batch size must divide exactly across DDP ranks"
        )
    local_batch_size = global_batch_size // distributed_runtime.world_size
    seed = 24731 + 1009 * (int(args.seed_index) - 1)
    set_training_seed(seed)
    model = build_variant_hierarchy_aware_tagger(
        variant_name, num_classes=10, smoke=bool(args.smoke)
    ).to(device)
    objective_config = tagging_objective_config(variant_name)
    initialization = str(resolved["training"]["initialization"])
    initialization_report = _initialize_tagger(
        model, root=root, variant_name=variant_name, initialization=initialization
    )
    if run_id == "E1":
        e0_payload = _torch_load(
            root / "runs" / "E0_pseudo_only" / "best_model_val.pt"
        )
        e0_state = _state_dict(e0_payload)
        pseudo_state = {
            name.removeprefix("pseudo_backbone."): value
            for name, value in e0_state.items()
            if name.startswith("pseudo_backbone.")
        }
        if not pseudo_state:
            raise RuntimeError("E1 requires the independently trained E0 pseudo branch")
        model.pseudo_backbone.load_state_dict(pseudo_state, strict=True)
        initialization_report = {
            **initialization_report,
            "selected_E0_pseudo_branch_loaded": True,
        }
    if resolved["variant"]["tier"] == "F" and run_id not in {"F6", "F7", "F8", "F9"}:
        selected_name = (
            "E7_dual_hierarchy_dualcross"
            if bool(resolved["model"]["fusion"].get("dual_hierarchy"))
            else "E5_kt32_mh4_dualcross"
        )
        _copy_selected_tagger(
            model, root / "runs" / selected_name / "best_model_val.pt"
        )
        initialization_report = {
            **initialization_report,
            f"selected_{selected_name.split('_', 1)[0]}_loaded": True,
            "selected_tagger_variant": selected_name,
        }
    elif run_id == "G0":
        _copy_selected_tagger(
            model, root / "runs" / "E5_kt32_mh4_dualcross" / "best_model_val.pt"
        )
        initialization_report = {**initialization_report, "selected_E5_loaded": True}
    elif run_id == "G1":
        _copy_selected_tagger(
            model,
            root / "runs" / "E7_dual_hierarchy_dualcross" / "best_model_val.pt",
        )
        initialization_report = {**initialization_report, "selected_E7_loaded": True}

    independent = run_id == "E11"
    if bool(resolved["model"]["fusion"].get("dual_hierarchy")) and not independent:
        pseudo_variant = "E7_shared_root_dual"
        cache_names = (pseudo_variant,)
    elif independent:
        cache_names = ("D1_kt32_mh4_particles", "D2_ca32_mh4_particles")
    elif resolved["data"].get("pseudo_sources"):
        cache_names = tuple(str(name) for name in resolved["data"]["pseudo_sources"])
        if len(cache_names) != 1:
            raise ValueError("single-hierarchy supplemental taggers require one pseudo source")
        source_spec = variant_spec(cache_names[0])
        if source_spec.tier != "D" or cache_names[0] not in ABPH_FROZEN_SINGLE_PSEUDO_SOURCES:
            raise ValueError(
                f"unsupported configured pseudo source for {variant_name}: {cache_names[0]}"
            )
    elif str(resolved["model"]["hierarchy"].get("grouping")) == "cambridge_aachen":
        cache_names = ("D2_ca32_mh4_particles",)
    else:
        cache_names = ("D1_kt32_mh4_particles",)

    joint = resolved["variant"]["tier"] == "F" and run_id not in {"F1", "F4"}
    maximum_batches = 1 if bool(args.smoke) else int(os.environ.get("ABPH_MAX_TAGGER_BATCHES", "0")) or None
    frozen_train = None
    frozen_val = None
    if not joint:
        if streaming_storage_enabled():
            frozen_generators = build_frozen_pseudo_generators(
                root, cache_names, device=device, smoke=bool(args.smoke)
            )
            workspace_rank = distributed_runtime.rank
            workspace = _active_ram_workspace(workspace_rank)
            frozen_train = build_frozen_reconstructor_ram_source(
                root,
                cache_names,
                split="model_train",
                batch_size=local_batch_size,
                device=device,
                smoke=bool(args.smoke),
                independent_roots=independent,
                maximum_batches=maximum_batches,
                generators=frozen_generators,
                workspace=workspace,
            )
            frozen_val = build_frozen_reconstructor_ram_source(
                root,
                cache_names,
                split="model_val",
                batch_size=local_batch_size,
                device=device,
                smoke=bool(args.smoke),
                independent_roots=independent,
                maximum_batches=maximum_batches,
                generators=frozen_generators,
                workspace=workspace,
            )
        else:
            frozen_train = FrozenPseudoBatchSource(
                hlt_cache_dir=root / "inputs" / "hlt_cache",
                cache_dirs=tuple(root / "pseudo_predictions" / name / "model_train" for name in cache_names),
                split="model_train",
                batch_size=local_batch_size,
                independent_roots=independent,
                maximum_batches=maximum_batches,
            )
            frozen_val = FrozenPseudoBatchSource(
                hlt_cache_dir=root / "inputs" / "hlt_cache",
                cache_dirs=tuple(root / "pseudo_predictions" / name / "model_val" for name in cache_names),
                split="model_val",
                batch_size=local_batch_size,
                independent_roots=independent,
                maximum_batches=maximum_batches,
            )

    reconstructor = None
    joint_train = None
    joint_val = None
    if joint:
        target_source_kwargs = campaign_target_source_kwargs(root)
        dual_joint = bool(resolved["model"]["fusion"].get("dual_hierarchy"))
        if dual_joint:
            reconstructor = build_shared_root_dual_reconstructor(
                root / "runs" / "D1_kt32_mh4_particles" / "best_model_val.pt",
                root / "runs" / "D2_ca32_mh4_particles" / "best_model_val.pt",
                device=device,
                smoke=bool(args.smoke),
            )

            def paired_source(split: str, *, shuffle: bool):
                return _PairedHierarchyTargetBatchSource(
                    {
                        grouping: AdaptiveBinaryTargetBatchSource(
                            hlt_cache_dir=root / "inputs" / "hlt_cache",
                            target_cache_dir=root / "targets",
                            split=split,
                            grouping=grouping,
                            batch_size=local_batch_size,
                            shuffle_shards=shuffle,
                            seed=seed,
                            maximum_batches=maximum_batches,
                            rank=distributed_runtime.rank,
                            world_size=distributed_runtime.world_size,
                            **target_source_kwargs,
                        )
                        for grouping in (
                            "exclusive_kt",
                            "cambridge_aachen",
                        )
                    }
                )

            joint_train = paired_source("model_train", shuffle=True)
            joint_val = paired_source("model_val", shuffle=False)
        else:
            reconstructor = load_selected_reconstructor(
                root / "runs" / "D1_kt32_mh4_particles" / "best_model_val.pt",
                variant_name="D1_kt32_mh4_particles",
                device=device,
                smoke=bool(args.smoke),
            )
            joint_train = AdaptiveBinaryTargetBatchSource(
                hlt_cache_dir=root / "inputs" / "hlt_cache",
                target_cache_dir=root / "targets",
                split="model_train",
                grouping="exclusive_kt",
                batch_size=local_batch_size,
                shuffle_shards=True,
                seed=seed,
                maximum_batches=maximum_batches,
                rank=distributed_runtime.rank,
                world_size=distributed_runtime.world_size,
                **target_source_kwargs,
            )
            joint_val = AdaptiveBinaryTargetBatchSource(
                hlt_cache_dir=root / "inputs" / "hlt_cache",
                target_cache_dir=root / "targets",
                split="model_val",
                grouping="exclusive_kt",
                batch_size=local_batch_size,
                shuffle_shards=False,
                seed=seed,
                maximum_batches=maximum_batches,
                rank=distributed_runtime.rank,
                world_size=distributed_runtime.world_size,
                **target_source_kwargs,
            )

    teacher_train = _teacher_logits(root, "model_train", required=objective_config.requires_teacher_logits)
    teacher_val = _teacher_logits(root, "model_val", required=objective_config.requires_teacher_logits)
    epochs = 1 if bool(args.smoke) else int(os.environ.get("ABPH_TAGGER_EPOCHS", "20"))
    scheduling_jets = (
        len(joint_train.hlt_view.labels)
        if joint
        else len(frozen_train.hlt.labels)
    )
    total_steps = max(epochs * max(1, math.ceil(scheduling_jets / int(args.batch_size))), 1)
    best_accuracy = -1.0
    best_loss = float("inf")
    best_val_metrics: Mapping[str, Any] | None = None
    best_path = output_dir / "best_model_val.pt"
    curves: list[dict[str, Any]] = []
    validation_hlt_view = joint_val.hlt_view if joint else frozen_val.hlt
    joint_consumer_schema_hash: str | None = None

    def write_selected_tagger(payload: Mapping[str, Any]) -> None:
        if not streaming_storage_enabled():
            torch.save(dict(payload), best_path)
            return
        reconstructor_state = payload.get("reconstructor_state_dict")
        components = (
            {"reconstructor": reconstructor_state}
            if isinstance(reconstructor_state, Mapping)
            else {}
        )
        compact = build_compact_selected_checkpoint(
            model_state_dict=payload["model_state_dict"],
            component_state_dicts=components,
            checkpoint_role="best_model_val",
            model_metadata={
                "model_class": f"{model.__class__.__module__}.{model.__class__.__qualname__}",
                "reconstructor_hierarchy_names": payload.get(
                    "reconstructor_hierarchy_names", []
                ),
            },
            resolved_variant_config=resolved,
            resolved_variant_config_hash=str(resolved["resolved_config_hash"]),
            validation=payload.get("model_val"),
            provenance={
                "variant_name": variant_name,
                "hlt_model_val_content_hash": validation_hlt_view.metadata.get(
                    "hlt_content_hash"
                ),
                "target_provenance": _selected_target_provenance(root, cache_names),
            },
            runtime_contracts={"contract": ABPH_TAGGER_RUNTIME_CONTRACT},
            schedule_contracts={
                "epochs": int(epochs),
                "scheduler": "cosine_annealing",
            },
            extra_metadata={
                "variant_name": variant_name,
                "teacher_logits_loaded": bool(
                    payload.get("teacher_logits_loaded", False)
                ),
            },
        )
        write_selected_checkpoint(
            best_path,
            compact,
            campaign_root=root,
            artifact_role="selected_tagger_checkpoint",
            run_id=variant_name,
        )

    def forward_tagger_step(
        active_model: Any,
        active_reconstructor: Any | None,
        cpu_batch: Any,
        split: str,
        validation: bool,
    ):
        nonlocal joint_consumer_schema_hash
        if not joint:
            tokens, mask, labels, pseudo, independent_roots = _move_frozen_batch(
                cpu_batch, device
            )
            output = active_model(
                tokens,
                mask,
                pseudo,
                independent_root_ledgers=independent_roots,
            )
            return (
                output,
                labels,
                None,
                torch.as_tensor(cpu_batch.indices, device=device).long(),
            )
        if active_reconstructor is None:
            raise RuntimeError("joint tagger forward lacks its reconstructor")
        tokens = cpu_batch["hlt_tokens"].to(device)
        mask = cpu_batch["hlt_mask"].to(device).bool()
        labels = cpu_batch["labels"].to(device).long()
        shared_forward = active_reconstructor.prepare_shared_reconstruction_forward(
            tokens, mask
        )
        deployed = active_reconstructor.deploy_from_shared_reconstruction_forward(
            shared_forward, evaluation_seed=24731
        )
        pseudo = package_trainable_pseudo_views(deployed)
        candidate_schema_hash = str(
            pseudo.diagnostics["consumer_pseudo_schema_hash"]
        )
        if joint_consumer_schema_hash not in (None, candidate_schema_hash):
            raise ValueError("joint differentiable pseudo schema changed between batches")
        joint_consumer_schema_hash = candidate_schema_hash
        output = active_model(tokens, mask, pseudo)
        reconstruction = None
        if objective_config.joint_reconstruction > 0.0:
            reconstruction = _joint_reconstruction_loss(
                active_reconstructor,
                {
                    **cpu_batch,
                    "hlt_tokens": tokens,
                    "hlt_mask": mask,
                    "shared_reconstructor_forward": shared_forward,
                    "shared_deployment_output": deployed,
                },
                split=split,
                validation=validation,
            )
        return output, labels, reconstruction, cpu_batch["indices"].to(device)

    if reconstructor is not None:
        _set_joint_trainability(
            reconstructor, epoch=0, joint_from_start=(run_id == "F9")
        )
    training_module = TaggerTrainingModule(
        model,
        reconstructor,
        forward_tagger_step,
        compute_tagging_objective,
        objective_config,
    ).to(device)

    def materialize_dynamic_state() -> Any:
        if joint:
            representative_batch = next(iter(joint_train.iter_epoch()))
        else:
            representative_batch = next(
                frozen_train.iter_batches(shuffle=False, seed=seed)
            )
        return forward_tagger_step(
            model,
            reconstructor,
            representative_batch,
            "model_train",
            True,
        )

    dynamic_state_materialization = _materialize_tagger_dynamic_state(
        training_module,
        materialize_dynamic_state,
        distributed_runtime=distributed_runtime,
        seed=seed,
    )
    initial_parameter_state_hash = verify_common_parameter_state(
        distributed_runtime, training_module
    )
    parameters = list(training_module.parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=2.0e-4, betas=(0.9, 0.95), weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1.0e-5
    )
    ddp_wrapper: Any = build_tagger_ddp_wrapper(
        training_module, distributed_runtime, device=device
    )

    ddp_trainability_signature = tuple(
        (name, bool(parameter.requires_grad))
        for name, parameter in training_module.named_parameters()
    )

    def run_epoch(*, training: bool, epoch: int) -> dict[str, Any]:
        nonlocal ddp_wrapper, ddp_trainability_signature
        model.train(training)
        if reconstructor is not None:
            reconstructor.train(training)
            trainability = _set_joint_trainability(
                reconstructor, epoch=epoch, joint_from_start=(run_id == "F9")
            )
        else:
            trainability = {}
        current_signature = tuple(
            (name, bool(parameter.requires_grad))
            for name, parameter in training_module.named_parameters()
        )
        if distributed_runtime.distributed and current_signature != ddp_trainability_signature:
            barrier(distributed_runtime)
            del ddp_wrapper
            ddp_wrapper = build_tagger_ddp_wrapper(
                training_module, distributed_runtime, device=device
            )
            barrier(distributed_runtime)
            ddp_trainability_signature = current_signature

        split = "model_train" if training else "model_val"
        source = (joint_train if training else joint_val) if joint else (
            frozen_train if training else frozen_val
        )
        if training:
            expected_steps = math.ceil(scheduling_jets / global_batch_size)
            if maximum_batches is not None:
                expected_steps = min(expected_steps, int(maximum_batches))
            iterator = (
                None
                if joint
                else iter(
                    source.iter_batches(shuffle=True, seed=seed + epoch)
                )
            )
        else:
            expected_steps = None
            iterator = iter(
                source.iter_epoch()
                if joint
                else source.iter_batches(shuffle=False, seed=seed + epoch)
            )

        total_loss = 0.0
        correct = 0
        seen = 0
        batches_seen = 0
        evaluation_logits: list[np.ndarray] = []
        evaluation_labels: list[np.ndarray] = []
        evaluation_indices: list[np.ndarray] = []
        root_provenance = None
        plan_hashes: list[str] = []
        accumulator = TypedValidationAccumulator() if not training else None
        required_terms: tuple[str, ...] = ()

        while True:
            if training and batches_seen >= int(expected_steps or 0):
                break
            local_input_error: BaseException | None = None
            cpu_batch = None
            try:
                if training and joint:
                    cpu_batch = source.next_batch()
                else:
                    cpu_batch = next(iterator)
            except StopIteration:
                cpu_batch = None
            except BaseException as exc:
                local_input_error = exc
            input_ok = local_input_error is None and cpu_batch is not None
            available_ranks = all_reduce_sum_int(
                distributed_runtime, int(input_ok), device=device
            )
            if available_ranks == 0 and not training:
                break
            if available_ranks != distributed_runtime.world_size:
                summaries = gather_error_summaries(
                    distributed_runtime,
                    phase="tagger_input",
                    error=local_input_error or "rank stream ended early",
                    structural=True,
                )
                abort_distributed_runtime(distributed_runtime)
                raise RuntimeError(f"tagger input ranks diverged: {summaries}")
            assert cpu_batch is not None

            indices_cpu = (
                np.asarray(cpu_batch["indices"], dtype=np.int64)
                if joint
                else np.asarray(cpu_batch.indices, dtype=np.int64)
            )
            if training:
                immutable_range = (
                    None
                    if joint
                    else (source.rank_start, source.rank_stop)
                )
                upstream_hash = (
                    cpu_batch.get("global_batch_plan_hash") if joint else None
                )
                plan = compile_tagger_global_batch_plan(
                    distributed_runtime,
                    split=split,
                    epoch=epoch,
                    global_update=batches_seen,
                    indices=indices_cpu,
                    jet_ids=(source.hlt_view.jet_ids if joint else source.hlt.jet_ids),
                    immutable_rank_range=immutable_range,
                    upstream_plan_hash=upstream_hash,
                )
                if int(plan["global_effective_batch"]) != global_batch_size:
                    raise RuntimeError(
                        "tagger optimizer window differs from the declared global batch"
                    )
                plan_hashes.append(str(plan["plan_hash"]))

            teacher = teacher_train if training else teacher_val
            teacher_batch = (
                None
                if teacher is None
                else torch.as_tensor(teacher[indices_cpu], device=device)
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
            local_forward_error: BaseException | None = None
            tensors = None
            try:
                with torch.set_grad_enabled(training):
                    tensors = (ddp_wrapper if training else training_module)(
                        cpu_batch, teacher_batch, split, not training
                    )
                    tensors = require_tagger_tensor_mapping(tensors)
                    if not tagger_tensor_mapping_is_finite(tensors):
                        raise FloatingPointError("tagger forward produced nonfinite tensors")
            except BaseException as exc:
                local_forward_error = exc
            globally_forward_ok = all_reduce_min_bool(
                distributed_runtime,
                local_forward_error is None,
                device=device,
            )
            if not globally_forward_ok:
                summaries = gather_error_summaries(
                    distributed_runtime,
                    phase="tagger_forward",
                    error=local_forward_error,
                    structural=True,
                )
                abort_distributed_runtime(distributed_runtime)
                raise RuntimeError(f"distributed tagger forward failed: {summaries}")
            assert tensors is not None
            metadata = training_module.last_metadata or {}
            required_terms = tuple(metadata.get("required_terms") or ())
            if root_provenance is None and isinstance(
                metadata.get("root_provenance"), Mapping
            ):
                root_provenance = dict(metadata["root_provenance"])

            batch_count = int(tensors["batch_size_tensor"].item())
            batch_loss = float(tensors["total_loss"].detach().cpu())
            batch_correct = int(
                (
                    tensors["logits"].argmax(dim=-1) == tensors["labels"]
                ).sum().detach().cpu()
            )
            if training:
                try:
                    tensors["total_loss"].backward()
                except BaseException as exc:
                    abort_distributed_runtime(distributed_runtime)
                    raise RuntimeError(
                        "rank-local DDP tagger backward failed; process group aborted"
                    ) from exc
                gradients_finite = all(
                    parameter.grad is None
                    or bool(torch.isfinite(parameter.grad).all())
                    for parameter in parameters
                )
                if not all_reduce_min_bool(
                    distributed_runtime, gradients_finite, device=device
                ):
                    abort_distributed_runtime(distributed_runtime)
                    raise FloatingPointError(
                        "distributed tagger gradients are nonfinite"
                    )
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                scheduler.step()
            else:
                assert accumulator is not None
                accumulator.add_mean(
                    "loss.total", batch_loss, batch_count, selection_eligible=True
                )
                for name, value in tensors["raw_loss_terms"].items():
                    accumulator.add_mean(
                        f"loss.raw.{name}",
                        float(value.detach().cpu()),
                        batch_count,
                    )
                for name, value in tensors["weighted_loss_terms"].items():
                    accumulator.add_mean(
                        f"loss.weighted.{name}",
                        float(value.detach().cpu()),
                        batch_count,
                    )
                accumulator.add_ratio(
                    "accuracy",
                    batch_correct,
                    batch_count,
                    denominator_semantics="jets",
                )
                accumulator.finish_batch(batch_count)
                evaluation_logits.append(tensors["logits"].detach().cpu().numpy())
                evaluation_labels.append(tensors["labels"].detach().cpu().numpy())
                evaluation_indices.append(tensors["indices"].detach().cpu().numpy())
            total_loss += batch_loss * batch_count
            correct += batch_correct
            seen += batch_count
            batches_seen += 1

        if seen == 0:
            raise RuntimeError(f"{split} produced no tagger batches")
        if training:
            reduced_loss, reduced_seen = all_reduce_float64_pair(
                distributed_runtime, total_loss, seen, device=device
            )
            global_correct = all_reduce_sum_int(
                distributed_runtime, correct, device=device
            )
            result = {
                "loss": reduced_loss / reduced_seen,
                "accuracy": global_correct / reduced_seen,
                "n_jets": int(reduced_seen),
                "n_batches": batches_seen,
                "rank_batches": batches_seen * distributed_runtime.world_size,
                "reconstructor_trainability": trainability,
                "root_provenance": root_provenance,
                "global_batch_plan_hashes": plan_hashes,
                "global_effective_batch": global_batch_size,
            }
        else:
            local_evaluation_indices = np.concatenate(
                evaluation_indices, axis=0
            ).astype(np.int64, copy=False)
            gathered_indices = all_gather_objects(
                distributed_runtime, local_evaluation_indices.tolist()
            )
            if maximum_batches is not None:
                ordered_indices = tuple(
                    int(index) for row in gathered_indices for index in row
                )
                if len(ordered_indices) != len(set(ordered_indices)):
                    raise RuntimeError(
                        "bounded tagger validation overlaps identities across ranks"
                    )
                partial_jet_ids = tuple(
                    validation_hlt_view.jet_ids[index] for index in ordered_indices
                )
                local_start = sum(
                    len(gathered_indices[rank])
                    for rank in range(distributed_runtime.rank)
                )
                local_stop = local_start + len(local_evaluation_indices)
                validation_range = validation_range_row(
                    split=split,
                    rank=distributed_runtime.rank,
                    start=local_start,
                    stop=local_stop,
                    jet_ids=partial_jet_ids,
                )
                expected_validation_jet_ids = partial_jet_ids
            else:
                validation_range = (
                    source.last_validation_range
                    if joint
                    else validation_range_row(
                        split=split,
                        rank=distributed_runtime.rank,
                        start=source.rank_start,
                        stop=source.rank_stop,
                        jet_ids=source.hlt.jet_ids,
                    )
                )
                expected_validation_jet_ids = validation_hlt_view.jet_ids
            if validation_range is None:
                raise RuntimeError("tagger validation lacks immutable range coverage")
            weights = {
                name: float(
                    {
                        "label_ce": objective_config.label_ce,
                        "hlt_anchor_ce": objective_config.hlt_anchor_ce,
                        "joint_reconstruction": objective_config.joint_reconstruction,
                        "pseudo_aux_ce": objective_config.pseudo_aux_ce,
                        "hierarchy_aux_ce": objective_config.hierarchy_aux_ce,
                        "cross_view_agreement": objective_config.cross_view_agreement,
                        "offline_logit_kd": objective_config.offline_logit_kd,
                    }[name]
                )
                for name in required_terms
            }
            reduced = finalize_typed_validation(
                accumulator,
                runtime=distributed_runtime,
                device=device,
                required_losses=required_terms,
                effective_weights=weights,
                validation_range=validation_range,
                expected_jet_ids=expected_validation_jet_ids,
                split=split,
            )
            gathered = all_gather_objects(
                distributed_runtime,
                {
                    "logits": np.concatenate(evaluation_logits, axis=0),
                    "labels": np.concatenate(evaluation_labels, axis=0),
                },
            )
            detail = _detailed_eval_metrics(
                np.concatenate([row["logits"] for row in gathered], axis=0),
                np.concatenate([row["labels"] for row in gathered], axis=0),
            )
            roots = all_gather_objects(distributed_runtime, root_provenance)
            result = {
                "loss": float(reduced["selection_score"]),
                "accuracy": float(reduced["metrics"]["accuracy"]),
                "n_jets": int(reduced["n_jets"]),
                "n_batches": int(reduced["n_batches"]),
                "reconstructor_trainability": trainability,
                "root_provenance": next(
                    (row for row in roots if isinstance(row, Mapping)), None
                ),
                "validation_reduction": reduced,
                **detail,
            }
        return result

    training_started = time.perf_counter()

    # E1 is the declared untrained arithmetic mean. It still writes a selected
    # checkpoint so prediction/report tooling has one immutable artifact.
    effective_epochs = 0 if run_id == "E1" else epochs
    for epoch in range(effective_epochs):
        train_metrics = run_epoch(training=True, epoch=epoch)
        val_metrics = run_epoch(training=False, epoch=epoch)
        curves.append({"epoch": epoch, "model_train": train_metrics, "model_val": val_metrics})
        selected_this_epoch = val_metrics["accuracy"] > best_accuracy or (
            np.isclose(val_metrics["accuracy"], best_accuracy) and val_metrics["loss"] < best_loss
        )
        selected_this_epoch = bool(
            broadcast_object(
                distributed_runtime,
                selected_this_epoch if distributed_runtime.is_primary else None,
            )
        )
        if selected_this_epoch:
            best_accuracy = float(val_metrics["accuracy"])
            best_loss = float(val_metrics["loss"])
            best_val_metrics = dict(val_metrics)
            if distributed_runtime.is_primary:
                write_selected_tagger(
                    {
                        "checkpoint_contract": ABPH_TAGGER_RUNTIME_CONTRACT,
                        "checkpoint_role": "best_model_val",
                        "variant_name": variant_name,
                        "resolved_variant_config_hash": resolved["resolved_config_hash"],
                        "model_state_dict": model.state_dict(),
                        "reconstructor_state_dict": None if reconstructor is None else reconstructor.state_dict(),
                        "reconstructor_hierarchy_names": (
                            [] if reconstructor is None else list(reconstructor.hierarchy_names)
                        ),
                        "model_val": val_metrics,
                        "final_test_loaded": False,
                        "teacher_logits_loaded": objective_config.requires_teacher_logits,
                    },
                )
            barrier(distributed_runtime)
    if effective_epochs == 0:
        val_metrics = run_epoch(training=False, epoch=0)
        best_accuracy = float(val_metrics["accuracy"])
        best_loss = float(val_metrics["loss"])
        best_val_metrics = dict(val_metrics)
        if distributed_runtime.is_primary:
            write_selected_tagger(
                {
                    "checkpoint_contract": ABPH_TAGGER_RUNTIME_CONTRACT,
                    "checkpoint_role": "best_model_val",
                    "variant_name": variant_name,
                    "resolved_variant_config_hash": resolved["resolved_config_hash"],
                    "model_state_dict": model.state_dict(),
                    "reconstructor_state_dict": None,
                    "model_val": val_metrics,
                    "final_test_loaded": False,
                    "teacher_logits_loaded": False,
                },
            )
        barrier(distributed_runtime)
        curves.append({"epoch": None, "model_train": None, "model_val": val_metrics})
    if distributed_runtime.is_primary:
        (output_dir / "training_curves.json").write_text(
            json.dumps(curves, indent=2, sort_keys=True), encoding="utf-8"
        )
    barrier(distributed_runtime)
    elapsed_rows = all_gather_objects(
        distributed_runtime, float(time.perf_counter() - training_started)
    )
    training_wall_seconds = max(float(value) for value in elapsed_rows)
    all_plan_hashes = tuple(
        str(plan_hash)
        for row in curves
        if isinstance(row.get("model_train"), Mapping)
        for plan_hash in row["model_train"].get("global_batch_plan_hashes", ())
    )
    global_batch_plan_ledger = {
        "contract": "adaptive_binary_tagger_global_batch_plan_ledger_v1",
        "count": len(all_plan_hashes),
        "plan_hashes": list(all_plan_hashes),
        "aggregate_hash": canonical_hash(list(all_plan_hashes)),
    }
    target_provenance = (
        joint_val.target_provenance
        if isinstance(joint_val, _PairedHierarchyTargetBatchSource)
        else _selected_target_provenance(root, cache_names)
    )
    provenance = _source_provenance(
        joint_val if joint else frozen_val,
        target_provenance=target_provenance,
    )
    if joint:
        if not joint_consumer_schema_hash:
            raise RuntimeError("joint tagger produced no consumer pseudo schema hash")
        provenance.update(
            {
                "consumer_pseudo_contract": ABPH_CONSUMER_PSEUDO_CONTRACT,
                "consumer_pseudo_schema_hash": joint_consumer_schema_hash,
                "consumer_pseudo_member_schema_hashes": {
                    "joint_differentiable": joint_consumer_schema_hash
                },
                "consumer_only_pseudo_at_tagger_boundary": True,
            }
        )
    if best_val_metrics is None:
        raise RuntimeError("tagger training selected no model-validation metrics")
    checkpoint_hash = _sha256_file(best_path)
    checkpoint_payload = _torch_load(best_path, device="cpu")
    checkpoint_provenance = selected_checkpoint_provenance(
        best_path, checkpoint_payload
    )
    if joint:
        pseudo_execution = {
            "execution_mode": "joint_differentiable",
            "consumer_pseudo_schema_hash": joint_consumer_schema_hash,
            "pseudo_representations_written_persistently": False,
            "regenerated_shards": 0,
            "cache_hit_rate": None,
        }
    else:
        pseudo_execution = {
            "model_train": (
                frozen_train.telemetry()
                if hasattr(frozen_train, "telemetry")
                else {"execution_mode": "persistent_legacy_cache"}
            ),
            "model_val": (
                frozen_val.telemetry()
                if hasattr(frozen_val, "telemetry")
                else {"execution_mode": "persistent_legacy_cache"}
            ),
            "pseudo_representations_written_persistently": not streaming_storage_enabled(),
        }
    pseudo_execution_rows = all_gather_objects(
        distributed_runtime,
        {
            "rank": distributed_runtime.rank,
            "source": pseudo_execution,
        },
    )
    pseudo_execution = {
        "contract": "adaptive_binary_tagger_rank_local_pseudo_execution_v1",
        "world_size": distributed_runtime.world_size,
        "rank_sources": list(pseudo_execution_rows),
        "rank_source_aggregate_hash": canonical_hash(pseudo_execution_rows),
        "pseudo_representations_written_persistently": any(
            bool(row["source"].get("pseudo_representations_written_persistently"))
            for row in pseudo_execution_rows
        ),
    }
    report = {
        "contract": ABPH_TAGGER_RUNTIME_CONTRACT,
        "ok": True,
        "storage_profile": active_storage_profile(),
        "variant_name": variant_name,
        "variant": dict(resolved["variant"]),
        "resolved_variant_config_hash": resolved["resolved_config_hash"],
        "selected_checkpoint_hash": checkpoint_hash,
        "selected_checkpoint_provenance": checkpoint_provenance,
        "distributed_runtime": {
            "contract": "adaptive_binary_tagger_distributed_runtime_v1",
            "world_size": distributed_runtime.world_size,
            "backend": distributed_runtime.backend,
            "global_effective_batch": global_batch_size,
            "local_batch_size": local_batch_size,
            "training_wall_seconds": training_wall_seconds,
            "initial_parameter_state_hash": initial_parameter_state_hash,
            "dynamic_state_materialization": dynamic_state_materialization,
            "global_batch_plan_ledger": global_batch_plan_ledger,
            "validation_coverage_hash": best_val_metrics.get(
                "validation_reduction", {}
            ).get("validation_coverage", {}).get("validation_coverage_hash"),
            "rank_zero_only_persistent_writes": True,
            "bounded_failure_abort": True,
        },
        "source_git_commit": os.environ.get("FRESH_SOURCE_COMMIT", "recorded_by_slurm_run_config"),
        "source_status_hash": os.environ.get("FRESH_SOURCE_STATUS_HASH", "recorded_by_slurm_run_config"),
        "metrics": {
            "model_val": {
                "available": True,
                "accuracy": best_accuracy,
                "loss": best_loss,
                "n_jets": best_val_metrics["n_jets"],
                "macro_ovr_auc": best_val_metrics.get("macro_ovr_auc"),
                "macro_per_class_accuracy": best_val_metrics.get(
                    "macro_per_class_accuracy"
                ),
                "per_class": best_val_metrics.get("per_class"),
                "per_class_accuracy": best_val_metrics.get(
                    "per_class_accuracy"
                ),
                "confusion_matrix": best_val_metrics.get(
                    "confusion_matrix"
                ),
            }
        },
        "provenance": {
            "model_val": provenance,
            "artifact": {
                "resolved_variant_config_hash": resolved["resolved_config_hash"],
                "selected_checkpoint_hash": checkpoint_hash,
                "selected_checkpoint_content_hash": checkpoint_provenance.get(
                    "content_hash"
                ),
                "source_git_commit": os.environ.get("FRESH_SOURCE_COMMIT", "recorded_by_slurm_run_config"),
                "source_status_hash": os.environ.get("FRESH_SOURCE_STATUS_HASH", "recorded_by_slurm_run_config"),
            },
        },
        "diagnostics": {
            "model_val": {
                "offline_inputs_loaded": False,
                "teacher_logits_loaded": objective_config.requires_teacher_logits,
                "final_test_loaded": False,
                "pseudo_cache_members": list(cache_names),
                "joint_reconstructor": joint,
                "joint_reconstructor_hierarchy_names": (
                    [] if reconstructor is None else list(reconstructor.hierarchy_names)
                ),
                "dual_hierarchy_joint_training": bool(
                    reconstructor is not None and len(reconstructor.hierarchy_names) == 2
                ),
                "root_provenance": curves[-1]["model_val"].get("root_provenance"),
                "pseudo_execution": pseudo_execution,
            },
            "initialization": initialization_report,
            "objective": objective_config.to_dict(),
            "capacity": {
                "tagger_trainable_parameter_count": int(
                    sum(
                        parameter.numel()
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    )
                ),
                "reconstructor_trainable_parameter_count": int(
                    0
                    if reconstructor is None
                    else sum(
                        parameter.numel()
                        for parameter in reconstructor.parameters()
                        if parameter.requires_grad
                    )
                ),
                "combined_trainable_parameter_count": int(
                    sum(parameter.numel() for parameter in parameters)
                ),
            },
        },
    }
    if not joint:
        for source in (frozen_train, frozen_val):
            if hasattr(source, "close"):
                source.close()
    barrier(distributed_runtime)
    destroy_distributed_runtime(distributed_runtime)
    return report


__all__ = [
    "ABPH_TAGGER_RUNTIME_CONTRACT",
    "FrozenPseudoBatch",
    "FrozenPseudoBatchSource",
    "FrozenReconstructorRamSource",
    "build_frozen_pseudo_generators",
    "build_frozen_reconstructor_ram_source",
    "train_tagger_variant",
]
