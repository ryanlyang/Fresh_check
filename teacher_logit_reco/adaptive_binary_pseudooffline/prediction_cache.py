"""Deployable HLT-only pseudo-view prediction caches.

This module is deliberately strict at the deployment boundary.  It accepts a
validated HLT batch source and a model-val-selected reconstructor checkpoint,
writes bounded-size shards, and binds every shard to the active HLT cache,
split manifest, checkpoint, variant configuration, and ordered jet identities.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Callable, Iterator, Mapping, Protocol, Sequence
import uuid

import numpy as np

from jetclass_fresh.hlt_cache import hash_arrays, jet_identity_hash
from jetclass_fresh.jetclass_data import JetIdentity, LABEL_NAMES

from .config import (
    ABPH_CAMPAIGN_MODES,
    ABPH_HLT_DEGRADATION_STRENGTH,
    ABPH_HLT_PROFILE,
    ABPH_HLT_PROFILE_VERSION,
    ABPH_SPLIT_ORDER,
    abph_hlt_params_dict,
    canonical_hash,
)
from .conditional_latent import ABPH_FIXED_EVALUATION_SEED
from .hypothesis_distribution import (
    ABPH_PRIMARY_HYPOTHESIS_NAMES,
    MultiHypothesisHierarchyOutput,
)
from .inputs import AdaptiveBinaryHLTOnlyDataset, load_hlt_only_dataset
from .particle_renderer import RenderedParticleBatch
from .schemas import ABPH_MAX_PARTICLES
from .training import (
    describe_reconstructor_model,
    load_reconstructor_curriculum_checkpoint,
)


ABPH_PSEUDO_VIEW_CACHE_CONTRACT = "adaptive_binary_pseudooffline_prediction_cache_v1"
ABPH_PSEUDO_VIEW_CACHE_SET_FILENAME = "pseudo_view_cache_manifest.json"
ABPH_PSEUDO_VIEW_SCHEMA_VERSION = "1.0"

_IDENTITY_ARRAY_KEYS = (
    "labels",
    "jet_file_indices",
    "jet_entries",
    "source_indices",
)
_FRONTIER_FIELDS = (
    "ledger",
    "hidden",
    "support",
    "uncertainty",
    "mask",
    "topology",
    "parent_indices",
    "source_child_indices",
)
_PREDICTION_REQUIRED_GLOBAL_KEYS = (
    "shared_root_ledger",
    "hypothesis_latent",
    "hypothesis_prior_log_prob",
)
_PREDICTION_REQUIRED_PARTICLE_FIELDS = (
    "canonical_features",
    "side_channels",
    "four_vector",
    "mass",
    "mask",
    "group_indices",
    "local_slot_indices",
    "uncertainty",
    "slot_hidden",
)


def _json_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _update_identity_hash(hasher: Any, identity: JetIdentity) -> None:
    hasher.update(str(identity.file).encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(str(int(identity.entry)).encode("ascii"))
    hasher.update(b"\0")
    hasher.update(str(int(identity.label)).encode("ascii"))
    hasher.update(b"\n")


def _class_mapping_hash() -> str:
    return canonical_hash({"label_names": list(LABEL_NAMES)})


def _prediction_key(hierarchy: str, category: str, field: str, depth: int | None = None) -> str:
    hierarchy_name = str(hierarchy)
    if not hierarchy_name or "__" in hierarchy_name:
        raise ValueError("hierarchy names must be nonempty and cannot contain '__'")
    if category == "particle":
        return f"particle__{hierarchy_name}__{field}"
    if category == "frontier":
        if depth is None or int(depth) < 0:
            raise ValueError("frontier keys require a nonnegative depth")
        return f"frontier__{hierarchy_name}__depth_{int(depth):02d}__{field}"
    raise ValueError(f"unknown prediction category {category!r}")


def _to_numpy(value: Any) -> np.ndarray:
    tensor = value
    if hasattr(tensor, "detach"):
        tensor = tensor.detach()
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    if hasattr(tensor, "numpy"):
        tensor = tensor.numpy()
    return np.asarray(tensor)


def _stack_numpy(values: Sequence[Any], axis: int = 1) -> np.ndarray:
    if not values:
        raise ValueError("cannot stack an empty prediction sequence")
    return np.stack(tuple(_to_numpy(value) for value in values), axis=axis)


def _schema_for_arrays(arrays: Mapping[str, np.ndarray], batch_size: int) -> dict[str, Any]:
    return {
        name: {
            "dtype": str(np.asarray(array).dtype),
            "shape_after_batch": list(np.asarray(array).shape[1:]),
        }
        for name, array in sorted(arrays.items())
        if int(np.asarray(array).shape[0]) == int(batch_size)
    }


def _schema_hash(schema: Mapping[str, Any]) -> str:
    return canonical_hash(
        {
            "contract": ABPH_PSEUDO_VIEW_CACHE_CONTRACT,
            "schema_version": ABPH_PSEUDO_VIEW_SCHEMA_VERSION,
            "arrays": dict(schema),
        }
    )


@dataclass(frozen=True)
class DeployablePseudoViewBatch:
    """One HLT-only batch containing every fixed deployment hypothesis."""

    arrays: Mapping[str, np.ndarray]
    view_names: tuple[str, ...]
    hierarchy_names: tuple[str, ...]
    frontier_depths: Mapping[str, int]
    diagnostics: Mapping[str, Any]

    @property
    def batch_size(self) -> int:
        return int(np.asarray(self.arrays["shared_root_ledger"]).shape[0])

    def validate(self) -> dict[str, Any]:
        arrays = {str(name): np.asarray(value) for name, value in self.arrays.items()}
        if tuple(self.view_names) != ABPH_PRIMARY_HYPOTHESIS_NAMES:
            raise ValueError(
                "deployable pseudo views must be mean plus the four fixed stochastic hypotheses"
            )
        if not self.hierarchy_names or len(self.hierarchy_names) != len(set(self.hierarchy_names)):
            raise ValueError("prediction batch requires unique hierarchy names")
        missing = [name for name in _PREDICTION_REQUIRED_GLOBAL_KEYS if name not in arrays]
        if missing:
            raise KeyError(f"prediction batch is missing global arrays: {missing}")
        root = arrays["shared_root_ledger"]
        if root.ndim != 2 or root.shape[0] <= 0:
            raise ValueError("shared_root_ledger must have shape [B, root_dim]")
        batch_size = int(root.shape[0])
        n_views = len(self.view_names)
        for name, array in arrays.items():
            if array.ndim == 0 or int(array.shape[0]) != batch_size:
                raise ValueError(f"prediction array {name!r} does not share the batch axis")
            if array.dtype.kind in "fc" and not np.isfinite(array).all():
                raise ValueError(f"prediction array {name!r} contains nonfinite values")
        for name in ("hypothesis_latent", "hypothesis_prior_log_prob"):
            if arrays[name].ndim < 2 or int(arrays[name].shape[1]) != n_views:
                raise ValueError(f"{name} does not contain all fixed views")
        for hierarchy in self.hierarchy_names:
            for field in _PREDICTION_REQUIRED_PARTICLE_FIELDS:
                key = _prediction_key(hierarchy, "particle", field)
                if key not in arrays:
                    raise KeyError(f"prediction batch is missing {key}")
                if arrays[key].ndim < 3 or int(arrays[key].shape[1]) != n_views:
                    raise ValueError(f"{key} does not contain all fixed views")
            particle_mask = arrays[_prediction_key(hierarchy, "particle", "mask")]
            if particle_mask.dtype != np.bool_:
                raise TypeError(f"particle mask for {hierarchy} must be bool")
            if particle_mask.ndim != 3 or int(particle_mask.shape[2]) != ABPH_MAX_PARTICLES:
                raise ValueError(
                    f"particle mask for {hierarchy} must preserve the {ABPH_MAX_PARTICLES}-slot contract"
                )
            depth_count = int(self.frontier_depths.get(hierarchy, -1))
            if depth_count <= 0:
                raise ValueError(f"hierarchy {hierarchy!r} has no frontier depths")
            for depth in range(depth_count):
                for field in _FRONTIER_FIELDS:
                    key = _prediction_key(hierarchy, "frontier", field, depth)
                    if key not in arrays:
                        raise KeyError(f"prediction batch is missing {key}")
                    if arrays[key].ndim < 3 or int(arrays[key].shape[1]) != n_views:
                        raise ValueError(f"{key} does not contain all fixed views")
                frontier_mask = arrays[
                    _prediction_key(hierarchy, "frontier", "mask", depth)
                ]
                if frontier_mask.dtype != np.bool_:
                    raise TypeError(f"frontier mask for {hierarchy}/depth{depth} must be bool")
        if self.diagnostics.get("offline_inputs_loaded") is not False:
            raise ValueError("deployable prediction must attest offline_inputs_loaded=false")
        if self.diagnostics.get("teacher_logits_loaded") is not False:
            raise ValueError("deployable prediction must attest teacher_logits_loaded=false")
        if self.diagnostics.get("offline_target_selected_hypothesis") is not False:
            raise ValueError(
                "deployable prediction must attest offline_target_selected_hypothesis=false"
            )
        schema = _schema_for_arrays(arrays, batch_size)
        return {
            "ok": True,
            "batch_size": batch_size,
            "view_names": list(self.view_names),
            "hierarchy_names": list(self.hierarchy_names),
            "frontier_depths": {
                name: int(self.frontier_depths[name]) for name in self.hierarchy_names
            },
            "schema": schema,
            "schema_hash": _schema_hash(schema),
        }

    def slice(self, start: int, stop: int) -> "DeployablePseudoViewBatch":
        begin = int(start)
        end = int(stop)
        if not 0 <= begin < end <= self.batch_size:
            raise ValueError("prediction batch slice is invalid")
        return DeployablePseudoViewBatch(
            arrays={name: np.asarray(value)[begin:end] for name, value in self.arrays.items()},
            view_names=self.view_names,
            hierarchy_names=self.hierarchy_names,
            frontier_depths=dict(self.frontier_depths),
            diagnostics=dict(self.diagnostics),
        )


def package_deployable_pseudo_views(
    hierarchy_output: MultiHypothesisHierarchyOutput,
    rendered_views: Mapping[str, Sequence[RenderedParticleBatch]],
) -> DeployablePseudoViewBatch:
    """Convert rollout and renderer outputs into the stable Step-9 tensor schema."""

    hypotheses = tuple(hierarchy_output.hypotheses)
    view_names = tuple(hypothesis.identity.name for hypothesis in hypotheses)
    if view_names != ABPH_PRIMARY_HYPOTHESIS_NAMES:
        raise ValueError(f"unexpected deployment hypothesis identities: {view_names}")
    hierarchy_names = tuple(str(name) for name in rendered_views)
    expected_hierarchies = tuple(hypotheses[0].hierarchy_outputs)
    if set(hierarchy_names) != set(expected_hierarchies):
        raise ValueError("rendered hierarchy names differ from rollout hierarchy names")
    arrays: dict[str, np.ndarray] = {
        "shared_root_ledger": _to_numpy(hierarchy_output.shared_root_ledger),
        "hypothesis_latent": _stack_numpy(
            [hypothesis.latent for hypothesis in hypotheses]
        ),
        "hypothesis_prior_log_prob": _stack_numpy(
            [hypothesis.prior_log_prob for hypothesis in hypotheses]
        ),
    }
    frontier_depths: dict[str, int] = {}
    shared_root = arrays["shared_root_ledger"]
    for hierarchy in hierarchy_names:
        rendered = tuple(rendered_views[hierarchy])
        if len(rendered) != len(hypotheses):
            raise ValueError(f"rendered view count differs for hierarchy {hierarchy}")
        for view_index, particle_batch in enumerate(rendered):
            diagnostics = dict(particle_batch.diagnostics)
            if diagnostics.get("offline_inputs_consumed") is not False:
                raise ValueError("particle renderer consumed offline inputs in deployable mode")
            if diagnostics.get("hlt_only_deployment_inputs") is not True:
                raise ValueError("particle renderer lacks its HLT-only deployment attestation")
            if int(diagnostics.get("hypothesis_index", -1)) != view_index:
                raise ValueError("particle renderer hypothesis identity mismatch")
        for field in _PREDICTION_REQUIRED_PARTICLE_FIELDS:
            arrays[_prediction_key(hierarchy, "particle", field)] = _stack_numpy(
                [getattr(particle_batch, field) for particle_batch in rendered]
            )
        outputs = [hypothesis.hierarchy_outputs[hierarchy] for hypothesis in hypotheses]
        for output in outputs:
            candidate_root = _to_numpy(output.root_frontier.ledger)
            if candidate_root.shape != (shared_root.shape[0], 1, shared_root.shape[1]):
                raise ValueError("hierarchy root frontier shape differs from the shared root")
            if not np.array_equal(candidate_root[:, 0], shared_root):
                raise RuntimeError("hierarchy rollout does not use the exact shared compiled root")
        depth_count = 1 + len(outputs[0].levels)
        if any(1 + len(output.levels) != depth_count for output in outputs):
            raise ValueError("hierarchy depth differs across deployment hypotheses")
        frontier_depths[hierarchy] = depth_count
        for depth in range(depth_count):
            frontiers = [
                output.root_frontier if depth == 0 else output.levels[depth - 1].next_frontier
                for output in outputs
            ]
            for field in _FRONTIER_FIELDS:
                arrays[_prediction_key(hierarchy, "frontier", field, depth)] = _stack_numpy(
                    [getattr(frontier, field) for frontier in frontiers]
                )
    batch = DeployablePseudoViewBatch(
        arrays=arrays,
        view_names=view_names,
        hierarchy_names=hierarchy_names,
        frontier_depths=frontier_depths,
        diagnostics={
            "contract": ABPH_PSEUDO_VIEW_CACHE_CONTRACT,
            "offline_inputs_loaded": False,
            "teacher_logits_loaded": False,
            "offline_target_selected_hypothesis": False,
            "fixed_evaluation_hypotheses": True,
            "shared_root_exact_across_views_and_hierarchies": bool(
                hierarchy_output.diagnostics.get(
                    "exact_root_identity_across_all_hypotheses_and_hierarchies"
                )
            ),
        },
    )
    report = batch.validate()
    if not batch.diagnostics["shared_root_exact_across_views_and_hierarchies"]:
        raise RuntimeError("deployable views do not preserve the exact shared root")
    if not report["ok"]:  # pragma: no cover - validate raises before this branch
        raise RuntimeError("deployable prediction packaging failed")
    return batch


@dataclass(frozen=True)
class DeployableHLTBatch:
    tokens: np.ndarray
    mask: np.ndarray
    labels: np.ndarray
    indices: np.ndarray
    jet_ids: tuple[JetIdentity, ...]

    @property
    def batch_size(self) -> int:
        return int(np.asarray(self.labels).shape[0])

    def validate(self) -> None:
        tokens = np.asarray(self.tokens)
        mask = np.asarray(self.mask)
        labels = np.asarray(self.labels)
        indices = np.asarray(self.indices)
        size = int(labels.shape[0])
        if tokens.ndim != 3 or mask.shape != tokens.shape[:2] or mask.dtype != np.bool_:
            raise ValueError("HLT batch tokens/mask shapes are invalid")
        if int(tokens.shape[1]) != ABPH_MAX_PARTICLES:
            raise ValueError(
                f"HLT batch must preserve the {ABPH_MAX_PARTICLES}-particle input contract"
            )
        if labels.shape != (size,) or indices.shape != (size,) or len(self.jet_ids) != size:
            raise ValueError("HLT batch metadata axes are inconsistent")
        if not np.array_equal(
            labels.astype(np.int64, copy=False),
            np.asarray([identity.label for identity in self.jet_ids], dtype=np.int64),
        ):
            raise ValueError("HLT batch labels differ from jet identities")
        if not np.isfinite(tokens[mask]).all():
            raise ValueError("HLT batch contains nonfinite valid particles")


class DeployableHLTBatchSource(Protocol):
    provenance: Mapping[str, Any]
    streaming: bool
    resident_bytes: int

    def __len__(self) -> int: ...
    def iter_batches(self) -> Iterator[DeployableHLTBatch]: ...


class HLTOnlyArrayBatchSource:
    """Validated HLT-only arrays exposed as deterministic bounded batches."""

    def __init__(
        self,
        dataset: AdaptiveBinaryHLTOnlyDataset,
        *,
        batch_size: int,
    ) -> None:
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        validation = dict(dataset.metadata["input_validation"])
        self.provenance = {
            "source_manifest_hash": validation["source_manifest_hash"],
            "hlt_content_hash": validation["hlt_content_hash"],
            "jet_identity_hash": jet_identity_hash(dataset.jet_ids),
            "label_hash": hashlib.sha256(
                np.ascontiguousarray(dataset.labels, dtype=np.int64).tobytes()
            ).hexdigest(),
            "class_mapping_hash": _class_mapping_hash(),
            "hlt_profile": validation["hlt_profile"],
            "hlt_profile_version": validation["hlt_profile_version"],
            "hlt_degradation_strength": validation["hlt_degradation_strength"],
            "hlt_params_hash": canonical_hash(abph_hlt_params_dict()),
            "split": str(dataset.metadata["split"]),
            "n_jets": len(dataset),
            "target_cache_loaded": False,
            "offline_inputs_loaded": False,
            "teacher_logits_loaded": False,
            "source_kind": "validated_hlt_cache_arrays",
        }
        self.streaming = True
        self.resident_bytes = int(
            np.asarray(dataset.tokens).nbytes
            + np.asarray(dataset.mask).nbytes
            + np.asarray(dataset.labels).nbytes
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def iter_batches(self) -> Iterator[DeployableHLTBatch]:
        for start in range(0, len(self), self.batch_size):
            stop = min(start + self.batch_size, len(self))
            batch = DeployableHLTBatch(
                tokens=np.asarray(self.dataset.tokens[start:stop]),
                mask=np.asarray(self.dataset.mask[start:stop], dtype=bool),
                labels=np.asarray(self.dataset.labels[start:stop], dtype=np.int64),
                indices=np.arange(start, stop, dtype=np.int64),
                jet_ids=tuple(self.dataset.jet_ids[start:stop]),
            )
            batch.validate()
            yield batch


def load_hlt_prediction_source(
    manifest_path: str | Path,
    hlt_cache_dir: str | Path,
    split: str,
    *,
    campaign_mode: str = "highdata",
    expected_split_sizes: Mapping[str, int] | None = None,
    expected_n_jets: int | None = None,
    max_jets: int | None = None,
    batch_size: int = 256,
) -> HLTOnlyArrayBatchSource:
    dataset = load_hlt_only_dataset(
        manifest_path,
        hlt_cache_dir,
        split,
        campaign_mode=campaign_mode,
        expected_split_sizes=expected_split_sizes,
        expected_n_jets=expected_n_jets,
        max_jets=max_jets,
        verify_hash=True,
    )
    return HLTOnlyArrayBatchSource(dataset, batch_size=batch_size)


@dataclass(frozen=True)
class DeployablePseudoViewCacheConfig:
    checkpoint_path: str | Path
    output_cache_dir: str | Path
    split: str
    resolved_variant_config_hash: str
    campaign_mode: str = "highdata"
    device: str = "cuda"
    evaluation_seed: int = ABPH_FIXED_EVALUATION_SEED
    shard_size: int = 1_024
    feature_dtype: str = "float16"
    overwrite: bool = False
    reuse_existing: bool = False
    source_git_commit: str | None = None
    source_status_hash: str | None = None
    offline_cache_dir: str | Path | None = None
    target_cache_dir: str | Path | None = None
    teacher_logits_dir: str | Path | None = None

    def __post_init__(self) -> None:
        if self.split not in ABPH_SPLIT_ORDER:
            raise ValueError(f"unknown prediction split {self.split!r}")
        if self.campaign_mode not in ABPH_CAMPAIGN_MODES:
            raise ValueError(f"unknown campaign mode {self.campaign_mode!r}")
        if int(self.shard_size) <= 0:
            raise ValueError("shard_size must be positive")
        if np.dtype(self.feature_dtype) not in (np.dtype("float16"), np.dtype("float32")):
            raise ValueError("feature_dtype must be float16 or float32")
        if not str(self.resolved_variant_config_hash):
            raise ValueError("resolved_variant_config_hash is required")
        privileged = {
            "offline_cache_dir": self.offline_cache_dir,
            "target_cache_dir": self.target_cache_dir,
            "teacher_logits_dir": self.teacher_logits_dir,
        }
        mounted = [name for name, value in privileged.items() if value is not None]
        if mounted:
            raise ValueError(
                "deployable pseudo prediction rejects privileged paths: " + ", ".join(mounted)
            )
        if self.overwrite and self.reuse_existing:
            raise ValueError("overwrite and reuse_existing are mutually exclusive")


DeployablePredictor = Callable[
    [Any, DeployableHLTBatch, int, str], DeployablePseudoViewBatch
]


def _default_predictor(
    model: Any,
    batch: DeployableHLTBatch,
    evaluation_seed: int,
    device: str,
) -> DeployablePseudoViewBatch:
    method = getattr(model, "predict_deployable_pseudo_views", None)
    if method is None or not callable(method):
        raise TypeError(
            "model must implement predict_deployable_pseudo_views or a predictor callback must be supplied"
        )
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - model inference requires torch
        raise RuntimeError("PyTorch is required for deployable pseudo prediction") from exc
    result = method(
        tokens=torch.as_tensor(batch.tokens, device=device),
        mask=torch.as_tensor(batch.mask, device=device),
        indices=torch.as_tensor(batch.indices, device=device),
        evaluation_seed=int(evaluation_seed),
    )
    if not isinstance(result, DeployablePseudoViewBatch):
        raise TypeError("predict_deployable_pseudo_views returned the wrong type")
    return result


def _validate_source(source: DeployableHLTBatchSource, split: str) -> dict[str, Any]:
    provenance = dict(source.provenance)
    required = (
        "source_manifest_hash",
        "hlt_content_hash",
        "jet_identity_hash",
        "label_hash",
        "class_mapping_hash",
        "hlt_profile",
        "hlt_profile_version",
        "hlt_degradation_strength",
        "hlt_params_hash",
        "split",
        "n_jets",
    )
    missing = [name for name in required if not provenance.get(name) and provenance.get(name) != 0]
    if missing:
        raise ValueError(f"HLT prediction source lacks provenance: {missing}")
    if provenance["split"] != split or int(provenance["n_jets"]) != len(source):
        raise ValueError("HLT prediction source split/size provenance mismatch")
    if provenance["hlt_profile"] != ABPH_HLT_PROFILE:
        raise ValueError("HLT prediction source profile mismatch")
    if provenance["hlt_profile_version"] != ABPH_HLT_PROFILE_VERSION:
        raise ValueError("HLT prediction source profile version mismatch")
    if abs(float(provenance["hlt_degradation_strength"]) - ABPH_HLT_DEGRADATION_STRENGTH) > 1.0e-12:
        raise ValueError("HLT prediction source degradation strength mismatch")
    if provenance["hlt_params_hash"] != canonical_hash(abph_hlt_params_dict()):
        raise ValueError("HLT prediction source parameter hash mismatch")
    if provenance["class_mapping_hash"] != _class_mapping_hash():
        raise ValueError("HLT prediction source class mapping mismatch")
    for key in ("target_cache_loaded", "offline_inputs_loaded", "teacher_logits_loaded"):
        if provenance.get(key) is not False:
            raise ValueError(f"HLT prediction source must attest {key}=false")
    if not bool(getattr(source, "streaming", False)):
        raise ValueError("deployable prediction requires a streaming batch source")
    return provenance


def _prepare_checkpoint(
    model: Any,
    module_groups: Mapping[str, Any],
    config: DeployablePseudoViewCacheConfig,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    checkpoint_path = Path(config.checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    payload = load_reconstructor_curriculum_checkpoint(
        checkpoint_path, device="cpu", require_selected=True
    )
    if payload.get("checkpoint_role") != "best_model_val":
        raise ValueError(
            "deployable pseudo prediction requires the completed best_model_val checkpoint"
        )
    checkpoint_metadata = dict(payload.get("model_metadata") or {})
    expected_model = describe_reconstructor_model(model, module_groups)
    if checkpoint_metadata.get("model_metadata_hash") != expected_model["model_metadata_hash"]:
        raise ValueError("reconstructor model metadata differs from the selected checkpoint")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    checkpoint_provenance = dict(payload.get("provenance") or {})
    git_commit = config.source_git_commit or checkpoint_provenance.get("source_git_commit")
    status_hash = config.source_status_hash or checkpoint_provenance.get("source_status_hash")
    if not git_commit or not status_hash:
        raise ValueError("deployable cache requires source git commit and dirty-status hash")
    checkpoint_config = dict(payload.get("config") or {})
    return payload, {
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "checkpoint_role": payload["checkpoint_role"],
        "checkpoint_model_metadata_hash": expected_model["model_metadata_hash"],
        "checkpoint_config_hash": canonical_hash(checkpoint_config),
        "source_git_commit": str(git_commit),
        "source_status_hash": str(status_hash),
    }


def _cast_prediction_arrays(
    arrays: Mapping[str, np.ndarray], feature_dtype: np.dtype
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name, value in arrays.items():
        array = np.asarray(value)
        if array.dtype.kind == "f":
            array = array.astype(feature_dtype, copy=False)
        elif array.dtype == np.bool_:
            array = array.astype(bool, copy=False)
        elif array.dtype.kind in "iu":
            array = array.astype(np.int64 if array.dtype.itemsize > 4 else np.int32, copy=False)
        else:
            raise TypeError(f"unsupported prediction dtype for {name}: {array.dtype}")
        result[str(name)] = np.ascontiguousarray(array)
    return result


def _identity_arrays_for_shard(
    jet_ids: Sequence[JetIdentity],
    file_lookup: dict[str, int],
    jet_files: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    file_indices = []
    entries = []
    for identity in jet_ids:
        filename = str(identity.file)
        if filename not in file_lookup:
            file_lookup[filename] = len(jet_files)
            jet_files.append(filename)
        file_indices.append(file_lookup[filename])
        entries.append(int(identity.entry))
    return np.asarray(file_indices, dtype=np.int32), np.asarray(entries, dtype=np.int64)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sidecar_hash(payload: Mapping[str, Any]) -> str:
    return _json_hash({key: value for key, value in payload.items() if key != "sidecar_hash"})


def _cache_hash_payload(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: metadata[key]
        for key in (
            "cache_contract",
            "schema_version",
            "split",
            "campaign_mode",
            "n_jets",
            "n_shards",
            "feature_dtype",
            "view_names",
            "hierarchy_names",
            "frontier_depths",
            "prediction_schema_hash",
            "source_manifest_hash",
            "hlt_content_hash",
            "jet_identity_hash",
            "label_hash",
            "class_mapping_hash",
            "hlt_profile",
            "hlt_profile_version",
            "hlt_degradation_strength",
            "hlt_params_hash",
            "checkpoint_sha256",
            "checkpoint_role",
            "checkpoint_model_metadata_hash",
            "checkpoint_config_hash",
            "resolved_variant_config_hash",
            "evaluation_seed",
            "source_git_commit",
            "source_status_hash",
            "final_test_attestation",
            "shards",
        )
    }


def generate_deployable_pseudo_view_cache(
    model: Any,
    module_groups: Mapping[str, Any],
    source: DeployableHLTBatchSource,
    config: DeployablePseudoViewCacheConfig,
    *,
    predictor: DeployablePredictor | None = None,
) -> dict[str, Any]:
    """Generate one split without ever accepting offline or teacher inputs."""

    source_provenance = _validate_source(source, config.split)
    _, checkpoint = _prepare_checkpoint(model, module_groups, config)
    output_dir = Path(config.output_cache_dir)
    expected_bindings = {
        "source_manifest_hash": source_provenance["source_manifest_hash"],
        "hlt_content_hash": source_provenance["hlt_content_hash"],
        "jet_identity_hash": source_provenance["jet_identity_hash"],
        "label_hash": source_provenance["label_hash"],
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "resolved_variant_config_hash": config.resolved_variant_config_hash,
    }
    if output_dir.exists():
        if config.reuse_existing:
            report = audit_deployable_pseudo_view_cache(
                output_dir, expected_bindings=expected_bindings, verify_hash=True
            )
            if not report["ok"]:
                raise ValueError("existing pseudo-view cache is stale or partial: " + "; ".join(report["problems"]))
            return {**report["metadata"], "reused": True}
        if not config.overwrite:
            raise FileExistsError(output_dir)

    temporary_dir = output_dir.parent / f".{output_dir.name}.{uuid.uuid4().hex}.building"
    if temporary_dir.exists():  # effectively impossible, but fail safely
        raise FileExistsError(temporary_dir)
    temporary_dir.mkdir(parents=True, exist_ok=False)
    feature_dtype = np.dtype(config.feature_dtype)
    prediction_function = predictor or _default_predictor
    schema: dict[str, Any] | None = None
    schema_sha: str | None = None
    view_names: tuple[str, ...] | None = None
    hierarchy_names: tuple[str, ...] | None = None
    frontier_depths: dict[str, int] | None = None
    buffer_arrays: dict[str, list[np.ndarray]] = {}
    buffer_labels: list[np.ndarray] = []
    buffer_indices: list[np.ndarray] = []
    buffer_jet_ids: list[JetIdentity] = []
    buffered_rows = 0
    maximum_buffered_rows = 0
    maximum_buffered_bytes = 0
    maximum_flush_live_bytes_estimate = 0
    buffered_payload_bytes = 0
    maximum_prediction_batch_rows = 0
    maximum_prediction_batch_bytes = 0
    source_rows_seen = 0
    output_bytes = 0
    shard_reports: list[dict[str, Any]] = []
    jet_files: list[str] = []
    file_lookup: dict[str, int] = {}
    identity_hasher = hashlib.sha256()
    label_hasher = hashlib.sha256()
    class_counts = np.zeros(len(LABEL_NAMES), dtype=np.int64)

    try:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - inference requires torch
            raise RuntimeError("PyTorch is required for deployable pseudo prediction") from exc
        device = torch.device(config.device)
        model.to(device)
        model.eval()

        def flush_shard() -> None:
            nonlocal buffered_rows, buffered_payload_bytes, maximum_buffered_bytes
            nonlocal maximum_flush_live_bytes_estimate, output_bytes
            if buffered_rows == 0:
                return
            shard_index = len(shard_reports)
            start = sum(int(row["n_jets"]) for row in shard_reports)
            stop = start + buffered_rows
            predictions = {
                name: np.concatenate(parts, axis=0)
                for name, parts in buffer_arrays.items()
            }
            labels = np.concatenate(buffer_labels, axis=0).astype(np.int64, copy=False)
            indices = np.concatenate(buffer_indices, axis=0).astype(np.int64, copy=False)
            if not np.array_equal(indices, np.arange(start, stop, dtype=np.int64)):
                raise ValueError("prediction source indices are not contiguous and ordered")
            file_indices, entries = _identity_arrays_for_shard(
                buffer_jet_ids, file_lookup, jet_files
            )
            arrays = {
                **predictions,
                "labels": labels,
                "jet_file_indices": file_indices,
                "jet_entries": entries,
                "source_indices": indices,
            }
            local_bytes = int(sum(np.asarray(value).nbytes for value in arrays.values()))
            maximum_buffered_bytes = max(maximum_buffered_bytes, local_bytes)
            maximum_flush_live_bytes_estimate = max(
                maximum_flush_live_bytes_estimate,
                int(buffered_payload_bytes + local_bytes),
            )
            content_hash = hash_arrays(arrays)
            filename = f"shard_{shard_index:06d}.npz"
            array_path = temporary_dir / filename
            np.savez_compressed(array_path, **arrays)
            file_sha = _sha256_file(array_path)
            output_bytes += int(array_path.stat().st_size)
            local_identity_hash = jet_identity_hash(tuple(buffer_jet_ids))
            sidecar: dict[str, Any] = {
                "cache_contract": ABPH_PSEUDO_VIEW_CACHE_CONTRACT,
                "schema_version": ABPH_PSEUDO_VIEW_SCHEMA_VERSION,
                "split": config.split,
                "shard_index": shard_index,
                "start": start,
                "stop": stop,
                "n_jets": buffered_rows,
                "filename": filename,
                "content_hash": content_hash,
                "file_sha256": file_sha,
                "jet_identity_hash": local_identity_hash,
                "source_manifest_hash": source_provenance["source_manifest_hash"],
                "hlt_content_hash": source_provenance["hlt_content_hash"],
                "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                "resolved_variant_config_hash": config.resolved_variant_config_hash,
                "prediction_schema_hash": schema_sha,
                "offline_inputs_loaded": False,
                "teacher_logits_loaded": False,
                "offline_target_selected_hypothesis": False,
            }
            sidecar["sidecar_hash"] = _sidecar_hash(sidecar)
            sidecar_filename = f"shard_{shard_index:06d}.json"
            _atomic_json(temporary_dir / sidecar_filename, sidecar)
            output_bytes += int((temporary_dir / sidecar_filename).stat().st_size)
            shard_reports.append(
                {
                    "filename": filename,
                    "sidecar_filename": sidecar_filename,
                    "shard_index": shard_index,
                    "start": start,
                    "stop": stop,
                    "n_jets": buffered_rows,
                    "content_hash": content_hash,
                    "file_sha256": file_sha,
                    "sidecar_hash": sidecar["sidecar_hash"],
                    "jet_identity_hash": local_identity_hash,
                }
            )
            buffer_arrays.clear()
            buffer_labels.clear()
            buffer_indices.clear()
            buffer_jet_ids.clear()
            buffered_rows = 0
            buffered_payload_bytes = 0

        for hlt_batch in source.iter_batches():
            hlt_batch.validate()
            if hlt_batch.batch_size <= 0:
                raise ValueError("prediction source yielded an empty batch")
            expected_indices = np.arange(
                source_rows_seen, source_rows_seen + hlt_batch.batch_size, dtype=np.int64
            )
            if not np.array_equal(np.asarray(hlt_batch.indices, dtype=np.int64), expected_indices):
                raise ValueError("prediction source changed jet order or yielded duplicate indices")
            with torch.no_grad():
                predicted = prediction_function(
                    model,
                    hlt_batch,
                    int(config.evaluation_seed),
                    str(device),
                )
            if not isinstance(predicted, DeployablePseudoViewBatch):
                raise TypeError("deployable predictor returned the wrong type")
            batch_report = predicted.validate()
            if predicted.batch_size != hlt_batch.batch_size:
                raise ValueError("prediction batch size differs from HLT batch size")
            cast_arrays = _cast_prediction_arrays(predicted.arrays, feature_dtype)
            cast_batch = DeployablePseudoViewBatch(
                arrays=cast_arrays,
                view_names=predicted.view_names,
                hierarchy_names=predicted.hierarchy_names,
                frontier_depths=predicted.frontier_depths,
                diagnostics=predicted.diagnostics,
            )
            cast_report = cast_batch.validate()
            if schema is None:
                schema = cast_report["schema"]
                schema_sha = cast_report["schema_hash"]
                view_names = cast_batch.view_names
                hierarchy_names = cast_batch.hierarchy_names
                frontier_depths = dict(cast_batch.frontier_depths)
            elif (
                cast_report["schema_hash"] != schema_sha
                or cast_batch.view_names != view_names
                or cast_batch.hierarchy_names != hierarchy_names
                or dict(cast_batch.frontier_depths) != frontier_depths
            ):
                raise ValueError("deployable prediction schema changed between batches")
            prediction_bytes = int(sum(value.nbytes for value in cast_arrays.values()))
            maximum_prediction_batch_rows = max(
                maximum_prediction_batch_rows, hlt_batch.batch_size
            )
            maximum_prediction_batch_bytes = max(
                maximum_prediction_batch_bytes, prediction_bytes
            )

            batch_offset = 0
            while batch_offset < hlt_batch.batch_size:
                available = int(config.shard_size) - buffered_rows
                take = min(available, hlt_batch.batch_size - batch_offset)
                part = cast_batch.slice(batch_offset, batch_offset + take)
                for name, value in part.arrays.items():
                    buffer_arrays.setdefault(name, []).append(np.asarray(value))
                labels_part = np.asarray(
                    hlt_batch.labels[batch_offset : batch_offset + take], dtype=np.int64
                )
                indices_part = np.asarray(
                    hlt_batch.indices[batch_offset : batch_offset + take], dtype=np.int64
                )
                ids_part = tuple(hlt_batch.jet_ids[batch_offset : batch_offset + take])
                buffer_labels.append(labels_part)
                buffer_indices.append(indices_part)
                buffer_jet_ids.extend(ids_part)
                buffered_payload_bytes += int(
                    sum(np.asarray(value).nbytes for value in part.arrays.values())
                    + labels_part.nbytes
                    + indices_part.nbytes
                )
                for identity in ids_part:
                    _update_identity_hash(identity_hasher, identity)
                label_hasher.update(np.ascontiguousarray(labels_part).tobytes())
                class_counts += np.bincount(labels_part, minlength=len(LABEL_NAMES))
                buffered_rows += take
                maximum_buffered_rows = max(maximum_buffered_rows, buffered_rows)
                batch_offset += take
                if buffered_rows == int(config.shard_size):
                    flush_shard()
            source_rows_seen += hlt_batch.batch_size
        flush_shard()

        if source_rows_seen != len(source) or source_rows_seen == 0:
            raise ValueError("prediction source did not yield its declared number of jets")
        if identity_hasher.hexdigest() != source_provenance["jet_identity_hash"]:
            raise ValueError("generated prediction identities differ from HLT source provenance")
        if label_hasher.hexdigest() != source_provenance["label_hash"]:
            raise ValueError("generated prediction labels differ from HLT source provenance")
        if schema is None or schema_sha is None or view_names is None or hierarchy_names is None:
            raise RuntimeError("deployable prediction produced no schema")

        metadata: dict[str, Any] = {
            "cache_contract": ABPH_PSEUDO_VIEW_CACHE_CONTRACT,
            "schema_version": ABPH_PSEUDO_VIEW_SCHEMA_VERSION,
            "split": config.split,
            "campaign_mode": config.campaign_mode,
            "n_jets": source_rows_seen,
            "n_shards": len(shard_reports),
            "shard_size": int(config.shard_size),
            "feature_dtype": str(feature_dtype),
            "view_names": list(view_names),
            "hierarchy_names": list(hierarchy_names),
            "frontier_depths": frontier_depths,
            "prediction_schema": schema,
            "prediction_schema_hash": schema_sha,
            "source_manifest_hash": source_provenance["source_manifest_hash"],
            "hlt_content_hash": source_provenance["hlt_content_hash"],
            "jet_identity_hash": source_provenance["jet_identity_hash"],
            "label_hash": source_provenance["label_hash"],
            "class_mapping_hash": source_provenance["class_mapping_hash"],
            "class_counts": {
                name: int(class_counts[index]) for index, name in enumerate(LABEL_NAMES)
            },
            "jet_files": jet_files,
            "hlt_profile": source_provenance["hlt_profile"],
            "hlt_profile_version": source_provenance["hlt_profile_version"],
            "hlt_degradation_strength": source_provenance["hlt_degradation_strength"],
            "hlt_params_hash": source_provenance["hlt_params_hash"],
            **checkpoint,
            "resolved_variant_config_hash": config.resolved_variant_config_hash,
            "evaluation_seed": int(config.evaluation_seed),
            "final_test_attestation": {
                "offline_inputs_loaded": False,
                "teacher_logits_loaded": False,
                "offline_target_selected_hypothesis": False,
                "fusion_fitted_on_final_test": False,
            },
            "memory_audit": {
                "streaming_batch_source": True,
                "source_kind": source_provenance.get("source_kind", "custom_streaming_source"),
                "source_resident_bytes": int(getattr(source, "resident_bytes", 0)),
                "full_prediction_materialized": False,
                "maximum_prediction_batch_rows": maximum_prediction_batch_rows,
                "maximum_prediction_batch_bytes": maximum_prediction_batch_bytes,
                "maximum_buffered_rows": maximum_buffered_rows,
                "maximum_buffered_bytes": maximum_buffered_bytes,
                "maximum_flush_live_bytes_estimate": maximum_flush_live_bytes_estimate,
                "configured_shard_size": int(config.shard_size),
                "bounded_buffer_verified": maximum_buffered_rows <= int(config.shard_size),
                "compressed_output_bytes": output_bytes,
            },
            "shards": shard_reports,
            "complete": True,
        }
        metadata["prediction_content_hash"] = _json_hash(_cache_hash_payload(metadata))
        _atomic_json(temporary_dir / ABPH_PSEUDO_VIEW_CACHE_SET_FILENAME, metadata)

        if output_dir.exists():
            if not config.overwrite:
                raise FileExistsError(output_dir)
            shutil.rmtree(output_dir)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary_dir, output_dir)
        audit = audit_deployable_pseudo_view_cache(
            output_dir, expected_bindings=expected_bindings, verify_hash=True
        )
        if not audit["ok"]:
            raise RuntimeError("new pseudo-view cache failed its own audit: " + "; ".join(audit["problems"]))
        return {**metadata, "reused": False}
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise


def _load_cache_metadata(cache_dir: str | Path) -> dict[str, Any]:
    path = Path(cache_dir) / ABPH_PSEUDO_VIEW_CACHE_SET_FILENAME
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("cache_contract") != ABPH_PSEUDO_VIEW_CACHE_CONTRACT:
        raise ValueError("pseudo-view cache contract mismatch")
    if metadata.get("schema_version") != ABPH_PSEUDO_VIEW_SCHEMA_VERSION:
        raise ValueError("pseudo-view cache schema version mismatch")
    expected = _json_hash(_cache_hash_payload(metadata))
    if metadata.get("prediction_content_hash") != expected:
        raise ValueError("pseudo-view aggregate content hash mismatch")
    return metadata


def _identities_from_arrays(
    jet_files: Sequence[str], arrays: Mapping[str, np.ndarray]
) -> tuple[JetIdentity, ...]:
    labels = np.asarray(arrays["labels"], dtype=np.int64)
    file_indices = np.asarray(arrays["jet_file_indices"], dtype=np.int64)
    entries = np.asarray(arrays["jet_entries"], dtype=np.int64)
    if labels.shape != file_indices.shape or labels.shape != entries.shape:
        raise ValueError("cached identity arrays have inconsistent shapes")
    result = []
    for file_index, entry, label in zip(file_indices, entries, labels):
        if not 0 <= int(file_index) < len(jet_files):
            raise ValueError("cached jet file index is out of range")
        result.append(
            JetIdentity(
                file=str(jet_files[int(file_index)]),
                entry=int(entry),
                label=int(label),
            )
        )
    return tuple(result)


def _validate_arrays_against_schema(
    arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any], n_rows: int
) -> None:
    expected_keys = set(metadata["prediction_schema"]) | set(_IDENTITY_ARRAY_KEYS)
    if set(arrays) != expected_keys:
        missing = sorted(expected_keys - set(arrays))
        extra = sorted(set(arrays) - expected_keys)
        raise ValueError(f"pseudo-view shard keys differ; missing={missing}, extra={extra}")
    for name, specification in metadata["prediction_schema"].items():
        array = np.asarray(arrays[name])
        expected_shape = (int(n_rows), *tuple(specification["shape_after_batch"]))
        if tuple(array.shape) != expected_shape or str(array.dtype) != specification["dtype"]:
            raise ValueError(
                f"pseudo-view array {name} shape/dtype {array.shape}/{array.dtype} "
                f"!= {expected_shape}/{specification['dtype']}"
            )


def audit_deployable_pseudo_view_cache(
    cache_dir: str | Path,
    *,
    expected_bindings: Mapping[str, Any] | None = None,
    verify_hash: bool = True,
) -> dict[str, Any]:
    """Recompute all hashes and reject partial, stale, reordered, or extra shards."""

    root = Path(cache_dir)
    problems: list[str] = []
    try:
        metadata = _load_cache_metadata(root)
    except Exception as exc:
        return {"ok": False, "problems": [str(exc)], "metadata": {}}
    bindings = dict(expected_bindings or {})
    for name, expected in bindings.items():
        if metadata.get(name) != expected:
            problems.append(f"{name} mismatch: {metadata.get(name)!r} != {expected!r}")
    if metadata.get("complete") is not True:
        problems.append("cache is not marked complete")
    attestation = dict(metadata.get("final_test_attestation") or {})
    for name in (
        "offline_inputs_loaded",
        "teacher_logits_loaded",
        "offline_target_selected_hypothesis",
        "fusion_fitted_on_final_test",
    ):
        if attestation.get(name) is not False:
            problems.append(f"final-test attestation {name} is not false")
    memory = dict(metadata.get("memory_audit") or {})
    if memory.get("full_prediction_materialized") is not False:
        problems.append("memory audit does not attest streaming prediction")
    if memory.get("bounded_buffer_verified") is not True:
        problems.append("memory audit does not attest a bounded shard buffer")
    if metadata.get("prediction_schema_hash") != _schema_hash(
        dict(metadata.get("prediction_schema") or {})
    ):
        problems.append("prediction schema hash mismatch")
    if tuple(metadata.get("view_names") or ()) != ABPH_PRIMARY_HYPOTHESIS_NAMES:
        problems.append("fixed deployment hypothesis identities mismatch")
    hierarchy_names = tuple(metadata.get("hierarchy_names") or ())
    frontier_depths = dict(metadata.get("frontier_depths") or {})
    if not hierarchy_names or set(frontier_depths) != set(hierarchy_names):
        problems.append("hierarchy names/depth metadata is incomplete")
    if metadata.get("class_mapping_hash") != _class_mapping_hash():
        problems.append("class mapping hash mismatch")
    if metadata.get("hlt_profile") != ABPH_HLT_PROFILE:
        problems.append("HLT profile mismatch")
    if metadata.get("hlt_profile_version") != ABPH_HLT_PROFILE_VERSION:
        problems.append("HLT profile version mismatch")
    if abs(
        float(metadata.get("hlt_degradation_strength", -1.0))
        - ABPH_HLT_DEGRADATION_STRENGTH
    ) > 1.0e-12:
        problems.append("HLT degradation strength mismatch")
    if metadata.get("hlt_params_hash") != canonical_hash(abph_hlt_params_dict()):
        problems.append("HLT parameter hash mismatch")

    shard_records = list(metadata.get("shards") or [])
    if len(shard_records) != int(metadata.get("n_shards", -1)) or not shard_records:
        problems.append("shard record count is invalid")
    expected_files = {ABPH_PSEUDO_VIEW_CACHE_SET_FILENAME}
    for row in shard_records:
        expected_files.add(str(row.get("filename")))
        expected_files.add(str(row.get("sidecar_filename")))
    observed_files = {path.name for path in root.iterdir() if path.is_file()}
    missing_files = sorted(expected_files - observed_files)
    extra_shards = sorted(
        name
        for name in observed_files - expected_files
        if name.startswith("shard_") and (name.endswith(".npz") or name.endswith(".json"))
    )
    if missing_files:
        problems.append(f"missing cache files: {missing_files}")
    if extra_shards:
        problems.append(f"unexpected stale shard files: {extra_shards}")

    identity_hasher = hashlib.sha256()
    label_hasher = hashlib.sha256()
    expected_start = 0
    rows_seen = 0
    for index, row in enumerate(shard_records):
        prefix = f"shard {index}"
        try:
            if int(row.get("shard_index", -1)) != index:
                raise ValueError("metadata index is noncontiguous")
            start = int(row["start"])
            stop = int(row["stop"])
            n_rows = int(row["n_jets"])
            if start != expected_start or stop - start != n_rows or n_rows <= 0:
                raise ValueError("row interval is noncontiguous or empty")
            array_path = root / row["filename"]
            sidecar_path = root / row["sidecar_filename"]
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            if sidecar.get("sidecar_hash") != _sidecar_hash(sidecar):
                raise ValueError("sidecar hash mismatch")
            for name in (
                "sidecar_hash",
                "content_hash",
                "file_sha256",
                "jet_identity_hash",
                "shard_index",
                "start",
                "stop",
                "n_jets",
                "filename",
            ):
                if sidecar.get(name) != row.get(name):
                    raise ValueError(f"sidecar {name} differs from aggregate metadata")
            for name in (
                "source_manifest_hash",
                "hlt_content_hash",
                "checkpoint_sha256",
                "resolved_variant_config_hash",
                "prediction_schema_hash",
            ):
                if sidecar.get(name) != metadata.get(name):
                    raise ValueError(f"sidecar {name} binding mismatch")
            for name in (
                "offline_inputs_loaded",
                "teacher_logits_loaded",
                "offline_target_selected_hypothesis",
            ):
                if sidecar.get(name) is not False:
                    raise ValueError(f"sidecar {name} is not false")
            if verify_hash and _sha256_file(array_path) != row["file_sha256"]:
                raise ValueError("NPZ file hash mismatch")
            with np.load(array_path, allow_pickle=False) as source:
                arrays = {name: np.asarray(source[name]) for name in source.files}
            _validate_arrays_against_schema(arrays, metadata, n_rows)
            if verify_hash and hash_arrays(arrays) != row["content_hash"]:
                raise ValueError("numeric content hash mismatch")
            indices = np.asarray(arrays["source_indices"], dtype=np.int64)
            if not np.array_equal(indices, np.arange(start, stop, dtype=np.int64)):
                raise ValueError("source indices are reordered or duplicated")
            identities = _identities_from_arrays(metadata["jet_files"], arrays)
            if jet_identity_hash(identities) != row["jet_identity_hash"]:
                raise ValueError("shard identity hash mismatch")
            labels = np.asarray(arrays["labels"], dtype=np.int64)
            if not np.array_equal(labels, np.asarray([item.label for item in identities])):
                raise ValueError("cached labels differ from identities")
            for identity in identities:
                _update_identity_hash(identity_hasher, identity)
            label_hasher.update(np.ascontiguousarray(labels).tobytes())
            expected_start = stop
            rows_seen += n_rows
        except Exception as exc:
            problems.append(f"{prefix}: {exc}")
    if rows_seen != int(metadata.get("n_jets", -1)):
        problems.append("shards do not cover the declared number of jets")
    if identity_hasher.hexdigest() != metadata.get("jet_identity_hash"):
        problems.append("full ordered jet identity hash mismatch")
    if label_hasher.hexdigest() != metadata.get("label_hash"):
        problems.append("full label hash mismatch")
    return {
        "ok": not problems,
        "problems": problems,
        "metadata": metadata,
        "n_jets_verified": rows_seen,
        "n_shards_verified": len(shard_records),
    }


def require_deployable_pseudo_view_cache(
    cache_dir: str | Path,
    *,
    expected_bindings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    report = audit_deployable_pseudo_view_cache(
        cache_dir, expected_bindings=expected_bindings, verify_hash=True
    )
    if not report["ok"]:
        raise ValueError("pseudo-view cache validation failed: " + "; ".join(report["problems"]))
    return report


@dataclass(frozen=True)
class ReassembledPseudoViewCache:
    arrays: Mapping[str, np.ndarray]
    labels: np.ndarray
    indices: np.ndarray
    jet_ids: tuple[JetIdentity, ...]
    metadata: Mapping[str, Any]


def iter_deployable_pseudo_view_shards(
    cache_dir: str | Path,
    *,
    verify_hash: bool = True,
) -> Iterator[tuple[dict[str, np.ndarray], tuple[JetIdentity, ...], Mapping[str, Any]]]:
    report = audit_deployable_pseudo_view_cache(cache_dir, verify_hash=verify_hash)
    if not report["ok"]:
        raise ValueError("pseudo-view cache validation failed: " + "; ".join(report["problems"]))
    metadata = report["metadata"]
    root = Path(cache_dir)
    for row in metadata["shards"]:
        with np.load(root / row["filename"], allow_pickle=False) as source:
            arrays = {name: np.asarray(source[name]) for name in source.files}
        identities = _identities_from_arrays(metadata["jet_files"], arrays)
        yield arrays, identities, row


def reassemble_deployable_pseudo_view_cache(
    cache_dir: str | Path,
    *,
    verify_hash: bool = True,
) -> ReassembledPseudoViewCache:
    """Materialize a cache for tests/small analyses after a full streaming audit."""

    metadata = _load_cache_metadata(cache_dir)
    prediction_parts: dict[str, list[np.ndarray]] = {
        name: [] for name in metadata["prediction_schema"]
    }
    labels: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    identities: list[JetIdentity] = []
    for arrays, shard_ids, _ in iter_deployable_pseudo_view_shards(
        cache_dir, verify_hash=verify_hash
    ):
        for name in prediction_parts:
            prediction_parts[name].append(np.asarray(arrays[name]))
        labels.append(np.asarray(arrays["labels"], dtype=np.int64))
        indices.append(np.asarray(arrays["source_indices"], dtype=np.int64))
        identities.extend(shard_ids)
    return ReassembledPseudoViewCache(
        arrays={name: np.concatenate(parts, axis=0) for name, parts in prediction_parts.items()},
        labels=np.concatenate(labels, axis=0),
        indices=np.concatenate(indices, axis=0),
        jet_ids=tuple(identities),
        metadata=metadata,
    )


__all__ = [
    "ABPH_PSEUDO_VIEW_CACHE_CONTRACT",
    "ABPH_PSEUDO_VIEW_CACHE_SET_FILENAME",
    "ABPH_PSEUDO_VIEW_SCHEMA_VERSION",
    "DeployableHLTBatch",
    "DeployableHLTBatchSource",
    "DeployablePredictor",
    "DeployablePseudoViewBatch",
    "DeployablePseudoViewCacheConfig",
    "HLTOnlyArrayBatchSource",
    "ReassembledPseudoViewCache",
    "audit_deployable_pseudo_view_cache",
    "generate_deployable_pseudo_view_cache",
    "iter_deployable_pseudo_view_shards",
    "load_hlt_prediction_source",
    "package_deployable_pseudo_views",
    "reassemble_deployable_pseudo_view_cache",
    "require_deployable_pseudo_view_cache",
]
