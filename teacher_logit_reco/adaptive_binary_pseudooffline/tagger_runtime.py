"""Production E/F/G0/G1 training runtime for adaptive-binary pseudo views."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

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
from .production import (
    AdaptiveBinaryReconstructorModel,
    AdaptiveBinaryTargetBatchSource,
    build_shared_root_dual_reconstructor,
    load_selected_reconstructor,
    package_trainable_pseudo_views,
    reconstructor_step,
)
from .tagger import (
    HierarchyAwareDualStreamTagger,
    PseudoViewInputs,
    build_variant_hierarchy_aware_tagger,
    load_dual_stream_warm_starts,
)
from .tagging_objectives import compute_tagging_objective, tagging_objective_config
from .training import (
    CurriculumState,
    ReconstructionLossWeights,
    ReconstructorStepContext,
    compose_reconstruction_loss,
)


ABPH_TAGGER_RUNTIME_CONTRACT = "adaptive_binary_pseudooffline_tagger_runtime_v1"


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
    torch = require_torch()
    try:
        payload = torch.load(Path(path), map_location=device, weights_only=False)
    except TypeError:  # pragma: no cover - old research PyTorch
        payload = torch.load(Path(path), map_location=device)
    if not isinstance(payload, Mapping):
        raise TypeError(f"checkpoint {path} is not a mapping")
    return payload


def _state_dict(payload: Mapping[str, Any]) -> Mapping[str, Any]:
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
    return DeployablePseudoViewBatch(
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
    merged = DeployablePseudoViewBatch(
        arrays=arrays,
        view_names=first.view_names,
        hierarchy_names=(kt_name, ca_name),
        frontier_depths={
            kt_name: int(first.frontier_depths[kt_name]),
            ca_name: int(second.frontier_depths[ca_name]),
        },
        diagnostics=dict(first.diagnostics),
    )
    merged.validate()
    return merged, independent


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
        for name, source in self.sources.items():
            if source.split != primary.split:
                raise ValueError(f"dual target split mismatch for {name}")
            if source.hlt_view.metadata.get("hlt_content_hash") != primary_hash:
                raise ValueError(f"dual target HLT hash mismatch for {name}")
            if jet_identity_hash(source.hlt_view.jet_ids) != primary_ids:
                raise ValueError(f"dual target identity mismatch for {name}")

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
    hlt = source.hlt if isinstance(source, FrozenPseudoBatchSource) else source.hlt_view
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
    return result


def _selected_target_provenance(root: Path, cache_names: Sequence[str]) -> Mapping[str, Any]:
    source_name = next(
        (name for name in cache_names if name.startswith("D1") or name.startswith("D2")),
        "D1_kt32_mh4_particles",
    )
    if source_name == "E7_shared_root_dual":
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
        common_fields = (
            "offline_cache_content_hash",
            "hierarchy_target_schema_hash",
            "root_ledger_schema_hash",
            "normalization_hash",
        )
        primary = branches["exclusive_kt"]
        result: dict[str, Any] = {}
        for field in common_fields:
            values = {name: value.get(field) for name, value in branches.items()}
            if None in values.values() or len(set(values.values())) != 1:
                raise ValueError(
                    f"dual hierarchy target provenance conflicts on {field}: {values}"
                )
            result[field] = primary[field]
        for field in ("hierarchy_target_content_hash", "grouping_algorithm_hash"):
            values = {name: value.get(field) for name, value in branches.items()}
            if None in values.values():
                raise ValueError(f"dual hierarchy target provenance lacks {field}")
            result[field] = canonical_hash(values)
        result["hierarchy_branches"] = {
            name: {
                field: provenance.get(field)
                for field in (
                    "hierarchy_target_content_hash",
                    "grouping_algorithm_hash",
                )
            }
            for name, provenance in branches.items()
        }
        return result
    report_path = root / "runs" / source_name / "run_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    provenance = payload.get("provenance", {}).get("model_val")
    if not isinstance(provenance, Mapping):
        raise ValueError(f"{source_name} lacks model_val target provenance")
    return provenance


def train_tagger_variant(args: Any, resolved: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    """Train one registry-resolved E/F/G0/G1 model with real pseudo inputs."""

    torch = require_torch()
    root = Path(args.campaign_root)
    variant_name = str(args.variant)
    run_id = str(resolved["variant"]["run_id"])
    device = resolve_device(str(args.device))
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
    elif str(resolved["model"]["hierarchy"].get("grouping")) == "cambridge_aachen":
        cache_names = ("D2_ca32_mh4_particles",)
    else:
        cache_names = ("D1_kt32_mh4_particles",)

    maximum_batches = 1 if bool(args.smoke) else int(os.environ.get("ABPH_MAX_TAGGER_BATCHES", "0")) or None
    frozen_train = FrozenPseudoBatchSource(
        hlt_cache_dir=root / "inputs" / "hlt_cache",
        cache_dirs=tuple(root / "pseudo_predictions" / name / "model_train" for name in cache_names),
        split="model_train",
        batch_size=int(args.batch_size),
        independent_roots=independent,
        maximum_batches=maximum_batches,
    )
    frozen_val = FrozenPseudoBatchSource(
        hlt_cache_dir=root / "inputs" / "hlt_cache",
        cache_dirs=tuple(root / "pseudo_predictions" / name / "model_val" for name in cache_names),
        split="model_val",
        batch_size=int(args.batch_size),
        independent_roots=independent,
        maximum_batches=maximum_batches,
    )

    joint = resolved["variant"]["tier"] == "F" and run_id not in {"F1", "F4"}
    reconstructor = None
    joint_train = None
    joint_val = None
    if joint:
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
                            batch_size=int(args.batch_size),
                            shuffle_shards=shuffle,
                            seed=seed,
                            maximum_batches=maximum_batches,
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
                batch_size=int(args.batch_size),
                shuffle_shards=True,
                seed=seed,
                maximum_batches=maximum_batches,
            )
            joint_val = AdaptiveBinaryTargetBatchSource(
                hlt_cache_dir=root / "inputs" / "hlt_cache",
                target_cache_dir=root / "targets",
                split="model_val",
                grouping="exclusive_kt",
                batch_size=int(args.batch_size),
                shuffle_shards=False,
                seed=seed,
                maximum_batches=maximum_batches,
            )

    teacher_train = _teacher_logits(root, "model_train", required=objective_config.requires_teacher_logits)
    teacher_val = _teacher_logits(root, "model_val", required=objective_config.requires_teacher_logits)
    parameters = list(model.parameters()) + ([] if reconstructor is None else list(reconstructor.parameters()))
    optimizer = torch.optim.AdamW(parameters, lr=2.0e-4, betas=(0.9, 0.95), weight_decay=0.01)
    epochs = 1 if bool(args.smoke) else int(os.environ.get("ABPH_TAGGER_EPOCHS", "20"))
    total_steps = max(epochs * max(1, math.ceil(len(frozen_train.hlt.labels) / int(args.batch_size))), 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1.0e-5
    )
    best_accuracy = -1.0
    best_loss = float("inf")
    best_val_metrics: Mapping[str, Any] | None = None
    best_path = output_dir / "best_model_val.pt"
    curves: list[dict[str, Any]] = []

    def forward_joint(cpu_batch: Mapping[str, Any], *, split: str, validation: bool):
        tokens = cpu_batch["hlt_tokens"].to(device)
        mask = cpu_batch["hlt_mask"].to(device).bool()
        labels = cpu_batch["labels"].to(device).long()
        deployed = reconstructor.deploy(tokens, mask, evaluation_seed=24731)
        pseudo = package_trainable_pseudo_views(deployed)
        output = model(tokens, mask, pseudo)
        reconstruction = None
        if objective_config.joint_reconstruction > 0.0:
            reconstruction = _joint_reconstruction_loss(
                reconstructor, {**cpu_batch, "hlt_tokens": tokens, "hlt_mask": mask}, split=split, validation=validation
            )
        return output, labels, reconstruction, cpu_batch["indices"].to(device)

    def run_epoch(*, training: bool, epoch: int) -> dict[str, Any]:
        model.train(training)
        if reconstructor is not None:
            reconstructor.train(training)
            trainability = _set_joint_trainability(
                reconstructor, epoch=epoch, joint_from_start=(run_id == "F9")
            )
        else:
            trainability = {}
        total_loss = 0.0
        correct = 0
        seen = 0
        batches_seen = 0
        evaluation_logits: list[np.ndarray] = []
        evaluation_labels: list[np.ndarray] = []
        root_provenance = None
        split = "model_train" if training else "model_val"
        if joint:
            iterator: Iterable[Any] = (
                iter(lambda: joint_train.next_batch(), None)
                if training
                else joint_val.iter_epoch()
            )
        else:
            source = frozen_train if training else frozen_val
            iterator = source.iter_batches(shuffle=training, seed=seed + epoch)
        maximum = maximum_batches
        for cpu_batch in iterator:
            if maximum is not None and batches_seen >= maximum:
                break
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                if joint:
                    output, labels, reconstruction, indices = forward_joint(
                        cpu_batch, split=split, validation=not training
                    )
                else:
                    tokens, mask, labels, pseudo, independent_roots = _move_frozen_batch(cpu_batch, device)
                    output = model(
                        tokens,
                        mask,
                        pseudo,
                        independent_root_ledgers=independent_roots,
                    )
                    reconstruction = None
                    indices = torch.as_tensor(cpu_batch.indices, device=device).long()
                teacher = teacher_train if training else teacher_val
                teacher_batch = None if teacher is None else torch.as_tensor(teacher[indices.detach().cpu().numpy()], device=device)
                objective = compute_tagging_objective(
                    output,
                    labels,
                    objective_config,
                    split=split,
                    reconstruction_loss=reconstruction,
                    auxiliary_logits=output.auxiliary_logits,
                    teacher_logits=teacher_batch,
                )
                if training:
                    objective.total.backward()
                    torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                    optimizer.step()
                    scheduler.step()
            candidate_root = output.diagnostics.get("root_provenance")
            if isinstance(candidate_root, Mapping) and root_provenance is None:
                root_provenance = dict(candidate_root)
            batch_size = int(labels.shape[0])
            total_loss += float(objective.total.detach().cpu()) * batch_size
            correct += int((output.logits.argmax(dim=-1) == labels).sum().detach().cpu())
            if not training:
                evaluation_logits.append(output.logits.detach().cpu().numpy())
                evaluation_labels.append(labels.detach().cpu().numpy())
            seen += batch_size
            batches_seen += 1
            if joint and training and maximum is None and batches_seen >= math.ceil(len(joint_train.hlt_view.labels) / int(args.batch_size)):
                break
        if seen == 0:
            raise RuntimeError(f"{split} produced no tagger batches")
        result = {
            "loss": total_loss / seen,
            "accuracy": correct / seen,
            "n_jets": seen,
            "n_batches": batches_seen,
            "reconstructor_trainability": trainability,
            "root_provenance": root_provenance,
        }
        if not training:
            result.update(
                _detailed_eval_metrics(
                    np.concatenate(evaluation_logits, axis=0),
                    np.concatenate(evaluation_labels, axis=0),
                )
            )
        return result

    # E1 is the declared untrained arithmetic mean. It still writes a selected
    # checkpoint so prediction/report tooling has one immutable artifact.
    effective_epochs = 0 if run_id == "E1" else epochs
    for epoch in range(effective_epochs):
        train_metrics = run_epoch(training=True, epoch=epoch)
        val_metrics = run_epoch(training=False, epoch=epoch)
        curves.append({"epoch": epoch, "model_train": train_metrics, "model_val": val_metrics})
        if val_metrics["accuracy"] > best_accuracy or (
            np.isclose(val_metrics["accuracy"], best_accuracy) and val_metrics["loss"] < best_loss
        ):
            best_accuracy = float(val_metrics["accuracy"])
            best_loss = float(val_metrics["loss"])
            best_val_metrics = dict(val_metrics)
            torch.save(
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
                best_path,
            )
    if effective_epochs == 0:
        val_metrics = run_epoch(training=False, epoch=0)
        best_accuracy = float(val_metrics["accuracy"])
        best_loss = float(val_metrics["loss"])
        best_val_metrics = dict(val_metrics)
        torch.save(
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
            best_path,
        )
        curves.append({"epoch": None, "model_train": None, "model_val": val_metrics})
    (output_dir / "training_curves.json").write_text(
        json.dumps(curves, indent=2, sort_keys=True), encoding="utf-8"
    )
    provenance = _source_provenance(
        frozen_val,
        target_provenance=_selected_target_provenance(root, cache_names),
    )
    if best_val_metrics is None:
        raise RuntimeError("tagger training selected no model-validation metrics")
    checkpoint_hash = _sha256_file(best_path)
    return {
        "contract": ABPH_TAGGER_RUNTIME_CONTRACT,
        "ok": True,
        "variant_name": variant_name,
        "variant": dict(resolved["variant"]),
        "resolved_variant_config_hash": resolved["resolved_config_hash"],
        "selected_checkpoint_hash": checkpoint_hash,
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


__all__ = [
    "ABPH_TAGGER_RUNTIME_CONTRACT",
    "FrozenPseudoBatch",
    "FrozenPseudoBatchSource",
    "train_tagger_variant",
]
