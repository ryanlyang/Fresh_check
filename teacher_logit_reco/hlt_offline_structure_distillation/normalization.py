"""Train-only HOSD target, residual, and latent statistics."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from teacher_logit_reco.relational_part.relation_track import (
    build_track_compatibility,
)

from .contracts import (
    CONDITIONAL_RESIDUAL_CONTRACT,
    HETEROSCEDASTIC_METADATA_CONTRACT,
    LATENT_WHITENING_CONTRACT,
    RIDGE_ADAPTER_CONTRACT,
    TARGET_NORMALIZER_CONTRACT,
    STREAMED_TARGET_NORMALIZER_CONTRACT,
    require_sha256,
    validate_content_hash,
    with_content_hash,
)
from teacher_logit_reco.relational_part.normalization import (
    NORMALIZATION_JET_LIMIT,
    NORMALIZATION_JET_SALT,
    NORMALIZATION_PAIR_LIMIT_PER_JET,
    NORMALIZATION_PAIR_SALT,
)
from .extractors import TargetBatch
from .target_cache import LoadedTargetCache, canonicalize_identities


NORMALIZED_CLIP = (-12.0, 12.0)
ROBUST_SCALE_FLOOR = 1.0e-6
WHITENING_EIGENVALUE_FLOOR = 1.0e-5
RIDGE_LAMBDA = 1.0e-4
HET_LOG_VARIANCE_CLIP = (-8.0, 5.0)
CONDITIONAL_BIN_COUNTS = (8, 4, 4, 4)
CONDITIONAL_FEATURES = (
    "log_jet_pt",
    "absolute_jet_eta",
    "valid_multiplicity",
    "valid_track_fraction",
)
BACKOFF_PATH = (
    "drop_valid_track_fraction",
    "drop_valid_multiplicity",
    "drop_absolute_jet_eta",
    "drop_log_jet_pt",
    "global_mean",
)
CONDITIONAL_CONTEXT_CONVENTION = {
    "log_jet_pt": "natural_log(max(vector_sum_pt,1e-6))",
    "absolute_jet_eta": "abs(asinh(sum(pt*sinh(eta))/max(vector_sum_pt,1e-6)))",
    "valid_multiplicity": "count_true_particle_mask",
    "valid_track_fraction": (
        "registered_TRACK_valid_count/max(valid_particle_count,1)"
    ),
}


def _validate_fit_population(split: str, fitting_population: str) -> None:
    expected = "model_train" if fitting_population == "target_500k" else (
        "scale_train" if fitting_population == "target_scale" else None
    )
    if expected is None:
        raise ValueError("fitting_population must be target_500k or target_scale")
    if split != expected:
        raise ValueError(
            f"{fitting_population} statistics must be fit on {expected}, got {split}"
        )


def _linear_quantile(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(values, probability, method="linear"))


def fit_target_normalizer(
    cache: LoadedTargetCache,
    *,
    fitting_population: str,
    source: Mapping[str, Any],
    component_kinds: Mapping[str, Sequence[str]] | None = None,
    normalization_role: str = "target",
) -> dict[str, Any]:
    """Fit identity-free per-component robust statistics on train only."""

    split = str(cache.manifest["split"])
    _validate_fit_population(split, fitting_population)
    if normalization_role not in {"target", "residual"}:
        raise ValueError("normalization_role must be target or residual")
    targets = []
    for target_id in cache.manifest["persisted_target_ids"]:
        values = np.asarray(cache.values[target_id], dtype=np.float64)
        masks = np.asarray(cache.masks[target_id], dtype=bool)
        components = []
        kinds = tuple(
            (component_kinds or {}).get(
                target_id, ("continuous",) * values.shape[1]
            )
        )
        if len(kinds) != values.shape[1]:
            raise ValueError(f"{target_id} component-kind count differs")
        for index, kind in enumerate(kinds):
            selected = values[:, index][masks[:, index]]
            if selected.size == 0:
                raise ValueError(
                    f"{target_id} component {index} has no model-train observations"
                )
            if not np.all(np.isfinite(selected)):
                raise ValueError("normalizer input contains non-finite values")
            continuous = kind == "continuous"
            median = _linear_quantile(selected, 0.5)
            q25 = _linear_quantile(selected, 0.25)
            q75 = _linear_quantile(selected, 0.75)
            scale = max((q75 - q25) / 1.349, ROBUST_SCALE_FLOOR)
            components.append(
                {
                    "component_index": index,
                    "kind": kind,
                    "normalize": continuous,
                    "center": median if continuous else 0.0,
                    "scale": scale if continuous else 1.0,
                    "unconditional_mean": float(selected.mean()),
                    "q25": q25,
                    "q50": median,
                    "q75": q75,
                    "valid_count": int(selected.size),
                    "clip": list(NORMALIZED_CLIP) if continuous else None,
                }
            )
        targets.append(
            {
                "target_id": target_id,
                "component_count": values.shape[1],
                "components": components,
            }
        )
    return with_content_hash(
        {
            "contract": TARGET_NORMALIZER_CONTRACT,
            "schema_version": 1,
            "fitting_population": fitting_population,
            "normalization_role": normalization_role,
            "fit_split": split,
            "population_role": (
                "model_train_only" if split == "model_train" else "scale_train_only"
            ),
            "cache_manifest_sha256": cache.manifest["content_hash"],
            "identity_order_sha256": cache.manifest[
                "canonical_identity_order_sha256"
            ],
            "event_count": cache.manifest["event_count"],
            "identity_values_stored": False,
            "quantile_method": "numpy_linear",
            "continuous_scale": "max((q75-q25)/1.349,1e-6)",
            "normalized_clipping": list(NORMALIZED_CLIP),
            "targets": targets,
            "source": dict(source),
        }
    )


def fit_sharded_target_normalizer(
    caches: Sequence[LoadedTargetCache],
    *,
    target_id: str,
    fitting_population: str,
    source: Mapping[str, Any],
    component_kinds: Sequence[str],
    normalization_role: str = "target",
    workspace: str | Path | None = None,
    batch_size: int = 2048,
) -> dict[str, Any]:
    """Fit exact robust statistics with one target shard plus one component resident."""

    if not caches or batch_size <= 0:
        raise ValueError("sharded target normalizer requires caches and a batch size")
    split = str(caches[0].manifest["split"])
    _validate_fit_population(split, fitting_population)
    if normalization_role not in {"target", "residual"}:
        raise ValueError("normalization_role must be target or residual")
    kinds = tuple(str(value) for value in component_kinds)
    component_count = None
    for cache in caches:
        if (
            cache.manifest.get("split") != split
            or cache.manifest.get("source") != dict(source)
            or target_id not in cache.values
        ):
            raise ValueError("sharded normalizer cache lineage/coverage differs")
        shape = cache.values[target_id].shape
        if len(shape) != 2 or (component_count is not None and shape[1] != component_count):
            raise ValueError("sharded normalizer component shapes differ")
        component_count = int(shape[1])
    if len(kinds) != component_count:
        raise ValueError("sharded normalizer component-kind count differs")
    parent_hashes = [str(cache.manifest["content_hash"]) for cache in caches]
    binding = (
        parent_hashes[0]
        if len(parent_hashes) == 1
        else hashlib.sha256("".join(parent_hashes).encode("ascii")).hexdigest()
    )
    identity_hashes = [
        str(cache.manifest["canonical_identity_order_sha256"])
        for cache in caches
    ]
    identity_binding = (
        identity_hashes[0]
        if len(identity_hashes) == 1
        else hashlib.sha256(
            ("hosd_sharded_normalizer_v1\0" + "\0".join(identity_hashes)).encode(
                "ascii"
            )
        ).hexdigest()
    )
    event_count = sum(int(cache.values[target_id].shape[0]) for cache in caches)
    base = None if workspace is None else Path(workspace)
    if base is not None:
        base.mkdir(parents=True, exist_ok=True)
    components = []
    with tempfile.TemporaryDirectory(dir=base, prefix="hosd-normalizer-") as temporary:
        temporary_path = Path(temporary)
        for index, kind in enumerate(kinds):
            valid_count = 0
            for cache in caches:
                count = int(cache.values[target_id].shape[0])
                for start in range(0, count, batch_size):
                    stop = min(start + batch_size, count)
                    valid_count += int(
                        np.asarray(cache.masks[target_id][start:stop], dtype=bool)[:, index].sum()
                    )
            if valid_count == 0:
                raise ValueError(f"{target_id} component {index} has no observations")
            sample_path = temporary_path / f"component_{index:04d}.npy"
            selected = np.lib.format.open_memmap(
                sample_path, mode="w+", dtype=np.float64, shape=(valid_count,)
            )
            cursor = 0
            for cache in caches:
                count = int(cache.values[target_id].shape[0])
                for start in range(0, count, batch_size):
                    stop = min(start + batch_size, count)
                    masks = np.asarray(cache.masks[target_id][start:stop], dtype=bool)[:, index]
                    values = np.asarray(cache.values[target_id][start:stop], dtype=np.float64)[:, index]
                    observed = values[masks]
                    if not np.isfinite(observed).all():
                        raise ValueError("normalizer input contains non-finite values")
                    selected[cursor : cursor + observed.size] = observed
                    cursor += observed.size
            if cursor != valid_count:
                raise RuntimeError("sharded normalizer observation count changed")
            selected.flush()
            continuous = kind == "continuous"
            median = _linear_quantile(selected, 0.5)
            q25 = _linear_quantile(selected, 0.25)
            q75 = _linear_quantile(selected, 0.75)
            scale = max((q75 - q25) / 1.349, ROBUST_SCALE_FLOOR)
            components.append(
                {
                    "component_index": index,
                    "kind": kind,
                    "normalize": continuous,
                    "center": median if continuous else 0.0,
                    "scale": scale if continuous else 1.0,
                    "unconditional_mean": float(selected.mean()),
                    "q25": q25,
                    "q50": median,
                    "q75": q75,
                    "valid_count": valid_count,
                    "clip": list(NORMALIZED_CLIP) if continuous else None,
                }
            )
            del selected
    return with_content_hash(
        {
            "contract": TARGET_NORMALIZER_CONTRACT,
            "schema_version": 1,
            "fitting_population": fitting_population,
            "normalization_role": normalization_role,
            "fit_split": split,
            "population_role": "model_train_only" if split == "model_train" else "scale_train_only",
            "cache_manifest_sha256": binding,
            "cache_manifest_hashes": parent_hashes,
            "identity_order_sha256": identity_binding,
            "event_count": event_count,
            "identity_values_stored": False,
            "quantile_method": "numpy_linear",
            "continuous_scale": "max((q75-q25)/1.349,1e-6)",
            "normalized_clipping": list(NORMALIZED_CLIP),
            "bounded_statistics": "disk_backed_one_component_one_shard_v1",
            "targets": [{"target_id": target_id, "component_count": component_count, "components": components}],
            "source": dict(source),
        }
    )


def fit_streamed_target_normalizer(
    *,
    target_id: str,
    component_names: Sequence[str],
    component_kinds: Sequence[str],
    component_samples: Sequence[np.ndarray],
    fitting_population: str,
    split: str,
    selected_jet_count: int,
    selected_jet_identity_sha256: str,
    sampled_pair_identity_sha256: str,
    parent_hashes: Mapping[str, str],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Fit robust statistics from a deterministic, non-persisted pair stream.

    Pair matrices remain absent from the artifact.  The caller supplies only
    the per-component values selected by the inherited salted 50k-jet,
    64-pair-per-jet normalization coordinate contract.
    """

    _validate_fit_population(split, fitting_population)
    names = tuple(str(value) for value in component_names)
    kinds = tuple(str(value) for value in component_kinds)
    samples = tuple(np.asarray(value, dtype=np.float64) for value in component_samples)
    if (
        not names
        or len(names) != len(set(names))
        or len(names) != len(kinds)
        or len(names) != len(samples)
        or int(selected_jet_count) <= 0
    ):
        raise ValueError("streamed target normalizer component declaration differs")
    checked_parents = {
        str(key): require_sha256(value, name=f"parent_hashes.{key}")
        for key, value in sorted(parent_hashes.items())
    }
    if not checked_parents:
        raise ValueError("streamed target normalizer requires parent hashes")
    components = []
    for index, (name, kind, sample) in enumerate(zip(names, kinds, samples)):
        if sample.ndim != 1 or sample.size == 0 or not np.isfinite(sample).all():
            raise ValueError(
                f"streamed target {target_id} component {name} has no finite sample"
            )
        continuous = kind == "continuous"
        median = _linear_quantile(sample, 0.5)
        q25 = _linear_quantile(sample, 0.25)
        q75 = _linear_quantile(sample, 0.75)
        scale = max((q75 - q25) / 1.349, ROBUST_SCALE_FLOOR)
        components.append(
            {
                "component_index": index,
                "component_name": name,
                "kind": kind,
                "normalize": continuous,
                "center": median if continuous else 0.0,
                "scale": scale if continuous else 1.0,
                "unconditional_mean": float(sample.mean()),
                "q25": q25,
                "q50": median,
                "q75": q75,
                "valid_count": int(sample.size),
                "clip": list(NORMALIZED_CLIP) if continuous else None,
            }
        )
    return with_content_hash(
        {
            "contract": STREAMED_TARGET_NORMALIZER_CONTRACT,
            "schema_version": 1,
            "fitting_population": fitting_population,
            "normalization_role": "target",
            "fit_split": split,
            "population_role": (
                "model_train_only" if split == "model_train" else "scale_train_only"
            ),
            "target_id": str(target_id),
            "targets": [
                {
                    "target_id": str(target_id),
                    "component_count": len(names),
                    "components": components,
                }
            ],
            "parent_hashes": checked_parents,
            "sampling_contract": {
                "jet_limit": NORMALIZATION_JET_LIMIT,
                "jet_salt": NORMALIZATION_JET_SALT,
                "pair_limit_per_jet": NORMALIZATION_PAIR_LIMIT_PER_JET,
                "pair_salt": NORMALIZATION_PAIR_SALT,
                "selected_jet_count": int(selected_jet_count),
                "selected_jet_identity_sha256": require_sha256(
                    selected_jet_identity_sha256,
                    name="selected_jet_identity_sha256",
                ),
                "sampled_pair_identity_sha256": require_sha256(
                    sampled_pair_identity_sha256,
                    name="sampled_pair_identity_sha256",
                ),
                "sampling_precedes_value_inspection": True,
            },
            "dense_pair_target_persisted": False,
            "identity_values_stored": False,
            "quantile_method": "numpy_linear",
            "continuous_scale": "max((q75-q25)/1.349,1e-6)",
            "normalized_clipping": list(NORMALIZED_CLIP),
            "source": dict(source),
        }
    )


def validate_target_normalizer(normalizer: Mapping[str, Any]) -> str:
    contract = normalizer.get("contract")
    if contract not in {
        TARGET_NORMALIZER_CONTRACT,
        STREAMED_TARGET_NORMALIZER_CONTRACT,
    }:
        raise ValueError("target normalizer contract differs")
    return validate_content_hash(normalizer, expected_contract=str(contract))


def _normalizer_target(
    normalizer: Mapping[str, Any], target_id: str
) -> Mapping[str, Any]:
    validate_target_normalizer(normalizer)
    matches = [row for row in normalizer["targets"] if row["target_id"] == target_id]
    if len(matches) != 1:
        raise ValueError(f"normalizer has no unique row for {target_id}")
    return matches[0]


def normalize_target(
    values: np.ndarray,
    masks: np.ndarray,
    *,
    target_id: str,
    normalizer: Mapping[str, Any],
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    masks = np.asarray(masks, dtype=bool)
    if values.shape != masks.shape:
        raise ValueError("target values/masks differ in shape")
    row = _normalizer_target(normalizer, target_id)
    output = np.zeros_like(values)
    for component in row["components"]:
        index = int(component["component_index"])
        selected = masks[:, index]
        if component["normalize"]:
            output[selected, index] = np.clip(
                (
                    values[selected, index] - float(component["center"])
                )
                / float(component["scale"]),
                *NORMALIZED_CLIP,
            )
        else:
            output[selected, index] = values[selected, index]
    return output


def target_mean_values(
    masks: np.ndarray,
    *,
    target_id: str,
    normalizer: Mapping[str, Any],
) -> np.ndarray:
    masks = np.asarray(masks, dtype=bool)
    row = _normalizer_target(normalizer, target_id)
    if masks.ndim != 2 or masks.shape[1] != row["component_count"]:
        raise ValueError("target-mean mask shape differs from schema")
    output = np.zeros(masks.shape, dtype=np.float32)
    for component in row["components"]:
        index = int(component["component_index"])
        output[masks[:, index], index] = float(component["unconditional_mean"])
    return output


def build_heteroscedastic_metadata(
    normalizer: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    validate_target_normalizer(normalizer)
    return with_content_hash(
        {
            "contract": HETEROSCEDASTIC_METADATA_CONTRACT,
            "schema_version": 1,
            "target_normalizer_sha256": normalizer["content_hash"],
            "eligible_component_kind": "continuous",
            "prediction_parameterization": "normalized_mean_and_log_variance",
            "log_variance_clip": list(HET_LOG_VARIANCE_CLIP),
            "variance_floor": math.exp(HET_LOG_VARIANCE_CLIP[0]),
            "variance_ceiling": math.exp(HET_LOG_VARIANCE_CLIP[1]),
            "loss": "masked_gaussian_nll_in_normalized_target_coordinates",
            "identity_values_stored": False,
            "source": dict(source),
        }
    )


def residual_batches(
    offline: LoadedTargetCache,
    hlt: LoadedTargetCache,
    *,
    target_pairs: Mapping[str, str],
    output_target_ids: Mapping[str, str] | None = None,
) -> dict[str, TargetBatch]:
    """Build exact per-replica offline-minus-HLT targets."""

    if offline.identities != hlt.identities:
        raise ValueError("offline and HLT residual caches have different identities")
    hlt_replica = hlt.manifest.get("hlt_replica_id")
    if hlt_replica is None:
        # hlt_replica_id is on the cache spec; manifests created by v1 retain kind.
        hlt_replica = "bound_in_hlt_cache_spec"
    output = {}
    for offline_id, hlt_id in sorted(target_pairs.items()):
        if offline.values[offline_id].shape != hlt.values[hlt_id].shape:
            raise ValueError("residual component schemas differ")
        mask = offline.masks[offline_id] & hlt.masks[hlt_id]
        values = np.zeros_like(offline.values[offline_id], dtype=np.float32)
        values[mask] = (
            offline.values[offline_id][mask] - hlt.values[hlt_id][mask]
        )
        target_id = (
            output_target_ids or {}
        ).get(offline_id, f"{offline_id}__RES")
        output[target_id] = TargetBatch(
            target_id=target_id,
            component_names=tuple(
                offline.manifest["target_components"][offline_id]
            ),
            availability_groups=("offline_and_hlt_target_available",),
            values=torch.from_numpy(values),
            loss_mask=torch.from_numpy(mask),
            diagnostics={
                "parameterization": "RES",
                "residual_definition": "offline_minus_exact_hlt_replica",
                "hlt_replica_id": hlt_replica,
                "offline_cache_sha256": offline.manifest["content_hash"],
                "hlt_cache_sha256": hlt.manifest["content_hash"],
            },
        )
    return output


def _quantile_edges(values: np.ndarray, bins: int) -> list[float]:
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("conditional-residual feature values must be finite")
    return [
        _linear_quantile(values, index / bins)
        for index in range(1, bins)
    ]


def build_hlt_conditional_context(
    raw_tokens: np.ndarray,
    mask: np.ndarray,
    *,
    d0_uncertainty_floor: float,
    dz_uncertainty_floor: float,
    sentinel_policy: Mapping[str, Any] | None,
) -> np.ndarray:
    """Compute the globally frozen four deployable HLT conditioning scalars."""

    raw = np.asarray(raw_tokens, dtype=np.float32)
    valid = np.asarray(mask, dtype=bool)
    if raw.ndim != 3 or raw.shape[2] != 14 or valid.shape != raw.shape[:2]:
        raise ValueError("HLT conditional context requires raw [B,N,14] and mask [B,N]")
    pt = np.where(valid, raw[:, :, 0], 0.0).astype(np.float64)
    eta = np.where(valid, raw[:, :, 1], 0.0).astype(np.float64)
    phi = np.where(valid, raw[:, :, 2], 0.0).astype(np.float64)
    px = np.sum(pt * np.cos(phi), axis=1)
    py = np.sum(pt * np.sin(phi), axis=1)
    pz = np.sum(pt * np.sinh(eta), axis=1)
    jet_pt = np.hypot(px, py)
    jet_eta = np.arcsinh(pz / np.maximum(jet_pt, 1.0e-6))
    track = build_track_compatibility(
        torch.from_numpy(raw),
        torch.from_numpy(valid[:, None, :]),
        d0_uncertainty_floor=d0_uncertainty_floor,
        dz_uncertainty_floor=dz_uncertainty_floor,
        sentinel_policy=sentinel_policy,
    )["track_valid"].cpu().numpy()
    multiplicity = valid.sum(axis=1)
    return np.stack(
        [
            np.log(np.maximum(jet_pt, 1.0e-6)),
            np.abs(jet_eta),
            multiplicity.astype(np.float64),
            track.sum(axis=1) / np.maximum(multiplicity, 1),
        ],
        axis=1,
    ).astype(np.float64)


def align_conditional_context_to_cache(
    source_identities: Sequence[str],
    context: np.ndarray,
    cache_identities: Sequence[str],
) -> np.ndarray:
    """Align compact HLT context rows to canonical target-cache identities."""

    values = np.asarray(context, dtype=np.float64)
    source = tuple(str(value) for value in source_identities)
    expected = tuple(str(value) for value in cache_identities)
    if values.shape != (len(source), 4):
        raise ValueError("conditional context/source identity shape differs")
    canonical, canonical_to_source = canonicalize_identities(source)
    if canonical != expected:
        raise ValueError("conditional context/cache identity coverage differs")
    if np.array_equal(
        canonical_to_source,
        np.arange(len(source), dtype=np.int64),
    ):
        return values
    # Only the compact four-scalar context is reordered.  The population-wide
    # particle tensor remains in its authenticated source order.
    return np.take(values, canonical_to_source, axis=0)


def _bin_index(value: float, edges: Sequence[float]) -> int:
    return int(np.searchsorted(np.asarray(edges), value, side="right"))


def fit_conditional_residual(
    residual_values: np.ndarray,
    residual_masks: np.ndarray,
    hlt_context: np.ndarray,
    *,
    target_id: str,
    train_cache_hashes: Mapping[str, str],
    source: Mapping[str, Any],
    fitting_population: str = "target_500k",
) -> dict[str, Any]:
    """Fit the globally frozen four-axis coarse conditional mean diagnostic."""

    residual_values = np.asarray(residual_values, dtype=np.float64)
    residual_masks = np.asarray(residual_masks, dtype=bool)
    hlt_context = np.asarray(hlt_context, dtype=np.float64)
    if residual_values.shape != residual_masks.shape or residual_values.ndim != 2:
        raise ValueError("conditional residual target must be [event,component]")
    if hlt_context.shape != (residual_values.shape[0], 4):
        raise ValueError("conditional residual context must be [event,4]")
    if not np.all(np.isfinite(hlt_context)):
        raise ValueError("conditional residual context contains non-finite values")
    if fitting_population not in {"target_500k", "target_scale"}:
        raise ValueError("invalid conditional-residual fitting population")
    edges = [
        _quantile_edges(hlt_context[:, axis], bins)
        for axis, bins in enumerate(CONDITIONAL_BIN_COUNTS)
    ]
    coordinates = np.stack(
        [
            np.asarray([_bin_index(value, edges[axis]) for value in hlt_context[:, axis]])
            for axis in range(4)
        ],
        axis=1,
    )
    levels = []
    for retained_axes in (4, 3, 2, 1, 0):
        cells = []
        keys = (
            sorted({tuple(row[:retained_axes]) for row in coordinates})
            if retained_axes
            else [()]
        )
        for key in keys:
            selected_rows = (
                np.all(coordinates[:, :retained_axes] == key, axis=1)
                if retained_axes
                else np.ones(coordinates.shape[0], dtype=bool)
            )
            component_counts = []
            component_means = []
            for component in range(residual_values.shape[1]):
                selected = selected_rows & residual_masks[:, component]
                component_counts.append(int(selected.sum()))
                component_means.append(
                    float(residual_values[selected, component].mean())
                    if selected.any()
                    else 0.0
                )
            if any(component_counts):
                cells.append(
                    {
                        "bin_prefix": [int(value) for value in key],
                        "event_count": int(selected_rows.sum()),
                        "component_valid_counts": component_counts,
                        "component_means": component_means,
                    }
                )
        levels.append({"retained_axes": retained_axes, "cells": cells})
    return with_content_hash(
        {
            "contract": CONDITIONAL_RESIDUAL_CONTRACT,
            "schema_version": 1,
            "target_id": target_id,
            "fitting_population": fitting_population,
            "fit_split": (
                "model_train" if fitting_population == "target_500k" else "scale_train"
            ),
            "feature_order": list(CONDITIONAL_FEATURES),
            "feature_convention": dict(CONDITIONAL_CONTEXT_CONVENTION),
            "bin_counts": list(CONDITIONAL_BIN_COUNTS),
            "bin_edges": edges,
            "edge_quantile_method": "numpy_linear",
            "endpoint_rule": "searchsorted_right_including_outer_infinite_ranges",
            "backoff_path": list(BACKOFF_PATH),
            "levels": levels,
            "train_cache_hashes": {
                str(key): str(value) for key, value in sorted(train_cache_hashes.items())
            },
            "role": "train_fitted_identity_free_deployable_statistic",
            "identity_values_stored": False,
            "event_values_stored": False,
            "source": dict(source),
        }
    )


def apply_conditional_residual(
    hlt_context: np.ndarray,
    *,
    artifact: Mapping[str, Any],
) -> np.ndarray:
    validate_content_hash(
        artifact, expected_contract=CONDITIONAL_RESIDUAL_CONTRACT
    )
    context = np.asarray(hlt_context, dtype=np.float64)
    if context.ndim != 2 or context.shape[1] != 4:
        raise ValueError("conditional residual context must be [event,4]")
    edges = artifact["bin_edges"]
    coordinates = np.stack(
        [
            np.asarray([_bin_index(value, edges[axis]) for value in context[:, axis]])
            for axis in range(4)
        ],
        axis=1,
    )
    levels = {
        int(level["retained_axes"]): {
            tuple(cell["bin_prefix"]): cell for cell in level["cells"]
        }
        for level in artifact["levels"]
    }
    component_count = len(levels[0][()]["component_means"])
    output = np.zeros((context.shape[0], component_count), dtype=np.float32)
    for row, coordinate in enumerate(coordinates):
        unresolved = set(range(component_count))
        for retained_axes in (4, 3, 2, 1, 0):
            cell = levels[retained_axes].get(tuple(coordinate[:retained_axes]))
            if cell is None:
                continue
            for component in tuple(unresolved):
                if int(cell["component_valid_counts"][component]) > 0:
                    output[row, component] = float(cell["component_means"][component])
                    unresolved.remove(component)
            if not unresolved:
                break
        if unresolved:
            raise ValueError("conditional residual has no global fallback")
    return output


def _canonical_eigenspace_basis(
    vectors: np.ndarray, eigenvalues: np.ndarray
) -> np.ndarray:
    """Fix arbitrary rotations inside numerically degenerate eigenspaces."""

    dimension = vectors.shape[0]
    output = np.empty_like(vectors)
    tolerance = 1.0e-12 * max(1.0, float(np.max(np.abs(eigenvalues))))
    start = 0
    while start < dimension:
        stop = start + 1
        while (
            stop < dimension
            and abs(float(eigenvalues[stop] - eigenvalues[start])) <= tolerance
        ):
            stop += 1
        subspace = vectors[:, start:stop]
        projector = subspace @ subspace.T
        basis: list[np.ndarray] = []
        for axis in range(dimension):
            candidate = projector[:, axis].copy()
            for existing in basis:
                candidate -= existing * float(existing @ candidate)
            norm = float(np.linalg.norm(candidate))
            if norm > 1.0e-10:
                basis.append(candidate / norm)
            if len(basis) == stop - start:
                break
        if len(basis) != stop - start:
            raise RuntimeError("failed to canonicalize degenerate eigenspace")
        output[:, start:stop] = np.stack(basis, axis=1)
        start = stop
    for column in range(dimension):
        vector = output[:, column]
        pivot = int(np.argmax(np.abs(vector)))
        if vector[pivot] < 0:
            output[:, column] *= -1.0
    return output


def fit_latent_whitening(
    latents: np.ndarray,
    *,
    teacher_lock_sha256: str,
    fitting_population: str,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    values = np.asarray(latents, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] != 128:
        raise ValueError("teacher latent whitening requires [N>=2,128]")
    if not np.all(np.isfinite(values)):
        raise ValueError("teacher latents contain non-finite values")
    if fitting_population not in {"target_500k", "target_scale"}:
        raise ValueError("invalid latent-whitening fitting population")
    mean = values.mean(axis=0)
    centered = values - mean
    covariance = centered.T @ centered / values.shape[0]
    return _build_latent_whitening_artifact(
        mean=mean,
        covariance=covariance,
        event_count=values.shape[0],
        teacher_lock_sha256=teacher_lock_sha256,
        fitting_population=fitting_population,
        source=source,
        accumulation="population_resident_reference_v2",
    )


def fit_sharded_latent_whitening(
    latents: Any,
    *,
    teacher_lock_sha256: str,
    fitting_population: str,
    source: Mapping[str, Any],
    batch_size: int = 2048,
) -> dict[str, Any]:
    """Fit latent whitening from a lazy array with one bounded slice resident."""

    shape = tuple(latents.shape)
    if len(shape) != 2 or shape[0] < 2 or shape[1] != 128 or batch_size <= 0:
        raise ValueError("teacher latent whitening requires lazy [N>=2,128]")
    if fitting_population not in {"target_500k", "target_scale"}:
        raise ValueError("invalid latent-whitening fitting population")
    total = np.zeros(128, dtype=np.float64)
    cross = np.zeros((128, 128), dtype=np.float64)
    count = 0
    for start in range(0, shape[0], batch_size):
        values = np.asarray(
            latents[start : min(start + batch_size, shape[0])], dtype=np.float64
        )
        if values.ndim != 2 or values.shape[1] != 128 or not np.isfinite(values).all():
            raise ValueError("teacher latents contain non-finite values")
        total += values.sum(axis=0, dtype=np.float64)
        cross += values.T @ values
        count += values.shape[0]
    if count != shape[0]:
        raise RuntimeError("latent population changed during streamed whitening")
    mean = total / count
    covariance = cross / count - np.outer(mean, mean)
    covariance = (covariance + covariance.T) * 0.5
    return _build_latent_whitening_artifact(
        mean=mean,
        covariance=covariance,
        event_count=count,
        teacher_lock_sha256=teacher_lock_sha256,
        fitting_population=fitting_population,
        source=source,
        accumulation="float64_shard_sum_and_cross_product_v2",
    )


def _build_latent_whitening_artifact(
    *,
    mean: np.ndarray,
    covariance: np.ndarray,
    event_count: int,
    teacher_lock_sha256: str,
    fitting_population: str,
    source: Mapping[str, Any],
    accumulation: str,
) -> dict[str, Any]:
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues, kind="stable")[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = _canonical_eigenspace_basis(eigenvectors[:, order], eigenvalues)
    floor = WHITENING_EIGENVALUE_FLOOR * max(float(eigenvalues[0]), 1.0e-30)
    floored = np.maximum(eigenvalues, floor)
    return with_content_hash(
        {
            "contract": LATENT_WHITENING_CONTRACT,
            "schema_version": 2,
            "teacher_lock_sha256": teacher_lock_sha256,
            "fitting_population": fitting_population,
            "fit_split": (
                "model_train" if fitting_population == "target_500k" else "scale_train"
            ),
            "event_count": int(event_count),
            "dimension": 128,
            "mean": mean.tolist(),
            "eigenvalues_descending": eigenvalues.tolist(),
            "floored_eigenvalues": floored.tolist(),
            "eigenvectors_as_columns": eigenvectors.tolist(),
            "eigenvalue_floor_fraction_of_largest": WHITENING_EIGENVALUE_FLOOR,
            "degenerate_eigenspace_rule": (
                "projected_canonical_basis_gram_schmidt_axis_order"
            ),
            "eigenvector_sign_rule": (
                "largest_absolute_component_positive_tie_smallest_index"
            ),
            "identity_values_stored": False,
            "covariance_accumulation": accumulation,
            "source": dict(source),
        }
    )


def whiten_latents(
    values: np.ndarray, *, whitening: Mapping[str, Any]
) -> np.ndarray:
    validate_content_hash(whitening, expected_contract=LATENT_WHITENING_CONTRACT)
    raw = np.asarray(values, dtype=np.float64)
    mean = np.asarray(whitening["mean"], dtype=np.float64)
    vectors = np.asarray(whitening["eigenvectors_as_columns"], dtype=np.float64)
    eigenvalues = np.asarray(whitening["floored_eigenvalues"], dtype=np.float64)
    return ((raw - mean) @ vectors / np.sqrt(eigenvalues)).astype(np.float32)


def fit_latent_ridge_adapter(
    predicted_whitened: np.ndarray,
    teacher_unwhitened: np.ndarray,
    *,
    whitening_sha256: str,
    teacher_lock_sha256: str,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    x = np.asarray(predicted_whitened, dtype=np.float64)
    y = np.asarray(teacher_unwhitened, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("ridge inputs must have equal [event,dimension] shapes")
    design = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
    penalty = np.eye(design.shape[1], dtype=np.float64) * RIDGE_LAMBDA
    penalty[-1, -1] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + penalty, design.T @ y
    )
    return with_content_hash(
        {
            "contract": RIDGE_ADAPTER_CONTRACT,
            "schema_version": 1,
            "whitening_sha256": whitening_sha256,
            "teacher_lock_sha256": teacher_lock_sha256,
            "event_count": x.shape[0],
            "input_dimension": x.shape[1],
            "output_dimension": y.shape[1],
            "lambda": RIDGE_LAMBDA,
            "solver": "float64_normal_equations_numpy_solve",
            "intercept": "unregularized_last_design_column",
            "coefficients_with_intercept_last_row": coefficients.tolist(),
            "labels_used": False,
            "checkpoint_selection_used": False,
            "identity_values_stored": False,
            "source": dict(source),
        }
    )


__all__ = [
    "align_conditional_context_to_cache",
    "BACKOFF_PATH",
    "CONDITIONAL_BIN_COUNTS",
    "CONDITIONAL_FEATURES",
    "CONDITIONAL_CONTEXT_CONVENTION",
    "HET_LOG_VARIANCE_CLIP",
    "NORMALIZED_CLIP",
    "RIDGE_LAMBDA",
    "WHITENING_EIGENVALUE_FLOOR",
    "apply_conditional_residual",
    "build_heteroscedastic_metadata",
    "build_hlt_conditional_context",
    "fit_conditional_residual",
    "fit_latent_ridge_adapter",
    "fit_latent_whitening",
    "fit_target_normalizer",
    "fit_streamed_target_normalizer",
    "normalize_target",
    "residual_batches",
    "target_mean_values",
    "validate_target_normalizer",
    "whiten_latents",
]
