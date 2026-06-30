"""Checkpoint loading and warm-start helpers for local-compression ParT.

Step 11 is deliberately narrow: load a frozen HLT ParT baseline checkpoint into
the exact ParT backbone inside the local-compression wrapper, verify the parts
of the protocol metadata that are available, and preserve enough provenance for
later training/reporting.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from jetclass_fresh.hlt_baseline import ParticleTransformerHLTClassifier, require_torch

from .config import (
    LOCAL_COMPRESSION_BINARY_LABEL_FILTER,
    LOCAL_COMPRESSION_PART_CONTRACT,
    LOCAL_COMPRESSION_PART_HLT_DEGRADATION_STRENGTH,
    LOCAL_COMPRESSION_PRIMARY_METRIC,
    LOCAL_COMPRESSION_SOURCE_LABEL_NAMES,
)


LOCAL_COMPRESSION_CHECKPOINT_STEP = "local_compression_part_step11_checkpoint_loading"
LOCAL_COMPRESSION_CHECKPOINT_CONTRACT = f"{LOCAL_COMPRESSION_PART_CONTRACT}_checkpoint_warm_start_v1"


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a SHA256 hash for a checkpoint/report file."""

    resolved = Path(path)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        while True:
            chunk = handle.read(int(chunk_size))
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _json_load_if_exists(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        return None
    return payload


def _nested_get(payload: Mapping[str, Any], paths: tuple[tuple[str, ...], ...]) -> Any:
    for path in paths:
        current: Any = payload
        found = True
        for key in path:
            if not isinstance(current, Mapping) or key not in current:
                found = False
                break
            current = current[key]
        if found:
            return current
    return None


def _first_metadata_value(*payloads: Mapping[str, Any] | None, paths: tuple[tuple[str, ...], ...]) -> Any:
    for payload in payloads:
        if isinstance(payload, Mapping):
            value = _nested_get(payload, paths)
            if value is not None:
                return value
    return None


def _load_torch_payload(path: Path, *, map_location: Any = "cpu") -> Any:
    torch = require_torch()
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # pragma: no cover - older torch
        return torch.load(path, map_location=map_location)


def _extract_state_dict(payload: Any) -> Mapping[str, Any]:
    if isinstance(payload, Mapping):
        for key in ("model_state_dict", "state_dict", "model"):
            value = payload.get(key)
            if isinstance(value, Mapping):
                return value
        if payload and all(isinstance(key, str) for key in payload.keys()):
            tensor_like = [value for value in payload.values() if hasattr(value, "shape")]
            if tensor_like:
                return payload
    raise ValueError("checkpoint does not contain a recognizable model state dict")


def _strip_wrapper_prefixes(key: str) -> str:
    clean = str(key)
    changed = True
    while changed:
        changed = False
        for prefix in ("module.", "_orig_mod."):
            if clean.startswith(prefix):
                clean = clean[len(prefix) :]
                changed = True
    return clean


def _part_model_candidate_keys(source_key: str) -> tuple[str, ...]:
    clean = _strip_wrapper_prefixes(source_key)
    candidates = [clean]
    for prefix in ("part_model.", "model.", "classifier."):
        if clean.startswith(prefix):
            candidates.append(clean[len(prefix) :])
    output: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        candidate = _strip_wrapper_prefixes(candidate)
        if candidate not in seen:
            seen.add(candidate)
            output.append(candidate)
    return tuple(output)


def _coerce_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _coerce_str_tuple_or_none(value: Any) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return (value,)
    try:
        return tuple(str(item) for item in value)
    except TypeError:
        return None


def _coerce_int_tuple_or_none(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        values = [part.strip() for part in value.split(",") if part.strip()]
    else:
        try:
            values = list(value)
        except TypeError:
            values = [value]
    output: list[int] = []
    try:
        for item in values:
            output.append(int(item))
    except (TypeError, ValueError):
        return None
    return tuple(output)


def _coerce_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _sidecar_report_paths(checkpoint_path: Path, explicit: str | Path | None = None) -> tuple[Path, ...]:
    if explicit is not None:
        return (Path(explicit),)
    parent = checkpoint_path.parent
    return (
        parent / "run_report.json",
        parent / "model_val_report.json",
        parent / "config.json",
        parent / "slurm_run_config.json",
    )


def _metadata_from_checkpoint_and_reports(
    checkpoint_path: Path,
    checkpoint_payload: Any,
    *,
    run_report_path: str | Path | None = None,
) -> dict[str, Any]:
    checkpoint_mapping = checkpoint_payload if isinstance(checkpoint_payload, Mapping) else None
    sidecars: list[dict[str, Any]] = []
    loaded_paths: list[str] = []
    for path in _sidecar_report_paths(checkpoint_path, explicit=run_report_path):
        payload = _json_load_if_exists(path)
        if payload is not None:
            sidecars.append(payload)
            loaded_paths.append(str(path))

    payloads: tuple[Mapping[str, Any] | None, ...] = (checkpoint_mapping, *sidecars)
    selection_metric = _first_metadata_value(
        *payloads,
        paths=(
            ("selection_metric",),
            ("best_model_selection_metric",),
            ("config", "selection_metric"),
            ("training_config", "selection_metric"),
            ("run_config", "selection_metric"),
        ),
    )
    hlt_strength = _first_metadata_value(
        *payloads,
        paths=(
            ("hlt_degradation_strength",),
            ("expected_hlt_degradation_strength",),
            ("config", "hlt_degradation_strength"),
            ("config", "expected_hlt_degradation_strength"),
            ("training_config", "expected_hlt_degradation_strength"),
            ("run_config", "expected_hlt_degradation_strength"),
            ("metadata", "hlt_degradation_strength"),
            ("binary_inputs", "hlt_degradation_strength"),
        ),
    )
    split_manifest_hash = _first_metadata_value(
        *payloads,
        paths=(
            ("split_manifest_hash",),
            ("manifest_hash",),
            ("config", "split_manifest_hash"),
            ("run_config", "split_manifest_hash"),
            ("dataset", "split_manifest_hash"),
            ("binary_inputs", "split_manifest_hash"),
        ),
    )
    part_config = _first_metadata_value(
        *payloads,
        paths=(("part_config",), ("part_model_config",), ("model_config",), ("config", "part_model_config")),
    )
    label_names = _first_metadata_value(
        *payloads,
        paths=(
            ("label_names",),
            ("config", "label_names"),
            ("training_config", "label_names"),
            ("run_config", "label_names"),
            ("metadata", "label_names"),
        ),
    )
    label_filter = _first_metadata_value(
        *payloads,
        paths=(
            ("label_filter",),
            ("source_label_indices",),
            ("config", "label_filter"),
            ("training_config", "label_filter"),
            ("run_config", "label_filter"),
            ("metadata", "label_filter"),
        ),
    )
    num_classes = _first_metadata_value(
        *payloads,
        paths=(
            ("num_classes",),
            ("config", "num_classes"),
            ("training_config", "num_classes"),
            ("run_config", "num_classes"),
            ("metadata", "num_classes"),
        ),
    )
    return {
        "sidecar_report_paths": loaded_paths,
        "selection_metric": _coerce_str_or_none(selection_metric),
        "hlt_degradation_strength": _coerce_float_or_none(hlt_strength),
        "split_manifest_hash": _coerce_str_or_none(split_manifest_hash),
        "part_config": dict(part_config) if isinstance(part_config, Mapping) else {},
        "label_names": _coerce_str_tuple_or_none(label_names),
        "label_filter": _coerce_int_tuple_or_none(label_filter),
        "num_classes": _coerce_int_or_none(num_classes),
    }


def _check_optional_equal(name: str, observed: Any, expected: Any, *, require_metadata: bool = False) -> None:
    if expected is None:
        return
    if observed is None:
        if require_metadata:
            raise ValueError(f"baseline checkpoint is missing required metadata field {name}")
        return
    if isinstance(expected, float):
        if abs(float(observed) - float(expected)) > 1.0e-12:
            raise ValueError(f"baseline checkpoint {name} mismatch: expected {expected}, got {observed}")
        return
    if str(observed) != str(expected):
        raise ValueError(f"baseline checkpoint {name} mismatch: expected {expected!r}, got {observed!r}")


@dataclass(frozen=True)
class LocalCompressionBaselineCheckpointReport:
    """Provenance and load summary for a warm-started HLT ParT baseline."""

    contract: str
    step: str
    baseline_checkpoint_path: str
    baseline_checkpoint_hash: str
    baseline_checkpoint_selection_metric: str | None
    baseline_checkpoint_hlt_degradation_strength: float | None
    baseline_checkpoint_split_manifest_hash: str | None
    baseline_checkpoint_label_names: tuple[str, ...] | None
    baseline_checkpoint_label_filter: tuple[int, ...] | None
    baseline_checkpoint_num_classes: int | None
    baseline_checkpoint_sidecar_reports: tuple[str, ...]
    part_config: Mapping[str, Any]
    adapter_config: Mapping[str, Any]
    loaded_key_count: int
    missing_key_count: int
    unexpected_key_count: int
    shape_mismatch_count: int
    non_tensor_key_count: int
    loaded_keys_sample: tuple[Mapping[str, str], ...]
    missing_keys_sample: tuple[str, ...]
    unexpected_keys_sample: tuple[str, ...]
    shape_mismatches_sample: tuple[Mapping[str, Any], ...]
    non_tensor_keys_sample: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(asdict(self), sort_keys=True))


def _resolve_part_model(model_or_part_model: Any) -> ParticleTransformerHLTClassifier:
    candidate = getattr(model_or_part_model, "part_model", model_or_part_model)
    if not isinstance(candidate, ParticleTransformerHLTClassifier):
        raise ValueError("baseline checkpoint can only be loaded into ParticleTransformerHLTClassifier")
    return candidate


def load_hlt_part_baseline_checkpoint(
    checkpoint_path: str | Path,
    model_or_part_model: Any,
    *,
    map_location: Any = "cpu",
    run_report_path: str | Path | None = None,
    expected_selection_metric: str | None = LOCAL_COMPRESSION_PRIMARY_METRIC,
    expected_hlt_degradation_strength: float | None = LOCAL_COMPRESSION_PART_HLT_DEGRADATION_STRENGTH,
    expected_split_manifest_hash: str | None = None,
    expected_label_names: Sequence[str] | None = LOCAL_COMPRESSION_SOURCE_LABEL_NAMES,
    expected_label_filter: Sequence[int] | None = LOCAL_COMPRESSION_BINARY_LABEL_FILTER,
    expected_num_classes: int | None = 2,
    require_metadata: bool = False,
    require_all_part_keys: bool = True,
) -> LocalCompressionBaselineCheckpointReport:
    """Load a baseline HLT ParT checkpoint into a ParT model or wrapper.

    The source may be a direct HLT ParT checkpoint with keys like ``mod.*`` or a
    local-adapter checkpoint with keys like ``part_model.mod.*``.  Extra
    non-backbone keys are reported as unexpected source keys rather than loaded.
    """

    torch = require_torch()
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(path)
    part_model = _resolve_part_model(model_or_part_model)
    payload = _load_torch_payload(path, map_location=map_location)
    source_state = _extract_state_dict(payload)
    target_state = part_model.state_dict()
    updated_state = dict(target_state)
    loaded: list[dict[str, str]] = []
    loaded_targets: set[str] = set()
    unexpected: list[str] = []
    shape_mismatches: list[dict[str, Any]] = []
    non_tensor: list[str] = []

    for source_key, source_value in source_state.items():
        if not hasattr(source_value, "shape"):
            non_tensor.append(str(source_key))
            continue
        matched_key = None
        for candidate in _part_model_candidate_keys(str(source_key)):
            if candidate not in target_state:
                continue
            target_value = target_state[candidate]
            if tuple(source_value.shape) != tuple(target_value.shape):
                shape_mismatches.append(
                    {
                        "source_key": str(source_key),
                        "target_key": str(candidate),
                        "source_shape": list(source_value.shape),
                        "target_shape": list(target_value.shape),
                    }
                )
                continue
            updated_state[candidate] = source_value.detach().to(device=target_value.device, dtype=target_value.dtype)
            matched_key = candidate
            loaded_targets.add(str(candidate))
            loaded.append({"source_key": str(source_key), "target_key": str(candidate)})
            break
        if matched_key is None:
            unexpected.append(str(source_key))

    missing = sorted(str(key) for key in target_state.keys() if str(key) not in loaded_targets)
    if not loaded:
        raise ValueError(f"warm start loaded zero HLT ParT keys from {path}")
    if missing and bool(require_all_part_keys):
        raise ValueError(
            f"baseline checkpoint did not load all HLT ParT keys from {path}; "
            f"missing {len(missing)} keys, first few: {missing[:8]}"
        )
    part_model.load_state_dict(updated_state, strict=True)

    metadata = _metadata_from_checkpoint_and_reports(path, payload, run_report_path=run_report_path)
    _check_optional_equal(
        "selection_metric",
        metadata["selection_metric"],
        expected_selection_metric,
        require_metadata=bool(require_metadata),
    )
    _check_optional_equal(
        "hlt_degradation_strength",
        metadata["hlt_degradation_strength"],
        expected_hlt_degradation_strength,
        require_metadata=bool(require_metadata),
    )
    _check_optional_equal(
        "split_manifest_hash",
        metadata["split_manifest_hash"],
        expected_split_manifest_hash,
        require_metadata=bool(require_metadata and expected_split_manifest_hash is not None),
    )
    _check_optional_equal(
        "label_names",
        metadata["label_names"],
        None if expected_label_names is None else tuple(str(name) for name in expected_label_names),
        require_metadata=bool(require_metadata),
    )
    _check_optional_equal(
        "label_filter",
        metadata["label_filter"],
        None if expected_label_filter is None else tuple(int(index) for index in expected_label_filter),
        require_metadata=bool(require_metadata),
    )
    _check_optional_equal(
        "num_classes",
        metadata["num_classes"],
        expected_num_classes,
        require_metadata=bool(require_metadata),
    )

    adapter_config = {}
    if hasattr(model_or_part_model, "config") and hasattr(getattr(model_or_part_model, "config"), "to_dict"):
        adapter_config = getattr(model_or_part_model, "config").to_dict()
    elif hasattr(model_or_part_model, "config") and isinstance(getattr(model_or_part_model, "config"), Mapping):
        adapter_config = dict(getattr(model_or_part_model, "config"))

    report = LocalCompressionBaselineCheckpointReport(
        contract=LOCAL_COMPRESSION_CHECKPOINT_CONTRACT,
        step=LOCAL_COMPRESSION_CHECKPOINT_STEP,
        baseline_checkpoint_path=str(path),
        baseline_checkpoint_hash=sha256_file(path),
        baseline_checkpoint_selection_metric=metadata["selection_metric"],
        baseline_checkpoint_hlt_degradation_strength=metadata["hlt_degradation_strength"],
        baseline_checkpoint_split_manifest_hash=metadata["split_manifest_hash"],
        baseline_checkpoint_label_names=metadata["label_names"],
        baseline_checkpoint_label_filter=metadata["label_filter"],
        baseline_checkpoint_num_classes=metadata["num_classes"],
        baseline_checkpoint_sidecar_reports=tuple(metadata["sidecar_report_paths"]),
        part_config=dict(getattr(part_model, "config", {}) or metadata["part_config"] or {}),
        adapter_config=adapter_config,
        loaded_key_count=len(loaded),
        missing_key_count=len(missing),
        unexpected_key_count=len(unexpected),
        shape_mismatch_count=len(shape_mismatches),
        non_tensor_key_count=len(non_tensor),
        loaded_keys_sample=tuple(loaded[:50]),
        missing_keys_sample=tuple(missing[:50]),
        unexpected_keys_sample=tuple(unexpected[:50]),
        shape_mismatches_sample=tuple(shape_mismatches[:50]),
        non_tensor_keys_sample=tuple(non_tensor[:50]),
    )
    if hasattr(model_or_part_model, "baseline_checkpoint_report"):
        setattr(model_or_part_model, "baseline_checkpoint_report", report.to_dict())
    return report


def warm_start_local_compression_part_model(
    model: Any,
    checkpoint_path: str | Path,
    **kwargs: Any,
) -> LocalCompressionBaselineCheckpointReport:
    """Warm-start the wrapper's HLT ParT backbone and attach provenance."""

    return load_hlt_part_baseline_checkpoint(checkpoint_path, model, **kwargs)


def compute_init_logit_diff_vs_baseline(
    model: Any,
    tokens_or_batch: Any,
    mask: Any | None = None,
    *,
    max_constits: int | None = None,
    weight_threshold: float = 0.0,
    attach: bool = True,
) -> dict[str, Any]:
    """Compare local-compression logits to the same ParT on unmodified inputs."""

    torch = require_torch()
    if not hasattr(model, "part_model") or not hasattr(model, "build_canonical_inputs"):
        raise ValueError("init-logit comparison requires LocalCompressionFeatureAdapterParT")
    was_training = bool(model.training)
    model.eval()
    with torch.no_grad():
        from .model import _coerce_tokens_and_mask

        raw_tokens, raw_mask = _coerce_tokens_and_mask(tokens_or_batch, mask)
        if max_constits is None:
            max_constits = int(raw_tokens.shape[1])
        canonical = model.build_canonical_inputs(
            raw_tokens,
            raw_mask,
            max_constits=int(max_constits),
            weight_threshold=float(weight_threshold),
        )
        baseline_logits = model.part_model(
            canonical.points,
            canonical.features,
            canonical.lorentz_vectors,
            canonical.mask,
        )
        output = model(
            raw_tokens,
            raw_mask,
            return_outputs=True,
            max_constits=int(max_constits),
            weight_threshold=float(weight_threshold),
        )
        diff = (output.logits - baseline_logits).detach().float()
        result = {
            "contract": LOCAL_COMPRESSION_CHECKPOINT_CONTRACT,
            "baseline_logits_shape": list(baseline_logits.shape),
            "adapter_logits_shape": list(output.logits.shape),
            "max_abs_logit_diff": float(diff.abs().max().cpu().item()) if diff.numel() else 0.0,
            "mean_abs_logit_diff": float(diff.abs().mean().cpu().item()) if diff.numel() else 0.0,
            "allclose_atol_1e_6": bool(torch.allclose(output.logits, baseline_logits, atol=1.0e-6, rtol=1.0e-6)),
        }
    if was_training:
        model.train()
    if bool(attach) and hasattr(model, "init_logit_diff_vs_baseline"):
        setattr(model, "init_logit_diff_vs_baseline", dict(result))
    return result


__all__ = [
    "LOCAL_COMPRESSION_CHECKPOINT_CONTRACT",
    "LOCAL_COMPRESSION_CHECKPOINT_STEP",
    "LocalCompressionBaselineCheckpointReport",
    "compute_init_logit_diff_vs_baseline",
    "load_hlt_part_baseline_checkpoint",
    "sha256_file",
    "warm_start_local_compression_part_model",
]
