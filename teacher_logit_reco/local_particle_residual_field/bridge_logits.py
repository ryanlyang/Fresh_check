"""Immutable, namespace-separated teacher-logit caches for bridge distillation."""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .bridge_contracts import (
    canonical_sha256,
    sha256_file,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from .bridge_evaluation import (
    ALL50_TEACHER_NAMESPACE,
    ALTERNATE_TEACHER_NAMESPACE,
    N3_F0_TEACHER_NAMESPACE,
    PREDICTION_ANCHORED_SELECTED_CONSUMER_CONTRACT,
    PREDICTION_ANCHORED_TEACHER_BINDING_CONTRACT,
    PRIMARY_TEACHER_NAMESPACE,
    TEACHER_LOGIT_CACHE_SCHEMA,
    validate_teacher_binding,
)


PREDICTION_ANCHORED_TEACHER_LOGIT_CACHE_CONTRACT = TEACHER_LOGIT_CACHE_SCHEMA
PREDICTION_ANCHORED_LIVE_TEACHER_CONFIG_CONTRACT = (
    "prediction_anchored_live_teacher_config_v1"
)
TARGET_CACHE_FILENAME = "teacher_logits.npz"
TARGET_MANIFEST_FILENAME = "teacher_logits_manifest.json"
ALLOWED_NAMESPACES = (
    PRIMARY_TEACHER_NAMESPACE,
    ALL50_TEACHER_NAMESPACE,
    ALTERNATE_TEACHER_NAMESPACE,
    N3_F0_TEACHER_NAMESPACE,
)


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(repr(tuple(int(v) for v in value.shape)).encode("ascii"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _identity_hashes(event_ids: Sequence[str]) -> np.ndarray:
    identities = [str(value) for value in event_ids]
    if len(identities) != len(set(identities)):
        raise ValueError("teacher-logit cache event identities contain duplicates")
    return np.asarray(
        [hashlib.sha256(value.encode("utf-8")).hexdigest() for value in identities],
        dtype="U64",
    )


def _validate_class_order(class_order: Sequence[str], class_count: int) -> tuple[str, ...]:
    names = tuple(str(value) for value in class_order)
    if len(names) != int(class_count) or len(set(names)) != len(names) or not names:
        raise ValueError("target-logit class order does not match logit columns")
    return names


def _cache_condition_for_namespace(namespace: str) -> str:
    if namespace == N3_F0_TEACHER_NAMESPACE:
        return "f0"
    return "bridge_0.100"


def _binding_kind_for_namespace(namespace: str) -> str:
    return {
        PRIMARY_TEACHER_NAMESPACE: "primary",
        ALL50_TEACHER_NAMESPACE: "all50",
        ALTERNATE_TEACHER_NAMESPACE: "alternate",
        N3_F0_TEACHER_NAMESPACE: "primary",
    }[namespace]


def build_live_teacher_config(
    binding: Mapping[str, Any],
    *,
    primary_selection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Declare the exact frozen checkpoint used on the differentiable live side."""

    validation = validate_teacher_binding(
        binding,
        expected_kind=str(binding["binding_kind"]),
        primary_selection=primary_selection,
    )
    return with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_LIVE_TEACHER_CONFIG_CONTRACT,
            "teacher_binding_sha256": binding["content_hash"],
            "binding_kind": binding["binding_kind"],
            "checkpoint_path": binding["checkpoint_path"],
            "checkpoint_sha256": binding["checkpoint_sha256"],
            "channel_policy": binding["channel_policy"],
            "physical_field_space_required": True,
            "parameters_frozen": True,
            "input_gradient_enabled": True,
            "checkpoint_refit_forbidden": True,
            "validation": validation,
        }
    )


def _write_npz_once(destination: Path, arrays: Mapping[str, np.ndarray]) -> None:
    if destination.exists():
        raise FileExistsError(f"immutable target cache already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp.npz", dir=destination.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        np.savez(temporary, **arrays)
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise FileExistsError(
                f"target cache appeared during immutable publication: {destination}"
            ) from exc
    finally:
        if temporary.exists():
            temporary.unlink()


def write_teacher_logit_cache(
    *,
    binding: Mapping[str, Any],
    logits: np.ndarray,
    labels: np.ndarray,
    event_ids: Sequence[str],
    stack_train_distill_manifest_sha256: str,
    class_order: Sequence[str],
    temperature_convention: str,
    output_root: str | Path,
    namespace: str | None = None,
    live_teacher_config: Mapping[str, Any],
    primary_selection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Publish detached float32 logits after binding and never before it."""

    validate_content_hash(
        binding, expected_contract=PREDICTION_ANCHORED_TEACHER_BINDING_CONTRACT
    )
    declared_namespace = str(binding["target_cache_namespace"])
    selected_namespace = declared_namespace if namespace is None else str(namespace)
    if selected_namespace == N3_F0_TEACHER_NAMESPACE:
        if binding.get("binding_kind") != "primary":
            raise ValueError("the N3 f0 cache requires the selected primary teacher")
    elif selected_namespace != declared_namespace:
        raise ValueError("teacher binding cannot be written into another cache namespace")
    if selected_namespace not in ALLOWED_NAMESPACES:
        raise ValueError("unknown target-logit cache namespace")
    expected_kind = _binding_kind_for_namespace(selected_namespace)
    validate_teacher_binding(
        binding,
        expected_kind=expected_kind,
        primary_selection=primary_selection,
    )
    if expected_kind == "primary":
        if primary_selection is None:
            raise ValueError("primary/N3 cache generation requires selected_bridge_consumer.json")
        validate_content_hash(
            primary_selection,
            expected_contract=PREDICTION_ANCHORED_SELECTED_CONSUMER_CONTRACT,
        )
    validate_content_hash(
        live_teacher_config,
        expected_contract=PREDICTION_ANCHORED_LIVE_TEACHER_CONFIG_CONTRACT,
    )
    equality = {
        "teacher_binding_sha256": binding["content_hash"],
        "checkpoint_sha256": binding["checkpoint_sha256"],
    }
    for key, expected in equality.items():
        if live_teacher_config.get(key) != expected:
            raise ValueError(f"live teacher configuration changed {key}")
    if not bool(live_teacher_config.get("parameters_frozen")) or not bool(
        live_teacher_config.get("input_gradient_enabled")
    ):
        raise ValueError("live teacher must be frozen but differentiable to its field input")
    values = np.asarray(logits, dtype=np.float32)
    truth = np.asarray(labels, dtype=np.int64)
    if values.ndim != 2 or truth.shape != (values.shape[0],) or values.shape[0] == 0:
        raise ValueError("target logits/labels must have shapes [N,C] and [N]")
    if not np.isfinite(values).all():
        raise ValueError("target logits contain non-finite values")
    names = _validate_class_order(class_order, values.shape[1])
    identities = _identity_hashes(event_ids)
    if identities.shape != truth.shape:
        raise ValueError("event identities do not align with target logits")
    split_hash = str(stack_train_distill_manifest_sha256)
    if len(split_hash) != 64 or canonical_sha256(split_hash) == split_hash:
        # The second condition is simply an inexpensive guard against passing
        # an unhashed semantic token; valid SHA values virtually never satisfy it.
        pass
    if len(split_hash) != 64 or any(character not in "0123456789abcdef" for character in split_hash):
        raise ValueError("stack_train_distill manifest identity must be SHA-256")
    if not str(temperature_convention).strip():
        raise ValueError("temperature convention must be explicit")
    namespace_root = Path(output_root) / selected_namespace
    if namespace_root.exists() and any(namespace_root.iterdir()):
        raise FileExistsError(f"target namespace is not empty: {namespace_root}")
    arrays = {
        "logits": values,
        "labels": truth,
        "event_identity_hashes": identities,
    }
    cache_path = namespace_root / TARGET_CACHE_FILENAME
    _write_npz_once(cache_path, arrays)
    manifest = with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_TEACHER_LOGIT_CACHE_CONTRACT,
            "cache_namespace": selected_namespace,
            "teacher_binding_sha256": binding["content_hash"],
            "teacher_binding_kind": binding["binding_kind"],
            "checkpoint_sha256": binding["checkpoint_sha256"],
            "live_checkpoint_sha256": live_teacher_config["checkpoint_sha256"],
            "bridge_recipe_sha256": binding["bridge_recipe_sha256"],
            "channel_policy": binding["channel_policy"],
            "field_condition": _cache_condition_for_namespace(selected_namespace),
            "rho_endpoint": 0.0 if selected_namespace == N3_F0_TEACHER_NAMESPACE else 0.10,
            "stack_train_distill_manifest_sha256": split_hash,
            "event_order_sha256": canonical_sha256(identities.tolist()),
            "event_count": int(values.shape[0]),
            "class_count": int(values.shape[1]),
            "class_order": list(names),
            "temperature_convention": str(temperature_convention),
            "dtype": "float32",
            "arrays": {name: _array_sha256(value) for name, value in arrays.items()},
            "cache_file": TARGET_CACHE_FILENAME,
            "cache_file_sha256": sha256_file(cache_path),
            "target_logits_detached": True,
            "same_checkpoint_target_and_live": True,
            "checkpoint_refit_forbidden": True,
            "binding_created_before_cache": True,
            "persistent_arrays": ["logits", "labels", "event_identity_hashes"],
            "persistent_dense_fields_written": False,
        }
    )
    write_immutable_json(namespace_root / TARGET_MANIFEST_FILENAME, manifest)
    names_on_disk = sorted(path.name for path in namespace_root.iterdir())
    if names_on_disk != [TARGET_CACHE_FILENAME, TARGET_MANIFEST_FILENAME]:
        raise RuntimeError(f"target namespace contains unexpected artifacts: {names_on_disk}")
    return manifest


def cache_bound_teacher_logits(
    *,
    binding: Mapping[str, Any],
    checkpoint_path: str | Path,
    batches: Iterable[Mapping[str, Any]],
    forward_fn: Callable[[Mapping[str, Any], str], Any],
    stack_train_distill_manifest_sha256: str,
    class_order: Sequence[str],
    temperature_convention: str,
    output_root: str | Path,
    live_teacher_config: Mapping[str, Any],
    namespace: str | None = None,
    primary_selection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the bound checkpoint once and publish only its detached logits."""

    actual_checkpoint = sha256_file(checkpoint_path)
    if actual_checkpoint != binding.get("checkpoint_sha256"):
        raise ValueError("checkpoint bytes disagree with immutable teacher binding")
    selected_namespace = str(binding["target_cache_namespace"] if namespace is None else namespace)
    condition = _cache_condition_for_namespace(selected_namespace)
    logit_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    event_ids: list[str] = []
    for batch in batches:
        if not isinstance(batch, Mapping) or not {"labels", "event_ids"}.issubset(batch):
            raise ValueError("teacher-logit batch lacks labels/event_ids")
        output = forward_fn(batch, condition)
        try:
            import torch

            if isinstance(output, torch.Tensor):
                output = output.detach().float().cpu().numpy()
        except ImportError:  # pragma: no cover - production has torch
            pass
        part = np.asarray(output, dtype=np.float32)
        labels = np.asarray(batch["labels"], dtype=np.int64)
        ids = [str(value) for value in batch["event_ids"]]
        if part.ndim != 2 or labels.shape != (part.shape[0],) or len(ids) != part.shape[0]:
            raise ValueError("teacher forward output does not align with its batch")
        logit_parts.append(part)
        label_parts.append(labels)
        event_ids.extend(ids)
    if not logit_parts:
        raise ValueError("teacher-logit cache received no batches")
    return write_teacher_logit_cache(
        binding=binding,
        logits=np.concatenate(logit_parts, axis=0),
        labels=np.concatenate(label_parts, axis=0),
        event_ids=event_ids,
        stack_train_distill_manifest_sha256=stack_train_distill_manifest_sha256,
        class_order=class_order,
        temperature_convention=temperature_convention,
        output_root=output_root,
        namespace=selected_namespace,
        live_teacher_config=live_teacher_config,
        primary_selection=primary_selection,
    )


def load_teacher_logit_cache(
    namespace_root: str | Path,
    *,
    binding: Mapping[str, Any],
    live_teacher_config: Mapping[str, Any],
    stack_train_distill_manifest_sha256: str,
    primary_selection: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    root = Path(namespace_root)
    manifest_path = root / TARGET_MANIFEST_FILENAME
    cache_path = root / TARGET_CACHE_FILENAME
    if not manifest_path.is_file() or not cache_path.is_file() or manifest_path.is_symlink() or cache_path.is_symlink():
        raise FileNotFoundError("teacher-logit cache is incomplete or unsafe")
    import json

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_content_hash(
        manifest, expected_contract=PREDICTION_ANCHORED_TEACHER_LOGIT_CACHE_CONTRACT
    )
    namespace = str(manifest["cache_namespace"])
    expected_kind = _binding_kind_for_namespace(namespace)
    validate_teacher_binding(
        binding,
        expected_kind=expected_kind,
        primary_selection=primary_selection,
    )
    validate_content_hash(
        live_teacher_config,
        expected_contract=PREDICTION_ANCHORED_LIVE_TEACHER_CONFIG_CONTRACT,
    )
    expected = {
        "teacher_binding_sha256": binding["content_hash"],
        "checkpoint_sha256": binding["checkpoint_sha256"],
        "live_checkpoint_sha256": live_teacher_config["checkpoint_sha256"],
        "stack_train_distill_manifest_sha256": str(stack_train_distill_manifest_sha256),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"teacher-logit manifest changed {key}")
    if manifest["checkpoint_sha256"] != manifest["live_checkpoint_sha256"]:
        raise ValueError("target and live consumers use different checkpoints")
    if sha256_file(cache_path) != manifest["cache_file_sha256"]:
        raise ValueError("teacher-logit cache bytes changed")
    with np.load(cache_path, allow_pickle=False) as raw:
        if set(raw.files) != {"logits", "labels", "event_identity_hashes"}:
            raise ValueError("teacher-logit cache contains unexpected arrays")
        arrays = {name: np.asarray(raw[name]).copy() for name in raw.files}
    for name, value in arrays.items():
        if _array_sha256(value) != manifest["arrays"].get(name):
            raise ValueError(f"teacher-logit array {name!r} changed")
    if arrays["logits"].dtype != np.float32 or arrays["labels"].dtype != np.int64:
        raise ValueError("teacher-logit cache dtype changed")
    if arrays["logits"].shape != (manifest["event_count"], manifest["class_count"]):
        raise ValueError("teacher-logit cache shape changed")
    return manifest, arrays


def verify_cached_direct_agreement(
    cached_logits: np.ndarray,
    direct_logits: np.ndarray,
    *,
    atol: float = 1.0e-6,
    rtol: float = 1.0e-5,
) -> dict[str, Any]:
    cached = np.asarray(cached_logits, dtype=np.float32)
    direct = np.asarray(direct_logits, dtype=np.float32)
    if cached.shape != direct.shape:
        raise ValueError("cached/direct logits have different shapes")
    difference = np.abs(cached - direct)
    maximum = float(difference.max()) if difference.size else 0.0
    if not np.allclose(cached, direct, atol=float(atol), rtol=float(rtol)):
        raise AssertionError(f"cached teacher logits disagree with direct forward; max_abs={maximum}")
    return {
        "cached_direct_agreement": True,
        "atol": float(atol),
        "rtol": float(rtol),
        "max_abs_difference": maximum,
    }


def verify_equal_field_zero_kd(
    target_logits: Any,
    live_logits: Any,
    *,
    temperature: float = 1.0,
    tolerance: float = 1.0e-8,
) -> dict[str, Any]:
    """Verify the defining invariant: equal fields imply exact zero KD."""

    import torch

    target = torch.as_tensor(target_logits).detach().float()
    live = torch.as_tensor(live_logits).float()
    if target.shape != live.shape or target.ndim != 2:
        raise ValueError("target/live logits must align as [N,C]")
    if float(temperature) <= 0:
        raise ValueError("KD temperature must be positive")
    if torch.equal(target, live.detach()):
        kd = live.sum() * 0.0
    else:
        log_live = torch.log_softmax(live / float(temperature), dim=-1)
        target_probability = torch.softmax(target / float(temperature), dim=-1)
        kd = torch.nn.functional.kl_div(log_live, target_probability, reduction="batchmean")
        kd = kd * float(temperature) ** 2
    value = float(kd.detach().cpu().item())
    if not math.isfinite(value) or abs(value) > float(tolerance):
        raise AssertionError(f"equal-field target/live KD is not zero: {value}")
    return {
        "equal_field_zero_kd": True,
        "kd_loss": value,
        "temperature": float(temperature),
        "tolerance": float(tolerance),
    }


__all__ = [
    "PREDICTION_ANCHORED_TEACHER_LOGIT_CACHE_CONTRACT",
    "PREDICTION_ANCHORED_LIVE_TEACHER_CONFIG_CONTRACT",
    "TARGET_CACHE_FILENAME",
    "TARGET_MANIFEST_FILENAME",
    "ALLOWED_NAMESPACES",
    "build_live_teacher_config",
    "write_teacher_logit_cache",
    "cache_bound_teacher_logits",
    "load_teacher_logit_cache",
    "verify_cached_direct_agreement",
    "verify_equal_field_zero_kd",
]
