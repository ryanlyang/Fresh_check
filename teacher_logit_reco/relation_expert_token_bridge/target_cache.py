"""Identity-bound, audited, resumable offline target caches for RETB."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .contracts import (
    bind_source,
    canonical_json_bytes,
    load_hashed_json,
    require_sha256,
    validate_content_hash,
    with_content_hash,
    write_immutable_bytes,
    write_immutable_json,
)
from .registry import EXPERT_ORDER


TARGET_CACHE_POLICY_CONTRACT = "retb_offline_target_cache_policy_v1"
TARGET_CACHE_SPEC_CONTRACT = "retb_offline_target_cache_specification_v1"
TARGET_STORAGE_AUDIT_CONTRACT = "retb_target_storage_audit_v1"
TARGET_SHARD_CONTRACT = "retb_offline_target_shard_v1"
TARGET_CACHE_MANIFEST_CONTRACT = "retb_offline_target_cache_manifest_v1"
TARGET_NORMALIZER_CONTRACT = "retb_target_token_normalizer_v1"
TARGET_NORMALIZER_SET_CONTRACT = "retb_target_normalizer_set_v1"
SEALED_INPUT_PREPARATION_CONTRACT = "retb_sealed_input_preparation_v1"

PRELOCK_TARGET_SPLITS = ("model_train", "val_stop", "val_design")
TARGET_MODES = (
    "T0_PURE",
    "T1_ANCHORED_BRIDGE",
    "T1_TASK_BRIDGE",
    "T2_PROJECT",
)
DEFAULT_SHARD_SIZE = 2048
AUDIT_SAMPLE_SIZE = 4096


def _array_hash(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(b"retb_numpy_array_v1\0")
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical_json_bytes(list(array.shape)))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def identity_order_sha256(
    identities: Sequence[str], labels: np.ndarray
) -> str:
    canonical = tuple(str(value) for value in identities)
    truth = np.asarray(labels, dtype=np.int64)
    if (
        not canonical
        or len(canonical) != len(set(canonical))
        or truth.shape != (len(canonical),)
        or bool(((truth < 0) | (truth >= 10)).any())
    ):
        raise ValueError("target-cache identity population differs")
    digest = hashlib.sha256()
    digest.update(b"retb_target_identity_order_v1\0")
    for index, (identity, label) in enumerate(zip(canonical, truth)):
        digest.update(str(index).encode("ascii"))
        digest.update(b"\0")
        digest.update(identity.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(int(label)).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def build_target_cache_policy() -> dict[str, Any]:
    return with_content_hash(
        {
            "contract": TARGET_CACHE_POLICY_CONTRACT,
            "schema_version": 1,
            "prelock_splits": list(PRELOCK_TARGET_SPLITS),
            "target_modes": list(TARGET_MODES),
            "forbidden_prelock_modes": ["T3_LOGIT"],
            "default_shard_size": DEFAULT_SHARD_SIZE,
            "storage_audit": {
                "sample": "first_min_4096_canonical_identity_indices",
                "maximum_sample_size": AUDIT_SAMPLE_SIZE,
                "float16_token_roundtrip_absolute_error_max": 5.0e-4,
                "float16_expert_logit_absolute_error_max": 2.0e-4,
                "predicted_class_identities_exact": True,
                "failed_float16_falls_back_to_float32": True,
                "scientific_failure": False,
            },
            "payload": {
                "tokens": "audited_float16_or_float32_per_expert",
                "expert_logits": "float32",
                "labels": "int64",
                "canonical_identity_indices": "int64",
                "consumer_token_dtype": "float32",
            },
            "normalization": {
                "fit_split": "model_train_only",
                "granularity": "expert_slot_channel",
                "statistics_dtype": "float32",
                "accumulator_dtype": "float64",
                "standard_deviation_floor": 1.0e-4,
                "validation_or_test_fit_forbidden": True,
            },
            "publication": {
                "atomic": True,
                "completed_shards_reused_by_manifest_and_file_hash": True,
                "partial_final_paths_rejected": True,
                "complete_identity_coverage_required": True,
            },
            "performance_based_termination": False,
        }
    )


def _canonical_allocation(
    allocation: Mapping[str, Sequence[int]]
) -> dict[str, list[int]]:
    if set(allocation) != set(EXPERT_ORDER):
        raise ValueError("target-cache allocation coverage differs")
    output = {}
    for expert in EXPERT_ORDER:
        shape = list(allocation[expert])
        if (
            len(shape) != 2
            or int(shape[0]) not in {1, 2, 4, 8, 16}
            or int(shape[1]) not in {64, 128}
        ):
            raise ValueError("target-cache expert shape is unregistered")
        output[expert] = [int(shape[0]), int(shape[1])]
    return output


def _target_descriptor(
    expert: str,
    mode: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if mode not in TARGET_MODES:
        raise ValueError("target-cache mode is not token-valued")
    required = {
        "checkpoint_sha256",
        "registration_sha256",
        "slot_query_sha256",
        "eligibility_sha256",
    }
    if not required.issubset(payload):
        raise ValueError(f"target descriptor for {expert} lacks parents")
    result = {
        "target_mode": mode,
        **{
            name: require_sha256(
                payload[name], name=f"target_descriptors.{expert}.{name}"
            )
            for name in sorted(required)
        },
        "pilot_checkpoint_sha256": None,
        "content_certification_sha256": None,
        "noninferiority_sha256": None,
    }
    if mode != "T0_PURE":
        for name in (
            "pilot_checkpoint_sha256",
            "content_certification_sha256",
            "noninferiority_sha256",
        ):
            result[name] = require_sha256(
                payload.get(name),
                name=f"target_descriptors.{expert}.{name}",
            )
    elif any(
        payload.get(name) is not None
        for name in (
            "pilot_checkpoint_sha256",
            "content_certification_sha256",
            "noninferiority_sha256",
        )
    ):
        raise ValueError("T0 target descriptor cannot bind bridge certificates")
    return result


def build_target_cache_specification(
    *,
    split: str,
    pipeline_seed: int,
    shape_id: str,
    allocation: Mapping[str, Sequence[int]],
    target_tuple: Sequence[str],
    target_descriptors: Mapping[str, Mapping[str, Any]],
    selected_target_lineage_sha256: str,
    target_cache_namespace: str,
    locked_coordinate_contract_sha256: str,
    locked_coordinate_selection_sha256: str,
    offline_fusion_checkpoint_sha256: str,
    offline_fusion_registration_sha256: str,
    normalizer_set_sha256: str,
    identity_manifest_sha256: str,
    identity_order_sha256: str,
    event_count: int,
) -> dict[str, Any]:
    if (
        split not in PRELOCK_TARGET_SPLITS
        or int(pipeline_seed) not in {101, 202, 303}
        or int(event_count) <= 0
        or not str(target_cache_namespace).strip()
        or len(target_tuple) != len(EXPERT_ORDER)
        or set(target_descriptors) != set(EXPERT_ORDER)
    ):
        raise ValueError("target-cache specification identity differs")
    modes = tuple(str(value) for value in target_tuple)
    if any(mode not in TARGET_MODES for mode in modes):
        raise ValueError("target-cache tuple contains a non-token target")
    descriptors = {
        expert: _target_descriptor(
            expert, modes[index], target_descriptors[expert]
        )
        for index, expert in enumerate(EXPERT_ORDER)
    }
    return with_content_hash(
        {
            "contract": TARGET_CACHE_SPEC_CONTRACT,
            "schema_version": 1,
            "split": split,
            "pipeline_seed": int(pipeline_seed),
            "shape_id": str(shape_id),
            "expert_order": list(EXPERT_ORDER),
            "allocation": _canonical_allocation(allocation),
            "target_tuple": list(modes),
            "target_descriptors": descriptors,
            "selected_target_lineage_sha256": require_sha256(
                selected_target_lineage_sha256,
                name="selected_target_lineage_sha256",
            ),
            "target_cache_namespace": str(target_cache_namespace),
            "locked_coordinate_contract_sha256": require_sha256(
                locked_coordinate_contract_sha256,
                name="locked_coordinate_contract_sha256",
            ),
            "locked_coordinate_selection_sha256": require_sha256(
                locked_coordinate_selection_sha256,
                name="locked_coordinate_selection_sha256",
            ),
            "offline_fusion_checkpoint_sha256": require_sha256(
                offline_fusion_checkpoint_sha256,
                name="offline_fusion_checkpoint_sha256",
            ),
            "offline_fusion_registration_sha256": require_sha256(
                offline_fusion_registration_sha256,
                name="offline_fusion_registration_sha256",
            ),
            "normalizer_set_sha256": require_sha256(
                normalizer_set_sha256, name="normalizer_set_sha256"
            ),
            "identity_manifest_sha256": require_sha256(
                identity_manifest_sha256, name="identity_manifest_sha256"
            ),
            "identity_order_sha256": require_sha256(
                identity_order_sha256, name="identity_order_sha256"
            ),
            "event_count": int(event_count),
            "prelock_target_cache": True,
            "selection_eligible": False,
            "performance_based_termination": False,
        }
    )


def validate_target_cache_specification(payload: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        payload, expected_contract=TARGET_CACHE_SPEC_CONTRACT
    )
    expected = build_target_cache_specification(
        split=payload["split"],
        pipeline_seed=int(payload["pipeline_seed"]),
        shape_id=payload["shape_id"],
        allocation=payload["allocation"],
        target_tuple=payload["target_tuple"],
        target_descriptors=payload["target_descriptors"],
        selected_target_lineage_sha256=payload[
            "selected_target_lineage_sha256"
        ],
        target_cache_namespace=payload["target_cache_namespace"],
        locked_coordinate_contract_sha256=payload[
            "locked_coordinate_contract_sha256"
        ],
        locked_coordinate_selection_sha256=payload[
            "locked_coordinate_selection_sha256"
        ],
        offline_fusion_checkpoint_sha256=payload[
            "offline_fusion_checkpoint_sha256"
        ],
        offline_fusion_registration_sha256=payload[
            "offline_fusion_registration_sha256"
        ],
        normalizer_set_sha256=payload["normalizer_set_sha256"],
        identity_manifest_sha256=payload["identity_manifest_sha256"],
        identity_order_sha256=payload["identity_order_sha256"],
        event_count=int(payload["event_count"]),
    )
    actual = dict(payload)
    actual.pop("content_hash", None)
    actual.pop("source", None)
    expected.pop("content_hash")
    if actual != expected:
        raise ValueError("target-cache specification semantics differ")
    return digest


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value)


def audit_target_storage(
    *,
    expert_id: str,
    tokens: np.ndarray,
    stored_expert_logits: np.ndarray,
    reproduce_logits: Callable[[np.ndarray], Any],
    checkpoint_sha256: str,
    slot_query_sha256: str,
    identity_order_sha256: str,
    sample_start: int = 0,
) -> dict[str, Any]:
    source = np.asarray(tokens, dtype=np.float32)
    reference_logits = np.asarray(stored_expert_logits, dtype=np.float32)
    if (
        expert_id not in EXPERT_ORDER
        or source.ndim != 3
        or source.shape[0] == 0
        or reference_logits.shape != (len(source), 10)
        or not np.isfinite(source).all()
        or not np.isfinite(reference_logits).all()
    ):
        raise ValueError("target-storage audit population differs")
    original_logits = _as_numpy(reproduce_logits(source)).astype(
        np.float32, copy=False
    )
    if (
        original_logits.shape != reference_logits.shape
        or not np.isfinite(original_logits).all()
    ):
        raise ValueError("target-storage logit reproducer differs")
    source_logit_error = float(
        np.max(np.abs(original_logits - reference_logits))
    )
    if source_logit_error > 2.0e-6 or not np.array_equal(
        original_logits.argmax(axis=1), reference_logits.argmax(axis=1)
    ):
        raise RuntimeError(
            "float32 target tokens do not reproduce frozen expert logits"
        )
    roundtrip = source.astype(np.float16).astype(np.float32)
    roundtrip_logits = _as_numpy(reproduce_logits(roundtrip)).astype(
        np.float32, copy=False
    )
    token_error = float(np.max(np.abs(roundtrip - source)))
    logit_error = float(
        np.max(np.abs(roundtrip_logits - reference_logits))
    )
    class_exact = bool(
        np.array_equal(
            roundtrip_logits.argmax(axis=1),
            reference_logits.argmax(axis=1),
        )
    )
    passed = (
        token_error <= 5.0e-4
        and logit_error <= 2.0e-4
        and class_exact
    )
    return with_content_hash(
        {
            "contract": TARGET_STORAGE_AUDIT_CONTRACT,
            "schema_version": 1,
            "expert_id": expert_id,
            "checkpoint_sha256": require_sha256(
                checkpoint_sha256, name="checkpoint_sha256"
            ),
            "slot_query_sha256": require_sha256(
                slot_query_sha256, name="slot_query_sha256"
            ),
            "identity_order_sha256": require_sha256(
                identity_order_sha256, name="identity_order_sha256"
            ),
            "sample_start": int(sample_start),
            "sample_count": int(len(source)),
            "sample_rule": "first_min_4096_canonical_identity_indices",
            "source_float32_logit_max_absolute_error": source_logit_error,
            "float16_token_roundtrip_max_absolute_error": token_error,
            "float16_expert_logit_max_absolute_error": logit_error,
            "predicted_class_identities_exact": class_exact,
            "thresholds": {
                "token_absolute_error_max": 5.0e-4,
                "expert_logit_absolute_error_max": 2.0e-4,
            },
            "float16_audit_passed": passed,
            "selected_storage_dtype": "float16" if passed else "float32",
            "float16_failure_stops_workflow": False,
        }
    )


def validate_target_storage_audit(payload: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        payload, expected_contract=TARGET_STORAGE_AUDIT_CONTRACT
    )
    passed = (
        float(payload["float16_token_roundtrip_max_absolute_error"])
        <= 5.0e-4
        and float(payload["float16_expert_logit_max_absolute_error"])
        <= 2.0e-4
        and bool(payload["predicted_class_identities_exact"])
    )
    if (
        bool(payload["float16_audit_passed"]) != passed
        or payload["selected_storage_dtype"]
        != ("float16" if passed else "float32")
        or int(payload["sample_count"]) <= 0
        or int(payload["sample_count"]) > AUDIT_SAMPLE_SIZE
    ):
        raise ValueError("target-storage audit semantics differ")
    return digest


def _validate_generated(
    generated: Mapping[str, Any],
    *,
    count: int,
    allocation: Mapping[str, Sequence[int]],
) -> dict[str, dict[str, np.ndarray]]:
    if set(generated) != {"tokens", "expert_logits"}:
        raise ValueError("target generator fields differ")
    if set(generated["tokens"]) != set(EXPERT_ORDER) or set(
        generated["expert_logits"]
    ) != set(EXPERT_ORDER):
        raise ValueError("target generator expert coverage differs")
    output = {"tokens": {}, "expert_logits": {}}
    for expert in EXPERT_ORDER:
        tokens = _as_numpy(generated["tokens"][expert]).astype(
            np.float32, copy=False
        )
        logits = _as_numpy(generated["expert_logits"][expert]).astype(
            np.float32, copy=False
        )
        if (
            tokens.shape
            != (
                count,
                int(allocation[expert][0]),
                int(allocation[expert][1]),
            )
            or logits.shape != (count, 10)
            or not np.isfinite(tokens).all()
            or not np.isfinite(logits).all()
        ):
            raise ValueError(f"generated target arrays differ for {expert}")
        output["tokens"][expert] = tokens
        output["expert_logits"][expert] = logits
    return output


def _npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_token_head_reproducer(
    *,
    checkpoint_path: str | Path,
    expected_checkpoint_sha256: str,
    target_mode: str,
    token_dimension: int,
) -> Callable[[np.ndarray], np.ndarray]:
    """Load only the immutable token head needed for cache parity checks."""
    try:
        import torch
    except ImportError as error:  # pragma: no cover - deployment dependency
        raise RuntimeError("PyTorch is required to load RETB target heads") from error

    from .summary_tokens import TokenOnlyExpertHead

    path = Path(checkpoint_path)
    expected = require_sha256(
        expected_checkpoint_sha256, name="expected_checkpoint_sha256"
    )
    if (
        not path.is_file()
        or path.is_symlink()
        or _file_sha256(path) != expected
        or target_mode not in TARGET_MODES
    ):
        raise ValueError("target-head checkpoint identity differs")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if target_mode == "T0_PURE":
        state = payload.get("model_state_dict", payload)
        prefixes = ("head.", "expert_model.head.")
    else:
        state = payload.get("offline_target_state_dict")
        if state is None:
            raise ValueError("bridge checkpoint lacks offline target state")
        prefixes = (
            ("projected_expert_head.",)
            if target_mode == "T2_PROJECT"
            else ("expert_model.head.",)
        )
    if not isinstance(state, Mapping):
        raise ValueError("target-head checkpoint state differs")
    selected: dict[str, Any] | None = None
    for prefix in prefixes:
        candidate = {
            str(name)[len(prefix) :]: value
            for name, value in state.items()
            if str(name).startswith(prefix)
        }
        if candidate:
            selected = candidate
            break
    if selected is None:
        raise ValueError("target-head parameters are absent from checkpoint")
    head = TokenOnlyExpertHead(
        token_dimension=int(token_dimension), num_classes=10
    )
    head.load_state_dict(selected, strict=True)
    head.eval()

    def reproduce(tokens: np.ndarray) -> np.ndarray:
        values = np.asarray(tokens, dtype=np.float32)
        with torch.no_grad():
            output = head(torch.from_numpy(values))
        return output.detach().cpu().numpy().astype(np.float32, copy=False)

    return reproduce


def _reuse_shard(
    *,
    manifest_path: Path,
    expected_spec_sha256: str,
    expected_index: int,
) -> dict[str, Any]:
    manifest = load_hashed_json(
        manifest_path, expected_contract=TARGET_SHARD_CONTRACT
    )
    data_path = manifest_path.parent / manifest["npz_filename"]
    if (
        manifest["target_cache_specification_sha256"]
        != expected_spec_sha256
        or int(manifest["shard_index"]) != expected_index
        or not data_path.is_file()
        or data_path.is_symlink()
        or _file_sha256(data_path) != manifest["npz_sha256"]
    ):
        raise ValueError("reusable target shard differs")
    return manifest


def publish_offline_target_cache(
    *,
    output_dir: str | Path,
    specification: Mapping[str, Any],
    identities: Sequence[str],
    labels: np.ndarray,
    generator: Callable[[int, int], Mapping[str, Any]],
    logit_reproducers: Mapping[str, Callable[[np.ndarray], Any]],
    source_snapshot: Mapping[str, Any],
    shard_size: int = DEFAULT_SHARD_SIZE,
) -> dict[str, Any]:
    spec_sha = validate_target_cache_specification(specification)
    if int(shard_size) <= 0 or set(logit_reproducers) != set(EXPERT_ORDER):
        raise ValueError("target-cache builder configuration differs")
    canonical_ids = tuple(str(value) for value in identities)
    truth = np.asarray(labels, dtype=np.int64)
    order_sha = identity_order_sha256(canonical_ids, truth)
    if (
        len(canonical_ids) != int(specification["event_count"])
        or order_sha != specification["identity_order_sha256"]
    ):
        raise ValueError("target-cache identity order differs from specification")
    root = Path(output_dir)
    if root.exists() and root.is_symlink():
        raise ValueError("target-cache output cannot be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    final_manifest = root / "target_cache_manifest.json"
    if final_manifest.exists():
        manifest = load_hashed_json(
            final_manifest, expected_contract=TARGET_CACHE_MANIFEST_CONTRACT
        )
        validate_offline_target_cache(
            final_manifest,
            expected_pipeline_seed=int(specification["pipeline_seed"]),
            expected_specification_sha256=spec_sha,
        )
        return manifest
    audit_count = min(AUDIT_SAMPLE_SIZE, len(canonical_ids))
    audit_generated = _validate_generated(
        generator(0, audit_count),
        count=audit_count,
        allocation=specification["allocation"],
    )
    audits = {}
    for expert in EXPERT_ORDER:
        descriptor = specification["target_descriptors"][expert]
        audits[expert] = bind_source(
            audit_target_storage(
                expert_id=expert,
                tokens=audit_generated["tokens"][expert],
                stored_expert_logits=audit_generated["expert_logits"][expert],
                reproduce_logits=logit_reproducers[expert],
                checkpoint_sha256=descriptor["checkpoint_sha256"],
                slot_query_sha256=descriptor["slot_query_sha256"],
                identity_order_sha256=order_sha,
            ),
            source_snapshot=source_snapshot,
        )
        write_immutable_json(
            root / f"storage_audit_{expert}.json", audits[expert]
        )
    shard_manifests = []
    for shard_index, start in enumerate(
        range(0, len(canonical_ids), int(shard_size))
    ):
        stop = min(start + int(shard_size), len(canonical_ids))
        shard_manifest_path = root / f"shard_{shard_index:06d}.json"
        if shard_manifest_path.exists():
            shard_manifests.append(
                _reuse_shard(
                    manifest_path=shard_manifest_path,
                    expected_spec_sha256=spec_sha,
                    expected_index=shard_index,
                )
            )
            continue
        npz_path = root / f"shard_{shard_index:06d}.npz"
        if npz_path.exists():
            raise FileExistsError(
                "target shard data exists without its immutable manifest"
            )
        generated = _validate_generated(
            generator(start, stop),
            count=stop - start,
            allocation=specification["allocation"],
        )
        arrays: dict[str, np.ndarray] = {
            "canonical_identity_indices": np.arange(
                start, stop, dtype=np.int64
            ),
            "labels": truth[start:stop].astype(np.int64, copy=False),
        }
        token_hashes, logit_hashes, storage_dtypes = {}, {}, {}
        normalization_moment_hashes = {}
        for expert in EXPERT_ORDER:
            dtype = audits[expert]["selected_storage_dtype"]
            stored_tokens = generated["tokens"][expert].astype(
                np.float16 if dtype == "float16" else np.float32
            )
            stored_logits = generated["expert_logits"][expert].astype(
                np.float32, copy=False
            )
            arrays[f"tokens_{expert}"] = stored_tokens
            arrays[f"logits_{expert}"] = stored_logits
            source64 = generated["tokens"][expert].astype(np.float64)
            arrays[f"normalization_sum_{expert}"] = source64.sum(
                axis=0, dtype=np.float64
            )
            arrays[f"normalization_sum_square_{expert}"] = np.square(
                source64
            ).sum(axis=0, dtype=np.float64)
            token_hashes[expert] = _array_hash(stored_tokens)
            logit_hashes[expert] = _array_hash(stored_logits)
            normalization_moment_hashes[expert] = {
                "sum": _array_hash(arrays[f"normalization_sum_{expert}"]),
                "sum_square": _array_hash(
                    arrays[f"normalization_sum_square_{expert}"]
                ),
            }
            storage_dtypes[expert] = dtype
        encoded = _npz_bytes(arrays)
        npz_publication = write_immutable_bytes(npz_path, encoded)
        shard_manifest = bind_source(
            with_content_hash(
                {
                    "contract": TARGET_SHARD_CONTRACT,
                    "schema_version": 1,
                    "target_cache_specification_sha256": spec_sha,
                    "pipeline_seed": int(specification["pipeline_seed"]),
                    "split": specification["split"],
                    "shape_id": specification["shape_id"],
                    "target_tuple": list(specification["target_tuple"]),
                    "shard_index": shard_index,
                    "start_index": start,
                    "stop_index_exclusive": stop,
                    "event_count": stop - start,
                    "npz_filename": npz_path.name,
                    "npz_sha256": npz_publication["file_sha256"],
                    "storage_dtype_by_expert": storage_dtypes,
                    "token_content_hashes": token_hashes,
                    "expert_logit_content_hashes": logit_hashes,
                    "normalization_moment_hashes": (
                        normalization_moment_hashes
                    ),
                    "identity_index_content_hash": _array_hash(
                        arrays["canonical_identity_indices"]
                    ),
                    "label_content_hash": _array_hash(arrays["labels"]),
                    "storage_audit_hashes": {
                        expert: audits[expert]["content_hash"]
                        for expert in EXPERT_ORDER
                    },
                    "consumer_token_dtype": "float32",
                    "complete": True,
                }
            ),
            source_snapshot=source_snapshot,
        )
        write_immutable_json(shard_manifest_path, shard_manifest)
        shard_manifests.append(shard_manifest)
    manifest = bind_source(
        with_content_hash(
            {
                "contract": TARGET_CACHE_MANIFEST_CONTRACT,
                "schema_version": 1,
                "target_cache_specification_sha256": spec_sha,
                "pipeline_seed": int(specification["pipeline_seed"]),
                "split": specification["split"],
                "shape_id": specification["shape_id"],
                "target_tuple": list(specification["target_tuple"]),
                "allocation": dict(specification["allocation"]),
                "event_count": len(canonical_ids),
                "identity_manifest_sha256": specification[
                    "identity_manifest_sha256"
                ],
                "identity_order_sha256": order_sha,
                "offline_fusion_checkpoint_sha256": specification[
                    "offline_fusion_checkpoint_sha256"
                ],
                "normalizer_set_sha256": specification[
                    "normalizer_set_sha256"
                ],
                "shard_size": int(shard_size),
                "shard_count": len(shard_manifests),
                "shards": [
                    {
                        "index": int(row["shard_index"]),
                        "manifest_filename": (
                            f"shard_{int(row['shard_index']):06d}.json"
                        ),
                        "manifest_sha256": row["content_hash"],
                        "npz_sha256": row["npz_sha256"],
                        "start_index": int(row["start_index"]),
                        "stop_index_exclusive": int(
                            row["stop_index_exclusive"]
                        ),
                    }
                    for row in shard_manifests
                ],
                "storage_audit_hashes": {
                    expert: audits[expert]["content_hash"]
                    for expert in EXPERT_ORDER
                },
                "complete_coverage": True,
                "tokens_load_as_float32": True,
                "performance_based_termination": False,
            }
        ),
        source_snapshot=source_snapshot,
    )
    write_immutable_json(final_manifest, manifest)
    return manifest


def _load_shard_arrays(
    root: Path,
    shard: Mapping[str, Any],
    *,
    expected_seed: int,
    expected_specification_sha256: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest_path = root / shard["manifest_filename"]
    manifest = load_hashed_json(
        manifest_path, expected_contract=TARGET_SHARD_CONTRACT
    )
    if (
        manifest["content_hash"] != shard["manifest_sha256"]
        or manifest["target_cache_specification_sha256"]
        != expected_specification_sha256
        or int(manifest["pipeline_seed"]) != int(expected_seed)
    ):
        raise ValueError("target shard seed/specification lineage differs")
    npz_path = root / manifest["npz_filename"]
    if (
        not npz_path.is_file()
        or npz_path.is_symlink()
        or _file_sha256(npz_path) != manifest["npz_sha256"]
        or manifest["npz_sha256"] != shard["npz_sha256"]
    ):
        raise ValueError("target shard bytes differ")
    with np.load(npz_path, allow_pickle=False) as payload:
        expected_fields = {"canonical_identity_indices", "labels"} | {
            f"{kind}_{expert}"
            for expert in EXPERT_ORDER
            for kind in (
                "tokens",
                "logits",
                "normalization_sum",
                "normalization_sum_square",
            )
        }
        if set(payload.files) != expected_fields:
            raise ValueError("target shard NPZ fields differ")
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    count = int(manifest["event_count"])
    expected_indices = np.arange(
        int(manifest["start_index"]),
        int(manifest["stop_index_exclusive"]),
        dtype=np.int64,
    )
    if (
        len(expected_indices) != count
        or not np.array_equal(
            arrays["canonical_identity_indices"], expected_indices
        )
        or arrays["labels"].dtype != np.int64
        or arrays["labels"].shape != (count,)
        or _array_hash(arrays["canonical_identity_indices"])
        != manifest["identity_index_content_hash"]
        or _array_hash(arrays["labels"]) != manifest["label_content_hash"]
    ):
        raise ValueError("target shard identity/label payload differs")
    for expert in EXPERT_ORDER:
        tokens = arrays[f"tokens_{expert}"]
        logits = arrays[f"logits_{expert}"]
        expected_dtype = np.dtype(manifest["storage_dtype_by_expert"][expert])
        if (
            tokens.dtype != expected_dtype
            or logits.dtype != np.float32
            or _array_hash(tokens) != manifest["token_content_hashes"][expert]
            or _array_hash(logits)
            != manifest["expert_logit_content_hashes"][expert]
        ):
            raise ValueError(f"target shard payload differs for {expert}")
        if (
            arrays[f"normalization_sum_{expert}"].dtype != np.float64
            or arrays[f"normalization_sum_square_{expert}"].dtype
            != np.float64
            or _array_hash(arrays[f"normalization_sum_{expert}"])
            != manifest["normalization_moment_hashes"][expert]["sum"]
            or _array_hash(
                arrays[f"normalization_sum_square_{expert}"]
            )
            != manifest["normalization_moment_hashes"][expert][
                "sum_square"
            ]
        ):
            raise ValueError(
                f"target normalization moments differ for {expert}"
            )
        arrays[f"tokens_{expert}"] = tokens.astype(np.float32)
    return manifest, arrays


def validate_offline_target_cache(
    manifest_path: str | Path,
    *,
    expected_pipeline_seed: int,
    expected_specification_sha256: str,
) -> str:
    path = Path(manifest_path)
    manifest = load_hashed_json(
        path, expected_contract=TARGET_CACHE_MANIFEST_CONTRACT
    )
    digest = validate_content_hash(manifest)
    if (
        int(expected_pipeline_seed) not in {101, 202, 303}
        or int(manifest["pipeline_seed"]) != int(expected_pipeline_seed)
        or manifest["target_cache_specification_sha256"]
        != require_sha256(
            expected_specification_sha256,
            name="expected_specification_sha256",
        )
        or not manifest["complete_coverage"]
    ):
        raise ValueError("target-cache seed/specification differs")
    audit_hashes = {}
    for expert in EXPERT_ORDER:
        audit = load_hashed_json(
            path.parent / f"storage_audit_{expert}.json",
            expected_contract=TARGET_STORAGE_AUDIT_CONTRACT,
        )
        audit_hashes[expert] = validate_target_storage_audit(audit)
        if (
            audit["expert_id"] != expert
            or int(audit.get("sample_start", -1)) != 0
            or audit["identity_order_sha256"]
            != manifest["identity_order_sha256"]
            or audit.get("source") != manifest.get("source")
        ):
            raise ValueError("target-cache storage audit lineage differs")
    if audit_hashes != manifest["storage_audit_hashes"]:
        raise ValueError("target-cache storage audit hashes differ")
    cursor = 0
    seen_shards = set()
    for expected_index, shard in enumerate(manifest["shards"]):
        if (
            int(shard["index"]) != expected_index
            or int(shard["start_index"]) != cursor
            or shard["manifest_filename"] in seen_shards
        ):
            raise ValueError("target-cache shard ordering differs")
        seen_shards.add(shard["manifest_filename"])
        shard_manifest, arrays = _load_shard_arrays(
            path.parent,
            shard,
            expected_seed=int(expected_pipeline_seed),
            expected_specification_sha256=expected_specification_sha256,
        )
        if (
            int(shard_manifest["start_index"]) != int(shard["start_index"])
            or int(shard_manifest["stop_index_exclusive"])
            != int(shard["stop_index_exclusive"])
            or shard_manifest["storage_audit_hashes"] != audit_hashes
            or shard_manifest.get("source") != manifest.get("source")
        ):
            raise ValueError("target-cache shard summary differs")
        count = int(shard_manifest["event_count"])
        if bool(((arrays["labels"] < 0) | (arrays["labels"] >= 10)).any()):
            raise ValueError("target-cache labels are outside the ten classes")
        for expert in EXPERT_ORDER:
            if (
                arrays[f"tokens_{expert}"].shape
                != (
                    count,
                    int(manifest["allocation"][expert][0]),
                    int(manifest["allocation"][expert][1]),
                )
                or arrays[f"logits_{expert}"].shape != (count, 10)
                or not np.isfinite(arrays[f"tokens_{expert}"]).all()
                or not np.isfinite(arrays[f"logits_{expert}"]).all()
            ):
                raise ValueError("target-cache shard array shape differs")
        cursor = int(shard_manifest["stop_index_exclusive"])
    if cursor != int(manifest["event_count"]):
        raise ValueError("target-cache identity coverage is incomplete")
    return digest


def load_offline_target_cache(
    manifest_path: str | Path,
    *,
    expected_pipeline_seed: int,
    expected_specification_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(manifest_path)
    validate_offline_target_cache(
        path,
        expected_pipeline_seed=expected_pipeline_seed,
        expected_specification_sha256=expected_specification_sha256,
    )
    manifest = load_hashed_json(
        path, expected_contract=TARGET_CACHE_MANIFEST_CONTRACT
    )
    pieces = list(
        iter_offline_target_cache(
            path,
            expected_pipeline_seed=expected_pipeline_seed,
            expected_specification_sha256=expected_specification_sha256,
        )
    )
    return manifest, {
        "canonical_identity_indices": np.concatenate(
            [row["canonical_identity_indices"] for row in pieces]
        ),
        "labels": np.concatenate([row["labels"] for row in pieces]),
        "tokens": {
            expert: np.concatenate(
                [row[f"tokens_{expert}"] for row in pieces]
            ).astype(np.float32, copy=False)
            for expert in EXPERT_ORDER
        },
        "expert_logits": {
            expert: np.concatenate(
                [row[f"logits_{expert}"] for row in pieces]
            ).astype(np.float32, copy=False)
            for expert in EXPERT_ORDER
        },
    }


def iter_offline_target_cache(
    manifest_path: str | Path,
    *,
    expected_pipeline_seed: int,
    expected_specification_sha256: str,
):
    path = Path(manifest_path)
    validate_offline_target_cache(
        path,
        expected_pipeline_seed=expected_pipeline_seed,
        expected_specification_sha256=expected_specification_sha256,
    )
    manifest = load_hashed_json(
        path, expected_contract=TARGET_CACHE_MANIFEST_CONTRACT
    )
    for shard in manifest["shards"]:
        _, arrays = _load_shard_arrays(
            path.parent,
            shard,
            expected_seed=int(expected_pipeline_seed),
            expected_specification_sha256=expected_specification_sha256,
        )
        yield arrays


def verify_target_batch_logits(
    arrays: Mapping[str, np.ndarray],
    *,
    logit_reproducers: Mapping[str, Callable[[np.ndarray], Any]],
    storage_dtype_by_expert: Mapping[str, str],
) -> dict[str, Any]:
    if set(logit_reproducers) != set(EXPERT_ORDER) or set(
        storage_dtype_by_expert
    ) != set(EXPERT_ORDER):
        raise ValueError("target batch verifier expert coverage differs")
    diagnostics = {}
    for expert in EXPERT_ORDER:
        tokens = np.asarray(arrays[f"tokens_{expert}"], dtype=np.float32)
        stored = np.asarray(arrays[f"logits_{expert}"], dtype=np.float32)
        reproduced = _as_numpy(logit_reproducers[expert](tokens)).astype(
            np.float32, copy=False
        )
        if reproduced.shape != stored.shape or not np.isfinite(
            reproduced
        ).all():
            raise ValueError("target batch reproduced-logit shape differs")
        maximum = float(np.max(np.abs(reproduced - stored)))
        tolerance = (
            2.0e-4
            if storage_dtype_by_expert[expert] == "float16"
            else 2.0e-6
        )
        class_exact = bool(
            np.array_equal(
                reproduced.argmax(axis=1), stored.argmax(axis=1)
            )
        )
        diagnostics[expert] = {
            "maximum_absolute_error": maximum,
            "absolute_tolerance": tolerance,
            "predicted_class_identities_exact": class_exact,
            "passed": maximum <= tolerance and class_exact,
        }
    if not all(row["passed"] for row in diagnostics.values()):
        raise RuntimeError(
            "target batch does not reproduce frozen expert logits"
        )
    return diagnostics


def fit_target_normalizers(
    *,
    model_train_manifest_path: str | Path,
    expected_pipeline_seed: int,
    expected_specification_sha256: str,
    source_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    path = Path(model_train_manifest_path)
    manifest = load_hashed_json(
        path, expected_contract=TARGET_CACHE_MANIFEST_CONTRACT
    )
    if manifest["split"] != "model_train":
        raise ValueError("target normalizers fit model_train only")
    validate_offline_target_cache(
        path,
        expected_pipeline_seed=expected_pipeline_seed,
        expected_specification_sha256=expected_specification_sha256,
    )
    count = int(manifest["event_count"])
    sums = {
        expert: np.zeros(manifest["allocation"][expert], dtype=np.float64)
        for expert in EXPERT_ORDER
    }
    sums2 = {expert: np.zeros_like(sums[expert]) for expert in EXPERT_ORDER}
    for shard in manifest["shards"]:
        _, arrays = _load_shard_arrays(
            path.parent,
            shard,
            expected_seed=expected_pipeline_seed,
            expected_specification_sha256=expected_specification_sha256,
        )
        for expert in EXPERT_ORDER:
            sums[expert] += arrays[f"normalization_sum_{expert}"]
            sums2[expert] += arrays[
                f"normalization_sum_square_{expert}"
            ]
    normalizers = {}
    for expert in EXPERT_ORDER:
        mean64 = sums[expert] / count
        variance64 = np.maximum(sums2[expert] / count - mean64**2, 0.0)
        mean = mean64.astype(np.float32)
        standard_deviation = np.sqrt(variance64).astype(np.float32)
        normalizers[expert] = bind_source(
            with_content_hash(
                {
                    "contract": TARGET_NORMALIZER_CONTRACT,
                    "schema_version": 1,
                    "expert_id": expert,
                    "target_mode": manifest["target_tuple"][
                        EXPERT_ORDER.index(expert)
                    ],
                    "pipeline_seed": int(expected_pipeline_seed),
                    "shape_id": manifest["shape_id"],
                    "fit_split": "model_train",
                    "fit_event_count": count,
                    "target_cache_manifest_sha256": manifest["content_hash"],
                    "target_cache_specification_sha256": (
                        expected_specification_sha256
                    ),
                    "mean": mean.tolist(),
                    "standard_deviation": standard_deviation.tolist(),
                    "statistics_dtype": "float32",
                    "accumulator_dtype": "float64",
                    "population_variance_denominator": count,
                    "standard_deviation_floor": 1.0e-4,
                    "normalization_mode": "N_UNCLIPPED",
                }
            ),
            source_snapshot=source_snapshot,
        )
    normalizer_set = bind_source(
        with_content_hash(
            {
                "contract": TARGET_NORMALIZER_SET_CONTRACT,
                "schema_version": 1,
                "pipeline_seed": int(expected_pipeline_seed),
                "shape_id": manifest["shape_id"],
                "target_tuple": list(manifest["target_tuple"]),
                "model_train_target_cache_manifest_sha256": manifest[
                    "content_hash"
                ],
                "normalizer_hashes": {
                    expert: normalizers[expert]["content_hash"]
                    for expert in EXPERT_ORDER
                },
                "validation_or_test_statistics_consumed": False,
            }
        ),
        source_snapshot=source_snapshot,
    )
    return {**normalizers, "normalizer_set": normalizer_set}


def validate_target_normalizer_set(
    artifacts: Mapping[str, Mapping[str, Any]]
) -> str:
    if set(artifacts) != set(EXPERT_ORDER) | {"normalizer_set"}:
        raise ValueError("target normalizer-set members differ")
    normalizer_set = artifacts["normalizer_set"]
    digest = validate_content_hash(
        normalizer_set, expected_contract=TARGET_NORMALIZER_SET_CONTRACT
    )
    hashes = {}
    for expert in EXPERT_ORDER:
        artifact = artifacts[expert]
        hashes[expert] = validate_content_hash(
            artifact, expected_contract=TARGET_NORMALIZER_CONTRACT
        )
        mean = np.asarray(artifact["mean"], dtype=np.float32)
        standard_deviation = np.asarray(
            artifact["standard_deviation"], dtype=np.float32
        )
        if (
            artifact["expert_id"] != expert
            or artifact["fit_split"] != "model_train"
            or artifact["pipeline_seed"] != normalizer_set["pipeline_seed"]
            or artifact["shape_id"] != normalizer_set["shape_id"]
            or mean.ndim != 2
            or standard_deviation.shape != mean.shape
            or not np.isfinite(mean).all()
            or not np.isfinite(standard_deviation).all()
            or bool((standard_deviation < 0).any())
            or artifact["statistics_dtype"] != "float32"
            or artifact["accumulator_dtype"] != "float64"
        ):
            raise ValueError("target normalizer lineage differs")
    if (
        normalizer_set["normalizer_hashes"] != hashes
        or len({repr(row.get("source")) for row in artifacts.values()}) != 1
    ):
        raise ValueError("target normalizer-set hashes/source differ")
    return digest


def publish_target_normalizers(
    *,
    output_dir: str | Path,
    artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    digest = validate_target_normalizer_set(artifacts)
    root = Path(output_dir)
    publications = {
        expert: write_immutable_json(
            root / f"target_normalizer_{expert}.json", artifacts[expert]
        )
        for expert in EXPERT_ORDER
    }
    publications["normalizer_set"] = write_immutable_json(
        root / "target_normalizer_set.json", artifacts["normalizer_set"]
    )
    return {
        "target_normalizer_set_sha256": digest,
        "publications": publications,
    }


def build_sealed_input_preparation(
    *,
    split: str,
    identity_manifest_sha256: str,
    raw_input_manifest_sha256: str,
    degraded_hlt_input_manifest_sha256: str,
    relation_sidecar_manifest_sha256: str,
    region_sidecar_manifest_sha256: str,
) -> dict[str, Any]:
    if split not in {"stack_val", "final_test"}:
        raise ValueError("sealed preparation split differs")
    return with_content_hash(
        {
            "contract": SEALED_INPUT_PREPARATION_CONTRACT,
            "schema_version": 1,
            "split": split,
            "parents": {
                "identity_manifest": require_sha256(
                    identity_manifest_sha256,
                    name="identity_manifest_sha256",
                ),
                "raw_input_manifest": require_sha256(
                    raw_input_manifest_sha256,
                    name="raw_input_manifest_sha256",
                ),
                "degraded_HLT_input_manifest": require_sha256(
                    degraded_hlt_input_manifest_sha256,
                    name="degraded_hlt_input_manifest_sha256",
                ),
                "relation_sidecar_manifest": require_sha256(
                    relation_sidecar_manifest_sha256,
                    name="relation_sidecar_manifest_sha256",
                ),
                "REGION_sidecar_manifest": require_sha256(
                    region_sidecar_manifest_sha256,
                    name="region_sidecar_manifest_sha256",
                ),
            },
            "checkpoint_loading_permitted": False,
            "model_outputs_present": False,
            "offline_targets_present": False,
            "oracle_logits_present": False,
            "labels_joined_to_model_outputs": False,
            "allowed_before_finalist_lock": True,
        }
    )


def validate_sealed_input_preparation(payload: Mapping[str, Any]) -> str:
    digest = validate_content_hash(
        payload, expected_contract=SEALED_INPUT_PREPARATION_CONTRACT
    )
    if (
        payload["split"] not in {"stack_val", "final_test"}
        or payload["checkpoint_loading_permitted"]
        or payload["model_outputs_present"]
        or payload["offline_targets_present"]
        or payload["oracle_logits_present"]
        or payload["labels_joined_to_model_outputs"]
    ):
        raise ValueError("sealed input preparation contains forbidden outputs")
    return digest


__all__ = [
    "AUDIT_SAMPLE_SIZE",
    "DEFAULT_SHARD_SIZE",
    "SEALED_INPUT_PREPARATION_CONTRACT",
    "TARGET_CACHE_MANIFEST_CONTRACT",
    "TARGET_CACHE_POLICY_CONTRACT",
    "TARGET_CACHE_SPEC_CONTRACT",
    "TARGET_NORMALIZER_CONTRACT",
    "TARGET_NORMALIZER_SET_CONTRACT",
    "TARGET_SHARD_CONTRACT",
    "TARGET_STORAGE_AUDIT_CONTRACT",
    "audit_target_storage",
    "build_sealed_input_preparation",
    "build_target_cache_policy",
    "build_target_cache_specification",
    "fit_target_normalizers",
    "identity_order_sha256",
    "iter_offline_target_cache",
    "load_offline_target_cache",
    "load_frozen_token_head_reproducer",
    "publish_offline_target_cache",
    "publish_target_normalizers",
    "validate_offline_target_cache",
    "validate_sealed_input_preparation",
    "validate_target_cache_specification",
    "validate_target_normalizer_set",
    "validate_target_storage_audit",
    "verify_target_batch_logits",
]
