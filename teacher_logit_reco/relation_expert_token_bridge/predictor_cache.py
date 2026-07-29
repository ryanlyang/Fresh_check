"""Identity-bound predictor outputs and label-free uncertainty calibration."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any, Mapping, Sequence

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
from .predictor_losses import (
    UNCERTAINTY_CALIBRATION_CONTRACT,
    fit_uncertainty_calibration,
)
from .predictors import UNCERTAINTY_HEADS, uncertainty_width
from .registry import EXPERT_ORDER


PREDICTOR_INFERENCE_MANIFEST_CONTRACT = "retb_predictor_inference_cache_v1"


def _array_hash(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(b"retb_predictor_numpy_array_v1\0")
    digest.update(str(array.dtype).encode())
    digest.update(b"\0")
    digest.update(canonical_json_bytes(list(array.shape)))
    digest.update(b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def predictor_identity_order_sha256(identities: Sequence[str]) -> str:
    canonical = tuple(str(value) for value in identities)
    if not canonical or len(canonical) != len(set(canonical)):
        raise ValueError("predictor cache identities are empty/duplicated")
    digest = hashlib.sha256()
    digest.update(b"retb_predictor_identity_order_v1\0")
    for index, identity in enumerate(canonical):
        digest.update(str(index).encode())
        digest.update(b"\0")
        digest.update(identity.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def publish_predictor_inference_cache(
    *,
    output_dir: str | Path,
    split: str,
    pipeline_seed: int,
    expert_id: str,
    uncertainty_head: str,
    identities: Sequence[str],
    predicted_tokens: np.ndarray,
    normalized_predicted_tokens: np.ndarray,
    log_variance: np.ndarray,
    expert_logits: np.ndarray,
    hybrid_logits: np.ndarray,
    predictor_registration_sha256: str,
    predictor_checkpoint_sha256: str,
    target_cache_manifest_sha256: str,
    target_normalizer_sha256: str,
    identity_manifest_sha256: str,
    source_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        split not in {"val_stop", "val_design"}
        or int(pipeline_seed) not in {101, 202, 303}
        or expert_id not in EXPERT_ORDER
        or uncertainty_head not in UNCERTAINTY_HEADS
    ):
        raise ValueError("predictor inference cache identity differs")
    ids = tuple(str(value) for value in identities)
    identity_order = predictor_identity_order_sha256(ids)
    tokens = np.asarray(predicted_tokens, dtype=np.float32)
    normalized_tokens = np.asarray(
        normalized_predicted_tokens, dtype=np.float32
    )
    variance = np.asarray(log_variance, dtype=np.float32)
    expert = np.asarray(expert_logits, dtype=np.float32)
    hybrid = np.asarray(hybrid_logits, dtype=np.float32)
    if (
        tokens.ndim != 3
        or tokens.shape[0] != len(ids)
        or normalized_tokens.shape != tokens.shape
        or variance.shape
        != tokens.shape[:2]
        + (uncertainty_width(uncertainty_head, tokens.shape[-1]),)
        or expert.shape != (len(ids), 10)
        or hybrid.shape != (len(ids), 10)
        or not all(
            np.isfinite(value).all()
            for value in (
                tokens,
                normalized_tokens,
                variance,
                expert,
                hybrid,
            )
        )
        or bool((variance < -8.0).any())
        or bool((variance > 4.0).any())
    ):
        raise ValueError("predictor inference cache arrays differ")
    arrays = {
        "identities": np.asarray(ids, dtype=np.str_),
        "predicted_tokens": tokens,
        "normalized_predicted_tokens": normalized_tokens,
        "log_variance": variance,
        "expert_logits": expert,
        "hybrid_logits": hybrid,
    }
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    root = Path(output_dir)
    publication = write_immutable_bytes(
        root / "predictor_outputs.npz", stream.getvalue()
    )
    manifest = bind_source(
        with_content_hash(
            {
                "contract": PREDICTOR_INFERENCE_MANIFEST_CONTRACT,
                "schema_version": 1,
                "split": split,
                "pipeline_seed": int(pipeline_seed),
                "expert_id": expert_id,
                "uncertainty_head": uncertainty_head,
                "event_count": len(ids),
                "identity_order_sha256": identity_order,
                "npz_filename": "predictor_outputs.npz",
                "npz_sha256": publication["file_sha256"],
                "array_hashes": {
                    name: _array_hash(value) for name, value in arrays.items()
                },
                "parents": {
                    "predictor_registration": require_sha256(
                        predictor_registration_sha256,
                        name="predictor_registration_sha256",
                    ),
                    "predictor_checkpoint": require_sha256(
                        predictor_checkpoint_sha256,
                        name="predictor_checkpoint_sha256",
                    ),
                    "target_cache_manifest": require_sha256(
                        target_cache_manifest_sha256,
                        name="target_cache_manifest_sha256",
                    ),
                    "target_normalizer": require_sha256(
                        target_normalizer_sha256,
                        name="target_normalizer_sha256",
                    ),
                    "identity_manifest": require_sha256(
                        identity_manifest_sha256,
                        name="identity_manifest_sha256",
                    ),
                },
                "labels_present": False,
                "offline_inputs_present": False,
                "tokens_dtype": "float32",
                "token_coordinates": {
                    "predicted_tokens": (
                        "original_offline_token_coordinates"
                    ),
                    "normalized_predicted_tokens": (
                        "model_train_normalized_predictor_coordinates"
                    ),
                    "log_variance": (
                        "model_train_normalized_predictor_coordinates"
                    ),
                },
                "log_variance_clip": [-8.0, 4.0],
                "complete_coverage": True,
            }
        ),
        source_snapshot=source_snapshot,
    )
    write_immutable_json(root / "predictor_outputs_manifest.json", manifest)
    return manifest


def load_predictor_inference_cache(
    manifest_path: str | Path,
    *,
    expected_pipeline_seed: int,
    expected_registration_sha256: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    path = Path(manifest_path)
    manifest = load_hashed_json(
        path, expected_contract=PREDICTOR_INFERENCE_MANIFEST_CONTRACT
    )
    if (
        int(manifest["pipeline_seed"]) != int(expected_pipeline_seed)
        or manifest["parents"]["predictor_registration"]
        != require_sha256(
            expected_registration_sha256,
            name="expected_registration_sha256",
        )
        or set(manifest["parents"])
        != {
            "predictor_registration",
            "predictor_checkpoint",
            "target_cache_manifest",
            "target_normalizer",
            "identity_manifest",
        }
        or manifest.get("token_coordinates")
        != {
            "predicted_tokens": "original_offline_token_coordinates",
            "normalized_predicted_tokens": (
                "model_train_normalized_predictor_coordinates"
            ),
            "log_variance": "model_train_normalized_predictor_coordinates",
        }
    ):
        raise ValueError("predictor inference cache seed/registration differs")
    npz_path = path.parent / manifest["npz_filename"]
    if (
        not npz_path.is_file()
        or npz_path.is_symlink()
        or hashlib.sha256(npz_path.read_bytes()).hexdigest()
        != manifest["npz_sha256"]
    ):
        raise ValueError("predictor inference cache bytes differ")
    with np.load(npz_path, allow_pickle=False) as payload:
        expected = {
            "identities",
            "predicted_tokens",
            "normalized_predicted_tokens",
            "log_variance",
            "expert_logits",
            "hybrid_logits",
        }
        if set(payload.files) != expected:
            raise ValueError("predictor inference cache fields differ")
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    if (
        any(_array_hash(arrays[name]) != digest for name, digest in manifest[
            "array_hashes"
        ].items())
        or predictor_identity_order_sha256(arrays["identities"].tolist())
        != manifest["identity_order_sha256"]
        or len(arrays["identities"]) != int(manifest["event_count"])
        or arrays["predicted_tokens"].dtype != np.float32
        or arrays["predicted_tokens"].ndim != 3
        or arrays["normalized_predicted_tokens"].dtype != np.float32
        or arrays["normalized_predicted_tokens"].shape
        != arrays["predicted_tokens"].shape
        or arrays["log_variance"].dtype != np.float32
        or arrays["log_variance"].shape
        != arrays["predicted_tokens"].shape[:2]
        + (
            uncertainty_width(
                manifest["uncertainty_head"],
                arrays["predicted_tokens"].shape[-1],
            ),
        )
        or arrays["expert_logits"].shape
        != (int(manifest["event_count"]), 10)
        or arrays["hybrid_logits"].shape
        != (int(manifest["event_count"]), 10)
        or bool((arrays["log_variance"] < -8.0).any())
        or bool((arrays["log_variance"] > 4.0).any())
        or not all(
            np.isfinite(arrays[name]).all()
            for name in (
                "predicted_tokens",
                "normalized_predicted_tokens",
                "log_variance",
                "expert_logits",
                "hybrid_logits",
            )
        )
    ):
        raise ValueError("predictor inference cache content differs")
    return manifest, arrays


def calibrate_predictor_inference_cache(
    *,
    manifest_path: str | Path,
    expected_pipeline_seed: int,
    expected_registration_sha256: str,
    target_tokens: np.ndarray,
    identity_order_sha256: str,
    source_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    manifest, arrays = load_predictor_inference_cache(
        manifest_path,
        expected_pipeline_seed=expected_pipeline_seed,
        expected_registration_sha256=expected_registration_sha256,
    )
    if (
        manifest["split"] != "val_design"
        or manifest["identity_order_sha256"]
        != require_sha256(
            identity_order_sha256, name="identity_order_sha256"
        )
    ):
        raise ValueError("uncertainty calibration population/identity differs")
    calibration = bind_source(
        fit_uncertainty_calibration(
            expert_id=manifest["expert_id"],
            uncertainty_head=manifest["uncertainty_head"],
            predicted_tokens=arrays["normalized_predicted_tokens"],
            target_tokens=np.asarray(target_tokens, dtype=np.float32),
            log_variance=arrays["log_variance"],
            predictor_checkpoint_sha256=manifest["parents"][
                "predictor_checkpoint"
            ],
            predictor_registration_sha256=manifest["parents"][
                "predictor_registration"
            ],
            predictor_inference_manifest_sha256=manifest["content_hash"],
            target_cache_manifest_sha256=manifest["parents"][
                "target_cache_manifest"
            ],
            identity_manifest_sha256=manifest["parents"]["identity_manifest"],
        ),
        source_snapshot=source_snapshot,
    )
    validate_content_hash(
        calibration, expected_contract=UNCERTAINTY_CALIBRATION_CONTRACT
    )
    return calibration


__all__ = [
    "PREDICTOR_INFERENCE_MANIFEST_CONTRACT",
    "calibrate_predictor_inference_cache",
    "load_predictor_inference_cache",
    "predictor_identity_order_sha256",
    "publish_predictor_inference_cache",
]
