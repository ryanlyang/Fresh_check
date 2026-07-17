"""Fixed-row paired-bootstrap sanity gates for accelerated C2F reconstructors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from jetclass_fresh.fusion import load_prediction_block, validate_prediction_alignment
from jetclass_fresh.hlt_cache import hash_arrays, jet_identity_hash
from .runtime_profiles import resolve_execution, validate_runtime_profile


RUNTIME_TAGGER_SANITY_CONTRACT = "constrained_c2f_runtime_tagger_sanity_v1"
_PATH_TO_VARIANT = {"C5-B3": "D5", "C6": "D6"}


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _cross_entropy(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    shifted = values - np.max(values, axis=1, keepdims=True)
    log_normalizer = np.log(np.exp(shifted).sum(axis=1))
    return -shifted[np.arange(targets.size), targets] + log_normalizer


def _array_hash(arrays: Mapping[str, np.ndarray]) -> str:
    """Hash an explicit paired-evaluation payload with stable array ordering."""

    return hash_arrays({name: np.ascontiguousarray(value) for name, value in sorted(arrays.items())})


def _reconstructor_checkpoint_metadata(
    path: Path,
    *,
    expected_variant: str,
    expected_runtime_profile: str,
    expected_runtime_profile_hash: str | None,
    require_thirty_epoch_reference: bool,
) -> dict[str, Any]:
    """Load only the provenance-bearing checkpoint payload for the Step 9 contract."""

    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older PyTorch compatibility
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError(f"reconstructor checkpoint {path} is not a mapping payload")
    runtime_profile = payload.get("runtime_profile")
    config = payload.get("config")
    problems: list[str] = []
    if payload.get("checkpoint_role") != "best_model_val":
        problems.append("checkpoint_role is not best_model_val")
    if payload.get("family") != "C" or payload.get("variant") != expected_variant:
        problems.append(f"expected C/{expected_variant} checkpoint")
    if not isinstance(runtime_profile, Mapping) or runtime_profile.get("name") != expected_runtime_profile:
        problems.append(f"expected runtime profile {expected_runtime_profile}")
    if expected_runtime_profile_hash is not None and payload.get("runtime_profile_hash") != expected_runtime_profile_hash:
        problems.append("runtime profile hash does not match the selected candidate execution")
    if require_thirty_epoch_reference:
        if not isinstance(config, Mapping) or int(config.get("epochs", 0)) != 30 or not bool(config.get("fixed_horizon")):
            problems.append("FP32 reference is not a fixed-horizon 30-epoch checkpoint")
    return {
        "path": str(path),
        "sha256": _file_hash(path),
        "ok": not problems,
        "problems": problems,
        "provenance": payload.get("provenance"),
        "code_environment": payload.get("code_environment"),
        "runtime_profile": runtime_profile,
        "configuration": {
            "epochs": None if not isinstance(config, Mapping) else config.get("epochs"),
            "fixed_horizon": None if not isinstance(config, Mapping) else config.get("fixed_horizon"),
        },
    }


@dataclass(frozen=True)
class PairedBootstrapConfig:
    seed: int = 48271
    replicates: int = 10_000
    confidence: float = 0.95
    chunk_size: int = 128
    max_accuracy_loss: float = 0.005
    max_ce_increase: float = 0.010

    def __post_init__(self) -> None:
        if self.replicates <= 0 or self.chunk_size <= 0:
            raise ValueError("replicates and chunk_size must be positive")
        if not 0.5 < self.confidence < 1.0:
            raise ValueError("confidence must be in (0.5, 1)")


def paired_stratified_bootstrap(
    *,
    candidate_logits: np.ndarray,
    reference_logits: np.ndarray,
    labels: np.ndarray,
    config: PairedBootstrapConfig,
) -> dict[str, Any]:
    """Return deterministic paired one-sided upper bounds without materializing all resamples."""

    candidate_logits = np.asarray(candidate_logits, dtype=np.float64)
    reference_logits = np.asarray(reference_logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if candidate_logits.shape != reference_logits.shape or candidate_logits.ndim != 2:
        raise ValueError("candidate/reference logits must be equally shaped rank-2 arrays")
    if labels.shape != (candidate_logits.shape[0],):
        raise ValueError("labels do not align with logits")
    if not np.isfinite(candidate_logits).all() or not np.isfinite(reference_logits).all():
        raise ValueError("paired bootstrap requires finite logits")
    candidate_accuracy_rows = (candidate_logits.argmax(axis=1) == labels).astype(np.float64)
    reference_accuracy_rows = (reference_logits.argmax(axis=1) == labels).astype(np.float64)
    accuracy_delta_rows = candidate_accuracy_rows - reference_accuracy_rows
    ce_delta_rows = _cross_entropy(candidate_logits, labels) - _cross_entropy(reference_logits, labels)
    classes = np.unique(labels)
    if classes.size < 2:
        raise ValueError("paired label-stratified bootstrap requires at least two represented classes")
    class_indices = [np.flatnonzero(labels == label) for label in classes]
    if any(indices.size == 0 for indices in class_indices):
        raise AssertionError("unique labels produced an empty bootstrap stratum")
    rng = np.random.RandomState(int(config.seed))
    accuracy_samples = np.empty(int(config.replicates), dtype=np.float64)
    ce_samples = np.empty(int(config.replicates), dtype=np.float64)
    for start in range(0, int(config.replicates), int(config.chunk_size)):
        stop = min(start + int(config.chunk_size), int(config.replicates))
        count = stop - start
        accuracy = np.zeros(count, dtype=np.float64)
        ce = np.zeros(count, dtype=np.float64)
        for indices in class_indices:
            sampled = rng.randint(0, indices.size, size=(count, indices.size))
            rows = indices[sampled]
            # Each class has equal aggregate weight, matching label-stratified
            # resampling while retaining paired candidate/reference rows.
            accuracy += accuracy_delta_rows[rows].mean(axis=1) / len(class_indices)
            ce += ce_delta_rows[rows].mean(axis=1) / len(class_indices)
        accuracy_samples[start:stop] = accuracy
        ce_samples[start:stop] = ce
    quantile = float(config.confidence)
    observed_accuracy_delta = float(accuracy_delta_rows.mean())
    observed_ce_delta = float(ce_delta_rows.mean())
    upper_accuracy_loss = float(np.quantile(-accuracy_samples, quantile))
    upper_ce_increase = float(np.quantile(ce_samples, quantile))
    return {
        "bootstrap": asdict(config),
        "n_jets": int(labels.size),
        "class_counts": {str(int(label)): int(indices.size) for label, indices in zip(classes, class_indices)},
        "observed_delta_accuracy": observed_accuracy_delta,
        "observed_delta_ce": observed_ce_delta,
        "upper_bound_negative_delta_accuracy": upper_accuracy_loss,
        "upper_bound_delta_ce": upper_ce_increase,
        "pass_accuracy": upper_accuracy_loss <= float(config.max_accuracy_loss),
        "pass_ce": upper_ce_increase <= float(config.max_ce_increase),
        "ok": upper_accuracy_loss <= float(config.max_accuracy_loss) and upper_ce_increase <= float(config.max_ce_increase),
    }


def _normalized_tagger_config(config: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(config)
    payload.pop("output_dir", None)
    payload.pop("reconstructor_sources", None)
    return payload


def _source_checkpoint_hashes(source_metadata: Mapping[str, Any]) -> set[str]:
    reconstructors = source_metadata.get("reconstructors")
    sources = reconstructors.get("sources") if isinstance(reconstructors, Mapping) else None
    if not isinstance(sources, Mapping):
        return set()
    return {
        str(row.get("checkpoint_sha256"))
        for row in sources.values()
        if isinstance(row, Mapping) and isinstance(row.get("checkpoint_sha256"), str)
    }


def _validate_tagger_pair(
    *,
    path: str,
    candidate_tagger_dir: Path,
    reference_tagger_dir: Path,
    candidate_reconstructor_hash: str,
    reference_reconstructor_hash: str,
) -> dict[str, Any]:
    expected_variant = _PATH_TO_VARIANT.get(path)
    if expected_variant is None:
        raise ValueError(f"unsupported sanity path {path!r}")
    candidate_report = _read_json(candidate_tagger_dir / "run_report.json")
    reference_report = _read_json(reference_tagger_dir / "run_report.json")
    candidate_source = _read_json(candidate_tagger_dir / "source_metadata.json")
    reference_source = _read_json(reference_tagger_dir / "source_metadata.json")
    candidate_config = _read_json(candidate_tagger_dir / "config.json")
    reference_config = _read_json(reference_tagger_dir / "config.json")
    problems: list[str] = []
    for name, report in (("accelerated", candidate_report), ("fp32_reference", reference_report)):
        if not bool(report.get("ok")):
            problems.append(f"{name} tagger report has ok=false")
        if report.get("variant") != expected_variant:
            problems.append(f"{name} tagger variant mismatch")
        if report.get("selection_split") != "model_val" or bool(report.get("final_test_evaluated")):
            problems.append(f"{name} tagger violates model-val-only selection")
        if not bool(report.get("deployable_hlt_only")):
            problems.append(f"{name} tagger is not deployable HLT-only")
    if _normalized_tagger_config(candidate_config) != _normalized_tagger_config(reference_config):
        problems.append("tagger config differs beyond reconstructor source paths")
    for key in (
        "provenance",
        "trusted_hlt_warm_start",
        "selection_split",
        "input_view",
        "runtime_compatibility",
        "source_state",
    ):
        if candidate_source.get(key) != reference_source.get(key):
            problems.append(f"tagger source metadata differs for {key}")
    if candidate_report.get("phase_history") != reference_report.get("phase_history"):
        problems.append("tagger phase histories differ")
    if _source_checkpoint_hashes(candidate_source) != {candidate_reconstructor_hash}:
        problems.append("accelerated tagger did not load exactly the declared accelerated reconstructor")
    if _source_checkpoint_hashes(reference_source) != {reference_reconstructor_hash}:
        problems.append("FP32 tagger did not load exactly the declared FP32 reconstructor")
    return {
        "ok": not problems,
        "problems": problems,
        "path": path,
        "variant": expected_variant,
        "accelerated_tagger": {
            "dir": str(candidate_tagger_dir),
            "checkpoint_sha256": candidate_report.get("checkpoint_sha256"),
            "configuration_hash": candidate_report.get("configuration_hash"),
        },
        "fp32_tagger": {
            "dir": str(reference_tagger_dir),
            "checkpoint_sha256": reference_report.get("checkpoint_sha256"),
            "configuration_hash": reference_report.get("configuration_hash"),
        },
    }


def _paired_arrays(
    *,
    candidate_logits: np.ndarray,
    reference_logits: np.ndarray,
    labels: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "labels": np.ascontiguousarray(labels, dtype=np.int64),
        "accelerated_logits": np.ascontiguousarray(candidate_logits, dtype=np.float32),
        "fp32_logits": np.ascontiguousarray(reference_logits, dtype=np.float32),
        "accelerated_ce": np.ascontiguousarray(_cross_entropy(candidate_logits, labels), dtype=np.float64),
        "fp32_ce": np.ascontiguousarray(_cross_entropy(reference_logits, labels), dtype=np.float64),
    }


def _bootstrap_config(payload: Mapping[str, Any]) -> PairedBootstrapConfig:
    required = {
        "seed",
        "replicates",
        "confidence",
        "chunk_size",
        "max_accuracy_loss",
        "max_ce_increase",
    }
    if set(payload) != required:
        raise ValueError("tagger sanity bootstrap configuration is malformed")
    return PairedBootstrapConfig(
        seed=int(payload["seed"]),
        replicates=int(payload["replicates"]),
        confidence=float(payload["confidence"]),
        chunk_size=int(payload["chunk_size"]),
        max_accuracy_loss=float(payload["max_accuracy_loss"]),
        max_ce_increase=float(payload["max_ce_increase"]),
    )


def validate_tagger_sanity_report(report_path: str | Path) -> dict[str, Any]:
    """Recompute a persisted sanity gate from the live model-val predictions."""

    evidence_path = Path(report_path)
    report = _read_json(evidence_path)
    if report.get("contract") != RUNTIME_TAGGER_SANITY_CONTRACT:
        raise ValueError("tagger sanity report has the wrong contract")
    if report.get("promotion_evidence_kind") != "tagger_sanity" or report.get("selection_split") != "model_val":
        raise ValueError("tagger sanity report has the wrong evaluation contract")
    if report.get("final_test_loaded") is not False:
        raise ValueError("tagger sanity report must not load final-test data")
    path = str(report.get("path", ""))
    if path not in _PATH_TO_VARIANT:
        raise ValueError(f"unsupported tagger sanity path {path!r}")
    sources = report.get("source_artifacts")
    provenance = report.get("prediction_provenance")
    if not isinstance(sources, Mapping) or not isinstance(provenance, Mapping):
        raise ValueError("tagger sanity report lacks source artifact provenance")
    candidate_checkpoint = Path(str(sources.get("candidate_reconstructor_checkpoint", "")))
    reference_checkpoint = Path(str(sources.get("fp32_reconstructor_checkpoint", "")))
    if not candidate_checkpoint.is_file() or not reference_checkpoint.is_file():
        raise ValueError("tagger sanity report references a missing reconstructor checkpoint")
    candidate_checkpoint_hash = _file_hash(candidate_checkpoint)
    reference_checkpoint_hash = _file_hash(reference_checkpoint)
    if candidate_checkpoint_hash != provenance.get("accelerated_reconstructor_checkpoint_sha256"):
        raise ValueError("tagger sanity accelerated checkpoint hash mismatch")
    if reference_checkpoint_hash != provenance.get("fp32_reconstructor_checkpoint_sha256"):
        raise ValueError("tagger sanity FP32 checkpoint hash mismatch")
    candidate_block = load_prediction_block(
        Path(str(sources.get("candidate_prediction_dir", ""))),
        str(sources.get("candidate_model_name", "")),
        "model_val",
    )
    reference_block = load_prediction_block(
        Path(str(sources.get("fp32_prediction_dir", ""))),
        str(sources.get("fp32_model_name", "")),
        "model_val",
    )
    validate_prediction_alignment((candidate_block, reference_block))
    expected_arrays = _paired_arrays(
        candidate_logits=candidate_block.logits,
        reference_logits=reference_block.logits,
        labels=candidate_block.labels,
    )
    if _array_hash(expected_arrays) != provenance.get("paired_arrays_content_hash"):
        raise ValueError("tagger sanity paired-array provenance does not match live predictions")
    paired_path = Path(str(report.get("paired_arrays_path", "")))
    if not paired_path.is_file() or _file_hash(paired_path) != report.get("paired_arrays_file_sha256"):
        raise ValueError("tagger sanity paired-array file hash mismatch")
    with np.load(paired_path, allow_pickle=False) as paired:
        if set(paired.files) != set(expected_arrays):
            raise ValueError("tagger sanity paired-array membership mismatch")
        for name, expected in expected_arrays.items():
            observed = np.ascontiguousarray(paired[name])
            if observed.dtype != expected.dtype or observed.shape != expected.shape or not np.array_equal(observed, expected):
                raise ValueError(f"tagger sanity paired array {name} does not match live predictions")
    bootstrap_payload = report.get("paired_bootstrap")
    if not isinstance(bootstrap_payload, Mapping):
        raise ValueError("tagger sanity report lacks paired bootstrap")
    config_payload = bootstrap_payload.get("bootstrap")
    if not isinstance(config_payload, Mapping):
        raise ValueError("tagger sanity bootstrap lacks its configuration")
    bootstrap_report = paired_stratified_bootstrap(
        candidate_logits=expected_arrays["accelerated_logits"],
        reference_logits=expected_arrays["fp32_logits"],
        labels=expected_arrays["labels"],
        config=_bootstrap_config(config_payload),
    )
    if bootstrap_report != bootstrap_payload:
        raise ValueError("tagger sanity paired bootstrap does not match recomputation")
    pair = _validate_tagger_pair(
        path=path,
        candidate_tagger_dir=Path(str(sources.get("candidate_tagger_dir", ""))),
        reference_tagger_dir=Path(str(sources.get("fp32_tagger_dir", ""))),
        candidate_reconstructor_hash=candidate_checkpoint_hash,
        reference_reconstructor_hash=reference_checkpoint_hash,
    )
    if pair != report.get("pair_validation"):
        raise ValueError("tagger sanity tagger-pair validation mismatch")
    gate = {"ok": bool(pair["ok"] and bootstrap_report["ok"] and not report.get("problems"))}
    if gate != report.get("promotion_gate") or bool(report.get("ok")) != gate["ok"]:
        raise ValueError("tagger sanity promotion gate does not match recomputation")
    return {
        "report": dict(report),
        "pair_validation": pair,
        "paired_bootstrap": bootstrap_report,
        "paired_arrays": expected_arrays,
    }


def write_tagger_sanity_report(
    *,
    path: str,
    candidate_tagger_dir: str | Path,
    reference_tagger_dir: str | Path,
    candidate_prediction_dir: str | Path,
    reference_prediction_dir: str | Path,
    candidate_model_name: str,
    reference_model_name: str,
    candidate_reconstructor_checkpoint: str | Path,
    reference_reconstructor_checkpoint: str | Path,
    output_path: str | Path,
    bootstrap: PairedBootstrapConfig = PairedBootstrapConfig(),
    candidate_profile_path: str | Path,
) -> dict[str, Any]:
    """Validate a matched D5/D6 pair and write its fixed-row bootstrap gate."""

    candidate_tagger_dir = Path(candidate_tagger_dir)
    reference_tagger_dir = Path(reference_tagger_dir)
    candidate_checkpoint = Path(candidate_reconstructor_checkpoint)
    reference_checkpoint = Path(reference_reconstructor_checkpoint)
    for required in (candidate_checkpoint, reference_checkpoint):
        if not required.is_file():
            raise FileNotFoundError(required)
    candidate_checkpoint_hash = _file_hash(candidate_checkpoint)
    reference_checkpoint_hash = _file_hash(reference_checkpoint)
    if candidate_checkpoint_hash == reference_checkpoint_hash:
        raise ValueError("accelerated and FP32 reference reconstructors must be distinct checkpoints")
    profile_path = Path(candidate_profile_path)
    validated_profile = validate_runtime_profile(profile_path, expected_status="accelerated_candidate_v1")
    profile = validated_profile["profile"]
    profile_hash = validated_profile["file_sha256"]
    expected_candidate_runtime_hash = str(resolve_execution(profile, path)["runtime_profile_hash"])
    candidate_reconstructor = _reconstructor_checkpoint_metadata(
        candidate_checkpoint,
        expected_variant=path,
        expected_runtime_profile="accelerated_candidate_v1",
        expected_runtime_profile_hash=expected_candidate_runtime_hash,
        require_thirty_epoch_reference=False,
    )
    reference_reconstructor = _reconstructor_checkpoint_metadata(
        reference_checkpoint,
        expected_variant=path,
        expected_runtime_profile="fp32_reference",
        expected_runtime_profile_hash=None,
        require_thirty_epoch_reference=True,
    )
    candidate_block = load_prediction_block(candidate_prediction_dir, candidate_model_name, "model_val")
    reference_block = load_prediction_block(reference_prediction_dir, reference_model_name, "model_val")
    validate_prediction_alignment((candidate_block, reference_block))
    pair = _validate_tagger_pair(
        path=path,
        candidate_tagger_dir=candidate_tagger_dir,
        reference_tagger_dir=reference_tagger_dir,
        candidate_reconstructor_hash=candidate_checkpoint_hash,
        reference_reconstructor_hash=reference_checkpoint_hash,
    )
    bootstrap_report = paired_stratified_bootstrap(
        candidate_logits=candidate_block.logits,
        reference_logits=reference_block.logits,
        labels=candidate_block.labels,
        config=bootstrap,
    )
    candidate_metadata = candidate_block.metadata
    reference_metadata = reference_block.metadata
    provenance_problems: list[str] = []
    for label, metadata in (("accelerated", candidate_reconstructor), ("fp32_reference", reference_reconstructor)):
        if not metadata["ok"]:
            provenance_problems.extend(f"{label} reconstructor: {problem}" for problem in metadata["problems"])
    if candidate_reconstructor["provenance"] != reference_reconstructor["provenance"]:
        provenance_problems.append("candidate/reference reconstructor provenance differs")
    candidate_environment = candidate_reconstructor["code_environment"]
    if candidate_environment != reference_reconstructor["code_environment"]:
        provenance_problems.append("candidate/reference reconstructor code environment differs")
    if candidate_environment != profile.get("code_environment"):
        provenance_problems.append("candidate reconstructor code environment differs from candidate profile")
    for key in ("source_manifest_hash", "hlt_content_hash", "jet_identity_hash"):
        if candidate_metadata.get(key) != reference_metadata.get(key):
            provenance_problems.append(f"prediction metadata mismatch for {key}")
    expected_identity = jet_identity_hash(candidate_block.jet_ids)
    if candidate_metadata.get("jet_identity_hash") != expected_identity:
        provenance_problems.append("candidate prediction identity hash does not match rows")
    if reference_metadata.get("jet_identity_hash") != expected_identity:
        provenance_problems.append("reference prediction identity hash does not match rows")
    paired_arrays = _paired_arrays(
        candidate_logits=candidate_block.logits,
        reference_logits=reference_block.logits,
        labels=candidate_block.labels,
    )
    report = {
        "contract": RUNTIME_TAGGER_SANITY_CONTRACT,
        "ok": bool(pair["ok"] and bootstrap_report["ok"] and not provenance_problems),
        "promotion_evidence_kind": "tagger_sanity",
        "promotion_gate": {"ok": bool(pair["ok"] and bootstrap_report["ok"] and not provenance_problems)},
        "path": path,
        "selection_split": "model_val",
        "final_test_loaded": False,
        "pair_validation": pair,
        "paired_bootstrap": bootstrap_report,
        "prediction_provenance": {
            "jet_identity_hash": expected_identity,
            "labels_hash": hashlib.sha256(np.ascontiguousarray(candidate_block.labels).tobytes()).hexdigest(),
            "accelerated_prediction_content_hash": candidate_metadata.get("prediction_content_hash"),
            "fp32_prediction_content_hash": reference_metadata.get("prediction_content_hash"),
            "accelerated_reconstructor_checkpoint_sha256": candidate_checkpoint_hash,
            "fp32_reconstructor_checkpoint_sha256": reference_checkpoint_hash,
            "paired_arrays_content_hash": _array_hash(paired_arrays),
            "accelerated_reconstructor": candidate_reconstructor,
            "fp32_reconstructor": reference_reconstructor,
        },
        "candidate_profile_path": str(profile_path),
        "candidate_profile_file_sha256": profile_hash,
        "candidate_profile_hash": profile.get("candidate_profile_hash"),
        "code_environment_hash": profile.get("code_environment_hash"),
        "candidate_execution_runtime_profile_hash": expected_candidate_runtime_hash,
        "source_artifacts": {
            "candidate_tagger_dir": str(candidate_tagger_dir.resolve()),
            "fp32_tagger_dir": str(reference_tagger_dir.resolve()),
            "candidate_prediction_dir": str(Path(candidate_prediction_dir).resolve()),
            "fp32_prediction_dir": str(Path(reference_prediction_dir).resolve()),
            "candidate_model_name": candidate_model_name,
            "fp32_model_name": reference_model_name,
            "candidate_reconstructor_checkpoint": str(candidate_checkpoint.resolve()),
            "fp32_reconstructor_checkpoint": str(reference_checkpoint.resolve()),
        },
        "problems": provenance_problems,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    paired_arrays_path = output.with_name(f"{output.stem}_paired_arrays.npz")
    np.savez_compressed(paired_arrays_path, **paired_arrays)
    report["paired_arrays_path"] = str(paired_arrays_path)
    report["paired_arrays_file_sha256"] = _file_hash(paired_arrays_path)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return validate_tagger_sanity_report(output)["report"]


__all__ = [
    "PairedBootstrapConfig",
    "RUNTIME_TAGGER_SANITY_CONTRACT",
    "paired_stratified_bootstrap",
    "validate_tagger_sanity_report",
    "write_tagger_sanity_report",
]
