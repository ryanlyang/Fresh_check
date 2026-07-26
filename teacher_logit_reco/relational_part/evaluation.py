"""Deterministic metrics and inference for the relational ParT campaign."""

from __future__ import annotations

import inspect
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .contracts import (
    canonical_sha256,
    require_sha256,
    sha256_file,
    validate_content_hash,
    with_content_hash,
)

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


EVALUATION_CONTRACT = "relational_part_evaluation_v1"
FINAL_EVALUATION_CONTRACT = "relational_part_final_evaluation_v2"
FINAL_PREDICTION_CONTRACT = "relational_part_final_predictions_v1"
PAIRED_STATISTICS_CONTRACT = "relational_part_paired_statistics_v1"
CLASS_NAMES = (
    "QCD",
    "Hbb",
    "Hcc",
    "Hgg",
    "H4q",
    "Hqql",
    "Zqq",
    "Wqq",
    "Tbqq",
    "Tbl",
)


def stable_probabilities(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(CLASS_NAMES):
        raise ValueError("logits must have shape [events,10]")
    if not np.isfinite(values).all():
        raise FloatingPointError("logits contain NaN or infinity")
    shifted = values - values.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    return ranks


def _binary_auc(scores: np.ndarray, positive: np.ndarray) -> float:
    positive = np.asarray(positive, dtype=bool)
    positives = int(positive.sum())
    negatives = int(len(positive) - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("AUC requires positive and negative support")
    ranks = _average_ranks(np.asarray(scores, dtype=np.float64))
    positive_rank_sum = float(ranks[positive].sum(dtype=np.float64))
    return (
        positive_rank_sum - positives * (positives + 1) / 2
    ) / (positives * negatives)


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correct = prediction == labels
    edges = np.linspace(0.0, 1.0, 16, dtype=np.float64)
    bins: list[dict[str, Any]] = []
    ece = np.float64(0.0)
    for index in range(15):
        selected = (confidence >= edges[index]) & (
            (confidence < edges[index + 1])
            if index < 14
            else (confidence <= edges[index + 1])
        )
        count = int(selected.sum())
        if count:
            accuracy = float(correct[selected].mean(dtype=np.float64))
            mean_confidence = float(confidence[selected].mean(dtype=np.float64))
            contribution = (
                count / len(labels) * abs(accuracy - mean_confidence)
            )
        else:
            accuracy = None
            mean_confidence = None
            contribution = 0.0
        ece += np.float64(contribution)
        bins.append(
            {
                "index": index,
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "right_inclusive": index == 14,
                "count": count,
                "accuracy": accuracy,
                "mean_confidence": mean_confidence,
                "contribution": float(contribution),
            }
        )
    return {"value": float(ece), "bins": bins}


def qcd_signal_rejection(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    signal_index: int,
    target_efficiency: float,
) -> dict[str, Any]:
    values = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if not 1 <= int(signal_index) < len(CLASS_NAMES):
        raise ValueError("signal_index must identify a non-QCD class")
    signal = labels == int(signal_index)
    qcd = labels == 0
    signal_support = int(signal.sum())
    qcd_support = int(qcd.sum())
    if not signal_support or not qcd_support:
        raise ValueError("QCD rejection requires signal and QCD support")
    if not 0 < float(target_efficiency) <= 1:
        raise ValueError("target efficiency must lie in (0,1]")
    score = values[:, signal_index] - values[:, 0]
    rank = int(math.ceil(float(target_efficiency) * signal_support))
    threshold = float(np.sort(score[signal])[::-1][rank - 1])
    passed = score >= threshold
    signal_pass = int((passed & signal).sum())
    qcd_pass = int((passed & qcd).sum())
    false_positive_rate = qcd_pass / qcd_support
    return {
        "signal_class": CLASS_NAMES[signal_index],
        "target_signal_efficiency": float(target_efficiency),
        "achieved_signal_efficiency": signal_pass / signal_support,
        "discriminant": "logit_signal_minus_logit_QCD",
        "threshold": threshold,
        "pass_rule": "score_greater_than_or_equal_to_threshold",
        "signal_support": signal_support,
        "signal_pass_count": signal_pass,
        "qcd_support": qcd_support,
        "qcd_false_positive_count": qcd_pass,
        "qcd_false_positive_rate": false_positive_rate,
        "background_rejection": (
            None if qcd_pass == 0 else 1.0 / false_positive_rate
        ),
        "background_rejection_is_infinite": qcd_pass == 0,
    }


def evaluate_logits(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    split: str,
) -> dict[str, Any]:
    values = np.asarray(logits, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int64)
    if values.ndim != 2 or values.shape != (len(truth), len(CLASS_NAMES)):
        raise ValueError("logits and labels have incompatible shapes")
    if len(truth) == 0:
        raise ValueError("evaluation requires at least one event")
    if not np.isfinite(values).all():
        raise FloatingPointError("evaluation logits are nonfinite")
    if bool(((truth < 0) | (truth >= len(CLASS_NAMES))).any()):
        raise ValueError("labels lie outside the canonical class order")
    probabilities = stable_probabilities(values)
    predictions = probabilities.argmax(axis=1)
    log_norm = np.log(np.exp(values - values.max(1, keepdims=True)).sum(1))
    log_norm += values.max(1)
    cross_entropy = float(
        (log_norm - values[np.arange(len(truth)), truth]).mean(dtype=np.float64)
    )
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    np.add.at(confusion, (truth, predictions), 1)
    supports = confusion.sum(axis=1)
    if bool((supports == 0).any()):
        missing = [CLASS_NAMES[index] for index in np.flatnonzero(supports == 0)]
        raise ValueError(f"evaluation lacks canonical class support: {missing}")
    efficiencies = np.diag(confusion) / supports
    one_hot = np.eye(len(CLASS_NAMES), dtype=np.float64)[truth]
    brier = float(
        np.square(probabilities - one_hot).sum(axis=1).mean(dtype=np.float64)
    )
    auc = {
        name: _binary_auc(probabilities[:, index], truth == index)
        for index, name in enumerate(CLASS_NAMES)
    }
    rejection = {
        CLASS_NAMES[index]: {
            str(target): qcd_signal_rejection(
                values,
                truth,
                signal_index=index,
                target_efficiency=target,
            )
            for target in (0.30, 0.50)
        }
        for index in range(1, len(CLASS_NAMES))
    }
    calibration = expected_calibration_error(probabilities, truth)
    return with_content_hash(
        {
            "contract": EVALUATION_CONTRACT,
            "schema_version": 1,
            "split": str(split),
            "event_count": len(truth),
            "class_order": list(CLASS_NAMES),
            "accuracy": float((predictions == truth).mean(dtype=np.float64)),
            "cross_entropy": cross_entropy,
            "macro_per_class_accuracy": float(
                efficiencies.mean(dtype=np.float64)
            ),
            "per_class_efficiency": {
                name: float(efficiencies[index])
                for index, name in enumerate(CLASS_NAMES)
            },
            "one_vs_rest_auc": auc,
            "ece_15_bin_top_label": calibration,
            "brier_score": brier,
            "confusion_matrix": confusion.tolist(),
            "qcd_signal_rejection": rejection,
            "calculation_dtype": "float64",
        }
    )


def _move_batch(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("PyTorch is required for model evaluation")
    moved: dict[str, Any] = {}
    for name, value in batch.items():
        moved[name] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


def model_forward(model: Any, batch: Mapping[str, Any]) -> Any:
    parameters = inspect.signature(model.forward).parameters
    aliases = {
        "lorentz_vectors": ("lorentz_vectors", "vectors"),
        "raw_tokens": ("raw_tokens", "tokens"),
    }
    kwargs: dict[str, Any] = {}
    for name in parameters:
        candidates = aliases.get(name, (name,))
        for candidate in candidates:
            if candidate in batch:
                kwargs[name] = batch[candidate]
                break
    required = [
        name
        for name, parameter in parameters.items()
        if name != "self"
        and parameter.default is inspect.Parameter.empty
        and parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        and name not in kwargs
    ]
    if required:
        raise ValueError(f"batch lacks required model inputs: {required}")
    return model(**kwargs)


def evaluate_model(
    model: Any,
    loader: Iterable[Mapping[str, Any]],
    *,
    split: str,
    device: str | Any = "cpu",
) -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("PyTorch is required for model evaluation")
    resolved_device = torch.device(device)
    was_training = bool(model.training)
    model.eval()
    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for raw in loader:
            batch = _move_batch(raw, resolved_device)
            if "labels" not in batch:
                raise ValueError("evaluation batch lacks labels")
            output = model_forward(model, batch)
            if output.ndim != 2 or int(output.shape[1]) != len(CLASS_NAMES):
                raise ValueError("model output must have shape [events,10]")
            logits.append(output.detach().float().cpu().numpy())
            labels.append(batch["labels"].detach().long().cpu().numpy())
    if was_training:
        model.train()
    if not logits:
        raise ValueError("evaluation loader is empty")
    return evaluate_logits(
        np.concatenate(logits, axis=0),
        np.concatenate(labels, axis=0),
        split=split,
    )


def collect_model_predictions(
    model: Any,
    loader: Iterable[Mapping[str, Any]],
    *,
    device: str | Any = "cpu",
    required_split: str = "final_test",
) -> dict[str, Any]:
    """Collect HLT-only event-aligned outputs without opening other sources."""

    if torch is None:
        raise RuntimeError("PyTorch is required for final evaluation")
    if required_split != "final_test":
        raise ValueError("sealed prediction collection is final_test-only")
    resolved_device = torch.device(device)
    was_training = bool(model.training)
    model.eval()
    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    identities: list[str] = []
    forbidden = ("offline", "teacher", "target_logits", "reco_view")
    with torch.no_grad():
        for raw in loader:
            if any(name in raw for name in forbidden):
                raise ValueError("final evaluation batch exposes a forbidden source")
            if "labels" not in raw or "event_identities" not in raw:
                raise ValueError(
                    "final evaluation requires labels and bound event identities"
                )
            batch = _move_batch(raw, resolved_device)
            output = model_forward(model, batch)
            logits.append(output.detach().float().cpu().numpy())
            labels.append(batch["labels"].detach().long().cpu().numpy())
            identities.extend(str(value) for value in raw["event_identities"])
    if was_training:
        model.train()
    if not logits:
        raise ValueError("final-test loader is empty")
    values = np.concatenate(logits)
    truth = np.concatenate(labels)
    if len(identities) != len(truth) or len(set(identities)) != len(identities):
        raise ValueError("final-test event identities are absent or duplicated")
    return {
        "logits": values,
        "labels": truth,
        "predictions": values.argmax(axis=1).astype(np.int16),
        "event_identities": np.asarray(identities, dtype=np.str_),
        "event_identity_sha256": canonical_sha256(identities),
    }


def write_final_predictions(
    path: str | Path,
    predictions: Mapping[str, Any],
    *,
    run_id: str,
    seed: int,
    checkpoint_sha256: str,
    locked_finalists_sha256: str,
) -> dict[str, Any]:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError("final prediction artifact already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    logits = np.asarray(predictions["logits"], dtype=np.float32)
    labels = np.asarray(predictions["labels"], dtype=np.int16)
    predicted = np.asarray(predictions["predictions"], dtype=np.int16)
    identities = np.asarray(predictions["event_identities"], dtype=np.str_)
    if not (
        logits.shape == (len(labels), len(CLASS_NAMES))
        and predicted.shape == labels.shape
        and identities.shape == labels.shape
    ):
        raise ValueError("prediction artifact arrays have incompatible shapes")
    metadata = with_content_hash(
        {
            "contract": FINAL_PREDICTION_CONTRACT,
            "schema_version": 1,
            "run_id": str(run_id),
            "seed": int(seed),
            "checkpoint_sha256": require_sha256(
                checkpoint_sha256, name="checkpoint_sha256"
            ),
            "locked_finalists_sha256": require_sha256(
                locked_finalists_sha256, name="locked_finalists_sha256"
            ),
            "event_count": len(labels),
            "event_identity_sha256": str(
                predictions["event_identity_sha256"]
            ),
            "logits_dtype": "float32",
            "labels_dtype": "int16",
            "prediction_tie_rule": "lowest_canonical_class_index",
            "hlt_only": True,
        }
    )
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(handle, "wb") as stream:
            np.savez_compressed(
                stream,
                logits=logits,
                labels=labels,
                predictions=predicted,
                event_identities=identities,
                metadata=np.asarray(
                    [
                        __import__("json").dumps(
                            metadata,
                            allow_nan=False,
                            separators=(",", ":"),
                            sort_keys=True,
                        )
                    ],
                    dtype=np.str_,
                ),
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return {
        "path": str(destination.resolve()),
        "sha256": sha256_file(destination),
        "metadata": metadata,
    }


def evaluate_locked_finalist(
    model: Any,
    loader: Iterable[Mapping[str, Any]],
    *,
    run_id: str,
    seed: int,
    checkpoint_registration: Mapping[str, Any],
    locked_finalists: Mapping[str, Any],
    campaign_spec_sha256: str,
    split_manifest_sha256: str,
    hlt_cache_hashes: Mapping[str, str],
    output_dir: str | Path,
    device: str | Any = "cpu",
    expected_event_count: int = 500_000,
) -> dict[str, Any]:
    from .selection import validate_locked_finalists

    lock_sha = validate_locked_finalists(
        locked_finalists,
        campaign_spec_sha256=campaign_spec_sha256,
        split_manifest_sha256=split_manifest_sha256,
        hlt_cache_hashes=hlt_cache_hashes,
    )
    rows = {
        str(row["run_id"]): row for row in locked_finalists["evaluation_rows"]
    }
    if run_id not in rows:
        raise ValueError("run is absent from the immutable finalist lock")
    expected_checkpoint = rows[run_id]["checkpoint_hashes"].get(str(int(seed)))
    registration_sha = validate_content_hash(
        checkpoint_registration,
        expected_contract="relational_part_checkpoint_registration_v1",
    )
    if registration_sha != rows[run_id][
        "checkpoint_registration_hashes"
    ].get(str(int(seed))):
        raise ValueError("checkpoint registration differs from finalist lock")
    observed_checkpoint = require_sha256(
        checkpoint_registration.get("checkpoint_sha256"),
        name="checkpoint_registration.checkpoint_sha256",
    )
    if observed_checkpoint != expected_checkpoint:
        raise ValueError("checkpoint differs from the locked seed artifact")
    if checkpoint_registration.get("run_id") != run_id or int(
        checkpoint_registration.get("seed", -1)
    ) != int(seed):
        raise ValueError("checkpoint registration run/seed mismatch")
    if checkpoint_registration.get("val_select_used_for_checkpoint_selection") is not False:
        raise ValueError("checkpoint selection provenance is invalid")
    if checkpoint_registration.get("hlt_only_inference") is not True:
        raise ValueError("final checkpoint is not declared HLT-only")
    if checkpoint_registration.get("offline_or_teacher_required") is not False:
        raise ValueError("final checkpoint declares a forbidden dependency")
    if checkpoint_registration.get("model_contract_sha256") != rows[run_id][
        "model_contract_sha256"
    ]:
        raise ValueError("final checkpoint model contract differs from lock")
    registration_lineage = checkpoint_registration.get("lineage_hashes")
    if (
        not isinstance(registration_lineage, Mapping)
        or dict(registration_lineage) != dict(rows[run_id]["lineage_hashes"])
    ):
        raise ValueError("final checkpoint lineage differs from finalist lock")
    profile = checkpoint_registration.get("parameter_and_flop_profile")
    if not isinstance(profile, Mapping):
        raise ValueError("locked checkpoint lacks its resource profile")
    required_profile = {
        "trainable_parameters",
        "forward_flops_per_event",
        "latency_ms",
        "peak_incremental_device_memory_bytes",
    }
    if not required_profile.issubset(profile):
        raise ValueError("locked checkpoint resource profile is incomplete")
    prediction_arrays = collect_model_predictions(
        model, loader, device=device, required_split="final_test"
    )
    if len(prediction_arrays["labels"]) != int(expected_event_count):
        raise ValueError(
            "sealed final-test count differs from the locked campaign: "
            f"{len(prediction_arrays['labels'])} != {int(expected_event_count)}"
        )
    metrics = evaluate_logits(
        prediction_arrays["logits"],
        prediction_arrays["labels"],
        split="final_test",
    )
    destination = Path(output_dir)
    prediction_publication = write_final_predictions(
        destination / "predictions.npz",
        prediction_arrays,
        run_id=run_id,
        seed=seed,
        checkpoint_sha256=observed_checkpoint,
        locked_finalists_sha256=lock_sha,
    )
    return with_content_hash(
        {
            "contract": FINAL_EVALUATION_CONTRACT,
            "schema_version": 2,
            "run_id": run_id,
            "seed": int(seed),
            "configuration_role": rows[run_id]["configuration_role"],
            "relational_selection_eligible": rows[run_id][
                "relational_selection_eligible"
            ],
            "locked_finalists_sha256": lock_sha,
            "checkpoint_sha256": observed_checkpoint,
            "checkpoint_registration_sha256": registration_sha,
            "model_contract_sha256": checkpoint_registration[
                "model_contract_sha256"
            ],
            "checkpoint_lineage_hashes": dict(registration_lineage),
            "lineage_authenticated": True,
            "campaign_spec_sha256": require_sha256(
                campaign_spec_sha256, name="campaign_spec_sha256"
            ),
            "split_manifest_sha256": require_sha256(
                split_manifest_sha256, name="split_manifest_sha256"
            ),
            "final_test_hlt_cache_sha256": require_sha256(
                hlt_cache_hashes["final_test"],
                name="hlt_cache_hashes.final_test",
            ),
            "metrics": metrics,
            "event_count": int(expected_event_count),
            "prediction_file": "predictions.npz",
            "prediction_file_sha256": prediction_publication["sha256"],
            "prediction_metadata": prediction_publication["metadata"],
            "parameter_and_flop_profile": checkpoint_registration.get(
                "parameter_and_flop_profile"
            ),
            "hlt_only_inference": True,
            "final_test_used_for_selection": False,
        }
    )


def load_final_predictions(path: str | Path) -> dict[str, Any]:
    artifact = Path(path)
    if not artifact.is_file() or artifact.is_symlink():
        raise FileNotFoundError(f"prediction artifact is absent or unsafe: {artifact}")
    import json

    with np.load(artifact, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata"][0]))
        from .contracts import validate_content_hash

        validate_content_hash(
            metadata, expected_contract=FINAL_PREDICTION_CONTRACT
        )
        result = {
            "logits": np.asarray(payload["logits"], dtype=np.float32),
            "labels": np.asarray(payload["labels"], dtype=np.int16),
            "predictions": np.asarray(payload["predictions"], dtype=np.int16),
            "event_identities": np.asarray(payload["event_identities"], dtype=np.str_),
            "metadata": metadata,
            "file_sha256": sha256_file(artifact),
        }
    identities = [str(value) for value in result["event_identities"]]
    if canonical_sha256(identities) != metadata["event_identity_sha256"]:
        raise ValueError("prediction event-identity hash mismatch")
    return result


def paired_prediction_statistics(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 917_301,
) -> dict[str, Any]:
    return paired_prediction_statistics_many(
        {"candidate": candidate},
        baseline,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )["candidate"]


def paired_prediction_statistics_many(
    candidates: Mapping[str, Mapping[str, Any]],
    baseline: Mapping[str, Any],
    *,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 917_301,
    replicate_chunk_size: int = 16,
) -> dict[str, dict[str, Any]]:
    """Reuse the contract-identical bootstrap draws across many candidates."""

    if not candidates:
        raise ValueError("paired statistics require at least one candidate")
    baseline_metadata = baseline.get("metadata", {})
    baseline_seed = baseline_metadata.get("seed")
    baseline_ids = np.asarray(baseline["event_identities"], dtype=np.str_)
    labels = np.asarray(baseline["labels"], dtype=np.int64)
    baseline_prediction = np.asarray(baseline["predictions"], dtype=np.int64)
    baseline_correct = baseline_prediction == labels
    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap replicate count must be positive")
    if replicate_chunk_size <= 0:
        raise ValueError("bootstrap chunk size must be positive")
    class_indices = [
        np.flatnonzero(labels == index) for index in range(len(CLASS_NAMES))
    ]
    if any(len(indices) == 0 for indices in class_indices):
        raise ValueError("paired bootstrap requires every balanced class")
    class_count = len(class_indices[0])
    if any(len(indices) != class_count for indices in class_indices):
        raise ValueError(
            "paired bootstrap requires the locked exactly balanced final test"
        )
    names = list(candidates)
    candidate_correct_rows = []
    differences = []
    metadata_rows = []
    for name in names:
        candidate = candidates[name]
        candidate_metadata = candidate.get("metadata", {})
        candidate_seed = candidate_metadata.get("seed")
        if candidate_seed is not None and baseline_seed is not None and int(
            candidate_seed
        ) != int(baseline_seed):
            raise ValueError("paired predictions belong to different seeds")
        candidate_ids = np.asarray(
            candidate["event_identities"], dtype=np.str_
        )
        if not np.array_equal(candidate_ids, baseline_ids):
            raise ValueError("paired predictions are not identity-aligned")
        if not np.array_equal(
            np.asarray(candidate["labels"], dtype=np.int64), labels
        ):
            raise ValueError("paired predictions have different true labels")
        candidate_prediction = np.asarray(
            candidate["predictions"], dtype=np.int64
        )
        candidate_correct = candidate_prediction == labels
        candidate_correct_rows.append(candidate_correct)
        differences.append(
            candidate_correct.astype(np.int8)
            - baseline_correct.astype(np.int8)
        )
        metadata_rows.append(candidate_metadata)
    difference_matrix = np.stack(differences)
    generator = np.random.Generator(np.random.PCG64(int(bootstrap_seed)))
    bootstrap_sums = np.zeros(
        (len(names), int(bootstrap_replicates)), dtype=np.int64
    )
    total = len(labels)
    for start in range(0, int(bootstrap_replicates), replicate_chunk_size):
        stop = min(start + replicate_chunk_size, int(bootstrap_replicates))
        chunk = stop - start
        draws = generator.integers(
            0,
            class_count,
            size=(chunk, len(CLASS_NAMES), class_count),
            endpoint=False,
            dtype=np.int64,
        )
        for class_index, indices in enumerate(class_indices):
            class_difference = difference_matrix[:, indices]
            bootstrap_sums[:, start:stop] += class_difference[
                :, draws[:, class_index]
            ].sum(axis=-1, dtype=np.int64)
    bootstrap = bootstrap_sums.astype(np.float64) / total
    baseline_accuracy = float(baseline_correct.mean(dtype=np.float64))
    baseline_error = 1.0 - baseline_accuracy
    output = {}
    identity_sha = canonical_sha256(
        [str(value) for value in baseline_ids]
    )
    for row_index, name in enumerate(names):
        candidate_correct = candidate_correct_rows[row_index]
        difference = difference_matrix[row_index]
        candidate_metadata = metadata_rows[row_index]
        candidate_seed = candidate_metadata.get("seed")
        candidate_accuracy = float(
            candidate_correct.mean(dtype=np.float64)
        )
        interval = np.quantile(
            bootstrap[row_index], [0.025, 0.975], method="linear"
        )
        per_class = {
            class_name: float(
                difference[labels == index].mean(dtype=np.float64)
            )
            for index, class_name in enumerate(CLASS_NAMES)
        }
        output[name] = with_content_hash({
            "contract": PAIRED_STATISTICS_CONTRACT,
            "schema_version": 1,
            "event_count": len(labels),
            "candidate_run_id": candidate_metadata.get("run_id"),
            "baseline_run_id": baseline_metadata.get("run_id"),
            "seed": (
                None if candidate_seed is None else int(candidate_seed)
            ),
            "candidate_checkpoint_sha256": candidate_metadata.get(
                "checkpoint_sha256"
            ),
            "baseline_checkpoint_sha256": baseline_metadata.get(
                "checkpoint_sha256"
            ),
            "event_identity_sha256": identity_sha,
            "candidate_accuracy": candidate_accuracy,
            "baseline_accuracy": baseline_accuracy,
            "paired_absolute_accuracy_difference": (
                candidate_accuracy - baseline_accuracy
            ),
            "paired_bootstrap": {
                "seed": int(bootstrap_seed),
                "replicates": int(bootstrap_replicates),
                "sampling_unit": "aligned_event_identity",
                "class_stratified_balanced": True,
                "execution_chunk_replicates": int(replicate_chunk_size),
                "lower_2p5_percent": float(interval[0]),
                "upper_97p5_percent": float(interval[1]),
                "quantile_method": "Hyndman_Fan_type_7_linear",
            },
            "mcnemar": {
                "candidate_correct_baseline_wrong": int(
                    (candidate_correct & ~baseline_correct).sum()
                ),
                "candidate_wrong_baseline_correct": int(
                    (~candidate_correct & baseline_correct).sum()
                ),
            },
            "relative_error_reduction": (
                None
                if baseline_error == 0.0
                else (candidate_accuracy - baseline_accuracy) / baseline_error
            ),
            "relative_error_reduction_undefined": baseline_error == 0.0,
            "per_class_paired_accuracy_difference": per_class,
        })
    return output


def build_evaluation_contract(*, global_determinism_sha256: str) -> dict[str, Any]:
    return with_content_hash(
        {
            "contract": EVALUATION_CONTRACT,
            "schema_version": 1,
            "global_determinism_sha256": require_sha256(
                global_determinism_sha256, name="global_determinism_sha256"
            ),
            "class_order": list(CLASS_NAMES),
            "metrics": [
                "accuracy",
                "cross_entropy",
                "macro_per_class_accuracy",
                "per_class_efficiency",
                "one_vs_rest_auc",
                "15_bin_top_label_ECE",
                "multiclass_Brier",
                "confusion_matrix",
                "QCD_signal_rejection_at_0.30_and_0.50",
            ],
            "val_stop_selects_checkpoint": True,
            "val_select_selects_checkpoint": False,
            "val_select_evaluations_per_checkpoint": 1,
            "calculation_dtype": "float64",
        }
    )


__all__ = [
    "CLASS_NAMES",
    "EVALUATION_CONTRACT",
    "FINAL_EVALUATION_CONTRACT",
    "FINAL_PREDICTION_CONTRACT",
    "PAIRED_STATISTICS_CONTRACT",
    "build_evaluation_contract",
    "evaluate_logits",
    "evaluate_model",
    "evaluate_locked_finalist",
    "expected_calibration_error",
    "model_forward",
    "qcd_signal_rejection",
    "collect_model_predictions",
    "load_final_predictions",
    "paired_prediction_statistics",
    "paired_prediction_statistics_many",
    "stable_probabilities",
    "write_final_predictions",
]
