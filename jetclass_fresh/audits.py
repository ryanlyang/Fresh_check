"""Step 12 leakage and sanity audits for the fresh JetClass replication."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
import inspect
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from . import dual_view, part_inputs
from .fusion import (
    DEFAULT_C_GRID,
    STACK_SPLITS,
    PredictionBlock,
    classification_metrics_from_probs,
    fit_logistic_stacker,
    load_blocks_for_split,
    load_prediction_block,
    prediction_paths,
    softmax_np,
    stack_feature_matrix,
)
from .hlt_baseline import save_json
from .hlt_cache import jet_identity_hash, load_hlt_metadata
from .jetclass_data import SPLIT_ORDER, SplitManifest, audit_split_manifest, load_split_manifest, manifest_hash


ALLOWED_PREDICTION_ARRAY_KEYS = {
    "logits",
    "probs",
    "labels",
    "jet_file_indices",
    "jet_entries",
}

ALLOWED_MODEL_INPUTS = {
    "cached_fixed_hlt_only",
    "cached_fixed_hlt_and_reconstruction_from_cached_fixed_hlt",
}

FORBIDDEN_FUSION_SOURCE_PATTERNS = [
    "offline_teacher",
    "teacher_logits",
    "teacher_probs",
    "teacher_probability",
    "teacher_target",
    "target_score",
    "oracle",
    "reconstruction_loss",
    "offline_logits",
    "offline_probs",
]


@dataclass
class AuditRunConfig:
    """Configuration for the Step 12 audit suite."""

    manifest_path: str
    prediction_dir: str
    output_dir: str
    hlt_cache_dir: str | None = None
    fusion_report_path: str | None = None
    model_names: list[str] = field(default_factory=list)
    splits: list[str] = field(default_factory=lambda: list(STACK_SPLITS))
    require_file_disjoint: bool = True
    verify_hlt_cache_arrays: bool = False
    seed: int = 1701
    C_grid: list[float] = field(default_factory=lambda: list(DEFAULT_C_GRID))
    feature_mode: str = "logits_probs"
    max_iter: int = 500
    permutation_accuracy_slack: float = 0.05
    holdout_max_accuracy_gap: float = 0.10
    block_shuffle_model: str | None = None


def _load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _report_model_names(fusion_report_path: str | Path | None) -> list[str]:
    if fusion_report_path is None:
        return []
    report = _load_json(fusion_report_path)
    return [str(name) for name in report.get("model_names", [])]


def _split_files(manifest: SplitManifest) -> dict[str, set[str]]:
    return {
        split: {identity.file for identity in manifest.splits.get(split, [])}
        for split in SPLIT_ORDER
    }


def audit_file_split(manifest: SplitManifest, *, require_disjoint: bool = True) -> Dict[str, Any]:
    """Measure ROOT-file overlap across partitions."""

    files_by_split = _split_files(manifest)
    pair_reports: Dict[str, Any] = {}
    overlap_count = 0
    for index, split_a in enumerate(SPLIT_ORDER):
        for split_b in SPLIT_ORDER[index + 1 :]:
            overlap = sorted(files_by_split[split_a] & files_by_split[split_b])
            overlap_count += len(overlap)
            pair_reports[f"{split_a}__{split_b}"] = {
                "count": int(len(overlap)),
                "examples": overlap[:10],
            }

    ok = (overlap_count == 0) if require_disjoint else True
    return {
        "ok": bool(ok),
        "require_disjoint": bool(require_disjoint),
        "total_pairwise_file_overlap_count": int(overlap_count),
        "file_counts_by_split": {split: int(len(files)) for split, files in files_by_split.items()},
        "pair_overlaps": pair_reports,
        "note": (
            "File-level disjointness is a hard pass criterion when require_disjoint=True. "
            "Jet-level split manifests may still pass the separate jet_identity audit."
        ),
    }


def audit_jet_identity_splits(manifest: SplitManifest) -> Dict[str, Any]:
    """Verify no stable `(file, entry)` identity appears in multiple partitions."""

    base = audit_split_manifest(manifest)
    base.update(
        {
            "manifest_hash": manifest_hash(manifest),
            "jet_identity_hash_by_split": {
                split: jet_identity_hash(manifest.splits.get(split, []))
                for split in SPLIT_ORDER
            },
        }
    )
    base["ok"] = bool(
        base.get("expected_counts_ok")
        and int(base.get("duplicate_within_split_count", 0)) == 0
        and int(base.get("cross_split_overlap_count", 0)) == 0
    )
    return base


def audit_fusion_report_roles(fusion_report: Mapping[str, Any]) -> Dict[str, Any]:
    """Check that the saved fusion report exposes the intended stack partition roles."""

    problems: list[str] = []
    splits = list(fusion_report.get("splits", []))
    if splits != list(STACK_SPLITS):
        problems.append(f"fusion report splits are {splits}, expected {STACK_SPLITS}")
    if not fusion_report.get("final_test_evaluated"):
        problems.append("fusion report does not mark final_test_evaluated=True")

    stack_section = fusion_report.get("stacked_logistic_regression", {})
    metrics = stack_section.get("metrics", {}) if isinstance(stack_section, Mapping) else {}
    missing_metric_splits = [split for split in STACK_SPLITS if split not in metrics]
    if missing_metric_splits:
        problems.append(f"stacked logistic metrics are missing splits: {missing_metric_splits}")

    return {
        "ok": not problems,
        "problems": problems,
        "reported_splits": splits,
        "expected_stack_fit_split": "stack_train",
        "expected_stack_selection_split": "stack_val",
        "expected_locked_report_split": "final_test",
    }


def _load_prediction_metadata(prediction_dir: str | Path, model_name: str, split: str) -> Dict[str, Any]:
    _, metadata_path = prediction_paths(prediction_dir, model_name, split)
    return _load_json(metadata_path)


def audit_hlt_sharing(
    prediction_dir: str | Path,
    model_names: Sequence[str],
    *,
    hlt_cache_dir: str | Path | None = None,
    splits: Sequence[str] = STACK_SPLITS,
) -> Dict[str, Any]:
    """Verify all frozen stack inputs for each split came from one shared HLT cache."""

    split_reports: Dict[str, Any] = {}
    ok = True
    for split in splits:
        split_ok = True
        problems: list[str] = []
        expected_hash = None
        expected_seed = None
        if hlt_cache_dir is not None:
            try:
                hlt_metadata = load_hlt_metadata(hlt_cache_dir, split)
                expected_hash = hlt_metadata.get("hlt_content_hash")
                expected_seed = hlt_metadata.get("seed")
            except Exception as exc:  # pragma: no cover - exercised by CLI failures
                split_ok = False
                problems.append(f"could not load HLT metadata: {exc}")

        hlt_hashes: Dict[str, Any] = {}
        allowed_inputs: Dict[str, Any] = {}
        identity_hashes: Dict[str, Any] = {}
        for model_name in model_names:
            try:
                metadata = _load_prediction_metadata(prediction_dir, model_name, split)
            except Exception as exc:
                split_ok = False
                problems.append(f"{model_name}: missing prediction metadata: {exc}")
                continue
            hlt_hashes[model_name] = metadata.get("hlt_content_hash")
            identity_hashes[model_name] = metadata.get("jet_identity_hash")
            allowed_inputs[model_name] = metadata.get("allowed_inputs")
            if metadata.get("hlt_content_hash") is None:
                split_ok = False
                problems.append(f"{model_name}: missing hlt_content_hash")
            if expected_hash is not None and metadata.get("hlt_content_hash") != expected_hash:
                split_ok = False
                problems.append(f"{model_name}: prediction HLT hash does not match cache metadata")
            if metadata.get("allowed_inputs") not in ALLOWED_MODEL_INPUTS:
                split_ok = False
                problems.append(f"{model_name}: unexpected allowed_inputs={metadata.get('allowed_inputs')!r}")

        unique_hashes = {value for value in hlt_hashes.values() if value is not None}
        if len(unique_hashes) != 1 and model_names:
            split_ok = False
            problems.append(f"models do not share one HLT hash: {sorted(unique_hashes)}")

        try:
            load_blocks_for_split(prediction_dir, model_names, split)
        except Exception as exc:
            split_ok = False
            problems.append(f"prediction blocks are not row-aligned: {exc}")

        split_reports[split] = {
            "ok": bool(split_ok),
            "problems": problems,
            "expected_hlt_content_hash": expected_hash,
            "expected_hlt_seed": expected_seed,
            "model_hlt_content_hashes": hlt_hashes,
            "model_jet_identity_hashes": identity_hashes,
            "model_allowed_inputs": allowed_inputs,
        }
        ok = ok and split_ok

    return {"ok": bool(ok), "split_reports": split_reports}


def _npz_keys(path: str | Path) -> list[str]:
    with np.load(path, allow_pickle=False) as data:
        return sorted(str(key) for key in data.files)


def _metadata_forbidden_hits(metadata: Mapping[str, Any]) -> list[str]:
    hits: list[str] = []
    for key in metadata:
        key_lower = str(key).lower()
        for pattern in FORBIDDEN_FUSION_SOURCE_PATTERNS:
            if pattern in key_lower:
                hits.append(str(key))
                break
    return hits


def audit_fusion_source(
    prediction_dir: str | Path,
    model_names: Sequence[str],
    *,
    splits: Sequence[str] = STACK_SPLITS,
    feature_mode: str = "logits_probs",
) -> Dict[str, Any]:
    """Verify stack inputs are only frozen logits/probabilities plus labels/identities."""

    split_reports: Dict[str, Any] = {}
    ok = True
    for split in splits:
        split_ok = True
        problems: list[str] = []
        model_reports: Dict[str, Any] = {}
        blocks: list[PredictionBlock] = []
        for model_name in model_names:
            npz_path, _ = prediction_paths(prediction_dir, model_name, split)
            try:
                keys = set(_npz_keys(npz_path))
                extra = sorted(keys - ALLOWED_PREDICTION_ARRAY_KEYS)
                missing = sorted(ALLOWED_PREDICTION_ARRAY_KEYS - keys)
                metadata = _load_prediction_metadata(prediction_dir, model_name, split)
                forbidden_metadata = _metadata_forbidden_hits(metadata)
                block = load_prediction_block(prediction_dir, model_name, split)
                blocks.append(block)
            except Exception as exc:
                split_ok = False
                problems.append(f"{model_name}: could not inspect prediction block: {exc}")
                continue
            if extra:
                split_ok = False
                problems.append(f"{model_name}: forbidden array keys in prediction npz: {extra}")
            if missing:
                split_ok = False
                problems.append(f"{model_name}: missing required prediction arrays: {missing}")
            if forbidden_metadata:
                split_ok = False
                problems.append(f"{model_name}: forbidden metadata keys: {forbidden_metadata}")
            if not np.allclose(block.probs, softmax_np(block.logits), atol=1.0e-5):
                split_ok = False
                problems.append(f"{model_name}: probs are not softmax(logits)")
            model_reports[model_name] = {
                "array_keys": sorted(keys),
                "forbidden_metadata_keys": forbidden_metadata,
                "num_classes": int(block.logits.shape[1]),
                "n_jets": int(len(block.labels)),
            }

        if blocks:
            try:
                features = stack_feature_matrix(blocks, feature_mode=feature_mode)
                classes_per_model = int(blocks[0].logits.shape[1])
                multiplier = 2 if feature_mode == "logits_probs" else 1
                expected_width = int(len(model_names) * classes_per_model * multiplier)
                if features.shape[1] != expected_width:
                    split_ok = False
                    problems.append(f"stack feature width {features.shape[1]} != expected {expected_width}")
            except Exception as exc:
                split_ok = False
                problems.append(f"could not build allowed stack feature matrix: {exc}")

        split_reports[split] = {"ok": bool(split_ok), "problems": problems, "models": model_reports}
        ok = ok and split_ok

    return {
        "ok": bool(ok),
        "allowed_prediction_array_keys": sorted(ALLOWED_PREDICTION_ARRAY_KEYS),
        "feature_mode": feature_mode,
        "split_reports": split_reports,
    }


def audit_offline_leakage_interfaces() -> Dict[str, Any]:
    """Static interface check for HLT-side Particle Transformer input builders."""

    checks = {
        "part_inputs.compute_view_jet_features": inspect.signature(part_inputs.compute_view_jet_features),
        "part_inputs.build_particle_transformer_inputs_from_tokens": inspect.signature(
            part_inputs.build_particle_transformer_inputs_from_tokens
        ),
        "dual_view.build_part_inputs_torch": inspect.signature(dual_view.build_part_inputs_torch),
    }
    forbidden_names = {
        "offline_tokens",
        "offline_mask",
        "offline_jet_pt",
        "offline_jet_eta",
        "offline_jet_phi",
        "offline_jet_energy",
        "offline_mass",
        "teacher_logits",
        "teacher_probs",
    }
    problems: list[str] = []
    signature_report: Dict[str, Any] = {}
    for name, signature in checks.items():
        parameters = list(signature.parameters)
        forbidden = sorted(set(parameters) & forbidden_names)
        if forbidden:
            problems.append(f"{name} exposes forbidden parameters: {forbidden}")
        signature_report[name] = {"parameters": parameters, "forbidden_parameters": forbidden}

    return {
        "ok": not problems,
        "problems": problems,
        "signature_report": signature_report,
        "note": (
            "This static audit checks the HLT-side input builder interfaces. "
            "Stage A reconstruction training may use offline targets; HLT-side tagger/fusion inference may not."
        ),
    }


def _copy_block(
    block: PredictionBlock,
    *,
    logits: np.ndarray | None = None,
    probs: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    jet_ids: Sequence[Any] | None = None,
    split: str | None = None,
    metadata_update: Mapping[str, Any] | None = None,
) -> PredictionBlock:
    metadata = dict(block.metadata)
    if metadata_update:
        metadata.update(dict(metadata_update))
    return PredictionBlock(
        model_name=block.model_name,
        split=block.split if split is None else split,
        logits=block.logits.copy() if logits is None else np.asarray(logits, dtype=np.float32),
        probs=block.probs.copy() if probs is None else np.asarray(probs, dtype=np.float32),
        labels=block.labels.copy() if labels is None else np.asarray(labels, dtype=np.int64),
        jet_ids=list(block.jet_ids if jet_ids is None else jet_ids),
        metadata=metadata,
    )


def _subset_blocks(blocks: Sequence[PredictionBlock], indices: np.ndarray, *, split: str) -> list[PredictionBlock]:
    selected = np.asarray(indices, dtype=np.int64)
    return [
        _copy_block(
            block,
            logits=block.logits[selected],
            probs=block.probs[selected],
            labels=block.labels[selected],
            jet_ids=[block.jet_ids[int(index)] for index in selected],
            split=split,
        )
        for block in blocks
    ]


def _evaluate_stacker_on_blocks(stacker, blocks: Sequence[PredictionBlock], *, feature_mode: str) -> Dict[str, float]:
    features = stack_feature_matrix(blocks, feature_mode=feature_mode)
    return classification_metrics_from_probs(stacker.predict_probs(features), blocks[0].labels)


def _load_stack_blocks(prediction_dir: str | Path, model_names: Sequence[str]) -> dict[str, list[PredictionBlock]]:
    return {
        split: load_blocks_for_split(prediction_dir, model_names, split)
        for split in STACK_SPLITS
    }


def reference_stacked_metrics(
    prediction_dir: str | Path,
    model_names: Sequence[str],
    *,
    C_grid: Sequence[float] = DEFAULT_C_GRID,
    feature_mode: str = "logits_probs",
    max_iter: int = 500,
) -> Dict[str, Any]:
    """Fit the normal stacker from saved predictions and report metrics."""

    blocks_by_split = _load_stack_blocks(prediction_dir, model_names)
    stacker, selection = fit_logistic_stacker(
        blocks_by_split["stack_train"],
        blocks_by_split["stack_val"],
        C_grid=C_grid,
        feature_mode=feature_mode,
        max_iter=max_iter,
    )
    metrics = {
        split: _evaluate_stacker_on_blocks(stacker, blocks, feature_mode=feature_mode)
        for split, blocks in blocks_by_split.items()
    }
    return {"selection": selection, "metrics": metrics}


def permutation_label_audit(
    prediction_dir: str | Path,
    model_names: Sequence[str],
    *,
    seed: int = 1701,
    C_grid: Sequence[float] = DEFAULT_C_GRID,
    feature_mode: str = "logits_probs",
    max_iter: int = 500,
    accuracy_slack: float = 0.05,
) -> Dict[str, Any]:
    """Shuffle stack_train labels before fitting the stacker."""

    blocks_by_split = _load_stack_blocks(prediction_dir, model_names)
    rng = np.random.RandomState(int(seed))
    permuted_labels = blocks_by_split["stack_train"][0].labels.copy()
    rng.shuffle(permuted_labels)
    shuffled_train = [
        _copy_block(block, labels=permuted_labels, metadata_update={"audit_label_permutation_seed": int(seed)})
        for block in blocks_by_split["stack_train"]
    ]
    stacker, selection = fit_logistic_stacker(
        shuffled_train,
        blocks_by_split["stack_val"],
        C_grid=C_grid,
        feature_mode=feature_mode,
        max_iter=max_iter,
    )
    metrics = {
        split: _evaluate_stacker_on_blocks(stacker, blocks, feature_mode=feature_mode)
        for split, blocks in blocks_by_split.items()
    }
    num_classes = int(blocks_by_split["stack_train"][0].logits.shape[1])
    chance_accuracy = 1.0 / float(num_classes)
    threshold = chance_accuracy + float(accuracy_slack)
    checked_splits = ["stack_val", "final_test"]
    max_checked_accuracy = max(metrics[split]["accuracy"] for split in checked_splits)
    return {
        "ok": bool(max_checked_accuracy <= threshold),
        "seed": int(seed),
        "chance_accuracy": chance_accuracy,
        "accuracy_threshold": threshold,
        "max_checked_accuracy": float(max_checked_accuracy),
        "selection": selection,
        "metrics": metrics,
        "note": "stack_train labels are permuted; stack_val/final_test accuracy should collapse near chance.",
    }


def holdout_stack_audit(
    prediction_dir: str | Path,
    model_names: Sequence[str],
    *,
    seed: int = 1701,
    C_grid: Sequence[float] = DEFAULT_C_GRID,
    feature_mode: str = "logits_probs",
    max_iter: int = 500,
    max_accuracy_gap: float = 0.10,
) -> Dict[str, Any]:
    """Fit the stacker on half of stack_train and evaluate the other half."""

    blocks_by_split = _load_stack_blocks(prediction_dir, model_names)
    n_rows = int(len(blocks_by_split["stack_train"][0].labels))
    rng = np.random.RandomState(int(seed))
    order = rng.permutation(n_rows)
    split_at = n_rows // 2
    fit_indices = order[:split_at]
    holdout_indices = order[split_at:]
    fit_blocks = _subset_blocks(blocks_by_split["stack_train"], fit_indices, split="stack_train_fit_half")
    holdout_blocks = _subset_blocks(blocks_by_split["stack_train"], holdout_indices, split="stack_train_holdout_half")
    stacker, selection = fit_logistic_stacker(
        fit_blocks,
        blocks_by_split["stack_val"],
        C_grid=C_grid,
        feature_mode=feature_mode,
        max_iter=max_iter,
    )
    metrics = {
        "stack_train_fit_half": _evaluate_stacker_on_blocks(stacker, fit_blocks, feature_mode=feature_mode),
        "stack_train_holdout_half": _evaluate_stacker_on_blocks(stacker, holdout_blocks, feature_mode=feature_mode),
        "stack_val": _evaluate_stacker_on_blocks(stacker, blocks_by_split["stack_val"], feature_mode=feature_mode),
        "final_test": _evaluate_stacker_on_blocks(stacker, blocks_by_split["final_test"], feature_mode=feature_mode),
    }
    reference_accuracy = metrics["stack_train_holdout_half"]["accuracy"]
    gaps = {
        split: abs(float(metrics[split]["accuracy"]) - float(reference_accuracy))
        for split in ["stack_val", "final_test"]
    }
    return {
        "ok": bool(max(gaps.values()) <= float(max_accuracy_gap)),
        "seed": int(seed),
        "fit_rows": int(len(fit_indices)),
        "holdout_rows": int(len(holdout_indices)),
        "max_accuracy_gap": float(max(gaps.values())),
        "allowed_max_accuracy_gap": float(max_accuracy_gap),
        "accuracy_gaps_vs_holdout": gaps,
        "selection": selection,
        "metrics": metrics,
        "note": "Regularization is selected on stack_val; the other half of stack_train is never used for fitting.",
    }


def block_shuffle_audit(
    prediction_dir: str | Path,
    model_names: Sequence[str],
    *,
    model_name: str | None = None,
    seed: int = 1701,
    C_grid: Sequence[float] = DEFAULT_C_GRID,
    feature_mode: str = "logits_probs",
    max_iter: int = 500,
) -> Dict[str, Any]:
    """Permute one model's rows relative to labels and the other models."""

    if not model_names:
        raise ValueError("block shuffle audit requires at least one model")
    shuffled_model = str(model_name or model_names[-1])
    if shuffled_model not in model_names:
        raise ValueError(f"Unknown block_shuffle_model {shuffled_model!r}; expected one of {list(model_names)}")

    blocks_by_split = _load_stack_blocks(prediction_dir, model_names)
    rng = np.random.RandomState(int(seed))
    shuffled_by_split: dict[str, list[PredictionBlock]] = {}
    for split, blocks in blocks_by_split.items():
        split_blocks: list[PredictionBlock] = []
        permutation = rng.permutation(len(blocks[0].labels))
        for block in blocks:
            if block.model_name != shuffled_model:
                split_blocks.append(block)
                continue
            split_blocks.append(
                _copy_block(
                    block,
                    logits=block.logits[permutation],
                    probs=block.probs[permutation],
                    labels=block.labels,
                    jet_ids=block.jet_ids,
                    metadata_update={
                        "audit_block_shuffle_seed": int(seed),
                        "audit_block_shuffle_model": shuffled_model,
                    },
                )
            )
        shuffled_by_split[split] = split_blocks

    reference = reference_stacked_metrics(
        prediction_dir,
        model_names,
        C_grid=C_grid,
        feature_mode=feature_mode,
        max_iter=max_iter,
    )
    stacker, selection = fit_logistic_stacker(
        shuffled_by_split["stack_train"],
        shuffled_by_split["stack_val"],
        C_grid=C_grid,
        feature_mode=feature_mode,
        max_iter=max_iter,
    )
    shuffled_metrics = {
        split: _evaluate_stacker_on_blocks(stacker, blocks, feature_mode=feature_mode)
        for split, blocks in shuffled_by_split.items()
    }
    deltas = {
        split: float(shuffled_metrics[split]["accuracy"] - reference["metrics"][split]["accuracy"])
        for split in STACK_SPLITS
    }
    return {
        "ok": bool(deltas["stack_val"] <= 0.0 and deltas["final_test"] <= 0.0),
        "seed": int(seed),
        "shuffled_model": shuffled_model,
        "reference_stacked_logistic_metrics": reference["metrics"],
        "shuffled_stacked_logistic_selection": selection,
        "shuffled_stacked_logistic_metrics": shuffled_metrics,
        "accuracy_delta_shuffled_minus_reference": deltas,
        "note": "One model block is row-permuted within every stack split while labels/other models stay fixed.",
    }


def run_audit_suite(config: AuditRunConfig) -> Dict[str, Any]:
    """Run the full Step 12 audit suite and save `audit_report.json`."""

    manifest = load_split_manifest(config.manifest_path)
    fusion_report = _load_json(config.fusion_report_path) if config.fusion_report_path else {}
    model_names = list(config.model_names) or _report_model_names(config.fusion_report_path)
    if not model_names:
        raise ValueError("No model names provided; pass --model-names or a fusion report with model_names")

    audits: Dict[str, Any] = {
        "file_split": audit_file_split(manifest, require_disjoint=config.require_file_disjoint),
        "jet_identity": audit_jet_identity_splits(manifest),
        "offline_leakage_interfaces": audit_offline_leakage_interfaces(),
        "fusion_source": audit_fusion_source(
            config.prediction_dir,
            model_names,
            splits=config.splits,
            feature_mode=config.feature_mode,
        ),
        "hlt_sharing": audit_hlt_sharing(
            config.prediction_dir,
            model_names,
            hlt_cache_dir=config.hlt_cache_dir,
            splits=config.splits,
        ),
        "permutation_label": permutation_label_audit(
            config.prediction_dir,
            model_names,
            seed=config.seed,
            C_grid=config.C_grid,
            feature_mode=config.feature_mode,
            max_iter=config.max_iter,
            accuracy_slack=config.permutation_accuracy_slack,
        ),
        "holdout_stack": holdout_stack_audit(
            config.prediction_dir,
            model_names,
            seed=config.seed,
            C_grid=config.C_grid,
            feature_mode=config.feature_mode,
            max_iter=config.max_iter,
            max_accuracy_gap=config.holdout_max_accuracy_gap,
        ),
        "block_shuffle": block_shuffle_audit(
            config.prediction_dir,
            model_names,
            model_name=config.block_shuffle_model,
            seed=config.seed,
            C_grid=config.C_grid,
            feature_mode=config.feature_mode,
            max_iter=config.max_iter,
        ),
    }
    if fusion_report:
        audits["partition_roles"] = audit_fusion_report_roles(fusion_report)

    if config.verify_hlt_cache_arrays:
        from .hlt_cache import audit_hlt_cache

        audits["hlt_cache_arrays"] = audit_hlt_cache(manifest, config.hlt_cache_dir, splits=SPLIT_ORDER)

    overall_ok = all(bool(report.get("ok")) for report in audits.values())
    output = {
        "experiment_step": "step12_leakage_audits",
        "ok": bool(overall_ok),
        "config": asdict(config),
        "model_names": model_names,
        "manifest_hash": manifest_hash(manifest),
        "audits": audits,
        "interpretation_lock": "Do not interpret final physics result until these audit reports are reviewed.",
    }
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "audit_report.json", output)
    return output
