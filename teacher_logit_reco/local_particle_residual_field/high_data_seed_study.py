"""Locked 3M-jet matched-seed A0/P7b scaling-study contracts."""

from __future__ import annotations

from dataclasses import asdict
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Mapping

from jetclass_fresh.jetclass_data import SPLIT_ORDER, load_split_manifest, manifest_hash

from .curriculum import (
    LOCAL_RESIDUAL_FIELD_SELECTED_CONSUMER_CONTRACT,
    load_selected_consumer_record,
)
from .fusion_campaign import stable_fusion_json_hash
from .fusion_seed_control import extract_a0_train_config
from .tagger import RESIDUAL_FIELD_SOURCE_HLT_ONLY
from .tagger_train import LocalResidualFieldTaggerTrainConfig


HIGH_DATA_STUDY_MANIFEST_CONTRACT = "local_residual_field_high_data_seed_study_manifest_v1"
HIGH_DATA_RUN_COMPLETION_CONTRACT = "local_residual_field_high_data_seed_study_run_v1"
HIGH_DATA_VALIDATION_REPORT_CONTRACT = "local_residual_field_high_data_validation_report_v1"
HIGH_DATA_FINAL_TEST_REPORT_CONTRACT = "local_residual_field_high_data_final_test_report_v1"

HIGH_DATA_MATCHED_SEEDS = (20421, 20522, 20623)
HIGH_DATA_REUSED_A0_SEEDS = (20421,)
HIGH_DATA_TRAINED_A0_SEEDS = (20522, 20623)
HIGH_DATA_SPLIT_COUNTS = {
    "model_train": 3_000_000,
    "model_val": 250_000,
    "stack_train": 0,
    "stack_val": 500_000,
    "final_test": 1_000_000,
}
HIGH_DATA_HLT_SPLITS = ("model_train", "model_val", "stack_val", "final_test")
HIGH_DATA_PRIVILEGED_SPLITS = ("model_train", "model_val", "stack_val")
T_CRITICAL_95_DF2 = 4.302652729696142

_A0_FORBIDDEN_INPUTS = (
    "baseline_checkpoint",
    "reconstructor_checkpoint",
    "teacher_logits_dir",
    "teacher_logits_train_path",
    "teacher_logits_val_path",
    "teacher_logits_stack_val_path",
)
_P7B_ALLOWED_CONFIG_DIFFERENCES = (
    "hlt_cache_dir",
    "manifest_path",
    "oracle_run_report_path",
    "oracle_teacher_checkpoint",
    "oracle_teacher_config_path",
    "oracle_teacher_logits_dir",
    "output_dir",
    "predictor_warm_start_checkpoint",
    "seed",
    "selected_consumer_json",
    "student_warm_start_checkpoint",
    "target_cache_dir",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_object(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(payload)


def save_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")


def _require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"missing {label}: {path}")


def _source_binding(path: Path) -> dict[str, str]:
    _require_file(path, "source artifact")
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def build_frozen_high_data_selection(
    source_path: str | Path,
    *,
    output_path: str | Path,
) -> dict[str, Any]:
    """Freeze the successful low-data P7b consumer choice without high-data reselection."""

    source_path = Path(source_path).resolve()
    source = load_selected_consumer_record(source_path)
    if source.selected_consumer_id != "Orobust_light":
        raise ValueError("the frozen P7b scaling recipe requires Orobust_light")
    if not math.isclose(float(source.selected_alpha_endpoint or -1.0), 1.0, abs_tol=1.0e-8):
        raise ValueError("the frozen P7b scaling recipe requires selected alpha endpoint 1.0")
    payload: dict[str, Any] = {
        "contract": LOCAL_RESIDUAL_FIELD_SELECTED_CONSUMER_CONTRACT,
        "selected_consumer_id": "Orobust_light",
        "selected_alpha_endpoint": 1.0,
        "source_path": str(source_path),
        "source_hash": sha256_file(source_path),
        "selection_source": "frozen_low_data_p7b_scaling_recipe",
        "selection_reason": (
            "The high-data scaling study freezes the already selected P7b recipe; "
            "it does not select a consumer using high-data validation results."
        ),
        "model_val_alpha_curve": dict(source.model_val_alpha_curve),
        "stack_val_alpha_curve": dict(source.stack_val_alpha_curve),
        "payload": {
            "high_data_reselection_performed": False,
            "recipe_name": "P7b",
            "source_selection_path": str(source_path),
            "source_selection_sha256": sha256_file(source_path),
        },
    }
    # Exercise the production loader before writing the immutable campaign manifest.
    save_json(output_path, payload)
    loaded = load_selected_consumer_record(output_path)
    if loaded.selected_consumer_id != "Orobust_light":
        raise AssertionError("written high-data selection did not round-trip")
    return payload


def _validate_split_inventory(manifest_path: Path) -> tuple[str, dict[str, int]]:
    split_manifest = load_split_manifest(manifest_path)
    observed = {
        split: len(split_manifest.splits.get(split, ()))
        for split in SPLIT_ORDER
    }
    if observed != HIGH_DATA_SPLIT_COUNTS:
        raise ValueError(
            f"high-data split inventory mismatch: expected {HIGH_DATA_SPLIT_COUNTS}, "
            f"observed {observed}"
        )
    return manifest_hash(split_manifest), observed


def _require_cache_files(
    root: Path,
    *,
    splits: Iterable[str],
    data_suffix: str,
    metadata_suffix: str,
    label: str,
    expected_manifest_hash: str,
) -> None:
    _require_dir(root, label)
    for split in splits:
        _require_file(root / f"{split}{data_suffix}", f"{label}/{split} data")
        metadata_path = root / f"{split}{metadata_suffix}"
        _require_file(metadata_path, f"{label}/{split} metadata")
        metadata = load_json_object(metadata_path)
        if metadata.get("source_manifest_hash") != expected_manifest_hash:
            raise ValueError(f"{label}/{split} is bound to a different split manifest")
        if metadata.get("split") != split:
            raise ValueError(f"{label}/{split} metadata declares split={metadata.get('split')!r}")
        if int(metadata.get("n_jets", -1)) != HIGH_DATA_SPLIT_COUNTS[split]:
            raise ValueError(
                f"{label}/{split} contains {metadata.get('n_jets')} jets, "
                f"expected {HIGH_DATA_SPLIT_COUNTS[split]}"
            )


def build_high_data_study_manifest(
    *,
    campaign_id: str,
    campaign_root: str | Path,
    reference_curriculum_root: str | Path,
) -> dict[str, Any]:
    """Audit completed shared 3M sources and freeze the three-seed comparison."""

    root = Path(campaign_root).resolve()
    reference = Path(reference_curriculum_root).resolve()
    manifest_path = root / "inputs" / "split_manifest" / "split_manifest.json.gz"
    hlt_cache = root / "inputs" / "hlt_cache"
    offline_cache = root / "inputs" / "offline_cache"
    target_cache = root / "targets"
    c0_dir = root / "reconstructors" / "C0"
    a0_dir = root / "taggers" / "A0"
    consumer_dir = root / "taggers" / "Orobust_light"
    selected_path = root / "selected_consumer.json"
    reference_selected_path = reference / "selected_consumer.json"
    reference_p7b_dir = reference / "curriculum" / "P7b"

    _require_file(manifest_path, "split manifest")
    manifest_sha, observed_counts = _validate_split_inventory(manifest_path)
    _require_cache_files(
        hlt_cache,
        splits=HIGH_DATA_HLT_SPLITS,
        data_suffix="_fixed_hlt.npz",
        metadata_suffix="_fixed_hlt_metadata.json",
        label="HLT cache",
        expected_manifest_hash=manifest_sha,
    )
    for split in HIGH_DATA_HLT_SPLITS:
        hlt_metadata = load_json_object(hlt_cache / f"{split}_fixed_hlt_metadata.json")
        if hlt_metadata.get("hlt_profile") != "fixed_hlt_v2_realistic":
            raise ValueError(f"HLT cache/{split} does not use fixed_hlt_v2_realistic")
        if not math.isclose(
            float(hlt_metadata.get("hlt_degradation_strength", -1.0)),
            2.5,
            rel_tol=0.0,
            abs_tol=1.0e-8,
        ):
            raise ValueError(f"HLT cache/{split} does not use degradation strength 2.5")
    _require_cache_files(
        offline_cache,
        splits=HIGH_DATA_PRIVILEGED_SPLITS,
        data_suffix="_offline.npz",
        metadata_suffix="_offline_metadata.json",
        label="offline cache",
        expected_manifest_hash=manifest_sha,
    )
    _require_cache_files(
        target_cache,
        splits=HIGH_DATA_PRIVILEGED_SPLITS,
        data_suffix="_local_particle_residual_fields.npz",
        metadata_suffix="_local_particle_residual_fields_metadata.json",
        label="residual-field target cache",
        expected_manifest_hash=manifest_sha,
    )
    for forbidden in (
        offline_cache / "final_test_offline.npz",
        offline_cache / "final_test_offline_metadata.json",
        target_cache / "final_test_local_particle_residual_fields.npz",
        target_cache / "final_test_local_particle_residual_fields_metadata.json",
    ):
        if forbidden.exists():
            raise ValueError(f"privileged final-test artifact is forbidden: {forbidden}")

    required_files = {
        "c0_checkpoint": c0_dir / "best_model_val.pt",
        "c0_run_report": c0_dir / "run_report.json",
        "a0_checkpoint": a0_dir / "best_model_val.pt",
        "a0_source_metadata": a0_dir / "source_metadata.json",
        "a0_run_report": a0_dir / "run_report.json",
        "consumer_checkpoint": consumer_dir / "best_model_val.pt",
        "consumer_teacher_config": consumer_dir / "teacher_config.json",
        "consumer_run_report": consumer_dir / "run_report.json",
        "reference_selected_consumer": reference_selected_path,
        "reference_p7b_source_metadata": reference_p7b_dir / "source_metadata.json",
        "reference_p7b_run_report": reference_p7b_dir / "run_report.json",
    }
    for label, path in required_files.items():
        _require_file(path, label)

    a0_source = load_json_object(required_files["a0_source_metadata"])
    a0_config = extract_a0_train_config(a0_source)
    if int(a0_config.get("seed", -1)) != HIGH_DATA_MATCHED_SEEDS[0]:
        raise ValueError("shared high-data A0 must use the first locked matched seed")
    if a0_config.get("field_source") != RESIDUAL_FIELD_SOURCE_HLT_ONLY:
        raise ValueError("shared high-data A0 must be HLT-only")
    if Path(str(a0_config.get("manifest_path"))).resolve() != manifest_path:
        raise ValueError("shared high-data A0 is not bound to the active split manifest")

    if selected_path.exists():
        existing = load_selected_consumer_record(selected_path)
        if existing.selection_source != "frozen_low_data_p7b_scaling_recipe":
            raise ValueError("existing selected_consumer.json is not the frozen high-data recipe")
    else:
        build_frozen_high_data_selection(reference_selected_path, output_path=selected_path)

    reference_source = load_json_object(required_files["reference_p7b_source_metadata"])
    reference_config = reference_source.get("config")
    if not isinstance(reference_config, Mapping):
        raise ValueError("reference P7b source metadata is missing config")
    if bool(reference_config.get("evaluate_final_test")):
        raise ValueError("reference P7b unexpectedly evaluated final_test during training")

    run_dirs = {
        str(seed): {
            "A0": str(
                (
                    a0_dir
                    if seed == HIGH_DATA_MATCHED_SEEDS[0]
                    else root / "runs" / f"seed_{seed}" / "A0"
                ).resolve()
            ),
            "P7b": str((root / "runs" / f"seed_{seed}" / "P7b").resolve()),
        }
        for seed in HIGH_DATA_MATCHED_SEEDS
    }
    payload: dict[str, Any] = {
        "contract": HIGH_DATA_STUDY_MANIFEST_CONTRACT,
        "ok": True,
        "campaign_id": str(campaign_id),
        "campaign_root": str(root),
        "data_profile": "3m_train_250k_model_val_500k_stack_val_1m_final_test",
        "split_counts": observed_counts,
        "split_manifest_hash": manifest_sha,
        "matched_seeds": list(HIGH_DATA_MATCHED_SEEDS),
        "reused_a0_seeds": list(HIGH_DATA_REUSED_A0_SEEDS),
        "trained_a0_seeds": list(HIGH_DATA_TRAINED_A0_SEEDS),
        "selection_split": "model_val",
        "primary_comparison_split": "stack_val",
        "final_test_policy": "sealed_until_explicit_confirmation",
        "final_test_runtime_inputs": "HLT_only",
        "privileged_final_test_artifacts_present": False,
        "selected_consumer_id": "Orobust_light",
        "selected_alpha_endpoint": 1.0,
        "oracle_execution_mode": "frozen_checkpoint_online",
        "oracle_shared_across_student_seeds": True,
        "predictor_shared_across_student_seeds": True,
        "reference_curriculum_root": str(reference),
        "reference_p7b_dir": str(reference_p7b_dir),
        "p7b_allowed_config_differences": list(_P7B_ALLOWED_CONFIG_DIFFERENCES),
        "paths": {
            "split_manifest": str(manifest_path),
            "hlt_cache_dir": str(hlt_cache),
            "offline_cache_dir": str(offline_cache),
            "target_cache_dir": str(target_cache),
            "selected_consumer_json": str(selected_path.resolve()),
            "c0_checkpoint": str(required_files["c0_checkpoint"].resolve()),
            "consumer_checkpoint": str(required_files["consumer_checkpoint"].resolve()),
            "consumer_teacher_config": str(required_files["consumer_teacher_config"].resolve()),
            "consumer_run_report": str(required_files["consumer_run_report"].resolve()),
            "a0_source_metadata": str(required_files["a0_source_metadata"].resolve()),
            "reference_p7b_source_metadata": str(
                required_files["reference_p7b_source_metadata"].resolve()
            ),
        },
        "source_bindings": {
            "split_manifest": _source_binding(manifest_path),
            "selected_consumer": _source_binding(selected_path),
            "c0_checkpoint": _source_binding(required_files["c0_checkpoint"]),
            "consumer_checkpoint": _source_binding(required_files["consumer_checkpoint"]),
            "consumer_teacher_config": _source_binding(
                required_files["consumer_teacher_config"]
            ),
            "a0_source_metadata": _source_binding(required_files["a0_source_metadata"]),
            "reference_p7b_source_metadata": _source_binding(
                required_files["reference_p7b_source_metadata"]
            ),
        },
        "run_dirs": run_dirs,
        "extension_rule": {
            "description": (
                "Extend to seeds 20724 and 20825 only after the three-seed validation "
                "report is frozen."
            ),
            "condition": "P7b stack_val wins 3/3 and mean paired accuracy delta > 0",
        },
    }
    payload["artifact_hash"] = stable_fusion_json_hash(payload)
    return payload


def require_high_data_study_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    manifest = dict(payload)
    if (
        manifest.get("contract") != HIGH_DATA_STUDY_MANIFEST_CONTRACT
        or manifest.get("ok") is not True
    ):
        raise ValueError("invalid high-data study manifest")
    expected_hash = manifest.get("artifact_hash")
    unhashed = {key: value for key, value in manifest.items() if key != "artifact_hash"}
    if expected_hash != stable_fusion_json_hash(unhashed):
        raise ValueError("high-data study manifest hash mismatch")
    if tuple(int(seed) for seed in manifest.get("matched_seeds") or ()) != HIGH_DATA_MATCHED_SEEDS:
        raise ValueError("high-data study does not use the locked matched seeds")
    if dict(manifest.get("split_counts") or {}) != HIGH_DATA_SPLIT_COUNTS:
        raise ValueError("high-data study split counts changed")
    if manifest.get("final_test_policy") != "sealed_until_explicit_confirmation":
        raise ValueError("high-data final test is not sealed")
    if manifest.get("privileged_final_test_artifacts_present") is not False:
        raise ValueError("high-data manifest permits privileged final-test artifacts")
    return manifest


def build_high_data_a0_config(
    source_metadata: Mapping[str, Any],
    *,
    seed: int,
    output_dir: str | Path,
) -> tuple[LocalResidualFieldTaggerTrainConfig, dict[str, Any]]:
    if int(seed) not in HIGH_DATA_TRAINED_A0_SEEDS:
        raise ValueError(f"seed {seed} is not a new high-data A0 seed")
    source = extract_a0_train_config(source_metadata)
    candidate = dict(source)
    candidate["seed"] = int(seed)
    candidate["output_dir"] = str(Path(output_dir).resolve())
    config = LocalResidualFieldTaggerTrainConfig(**candidate)
    normalized = json.loads(json.dumps(asdict(config), sort_keys=True, allow_nan=False))
    differences = sorted(
        key for key in set(source) | set(normalized) if source.get(key) != normalized.get(key)
    )
    problems: list[str] = []
    if int(source.get("seed", -1)) != HIGH_DATA_MATCHED_SEEDS[0]:
        problems.append("source A0 does not use the first locked seed")
    if differences != ["output_dir", "seed"]:
        problems.append(f"A0 clone changed non-allowlisted fields: {differences}")
    if normalized.get("field_source") != RESIDUAL_FIELD_SOURCE_HLT_ONLY:
        problems.append("A0 clone is not HLT-only")
    if bool(normalized.get("require_baseline_warm_start")):
        problems.append("A0 clone must train from scratch")
    for name in _A0_FORBIDDEN_INPUTS:
        if normalized.get(name) not in (None, ""):
            problems.append(f"A0 clone unexpectedly sets {name}")
    audit: dict[str, Any] = {
        "contract": "local_residual_field_high_data_a0_recipe_audit_v1",
        "ok": not problems,
        "problems": problems,
        "seed": int(seed),
        "allowed_differences": ["output_dir", "seed"],
        "observed_differences": differences,
        "source_config_hash": stable_fusion_json_hash(source),
        "candidate_config_hash": stable_fusion_json_hash(normalized),
        "from_scratch": True,
        "runtime_inputs": "HLT_only",
    }
    audit["artifact_hash"] = stable_fusion_json_hash(audit)
    if problems:
        raise ValueError(f"high-data A0 recipe audit failed: {problems}")
    return config, audit


def validate_high_data_p7b_run(
    manifest: Mapping[str, Any],
    *,
    seed: int,
    output_dir: str | Path,
) -> dict[str, Any]:
    manifest = require_high_data_study_manifest(manifest)
    output_dir = Path(output_dir).resolve()
    expected_dir = Path(manifest["run_dirs"][str(seed)]["P7b"]).resolve()
    if output_dir != expected_dir:
        raise ValueError(f"P7b output directory differs from manifest: {output_dir}")
    report_path = output_dir / "run_report.json"
    source_path = output_dir / "source_metadata.json"
    checkpoint_path = output_dir / "best_model_val.pt"
    for label, path in (
        ("P7b run report", report_path),
        ("P7b source metadata", source_path),
        ("P7b checkpoint", checkpoint_path),
    ):
        _require_file(path, label)
    report = load_json_object(report_path)
    source = load_json_object(source_path)
    config = source.get("config")
    if not isinstance(config, Mapping) or int(config.get("seed", -1)) != int(seed):
        raise ValueError("P7b source seed mismatch")
    reference_source = load_json_object(manifest["paths"]["reference_p7b_source_metadata"])
    reference_config = reference_source.get("config")
    if not isinstance(reference_config, Mapping):
        raise ValueError("reference P7b source is missing config")
    differences = sorted(
        key
        for key in set(reference_config) | set(config)
        if reference_config.get(key) != config.get(key)
    )
    expected_differences = list(manifest["p7b_allowed_config_differences"])
    if differences != expected_differences:
        raise ValueError(
            "high-data P7b changed non-allowlisted recipe fields; "
            f"expected {expected_differences}, observed {differences}"
        )
    if report.get("run_id") != "P7b" or report.get("deployable") is not True:
        raise ValueError("run is not a deployable P7b")
    if report.get("runtime_inputs") != "HLT_only":
        raise ValueError("P7b runtime inputs are not HLT-only")
    if report.get("final_test") not in (None, {}):
        raise ValueError("validation training opened final_test")
    if dict(report.get("oracle_teacher_logits_paths") or {}):
        raise ValueError("high-data P7b unexpectedly used an oracle-logit cache")
    if bool(config.get("oracle_logit_only_fallback")):
        raise ValueError("high-data P7b must use the online frozen-oracle forward")
    if report.get("selected_consumer_id") != "Orobust_light":
        raise ValueError("high-data P7b did not use Orobust_light")

    expected_oracle_hash = sha256_file(manifest["paths"]["consumer_checkpoint"])
    expected_predictor_hash = sha256_file(manifest["paths"]["c0_checkpoint"])
    observed_oracle_hash = report.get("oracle_teacher_checkpoint_hash")
    observed_student_hash = (report.get("student_initialization") or {}).get(
        "student_init_checkpoint_hash"
    )
    observed_predictor_hash = (report.get("predictor_warm_start") or {}).get(
        "checkpoint_hash"
    )
    if observed_oracle_hash != expected_oracle_hash or observed_student_hash != expected_oracle_hash:
        raise ValueError("P7b oracle/student initialization is not the frozen 3M consumer")
    if observed_predictor_hash != expected_predictor_hash:
        raise ValueError("P7b predictor initialization is not the frozen 3M C0")
    completion: dict[str, Any] = {
        "contract": HIGH_DATA_RUN_COMPLETION_CONTRACT,
        "ok": True,
        "study_manifest_hash": manifest["artifact_hash"],
        "recipe": "P7b",
        "seed": int(seed),
        "runtime_inputs": "HLT_only",
        "deployable": True,
        "final_test_evaluated": False,
        "oracle_execution_mode": "frozen_checkpoint_online",
        "recipe_difference_paths": differences,
        "oracle_teacher_checkpoint_sha256": expected_oracle_hash,
        "predictor_checkpoint_sha256": expected_predictor_hash,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "run_report_path": str(report_path),
        "run_report_sha256": sha256_file(report_path),
    }
    completion["artifact_hash"] = stable_fusion_json_hash(completion)
    return completion


def _metric_row(report: Mapping[str, Any], split: str, *, expected_n: int) -> dict[str, Any]:
    metrics = report.get("best_model_val" if split == "model_val" else split)
    if not isinstance(metrics, Mapping):
        raise ValueError(f"run report is missing {split}")
    if metrics.get("valid_for_selection") is not True:
        raise ValueError(f"{split} is not valid for selection")
    if int(metrics.get("n_jets", -1)) != int(expected_n):
        raise ValueError(
            f"{split} has {metrics.get('n_jets')} jets, expected {expected_n}"
        )
    return {
        "accuracy": float(metrics["accuracy"]),
        "cross_entropy": float(metrics["cross_entropy"]),
        "n_jets": int(metrics["n_jets"]),
    }


def _read_validation_run(
    run_dir: Path,
    *,
    recipe: str,
    seed: int,
) -> dict[str, Any]:
    report_path = run_dir / "run_report.json"
    source_path = run_dir / "source_metadata.json"
    checkpoint_path = run_dir / "best_model_val.pt"
    for label, path in (
        ("run report", report_path),
        ("source metadata", source_path),
        ("checkpoint", checkpoint_path),
    ):
        _require_file(path, f"{recipe} {label}")
    report = load_json_object(report_path)
    source = load_json_object(source_path)
    config = source.get("config")
    if not isinstance(config, Mapping) or int(config.get("seed", -1)) != int(seed):
        raise ValueError(f"{recipe} seed mismatch at {run_dir}")
    if recipe == "P7b":
        if report.get("deployable") is not True or report.get("runtime_inputs") != "HLT_only":
            raise ValueError("P7b validation row is not deployable HLT-only")
        if report.get("final_test") not in (None, {}):
            raise ValueError("P7b validation row opened final_test")
    elif config.get("field_source") != RESIDUAL_FIELD_SOURCE_HLT_ONLY:
        raise ValueError("A0 validation row is not HLT-only")
    return {
        "run_dir": str(run_dir),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "run_report_sha256": sha256_file(report_path),
        "best_epoch": int(report["best_epoch"]),
        "model_val": _metric_row(
            report, "model_val", expected_n=HIGH_DATA_SPLIT_COUNTS["model_val"]
        ),
        "stack_val": _metric_row(
            report, "stack_val", expected_n=HIGH_DATA_SPLIT_COUNTS["stack_val"]
        ),
    }


def _summary(values: Iterable[float], *, t_critical: float = T_CRITICAL_95_DF2) -> dict[str, Any]:
    rows = [float(value) for value in values]
    result: dict[str, Any] = {"n": len(rows), "mean": mean(rows)}
    if len(rows) > 1:
        result["sample_std"] = stdev(rows)
        half_width = float(t_critical) * result["sample_std"] / math.sqrt(len(rows))
        result["mean_interval_95_t"] = [
            result["mean"] - half_width,
            result["mean"] + half_width,
        ]
    else:
        result["sample_std"] = None
        result["mean_interval_95_t"] = None
    return result


def _summarize_pairs(pairs: list[dict[str, Any]], split: str) -> dict[str, Any]:
    a0_accuracy = [pair["runs"]["A0"][split]["accuracy"] for pair in pairs]
    p7b_accuracy = [pair["runs"]["P7b"][split]["accuracy"] for pair in pairs]
    accuracy_deltas = [pair[split]["accuracy_delta_p7b_minus_a0"] for pair in pairs]
    ce_deltas = [pair[split]["cross_entropy_delta_p7b_minus_a0"] for pair in pairs]
    return {
        "A0_accuracy": _summary(a0_accuracy),
        "P7b_accuracy": _summary(p7b_accuracy),
        "paired_accuracy_delta_p7b_minus_a0": _summary(accuracy_deltas),
        "paired_cross_entropy_delta_p7b_minus_a0": _summary(ce_deltas),
        "p7b_accuracy_wins": sum(delta > 0.0 for delta in accuracy_deltas),
        "ties": sum(delta == 0.0 for delta in accuracy_deltas),
        "a0_accuracy_wins": sum(delta < 0.0 for delta in accuracy_deltas),
    }


def build_high_data_validation_report(
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = require_high_data_study_manifest(manifest)
    pairs: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for raw_seed in manifest["matched_seeds"]:
        seed = int(raw_seed)
        runs: dict[str, dict[str, Any]] = {}
        for recipe in ("A0", "P7b"):
            run = _read_validation_run(
                Path(manifest["run_dirs"][str(seed)][recipe]),
                recipe=recipe,
                seed=seed,
            )
            runs[recipe] = run
            for split in ("model_val", "stack_val"):
                rows.append(
                    {
                        "seed": seed,
                        "recipe": recipe,
                        "split": split,
                        "accuracy": run[split]["accuracy"],
                        "cross_entropy": run[split]["cross_entropy"],
                        "n_jets": run[split]["n_jets"],
                        "best_epoch": run["best_epoch"],
                        "run_dir": run["run_dir"],
                    }
                )
        pair: dict[str, Any] = {"seed": seed, "runs": runs}
        for split in ("model_val", "stack_val"):
            pair[split] = {
                "accuracy_delta_p7b_minus_a0": (
                    runs["P7b"][split]["accuracy"] - runs["A0"][split]["accuracy"]
                ),
                "cross_entropy_delta_p7b_minus_a0": (
                    runs["P7b"][split]["cross_entropy"]
                    - runs["A0"][split]["cross_entropy"]
                ),
            }
        pairs.append(pair)

    summaries = {
        split: _summarize_pairs(pairs, split)
        for split in ("model_val", "stack_val")
    }
    primary = summaries["stack_val"]
    extend = (
        int(primary["p7b_accuracy_wins"]) == len(HIGH_DATA_MATCHED_SEEDS)
        and float(primary["paired_accuracy_delta_p7b_minus_a0"]["mean"]) > 0.0
    )
    payload: dict[str, Any] = {
        "contract": HIGH_DATA_VALIDATION_REPORT_CONTRACT,
        "ok": True,
        "campaign_id": manifest["campaign_id"],
        "manifest_artifact_hash": manifest["artifact_hash"],
        "matched_seeds": list(manifest["matched_seeds"]),
        "selection_split": "model_val",
        "primary_comparison_split": "stack_val",
        "final_test_evaluated": False,
        "final_test_status": "SEALED",
        "pairs": pairs,
        "summaries": summaries,
        "extension_recommendation": {
            "extend_to_five_seeds": bool(extend),
            "rule": manifest["extension_rule"]["condition"],
        },
    }
    payload["artifact_hash"] = stable_fusion_json_hash(payload)
    return payload, rows


def build_high_data_final_test_report(
    manifest: Mapping[str, Any],
    *,
    validation_report: Mapping[str, Any],
    predictions_root: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = require_high_data_study_manifest(manifest)
    if validation_report.get("contract") != HIGH_DATA_VALIDATION_REPORT_CONTRACT:
        raise ValueError("final-test report requires the frozen validation report")
    if validation_report.get("artifact_hash") != stable_fusion_json_hash(
        {key: value for key, value in validation_report.items() if key != "artifact_hash"}
    ):
        raise ValueError("validation report hash mismatch")
    predictions_root = Path(predictions_root).resolve()
    pairs: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for raw_seed in manifest["matched_seeds"]:
        seed = int(raw_seed)
        runs: dict[str, dict[str, Any]] = {}
        for recipe in ("A0", "P7b"):
            metadata_path = (
                predictions_root
                / f"seed_{seed}"
                / recipe
                / "final_test_predictions_metadata.json"
            )
            _require_file(metadata_path, f"{recipe} final-test metadata")
            metadata = load_json_object(metadata_path)
            dataset = metadata.get("dataset_metadata")
            metrics = metadata.get("metrics")
            if metadata.get("deployable") is not True or not isinstance(dataset, Mapping):
                raise ValueError(f"{recipe} seed {seed} final-test row is not deployable")
            if dataset.get("allowed_inputs") != "HLT_particles_only_deployable_final_test":
                raise ValueError(f"{recipe} seed {seed} final test is not HLT-only")
            if dataset.get("target_fields_present") is not False:
                raise ValueError("target fields appeared in deployable final-test evaluation")
            if dataset.get("teacher_logits_present") is not False:
                raise ValueError("teacher logits appeared in deployable final-test evaluation")
            if not isinstance(metrics, Mapping) or int(metrics.get("n_jets", -1)) != 1_000_000:
                raise ValueError(f"{recipe} seed {seed} final-test coverage is not 1M")
            checkpoint = Path(manifest["run_dirs"][str(seed)][recipe]) / "best_model_val.pt"
            if metadata.get("checkpoint_hash") != sha256_file(checkpoint):
                raise ValueError(f"{recipe} seed {seed} final-test checkpoint hash mismatch")
            row = {
                "accuracy": float(metrics["accuracy"]),
                "cross_entropy": float(metrics["cross_entropy"]),
                "n_jets": int(metrics["n_jets"]),
                "metadata_path": str(metadata_path),
                "metadata_sha256": sha256_file(metadata_path),
            }
            runs[recipe] = row
            rows.append({"seed": seed, "recipe": recipe, "split": "final_test", **row})
        pair = {
            "seed": seed,
            "runs": runs,
            "final_test": {
                "accuracy_delta_p7b_minus_a0": (
                    runs["P7b"]["accuracy"] - runs["A0"]["accuracy"]
                ),
                "cross_entropy_delta_p7b_minus_a0": (
                    runs["P7b"]["cross_entropy"] - runs["A0"]["cross_entropy"]
                ),
            },
        }
        pairs.append(pair)
    summary_pairs = [
        {
            "runs": {
                recipe: {"final_test": pair["runs"][recipe]}
                for recipe in ("A0", "P7b")
            },
            "final_test": pair["final_test"],
        }
        for pair in pairs
    ]
    payload: dict[str, Any] = {
        "contract": HIGH_DATA_FINAL_TEST_REPORT_CONTRACT,
        "ok": True,
        "campaign_id": manifest["campaign_id"],
        "manifest_artifact_hash": manifest["artifact_hash"],
        "validation_report_artifact_hash": validation_report["artifact_hash"],
        "matched_seeds": list(manifest["matched_seeds"]),
        "final_test_evaluated": True,
        "final_test_runtime_inputs": "HLT_only",
        "oracle_diagnostics_included": False,
        "pairs": pairs,
        "summary": _summarize_pairs(summary_pairs, "final_test"),
    }
    payload["artifact_hash"] = stable_fusion_json_hash(payload)
    return payload, rows


def write_rows_csv(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = [dict(row) for row in rows]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


__all__ = [
    "HIGH_DATA_FINAL_TEST_REPORT_CONTRACT",
    "HIGH_DATA_HLT_SPLITS",
    "HIGH_DATA_MATCHED_SEEDS",
    "HIGH_DATA_PRIVILEGED_SPLITS",
    "HIGH_DATA_REUSED_A0_SEEDS",
    "HIGH_DATA_RUN_COMPLETION_CONTRACT",
    "HIGH_DATA_SPLIT_COUNTS",
    "HIGH_DATA_STUDY_MANIFEST_CONTRACT",
    "HIGH_DATA_TRAINED_A0_SEEDS",
    "HIGH_DATA_VALIDATION_REPORT_CONTRACT",
    "build_frozen_high_data_selection",
    "build_high_data_a0_config",
    "build_high_data_final_test_report",
    "build_high_data_study_manifest",
    "build_high_data_validation_report",
    "load_json_object",
    "require_high_data_study_manifest",
    "save_json",
    "sha256_file",
    "validate_high_data_p7b_run",
    "write_rows_csv",
]
