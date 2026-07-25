"""Matched-seed A0/P7b study contracts and reporting helpers."""

from __future__ import annotations

from dataclasses import asdict
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Mapping

from .fusion_campaign import stable_fusion_json_hash
from .fusion_seed_control import A0_TRAINING_SEED, extract_a0_train_config
from .tagger import RESIDUAL_FIELD_SOURCE_HLT_ONLY
from .tagger_train import LocalResidualFieldTaggerTrainConfig


SEED_STUDY_MANIFEST_CONTRACT = "local_residual_field_seed_study_manifest_v1"
SEED_STUDY_RUN_COMPLETION_CONTRACT = "local_residual_field_seed_study_run_completion_v1"
SEED_STUDY_REPORT_CONTRACT = "local_residual_field_seed_study_report_v1"
MATCHED_SEEDS = (20421, 20522, 20623, 20724, 20825)
REUSED_A0_SEEDS = (20421, 20522)
TRAINED_A0_SEEDS = tuple(seed for seed in MATCHED_SEEDS if seed not in REUSED_A0_SEEDS)
P7B_REFERENCE_SEED = 30421
T_CRITICAL_95_DF4 = 2.7764451051977987

_A0_FORBIDDEN_INPUTS = (
    "baseline_checkpoint",
    "reconstructor_checkpoint",
    "teacher_logits_dir",
    "teacher_logits_train_path",
    "teacher_logits_val_path",
    "teacher_logits_stack_val_path",
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


def _source_binding(path: Path) -> dict[str, Any]:
    _require_file(path, "source artifact")
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def build_seed_study_manifest(
    *,
    campaign_id: str,
    campaign_root: str | Path,
    curriculum_root: str | Path,
    fusion_root: str | Path,
) -> dict[str, Any]:
    """Validate and bind the immutable inputs reused by the matched study."""

    campaign_root = Path(campaign_root).resolve()
    curriculum_root = Path(curriculum_root).resolve()
    fusion_root = Path(fusion_root).resolve()
    split_manifest = curriculum_root / "inputs" / "split_manifest" / "split_manifest.json.gz"
    hlt_cache = curriculum_root / "inputs" / "hlt_cache"
    target_cache = curriculum_root / "targets"
    selected_consumer_path = curriculum_root / "selected_consumer.json"
    c0_checkpoint = curriculum_root / "reconstructors" / "C0" / "best_model_val.pt"
    a0_dir = curriculum_root / "taggers" / "A0"
    a0_seed1_dir = fusion_root / "taggers" / "A0_seed1"
    p7b_reference_dir = curriculum_root / "curriculum" / "P7b"

    _require_file(split_manifest, "split manifest")
    _require_dir(hlt_cache, "HLT cache")
    _require_dir(target_cache, "target cache")
    _require_file(selected_consumer_path, "selected consumer")
    selected_consumer = load_json_object(selected_consumer_path)
    consumer_id = str(selected_consumer.get("selected_consumer_id") or "")
    if consumer_id not in {"Ofull", "Orobust_light"}:
        raise ValueError(f"unsupported selected consumer: {consumer_id!r}")
    consumer_dir = curriculum_root / "taggers" / consumer_id
    oracle_logits_dir = curriculum_root / "oracle_teacher_logits" / consumer_id

    required_files = {
        "c0_checkpoint": c0_checkpoint,
        "a0_checkpoint": a0_dir / "best_model_val.pt",
        "a0_source_metadata": a0_dir / "source_metadata.json",
        "a0_run_report": a0_dir / "run_report.json",
        "a0_seed1_checkpoint": a0_seed1_dir / "best_model_val.pt",
        "a0_seed1_source_metadata": a0_seed1_dir / "source_metadata.json",
        "a0_seed1_run_report": a0_seed1_dir / "run_report.json",
        "consumer_checkpoint": consumer_dir / "best_model_val.pt",
        "consumer_teacher_config": consumer_dir / "teacher_config.json",
        "consumer_run_report": consumer_dir / "run_report.json",
        "p7b_reference_run_report": p7b_reference_dir / "run_report.json",
        "p7b_reference_source_metadata": p7b_reference_dir / "source_metadata.json",
    }
    for label, path in required_files.items():
        _require_file(path, label)

    a0_seed = int(extract_a0_train_config(load_json_object(required_files["a0_source_metadata"]))["seed"])
    a0_seed1 = int(
        extract_a0_train_config(load_json_object(required_files["a0_seed1_source_metadata"]))["seed"]
    )
    if a0_seed != MATCHED_SEEDS[0] or a0_seed1 != MATCHED_SEEDS[1]:
        raise ValueError(
            f"reused A0 seeds must be {MATCHED_SEEDS[:2]}, observed {(a0_seed, a0_seed1)}"
        )
    p7b_reference_source = load_json_object(required_files["p7b_reference_source_metadata"])
    p7b_reference_config = p7b_reference_source.get("config")
    if not isinstance(p7b_reference_config, Mapping):
        raise ValueError("P7b reference source metadata is missing its config")
    if int(p7b_reference_config.get("seed", -1)) != P7B_REFERENCE_SEED:
        raise ValueError(f"P7b reference seed must be {P7B_REFERENCE_SEED}")
    if bool(p7b_reference_config.get("oracle_logit_only_fallback")):
        raise ValueError("P7b reference unexpectedly used oracle-logit-only fallback")
    p7b_reference_report = load_json_object(required_files["p7b_reference_run_report"])
    if dict(p7b_reference_report.get("oracle_teacher_logits_paths") or {}):
        raise ValueError("P7b reference unexpectedly resolved cached oracle teacher logits")

    run_dirs: dict[str, dict[str, str]] = {}
    for seed in MATCHED_SEEDS:
        run_dirs[str(seed)] = {
            "A0": str(
                (
                    a0_dir
                    if seed == MATCHED_SEEDS[0]
                    else a0_seed1_dir
                    if seed == MATCHED_SEEDS[1]
                    else campaign_root / "runs" / f"seed_{seed}" / "A0"
                ).resolve()
            ),
            "P7b": str((campaign_root / "runs" / f"seed_{seed}" / "P7b").resolve()),
        }

    payload: dict[str, Any] = {
        "contract": SEED_STUDY_MANIFEST_CONTRACT,
        "ok": True,
        "campaign_id": str(campaign_id),
        "campaign_root": str(campaign_root),
        "curriculum_root": str(curriculum_root),
        "fusion_root": str(fusion_root),
        "matched_seeds": list(MATCHED_SEEDS),
        "reused_a0_seeds": list(REUSED_A0_SEEDS),
        "trained_a0_seeds": list(TRAINED_A0_SEEDS),
        "p7b_reference_seed": P7B_REFERENCE_SEED,
        "p7b_reference_dir": str(p7b_reference_dir.resolve()),
        "final_test_policy": "forbidden",
        "oracle_execution_mode": "frozen_checkpoint_online",
        "oracle_logit_cache_required": False,
        "oracle_logit_cache_present": oracle_logits_dir.is_dir(),
        "selection_split": "model_val",
        "comparison_split": "stack_val",
        "selected_consumer_id": consumer_id,
        "paths": {
            "split_manifest": str(split_manifest.resolve()),
            "hlt_cache_dir": str(hlt_cache.resolve()),
            "target_cache_dir": str(target_cache.resolve()),
            "selected_consumer_json": str(selected_consumer_path.resolve()),
            "c0_checkpoint": str(c0_checkpoint.resolve()),
            "consumer_checkpoint": str(required_files["consumer_checkpoint"].resolve()),
            "consumer_teacher_config": str(required_files["consumer_teacher_config"].resolve()),
            "consumer_run_report": str(required_files["consumer_run_report"].resolve()),
            "oracle_teacher_logits_dir": (
                str(oracle_logits_dir.resolve()) if oracle_logits_dir.is_dir() else None
            ),
            "a0_source_metadata": str(required_files["a0_source_metadata"].resolve()),
            "p7b_reference_source_metadata": str(
                required_files["p7b_reference_source_metadata"].resolve()
            ),
        },
        "source_bindings": {
            "split_manifest": _source_binding(split_manifest),
            "selected_consumer": _source_binding(selected_consumer_path),
            "a0_source_metadata": _source_binding(required_files["a0_source_metadata"]),
            "a0_seed1_source_metadata": _source_binding(required_files["a0_seed1_source_metadata"]),
            "p7b_reference_run_report": _source_binding(required_files["p7b_reference_run_report"]),
            "p7b_reference_source_metadata": _source_binding(
                required_files["p7b_reference_source_metadata"]
            ),
        },
        "run_dirs": run_dirs,
    }
    payload["artifact_hash"] = stable_fusion_json_hash(payload)
    return payload


def require_seed_study_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    manifest = dict(payload)
    if manifest.get("contract") != SEED_STUDY_MANIFEST_CONTRACT or manifest.get("ok") is not True:
        raise ValueError("invalid seed-study manifest")
    expected = manifest.get("artifact_hash")
    without_hash = {key: value for key, value in manifest.items() if key != "artifact_hash"}
    if expected != stable_fusion_json_hash(without_hash):
        raise ValueError("seed-study manifest hash mismatch")
    if tuple(int(seed) for seed in manifest.get("matched_seeds") or ()) != MATCHED_SEEDS:
        raise ValueError("seed-study manifest does not use the locked matched seeds")
    if manifest.get("final_test_policy") != "forbidden":
        raise ValueError("seed-study manifest must forbid final-test evaluation")
    return manifest


def build_a0_seed_study_config(
    source_metadata: Mapping[str, Any],
    *,
    seed: int,
    output_dir: str | Path,
) -> tuple[LocalResidualFieldTaggerTrainConfig, dict[str, Any]]:
    """Clone the audited A0 recipe, changing only output_dir and training seed."""

    if int(seed) not in MATCHED_SEEDS:
        raise ValueError(f"seed {seed} is outside the locked study seeds")
    source = extract_a0_train_config(source_metadata)
    candidate = dict(source)
    candidate["seed"] = int(seed)
    candidate["output_dir"] = str(Path(output_dir).resolve())
    config = LocalResidualFieldTaggerTrainConfig(**candidate)
    normalized = json.loads(json.dumps(asdict(config), sort_keys=True, allow_nan=False))
    differences = [
        key
        for key in sorted(set(source) | set(normalized))
        if source.get(key) != normalized.get(key)
    ]
    problems: list[str] = []
    if int(source.get("seed", -1)) != A0_TRAINING_SEED:
        problems.append(f"source A0 seed must be {A0_TRAINING_SEED}")
    if differences != ["output_dir", "seed"]:
        problems.append(f"A0 clone changed non-allowlisted fields: {differences}")
    if normalized.get("field_source") != RESIDUAL_FIELD_SOURCE_HLT_ONLY:
        problems.append("A0 clone must be HLT-only")
    if bool(normalized.get("require_baseline_warm_start")):
        problems.append("A0 clone must train from scratch")
    for field_name in _A0_FORBIDDEN_INPUTS:
        if normalized.get(field_name) not in (None, ""):
            problems.append(f"A0 clone unexpectedly sets {field_name}")
    for field_name in ("kd_loss_weight", "reconstructor_loss_weight"):
        if float(normalized.get(field_name, 0.0)) != 0.0:
            problems.append(f"A0 clone must use {field_name}=0")
    audit: dict[str, Any] = {
        "contract": "local_residual_field_seed_study_a0_recipe_audit_v1",
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
        raise ValueError(f"A0 seed-study recipe audit failed: {problems}")
    return config, audit


def _metric_row(report: Mapping[str, Any], split: str) -> dict[str, Any]:
    metrics = report.get("best_model_val" if split == "model_val" else split)
    if not isinstance(metrics, Mapping):
        raise ValueError(f"run report is missing {split} metrics")
    if metrics.get("valid_for_selection") is not True:
        raise ValueError(f"{split} metrics are not valid for selection")
    return {
        "accuracy": float(metrics["accuracy"]),
        "cross_entropy": float(metrics["cross_entropy"]),
        "n_jets": int(metrics["n_jets"]),
    }


def _read_run(run_dir: Path, *, recipe: str, expected_seed: int) -> dict[str, Any]:
    report_path = run_dir / "run_report.json"
    source_path = run_dir / "source_metadata.json"
    _require_file(report_path, f"{recipe} run report")
    _require_file(source_path, f"{recipe} source metadata")
    report = load_json_object(report_path)
    source = load_json_object(source_path)
    config = source.get("config")
    if not isinstance(config, Mapping) or int(config.get("seed", -1)) != int(expected_seed):
        raise ValueError(f"{recipe} at {run_dir} does not use seed {expected_seed}")
    if recipe == "P7b":
        if report.get("run_id") != "P7b":
            raise ValueError(f"unexpected P7b run_id at {run_dir}")
        if report.get("deployable") is not True or report.get("runtime_inputs") != "HLT_only":
            raise ValueError(f"P7b at {run_dir} is not a deployable HLT-only checkpoint")
        final_test = report.get("final_test")
        if final_test not in (None, {}):
            raise ValueError(f"P7b at {run_dir} unexpectedly evaluated final_test")
    elif source.get("config", {}).get("field_source") != RESIDUAL_FIELD_SOURCE_HLT_ONLY:
        raise ValueError(f"A0 at {run_dir} is not HLT-only")
    return {
        "run_dir": str(run_dir),
        "run_report_sha256": sha256_file(report_path),
        "checkpoint_sha256": sha256_file(run_dir / "best_model_val.pt"),
        "best_epoch": int(report["best_epoch"]),
        "model_val": _metric_row(report, "model_val"),
        "stack_val": _metric_row(report, "stack_val"),
    }


def _summary(values: Iterable[float]) -> dict[str, Any]:
    rows = [float(value) for value in values]
    result: dict[str, Any] = {"n": len(rows), "mean": mean(rows)}
    if len(rows) > 1:
        result["sample_std"] = stdev(rows)
        half_width = T_CRITICAL_95_DF4 * result["sample_std"] / math.sqrt(len(rows))
        result["mean_interval_95_t"] = [result["mean"] - half_width, result["mean"] + half_width]
    else:
        result["sample_std"] = None
        result["mean_interval_95_t"] = None
    return result


def build_seed_study_report(manifest: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = require_seed_study_manifest(manifest)
    rows: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for raw_seed in manifest["matched_seeds"]:
        seed = int(raw_seed)
        run_dirs = manifest["run_dirs"][str(seed)]
        pair_runs: dict[str, dict[str, Any]] = {}
        for recipe in ("A0", "P7b"):
            run = _read_run(Path(run_dirs[recipe]), recipe=recipe, expected_seed=seed)
            pair_runs[recipe] = run
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
        pair: dict[str, Any] = {"seed": seed, "runs": pair_runs}
        for split in ("model_val", "stack_val"):
            pair[split] = {
                "accuracy_delta_p7b_minus_a0": (
                    pair_runs["P7b"][split]["accuracy"] - pair_runs["A0"][split]["accuracy"]
                ),
                "cross_entropy_delta_p7b_minus_a0": (
                    pair_runs["P7b"][split]["cross_entropy"]
                    - pair_runs["A0"][split]["cross_entropy"]
                ),
            }
        pairs.append(pair)

    summaries: dict[str, Any] = {}
    for split in ("model_val", "stack_val"):
        a0_accuracy = [pair["runs"]["A0"][split]["accuracy"] for pair in pairs]
        p7b_accuracy = [pair["runs"]["P7b"][split]["accuracy"] for pair in pairs]
        accuracy_deltas = [pair[split]["accuracy_delta_p7b_minus_a0"] for pair in pairs]
        ce_deltas = [pair[split]["cross_entropy_delta_p7b_minus_a0"] for pair in pairs]
        summaries[split] = {
            "A0_accuracy": _summary(a0_accuracy),
            "P7b_accuracy": _summary(p7b_accuracy),
            "paired_accuracy_delta_p7b_minus_a0": _summary(accuracy_deltas),
            "paired_cross_entropy_delta_p7b_minus_a0": _summary(ce_deltas),
            "p7b_accuracy_wins": sum(delta > 0.0 for delta in accuracy_deltas),
            "ties": sum(delta == 0.0 for delta in accuracy_deltas),
            "a0_accuracy_wins": sum(delta < 0.0 for delta in accuracy_deltas),
        }

    payload: dict[str, Any] = {
        "contract": SEED_STUDY_REPORT_CONTRACT,
        "ok": True,
        "campaign_id": manifest["campaign_id"],
        "manifest_artifact_hash": manifest["artifact_hash"],
        "matched_seeds": list(manifest["matched_seeds"]),
        "selection_split": "model_val",
        "primary_comparison_split": "stack_val",
        "final_test_evaluated": False,
        "pairs": pairs,
        "summaries": summaries,
    }
    payload["artifact_hash"] = stable_fusion_json_hash(payload)
    return payload, rows


def write_rows_csv(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "seed",
        "recipe",
        "split",
        "accuracy",
        "cross_entropy",
        "n_jets",
        "best_epoch",
        "run_dir",
    )
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


__all__ = [
    "MATCHED_SEEDS",
    "REUSED_A0_SEEDS",
    "TRAINED_A0_SEEDS",
    "P7B_REFERENCE_SEED",
    "SEED_STUDY_MANIFEST_CONTRACT",
    "SEED_STUDY_RUN_COMPLETION_CONTRACT",
    "SEED_STUDY_REPORT_CONTRACT",
    "build_seed_study_manifest",
    "require_seed_study_manifest",
    "build_a0_seed_study_config",
    "build_seed_study_report",
    "load_json_object",
    "save_json",
    "sha256_file",
    "write_rows_csv",
]
