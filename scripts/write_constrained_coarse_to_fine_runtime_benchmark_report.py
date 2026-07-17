#!/usr/bin/env python3
"""Normalize and fail-close C2F runtime benchmark outputs for Step 7."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.jetclass_data import manifest_hash, load_split_manifest  # noqa: E402
from scripts.build_constrained_coarse_to_fine_runtime_benchmark_plan import (  # noqa: E402
    BENCHMARK_PLAN_CONTRACT,
)


def _read_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _nested(payload: Mapping[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _epoch_summary(epoch: Mapping[str, Any]) -> dict[str, Any]:
    train = epoch.get("train") if isinstance(epoch.get("train"), Mapping) else {}
    val = epoch.get("model_val") if isinstance(epoch.get("model_val"), Mapping) else {}
    wanted = (
        "selection.reconstruction_score",
        "loss.total",
        "loss.hierarchy",
        "loss.slot",
        "slot.metric.matched_pT_mae",
        "slot.metric.matched_eta_mae",
        "slot.metric.matched_phi_mae",
        "slot.metric.matched_pid_accuracy",
        "runtime.elapsed_seconds",
        "runtime.batches_per_second",
        "runtime.jets_per_second",
        "runtime.batch_loading_seconds",
        "runtime.batch_loading_fraction",
        "runtime.cpu_process_seconds",
        "runtime.cpu_process_utilization",
        "runtime.host_max_rss_bytes",
        "runtime.cuda_peak_allocated_bytes",
        "runtime.cuda_peak_reserved_bytes",
        "nonfinite_batches_skipped",
        "n_jets",
    )
    result: dict[str, Any] = {"epoch": epoch.get("epoch")}
    for split_name, split_metrics in (("train", train), ("model_val", val)):
        result[split_name] = {key: split_metrics.get(key) for key in wanted if key in split_metrics}
        # Step 8 compares all named loss/accounting components, not only the
        # compact human-facing table above. Keep finite scalar metrics intact.
        result[f"{split_name}_numeric_metrics"] = {
            str(key): value
            for key, value in split_metrics.items()
            if _finite(value) is not None
        }
    return result


def _weighted_rate(epochs: list[Mapping[str, Any]], split: str, rate_key: str) -> float | None:
    numerator = 0.0
    denominator = 0.0
    for epoch in epochs:
        metrics = epoch.get(split)
        if not isinstance(metrics, Mapping):
            continue
        elapsed = _finite(metrics.get("runtime.elapsed_seconds"))
        rate = _finite(metrics.get(rate_key))
        if elapsed is None or rate is None:
            continue
        numerator += elapsed * rate
        denominator += elapsed
    return numerator / denominator if denominator else None


def _max_metric(epochs: list[Mapping[str, Any]], split: str, metric: str) -> float | None:
    values = [_finite(_nested(epoch, split, metric)) for epoch in epochs]
    values = [value for value in values if value is not None]
    return max(values) if values else None


def _sum_metric(epochs: list[Mapping[str, Any]], split: str, metric: str) -> float | None:
    values = [_finite(_nested(epoch, split, metric)) for epoch in epochs]
    values = [value for value in values if value is not None]
    return sum(values) if values else None


def _validate_run(
    row: Mapping[str, Any],
    run_dir: Path,
    *,
    calibration_manifest_hash: str,
    requested_epochs: int,
) -> tuple[dict[str, Any], list[str]]:
    problems: list[str] = []
    run_report_path = run_dir / "run_report.json"
    curves_path = run_dir / "training_curves.json"
    memory_path = run_dir / "memory_preflight.json"
    for path in (run_report_path, curves_path, memory_path):
        if not path.is_file():
            problems.append(f"missing {path.name}")
    if problems:
        return {"run_id": row["run_id"], "output_dir": str(run_dir)}, problems
    report = _read_json(run_report_path)
    curves_payload = _read_json(curves_path)
    memory_payload = _read_json(memory_path)
    if not bool(report.get("ok")):
        problems.append("run_report.ok is false")
    for key in ("variant", "runtime_profile"):
        expected = row["variant"] if key == "variant" else row["runtime_profile"]
        if report.get(key) != expected:
            problems.append(f"{key}={report.get(key)!r}, expected {expected!r}")
    config = report.get("training_config")
    if not isinstance(config, Mapping):
        problems.append("missing training_config")
        config = {}
    expected_config = {
        "precision_mode": row["precision_mode"],
        "batch_size": row["train_batch_size"],
        "eval_batch_size": row["eval_batch_size"],
        "num_workers": row["num_workers"],
        "lr_schedule": row["lr_schedule"],
        "learning_rate": row["learning_rate"],
        "hlt_encoder_lr_scale": row["hlt_encoder_lr_scale"],
        "hungarian_executor": row["hungarian_executor"],
        "hungarian_workers": row["hungarian_workers"],
        "fixed_horizon": True,
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            problems.append(f"training_config.{key}={config.get(key)!r}, expected {expected!r}")
    provenance = report.get("provenance")
    if not isinstance(provenance, Mapping):
        problems.append("missing provenance")
    else:
        for split in ("model_train", "model_val"):
            source = provenance.get(split)
            actual = source.get("source_manifest_hash") if isinstance(source, Mapping) else None
            if actual != calibration_manifest_hash:
                problems.append(f"{split}.source_manifest_hash mismatch")
    epochs = curves_payload.get("epochs")
    if not isinstance(epochs, list) or len(epochs) != requested_epochs:
        problems.append(f"training curves contain {len(epochs) if isinstance(epochs, list) else 'invalid'} epochs; expected {requested_epochs}")
        epochs = epochs if isinstance(epochs, list) else []
    if report.get("stop_reason") != "fixed_horizon_completed":
        problems.append(f"stop_reason={report.get('stop_reason')!r}, expected fixed_horizon_completed")
    if report.get("completed_epochs") != requested_epochs:
        problems.append("completed_epochs mismatch")
    for epoch in epochs:
        if not isinstance(epoch, Mapping):
            problems.append("invalid epoch record")
            continue
        for split in ("train", "model_val"):
            if not isinstance(epoch.get(split), Mapping):
                problems.append(f"epoch missing {split} metrics")
    final_epoch = epochs[-1] if epochs and isinstance(epochs[-1], Mapping) else {}
    resource = {
        "train_batches_per_second_weighted": _weighted_rate(epochs, "train", "runtime.batches_per_second"),
        "train_jets_per_second_weighted": _weighted_rate(epochs, "train", "runtime.jets_per_second"),
        "train_elapsed_seconds": _sum_metric(epochs, "train", "runtime.elapsed_seconds"),
        "model_val_elapsed_seconds": _sum_metric(epochs, "model_val", "runtime.elapsed_seconds"),
        "peak_cuda_allocated_bytes": _max_metric(epochs, "train", "runtime.cuda_peak_allocated_bytes"),
        "peak_cuda_reserved_bytes": _max_metric(epochs, "train", "runtime.cuda_peak_reserved_bytes"),
        "peak_host_max_rss_bytes": _max_metric(epochs, "train", "runtime.host_max_rss_bytes"),
        "total_batch_loading_seconds": _sum_metric(epochs, "train", "runtime.batch_loading_seconds"),
        "total_cpu_process_seconds": _sum_metric(epochs, "train", "runtime.cpu_process_seconds"),
        "mean_cpu_process_utilization": _weighted_rate(epochs, "train", "runtime.cpu_process_utilization"),
        "nonfinite_batches_skipped": _sum_metric(epochs, "train", "nonfinite_batches_skipped"),
    }
    result = {
        "run_id": row["run_id"],
        "matrix_case": row["matrix_case"],
        "variant": row["variant"],
        "requested_execution": dict(row),
        "output_dir": str(run_dir),
        "checkpoint_sha256": report.get("checkpoint_sha256"),
        "runtime_profile_hash": report.get("runtime_profile_hash"),
        "code_environment": report.get("code_environment"),
        "provenance": provenance,
        "memory_preflight": memory_payload,
        "final_epoch": _epoch_summary(final_epoch),
        "best_model_val": report.get("best_model_val"),
        "resource": resource,
        "epochs": [_epoch_summary(epoch) for epoch in epochs if isinstance(epoch, Mapping)],
    }
    return result, problems


def write_report(*, plan_path: Path, benchmark_root: Path, output: Path) -> dict[str, Any]:
    plan = _read_json(plan_path)
    if plan.get("contract") != BENCHMARK_PLAN_CONTRACT:
        raise ValueError("unsupported benchmark plan contract")
    manifest_path = Path(str(plan["calibration_manifest"]))
    calibration_hash = manifest_hash(load_split_manifest(manifest_path))
    calibration_validation_path = Path(str(plan["calibration_root"])) / "runtime_benchmark_calibration_validation.json"
    calibration_validation = _read_json(calibration_validation_path)
    if not bool(calibration_validation.get("ok")):
        raise ValueError("calibration validation is not ok")
    if calibration_validation.get("calibration_manifest_hash") != calibration_hash:
        raise ValueError("calibration validation manifest hash mismatch")
    calibration_inputs = calibration_validation.get("splits")
    if not isinstance(calibration_inputs, Mapping) or set(calibration_inputs) != {"model_train", "model_val"}:
        raise ValueError("calibration validation lacks model_train/model_val input provenance")
    for split, row in calibration_inputs.items():
        if not isinstance(row, Mapping) or any(row.get(key) in (None, "") for key in (
            "hlt_content_hash", "offline_content_hash", "target_content_hash", "jet_identity_hash"
        )):
            raise ValueError(f"calibration validation has incomplete {split} input provenance")
    rows = plan.get("runs")
    if not isinstance(rows, list) or not rows:
        raise ValueError("benchmark plan has no runs")
    records: list[dict[str, Any]] = []
    problems: dict[str, list[str]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("benchmark plan has invalid run record")
        run_id = str(row["run_id"])
        result, run_problems = _validate_run(
            row,
            benchmark_root / "reconstructors" / run_id,
            calibration_manifest_hash=calibration_hash,
            requested_epochs=int(plan["epochs"]),
        )
        records.append(result)
        if run_problems:
            problems[run_id] = run_problems
    payload: dict[str, Any] = {
        "contract": "constrained_c2f_runtime_benchmark_report_v1",
        "ok": not problems,
        "benchmark_plan": str(plan_path),
        "benchmark_plan_hash": plan.get("benchmark_plan_hash"),
        "benchmark_root": str(benchmark_root),
        "calibration_manifest": str(manifest_path),
        "calibration_manifest_hash": calibration_hash,
        "calibration_validation_path": str(calibration_validation_path),
        "calibration_validation_sha256": hashlib.sha256(calibration_validation_path.read_bytes()).hexdigest(),
        "calibration_input_provenance": calibration_inputs,
        "requested_epochs": int(plan["epochs"]),
        "records": records,
        "problems": problems,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = output.with_suffix(".md")
    lines = ["# C2F Runtime Benchmark", "", f"Status: {'OK' if payload['ok'] else 'FAILED'}", "", "| Run | Variant | Case | Train jets/s | CUDA reserved | Problems |", "|---|---|---:|---:|---:|---:|"]
    for record in records:
        resource = record.get("resource", {})
        run_problems = "; ".join(problems.get(record["run_id"], [])) or ""
        lines.append(
            f"| {record['run_id']} | {record.get('variant', '')} | {record.get('matrix_case', '')} | "
            f"{resource.get('train_jets_per_second_weighted', '')} | {resource.get('peak_cuda_reserved_bytes', '')} | {run_problems} |"
        )
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = write_report(plan_path=Path(args.plan), benchmark_root=Path(args.benchmark_root), output=Path(args.output))
    print(json.dumps({"ok": report["ok"], "output": args.output, "runs": len(report["records"])}, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
