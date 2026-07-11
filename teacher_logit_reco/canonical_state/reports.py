"""Step 9 report builder for Canonical Multi-Scale Jet State campaigns."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import (
    CANONICAL_STATE_HLT_DEGRADATION_STRENGTH,
    CANONICAL_STATE_HLT_PROFILE,
    CANONICAL_STATE_HLT_PROFILE_VERSION,
    CANONICAL_STATE_LABEL_FILTER,
    CANONICAL_STATE_LABEL_NAMES,
)
from .variants import (
    CANONICAL_STATE_EXPECTED_RUN_IDS,
    FINAL_TEST_POLICY_MODEL_VAL_ONLY,
    FINAL_TEST_POLICY_PRIMARY_TEACHER_FREE,
    FINAL_TEST_POLICY_REPORT_ONLY,
    FINAL_TEST_POLICY_STACK_ONLY,
    MODEL_KIND_STATE_PREDICTOR_ONLY,
    canonical_state_diagnostic_run_ids,
    canonical_state_fusion_run_ids,
    canonical_state_oracle_run_ids,
    canonical_state_primary_run_ids,
    canonical_state_required_dependencies,
    canonical_state_variant_registry,
    canonical_state_variant_spec,
)


CANONICAL_STATE_REPORT_CONTRACT = "canonical_state_step9_metrics_and_reports_v1"
CANONICAL_STATE_REPORT_STEP = "canonical_state_step9_metrics_and_reports"
CANONICAL_STATE_REPORT_JSON = "canonical_state_report.json"
CANONICAL_STATE_REPORT_MD = "canonical_state_report.md"
CANONICAL_STATE_RUN_REPORT_NAME = "run_report.json"
CANONICAL_STATE_REPORT_SPLITS = ("model_val", "stack_val", "final_test")
CANONICAL_STATE_REQUIRED_PROVENANCE_FIELDS = (
    "manifest_hash",
    "hlt_profile",
    "hlt_profile_version",
    "hlt_degradation_strength",
    "label_names",
    "label_filter",
    "model_val_hlt_content_hash",
    "model_val_phi_hlt_content_hash",
    "model_val_phi_hlt_source_cache_hash",
    "model_val_jet_identity_hash",
)
CANONICAL_STATE_OPTIONAL_PROVENANCE_FIELDS = (
    "model_val_phi_offline_content_hash",
    "model_val_phi_offline_source_cache_hash",
)
CANONICAL_STATE_REPORT_TABLES = {
    "tagging_metrics": "tagging_metrics.csv",
    "single_model_tagging_metrics": "single_model_tagging_metrics.csv",
    "state_prediction_metrics": "state_prediction_metrics.csv",
    "per_token_family_residual_metrics": "per_token_family_residual_metrics.csv",
    "per_field_residual_metrics": "per_field_residual_metrics.csv",
    "per_class_metrics": "per_class_metrics.csv",
    "control_gaps": "control_gaps.csv",
    "oracle_gaps": "oracle_gaps.csv",
    "fusion_comparison": "fusion_comparison.csv",
    "seed_ensemble_comparison": "seed_ensemble_comparison.csv",
    "provenance": "provenance.csv",
}


@dataclass(frozen=True)
class CanonicalStateReportConfig:
    """Locations and strictness policy for the Step 9 report."""

    output_dir: str | Path
    run_root: str | Path
    run_ids: tuple[str, ...] = field(default_factory=lambda: tuple(CANONICAL_STATE_EXPECTED_RUN_IDS))
    baseline_run_id: str = "A0"
    require_all_runs: bool = True
    allow_missing_runs: bool = False
    confirm_final_test: bool = False

    def __post_init__(self) -> None:
        if not bool(self.confirm_final_test):
            raise ValueError("Canonical-state Step 9 report requires confirm_final_test=True")
        registry = canonical_state_variant_registry()
        run_ids = tuple(str(run_id) for run_id in self.run_ids)
        if not run_ids:
            run_ids = tuple(CANONICAL_STATE_EXPECTED_RUN_IDS)
        unknown = [run_id for run_id in run_ids if run_id not in registry]
        if unknown:
            raise ValueError(f"unknown canonical-state report run IDs: {unknown}")
        if len(set(run_ids)) != len(run_ids):
            raise ValueError("run_ids contains duplicates")
        baseline = str(self.baseline_run_id)
        if baseline not in registry:
            raise ValueError(f"unknown baseline run ID {baseline!r}")
        if baseline not in run_ids:
            run_ids = (baseline, *run_ids)
        object.__setattr__(self, "run_ids", run_ids)
        object.__setattr__(self, "baseline_run_id", baseline)
        object.__setattr__(self, "output_dir", str(self.output_dir))
        object.__setattr__(self, "run_root", str(self.run_root))


def _read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _csv_value(value: Any) -> Any:
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(_jsonable(value), sort_keys=True)
    return _jsonable(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(str(key))
                keys.append(str(key))
    if not keys:
        keys = ["available"]
        rows = [{"available": False}]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in keys})


def _float(value: Any) -> float | None:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    return output if math.isfinite(output) else None


def _nested_value(payload: Mapping[str, Any] | None, path: Sequence[str]) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _first_nested_value(payload: Mapping[str, Any], paths: Sequence[Sequence[str]]) -> Any:
    for path in paths:
        value = _nested_value(payload, path)
        if value not in (None, ""):
            return value
    return None


def _stable_value(value: Any) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(_jsonable(value), sort_keys=True)
    return str(value)


def _string_tuple(value: Any) -> tuple[str, ...] | None:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            decoded = [item.strip() for item in value.split(",") if item.strip()]
        value = decoded
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        return None
    return tuple(str(item) for item in value)


def _int_tuple(value: Any) -> tuple[int, ...] | None:
    sequence = _string_tuple(value)
    if sequence is None:
        return None
    try:
        return tuple(int(item) for item in sequence)
    except (TypeError, ValueError):
        return None


def _metrics_for_split(report: Mapping[str, Any], split: str) -> Mapping[str, Any] | None:
    candidates: tuple[Sequence[str], ...]
    if split == "model_val":
        candidates = (
            ("best_model_val_metrics",),
            ("model_val_metrics",),
            ("metrics", "model_val"),
            ("evaluation", "model_val", "metrics"),
        )
    elif split == "stack_val":
        candidates = (
            ("stack_val_metrics",),
            ("best_stack_val_metrics",),
            ("metrics", "stack_val"),
            ("evaluation", "stack_val", "metrics"),
        )
    elif split == "final_test":
        candidates = (
            ("final_test_metrics",),
            ("metrics", "final_test"),
            ("evaluation", "final_test", "metrics"),
        )
    else:
        candidates = ((f"{split}_metrics",), ("metrics", split))
    for path in candidates:
        value = _nested_value(report, path)
        if isinstance(value, Mapping):
            return value
    return None


def _metric(metrics: Mapping[str, Any] | None, name: str) -> float | None:
    if not isinstance(metrics, Mapping):
        return None
    aliases = {
        "cross_entropy": ("cross_entropy", "ce_loss", "loss"),
        "accuracy": ("accuracy", "acc"),
        "macro_auc": ("macro_auc", "auc_macro", "auc"),
    }.get(name, (name,))
    for key in aliases:
        value = _float(metrics.get(key))
        if value is not None:
            return value
    binary = metrics.get("binary_metrics")
    if isinstance(binary, Mapping):
        for key in aliases:
            value = _float(binary.get(key))
            if value is not None:
                return value
    return None


def _relative_error_reduction(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    baseline_error = 1.0 - float(baseline)
    if baseline_error <= 0.0:
        return None
    return (float(value) - float(baseline)) / baseline_error


def _state_metrics_for_split(report: Mapping[str, Any], split: str) -> Mapping[str, Any] | None:
    candidates = (
        ("state_prediction_metrics", split),
        ("state_metrics", split),
        (f"{split}_state_prediction_metrics",),
        (f"{split}_state_metrics",),
        ("metrics", split, "state_prediction"),
    )
    for path in candidates:
        value = _nested_value(report, path)
        if isinstance(value, Mapping):
            return value
    return None


def _state_loss_value(metrics: Mapping[str, Any] | None) -> float | None:
    if not isinstance(metrics, Mapping):
        return None
    for key in ("state_huber", "state_l1", "loss", "total_loss", "delta_l2", "mse"):
        value = _float(metrics.get(key))
        if value is not None:
            return value
    return None


def _provenance_value(report: Mapping[str, Any], field: str) -> Any:
    if field == "manifest_hash":
        return _first_nested_value(
            report,
            (
                ("manifest", "manifest_hash"),
                ("input_contract", "manifest_hash"),
                ("source_manifest_hash",),
                ("model_val_dataset", "source_manifest_hash"),
                ("val_dataset", "source_manifest_hash"),
            ),
        )
    if field == "hlt_profile":
        return _first_nested_value(
            report,
            (
                ("hlt_input_contract", "hlt_profile"),
                ("input_contract", "hlt_profile"),
                ("config", "hlt_profile"),
                ("model_val_dataset", "hlt_profile"),
                ("val_dataset", "hlt_profile"),
            ),
        )
    if field == "hlt_profile_version":
        return _first_nested_value(
            report,
            (
                ("hlt_input_contract", "hlt_profile_version"),
                ("input_contract", "hlt_profile_version"),
                ("config", "hlt_profile_version"),
                ("model_val_dataset", "hlt_profile_version"),
                ("val_dataset", "hlt_profile_version"),
            ),
        )
    if field == "hlt_degradation_strength":
        return _first_nested_value(
            report,
            (
                ("hlt_input_contract", "hlt_degradation_strength"),
                ("input_contract", "hlt_degradation_strength"),
                ("config", "hlt_degradation_strength"),
                ("model_val_dataset", "hlt_degradation_strength"),
                ("val_dataset", "hlt_degradation_strength"),
            ),
        )
    if field == "label_names":
        return _first_nested_value(report, (("label_names",), ("config", "label_names"), ("input_contract", "label_names")))
    if field == "label_filter":
        return _first_nested_value(report, (("label_filter",), ("config", "label_filter"), ("input_contract", "label_filter")))
    split_map = {
        "model_train": ("model_train_dataset", "train_dataset"),
        "model_val": ("model_val_dataset", "val_dataset"),
        "stack_train": ("stack_train_dataset",),
        "stack_val": ("stack_val_dataset",),
        "final_test": ("final_test_dataset",),
    }
    for split, dataset_keys in split_map.items():
        prefix = f"{split}_"
        if field.startswith(prefix):
            source_field = field.removeprefix(prefix)
            for key in dataset_keys:
                value = _nested_value(report, (key, source_field))
                if value not in (None, ""):
                    return value
            return None
    return _first_nested_value(report, ((field,), ("provenance", field)))


def _load_run_reports(config: CanonicalStateReportConfig, problems: list[str]) -> dict[str, Mapping[str, Any]]:
    root = Path(config.run_root)
    reports: dict[str, Mapping[str, Any]] = {}
    for run_id in config.run_ids:
        path = root / run_id / CANONICAL_STATE_RUN_REPORT_NAME
        payload = _read_json(path)
        if not isinstance(payload, Mapping):
            if config.require_all_runs and not config.allow_missing_runs:
                problems.append(f"missing required canonical-state run report for {run_id}: {path}")
            continue
        declared = payload.get("run_id") or payload.get("variant") or payload.get("canonical_state_run_id") or run_id
        if str(declared) != run_id:
            problems.append(f"run ID mismatch in {path}: expected {run_id}, found {declared}")
            continue
        if payload.get("ok") is False:
            problems.append(f"{run_id} run_report has ok=false")
        enriched = dict(payload)
        enriched["_run_report_path"] = str(path)
        reports[run_id] = enriched
    if config.baseline_run_id not in reports:
        problems.append(f"missing required baseline run {config.baseline_run_id}")
    return reports


def _check_report_provenance(reports: Mapping[str, Mapping[str, Any]], problems: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_id, report in reports.items():
        for field in (*CANONICAL_STATE_REQUIRED_PROVENANCE_FIELDS, *CANONICAL_STATE_OPTIONAL_PROVENANCE_FIELDS):
            value = _provenance_value(report, field)
            rows.append({"run_id": run_id, "field": field, "value": value})
            if field in CANONICAL_STATE_REQUIRED_PROVENANCE_FIELDS and value in (None, ""):
                problems.append(f"{run_id} is missing required provenance field {field}")
    for field in (*CANONICAL_STATE_REQUIRED_PROVENANCE_FIELDS, *CANONICAL_STATE_OPTIONAL_PROVENANCE_FIELDS):
        observed = {
            run_id: _stable_value(_provenance_value(report, field))
            for run_id, report in reports.items()
            if _stable_value(_provenance_value(report, field)) is not None
        }
        if len(set(observed.values())) > 1:
            details = ", ".join(f"{run_id}={value}" for run_id, value in sorted(observed.items()))
            problems.append(f"canonical-state reports disagree on {field}: {details}")
    for run_id, report in reports.items():
        manifest_hash = _stable_value(_provenance_value(report, "manifest_hash"))
        if manifest_hash is None:
            continue
        for split in ("model_train", "model_val", "stack_train", "stack_val", "final_test"):
            dataset_hash = _stable_value(_provenance_value(report, f"{split}_source_manifest_hash"))
            if dataset_hash is not None and dataset_hash != manifest_hash:
                problems.append(
                    f"{run_id} {split}.source_manifest_hash does not match manifest.manifest_hash: "
                    f"{dataset_hash} != {manifest_hash}"
                )
            hlt_hash = _stable_value(_provenance_value(report, f"{split}_hlt_content_hash"))
            phi_source_hash = _stable_value(_provenance_value(report, f"{split}_phi_hlt_source_cache_hash"))
            if hlt_hash is not None and phi_source_hash is not None and phi_source_hash != hlt_hash:
                problems.append(
                    f"{run_id} {split}.phi_hlt_source_cache_hash does not match "
                    f"{split}.hlt_content_hash: {phi_source_hash} != {hlt_hash}"
                )
        profile = _provenance_value(report, "hlt_profile")
        if profile not in (None, "") and str(profile) != CANONICAL_STATE_HLT_PROFILE:
            problems.append(f"{run_id} hlt_profile is {profile!r}, expected {CANONICAL_STATE_HLT_PROFILE!r}")
        profile_version = _provenance_value(report, "hlt_profile_version")
        if profile_version not in (None, "") and str(profile_version) != CANONICAL_STATE_HLT_PROFILE_VERSION:
            problems.append(
                f"{run_id} hlt_profile_version is {profile_version!r}, expected {CANONICAL_STATE_HLT_PROFILE_VERSION!r}"
            )
        strength = _float(_provenance_value(report, "hlt_degradation_strength"))
        if strength is not None and abs(strength - float(CANONICAL_STATE_HLT_DEGRADATION_STRENGTH)) > 1.0e-12:
            problems.append(
                f"{run_id} hlt_degradation_strength is {strength:g}, "
                f"expected {float(CANONICAL_STATE_HLT_DEGRADATION_STRENGTH):g}"
            )
        label_names = _string_tuple(_provenance_value(report, "label_names"))
        if label_names is not None and label_names != CANONICAL_STATE_LABEL_NAMES:
            problems.append(f"{run_id} label_names do not match canonical JetClass order")
        raw_label_filter = _provenance_value(report, "label_filter")
        label_filter = _int_tuple(raw_label_filter)
        if raw_label_filter not in (None, "") and label_filter is None:
            problems.append(f"{run_id} label_filter must be canonical integer class IDs, found {raw_label_filter!r}")
        elif label_filter is not None and label_filter != CANONICAL_STATE_LABEL_FILTER:
            problems.append(f"{run_id} label_filter is {label_filter}, expected {CANONICAL_STATE_LABEL_FILTER}")
        final_metrics = _metrics_for_split(report, "final_test")
        if _metrics_available(final_metrics):
            for field in (
                "final_test_hlt_content_hash",
                "final_test_phi_hlt_content_hash",
                "final_test_phi_hlt_source_cache_hash",
            ):
                value = _provenance_value(report, field)
                rows.append({"run_id": run_id, "field": field, "value": value})
                if value in (None, ""):
                    problems.append(f"{run_id} has final_test metrics but is missing required provenance field {field}")
    return rows


def _metrics_available(metrics: Mapping[str, Any] | None) -> bool:
    if not isinstance(metrics, Mapping):
        return False
    if metrics.get("available") is False:
        return False
    return _metric(metrics, "accuracy") is not None


def _final_test_metrics_present(report: Mapping[str, Any]) -> bool:
    return _metrics_available(_metrics_for_split(report, "final_test"))


def _check_final_test_policy(reports: Mapping[str, Mapping[str, Any]], config: CanonicalStateReportConfig, problems: list[str]) -> None:
    for run_id, report in reports.items():
        spec = canonical_state_variant_spec(run_id)
        if bool(report.get("final_test_uses_teacher_logits")) or bool(report.get("final_test_uses_offline_phi")):
            problems.append(f"{run_id} final_test used teacher/offline-oracle tensors")
        if bool(report.get("final_test_loaded_oracle_inputs")) or bool(report.get("final_test_loaded_teacher_caches")):
            problems.append(f"{run_id} final_test loaded privileged/oracle caches")
        final_metrics = _metrics_for_split(report, "final_test")
        has_final = _metrics_available(final_metrics)
        if isinstance(final_metrics, Mapping) and final_metrics.get("available") is False and spec.allows_primary_final_test():
            problems.append(f"{run_id} final_test metrics are marked unavailable")
        if config.confirm_final_test and spec.allows_primary_final_test() and not has_final:
            problems.append(f"{run_id} is primary but missing final_test metrics")
        if spec.final_test_policy in {FINAL_TEST_POLICY_MODEL_VAL_ONLY, FINAL_TEST_POLICY_STACK_ONLY, FINAL_TEST_POLICY_REPORT_ONLY} and has_final:
            problems.append(f"{run_id} has policy {spec.final_test_policy} but includes final_test metrics")


def _tagging_rows(
    reports: Mapping[str, Mapping[str, Any]],
    *,
    baseline_run_id: str,
    include_fusion: bool,
) -> list[dict[str, Any]]:
    baseline_report = reports.get(baseline_run_id)
    baseline_metrics = {
        split: _metrics_for_split(baseline_report, split) if isinstance(baseline_report, Mapping) else None
        for split in CANONICAL_STATE_REPORT_SPLITS
    }
    rows: list[dict[str, Any]] = []
    for run_id, report in reports.items():
        spec = canonical_state_variant_spec(run_id)
        if spec.model_kind == MODEL_KIND_STATE_PREDICTOR_ONLY:
            continue
        if spec.is_fusion != bool(include_fusion):
            continue
        for split in CANONICAL_STATE_REPORT_SPLITS:
            metrics = _metrics_for_split(report, split)
            if not isinstance(metrics, Mapping):
                continue
            accuracy = _metric(metrics, "accuracy")
            baseline_accuracy = _metric(baseline_metrics.get(split), "accuracy")
            rows.append(
                {
                    "run_id": run_id,
                    "tier": spec.tier,
                    "title": spec.title,
                    "split": split,
                    "model_kind": spec.model_kind,
                    "primary": bool(spec.primary),
                    "diagnostic_only": bool(spec.diagnostic_only),
                    "is_fusion": bool(spec.is_fusion),
                    "is_oracle": bool(spec.is_oracle),
                    "final_test_policy": spec.final_test_policy,
                    "accuracy": accuracy,
                    "cross_entropy": _metric(metrics, "cross_entropy"),
                    "macro_auc": _metric(metrics, "macro_auc"),
                    "n_jets": _metric(metrics, "n_jets"),
                    "accuracy_gain_vs_baseline": None if accuracy is None or baseline_accuracy is None else accuracy - baseline_accuracy,
                    "relative_error_reduction_vs_baseline": _relative_error_reduction(accuracy, baseline_accuracy),
                }
            )
    return rows


def _state_prediction_rows(reports: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_id, report in reports.items():
        spec = canonical_state_variant_spec(run_id)
        for split in ("model_val", "stack_val"):
            metrics = _state_metrics_for_split(report, split)
            if not isinstance(metrics, Mapping):
                continue
            rows.append(
                {
                    "run_id": run_id,
                    "tier": spec.tier,
                    "title": spec.title,
                    "split": split,
                    "predictor_variant": spec.predictor_variant,
                    "model_kind": spec.model_kind,
                    "state_loss": _state_loss_value(metrics),
                    "state_huber": _float(metrics.get("state_huber")),
                    "state_l1": _float(metrics.get("state_l1")),
                    "delta_l2": _float(metrics.get("delta_l2")),
                    "smoothness": _float(metrics.get("smoothness")),
                    "uncertainty_nll": _float(metrics.get("uncertainty_nll")),
                    "cosine": _float(metrics.get("cosine")),
                    "n_tokens": _float(metrics.get("n_tokens")),
                }
            )
    return rows


def _mapping_table_rows(
    reports: Mapping[str, Mapping[str, Any]],
    *,
    report_key: str,
    row_key: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_id, report in reports.items():
        source = report.get(report_key)
        if not isinstance(source, Mapping):
            continue
        for split, split_payload in source.items():
            if not isinstance(split_payload, Mapping):
                continue
            for name, metrics in split_payload.items():
                if not isinstance(metrics, Mapping):
                    continue
                row = {"run_id": run_id, "split": split, row_key: name}
                row.update({str(key): value for key, value in metrics.items()})
                rows.append(row)
    return rows


def _per_class_rows(reports: Mapping[str, Mapping[str, Any]], *, baseline_run_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    baseline_report = reports.get(baseline_run_id)
    baseline_by_split: dict[str, Mapping[str, Any]] = {}
    if isinstance(baseline_report, Mapping):
        for split in CANONICAL_STATE_REPORT_SPLITS:
            metrics = _metrics_for_split(baseline_report, split)
            per_class = metrics.get("per_class_accuracy") if isinstance(metrics, Mapping) else None
            if isinstance(per_class, Mapping):
                baseline_by_split[split] = per_class
    for run_id, report in reports.items():
        for split in CANONICAL_STATE_REPORT_SPLITS:
            metrics = _metrics_for_split(report, split)
            if not isinstance(metrics, Mapping):
                continue
            per_class = metrics.get("per_class_accuracy")
            if isinstance(per_class, Sequence) and not isinstance(per_class, (str, bytes, bytearray)):
                per_class = {str(index): value for index, value in enumerate(per_class)}
            if not isinstance(per_class, Mapping):
                continue
            for class_id, value in per_class.items():
                accuracy = _float(value)
                baseline_accuracy = _float(baseline_by_split.get(split, {}).get(str(class_id)))
                rows.append(
                    {
                        "run_id": run_id,
                        "split": split,
                        "class_id": str(class_id),
                        "accuracy": accuracy,
                        "accuracy_gain_vs_baseline": None if accuracy is None or baseline_accuracy is None else accuracy - baseline_accuracy,
                    }
                )
    return rows


def _comparison_value(run_id: str, report: Mapping[str, Any]) -> tuple[str, float | None, bool]:
    spec = canonical_state_variant_spec(run_id)
    if spec.model_kind == MODEL_KIND_STATE_PREDICTOR_ONLY:
        metrics = _state_metrics_for_split(report, "model_val")
        return ("model_val_state_loss", _state_loss_value(metrics), False)
    split = "model_val" if spec.final_test_policy == FINAL_TEST_POLICY_MODEL_VAL_ONLY else "final_test"
    metrics = _metrics_for_split(report, split) or _metrics_for_split(report, "model_val")
    return (f"{split}_accuracy", _metric(metrics, "accuracy"), True)


def _comparison_rows(reports: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    comparisons = [
        ("D2_vs_A0", "D2", "A0", "main clean residual-state versus HLT ParT"),
        ("D2_vs_A1", "D2", "A1", "main clean residual-state versus fine-tune-only"),
        ("D2_vs_A2", "D2", "A2", "main clean residual-state versus canonical feature MLP"),
        ("D2_vs_A3", "D2", "A3", "main clean residual-state versus seed ensemble"),
        ("D3_vs_A0", "D3", "A0", "residual-state repeat versus HLT ParT"),
        ("D3_vs_A1", "D3", "A1", "residual-state repeat versus fine-tune-only"),
        ("D3_vs_A2", "D3", "A2", "residual-state repeat versus canonical feature MLP"),
        ("D3_vs_A3", "D3", "A3", "residual-state repeat versus seed ensemble"),
        ("B0_vs_A0", "B0", "A0", "raw HLT state context versus HLT ParT"),
        ("D2_vs_B0", "D2", "B0", "residual state versus raw HLT state"),
        ("D3_vs_B0", "D3", "B0", "residual-state repeat versus raw HLT state"),
        ("D5_vs_D2", "D5", "D2", "CE-only D2 architecture versus supervised residual state"),
        ("C0_vs_C1", "C0", "C1", "geometry predictor versus no-geometry predictor"),
        ("C0_vs_C2", "C0", "C2", "geometry predictor versus DeepSets predictor"),
        ("C0_vs_C3", "C0", "C3", "geometry predictor versus state-only predictor"),
        ("E3_vs_E4", "E3", "E4", "from-scratch canonical-state versus from-scratch ParT"),
        ("F1_vs_Fseed", "F1", "Fseed", "core fusion versus seed ensemble"),
        ("F3_vs_Fseed", "F3", "Fseed", "state-token fusion prototype versus seed ensemble"),
        ("F4_vs_Fseed", "F4", "Fseed", "particle-view fusion prototype versus seed ensemble"),
    ]
    rows: list[dict[str, Any]] = []
    for comparison_id, candidate, reference, question in comparisons:
        left_report = reports.get(candidate)
        right_report = reports.get(reference)
        if not isinstance(left_report, Mapping) or not isinstance(right_report, Mapping):
            rows.append(
                {
                    "comparison": comparison_id,
                    "candidate": candidate,
                    "reference": reference,
                    "question": question,
                    "available": False,
                }
            )
            continue
        metric_name, left_value, higher_is_better = _comparison_value(candidate, left_report)
        _, right_value, _ = _comparison_value(reference, right_report)
        if left_value is None or right_value is None:
            improvement = None
        elif higher_is_better:
            improvement = left_value - right_value
        else:
            improvement = right_value - left_value
        rows.append(
            {
                "comparison": comparison_id,
                "candidate": candidate,
                "reference": reference,
                "question": question,
                "available": left_value is not None and right_value is not None,
                "metric": metric_name,
                "higher_is_better": higher_is_better,
                "candidate_value": left_value,
                "reference_value": right_value,
                "improvement": improvement,
            }
        )
    return rows


def _oracle_gap_rows(reports: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for oracle_run in ("G0", "G1"):
        oracle_report = reports.get(oracle_run)
        if not isinstance(oracle_report, Mapping):
            continue
        oracle_metrics = _metrics_for_split(oracle_report, "model_val")
        oracle_value = _metric(oracle_metrics, "accuracy")
        for reference in ("D2", "D3"):
            ref_report = reports.get(reference)
            if not isinstance(ref_report, Mapping):
                continue
            ref_metrics = _metrics_for_split(ref_report, "model_val")
            ref_value = _metric(ref_metrics, "accuracy")
            gap = None if oracle_value is None or ref_value is None else oracle_value - ref_value
            rows.append(
                {
                    "oracle_run_id": oracle_run,
                    "reference_run_id": reference,
                    "metric": "model_val_accuracy",
                    "higher_is_better": True,
                    "oracle_value": oracle_value,
                    "reference_value": ref_value,
                    "oracle_gap": gap,
                    "deployable": False,
                }
            )
    return rows


def _fusion_rows(tagging_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in tagging_rows if bool(row.get("is_fusion"))]


def _seed_rows(tagging_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in tagging_rows if row.get("run_id") in {"A3", "Fseed"}]


def _best_final_row(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    candidates = [
        row for row in rows
        if row.get("split") == "final_test" and _float(row.get("accuracy")) is not None
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda row: _float(row.get("accuracy")) or float("-inf"))


def _markdown_summary(report: Mapping[str, Any]) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), Mapping) else {}
    problems = report.get("problems") if isinstance(report.get("problems"), Sequence) else []
    lines = [
        "# Canonical Multi-Scale Jet State Report",
        "",
        f"- ok: `{bool(report.get('ok'))}`",
        f"- baseline: `{summary.get('baseline_run_id')}`",
        f"- best final-test run: `{summary.get('best_final_test_run_id')}`",
        f"- best final-test accuracy: `{summary.get('best_final_test_accuracy')}`",
        "",
        "## Problems",
        "",
    ]
    if problems:
        lines.extend(f"- {problem}" for problem in problems)
    else:
        lines.append("- none")
    lines.extend(["", "## Outputs", ""])
    outputs = report.get("outputs") if isinstance(report.get("outputs"), Mapping) else {}
    for key, value in outputs.items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    return "\n".join(lines)


def build_canonical_state_report(config: CanonicalStateReportConfig) -> dict[str, Any]:
    """Build and write the Step 9 canonical-state report."""

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    problems: list[str] = []
    reports = _load_run_reports(config, problems)
    _check_report_provenance(reports, problems)
    _check_final_test_policy(reports, config, problems)

    single_rows = _tagging_rows(reports, baseline_run_id=config.baseline_run_id, include_fusion=False)
    fusion_rows = _tagging_rows(reports, baseline_run_id=config.baseline_run_id, include_fusion=True)
    tagging_rows = [*single_rows, *fusion_rows]
    state_rows = _state_prediction_rows(reports)
    token_rows = _mapping_table_rows(
        reports,
        report_key="per_token_family_residual_metrics",
        row_key="token_family",
    )
    field_rows = _mapping_table_rows(
        reports,
        report_key="per_field_residual_metrics",
        row_key="field",
    )
    per_class_rows = _per_class_rows(reports, baseline_run_id=config.baseline_run_id)
    control_rows = _comparison_rows(reports)
    oracle_rows = _oracle_gap_rows(reports)
    fusion_comparison_rows = _fusion_rows(tagging_rows)
    seed_rows = _seed_rows(tagging_rows)
    provenance_rows = _check_report_provenance(reports, [])

    best_final = _best_final_row([row for row in tagging_rows if bool(row.get("primary"))])
    baseline_final = next(
        (
            row for row in tagging_rows
            if row.get("run_id") == config.baseline_run_id and row.get("split") == "final_test"
        ),
        {},
    )
    missing_primary = [
        run_id
        for run_id in canonical_state_primary_run_ids()
        if run_id in config.run_ids and run_id not in reports and not config.allow_missing_runs
    ]
    if missing_primary:
        problems.append(f"missing primary canonical-state runs: {' '.join(missing_primary)}")
    missing_dependencies: list[str] = []
    for run_id in reports:
        for dependency in canonical_state_required_dependencies(run_id):
            if dependency in config.run_ids and dependency not in reports:
                missing_dependencies.append(f"{run_id}->{dependency}")
    if missing_dependencies and not config.allow_missing_runs:
        problems.append(f"missing required dependencies: {' '.join(sorted(set(missing_dependencies)))}")

    outputs = {
        "report_json": str(output_dir / CANONICAL_STATE_REPORT_JSON),
        "report_md": str(output_dir / CANONICAL_STATE_REPORT_MD),
        **{key: str(output_dir / name) for key, name in CANONICAL_STATE_REPORT_TABLES.items()},
    }
    report = {
        "experiment_step": CANONICAL_STATE_REPORT_STEP,
        "report_contract": CANONICAL_STATE_REPORT_CONTRACT,
        "ok": not problems,
        "problems": problems,
        "config": asdict(config),
        "run_reports": {run_id: payload.get("_run_report_path") for run_id, payload in reports.items()},
        "summary": {
            "baseline_run_id": config.baseline_run_id,
            "baseline_final_test_accuracy": baseline_final.get("accuracy"),
            "best_final_test_run_id": None if best_final is None else best_final.get("run_id"),
            "best_final_test_accuracy": None if best_final is None else best_final.get("accuracy"),
            "best_final_test_accuracy_gain_vs_baseline": None if best_final is None else best_final.get("accuracy_gain_vs_baseline"),
            "n_loaded_reports": len(reports),
            "n_requested_reports": len(config.run_ids),
            "primary_run_ids": list(canonical_state_primary_run_ids()),
            "diagnostic_run_ids": list(canonical_state_diagnostic_run_ids()),
            "oracle_run_ids": list(canonical_state_oracle_run_ids()),
            "fusion_run_ids": list(canonical_state_fusion_run_ids()),
        },
        "tables": {
            "tagging_metrics": tagging_rows,
            "single_model_tagging_metrics": single_rows,
            "state_prediction_metrics": state_rows,
            "per_token_family_residual_metrics": token_rows,
            "per_field_residual_metrics": field_rows,
            "per_class_metrics": per_class_rows,
            "control_gaps": control_rows,
            "oracle_gaps": oracle_rows,
            "fusion_comparison": fusion_comparison_rows,
            "seed_ensemble_comparison": seed_rows,
            "provenance": provenance_rows,
        },
        "outputs": outputs,
    }
    table_payloads = report["tables"]
    for key, filename in CANONICAL_STATE_REPORT_TABLES.items():
        _write_csv(output_dir / filename, table_payloads[key])
    _write_json(output_dir / CANONICAL_STATE_REPORT_JSON, report)
    (output_dir / CANONICAL_STATE_REPORT_MD).write_text(_markdown_summary(report), encoding="utf-8")
    return _jsonable(report)


__all__ = [
    "CANONICAL_STATE_REPORT_CONTRACT",
    "CANONICAL_STATE_REPORT_JSON",
    "CANONICAL_STATE_REPORT_MD",
    "CANONICAL_STATE_REPORT_SPLITS",
    "CANONICAL_STATE_REPORT_STEP",
    "CANONICAL_STATE_REPORT_TABLES",
    "CanonicalStateReportConfig",
    "build_canonical_state_report",
]
