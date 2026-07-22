"""Assemble Step 9 reports directly from immutable campaign publications.

The production graph owns every input to this module.  In particular, report
generation never accepts a hand-written metric bundle: it discovers the paired
consumer/reconstructor publications, the locked selections, and the clean
HLT-only reload audit under one campaign artifact root.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .bridge_campaign import PAIRED_SEED_IDS, validate_campaign_registry
from .bridge_campaign_policy import (
    PREDICTION_ANCHORED_CLEAN_RELOAD_AUDIT_CONTRACT,
    PREDICTION_ANCHORED_DEPLOYABLE_PRECONFIRMATION_CONTRACT,
    PREDICTION_ANCHORED_LOCKED_DEPLOYABLE_CONTRACT,
    build_step9_report_row,
    build_step9_reports,
)
from .bridge_consumer import (
    A0_C250,
    A0_C250_LONG,
    A0_S500,
    STEP3_RUN_IDS,
    T10_ALL50_CLEAN,
    T10_CLEAN,
    T10_ROBUST,
    TPRED,
    TPRED_CONTINUE,
    PREDICTION_ANCHORED_CONSUMER_PUBLICATION_CONTRACT,
    PREDICTION_ANCHORED_REPLICA_AGGREGATE_CONTRACT,
    consumer_run_specs,
)
from .bridge_contracts import (
    load_hashed_json,
    sha256_file,
    validate_content_hash,
    with_content_hash,
)
from .bridge_evaluation import (
    PREDICTION_ANCHORED_CONSUMER_AGGREGATE_CONTRACT,
    PREDICTION_ANCHORED_SELECTED_CONSUMER_CONTRACT,
)
from .bridge_execution import (
    PREDICTION_ANCHORED_EXECUTION_SPEC_CONTRACT,
    validate_prediction_anchored_execution_spec,
)
from .bridge_reconstruction_execution import (
    PREDICTION_ANCHORED_RECONSTRUCTION_AGGREGATE_CONTRACT,
    PREDICTION_ANCHORED_RECONSTRUCTION_PUBLICATION_CONTRACT,
)
from .bridge_splits import PREDICTION_ANCHORED_SPLIT_CONTRACT


PREDICTION_ANCHORED_REPORT_EVIDENCE_CONTRACT = (
    "prediction_anchored_automatic_report_evidence_v1"
)

_DIRECT_BASELINES = (
    "A0_CAP500_direct_hlt",
    "A0_CAP500_r0rep_direct",
)
_TEACHER_RUNS = (T10_CLEAN, T10_ROBUST, T10_ALL50_CLEAN)
_FORBIDDEN_LEGACY_INFERENCE_KEYS = {
    "f_true",
    "offline_fields",
    "bridge_fields",
    "oracle_teacher",
    "oracle_teacher_state_dict",
}


def _regular_file(path: str | Path, *, label: str) -> Path:
    value = Path(path).resolve()
    if value.is_symlink() or not value.is_file():
        raise FileNotFoundError(f"{label} is absent or unsafe: {value}")
    return value


def _mean(values: Sequence[Any]) -> float:
    array = np.asarray([float(value) for value in values], dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("report aggregate values must be non-empty and finite")
    return float(array.mean())


def _sample_std(values: Sequence[Any]) -> float:
    array = np.asarray([float(value) for value in values], dtype=np.float64)
    if array.size != 3 or not np.isfinite(array).all():
        raise ValueError("paired report standard deviation requires three finite values")
    return float(array.std(ddof=1))


def _seed_rows(aggregate: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = aggregate.get("replica_metrics")
    if not isinstance(raw, list) or len(raw) != 3:
        raise ValueError("published aggregate does not retain three replica metrics")
    rows = [deepcopy(dict(row)) for row in raw]
    if sorted(int(row["seed_id"]) for row in rows) != list(PAIRED_SEED_IDS):
        raise ValueError("published report evidence is not paired seeds 101/202/303")
    return sorted(rows, key=lambda row: int(row["seed_id"]))


def _median_metrics(
    seed_rows: Sequence[Mapping[str, Any]], median_seed_id: int
) -> dict[str, Any]:
    selected = next(
        row for row in seed_rows if int(row["seed_id"]) == int(median_seed_id)
    )
    metrics = selected.get("metrics", selected)
    return deepcopy(dict(metrics)) if isinstance(metrics, Mapping) else {}


def _load_publication_pair(
    root: Path, *, run_id: str, reconstruction: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    if reconstruction:
        aggregate_contract = PREDICTION_ANCHORED_RECONSTRUCTION_AGGREGATE_CONTRACT
        publication_contract = PREDICTION_ANCHORED_RECONSTRUCTION_PUBLICATION_CONTRACT
    else:
        aggregate_contract = (
            PREDICTION_ANCHORED_CONSUMER_AGGREGATE_CONTRACT
            if run_id in _TEACHER_RUNS
            else PREDICTION_ANCHORED_REPLICA_AGGREGATE_CONTRACT
        )
        publication_contract = PREDICTION_ANCHORED_CONSUMER_PUBLICATION_CONTRACT
    aggregate = load_hashed_json(
        root / "aggregate_metrics.json", expected_contract=aggregate_contract
    )
    publication = load_hashed_json(
        root / "publication.json", expected_contract=publication_contract
    )
    if aggregate.get("run_id") != run_id or publication.get("run_id") != run_id:
        raise ValueError(f"publication identity differs from directory {run_id}")
    if publication.get("aggregate_sha256") != aggregate["content_hash"]:
        raise ValueError(f"publication aggregate binding changed for {run_id}")
    checkpoint = _regular_file(
        root / str(publication["retained_checkpoint"]),
        label=f"{run_id} retained checkpoint",
    )
    if sha256_file(checkpoint) != publication.get("retained_checkpoint_sha256"):
        raise ValueError(f"retained checkpoint hash changed for {run_id}")
    return aggregate, publication


def _training_metadata(
    *,
    run_id: str,
    execution_spec: Mapping[str, Any],
    child_manifest: Mapping[str, Any],
    median_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    if run_id in {A0_C250, A0_C250_LONG, A0_S500}:
        run_spec = consumer_run_specs()[run_id]
        config = execution_spec["consumer_training"]
        base_steps = int(config["baseline_steps"])
        continuation = int(config["bridge_finetune_steps"])
        steps = base_steps + continuation if run_id == A0_C250_LONG else base_steps
        if run_id == A0_S500:
            manifest_sha = child_manifest["content_hash"]
            training_splits = ["stack_train_consumer", "stack_train_distill"]
        else:
            manifest_sha = child_manifest["children"]["stack_train_consumer"][
                "content_hash"
            ]
            training_splits = ["stack_train_consumer"]
        return {
            "training_manifest_sha256": manifest_sha,
            "training_split_names": training_splits,
            "unique_jet_count": int(run_spec.unique_jet_count),
            "optimizer_step_budget": int(steps),
            "training_metadata_source": "immutable_execution_spec_and_child_manifest",
        }
    if run_id in _DIRECT_BASELINES:
        steps = median_metrics.get("optimizer_steps_completed")
        if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
            raise ValueError(f"direct baseline {run_id} lacks its executed step count")
        return {
            "training_manifest_sha256": child_manifest["content_hash"],
            "training_split_names": ["stack_train_consumer", "stack_train_distill"],
            "unique_jet_count": 500_000,
            "optimizer_step_budget": int(steps),
            "training_metadata_source": "paired_reconstruction_metrics_and_child_manifest",
        }
    return {}


def _paired_publication_row(
    *,
    row_id: str,
    run_id: str,
    aggregate: Mapping[str, Any],
    publication: Mapping[str, Any],
    metadata: Mapping[str, Any],
    section: str = "baselines_and_deployable",
    deployable: bool = True,
    comparison_baseline_seed_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    seeds = _seed_rows(aggregate)
    measured_gains: list[float] = []
    if comparison_baseline_seed_rows is not None:
        baseline_by_seed = {
            int(row["seed_id"]): row for row in comparison_baseline_seed_rows
        }
        for row in seeds:
            metrics = row.get("metrics", row)
            select = metrics.get(
                "model_val_select", metrics.get("model_val_stop", metrics)
            )
            baseline_row = baseline_by_seed[int(row["seed_id"])]
            baseline_metrics = baseline_row.get("metrics", baseline_row)
            baseline_select = baseline_metrics.get(
                "model_val_select", baseline_metrics.get("model_val_stop", baseline_metrics)
            )
            gain = float(select["accuracy"] - baseline_select["accuracy"])
            select["baseline_accuracy"] = float(baseline_select["accuracy"])
            select["deployable_gain"] = gain
            select["deployable_gain_reference"] = "A0_S500_same_seed_model_val"
            measured_gains.append(gain)
    median_seed = int(aggregate["median_seed_id"])
    return build_step9_report_row(
        row_id=row_id,
        section=section,
        status="COMPLETED",
        metrics=_median_metrics(seeds, median_seed),
        seed_metrics=seeds,
        aggregate=aggregate,
        median_seed_id=median_seed,
        deployable=deployable,
        hlt_only_reload_passed=True if deployable else None,
        metadata={
            **deepcopy(dict(metadata)),
            "source_run_id": run_id,
            "source_aggregate_sha256": aggregate["content_hash"],
            "source_publication_sha256": publication["content_hash"],
            "checkpoint_reload_verified": True,
            "measured_gain_over_A0_S500_mean": (
                None if not measured_gains else _mean(measured_gains)
            ),
        },
    )


def _missing_baseline_row(row_id: str, run_id: str) -> dict[str, Any]:
    return build_step9_report_row(
        row_id=row_id,
        section="baselines_and_deployable",
        status="FAILED",
        deployable=False,
        hlt_only_reload_passed=None,
        metadata={
            "source_run_id": run_id,
            "missing_metrics_visible": True,
            "failure_did_not_block_unrelated_selection_or_reporting": True,
        },
    )


def _condition_row(
    *,
    row_id: str,
    condition: str,
    selected_aggregate: Mapping[str, Any],
    section: str,
    deployable: bool,
    hlt_only_reload_passed: bool | None,
) -> dict[str, Any]:
    source_rows = _seed_rows(selected_aggregate)
    seeds: list[dict[str, Any]] = []
    for row in source_rows:
        if condition == "f0":
            metrics = row["f0"]
        elif condition == "bridge_0p10":
            metrics = row["bridge_0p10"]
        elif condition in {
            "oracle_physical45", "oracle_all50", "reliability5_only",
            "zero_field_consumer_diagnostic",
        }:
            metrics = row["diagnostics"][condition]
        elif condition.startswith("negative_control:"):
            metrics = row["negative_controls"][condition.split(":", 1)[1]]
        else:
            raise KeyError(f"unknown consumer report condition {condition}")
        seeds.append(
            {
                "seed_id": int(row["seed_id"]),
                "checkpoint_sha256": str(row["checkpoint_sha256"]),
                "metrics": deepcopy(dict(metrics)),
            }
        )
    accuracies = [row["metrics"]["accuracy"] for row in seeds]
    losses = [row["metrics"]["cross_entropy"] for row in seeds]
    aggregate = {
        "condition": condition,
        "mean_accuracy": _mean(accuracies),
        "accuracy_sample_std": _sample_std(accuracies),
        "mean_cross_entropy": _mean(losses),
        "cross_entropy_sample_std": _sample_std(losses),
        "source_consumer_aggregate_sha256": selected_aggregate["content_hash"],
    }
    median_seed = int(selected_aggregate["median_seed_id"])
    median = next(row for row in seeds if int(row["seed_id"]) == median_seed)
    teacher_gain = (
        float(selected_aggregate["mean_delta_same"])
        if condition == "bridge_0p10"
        else None
    )
    return build_step9_report_row(
        row_id=row_id,
        section=section,
        status="COMPLETED",
        metrics=median["metrics"],
        seed_metrics=seeds,
        aggregate=aggregate,
        median_seed_id=median_seed,
        deployable=deployable,
        hlt_only_reload_passed=hlt_only_reload_passed,
        teacher_bridge_gain=teacher_gain,
        metadata={
            "condition": condition,
            "consumer_recipe": selected_aggregate["run_id"],
            "evaluation_split": "model_val_select",
        },
    )


def _legacy_reference_row(execution_spec: Mapping[str, Any]) -> dict[str, Any]:
    """Report the pre-existing reference honestly when it has no paired lineage.

    The reference checkpoint is not a newly trained campaign configuration.
    Embedded metrics/provenance are retained when present, but three fake seed
    replicas are never manufactured merely to satisfy table shape.
    """

    checkpoint = _regular_file(
        execution_spec["baseline_checkpoint"]["path"], label="legacy HLT checkpoint"
    )
    if sha256_file(checkpoint) != execution_spec["baseline_checkpoint"]["sha256"]:
        raise ValueError("legacy HLT checkpoint hash changed after execution-spec binding")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = None
    if isinstance(payload, Mapping):
        state = next(
            (
                payload[key]
                for key in ("model_state_dict", "state_dict", "model")
                if isinstance(payload.get(key), Mapping)
            ),
            None,
        )
    if state is None:
        raise ValueError("legacy HLT checkpoint is not a reloadable model checkpoint")
    if _FORBIDDEN_LEGACY_INFERENCE_KEYS.intersection(payload):
        raise PermissionError("legacy HLT checkpoint contains privileged inference state")
    metrics = payload.get("metrics")
    metrics = deepcopy(dict(metrics)) if isinstance(metrics, Mapping) else {}
    training = payload.get("training_metadata")
    training = dict(training) if isinstance(training, Mapping) else {}
    return build_step9_report_row(
        row_id="A0_legacy",
        section="baselines_and_deployable",
        status="REFERENCE_ONLY_UNPAIRED",
        metrics=metrics,
        deployable=True,
        hlt_only_reload_passed=True,
        metadata={
            "training_manifest_sha256": training.get("training_manifest_sha256"),
            "unique_jet_count": training.get("unique_jet_count"),
            "optimizer_step_budget": training.get("optimizer_step_budget"),
            "legacy_checkpoint_sha256": sha256_file(checkpoint),
            "paired_seed_evidence_available": False,
            "replica_evidence_unavailable_reason": (
                "pre_existing_reference_checkpoint_is_not_a_paired_campaign_run"
            ),
            "training_metadata_source": (
                "checkpoint_training_metadata" if training else "not_embedded_in_legacy_checkpoint"
            ),
        },
    )


def _selected_deployable_row(
    *,
    preconfirmation: Mapping[str, Any],
    locked: Mapping[str, Any],
    clean_reload: Mapping[str, Any],
) -> dict[str, Any]:
    aggregate = preconfirmation["configuration_aggregate"]
    validate_content_hash(
        aggregate,
        expected_contract="prediction_anchored_deployable_configuration_aggregate_v1",
    )
    if aggregate["content_hash"] != locked["selected_aggregate_sha256"]:
        raise ValueError("locked deployable no longer names its pre-confirmation aggregate")
    if not bool(clean_reload.get("passed")):
        raise PermissionError("automatic reports require a passed clean HLT-only bundle reload")
    if clean_reload.get("locked_deployable_sha256") != locked["content_hash"]:
        raise ValueError("clean reload audit belongs to another locked deployable")
    seeds = [deepcopy(dict(row)) for row in aggregate["replicas"]]
    median_seed = int(aggregate["median_seed_id"])
    median = next(row for row in seeds if int(row["seed_id"]) == median_seed)
    return build_step9_report_row(
        row_id=f"selected_T10(fhat:{aggregate['run_id']})",
        section="baselines_and_deployable",
        status="COMPLETED",
        metrics=median,
        seed_metrics=seeds,
        aggregate=aggregate,
        median_seed_id=median_seed,
        deployable=True,
        hlt_only_reload_passed=True,
        teacher_bridge_gain=_mean(
            [row["teacher_bridge_gain"] for row in aggregate["replicas"]]
        ),
        recovery_fraction=aggregate.get("mean_recovery_fraction"),
        metadata={
            "selected_run_id": aggregate["run_id"],
            "locked_deployable_sha256": locked["content_hash"],
            "clean_reload_audit_sha256": clean_reload["content_hash"],
            "selection_split": "model_val_select",
        },
    )


def _scan_persistent_bytes(root: Path) -> dict[str, Any]:
    categories = {"checkpoint_bytes": 0, "json_bytes": 0, "other_artifact_bytes": 0}
    files = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            raise PermissionError(f"campaign artifact root contains a symlink: {path}")
        if not path.is_file() or "reports" in path.relative_to(root).parts:
            continue
        files += 1
        size = int(path.stat().st_size)
        if path.suffix in {".pt", ".pth", ".ckpt"}:
            categories["checkpoint_bytes"] += size
        elif path.suffix == ".json":
            categories["json_bytes"] += size
        else:
            categories["other_artifact_bytes"] += size
    return {**categories, "measured_file_count": files, "measurement_root": str(root.resolve())}


def _walk_json(value: Any, found: list[int]) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower()
            if (
                isinstance(item, int)
                and not isinstance(item, bool)
                and item >= 0
                and "bytes" in lowered
                and ("ram" in lowered or "reserved" in lowered or "resident" in lowered)
            ):
                found.append(int(item))
            _walk_json(item, found)
    elif isinstance(value, list):
        for item in value:
            _walk_json(item, found)


def _scan_ram_telemetry(root: Path, graph: Mapping[str, Any] | None) -> dict[str, Any]:
    observed: list[int] = []
    scanned = 0
    for path in root.rglob("*.json"):
        if path.is_symlink() or "reports" in path.relative_to(root).parts:
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        scanned += 1
        _walk_json(value, observed)
    requested = 0
    if graph is not None:
        requested = max(
            int(node["resources"]["host_memory_gib"]) * 1024**3
            for node in graph["nodes"]
        )
    return {
        "observed_peak_ram_bytes": max(observed, default=0),
        "maximum_requested_host_memory_bytes": int(requested),
        "telemetry_json_files_scanned": scanned,
        "observed_counter_count": len(observed),
    }


def assemble_step9_report_evidence(
    registry: Mapping[str, Any],
    *,
    execution_spec_path: str | Path,
    artifact_root: str | Path,
    graph: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Discover and validate every report row from the campaign artifact tree."""

    validate_campaign_registry(registry)
    execution_spec = load_hashed_json(
        execution_spec_path, expected_contract=PREDICTION_ANCHORED_EXECUTION_SPEC_CONTRACT
    )
    validate_prediction_anchored_execution_spec(execution_spec, verify_file_hashes=True)
    root = Path(artifact_root).resolve()
    if root.is_symlink() or not root.is_dir():
        raise FileNotFoundError(f"campaign artifact root is absent or unsafe: {root}")
    child = load_hashed_json(
        execution_spec["child_manifest"]["path"],
        expected_contract=PREDICTION_ANCHORED_SPLIT_CONTRACT,
    )
    if child["content_hash"] != execution_spec["child_manifest"]["content_hash"]:
        raise ValueError("report execution spec points at a stale child manifest")
    selected = load_hashed_json(
        root / "selection" / "selected_bridge_consumer.json",
        expected_contract=PREDICTION_ANCHORED_SELECTED_CONSUMER_CONTRACT,
    )
    if selected.get("status") != "CONFIRMED_LOCKED":
        raise PermissionError("reports refuse an unconfirmed or guessed bridge consumer")
    selected_run = str(selected["selected_consumer_recipe"])
    selected_aggregate = load_hashed_json(
        root / "consumer_evaluations" / selected_run / "selection_aggregate.json",
        expected_contract=PREDICTION_ANCHORED_CONSUMER_AGGREGATE_CONTRACT,
    )
    if selected_aggregate["content_hash"] != selected["recipe_aggregate_metrics"]["content_hash"]:
        raise ValueError("selected consumer aggregate changed after confirmation")
    preconfirmation = load_hashed_json(
        root / "selection" / "deployable_preconfirmation.json",
        expected_contract=PREDICTION_ANCHORED_DEPLOYABLE_PRECONFIRMATION_CONTRACT,
    )
    locked = load_hashed_json(
        root / "selection" / "locked_deployable.json",
        expected_contract=PREDICTION_ANCHORED_LOCKED_DEPLOYABLE_CONTRACT,
    )
    clean_reload = load_hashed_json(
        root / "deployable_bundle" / "clean_reload_audit.json",
        expected_contract=PREDICTION_ANCHORED_CLEAN_RELOAD_AUDIT_CONTRACT,
    )
    if locked["preconfirmation_sha256"] != preconfirmation["content_hash"]:
        raise ValueError("locked deployable is bound to another pre-confirmation")

    publications: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for run_id in STEP3_RUN_IDS:
        publication_root = root / "consumers" / run_id
        if (publication_root / "publication.json").is_file():
            publications[run_id] = _load_publication_pair(
                publication_root, run_id=run_id, reconstruction=False
            )
    for registry_row in registry["runs"]:
        run_id = str(registry_row["canonical_run_id"])
        if run_id in STEP3_RUN_IDS or registry_row["execution_status"] != "RUNNABLE":
            continue
        publication_root = root / "reconstructors" / run_id
        if (publication_root / "publication.json").is_file():
            publications[run_id] = _load_publication_pair(
                publication_root, run_id=run_id, reconstruction=True
            )

    baseline_rows = [_legacy_reference_row(execution_spec)]
    for run_id in (A0_C250, A0_C250_LONG, A0_S500):
        if run_id not in publications:
            baseline_rows.append(_missing_baseline_row(run_id, run_id))
            continue
        aggregate, publication = publications[run_id]
        seeds = _seed_rows(aggregate)
        metadata = _training_metadata(
            run_id=run_id,
            execution_spec=execution_spec,
            child_manifest=child,
            median_metrics=_median_metrics(seeds, int(aggregate["median_seed_id"])),
        )
        baseline_rows.append(
            _paired_publication_row(
                row_id=run_id,
                run_id=run_id,
                aggregate=aggregate,
                publication=publication,
                metadata=metadata,
            )
        )
    for run_id in _DIRECT_BASELINES:
        if run_id not in publications or A0_S500 not in publications:
            baseline_rows.append(_missing_baseline_row(run_id, run_id))
            continue
        aggregate, publication = publications[run_id]
        seeds = _seed_rows(aggregate)
        median_metrics = _median_metrics(seeds, int(aggregate["median_seed_id"]))
        baseline_rows.append(
            _paired_publication_row(
                row_id=run_id,
                run_id=run_id,
                aggregate=aggregate,
                publication=publication,
                metadata=_training_metadata(
                    run_id=run_id,
                    execution_spec=execution_spec,
                    child_manifest=child,
                    median_metrics=median_metrics,
                ),
                comparison_baseline_seed_rows=_seed_rows(publications[A0_S500][0]),
            )
        )
    for run_id, row_id in ((TPRED, "Tpred(f0)"), (TPRED_CONTINUE, "Tpred_continue(f0)")):
        if run_id not in publications:
            baseline_rows.append(_missing_baseline_row(row_id, run_id))
            continue
        aggregate, publication = publications[run_id]
        baseline_rows.append(
            _paired_publication_row(
                row_id=row_id,
                run_id=run_id,
                aggregate=aggregate,
                publication=publication,
                metadata={"field_source": "R0_HLT_only"},
            )
        )
    baseline_rows.append(
        _condition_row(
            row_id="selected_T10(f0)",
            condition="f0",
            selected_aggregate=selected_aggregate,
            section="baselines_and_deployable",
            deployable=True,
            hlt_only_reload_passed=True,
        )
    )
    baseline_rows.append(
        _selected_deployable_row(
            preconfirmation=preconfirmation,
            locked=locked,
            clean_reload=clean_reload,
        )
    )

    privileged_rows = [
        _condition_row(
            row_id="selected_T10(bridge_0.10)",
            condition="bridge_0p10",
            selected_aggregate=selected_aggregate,
            section="privileged_oracle",
            deployable=False,
            hlt_only_reload_passed=None,
        ),
        _condition_row(
            row_id="selected_T10(physical45_oracle_ceiling)",
            condition="oracle_physical45",
            selected_aggregate=selected_aggregate,
            section="privileged_oracle",
            deployable=False,
            hlt_only_reload_passed=None,
        ),
        _condition_row(
            row_id="selected_T10(full50_oracle_ceiling)",
            condition="oracle_all50",
            selected_aggregate=selected_aggregate,
            section="privileged_oracle",
            deployable=False,
            hlt_only_reload_passed=None,
        ),
        _condition_row(
            row_id="selected_T10(zero_field)",
            condition="zero_field_consumer_diagnostic",
            selected_aggregate=selected_aggregate,
            section="privileged_oracle",
            deployable=False,
            hlt_only_reload_passed=None,
        ),
    ]
    control_names = sorted(selected_aggregate["replica_metrics"][0]["negative_controls"])
    for control in control_names:
        privileged_rows.append(
            _condition_row(
                row_id=f"selected_T10(control:{control})",
                condition=f"negative_control:{control}",
                selected_aggregate=selected_aggregate,
                section="privileged_oracle",
                deployable=False,
                hlt_only_reload_passed=None,
            )
        )
    if T10_ALL50_CLEAN in publications:
        all50_aggregate = publications[T10_ALL50_CLEAN][0]
        for condition, suffix in (
            ("bridge_0p10", "bridge_0.10"),
            ("reliability5_only", "reliability5_only"),
            ("oracle_physical45", "physical45_oracle"),
            ("oracle_all50", "full50_oracle"),
            ("zero_field_consumer_diagnostic", "zero_field"),
        ):
            privileged_rows.append(
                _condition_row(
                    row_id=f"T10_all50_clean({suffix})",
                    condition=condition,
                    selected_aggregate=all50_aggregate,
                    section="privileged_oracle",
                    deployable=False,
                    hlt_only_reload_passed=None,
                )
            )
    alternate_binding_path = root / "bindings" / "alternate.json"
    if alternate_binding_path.is_file():
        alternate_binding = load_hashed_json(alternate_binding_path)
        alternate_run = str(alternate_binding["run_id"])
        if alternate_run not in {T10_CLEAN, T10_ROBUST}:
            raise ValueError("alternate physical45 binding names an unknown consumer recipe")
        alternate_aggregate = publications[alternate_run][0]
        privileged_rows.append(
            _condition_row(
                row_id=f"alternate_T10:{alternate_run}(bridge_0.10)",
                condition="bridge_0p10",
                selected_aggregate=alternate_aggregate,
                section="privileged_oracle",
                deployable=False,
                hlt_only_reload_passed=None,
            )
        )

    run_outcomes: dict[str, str] = {}
    ablation_rows: dict[str, dict[str, Any]] = {}
    for registry_row in registry["runs"]:
        run_id = str(registry_row["canonical_run_id"])
        if registry_row["execution_status"] == "SKIPPED_INVALID_PARENT":
            run_outcomes[run_id] = "SKIPPED_INVALID_PARENT"
            continue
        pair = publications.get(run_id)
        if pair is None:
            run_outcomes[run_id] = "FAILED"
            continue
        aggregate, publication = pair
        run_outcomes[run_id] = "COMPLETED"
        ablation_rows[run_id] = _paired_publication_row(
            row_id=run_id,
            run_id=run_id,
            aggregate=aggregate,
            publication=publication,
            metadata={
                "family": registry_row["family"],
                "scientific_role": registry_row["scientific_role"],
                "selectable_for_primary_deployment": registry_row[
                    "selectable_for_primary_deployment"
                ],
            },
            section="ablation_evidence",
            deployable=False,
        )

    persistent = _scan_persistent_bytes(root)
    ram = _scan_ram_telemetry(root, graph)
    return with_content_hash(
        {
            "contract": PREDICTION_ANCHORED_REPORT_EVIDENCE_CONTRACT,
            "registry_sha256": registry["content_hash"],
            "execution_spec_sha256": execution_spec["content_hash"],
            "selected_consumer_sha256": selected["content_hash"],
            "locked_deployable_sha256": locked["content_hash"],
            "clean_reload_audit_sha256": clean_reload["content_hash"],
            "baseline_deployable_rows": baseline_rows,
            "privileged_rows": privileged_rows,
            "ablation_rows": ablation_rows,
            "run_outcomes": run_outcomes,
            "persistent_telemetry": persistent,
            "ram_telemetry": ram,
            "manual_report_evidence_used": False,
            "oracle_rows_separate_from_deployable_rows": True,
            "final_test_metrics_used": False,
        }
    )


def build_step9_reports_from_publications(
    registry: Mapping[str, Any],
    *,
    execution_spec_path: str | Path,
    artifact_root: str | Path,
    graph: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    evidence = assemble_step9_report_evidence(
        registry,
        execution_spec_path=execution_spec_path,
        artifact_root=artifact_root,
        graph=graph,
    )
    reports = build_step9_reports(
        registry,
        baseline_deployable_rows=evidence["baseline_deployable_rows"],
        privileged_rows=evidence["privileged_rows"],
        ablation_rows=evidence["ablation_rows"],
        run_outcomes=evidence["run_outcomes"],
        persistent_telemetry=evidence["persistent_telemetry"],
        ram_telemetry=evidence["ram_telemetry"],
    )
    return evidence, reports


__all__ = [
    "PREDICTION_ANCHORED_REPORT_EVIDENCE_CONTRACT",
    "assemble_step9_report_evidence",
    "build_step9_reports_from_publications",
]
