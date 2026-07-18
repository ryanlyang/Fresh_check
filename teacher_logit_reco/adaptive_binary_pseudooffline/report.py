"""Fail-closed Step-11 campaign reporting and provenance comparison."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import (
    ABPH_HLT_DEGRADATION_STRENGTH,
    ABPH_HLT_PROFILE,
    ABPH_HLT_PROFILE_VERSION,
    canonical_hash,
)
from .convergence_schedule import ABPH_ACCELERATED_SCHEDULE_CONTRACT
from .diagnostics import ABPH_TAGGER_DIAGNOSTIC_CONTRACT
from .fusion import (
    ABPH_FUSION_FIT_SPLIT,
    ABPH_FUSION_SELECTION_SPLIT,
    load_frozen_fusion_artifact,
)
from .tagger import ABPH_PRIMARY_HIERARCHY_NAMES
from .variants import (
    ABPH_EXPECTED_VARIANT_NAMES,
    resolve_variant_config,
    variant_spec,
)


ABPH_CAMPAIGN_REPORT_CONTRACT = "adaptive_binary_pseudooffline_campaign_report_v1"
ABPH_REPORT_SPLITS: tuple[str, ...] = ("model_val", "stack_val", "final_test")
ABPH_POSTHOC_FUSION_VARIANTS: tuple[str, ...] = (
    "G2_kt_ca_logit_fusion",
    "G3_particle_and_logit_fusion",
    "G4_seed_ensemble_primary",
    "G5_best_complementary_ensemble",
)
ABPH_COMMON_SPLIT_PROVENANCE_FIELDS: tuple[str, ...] = (
    "source_manifest_hash",
    "jet_identity_hash",
    "label_hash",
    "class_mapping_hash",
    "hlt_content_hash",
    "hlt_profile",
    "hlt_profile_version",
    "hlt_degradation_strength",
    "hlt_params_hash",
)
ABPH_TARGET_PROVENANCE_FIELDS: tuple[str, ...] = (
    "offline_cache_content_hash",
    "hierarchy_target_content_hash",
    "hierarchy_target_schema_hash",
    "grouping_algorithm_hash",
    "root_ledger_schema_hash",
    "normalization_hash",
)
ABPH_ARTIFACT_PROVENANCE_FIELDS: tuple[str, ...] = (
    "resolved_variant_config_hash",
    "selected_checkpoint_hash",
    "source_git_commit",
    "source_status_hash",
)
ABPH_FINAL_TEST_ATTESTATIONS: Mapping[str, bool] = {
    "offline_inputs_loaded": False,
    "teacher_logits_loaded": False,
    "hypothesis_selection_used_offline_target": False,
    "fusion_fitted_on_final_test": False,
}
ABPH_REQUIRED_DIAGNOSTIC_VARIANTS: tuple[str, ...] = (
    "E5_kt32_mh4_dualcross",
    "E7_dual_hierarchy_dualcross",
)


def _read_json(path: Path) -> Mapping[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON mapping")
    return payload


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    materialized = [dict(row) for row in rows]
    fields = sorted({str(key) for row in materialized for key in row}) or ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in materialized:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, sort_keys=True)
                        if isinstance(value, (dict, list, tuple))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def _run_report_candidates(root: Path, variant_name: str) -> tuple[Path, ...]:
    spec = variant_spec(variant_name)
    return (
        root / "runs" / variant_name / "run_report.json",
        root / "runs" / spec.run_id / "run_report.json",
        root / spec.tier.lower() / variant_name / "run_report.json",
        root / spec.tier / variant_name / "run_report.json",
        root / variant_name / "run_report.json",
        root / spec.run_id / "run_report.json",
    )


def _find_run_report(root: Path, variant_name: str) -> tuple[Path | None, Mapping[str, Any] | None]:
    for path in _run_report_candidates(root, variant_name):
        payload = _read_json(path)
        if payload is not None:
            return path, payload
    return None, None


def _split_metrics(report: Mapping[str, Any], split: str) -> Mapping[str, Any] | None:
    candidates = (
        report.get("metrics", {}).get(split) if isinstance(report.get("metrics"), Mapping) else None,
        report.get(f"{split}_metrics"),
        report.get("evaluation", {}).get(split) if isinstance(report.get("evaluation"), Mapping) else None,
    )
    for value in candidates:
        if isinstance(value, Mapping):
            return value
    return None


def _metrics_available(metrics: Mapping[str, Any] | None) -> bool:
    return bool(metrics) and metrics.get("available") is not False and metrics.get("ok") is not False


def _metric(metrics: Mapping[str, Any] | None, *names: str) -> float | None:
    if not isinstance(metrics, Mapping):
        return None
    for name in names:
        value = metrics.get(name)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def _split_provenance(report: Mapping[str, Any], split: str) -> Mapping[str, Any]:
    provenance = report.get("provenance")
    if isinstance(provenance, Mapping):
        split_payload = provenance.get(split)
        if isinstance(split_payload, Mapping):
            return split_payload
    return {}


def _primary_dual_target_provenance_problems(
    provenance: Mapping[str, Any],
) -> list[str]:
    prefix = "F0 model_val dual-target provenance"
    problems: list[str] = []
    if provenance.get("dual_target_provenance") is not True:
        return [f"{prefix} is not attested"]
    branches = provenance.get("hierarchy_branches")
    if not isinstance(branches, Mapping):
        return [f"{prefix} lacks hierarchy_branches"]
    expected = set(ABPH_PRIMARY_HIERARCHY_NAMES)
    if set(branches) != expected:
        return [
            f"{prefix} must contain exact kT/C-A branches; got {sorted(branches)}"
        ]
    required_branch_fields = (
        "grouping",
        "hierarchy_target_content_hash",
        "grouping_algorithm_hash",
        "offline_cache_content_hash",
        "source_manifest_hash",
        "hlt_content_hash",
        "jet_identity_hash",
    )
    for hierarchy_name in ABPH_PRIMARY_HIERARCHY_NAMES:
        branch = branches.get(hierarchy_name)
        if not isinstance(branch, Mapping):
            problems.append(f"{prefix}/{hierarchy_name} is not a mapping")
            continue
        if branch.get("grouping") != hierarchy_name:
            problems.append(f"{prefix}/{hierarchy_name} grouping mismatch")
        for field in required_branch_fields:
            if branch.get(field) in (None, ""):
                problems.append(f"{prefix}/{hierarchy_name} lacks {field}")
    if problems:
        return problems
    common_fields = (
        "offline_cache_content_hash",
        "source_manifest_hash",
        "hlt_content_hash",
        "jet_identity_hash",
    )
    for field in common_fields:
        values = {
            str(branches[name][field]) for name in ABPH_PRIMARY_HIERARCHY_NAMES
        }
        if len(values) != 1:
            problems.append(f"{prefix} conflicts on {field}")
    target_hashes = {
        name: branches[name]["hierarchy_target_content_hash"]
        for name in ABPH_PRIMARY_HIERARCHY_NAMES
    }
    grouping_hashes = {
        name: branches[name]["grouping_algorithm_hash"]
        for name in ABPH_PRIMARY_HIERARCHY_NAMES
    }
    if provenance.get("hierarchy_target_content_hash") != canonical_hash(target_hashes):
        problems.append(f"{prefix} aggregate hierarchy target hash mismatch")
    if provenance.get("grouping_algorithm_hash") != canonical_hash(grouping_hashes):
        problems.append(f"{prefix} aggregate grouping hash mismatch")
    return problems


def _artifact_value(report: Mapping[str, Any], field: str) -> Any:
    if report.get(field) is not None:
        return report.get(field)
    provenance = report.get("provenance")
    if isinstance(provenance, Mapping):
        artifact = provenance.get("artifact")
        if isinstance(artifact, Mapping) and artifact.get(field) is not None:
            return artifact.get(field)
        if provenance.get(field) is not None:
            return provenance.get(field)
    if field == "selected_checkpoint_hash":
        for alias in ("best_model_val_checkpoint_sha256", "checkpoint_sha256", "source_checkpoint_hash"):
            if report.get(alias) is not None:
                return report.get(alias)
    return None


def _diagnostics(report: Mapping[str, Any], split: str) -> Mapping[str, Any]:
    metrics = _split_metrics(report, split)
    if isinstance(metrics, Mapping) and isinstance(metrics.get("diagnostics"), Mapping):
        return metrics["diagnostics"]
    diagnostics = report.get("diagnostics")
    if isinstance(diagnostics, Mapping):
        candidate = diagnostics.get(split)
        if isinstance(candidate, Mapping):
            return candidate
        return diagnostics
    return {}


def _root_provenance(report: Mapping[str, Any], split: str) -> Mapping[str, Any] | None:
    diagnostics = _diagnostics(report, split)
    root = diagnostics.get("root_provenance")
    if isinstance(root, Mapping):
        return root
    root = report.get("root_provenance")
    return root if isinstance(root, Mapping) else None


def _final_claim_defaults() -> tuple[str, ...]:
    names: list[str] = []
    for name in ABPH_EXPECTED_VARIANT_NAMES:
        spec = variant_spec(name)
        resolved = resolve_variant_config(name)
        if spec.primary and bool(resolved["evaluation"].get("final_test_eligible")):
            names.append(name)
    return tuple(names)


@dataclass(frozen=True)
class AdaptiveBinaryCampaignReportConfig:
    campaign_root: str | Path
    output_dir: str | Path | None = None
    required_variants: tuple[str, ...] = ABPH_EXPECTED_VARIANT_NAMES
    final_claim_variants: tuple[str, ...] = _final_claim_defaults()
    confirm_final_test: bool = False
    frozen_fusion_artifacts: Mapping[str, str | Path] | None = None

    def __post_init__(self) -> None:
        required = tuple(str(name) for name in self.required_variants)
        claims = tuple(str(name) for name in self.final_claim_variants)
        unknown = sorted((set(required) | set(claims)) - set(ABPH_EXPECTED_VARIANT_NAMES))
        if unknown:
            raise ValueError(f"campaign report contains unknown variants {unknown}")
        missing = [name for name in ABPH_EXPECTED_VARIANT_NAMES if name not in required]
        duplicates = len(required) != len(set(required))
        if missing or duplicates or len(required) != len(ABPH_EXPECTED_VARIANT_NAMES):
            raise ValueError(
                "full campaign report must check the complete frozen A0-G5 registry; "
                f"missing={missing}, duplicates={duplicates}"
            )
        if not set(claims).issubset(required):
            raise ValueError("final claim variants must be required campaign members")
        object.__setattr__(self, "required_variants", required)
        object.__setattr__(self, "final_claim_variants", claims)


def _fusion_path(config: AdaptiveBinaryCampaignReportConfig, variant_name: str) -> Path:
    supplied = dict(config.frozen_fusion_artifacts or {})
    if variant_name in supplied:
        return Path(supplied[variant_name])
    root = Path(config.campaign_root)
    return root / "fusion" / variant_name / "frozen_fusion.json"


def _audit_e7_root(
    variant_name: str,
    report: Mapping[str, Any],
    problems: list[str],
    rows: list[dict[str, Any]],
) -> None:
    root = _root_provenance(report, "model_val")
    if variant_name == "E7_dual_hierarchy_dualcross":
        if not isinstance(root, Mapping):
            problems.append("E7 model_val lacks shared-root provenance")
            return
        branch = root.get("branch_root_hashes")
        hashes = list(branch.values()) if isinstance(branch, Mapping) else []
        shared_hash = root.get("root_hash")
        ok = (
            root.get("shared_root") is True
            and int(root.get("root_hash_count", -1)) == 1
            and len(hashes) == 2
            and set(branch) == set(ABPH_PRIMARY_HIERARCHY_NAMES)
            and shared_hash is not None
            and all(value == shared_hash for value in hashes)
        )
        if not ok:
            problems.append("E7 branches do not prove exact one-root identity")
        rows.append(
            {
                "variant": variant_name,
                "shared_root": root.get("shared_root"),
                "root_hash": shared_hash,
                "root_hash_count": root.get("root_hash_count"),
                "branch_root_hashes": branch,
                "ok": ok,
            }
        )
    elif variant_name == "E11_independent_root_dual_hierarchy_diagnostic" and isinstance(root, Mapping):
        if root.get("shared_root") is not False or int(root.get("root_hash_count", -1)) != 2:
            problems.append("E11 does not attest two independent roots")


def _audit_fusion(
    config: AdaptiveBinaryCampaignReportConfig,
    variant_name: str,
    report: Mapping[str, Any],
    problems: list[str],
    rows: list[dict[str, Any]],
) -> None:
    path = _fusion_path(config, variant_name)
    try:
        artifact = load_frozen_fusion_artifact(path)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        problems.append(f"{variant_name} frozen fusion artifact is invalid: {exc}")
        return
    payload = artifact.to_dict()
    fusion_report = report.get("fusion")
    if not isinstance(fusion_report, Mapping):
        problems.append(f"{variant_name} run_report lacks frozen fusion metadata")
        return
    if tuple(fusion_report.get("members", ())) != artifact.members:
        problems.append(f"{variant_name} run membership differs from its frozen declaration")
    if fusion_report.get("membership_hash") != artifact.membership_hash:
        problems.append(f"{variant_name} run membership hash differs from frozen artifact")
    if fusion_report.get("artifact_hash") != payload["artifact_hash"]:
        problems.append(f"{variant_name} run references the wrong frozen fusion artifact")
    rows.append(
        {
            "variant": variant_name,
            "members": list(artifact.members),
            "membership_hash": artifact.membership_hash,
            "artifact_hash": payload["artifact_hash"],
            "fit_split": ABPH_FUSION_FIT_SPLIT,
            "selection_split": ABPH_FUSION_SELECTION_SPLIT,
        }
    )


def write_adaptive_binary_campaign_report(
    config: AdaptiveBinaryCampaignReportConfig,
) -> dict[str, Any]:
    root = Path(config.campaign_root)
    output_dir = Path(config.output_dir) if config.output_dir is not None else root / "report"
    problems: list[str] = []
    metric_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    confusion_matrices: dict[str, Any] = {}
    provenance_rows: list[dict[str, Any]] = []
    root_rows: list[dict[str, Any]] = []
    fusion_rows: list[dict[str, Any]] = []
    reports: dict[str, Mapping[str, Any]] = {}
    schedule_rows: list[dict[str, Any]] = []
    report_paths: dict[str, str] = {}
    common_values: dict[tuple[str, str], dict[str, Any]] = {}

    diagnostic_path = root / "diagnostics" / "tagger_use_report.json"
    diagnostic_report = _read_json(diagnostic_path)
    if diagnostic_report is None:
        problems.append("missing required non-selection tagger-use diagnostics")
    else:
        if diagnostic_report.get("contract") != ABPH_TAGGER_DIAGNOSTIC_CONTRACT:
            problems.append("tagger-use diagnostic contract mismatch")
        if diagnostic_report.get("ok") is not True:
            problems.append("tagger-use diagnostics have ok!=true")
        if diagnostic_report.get("selection_eligible") is not False:
            problems.append("tagger-use diagnostics must be selection-ineligible")
        if diagnostic_report.get("split") != "model_val":
            problems.append("tagger-use diagnostics must use model_val only")
        if diagnostic_report.get("final_test_loaded") is not False:
            problems.append("tagger-use diagnostics accessed final_test")
        variants = tuple(str(value) for value in diagnostic_report.get("variants", ()))
        if variants != ABPH_REQUIRED_DIAGNOSTIC_VARIANTS:
            problems.append(
                "tagger-use diagnostic membership differs from the frozen E5/E7 contract"
            )
        rows = diagnostic_report.get("rows")
        covered = {
            str(row.get("variant"))
            for row in rows
            if isinstance(row, Mapping)
        } if isinstance(rows, list) else set()
        if not set(ABPH_REQUIRED_DIAGNOSTIC_VARIANTS).issubset(covered):
            problems.append("tagger-use diagnostics contain no rows for every required model")

    for variant_name in config.required_variants:
        path, report = _find_run_report(root, variant_name)
        if report is None or path is None:
            problems.append(f"missing required run_report for {variant_name}")
            continue
        reports[variant_name] = report
        report_paths[variant_name] = str(path)
        if report.get("ok") is not True:
            problems.append(f"{variant_name} run_report has ok!=true")
        schedule = report.get("schedule")
        schedule_required = (
            variant_spec(variant_name).tier in {"B", "C", "D"}
            and variant_name
            not in {"B4_oracle_root_diagnostic", "D6_true_offline_particles"}
        )
        if schedule_required and not isinstance(schedule, Mapping):
            problems.append(f"{variant_name} lacks accelerated schedule provenance")
        if isinstance(schedule, Mapping):
            if schedule.get("contract") != ABPH_ACCELERATED_SCHEDULE_CONTRACT:
                problems.append(f"{variant_name} schedule contract mismatch")
            if schedule.get("policy_label") != "accelerated_screening_v1":
                problems.append(f"{variant_name} schedule policy is not accelerated_screening_v1")
            schedule_rows.append(
                {
                    "variant": variant_name,
                    "campaign_profile": schedule.get("campaign_profile"),
                    "schedule_truncated": bool(schedule.get("schedule_truncated")),
                    "negative_mechanism_conclusion_valid": bool(
                        schedule.get("negative_mechanism_conclusion_valid")
                    ),
                    "automatic_highdata_promotion_allowed": bool(
                        schedule.get("automatic_highdata_promotion_allowed")
                    ),
                    "stages": schedule.get("stages", {}),
                }
            )
        resolved = resolve_variant_config(variant_name)
        reported_name = report.get("variant_name")
        if reported_name is None and isinstance(report.get("variant"), Mapping):
            reported_name = report["variant"].get("name")
        if reported_name not in (None, variant_name):
            problems.append(f"{variant_name} run_report identifies {reported_name}")
        for field in ABPH_ARTIFACT_PROVENANCE_FIELDS:
            value = _artifact_value(report, field)
            if value in (None, ""):
                problems.append(f"{variant_name} lacks required artifact provenance {field}")
            elif field == "resolved_variant_config_hash" and value != resolved["resolved_config_hash"]:
                problems.append(f"{variant_name} resolved configuration hash mismatch")

        evaluation = resolved["evaluation"]
        model_val_diagnostics = _diagnostics(report, "model_val")
        if variant_name == "F0_ce_reco_primary":
            hierarchy_names = tuple(
                model_val_diagnostics.get(
                    "joint_reconstructor_hierarchy_names", ()
                )
            )
            if (
                model_val_diagnostics.get("dual_hierarchy_joint_training") is not True
                or hierarchy_names != ABPH_PRIMARY_HIERARCHY_NAMES
            ):
                problems.append(
                    "F0 must attest joint kT/C-A shared-root reconstructor training"
                )
            problems.extend(
                _primary_dual_target_provenance_problems(
                    _split_provenance(report, "model_val")
                )
            )
        if variant_name == "D6_true_offline_particles":
            if (
                model_val_diagnostics.get("actual_pseudo_branch_forward_pass") is not True
                or model_val_diagnostics.get("copied_A4_metrics") is not False
                or model_val_diagnostics.get("pure_offline_architecture_ceiling") is not False
                or model_val_diagnostics.get("retains_reconstructed_hierarchy_context") is not True
            ):
                problems.append(
                    "D6 must attest oracle particle-feature injection through the reconstructed pseudo branch"
                )
        capacity = report.get("diagnostics", {}).get("capacity", {})
        if not isinstance(capacity, Mapping):
            capacity = {}
        expected_metric_split = (
            "stack_val" if variant_name in ABPH_POSTHOC_FUSION_VARIANTS else "model_val"
        )
        metrics = _split_metrics(report, expected_metric_split)
        if not _metrics_available(metrics):
            problems.append(f"{variant_name} lacks available {expected_metric_split} metrics")
        for split in ABPH_REPORT_SPLITS:
            split_metrics = _split_metrics(report, split)
            if split_metrics is None:
                continue
            if not _metrics_available(split_metrics):
                problems.append(f"{variant_name}/{split} metrics are unavailable")
                continue
            if report.get("ok") is not True:
                problems.append(f"{variant_name}/{split} metrics come from a failed run")
            if split == "final_test":
                if not config.confirm_final_test:
                    problems.append(f"{variant_name} accessed final_test without report confirmation")
                if not bool(evaluation.get("final_test_eligible")):
                    problems.append(f"{variant_name} is not eligible for final-test claims")
            metric_rows.append(
                {
                    "variant": variant_name,
                    "run_id": variant_spec(variant_name).run_id,
                    "tier": variant_spec(variant_name).tier,
                    "split": split,
                    "accuracy": _metric(split_metrics, "accuracy"),
                    "loss": _metric(split_metrics, "loss", "cross_entropy"),
                    "macro_ovr_auc": _metric(split_metrics, "macro_ovr_auc", "macro_auc"),
                    "n_jets": _metric(split_metrics, "n_jets"),
                    "tagger_trainable_parameter_count": capacity.get(
                        "tagger_trainable_parameter_count",
                        capacity.get("trainable_parameter_count"),
                    ),
                    "reconstructor_trainable_parameter_count": capacity.get(
                        "reconstructor_trainable_parameter_count"
                    ),
                    "combined_trainable_parameter_count": capacity.get(
                        "combined_trainable_parameter_count"
                    ),
                }
            )
            per_class = split_metrics.get("per_class")
            if not isinstance(per_class, list):
                per_class = split_metrics.get("per_class_accuracy")
            if isinstance(per_class, list):
                per_class_rows.extend(
                    {
                        "variant": variant_name,
                        "split": split,
                        **dict(row),
                    }
                    for row in per_class
                    if isinstance(row, Mapping)
                )
            confusion = split_metrics.get("confusion_matrix")
            if isinstance(confusion, list):
                confusion_matrices[f"{variant_name}/{split}"] = confusion

        provenance_splits = [expected_metric_split]
        if _split_metrics(report, "final_test") is not None:
            provenance_splits.append("final_test")
        for split in provenance_splits:
            provenance = _split_provenance(report, split)
            for field in ABPH_COMMON_SPLIT_PROVENANCE_FIELDS:
                value = provenance.get(field)
                if value in (None, ""):
                    problems.append(f"{variant_name}/{split} lacks provenance {field}")
                else:
                    common_values.setdefault((split, field), {})[variant_name] = value
            if variant_spec(variant_name).tier in {"E", "F"}:
                if provenance.get("consumer_pseudo_schema_hash") in (None, ""):
                    problems.append(
                        f"{variant_name}/{split} lacks consumer-only pseudo schema hash"
                    )
                if provenance.get("consumer_only_pseudo_at_tagger_boundary") is not True:
                    problems.append(
                        f"{variant_name}/{split} does not attest consumer-only pseudo inputs"
                    )
                if report.get("storage_profile") == "streaming_30gb_v1":
                    execution = _diagnostics(report, "model_val").get(
                        "pseudo_execution"
                    )
                    if not isinstance(execution, Mapping):
                        problems.append(
                            f"{variant_name}/model_val lacks frozen pseudo RAM telemetry"
                        )
                    elif execution.get("pseudo_representations_written_persistently") is not False:
                        problems.append(
                            f"{variant_name}/model_val persisted pseudo representations"
                        )
                    elif execution.get("execution_mode") == "joint_differentiable":
                        pass
                    else:
                        for source_split in ("model_train", "model_val"):
                            row = execution.get(source_split)
                            if not isinstance(row, Mapping) or row.get(
                                "execution_mode"
                            ) not in {"full_rank_cache", "bounded_lru"}:
                                problems.append(
                                    f"{variant_name}/{source_split} lacks an approved RAM pseudo mode"
                                )
            if (
                bool(resolved["data"].get("requires_offline_targets"))
                and variant_spec(variant_name).tier != "A"
                and split != "final_test"
            ):
                for field in ABPH_TARGET_PROVENANCE_FIELDS:
                    if provenance.get(field) in (None, ""):
                        problems.append(f"{variant_name}/{split} lacks target provenance {field}")
            provenance_rows.append(
                {"variant": variant_name, "split": split, **dict(provenance)}
            )

        final_metrics = _split_metrics(report, "final_test")
        if _metrics_available(final_metrics):
            attestations = _diagnostics(report, "final_test")
            for field, expected in ABPH_FINAL_TEST_ATTESTATIONS.items():
                if attestations.get(field) is not expected:
                    problems.append(
                        f"{variant_name}/final_test must attest {field}={str(expected).lower()}"
                    )
        _audit_e7_root(variant_name, report, problems, root_rows)
        if variant_name in ABPH_POSTHOC_FUSION_VARIANTS:
            _audit_fusion(config, variant_name, report, problems, fusion_rows)

    for (split, field), values in common_values.items():
        unique = {json.dumps(value, sort_keys=True) for value in values.values()}
        if len(unique) > 1:
            problems.append(f"campaign provenance conflicts on {split}/{field}: {values}")

    if config.confirm_final_test:
        for variant_name in config.final_claim_variants:
            report = reports.get(variant_name)
            if report is None or report.get("ok") is not True:
                problems.append(f"final claim {variant_name} lacks a successful run_report")
                continue
            if not _metrics_available(_split_metrics(report, "final_test")):
                problems.append(f"final claim {variant_name} lacks available final_test metrics")

    baseline = next(
        (
            row
            for row in metric_rows
            if row["variant"] == "A0_hlt_part" and row["split"] in ("final_test", "model_val")
        ),
        None,
    )
    baseline_accuracy = None if baseline is None else baseline.get("accuracy")
    for row in metric_rows:
        value = row.get("accuracy")
        row["accuracy_gap_vs_A0"] = (
            None
            if value is None or baseline_accuracy is None
            else float(value) - float(baseline_accuracy)
        )

    outputs = {
        "metrics_csv": str(output_dir / "metrics.csv"),
        "per_class_metrics_csv": str(output_dir / "per_class_metrics.csv"),
        "confusion_matrices_json": str(output_dir / "confusion_matrices.json"),
        "provenance_csv": str(output_dir / "provenance.csv"),
        "root_identity_csv": str(output_dir / "root_identity.csv"),
        "fusion_membership_csv": str(output_dir / "fusion_membership.csv"),
        "final_report_json": str(output_dir / "final_report.json"),
    }
    report_payload = {
        "contract": ABPH_CAMPAIGN_REPORT_CONTRACT,
        "ok": not problems,
        "problems": problems,
        "campaign_root": str(root),
        "required_variants": list(config.required_variants),
        "checked_variant_count": len(reports),
        "all_tiers_checked": sorted({variant_spec(name).tier for name in reports}),
        "run_report_paths": report_paths,
        "hlt_contract": {
            "profile": ABPH_HLT_PROFILE,
            "profile_version": ABPH_HLT_PROFILE_VERSION,
            "degradation_strength": ABPH_HLT_DEGRADATION_STRENGTH,
        },
        "final_test_policy": {
            "confirmed": bool(config.confirm_final_test),
            "claim_variants": list(config.final_claim_variants),
            "attestations": dict(ABPH_FINAL_TEST_ATTESTATIONS),
        },
        "metrics": metric_rows,
        "per_class_metrics": per_class_rows,
        "confusion_matrices": confusion_matrices,
        "provenance": provenance_rows,
        "root_identity": root_rows,
        "fusion_membership": fusion_rows,
        "schedule_screening": {
            "policy_label": "accelerated_screening_v1",
            "runs": schedule_rows,
            "truncated_variants": [
                row["variant"] for row in schedule_rows if row["schedule_truncated"]
            ],
            "negative_mechanism_conclusion_valid": bool(
                schedule_rows
                and all(
                    row["negative_mechanism_conclusion_valid"]
                    for row in schedule_rows
                )
            ),
            "automatic_highdata_promotion_allowed": bool(
                schedule_rows
                and all(
                    row["automatic_highdata_promotion_allowed"]
                    for row in schedule_rows
                )
            ),
        },
        "tagger_use_diagnostics": {
            "path": str(diagnostic_path),
            "report": diagnostic_report,
        },
        "outputs": outputs,
    }
    report_payload["report_content_hash"] = canonical_hash(
        {key: value for key, value in report_payload.items() if key != "outputs"}
    )
    _write_csv(Path(outputs["metrics_csv"]), metric_rows)
    _write_csv(Path(outputs["per_class_metrics_csv"]), per_class_rows)
    _atomic_json(
        Path(outputs["confusion_matrices_json"]),
        {"matrices": confusion_matrices},
    )
    _write_csv(Path(outputs["provenance_csv"]), provenance_rows)
    _write_csv(Path(outputs["root_identity_csv"]), root_rows)
    _write_csv(Path(outputs["fusion_membership_csv"]), fusion_rows)
    _atomic_json(Path(outputs["final_report_json"]), report_payload)
    return report_payload


def require_successful_campaign_report(report: Mapping[str, Any]) -> None:
    if report.get("contract") != ABPH_CAMPAIGN_REPORT_CONTRACT:
        raise ValueError("ABPH campaign report contract mismatch")
    if report.get("ok") is not True:
        raise RuntimeError(
            "ABPH campaign report failed closed: " + "; ".join(str(row) for row in report.get("problems", ()))
        )


__all__ = [
    "ABPH_ARTIFACT_PROVENANCE_FIELDS",
    "ABPH_CAMPAIGN_REPORT_CONTRACT",
    "ABPH_COMMON_SPLIT_PROVENANCE_FIELDS",
    "ABPH_FINAL_TEST_ATTESTATIONS",
    "ABPH_POSTHOC_FUSION_VARIANTS",
    "ABPH_REPORT_SPLITS",
    "ABPH_TARGET_PROVENANCE_FIELDS",
    "AdaptiveBinaryCampaignReportConfig",
    "require_successful_campaign_report",
    "write_adaptive_binary_campaign_report",
]
