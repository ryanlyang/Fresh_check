"""Separated scientific reports and one-time HLT-only final-test execution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from jetclass_fresh.jetclass_data import LABEL_NAMES

from .contracts import (
    require_sha256,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from .deployment import (
    PARTICLE_VIEW_BUNDLE_EXPORT_CONTRACT,
    PARTICLE_VIEW_BUNDLE_INPUT_NAMES,
    PARTICLE_VIEW_FRESH_RELOAD_AUDIT_CONTRACT,
    load_exported_particle_view_bundle,
    validate_particle_view_bundle_export,
)
from .metrics import classification_metrics
from .selection import (
    PARTICLE_VIEW_SPLIT_AUTHORIZATION_CONTRACT,
    PARTICLE_VIEW_STACK_REPORT_CONTRACT,
    PARTICLE_VIEW_WINNER_SELECTION_CONTRACT,
)


PARTICLE_VIEW_REPORT_CONTRACT = "particle_view_separated_campaign_report_v1"
PARTICLE_VIEW_FINAL_TEST_PERMIT_CONTRACT = (
    "particle_view_final_test_permit_v1"
)
PARTICLE_VIEW_FINAL_TEST_RESULT_CONTRACT = (
    "particle_view_hlt_only_final_test_result_v1"
)

PARTICLE_VIEW_REPORT_SECTIONS = (
    "offline_oracle_diagnostics",
    "privileged_true_view_diagnostics",
    "frozen_consumer_hlt_deployable",
    "joint_hlt_deployable",
    "pre_stage_g_hlt_performance_controls",
    "fusion_ensemble_results",
)

_ORACLE_SECTIONS = {
    "offline_oracle_diagnostics",
    "privileged_true_view_diagnostics",
}
_DEPLOYABLE_SECTIONS = {
    "frozen_consumer_hlt_deployable",
    "joint_hlt_deployable",
    "pre_stage_g_hlt_performance_controls",
    "fusion_ensemble_results",
}
_FORBIDDEN_FINAL_BATCH_FRAGMENTS = (
    "offline",
    "true_view",
    "oracle",
    "teacher",
    "target_logit",
    "selected_view",
    "gview",
    "attention_map",
)


def _normalize_report_row(
    section: str, row: Mapping[str, Any]
) -> dict[str, Any]:
    required = {
        "row_id",
        "artifact_sha256",
        "split",
        "metrics",
        "requires_oracle",
        "deployable",
    }
    if set(row) != required:
        raise ValueError(f"{section} report row schema mismatch")
    if not isinstance(row["row_id"], str) or not row["row_id"]:
        raise ValueError("report row_id must be non-empty")
    require_sha256("artifact_sha256", row["artifact_sha256"])
    if not isinstance(row["metrics"], Mapping):
        raise ValueError("report row metrics must be an object")
    if not isinstance(row["requires_oracle"], bool) or not isinstance(
        row["deployable"], bool
    ):
        raise ValueError("report row oracle/deployable flags must be boolean")
    if section in _ORACLE_SECTIONS:
        if row["requires_oracle"] is not True or row["deployable"] is not False:
            raise ValueError("oracle diagnostics cannot be deployable rows")
        if row["split"] == "final_test":
            raise PermissionError("oracle diagnostics cannot access final_test")
    if section in _DEPLOYABLE_SECTIONS:
        if row["requires_oracle"] is not False or row["deployable"] is not True:
            raise ValueError("deployable report rows must be HLT-only")
    return dict(row)


def build_separated_campaign_report(
    *,
    sections: Mapping[str, Sequence[Mapping[str, Any]]],
    selection_sha256: str,
    stack_report_sha256: str,
    fairness_ledger_sha256: str,
    label_exposure_ledger_sha256: str,
    storage_reservation_sha256: str,
    lineage_graph_sha256: str,
    deployment_export_sha256: Sequence[str],
    aggregate_warning_summary_sha256: str,
    final_test_permit_sha256: str | None = None,
    final_test_result_sha256: Sequence[str] = (),
) -> dict[str, Any]:
    """Build a report that cannot mix oracle and deployable result rows."""

    if set(sections) != set(PARTICLE_VIEW_REPORT_SECTIONS):
        raise ValueError("campaign report section inventory mismatch")
    parents = {
        "selection_sha256": selection_sha256,
        "stack_report_sha256": stack_report_sha256,
        "fairness_ledger_sha256": fairness_ledger_sha256,
        "label_exposure_ledger_sha256": label_exposure_ledger_sha256,
        "storage_reservation_sha256": storage_reservation_sha256,
        "lineage_graph_sha256": lineage_graph_sha256,
        "aggregate_warning_summary_sha256": aggregate_warning_summary_sha256,
    }
    for name, value in parents.items():
        require_sha256(name, value)
    exports = sorted(
        {
            require_sha256("deployment_export_sha256", value)
            for value in deployment_export_sha256
        }
    )
    if not exports:
        raise ValueError("campaign report requires a deployment export")
    normalized = {
        section: sorted(
            [_normalize_report_row(section, row) for row in sections[section]],
            key=lambda row: row["row_id"],
        )
        for section in PARTICLE_VIEW_REPORT_SECTIONS
    }
    final_rows = [
        row
        for rows in normalized.values()
        for row in rows
        if row["split"] == "final_test"
    ]
    final_results = sorted(
        {
            require_sha256("final_test_result_sha256", value)
            for value in final_test_result_sha256
        }
    )
    if final_rows:
        if final_test_permit_sha256 is None or not final_results:
            raise ValueError(
                "final report rows require their permit and result artifacts"
            )
        require_sha256(
            "final_test_permit_sha256", final_test_permit_sha256
        )
    elif final_test_permit_sha256 is not None or final_results:
        raise ValueError("pre-final report cannot bind unused final-test artifacts")
    return with_content_hash(
        {
            "contract": PARTICLE_VIEW_REPORT_CONTRACT,
            **parents,
            "deployment_export_sha256": exports,
            "final_test_permit_sha256": final_test_permit_sha256,
            "final_test_result_sha256": final_results,
            "sections": normalized,
            "section_order": list(PARTICLE_VIEW_REPORT_SECTIONS),
            "oracle_and_deployable_rows_separated": True,
            "final_test_deployable_rows_hlt_only": True,
            "warnings_are_non_gating": True,
        }
    )


def build_final_test_permit(
    *,
    selection: Mapping[str, Any],
    split_authorization: Mapping[str, Any],
    stack_report: Mapping[str, Any],
    bundle_export_manifests: Sequence[str | Path],
    fresh_reload_audits: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind median winner exports to the already frozen sealed-split decision."""

    validate_content_hash(
        selection, expected_contract=PARTICLE_VIEW_WINNER_SELECTION_CONTRACT
    )
    validate_content_hash(
        split_authorization,
        expected_contract=PARTICLE_VIEW_SPLIT_AUTHORIZATION_CONTRACT,
    )
    validate_content_hash(
        stack_report, expected_contract=PARTICLE_VIEW_STACK_REPORT_CONTRACT
    )
    if split_authorization["selection_sha256"] != selection["content_hash"]:
        raise ValueError("split authorization belongs to a different selection")
    if (
        stack_report["selection_sha256"] != selection["content_hash"]
        or stack_report["authorization_sha256"]
        != split_authorization["content_hash"]
    ):
        raise ValueError("stack report belongs to a different selection")
    if stack_report.get("selection_changed") is not False:
        raise ValueError("stack validation cannot replace the selected winners")
    final_section = split_authorization["final_test"]
    if (
        final_section.get("hlt_only_required") is not True
        or final_section.get("stage_g_controls_forbidden") is not True
    ):
        raise ValueError("final-test sealed-split policy changed")
    baselines = []
    for row in final_section.get("authorized_hlt_baselines", []):
        source_sha = require_sha256(
            "authorized_hlt_baseline.bundle_sha256",
            row.get("bundle_sha256"),
        )
        if (
            int(row.get("seed")) not in {101, 202, 303}
            or row.get("role") != "matched_a0_final_baseline"
            or not isinstance(row.get("winner_families"), list)
        ):
            raise ValueError("final-test HLT baseline authorization changed")
        baselines.append(
            {
                "bundle_sha256": source_sha,
                "seed": int(row["seed"]),
                "role": "matched_a0_final_baseline",
                "winner_families": sorted(row["winner_families"]),
                "requires_oracle": False,
                "required_inputs": list(PARTICLE_VIEW_BUNDLE_INPUT_NAMES),
            }
        )
    fusion_recipes = sorted(
        {
            require_sha256("authorized_fusion_recipe_sha256", value)
            for value in final_section.get("authorized_fusion_recipes", [])
        }
    )
    authorized_source = {
        row["bundle_sha256"]: row
        for row in final_section["authorized_bundles"]
    }
    export_by_source: dict[str, tuple[dict[str, Any], Path]] = {}
    for raw_path in bundle_export_manifests:
        path = Path(raw_path)
        validation = validate_particle_view_bundle_export(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        source = validation["source_bundle_sha256"]
        if source in export_by_source:
            raise ValueError("duplicate deployment export for a selected bundle")
        export_by_source[source] = (payload, path)
    audit_by_export = {}
    for audit in fresh_reload_audits:
        validate_content_hash(
            audit, expected_contract=PARTICLE_VIEW_FRESH_RELOAD_AUDIT_CONTRACT
        )
        if audit.get("passed") is not True or audit.get(
            "only_hlt_inputs_visible"
        ) is not True:
            raise ValueError("fresh-process deployment audit did not pass")
        audit_by_export[audit["bundle_export_sha256"]] = dict(audit)
    winner_family_map = {
        "selected_privileged_scientific_model": "privileged_scientific",
        "selected_pre_stage_g_hlt_deployable_model": "pre_stage_g_deployable",
    }
    entries_by_source = {}
    for selection_family, deployment_family in winner_family_map.items():
        winner = selection[selection_family]
        source_sha = winner["representative_bundle_sha256"]
        if source_sha not in authorized_source:
            raise PermissionError("median winner is not sealed-split authorized")
        if source_sha not in export_by_source:
            raise FileNotFoundError("median winner has no deployment export")
        export, path = export_by_source[source_sha]
        if source_sha not in entries_by_source and (
            export["deployment_manifest"]["winner_family"]
            != deployment_family
        ):
            raise ValueError("deployment export winner family mismatch")
        audit = audit_by_export.get(export["content_hash"])
        if audit is None:
            raise ValueError("deployment export has no fresh-process reload audit")
        entry = entries_by_source.setdefault(
            source_sha,
            {
                "source_bundle_sha256": source_sha,
                "bundle_export_sha256": export["content_hash"],
                "bundle_manifest_name": path.name,
                "archive_sha256": export["archive_sha256"],
                "fresh_reload_audit_sha256": audit["content_hash"],
                "result_file": (
                    f"final_test_{export['content_hash'][:16]}.json"
                ),
                "seed": winner["representative_seed"],
                "winner_families": [],
                "requires_oracle": False,
                "required_inputs": list(PARTICLE_VIEW_BUNDLE_INPUT_NAMES),
            },
        )
        entry["winner_families"].append(selection_family)
    entries = []
    for source_sha in sorted(entries_by_source):
        entry = entries_by_source[source_sha]
        entry["winner_families"].sort()
        entries.append(entry)
    return with_content_hash(
        {
            "contract": PARTICLE_VIEW_FINAL_TEST_PERMIT_CONTRACT,
            "selection_sha256": selection["content_hash"],
            "split_authorization_sha256": split_authorization["content_hash"],
            "stack_report_sha256": stack_report["content_hash"],
            "final_test_split_sha256": final_section["split_sha256"],
            "authorized_exports": entries,
            "authorized_export_count": len(entries),
            "authorized_hlt_baselines": baselines,
            "authorized_fusion_recipe_sha256": fusion_recipes,
            "only_preselected_median_representatives": True,
            "stage_g_controls_authorized": False,
            "oracle_diagnostics_authorized": False,
            "hlt_only_required": True,
            "winner_selection_frozen": True,
        }
    )


def _validate_final_test_permit(payload: Mapping[str, Any]) -> None:
    validate_content_hash(
        payload, expected_contract=PARTICLE_VIEW_FINAL_TEST_PERMIT_CONTRACT
    )
    for name in (
        "selection_sha256",
        "split_authorization_sha256",
        "stack_report_sha256",
        "final_test_split_sha256",
    ):
        require_sha256(name, payload[name])
    if (
        payload.get("only_preselected_median_representatives") is not True
        or payload.get("stage_g_controls_authorized") is not False
        or payload.get("oracle_diagnostics_authorized") is not False
        or payload.get("hlt_only_required") is not True
    ):
        raise ValueError("final-test permit policy changed")
    for row in payload.get("authorized_hlt_baselines", []):
        require_sha256(
            "authorized_hlt_baseline.bundle_sha256", row["bundle_sha256"]
        )
        if (
            row["requires_oracle"] is not False
            or row["required_inputs"]
            != list(PARTICLE_VIEW_BUNDLE_INPUT_NAMES)
        ):
            raise ValueError("final-test baseline permit is not HLT-only")
    for value in payload.get("authorized_fusion_recipe_sha256", []):
        require_sha256("authorized_fusion_recipe_sha256", value)


def evaluate_authorized_final_test(
    *,
    bundle_manifest_path: str | Path,
    permit: Mapping[str, Any],
    loader,
    output_path: str | Path,
    class_names: Sequence[str] = LABEL_NAMES,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Evaluate one authorized export once; model execution sees only HLT."""

    _validate_final_test_permit(permit)
    manifest_path = Path(bundle_manifest_path)
    validation = validate_particle_view_bundle_export(manifest_path)
    entry = next(
        (
            row
            for row in permit["authorized_exports"]
            if row["bundle_export_sha256"] == validation["content_hash"]
        ),
        None,
    )
    if entry is None:
        raise PermissionError("deployment export is not final-test authorized")
    if entry["requires_oracle"] or entry["required_inputs"] != list(
        PARTICLE_VIEW_BUNDLE_INPUT_NAMES
    ):
        raise PermissionError("final-test permit is not HLT-only")
    destination = Path(output_path)
    if destination.name != entry["result_file"]:
        raise ValueError("final-test output path differs from its permit")
    if destination.exists():
        raise FileExistsError("final-test result already exists")
    module = load_exported_particle_view_bundle(
        manifest_path,
        expected_source_bundle_sha256=entry["source_bundle_sha256"],
        device=device,
    )
    logits = []
    labels = []
    with torch.no_grad():
        for raw in loader:
            if not isinstance(raw, Mapping):
                raise TypeError("final-test batches must be mappings")
            lowered = {str(name).lower() for name in raw}
            if any(
                fragment in name
                for name in lowered
                for fragment in _FORBIDDEN_FINAL_BATCH_FRAGMENTS
            ):
                raise PermissionError(
                    "final-test batch exposes an offline/oracle field"
                )
            required = set(PARTICLE_VIEW_BUNDLE_INPUT_NAMES) | {"labels"}
            if not required.issubset(raw):
                raise ValueError("final-test batch is missing HLT inputs or labels")
            inputs = tuple(
                raw[name].to(device)
                for name in PARTICLE_VIEW_BUNDLE_INPUT_NAMES
            )
            output = module(*inputs)
            if not torch.isfinite(output).all():
                raise FloatingPointError("final-test logits are non-finite")
            logits.append(output.detach().cpu().numpy())
            labels.append(raw["labels"].detach().cpu().numpy())
    if not logits:
        raise ValueError("final-test loader is empty")
    metrics = classification_metrics(
        np.concatenate(logits),
        np.concatenate(labels),
        split="final_test",
        class_names=class_names,
    )
    result = with_content_hash(
        {
            "contract": PARTICLE_VIEW_FINAL_TEST_RESULT_CONTRACT,
            "permit_sha256": permit["content_hash"],
            "bundle_export_sha256": validation["content_hash"],
            "source_bundle_sha256": entry["source_bundle_sha256"],
            "final_test_split_sha256": permit["final_test_split_sha256"],
            "winner_families": entry["winner_families"],
            "seed": entry["seed"],
            "metrics": metrics,
            "model_input_names": list(PARTICLE_VIEW_BUNDLE_INPUT_NAMES),
            "labels_used_for_evaluation_only": True,
            "offline_inputs_loaded": False,
            "selected_view_cache_loaded": False,
            "oracle_model_loaded": False,
            "selection_changed": False,
            "one_time_evaluation": True,
        }
    )
    write_immutable_json(destination, result)
    return result


__all__ = [
    "PARTICLE_VIEW_FINAL_TEST_PERMIT_CONTRACT",
    "PARTICLE_VIEW_FINAL_TEST_RESULT_CONTRACT",
    "PARTICLE_VIEW_REPORT_CONTRACT",
    "PARTICLE_VIEW_REPORT_SECTIONS",
    "build_final_test_permit",
    "build_separated_campaign_report",
    "evaluate_authorized_final_test",
]
