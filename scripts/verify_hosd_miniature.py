#!/usr/bin/env python3
"""Semantically verify a real HOSD miniature and issue all nine receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    build_miniature_acceptance,
    build_miniature_check_receipt,
    load_and_validate_campaign,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    PRODUCTION_EXECUTION_PLAN_CONTRACT,
    RUNTIME_MANIFEST_CONTRACT,
    load_hashed_json,
    write_immutable_json,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hashed(root: Path, relative: str, source) -> tuple[dict, str]:
    path = root / relative
    artifact = load_hashed_json(path)
    if artifact.get("source") not in (None, source):
        raise ValueError(f"miniature artifact source differs: {relative}")
    return artifact, _sha(path)


def _scheduler(path: Path, *, plan, expected_nodes: set[str]) -> tuple[dict, str]:
    evidence = load_hashed_json(
        path, expected_contract="hosd_miniature_scheduler_evidence_v1"
    )
    if (
        evidence.get("research_compute") is not True
        or evidence.get("scheduler") != "slurm"
        or evidence.get("execution_plan_sha256") != plan["content_hash"]
        or evidence.get("source") != plan["source"]
    ):
        raise ValueError("miniature scheduler evidence lineage differs")
    states = evidence.get("terminal_state_by_node", {})
    if set(states) != expected_nodes or any(
        value != "COMPLETED" for value in states.values()
    ):
        raise ValueError("miniature did not complete every registered node")
    for key in ("target_shard", "training_row"):
        row = evidence.get("interrupt_resume", {}).get(key, {})
        if (
            row.get("interrupted_state")
            not in {"CANCELLED", "FAILED", "TIMEOUT", "PREEMPTED"}
            or row.get("resumed_state") != "COMPLETED"
            or str(row.get("interrupted_job_id", ""))
            == str(row.get("resumed_job_id", ""))
            or not str(row.get("interrupted_job_id", "")).isdigit()
            or not str(row.get("resumed_job_id", "")).isdigit()
            or row.get("same_source_and_coordinate") is not True
            or row.get("partial_artifact_rejected") is not True
            or row.get("resume_reused_only_valid_state") is not True
        ):
            raise ValueError(f"{key} interrupt/resume evidence differs")
    return evidence, _sha(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--scheduler-evidence", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    root = args.campaign_root.resolve()
    campaign = load_and_validate_campaign(root, repo_root=REPO_ROOT)
    plan = load_hashed_json(
        root / "job_ledgers" / "production_execution_plan.json",
        expected_contract=PRODUCTION_EXECUTION_PLAN_CONTRACT,
    )
    if (
        plan["profile"] != "miniature_test"
        or plan.get("source") != campaign["source"]
    ):
        raise ValueError("semantic verifier requires the active miniature plan")
    nodes = {str(row["node_id"]) for row in plan["nodes"]}
    scheduler, scheduler_sha = _scheduler(
        args.scheduler_evidence, plan=plan, expected_nodes=nodes
    )
    # Every declared output must exist. JSON outputs are content-hash and
    # source validated, so a scheduler success alone can never attest a run.
    declared_hashes = {}
    for node in plan["nodes"]:
        for relative in node["outputs"]:
            path = root / relative
            if not path.is_file() or path.is_symlink():
                raise FileNotFoundError(
                    f"registered miniature output is absent: {relative}"
                )
            if path.suffix == ".json":
                artifact = load_hashed_json(path)
                if artifact.get("source") not in (None, campaign["source"]):
                    raise ValueError(
                        f"registered miniature output source differs: {relative}"
                    )
            declared_hashes[relative] = _sha(path)

    parent, parent_sha = _hashed(
        root, "inputs/resolved_inherited_parent_lock.json", campaign["source"]
    )
    required_parents = {
        "hlt_v3_cache",
        "hlt_v3_profile",
        "hlt_v3_degradation_audit",
        "raw_input_schema",
    }
    parent_rows = {
        row["parent_id"]: row for row in parent["requirements"]
    }
    if not required_parents.issubset(parent_rows) or any(
        parent_rows[key].get("reusable") is not True
        for key in required_parents
    ):
        raise ValueError("miniature lacks real JetClass/HLT-v3 parents")

    target_audit, target_audit_sha = _hashed(
        root, "targets/target_audit.json", campaign["source"]
    )
    if target_audit.get("coverage_exact") is not True:
        raise ValueError("miniature target-family coverage is incomplete")

    baseline, baseline_sha = _hashed(
        root, "baselines/baseline_completion.json", campaign["source"]
    )
    probe, probe_sha = _hashed(
        root, "probes/probe_completion.json", campaign["source"]
    )
    auxiliary_rows = [
        _hashed(root, relative, campaign["source"])
        for relative in (
            "auxiliary/primary_scientific_completion.json",
            "auxiliary/relation_het_completion.json",
            "auxiliary/hlt_self_control_completion.json",
            "auxiliary/null_control_completion.json",
        )
    ]
    feedback_rows = [
        _hashed(root, relative, campaign["source"])
        for relative in (
            "feedback/scientific_row_completion.json",
            "feedback/mechanism_control_completion.json",
        )
    ]
    if any(
        artifact.get("performance_based_termination") is True
        for artifact in (
            baseline,
            probe,
            *(row[0] for row in auxiliary_rows),
            *(row[0] for row in feedback_rows),
        )
    ):
        raise ValueError("miniature training wave used performance termination")

    scale_shortlist, shortlist_sha = _hashed(
        root, "selection/locked_scale_shortlist.json", campaign["source"]
    )
    if (
        scale_shortlist.get("performance_can_disable_scale") is not False
        or scale_shortlist.get("all_negative_campaign_still_shortlisted")
        is not True
    ):
        raise ValueError("all-negative selector continuation is not attested")

    export_audit, export_sha = _hashed(
        root, "scale_up/export_audit.json", campaign["source"]
    )
    if (
        export_audit.get("all_shortlisted_exports_valid") is not True
        or export_audit.get("hlt_only") is not True
    ):
        raise ValueError("miniature HLT-only exports are not valid")

    selector, selector_sha = _hashed(
        root, "selection/stack_selector_trace.json", campaign["source"]
    )
    finalist, finalist_sha = _hashed(
        root, "selection/locked_hosd_finalists.json", campaign["source"]
    )
    execution, execution_sha = _hashed(
        root, "selection/final_test_execution_lock.json", campaign["source"]
    )
    final, final_sha = _hashed(
        root, "final_test/final_evaluation.json", campaign["source"]
    )
    if (
        selector.get("final_test_consumed") is not False
        or finalist.get("final_test_inference_authorized") is not False
        or execution.get("final_test_inference_authorized") is not True
        or final.get("execution_lock_sha256") != execution["content_hash"]
    ):
        raise ValueError("miniature stack/two-lock traversal differs")

    runtime = load_hashed_json(
        root / "registry" / "runtime_manifest.json",
        expected_contract=RUNTIME_MANIFEST_CONTRACT,
    )
    if (
        runtime.get("execution_ready") is not True
        or runtime.get("manual_command_argv_allowed") is not False
        or plan.get("manual_command_bundle_consumed") is not False
        or scheduler.get("manual_artifact_injection") is not False
    ):
        raise ValueError("miniature used or permits manual artifact injection")
    verifier_sha = _sha(Path(__file__))
    common = {
        "scheduler_evidence": scheduler_sha,
        "declared_output_set": hashlib.sha256(
            json.dumps(
                declared_hashes, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest(),
    }
    definitions = {
        "real_jetclass_hlt_v3": (
            {**common, "resolved_parents": parent_sha},
            {
                "research_compute_slurm": True,
                "real_jetclass_parent": True,
                "hlt_v3_parent": True,
            },
        ),
        "all_current_family_groups": (
            {**common, "target_audit": target_audit_sha},
            {"target_audit_exact": True, "all_nodes_completed": True},
        ),
        "baseline_probe_aux_feedback_trained": (
            {
                **common,
                "baseline": baseline_sha,
                "probe": probe_sha,
                **{
                    f"auxiliary_{index}": row[1]
                    for index, row in enumerate(auxiliary_rows)
                },
                **{
                    f"feedback_{index}": row[1]
                    for index, row in enumerate(feedback_rows)
                },
            },
            {"all_training_waves_complete": True},
        ),
        "all_negative_selector_continued": (
            {**common, "scale_shortlist": shortlist_sha},
            {"no_minimum_gain_gate": True, "scale_continued": True},
        ),
        "target_shard_interrupt_resume": (
            {**common, "scheduler_target_resume": scheduler_sha},
            {"interrupted": True, "resumed_exact_coordinate": True},
        ),
        "training_row_interrupt_resume": (
            {**common, "scheduler_training_resume": scheduler_sha},
            {"interrupted": True, "resumed_exact_coordinate": True},
        ),
        "hlt_only_export_validated": (
            {**common, "export_audit": export_sha},
            {"hlt_only": True, "all_exports_valid": True},
        ),
        "stack_and_two_locks_traversed": (
            {
                **common,
                "selector": selector_sha,
                "finalist_lock": finalist_sha,
                "execution_lock": execution_sha,
                "final_evaluation": final_sha,
            },
            {"label_free_stack": True, "two_locks": True, "final_once": True},
        ),
        "no_manual_artifact_injection": (
            {
                **common,
                "runtime_manifest": runtime["content_hash"],
                "execution_plan": plan["content_hash"],
            },
            {"registered_factories_only": True, "all_outputs_declared": True},
        ),
    }
    output = args.output_dir or root / "job_ledgers" / "miniature_checks"
    receipts = []
    for check_id, (evidence, predicates) in definitions.items():
        receipt = build_miniature_check_receipt(
            execution_plan=plan,
            check_id=check_id,
            evidence_hashes=evidence,
            verifier="hosd_miniature_semantic_verifier_v1",
            verifier_source_sha256=verifier_sha,
            verified_predicates=predicates,
            source=campaign["source"],
        )
        write_immutable_json(output / f"{check_id}.json", receipt)
        receipts.append(receipt)
    acceptance = build_miniature_acceptance(
        execution_plan=plan,
        check_receipts=receipts,
        source=campaign["source"],
    )
    write_immutable_json(
        root / "job_ledgers" / "miniature_acceptance.json", acceptance
    )
    print(
        json.dumps(
            {
                "acceptance_sha256": acceptance["content_hash"],
                "receipt_count": len(receipts),
                "passed": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
