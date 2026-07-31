#!/usr/bin/env python3
"""Bind executable finalist controls after finalist locking."""

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
    load_and_validate_campaign,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    FINALIST_LOCK_CONTRACT,
    canonical_sha256,
    load_hashed_json,
    with_content_hash,
    write_immutable_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    root = args.campaign_root.resolve()
    campaign = load_and_validate_campaign(root, repo_root=REPO_ROOT)
    finalist = load_hashed_json(
        root / "selection" / "locked_hosd_finalists.json",
        expected_contract=FINALIST_LOCK_CONTRACT,
    )
    paths = {
        "capacity_controls": (
            root / "confirmation_500k" / "capacity_control_completion.json"
        ),
        "mechanism_control_summary": (
            root / "mechanism_controls" / "design_confirm_summary.json"
        ),
        "export_parity": root / "scale_up" / "export_audit.json",
        "latency_controls": (
            root / "scale_up" / "efficiency" / "completion.json"
        ),
        "matched_baselines": root / "confirmation_500k" / "summary.json",
    }
    controls = {}
    for key, path in paths.items():
        value = load_hashed_json(path)
        if value.get("source") != campaign["source"]:
            raise ValueError(f"finalist control source differs: {key}")
        controls[key] = value["content_hash"]
    confirmation_plan = load_hashed_json(
        root / "confirmation_500k" / "execution_plan.json"
    )
    final_control_rows = []
    for row in confirmation_plan["training_rows"]:
        if row["role"] not in {
            "H_BASE_LONG",
            "H_KD_O_BASE",
            "H_KD_O_FULLREL",
        }:
            continue
        result = load_hashed_json(
            root
            / "confirmation_500k"
            / "results"
            / f"{row['row_id']}.json"
        )
        final_control_rows.append(
            {
                "row_id": f"FINAL_CONTROL_{row['row_id']}",
                "graph_id": row["graph_id"],
                "seed": int(row["seed"]),
                "checkpoint_sha256": result["checkpoint_sha256"],
                "export_sha256": result["deployable_export_sha256"],
                "export_path": result["deployable_export_file"],
                "control_family": row["role"],
            }
        )
    capacity_plan = load_hashed_json(
        root / "confirmation_500k" / "capacity_execution_plan.json"
    )
    finalist_ids = set(finalist["unique_finalist_graph_ids"])
    for row in capacity_plan["rows"]:
        if row["parent_graph_id"] not in finalist_ids:
            continue
        result = load_hashed_json(
            root
            / "confirmation_500k"
            / "capacity_results"
            / f"{row['row_id']}.json"
        )
        final_control_rows.append(
            {
                "row_id": f"FINAL_CONTROL_{row['row_id']}",
                "graph_id": row["control_graph_id"],
                "seed": int(row["seed"]),
                "checkpoint_sha256": result["checkpoint_sha256"],
                "export_sha256": result["deployable_export_sha256"],
                "export_path": result["deployable_export_file"],
                "control_family": row["control_kind"],
                "matched_finalist_graph_id": row["parent_graph_id"],
            }
        )
    stage_e_plan = load_hashed_json(
        root / "job_ledgers" / "stage_e_execution_plan.json"
    )
    semantic_categories = {
        "AUX_ONLY": {"AUX_ONLY"},
        "DETACHED": {"DETACHED"},
        "SEMANTIC_LOSS_DISABLED": {
            "DISABLED_LOSS",
            "NO_SEMANTIC_LOSS",
        },
        "UNRESTRICTED": {"UNRESTRICTED", "UNRESTRICTED_MLP"},
        "SHUFFLED_PREDICTION": {"SHUFFLED_PREDICTION", "SHUFFLED"},
        "ZERO_FEEDBACK": {"ZERO", "ZERO_GATE"},
    }
    selected_feedback = load_hashed_json(
        root / "feedback" / "locked_feedback_choice.json"
    )["selected_feedback_definition"]
    if selected_feedback["parameterization"] == "HET":
        semantic_categories["MEAN_ONLY"] = {"MEAN_ONLY"}
    semantic_rows = []
    all_stage_e_rows = list(stage_e_plan["all_rows"])
    for family, accepted in semantic_categories.items():
        candidates = [
            row
            for row in all_stage_e_rows
            if (
                row.get("control") in accepted
                or (
                    family == "DETACHED"
                    and row.get("gradient_path") == "DETACHED"
                    and row.get("row_kind") == "SCIENTIFIC"
                )
            )
            and bool(row.get("deployable"))
        ]
        if family == "MEAN_ONLY":
            candidates = [
                row
                for row in candidates
                if row.get("parameterization") == "HET"
            ]
        if not candidates:
            raise ValueError(
                f"finalist semantic control is absent: {family}"
            )
        candidates.sort(
            key=lambda row: (
                row.get("target_id")
                != selected_feedback.get("target_id"),
                row.get("interface")
                != selected_feedback.get("interface"),
                row["row_id"],
            )
        )
        row = candidates[0]
        export_path = (
            root
            / "feedback"
            / row["row_id"]
            / "seed_101"
            / "deployable_control.pt"
        )
        manifest = load_hashed_json(export_path.with_suffix(".pt.json"))
        descriptor = manifest.get("descriptor", {})
        if (
            manifest.get("source") != campaign["source"]
            or descriptor.get("semantic_control_row", {}).get("row_id")
            != row["row_id"]
            or manifest.get("checkpoint_sha256") is None
            or not export_path.is_file()
            or hashlib.sha256(export_path.read_bytes()).hexdigest()
            != manifest.get("export_sha256")
        ):
            raise ValueError(
                f"semantic control export lineage differs: {family}"
            )
        semantic_rows.append(
            {
                "semantic_family": family,
                "semantic_control_row_id": row["row_id"],
                "checkpoint_sha256": manifest["checkpoint_sha256"],
                "export_sha256": manifest["content_hash"],
                "export_path": str(export_path.resolve()),
                "export_file_sha256": manifest["export_sha256"],
                "capacity_matched_to": (
                    "selected_stage_e_semantic_feedback_exemplar"
                ),
            }
        )
    semantic_bundle = with_content_hash(
        {
            "contract": "hosd_finalist_semantic_control_bundle_v1",
            "schema_version": 1,
            "source": dict(campaign["source"]),
            "stage_e_plan_sha256": stage_e_plan["content_hash"],
            "selected_feedback_definition": selected_feedback,
            "semantic_rows": semantic_rows,
            "semantic_families": sorted(semantic_categories),
            "coverage_exact": True,
            "selection_eligible": False,
            "reference_scope": (
                "shared_selected_stage_e_feedback_exemplar"
            ),
            "candidate_specific_capacity_claim": False,
        }
    )
    controls["semantic_controls"] = semantic_bundle["content_hash"]
    for finalist_graph_id in sorted(finalist_ids):
        for semantic in semantic_rows:
            final_control_rows.append(
                {
                    "row_id": (
                        "FINAL_SEMANTIC_"
                        + canonical_sha256(
                            [
                                finalist_graph_id,
                                semantic["semantic_family"],
                            ]
                        )[:16]
                    ),
                    "graph_id": semantic["semantic_control_row_id"],
                    "seed": 101,
                    "checkpoint_sha256": semantic[
                        "checkpoint_sha256"
                    ],
                    "export_sha256": semantic["export_sha256"],
                    "export_path": semantic["export_path"],
                    "control_family": (
                        "SEMANTIC_" + semantic["semantic_family"]
                    ),
                    "comparison_finalist_graph_id": finalist_graph_id,
                    "semantic_control_row_id": semantic[
                        "semantic_control_row_id"
                    ],
                    "semantic_reference_scope": (
                        "shared_selected_stage_e_feedback_exemplar"
                    ),
                }
            )
    artifact = with_content_hash(
        {
            "contract": "hosd_finalist_control_completion_v3",
            "schema_version": 3,
            "source": dict(campaign["source"]),
            "campaign_spec_sha256": campaign["content_hash"],
            "finalist_lock_sha256": finalist["content_hash"],
            "control_hashes": controls,
            "semantic_control_bundle": semantic_bundle,
            "final_control_rows": sorted(
                final_control_rows, key=lambda row: row["row_id"]
            ),
            "final_control_row_count": len(final_control_rows),
            "required_control_families": sorted(controls),
            "coverage_exact": True,
            "selection_eligible": False,
            "performance_based_termination": False,
        }
    )
    output = args.output or (
        root / "selection" / "finalist_control_completion.json"
    )
    publication = write_immutable_json(output, artifact)
    print(
        json.dumps(
            {
                "content_hash": artifact["content_hash"],
                "publication": publication["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
