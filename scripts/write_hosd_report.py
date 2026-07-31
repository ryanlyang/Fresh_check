#!/usr/bin/env python3
"""Write deterministic HOSD JSON and Markdown reports without manual edits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    build_hosd_report,
    load_and_validate_campaign,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    EFFICIENCY_PROFILE_CONTRACT,
    load_hashed_json,
    write_immutable_bytes,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.confirmation import (  # noqa: E402
    mean_log_selection_rejection,
)


def _efficiency_fields(root: Path, row, source) -> dict:
    path = (
        root
        / "scale_up"
        / "efficiency"
        / f"{row['graph_id']}__seed_{row['seed']}.json"
    )
    if not path.is_file():
        return {}
    profile = load_hashed_json(
        path, expected_contract=EFFICIENCY_PROFILE_CONTRACT
    )
    if (
        profile.get("source") != source
        or profile.get("graph_id") != row["graph_id"]
        or int(profile.get("seed", -1)) != int(row["seed"])
    ):
        raise ValueError("report efficiency profile lineage differs")
    return {
        "efficiency_profile_sha256": profile["content_hash"],
        "complete_trainable_parameters": profile[
            "complete_trainable_parameters"
        ],
        "deployed_trainable_parameters": profile[
            "deployed_trainable_parameters"
        ],
        "analytical_training_flops": profile[
            "analytical_training_flops"
        ],
        "analytical_inference_flops_by_batch": profile[
            "analytical_inference_flops_by_batch"
        ],
        "peak_gpu_memory_bytes": profile["peak_gpu_memory_bytes"],
        "target_cache_bytes_per_jet": profile[
            "target_cache_bytes_per_jet"
        ],
        "training_gpu_hours": profile["training_gpu_hours"],
        "latency": profile["latency"],
        "export_size_bytes": profile["export_size_bytes"],
        "target_head_removal_parameter_savings": profile[
            "target_head_removal_parameter_savings"
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--title", default="HLT Offline-Structure Distillation")
    parser.add_argument("--rows-json", required=True, type=Path)
    parser.add_argument("--artifact", action="append", default=[], metavar="NAME=SHA256")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign(args.campaign_root, repo_root=REPO_ROOT)
    hashes = {}
    for value in args.artifact:
        key, separator, digest = value.partition("=")
        if not separator or key in hashes:
            raise ValueError("artifacts must be unique NAME=SHA256")
        hashes[key] = digest
    rows_payload = json.loads(args.rows_json.read_text(encoding="utf-8"))
    rows = (
        rows_payload["rows"]
        if isinstance(rows_payload, dict) and "rows" in rows_payload
        else rows_payload
    )
    if rows and "classification_metrics" in rows[0]:
        h_base = [
            float(row["classification_metrics"]["macro_per_class_accuracy"])
            for row in rows
            if row["graph_id"] == "H_BASE"
        ]
        if not h_base:
            raise ValueError("final report lacks the locked H_BASE rows")
        h_base_mean = sum(h_base) / len(h_base)
        h_base_rejection = [
            mean_log_selection_rejection(row["classification_metrics"])
            for row in rows
            if row["graph_id"] == "H_BASE"
        ]
        h_base_rejection_mean = sum(h_base_rejection) / len(
            h_base_rejection
        )
        rows = [
            {
                **row,
                "split": "final_test",
                "balanced_accuracy": float(
                    row["classification_metrics"][
                        "macro_per_class_accuracy"
                    ]
                ),
                "accuracy_difference_vs_h_base": float(
                    row["classification_metrics"][
                        "macro_per_class_accuracy"
                    ]
                )
                - h_base_mean,
                "mean_log_rejection": float(
                    mean_log_selection_rejection(
                        row["classification_metrics"]
                    )
                ),
                "mean_log_rejection_difference_vs_h_base": float(
                    mean_log_selection_rejection(
                        row["classification_metrics"]
                    )
                )
                - h_base_rejection_mean,
                **_efficiency_fields(
                    args.campaign_root,
                    row,
                    campaign["source"],
                ),
            }
            for row in rows
        ]
    artifact, markdown = build_hosd_report(
        title=args.title,
        artifact_hashes=hashes,
        result_rows=rows,
        source=campaign["source"],
    )
    output_json = args.output_json or (
        args.campaign_root / "reports" / "final_report.json"
    )
    output_md = args.output_md or (
        args.campaign_root / "reports" / "final_report.md"
    )
    json_publication = write_immutable_json(output_json, artifact)
    md_publication = write_immutable_bytes(output_md, markdown.encode("utf-8"))
    print(json.dumps({"content_hash": artifact["content_hash"], "json": json_publication["status"], "markdown": md_publication["status"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
