#!/usr/bin/env python3
"""Rank Step-4 view targets and emit non-gating scientific warnings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from teacher_logit_reco.local_particle_residual_field.particle_view import (  # noqa: E402
    PARTICLE_VIEW_TARGET_METRICS_CONTRACT,
    TargetCandidateMetrics,
    build_scientific_warnings,
    select_target_candidates,
    validate_content_hash,
    write_immutable_json,
    write_scientific_warnings,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", action="append", required=True)
    parser.add_argument("--canonical-target-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--warnings-jsonl", required=True)
    parser.add_argument("--graph-node", default="stage_b_target_selection")
    parser.add_argument("--source-commit", required=True)
    return parser


def _load(path: str) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_content_hash(
        payload, expected_contract=PARTICLE_VIEW_TARGET_METRICS_CONTRACT
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    rows = [_load(path) for path in args.metrics]
    selection = select_target_candidates(
        rows, canonical_target_id=args.canonical_target_id
    )
    write_immutable_json(args.output, selection)
    warnings = []
    for row in rows:
        metrics = TargetCandidateMetrics(
            run_id=row["run_id"],
            target_id=row["target_id"],
            bottleneck_width=row["bottleneck_width"],
            predicted_view_gain=row["predicted_view_gain"],
            oracle_gain=row["oracle_gain"],
            predicted_view_cross_entropy=row[
                "predicted_view_cross_entropy"
            ],
            zero_view_accuracy=row["zero_view_accuracy"],
            predicted_view_accuracy=row["predicted_view_accuracy"],
            oracle_accuracy=row["oracle_accuracy"],
            a0_accuracy=row["a0_accuracy"],
            target_registration_sha256=row[
                "target_registration_sha256"
            ],
            selection_status=row["selection_status"],
        )
        warnings.extend(
            build_scientific_warnings(
                metrics,
                graph_node=args.graph_node,
                configuration_id=row["target_id"],
                seed=101,
                split="model_val_select",
                supporting_metric_sha256=row["content_hash"],
                source_commit=args.source_commit,
            )
        )
    write_scientific_warnings(args.warnings_jsonl, warnings)
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "selection": str(Path(args.output).resolve()),
                "selection_sha256": selection["content_hash"],
                "forwarded_target_ids": selection[
                    "forwarded_target_ids"
                ],
                "warning_count": len(warnings),
                "warnings_stop_execution": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
