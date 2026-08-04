#!/usr/bin/env python3
"""Aggregate OBASE7 and publish the five-bank supplemental comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    load_hashed_json,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.evaluation import (  # noqa: E402
    evaluate_classification,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.supplemental_offline_fusion import (  # noqa: E402
    BANK_DEFINITIONS,
    FUSION_VARIANTS,
    SEVEN_SEEDS,
    SUPPLEMENTAL_BANK_RESULT_CONTRACT,
    SUPPLEMENTAL_OBASE7_RESULT_CONTRACT,
    SUPPLEMENTAL_REPORT_CONTRACT,
    file_sha256,
    validate_supplemental_plan,
)


def _obase7(root: Path, plan: dict) -> dict:
    member_root = root / "runs/obase7/O_BASE"
    logits = []
    labels = None
    identities = None
    members = []
    for seed in SEVEN_SEEDS:
        current = member_root / f"member_seed_{seed}"
        registration = load_hashed_json(current / "training_registration.json")
        if (
            registration.get("control_id") != "O_BASE"
            or registration.get("lineage_hashes", {}).get(
                "supplemental_plan"
            )
            != plan["content_hash"]
            or registration.get("fixed_budget_completed") is not True
        ):
            raise ValueError("OBASE7 member registration lineage differs")
        prediction = current / "val_design_predictions.npz"
        with np.load(prediction, allow_pickle=False) as payload:
            current_logits = np.asarray(payload["logits"], dtype=np.float32)
            current_labels = np.asarray(payload["labels"], dtype=np.int64)
            current_identities = np.asarray(payload["identities"])
        if labels is None:
            labels = current_labels
            identities = current_identities
        elif not (
            np.array_equal(labels, current_labels)
            and np.array_equal(identities, current_identities)
        ):
            raise ValueError("OBASE7 member identities or labels differ")
        logits.append(current_logits)
        members.append(
            {
                "seed": seed,
                "training_registration_sha256": registration["content_hash"],
                "checkpoint_sha256": registration["checkpoint_sha256"],
                "prediction_sha256": file_sha256(prediction),
                "registration_source": registration.get("source"),
            }
        )
    scores = np.mean(np.stack(logits, axis=0), axis=0, dtype=np.float64).astype(
        np.float32
    )
    metrics = evaluate_classification(scores, labels, split="val_design")
    output = root / "runs/obase7/OBASE7_MEAN_LOGITS/ensemble"
    output.mkdir(parents=True, exist_ok=True)
    prediction = output / "val_design_predictions.npz"
    if not prediction.exists():
        with prediction.open("xb") as handle:
            np.savez_compressed(
                handle, logits=scores, labels=labels, identities=identities
            )
    result = bind_source(
        with_content_hash(
            {
                "contract": SUPPLEMENTAL_OBASE7_RESULT_CONTRACT,
                "schema_version": 2,
                "plan_sha256": plan["content_hash"],
                "control_id": "OBASE7_MEAN_LOGITS",
                "member_seeds": list(SEVEN_SEEDS),
                "members": members,
                "combiner": "arithmetic_mean_logits",
                "learned_combiner": False,
                "metrics": metrics,
                "val_design_predictions_sha256": file_sha256(prediction),
                "fixed_budget_completed": True,
                "performance_based_termination": False,
            }
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    write_immutable_json(output / "result.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ready-plan", required=True, type=Path)
    parser.add_argument("--late-plan", required=True, type=Path)
    parser.add_argument("--supplemental-root", required=True, type=Path)
    args = parser.parse_args(argv)
    ready_plan = load_hashed_json(args.ready_plan)
    late_plan = load_hashed_json(args.late_plan)
    validate_supplemental_plan(ready_plan)
    validate_supplemental_plan(late_plan)
    if (
        ready_plan.get("plan_role") != "ready"
        or late_plan.get("plan_role") != "late"
        or ready_plan.get("supplemental_id") != late_plan.get("supplemental_id")
        or ready_plan.get("parent_campaign_spec_sha256")
        != late_plan.get("parent_campaign_spec_sha256")
        or ready_plan.get("source") != late_plan.get("source")
    ):
        raise ValueError("supplemental ready/late plan lineage differs")
    root = args.supplemental_root.resolve()
    obase = _obase7(root, ready_plan)
    rows = []
    for bank_id in BANK_DEFINITIONS:
        plan = ready_plan if bank_id in ready_plan["banks"] else late_plan
        for variant in FUSION_VARIANTS:
            path = root / "runs/fusion_banks" / bank_id / variant / "result.json"
            result = load_hashed_json(
                path, expected_contract=SUPPLEMENTAL_BANK_RESULT_CONTRACT
            )
            if (
                result.get("plan_sha256") != plan["content_hash"]
                or result.get("bank_id") != bank_id
                or result.get("variant") != variant
                or result.get("fixed_budget_completed") is not True
                or result.get("source") != plan.get("source")
            ):
                raise ValueError("supplemental bank result lineage differs")
            rows.append(
                {
                    "candidate_id": f"{bank_id}/{variant}",
                    "bank_id": bank_id,
                    "variant": variant,
                    "result_sha256": result["content_hash"],
                    "accuracy": float(result["val_design_metrics"]["accuracy"]),
                    "cross_entropy": float(
                        result["val_design_metrics"]["cross_entropy"]
                    ),
                }
            )
    rows.append(
        {
            "candidate_id": "OBASE7/MEAN_LOGITS",
            "bank_id": "OBASE7",
            "variant": "MEAN_LOGITS",
            "result_sha256": obase["content_hash"],
            "accuracy": float(obase["metrics"]["accuracy"]),
            "cross_entropy": float(obase["metrics"]["cross_entropy"]),
        }
    )
    ranked = sorted(
        rows,
        key=lambda row: (
            -row["accuracy"], row["cross_entropy"], row["candidate_id"]
        ),
    )
    report = bind_source(
        with_content_hash(
            {
                "contract": SUPPLEMENTAL_REPORT_CONTRACT,
                "schema_version": 2,
                "ready_plan_sha256": ready_plan["content_hash"],
                "late_plan_sha256": late_plan["content_hash"],
                "comparison_split": "val_design",
                "rows": ranked,
                "best_available": ranked[0]["candidate_id"],
                "underperformance_blocks_completion": False,
                "final_test_accessed": False,
                "scientific_result_count": len(ranked),
            }
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    output = root / "reports/supplemental_offline_fusion_report.json"
    write_immutable_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
