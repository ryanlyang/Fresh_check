#!/usr/bin/env python3
"""Aggregate matched/hybrid specialist KD students and fixed controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    load_hashed_json,
    source_record,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.evaluation import (  # noqa: E402
    evaluate_classification,
)
from teacher_logit_reco.relation_expert_token_bridge.expert_training import (  # noqa: E402
    _publish_predictions,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.supplemental_offline_fusion import (  # noqa: E402
    SUPPLEMENTAL_BANK_RESULT_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.supplemental_specialist_kd import (  # noqa: E402
    SPECIALIST_CONDITIONS,
    SPECIALIST_EXPERTS,
    SPECIALIST_REPORT_CONTRACT,
    SPECIALIST_STUDENT_CONTRACT,
    file_sha256,
    pairwise_diversity,
    validate_specialist_kd_plan,
)


def _load_prediction(path: Path, expected_hash: str) -> dict[str, np.ndarray]:
    if file_sha256(path) != expected_hash:
        raise ValueError("specialist KD prediction bytes drifted")
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _control(plan: dict, name: str) -> dict:
    key = "common_kd4_result" if name == "COMMON_KD4" else "ce4_result"
    path = Path(plan["parent_artifacts"][key]["path"])
    if file_sha256(path) != plan["parent_artifacts"][key]["file_sha256"]:
        raise ValueError("specialist KD control result bytes drifted")
    result = load_hashed_json(path, expected_contract=SUPPLEMENTAL_BANK_RESULT_CONTRACT)
    if result.get("variant") != "MEAN_LOGITS":
        raise ValueError("specialist KD control is not mean-logit fusion")
    return {
        "result_sha256": result["content_hash"],
        "val_stop": result["val_stop_metrics"],
        "val_design": result["val_design_metrics"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--student-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    plan = load_hashed_json(args.plan)
    validate_specialist_kd_plan(plan, check_parent_bytes=False)
    source = source_snapshot(REPO_ROOT)
    if plan.get("source") != source_record(source):
        raise ValueError("specialist KD finalizer source differs from sealed plan")
    report_path = args.output_root / "specialist_kd_report.json"
    if report_path.is_file():
        report = load_hashed_json(report_path, expected_contract=SPECIALIST_REPORT_CONTRACT)
        if report.get("plan_sha256") != plan["content_hash"] or report.get("source") != plan.get("source"):
            raise ValueError("reusable specialist KD report differs")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    conditions = {}
    for condition in SPECIALIST_CONDITIONS:
        members = {}
        predictions_by_split = {"val_stop": {}, "val_design": {}}
        labels_by_split = {}
        identities_by_split = {}
        for expert in SPECIALIST_EXPERTS:
            result = load_hashed_json(
                args.student_root / condition / expert / "result.json",
                expected_contract=SPECIALIST_STUDENT_CONTRACT,
            )
            if result.get("plan_sha256") != plan["content_hash"] or result.get("coordinate") != {"condition": condition, "expert": expert, "seed": 101}:
                raise ValueError("specialist KD student lineage differs")
            if result.get("source") != plan.get("source"):
                raise ValueError("specialist KD student source differs")
            members[expert] = {
                "result_sha256": result["content_hash"],
                "selected_epoch": result["selected_epoch"],
                "val_stop": result["split_metrics"]["val_stop"],
                "val_design": result["split_metrics"]["val_design"],
            }
            for split in predictions_by_split:
                record = result["predictions"][split]
                arrays = _load_prediction(Path(record["path"]), record["file_sha256"])
                predictions_by_split[split][expert] = arrays["logits"]
                if split in labels_by_split:
                    if not np.array_equal(labels_by_split[split], arrays["labels"]) or not np.array_equal(identities_by_split[split], arrays["identities"]):
                        raise ValueError("specialist KD member event order differs")
                else:
                    labels_by_split[split] = arrays["labels"]
                    identities_by_split[split] = arrays["identities"]
        ensemble = {}
        diversity = {}
        for split in predictions_by_split:
            mean_logits = np.mean(
                np.stack([predictions_by_split[split][expert] for expert in SPECIALIST_EXPERTS]),
                axis=0,
            ).astype(np.float32)
            metrics = evaluate_classification(mean_logits, labels_by_split[split], split=split)
            path = args.output_root / condition / f"{split}_mean_logits.npz"
            publication = _publish_predictions(
                path, logits=mean_logits, labels=labels_by_split[split], identities=identities_by_split[split]
            )
            best_member = max(float(members[expert][split]["accuracy"]) for expert in SPECIALIST_EXPERTS)
            ensemble[split] = {
                "metrics": metrics,
                "prediction_path": str(path.resolve()),
                "prediction_file_sha256": publication["file_sha256"],
                "best_member_accuracy": best_member,
                "ensemble_bonus_accuracy": float(metrics["accuracy"]) - best_member,
            }
            rows = pairwise_diversity(predictions_by_split[split], labels_by_split[split])
            diversity[split] = {
                "pairs": rows,
                "means": {
                    key: float(np.mean([row[key] for row in rows]))
                    for key in (
                        "prediction_disagreement", "correctness_disagreement",
                        "double_fault", "centered_logit_correlation",
                    )
                },
            }
        conditions[condition] = {"members": members, "ensemble": ensemble, "diversity": diversity}
    controls = {name: _control(plan, name) for name in ("CE4", "COMMON_KD4")}
    for condition in conditions.values():
        accuracy = float(condition["ensemble"]["val_design"]["metrics"]["accuracy"])
        condition["val_design_accuracy_deltas"] = {
            name: accuracy - float(control["val_design"]["accuracy"])
            for name, control in controls.items()
        }
    report = bind_source(
        with_content_hash({
            "contract": SPECIALIST_REPORT_CONTRACT,
            "schema_version": 1,
            "plan_sha256": plan["content_hash"],
            "controls": controls,
            "conditions": conditions,
            "interpretation_policy": {
                "primary_comparison_split": "val_design",
                "scientific_underperformance_blocks_execution": False,
                "final_test_accessed": False,
                "descriptive_not_final_claim": True,
            },
            "all_eight_students_complete": True,
            "performance_based_termination": False,
            "final_test_accessed": False,
        }),
        source_snapshot=source,
    )
    write_immutable_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
