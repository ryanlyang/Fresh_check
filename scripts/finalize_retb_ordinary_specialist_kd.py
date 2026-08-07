#!/usr/bin/env python3
"""Compare ordinary-head and compact specialist-KD students."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

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
from teacher_logit_reco.relation_expert_token_bridge.supplemental_ordinary_specialist_kd import (  # noqa: E402,E501
    ORDINARY_SPECIALIST_REPORT_CONTRACT,
    ORDINARY_SPECIALIST_STUDENT_CONTRACT,
    validate_ordinary_specialist_kd_plan,
)
from teacher_logit_reco.relation_expert_token_bridge.supplemental_specialist_kd import (  # noqa: E402,E501
    SPECIALIST_CONDITIONS,
    SPECIALIST_EXPERTS,
    SPECIALIST_KD_PLAN_CONTRACT,
    SPECIALIST_REPORT_CONTRACT,
    SPECIALIST_STUDENT_CONTRACT,
    file_sha256,
    pairwise_diversity,
)


def _prediction(record: dict[str, Any]) -> dict[str, np.ndarray]:
    path = Path(record["path"])
    if file_sha256(path) != record["file_sha256"]:
        raise ValueError("ordinary specialist KD prediction bytes drifted")
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _compact_mean_logits(
    logits_by_expert: dict[str, np.ndarray],
) -> np.ndarray:
    """Reproduce the compact v1 finalizer's exact float32 reduction."""

    ordered = [
        np.asarray(logits_by_expert[expert]) for expert in SPECIALIST_EXPERTS
    ]
    if any(value.dtype != np.float32 for value in ordered):
        raise ValueError("compact specialist logits are not float32")
    # Do not pass dtype=float64 here.  The authenticated compact report was
    # produced by numpy's float32 accumulation, so even a more accurate
    # reduction would create different logits and cross entropy.
    return np.mean(np.stack(ordered), axis=0).astype(np.float32)


def _teacher_metrics(compact_plan: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    base_registration = load_hashed_json(
        Path(
            compact_plan["parent_artifacts"]["base4_teacher_registration"][
                "path"
            ]
        )
    )
    metrics["BASE4"] = {
        "val_stop": base_registration["selected_val_stop"],
        "val_design": None,
        "teacher_result_sha256": base_registration["content_hash"],
    }
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--student-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--corrective-source-recovery",
        action="store_true",
        help=(
            "Permit a finalizer-only source change while preserving the "
            "original plan and completed student lineage."
        ),
    )
    args = parser.parse_args()

    plan = load_hashed_json(args.plan)
    validate_ordinary_specialist_kd_plan(plan, check_parent_bytes=False)
    source = source_snapshot(REPO_ROOT)
    finalizer_source = source_record(source)
    source_matches_plan = plan.get("source") == finalizer_source
    if not source_matches_plan and not args.corrective_source_recovery:
        raise ValueError("ordinary specialist KD finalizer source differs")
    report_path = args.output_root / "ordinary_specialist_kd_report.json"
    if report_path.is_file():
        report = load_hashed_json(
            report_path, expected_contract=ORDINARY_SPECIALIST_REPORT_CONTRACT
        )
        finalization = report.get("source_finalization")
        if (
            report.get("plan_sha256") != plan["content_hash"]
            or not isinstance(finalization, dict)
            or finalization.get("training_and_plan_source")
            != plan.get("source")
            or finalization.get("report_finalizer_source")
            != report.get("source")
            or bool(finalization.get("source_matches_plan"))
            != (report.get("source") == plan.get("source"))
            or finalization.get("scientific_training_artifacts_modified")
            is not False
        ):
            raise ValueError("reusable ordinary specialist KD report differs")
        if report.get("source") != finalizer_source:
            raise ValueError(
                "reusable ordinary specialist KD report finalizer source differs"
            )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    compact_plan = load_hashed_json(
        Path(plan["parent_artifacts"]["compact_plan"]["path"]),
        expected_contract=SPECIALIST_KD_PLAN_CONTRACT,
    )
    compact_report = load_hashed_json(
        Path(plan["parent_artifacts"]["compact_report"]["path"]),
        expected_contract=SPECIALIST_REPORT_CONTRACT,
    )
    if (
        compact_plan["content_hash"] != plan["compact_plan_sha256"]
        or compact_report["content_hash"] != plan["compact_report_sha256"]
        or compact_report["plan_sha256"] != compact_plan["content_hash"]
    ):
        raise ValueError("ordinary specialist KD compact lineage differs")

    teacher_metrics = _teacher_metrics(compact_plan)
    for expert in ("PT", "TRACK", "REGION"):
        manifest = load_hashed_json(
            Path(plan["parent_artifacts"][f"teacher_{expert}"]["path"])
        )
        teacher_metrics[expert] = {
            "val_stop": manifest["split_metrics"]["val_stop"],
            "val_design": manifest["split_metrics"]["val_design"],
            "teacher_result_sha256": manifest["content_hash"],
        }

    ordinary_conditions: dict[str, Any] = {}
    decomposition: dict[str, Any] = {}
    for condition in SPECIALIST_CONDITIONS:
        members: dict[str, Any] = {}
        arrays_by_split: dict[str, dict[str, np.ndarray]] = {
            "val_stop": {},
            "val_design": {},
        }
        compact_arrays_by_split: dict[str, dict[str, np.ndarray]] = {
            "val_stop": {},
            "val_design": {},
        }
        labels_by_split: dict[str, np.ndarray] = {}
        identities_by_split: dict[str, np.ndarray] = {}
        member_decomposition: dict[str, Any] = {}
        for expert in SPECIALIST_EXPERTS:
            result = load_hashed_json(
                args.student_root / condition / expert / "result.json",
                expected_contract=ORDINARY_SPECIALIST_STUDENT_CONTRACT,
            )
            coordinate = {"condition": condition, "expert": expert, "seed": 101}
            compact_key = f"compact_{condition}_{expert}"
            if (
                result.get("plan_sha256") != plan["content_hash"]
                or result.get("coordinate") != coordinate
                or result.get("compact_counterpart_sha256")
                != plan["parent_artifacts"][compact_key]["content_hash"]
                or result.get("fixed_budget_completed") is not True
                or result.get("source") != plan.get("source")
            ):
                raise ValueError("ordinary specialist KD student lineage differs")
            members[expert] = {
                "result_sha256": result["content_hash"],
                "selected_epoch": result["selected_epoch"],
                "trainable_parameter_count": result["trainable_parameter_count"],
                "val_stop": result["split_metrics"]["val_stop"],
                "val_design": result["split_metrics"]["val_design"],
            }
            compact_member = compact_report["conditions"][condition]["members"][
                expert
            ]
            compact_result = load_hashed_json(
                Path(plan["parent_artifacts"][compact_key]["path"]),
                expected_contract=SPECIALIST_STUDENT_CONTRACT,
            )
            if (
                compact_result["content_hash"]
                != plan["parent_artifacts"][compact_key]["content_hash"]
                or compact_result.get("coordinate") != coordinate
            ):
                raise ValueError(
                    "ordinary specialist KD compact counterpart differs"
                )
            effects: dict[str, Any] = {}
            for split in ("val_stop", "val_design"):
                arrays = _prediction(result["predictions"][split])
                compact_arrays = _prediction(
                    compact_result["predictions"][split]
                )
                compact_prediction_record = plan["parent_artifacts"][
                    f"{compact_key}_{split}"
                ]
                if (
                    file_sha256(
                        Path(compact_result["predictions"][split]["path"])
                    )
                    != compact_prediction_record["file_sha256"]
                    or not np.array_equal(
                        arrays["identities"], compact_arrays["identities"]
                    )
                    or not np.array_equal(arrays["labels"], compact_arrays["labels"])
                ):
                    raise ValueError(
                        "ordinary and compact specialist event order differs"
                    )
                arrays_by_split[split][expert] = arrays["logits"]
                compact_arrays_by_split[split][expert] = compact_arrays["logits"]
                if split in labels_by_split:
                    if (
                        not np.array_equal(labels_by_split[split], arrays["labels"])
                        or not np.array_equal(
                            identities_by_split[split], arrays["identities"]
                        )
                    ):
                        raise ValueError(
                            "ordinary specialist KD member event order differs"
                        )
                else:
                    labels_by_split[split] = arrays["labels"]
                    identities_by_split[split] = arrays["identities"]
                ordinary_metrics = evaluate_classification(
                    arrays["logits"], arrays["labels"], split=split
                )
                ordinary_accuracy = float(ordinary_metrics["accuracy"])
                if (
                    ordinary_accuracy
                    != float(result["split_metrics"][split]["accuracy"])
                    or float(ordinary_metrics["cross_entropy"])
                    != float(result["split_metrics"][split]["cross_entropy"])
                ):
                    raise ValueError(
                        "ordinary specialist KD member metrics differ"
                    )
                compact_metrics = evaluate_classification(
                    compact_arrays["logits"], compact_arrays["labels"], split=split
                )
                compact_accuracy = float(compact_metrics["accuracy"])
                if (
                    compact_accuracy != float(compact_member[split]["accuracy"])
                    or compact_accuracy
                    != float(compact_result["split_metrics"][split]["accuracy"])
                    or float(compact_metrics["cross_entropy"])
                    != float(compact_member[split]["cross_entropy"])
                ):
                    raise ValueError(
                        "ordinary specialist KD compact metrics differ"
                    )
                teacher = teacher_metrics[expert][split]
                effects[split] = {
                    "ordinary_accuracy": ordinary_accuracy,
                    "compact_accuracy": compact_accuracy,
                    "compression_effect_compact_minus_ordinary_accuracy": (
                        compact_accuracy - ordinary_accuracy
                    ),
                    "ordinary_minus_CE_teacher_accuracy": (
                        None
                        if teacher is None
                        else ordinary_accuracy - float(teacher["accuracy"])
                    ),
                }
            member_decomposition[expert] = effects

        ensemble: dict[str, Any] = {}
        diversity: dict[str, Any] = {}
        for split in ("val_stop", "val_design"):
            logits = np.mean(
                np.stack(
                    [arrays_by_split[split][expert] for expert in SPECIALIST_EXPERTS]
                ),
                axis=0,
                dtype=np.float64,
            ).astype(np.float32)
            metrics = evaluate_classification(logits, labels_by_split[split], split=split)
            output = args.output_root / condition / f"{split}_mean_logits.npz"
            publication = _publish_predictions(
                output,
                logits=logits,
                labels=labels_by_split[split],
                identities=identities_by_split[split],
            )
            best = max(
                float(members[expert][split]["accuracy"])
                for expert in SPECIALIST_EXPERTS
            )
            ensemble[split] = {
                "metrics": metrics,
                "prediction_path": str(output.resolve()),
                "prediction_file_sha256": publication["file_sha256"],
                "best_member_accuracy": best,
                "ensemble_bonus_accuracy": float(metrics["accuracy"]) - best,
            }
            rows = pairwise_diversity(
                arrays_by_split[split], labels_by_split[split]
            )
            diversity[split] = {
                "pairs": rows,
                "means": {
                    key: float(np.mean([row[key] for row in rows]))
                    for key in (
                        "prediction_disagreement",
                        "correctness_disagreement",
                        "double_fault",
                        "centered_logit_correlation",
                    )
                },
            }
        controls = compact_report["controls"]
        ordinary_conditions[condition] = {
            "members": members,
            "ensemble": ensemble,
            "diversity": diversity,
            "val_design_accuracy_deltas": {
                name: float(ensemble["val_design"]["metrics"]["accuracy"])
                - float(control["val_design"]["accuracy"])
                for name, control in controls.items()
            },
        }
        compact_condition = compact_report["conditions"][condition]
        ensemble_effects = {}
        for split in ("val_stop", "val_design"):
            compact_logits = _compact_mean_logits(
                compact_arrays_by_split[split]
            )
            compact_ensemble = compact_condition["ensemble"][split]
            published_compact = _prediction(
                {
                    "path": compact_ensemble["prediction_path"],
                    "file_sha256": compact_ensemble[
                        "prediction_file_sha256"
                    ],
                }
            )
            if (
                not np.array_equal(
                    published_compact["identities"], identities_by_split[split]
                )
                or not np.array_equal(
                    published_compact["labels"], labels_by_split[split]
                )
                or not np.array_equal(
                    published_compact["logits"], compact_logits
                )
            ):
                raise ValueError(
                    "ordinary specialist KD compact ensemble bytes differ"
                )
            compact_metrics = evaluate_classification(
                published_compact["logits"],
                published_compact["labels"],
                split=split,
            )
            compact_recorded = compact_condition["ensemble"][split]["metrics"]
            if (
                float(compact_metrics["accuracy"])
                != float(compact_recorded["accuracy"])
                or float(compact_metrics["cross_entropy"])
                != float(compact_recorded["cross_entropy"])
            ):
                raise ValueError(
                    "ordinary specialist KD compact ensemble metrics differ"
                )
            compact_accuracy = float(compact_metrics["accuracy"])
            ordinary_accuracy = float(ensemble[split]["metrics"]["accuracy"])
            ensemble_effects[split] = {
                "ordinary_accuracy": ordinary_accuracy,
                "compact_accuracy": compact_accuracy,
                "compression_effect_compact_minus_ordinary_accuracy": (
                    compact_accuracy - ordinary_accuracy
                ),
            }
        decomposition[condition] = {
            "members": member_decomposition,
            "ensemble": ensemble_effects,
        }

    report = bind_source(
        with_content_hash(
            {
                "contract": ORDINARY_SPECIALIST_REPORT_CONTRACT,
                "schema_version": 2,
                "plan_sha256": plan["content_hash"],
                "compact_plan_sha256": compact_plan["content_hash"],
                "compact_report_sha256": compact_report["content_hash"],
                "teacher_metrics": teacher_metrics,
                "controls": compact_report["controls"],
                "ordinary_conditions": ordinary_conditions,
                "compact_conditions": compact_report["conditions"],
                "decomposition": decomposition,
                "interpretation_policy": {
                    "primary_split": "val_design",
                    "kd_effect": "ordinary_KD_minus_ordinary_CE_teacher",
                    "compression_effect": "compact_KD_minus_ordinary_KD",
                    "descriptive_not_final_claim": True,
                },
                "all_eight_ordinary_students_complete": True,
                "source_finalization": {
                    "training_and_plan_source": plan.get("source"),
                    "report_finalizer_source": finalizer_source,
                    "source_matches_plan": source_matches_plan,
                    "corrective_source_recovery": bool(
                        args.corrective_source_recovery
                    ),
                    "correction": (
                        None
                        if source_matches_plan
                        else "compact_v1_float32_mean_logit_reproduction"
                    ),
                    "scientific_training_artifacts_modified": False,
                },
                "performance_based_termination": False,
                "scientific_underperformance_blocks_execution": False,
                "final_test_accessed": False,
            }
        ),
        source_snapshot=source,
    )
    write_immutable_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
