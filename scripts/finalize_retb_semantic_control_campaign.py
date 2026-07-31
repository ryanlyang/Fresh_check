#!/usr/bin/env python3
"""Aggregate the complete, executable Section-28 semantic-control matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.execute_retb_semantic_control_campaign import CONTRACT, _run_id  # noqa: E402
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    SEMANTIC_CONTROL_POLICY, bind_source, canonical_sha256, load_hashed_json,
    require_sha256, with_content_hash, write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.final_consumers import (  # noqa: E402
    BYPASS_CONTROLS,
)
from teacher_logit_reco.relation_expert_token_bridge.late_plan_factories import (  # noqa: E402
    SEMANTIC_CONTROL_KINDS,
)
from teacher_logit_reco.relation_expert_token_bridge.oracle_substitutions import (  # noqa: E402
    STAGE_I_EVALUATION_CONTRACT, validate_stage_i_evaluation,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.predictor_bundle import (  # noqa: E402
    PIPELINE_SEEDS, validate_locked_target_coordinate,
    validate_predictor_candidate,
)
from teacher_logit_reco.relation_expert_token_bridge.predictor_cache import (  # noqa: E402
    PREDICTOR_INFERENCE_MANIFEST_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.step9 import (  # noqa: E402
    validate_materialized_predictor_run,
)
from teacher_logit_reco.relation_expert_token_bridge.step7 import STAGE_E_SHAPES  # noqa: E402
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)


def _metrics(row: Mapping[str, Any]) -> dict[str, float]:
    return {
        name: float(row[name])
        for name in ("accuracy", "cross_entropy", "macro_per_class_accuracy")
    }


def _delta(
    row: Mapping[str, Any], reference: Mapping[str, Any]
) -> dict[str, float]:
    return {
        f"{name}_control_minus_reference": float(row[name])
        - float(reference[name])
        for name in ("accuracy", "cross_entropy", "macro_per_class_accuracy")
    }


def _metric_path(root: Path, run_id: str) -> Path:
    run_root = root / "runs" / "final_consumers" / run_id
    preferred = run_root / "val_design" / "metrics.json"
    return preferred if preferred.is_file() else run_root / "reference_metrics.json"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _predictor_architecture_evidence(
    root: Path, *, campaign: Mapping[str, Any]
) -> dict[str, Any]:
    selection_root = root / "selection" / "predictor_bundle"
    index = load_hashed_json(
        selection_root / "bundle_input_index.json",
        expected_contract="retb_predictor_bundle_input_index_v1",
    )
    configuration_path = selection_root / "inputs" / "selector_configuration.json"
    if (
        set(index) != {
            "contract", "schema_version", "candidate_count", "coordinate_count",
            "candidate_manifest_hashes", "coordinate_manifest_hashes",
            "selector_configuration_sha256", "predictor_phase_plan_sha256",
            "scientific_underperformance_blocks_continuation", "source",
            "content_hash",
        }
        or int(index.get("schema_version", -1)) != 1
        or index.get("source") != campaign.get("source")
        or index.get("scientific_underperformance_blocks_continuation") is not False
        or _file_sha256(configuration_path)
        != index.get("selector_configuration_sha256")
    ):
        raise ValueError("predictor selector input-index lineage differs")
    configuration = json.loads(configuration_path.read_text("utf-8"))
    if set(configuration) != {
        "candidate_manifest_paths", "coordinate_manifest_paths",
        "inference_manifest_paths", "calibration_artifact_paths",
        "capacity_report_paths", "materialized_run_paths",
        "fusion_checkpoint_paths", "label_npz_paths",
        "label_manifest_paths_by_seed", "label_manifest_hashes_by_seed",
        "label_npz_hashes_by_seed",
    }:
        raise ValueError("predictor selector configuration fields differ")
    require_sha256(
        index["predictor_phase_plan_sha256"],
        name="bundle_input_index.predictor_phase_plan_sha256",
    )
    candidate_paths = [Path(path) for path in configuration["candidate_manifest_paths"]]
    coordinate_paths = [Path(path) for path in configuration["coordinate_manifest_paths"]]
    candidates = [load_hashed_json(path) for path in candidate_paths]
    coordinates = [load_hashed_json(path) for path in coordinate_paths]
    if (
        int(index.get("candidate_count", -1)) != len(candidates)
        or int(index.get("coordinate_count", -1)) != len(coordinates)
        or index.get("candidate_manifest_hashes")
        != [row["content_hash"] for row in candidates]
        or index.get("coordinate_manifest_hashes")
        != [row["content_hash"] for row in coordinates]
        or any(row.get("source") != campaign.get("source") for row in [*candidates, *coordinates])
    ):
        raise ValueError("predictor selector manifest index differs")
    for candidate in candidates:
        validate_predictor_candidate(candidate)
    for coordinate in coordinates:
        validate_locked_target_coordinate(coordinate)
    candidate_map = {str(row["candidate_id"]): row for row in candidates}
    if (
        len(candidate_map) != len(candidates)
        or set(configuration["materialized_run_paths"]) != set(candidate_map)
        or set(configuration["inference_manifest_paths"]) != set(candidate_map)
    ):
        raise ValueError("predictor selector candidate coverage differs")
    groups: dict[str, list[dict[str, Any]]] = {
        "direct": [], "gated": [], "logit_only": [], "non_logit": []
    }
    for candidate_id, candidate in sorted(candidate_map.items()):
        paths = configuration["materialized_run_paths"][candidate_id]
        inference_paths = configuration["inference_manifest_paths"][candidate_id]
        if {int(seed) for seed in paths} != set(PIPELINE_SEEDS) or {
            int(seed) for seed in inference_paths
        } != set(PIPELINE_SEEDS):
            raise ValueError("predictor selector seed coverage differs")
        for seed, path in sorted(paths.items(), key=lambda item: int(item[0])):
            run = load_hashed_json(Path(path))
            validate_materialized_predictor_run(run)
            inference = load_hashed_json(
                Path(inference_paths[seed]),
                expected_contract=PREDICTOR_INFERENCE_MANIFEST_CONTRACT,
            )
            metric_path = Path(inference_paths[seed]).parent / "val_design_metrics.json"
            metric = load_hashed_json(
                metric_path,
                expected_contract="retb_predictor_val_design_metrics_v1",
            )
            expected_seed = candidate["seed_artifacts"][str(seed)]
            candidate_metrics = candidate["metrics_by_seed"]
            if (
                run.get("source") != campaign.get("source")
                or inference.get("source") != campaign.get("source")
                or metric.get("source") != campaign.get("source")
                or run["content_hash"]
                != candidate["materialized_run_hashes"][str(seed)]
                or int(run["pipeline_seed"]) != int(seed)
                or run["expert_id"] != candidate["expert_id"]
                or run["architecture"] != candidate["configuration"]["architecture"]
                or run["objective_id"] != candidate["configuration"]["objective_id"]
                or inference["content_hash"] != expected_seed["inference_manifest"]
                or inference["parents"]["predictor_registration"]
                != expected_seed["predictor_registration"]
                or inference["parents"]["predictor_checkpoint"]
                != expected_seed["predictor_checkpoint"]
                or metric.get("run_sha256") != run["content_hash"]
                or metric.get("run_id") != run["run_id"]
                or int(metric.get("pipeline_seed", -1)) != int(seed)
                or metric.get("expert_id") != candidate["expert_id"]
                or metric.get("capacity_report_sha256")
                != expected_seed["capacity_report"]
                or metric.get("predictor_registration_sha256")
                != expected_seed["predictor_registration"]
                or float(metric["val_design_accuracy"])
                != float(candidate_metrics["hybrid_accuracy"][str(seed)])
                or float(metric["cross_entropy"])
                != float(candidate_metrics["hybrid_cross_entropy"][str(seed)])
                or float(metric["normalized_token_error"])
                != float(candidate_metrics["normalized_token_error"][str(seed)])
            ):
                raise ValueError("predictor architecture evidence lineage differs")
            record = {
                "record_type": "predictor_evaluation",
                "candidate_id": str(candidate_id),
                "pipeline_seed": int(seed),
                "architecture": str(run["architecture"]),
                "objective_id": str(run["objective_id"]),
                "metrics": {
                    "accuracy": float(metric["val_design_accuracy"]),
                    "cross_entropy": float(metric["cross_entropy"]),
                    "normalized_token_error": float(
                        metric["normalized_token_error"]
                    ),
                },
                "bundle_input_index_sha256": index["content_hash"],
                "selector_configuration_sha256": index[
                    "selector_configuration_sha256"
                ],
                "candidate_manifest_sha256": candidate["content_hash"],
                "materialized_run_sha256": run["content_hash"],
                "inference_manifest_sha256": inference["content_hash"],
                "metric_artifact_sha256": metric["content_hash"],
            }
            if run["architecture"] == "A3_SLOT_DECODER_DIRECT":
                groups["direct"].append(record)
            if run["architecture"] == "A4_SLOT_DECODER_GATED":
                groups["gated"].append(record)
            if run["objective_id"] == "W_LOGIT_ONLY":
                groups["logit_only"].append(record)
            else:
                groups["non_logit"].append(record)
    if any(not values for values in groups.values()):
        raise ValueError("predictor semantic comparison lacks real evaluation rows")
    return {
        "bundle_input_index_sha256": index["content_hash"],
        "selector_configuration_sha256": index[
            "selector_configuration_sha256"
        ],
        "candidate_manifest_hashes": index["candidate_manifest_hashes"],
        "coordinate_manifest_hashes": index["coordinate_manifest_hashes"],
        "direct_evaluations": groups["direct"],
        "gated_evaluations": groups["gated"],
        "W_LOGIT_ONLY_evaluations": groups["logit_only"],
        "non_logit_evaluations": groups["non_logit"],
        "direct_evaluations_sha256": canonical_sha256(groups["direct"]),
        "gated_evaluations_sha256": canonical_sha256(groups["gated"]),
        "W_LOGIT_ONLY_evaluations_sha256": canonical_sha256(groups["logit_only"]),
        "non_logit_evaluations_sha256": canonical_sha256(groups["non_logit"]),
        "all_evidence_is_materialized_val_design_inference": True,
    }


def _mean_predictor_metrics(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    if not records:
        raise ValueError("predictor comparison group is empty")
    return {
        name: sum(float(row["metrics"][name]) for row in records) / len(records)
        for name in ("accuracy", "cross_entropy", "normalized_token_error")
    }


def _predictor_delta(
    control: Mapping[str, Any], reference: Mapping[str, Any]
) -> dict[str, float]:
    return {
        f"{name}_control_minus_reference": float(control[name])
        - float(reference[name])
        for name in ("accuracy", "cross_entropy", "normalized_token_error")
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    root = args.campaign_root.resolve()
    campaign = load_and_validate_campaign_source(root, repo_root=REPO_ROOT)
    rows: list[dict[str, Any]] = []
    metric_records: dict[str, list[dict[str, Any]]] = {
        control_id: [] for control_id in SEMANTIC_CONTROL_KINDS
    }
    relation_predictor_hashes = []
    bypass_hashes = []
    reconstruction_hashes = []
    reconstruction_metric_records: list[dict[str, Any]] = []
    for role in STAGE_E_SHAPES:
        refiner_lock = load_hashed_json(
            root / "selection" / "final_consumers" / role / "token_refiner_lock.json"
        )
        if refiner_lock.get("source") != campaign.get("source"):
            raise ValueError("token-refiner lock source lineage differs")
        for seed in (101, 202, 303):
            semantic = load_hashed_json(
                root / "controls" / "semantics" / "relation_predictor"
                / role / f"seed_{seed}.json",
                expected_contract="retb_relation_predictor_semantic_controls_v3",
            )
            expected_keys = {
                "RELATION__active",
                *(f"RELATION__zero__{expert}" for expert in (
                    "PT", "TRACK", "PID", "CHARGE", "DENSITY", "REGION",
                )),
                *(f"RELATION__{mode}" for mode in (
                    "within_jet_cyclic",
                    "wrong_event_matched_multiplicity",
                    "directional_endpoint_swap",
                )),
                *(f"PREDICTOR__{mode}" for mode in (
                    "active", "zero_hlt_evidence",
                    "shuffle_hlt_evidence_between_events",
                    "remove_native_particle_context",
                    "remove_noncorresponding_expert_banks",
                )),
            }
            wrong_population = semantic["controls"][
                "RELATION__wrong_event_matched_multiplicity"
            ]["eligible_population"]
            if (
                semantic.get("source") != campaign.get("source")
                or semantic.get("semantic_control_policy")
                != campaign.get("semantic_control_policy")
                or semantic.get("semantic_control_policy")
                != SEMANTIC_CONTROL_POLICY
                or set(semantic["controls"]) != expected_keys
                or semantic.get("per_expert_relation_zero_coverage")
                != ["PT", "TRACK", "PID", "CHARGE", "DENSITY", "REGION"]
                or not semantic["wrong_event_relation_is_multiplicity_matched_and_never_self"]
                or any(
                    row.get("all_val_design_events_evaluated") is not True
                    for name, row in semantic["controls"].items()
                    if name != "RELATION__wrong_event_matched_multiplicity"
                )
                or int(wrong_population["evaluated_event_count"]) <= 0
                or int(wrong_population["total_event_count"])
                - int(wrong_population["evaluated_event_count"])
                != int(wrong_population["excluded_singleton_stratum_event_count"])
                or float(wrong_population["evaluated_fraction"])
                != int(wrong_population["evaluated_event_count"])
                / int(wrong_population["total_event_count"])
            ):
                raise ValueError("relation/predictor semantic coverage differs")
            relation_predictor_hashes.append(semantic["content_hash"])
            condition_to_control = {
                "RELATION__within_jet_cyclic": "RELATION_WITHIN_JET_SHUFFLE",
                "RELATION__wrong_event_matched_multiplicity": (
                    "RELATION_WRONG_EVENT_MATCHED_MULTIPLICITY"
                ),
                "RELATION__directional_endpoint_swap": (
                    "RELATION_DIRECTIONAL_ENDPOINT_SWAP"
                ),
                "PREDICTOR__zero_hlt_evidence": "PREDICTOR_ZERO_HLT_EVIDENCE",
                "PREDICTOR__shuffle_hlt_evidence_between_events": (
                    "PREDICTOR_SHUFFLE_HLT_EVIDENCE"
                ),
                "PREDICTOR__remove_native_particle_context": (
                    "PREDICTOR_REMOVE_NATIVE_PARTICLE_CONTEXT"
                ),
                "PREDICTOR__remove_noncorresponding_expert_banks": (
                    "PREDICTOR_REMOVE_NONCORRESPONDING_BANKS"
                ),
            }
            for condition_id, condition in semantic["controls"].items():
                control_id = (
                    "RELATION_ZERO"
                    if condition_id.startswith("RELATION__zero__")
                    else condition_to_control.get(condition_id)
                )
                if control_id is None:
                    continue
                reference = semantic["controls"][
                    "RELATION__active"
                    if condition_id.startswith("RELATION__")
                    else "PREDICTOR__active"
                ]["metrics"]
                computed_delta = _delta(condition["metrics"], reference)
                if {
                    name: float(value)
                    for name, value in condition["metric_deltas"].items()
                } != computed_delta:
                    raise ValueError("relation/predictor serialized delta differs")
                metric_records[control_id].append({
                    "record_type": "metric_comparison",
                    "shape_role": role,
                    "pipeline_seed": seed,
                    "condition_id": condition_id,
                    "relation_expert": condition.get("relation_expert"),
                    "event_count": int(condition["event_count"]),
                    "eligible_population": condition["eligible_population"],
                    "metrics": _metrics(condition["metrics"]),
                    "reference_metrics": _metrics(reference),
                    "metric_deltas": computed_delta,
                    "source_artifact_sha256": semantic["content_hash"],
                })

            bypass_controls: set[str] = set()
            for consumer_kind in ("HF_UNRESTRICTED", "HF_ADAPTER"):
                bypass = load_hashed_json(
                    root / "controls" / "semantics" / "bypass"
                    / role / f"seed_{seed}" / consumer_kind
                    / "bypass_controls.json"
                )
                if bypass.get("source") != campaign.get("source"):
                    raise ValueError("native-bypass source lineage differs")
                bypass_controls.update(bypass["controls"])
                bypass_hashes.append(bypass["content_hash"])
                normal = bypass["controls"]["NORMAL"]["metrics"]
                bypass_mapping = {
                    "NATIVE_BRANCH_REMOVED": "BYPASS_NATIVE_REMOVED",
                    "RECONSTRUCTED_BRANCH_REMOVED": (
                        "BYPASS_RECONSTRUCTED_REMOVED"
                    ),
                    "NATIVE_BRANCH_DROPPED_AT_EVALUATION": (
                        "BYPASS_NATIVE_DROPPED_EVAL"
                    ),
                    "RESIDUAL_GAMMA_ZERO": "BYPASS_GAMMA_ZERO",
                    "SOURCE_EMBEDDINGS_SWAPPED": (
                        "BYPASS_SOURCE_EMBEDDINGS_SWAPPED"
                    ),
                }
                for name, control_id in bypass_mapping.items():
                    if name not in bypass["controls"]:
                        continue
                    values = bypass["controls"][name]["metrics"]
                    metric_records[control_id].append({
                        "record_type": "metric_comparison",
                        "shape_role": role,
                        "pipeline_seed": seed,
                        "consumer_kind": consumer_kind,
                        "condition_id": name,
                        "metrics": _metrics(values),
                        "reference_metrics": _metrics(normal),
                        "metric_deltas": _delta(values, normal),
                        "source_artifact_sha256": bypass["content_hash"],
                    })
            if bypass_controls != set(BYPASS_CONTROLS):
                raise ValueError("native-bypass semantic coverage differs")
            pf_id = f"RETB_{role}_PF_FROZEN_PF_FROZEN_ND0_NONE_TOKEN_PREDICTED_S{seed}"
            tr_id = (
                f"RETB_{role}_TR_REFINE_{refiner_lock['selected_variant']}_"
                f"ND0_NONE_TOKEN_PREDICTED_S{seed}"
            )
            metrics = {
                name: load_hashed_json(_metric_path(root, run_id))
                for name, run_id in (
                    ("frozen_reconstruction", pf_id),
                    ("token_refiner", tr_id),
                    ("unrestricted_fusion", _run_id(role, seed)),
                )
            }
            if any(
                metric.get("source") != campaign.get("source")
                for metric in metrics.values()
            ):
                raise ValueError("reconstruction metric source lineage differs")
            reconstruction_hashes.append(canonical_sha256({
                name: metric["content_hash"] for name, metric in metrics.items()
            }))
            reconstruction_metric_records.append({
                "shape_role": role,
                "pipeline_seed": seed,
                "condition_metrics": {
                    name: _metrics(metric) for name, metric in metrics.items()
                },
                "condition_metric_artifact_hashes": {
                    name: metric["content_hash"]
                    for name, metric in metrics.items()
                },
                "condition_metric_deltas_vs_frozen_reconstruction": {
                    name: _delta(
                        metric,
                        metrics["frozen_reconstruction"],
                    )
                    for name, metric in metrics.items()
                    if name != "frozen_reconstruction"
                },
            })

    stage_i = load_hashed_json(root / "selection" / "stage_i" / "stage_i_index.json")
    if stage_i.get("source") != campaign.get("source"):
        raise ValueError("Stage-I index source lineage differs")
    token_hashes = []
    for seed in (101, 202, 303):
        evaluation = load_hashed_json(
            root / "selection" / "stage_i" / f"seed_{seed}" / "evaluation.json",
            expected_contract=STAGE_I_EVALUATION_CONTRACT,
        )
        validate_stage_i_evaluation(evaluation)
        if evaluation.get("source") != campaign.get("source"):
            raise ValueError("Stage-I evaluation source lineage differs")
        token_hashes.append(evaluation["content_hash"])
        reference = evaluation["conditions"]["ALL_PREDICTED"]["metrics"]
        token_mapping = {
            "SLOT_PERMUTED_TARGETS": "TOKEN_SLOT_PERMUTE",
            "MATCHED_GAUSSIAN_NOISE": "TOKEN_MATCHED_GAUSSIAN_NOISE",
        }
        prefix_mapping = {
            "WITHIN_CLASS_MEAN_TARGETS__": "TOKEN_WITHIN_CLASS_MEAN_BANK",
            "WITHIN_CLASS_WRONG_EVENT_BANK__": (
                "TOKEN_WRONG_EVENT_SAME_CLASS_BANK"
            ),
            "WRONG_EVENT_BANK__": "TOKEN_WRONG_EVENT_UNMATCHED_BANK",
            "ZERO_ORACLE_BANK__": "TOKEN_ZERO_BANK",
        }
        for condition_id, condition in evaluation["conditions"].items():
            control_id = token_mapping.get(condition_id)
            if control_id is None:
                control_id = next(
                    (mapped for prefix, mapped in prefix_mapping.items()
                     if condition_id.startswith(prefix)), None,
                )
            if control_id is None:
                continue
            values = condition["metrics"]
            metric_records[control_id].append({
                "record_type": "metric_comparison",
                "pipeline_seed": seed,
                "condition_id": condition_id,
                "metrics": _metrics(values),
                "reference_metrics": _metrics(reference),
                "metric_deltas": _delta(values, reference),
                "source_artifact_sha256": evaluation["content_hash"],
            })
        metric_records["TOKEN_NORM_COVARIANCE"].append({
            "record_type": "bank_diagnostic",
            "pipeline_seed": seed,
            "predicted_bank_diagnostics": evaluation["conditions"]
            ["ALL_PREDICTED"]["bank_diagnostics"],
            "oracle_bank_diagnostics": evaluation["conditions"]
            ["ALL_ORACLE"]["bank_diagnostics"],
            "source_artifact_sha256": evaluation["content_hash"],
        })
    if {
        row["evaluation_sha256"] for row in stage_i["pipeline_seed_records"]
    } != set(token_hashes):
        raise ValueError("Stage-I token semantic index differs")

    bias_scale = load_hashed_json(
        root / "controls" / "semantics" / "bias_scale.json",
        expected_contract="retb_bias_scale_semantic_evaluation_v1",
    )
    topologies = {row["topology"] for row in bias_scale["rows"]}
    if (
        bias_scale.get("source") != campaign.get("source")
        or topologies != {"B_DUAL_FIXED", "B_DUAL_GATED"}
        or len(bias_scale["rows"]) != 14
    ):
        raise ValueError("fixed-versus-learned bias-scale evidence is absent")
    architecture = _predictor_architecture_evidence(root, campaign=campaign)
    metric_records["PREDICTOR_DIRECT_VS_GATED"].extend(
        architecture["direct_evaluations"]
        + architecture["gated_evaluations"]
    )
    direct_mean = _mean_predictor_metrics(architecture["direct_evaluations"])
    gated_mean = _mean_predictor_metrics(architecture["gated_evaluations"])
    metric_records["PREDICTOR_DIRECT_VS_GATED"].append({
        "record_type": "aggregate_metric_comparison",
        "condition_id": "GATED_MEAN_VS_DIRECT_MEAN",
        "metrics": gated_mean,
        "reference_metrics": direct_mean,
        "metric_deltas": _predictor_delta(gated_mean, direct_mean),
        "aggregation": "arithmetic_mean_over_predeclared_val_design_rows",
    })
    metric_records["PREDICTOR_W_LOGIT_ONLY"].extend(
        architecture["W_LOGIT_ONLY_evaluations"]
    )
    logit_mean = _mean_predictor_metrics(
        architecture["W_LOGIT_ONLY_evaluations"]
    )
    non_logit_mean = _mean_predictor_metrics(
        architecture["non_logit_evaluations"]
    )
    metric_records["PREDICTOR_W_LOGIT_ONLY"].append({
        "record_type": "aggregate_metric_comparison",
        "condition_id": "W_LOGIT_ONLY_MEAN_VS_NON_LOGIT_MEAN",
        "metrics": logit_mean,
        "reference_metrics": non_logit_mean,
        "metric_deltas": _predictor_delta(logit_mean, non_logit_mean),
        "aggregation": "arithmetic_mean_over_predeclared_val_design_rows",
    })
    by_expert: dict[str, dict[str, Mapping[str, Any]]] = {}
    for bias_row in bias_scale["rows"]:
        by_expert.setdefault(str(bias_row["expert_id"]), {})[
            str(bias_row["topology"])
        ] = bias_row
    for expert, pair in sorted(by_expert.items()):
        fixed = pair["B_DUAL_FIXED"]
        gated = pair["B_DUAL_GATED"]
        metric_records["RELATION_FIXED_VS_LEARNED_SCALE"].append({
            "record_type": "metric_comparison",
            "condition_id": "B_DUAL_GATED_VS_B_DUAL_FIXED",
            "expert_id": expert,
            "fixed_run_id": fixed["run_id"],
            "learned_run_id": gated["run_id"],
            "reference_metrics": _metrics(fixed["metrics"]),
            "metrics": _metrics(gated["metrics"]),
            "metric_deltas": _delta(
                gated["metrics"], fixed["metrics"]
            ),
            "source_artifact_sha256": bias_scale["content_hash"],
        })
    evidence = {
        "relation_and_predictor_wave": canonical_sha256(relation_predictor_hashes),
        "token_substitutions": canonical_sha256(token_hashes),
        "native_bypass": canonical_sha256(bypass_hashes),
        "reconstruction_comparison": canonical_sha256(reconstruction_hashes),
        "fixed_vs_learned_bias_scale": bias_scale["content_hash"],
        "predictor_direct_gated_logit_only": canonical_sha256(architecture),
    }
    for control_id in SEMANTIC_CONTROL_KINDS:
        family = (
            "relation_and_predictor_wave"
            if control_id.startswith(("RELATION_ZERO", "RELATION_WITHIN", "RELATION_WRONG", "RELATION_DIRECTIONAL", "PREDICTOR_ZERO", "PREDICTOR_SHUFFLE", "PREDICTOR_REMOVE"))
            else "fixed_vs_learned_bias_scale"
            if control_id == "RELATION_FIXED_VS_LEARNED_SCALE"
            else "predictor_direct_gated_logit_only"
            if control_id in {"PREDICTOR_DIRECT_VS_GATED", "PREDICTOR_W_LOGIT_ONLY"}
            else "native_bypass"
            if control_id.startswith("BYPASS_")
            else "token_substitutions"
        )
        rows.append({
            "control_id": control_id,
            "evidence_family": family,
            "artifact_sha256": evidence[family],
            "metric_records": metric_records[control_id],
            "metric_record_count": len(metric_records[control_id]),
        })
    if any(not row["metric_records"] for row in rows):
        missing = [row["control_id"] for row in rows if not row["metric_records"]]
        raise ValueError(f"semantic controls lack reportable metrics: {missing}")
    artifact = bind_source(
        with_content_hash({
            "contract": CONTRACT,
            "schema_version": 5,
            "control_ids": list(SEMANTIC_CONTROL_KINDS),
            "semantic_control_policy": dict(SEMANTIC_CONTROL_POLICY),
            "rows": rows,
            "evidence_families": evidence,
            "predictor_architecture_evidence": architecture,
            "reconstruction_metric_records": reconstruction_metric_records,
            "complete_coverage": True,
            "all_section_28_controls_have_real_evaluation_evidence": True,
            "scientific_underperformance_blocks_continuation": False,
            "stack_val_consumed": False,
            "final_test_consumed": False,
        }), source_snapshot=source_snapshot(REPO_ROOT),
    )
    write_immutable_json(args.output, artifact)
    print(json.dumps({"semantic_controls_sha256": artifact["content_hash"], "control_count": len(rows)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
