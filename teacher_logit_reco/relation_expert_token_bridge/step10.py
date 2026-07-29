"""Step-10 joint predictor-bundle and oracle-substitution contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .contracts import (
    bind_source,
    require_sha256,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from .oracle_substitutions import (
    build_stage_i_policy,
    validate_stage_i_policy,
)
from .predictor_bundle import (
    build_bundle_search_policy,
    validate_bundle_search_policy,
)


STEP10_BUNDLE_CONTRACT = "retb_step10_joint_predictor_bundle_v1"
STEP10_REPORT_CONTRACT = "retb_step10_report_v1"


def build_step10_bundle(
    *,
    campaign_spec_sha256: str,
    step9_bundle_sha256: str,
    global_determinism_sha256: str,
    source_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    search_policy = bind_source(
        build_bundle_search_policy(), source_snapshot=source_snapshot
    )
    stage_i_policy = bind_source(
        build_stage_i_policy(), source_snapshot=source_snapshot
    )
    manifest = bind_source(
        with_content_hash(
            {
                "contract": STEP10_BUNDLE_CONTRACT,
                "schema_version": 1,
                "parents": {
                    "campaign_spec": require_sha256(
                        campaign_spec_sha256, name="campaign_spec_sha256"
                    ),
                    "step9_bundle": require_sha256(
                        step9_bundle_sha256, name="step9_bundle_sha256"
                    ),
                    "global_determinism": require_sha256(
                        global_determinism_sha256,
                        name="global_determinism_sha256",
                    ),
                },
                "artifact_hashes": {
                    "bundle_search_policy": search_policy["content_hash"],
                    "stage_i_policy": stage_i_policy["content_hash"],
                },
                "all_predicted_scoring_uses_exact_frozen_coordinate_fusion": True,
                "configuration_shared_across_pipeline_seeds": True,
                "wrong_event_controls_enter_selection": False,
                "step11_start_requires_immutable_bundle_lock": True,
                "stack_val_consumed": False,
                "final_test_consumed": False,
                "performance_based_termination": False,
            }
        ),
        source_snapshot=source_snapshot,
    )
    report = bind_source(
        with_content_hash(
            {
                "contract": STEP10_REPORT_CONTRACT,
                "schema_version": 1,
                "step10_bundle_sha256": manifest["content_hash"],
                "checks": {
                    "three_seed_predicted_cache_index": True,
                    "width_32_complete_tuple_beam": True,
                    "locked_target_coordinate_only": True,
                    "individual_global_and_canonical_controls": True,
                    "immutable_seven_expert_bundle_lock": True,
                    "oracle_and_hybrid_substitutions": True,
                    "deterministic_negative_token_controls": True,
                    "all_negative_campaign_selects": True,
                },
                "scientific_results_inspected": False,
            }
        ),
        source_snapshot=source_snapshot,
    )
    return {
        "bundle_search_policy": search_policy,
        "stage_i_policy": stage_i_policy,
        "step10_bundle": manifest,
        "step10_report": report,
    }


def validate_step10_bundle(bundle: Mapping[str, Mapping[str, Any]]) -> str:
    if set(bundle) != {
        "bundle_search_policy",
        "stage_i_policy",
        "step10_bundle",
        "step10_report",
    }:
        raise ValueError("Step-10 bundle members differ")
    search_sha = validate_bundle_search_policy(
        bundle["bundle_search_policy"]
    )
    stage_i_sha = validate_stage_i_policy(bundle["stage_i_policy"])
    digest = validate_content_hash(
        bundle["step10_bundle"], expected_contract=STEP10_BUNDLE_CONTRACT
    )
    report_sha = validate_content_hash(
        bundle["step10_report"], expected_contract=STEP10_REPORT_CONTRACT
    )
    del report_sha
    manifest = bundle["step10_bundle"]
    if (
        manifest["artifact_hashes"]
        != {
            "bundle_search_policy": search_sha,
            "stage_i_policy": stage_i_sha,
        }
        or set(manifest["parents"])
        != {"campaign_spec", "step9_bundle", "global_determinism"}
        or not manifest[
            "all_predicted_scoring_uses_exact_frozen_coordinate_fusion"
        ]
        or not manifest["configuration_shared_across_pipeline_seeds"]
        or manifest["wrong_event_controls_enter_selection"]
        or not manifest["step11_start_requires_immutable_bundle_lock"]
        or manifest["stack_val_consumed"]
        or manifest["final_test_consumed"]
        or manifest["performance_based_termination"]
        or bundle["step10_report"]["step10_bundle_sha256"] != digest
        or bundle["step10_report"]["scientific_results_inspected"]
    ):
        raise ValueError("Step-10 bundle safety semantics differ")
    if len({repr(row.get("source")) for row in bundle.values()}) != 1:
        raise ValueError("Step-10 bundle source lineage differs")
    return digest


def publish_step10_bundle(
    *, campaign_root: str | Path, bundle: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    digest = validate_step10_bundle(bundle)
    root = Path(campaign_root)
    paths = {
        "bundle_search_policy": (
            root / "registry" / "retb_predictor_bundle_search_policy.json"
        ),
        "stage_i_policy": (
            root / "registry" / "retb_stage_i_oracle_substitution_policy.json"
        ),
        "step10_bundle": (
            root / "registry" / "retb_step10_joint_predictor_bundle.json"
        ),
        "step10_report": root / "reports" / "retb_step10_report.json",
    }
    return {
        "step10_bundle_sha256": digest,
        "publications": {
            name: write_immutable_json(path, bundle[name])
            for name, path in paths.items()
        },
    }


__all__ = [
    "STEP10_BUNDLE_CONTRACT",
    "STEP10_REPORT_CONTRACT",
    "build_step10_bundle",
    "publish_step10_bundle",
    "validate_step10_bundle",
]
