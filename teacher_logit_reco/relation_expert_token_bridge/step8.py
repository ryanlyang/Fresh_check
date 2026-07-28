"""Step-8 target-cache lineage resolution and immutable contract bundle."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .bridge_certification import (
    validate_bridge_candidate_eligibility,
    validate_bridge_content_certification,
    validate_bridge_noninferiority,
)
from .bridge_selection import BRIDGE_COORDINATE_SELECTION_CONTRACT
from .contracts import (
    bind_source,
    require_sha256,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from .registry import EXPERT_ORDER
from .target_cache import (
    SEALED_INPUT_PREPARATION_CONTRACT,
    TARGET_CACHE_POLICY_CONTRACT,
    build_target_cache_policy,
    build_target_cache_specification,
)


STEP8_BUNDLE_CONTRACT = "retb_step8_target_cache_bundle_v1"
STEP8_REPORT_CONTRACT = "retb_step8_report_v1"
TARGET_LINEAGE_CONTRACT = "retb_selected_target_lineage_v1"
POSTLOCK_TARGET_POLICY_CONTRACT = "retb_postlock_target_policy_v1"


def build_postlock_target_policy() -> dict[str, Any]:
    return with_content_hash(
        {
            "contract": POSTLOCK_TARGET_POLICY_CONTRACT,
            "schema_version": 1,
            "prelock": {
                "stack_val_oracle_targets_permitted": False,
                "final_test_model_outputs_permitted": False,
                "sealed_input_preparation_contract": (
                    SEALED_INPUT_PREPARATION_CONTRACT
                ),
                "sealed_input_preparation_may_load_checkpoint": False,
            },
            "post_finalist_lock": {
                "stack_val_oracle_targets": {
                    "permitted": True,
                    "selection_eligible": False,
                },
                "final_test_oracle_targets": {
                    "permitted": True,
                    "requires_exact_stage_m_teacher": True,
                    "five_hundred_k_teacher_permitted": False,
                },
            },
            "final_test_inference_requires_execution_lock": True,
        }
    )


def _seed(registration: Mapping[str, Any]) -> int:
    return int(
        registration.get(
            "pipeline_seed", registration.get("seed", -1)
        )
    )


def build_selected_target_lineage(
    *,
    pipeline_seed: int,
    shape_id: str,
    target_tuple: Sequence[str],
    target_registrations: Mapping[str, Mapping[str, Any]],
    slot_query_hashes: Mapping[str, str],
    eligibility_artifacts: Mapping[str, Mapping[str, Any]],
    content_certifications: Mapping[str, Mapping[str, Any] | None],
    noninferiority_artifacts: Mapping[str, Mapping[str, Any] | None],
) -> dict[str, Any]:
    if (
        int(pipeline_seed) not in {101, 202, 303}
        or len(target_tuple) != len(EXPERT_ORDER)
        or set(target_registrations) != set(EXPERT_ORDER)
        or set(slot_query_hashes) != set(EXPERT_ORDER)
        or set(eligibility_artifacts) != set(EXPERT_ORDER)
        or set(content_certifications) != set(EXPERT_ORDER)
        or set(noninferiority_artifacts) != set(EXPERT_ORDER)
    ):
        raise ValueError("selected target lineage coverage differs")
    descriptors = {}
    for index, expert in enumerate(EXPERT_ORDER):
        mode = str(target_tuple[index])
        registration = target_registrations[expert]
        registration_sha = validate_content_hash(registration)
        eligibility = eligibility_artifacts[expert]
        eligibility_sha = validate_bridge_candidate_eligibility(eligibility)
        if (
            eligibility["expert_id"] != expert
            or eligibility["target_mode"] != mode
            or eligibility["shape_id"] != str(shape_id)
            or not eligibility["maximum_performance_eligible"]
            or _seed(registration) != int(pipeline_seed)
            or registration.get("expert_id") != expert
            or eligibility["checkpoint_hashes_by_seed"][
                str(int(pipeline_seed))
            ]
            != registration.get("checkpoint_sha256")
        ):
            raise ValueError("selected target seed/mode lineage differs")
        descriptor = {
            "checkpoint_sha256": require_sha256(
                registration.get("checkpoint_sha256"),
                name=f"target_registrations.{expert}.checkpoint_sha256",
            ),
            "registration_sha256": registration_sha,
            "slot_query_sha256": require_sha256(
                slot_query_hashes[expert],
                name=f"slot_query_hashes.{expert}",
            ),
            "eligibility_sha256": eligibility_sha,
        }
        content = content_certifications[expert]
        noninferiority = noninferiority_artifacts[expert]
        if mode == "T0_PURE":
            if content is not None or noninferiority is not None:
                raise ValueError("T0 selected lineage cannot bind bridge results")
        else:
            if content is None or noninferiority is None:
                raise ValueError("bridge target lineage lacks certification")
            content_sha = validate_bridge_content_certification(content)
            noninferiority_sha = validate_bridge_noninferiority(
                noninferiority
            )
            if (
                registration.get("target_mode") != mode
                or registration.get("shape_id") != str(shape_id)
                or content["target_mode"] != mode
                or content["expert_id"] != expert
                or int(content["pipeline_seed"]) != int(pipeline_seed)
                or content["parents"]["candidate_checkpoint"]
                != descriptor["checkpoint_sha256"]
                or noninferiority["target_mode"] != mode
                or not noninferiority["offline_noninferior"]
            ):
                raise ValueError("bridge target certification lineage differs")
            descriptor.update(
                {
                    "pilot_checkpoint_sha256": require_sha256(
                        registration.get(
                            "initial_pilot_checkpoint_sha256"
                        ),
                        name=(
                            f"target_registrations.{expert}."
                            "initial_pilot_checkpoint_sha256"
                        ),
                    ),
                    "content_certification_sha256": content_sha,
                    "noninferiority_sha256": noninferiority_sha,
                }
            )
        descriptors[expert] = descriptor
    return with_content_hash(
        {
            "contract": TARGET_LINEAGE_CONTRACT,
            "schema_version": 1,
            "pipeline_seed": int(pipeline_seed),
            "shape_id": str(shape_id),
            "expert_order": list(EXPERT_ORDER),
            "target_tuple": [str(value) for value in target_tuple],
            "target_descriptors": descriptors,
            "cross_seed_substitution_permitted": False,
        }
    )


def build_locked_target_cache_specification(
    *,
    split: str,
    pipeline_seed: int,
    shape_id: str,
    allocation: Mapping[str, Sequence[int]],
    coordinate_selection: Mapping[str, Any],
    coordinate_contract_sha256: str,
    target_lineage: Mapping[str, Any],
    fusion_registration: Mapping[str, Any],
    normalizer_set: Mapping[str, Any],
    identity_manifest_sha256: str,
    identity_order_sha256: str,
    event_count: int,
) -> dict[str, Any]:
    selection_sha = validate_content_hash(
        coordinate_selection,
        expected_contract=BRIDGE_COORDINATE_SELECTION_CONTRACT,
    )
    lineage_sha = validate_content_hash(
        target_lineage, expected_contract=TARGET_LINEAGE_CONTRACT
    )
    fusion_sha = validate_content_hash(fusion_registration)
    normalizer_sha = validate_content_hash(normalizer_set)
    locked = [
        row
        for row in coordinate_selection["locked_coordinate_systems"]
        if row["coordinate_contract_sha256"] == coordinate_contract_sha256
    ]
    if len(locked) != 1:
        raise ValueError("locked target coordinate is absent/duplicated")
    row = locked[0]
    if (
        int(target_lineage["pipeline_seed"]) != int(pipeline_seed)
        or target_lineage["shape_id"] != str(shape_id)
        or list(target_lineage["target_tuple"]) != list(row["target_tuple"])
        or fusion_registration.get("checkpoint_sha256")
        != row["fusion_sha256"]
        or normalizer_sha != row["normalizer_set_sha256"]
    ):
        raise ValueError("locked coordinate fusion/normalizer lineage differs")
    specification = build_target_cache_specification(
        split=split,
        pipeline_seed=pipeline_seed,
        shape_id=shape_id,
        allocation=allocation,
        target_tuple=row["target_tuple"],
        target_descriptors=target_lineage["target_descriptors"],
        selected_target_lineage_sha256=lineage_sha,
        target_cache_namespace=row["target_cache_namespace"],
        locked_coordinate_contract_sha256=coordinate_contract_sha256,
        locked_coordinate_selection_sha256=selection_sha,
        offline_fusion_checkpoint_sha256=fusion_registration[
            "checkpoint_sha256"
        ],
        offline_fusion_registration_sha256=fusion_sha,
        normalizer_set_sha256=normalizer_sha,
        identity_manifest_sha256=identity_manifest_sha256,
        identity_order_sha256=identity_order_sha256,
        event_count=event_count,
    )
    return specification


def build_step8_bundle(
    *,
    campaign_spec_sha256: str,
    step7_bundle_sha256: str,
    global_determinism_sha256: str,
    source_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    parents = {
        "campaign_spec": require_sha256(
            campaign_spec_sha256, name="campaign_spec_sha256"
        ),
        "step7_bundle": require_sha256(
            step7_bundle_sha256, name="step7_bundle_sha256"
        ),
        "global_determinism": require_sha256(
            global_determinism_sha256, name="global_determinism_sha256"
        ),
    }
    policy = bind_source(
        build_target_cache_policy(), source_snapshot=source_snapshot
    )
    postlock = bind_source(
        build_postlock_target_policy(), source_snapshot=source_snapshot
    )
    manifest = bind_source(
        with_content_hash(
            {
                "contract": STEP8_BUNDLE_CONTRACT,
                "schema_version": 1,
                "parents": parents,
                "artifact_hashes": {
                    "target_cache_policy": policy["content_hash"],
                    "postlock_target_policy": postlock["content_hash"],
                },
                "selected_coordinate_required": True,
                "cross_seed_cache_reuse_permitted": False,
                "float16_failure_stops_workflow": False,
                "performance_based_termination": False,
            }
        ),
        source_snapshot=source_snapshot,
    )
    report = bind_source(
        with_content_hash(
            {
                "contract": STEP8_REPORT_CONTRACT,
                "schema_version": 1,
                "step8_bundle_sha256": manifest["content_hash"],
                "checks": {
                    "selected_shape_and_mode_bound": True,
                    "pilot_and_content_parents_bound": True,
                    "float16_audit_and_fp32_fallback": True,
                    "per_seed_resumable_shards": True,
                    "consumer_loads_float32": True,
                    "sealed_test_preparation_has_no_checkpoint": True,
                    "cross_seed_mixing_rejected": True,
                },
                "scientific_results_inspected": False,
            }
        ),
        source_snapshot=source_snapshot,
    )
    return {
        "target_cache_policy": policy,
        "postlock_target_policy": postlock,
        "step8_bundle": manifest,
        "step8_report": report,
    }


def validate_step8_bundle(bundle: Mapping[str, Any]) -> str:
    if set(bundle) != {
        "target_cache_policy",
        "postlock_target_policy",
        "step8_bundle",
        "step8_report",
    }:
        raise ValueError("Step-8 bundle members differ")
    policy_sha = validate_content_hash(
        bundle["target_cache_policy"],
        expected_contract=TARGET_CACHE_POLICY_CONTRACT,
    )
    postlock_sha = validate_content_hash(
        bundle["postlock_target_policy"],
        expected_contract=POSTLOCK_TARGET_POLICY_CONTRACT,
    )
    digest = validate_content_hash(
        bundle["step8_bundle"], expected_contract=STEP8_BUNDLE_CONTRACT
    )
    if bundle["step8_bundle"]["artifact_hashes"] != {
        "target_cache_policy": policy_sha,
        "postlock_target_policy": postlock_sha,
    }:
        raise ValueError("Step-8 artifact hashes differ")
    validate_content_hash(
        bundle["step8_report"], expected_contract=STEP8_REPORT_CONTRACT
    )
    if bundle["step8_report"]["step8_bundle_sha256"] != digest:
        raise ValueError("Step-8 report parent differs")
    if len({repr(row.get("source")) for row in bundle.values()}) != 1:
        raise ValueError("Step-8 source lineage differs")
    return digest


def publish_step8_bundle(
    *, campaign_root: str | Path, bundle: Mapping[str, Any]
) -> dict[str, Any]:
    digest = validate_step8_bundle(bundle)
    root = Path(campaign_root)
    paths = {
        "target_cache_policy": (
            root / "registry" / "retb_target_cache_policy.json"
        ),
        "postlock_target_policy": (
            root / "registry" / "retb_postlock_target_policy.json"
        ),
        "step8_bundle": (
            root / "registry" / "retb_step8_target_cache_bundle.json"
        ),
        "step8_report": root / "reports" / "retb_step8_report.json",
    }
    return {
        "step8_bundle_sha256": digest,
        "publications": {
            name: write_immutable_json(paths[name], bundle[name])
            for name in paths
        },
    }


__all__ = [
    "POSTLOCK_TARGET_POLICY_CONTRACT",
    "STEP8_BUNDLE_CONTRACT",
    "STEP8_REPORT_CONTRACT",
    "TARGET_LINEAGE_CONTRACT",
    "build_locked_target_cache_specification",
    "build_postlock_target_policy",
    "build_selected_target_lineage",
    "build_step8_bundle",
    "publish_step8_bundle",
    "validate_step8_bundle",
]
