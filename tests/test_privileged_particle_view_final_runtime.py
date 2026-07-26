from __future__ import annotations

import hashlib

import numpy as np

from teacher_logit_reco.local_particle_residual_field.particle_view import (
    PARTICLE_VIEW_FINAL_EVALUATION_PLAN_CONTRACT,
    PARTICLE_VIEW_FINAL_TEST_PERMIT_CONTRACT,
    PARTICLE_VIEW_FUSION_RECIPE_CONTRACT,
    build_final_result_payloads,
    with_content_hash,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _permit() -> dict:
    return with_content_hash(
        {
            "contract": PARTICLE_VIEW_FINAL_TEST_PERMIT_CONTRACT,
            "selection_sha256": _sha("selection"),
            "split_authorization_sha256": _sha("authorization"),
            "stack_report_sha256": _sha("stack"),
            "final_test_split_sha256": _sha("final"),
            "authorized_exports": [
                {
                    "source_bundle_sha256": _sha("winner"),
                    "bundle_export_sha256": _sha("export"),
                    "bundle_manifest_name": "bundle.json",
                    "archive_sha256": _sha("archive"),
                    "fresh_reload_audit_sha256": _sha("reload"),
                    "result_file": "final_test_export.json",
                    "seed": 202,
                    "winner_families": [
                        "selected_privileged_scientific_model",
                        "selected_pre_stage_g_hlt_deployable_model",
                    ],
                    "requires_oracle": False,
                    "required_inputs": [
                        "points",
                        "features",
                        "lorentz_vectors",
                        "mask",
                    ],
                }
            ],
            "authorized_export_count": 1,
            "authorized_hlt_baselines": [
                {
                    "bundle_sha256": _sha("a0"),
                    "seed": 202,
                    "role": "matched_a0_final_baseline",
                    "winner_families": [
                        "selected_privileged_scientific_model",
                        "selected_pre_stage_g_hlt_deployable_model",
                    ],
                    "requires_oracle": False,
                    "required_inputs": [
                        "points",
                        "features",
                        "lorentz_vectors",
                        "mask",
                    ],
                }
            ],
            "authorized_fusion_recipe_sha256": [],
            "only_preselected_median_representatives": True,
            "stage_g_controls_authorized": False,
            "oracle_diagnostics_authorized": False,
            "hlt_only_required": True,
            "winner_selection_frozen": True,
        }
    )


def test_final_result_builder_deduplicates_shared_winner_and_scores_fusion():
    permit = _permit()
    recipe = with_content_hash(
        {
            "contract": PARTICLE_VIEW_FUSION_RECIPE_CONTRACT,
            "fusion_id": "PRIVILEGED_LOGIT_AVERAGE",
            "method": "logit_average",
            "source_bundle_sha256": [_sha("a0"), _sha("winner")],
            "class_order": ["a", "b", "c"],
            "stack_partition_sha256": _sha("partition"),
            "fit_identity_sha256": _sha("fit"),
            "evaluation_identity_sha256": _sha("evaluation"),
            "linear_parameters": None,
            "optional_p7b": False,
            "p7b_hlt_only_provenance": None,
            "hlt_only": True,
            "winner_selection_permitted": False,
        }
    )
    permit = with_content_hash(
        {
            **{key: value for key, value in permit.items() if key != "content_hash"},
            "authorized_fusion_recipe_sha256": [recipe["content_hash"]],
        }
    )
    plan = with_content_hash(
        {
            "contract": PARTICLE_VIEW_FINAL_EVALUATION_PLAN_CONTRACT,
            "permit_sha256": permit["content_hash"],
            "runtime_data_config_sha256": _sha("runtime"),
            "source_commit": "a" * 40,
            "final_test_split_sha256": _sha("final"),
            "final_test_identity_sha256": _sha("identity"),
            "parent_row_identity_sha256": _sha("parent_rows"),
            "event_count": 6,
            "authorized_source_bundle_sha256": [
                _sha("a0"),
                _sha("winner"),
            ],
            "bundle_kind_by_source": {
                _sha("winner"): "frozen_consumer"
            },
            "authorized_fusion_recipe_sha256": [recipe["content_hash"]],
            "distinct_sources_evaluated_once": True,
            "labels_used_for_evaluation_only": True,
            "offline_cache_forbidden": True,
            "oracle_models_forbidden": True,
            "selection_changed": False,
        }
    )
    labels = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64)
    a0 = np.asarray(
        [
            [3, 0, 0],
            [0, 3, 0],
            [0, 0, 3],
            [0, 2, 1],
            [2, 0, 1],
            [1, 2, 0],
        ],
        dtype=np.float64,
    )
    winner = np.asarray(
        [
            [4, 0, 0],
            [0, 4, 0],
            [0, 0, 4],
            [4, 0, 0],
            [0, 4, 0],
            [0, 0, 4],
        ],
        dtype=np.float64,
    )
    results = build_final_result_payloads(
        permit=permit,
        evaluation_plan=plan,
        logits_by_source={_sha("a0"): a0, _sha("winner"): winner},
        labels=labels,
        fusion_recipes=[recipe],
        class_names=["a", "b", "c"],
        bootstrap_replicates=25,
    )
    assert len(results["baselines"]) == 1
    assert len(results["standalone"]) == 1
    assert len(results["fusions"]) == 1
    assert results["standalone"][0]["metrics"]["accuracy"] == 1.0
    assert results["standalone"][0]["accuracy_gain_over_matched_a0"] > 0
    assert not results["standalone"][0]["offline_inputs_loaded"]
    assert not results["fusions"][0]["oracle_model_loaded"]
