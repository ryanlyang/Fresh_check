from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from teacher_logit_reco.local_particle_residual_field.particle_view import (
    PARTICLE_VIEW_DISTILLATION_REGISTRATION_CONTRACT,
    PARTICLE_VIEW_LABEL_EXPOSURE_LEDGER_CONTRACT,
    PARTICLE_VIEW_REPORT_SECTIONS,
    PARTICLE_VIEW_SPLIT_AUTHORIZATION_CONTRACT,
    PARTICLE_VIEW_STACK_REPORT_CONTRACT,
    PARTICLE_VIEW_TARGET_LOGIT_CACHE_CONTRACT,
    PARTICLE_VIEW_WINNER_SELECTION_CONTRACT,
    SELECTED_VIEW_MATERIALIZATION_POLICY,
    ParticleViewLineageNode,
    RetentionCandidate,
    build_diagnostic_budget,
    build_final_test_permit,
    build_fairness_budget_accounting,
    build_particle_view_deployment_manifest,
    build_particle_view_lineage_graph,
    build_retention_plan,
    build_separated_campaign_report,
    build_storage_reservation,
    build_tap_stage_reservation,
    build_view_coordinate_binding,
    canonical_sha256,
    evaluate_authorized_final_test,
    execute_retention_plan,
    export_hlt_only_particle_view_bundle,
    load_exported_particle_view_bundle,
    run_fresh_process_reload_audit,
    sha256_file,
    validate_coordinate_cache_consumer_logit_chain,
    validate_particle_view_lineage_graph,
    with_content_hash,
    write_bounded_json_diagnostic,
    write_hlt_reload_fixture,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def test_step9_measured_storage_ram_diagnostics_and_hash_safe_eviction(tmp_path):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    (campaign / "source.bin").write_bytes(b"s" * 101)
    (campaign / "checkpoint.pt").write_bytes(b"c" * 53)
    diagnostic_budget = build_diagnostic_budget(
        max_file_bytes=256,
        max_total_bytes=1024,
        max_attention_samples=2,
    )
    tap = build_tap_stage_reservation(
        source_role="offline_teacher",
        source_manifest_sha256=_sha("source"),
        logical_split_sha256=_sha("split"),
        ordered_identity_sha256=_sha("identity"),
        teacher_checkpoint_sha256=_sha("teacher"),
        tap_spec_sha256=_sha("tap"),
        jets=4,
        max_particles=8,
        token_width=16,
        identity_columns=2,
    )
    reservation = build_storage_reservation(
        campaign_root=campaign,
        measured_artifacts={
            "source_cache": "source.bin",
            "selected_checkpoint": "checkpoint.pt",
        },
        planned_persistent_bytes={
            "selected_view_caches": 2048,
            "target_logits": 1024,
        },
        tap_stage_reservations=[tap],
        persistent_budget_bytes=10_000,
        filesystem_available_bytes=20_000,
        allocation_ram_bytes=10_000,
        transient_ram_bytes=128,
        diagnostic_budget=diagnostic_budget,
        safety_margin_bytes=256,
    )
    assert reservation["preflight_passed"]
    assert not reservation["offline_contextual_tokens_persisted"]
    diagnostic_size = write_bounded_json_diagnostic(
        campaign / "diagnostic.json",
        {"status": "warning", "non_gating": True},
        budget=diagnostic_budget,
    )
    assert 0 < diagnostic_size <= 256
    with pytest.raises(ValueError, match="file byte cap"):
        write_bounded_json_diagnostic(
            campaign / "too_large.json",
            {"payload": "x" * 1000},
            budget=diagnostic_budget,
        )

    paths = {}
    for name, contents in {
        "metrics.json": b"{}",
        "best.pt": b"best",
        "old.pt": b"old",
        "optimizer.pt": b"optimizer",
        "attention.npz": b"attention",
    }.items():
        path = campaign / name
        path.write_bytes(contents)
        paths[name] = path
    plan = build_retention_plan(
        campaign_root=campaign,
        candidates=[
            RetentionCandidate(
                "metrics.json",
                "json_metric",
                sha256_file(paths["metrics.json"]),
                True,
            ),
            RetentionCandidate(
                "best.pt",
                "screen_checkpoint",
                sha256_file(paths["best.pt"]),
                True,
                selected=True,
                screen_run_id="screen_a",
            ),
            RetentionCandidate(
                "old.pt",
                "screen_checkpoint",
                sha256_file(paths["old.pt"]),
                True,
                screen_run_id="screen_a",
            ),
            RetentionCandidate(
                "optimizer.pt",
                "optimizer_state",
                sha256_file(paths["optimizer.pt"]),
                True,
            ),
            RetentionCandidate(
                "attention.npz",
                "attention_diagnostic",
                sha256_file(paths["attention.npz"]),
                True,
            ),
        ],
    )
    report = execute_retention_plan(
        campaign,
        plan,
        output_path=campaign / "eviction_report.json",
    )
    assert report["evicted_bytes"] > 0
    assert paths["metrics.json"].exists()
    assert paths["best.pt"].exists()
    assert not paths["old.pt"].exists()
    assert not paths["optimizer.pt"].exists()
    assert not paths["attention.npz"].exists()


def _coordinate():
    parent_names = (
        "source_manifest_sha256",
        "unified_split_manifest_sha256",
        "train_identity_sha256",
        "hlt_source_sha256",
        "offline_source_sha256",
        "a0_checkpoint_sha256",
        "a0_config_sha256",
        "a0_query_tap_sha256",
        "a0_input_normalization_sha256",
        "offline_teacher_checkpoint_sha256",
        "offline_teacher_config_sha256",
        "offline_tap_spec_sha256",
        "generator_checkpoint_sha256",
        "normalizer_sha256",
    )
    return build_view_coordinate_binding(
        parent_hashes={name: _sha(name) for name in parent_names},
        coordinate_definition={
            "offline_tap_layer": "block7",
            "offline_tap_tensor_location": "post_residual_pre_pool",
            "cross_attention_config_sha256": _sha("cross"),
            "pair_feature_schema_sha256": _sha("pair"),
            "centering_policy": "masked_particle_mean",
            "bounded_coordinate_policy": "tanh",
            "rate_budget_policy": "registered",
            "null_token_policy": "learned",
            "bottleneck_width": 4,
        },
    )


def test_step9_acyclic_graph_and_concrete_lineage_reject_stale_cache():
    ids = {
        name: _sha(name)
        for name in (
            "source",
            "base",
            "tap",
            "generator",
            "normalizer",
            "coordinate",
            "cache",
            "consumer",
            "pview0",
            "logits",
            "predictor",
            "deployment",
        )
    }
    graph = build_particle_view_lineage_graph(
        [
            ParticleViewLineageNode("source", "source_split", ids["source"]),
            ParticleViewLineageNode(
                "base", "a0_offline_teacher", ids["base"], ("source",)
            ),
            ParticleViewLineageNode("tap", "tap_spec", ids["tap"], ("base",)),
            ParticleViewLineageNode(
                "generator", "generator", ids["generator"], ("tap",)
            ),
            ParticleViewLineageNode(
                "normalizer", "normalizer", ids["normalizer"], ("generator",)
            ),
            ParticleViewLineageNode(
                "coordinate",
                "coordinate",
                ids["coordinate"],
                ("source", "base", "tap", "generator", "normalizer"),
            ),
            ParticleViewLineageNode(
                "cache",
                "selected_view_cache",
                ids["cache"],
                ("source", "coordinate"),
            ),
            ParticleViewLineageNode(
                "consumer",
                "consumer",
                ids["consumer"],
                ("coordinate", "cache"),
            ),
            ParticleViewLineageNode(
                "pview0",
                "pview0_or_robust_consumer",
                ids["pview0"],
                ("coordinate", "consumer"),
            ),
            ParticleViewLineageNode(
                "logits",
                "target_logits",
                ids["logits"],
                ("coordinate", "cache", "consumer"),
            ),
            ParticleViewLineageNode(
                "predictor",
                "final_predictor",
                ids["predictor"],
                ("coordinate", "consumer", "logits"),
            ),
            ParticleViewLineageNode(
                "deployment",
                "deployment",
                ids["deployment"],
                ("coordinate", "consumer", "predictor"),
            ),
        ]
    )
    assert validate_particle_view_lineage_graph(graph)["node_count"] == 12
    with pytest.raises(ValueError, match="circular|descendant"):
        build_particle_view_lineage_graph(
            [
                ParticleViewLineageNode(
                    "source", "source_split", _sha("cycle-source"), ("later",)
                ),
                ParticleViewLineageNode(
                    "later", "deployment", _sha("cycle-later"), ("source",)
                ),
            ]
        )

    coordinate = _coordinate()
    checkpoint_sha = _sha("consumer-checkpoint")
    caches = []
    logits = []
    for split in ("train", "model_val_stop", "model_val_select"):
        split_sha = _sha(f"{split}-identity")
        cache = with_content_hash(
            {
                "contract": "particle_view_selected_cache_v1",
                "target_id": "selected_target",
                "split": split,
                "split_sha256": split_sha,
                "ordered_identity_sha256": _sha(f"{split}-ordered"),
                "coordinate_binding_sha256": coordinate["content_hash"],
                "normalizer_sha256": coordinate["parents"][
                    "normalizer_sha256"
                ],
                "generator_checkpoint_sha256": coordinate["parents"][
                    "generator_checkpoint_sha256"
                ],
                "array_file": f"{split}.npz",
                "array_file_sha256": _sha(f"{split}-array"),
                "logical_content_sha256": _sha(f"{split}-logical"),
                "view_shape": [4, 128, 4],
                "mask_shape": [4, 128],
                "dtype": "float32",
                "byte_order": "little",
                "invalid_particles_exactly_zero": True,
                "operation_order": SELECTED_VIEW_MATERIALIZATION_POLICY[
                    "operation_order"
                ],
                "live_publication_max_abs_tolerance": 1.0e-6,
                "final_test_cache_forbidden": True,
            }
        )
        caches.append(cache)
    consumer = with_content_hash(
        {
            "contract": "test_particle_view_consumer_registration_v1",
            "coordinate_binding_sha256": coordinate["content_hash"],
            "checkpoint_sha256": checkpoint_sha,
        }
    )
    for cache in caches:
        logits.append(
            with_content_hash(
                {
                    "contract": PARTICLE_VIEW_TARGET_LOGIT_CACHE_CONTRACT,
                    "lineage": {
                        "split_identity_sha256": cache[
                            "ordered_identity_sha256"
                        ],
                        "coordinate_binding_sha256": coordinate["content_hash"],
                        "selected_view_cache_sha256": cache["content_hash"],
                        "consumer_registration_sha256": consumer["content_hash"],
                        "consumer_checkpoint_sha256": checkpoint_sha,
                    },
                }
            )
        )
    predictor = with_content_hash(
        {
            "contract": PARTICLE_VIEW_DISTILLATION_REGISTRATION_CONTRACT,
            "lineage": {
                "coordinate_binding_sha256": coordinate["content_hash"],
                "consumer_registration_sha256": consumer["content_hash"],
                "consumer_checkpoint_sha256": checkpoint_sha,
                "train_target_logit_cache_sha256": logits[0]["content_hash"],
                "model_val_stop_target_logit_cache_sha256": logits[1][
                    "content_hash"
                ],
                "model_val_select_target_logit_cache_sha256": logits[2][
                    "content_hash"
                ],
            },
        }
    )
    audit = validate_coordinate_cache_consumer_logit_chain(
        coordinate_binding=coordinate,
        selected_view_cache_manifests=caches,
        consumer_registration=consumer,
        consumer_checkpoint_sha256=checkpoint_sha,
        target_logit_cache_manifests=logits,
        predictor_registration=predictor,
    )
    assert audit["authenticated"]
    stale = dict(logits[0])
    stale["lineage"] = dict(stale["lineage"])
    stale["lineage"]["selected_view_cache_sha256"] = _sha("stale")
    stale = with_content_hash(
        {key: value for key, value in stale.items() if key != "content_hash"}
    )
    with pytest.raises(ValueError, match="stale selected view"):
        validate_coordinate_cache_consumer_logit_chain(
            coordinate_binding=coordinate,
            selected_view_cache_manifests=caches,
            consumer_registration=consumer,
            consumer_checkpoint_sha256=checkpoint_sha,
            target_logit_cache_manifests=[stale, *logits[1:]],
        )


def test_step9_fairness_budget_accounting_reconciles_exact_source_totals():
    train_sha = _sha("train-identities")
    label_ledger = with_content_hash(
        {
            "contract": PARTICLE_VIEW_LABEL_EXPOSURE_LEDGER_CONTRACT,
            "train_identity_sha256": train_sha,
            "totals_retained_deployable_path": {
                "optimizer_steps": 30,
                "label_bearing_steps": 20,
                "labeled_examples_processed": 200,
                "ce_bearing_steps": 12,
                "teacher_kd_steps": 10,
                "view_supervision_steps": 10,
                "training_flops": 1234,
            },
            "totals_all_training": {
                "optimizer_steps": 50,
                "label_bearing_steps": 40,
                "labeled_examples_processed": 400,
                "ce_bearing_steps": 20,
                "teacher_kd_steps": 10,
                "view_supervision_steps": 10,
                "training_flops": 4321,
            },
        }
    )
    replica = {
        "seed": 101,
        "training_ledger_sha256": label_ledger["content_hash"],
        "a0_view_long_deploy_exact_ce_updates": 12,
        "a0_view_total_label_budget_exact_updates": 40,
        "label_bearing_updates": 40,
        "optimizer_updates_retained_path": 30,
        "training_flops_retained_path": 1234,
    }
    fairness = with_content_hash(
        {
            "contract": "particle_view_selected_path_fairness_ledger_v1",
            "train_identity_sha256": train_sha,
            "entries": [
                {
                    "configuration_id": "winner",
                    "fairness_entry_sha256": _sha("entry"),
                    "replicas": [replica],
                }
            ],
        }
    )
    accounting = build_fairness_budget_accounting(
        selected_path_fairness_ledger=fairness,
        label_exposure_ledgers=[label_ledger],
    )
    assert accounting["all_control_budgets_exactly_reconciled"]
    stale_fairness = dict(fairness)
    stale_fairness["entries"] = [
        {
            **fairness["entries"][0],
            "replicas": [
                {
                    **replica,
                    "a0_view_long_deploy_exact_ce_updates": 11,
                }
            ],
        }
    ]
    stale_fairness = with_content_hash(
        {
            key: value
            for key, value in stale_fairness.items()
            if key != "content_hash"
        }
    )
    with pytest.raises(ValueError, match="differs from source ledger"):
        build_fairness_budget_accounting(
            selected_path_fairness_ledger=stale_fairness,
            label_exposure_ledgers=[label_ledger],
        )


class TinyViewPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 2)

    def forward(self, features, lorentz_vectors, mask):
        values = features.transpose(1, 2)
        return self.projection(values) * mask[:, 0, :, None]


class TinyViewConsumer(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(5, 10)

    def forward(
        self,
        points,
        features,
        lorentz_vectors,
        mask,
        view,
        *,
        augment_clean_view=False,
    ):
        valid = mask[:, 0, :, None].to(features.dtype)
        features_last = features.transpose(1, 2)
        pooled = torch.cat((features_last, view), dim=-1)
        pooled = (pooled * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        return self.classifier(pooled)


def _hlt_batch(batch=4):
    torch.manual_seed(9009)
    mask = torch.ones(batch, 1, 128, dtype=torch.bool)
    mask[:, :, -5:] = False
    momentum = torch.randn(batch, 3, 128)
    energy = (
        momentum.square().sum(dim=1, keepdim=True) + 1.0
    ).sqrt()
    return {
        "points": torch.randn(batch, 2, 128),
        "features": torch.randn(batch, 3, 128),
        "lorentz_vectors": torch.cat((momentum, energy), dim=1),
        "mask": mask,
    }


def _exported_bundle(tmp_path: Path):
    predictor = TinyViewPredictor().eval()
    consumer = TinyViewConsumer().eval()
    inputs = _hlt_batch()
    predictor_config = {"contract": "tiny_predictor_v1", "view_dim": 2}
    consumer_config = {"contract": "tiny_consumer_v1", "classes": 10}
    source_bundle_sha = _sha("selected-source-bundle")
    deployment = build_particle_view_deployment_manifest(
        bundle_id="tiny_bundle",
        bundle_kind="frozen_consumer",
        winner_family="privileged_scientific",
        unified_split_manifest_sha256=_sha("split-manifest"),
        train_identity_sha256=_sha("train"),
        predictor_config_sha256=canonical_sha256(predictor_config),
        predictor_checkpoint_sha256=_sha("predictor-checkpoint"),
        consumer_config_sha256=canonical_sha256(consumer_config),
        consumer_checkpoint_sha256=_sha("consumer-checkpoint"),
        hlt_preprocessing_sha256=_sha("preprocessing"),
        hlt_schema_sha256=_sha("schema"),
        view_normalizer_sha256=_sha("view-normalizer"),
        coordinate_binding_sha256=_sha("coordinate"),
        resource_profile_sha256=_sha("resource"),
        source_commit=_sha("commit"),
        bottleneck_width=2,
    )
    export = export_hlt_only_particle_view_bundle(
        tmp_path / "export",
        predictor=predictor,
        consumer=consumer,
        exemplar_hlt_inputs=inputs,
        deployment_manifest=deployment,
        source_bundle_sha256=source_bundle_sha,
        predictor_config=predictor_config,
        consumer_config=consumer_config,
        lineage_graph_sha256=_sha("lineage"),
        fairness_ledger_sha256=_sha("fairness"),
        split_authorization_sha256=_sha("authorization"),
    )
    manifest_path = tmp_path / "export" / "particle_view_bundle_manifest.json"
    module = load_exported_particle_view_bundle(manifest_path)
    with torch.no_grad():
        reference = consumer(
            inputs["points"],
            inputs["features"],
            inputs["lorentz_vectors"],
            inputs["mask"],
            predictor(
                inputs["features"], inputs["lorentz_vectors"], inputs["mask"]
            ),
            augment_clean_view=False,
        )
        observed = module(*(inputs[name] for name in ("points", "features", "lorentz_vectors", "mask")))
    assert torch.allclose(reference, observed, atol=1e-6, rtol=0)
    fixture = write_hlt_reload_fixture(
        tmp_path / "fixture",
        exemplar_hlt_inputs=inputs,
        reference_logits=reference,
        bundle_export_sha256=export["content_hash"],
    )
    audit = run_fresh_process_reload_audit(
        bundle_manifest_path=manifest_path,
        fixture_manifest_path=tmp_path / "fixture" / "hlt_reload_fixture.json",
        output_path=tmp_path / "fresh_reload_audit.json",
    )
    assert fixture["contains_offline_inputs"] is False
    assert audit["passed"]
    return export, manifest_path, audit, inputs, source_bundle_sha


def test_step9_hlt_only_export_fresh_reload_final_permit_and_reporting(tmp_path):
    export, manifest_path, reload_audit, inputs, source_sha = _exported_bundle(
        tmp_path
    )
    winner = {
        "representative_bundle_sha256": source_sha,
        "representative_seed": 202,
    }
    selection = with_content_hash(
        {
            "contract": PARTICLE_VIEW_WINNER_SELECTION_CONTRACT,
            "selected_privileged_scientific_model": winner,
            "selected_pre_stage_g_hlt_deployable_model": winner,
        }
    )
    authorization = with_content_hash(
        {
            "contract": PARTICLE_VIEW_SPLIT_AUTHORIZATION_CONTRACT,
            "selection_sha256": selection["content_hash"],
            "final_test": {
                "split_sha256": _sha("final-test"),
                "authorized_bundles": [
                    {
                        "bundle_sha256": source_sha,
                        "seed": 202,
                        "role": "preselected_median_winner",
                        "winner_families": [
                            "selected_pre_stage_g_hlt_deployable_model",
                            "selected_privileged_scientific_model",
                        ],
                    }
                ],
                "authorized_fusion_recipes": [],
                "hlt_only_required": True,
                "stage_g_controls_forbidden": True,
            },
        }
    )
    stack = with_content_hash(
        {
            "contract": PARTICLE_VIEW_STACK_REPORT_CONTRACT,
            "selection_sha256": selection["content_hash"],
            "authorization_sha256": authorization["content_hash"],
            "selection_changed": False,
        }
    )
    permit = build_final_test_permit(
        selection=selection,
        split_authorization=authorization,
        stack_report=stack,
        bundle_export_manifests=[manifest_path],
        fresh_reload_audits=[reload_audit],
    )
    assert permit["authorized_export_count"] == 1
    final_batch = {**inputs, "labels": torch.tensor([0, 1, 2, 3])}
    final_output = tmp_path / permit["authorized_exports"][0]["result_file"]
    with pytest.raises(PermissionError, match="offline/oracle"):
        evaluate_authorized_final_test(
            bundle_manifest_path=manifest_path,
            permit=permit,
            loader=[{**final_batch, "true_view": torch.zeros(4, 128, 2)}],
            output_path=final_output,
        )
    result = evaluate_authorized_final_test(
        bundle_manifest_path=manifest_path,
        permit=permit,
        loader=[final_batch],
        output_path=final_output,
    )
    assert result["offline_inputs_loaded"] is False
    assert result["selected_view_cache_loaded"] is False
    with pytest.raises(FileExistsError, match="already exists"):
        evaluate_authorized_final_test(
            bundle_manifest_path=manifest_path,
            permit=permit,
            loader=[final_batch],
            output_path=final_output,
        )

    sections = {name: [] for name in PARTICLE_VIEW_REPORT_SECTIONS}
    sections["offline_oracle_diagnostics"].append(
        {
            "row_id": "oracle",
            "artifact_sha256": _sha("oracle"),
            "split": "model_val_select",
            "metrics": {"accuracy": 0.9},
            "requires_oracle": True,
            "deployable": False,
        }
    )
    sections["frozen_consumer_hlt_deployable"].append(
        {
            "row_id": "deployable",
            "artifact_sha256": export["content_hash"],
            "split": "final_test",
            "metrics": result["metrics"],
            "requires_oracle": False,
            "deployable": True,
        }
    )
    report = build_separated_campaign_report(
        sections=sections,
        selection_sha256=selection["content_hash"],
        stack_report_sha256=stack["content_hash"],
        fairness_ledger_sha256=_sha("fairness"),
        label_exposure_ledger_sha256=_sha("label-ledger"),
        storage_reservation_sha256=_sha("storage"),
        lineage_graph_sha256=_sha("lineage"),
        deployment_export_sha256=[export["content_hash"]],
        aggregate_warning_summary_sha256=_sha("warnings"),
        final_test_permit_sha256=permit["content_hash"],
        final_test_result_sha256=[result["content_hash"]],
    )
    assert report["oracle_and_deployable_rows_separated"]
    bad_sections = {name: list(rows) for name, rows in sections.items()}
    bad_sections["offline_oracle_diagnostics"] = [
        {
            **bad_sections["offline_oracle_diagnostics"][0],
            "split": "final_test",
        }
    ]
    with pytest.raises(PermissionError, match="final_test"):
        build_separated_campaign_report(
            sections=bad_sections,
            selection_sha256=selection["content_hash"],
            stack_report_sha256=stack["content_hash"],
            fairness_ledger_sha256=_sha("fairness"),
            label_exposure_ledger_sha256=_sha("label-ledger"),
            storage_reservation_sha256=_sha("storage"),
            lineage_graph_sha256=_sha("lineage"),
            deployment_export_sha256=[export["content_hash"]],
            aggregate_warning_summary_sha256=_sha("warnings"),
            final_test_permit_sha256=permit["content_hash"],
            final_test_result_sha256=[result["content_hash"]],
        )
