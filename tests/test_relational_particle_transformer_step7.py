from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from jetclass_fresh.jetclass_data import JetIdentity
from jetclass_fresh.part_inputs import build_particle_transformer_inputs_from_tokens
from teacher_logit_reco.relational_part.data import RelationalJetDataset
from teacher_logit_reco.relational_part import (
    UnaryEndpointFeatureBuilder,
    build_angular_tree_resource_contract,
    build_normalization_contract,
    build_raw_input_schema_contract,
    build_reference_tree,
    fit_region_normalization,
    fit_relation_normalization,
)
from teacher_logit_reco.relational_part.contracts import with_content_hash
from teacher_logit_reco.relational_part.evaluation import (
    FINAL_EVALUATION_CONTRACT,
    evaluate_logits,
    evaluate_locked_finalist,
    load_final_predictions,
    paired_prediction_statistics,
)
from teacher_logit_reco.relational_part.registry import (
    build_confirmation_architecture_registry,
    build_relation_family_registry,
    build_screening_registry,
)
from teacher_logit_reco.relational_part.reporting import (
    build_relational_part_report,
    render_relational_part_markdown,
)
from teacher_logit_reco.relational_part.selection import (
    CONFIRMATION_SUMMARY_CONTRACT,
    LOCKED_FINALISTS_CONTRACT,
    aggregate_confirmation,
    build_confirmation_registry,
    build_screening_summary,
    build_selected_union_model_contract,
    validate_locked_finalists,
)
from teacher_logit_reco.relational_part.semantic_controls import (
    SEMANTIC_PERTURBATION_CONTRACT,
    SEMANTIC_RUN_DIAGNOSTICS_CONTRACT,
    build_semantic_diagnostics_bundle,
    build_semantic_perturbation_artifact,
    build_unary_control_registry,
    directional_swap_relations,
    evaluate_semantic_perturbations,
    semantic_diagnostic_rows,
    validate_semantic_diagnostics_bundle,
    select_unary_widths,
    zero_relation_family,
    zero_region_resolution,
    unary_adapter_parameter_count,
    within_jet_shuffled_relations,
    wrong_event_relations,
)
from teacher_logit_reco.relational_part.workflow import reject_final_test_paths
from scripts.run_relational_part_semantic_perturbation import (
    _load_reusable_artifact,
)
import scripts.run_relational_part_semantic_perturbation as semantic_runner
from teacher_logit_reco.relational_part import write_immutable_json


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _result_lineage(
    run_id: str,
    seed: int,
    hashes: dict[str, str],
    *,
    campaign: str = "5" * 64,
    split: str = "6" * 64,
) -> dict:
    return {
        "checkpoint_registration_sha256": _digest(
            f"registration-{run_id}-{seed}"
        ),
        "val_select_metrics_sha256": _digest(f"metrics-{run_id}-{seed}"),
        "model_contract_sha256": _digest(f"model-{run_id}"),
        "training_contract_sha256": _digest(f"training-{run_id}-{seed}"),
        "run_registry_sha256": _digest(f"registry-{run_id}-{seed}"),
        "relation_registry_sha256": _digest("relation-registry"),
        "lineage_hashes": {
            "campaign_spec": campaign,
            "split_manifest": split,
            "hlt_model_train": hashes["model_train"],
            "hlt_model_val": hashes["model_val"],
            "hlt_stack_val": hashes["stack_val"],
        },
        "lineage_authenticated": True,
    }


def _screening():
    relation = build_relation_family_registry()
    registry = build_screening_registry(
        relation_registry_sha256=relation["content_hash"]
    )
    hashes = {
        "model_train": "1" * 64,
        "model_val": "2" * 64,
        "stack_val": "3" * 64,
        "final_test": "4" * 64,
    }
    results = []
    for index, row in enumerate(registry["rows"]):
        run_id = row["run_id"]
        accuracy = 0.79 - index * 1e-5
        if run_id == "RPT_BASE":
            accuracy = 0.80
        if run_id == "RPT_PT":
            accuracy = 0.795
        if run_id == "RPT_PID":
            accuracy = 0.7949
        results.append(
            {
                "run_id": run_id,
                "seed": 101,
                "configuration_role": row["configuration_role"],
                "checkpoint_sha256": _digest(run_id),
                "parameter_count": 1000 + index,
                **_result_lineage(run_id, 101, hashes),
                "val_select": {
                    "split": "val_select",
                    "accuracy": accuracy,
                    "cross_entropy": 1.0 + index / 100,
                },
            }
        )
    summary = build_screening_summary(
        screening_registry=registry,
        results=results,
        campaign_spec_sha256="5" * 64,
        split_manifest_sha256="6" * 64,
        hlt_cache_hashes=hashes,
        results_envelope_sha256="7" * 64,
        expected_common_lineage_hashes=results[0]["lineage_hashes"],
    )
    architecture = build_confirmation_architecture_registry(
        relation_registry_sha256=relation["content_hash"],
        screening_registry_sha256=registry["content_hash"],
    )
    confirmation = build_confirmation_registry(
        screening_registry=registry,
        architecture_registry=architecture,
        screening_summary=summary,
    )
    return relation, registry, results, hashes, summary, confirmation


def test_screening_negative_campaign_and_confirmation_registry_are_complete() -> None:
    _, _, _, _, summary, confirmation = _screening()
    assert summary["screening_gain_positive"] is False
    assert summary["best_available_run_id"] == "RPT_PT"
    assert summary["selected_union_families"] == ["PT", "PID"]
    assert confirmation["selected_union"]["synthesized"] is True
    ids = {row["run_id"] for row in confirmation["rows"]}
    assert {
        "RPT_PT",
        "RPT_TRACK",
        "RPT_PID",
        "RPT_CHARGE",
        "RPT_DENSITY",
        "RPT_REGION",
        "RPT_BASE_WIDE_MAX",
        "RPT_FULL_ZERO_REL",
        "RPT_BASE_LAYERWISE",
        "RPT_BASE_EDGEVALUE",
        "RPT_SELECTED_LAYERWISE",
        "RPT_SELECTED_EDGEVALUE",
        "RPT_SELECTED_UNION",
    }.issubset(ids)
    union = build_selected_union_model_contract(
        confirmation_registry=confirmation,
        relation_normalization_sha256="1" * 64,
        relation_registry_sha256="2" * 64,
        pair_base_sha256="3" * 64,
        family_contract_sha256={
            family: _digest(family)
            for family in confirmation["selected_union"]["families"]
        },
        weaver_runtime_sha256="4" * 64,
        global_determinism_sha256="5" * 64,
        region_normalization_sha256="6" * 64,
    )
    assert union["run_id"] == "RPT_SELECTED_UNION"
    assert union["new_relation_families"] == confirmation[
        "selected_union"
    ]["families"]


def test_screening_rejects_stale_cross_campaign_checkpoint_lineage() -> None:
    _, registry, results, hashes, _, _ = _screening()
    stale = [dict(row) for row in results]
    stale[0] = {
        **stale[0],
        "lineage_hashes": {
            **stale[0]["lineage_hashes"],
            "campaign_spec": "0" * 64,
        },
    }
    with pytest.raises(ValueError, match="campaign_spec"):
        build_screening_summary(
            screening_registry=registry,
            results=stale,
            campaign_spec_sha256="5" * 64,
            split_manifest_sha256="6" * 64,
            hlt_cache_hashes=hashes,
            results_envelope_sha256="7" * 64,
            expected_common_lineage_hashes=results[0]["lineage_hashes"],
        )


def test_confirmation_negative_campaign_locks_and_stale_seal_fails_closed() -> None:
    _, _, _, hashes, _, confirmation = _screening()
    results = []
    for run_index, row in enumerate(confirmation["rows"]):
        for seed_index, seed in enumerate((101, 202, 303)):
            if row["seed_101"]["mode"] == "reuse_hash_exact" and seed == 101:
                checkpoint = row["seed_101"]["expected_checkpoint_sha256"]
            else:
                checkpoint = _digest(f"{row['run_id']}-{seed}")
            accuracy = 0.80 + seed_index * 1e-4
            if row["run_id"] != "RPT_BASE":
                accuracy -= 0.001 + run_index * 1e-6
            results.append(
                {
                    "run_id": row["run_id"],
                    "seed": seed,
                    "configuration_role": row["configuration_role"],
                    "relational_selection_eligible": row[
                        "relational_selection_eligible"
                    ],
                    "checkpoint_sha256": checkpoint,
                    "parameter_count": 1000 + run_index,
                    **_result_lineage(row["run_id"], seed, hashes),
                    "val_select": {
                        "split": "val_select",
                        "accuracy": accuracy,
                        "cross_entropy": 0.5 + run_index / 1000,
                        "per_class_efficiency": {
                            name: accuracy
                            for name in (
                                "QCD", "Hbb", "Hcc", "Hgg", "H4q",
                                "Hqql", "Zqq", "Wqq", "Tbqq", "Tbl",
                            )
                        },
                    },
                }
            )
    unary = [
        {
            "run_id": "RPT_SELECTED_UNARY",
            "seed": seed,
            "configuration_role": "semantic_control",
            "relational_selection_eligible": False,
            "checkpoint_sha256": _digest(f"unary-{seed}"),
            "parameter_count": 1100,
            **_result_lineage("RPT_SELECTED_UNARY", seed, hashes),
            "val_select": {
                "split": "val_select",
                "accuracy": 0.799,
                "cross_entropy": 0.51,
            },
        }
        for seed in (101, 202, 303)
    ]
    with pytest.raises(ValueError, match="unary control"):
        aggregate_confirmation(
            confirmation_registry=confirmation,
            results=results,
            campaign_spec_sha256="5" * 64,
            split_manifest_sha256="6" * 64,
            hlt_cache_hashes=hashes,
            results_envelope_sha256="9" * 64,
            expected_common_lineage_hashes=results[0]["lineage_hashes"],
        )
    preliminary, preliminary_lock = aggregate_confirmation(
        confirmation_registry=confirmation,
        results=results,
        campaign_spec_sha256="5" * 64,
        split_manifest_sha256="6" * 64,
        hlt_cache_hashes=hashes,
        results_envelope_sha256="9" * 64,
        expected_common_lineage_hashes=results[0]["lineage_hashes"],
        seal_finalists=False,
    )
    assert preliminary_lock is None
    summary, lock = aggregate_confirmation(
        confirmation_registry=confirmation,
        results=results,
        campaign_spec_sha256="5" * 64,
        split_manifest_sha256="6" * 64,
        hlt_cache_hashes=hashes,
        results_envelope_sha256="9" * 64,
        expected_common_lineage_hashes=results[0]["lineage_hashes"],
        semantic_unary_results=unary,
        unary_results_envelope_sha256="a" * 64,
        semantic_perturbation_sha256="7" * 64,
        unary_control_registry_sha256="8" * 64,
    )
    assert summary["content_hash"] == preliminary["content_hash"]
    assert summary["confirmation_gain_positive"] is False
    assert summary["nominal_relational_winner_id"] != "RPT_BASE"
    track_rows = [
        row for row in summary["rows"]
        if "TRACK" in row["new_relation_families"]
    ]
    assert track_rows
    assert all(
        set(row["track_required_class_gains"])
        == {"Hbb", "Hcc", "Tbqq", "Tbl"}
        for row in track_rows
    )
    assert lock["final_test_used_for_selection"] is False
    validate_locked_finalists(
        lock,
        campaign_spec_sha256="5" * 64,
        split_manifest_sha256="6" * 64,
        hlt_cache_hashes=hashes,
    )
    stale = dict(hashes)
    stale["final_test"] = "7" * 64
    with pytest.raises(ValueError, match="HLT-cache"):
        validate_locked_finalists(
            lock,
            campaign_spec_sha256="5" * 64,
            split_manifest_sha256="6" * 64,
            hlt_cache_hashes=stale,
        )
    with pytest.raises(ValueError, match="may not access final_test"):
        reject_final_test_paths((Path("campaign") / "final_test" / "metrics.json",))


def test_semantic_perturbations_preserve_base_and_exact_domains() -> None:
    torch.manual_seed(3)
    pairs = torch.randn(4, 7, 4, 4)
    mask = torch.tensor(
        [
            [[True, True, True, False]],
            [[True, True, True, False]],
            [[True, True, False, False]],
            [[True, True, False, False]],
        ]
    )
    pair_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
    pairs[:, 4:] = pairs[:, 4:].masked_fill(~pair_mask, 0)
    identities = [f"event-{index}" for index in range(4)]
    shuffled, shuffle_diag = within_jet_shuffled_relations(
        pairs, mask, identities
    )
    assert torch.equal(shuffled[:, :4], pairs[:, :4])
    assert shuffle_diag["fixed_point_count"] == 0
    torch.testing.assert_close(
        shuffled[:, 4:].flatten().sort().values,
        pairs[:, 4:].flatten().sort().values,
    )
    swapped, swap_diag = directional_swap_relations(pairs, mask)
    assert torch.equal(swapped[:, :4], pairs[:, :4])
    assert swap_diag["base4_unchanged"] is True

    vectors = torch.randn(4, 4, 4)
    tokens = torch.randn(4, 4, 14)
    tokens[:, :, 0] = torch.tensor([4.0, 3.0, 2.0, 1.0])
    wrong, diagnostic = wrong_event_relations(
        pairs, mask, identities, vectors, tokens
    )
    assert torch.equal(wrong[:, :4], pairs[:, :4])
    assert diagnostic["fixed_event_count"] == 0
    assert diagnostic["deranged_event_count"] == 4
    assert diagnostic["exact_valid_multiplicity"] is True


class _SemanticModel(torch.nn.Module):
    def __init__(self, families=()):
        super().__init__()
        self.families = tuple(families)

    def forward(
        self,
        points,
        features,
        lorentz_vectors,
        mask,
        raw_tokens,
        pair_transform=None,
    ):
        del points, features
        batch, _, particles = lorentz_vectors.shape
        channel_count = 10 if self.families == ("CHARGE",) else 7
        pairs = lorentz_vectors.new_zeros(
            batch, channel_count, particles, particles
        )
        pairs[:, 4] = (
            raw_tokens[:, :, 0].unsqueeze(-1)
            - raw_tokens[:, :, 0].unsqueeze(-2)
        )
        pairs[:, 5] = raw_tokens[:, :, 1].unsqueeze(-1)
        pairs[:, 6] = raw_tokens[:, :, 2].unsqueeze(-2)
        pair_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
        pairs[:, 4:] = pairs[:, 4:].masked_fill(~pair_mask, 0)
        if pair_transform is not None:
            pairs = pair_transform(
                pairs,
                mask=mask,
                features=None,
                lorentz_vectors=lorentz_vectors,
                raw_tokens=raw_tokens,
                region_trees=None,
            )
        return pairs.new_zeros(batch, 10)


def test_semantic_evaluator_uses_exact_global_multiplicity_strata() -> None:
    jets, particles = 31, 3
    tokens = np.zeros((jets, particles, 14), dtype=np.float32)
    mask = np.ones((jets, particles), dtype=bool)
    mask[-1, 1:] = False
    tokens[:, :, 0] = np.asarray([3.0, 2.0, 1.0])
    tokens[:, :, 1] = np.asarray([0.1, 0.2, 0.3])
    tokens[:, :, 2] = np.asarray([-0.1, 0.2, 0.4])
    tokens[:, :, 3] = tokens[:, :, 0] * np.cosh(tokens[:, :, 1])
    tokens[:, :, 5] = 1
    view = SimpleNamespace(
        tokens=tokens,
        mask=mask,
        labels=np.arange(jets, dtype=np.int64) % 10,
        jet_ids=[
            JetIdentity(file="semantic.root", entry=index, label=index % 10)
            for index in range(jets)
        ],
        split="stack_val",
    )
    dataset = RelationalJetDataset(view)
    metrics, diagnostics = evaluate_semantic_perturbations(
        _SemanticModel(), SimpleNamespace(dataset=dataset)
    )
    assert set(metrics) == {
        "full_model",
        "within_jet_shuffled_relations",
        "wrong_event_relations",
        "directional_swap",
    }
    assert diagnostics["within_jet_shuffled_relations"][
        "excluded_event_count"
    ] == 1
    assert diagnostics["wrong_event_relations"]["excluded_event_count"] == 1
    assert diagnostics["wrong_event_relations"]["fixed_point_count"] == 0


def test_charge_single_gets_matched_zero_charge_control() -> None:
    jets, particles = 20, 3
    tokens = np.zeros((jets, particles, 14), dtype=np.float32)
    mask = np.ones((jets, particles), dtype=bool)
    tokens[:, :, 0] = np.asarray([3.0, 2.0, 1.0])
    tokens[:, :, 1] = np.asarray([0.1, 0.2, 0.3])
    tokens[:, :, 2] = np.asarray([-0.1, 0.2, 0.4])
    tokens[:, :, 3] = tokens[:, :, 0] * np.cosh(tokens[:, :, 1])
    tokens[:, :, 5] = 1
    dataset = RelationalJetDataset(
        SimpleNamespace(
            tokens=tokens,
            mask=mask,
            labels=np.arange(jets, dtype=np.int64) % 10,
            jet_ids=[
                JetIdentity(
                    file="charge-semantic.root",
                    entry=index,
                    label=index % 10,
                )
                for index in range(jets)
            ],
            split="stack_val",
        )
    )
    metrics, diagnostics = evaluate_semantic_perturbations(
        _SemanticModel(("CHARGE",)),
        SimpleNamespace(dataset=dataset),
    )
    assert "family_dropout_CHARGE" in metrics
    assert "family_dropout_CHARGE" in diagnostics
    assert (
        "charge_gain_conditioned_on_pid_vs_charge_dropout"
        in metrics["full_model"]
    )


def test_semantic_bundle_requires_all_applicable_seed101_rows(
    tmp_path: Path,
) -> None:
    def row(run_id, families, role="scientific_finalist"):
        return {
            "run_id": run_id,
            "configuration_role": role,
            "new_relation_families": list(families),
            "checkpoint_hashes": {"101": _digest(f"checkpoint-{run_id}")},
            "checkpoint_registration_hashes": {
                "101": _digest(f"registration-{run_id}")
            },
            "model_contract_sha256": _digest(f"model-{run_id}"),
        }

    summary = with_content_hash(
        {
            "contract": CONFIRMATION_SUMMARY_CONTRACT,
            "schema_version": 3,
            "nominal_relational_winner_id": "RPT_PT",
            "rows": [
                row("RPT_PT", ("PT",)),
                row("RPT_CHARGE", ("CHARGE",)),
                row("RPT_PT_PID", ("PT", "PID")),
                row("RPT_REGION", ("REGION",)),
                row(
                    "RPT_FULL_ZERO_REL",
                    ("PT", "TRACK", "PID", "CHARGE", "DENSITY", "REGION"),
                    role="shape_control",
                ),
            ],
        }
    )
    selected = semantic_diagnostic_rows(summary)
    assert [value["run_id"] for value in selected] == [
        "RPT_PT",
        "RPT_PT_PID",
        "RPT_REGION",
    ]

    artifacts = []
    for value in selected:
        required_metrics = {
            "full_model": {},
            "within_jet_shuffled_relations": {},
            "wrong_event_relations": {},
            "directional_swap": {},
        }
        if len(value["families"]) >= 2:
            required_metrics.update(
                {
                    f"family_dropout_{family}": {}
                    for family in value["families"]
                }
            )
        if "REGION" in value["families"]:
            required_metrics.update(
                {
                    f"region_resolution_dropout_K{k}": {}
                    for k in (2, 4, 8)
                }
            )
        artifacts.append(
            build_semantic_perturbation_artifact(
                run_id=value["run_id"],
                families=value["families"],
                checkpoint_sha256=value["checkpoint_sha256"],
                checkpoint_registration_sha256=value[
                    "checkpoint_registration_sha256"
                ],
                model_contract_sha256=value["model_contract_sha256"],
                confirmation_summary_sha256=summary["content_hash"],
                metrics=required_metrics,
                diagnostics={
                    name: {} for name in required_metrics
                },
            )
        )
    assert all(
        value["contract"] == SEMANTIC_RUN_DIAGNOSTICS_CONTRACT
        for value in artifacts
    )
    with pytest.raises(ValueError, match="lack applicable"):
        build_semantic_diagnostics_bundle(
            confirmation_summary=summary,
            run_artifacts=artifacts[:-1],
        )
    bundle = build_semantic_diagnostics_bundle(
        confirmation_summary=summary,
        run_artifacts=artifacts,
    )
    assert bundle["contract"] == SEMANTIC_PERTURBATION_CONTRACT
    assert bundle["coverage_complete"] is True
    tampered = dict(bundle)
    tampered.pop("content_hash")
    tampered["runs"] = {
        name: dict(entry) for name, entry in bundle["runs"].items()
    }
    tampered["runs"]["RPT_PT"]["artifact_sha256"] = "0" * 64
    tampered = with_content_hash(tampered)
    with pytest.raises(ValueError, match="nested semantic hash"):
        validate_semantic_diagnostics_bundle(
            tampered, confirmation_summary=summary
        )
    reusable_path = tmp_path / "RPT_PT.json"
    write_immutable_json(reusable_path, artifacts[0])
    reused = _load_reusable_artifact(
        reusable_path,
        row=selected[0],
        confirmation_summary=summary,
    )
    assert reused["content_hash"] == artifacts[0]["content_hash"]
    stale_row = dict(selected[0])
    stale_row["checkpoint_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="coverage or lineage"):
        _load_reusable_artifact(
            reusable_path,
            row=stale_row,
            confirmation_summary=summary,
        )


def test_all_reused_semantic_restart_rejects_source_drift_before_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = with_content_hash(
        {
            "contract": "test_campaign",
            "schema_version": 1,
            "source": {
                "commit": "0" * 40,
                "status_sha256": "0" * 64,
                "dirty": False,
            },
        }
    )
    write_immutable_json(tmp_path / "campaign_spec.json", campaign)
    reusable = (
        tmp_path
        / "selection"
        / "semantic_controls"
        / "per_run"
        / "RPT_PT.json"
    )
    reusable.parent.mkdir(parents=True)
    reusable.write_text('{"authenticated_reusable_artifact": true}\n')

    def reject_drift(*args, **kwargs):
        raise ValueError(
            "active repository source snapshot differs from campaign_spec.json"
        )

    def reuse_must_not_be_considered(*args, **kwargs):
        pytest.fail("semantic reuse was considered before source validation")

    monkeypatch.setattr(
        semantic_runner, "validate_campaign_source", reject_drift
    )
    monkeypatch.setattr(
        semantic_runner, "_load_reusable_artifact", reuse_must_not_be_considered
    )
    with pytest.raises(ValueError, match="source snapshot differs"):
        semantic_runner.main(["--campaign-root", str(tmp_path)])
    assert reusable.is_file()


def test_combination_family_dropout_zeros_only_selected_channels() -> None:
    pairs = torch.randn(2, 20, 4, 4)
    mask = torch.tensor(
        [[[True, True, True, False]], [[True, True, False, False]]]
    )
    pairs = pairs.masked_fill(
        ~(mask.unsqueeze(-1) & mask.unsqueeze(-2)), 0.0
    )
    output, detail = zero_relation_family(
        pairs,
        mask,
        families=("PT", "PID"),
        family="PT",
    )
    assert torch.equal(output[:, :4], pairs[:, :4])
    assert output[:, 4:12].eq(0).all()
    valid_pairs = mask.unsqueeze(-1) & mask.unsqueeze(-2)
    torch.testing.assert_close(
        output[:, 12:].masked_select(valid_pairs.expand(-1, 8, -1, -1)),
        pairs[:, 12:].masked_select(valid_pairs.expand(-1, 8, -1, -1)),
    )
    assert detail["zeroed_family"] == "PT"


def test_region_resolution_ablation_reencodes_only_region_slice() -> None:
    class RegionEncoder:
        def __init__(self):
            self.disabled = None

        def __call__(
            self,
            raw_tokens,
            mask,
            region_trees,
            *,
            disabled_resolutions,
        ):
            del region_trees
            self.disabled = tuple(disabled_resolutions)
            batch, particles = raw_tokens.shape[:2]
            output = raw_tokens.new_full(
                (batch, 12, particles, particles),
                float(disabled_resolutions[0]),
            )
            return output.masked_fill(
                ~(mask.unsqueeze(-1) & mask.unsqueeze(-2)), 0.0
            )

    encoder = RegionEncoder()
    model = SimpleNamespace(
        families=("REGION",),
        pair_builder=SimpleNamespace(encoders={"REGION": encoder}),
    )
    pairs = torch.randn(2, 16, 3, 3)
    mask = torch.tensor(
        [[[True, True, False]], [[True, True, True]]]
    )
    pairs = pairs.masked_fill(
        ~(mask.unsqueeze(-1) & mask.unsqueeze(-2)), 0.0
    )
    output, detail = zero_region_resolution(
        pairs,
        mask,
        model=model,
        raw_tokens=torch.zeros(2, 3, 14),
        region_trees=({}, {}),
        resolution=4,
    )
    assert encoder.disabled == (4,)
    assert torch.equal(output[:, :4], pairs[:, :4])
    assert output[:, 4:].masked_select(
        (mask.unsqueeze(-1) & mask.unsqueeze(-2)).expand(-1, 12, -1, -1)
    ).eq(4).all()
    assert detail["zeroed_resolution"] == 4
    assert detail["normalized_zeroed_channel_indices"] == [
        1, 14, 15, 16, 17, 18, 19, 28, 29, 34, 35, 39
    ]


def test_unary_parameter_search_and_registry_are_exact() -> None:
    target = unary_adapter_parameter_count(3, 20, 10)
    selected = select_unary_widths(
        families=("PT",),
        reference_incremental_parameters=target,
    )
    assert selected["relative_incremental_mismatch"] <= 0.02
    registry = build_unary_control_registry(
        nominal_winner_run_id="RPT_SELECTED_EDGEVALUE",
        unary_reference_run_id="RPT_PT",
        families=("PT",),
        reference_incremental_parameters=target,
        reference_total_parameters=10_000 + target,
        base_total_parameters=10_000,
        confirmation_summary_sha256="8" * 64,
        relation_normalization_sha256="9" * 64,
    )
    assert registry["run_id"] == "RPT_SELECTED_UNARY"
    assert registry["new_pairwise_channels"] == 0
    assert registry["search"]["relative_incremental_mismatch"] <= 0.02


def test_unary_builder_enumerates_every_endpoint_family_without_padding() -> None:
    jets, particles = 4, 6
    tokens = np.zeros((jets, particles, 14), dtype=np.float32)
    mask = np.ones((jets, particles), dtype=bool)
    mask[-1, -1] = False
    for row in range(jets):
        pt = np.asarray([12, 8, 5, 3, 2, 1], dtype=np.float32)
        eta = np.linspace(-0.4, 0.5, particles, dtype=np.float32)
        phi = np.linspace(-2.5, 2.4, particles, dtype=np.float32)
        tokens[row, :, 0] = pt + row * 0.1
        tokens[row, :, 1] = eta
        tokens[row, :, 2] = phi
        tokens[row, :, 3] = tokens[row, :, 0] * np.cosh(eta)
        tokens[row, :, 4] = np.asarray([-1, 0, 1, -1, 0, 1])
        for particle in range(particles):
            tokens[row, particle, 5 + particle % 5] = 1
        tokens[row, :, 10] = np.linspace(-0.02, 0.03, particles)
        tokens[row, :, 11] = 0.002 + row * 0.0001
        tokens[row, :, 12] = np.linspace(-0.03, 0.05, particles)
        tokens[row, :, 13] = 0.004 + row * 0.0001
    identities = [
        JetIdentity(file="unary.root", entry=row, label=row)
        for row in range(jets)
    ]
    relation_registry = build_relation_family_registry()
    normalization = fit_relation_normalization(
        tokens,
        mask,
        identities,
        normalization_contract=build_normalization_contract(
            split_binding_sha256="1" * 64
        ),
        relation_registry=relation_registry,
        raw_input_schema=build_raw_input_schema_contract(),
        hlt_binding_sha256="2" * 64,
        source_manifest_sha256="3" * 64,
        hlt_model_train_content_sha256="4" * 64,
    )
    part = build_particle_transformer_inputs_from_tokens(
        tokens, mask, source_view="fixed_hlt"
    )
    vectors = np.asarray(part.pf_vectors).transpose(0, 2, 1)
    trees = [
        build_reference_tree(vectors[row], tokens[row], mask[row])
        for row in range(jets)
    ]
    region = fit_region_normalization(
        tokens,
        mask,
        identities,
        trees,
        relation_normalization_artifact=normalization,
        angular_tree_resource_sha256=build_angular_tree_resource_contract(
            split_binding_sha256="1" * 64
        )["content_hash"],
    )
    builder = UnaryEndpointFeatureBuilder(
        ("PT", "TRACK", "PID", "CHARGE", "DENSITY", "REGION"),
        normalization_artifact=normalization,
        region_normalization_artifact=region,
    )
    output = builder(
        torch.from_numpy(part.pf_features),
        torch.from_numpy(part.pf_vectors),
        torch.from_numpy(part.pf_mask),
        torch.from_numpy(tokens),
        trees,
    )
    assert builder.output_width == 63
    assert output.shape == (jets, particles, 63)
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output[-1, -1]) == 0


class _FinalModel(torch.nn.Module):
    def forward(self, points, features, lorentz_vectors, mask):
        del points, lorentz_vectors, mask
        return features[:, :10, 0]


def _final_loader():
    labels = torch.arange(20) % 10
    features = torch.zeros(20, 17, 2)
    features[torch.arange(20), labels, 0] = 2
    return [
        {
            "points": torch.zeros(20, 2, 2),
            "features": features,
            "lorentz_vectors": torch.ones(20, 4, 2),
            "mask": torch.ones(20, 1, 2, dtype=torch.bool),
            "labels": labels,
            "event_identities": [f"final-{index}" for index in range(20)],
        }
    ]


def test_sealed_final_evaluation_predictions_and_paired_bootstrap(tmp_path: Path) -> None:
    hashes = {
        "model_train": "1" * 64,
        "model_val": "2" * 64,
        "stack_val": "3" * 64,
        "final_test": "4" * 64,
    }
    checkpoint = "a" * 64
    lineage = {
        "campaign_spec": "5" * 64,
        "split_manifest": "6" * 64,
        "hlt_model_train": hashes["model_train"],
        "hlt_model_val": hashes["model_val"],
        "hlt_stack_val": hashes["stack_val"],
    }
    registration = with_content_hash(
        {
                "contract": "relational_part_checkpoint_registration_v2",
                "schema_version": 2,
            "run_id": "RPT_BASE",
            "seed": 101,
            "checkpoint_sha256": checkpoint,
            "model_contract_sha256": "d" * 64,
            "lineage_hashes": lineage,
            "val_select_used_for_checkpoint_selection": False,
            "hlt_only_inference": True,
            "offline_or_teacher_required": False,
            "parameter_and_flop_profile": {
                "trainable_parameters": 1,
                "forward_flops_per_event": 2,
                "latency_ms": {"mean": 1, "median": 1},
                "peak_incremental_device_memory_bytes": 3,
            },
        }
    )
    def locked_fields(run_id: str):
        registrations = {
            str(seed): _digest(f"locked-registration-{run_id}-{seed}")
            for seed in (101, 202, 303)
        }
        if run_id == "RPT_BASE":
            registrations["101"] = registration["content_hash"]
        return {
            "checkpoint_registration_hashes": registrations,
            "val_select_metrics_hashes": {
                str(seed): _digest(f"locked-metrics-{run_id}-{seed}")
                for seed in (101, 202, 303)
            },
            "model_contract_sha256": (
                "d" * 64 if run_id == "RPT_BASE" else _digest(f"model-{run_id}")
            ),
            "lineage_hashes": lineage,
            "lineage_authenticated": True,
        }
    lock = with_content_hash(
        {
                "contract": LOCKED_FINALISTS_CONTRACT,
                "schema_version": 3,
            "campaign_spec_sha256": "5" * 64,
            "split_manifest_sha256": "6" * 64,
            "hlt_cache_hashes": hashes,
            "confirmation_registry_sha256": "7" * 64,
            "confirmation_summary_sha256": "8" * 64,
            "semantic_perturbation_sha256": "9" * 64,
            "unary_control_registry_sha256": "0" * 64,
            "confirmation_results_envelope_sha256": "1" * 64,
            "unary_results_envelope_sha256": "2" * 64,
            "all_selection_lineage_authenticated": True,
            "baseline_id": "RPT_BASE",
            "evaluation_rows": [
                {
                    "run_id": "RPT_BASE",
                    "configuration_role": "reference_baseline",
                    "relational_selection_eligible": False,
                    "mean_matched_seed_accuracy_difference": 0.0,
                    "checkpoint_hashes": {
                        "101": checkpoint,
                        "202": "b" * 64,
                        "303": "c" * 64,
                    },
                    "new_relation_families": [],
                    **locked_fields("RPT_BASE"),
                },
                {
                    "run_id": "RPT_SELECTED_UNARY",
                    "configuration_role": "semantic_control",
                    "relational_selection_eligible": False,
                    "mean_matched_seed_accuracy_difference": 0.0,
                    "checkpoint_hashes": {
                        "101": "d" * 64,
                        "202": "e" * 64,
                        "303": "f" * 64,
                    },
                    "new_relation_families": ["PT"],
                    **locked_fields("RPT_SELECTED_UNARY"),
                },
                {
                    "run_id": "RPT_PT",
                    "configuration_role": "scientific_finalist",
                    "relational_selection_eligible": True,
                    "mean_matched_seed_accuracy_difference": -0.01,
                    "checkpoint_hashes": {
                        "101": "1" * 64,
                        "202": "2" * 64,
                        "303": "3" * 64,
                    },
                    "new_relation_families": ["PT"],
                    **locked_fields("RPT_PT"),
                },
            ],
            "nominal_relational_winner_id": "RPT_PT",
            "confirmation_gain_positive": False,
            "capacity_control_reproduces_gain": False,
            "selection_metrics": {},
            "selection_reason": "synthetic",
            "final_test_used_for_selection": False,
            "final_test_reporting_only": True,
        }
    )
    result = evaluate_locked_finalist(
        _FinalModel(),
        _final_loader(),
        run_id="RPT_BASE",
        seed=101,
        checkpoint_registration=registration,
        locked_finalists=lock,
        campaign_spec_sha256="5" * 64,
        split_manifest_sha256="6" * 64,
        hlt_cache_hashes=hashes,
        output_dir=tmp_path,
        expected_event_count=20,
    )
    assert result["contract"] == FINAL_EVALUATION_CONTRACT
    predictions = load_final_predictions(tmp_path / "predictions.npz")
    statistics = paired_prediction_statistics(
        predictions,
        predictions,
        bootstrap_replicates=50,
    )
    assert statistics["paired_absolute_accuracy_difference"] == 0
    assert statistics["paired_bootstrap"]["lower_2p5_percent"] == 0
    candidate = dict(predictions)
    candidate["predictions"] = predictions["predictions"].copy()
    candidate["predictions"][::3] = (
        candidate["predictions"][::3] + 1
    ) % 10
    candidate["metadata"] = {
        **predictions["metadata"],
        "run_id": "RPT_PT",
    }
    optimized = paired_prediction_statistics(
        candidate,
        predictions,
        bootstrap_replicates=37,
    )
    difference = (
        (candidate["predictions"] == candidate["labels"]).astype(np.int8)
        - (predictions["predictions"] == predictions["labels"]).astype(np.int8)
    )
    generator = np.random.Generator(np.random.PCG64(917_301))
    manual = []
    for _ in range(37):
        total = 0
        for class_index in range(10):
            indices = np.flatnonzero(candidate["labels"] == class_index)
            draw = generator.integers(
                0, len(indices), size=len(indices), dtype=np.int64
            )
            total += int(difference[indices[draw]].sum())
        manual.append(total / len(difference))
    interval = np.quantile(manual, [0.025, 0.975], method="linear")
    assert optimized["paired_bootstrap"]["lower_2p5_percent"] == interval[0]
    assert optimized["paired_bootstrap"]["upper_97p5_percent"] == interval[1]


def test_negative_json_and_markdown_report_remain_valid() -> None:
    checkpoints = {
        str(seed): _digest(f"checkpoint-{seed}") for seed in (101, 202, 303)
    }
    confirmation = with_content_hash(
        {
            "contract": CONFIRMATION_SUMMARY_CONTRACT,
            "schema_version": 3,
            "confirmation_registry_sha256": "1" * 64,
            "rows": [
                {
                    "run_id": "RPT_PT",
                    "configuration_role": "scientific_finalist",
                    "new_relation_families": ["PT"],
                    "checkpoint_hashes": checkpoints,
                    "checkpoint_registration_hashes": {
                        str(seed): _digest(
                            f"report-registration-RPT_PT-{seed}"
                        )
                        for seed in (101, 202, 303)
                    },
                    "model_contract_sha256": _digest(
                        "report-model-RPT_PT"
                    ),
                    "mean_matched_seed_accuracy_difference": -0.01,
                    "seeds_beating_matched_baseline": 0,
                }
            ],
            "scientific_finalist_ordering": ["RPT_PT"],
            "nominal_relational_winner_id": "RPT_PT",
            "confirmation_gain_positive": False,
            "capacity_control_reproduces_gain": False,
            "capacity_control_max_mean_delta": 0,
            "selection_reason": "synthetic",
            "negative_campaign_valid": True,
        }
    )
    report_lineage = {
        "campaign_spec": "2" * 64,
        "split_manifest": "3" * 64,
    }

    def report_locked_fields(run_id: str):
        return {
            "checkpoint_registration_hashes": {
                str(seed): _digest(f"report-registration-{run_id}-{seed}")
                for seed in (101, 202, 303)
            },
            "val_select_metrics_hashes": {
                str(seed): _digest(f"report-val-{run_id}-{seed}")
                for seed in (101, 202, 303)
            },
            "model_contract_sha256": _digest(f"report-model-{run_id}"),
            "lineage_hashes": report_lineage,
            "lineage_authenticated": True,
        }
    semantic_metrics = {
        name: {"accuracy": value}
        for name, value in (
            ("full_model", 0.50),
            ("within_jet_shuffled_relations", 0.40),
            ("wrong_event_relations", 0.35),
            ("directional_swap", 0.45),
        )
    }
    semantic_run = build_semantic_perturbation_artifact(
        run_id="RPT_PT",
        families=("PT",),
        checkpoint_sha256=checkpoints["101"],
        checkpoint_registration_sha256=_digest(
            "report-registration-RPT_PT-101"
        ),
        model_contract_sha256=_digest("report-model-RPT_PT"),
        confirmation_summary_sha256=confirmation["content_hash"],
        metrics=semantic_metrics,
        diagnostics={
            name: {"evaluated_event_count": 100}
            for name in semantic_metrics
        },
    )
    semantic = build_semantic_diagnostics_bundle(
        confirmation_summary=confirmation,
        run_artifacts=[semantic_run],
    )
    lock = with_content_hash(
        {
            "contract": LOCKED_FINALISTS_CONTRACT,
            "schema_version": 3,
            "campaign_spec_sha256": "2" * 64,
            "split_manifest_sha256": "3" * 64,
            "hlt_cache_hashes": {"final_test": "4" * 64},
            "confirmation_registry_sha256": "1" * 64,
            "confirmation_summary_sha256": confirmation["content_hash"],
            "semantic_perturbation_sha256": semantic["content_hash"],
            "unary_control_registry_sha256": "6" * 64,
            "confirmation_results_envelope_sha256": "7" * 64,
            "unary_results_envelope_sha256": "8" * 64,
            "all_selection_lineage_authenticated": True,
            "baseline_id": "RPT_BASE",
            "evaluation_rows": [
                {
                    "run_id": run_id,
                    "configuration_role": role,
                    "relational_selection_eligible": run_id == "RPT_PT",
                    "mean_matched_seed_accuracy_difference": (
                        -0.01 if run_id == "RPT_PT" else 0
                    ),
                    "checkpoint_hashes": checkpoints,
                    "new_relation_families": (
                        ["PT"] if run_id != "RPT_BASE" else []
                    ),
                    **report_locked_fields(run_id),
                }
                for run_id, role in (
                    ("RPT_BASE", "reference_baseline"),
                    ("RPT_PT", "scientific_finalist"),
                    ("RPT_SELECTED_UNARY", "semantic_control"),
                )
            ],
            "nominal_relational_winner_id": "RPT_PT",
            "confirmation_gain_positive": False,
            "capacity_control_reproduces_gain": False,
            "selection_metrics": {},
            "selection_reason": "synthetic",
            "final_test_used_for_selection": False,
            "final_test_reporting_only": True,
        }
    )
    evaluations = []
    truth = np.arange(100, dtype=np.int64) % 10
    for run_id, role, correct in (
        ("RPT_BASE", "reference_baseline", True),
        ("RPT_PT", "scientific_finalist", False),
        ("RPT_SELECTED_UNARY", "semantic_control", False),
    ):
        prediction = truth if correct else (truth + 1) % 10
        logits = np.full((len(truth), 10), -2.0)
        logits[np.arange(len(truth)), prediction] = 2.0
        metrics = evaluate_logits(logits, truth, split="final_test")
        for seed in (101, 202, 303):
            evaluations.append(
                with_content_hash(
                    {
                        "contract": FINAL_EVALUATION_CONTRACT,
                        "schema_version": 1,
                        "run_id": run_id,
                        "seed": seed,
                        "configuration_role": role,
                        "relational_selection_eligible": run_id == "RPT_PT",
                        "locked_finalists_sha256": lock["content_hash"],
                        "checkpoint_sha256": checkpoints[str(seed)],
                        "checkpoint_registration_sha256": (
                            _digest(
                                f"report-registration-{run_id}-{seed}"
                            )
                        ),
                        "model_contract_sha256": _digest(
                            f"report-model-{run_id}"
                        ),
                        "checkpoint_lineage_hashes": report_lineage,
                        "lineage_authenticated": True,
                        "campaign_spec_sha256": "2" * 64,
                        "split_manifest_sha256": "3" * 64,
                        "final_test_hlt_cache_sha256": "4" * 64,
                        "event_count": len(truth),
                        "metrics": metrics,
                        "parameter_and_flop_profile": {
                            "trainable_parameters": 1,
                            "forward_flops_per_event": 2,
                            "latency_ms": {"mean": 1, "median": 1},
                            "peak_incremental_device_memory_bytes": 3,
                        },
                        "hlt_only_inference": True,
                        "final_test_used_for_selection": False,
                    }
                )
            )
    paired = {
        run_id: {
            str(seed): with_content_hash(
                {
                    "contract": "relational_part_paired_statistics_v1",
                    "schema_version": 1,
                    "seed": seed,
                    "candidate_run_id": run_id,
                    "baseline_run_id": "RPT_BASE",
                    "candidate_accuracy": 0.0,
                    "baseline_accuracy": 1.0,
                    "paired_absolute_accuracy_difference": -1.0,
                }
            )
            for seed in (101, 202, 303)
        }
        for run_id in ("RPT_PT", "RPT_SELECTED_UNARY")
    }
    report = build_relational_part_report(
        locked_finalists=lock,
        confirmation_summary=confirmation,
        semantic_diagnostics=semantic,
        final_evaluations=evaluations,
        paired_statistics=paired,
    )
    assert report["fully_negative_campaign_completed_validly"] is True
    assert report["positive_architecture_result"] is False
    assert report["semantic_validation_diagnostics"]["run_ids"] == ["RPT_PT"]
    assert "Final-test results were used for reporting only" in (
        render_relational_part_markdown(report)
    )
    markdown = render_relational_part_markdown(report)
    assert "`wrong_event_relations` | 0.350000 | -0.150000" in markdown
