from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from scripts.fuse_adaptive_binary_pseudooffline import _detailed_metrics as fusion_metrics
from scripts.predict_adaptive_binary_pseudooffline import (
    _detailed_classifier_metrics as prediction_metrics,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.tagger_runtime import (
    _detailed_eval_metrics as tagger_metrics,
)

torch = pytest.importorskip("torch")

from teacher_logit_reco.adaptive_binary_pseudooffline import (
    ABPH_EXPECTED_VARIANT_NAMES,
    ABPH_POSTHOC_FUSION_VARIANTS,
    ABPH_TARGET_PROVENANCE_FIELDS,
    AdaptiveBinaryCampaignReportConfig,
    FrozenFusionArtifact,
    HierarchyAwareTaggerOutput,
    LogitPredictionBlock,
    PseudoViewInputs,
    ablate_pseudo_inputs,
    apply_frozen_fusion,
    compute_tagging_objective,
    canonical_hash,
    fit_frozen_stack_fusion,
    resolve_variant_config,
    tagging_objective_config,
    teacher_logits_for_objective,
    variant_spec,
    write_adaptive_binary_campaign_report,
    write_frozen_fusion_artifact,
)


def _tagger_output(batch: int = 5, classes: int = 3) -> HierarchyAwareTaggerOutput:
    baseline = torch.randn(batch, classes, requires_grad=True)
    logits = baseline + 0.1 * torch.randn(batch, classes, requires_grad=True)
    representation = torch.randn(batch, 8, requires_grad=True)
    return HierarchyAwareTaggerOutput(
        logits=logits,
        baseline_logits=baseline,
        representation=representation,
        baseline_representation=representation,
        diagnostics={"teacher_logits_loaded": False},
    )


def test_f_tier_objectives_fail_closed_on_teacher_logits() -> None:
    f0 = tagging_objective_config("F0")
    f4 = tagging_objective_config("F4")
    assert not f0.requires_teacher_logits
    assert f4.requires_teacher_logits

    calls: list[str] = []

    def loader(split: str) -> torch.Tensor:
        calls.append(split)
        return torch.randn(5, 3)

    assert teacher_logits_for_objective(f0, "model_train", loader) is None
    assert calls == []
    with pytest.raises(FileNotFoundError, match="requires offline teacher logits"):
        teacher_logits_for_objective(f4, "model_train", None)
    teacher = teacher_logits_for_objective(f4, "model_val", loader)
    assert teacher.shape == (5, 3)
    assert calls == ["model_val"]
    with pytest.raises(ValueError, match="forbidden"):
        teacher_logits_for_objective(f4, "final_test", loader)

    output = _tagger_output()
    labels = torch.tensor([0, 1, 2, 0, 1])
    with pytest.raises(FileNotFoundError, match="requires teacher logits"):
        compute_tagging_objective(
            output, labels, f4, split="model_train", reconstruction_loss=None
        )
    with pytest.raises(ValueError, match="must not load"):
        compute_tagging_objective(
            output,
            labels,
            f0,
            split="model_train",
            reconstruction_loss=torch.tensor(1.0),
            teacher_logits=torch.randn(5, 3),
        )


def test_primary_and_kd_objective_terms_are_not_silently_dropped() -> None:
    output = _tagger_output()
    labels = torch.tensor([0, 1, 2, 0, 1])
    primary = compute_tagging_objective(
        output,
        labels,
        tagging_objective_config("F0"),
        split="model_train",
        reconstruction_loss=torch.tensor(0.7, requires_grad=True),
    )
    assert set(primary.raw_terms) == {
        "label_ce",
        "hlt_anchor_ce",
        "joint_reconstruction",
    }
    assert primary.diagnostics["teacher_logits_loaded"] is False
    primary.total.backward()

    kd = compute_tagging_objective(
        _tagger_output(),
        labels,
        tagging_objective_config("F4"),
        split="model_val",
        teacher_logits=torch.randn(5, 3),
    )
    assert "offline_logit_kd" in kd.raw_terms
    assert "joint_reconstruction" not in kd.raw_terms
    assert kd.diagnostics["representation_kd_enabled"] is False


def test_prediction_and_fusion_reports_include_full_ten_class_diagnostics() -> None:
    labels = np.arange(40, dtype=np.int64) % 10
    logits = np.full((40, 10), -2.0, dtype=np.float64)
    logits[np.arange(40), labels] = 2.0
    for metrics in (
        prediction_metrics(logits, labels),
        fusion_metrics(logits, labels),
        tagger_metrics(logits, labels),
    ):
        assert metrics["accuracy"] == pytest.approx(1.0)
        assert metrics["macro_ovr_auc"] == pytest.approx(1.0)
        assert len(metrics["per_class_accuracy"]) == 10
        assert np.asarray(metrics["confusion_matrix"]).shape == (10, 10)


def _block(member: str, split: str, logits: np.ndarray, labels: np.ndarray) -> LogitPredictionBlock:
    return LogitPredictionBlock(
        member=member,
        split=split,
        logits=logits,
        labels=labels,
        jet_ids=np.arange(labels.size, dtype=np.int64),
        checkpoint_hash=f"checkpoint-{member}",
        resolved_config_hash=f"config-{member}",
        provenance={
            "source_manifest_hash": "manifest",
            "jet_identity_hash": f"jets-{split}",
            "label_hash": f"labels-{split}",
            "class_mapping_hash": "classes",
            "hlt_content_hash": f"hlt-{split}",
            "teacher_logits_loaded": False,
        },
    )


def test_stack_fusion_freezes_member_identity_and_never_fits_final_test() -> None:
    rng = np.random.default_rng(4)
    labels = np.arange(60, dtype=np.int64) % 3
    members = ("E5_kt32_mh4_dualcross", "E6_ca32_mh4_dualcross")
    train = tuple(
        _block(member, "stack_train", rng.normal(size=(60, 3)), labels)
        for member in members
    )
    val = tuple(
        _block(member, "stack_val", rng.normal(size=(60, 3)), labels)
        for member in members
    )
    artifact = fit_frozen_stack_fusion("G2_kt_ca_logit_fusion", members, train, val, updates=20)
    payload = artifact.to_dict()
    assert payload["fit_split"] == "stack_train"
    assert payload["selection_split"] == "stack_val"
    assert payload["final_test_loaded_during_fit"] is False
    assert apply_frozen_fusion(artifact, val).shape == (60, 3)
    with pytest.raises(ValueError, match="membership/order"):
        apply_frozen_fusion(artifact, tuple(reversed(val)))
    final = tuple(
        _block(member, "final_test", rng.normal(size=(60, 3)), labels)
        for member in members
    )
    assert apply_frozen_fusion(artifact, final).shape == (60, 3)
    with pytest.raises(ValueError, match="expected stack_train"):
        fit_frozen_stack_fusion("G2_kt_ca_logit_fusion", members, final, val, updates=2)


def test_prediction_block_accepts_model_val_without_permitting_fusion_fit() -> None:
    labels = np.arange(12, dtype=np.int64) % 3
    members = ("A0_hlt_part", "E5_kt32_mh4_dualcross")
    blocks = tuple(
        _block(
            member,
            "model_val",
            np.zeros((12, 3), dtype=np.float64),
            labels,
        )
        for member in members
    )
    for block in blocks:
        block.validate()
        assert block.prediction_hash
    with pytest.raises(ValueError, match="expected stack_train"):
        fit_frozen_stack_fusion(
            "G2_kt_ca_logit_fusion",
            members,
            blocks,
            tuple(
                _block(member, "stack_val", block.logits, labels)
                for member, block in zip(members, blocks)
            ),
            updates=2,
        )


def test_pseudo_diagnostic_ablation_preserves_internal_root_contract() -> None:
    batch, views, particles, root_dim = 3, 2, 128, 7
    root = torch.randn(batch, root_dim)
    arrays = {
        "shared_root_ledger": root,
        "hypothesis_latent": torch.randn(batch, views, 4),
        "hypothesis_prior_log_prob": torch.zeros(batch, views),
        "particle__exclusive_kt__canonical_features": torch.randn(batch, views, particles, 15),
        "particle__exclusive_kt__side_channels": torch.randn(batch, views, particles, 3),
        "particle__exclusive_kt__mask": torch.ones(batch, views, particles, dtype=torch.bool),
        "frontier__exclusive_kt__depth_00__ledger": root[:, None, None].expand(-1, views, 1, -1).clone(),
        "frontier__exclusive_kt__depth_00__uncertainty": torch.zeros(batch, views, 1),
        "frontier__exclusive_kt__depth_00__mask": torch.ones(batch, views, 1, dtype=torch.bool),
    }
    pseudo = PseudoViewInputs(
        arrays=arrays,
        view_names=("mean", "sample_1"),
        hierarchy_names=("exclusive_kt",),
        frontier_depths={"exclusive_kt": 1},
        diagnostics={
            "offline_inputs_loaded": False,
            "teacher_logits_loaded": False,
            "offline_target_selected_hypothesis": False,
        },
    )
    zeroed = ablate_pseudo_inputs(pseudo, "zero_pseudo_particles")
    assert not bool(zeroed.arrays["particle__exclusive_kt__mask"].any())
    assert zeroed.diagnostics["checkpoint_selection_eligible"] is False
    shuffled = ablate_pseudo_inputs(pseudo, "shuffle_hypotheses_within_jet", seed=9)
    shuffled.validate()


def _split_provenance(*, target: bool) -> dict[str, object]:
    payload: dict[str, object] = {
        "source_manifest_hash": "manifest",
        "jet_identity_hash": "jets-model-val",
        "label_hash": "labels-model-val",
        "class_mapping_hash": "classes",
        "hlt_content_hash": "hlt-model-val",
        "hlt_profile": "fixed_hlt_v2_realistic",
        "hlt_profile_version": "2.0",
        "hlt_degradation_strength": 2.5,
        "hlt_params_hash": "hlt-params",
    }
    if target:
        payload.update({field: f"target-{field}" for field in ABPH_TARGET_PROVENANCE_FIELDS})
    return payload


def _fusion_artifact(name: str) -> FrozenFusionArtifact:
    members = ("member_a", "member_b")
    return FrozenFusionArtifact(
        fusion_variant=name,
        members=members,
        member_checkpoint_hashes={member: f"checkpoint-{member}" for member in members},
        member_config_hashes={member: f"config-{member}" for member in members},
        kind="scalar_simplex",
        weights=(0.5, 0.5),
        stack_train_prediction_hashes={member: f"train-{member}" for member in members},
        stack_val_prediction_hashes={member: f"val-{member}" for member in members},
        stack_train_cross_entropy=1.0,
        stack_val_cross_entropy=1.1,
        selection_candidates={"scalar_simplex": 1.1},
        membership_hash=f"membership-{name}",
    )


def _write_complete_campaign(root: Path) -> None:
    for name in ABPH_EXPECTED_VARIANT_NAMES:
        resolved = resolve_variant_config(name)
        split = "stack_val" if name in ABPH_POSTHOC_FUSION_VARIANTS else "model_val"
        diagnostics: dict[str, object] = {}
        if name == "E7_dual_hierarchy_dualcross":
            diagnostics["root_provenance"] = {
                "shared_root": True,
                "root_hash": "one-root",
                "branch_root_hashes": {
                    "exclusive_kt": "one-root",
                    "cambridge_aachen": "one-root",
                },
                "root_hash_count": 1,
            }
        if name == "E11_independent_root_dual_hierarchy_diagnostic":
            diagnostics["root_provenance"] = {
                "shared_root": False,
                "root_hash_count": 2,
            }
        if name == "F0_ce_reco_primary":
            diagnostics.update(
                {
                    "dual_hierarchy_joint_training": True,
                    "joint_reconstructor_hierarchy_names": [
                        "exclusive_kt",
                        "cambridge_aachen",
                    ],
                }
            )
        if name == "D6_true_offline_particles":
            diagnostics.update(
                {
                    "actual_pseudo_branch_forward_pass": True,
                    "copied_A4_metrics": False,
                    "pure_offline_architecture_ceiling": False,
                    "retains_reconstructed_hierarchy_context": True,
                }
            )
        provenance = _split_provenance(
            target=bool(resolved["data"].get("requires_offline_targets"))
        )
        if name == "F0_ce_reco_primary":
            target_hashes = {
                "exclusive_kt": "target-kt",
                "cambridge_aachen": "target-ca",
            }
            grouping_hashes = {
                "exclusive_kt": "grouping-kt",
                "cambridge_aachen": "grouping-ca",
            }
            provenance.update(
                {
                    "dual_target_provenance": True,
                    "hierarchy_target_content_hash": canonical_hash(target_hashes),
                    "grouping_algorithm_hash": canonical_hash(grouping_hashes),
                    "hierarchy_branches": {
                        hierarchy_name: {
                            "grouping": hierarchy_name,
                            "hierarchy_target_content_hash": target_hashes[
                                hierarchy_name
                            ],
                            "grouping_algorithm_hash": grouping_hashes[
                                hierarchy_name
                            ],
                            "offline_cache_content_hash": provenance[
                                "offline_cache_content_hash"
                            ],
                            "source_manifest_hash": provenance[
                                "source_manifest_hash"
                            ],
                            "hlt_content_hash": provenance["hlt_content_hash"],
                            "jet_identity_hash": provenance[
                                "jet_identity_hash"
                            ],
                        }
                        for hierarchy_name in (
                            "exclusive_kt",
                            "cambridge_aachen",
                        )
                    },
                }
            )
        report: dict[str, object] = {
            "ok": True,
            "variant_name": name,
            "resolved_variant_config_hash": resolved["resolved_config_hash"],
            "selected_checkpoint_hash": f"checkpoint-{name}",
            "source_git_commit": "commit",
            "source_status_hash": "status",
            "metrics": {
                split: {
                    "available": True,
                    "accuracy": 0.60,
                    "loss": 1.0,
                    "n_jets": 30,
                    "diagnostics": diagnostics,
                }
            },
            "provenance": {split: provenance},
        }
        if name in ABPH_POSTHOC_FUSION_VARIANTS:
            artifact = _fusion_artifact(name)
            path = root / "fusion" / name / "frozen_fusion.json"
            write_frozen_fusion_artifact(path, artifact)
            artifact_payload = artifact.to_dict()
            report["fusion"] = {
                "members": list(artifact.members),
                "membership_hash": artifact.membership_hash,
                "artifact_hash": artifact_payload["artifact_hash"],
            }
        path = root / "runs" / name / "run_report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report), encoding="utf-8")
    diagnostic_path = root / "diagnostics" / "tagger_use_report.json"
    diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_path.write_text(
        json.dumps(
            {
                "contract": "adaptive_binary_pseudooffline_tagger_diagnostics_v1",
                "ok": True,
                "selection_eligible": False,
                "split": "model_val",
                "variants": [
                    "E5_kt32_mh4_dualcross",
                    "E7_dual_hierarchy_dualcross",
                ],
                "rows": [
                    {"variant": "E5_kt32_mh4_dualcross", "diagnostic": "unaltered"},
                    {"variant": "E7_dual_hierarchy_dualcross", "diagnostic": "unaltered"},
                ],
                "final_test_loaded": False,
                "offline_inputs_loaded": False,
                "teacher_logits_loaded": False,
            }
        ),
        encoding="utf-8",
    )


def test_full_campaign_report_checks_all_tiers_roots_and_frozen_membership(tmp_path: Path) -> None:
    _write_complete_campaign(tmp_path)
    report = write_adaptive_binary_campaign_report(
        AdaptiveBinaryCampaignReportConfig(campaign_root=tmp_path)
    )
    assert report["ok"], report["problems"]
    assert report["checked_variant_count"] == len(ABPH_EXPECTED_VARIANT_NAMES)
    assert report["all_tiers_checked"] == list("ABCDEFG")
    e7 = next(row for row in report["root_identity"] if row["variant"].startswith("E7_"))
    assert e7["ok"] is True

    g2_path = tmp_path / "runs" / "G2_kt_ca_logit_fusion" / "run_report.json"
    g2 = json.loads(g2_path.read_text(encoding="utf-8"))
    g2["fusion"]["members"].reverse()
    g2_path.write_text(json.dumps(g2), encoding="utf-8")
    failed = write_adaptive_binary_campaign_report(
        AdaptiveBinaryCampaignReportConfig(campaign_root=tmp_path, output_dir=tmp_path / "failed")
    )
    assert not failed["ok"]
    assert any("membership differs" in problem for problem in failed["problems"])


def test_primary_report_rejects_missing_ca_target_provenance(tmp_path: Path) -> None:
    _write_complete_campaign(tmp_path)
    path = tmp_path / "runs" / "F0_ce_reco_primary" / "run_report.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["provenance"]["model_val"]["hierarchy_branches"][
        "cambridge_aachen"
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")
    report = write_adaptive_binary_campaign_report(
        AdaptiveBinaryCampaignReportConfig(
            campaign_root=tmp_path,
            output_dir=tmp_path / "missing-ca",
        )
    )
    assert not report["ok"]
    assert any(
        "F0 model_val dual-target provenance" in problem
        for problem in report["problems"]
    )


def test_final_metrics_require_success_and_unconditional_teacher_free_attestation(tmp_path: Path) -> None:
    _write_complete_campaign(tmp_path)
    path = tmp_path / "runs" / "A0_hlt_part" / "run_report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["ok"] = False
    report["metrics"]["final_test"] = {
        "available": True,
        "accuracy": 0.5,
        "diagnostics": {
            "offline_inputs_loaded": False,
            "teacher_logits_loaded": True,
            "hypothesis_selection_used_offline_target": False,
            "fusion_fitted_on_final_test": False,
        },
    }
    report["provenance"]["final_test"] = {
        **_split_provenance(target=False),
        "jet_identity_hash": "jets-final",
        "label_hash": "labels-final",
        "hlt_content_hash": "hlt-final",
    }
    path.write_text(json.dumps(report), encoding="utf-8")
    final = write_adaptive_binary_campaign_report(
        AdaptiveBinaryCampaignReportConfig(
            campaign_root=tmp_path,
            output_dir=tmp_path / "final",
            confirm_final_test=True,
        )
    )
    assert not final["ok"]
    assert any("metrics come from a failed run" in problem for problem in final["problems"])
    assert any("teacher_logits_loaded=false" in problem for problem in final["problems"])


def test_report_config_requires_every_campaign_tier(tmp_path: Path) -> None:
    only_a = tuple(name for name in ABPH_EXPECTED_VARIANT_NAMES if variant_spec(name).tier == "A")
    with pytest.raises(ValueError, match="complete frozen A0-G5 registry"):
        AdaptiveBinaryCampaignReportConfig(campaign_root=tmp_path, required_variants=only_a)
