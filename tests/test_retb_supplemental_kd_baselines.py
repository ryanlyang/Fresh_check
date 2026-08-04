from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from teacher_logit_reco.relation_expert_token_bridge.expert_training import (
    EXPERT_LOSS_CANDIDATES,
    build_expert_loss_registry,
    build_teacher_logits_manifest,
    offline_expert_objective,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.supplemental_kd_baselines import (
    KD_BASELINE_ARCHITECTURES,
    KD_BASELINE_COORDINATES,
    KD_BASELINE_PLAN_CONTRACT_V1,
    KD_BASELINE_SEEDS,
    build_kd_baseline_plan,
    conventional_kd_configuration,
    file_sha256,
    validate_kd_baseline_plan,
)
from scripts import train_retb_supplemental_kd_baseline as kd_worker


SOURCE = {
    "source_commit": "1" * 40,
    "source_dirty": False,
    "source_status_sha256": "2" * 64,
}


def _parent_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "parent"
    campaign = with_content_hash(
        {
            "contract": "fixture_campaign",
            "campaign_id": "fixture",
            "campaign_profile": "production_500k_scale3m",
            "parent_artifact_hashes": {"global_determinism": "3" * 64},
            "source": {"fixture": True},
        }
    )
    write_immutable_json(root / "campaign_spec.json", campaign)
    write_immutable_json(
        root / "registry/retb_expert_losses.json", build_expert_loss_registry()
    )
    write_immutable_json(
        root / "registry/retb_step3_architecture_bundle.json",
        with_content_hash({"contract": "fixture_step3"}),
    )
    write_immutable_json(
        root / "selection/stage_a_strongest_teacher.json",
        with_content_hash(
            {"contract": "fixture_selection", "selected_teacher": "O_FULLREL"}
        ),
    )
    teacher = (
        root
        / "runs/stage_a/offline_controls/SELECTED_STRONGEST/seed_101/best_model_val.pt"
    )
    teacher.parent.mkdir(parents=True, exist_ok=True)
    teacher.write_bytes(b"teacher")
    teacher_sha = file_sha256(teacher)
    for control in KD_BASELINE_ARCHITECTURES:
        write_immutable_json(
            root
            / "runs/stage_a/offline_controls"
            / control
            / "seed_101/checkpoint_registration.json",
            with_content_hash(
                {
                    "contract": "fixture_registration",
                    "expert_id": control,
                    "seed": 101,
                    "loss_id": "ELOSS_CE",
                    "checkpoint_sha256": teacher_sha if control == "O_FULLREL" else "4" * 64,
                    "fixed_epoch_budget_completed": True,
                    "performance_based_termination": False,
                    "selected_val_stop": {
                        "accuracy": 0.8,
                        "cross_entropy": 0.5,
                    },
                    "source": {"control": control},
                }
            ),
        )
    cache = root / "inputs/teacher_logits/ELOSS_KD_DOMINANT"
    cache.mkdir(parents=True, exist_ok=True)
    train = cache / "model_train.npz"
    stop = cache / "val_stop.npz"
    train.write_bytes(b"train")
    stop.write_bytes(b"stop")
    manifest = build_teacher_logits_manifest(
        model_train_npz_sha256=file_sha256(train),
        val_stop_npz_sha256=file_sha256(stop),
        teacher_checkpoint_hashes={"SELECTED_STRONGEST": teacher_sha},
        teacher_fields=["SELECTED_STRONGEST"],
    )
    write_immutable_json(root / "inputs/teacher_logits/ELOSS_KD_DOMINANT.json", manifest)
    design = root / "inputs/offline/val_design/offline_inputs.npz"
    design.parent.mkdir(parents=True, exist_ok=True)
    design.write_bytes(b"design")
    write_immutable_json(
        design.parent / "offline_input_manifest.json",
        with_content_hash(
            {
                "contract": "fixture_input",
                "npz_filename": design.name,
                "npz_sha256": file_sha256(design),
            }
        ),
    )
    for name in ("relation", "region"):
        write_immutable_json(
            root / f"inputs/normalization/offline_500k/{name}.json",
            with_content_hash({"contract": f"fixture_{name}"}),
        )
    for split in ("model_train", "val_stop", "val_design"):
        write_immutable_json(
            root
            / "inputs/region_tree/offline"
            / f"{split}_exclusive_ca_v1/manifest.json",
            with_content_hash(
                {"contract": "fixture_tree_manifest", "split": split}
            ),
        )
    return root


def test_exact_six_conventional_kd_coordinates() -> None:
    assert KD_BASELINE_ARCHITECTURES == ("O_BASE", "O_FULLREL")
    assert KD_BASELINE_SEEDS == (101, 202, 303)
    assert KD_BASELINE_COORDINATES == (
        ("O_BASE", 101),
        ("O_BASE", 202),
        ("O_BASE", 303),
        ("O_FULLREL", 101),
        ("O_FULLREL", 202),
        ("O_FULLREL", 303),
    )
    base = conventional_kd_configuration("O_BASE")
    full = conventional_kd_configuration("O_FULLREL")
    assert base["tokenizer_mode"] == full["tokenizer_mode"] == "TOK_WEAVER_CLASS"
    assert base["summary_token_bottleneck_present"] is False
    assert full["summary_token_bottleneck_present"] is False
    assert base["relation_family"] is None
    assert full["relation_family"] == "ALL"


def test_plan_binds_teacher_caches_and_conventional_baselines(
    tmp_path: Path,
) -> None:
    parent = _parent_fixture(tmp_path)
    plan = build_kd_baseline_plan(
        parent_root=parent,
        supplemental_id="fixture-kd",
        source_snapshot=SOURCE,
    )
    validate_kd_baseline_plan(plan)
    assert plan["teacher_id"] == "O_FULLREL"
    assert plan["objective"] == {
        "cross_entropy_weight": 0.25,
        "kd_weight": 1.0,
        "temperature": 2.0,
        "kl_direction": "teacher_probability_to_student_probability",
    }
    assert set(plan["baseline_registrations"]) == set(KD_BASELINE_ARCHITECTURES)
    assert int(plan["schema_version"]) == 2
    assert {
        "region_tree_model_train_manifest",
        "region_tree_val_stop_manifest",
        "region_tree_val_design_manifest",
    }.issubset(plan["parent_artifacts"])
    teacher = Path(plan["parent_artifacts"]["teacher_checkpoint"]["path"])
    teacher.write_bytes(b"drift")
    try:
        validate_kd_baseline_plan(plan)
    except ValueError as error:
        assert "bytes drifted" in str(error)
    else:
        raise AssertionError("teacher checkpoint drift was accepted")


def test_dominant_kd_objective_is_the_existing_exact_objective() -> None:
    assert EXPERT_LOSS_CANDIDATES["ELOSS_KD_DOMINANT"] == {
        "cross_entropy_weight": 0.25,
        "kd_weight": 1.0,
        "teacher": "SELECTED_STRONGEST",
    }
    logits = torch.tensor([[0.2] * 10, [0.1] * 10], requires_grad=True)
    labels = torch.tensor([0, 1])
    teacher = {"SELECTED_STRONGEST": torch.zeros(2, 10)}
    loss, components = offline_expert_objective(
        logits,
        labels,
        loss_id="ELOSS_KD_DOMINANT",
        teacher_logits=teacher,
    )
    expected = 0.25 * components["cross_entropy"] + components[
        "knowledge_distillation"
    ]
    assert torch.allclose(loss.detach(), expected)


def test_legacy_v1_plan_remains_valid_only_for_targeted_recovery(
    tmp_path: Path,
) -> None:
    plan = build_kd_baseline_plan(
        parent_root=_parent_fixture(tmp_path),
        supplemental_id="legacy-recovery",
        source_snapshot=SOURCE,
    )
    legacy = dict(plan)
    legacy.pop("content_hash")
    legacy["contract"] = KD_BASELINE_PLAN_CONTRACT_V1
    legacy["schema_version"] = 1
    legacy["parent_artifacts"] = {
        name: row
        for name, row in legacy["parent_artifacts"].items()
        if not name.startswith("region_tree_")
    }
    legacy = with_content_hash(legacy)
    validate_kd_baseline_plan(legacy)


def test_fullrel_worker_loads_identity_aligned_authenticated_region_trees(
    tmp_path: Path, monkeypatch,
) -> None:
    parent = tmp_path / "parent"
    tree_root = (
        parent
        / "inputs/region_tree/offline/model_train_exclusive_ca_v1"
    )
    manifest = with_content_hash(
        {"contract": "fixture_tree_manifest", "split": "model_train"}
    )
    write_immutable_json(tree_root / "manifest.json", manifest)
    requested = {}

    def fake_loader(root: Path, identities: list[str]):
        requested["root"] = root
        requested["identities"] = identities
        return ([{"identity": value} for value in identities], manifest)

    monkeypatch.setattr(
        kd_worker, "load_authenticated_tree_selection", fake_loader
    )
    plan = {
        "parent_artifacts": {
            "region_tree_model_train_manifest": {
                "content_hash": manifest["content_hash"],
                "file_sha256": file_sha256(tree_root / "manifest.json"),
            }
        }
    }
    trees, manifest_sha256 = kd_worker._region_trees(
        parent=parent,
        plan=plan,
        split="model_train",
        arrays={"identities": np.asarray(["jet-2", "jet-1"])},
    )
    assert requested == {
        "root": tree_root,
        "identities": ["jet-2", "jet-1"],
    }
    assert trees == [{"identity": "jet-2"}, {"identity": "jet-1"}]
    assert manifest_sha256 == manifest["content_hash"]


def test_submission_is_unthrottled_six_job_array() -> None:
    submit = Path("sbatch/submit_retb_supplemental_kd_baselines.sh").read_text()
    row = Path("sbatch/run_retb_supplemental_kd_baseline_row.sh").read_text()
    assert "--array=0-5" in submit
    assert "%" not in "\n".join(
        line for line in submit.splitlines() if "--array=" in line
    )
    assert "status --porcelain" not in submit
    assert "architectures=(O_BASE O_BASE O_BASE O_FULLREL O_FULLREL O_FULLREL)" in row
    assert "seeds=(101 202 303 101 202 303)" in row
    assert 'output_namespace="runs_recovery"' in row
    assert "performance_based_termination" in Path(
        "teacher_logit_reco/relation_expert_token_bridge/supplemental_kd_baselines.py"
    ).read_text()


def test_recovery_replays_only_fullrel_and_waits_for_original_obase() -> None:
    recovery = Path(
        "sbatch/submit_retb_supplemental_kd_recovery.sh"
    ).read_text()
    assert "--array=3-5" in recovery
    assert "afterany:${RETB_SUPP_KD_ORIGINAL_ARRAY_JOB_ID}" in recovery
    assert "afterok:${recovery}" in recovery
    assert '"reused_original_task_ids": [0, 1, 2]' in recovery
    assert '"replayed_task_ids": [3, 4, 5]' in recovery
    assert "RETB_SUPP_KD_RECOVERY_MODE=1" in recovery
