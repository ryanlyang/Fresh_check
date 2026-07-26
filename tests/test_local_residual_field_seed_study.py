from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import pytest

from teacher_logit_reco.local_particle_residual_field.fusion_campaign import (
    stable_fusion_json_hash,
)
from teacher_logit_reco.local_particle_residual_field.seed_study import (
    MATCHED_SEEDS,
    REUSED_A0_SEEDS,
    TRAINED_A0_SEEDS,
    SEED_STUDY_MANIFEST_CONTRACT,
    build_a0_seed_study_config,
    build_seed_study_manifest,
    build_seed_study_report,
)
from teacher_logit_reco.local_particle_residual_field.tagger_train import (
    LocalResidualFieldTaggerTrainConfig,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _a0_source_config(tmp_path: Path) -> dict[str, object]:
    config = LocalResidualFieldTaggerTrainConfig(
        output_dir=str(tmp_path / "source_A0"),
        hlt_cache_dir=str(tmp_path / "hlt_cache"),
        target_cache_dir=str(tmp_path / "targets"),
        manifest_path=str(tmp_path / "split_manifest.json.gz"),
        seed=20421,
        field_source="hlt_only",
        teacher_logits_dir="/legacy/inactive/path",
        kd_loss_weight=0.0,
        reconstructor_loss_weight=0.0,
    )
    return {"config": asdict(config)}


def test_seed_matrix_and_a0_clone_are_locked(tmp_path: Path) -> None:
    assert MATCHED_SEEDS == (20421, 20522, 20623, 20724, 20825)
    assert REUSED_A0_SEEDS == (20421, 20522)
    assert TRAINED_A0_SEEDS == (20623, 20724, 20825)

    config, audit = build_a0_seed_study_config(
        _a0_source_config(tmp_path),
        seed=20623,
        output_dir=tmp_path / "study" / "seed_20623" / "A0",
    )
    assert audit["ok"] is True
    assert audit["observed_differences"] == ["output_dir", "seed"]
    assert config.seed == 20623
    assert config.field_source == "hlt_only"
    assert config.teacher_logits_dir is None
    assert config.baseline_checkpoint is None
    assert config.reconstructor_checkpoint is None
    assert config.kd_loss_weight == 0.0


def test_preflight_allows_deleted_semantically_inactive_oracle_logit_cache(
    tmp_path: Path,
) -> None:
    curriculum_root = tmp_path / "curriculum"
    fusion_root = tmp_path / "fusion"
    campaign_root = tmp_path / "study"
    (curriculum_root / "inputs" / "hlt_cache").mkdir(parents=True)
    (curriculum_root / "targets").mkdir()
    (curriculum_root / "inputs" / "split_manifest").mkdir()
    (curriculum_root / "inputs" / "split_manifest" / "split_manifest.json.gz").write_bytes(
        b"manifest"
    )
    _write_json(
        curriculum_root / "selected_consumer.json",
        {"selected_consumer_id": "Orobust_light"},
    )
    (curriculum_root / "reconstructors" / "C0").mkdir(parents=True)
    (curriculum_root / "reconstructors" / "C0" / "best_model_val.pt").write_bytes(b"c0")

    a0_config = _a0_source_config(tmp_path)["config"]
    a0_dir = curriculum_root / "taggers" / "A0"
    a0_dir.mkdir(parents=True)
    (a0_dir / "best_model_val.pt").write_bytes(b"a0")
    _write_json(a0_dir / "source_metadata.json", {"config": a0_config})
    _write_json(a0_dir / "run_report.json", {"ok": True})

    a0_seed1_dir = fusion_root / "taggers" / "A0_seed1"
    a0_seed1_dir.mkdir(parents=True)
    (a0_seed1_dir / "best_model_val.pt").write_bytes(b"a0-seed1")
    a0_seed1_config = dict(a0_config)
    a0_seed1_config["seed"] = 20522
    _write_json(a0_seed1_dir / "source_metadata.json", {"config": a0_seed1_config})
    _write_json(a0_seed1_dir / "run_report.json", {"ok": True})

    consumer_dir = curriculum_root / "taggers" / "Orobust_light"
    consumer_dir.mkdir(parents=True)
    for filename in ("best_model_val.pt",):
        (consumer_dir / filename).write_bytes(b"consumer")
    _write_json(consumer_dir / "teacher_config.json", {"ok": True})
    _write_json(consumer_dir / "run_report.json", {"ok": True})

    p7b_dir = curriculum_root / "curriculum" / "P7b"
    p7b_dir.mkdir(parents=True)
    _write_json(
        p7b_dir / "source_metadata.json",
        {
            "config": {
                "seed": 30421,
                "oracle_logit_only_fallback": False,
                "oracle_teacher_logits_dir": str(
                    curriculum_root / "oracle_teacher_logits" / "Orobust_light"
                ),
            }
        },
    )
    _write_json(
        p7b_dir / "run_report.json",
        {"ok": True, "oracle_teacher_logits_paths": {}},
    )

    manifest = build_seed_study_manifest(
        campaign_id="no_cache",
        campaign_root=campaign_root,
        curriculum_root=curriculum_root,
        fusion_root=fusion_root,
    )
    assert manifest["ok"] is True
    assert manifest["oracle_execution_mode"] == "frozen_checkpoint_online"
    assert manifest["oracle_logit_cache_required"] is False
    assert manifest["oracle_logit_cache_present"] is False
    assert manifest["paths"]["oracle_teacher_logits_dir"] is None


def _metric(accuracy: float, cross_entropy: float) -> dict[str, object]:
    return {
        "accuracy": accuracy,
        "cross_entropy": cross_entropy,
        "n_jets": 150_000,
        "valid_for_selection": True,
    }


def test_report_uses_five_paired_stack_val_deltas_and_no_final_test(tmp_path: Path) -> None:
    run_dirs: dict[str, dict[str, str]] = {}
    expected_deltas: list[float] = []
    for index, seed in enumerate(MATCHED_SEEDS):
        run_dirs[str(seed)] = {}
        a0_accuracy = 0.75 + index * 0.001
        p7b_accuracy = a0_accuracy + (index + 1) * 0.0002
        expected_deltas.append(p7b_accuracy - a0_accuracy)
        for recipe, accuracy in (("A0", a0_accuracy), ("P7b", p7b_accuracy)):
            run_dir = tmp_path / "runs" / f"seed_{seed}" / recipe
            run_dir.mkdir(parents=True)
            (run_dir / "best_model_val.pt").write_bytes(f"{recipe}-{seed}".encode())
            _write_json(
                run_dir / "source_metadata.json",
                {
                    "config": {
                        "seed": seed,
                        "field_source": "hlt_only",
                    }
                },
            )
            report = {
                "best_epoch": index + 1,
                "best_model_val": _metric(accuracy + 0.001, 0.7 - accuracy),
                "stack_val": _metric(accuracy, 0.71 - accuracy),
            }
            if recipe == "P7b":
                report.update(
                    {
                        "run_id": "P7b",
                        "deployable": True,
                        "runtime_inputs": "HLT_only",
                        "final_test": None,
                    }
                )
            _write_json(run_dir / "run_report.json", report)
            run_dirs[str(seed)][recipe] = str(run_dir)

    manifest = {
        "contract": SEED_STUDY_MANIFEST_CONTRACT,
        "ok": True,
        "campaign_id": "unit_test",
        "matched_seeds": list(MATCHED_SEEDS),
        "final_test_policy": "forbidden",
        "run_dirs": run_dirs,
    }
    manifest["artifact_hash"] = stable_fusion_json_hash(manifest)
    report, rows = build_seed_study_report(manifest)

    assert report["ok"] is True
    assert report["final_test_evaluated"] is False
    assert report["summaries"]["stack_val"]["p7b_accuracy_wins"] == 5
    assert report["summaries"]["stack_val"]["a0_accuracy_wins"] == 0
    assert report["summaries"]["stack_val"]["paired_accuracy_delta_p7b_minus_a0"][
        "mean"
    ] == pytest.approx(sum(expected_deltas) / 5)
    assert len(report["pairs"]) == 5
    assert len(rows) == 20


def test_tigris_submitter_queues_only_missing_a0_and_all_matched_p7b() -> None:
    submitter = (REPO_ROOT / "sbatch" / "submit_lprf_p7b_seed_study.sh").read_text()
    wrapper = (REPO_ROOT / "sbatch" / "submit_lprf_p7b_seed_study_tigris.sh").read_text()
    p7b_job = (
        REPO_ROOT / "sbatch" / "run_train_local_residual_field_seed_study_p7b.sh"
    ).read_text()

    assert "for seed in 20623 20724 20825" in submitter
    assert "for seed in 20421 20522 20623 20724 20825" in submitter
    assert "DependencyNeverSatisfied" in submitter
    assert "dependency_chain_changed" in submitter
    assert "reu-aisocial" in wrapper
    assert "/home/ryreu/miniforge3-aarch64" in wrapper
    assert "export PYTHONNOUSERSITE=1" in wrapper
    assert "--seed \"${SEED}\"" in p7b_job
    assert "--run-id P7b" in p7b_job
    assert "--evaluate-final-test" not in p7b_job
    assert "--confirm-final-test" not in p7b_job
    assert "--oracle-teacher-logits-dir" not in p7b_job
    assert "--student-warm-start-checkpoint \"${consumer_checkpoint}\"" in p7b_job
    assert "--predictor-warm-start-checkpoint \"${c0_checkpoint}\"" in p7b_job
    assert "Reusing completed P7b training artifacts" in p7b_job
    assert "reuse_completed_training" in p7b_job
