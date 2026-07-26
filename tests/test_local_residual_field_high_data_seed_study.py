from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import pytest

from teacher_logit_reco.local_particle_residual_field.curriculum import (
    LOCAL_RESIDUAL_FIELD_SELECTED_CONSUMER_CONTRACT,
)
from teacher_logit_reco.local_particle_residual_field.fusion_campaign import (
    stable_fusion_json_hash,
)
from teacher_logit_reco.local_particle_residual_field.high_data_seed_study import (
    HIGH_DATA_MATCHED_SEEDS,
    HIGH_DATA_SPLIT_COUNTS,
    HIGH_DATA_STUDY_MANIFEST_CONTRACT,
    build_frozen_high_data_selection,
    build_high_data_a0_config,
    build_high_data_final_test_report,
    build_high_data_validation_report,
    validate_high_data_p7b_run,
)
from teacher_logit_reco.local_particle_residual_field.tagger_train import (
    LocalResidualFieldTaggerTrainConfig,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _manifest(tmp_path: Path, run_dirs: dict[str, dict[str, str]]) -> dict[str, object]:
    payload: dict[str, object] = {
        "contract": HIGH_DATA_STUDY_MANIFEST_CONTRACT,
        "ok": True,
        "campaign_id": "unit_high_data",
        "campaign_root": str(tmp_path),
        "matched_seeds": list(HIGH_DATA_MATCHED_SEEDS),
        "split_counts": dict(HIGH_DATA_SPLIT_COUNTS),
        "final_test_policy": "sealed_until_explicit_confirmation",
        "privileged_final_test_artifacts_present": False,
        "run_dirs": run_dirs,
        "extension_rule": {
            "condition": "P7b stack_val wins 3/3 and mean paired accuracy delta > 0"
        },
    }
    payload["artifact_hash"] = stable_fusion_json_hash(payload)
    return payload


def _metric(accuracy: float, ce: float, n_jets: int) -> dict[str, object]:
    return {
        "accuracy": accuracy,
        "cross_entropy": ce,
        "n_jets": n_jets,
        "valid_for_selection": True,
    }


def test_high_data_inventory_and_a0_clone_are_locked(tmp_path: Path) -> None:
    assert HIGH_DATA_MATCHED_SEEDS == (20421, 20522, 20623)
    assert HIGH_DATA_SPLIT_COUNTS == {
        "model_train": 3_000_000,
        "model_val": 250_000,
        "stack_train": 0,
        "stack_val": 500_000,
        "final_test": 1_000_000,
    }
    source_config = LocalResidualFieldTaggerTrainConfig(
        output_dir=str(tmp_path / "A0_source"),
        hlt_cache_dir=str(tmp_path / "hlt"),
        target_cache_dir=str(tmp_path / "targets"),
        manifest_path=str(tmp_path / "manifest.json.gz"),
        seed=20421,
        field_source="hlt_only",
    )
    config, audit = build_high_data_a0_config(
        {"config": asdict(source_config)},
        seed=20522,
        output_dir=tmp_path / "A0_seed_20522",
    )
    assert audit["ok"] is True
    assert audit["observed_differences"] == ["output_dir", "seed"]
    assert config.seed == 20522
    assert config.field_source == "hlt_only"
    assert config.baseline_checkpoint is None
    assert config.reconstructor_checkpoint is None
    with pytest.raises(ValueError, match="not a new high-data A0 seed"):
        build_high_data_a0_config(
            {"config": asdict(source_config)},
            seed=20724,
            output_dir=tmp_path / "forbidden",
        )


def test_high_data_selection_freezes_low_data_p7b_without_reselection(
    tmp_path: Path,
) -> None:
    source = tmp_path / "low_data_selected_consumer.json"
    destination = tmp_path / "high_data_selected_consumer.json"
    _write_json(
        source,
        {
            "contract": LOCAL_RESIDUAL_FIELD_SELECTED_CONSUMER_CONTRACT,
            "selected_consumer_id": "Orobust_light",
            "selected_alpha_endpoint": 1.0,
            "selection_source": "unit",
            "selection_reason": "unit",
            "model_val_alpha_curve": {"0.0": {}, "1.0": {}},
            "stack_val_alpha_curve": {"0.0": {}, "1.0": {}},
        },
    )
    frozen = build_frozen_high_data_selection(source, output_path=destination)
    assert frozen["selected_consumer_id"] == "Orobust_light"
    assert frozen["selected_alpha_endpoint"] == 1.0
    assert frozen["selection_source"] == "frozen_low_data_p7b_scaling_recipe"
    assert frozen["payload"]["high_data_reselection_performed"] is False


def test_validation_report_uses_three_locked_pairs_and_keeps_test_sealed(
    tmp_path: Path,
) -> None:
    run_dirs: dict[str, dict[str, str]] = {}
    expected_stack_deltas: list[float] = []
    for index, seed in enumerate(HIGH_DATA_MATCHED_SEEDS):
        run_dirs[str(seed)] = {}
        a0_acc = 0.76 + index * 0.0005
        p7b_acc = a0_acc + 0.0004 + index * 0.0001
        expected_stack_deltas.append(p7b_acc - a0_acc)
        for recipe, accuracy in (("A0", a0_acc), ("P7b", p7b_acc)):
            run_dir = tmp_path / "runs" / f"seed_{seed}" / recipe
            run_dir.mkdir(parents=True)
            (run_dir / "best_model_val.pt").write_bytes(f"{recipe}-{seed}".encode())
            _write_json(
                run_dir / "source_metadata.json",
                {"config": {"seed": seed, "field_source": "hlt_only"}},
            )
            report: dict[str, object] = {
                "best_epoch": index + 1,
                "best_model_val": _metric(
                    accuracy + 0.001, 0.6 - accuracy, 250_000
                ),
                "stack_val": _metric(accuracy, 0.61 - accuracy, 500_000),
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
    report, rows = build_high_data_validation_report(_manifest(tmp_path, run_dirs))
    assert report["final_test_evaluated"] is False
    assert report["final_test_status"] == "SEALED"
    assert report["summaries"]["stack_val"]["p7b_accuracy_wins"] == 3
    assert report["summaries"]["stack_val"]["paired_accuracy_delta_p7b_minus_a0"][
        "mean"
    ] == pytest.approx(sum(expected_stack_deltas) / 3)
    assert report["extension_recommendation"]["extend_to_five_seeds"] is True
    assert len(rows) == 12


def test_high_data_p7b_audit_changes_only_data_and_shared_source_bindings(
    tmp_path: Path,
) -> None:
    seed = 20421
    run_dir = tmp_path / "runs" / f"seed_{seed}" / "P7b"
    run_dir.mkdir(parents=True)
    consumer = tmp_path / "consumer.pt"
    predictor = tmp_path / "c0.pt"
    consumer.write_bytes(b"3m-consumer")
    predictor.write_bytes(b"3m-c0")
    from teacher_logit_reco.local_particle_residual_field.high_data_seed_study import (
        sha256_file,
    )

    allowed = [
        "hlt_cache_dir",
        "manifest_path",
        "oracle_run_report_path",
        "oracle_teacher_checkpoint",
        "oracle_teacher_config_path",
        "oracle_teacher_logits_dir",
        "output_dir",
        "predictor_warm_start_checkpoint",
        "seed",
        "selected_consumer_json",
        "student_warm_start_checkpoint",
        "target_cache_dir",
    ]
    reference_config = {
        "epochs": 12,
        "run_id": "P7b",
        "oracle_logit_only_fallback": False,
        **{name: f"/old/{name}" for name in allowed},
        "seed": 30421,
    }
    candidate_config = dict(reference_config)
    for name in allowed:
        candidate_config[name] = f"/new/{name}"
    candidate_config["seed"] = seed
    candidate_config["oracle_teacher_logits_dir"] = None
    reference_source = tmp_path / "reference_source.json"
    _write_json(reference_source, {"config": reference_config})
    _write_json(run_dir / "source_metadata.json", {"config": candidate_config})
    checkpoint = run_dir / "best_model_val.pt"
    checkpoint.write_bytes(b"student")
    _write_json(
        run_dir / "run_report.json",
        {
            "run_id": "P7b",
            "deployable": True,
            "runtime_inputs": "HLT_only",
            "final_test": None,
            "oracle_teacher_logits_paths": {},
            "selected_consumer_id": "Orobust_light",
            "oracle_teacher_checkpoint_hash": sha256_file(consumer),
            "student_initialization": {
                "student_init_checkpoint_hash": sha256_file(consumer)
            },
            "predictor_warm_start": {"checkpoint_hash": sha256_file(predictor)},
        },
    )
    run_dirs = {
        str(item): {
            "A0": str(tmp_path / "unused" / str(item) / "A0"),
            "P7b": str(
                run_dir
                if item == seed
                else tmp_path / "unused" / str(item) / "P7b"
            ),
        }
        for item in HIGH_DATA_MATCHED_SEEDS
    }
    manifest = _manifest(tmp_path, run_dirs)
    manifest.update(
        {
            "p7b_allowed_config_differences": allowed,
            "paths": {
                "reference_p7b_source_metadata": str(reference_source),
                "consumer_checkpoint": str(consumer),
                "c0_checkpoint": str(predictor),
            },
        }
    )
    manifest["artifact_hash"] = stable_fusion_json_hash(
        {key: value for key, value in manifest.items() if key != "artifact_hash"}
    )
    completion = validate_high_data_p7b_run(
        manifest,
        seed=seed,
        output_dir=run_dir,
    )
    assert completion["ok"] is True
    assert completion["runtime_inputs"] == "HLT_only"
    assert completion["final_test_evaluated"] is False
    assert completion["recipe_difference_paths"] == allowed


def test_final_test_report_rejects_privileged_inputs_and_accepts_1m_hlt_only(
    tmp_path: Path,
) -> None:
    run_dirs: dict[str, dict[str, str]] = {}
    predictions = tmp_path / "final_test_predictions"
    for seed in HIGH_DATA_MATCHED_SEEDS:
        run_dirs[str(seed)] = {}
        for recipe in ("A0", "P7b"):
            run_dir = tmp_path / "runs" / f"seed_{seed}" / recipe
            run_dir.mkdir(parents=True)
            checkpoint = run_dir / "best_model_val.pt"
            checkpoint.write_bytes(f"{recipe}-{seed}".encode())
            run_dirs[str(seed)][recipe] = str(run_dir)
            from teacher_logit_reco.local_particle_residual_field.high_data_seed_study import (
                sha256_file,
            )

            _write_json(
                predictions
                / f"seed_{seed}"
                / recipe
                / "final_test_predictions_metadata.json",
                {
                    "deployable": True,
                    "checkpoint_hash": sha256_file(checkpoint),
                    "dataset_metadata": {
                        "allowed_inputs": "HLT_particles_only_deployable_final_test",
                        "target_fields_present": False,
                        "teacher_logits_present": False,
                    },
                    "metrics": {
                        "accuracy": 0.75 + (0.001 if recipe == "P7b" else 0.0),
                        "cross_entropy": 0.69 - (0.001 if recipe == "P7b" else 0.0),
                        "n_jets": 1_000_000,
                    },
                },
            )
    manifest = _manifest(tmp_path, run_dirs)
    validation = {
        "contract": "local_residual_field_high_data_validation_report_v1",
        "ok": True,
    }
    validation["artifact_hash"] = stable_fusion_json_hash(validation)
    report, rows = build_high_data_final_test_report(
        manifest,
        validation_report=validation,
        predictions_root=predictions,
    )
    assert report["final_test_evaluated"] is True
    assert report["final_test_runtime_inputs"] == "HLT_only"
    assert report["oracle_diagnostics_included"] is False
    assert report["summary"]["p7b_accuracy_wins"] == 3
    assert len(rows) == 6

    bad = (
        predictions
        / "seed_20421"
        / "P7b"
        / "final_test_predictions_metadata.json"
    )
    payload = json.loads(bad.read_text())
    payload["dataset_metadata"]["target_fields_present"] = True
    _write_json(bad, payload)
    with pytest.raises(ValueError, match="target fields"):
        build_high_data_final_test_report(
            manifest,
            validation_report=validation,
            predictions_root=predictions,
        )


def test_tigris_submitter_locks_data_graph_and_separates_final_test() -> None:
    submitter = (
        REPO_ROOT / "sbatch" / "submit_lprf_p7b_high_data_seed_study.sh"
    ).read_text(encoding="utf-8")
    wrapper = (
        REPO_ROOT / "sbatch" / "submit_lprf_p7b_high_data_seed_study_tigris.sh"
    ).read_text(encoding="utf-8")
    p7b = (
        REPO_ROOT / "sbatch" / "run_train_local_residual_field_high_data_p7b.sh"
    ).read_text(encoding="utf-8")
    final_runner = (
        REPO_ROOT
        / "sbatch"
        / "run_predict_local_residual_field_high_data_final_test.sh"
    ).read_text(encoding="utf-8")

    for expected in (
        "MODEL_TRAIN_SIZE=3000000",
        "MODEL_VAL_SIZE=250000",
        "STACK_TRAIN_SIZE=0",
        "STACK_VAL_SIZE=500000",
        "FINAL_TEST_SIZE=1000000",
        'HLT_SPLITS="model_train model_val stack_val final_test"',
        'ARCHITECTURE_VIEW_10CLASS_OFFLINE_SPLITS="model_train model_val stack_val"',
        'LOCAL_RESIDUAL_FIELD_TARGET_SPLITS="model_train model_val stack_val"',
        "LOCAL_RESIDUAL_FIELD_INCLUDE_FINAL_TEST_TARGETS=0",
        "for seed in 20522 20623",
        "for seed in 20421 20522 20623",
        "DependencyNeverSatisfied",
        "full_validation",
        "final_test",
        "CONFIRM_FINAL_TEST=1",
    ):
        assert expected in submitter
    assert "reu-aisocial" in wrapper
    assert "/home/ryreu/miniforge3-aarch64" in wrapper
    assert "export PYTHONNOUSERSITE=1" in wrapper
    assert "--run-id P7b" in p7b
    assert "--evaluate-final-test" not in p7b
    assert "--confirm-final-test" not in p7b
    assert "--oracle-teacher-logits-dir" not in p7b
    assert "--confirm-final-test" in final_runner
    assert "--target-cache-dir" not in final_runner
