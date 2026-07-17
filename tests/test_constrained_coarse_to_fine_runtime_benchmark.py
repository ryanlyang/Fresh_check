from __future__ import annotations

import json

from jetclass_fresh.jetclass_data import FileRecord, JetIdentity, SPLIT_ORDER, SplitManifest, manifest_hash, save_split_manifest
from scripts.build_constrained_coarse_to_fine_runtime_benchmark_plan import (
    BENCHMARK_PLAN_CONTRACT,
    build_benchmark_runs,
)
from scripts.write_constrained_coarse_to_fine_runtime_benchmark_report import write_report
from teacher_logit_reco.constrained_coarse_to_fine.runtime_selection import (
    RuntimeAcceptanceConfig,
    select_accelerated_candidate,
)


CODE_ENVIRONMENT = {
    "source_tree_clean": True,
    "code_environment_hash": "clean-test-environment",
}


def _manifest() -> SplitManifest:
    splits = {split: [] for split in SPLIT_ORDER}
    splits["model_train"] = [JetIdentity(file="a.root", entry=0, label=0)]
    splits["model_val"] = [JetIdentity(file="a.root", entry=1, label=0)]
    return SplitManifest(
        data_dir="toy",
        max_constits=8,
        class_names=["class0"],
        file_prefix_to_label={"class0": 0},
        split_sizes={split: len(rows) for split, rows in splits.items()},
        split_seeds={split: index for index, split in enumerate(SPLIT_ORDER)},
        file_records=[FileRecord(path="a.root", label=0, num_entries=2)],
        splits=splits,
        metadata={},
    )


def _epoch(epoch: int) -> dict[str, object]:
    metrics = {
        "selection.reconstruction_score": 1.0 / (epoch + 1),
        "loss.total": 1.0,
        "loss.hierarchy": 0.5,
        "loss.slot": 0.5,
        "hierarchy.component.global_accounting": 0.1,
        "hierarchy.component.level1_accounting": 0.1,
        "slot.component.match": 0.1,
        "hierarchy.metric.global_total_pT_relative_mae": 0.1,
        "hierarchy.metric.level1_accounting_mae": 0.1,
        "hierarchy.metric.level1_parent_child_consistency_max": 0.1,
        "slot.metric.matched_pT_mae": 0.1,
        "slot.metric.matched_eta_mae": 0.1,
        "slot.metric.matched_phi_mae": 0.1,
        "slot.metric.matched_pid_accuracy": 0.9,
        "train.grad_norm_before_clip": 1.0,
        "runtime.elapsed_seconds": 2.0,
        "runtime.batches_per_second": 3.0,
        "runtime.jets_per_second": 6.0,
        "runtime.cuda_peak_allocated_bytes": 100,
        "runtime.cuda_peak_reserved_bytes": 200,
        "runtime.host_max_rss_bytes": 300,
        "runtime.batch_loading_seconds": 0.25,
        "runtime.cpu_process_seconds": 1.0,
        "runtime.cpu_process_utilization": 50.0,
        "nonfinite_batches_skipped": 0,
        "n_jets": 12,
    }
    return {"epoch": epoch, "train": dict(metrics), "model_val": dict(metrics)}


def _write_run(root, row, calibration_hash: str) -> None:
    run_dir = root / "reconstructors" / row.run_id
    run_dir.mkdir(parents=True)
    config = {
        "precision_mode": row.precision_mode,
        "batch_size": row.train_batch_size,
        "eval_batch_size": row.eval_batch_size,
        "num_workers": row.num_workers,
        "lr_schedule": row.lr_schedule,
        "learning_rate": row.learning_rate,
        "hlt_encoder_lr_scale": row.hlt_encoder_lr_scale,
        "hungarian_executor": row.hungarian_executor,
        "hungarian_workers": row.hungarian_workers,
        "fixed_horizon": True,
    }
    provenance = {split: {"source_manifest_hash": calibration_hash} for split in ("model_train", "model_val")}
    (run_dir / "run_report.json").write_text(
        json.dumps(
            {
                "ok": True,
                "variant": row.variant,
                "runtime_profile": row.runtime_profile,
                "training_config": config,
                "provenance": provenance,
                "stop_reason": "fixed_horizon_completed",
                "completed_epochs": 3,
                "best_model_val": {"selection.reconstruction_score": 0.2},
                "code_environment": CODE_ENVIRONMENT,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "training_curves.json").write_text(json.dumps({"epochs": [_epoch(0), _epoch(1), _epoch(2)]}), encoding="utf-8")
    (run_dir / "memory_preflight.json").write_text(json.dumps({"ok": True}), encoding="utf-8")


def _write_calibration_validation(root, calibration_hash: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    splits = {
        split: {
            "hlt_content_hash": f"hlt-{split}",
            "offline_content_hash": f"offline-{split}",
            "target_content_hash": f"target-{split}",
            "jet_identity_hash": f"identity-{split}",
        }
        for split in ("model_train", "model_val")
    }
    (root / "runtime_benchmark_calibration_validation.json").write_text(
        json.dumps({"ok": True, "calibration_manifest_hash": calibration_hash, "splits": splits}),
        encoding="utf-8",
    )


def test_benchmark_matrix_covers_ad_and_c4_executor_sweep() -> None:
    rows = build_benchmark_runs(
        single_view_candidates=("32:64",),
        c6_candidates=("16:32",),
        c4_candidates=("16:32",),
        input_workers=(0, 4),
        c4_hungarian_workers=(1, 4),
    )
    assert {row.variant for row in rows} == {"C1", "C5-B3", "C6", "C4"}
    assert {row.matrix_case for row in rows if row.variant == "C1"} == {"A", "B", "C", "D"}
    assert {row.matrix_case for row in rows if row.variant == "C4"} == {"A", "B_serial", "B_thread", "C", "D"}
    assert len({row.run_id for row in rows}) == len(rows)
    assert all(row.prefetch_factor is None for row in rows if row.num_workers == 0)
    assert all(row.prefetch_factor == 4 for row in rows if row.num_workers > 0)


def test_runtime_benchmark_report_rejects_mismatched_execution_contract(tmp_path) -> None:
    manifest = _manifest()
    manifest_path = tmp_path / "calibration.json.gz"
    save_split_manifest(manifest, manifest_path)
    rows = build_benchmark_runs(
        single_view_candidates=("32:64",),
        c6_candidates=("16:32",),
        c4_candidates=("16:32",),
        input_workers=(0,),
        c4_hungarian_workers=(1,),
    )
    plan_path = tmp_path / "plan.json"
    calibration_root = tmp_path / "calibration"
    calibration_hash = manifest_hash(manifest)
    _write_calibration_validation(calibration_root, calibration_hash)
    plan_path.write_text(
        json.dumps(
            {
                "contract": BENCHMARK_PLAN_CONTRACT,
                "calibration_root": str(calibration_root),
                "calibration_manifest": str(manifest_path),
                "epochs": 3,
                "runs": [row.__dict__ for row in rows],
            }
        ),
        encoding="utf-8",
    )
    benchmark_root = tmp_path / "benchmarks"
    for row in rows:
        _write_run(benchmark_root, row, calibration_hash)
    output = benchmark_root / "runtime_benchmark_report.json"
    report = write_report(plan_path=plan_path, benchmark_root=benchmark_root, output=output)
    assert report["ok"]
    assert len(report["records"]) == len(rows)
    assert report["records"][0]["resource"]["peak_cuda_reserved_bytes"] == 200.0

    broken = benchmark_root / "reconstructors" / rows[0].run_id / "run_report.json"
    payload = json.loads(broken.read_text(encoding="utf-8"))
    payload["training_config"]["batch_size"] = 999
    broken.write_text(json.dumps(payload), encoding="utf-8")
    rejected = write_report(plan_path=plan_path, benchmark_root=benchmark_root, output=output)
    assert not rejected["ok"]
    assert "training_config.batch_size" in " ".join(rejected["problems"][rows[0].run_id])


def test_three_epoch_selection_requires_one_shared_lr_and_clean_environment(tmp_path) -> None:
    manifest = _manifest()
    manifest_path = tmp_path / "calibration.json.gz"
    save_split_manifest(manifest, manifest_path)
    rows = build_benchmark_runs(
        single_view_candidates=("32:64",),
        c6_candidates=("16:32",),
        c4_candidates=("16:32",),
        input_workers=(0,),
        c4_hungarian_workers=(1,),
        peak_learning_rates=(2.0e-4,),
    )
    plan_path = tmp_path / "plan.json"
    calibration_root = tmp_path / "calibration"
    calibration_hash = manifest_hash(manifest)
    _write_calibration_validation(calibration_root, calibration_hash)
    plan_path.write_text(
        json.dumps(
            {
                "contract": BENCHMARK_PLAN_CONTRACT,
                "calibration_root": str(calibration_root),
                "calibration_manifest": str(manifest_path),
                "epochs": 3,
                "runs": [row.__dict__ for row in rows],
            }
        ),
        encoding="utf-8",
    )
    benchmark_root = tmp_path / "benchmarks"
    for row in rows:
        _write_run(benchmark_root, row, calibration_hash)
    report = write_report(
        plan_path=plan_path,
        benchmark_root=benchmark_root,
        output=benchmark_root / "runtime_benchmark_report.json",
    )
    candidate = select_accelerated_candidate(
        report,
        config=RuntimeAcceptanceConfig(gpu_memory_bytes=1024**3),
        code_environment=CODE_ENVIRONMENT,
    )
    assert candidate["status"] == "accelerated_candidate_v1"
    assert candidate["shared_optimizer"]["learning_rate"] == 2.0e-4
    assert set(candidate["execution_by_variant"]) == {"C1", "C5-B3", "C6", "C4"}
