from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_adaptive_binary_bootstrap_storage_projection import (
    main as build_projection,
)
from scripts.compile_adaptive_binary_bootstrap_single_path_acceptance import (
    main as compile_single_path,
)
from teacher_logit_reco.adaptive_binary_pseudooffline import (
    ABPH_RUNTIME_PROFILE_BUCKETS,
    ABPH_RUNTIME_PROFILE_CONTRACT,
)
import scripts.prune_adaptive_binary_prepared_root as prepared_prune


def _json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_bootstrap_projection_is_bound_to_fresh_root(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared"
    fresh = tmp_path / "fresh"
    (prepared / "inputs" / "hlt_cache").mkdir(parents=True)
    (prepared / "inputs" / "hlt_cache" / "model_train.npz").write_bytes(b"h" * 1000)
    (prepared / "targets" / "model_train").mkdir(parents=True)
    (prepared / "targets" / "model_train" / "shard.npz").write_bytes(b"t" * 2000)
    _json(prepared / "audits" / "actual_target_feasibility.json", {"ok": True})
    _json(prepared / "runs" / "A0_hlt_part" / "run_report.json", {"ok": True})
    (prepared / "runs" / "A0_hlt_part" / "best_model_val.pt").write_bytes(b"c" * 128)
    output = tmp_path / "projection.json"
    assert build_projection(
        [
            "--prepared-root",
            str(prepared),
            "--campaign-root",
            str(fresh),
            "--output",
            str(output),
        ]
    ) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert Path(payload["campaign_root"]) == fresh.resolve()
    assert payload["storage_profile"] == "streaming_30gb_v1"
    assert payload["projected_peak_persistent_bytes"] < 24_000_000_000
    assert {row["artifact_family"] for row in payload["rows"]} >= {
        "transient_hlt_offline_inputs",
        "shared_transient_compact_targets",
        "reconstructor_renderer_selected_checkpoints",
    }


def _runtime_run(
    path: Path,
    *,
    profiled: bool,
    seconds: float,
    matched_pair_id: str = "12345:D1_kt32_mh4_particles",
    slurm_job_id: str = "12345",
) -> None:
    _json(path / "run_report.json", {"ok": True})
    _json(
        path / "training_curves.json",
        {"evaluations": [{"model_val_rollout": {"selection_score": 1.0}}]},
    )
    _json(
        path / "wall_time.json",
        {
            "contract": "adaptive_binary_runtime_walltime_v2",
            "elapsed_seconds": seconds,
            "runtime_profile_enabled": profiled,
            "allocation_identity": {
                "hostname": "gh-a-001.rc.rit.edu",
                "slurm_job_id": slurm_job_id,
                "slurm_job_nodelist": "gh-a-001",
                "matched_pair_id": matched_pair_id,
            },
        },
    )
    _json(
        path / "runtime_profile.json",
        {
            "contract": ABPH_RUNTIME_PROFILE_CONTRACT,
            "ok": profiled,
            "summary": {
                "sampled_training_updates": 20 if profiled else 0,
                "validation_count": 1 if profiled else 0,
            },
            "buckets": {
                name: {
                    "samples": (
                        1
                        if profiled
                        and name in {"optimizer_update_total", "full_validation"}
                        else 0
                    )
                }
                for name in ABPH_RUNTIME_PROFILE_BUCKETS
            },
        },
    )


def test_single_path_bootstrap_uses_measured_matched_runs(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    profiled = tmp_path / "profiled"
    _runtime_run(plain, profiled=False, seconds=100.0)
    _runtime_run(profiled, profiled=True, seconds=101.0)
    output = tmp_path / "single_path.json"
    assert compile_single_path(
        [
            "--uninstrumented-run",
            str(plain),
            "--instrumented-run",
            str(profiled),
            "--output",
            str(output),
        ]
    ) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["dense_instrumentation_overhead_fraction"] < 0.03
    assert payload["matched_allocation_identity"]["slurm_job_id"] == "12345"
    assert payload["checks"]["metric_and_checkpoint_parity"] is True
    assert set(payload["source_artifacts"]) >= {
        "uninstrumented_reference",
        "instrumented_reference",
        "instrumented_walltime",
    }


def test_single_path_accepts_measured_dense_profile_above_sparse_target(
    tmp_path: Path,
) -> None:
    plain = tmp_path / "plain"
    profiled = tmp_path / "profiled"
    _runtime_run(plain, profiled=False, seconds=100.0)
    _runtime_run(profiled, profiled=True, seconds=105.4)
    output = tmp_path / "single_path.json"
    assert compile_single_path(
        [
            "--uninstrumented-run",
            str(plain),
            "--instrumented-run",
            str(profiled),
            "--output",
            str(output),
        ]
    ) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert (
        payload["advisories"]["dense_instrumentation_overhead_target_below_3_percent"]
        is False
    )
    assert payload["checks"][
        "projected_sparse_instrumentation_overhead_below_3_percent"
    ] is True


def test_single_path_rejects_projected_sparse_overhead_above_ceiling(
    tmp_path: Path,
) -> None:
    plain = tmp_path / "plain"
    profiled = tmp_path / "profiled"
    _runtime_run(plain, profiled=False, seconds=100.0)
    _runtime_run(
        profiled,
        profiled=True,
        seconds=100.0 * 4.5,
    )
    output = tmp_path / "single_path.json"
    assert compile_single_path(
        [
            "--uninstrumented-run",
            str(plain),
            "--instrumented-run",
            str(profiled),
            "--output",
            str(output),
        ]
    ) == 1
    assert json.loads(output.read_text(encoding="utf-8"))["ok"] is False


def test_single_path_rejects_cross_allocation_comparison(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    profiled = tmp_path / "profiled"
    _runtime_run(plain, profiled=False, seconds=100.0)
    _runtime_run(
        profiled,
        profiled=True,
        seconds=101.0,
        slurm_job_id="67890",
        matched_pair_id="67890:D1_kt32_mh4_particles",
    )
    with pytest.raises(ValueError, match="allocation-matched"):
        compile_single_path(
            [
                "--uninstrumented-run",
                str(plain),
                "--instrumented-run",
                str(profiled),
                "--output",
                str(tmp_path / "single_path.json"),
            ]
        )


def test_prepared_prune_is_approved_exact_and_preserves_results(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "adaptive_binary_pseudooffline_prepared"
    evidence = root / "audits" / "bootstrap_test"
    evidence.mkdir(parents=True)
    acceptance = evidence / "runtime_acceptance.json"
    _json(acceptance, {"ok": True})
    for name in ("archives", "targets", "inputs"):
        (root / name).mkdir()
        (root / name / "payload.bin").write_bytes(b"x" * 100)
    (root / "runs").mkdir()
    (root / "runs" / "result.json").write_text("{}", encoding="utf-8")
    stale_evidence = root / "audits" / "bootstrap_stale"
    stale_evidence.mkdir()
    (stale_evidence / "old-profile.json").write_bytes(b"z" * 100)
    monkeypatch.setattr(
        prepared_prune, "require_runtime_acceptance", lambda *args, **kwargs: {"ok": True}
    )
    receipt = evidence / "receipt.json"
    assert prepared_prune.main(
        [
            "--prepared-root",
            str(root),
            "--bootstrap-evidence-root",
            str(evidence),
            "--runtime-acceptance",
            str(acceptance),
            "--maximum-retained-bytes",
            "1000000",
            "--approve-prune",
            "--output",
            str(receipt),
        ]
    ) == 0
    assert all(not (root / name).exists() for name in ("archives", "targets", "inputs"))
    assert not stale_evidence.exists()
    assert evidence.is_dir()
    assert (root / "runs" / "result.json").is_file()
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["removed_stale_bootstrap_directories"] == [
        {"bytes": 100, "path": str(stale_evidence.resolve())}
    ]
