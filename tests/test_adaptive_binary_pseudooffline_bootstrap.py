from __future__ import annotations

import json
from pathlib import Path

from scripts.build_adaptive_binary_bootstrap_storage_projection import (
    main as build_projection,
)
from scripts.compile_adaptive_binary_bootstrap_single_path_acceptance import (
    main as compile_single_path,
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


def _runtime_run(path: Path, *, profiled: bool, seconds: float) -> None:
    _json(path / "run_report.json", {"ok": True})
    _json(
        path / "training_curves.json",
        {"evaluations": [{"model_val_rollout": {"selection_score": 1.0}}]},
    )
    _json(
        path / "wall_time.json",
        {
            "contract": "adaptive_binary_runtime_walltime_v1",
            "elapsed_seconds": seconds,
            "runtime_profile_enabled": profiled,
        },
    )
    _json(
        path / "runtime_profile.json",
        {"ok": profiled, "summary": {"sampled_updates": 20 if profiled else 0}},
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
    assert payload["instrumentation_overhead_fraction"] < 0.03
    assert payload["checks"]["metric_and_checkpoint_parity"] is True
    assert set(payload["source_artifacts"]) >= {
        "uninstrumented_reference",
        "instrumented_reference",
        "instrumented_walltime",
    }


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
    assert (root / "runs" / "result.json").is_file()
    assert json.loads(receipt.read_text(encoding="utf-8"))["ok"] is True
