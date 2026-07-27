from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from teacher_logit_reco.adaptive_binary_pseudooffline import (
    ABPH_DDP_SMOKE_CONTRACT,
    ABPH_RUNTIME_BENCHMARK_CONTRACT,
    ABPH_RUNTIME_BENCHMARK_VALIDATION_POLICY,
    ABPH_RUNTIME_PROFILE_BUCKETS,
    ABPH_RUNTIME_PROFILE_CONTRACT,
    ABPH_SINGLE_PATH_ACCEPTANCE_CONTRACT,
    FullStepBatchMeasurement,
    build_extension_comparison_report,
    build_runtime_acceptance_report,
    calibrate_runtime_batch_contract,
    canonical_hash,
    require_runtime_acceptance,
    write_runtime_batch_contract,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.runtime_batch import (
    ABPH_RUNTIME_BATCH_MEASUREMENT_PRODUCER,
)


ROOT_VARIANT = "B1_semantic_query_root"
DEEP_VARIANT = "D1_kt32_mh4_particles"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _benchmark(
    path: Path,
    *,
    variant: str,
    world_size: int,
    sampled_jets: int,
    update_seconds: float,
    validation_seconds: float,
    score: float = 1.0,
) -> Path:
    validation_jets = 4_096
    ordered_ranges = [
        {
            "rank": rank,
            "start": validation_jets * rank // world_size,
            "stop": validation_jets * (rank + 1) // world_size,
        }
        for rank in range(world_size)
    ]
    coverage = {
        "contract": "adaptive_binary_validation_coverage_v1",
        "split": "model_val",
        "n_jets": validation_jets,
        "world_size": world_size,
        "ordered_ranges": ordered_ranges,
    }
    coverage["validation_coverage_hash"] = canonical_hash(coverage)
    coverage["expected_ordered_row_hash"] = canonical_hash(ordered_ranges)
    coverage["selection_eligible"] = True
    buckets = {
        name: {
            "cpu_total_seconds": 0.0,
            "cuda_total_seconds": None,
            "synchronized_wall_total_seconds": 0.0,
        }
        for name in ABPH_RUNTIME_PROFILE_BUCKETS
    }
    buckets["optimizer_update_total"]["cpu_total_seconds"] = update_seconds
    buckets["optimizer_update_total"][
        "synchronized_wall_total_seconds"
    ] = update_seconds
    buckets["full_validation"]["cpu_total_seconds"] = validation_seconds
    buckets["full_validation"][
        "synchronized_wall_total_seconds"
    ] = validation_seconds
    buckets["gradient_synchronization"]["cpu_total_seconds"] = (
        update_seconds * 0.20 if world_size > 1 else 0.0
    )
    seconds_per_update = float(update_seconds) / 20.0
    if variant == DEEP_VARIANT:
        stage_samples = {
            "phase1_root": 1,
            **{
                f"phase2_hierarchy_{capacity}": 1
                for capacity in (2, 4, 8, 16, 32)
            },
            "phase3_renderer": 1,
            "phase4_distribution": 13,
        }
    else:
        stage_samples = {"phase1_root": 20}
    profile = {
        "contract": ABPH_RUNTIME_PROFILE_CONTRACT,
        "ok": True,
        "buckets": buckets,
        "stages": {
            name: {
                "phase": (
                    1
                    if name == "phase1_root"
                    else 2
                    if name.startswith("phase2_")
                    else 3
                    if name == "phase3_renderer"
                    else 4
                ),
                "phase_name": name,
                "sampled_jets": sampled_jets * samples // 20,
                "sampled_updates": samples,
                "median_optimizer_update_seconds": seconds_per_update,
            }
            for name, samples in stage_samples.items()
        },
        "summary": {
            "communication_fraction": 0.20 if world_size > 1 else 0.0,
            "peak_reserved_bytes": 500,
        },
        "device": {"total_memory_bytes": 1000},
    }
    profile["profile_content_hash"] = canonical_hash(profile)
    _write_json(path / "runtime_profile.json", profile)
    report = {
        "ok": True,
        "variant_name": variant,
        "distributed_runtime": {"world_size": world_size},
        "runtime_profile": {
            "profile_content_hash": profile["profile_content_hash"]
        },
        "runtime_reference_benchmark": {
            "contract": ABPH_RUNTIME_BENCHMARK_CONTRACT,
            "validation_policy": ABPH_RUNTIME_BENCHMARK_VALIDATION_POLICY,
            "fixed_model_val_evaluations": 1,
            "curriculum_transition_validations": 0,
            "validation_jets": validation_jets,
        },
        "provenance": {"manifest_hash": "same", "hlt_cache_hash": "same"},
    }
    _write_json(path / "run_report.json", report)
    _write_json(
        path / "training_curves.json",
        {
            "evaluations": [
                {
                    "model_val_rollout": {
                        "selection_score": score,
                        "n_jets": validation_jets,
                        "validation_coverage": coverage,
                    }
                }
            ]
        },
    )
    return path


def _smoke(path: Path, *, world_size: int) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    state_path = path / "state.pt"
    state = {
        "model_state_dict": {
            "weight": torch.tensor([[0.25, -0.5]], dtype=torch.float32)
        },
        "ema_state_dict": {
            "weight": torch.tensor([[0.25, -0.5]], dtype=torch.float32)
        },
    }
    torch.save(state, state_path)
    checks = {
        "global_identity_coverage": True,
        "checkpoint_resume": True,
        "forward_failure_consensus": True,
        "wrapper_rebuild": True,
    }
    payload = {
        "contract": ABPH_DDP_SMOKE_CONTRACT,
        "ok": True,
        "world_size": world_size,
        "checks": checks,
        "global_batch_identity_hash": "same-global-batch",
        "validation_mean_loss": 0.75,
        "global_training_mean_loss": 0.5,
        "preclip_gradient_l2_norm": 0.25,
        "state_path": str(state_path.resolve()),
        "state_sha256": _sha256(state_path),
    }
    payload["smoke_content_hash"] = canonical_hash(payload)
    return _write_json(path / "smoke_report.json", payload)


def _measurement(
    family: str,
    local: int,
    accumulation: int,
    *,
    variant_name: str,
    world_size: int = 4,
) -> FullStepBatchMeasurement:
    return FullStepBatchMeasurement(
        stage_family=family,
        variant_name=variant_name,
        resolved_variant_config_hash="config",
        runtime_provenance_hash="runtime",
        measurement_producer=ABPH_RUNTIME_BATCH_MEASUREMENT_PRODUCER,
        slurm_job_id="12345",
        slurm_job_account="reu-aisocial",
        slurm_job_partition="tigris",
        local_batch_size=local,
        accumulation_steps=accumulation,
        requested_world_size=world_size,
        measured_world_size=world_size,
        rank_count=world_size,
        distributed_backend="nccl",
        successful=True,
        all_ranks_completed=True,
        process_group_initialized=True,
        ddp_gradient_buckets_initialized=True,
        find_unused_parameters_exercised=True,
        active_parameter_groups=("root",),
        forward_completed=True,
        backward_completed=True,
        gradients_unscaled=True,
        gradients_clipped=True,
        optimizer_step_completed=True,
        adamw_state_materialized=True,
        online_model_resident=True,
        ema_model_resident=True,
        prefetch_buffer_count=2,
        pinned_memory_staging=True,
        largest_path_exercised=True,
        total_device_memory_bytes=1000,
        peak_device_memory_bytes=700,
        free_device_memory_bytes_at_peak=300,
        rank_measurement_hashes=tuple(f"r{rank}" for rank in range(world_size)),
    )


def _batch_contract(path: Path, variant: str, *, world_size: int = 4) -> Path:
    contract = calibrate_runtime_batch_contract(
        variant_name=variant,
        resolved_variant_config_hash="config",
        runtime_provenance_hash="runtime",
        requested_world_size=world_size,
        probe=lambda family, local, accumulation: _measurement(
            family,
            local,
            accumulation,
            variant_name=variant,
            world_size=world_size,
        ),
    )
    return write_runtime_batch_contract(path, contract)


def _extension(path: Path, variant: str, *, truncated: bool = False) -> Path:
    report = build_extension_comparison_report(
        variant_name=variant,
        nominal_checkpoint_hash="nominal",
        extension_checkpoint_hash="extension",
        matched_a0_artifact_hash="a0",
        frozen_tagger_recipe_hash="tagger",
        nominal_best_loss=1.0,
        extension_best_loss=0.99 if truncated else 0.999,
        nominal_tagging_gain=0.003,
        extension_tagging_gain=-0.003 if truncated else 0.0031,
        initialization_seed=123,
        training_budget_hash="budget",
    )
    return _write_json(path, report)


def _pilot(path: Path) -> Path:
    payload = {
        "ok": True,
        "schedule_screening": {
            "automatic_highdata_promotion_allowed": True,
            "negative_mechanism_conclusion_valid": True,
        },
    }
    payload["report_content_hash"] = canonical_hash(payload)
    return _write_json(path, payload)


def _single_path(path: Path) -> Path:
    sources = {}
    for name in ("uninstrumented", "instrumented", "optimized"):
        source = path.parent / f"{name}.json"
        _write_json(source, {"name": name})
        sources[name] = {"path": str(source.resolve()), "sha256": _sha256(source)}
    payload = {
        "contract": ABPH_SINGLE_PATH_ACCEPTANCE_CONTRACT,
        "ok": True,
        "instrumentation_overhead_fraction": 0.02,
        "projected_production_instrumentation_overhead_fraction": 0.0002,
        "matched_allocation_identity": {
            "hostname": "gh-a-001.rc.rit.edu",
            "slurm_job_id": "12345",
            "slurm_job_nodelist": "gh-a-001",
            "matched_pair_id": "12345:D1_kt32_mh4_particles",
        },
        "deep_training_speedup": 1.4,
        "profiler_explanation": None,
        "checks": {
            "instrumentation": True,
            "coverage": True,
            "parity": True,
            "speed": True,
        },
        "source_artifacts": sources,
    }
    payload["report_content_hash"] = canonical_hash(payload)
    return _write_json(path, payload)


def _evidence(tmp_path: Path, *, deep_ddp_jets: int = 400):
    single = {
        ROOT_VARIANT: _benchmark(
            tmp_path / "single-root",
            variant=ROOT_VARIANT,
            world_size=1,
            sampled_jets=100,
            update_seconds=10.0,
            validation_seconds=10.0,
        ),
        DEEP_VARIANT: _benchmark(
            tmp_path / "single-deep",
            variant=DEEP_VARIANT,
            world_size=1,
            sampled_jets=100,
            update_seconds=10.0,
            validation_seconds=10.0,
        ),
    }
    ddp4 = {
        ROOT_VARIANT: _benchmark(
            tmp_path / "ddp-root",
            variant=ROOT_VARIANT,
            world_size=4,
            sampled_jets=400,
            update_seconds=10.0,
            validation_seconds=4.0,
        ),
        DEEP_VARIANT: _benchmark(
            tmp_path / "ddp-deep",
            variant=DEEP_VARIANT,
            world_size=4,
            sampled_jets=deep_ddp_jets,
            update_seconds=10.0,
            validation_seconds=4.0,
        ),
    }
    contracts = {
        variant: _batch_contract(tmp_path / f"{variant}-batch.json", variant)
        for variant in (ROOT_VARIANT, DEEP_VARIANT)
    }
    smokes = (
        _smoke(tmp_path / "single-smoke", world_size=1),
        _smoke(tmp_path / "ddp-smoke", world_size=4),
    )
    return single, ddp4, contracts, smokes, _single_path(tmp_path / "single-path.json")


def test_acceptance_separates_runtime_pilot_and_highdata_gates(tmp_path: Path):
    single, ddp4, contracts, smokes, single_path = _evidence(tmp_path)
    runtime_only = build_runtime_acceptance_report(
        single_run_dirs=single,
        ddp4_run_dirs=ddp4,
        single_smoke_path=smokes[0],
        ddp4_smoke_path=smokes[1],
        ddp4_batch_contracts=contracts,
        single_path_acceptance=single_path,
        expected_validation_jets=4_096,
    )
    assert runtime_only["promotion"]["ddp4_runtime_approved"] is True
    assert runtime_only["promotion"]["optimized_pilot_submission_allowed"] is False
    projection = runtime_only["runtime_gate"]["deep_stage_aware_projection"]
    assert projection["nominal_updates"] == 25_006
    assert projection["projected_validation_events"] == 19
    assert set(projection["stages"]) == {
        "phase1_root",
        "phase2_hierarchy_2",
        "phase2_hierarchy_4",
        "phase2_hierarchy_8",
        "phase2_hierarchy_16",
        "phase2_hierarchy_32",
        "phase3_renderer",
        "phase4_distribution",
    }
    assert "particle_render_projection" in runtime_only["runtime_gate"][
        "deep_bottleneck_buckets"
    ]

    extensions = {
        variant: _extension(tmp_path / f"{variant}-extension.json", variant)
        for variant in (ROOT_VARIANT, DEEP_VARIANT)
    }
    complete = build_runtime_acceptance_report(
        single_run_dirs=single,
        ddp4_run_dirs=ddp4,
        single_smoke_path=smokes[0],
        ddp4_smoke_path=smokes[1],
        ddp4_batch_contracts=contracts,
        single_path_acceptance=single_path,
        expected_validation_jets=4_096,
        extension_reports=extensions,
        optimized_pilot_report=_pilot(tmp_path / "pilot.json"),
    )
    path = _write_json(tmp_path / "acceptance.json", complete)
    assert require_runtime_acceptance(path, scope="highdata")["ok"] is True


def test_ddp8_seven_day_gate_is_bound_to_ddp4_and_world8_contracts(
    tmp_path: Path,
):
    single, ddp4, contracts, smokes, single_path = _evidence(tmp_path)
    ddp8 = {
        variant: _benchmark(
            tmp_path / f"ddp8-{variant}",
            variant=variant,
            world_size=8,
            sampled_jets=800,
            update_seconds=12.0,
            validation_seconds=2.5,
        )
        for variant in (ROOT_VARIANT, DEEP_VARIANT)
    }
    ddp8_contracts = {
        variant: _batch_contract(
            tmp_path / f"{variant}-ddp8-batch.json",
            variant,
            world_size=8,
        )
        for variant in (ROOT_VARIANT, DEEP_VARIANT)
    }
    report = build_runtime_acceptance_report(
        single_run_dirs=single,
        ddp4_run_dirs=ddp4,
        ddp8_run_dirs=ddp8,
        single_smoke_path=smokes[0],
        ddp4_smoke_path=smokes[1],
        ddp8_smoke_path=_smoke(tmp_path / "ddp8-smoke", world_size=8),
        ddp4_batch_contracts=contracts,
        ddp8_batch_contracts=ddp8_contracts,
        single_path_acceptance=single_path,
        expected_validation_jets=4_096,
    )
    assert report["ok"] is True
    assert report["promotion"]["ddp8_runtime_approved"] is True
    assert (
        report["promotion"]["production_reconstructor_parallelism"] == "ddp8"
    )
    assert (
        report["promotion"]["production_schedule_policy"]
        == "accelerated_screening_v2_7day"
    )
    projection = report["runtime_gate"]["ddp8_seven_day_projection"]
    assert projection["nominal_updates"] == 5_006
    assert projection["projected_validation_events"] == 9
    path = _write_json(tmp_path / "ddp8-acceptance.json", report)
    assert require_runtime_acceptance(path, scope="ddp8_runtime")["ok"] is True


@pytest.mark.parametrize(
    ("deep_ddp8_score", "expected_approved"),
    ((1.024, True), (1.026, False)),
)
def test_ddp8_stochastic_trajectory_tolerance_is_bounded(
    tmp_path: Path,
    deep_ddp8_score: float,
    expected_approved: bool,
) -> None:
    single, ddp4, contracts, smokes, single_path = _evidence(tmp_path)
    ddp8 = {
        variant: _benchmark(
            tmp_path / f"ddp8-{variant}",
            variant=variant,
            world_size=8,
            sampled_jets=800,
            update_seconds=12.0,
            validation_seconds=2.5,
            score=deep_ddp8_score if variant == DEEP_VARIANT else 1.0,
        )
        for variant in (ROOT_VARIANT, DEEP_VARIANT)
    }
    ddp8_contracts = {
        variant: _batch_contract(
            tmp_path / f"{variant}-ddp8-batch.json",
            variant,
            world_size=8,
        )
        for variant in (ROOT_VARIANT, DEEP_VARIANT)
    }
    report = build_runtime_acceptance_report(
        single_run_dirs=single,
        ddp4_run_dirs=ddp4,
        ddp8_run_dirs=ddp8,
        single_smoke_path=smokes[0],
        ddp4_smoke_path=smokes[1],
        ddp8_smoke_path=_smoke(tmp_path / "ddp8-smoke", world_size=8),
        ddp4_batch_contracts=contracts,
        ddp8_batch_contracts=ddp8_contracts,
        single_path_acceptance=single_path,
        expected_validation_jets=4_096,
    )
    assert (
        report["runtime_gate"]["checks"][
            "ddp8_validation_trajectories_compatible"
        ]
        is expected_approved
    )
    assert report["promotion"]["ddp8_runtime_approved"] is expected_approved


def test_deep_speedup_below_promotion_floor_fails_closed(tmp_path: Path):
    single, ddp4, contracts, smokes, single_path = _evidence(
        tmp_path, deep_ddp_jets=150
    )
    report = build_runtime_acceptance_report(
        single_run_dirs=single,
        ddp4_run_dirs=ddp4,
        single_smoke_path=smokes[0],
        ddp4_smoke_path=smokes[1],
        ddp4_batch_contracts=contracts,
        single_path_acceptance=single_path,
        expected_validation_jets=4_096,
    )
    assert report["ok"] is False
    assert report["runtime_gate"]["checks"]["deep_ddp4_speedup_at_least_1p8"] is False


def test_acceptance_rejects_references_with_repeated_full_validation(tmp_path: Path):
    single, ddp4, contracts, smokes, single_path = _evidence(tmp_path)
    report_path = single[DEEP_VARIANT] / "run_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["runtime_reference_benchmark"]["fixed_model_val_evaluations"] = 8
    _write_json(report_path, report)
    with pytest.raises(ValueError, match="exactly one fixed validation"):
        build_runtime_acceptance_report(
            single_run_dirs=single,
            ddp4_run_dirs=ddp4,
            single_smoke_path=smokes[0],
            ddp4_smoke_path=smokes[1],
            ddp4_batch_contracts=contracts,
            single_path_acceptance=single_path,
            expected_validation_jets=4_096,
        )


def test_acceptance_rehashes_evidence_when_reused(tmp_path: Path):
    single, ddp4, contracts, smokes, single_path = _evidence(tmp_path)
    report = build_runtime_acceptance_report(
        single_run_dirs=single,
        ddp4_run_dirs=ddp4,
        single_smoke_path=smokes[0],
        ddp4_smoke_path=smokes[1],
        ddp4_batch_contracts=contracts,
        single_path_acceptance=single_path,
    )
    path = _write_json(tmp_path / "acceptance.json", report)
    assert require_runtime_acceptance(path)["ok"] is True
    curves = single[ROOT_VARIANT] / "training_curves.json"
    curves.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="changed after review"):
        require_runtime_acceptance(path)


def test_acceptance_rejects_cpu_only_timing_without_synchronized_wall(tmp_path: Path):
    single, ddp4, contracts, smokes, single_path = _evidence(tmp_path)
    run_dir = single[ROOT_VARIANT]
    profile_path = run_dir / "runtime_profile.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["buckets"]["optimizer_update_total"].pop(
        "synchronized_wall_total_seconds"
    )
    profile.pop("profile_content_hash")
    profile["profile_content_hash"] = canonical_hash(profile)
    _write_json(profile_path, profile)
    report_path = run_dir / "run_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["runtime_profile"]["profile_content_hash"] = profile[
        "profile_content_hash"
    ]
    _write_json(report_path, report)

    with pytest.raises(ValueError, match="lacks measured throughput"):
        build_runtime_acceptance_report(
            single_run_dirs=single,
            ddp4_run_dirs=ddp4,
            single_smoke_path=smokes[0],
            ddp4_smoke_path=smokes[1],
            ddp4_batch_contracts=contracts,
            single_path_acceptance=single_path,
        )
