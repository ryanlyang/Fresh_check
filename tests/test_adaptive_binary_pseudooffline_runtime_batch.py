from __future__ import annotations

import json

import pytest
import torch

from teacher_logit_reco.adaptive_binary_pseudooffline import (
    ExponentialMovingAverage,
    FullStepBatchMeasurement,
    ReconstructorTrainerConfig,
    calibrate_runtime_batch_contract,
    exact_accumulation_steps,
    load_runtime_batch_contract,
    measure_full_optimizer_step,
    write_runtime_batch_contract,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.runtime_batch import (
    ABPH_RUNTIME_BATCH_MEASUREMENT_PRODUCER,
)


def _measurement(
    family: str,
    local: int,
    accumulation: int,
    *,
    requested_world: int = 4,
    measured_world: int = 4,
    free_fraction: float = 0.20,
    successful: bool = True,
    variant_name: str = "D1_kt32_mh4_particles",
    config_hash: str = "config-hash",
    provenance_hash: str = "provenance-hash",
) -> FullStepBatchMeasurement:
    total = 1000
    return FullStepBatchMeasurement(
        stage_family=family,
        variant_name=variant_name,
        resolved_variant_config_hash=config_hash,
        runtime_provenance_hash=provenance_hash,
        measurement_producer=ABPH_RUNTIME_BATCH_MEASUREMENT_PRODUCER,
        slurm_job_id="12345",
        slurm_job_account="reu-aisocial",
        slurm_job_partition="tigris",
        local_batch_size=local,
        accumulation_steps=accumulation,
        requested_world_size=requested_world,
        measured_world_size=measured_world,
        rank_count=measured_world,
        distributed_backend="nccl" if measured_world > 1 else "none",
        successful=successful,
        all_ranks_completed=successful,
        process_group_initialized=measured_world > 1,
        ddp_gradient_buckets_initialized=measured_world > 1,
        find_unused_parameters_exercised=True,
        active_parameter_groups=("hlt_encoder", "root"),
        forward_completed=successful,
        backward_completed=successful,
        gradients_unscaled=successful,
        gradients_clipped=successful,
        optimizer_step_completed=successful,
        adamw_state_materialized=successful,
        online_model_resident=True,
        ema_model_resident=True,
        prefetch_buffer_count=2,
        pinned_memory_staging=True,
        largest_path_exercised=True,
        total_device_memory_bytes=total,
        peak_device_memory_bytes=int(total * (1.0 - free_fraction)),
        free_device_memory_bytes_at_peak=int(total * free_fraction),
        rank_measurement_hashes=tuple(f"rank-{index}" for index in range(measured_world)),
        failure=None if successful else "CUDA out of memory",
    )


def test_locked_ddp4_batch_arithmetic_matches_the_plan():
    assert exact_accumulation_steps(
        "root_hierarchy", world_size=4, local_batch_size=128
    ) == 2
    assert exact_accumulation_steps(
        "renderer_distribution", world_size=4, local_batch_size=128
    ) == 1
    with pytest.raises(ValueError, match="does not exactly divide"):
        exact_accumulation_steps("renderer_distribution", world_size=4, local_batch_size=96)


def test_calibration_descends_after_headroom_failure_and_selects_largest_safe_batch():
    calls = []

    def probe(family, local, accumulation):
        calls.append((family, local))
        free = 0.10 if (family, local) == ("root_hierarchy", 256) else 0.20
        return _measurement(family, local, accumulation, free_fraction=free)

    contract = calibrate_runtime_batch_contract(
        variant_name="D1_kt32_mh4_particles",
        resolved_variant_config_hash="config-hash",
        runtime_provenance_hash="provenance-hash",
        requested_world_size=4,
        probe=probe,
    )
    assert contract.selections["root_hierarchy"].local_batch_size == 128
    assert contract.selections["root_hierarchy"].accumulation_steps == 2
    assert contract.selections["renderer_distribution"].local_batch_size == 128
    assert calls == [
        ("root_hierarchy", 256),
        ("root_hierarchy", 128),
        ("renderer_distribution", 128),
    ]


def test_single_rank_evidence_cannot_approve_a_requested_ddp4_contract():
    def probe(family, local, accumulation):
        return _measurement(
            family,
            local,
            accumulation,
            requested_world=4,
            measured_world=1,
        )

    with pytest.raises(RuntimeError, match="no root_hierarchy candidate"):
        calibrate_runtime_batch_contract(
            variant_name="D1_kt32_mh4_particles",
            resolved_variant_config_hash="config-hash",
            runtime_provenance_hash="provenance-hash",
            requested_world_size=4,
            probe=probe,
        )


def test_calibration_rejects_measurement_for_stale_variant_provenance():
    with pytest.raises(ValueError, match="wrong variant"):
        calibrate_runtime_batch_contract(
            variant_name="D1_kt32_mh4_particles",
            resolved_variant_config_hash="config-hash",
            runtime_provenance_hash="provenance-hash",
            requested_world_size=4,
            probe=lambda family, local, accumulation: _measurement(
                family,
                local,
                accumulation,
                variant_name="B1_root_deterministic",
            ),
        )


def test_contract_is_hash_bound_immutable_and_expected_input_checked(tmp_path):
    contract = calibrate_runtime_batch_contract(
        variant_name="B1_root_deterministic",
        resolved_variant_config_hash="config-hash",
        runtime_provenance_hash="provenance-hash",
        requested_world_size=4,
        probe=lambda family, local, accumulation: _measurement(
            family,
            local,
            accumulation,
            variant_name="B1_root_deterministic",
        ),
    )
    path = tmp_path / "runtime_batch_contract.json"
    write_runtime_batch_contract(path, contract)
    write_runtime_batch_contract(path, contract)
    loaded = load_runtime_batch_contract(
        path,
        expected_variant_name="B1_root_deterministic",
        expected_resolved_variant_config_hash="config-hash",
        expected_runtime_provenance_hash="provenance-hash",
        expected_world_size=4,
    )
    assert loaded.contract_hash == contract.contract_hash
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["selections"]["root_hierarchy"]["local_batch_size"] = 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_runtime_batch_contract(path)


def test_full_step_probe_materializes_gradients_adamw_and_ema_on_cpu():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    ema = ExponentialMovingAverage(model, 0.99)

    def batch_factory(batch_size):
        return torch.ones(batch_size, 2)

    def forward_loss(module, batch):
        prediction = module(batch)
        loss = prediction.square().mean()
        return {"total_loss": loss, "root_loss": loss}

    measurement = measure_full_optimizer_step(
        stage_family="root_hierarchy",
        variant_name="B1_root_deterministic",
        resolved_variant_config_hash="config-hash",
        runtime_provenance_hash="provenance-hash",
        slurm_job_id="12345",
        slurm_job_account="reu-aisocial",
        slurm_job_partition="tigris",
        local_batch_size=256,
        requested_world_size=1,
        model=model,
        optimizer=optimizer,
        ema=ema,
        batch_factory=batch_factory,
        forward_loss=forward_loss,
        active_parameter_groups=("root",),
        device=torch.device("cpu"),
        gradient_clip_norm=1.0,
        find_unused_parameters=False,
        largest_path_exercised=True,
        prefetch_buffers=({}, {}),
        pinned_memory_staging=True,
    )
    assert measurement.successful is True
    assert measurement.optimizer_step_completed is True
    assert measurement.adamw_state_materialized is True
    assert measurement.ema_model_resident is True
    assert measurement.production_rejections() == ()


def test_trainer_config_rejects_non_exact_phase_specific_contract():
    with pytest.raises(ValueError, match="runtime batch arithmetic"):
        ReconstructorTrainerConfig(
            output_dir="unused",
            distributed_world_size=4,
            root_hierarchy_local_batch_size=128,
            renderer_distribution_local_batch_size=128,
            root_hierarchy_gradient_accumulation_steps=1,
            renderer_distribution_gradient_accumulation_steps=1,
        )
