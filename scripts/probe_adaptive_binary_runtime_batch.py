#!/usr/bin/env python3
"""Measure one provenance-bound ABPH optimizer-step batch candidate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.hlt_baseline import require_torch, resolve_device  # noqa: E402
from teacher_logit_reco.adaptive_binary_pseudooffline import (  # noqa: E402
    ABPH_LEVEL_CAPACITIES,
    AdaptiveBinaryReconstructorModel,
    AdaptiveBinaryTargetBatchSource,
    CurriculumState,
    ExponentialMovingAverage,
    ReconstructorStepContext,
    ReconstructorTrainerConfig,
    ReconstructorTrainingModule,
    build_reconstructor_optimizer,
    build_stage_ddp_wrapper,
    canonical_hash,
    compose_reconstruction_loss,
    configure_reconstructor_optimizer,
    measure_full_optimizer_step,
    reconstructor_runtime_provenance,
    reconstructor_step,
    resolve_variant_config,
    write_full_step_measurement,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.distributed import (  # noqa: E402
    barrier,
    destroy_distributed_runtime,
    distributed_environment,
    initialize_distributed_runtime,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.input_pipeline import (  # noqa: E402
    _move_nested_to_device,
    prepare_contiguous_cpu_batch,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument(
        "--stage-family",
        choices=("root_hierarchy", "renderer_distribution"),
        required=True,
    )
    parser.add_argument("--local-batch-size", type=int, required=True)
    parser.add_argument("--expected-world-size", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    return parser


def _probe_state(resolved: dict, family: str) -> CurriculumState:
    hierarchy = resolved["model"]["hierarchy"]
    capacities = tuple(int(value) for value in hierarchy.get("capacities", ()))
    active_capacity = max(capacities, default=1)
    renderer = bool(resolved["model"]["renderer"].get("enabled"))
    distribution = bool(resolved["model"]["distribution"].get("enabled"))
    if family == "root_hierarchy":
        phase = 2 if active_capacity > 1 else 1
    elif distribution:
        phase = 4
    elif renderer:
        phase = 3
    else:
        phase = 2 if active_capacity > 1 else 1
    names = {
        1: "root_pretraining",
        2: "progressive_hierarchy",
        3: "particle_renderer",
        4: "distribution_refinement",
    }
    return CurriculumState(
        stage_index=0,
        stage_key=f"runtime_probe_{family}",
        phase=phase,
        phase_name=names[phase],
        global_update=1,
        stage_update=1,
        stage_maximum_updates=2,
        active_capacity=active_capacity,
        stage_progress=0.5,
        teacher_forcing_probability=0.0,
        distribution_weight=1.0,
        supervised_capacities=tuple(
            value for value in ABPH_LEVEL_CAPACITIES if value <= active_capacity
        ),
        stage_nominal_updates=2,
        stage_extension_updates=1,
        stage_hard_max_updates=2,
        schedule_contract="runtime_batch_probe_v1",
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    torch = require_torch()
    rank, world_size, _local_rank = distributed_environment()
    if world_size != int(args.expected_world_size):
        raise ValueError("Slurm world size differs from the requested probe topology")
    device = resolve_device(args.device)
    runtime = initialize_distributed_runtime(
        requested_world_size=world_size, device=device
    )
    if device.type == "cuda":
        device = torch.device("cuda", runtime.local_rank)

    root = Path(args.campaign_root)
    resolved = resolve_variant_config(args.variant)
    grouping = str(
        resolved["model"]["hierarchy"].get("grouping", "exclusive_kt")
    )
    source = AdaptiveBinaryTargetBatchSource(
        hlt_cache_dir=root / "inputs" / "hlt_cache",
        target_cache_dir=root / "targets",
        split="model_train",
        grouping=grouping,
        batch_size=int(args.local_batch_size),
        shuffle_shards=False,
        seed=24731,
        rank=rank,
        world_size=world_size,
    )
    provenance = reconstructor_runtime_provenance(
        variant_name=args.variant,
        target_metadata=source.metadata,
        hlt_metadata=source.hlt_view.metadata,
    )
    provenance_hash = canonical_hash(provenance)
    model = AdaptiveBinaryReconstructorModel(
        hierarchy_names=(grouping,), variant_name=args.variant, smoke=False
    ).to(device)
    module_groups = model.module_groups()
    optimizer = build_reconstructor_optimizer(model, module_groups)
    state = _probe_state(resolved, args.stage_family)
    group_rows = configure_reconstructor_optimizer(
        optimizer,
        state,
        ReconstructorTrainerConfig(
            output_dir=str(root / "audits" / "runtime_batch_probe_scratch"),
            device=str(device),
        ),
    )
    active_groups = tuple(
        row["group_name"] for row in group_rows if row["trainable"]
    )
    training_module = ReconstructorTrainingModule(
        model,
        reconstructor_step,
        compose_reconstruction_loss,
        None,
    ).to(device)
    wrapped = build_stage_ddp_wrapper(
        training_module,
        runtime,
        device=device,
        find_unused_parameters=True,
    )
    ema = ExponentialMovingAverage(model, 0.999)
    context = ReconstructorStepContext(
        curriculum=state,
        split="model_train",
        mode="rollout",
        validation=False,
        teacher_forcing_probability=0.0,
        stochastic_seed=24731,
    )

    cpu_buffers = [
        prepare_contiguous_cpu_batch(
            source.next_batch(), pin_memory=device.type == "cuda"
        )
        for _ in range(2)
    ]
    buffer_cursor = 0

    def batch_factory(local_batch_size: int):
        nonlocal buffer_cursor
        if int(local_batch_size) != int(args.local_batch_size):
            raise ValueError("probe requested an unexpected local batch size")
        if buffer_cursor < len(cpu_buffers):
            cpu_batch = cpu_buffers[buffer_cursor]
            buffer_cursor += 1
        else:
            cpu_batch = prepare_contiguous_cpu_batch(
                source.next_batch(), pin_memory=device.type == "cuda"
            )
        return _move_nested_to_device(
            cpu_batch, device=device, non_blocking=device.type == "cuda"
        )

    def forward_loss(active_model, batch):
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            return active_model(batch, context)

    measurement = measure_full_optimizer_step(
        stage_family=args.stage_family,
        variant_name=args.variant,
        resolved_variant_config_hash=resolved["resolved_config_hash"],
        runtime_provenance_hash=provenance_hash,
        slurm_job_id=os.environ.get("SLURM_JOB_ID", ""),
        slurm_job_account=os.environ.get("SLURM_JOB_ACCOUNT", ""),
        slurm_job_partition=os.environ.get("SLURM_JOB_PARTITION", ""),
        local_batch_size=int(args.local_batch_size),
        requested_world_size=world_size,
        model=wrapped,
        optimizer=optimizer,
        ema=ema,
        batch_factory=batch_factory,
        forward_loss=forward_loss,
        active_parameter_groups=active_groups,
        device=device,
        gradient_clip_norm=1.0,
        find_unused_parameters=True,
        largest_path_exercised=True,
        prefetch_buffers=cpu_buffers,
        pinned_memory_staging=device.type == "cuda",
    )
    if runtime.is_primary:
        write_full_step_measurement(args.output, measurement)
        print(json.dumps(measurement.to_dict(), indent=2, sort_keys=True))
    barrier(runtime)
    destroy_distributed_runtime(runtime)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
