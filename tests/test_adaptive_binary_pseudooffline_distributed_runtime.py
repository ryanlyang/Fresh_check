from __future__ import annotations

import gc
import json
import os
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from jetclass_fresh.jetclass_data import JetIdentity
from scripts.run_adaptive_binary_ddp_acceptance_smoke import (
    _parser as ddp_acceptance_smoke_parser,
)

from teacher_logit_reco.adaptive_binary_pseudooffline.distributed import (
    DistributedRuntime,
    ReconstructorTrainingModule,
    abort_distributed_runtime,
    all_reduce_min_bool,
    any_structural_error,
    barrier,
    build_stage_ddp_wrapper,
    destroy_distributed_runtime,
    distributed_environment,
    gather_error_summaries,
    initialize_distributed_runtime,
    parameter_state_hash,
    prepare_model_for_distributed_training,
    require_distributed_normalization_contract,
    require_standard_tensor_mapping,
    tensor_mapping_is_finite,
    verify_common_parameter_state,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.distributed_stream import (
    validation_range_row,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.distributed_validation import (
    TypedValidationAccumulator,
    finalize_typed_validation,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.training import (
    ABPH_RECONSTRUCTOR_MODULE_GROUPS,
    CurriculumController,
    ReconstructorCurriculumConfig,
    ReconstructorStepResult,
    ReconstructorTrainerConfig,
    RuntimeProfileConfig,
    evaluate_reconstructor_rollout,
    train_reconstructor_curriculum,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.runtime_batch_probe import (
    _gather_fixed_evidence,
    measure_full_optimizer_step,
)


def test_ddp_acceptance_smoke_cli_accepts_production_world_sizes() -> None:
    parser = ddp_acceptance_smoke_parser()
    for world_size in (1, 4, 8):
        parsed = parser.parse_args(
            [
                "--output-dir",
                "unused",
                "--expected-world-size",
                str(world_size),
            ]
        )
        assert parsed.expected_world_size == world_size


def _step(model, batch, _context):
    if batch.get("fail"):
        raise RuntimeError("injected rank-local compiler failure")
    prediction = model(batch["x"])
    loss = (prediction - batch["target"]).square().mean()
    return ReconstructorStepResult(
        loss_terms={"root": loss},
        metrics={"mean_prediction": prediction.detach().mean()},
        batch_size=int(prediction.shape[0]),
        tensors_to_check=(prediction,),
    )


def _compose(result, _context, _weights):
    loss = result.loss_terms["root"].float()
    return SimpleNamespace(
        total=loss,
        raw_terms={"root": loss},
        effective_weights={"root": 1.0},
        weighted_terms={"root": loss},
        required_terms=("root",),
    )


def test_training_module_anchors_every_stage_active_parameter() -> None:
    class BranchModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.left = torch.nn.Linear(2, 1, bias=False)
            self.right = torch.nn.Linear(2, 1, bias=False)

        def forward(self, values, use_left):
            return self.left(values) if use_left else self.right(values)

    def step(model, batch, _context):
        prediction = model(batch["x"], batch["use_left"])
        loss = prediction.square().mean()
        return ReconstructorStepResult(
            loss_terms={"root": loss},
            metrics={},
            batch_size=int(prediction.shape[0]),
            tensors_to_check=(prediction,),
        )

    model = BranchModel()
    module = ReconstructorTrainingModule(model, step, _compose, None)
    output = module(
        {"x": torch.ones(2, 2), "use_left": True},
        None,
    )
    output["total_loss"].backward()

    assert model.left.weight.grad is not None
    assert model.right.weight.grad is not None
    assert torch.count_nonzero(model.right.weight.grad) == 0
    assert module.last_metadata["complete_parameter_graph_anchored"] is True


class _FailBeforeReducer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, fail):
        ctx.fail = bool(fail)
        return value

    @staticmethod
    def backward(ctx, gradient):
        if ctx.fail:
            raise RuntimeError("injected failure before DDP reducer hooks")
        return gradient, None


def _rank_divergent_graph_worker(rank: int, world_size: int, port: int) -> None:
    os.environ.update(
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK=str(rank),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        ABPH_DDP_TIMEOUT_SECONDS="30",
    )
    runtime = initialize_distributed_runtime(
        requested_world_size=world_size, device=torch.device("cpu")
    )
    try:
        class BranchModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.left = torch.nn.Linear(2, 1, bias=False)
                self.right = torch.nn.Linear(2, 1, bias=False)

            def forward(self, values, use_left):
                return self.left(values) if use_left else self.right(values)

        def step(model, batch, _context):
            prediction = model(batch["x"], batch["use_left"])
            loss = prediction.square().mean()
            return ReconstructorStepResult(
                loss_terms={"root": loss},
                metrics={},
                batch_size=int(prediction.shape[0]),
                tensors_to_check=(prediction,),
            )

        torch.manual_seed(402)
        model = BranchModel()
        module = ReconstructorTrainingModule(model, step, _compose, None)
        wrapper = build_stage_ddp_wrapper(
            module,
            runtime,
            device=torch.device("cpu"),
            # The complete graph anchor makes reducer order independent of
            # rank-local topology even without unused-parameter discovery.
            find_unused_parameters=False,
        )
        output = wrapper(
            {"x": torch.ones(2, 2), "use_left": rank == 0},
            None,
        )
        output["total_loss"].backward()
        assert model.left.weight.grad is not None
        assert model.right.weight.grad is not None
        gathered = _gather_fixed_evidence(
            {"rank": rank, "backward_completed": True},
            measured_world=world_size,
            device=torch.device("cpu"),
        )
        assert tuple(row["rank"] for row in gathered) == tuple(range(world_size))
    finally:
        destroy_distributed_runtime(runtime)


def _real_ddp_backward_failure_worker(
    rank: int, world_size: int, port: int, root: str
) -> None:
    os.environ.update(
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK=str(rank),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        ABPH_DDP_TIMEOUT_SECONDS="8",
    )
    runtime = initialize_distributed_runtime(
        requested_world_size=world_size, device=torch.device("cpu")
    )
    started = time.monotonic()
    error = None
    try:
        model = torch.nn.parallel.DistributedDataParallel(
            torch.nn.Linear(2, 1, bias=False)
        )
        prediction = model(torch.ones(2, 2))
        loss = _FailBeforeReducer.apply(prediction, rank == 1).sum()
        loss.backward()
    except BaseException as exc:
        error = exc
        abort_distributed_runtime(runtime)
    elapsed = time.monotonic() - started
    result = {
        "rank": rank,
        "caught": error is not None,
        "error": None if error is None else f"{type(error).__name__}: {error}",
        "elapsed_seconds": elapsed,
    }
    Path(root, f"backward_failure_rank_{rank}.json").write_text(
        json.dumps(result), encoding="utf-8"
    )
    if error is None:
        abort_distributed_runtime(runtime)
        raise AssertionError("real DDP backward failure did not reach this rank")


def _runtime_probe_forward_failure_worker(
    rank: int, world_size: int, port: int, root: str
) -> None:
    os.environ.update(
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK=str(rank),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        ABPH_DDP_TIMEOUT_SECONDS="15",
    )
    runtime = initialize_distributed_runtime(
        requested_world_size=world_size, device=torch.device("cpu")
    )
    try:
        model = torch.nn.parallel.DistributedDataParallel(
            torch.nn.Linear(2, 1, bias=False)
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)

        def forward_loss(module, batch):
            prediction = module(batch)
            if rank == 1:
                raise RuntimeError("injected probe forward failure")
            loss = prediction.square().mean()
            return {"total_loss": loss, "root_loss": loss}

        measurement = measure_full_optimizer_step(
            stage_family="root_hierarchy",
            variant_name="B1_semantic_query_root",
            resolved_variant_config_hash="config",
            runtime_provenance_hash="provenance",
            slurm_job_id="probe",
            slurm_job_account="reu-aisocial",
            slurm_job_partition="tigris",
            local_batch_size=256,
            requested_world_size=world_size,
            model=model,
            optimizer=optimizer,
            ema=SimpleNamespace(shadow={}),
            batch_factory=lambda size: torch.ones(size, 2),
            forward_loss=forward_loss,
            active_parameter_groups=("root",),
            device=torch.device("cpu"),
            gradient_clip_norm=1.0,
            find_unused_parameters=False,
            largest_path_exercised=True,
            prefetch_buffers=(),
            pinned_memory_staging=False,
        )
        Path(root, f"probe_failure_rank_{rank}.json").write_text(
            json.dumps(measurement.to_dict()), encoding="utf-8"
        )
    finally:
        destroy_distributed_runtime(runtime)


def _distributed_worker(rank: int, world_size: int, port: int) -> None:
    os.environ.update(
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK=str(rank),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        ABPH_DDP_TIMEOUT_SECONDS="60",
    )
    runtime = initialize_distributed_runtime(
        requested_world_size=world_size, device=torch.device("cpu")
    )
    try:
        torch.manual_seed(91)
        model = torch.nn.Linear(2, 1, bias=False)
        module = ReconstructorTrainingModule(model, _step, _compose, None)
        wrapper = build_stage_ddp_wrapper(
            module,
            runtime,
            device=torch.device("cpu"),
            find_unused_parameters=False,
        )
        output = require_standard_tensor_mapping(
            wrapper(
                {
                    "x": torch.tensor([[1.0 + rank, 2.0]]),
                    "target": torch.ones(1, 1),
                },
                None,
            )
        )
        assert tensor_mapping_is_finite(output)
        diagnostic_gradients = torch.autograd.grad(
            output["weighted_loss_terms"]["root"],
            tuple(model.parameters()),
            retain_graph=True,
            allow_unused=True,
        )
        assert any(value is not None for value in diagnostic_gradients)
        output["total_loss"].backward()
        gradient = model.weight.grad.detach().clone()
        gathered = [torch.zeros_like(gradient) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, gradient)
        assert all(torch.allclose(item, gathered[0]) for item in gathered)

        model.zero_grad(set_to_none=True)
        local_error = None
        try:
            wrapper(
                {
                    "x": torch.ones(1, 2),
                    "target": torch.ones(1, 1),
                    "fail": rank == 1,
                },
                None,
            )
        except BaseException as exc:
            local_error = exc
        assert not all_reduce_min_bool(
            runtime, local_error is None, device=torch.device("cpu")
        )
        errors = gather_error_summaries(
            runtime,
            phase="forward",
            error=local_error,
            structural=local_error is not None,
        )
        assert any_structural_error(errors)
        barrier(runtime)
        wrapper = None
        gc.collect()
        barrier(runtime)
        wrapper = build_stage_ddp_wrapper(
            module,
            runtime,
            device=torch.device("cpu"),
            find_unused_parameters=False,
        )
        verify_common_parameter_state(runtime, model)
        barrier(runtime)
        backward_error = None
        try:
            scalar = torch.ones((), requires_grad=True)
            if rank == 1:
                scalar.register_hook(
                    lambda _gradient: (_ for _ in ()).throw(
                        RuntimeError("injected rank-local backward failure")
                    )
                )
            scalar.backward()
        except BaseException as exc:
            backward_error = exc
        assert not all_reduce_min_bool(
            runtime, backward_error is None, device=torch.device("cpu")
        )
        backward_errors = gather_error_summaries(
            runtime,
            phase="backward",
            error=backward_error,
            structural=True,
        )
        assert any_structural_error(backward_errors)
        assert any(row.get("phase") == "backward" for row in backward_errors)
        barrier(runtime)
        assert not all_reduce_min_bool(
            runtime, rank == 0, device=torch.device("cpu")
        )
        assert wrapper is not None
    finally:
        destroy_distributed_runtime(runtime)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def _distributed_validation_worker(rank: int, world_size: int, port: int) -> None:
    os.environ.update(
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK=str(rank),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        ABPH_DDP_TIMEOUT_SECONDS="60",
    )
    runtime = initialize_distributed_runtime(
        requested_world_size=world_size, device=torch.device("cpu")
    )
    try:
        identities = tuple(
            JetIdentity(file="HToBB_010.root", entry=index, label=index % 10)
            for index in range(5)
        )
        start, stop = ((0, 2), (2, 5))[rank]
        count = stop - start
        accumulator = TypedValidationAccumulator()
        accumulator.add_mean(
            "loss.total", 1.0 if rank == 0 else 3.0, count,
            selection_eligible=True,
        )
        accumulator.add_mean("loss.raw.root", 2.0 + rank, count)
        accumulator.add_mean("loss.weighted.root", 2.0 + rank, count)
        accumulator.add_sum("diagnostic.total_groups", 4.0 + rank)
        accumulator.add_count("diagnostic.accepted_groups", 2 + rank)
        accumulator.add_ratio(
            "diagnostic.acceptance",
            2 + rank,
            4 + rank,
            denominator_semantics="groups",
        )
        accumulator.add_non_additive("calibration.quantile", 0.2 + rank, count)
        accumulator.finish_batch(count)
        result = finalize_typed_validation(
            accumulator,
            runtime=runtime,
            device=torch.device("cpu"),
            required_losses=("root",),
            effective_weights={"root": 1.0},
            validation_range=validation_range_row(
                split="model_val",
                rank=rank,
                start=start,
                stop=stop,
                jet_ids=identities,
            ),
            expected_jet_ids=identities,
        )
        assert result["selection_score"] == pytest.approx(2.2)
        assert result["selection_numerator"] == pytest.approx(11.0)
        assert result["selection_denominator"] == 5
        assert result["n_jets"] == 5
        assert result["checkpoint_selection_eligible"] is True
        assert result["validation_coverage"]["n_jets"] == 5
        assert result["reduction_schema"]["loss.total"]["kind"] == "mean"
        assert result["metrics"]["diagnostic.total_groups"] == 9.0
        assert result["metrics"]["diagnostic.accepted_groups"] == 5.0
        assert result["metrics"]["diagnostic.acceptance"] == pytest.approx(5.0 / 9.0)
        assert result["reduction_schema"]["diagnostic.acceptance"]["kind"] == "ratio"
        assert (
            result["non_additive_diagnostics"]["calibration.quantile"][
                "selection_eligible"
            ]
            is False
        )
    finally:
        destroy_distributed_runtime(runtime)


def _distributed_validation_failure_worker(
    rank: int, world_size: int, port: int
) -> None:
    os.environ.update(
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK=str(rank),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        ABPH_DDP_TIMEOUT_SECONDS="60",
    )
    runtime = initialize_distributed_runtime(
        requested_world_size=world_size, device=torch.device("cpu")
    )
    try:
        model = torch.nn.Linear(2, 1)
        curriculum = ReconstructorCurriculumConfig(
            root_updates=1,
            hierarchy_updates_per_depth=1,
            renderer_updates=1,
            distribution_updates=1,
            maximum_capacity=1,
            hierarchy_capacities=(),
            renderer_enabled=False,
            distribution_enabled=False,
        )
        config = ReconstructorTrainerConfig(
            output_dir="unused",
            device="cpu",
            amp=False,
            distributed_world_size=2,
            pin_memory=False,
            curriculum=curriculum,
        )
        with pytest.raises(RuntimeError, match="failed before reduction"):
            evaluate_reconstructor_rollout(
                model,
                (
                    {
                        "x": torch.ones(1, 2),
                        "target": torch.zeros(1, 1),
                        "fail": rank == 1,
                    },
                ),
                CurriculumController(curriculum).state(),
                _step,
                config,
                torch.device("cpu"),
                distributed_runtime=runtime,
            )
    finally:
        destroy_distributed_runtime(runtime)


def _distributed_skewed_validation_worker(
    rank: int, world_size: int, port: int
) -> None:
    os.environ.update(
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK=str(rank),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        ABPH_DDP_TIMEOUT_SECONDS="2",
    )
    runtime = initialize_distributed_runtime(
        requested_world_size=world_size, device=torch.device("cpu")
    )
    try:
        identities = tuple(
            JetIdentity(file="HToBB_010.root", entry=index, label=index)
            for index in range(3)
        )
        start, stop = ((0, 1), (1, 3))[rank]
        owner = SimpleNamespace(
            hlt_view=SimpleNamespace(jet_ids=identities),
            last_validation_range=None,
        )

        def batches():
            for index in range(start, stop):
                if rank == 1:
                    # Each wait is below the collective timeout, while their sum is
                    # above it. Per-round heartbeats must prevent cumulative skew.
                    time.sleep(1.25)
                yield {
                    "x": torch.tensor([[1.0 + index, 2.0]]),
                    "target": torch.ones(1, 1),
                }
            owner.last_validation_range = validation_range_row(
                split="model_val",
                rank=rank,
                start=start,
                stop=stop,
                jet_ids=identities,
            )

        curriculum = ReconstructorCurriculumConfig(
            root_updates=1,
            hierarchy_updates_per_depth=1,
            renderer_updates=1,
            distribution_updates=1,
            maximum_capacity=1,
            hierarchy_capacities=(),
            renderer_enabled=False,
            distribution_enabled=False,
        )
        config = ReconstructorTrainerConfig(
            output_dir="unused",
            device="cpu",
            amp=False,
            distributed_world_size=2,
            pin_memory=False,
            curriculum=curriculum,
        )
        result = evaluate_reconstructor_rollout(
            torch.nn.Linear(2, 1),
            batches(),
            CurriculumController(curriculum).state(),
            _step,
            config,
            torch.device("cpu"),
            distributed_runtime=runtime,
            validation_owner=owner,
        )
        assert result["n_jets"] == 3
        assert result["validation_coverage"]["n_jets"] == 3
    finally:
        destroy_distributed_runtime(runtime)


@dataclass(frozen=True)
class _TrainerPlan:
    global_update: int
    accumulation_index: int
    start_cursor: int
    next_cursor: int

    @property
    def plan_hash(self):
        return f"trainer-{self.global_update}-{self.accumulation_index}-{self.start_cursor}"

    @property
    def rank_plans(self):
        return (
            SimpleNamespace(slices=(SimpleNamespace(shard_id=self.start_cursor),)),
            SimpleNamespace(slices=(SimpleNamespace(shard_id=self.start_cursor),)),
        )


class _TrainerSource:
    split = "model_train"
    grouping = "exclusive_kt"

    def __init__(self, rank):
        self.rank = rank
        self.world_size = 2
        self.cursor = 0
        self.batch_size = 1

    @property
    def current_cursor(self):
        return self.cursor

    def set_batch_size(self, value):
        self.batch_size = int(value)

    def set_runtime_profiler(self, _profiler):
        pass

    def set_plan_log_dir(self, _path):
        pass

    def derive_next_plan(self, *, global_update, accumulation_index, cursor=None):
        start = self.cursor if cursor is None else cursor
        return _TrainerPlan(global_update, accumulation_index, start, start + 1)

    def agree_plan_hash(self, plan):
        gathered = [None, None]
        torch.distributed.all_gather_object(gathered, plan.plan_hash)
        assert gathered == [plan.plan_hash, plan.plan_hash]

    def prepare_planned_batch(self, plan, *, background_worker=False):
        assert background_worker
        return {
            "x": torch.tensor([[1.0 + self.rank, 1.0 + plan.start_cursor]]),
            "target": torch.ones(1, 1),
            "global_batch_plan_hash": plan.plan_hash,
        }

    def commit_planned_batch(self, plan):
        assert plan.start_cursor == self.cursor
        self.cursor = plan.next_cursor

    def state_dict(self):
        return {"cursor": self.cursor}

    def load_state_dict(self, payload):
        self.cursor = int(payload["cursor"])


class _TrainerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        for name in ABPH_RECONSTRUCTOR_MODULE_GROUPS:
            setattr(self, name, torch.nn.Linear(2, 2, bias=False))

    def module_groups(self):
        return {name: getattr(self, name) for name in ABPH_RECONSTRUCTOR_MODULE_GROUPS}


def _trainer_step(model, batch, _context):
    evidence = model.hlt_encoder(batch["x"])
    prediction = model.root(evidence)
    loss = (prediction - batch["target"].expand_as(prediction)).square().mean()
    return ReconstructorStepResult(
        loss_terms={"root": loss},
        metrics={},
        batch_size=int(prediction.shape[0]),
        tensors_to_check=(prediction,),
    )


class _TrainerValidation:
    def __init__(self, rank: int):
        self.rank = rank
        self.world_size = 2
        self.batch_size = 1
        self.jet_ids = tuple(
            JetIdentity(file="HToBB_010.root", entry=index, label=index)
            for index in range(2)
        )
        self.hlt_view = SimpleNamespace(jet_ids=self.jet_ids)
        self.last_validation_range = None

    def set_batch_size(self, value):
        self.batch_size = int(value)

    def set_runtime_profiler(self, _profiler):
        pass

    def iter_epoch(self):
        start, stop = self.rank, self.rank + 1
        yield {
            "x": torch.tensor([[1.0 + self.rank, 2.0]]),
            "target": torch.ones(1, 1),
        }
        self.last_validation_range = validation_range_row(
            split="model_val",
            rank=self.rank,
            start=start,
            stop=stop,
            jet_ids=self.jet_ids,
        )


def _distributed_trainer_worker(rank: int, world_size: int, port: int, root: str):
    os.environ.update(
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK=str(rank),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        ABPH_DDP_TIMEOUT_SECONDS="60",
    )
    torch.manual_seed(41)
    model = _TrainerModel()
    source = _TrainerSource(rank)
    config = ReconstructorTrainerConfig(
        output_dir=str(Path(root) / f"rank_{rank}"),
        device="cpu",
        amp=False,
        distributed_world_size=2,
        root_hierarchy_local_batch_size=1,
        renderer_distribution_local_batch_size=1,
        root_hierarchy_gradient_accumulation_steps=2,
        renderer_distribution_gradient_accumulation_steps=2,
        root_hierarchy_effective_batch_size=4,
        renderer_distribution_effective_batch_size=4,
        pin_memory=False,
        save_last_checkpoint=False,
        runtime_profile=RuntimeProfileConfig(
            enabled=True,
            warmup_updates_per_stage=0,
            sample_interval=1,
            profile_validation=False,
        ),
        curriculum=ReconstructorCurriculumConfig(
            root_updates=3,
            hierarchy_updates_per_depth=1,
            renderer_updates=1,
            distribution_updates=1,
            evaluation_interval=100,
            maximum_capacity=1,
            hierarchy_capacities=(),
            renderer_enabled=False,
            distribution_enabled=False,
        ),
    )
    report = train_reconstructor_curriculum(
        model,
        model.module_groups(),
        source,
        lambda: (),
        _trainer_step,
        config,
        maximum_optimizer_updates=1,
    )
    assert report["ok"] is False
    assert report["distributed_runtime"]["world_size"] == 2
    assert source.cursor == 2
    verify_common_parameter_state(
        DistributedRuntime(rank, 2, rank, "gloo", "cpu"), model
    )
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


def _distributed_checkpoint_worker(rank: int, world_size: int, port: int, root: str):
    os.environ.update(
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK=str(rank),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        ABPH_DDP_TIMEOUT_SECONDS="60",
    )
    # Deliberately start each rank from a different live/EMA state. DDP makes
    # rank zero's live model canonical, and the trainer must do the same for EMA.
    torch.manual_seed(51 + rank)
    model = _TrainerModel()
    source = _TrainerSource(rank)
    validation = _TrainerValidation(rank)
    output = Path(root) / "shared"
    config = ReconstructorTrainerConfig(
        output_dir=str(output),
        device="cpu",
        amp=False,
        distributed_world_size=2,
        root_hierarchy_local_batch_size=1,
        renderer_distribution_local_batch_size=1,
        root_hierarchy_gradient_accumulation_steps=1,
        renderer_distribution_gradient_accumulation_steps=1,
        root_hierarchy_effective_batch_size=2,
        renderer_distribution_effective_batch_size=2,
        pin_memory=False,
        save_last_checkpoint=True,
        runtime_profile=RuntimeProfileConfig(enabled=False),
        curriculum=ReconstructorCurriculumConfig(
            root_updates=1,
            hierarchy_updates_per_depth=1,
            renderer_updates=1,
            distribution_updates=1,
            evaluation_interval=1,
            maximum_capacity=1,
            hierarchy_capacities=(),
            renderer_enabled=False,
            distribution_enabled=False,
        ),
    )
    report = train_reconstructor_curriculum(
        model,
        model.module_groups(),
        source,
        validation.iter_epoch,
        _trainer_step,
        config,
        provenance={"manifest_hash": "distributed-checkpoint-test"},
    )
    assert report["ok"] is True
    assert report["best_model_val_checkpoint_sha256"]
    resumed_model = _TrainerModel()
    resumed_source = _TrainerSource(rank)
    resumed_validation = _TrainerValidation(rank)
    resumed = train_reconstructor_curriculum(
        resumed_model,
        resumed_model.module_groups(),
        resumed_source,
        resumed_validation.iter_epoch,
        _trainer_step,
        config,
        provenance={"manifest_hash": "distributed-checkpoint-test"},
        resume_from=output / "last.pt",
    )
    assert resumed["ok"] is True
    assert resumed_source.cursor == 1
    torch.distributed.barrier()
    if rank == 0:
        checkpoint = torch.load(output / "last.pt", weights_only=False)
        distributed = checkpoint["distributed_checkpoint_state"]
        assert distributed["world_size"] == 2
        assert [row["rank"] for row in distributed["rank_states"]] == [0, 1]
        assert [row["train_source_state_dict"]["cursor"] for row in distributed["rank_states"]] == [1, 1]
        assert checkpoint["runtime_contracts"]["distributed_runtime_contract"]
        assert checkpoint["runtime_contracts"]["runtime_profile_snapshot_hash"]
        assert (output / "run_report.json").exists()
        assert (output / "training_curves.json").exists()
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


def test_environment_supports_slurm_fallback(monkeypatch):
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SLURM_PROCID", "2")
    monkeypatch.setenv("SLURM_NTASKS", "4")
    monkeypatch.setenv("SLURM_LOCALID", "0")
    assert distributed_environment(requested_world_size=4) == (2, 4, 0)


def test_tensor_mapping_rejects_custom_or_non_tensor_leaves():
    with pytest.raises(TypeError, match="standard mapping"):
        require_standard_tensor_mapping(SimpleNamespace(total_loss=torch.tensor(1.0)))
    with pytest.raises(TypeError, match="non-tensor leaf"):
        require_standard_tensor_mapping(
            {
                "total_loss": torch.tensor(1.0),
                "raw_loss_terms": {"root": torch.tensor(1.0)},
                "weighted_loss_terms": {"root": torch.tensor(1.0)},
                "finite_check_tensors": (torch.tensor(1.0),),
                "batch_size_tensor": 1,
            }
        )


def test_single_rank_wrapper_exposes_loss_terms_and_gradients():
    model = torch.nn.Linear(2, 1, bias=False)
    module = ReconstructorTrainingModule(model, _step, _compose, None)
    runtime = DistributedRuntime(0, 1, 0, "none", "cpu")
    wrapper = build_stage_ddp_wrapper(module, runtime, device=torch.device("cpu"))
    output = require_standard_tensor_mapping(
        wrapper({"x": torch.ones(2, 2), "target": torch.zeros(2, 1)}, None)
    )
    assert set(output) == {
        "total_loss",
        "raw_loss_terms",
        "weighted_loss_terms",
        "finite_check_tensors",
        "batch_size_tensor",
    }
    output["total_loss"].backward()
    assert model.weight.grad is not None
    before = parameter_state_hash(model)
    with torch.no_grad():
        model.weight.add_(1.0)
    assert parameter_state_hash(model) != before


def test_cuda_ddp_preparation_converts_rank_local_batch_norm():
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 4),
        torch.nn.BatchNorm1d(4),
        torch.nn.ReLU(),
    )
    state_names = tuple(model.state_dict())
    converted = prepare_model_for_distributed_training(
        model,
        requested_world_size=4,
        device="cuda",
    )
    assert isinstance(converted[1], torch.nn.SyncBatchNorm)
    assert tuple(converted.state_dict()) == state_names
    require_distributed_normalization_contract(
        converted,
        DistributedRuntime(0, 4, 0, "nccl", "cuda"),
        device="cuda",
    )


def test_cuda_ddp_contract_rejects_unconverted_batch_norm():
    model = torch.nn.Sequential(torch.nn.BatchNorm1d(4))
    with pytest.raises(RuntimeError, match="rank-local BatchNorm"):
        require_distributed_normalization_contract(
            model,
            DistributedRuntime(0, 4, 0, "nccl", "cuda"),
        device="cuda",
    )


def test_auto_cuda_ddp_preparation_converts_rank_local_batch_norm(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    model = torch.nn.Sequential(torch.nn.BatchNorm1d(4))

    converted = prepare_model_for_distributed_training(
        model,
        requested_world_size=8,
        device="auto",
    )

    assert isinstance(converted[0], torch.nn.SyncBatchNorm)
    require_distributed_normalization_contract(
        converted,
        DistributedRuntime(0, 8, 0, "nccl", "cuda"),
        device="cuda",
    )


def test_auto_cpu_ddp_preparation_leaves_batch_norm_unchanged(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = torch.nn.Sequential(torch.nn.BatchNorm1d(4))

    converted = prepare_model_for_distributed_training(
        model,
        requested_world_size=8,
        device="auto",
    )

    assert isinstance(converted[0], torch.nn.BatchNorm1d)
    assert not isinstance(converted[0], torch.nn.SyncBatchNorm)


@pytest.mark.skipif(
    not torch.distributed.is_available(), reason="torch.distributed is unavailable"
)
def test_two_rank_ddp_consensus_and_collective_rebuild():
    torch.multiprocessing.spawn(
        _distributed_worker,
        args=(2, _free_port()),
        nprocs=2,
        join=True,
    )


@pytest.mark.skipif(
    not torch.distributed.is_available(), reason="torch.distributed is unavailable"
)
def test_two_rank_divergent_topology_uses_one_reducer_sequence():
    torch.multiprocessing.spawn(
        _rank_divergent_graph_worker,
        args=(2, _free_port()),
        nprocs=2,
        join=True,
    )


@pytest.mark.skipif(
    not torch.distributed.is_available(), reason="torch.distributed is unavailable"
)
def test_two_rank_validation_reduces_typed_losses_and_exact_identity_coverage():
    torch.multiprocessing.spawn(
        _distributed_validation_worker,
        args=(2, _free_port()),
        nprocs=2,
        join=True,
    )


@pytest.mark.skipif(
    not torch.distributed.is_available(), reason="torch.distributed is unavailable"
)
def test_two_rank_validation_failure_is_consensual_before_reduction():
    torch.multiprocessing.spawn(
        _distributed_validation_failure_worker,
        args=(2, _free_port()),
        nprocs=2,
        join=True,
    )


@pytest.mark.skipif(
    not torch.distributed.is_available(), reason="torch.distributed is unavailable"
)
def test_two_rank_validation_heartbeats_bound_skew_and_pad_uneven_tails():
    torch.multiprocessing.spawn(
        _distributed_skewed_validation_worker,
        args=(2, _free_port()),
        nprocs=2,
        join=True,
    )


@pytest.mark.skipif(
    not torch.distributed.is_available(), reason="torch.distributed is unavailable"
)
def test_two_rank_curriculum_update_uses_no_sync_and_deferred_commit(tmp_path):
    torch.multiprocessing.spawn(
        _distributed_trainer_worker,
        args=(2, _free_port(), str(tmp_path)),
        nprocs=2,
        join=True,
    )
    profile = json.loads(
        (tmp_path / "rank_0" / "runtime_profile.json").read_text(
            encoding="utf-8"
        )
    )
    stage = next(iter(profile["stages"].values()))
    assert stage["sampled_updates"] == 1
    assert stage["sampled_jets"] == 4


@pytest.mark.skipif(
    not torch.distributed.is_available(), reason="torch.distributed is unavailable"
)
def test_real_ddp_backward_failure_aborts_all_ranks_within_timeout(tmp_path):
    torch.multiprocessing.spawn(
        _real_ddp_backward_failure_worker,
        args=(2, _free_port(), str(tmp_path)),
        nprocs=2,
        join=True,
    )
    rows = [
        json.loads(
            (tmp_path / f"backward_failure_rank_{rank}.json").read_text(
                encoding="utf-8"
            )
        )
        for rank in range(2)
    ]
    assert all(row["caught"] for row in rows)
    assert max(row["elapsed_seconds"] for row in rows) < 15.0
    assert "injected failure before DDP reducer hooks" in rows[1]["error"]


@pytest.mark.skipif(
    not torch.distributed.is_available(), reason="torch.distributed is unavailable"
)
def test_runtime_probe_reaches_consensus_before_ddp_backward(tmp_path):
    torch.multiprocessing.spawn(
        _runtime_probe_forward_failure_worker,
        args=(2, _free_port(), str(tmp_path)),
        nprocs=2,
        join=True,
    )
    rows = [
        json.loads(
            (tmp_path / f"probe_failure_rank_{rank}.json").read_text(
                encoding="utf-8"
            )
        )
        for rank in range(2)
    ]
    assert all(row["successful"] is False for row in rows)
    assert all(row["backward_completed"] is False for row in rows)
    assert any("injected probe forward failure" in row["failure"] for row in rows)
    assert any(
        "peer rank failed before synchronized backward" in row["failure"]
        for row in rows
    )


@pytest.mark.skipif(
    not torch.distributed.is_available(), reason="torch.distributed is unavailable"
)
def test_two_rank_checkpoint_is_rank_zero_owned_and_contains_all_rank_state(tmp_path):
    torch.multiprocessing.spawn(
        _distributed_checkpoint_worker,
        args=(2, _free_port(), str(tmp_path)),
        nprocs=2,
        join=True,
    )
