from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json
import os
import socket

import pytest
import torch

from jetclass_fresh.jetclass_data import JetIdentity
from teacher_logit_reco.adaptive_binary_pseudooffline.distributed import (
    DistributedRuntime,
    destroy_distributed_runtime,
    initialize_distributed_runtime,
    verify_common_parameter_state,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.tagger_distributed import (
    TaggerTrainingModule,
    build_tagger_ddp_wrapper,
    build_tagger_ddp_acceptance,
    compile_tagger_global_batch_plan,
    require_tagger_ddp_acceptance,
    require_tagger_tensor_mapping,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.orchestration import (
    AdaptiveBinarySubmissionConfig,
    build_submission_graph,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.storage_quota import (
    ABPH_STREAMING_STORAGE_PROFILE,
    StorageArtifactClass,
    StorageProjectionRow,
    build_storage_projection,
    write_storage_projection,
)


def _runtime() -> DistributedRuntime:
    return DistributedRuntime(
        rank=0,
        world_size=1,
        local_rank=0,
        backend="none",
        device_type="cpu",
    )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def _tagger_step(active_model, _reconstructor, batch, _split, _validation):
    logits = active_model(batch["features"])
    output = SimpleNamespace(
        logits=logits,
        auxiliary_logits={},
        diagnostics={"root_provenance": {"root": "shared"}},
    )
    return output, batch["labels"], None, batch["indices"]


def _tagger_objective(output, labels, _config, **_kwargs):
    loss = torch.nn.functional.cross_entropy(output.logits, labels)
    return SimpleNamespace(
        total=loss,
        raw_terms={"label_ce": loss},
        weighted_terms={"label_ce": loss},
        diagnostics={},
    )


def _two_rank_tagger_worker(
    rank: int, world_size: int, port: int, root: str
) -> None:
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
        torch.manual_seed(73)
        model = torch.nn.Linear(3, 2)
        module = TaggerTrainingModule(
            model, None, _tagger_step, _tagger_objective, object()
        )
        wrapper = build_tagger_ddp_wrapper(
            module, runtime, device=torch.device("cpu")
        )
        identities = tuple(
            JetIdentity(file="HToBB_010.root", entry=index, label=index % 2)
            for index in range(4)
        )
        indices = torch.tensor([2 * rank, 2 * rank + 1])
        plan = compile_tagger_global_batch_plan(
            runtime,
            split="model_train",
            epoch=0,
            global_update=0,
            indices=indices.tolist(),
            jet_ids=identities,
            immutable_rank_range=(2 * rank, 2 * rank + 2),
        )
        result = require_tagger_tensor_mapping(
            wrapper(
                {
                    "features": torch.tensor(
                        [[1.0 + rank, 0.5, -0.5], [0.25, 1.0 + rank, 0.5]]
                    ),
                    "labels": torch.tensor([0, 1]),
                    "indices": indices,
                },
                None,
                "model_train",
                False,
            )
        )
        result["total_loss"].backward()
        torch.optim.SGD(module.parameters(), lr=0.1).step()
        state_hash = verify_common_parameter_state(runtime, module)
        Path(root, f"rank_{rank}.json").write_text(
            json.dumps(
                {
                    "plan_hash": plan["plan_hash"],
                    "global_effective_batch": plan["global_effective_batch"],
                    "state_hash": state_hash,
                }
            ),
            encoding="utf-8",
        )
    finally:
        destroy_distributed_runtime(runtime)


def test_tagger_training_module_returns_ddp_visible_tensor_mapping() -> None:
    model = torch.nn.Linear(3, 2)

    def step(active_model, _reconstructor, batch, _split, _validation):
        logits = active_model(batch["features"])
        output = SimpleNamespace(
            logits=logits,
            auxiliary_logits={},
            diagnostics={"root_provenance": {"root": "shared"}},
        )
        return output, batch["labels"], None, batch["indices"]

    def objective(output, labels, _config, **_kwargs):
        loss = torch.nn.functional.cross_entropy(output.logits, labels)
        return SimpleNamespace(
            total=loss,
            raw_terms={"label_ce": loss},
            weighted_terms={"label_ce": loss},
            diagnostics={},
        )

    module = TaggerTrainingModule(model, None, step, objective, object())
    result = require_tagger_tensor_mapping(
        module(
            {
                "features": torch.randn(4, 3),
                "labels": torch.tensor([0, 1, 0, 1]),
                "indices": torch.arange(4),
            },
            None,
            "model_train",
            False,
        )
    )
    result["total_loss"].backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert result["batch_size_tensor"].item() == 4


def test_global_batch_plan_binds_ordered_identities() -> None:
    identities = tuple(
        JetIdentity(file="HToBB_010.root", entry=index, label=0)
        for index in range(8)
    )
    plan = compile_tagger_global_batch_plan(
        _runtime(),
        split="model_train",
        epoch=2,
        global_update=7,
        indices=(1, 3, 5, 7),
        jet_ids=identities,
        immutable_rank_range=(0, 8),
        upstream_plan_hash="upstream",
    )
    assert plan["global_effective_batch"] == 4
    assert plan["rank_plans"][0]["upstream_plan_hash"] == "upstream"
    assert len(plan["plan_hash"]) == 64


def _write_report(
    path: Path,
    *,
    variant: str,
    world_size: int,
    seconds: float,
    loss: float = 0.5,
    accuracy: float = 0.7,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "ok": True,
                "variant_name": variant,
                "metrics": {
                    "model_val": {"loss": loss, "accuracy": accuracy}
                },
                "provenance": {
                    "model_val": {"jet_identity_hash": "same-validation"}
                },
                "distributed_runtime": {
                    "world_size": world_size,
                    "training_wall_seconds": seconds,
                    "validation_coverage_hash": "coverage",
                },
            }
        ),
        encoding="utf-8",
    )


def test_tagger_ddp_gate_requires_parity_speed_and_immutable_evidence(
    tmp_path: Path,
) -> None:
    names = ("E7_dual_hierarchy_dualcross", "F0_ce_reco_primary")
    single = {}
    ddp4 = {}
    for name in names:
        single[name] = tmp_path / "single" / name / "run_report.json"
        ddp4[name] = tmp_path / "ddp4" / name / "run_report.json"
        _write_report(single[name], variant=name, world_size=1, seconds=100.0)
        _write_report(ddp4[name], variant=name, world_size=4, seconds=50.0)
    gate = build_tagger_ddp_acceptance(
        single_reports=single,
        ddp4_reports=ddp4,
    )
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    assert require_tagger_ddp_acceptance(gate_path)["production_mode"] == "ddp4"

    _write_report(
        ddp4[names[0]],
        variant=names[0],
        world_size=4,
        seconds=49.0,
    )
    with pytest.raises(PermissionError, match="missing, stale, or failed"):
        require_tagger_ddp_acceptance(gate_path)


def test_streaming_graph_promotes_only_tagger_jobs_after_gate(tmp_path: Path) -> None:
    names = ("E7_dual_hierarchy_dualcross", "F0_ce_reco_primary")
    single = {}
    ddp4 = {}
    for name in names:
        single[name] = tmp_path / "single" / name / "run_report.json"
        ddp4[name] = tmp_path / "ddp4" / name / "run_report.json"
        _write_report(single[name], variant=name, world_size=1, seconds=100.0)
        _write_report(ddp4[name], variant=name, world_size=4, seconds=50.0)
    gate_path = tmp_path / "tagger_gate.json"
    gate_path.write_text(
        json.dumps(
            build_tagger_ddp_acceptance(
                single_reports=single,
                ddp4_reports=ddp4,
            )
        ),
        encoding="utf-8",
    )
    root = tmp_path / "campaign"
    projection_path = tmp_path / "projection.json"
    projection = build_storage_projection(
        campaign_root=root,
        campaign_mode="pilot",
        profile=ABPH_STREAMING_STORAGE_PROFILE,
        rows=(
            StorageProjectionRow(
                artifact_family="test",
                artifact_class=StorageArtifactClass.PERSISTENT_ESSENTIAL,
                expected_bytes=1_000_000,
                active_from_wave=0,
                active_through_wave=6,
                retained=True,
                atomic_write_overhead_bytes=1_000,
                measurement_source="test",
            ),
        ),
        measurement_contract="test",
        sample_provenance_hash="test",
    )
    write_storage_projection(projection_path, projection)
    config = AdaptiveBinarySubmissionConfig(
        campaign_root=root,
        data_dir=tmp_path / "data",
        reconstructor_parallelism="single",
        allow_debug_single_reconstructor=True,
        storage_profile=ABPH_STREAMING_STORAGE_PROFILE,
        storage_projection_path=projection_path,
        tagger_ddp_acceptance_path=gate_path,
    )
    graph = build_submission_graph(config)
    taggers = [job for job in graph if job.stage in {"tagger", "tagger_seed"}]
    assert taggers
    assert all(job.launcher == "srun" and job.nodes == 4 for job in taggers)
    baselines = [job for job in graph if job.stage == "baseline"]
    assert baselines
    assert all(job.launcher == "direct" and job.nodes == 1 for job in baselines)


@pytest.mark.skipif(
    not torch.distributed.is_available(), reason="torch.distributed is unavailable"
)
def test_two_rank_tagger_ddp_has_one_global_plan_and_synchronized_update(
    tmp_path: Path,
) -> None:
    torch.multiprocessing.spawn(
        _two_rank_tagger_worker,
        args=(2, _free_port(), str(tmp_path)),
        nprocs=2,
        join=True,
    )
    rows = [
        json.loads((tmp_path / f"rank_{rank}.json").read_text(encoding="utf-8"))
        for rank in range(2)
    ]
    assert rows[0]["plan_hash"] == rows[1]["plan_hash"]
    assert rows[0]["state_hash"] == rows[1]["state_hash"]
    assert rows[0]["global_effective_batch"] == 4


def test_tigris_gate_submitter_is_dry_run_safe_and_four_rank() -> None:
    root = Path(__file__).resolve().parents[1]
    submitter = (
        root / "sbatch" / "submit_adaptive_binary_tagger_ddp_acceptance_tigris.sh"
    ).read_text(encoding="utf-8")
    worker = (root / "sbatch" / "run_adaptive_binary_variant.sh").read_text(
        encoding="utf-8"
    )
    assert 'if [[ "${DRY_RUN}" == "1" ]]' in submitter
    assert "--nodes=4 --ntasks=4 --ntasks-per-node=1" in submitter
    assert "ABPH_TAGGER_DISTRIBUTED_WORLD_SIZE=4" in submitter
    assert "ABPH_TAGGER_PARALLELISM" in worker
    assert "tagger ddp4 requires four nodes and four tasks" in worker
