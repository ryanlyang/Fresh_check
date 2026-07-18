#!/usr/bin/env python3
"""Run the fixed one-update ABPH DDP transport/resume/failure smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.adaptive_binary_pseudooffline import canonical_hash  # noqa: E402
from teacher_logit_reco.adaptive_binary_pseudooffline.distributed import (  # noqa: E402
    all_gather_objects,
    all_reduce_float64_pair,
    all_reduce_min_bool,
    barrier,
    destroy_distributed_runtime,
    initialize_distributed_runtime,
    verify_common_parameter_state,
)
from teacher_logit_reco.adaptive_binary_pseudooffline.runtime_acceptance import (  # noqa: E402
    ABPH_DDP_SMOKE_CONTRACT,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-world-size", type=int, choices=(1, 4), required=True)
    parser.add_argument("--device", default="cuda")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    import torch

    args = _parser().parse_args(argv)
    requested_world = int(args.expected_world_size)
    device = torch.device(args.device)
    runtime = initialize_distributed_runtime(
        requested_world_size=requested_world, device=device
    )
    if device.type == "cuda":
        device = torch.device("cuda", runtime.local_rank)
    torch.manual_seed(41027)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(41027)

    model = torch.nn.Sequential(
        torch.nn.Linear(6, 16),
        torch.nn.GELU(),
        torch.nn.Linear(16, 3),
    ).to(device)
    wrapped = (
        torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[runtime.local_rank] if device.type == "cuda" else None,
            output_device=runtime.local_rank if device.type == "cuda" else None,
            broadcast_buffers=False,
        )
        if runtime.distributed
        else model
    )
    optimizer = torch.optim.AdamW(wrapped.parameters(), lr=2.0e-4, weight_decay=0.01)
    generator = torch.Generator(device="cpu").manual_seed(73019)
    global_x = torch.randn(64, 6, generator=generator)
    global_y = torch.randn(64, 3, generator=generator)
    start = 64 * runtime.rank // runtime.world_size
    stop = 64 * (runtime.rank + 1) // runtime.world_size
    local_x = global_x[start:stop].to(device)
    local_y = global_y[start:stop].to(device)
    identity_hash = canonical_hash({"ordered_indices": list(range(64)), "seed": 73019})

    optimizer.zero_grad(set_to_none=True)
    prediction = wrapped(local_x)
    loss = (prediction - local_y).square().mean()
    training_loss_sum, training_loss_weight = all_reduce_float64_pair(
        runtime,
        float(loss.detach().cpu()) * float(stop - start),
        float(stop - start),
        device=device,
    )
    global_training_mean_loss = training_loss_sum / training_loss_weight
    loss.backward()
    gradient_finite = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in wrapped.parameters()
    )
    gradient_finite = all_reduce_min_bool(runtime, gradient_finite, device=device)
    if not gradient_finite:
        raise FloatingPointError("DDP smoke gradients are nonfinite")
    gradient_l2_norm = float(
        torch.sqrt(
            sum(
                parameter.grad.detach().to(torch.float64).square().sum()
                for parameter in wrapped.parameters()
                if parameter.grad is not None
            )
        ).cpu()
    )
    torch.nn.utils.clip_grad_norm_(wrapped.parameters(), 1.0)
    optimizer.step()
    parameter_hash = verify_common_parameter_state(runtime, model)
    ema_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }

    with torch.no_grad():
        validation_prediction = wrapped(local_x)
        validation_sum = float(
            (validation_prediction - local_y).square().sum().detach().cpu()
        )
        validation_count = int(local_y.numel())
    validation_sum, validation_count_float = all_reduce_float64_pair(
        runtime,
        validation_sum,
        float(validation_count),
        device=device,
    )
    validation_mean = validation_sum / validation_count_float

    rank_rows = all_gather_objects(
        runtime,
        {
            "rank": runtime.rank,
            "start": start,
            "stop": stop,
            "n_examples": stop - start,
        },
    )
    coverage_ok = [row["rank"] for row in rank_rows] == list(range(runtime.world_size))
    coverage_ok = coverage_ok and rank_rows[0]["start"] == 0 and rank_rows[-1]["stop"] == 64
    coverage_ok = coverage_ok and all(
        rank_rows[index]["stop"] == rank_rows[index + 1]["start"]
        for index in range(len(rank_rows) - 1)
    )

    output_dir = Path(args.output_dir).resolve()
    state_path = output_dir / "post_step_state.pt"
    checkpoint_path = output_dir / "resume_checkpoint.pt"
    if runtime.is_primary:
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": {
                    name: value.detach().cpu() for name, value in model.state_dict().items()
                },
                "ema_state_dict": {
                    name: value.detach().cpu() for name, value in ema_state.items()
                },
            },
            state_path,
        )
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "parameter_hash": parameter_hash,
                "rank_rows": rank_rows,
            },
            checkpoint_path,
        )
    barrier(runtime)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:  # pragma: no cover - older research PyTorch
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    resume_ok = verify_common_parameter_state(runtime, model) == checkpoint["parameter_hash"]

    local_forward_success = runtime.rank != 0
    consensus_success = all_reduce_min_bool(
        runtime, local_forward_success, device=device
    )
    forward_failure_consensus = consensus_success is False
    barrier(runtime)
    del wrapped
    barrier(runtime)
    rebuilt = (
        torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[runtime.local_rank] if device.type == "cuda" else None,
            output_device=runtime.local_rank if device.type == "cuda" else None,
            broadcast_buffers=False,
        )
        if runtime.distributed
        else model
    )
    rebuild_ok = verify_common_parameter_state(runtime, model) == parameter_hash
    with torch.no_grad():
        rebuild_ok = rebuild_ok and bool(torch.isfinite(rebuilt(local_x)).all())

    checks = {
        "rank_partition_full_union": bool(coverage_ok),
        "finite_synchronized_gradients": bool(gradient_finite),
        "post_step_rank_parameter_hash_agreement": bool(parameter_hash),
        "rank_zero_checkpoint_only": checkpoint_path.is_file(),
        "checkpoint_resume": bool(resume_ok),
        "forward_failure_consensus": bool(forward_failure_consensus),
        "wrapper_rebuild": bool(rebuild_ok),
        "validation_numerator_denominator_reduction": validation_count_float
        == float(global_y.numel()),
    }
    barrier(runtime)
    if runtime.is_primary:
        report = {
            "contract": ABPH_DDP_SMOKE_CONTRACT,
            "ok": all(checks.values()),
            "world_size": runtime.world_size,
            "runtime": runtime.to_dict(),
            "global_batch_identity_hash": identity_hash,
            "rank_ranges": list(rank_rows),
            "validation_numerator": validation_sum,
            "validation_denominator": validation_count_float,
            "validation_mean_loss": validation_mean,
            "global_training_mean_loss": global_training_mean_loss,
            "preclip_gradient_l2_norm": gradient_l2_norm,
            "parameter_state_hash": parameter_hash,
            "state_path": str(state_path),
            "state_sha256": _sha256(state_path),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "checks": checks,
            "final_test_loaded": False,
        }
        report["smoke_content_hash"] = canonical_hash(report)
        _atomic_json(output_dir / "smoke_report.json", report)
        print(json.dumps(report, indent=2, sort_keys=True))
    barrier(runtime)
    destroy_distributed_runtime(runtime)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
