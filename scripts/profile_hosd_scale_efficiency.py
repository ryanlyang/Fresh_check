#!/usr/bin/env python3
"""Profile one scaled graph and finalize the exhaustive efficiency wave."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    build_label_free_hlt_loader,
    load_and_validate_campaign,
    load_deployable_graph,
    measure_deployable_efficiency,
    native_relation_target_ids,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    EFFICIENCY_PROFILE_CONTRACT,
    SCALE_EXECUTION_PLAN_CONTRACT,
    SCALE_ROW_RESULT_CONTRACT,
    SCALE_TRAINING_COMPLETION_CONTRACT,
    SLURM_SUBMISSION_LEDGER_CONTRACT,
    load_hashed_json,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.hlt_cache import (  # noqa: E402
    load_hlt_v3_cache,
)


def _latest_training_job(root: Path, plan, source) -> tuple[str, str]:
    candidates = []
    for path in sorted(
        (root / "job_ledgers").glob("slurm_submission_attempt_*.json")
    ):
        ledger = load_hashed_json(
            path, expected_contract=SLURM_SUBMISSION_LEDGER_CONTRACT
        )
        if (
            ledger.get("source") != source
            or ledger.get("execution_plan_sha256") != plan["content_hash"]
        ):
            raise ValueError("efficiency submission ledger lineage differs")
        job_id = ledger.get("jobs", {}).get("scale_graph_train")
        if job_id is not None:
            candidates.append(
                (
                    int(ledger["attempt"]),
                    str(job_id),
                    ledger["content_hash"],
                )
            )
    if not candidates:
        raise ValueError("efficiency lacks the scale training Slurm job")
    _, job_id, ledger_hash = max(candidates)
    return job_id, ledger_hash


def _training_hours(job_id: str, coordinate: int) -> tuple[float, dict]:
    task_id = f"{job_id}_{int(coordinate)}"
    completed = subprocess.run(
        [
            "sacct",
            "-X",
            "-n",
            "-P",
            "-j",
            task_id,
            "-o",
            "JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    rows = [
        raw.split("|")
        for raw in completed.stdout.splitlines()
        if raw.strip()
    ]
    matches = [row for row in rows if row[0] == task_id]
    if len(matches) != 1:
        raise ValueError("scale training Slurm task evidence differs")
    row = matches[0]
    if (
        not row[1].rstrip("+").startswith("COMPLETED")
        or not row[2].startswith("0:")
        or int(row[3]) <= 0
    ):
        raise ValueError("scale training Slurm task did not complete")
    gpu_count = None
    for field in row[4].split(","):
        if field.startswith("gres/gpu="):
            gpu_count = int(field.split("=", 1)[1])
    if gpu_count is None or gpu_count <= 0:
        raise ValueError("scale training Slurm task lacks allocated GPU evidence")
    return int(row[3]) * gpu_count / 3600.0, {
        "slurm_task_id": task_id,
        "state": row[1],
        "exit_code": row[2],
        "elapsed_seconds": int(row[3]),
        "allocated_tres": row[4],
        "allocated_gpu_count": gpu_count,
    }


def _target_ids(descriptor) -> set[str]:
    kind = descriptor["graph_kind"]
    if kind in {"AUXILIARY", "FEEDBACK"}:
        return {str(descriptor["row"]["target_id"])}
    if kind == "COMBINATION":
        values = {
            str(row["target_id"])
            for row in descriptor["graph"]["members"]
        }
        if descriptor["graph"].get("native_relation_auxiliary") is not None:
            values.update(native_relation_target_ids())
        return values
    if (
        kind == "BASELINE"
        and descriptor.get("baseline_id") == "H_NATIVE_REL_AUX"
    ):
        return set(native_relation_target_ids())
    return set()


def _target_bytes_per_jet(
    root: Path, descriptor, event_count: int
) -> tuple[float, dict[str, int]]:
    sizes = {}
    for target_id in sorted(_target_ids(descriptor)):
        target_root = root / "scale_up" / "targets" / target_id
        if not target_root.is_dir():
            raise FileNotFoundError(
                f"scale target storage is absent: {target_id}"
            )
        sizes[target_id] = sum(
            path.stat().st_size
            for path in target_root.rglob("*")
            if path.is_file() and not path.is_symlink()
        )
    if _target_ids(descriptor) & set(native_relation_target_ids()):
        native_root = root / "scale_up" / "targets" / "native_relations"
        if native_root.is_dir():
            sizes["MATERIALIZED_NATIVE_RELATION_BUNDLE"] = sum(
                path.stat().st_size
                for path in native_root.rglob("*")
                if path.is_file() and not path.is_symlink()
            )
    return (
        sum(sizes.values()) / int(event_count) if sizes else 0.0,
        sizes,
    )


def _try_finalize(root: Path, plan, source):
    hashes = {}
    for row in plan["graph_rows"]:
        row_id = f"{row['graph_id']}__seed_{row['seed']}"
        path = root / "scale_up" / "efficiency" / f"{row_id}.json"
        if not path.is_file():
            return None
        artifact = load_hashed_json(
            path, expected_contract=EFFICIENCY_PROFILE_CONTRACT
        )
        if (
            artifact.get("source") != source
            or artifact.get("graph_id") != row["graph_id"]
            or int(artifact.get("seed", -1)) != int(row["seed"])
        ):
            raise ValueError("scale efficiency wave coordinate differs")
        hashes[row_id] = artifact["content_hash"]
    completion = with_content_hash(
        {
            "contract": "hosd_scale_efficiency_completion_v1",
            "schema_version": 1,
            "source": dict(source),
            "scale_execution_plan_sha256": plan["content_hash"],
            "profile_hashes": dict(sorted(hashes.items())),
            "profile_count": len(hashes),
            "coverage_exact": True,
            "latency_selection_tiebreaker": False,
            "performance_based_termination": False,
        }
    )
    write_immutable_json(
        root / "scale_up" / "efficiency" / "completion.json",
        completion,
    )
    return completion


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--graph-id", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--coordinate", required=True, type=int)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument(
        "--production-batch-size", required=True, type=int
    )
    parser.add_argument("--clock-power-mode", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    root = args.campaign_root.resolve()
    campaign = load_and_validate_campaign(root, repo_root=REPO_ROOT)
    plan = load_hashed_json(
        root / "scale_up" / "execution_plan.json",
        expected_contract=SCALE_EXECUTION_PLAN_CONTRACT,
    )
    expected_rows = [
        row
        for row in plan["graph_rows"]
        if row["graph_id"] == args.graph_id
        and int(row["seed"]) == int(args.seed)
    ]
    if len(expected_rows) != 1:
        raise ValueError("scale efficiency coordinate is absent or duplicated")
    if (
        int(args.coordinate) < 0
        or int(args.coordinate) >= len(plan["graph_rows"])
        or plan["graph_rows"][int(args.coordinate)] != expected_rows[0]
    ):
        raise ValueError("scale efficiency coordinate ordinal differs")
    export_path = (
        root
        / "scale_up"
        / "exports"
        / f"{args.graph_id}__seed_{args.seed}.pt"
    )
    export = load_hashed_json(export_path.with_suffix(".pt.json"))
    scale_result = load_hashed_json(
        root
        / "scale_up"
        / "results"
        / f"{args.graph_id}__seed_{args.seed}.json",
        expected_contract=SCALE_ROW_RESULT_CONTRACT,
    )
    module = importlib.import_module("weaver.nn.model.ParticleTransformer")
    model, payload = load_deployable_graph(
        export_path, weaver_module=module, source=campaign["source"]
    )
    if (
        export.get("content_hash")
        != scale_result.get("deployable_export_sha256")
        or payload["descriptor"].get("graph_id") != args.graph_id
        or scale_result.get("source") != campaign["source"]
    ):
        raise ValueError("scale efficiency export lineage differs")
    arrays, metadata = load_hlt_v3_cache(args.cache)
    identities = tuple(str(value) for value in arrays["identities"].tolist())
    production_batch = int(args.production_batch_size)
    if production_batch <= 0 or len(identities) < max(128, production_batch):
        raise ValueError("scale efficiency cache is too small")
    representative = {}
    for batch_size in {1, 128, production_batch}:
        loader = build_label_free_hlt_loader(
            cache_paths={0: args.cache},
            identities=identities[:batch_size],
            logical_role=str(metadata["logical_role"]),
            realization_policy="R_FIXED",
            batch_size=batch_size,
        )
        representative[batch_size] = next(iter(loader))
    train_count = int(campaign["split_roles"]["scale_train"])
    val_count = int(campaign["split_roles"]["val_stop"])
    confirm_count = int(campaign["split_roles"]["design_confirm"])
    completion = load_hashed_json(
        root
        / "scale_up"
        / "runs"
        / f"{args.graph_id}__seed_{args.seed}"
        / "training_completion.json",
        expected_contract=SCALE_TRAINING_COMPLETION_CONTRACT,
    )
    epochs = int(completion["epochs_completed"])
    forward_flops = int(
        export["analytical_inference_flops_batch1_n128"]
    )
    role_flops = {
        key: int(value)
        for key, value in scale_result[
            "analytical_forward_flops_by_role"
        ].items()
    }
    analytical_training_flops = (
        3 * role_flops["model_train"] * train_count * epochs
        + role_flops["val_stop"] * val_count * epochs
        + role_flops["design_confirm"] * confirm_count
    )
    target_bytes, target_sizes = _target_bytes_per_jet(
        root, payload["descriptor"], train_count
    )
    training_job_id, submission_hash = _latest_training_job(
        root,
        load_hashed_json(
            root / "job_ledgers" / "production_execution_plan.json"
        ),
        campaign["source"],
    )
    training_hours, training_evidence = _training_hours(
        training_job_id, args.coordinate
    )
    training_evidence.update(
        {
            "submission_ledger_sha256": submission_hash,
            "training_completion_sha256": completion["content_hash"],
            "target_storage_bytes_by_target": target_sizes,
        }
    )
    artifact = measure_deployable_efficiency(
        model=model,
        representative_batches=representative,
        production_batch_size=production_batch,
        graph_id=args.graph_id,
        seed=args.seed,
        export_path=export_path,
        export_sha256=hashlib.sha256(export_path.read_bytes()).hexdigest(),
        analytical_training_flops=analytical_training_flops,
        analytical_inference_flops_by_batch={
            batch_size: forward_flops * batch_size
            for batch_size in {1, 128, production_batch}
        },
        complete_parameter_count=int(
            export["complete_trainable_parameters"]
        ),
        target_cache_bytes_per_jet=target_bytes,
        training_gpu_hours=training_hours,
        clock_power_mode=args.clock_power_mode,
        training_evidence=training_evidence,
        analytical_training_flop_convention=(
            "3*training_graph_forward_flops*train_events*epochs+"
            "training_graph_validation_forward_flops*val_stop_events*epochs+"
            "deployed_forward_flops*design_confirm_events"
        ),
        source=campaign["source"],
        device="cuda",
    )
    publication = write_immutable_json(args.output, artifact)
    wave = _try_finalize(root, plan, campaign["source"])
    print(
        json.dumps(
            {
                "content_hash": artifact["content_hash"],
                "publication": publication["status"],
                "wave_complete": wave is not None,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
