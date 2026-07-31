#!/usr/bin/env python3
"""Train and attest one immutable graph-specific Stage-I capacity control."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_hosd_baseline import _dataset, _mapping  # noqa: E402
from teacher_logit_reco.hlt_offline_structure_distillation import (  # noqa: E402
    HOSDTrainingProtocol,
    build_capacity_control_result,
    build_capacity_model,
    build_label_free_hlt_loader,
    component_seed,
    export_deployable_graph,
    load_and_validate_campaign,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    CAPACITY_CONTROL_EXECUTION_PLAN_CONTRACT,
    CAPACITY_CONTROL_RESULT_CONTRACT,
    load_hashed_json,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.hlt_offline_structure_distillation.stage_c_training import (  # noqa: E402
    train_stage_c_baseline,
)
from teacher_logit_reco.relation_expert_token_bridge.evaluation import (  # noqa: E402
    evaluate_classification,
)
from teacher_logit_reco.relation_expert_token_bridge.hlt_experts import (  # noqa: E402
    make_native_hlt_expert_loader,
)


def _inference(model, loader, *, device):
    import torch

    resolved = torch.device(device)
    model.to(resolved).eval()
    identities, logits, labels = [], [], []
    with torch.no_grad():
        for raw in loader:
            batch = {
                key: value.to(resolved) if hasattr(value, "to") else value
                for key, value in raw.items()
            }
            vectors = batch.get("lorentz_vectors", batch.get("vectors"))
            points = batch.get("points")
            if points is None:
                points = batch["features"][:, 15:17]
            values = model(points, batch["features"], vectors, batch["mask"])
            if not bool(torch.isfinite(values).all()):
                raise FloatingPointError("capacity-control logits are nonfinite")
            identities.extend(str(value) for value in raw["event_identities"])
            logits.append(values.float().cpu().numpy())
            labels.append(raw["labels"].cpu().numpy())
    return (
        tuple(identities),
        np.concatenate(logits),
        np.concatenate(labels),
    )


def _atomic_predictions(path: Path, identities, logits) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                identities=np.asarray(identities, dtype="U"),
                logits=np.asarray(logits, dtype=np.float32),
            )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finalize(root: Path, plan, source):
    hashes = {}
    for row in plan["rows"]:
        path = (
            root
            / "confirmation_500k"
            / "capacity_results"
            / f"{row['row_id']}.json"
        )
        if not path.is_file():
            return None
        result = load_hashed_json(
            path, expected_contract=CAPACITY_CONTROL_RESULT_CONTRACT
        )
        if (
            result.get("source") != source
            or result.get("capacity_execution_plan_sha256")
            != plan["content_hash"]
            or result.get("row_id") != row["row_id"]
        ):
            raise ValueError("capacity-control completion coverage differs")
        hashes[row["row_id"]] = result["content_hash"]
    artifact = with_content_hash(
        {
            "contract": "hosd_capacity_control_completion_v1",
            "schema_version": 1,
            "source": dict(source),
            "capacity_execution_plan_sha256": plan["content_hash"],
            "result_hashes": {key: hashes[key] for key in sorted(hashes)},
            "row_count": len(hashes),
            "coverage_exact": True,
            "performance_based_termination": False,
        }
    )
    write_immutable_json(
        root / "confirmation_500k" / "capacity_control_completion.json",
        artifact,
    )
    return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--row-id", required=True)
    parser.add_argument("--train-cache", action="append", default=[])
    parser.add_argument("--val-stop-cache", action="append", default=[])
    parser.add_argument("--design-confirm-cache", action="append", default=[])
    parser.add_argument("--train-labels", required=True, type=Path)
    parser.add_argument("--val-stop-labels", required=True, type=Path)
    parser.add_argument("--design-confirm-labels", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    root = args.campaign_root.resolve()
    campaign = load_and_validate_campaign(root, repo_root=REPO_ROOT)
    plan = load_hashed_json(
        root / "confirmation_500k" / "capacity_execution_plan.json",
        expected_contract=CAPACITY_CONTROL_EXECUTION_PLAN_CONTRACT,
    )
    matches = [row for row in plan["rows"] if row["row_id"] == args.row_id]
    if len(matches) != 1:
        raise ValueError("capacity-control row is absent or duplicated")
    row = matches[0]
    if args.dry_run:
        print(json.dumps({"row_id": args.row_id, "executed": False}, indent=2))
        return 0
    definition = row["control_definition"]
    model = build_capacity_model(
        str(definition["kind"]), definition["configuration"]
    )
    seed = int(row["seed"])
    task_seed = component_seed(seed, "capacity_control", row["control_graph_id"])
    train = _dataset(
        _mapping(args.train_cache, name="--train-cache"),
        args.train_labels,
        role="model_train",
        teacher_logits=None,
        native_relation_targets=None,
    )
    val = _dataset(
        _mapping(
            args.val_stop_cache,
            name="--val-stop-cache",
            required_replicas={0},
        ),
        args.val_stop_labels,
        role="val_stop",
        teacher_logits=None,
        native_relation_targets=None,
    )
    confirm = _dataset(
        _mapping(
            args.design_confirm_cache,
            name="--design-confirm-cache",
            required_replicas={0},
        ),
        args.design_confirm_labels,
        role="design_confirm",
        teacher_logits=None,
        native_relation_targets=None,
    )
    miniature = campaign["campaign_profile"] == "miniature_test"
    protocol = HOSDTrainingProtocol(
        maximum_epochs=2 if miniature else 40,
        campaign_profile="miniature_test" if miniature else "production",
    )
    import torch

    device = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    run = root / "confirmation_500k" / "capacity_runs" / row["row_id"]
    train_stage_c_baseline(
        model=model,
        train_loader=make_native_hlt_expert_loader(
            train, seed=task_seed, training=True, batch_size=64
        ),
        val_stop_loader=make_native_hlt_expert_loader(
            val, seed=task_seed, training=False, batch_size=64
        ),
        output_dir=run,
        baseline_id=row["control_graph_id"],
        seed=seed,
        component_seed=task_seed,
        baseline_registry_sha256=plan["content_hash"],
        campaign_spec_sha256=campaign["content_hash"],
        lineage_hashes={
            "capacity_execution_plan": plan["content_hash"],
            "capacity_compilation": row["capacity_compilation_sha256"],
        },
        protocol=protocol,
        device=device,
        source=campaign["source"],
    )
    ids, logits, labels = _inference(
        model,
        make_native_hlt_expert_loader(
            confirm, seed=task_seed, training=False, batch_size=64
        ),
        device=device,
    )
    metrics = evaluate_classification(logits, labels, split="design_confirm")
    prediction_sha = _atomic_predictions(run / "design_confirm_predictions.npz", ids, logits)
    checkpoint = run / "best_model_val.pt"
    import importlib

    module = importlib.import_module("weaver.nn.model.ParticleTransformer")
    export = export_deployable_graph(
        descriptor={
            "graph_id": row["control_graph_id"],
            "graph_kind": "CAPACITY",
            "capacity_kind": str(definition["kind"]),
            "configuration": dict(definition["configuration"]),
            "seed": seed,
        },
        research_model=model,
        representative_batch=next(
            iter(
                build_label_free_hlt_loader(
                    cache_paths=_mapping(
                        args.val_stop_cache,
                        name="--val-stop-cache",
                        required_replicas={0},
                    ),
                    identities=val.identities[:8],
                    logical_role="val_stop",
                    realization_policy="R_FIXED",
                    batch_size=8,
                )
            )
        ),
        output_path=(
            root
            / "confirmation_500k"
            / "capacity_exports"
            / f"{row['row_id']}.pt"
        ),
        checkpoint_sha256=hashlib.sha256(
            checkpoint.read_bytes()
        ).hexdigest(),
        lineage_hashes={
            "capacity_execution_plan": plan["content_hash"],
            "capacity_compilation": row["capacity_compilation_sha256"],
            "design_confirm_predictions": prediction_sha,
        },
        source=campaign["source"],
        weaver_module=module,
        analytical_inference_flops_batch1_n128=int(
            row["control_definition"]["analytical_flops_batch1_n128"]
        ),
    )
    result = build_capacity_control_result(
        execution_plan=plan,
        row_id=row["row_id"],
        classification_metrics=metrics,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        prediction_sha256=prediction_sha,
        deployable_export_sha256=export["content_hash"],
        deployable_export_file=str(
            (
                root
                / "confirmation_500k"
                / "capacity_exports"
                / f"{row['row_id']}.pt"
            ).resolve()
        ),
        source=campaign["source"],
    )
    output = (
        root
        / "confirmation_500k"
        / "capacity_results"
        / f"{row['row_id']}.json"
    )
    write_immutable_json(output, result)
    coverage = _finalize(root, plan, campaign["source"])
    print(
        json.dumps(
            {
                "row_id": row["row_id"],
                "result_sha256": result["content_hash"],
                "coverage_complete": coverage is not None,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
