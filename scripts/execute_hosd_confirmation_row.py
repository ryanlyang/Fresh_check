#!/usr/bin/env python3
"""Retrain one locked HOSD graph and evaluate it only on design-confirm."""

from __future__ import annotations

import argparse
import hashlib
import importlib
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
    build_auxiliary_model,
    build_baseline_model,
    build_combination_loader_manifest,
    build_combination_model,
    build_confirmation_result,
    build_feedback_model,
    build_label_free_hlt_loader,
    build_stage_d_loader_manifest,
    component_seed,
    load_and_validate_campaign,
    load_combination_loaders,
    export_deployable_graph,
)
from teacher_logit_reco.hlt_offline_structure_distillation.contracts import (  # noqa: E402
    CONFIRMATION_PLAN_CONTRACT,
    CONFIRMATION_RESULT_CONTRACT,
    CONFIRMATION_TRAINING_CHECKPOINT_CONTRACT,
    CONFIRMATION_TRAINING_COMPLETION_CONTRACT,
    CONFIRMATION_TRAINING_PREDICTION_CONTRACT,
    CONFIRMATION_WAVE_COMPLETION_CONTRACT,
    load_hashed_json,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.hlt_offline_structure_distillation.stage_c_training import (  # noqa: E402
    train_stage_c_baseline,
)
from teacher_logit_reco.hlt_offline_structure_distillation.stage_d_data_factory import (  # noqa: E402
    load_stage_d_loaders_from_manifest,
)
from teacher_logit_reco.hlt_offline_structure_distillation.stage_d_training import (  # noqa: E402
    train_stage_d_auxiliary,
)
from teacher_logit_reco.hlt_offline_structure_distillation.combination_runtime import (  # noqa: E402
    train_combination,
)
from teacher_logit_reco.relation_expert_token_bridge.evaluation import (  # noqa: E402
    evaluate_classification,
)
from teacher_logit_reco.relation_expert_token_bridge.hlt_experts import (  # noqa: E402
    make_native_hlt_expert_loader,
)


def _device(name: str):
    import torch

    return (
        "cuda"
        if name == "auto" and torch.cuda.is_available()
        else "cpu"
        if name == "auto"
        else name
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
                raise FloatingPointError("confirmation logits are nonfinite")
            identities.extend(str(value) for value in raw["event_identities"])
            logits.append(values.float().cpu().numpy())
            labels.append(raw["labels"].cpu().numpy())
    return tuple(identities), np.concatenate(logits), np.concatenate(labels)


def _predictions(path: Path, identities, logits) -> str:
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


def _confirmation_manifest(
    *,
    base_path: Path,
    row,
    design_confirm_labels: Path,
    campaign,
    output: Path,
):
    base = load_hashed_json(base_path)
    if base.get("source") != campaign["source"]:
        raise ValueError("confirmation base loader source differs")
    definitions = {
        "model_train": dict(base["roles"]["model_train"]),
        "val_stop": dict(base["roles"]["val_stop"]),
        "design_confirm": {
            **dict(base["roles"]["design_select"]),
            "labels": str(design_confirm_labels.resolve()),
        },
    }
    manifest = build_stage_d_loader_manifest(
        row=row,
        role_definitions=definitions,
        campaign_spec_sha256=campaign["content_hash"],
        source=campaign["source"],
        evaluation_role="design_confirm",
    )
    write_immutable_json(output, manifest)
    return output


def _row_with_seed(definition, seed: int):
    kind, graph_id = definition["graph_kind"], definition["graph_id"]
    if kind == "AUXILIARY":
        row = dict(definition["row"])
        row.update(
            {
                "row_id": graph_id,
                "pipeline_seed": seed,
                "encoder_component_seed": component_seed(seed, "encoder", "H_BASE"),
                "head_component_seed": component_seed(seed, "head", graph_id),
                "resolved": True,
            }
        )
        return row
    if kind == "FEEDBACK":
        row = dict(definition["row"])
        row.update(
            {
                "row_id": graph_id,
                "pipeline_seed": seed,
                "encoder_component_seed": component_seed(seed, "encoder", "H_BASE"),
                "feedback_component_seed": component_seed(
                    seed, "feedback", graph_id
                ),
                "resolved": True,
                "row_kind": row.get("row_kind", "SCIENTIFIC"),
                "selection_eligible": False,
            }
        )
        return row
    raise ValueError("confirmation row kind has no auxiliary row")


def _selected_loader_path(loader_root: Path, graph_row) -> Path:
    return (
        loader_root
        / f"{graph_row['selected_row_id']}.json"
    )


def _load_auxiliary_or_feedback(
    *,
    root,
    definition,
    seed,
    stage_d_loader_root,
    design_confirm_labels,
    campaign,
    target_registry,
    output_root,
):
    row = _row_with_seed(definition, seed)
    if definition["graph_kind"] == "AUXILIARY":
        base_path = stage_d_loader_root / f"{definition['graph_id']}.json"
    else:
        single = load_hashed_json(
            root / "auxiliary" / "locked_single_family_choices.json"
        )
        selected = single["selected_row_by_target"][row["target_id"]]
        base_path = stage_d_loader_root / f"{selected}.json"
    manifest_path = _confirmation_manifest(
        base_path=base_path,
        row=row,
        design_confirm_labels=design_confirm_labels,
        campaign=campaign,
        output=output_root / "loader.json",
    )
    loaded = load_stage_d_loaders_from_manifest(
        manifest_path=manifest_path,
        campaign_root=root,
        row=row,
        campaign=campaign,
        target_registry=target_registry,
    )
    return row, loaded


def _load_combination(
    *,
    root,
    definition,
    seed,
    stage_d_loader_root,
    design_confirm_labels,
    campaign,
    target_registry,
    output_root,
    native_relation_design_confirm,
):
    graph = dict(definition["graph"])
    graph["graph_id"] = definition["graph_id"]
    member_manifests = {}
    for member in graph["members"]:
        row = {
            **dict(member),
            "row_id": member["selected_row_id"],
            "row_kind": "SCIENTIFIC",
            "resolved": True,
        }
        path = output_root / "members" / f"{member['target_id']}.json"
        _confirmation_manifest(
            base_path=_selected_loader_path(stage_d_loader_root, member),
            row=row,
            design_confirm_labels=design_confirm_labels,
            campaign=campaign,
            output=path,
        )
        member_manifests[member["target_id"]] = path
    native = None
    if graph.get("native_relation_auxiliary") is not None:
        discovery = load_hashed_json(
            root
            / "job_ledgers"
            / "loaders"
            / "combinations"
            / f"{definition['graph_id']}.json"
        )
        native = {
            **{
                f"model_train:{replica}": discovery[
                    "native_relation_target_files"
                ]["model_train"][str(replica)]["path"]
                for replica in range(4)
            },
            "val_stop:0": discovery["native_relation_target_files"][
                "val_stop"
            ]["0"]["path"],
            "design_confirm:0": (
                str(native_relation_design_confirm.resolve())
                if native_relation_design_confirm is not None
                else discovery["native_relation_target_files"][
                    "design_select"
                ]["0"]["path"]
            ),
        }
    manifest = build_combination_loader_manifest(
        graph=graph,
        member_loader_manifests=member_manifests,
        native_relation_target_files=native,
        campaign_spec_sha256=campaign["content_hash"],
        source=campaign["source"],
        evaluation_role="design_confirm",
    )
    manifest_path = output_root / "loader.json"
    write_immutable_json(manifest_path, manifest)
    return graph, load_combination_loaders(
        manifest=manifest,
        graph=graph,
        campaign_root=root,
        campaign=campaign,
        target_registry=target_registry,
    )


def _finalize(root, plan, source):
    hashes = {}
    for row in plan["training_rows"]:
        path = (
            root
            / "confirmation_500k"
            / "results"
            / f"{row['row_id']}.json"
        )
        if not path.is_file():
            return None
        result = load_hashed_json(
            path, expected_contract=CONFIRMATION_RESULT_CONTRACT
        )
        if (
            result.get("source") != source
            or result.get("confirmation_plan_sha256") != plan["content_hash"]
            or result.get("row_id") != row["row_id"]
        ):
            raise ValueError("confirmation training coverage differs")
        hashes[row["row_id"]] = result["content_hash"]
    artifact = with_content_hash(
        {
            "contract": CONFIRMATION_WAVE_COMPLETION_CONTRACT,
            "schema_version": 1,
            "source": dict(source),
            "confirmation_plan_sha256": plan["content_hash"],
            "result_hashes": {key: hashes[key] for key in sorted(hashes)},
            "row_count": len(hashes),
            "coverage_exact": True,
            "performance_based_termination": False,
        }
    )
    write_immutable_json(
        root / "confirmation_500k" / "training_completion.json", artifact
    )
    return artifact


def _capacity(root: Path, graph_id: str) -> tuple[float, int]:
    manifest = load_hashed_json(
        root
        / "confirmation_500k"
        / "discovery_exports"
        / f"{graph_id}.pt.json"
    )
    if manifest.get("descriptor", {}).get("graph_id") != graph_id:
        raise ValueError("confirmation discovery-export graph differs")
    return (
        float(manifest["analytical_inference_flops_batch1_n128"]),
        int(manifest["deployed_trainable_parameters"]),
    )


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
    parser.add_argument("--train-teacher-logits-o-base", type=Path)
    parser.add_argument("--train-teacher-logits-o-fullrel", type=Path)
    parser.add_argument("--stage-d-loader-root", required=True, type=Path)
    parser.add_argument("--native-relation-design-confirm", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    root = args.campaign_root.resolve()
    campaign = load_and_validate_campaign(root, repo_root=REPO_ROOT)
    plan = load_hashed_json(
        root / "confirmation_500k" / "execution_plan.json",
        expected_contract=CONFIRMATION_PLAN_CONTRACT,
    )
    matches = [row for row in plan["training_rows"] if row["row_id"] == args.row_id]
    if len(matches) != 1:
        raise ValueError("confirmation training row is absent or duplicated")
    plan_row = matches[0]
    definition = dict(plan_row["graph_definition"])
    seed = int(plan_row["seed"])
    if args.dry_run:
        print(json.dumps({"row_id": args.row_id, "executed": False}, indent=2))
        return 0
    target_registry = load_hashed_json(
        root / "registry" / "structure_target_registry.json"
    )
    miniature = campaign["campaign_profile"] == "miniature_test"
    protocol = HOSDTrainingProtocol(
        maximum_epochs=2 if miniature else 40,
        campaign_profile="miniature_test" if miniature else "production",
    )
    run = root / "confirmation_500k" / "runs" / plan_row["row_id"]
    run.mkdir(parents=True, exist_ok=True)
    module = importlib.import_module("weaver.nn.model.ParticleTransformer")
    kind = definition["graph_kind"]
    device = _device(args.device)
    deployed_flops, deployed_parameters = _capacity(
        root, definition["graph_id"]
    )
    if kind == "BASELINE":
        baseline_id = definition["baseline_id"]
        teacher_logits = (
            args.train_teacher_logits_o_base
            if baseline_id == "H_KD_LOGIT_O_BASE"
            else args.train_teacher_logits_o_fullrel
            if baseline_id == "H_KD_LOGIT_O_FULLREL"
            else None
        )
        if baseline_id.startswith("H_KD_") and teacher_logits is None:
            raise ValueError("confirmation KD baseline lacks teacher logits")
        model = build_baseline_model(baseline_id, weaver_module=module)
        task_seed = component_seed(seed, "baseline", baseline_id)
        train = _dataset(
            _mapping(args.train_cache, name="--train-cache"),
            args.train_labels,
            role="model_train",
            teacher_logits=teacher_logits,
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
        train_loader = make_native_hlt_expert_loader(
            train, seed=task_seed, training=True, batch_size=64
        )
        val_loader = make_native_hlt_expert_loader(
            val, seed=task_seed, training=False, batch_size=64
        )
        evaluation_loader = make_native_hlt_expert_loader(
            confirm, seed=task_seed, training=False, batch_size=64
        )
        completion = train_stage_c_baseline(
            model=model,
            train_loader=train_loader,
            val_stop_loader=val_loader,
            output_dir=run,
            baseline_id=baseline_id,
            seed=seed,
            component_seed=task_seed,
            baseline_registry_sha256=plan["content_hash"],
            campaign_spec_sha256=campaign["content_hash"],
            lineage_hashes={"confirmation_plan": plan["content_hash"]},
            protocol=protocol,
            teacher_id=(
                "O_BASE"
                if baseline_id == "H_KD_LOGIT_O_BASE"
                else "O_FULLREL"
                if baseline_id == "H_KD_LOGIT_O_FULLREL"
                else None
            ),
            teacher_logit_key=(
                "offline_target_logits"
                if baseline_id.startswith("H_KD_")
                else None
            ),
            device=device,
            source=campaign["source"],
        )
    elif kind in {"AUXILIARY", "FEEDBACK"}:
        row, loaded = _load_auxiliary_or_feedback(
            root=root,
            definition=definition,
            seed=seed,
            stage_d_loader_root=args.stage_d_loader_root.resolve(),
            design_confirm_labels=args.design_confirm_labels,
            campaign=campaign,
            target_registry=target_registry,
            output_root=run,
        )
        model = (
            build_auxiliary_model(row, weaver_module=module)[0]
            if kind == "AUXILIARY"
            else build_feedback_model(row, weaver_module=module)
        )
        completion = train_stage_d_auxiliary(
            model=model,
            train_loader=loaded["train_loader"],
            val_stop_loader=loaded["val_stop_loader"],
            design_select_loader=loaded["design_confirm_loader"],
            output_dir=run,
            row=row,
            component_group_ids=loaded["component_group_ids"],
            stage_d_plan_sha256=plan["content_hash"],
            campaign_spec_sha256=campaign["content_hash"],
            lineage_hashes=loaded["lineage_hashes"],
            protocol=protocol,
            source=campaign["source"],
            deployed_analytical_flops=deployed_flops,
            deployed_parameter_count=deployed_parameters,
            device=device,
            checkpoint_contract=CONFIRMATION_TRAINING_CHECKPOINT_CONTRACT,
            completion_contract=CONFIRMATION_TRAINING_COMPLETION_CONTRACT,
            prediction_contract=CONFIRMATION_TRAINING_PREDICTION_CONTRACT,
            plan_hash_field="confirmation_plan_sha256",
            stage_label="Stage-I-confirmation",
            completion_filename="training_completion.json",
            curves_contract="hosd_confirmation_training_curves_v1",
            evaluation_split="design_confirm",
        )
        evaluation_loader = loaded["design_confirm_loader"]
    elif kind == "COMBINATION":
        graph, loaded = _load_combination(
            root=root,
            definition=definition,
            seed=seed,
            stage_d_loader_root=args.stage_d_loader_root.resolve(),
            design_confirm_labels=args.design_confirm_labels,
            campaign=campaign,
            target_registry=target_registry,
            output_root=run,
            native_relation_design_confirm=args.native_relation_design_confirm,
        )
        model = build_combination_model(graph, seed=seed, weaver_module=module)
        completion = train_combination(
            model=model,
            train_loader=loaded["train_loader"],
            val_stop_loader=loaded["val_stop_loader"],
            design_select_loader=loaded["design_confirm_loader"],
            graph=graph,
            output_dir=run,
            stage_f_plan_sha256=plan["content_hash"],
            campaign_spec_sha256=campaign["content_hash"],
            lineage_hashes=loaded["lineage_hashes"],
            protocol=protocol,
            source=campaign["source"],
            deployed_analytical_flops=deployed_flops,
            deployed_parameter_count=deployed_parameters,
            device=device,
            evaluation_split="design_confirm",
            checkpoint_contract=CONFIRMATION_TRAINING_CHECKPOINT_CONTRACT,
            completion_contract=CONFIRMATION_TRAINING_COMPLETION_CONTRACT,
            result_contract=CONFIRMATION_TRAINING_PREDICTION_CONTRACT,
            plan_hash_field="confirmation_plan_sha256",
            completion_filename="training_completion.json",
        )
        evaluation_loader = loaded["design_confirm_loader"]
    else:
        raise ValueError("confirmation graph kind differs")
    identities, logits, labels = _inference(
        model, evaluation_loader, device=device
    )
    metrics = evaluate_classification(logits, labels, split="design_confirm")
    prediction_sha = _predictions(
        run / "design_confirm_predictions.npz", identities, logits
    )
    checkpoint = run / "best_model_val.pt"
    validation_caches = _mapping(
        args.val_stop_cache,
        name="--val-stop-cache",
        required_replicas={0},
    )
    with np.load(args.val_stop_labels, allow_pickle=False) as payload:
        validation_identities = tuple(
            str(value) for value in payload["identities"].tolist()
        )
    export_loader = build_label_free_hlt_loader(
        cache_paths=validation_caches,
        identities=validation_identities[:8],
        logical_role="val_stop",
        realization_policy="R_FIXED",
        batch_size=8,
    )
    export = export_deployable_graph(
        descriptor={**definition, "seed": seed},
        research_model=model,
        representative_batch=next(iter(export_loader)),
        output_path=(
            root
            / "confirmation_500k"
            / "exports"
            / f"{plan_row['row_id']}.pt"
        ),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        lineage_hashes={
            "confirmation_plan": plan["content_hash"],
            "training_completion": completion["content_hash"],
            "design_confirm_predictions": prediction_sha,
        },
        source=campaign["source"],
        weaver_module=module,
        analytical_inference_flops_batch1_n128=int(deployed_flops),
    )
    result = build_confirmation_result(
        plan=plan,
        row_id=plan_row["row_id"],
        classification_metrics=metrics,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        prediction_sha256=prediction_sha,
        training_completion_sha256=completion["content_hash"],
        deployable_export_sha256=export["content_hash"],
        deployable_export_file=str(
            (
                root
                / "confirmation_500k"
                / "exports"
                / f"{plan_row['row_id']}.pt"
            ).resolve()
        ),
        source=campaign["source"],
    )
    output = root / "confirmation_500k" / "results" / f"{plan_row['row_id']}.json"
    write_immutable_json(output, result)
    coverage = _finalize(root, plan, campaign["source"])
    print(
        json.dumps(
            {
                "row_id": plan_row["row_id"],
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
