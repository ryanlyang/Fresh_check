#!/usr/bin/env python3
"""Train, export, and attest one locked Stage-M graph/seed at scale."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.confirmation import (  # noqa: E402
    SCALE_SHORTLIST_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    load_hashed_json,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.deployment import (  # noqa: E402
    COMPLETE_GRAPH_CAPACITY_CONTRACT,
    DEPLOYABLE_EXPORT_CONTRACT,
)
from teacher_logit_reco.relation_expert_token_bridge.evaluation import (  # noqa: E402
    evaluate_classification,
)
from teacher_logit_reco.relation_expert_token_bridge.final_consumer_training import (  # noqa: E402
    evaluate_final_consumer,
    load_final_consumer_dataset,
    load_final_consumer_template,
    make_final_consumer_loader,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.scale_execution import (  # noqa: E402
    SCALE_COMPONENT_INDEX_CONTRACT,
    SCALE_JOINT_COMPLETION_CONTRACT,
    validate_scale_component_index,
)
from teacher_logit_reco.relation_expert_token_bridge.scale_up import (  # noqa: E402
    SCALE_COMPONENT_KEYS,
    SCALE_REFIT_BUNDLE_CONTRACT,
    build_scale_graph_run,
    validate_scale_graph_run,
    validate_scale_refit_bundle,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    authorize_dataset_access,
    load_and_validate_campaign_source,
)


REFINER_CONTRACT = "retb_scale_selected_token_refiner_v1"
COMPONENT_CONTRACT = "retb_scale_component_lineage_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(arguments: Sequence[str], *, expected: Sequence[Path]) -> None:
    outputs = [Path(path) for path in expected]
    if outputs and all(path.is_file() and not path.is_symlink() for path in outputs):
        for path in outputs:
            if path.suffix == ".json":
                load_hashed_json(path)
        return
    completed = subprocess.run(
        [sys.executable, *map(str, arguments)],
        cwd=REPO_ROOT,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"scale graph phase failed ({completed.returncode}): "
            f"{' '.join(map(str, arguments))}"
        )
    if any(not path.is_file() or path.is_symlink() for path in outputs):
        raise RuntimeError("scale graph phase omitted an expected output")


def _refiner_base_run(
    root: Path, *, role: str, seed: int
) -> tuple[str, str]:
    lock = load_hashed_json(
        root
        / "selection"
        / "final_consumers"
        / role
        / "token_refiner_lock.json"
    )
    registry = load_hashed_json(
        root / "registry" / "retb_final_consumer_registry.json"
    )
    rows = [
        row
        for row in registry["rows"]
        if row["carried_shape_role"] == role
        and int(row["pipeline_seed"]) == seed
        and row["consumer_kind"] == "TR_REFINE"
        and row["model_variant"] == lock["selected_variant"]
        and row["token_input"] == "TOKEN_PREDICTED"
    ]
    if len(rows) != 1:
        raise ValueError("locked scale token-refiner base row differs")
    return rows[0]["run_id"], lock["selected_variant"]


def _component(
    *,
    name: str,
    upstream_sha256: str,
    graph_id: str,
    seed: int,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    return bind_source(
        with_content_hash(
            {
                "contract": COMPONENT_CONTRACT,
                "schema_version": 1,
                "component": name,
                "upstream_artifact_sha256": upstream_sha256,
                "graph_id": graph_id,
                "pipeline_seed": seed,
                "training_population": "scale_train",
                "stack_val_consumed": False,
                "final_test_consumed": False,
            }
        ),
        source_snapshot=source,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--locked-scale-shortlist", required=True, type=Path)
    parser.add_argument("--component-index", required=True, type=Path)
    parser.add_argument("--scale-refit-bundle", required=True, type=Path)
    parser.add_argument("--graph-id", required=True)
    parser.add_argument("--pipeline-seed", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    root = args.campaign_root.resolve()
    output = args.output_dir.resolve()
    campaign = load_and_validate_campaign_source(root, repo_root=REPO_ROOT)
    shortlist = load_hashed_json(
        args.locked_scale_shortlist,
        expected_contract=SCALE_SHORTLIST_CONTRACT,
    )
    components = load_hashed_json(
        args.component_index,
        expected_contract=SCALE_COMPONENT_INDEX_CONTRACT,
    )
    validate_scale_component_index(components)
    refit = load_hashed_json(
        args.scale_refit_bundle,
        expected_contract=SCALE_REFIT_BUNDLE_CONTRACT,
    )
    validate_scale_refit_bundle(refit)
    if (
        shortlist.get("source") != campaign.get("source")
        or components.get("source") != campaign.get("source")
        or refit.get("source") != campaign.get("source")
        or args.graph_id
        not in shortlist.get(
            "SCALE_TRAINING_GRAPHS", shortlist["SCALE_SHORTLIST"]
        )
        or components["graph_id"] != args.graph_id
        or refit["graph_id"] != args.graph_id
        or int(components["pipeline_seed"]) != args.pipeline_seed
        or int(refit["pipeline_seed"]) != args.pipeline_seed
        or refit["locked_scale_shortlist_sha256"]
        != shortlist["content_hash"]
    ):
        raise ValueError("scale graph pipeline lineage differs")
    for resource in ("scale_train", "val_stop", "val_design"):
        authorize_dataset_access(
            worker_role=(
                "design_worker"
                if resource == "val_design"
                else "scale_training_worker"
            ),
            requested_resource=resource,
        )

    definition = shortlist["locked_graph_definitions"][args.graph_id]
    role = definition["configuration"]["source_carried_shape_role"]
    base_run_id = definition["configuration"]["run_ids_by_seed"][
        str(args.pipeline_seed)
    ]
    role_lock = (
        root / "selection" / "predictor_bundle" / "carried" / f"{role}.json"
    )
    j4_selection = root / "selection" / "joint" / role / "j4_blocks.json"
    j4 = load_hashed_json(j4_selection)
    selected_j4 = (
        root
        / "runs"
        / "joint"
        / role
        / (
            f"RETB_J4_BRIDGE_FINETUNE_S{args.pipeline_seed}_"
            f"N{int(j4['selected_final_particle_blocks'])}"
        )
    )

    joint_root = output / "joint"
    joint_completion_path = output / "joint_completion.json"
    _run(
        [
            "scripts/execute_retb_joint_training_row.py",
            "--campaign-root",
            str(root),
            "--variant",
            "J5_END_TO_END",
            "--pipeline-seed",
            str(args.pipeline_seed),
            "--predictor-bundle-lock",
            str(role_lock),
            "--j4-selection",
            str(j4_selection),
            "--selected-j4-output",
            str(selected_j4),
            "--scale-component-index",
            str(args.component_index),
            "--scale-joint-completion-output",
            str(joint_completion_path),
            "--output-dir",
            str(joint_root),
            "--device",
            args.device,
        ],
        expected=[
            joint_root / "registration.json",
            joint_root / "best_model_val.pt",
            joint_root / "training_curves.json",
            joint_completion_path,
        ],
    )
    joint_completion = load_hashed_json(
        joint_completion_path,
        expected_contract=SCALE_JOINT_COMPLETION_CONTRACT,
    )

    dataset_root = output / "datasets"
    seed_dataset = dataset_root / role / f"seed_{args.pipeline_seed}"
    expected_datasets = [
        seed_dataset / split / "final_consumer_dataset.json"
        for split in ("scale_train", "val_stop", "val_design")
    ]
    _run(
        [
            "scripts/prepare_retb_final_consumer_seed.py",
            "--campaign-root",
            str(root),
            "--pipeline-seed",
            str(args.pipeline_seed),
            "--carried-shape-role",
            role,
            "--scale-joint-root",
            str(joint_root),
            "--scale-component-index",
            str(args.component_index),
            "--scale-joint-completion",
            str(joint_completion_path),
            "--output-dir",
            str(seed_dataset),
            "--device",
            args.device,
        ],
        expected=expected_datasets,
    )

    selected_refiner_path = None
    if definition["configuration"]["token_input"] == "TOKEN_REFINED_SELECTED":
        refiner_base, refiner_variant = _refiner_base_run(
            root, role=role, seed=args.pipeline_seed
        )
        refiner_root = output / "token_refiner"
        _run(
            [
                "scripts/execute_retb_final_consumer_row.py",
                "--campaign-root",
                str(root),
                "--run-id",
                refiner_base,
                "--dataset-root",
                str(dataset_root),
                "--output-dir",
                str(refiner_root),
                "--graph-id",
                args.graph_id,
                "--scale-component-index",
                str(args.component_index),
                "--scale-joint-completion",
                str(joint_completion_path),
                "--scale-joint-root",
                str(joint_root),
                "--scale-base-run-id",
                refiner_base,
                "--scale-run-id",
                (
                    f"RETB_SCALE_{args.graph_id}_REFINER_"
                    f"S{args.pipeline_seed}"
                ),
                "--device",
                args.device,
            ],
            expected=[
                refiner_root / "registration.json",
                refiner_root / "best_model_val.pt",
                refiner_root / "training_curves.json",
            ],
        )
        registration = load_hashed_json(
            refiner_root / "registration.json"
        )
        selected_refiner = bind_source(
            with_content_hash(
                {
                    "contract": REFINER_CONTRACT,
                    "schema_version": 1,
                    "graph_id": args.graph_id,
                    "pipeline_seed": args.pipeline_seed,
                    "variant": refiner_variant,
                    "base_run_id": refiner_base,
                    "checkpoint_path": str(
                        (refiner_root / "best_model_val.pt").resolve()
                    ),
                    "checkpoint_sha256": registration[
                        "checkpoint_sha256"
                    ],
                    "registration_sha256": registration["content_hash"],
                    "training_population": "scale_train",
                    "architecture_reselection_performed": False,
                }
            ),
            source_snapshot=source_snapshot(REPO_ROOT),
        )
        selected_refiner_path = output / "selected_token_refiner.json"
        write_immutable_json(selected_refiner_path, selected_refiner)

    final_root = output / "final_consumer"
    final_args = [
        "scripts/execute_retb_final_consumer_row.py",
        "--campaign-root",
        str(root),
        "--run-id",
        base_run_id,
        "--dataset-root",
        str(dataset_root),
        "--output-dir",
        str(final_root),
        "--graph-id",
        args.graph_id,
        "--scale-component-index",
        str(args.component_index),
        "--scale-joint-completion",
        str(joint_completion_path),
        "--scale-joint-root",
        str(joint_root),
        "--scale-run-id",
        f"RETB_SCALE_{args.graph_id}_S{args.pipeline_seed}",
        "--device",
        args.device,
    ]
    if selected_refiner_path is not None:
        final_args.extend(
            ["--scale-selected-refiner", str(selected_refiner_path)]
        )
    final_registration_name = (
        "registration.json"
        if definition["configuration"]["trainable"]
        else "reference_registration.json"
    )
    final_metric_name = (
        "val_design/metrics.json"
        if definition["configuration"]["trainable"]
        else "reference_metrics.json"
    )
    final_prediction_name = (
        "val_design/final_consumer_predictions_manifest.json"
        if definition["configuration"]["trainable"]
        else "final_consumer_predictions_manifest.json"
    )
    _run(
        final_args,
        expected=[
            final_root / final_registration_name,
            final_root / final_metric_name,
            final_root / final_prediction_name,
        ],
    )

    export_root = output / "export"
    _run(
        [
            "scripts/execute_retb_deployable_export_row.py",
            "--campaign-root",
            str(root),
            "--run-root",
            str(final_root),
            "--output-dir",
            str(export_root),
            "--scale-component-index",
            str(args.component_index),
            "--scale-joint-root",
            str(joint_root),
            "--scale-joint-completion",
            str(joint_completion_path),
        ],
        expected=[
            export_root / "deployable_retb_graph.json",
            export_root / "complete_graph_capacity.json",
            export_root / "research_graph_parity.json",
        ],
    )

    prediction = load_hashed_json(final_root / final_prediction_name)
    prediction_npz = (final_root / final_prediction_name).parent / prediction[
        "npz_filename"
    ]
    if _sha256(prediction_npz) != prediction["npz_sha256"]:
        raise ValueError("scale graph prediction bytes differ")
    with np.load(prediction_npz, allow_pickle=False) as values:
        logits = np.asarray(values["logits"], dtype=np.float32)
        identities = tuple(
            str(value) for value in values["identities"].tolist()
        )
    _, dataset = load_final_consumer_dataset(
        seed_dataset / "val_design" / "final_consumer_dataset.json",
        expected_split="val_design",
        expected_source=campaign["source"],
    )
    if identities != tuple(dataset.identities):
        raise ValueError("scale graph metric identity order differs")
    metrics = bind_source(
        evaluate_classification(
            logits,
            np.asarray(dataset.labels, dtype=np.int64),
            split="val_design",
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    metrics_path = output / "pre_stack_metrics.json"
    write_immutable_json(metrics_path, metrics)

    capacity = load_hashed_json(
        export_root / "complete_graph_capacity.json",
        expected_contract=COMPLETE_GRAPH_CAPACITY_CONTRACT,
    )
    deployment = load_hashed_json(
        export_root / "deployable_retb_graph.json",
        expected_contract=DEPLOYABLE_EXPORT_CONTRACT,
    )
    final_registration = load_hashed_json(
        final_root / final_registration_name
    )
    curves_path = final_root / "training_curves.json"
    if curves_path.is_file():
        curves = load_hashed_json(curves_path)
        selected_epoch = int(curves["selected_epoch"])
        selected = next(
            row
            for row in curves["rows"]
            if int(row["epoch"]) == selected_epoch
        )
        val_stop_accuracy = float(selected["val_stop"]["accuracy"])
        val_stop_ce = float(selected["val_stop"]["cross_entropy"])
        curve_sha = curves["content_hash"]
    else:
        selected_epoch = 1
        final_run = load_hashed_json(final_root / "assets" / "run.json")
        _, model, heads, fusion, refiner = load_final_consumer_template(
            final_root
            / "assets"
            / "template"
            / "final_consumer_template.json",
            expected_run_record_sha256=final_run["content_hash"],
            expected_source=campaign["source"],
        )
        _, val_stop_dataset = load_final_consumer_dataset(
            seed_dataset / "val_stop" / "final_consumer_dataset.json",
            expected_split="val_stop",
            expected_source=campaign["source"],
        )
        device = torch.device(
            "cuda"
            if args.device == "auto" and torch.cuda.is_available()
            else "cpu"
            if args.device == "auto"
            else args.device
        )
        model.to(device).eval()
        validation = evaluate_final_consumer(
            model=model,
            consumer_kind=final_run["consumer_kind"],
            loader=make_final_consumer_loader(
                val_stop_dataset,
                batch_size=128,
                seed=args.pipeline_seed,
                training=False,
            ),
            frozen_expert_heads=heads,
            frozen_offline_fusion=fusion,
            device=device,
            refiner=refiner,
        )
        val_stop_accuracy = float(validation["metrics"]["accuracy"])
        val_stop_ce = float(validation["metrics"]["cross_entropy"])
        curve_sha = final_registration["content_hash"]

    upstream = {
        "offline_experts": with_content_hash(
            components["offline_experts"]
        )["content_hash"],
        "offline_fusion": components[
            "offline_fusion_registration_sha256"
        ],
        "scale_offline_target_cache": load_hashed_json(
            Path(components["target_cache_root"])
            / "scale_train"
            / "target_cache_manifest.json"
        )["content_hash"],
        "scale_target_token_normalizer": refit["refits"][
            "target_token"
        ]["artifact_sha256"],
        "native_HLT_experts": with_content_hash(
            components["native_hlt_experts"]
        )["content_hash"],
        "native_HLT_fusion": components["component_hashes"][
            "native_hlt_fusion"
        ],
        "predictor_bundle": components["component_hashes"][
            "predictor_set"
        ],
        "token_refiner_or_identity": (
            final_registration["content_hash"]
            if selected_refiner_path is None
            else load_hashed_json(selected_refiner_path)["content_hash"]
        ),
        "final_consumer": final_registration["content_hash"],
        "deployable_export": deployment["content_hash"],
        "complete_graph_capacity": capacity["content_hash"],
        "training_curve": curve_sha,
        "val_stop_metrics": with_content_hash(
            {
                "accuracy": val_stop_accuracy,
                "cross_entropy": val_stop_ce,
            }
        )["content_hash"],
        "pre_stack_val_confirmation_prediction": prediction["content_hash"],
        "pre_stack_val_confirmation_metrics": metrics["content_hash"],
    }
    if set(upstream) != set(SCALE_COMPONENT_KEYS):
        raise RuntimeError("scale graph component coverage differs")
    component_root = output / "component_evidence"
    component_hashes = {}
    for name, digest in upstream.items():
        artifact = _component(
            name=name,
            upstream_sha256=digest,
            graph_id=args.graph_id,
            seed=args.pipeline_seed,
            source=source_snapshot(REPO_ROOT),
        )
        path = component_root / f"{name}.json"
        write_immutable_json(path, artifact)
        component_hashes[name] = artifact["content_hash"]
    # The scale contract binds the actual metric, not its lineage proxy.
    component_hashes["pre_stack_val_confirmation_metrics"] = metrics[
        "content_hash"
    ]
    component_hashes["scale_target_token_normalizer"] = refit["refits"][
        "target_token"
    ]["artifact_sha256"]
    artifact = bind_source(
        build_scale_graph_run(
            locked_scale_shortlist=shortlist,
            graph_id=args.graph_id,
            pipeline_seed=args.pipeline_seed,
            scale_refit_bundle=refit,
            component_hashes=component_hashes,
            selected_epoch=selected_epoch,
            val_stop_accuracy=val_stop_accuracy,
            val_stop_cross_entropy=val_stop_ce,
            analytical_flops_batch1=int(
                capacity["totals"]["analytical_inference_flops_batch1"]
            ),
            analytical_flops_batch128=int(
                capacity["totals"]["analytical_inference_flops_batch128"]
            ),
            parameter_count=int(capacity["totals"]["parameter_count"]),
            pre_stack_confirmation_metrics=metrics,
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    validate_scale_graph_run(
        artifact, locked_scale_shortlist=shortlist
    )
    publication = write_immutable_json(
        output / "scale_graph_run.json", artifact
    )
    print(json.dumps(publication, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
