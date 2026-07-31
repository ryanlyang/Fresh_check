#!/usr/bin/env python3
"""Build genuine Stage-L evidence from one completed matched-seed graph."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.confirmation import (  # noqa: E402
    SEED_COMPONENT_KEYS,
    STAGE_L_GRAPH_REGISTRY_CONTRACT,
    validate_stage_l_graph_registry,
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
from teacher_logit_reco.relation_expert_token_bridge.paired_statistics import (  # noqa: E402
    build_paired_confirmation_statistics,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prediction(root: Path, run_id: str) -> tuple[dict[str, Any], np.ndarray, tuple[str, ...]]:
    run_root = root / "runs" / "final_consumers" / run_id
    candidates = (
        run_root / "val_design" / "final_consumer_predictions_manifest.json",
        run_root / "final_consumer_predictions_manifest.json",
    )
    paths = [path for path in candidates if path.is_file()]
    if len(paths) != 1:
        raise ValueError("Stage-L graph prediction manifest is absent/ambiguous")
    manifest = load_hashed_json(paths[0])
    npz_path = paths[0].parent / manifest["npz_filename"]
    if _sha256(npz_path) != manifest["npz_sha256"]:
        raise ValueError("Stage-L graph prediction payload bytes differ")
    with np.load(npz_path, allow_pickle=False) as payload:
        identities = tuple(str(value) for value in payload["identities"].tolist())
        logits = np.asarray(payload["logits"], dtype=np.float32)
    if len(identities) != len(set(identities)) or logits.shape != (
        len(identities),
        10,
    ):
        raise ValueError("Stage-L prediction identity/logit coverage differs")
    return manifest, logits, identities


def _labels(
    root: Path,
    *,
    seed: int,
    identities: tuple[str, ...],
) -> tuple[np.ndarray, str]:
    lock = load_hashed_json(
        root / "selection" / "predictor_bundle" / "predictor_bundle_lock.json"
    )
    configuration = json.loads(
        (
            root
            / "selection"
            / "predictor_bundle"
            / "inputs"
            / "selector_configuration.json"
        ).read_text("utf-8")
    )
    manifest_path = Path(
        configuration["label_manifest_paths_by_seed"][str(seed)]
    )
    label_manifest = load_hashed_json(manifest_path)
    if (
        label_manifest["content_hash"]
        != lock["selection_data_hashes"]["label_manifests"][str(seed)]
    ):
        raise ValueError("Stage-L label manifest differs from bundle lock")
    npz_path = Path(configuration["label_npz_paths"][str(seed)])
    if _sha256(npz_path) != label_manifest["label_npz_sha256"]:
        raise ValueError("Stage-L label payload bytes differ")
    with np.load(npz_path, allow_pickle=False) as payload:
        label_ids = tuple(str(value) for value in payload["identities"].tolist())
        labels = np.asarray(payload["labels"], dtype=np.int64)
    mapping = {identity: int(label) for identity, label in zip(label_ids, labels)}
    if set(mapping) != set(identities):
        raise ValueError("Stage-L label/prediction identity sets differ")
    return np.asarray([mapping[value] for value in identities]), label_manifest[
        "content_hash"
    ]


def _component_proxy(
    *,
    name: str,
    upstream_sha256: str,
    graph_id: str,
    seed: int,
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    return bind_source(
        with_content_hash(
            {
                "contract": "retb_stage_l_component_lineage_v1",
                "schema_version": 1,
                "component": name,
                "upstream_artifact_sha256": upstream_sha256,
                "graph_id": graph_id,
                "pipeline_seed": seed,
                "training_population": "model_train_500k",
                "stack_val_consumed": False,
                "final_test_consumed": False,
            }
        ),
        source_snapshot=snapshot,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--graph-registry", required=True, type=Path)
    parser.add_argument("--graph-id", required=True)
    parser.add_argument("--pipeline-seed", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    root = args.campaign_root.resolve()
    campaign = load_and_validate_campaign_source(root, repo_root=REPO_ROOT)
    registry = load_hashed_json(
        args.graph_registry,
        expected_contract=STAGE_L_GRAPH_REGISTRY_CONTRACT,
    )
    validate_stage_l_graph_registry(registry)
    definitions = {
        row["graph_id"]: row for row in registry["definitions"]
    }
    definition = definitions.get(args.graph_id)
    seed = int(args.pipeline_seed)
    if definition is None or seed not in {101, 202, 303}:
        raise ValueError("Stage-L evidence graph/seed is not registered")
    run_id = definition["configuration"]["run_ids_by_seed"][str(seed)]
    baseline_id = definition["named_baseline_graph_id"]
    baseline_run = definitions[baseline_id]["configuration"][
        "run_ids_by_seed"
    ][str(seed)]
    prediction, logits, identities = _prediction(root, run_id)
    baseline_prediction, baseline_logits, baseline_ids = _prediction(
        root, baseline_run
    )
    order = np.argsort(np.asarray(identities), kind="stable")
    baseline_map = {
        identity: baseline_logits[index]
        for index, identity in enumerate(baseline_ids)
    }
    canonical_ids = tuple(identities[index] for index in order)
    if set(baseline_map) != set(canonical_ids):
        raise ValueError("Stage-L paired graph identity sets differ")
    candidate_logits = logits[order]
    baseline_values = np.stack(
        [baseline_map[identity] for identity in canonical_ids]
    )
    labels, label_manifest_sha = _labels(
        root, seed=seed, identities=canonical_ids
    )
    snapshot = source_snapshot(REPO_ROOT)
    metrics = bind_source(
        evaluate_classification(
            candidate_logits, labels, split="val_design"
        ),
        source_snapshot=snapshot,
    )
    paired = bind_source(
        build_paired_confirmation_statistics(
            identities=canonical_ids,
            labels=labels,
            candidate_logits=candidate_logits,
            baseline_logits=baseline_values,
            candidate_graph_id=args.graph_id,
            baseline_graph_id=baseline_id,
            pipeline_seed=seed,
            candidate_prediction_sha256=prediction["content_hash"],
            baseline_prediction_sha256=baseline_prediction["content_hash"],
        ),
        source_snapshot=snapshot,
    )
    export_root = root / "exports" / run_id
    deployment = load_hashed_json(
        export_root / "deployable_retb_graph.json",
        expected_contract=DEPLOYABLE_EXPORT_CONTRACT,
    )
    capacity = load_hashed_json(
        export_root / "complete_graph_capacity.json",
        expected_contract=COMPLETE_GRAPH_CAPACITY_CONTRACT,
    )
    run = load_hashed_json(
        root / "runs" / "final_consumers" / run_id / "assets" / "run.json"
    )
    registration_candidates = (
        root / "runs" / "final_consumers" / run_id / "registration.json",
        root
        / "runs"
        / "final_consumers"
        / run_id
        / "reference_registration.json",
    )
    registration_paths = [
        path for path in registration_candidates if path.is_file()
    ]
    if len(registration_paths) != 1:
        raise ValueError("Stage-L final-consumer registration is ambiguous")
    registration = load_hashed_json(registration_paths[0])
    upstream = {
        "offline_experts": run["parent_hashes"]["frozen_offline_expert_heads"],
        "offline_fusion": run["parent_hashes"]["frozen_offline_fusion"],
        "offline_target_cache": run["parent_hashes"]["offline_target_cache"],
        "native_hlt_experts": run["parent_hashes"][
            "native_HLT_checkpoint_bundle"
        ],
        "native_hlt_fusion": run["parent_hashes"][
            "joint_prediction_checkpoint"
        ],
        "predictor_bundle": run["predictor_bundle_lock_sha256"],
        "refiner_or_identity": run["parent_hashes"].get(
            "selected_token_refiner", registration["content_hash"]
        ),
        "final_consumer": registration["content_hash"],
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    component_paths: dict[str, Path] = {}
    for name, upstream_sha in upstream.items():
        artifact = _component_proxy(
            name=name,
            upstream_sha256=upstream_sha,
            graph_id=args.graph_id,
            seed=seed,
            snapshot=snapshot,
        )
        path = output / f"{name}.json"
        write_immutable_json(path, artifact)
        component_paths[name] = path
    fixed = {
        "deployable_export": export_root / "deployable_retb_graph.json",
        "complete_graph_capacity": export_root / "complete_graph_capacity.json",
        "prediction_manifest": (
            root
            / "runs"
            / "final_consumers"
            / run_id
            / (
                "val_design/final_consumer_predictions_manifest.json"
                if (
                    root
                    / "runs"
                    / "final_consumers"
                    / run_id
                    / "val_design/final_consumer_predictions_manifest.json"
                ).is_file()
                else "final_consumer_predictions_manifest.json"
            )
        ),
        "metrics_artifact": output / "metrics.json",
        "paired_statistics": output / "paired_statistics.json",
    }
    component_paths.update(fixed)
    if set(component_paths) != set(SEED_COMPONENT_KEYS):
        raise RuntimeError("Stage-L component evidence coverage differs")
    write_immutable_json(fixed["metrics_artifact"], metrics)
    write_immutable_json(fixed["paired_statistics"], paired)
    token_errors = prediction.get("token_RMSE_after", {})
    normalized_error = (
        float(np.mean([float(value) for value in token_errors.values()]))
        if definition["predicts_tokens"]
        else None
    )
    epochs = 2 if campaign["campaign_profile"] == "miniature_test" else 40
    curves_path = (
        root / "runs" / "final_consumers" / run_id / "training_curves.json"
    )
    update_count = 1
    if curves_path.is_file():
        curves = load_hashed_json(curves_path)
        update_count = int(
            curves["planned_update_counts"]["total_optimizer_updates"]
        )
    summary = bind_source(
        with_content_hash(
            {
                "contract": "retb_stage_l_training_summary_v1",
                "schema_version": 1,
                "graph_id": args.graph_id,
                "pipeline_seed": seed,
                "training_population": "model_train_500k",
                "training_epochs": epochs,
                "optimizer_update_count": update_count,
                "normalized_token_error": normalized_error,
                "analytical_flops_batch1": capacity["totals"][
                    "analytical_inference_flops_batch1"
                ],
                "parameter_count": capacity["totals"]["parameter_count"],
                "actual_training_executed": True,
            }
        ),
        source_snapshot=snapshot,
    )
    write_immutable_json(output / "training_summary.json", summary)
    index = bind_source(
        with_content_hash(
            {
                "contract": "retb_stage_l_evidence_index_v1",
                "schema_version": 1,
                "graph_id": args.graph_id,
                "pipeline_seed": seed,
                "stage_l_graph_registry_sha256": registry["content_hash"],
                "component_artifacts": {
                    name: str(path)
                    for name, path in sorted(component_paths.items())
                },
                "training_summary": str(output / "training_summary.json"),
                "val_design_label_manifest_sha256": label_manifest_sha,
                "all_existing_training_outputs_reauthenticated": True,
                "performance_based_termination": False,
            }
        ),
        source_snapshot=snapshot,
    )
    write_immutable_json(output / "evidence_index.json", index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
