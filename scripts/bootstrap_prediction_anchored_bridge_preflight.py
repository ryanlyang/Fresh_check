#!/usr/bin/env python3
"""Build every immutable clean-start control needed to submit the full pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM, load_split_manifest
from teacher_logit_reco.local_particle_residual_field import (
    ARCH_A3_HLG_PRIMARY,
    BRIDGE_CHANNEL_ALL50,
    BRIDGE_CHANNEL_PHYSICAL45,
    STEP3_RUN_IDS,
    ConsumerCampaignConfig,
    StreamedR0TrainConfig,
    build_representative_architecture_resource_reference,
    build_campaign_registry,
    build_child_split_manifest,
    build_clean_start_step8_fixed_storage,
    build_prediction_anchored_execution_spec,
    build_prediction_anchored_tigris_graph,
    build_step3_consumer_model,
    build_step7_hlg_correction_model,
    fit_absolute_output_scaler,
    fit_bridge_scalers,
    initialize_step3_root_from_reference,
    measure_step8_registry_states,
    record_step3_registry_measurements,
)
from teacher_logit_reco.local_particle_residual_field.bridge_campaign_policy import (
    build_campaign_reservations,
)
from teacher_logit_reco.local_particle_residual_field.bridge_contracts import (
    canonical_sha256,
    sha256_file,
    write_immutable_json,
)
from teacher_logit_reco.local_particle_residual_field.bridge_execution import (
    write_prediction_anchored_execution_spec,
)
from teacher_logit_reco.local_particle_residual_field.model import (
    LocalResidualFieldReconstructorConfig,
    build_local_residual_field_reconstructor,
)
from teacher_logit_reco.local_particle_residual_field.targets import (
    local_particle_residual_field_layout,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-manifest", required=True)
    parser.add_argument("--hlt-cache-dir", required=True)
    parser.add_argument("--offline-cache-dir", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--budget-gib", type=int, choices=(5, 6), default=5)
    return parser


def _baseline_model_size(path: str | Path) -> str:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("model_config", "config"):
        value = payload.get(key) if isinstance(payload, dict) else None
        if isinstance(value, dict) and value.get("model_size") in {"tiny", "base", "large"}:
            return str(value["model_size"])
    raise ValueError("baseline checkpoint does not declare a supported model_size")


def _representative_scalers(source_sha: str):
    rng = np.random.default_rng(810_404)
    f0 = rng.normal(size=(8, 16, 50)).astype(np.float32)
    f_true = f0 + rng.normal(scale=0.2, size=f0.shape).astype(np.float32)
    mask = np.ones((8, 16), dtype=bool)
    parents = {"source_manifest_sha256": source_sha}
    physical = fit_bridge_scalers(
        [(f0, f_true, mask)], parent_hashes=parents,
        channel_policy=BRIDGE_CHANNEL_PHYSICAL45,
    ).to_artifact()
    all50 = fit_bridge_scalers(
        [(f0, f_true, mask)], parent_hashes=parents,
        channel_policy=BRIDGE_CHANNEL_ALL50,
    ).to_artifact()
    recipe_sha = canonical_sha256({"clean_start_representative_recipe": True})
    bridge = f0 + np.float32(0.1) * (f_true - f0)
    absolute = fit_absolute_output_scaler(
        [(bridge, mask)], source_manifest_sha256=source_sha,
        bridge_recipe_sha256=recipe_sha, epsilon=np.asarray(physical["epsilon"]),
    )
    return physical, all50, absolute


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"preflight output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    parent_path = Path(args.parent_manifest).resolve()
    parent = load_split_manifest(parent_path)
    source_sha = sha256_file(parent_path)
    child = build_child_split_manifest(parent)
    registry0 = build_campaign_registry(alternate_teacher_valid=False)
    child_path = output / "prediction_anchored_child_splits.json"
    registry0_path = output / "campaign_registry_step1.json"
    write_immutable_json(child_path, child)
    write_immutable_json(registry0_path, registry0)

    model_size = _baseline_model_size(args.baseline_checkpoint)
    execution = build_prediction_anchored_execution_spec(
        parent_manifest_path=parent_path,
        child_manifest_path=child_path,
        hlt_cache_dir=args.hlt_cache_dir,
        offline_cache_dir=args.offline_cache_dir,
        baseline_checkpoint_path=args.baseline_checkpoint,
        r0_config=StreamedR0TrainConfig(output_dir="__RUNTIME_OUTPUT_DIR__"),
        consumer_config=ConsumerCampaignConfig(
            baseline_steps=10_000,
            bridge_finetune_steps=2_000,
            batch_size=128,
            evaluation_interval_steps=200,
            model_size=model_size,
        ),
    )
    execution_path = output / "prediction_anchored_execution_spec.json"
    write_prediction_anchored_execution_spec(execution_path, execution)

    representative = {}
    for run_id in STEP3_RUN_IDS:
        model = build_step3_consumer_model(run_id, model_size=model_size)
        if run_id == "A0_C250":
            initialize_step3_root_from_reference(
                model, args.baseline_checkpoint, run_id=run_id, map_location="cpu"
            )
        representative[run_id] = {"model_state_dict": model.state_dict()}
    registry3, step3_measurement = record_step3_registry_measurements(
        registry0, representative
    )
    del representative

    physical, all50, absolute = _representative_scalers(source_sha)
    write_immutable_json(output / "representative_scaler_physical45.json", physical)
    write_immutable_json(output / "representative_scaler_all50.json", all50)
    write_immutable_json(output / "representative_absolute_scaler.json", absolute)

    a3 = build_step7_hlg_correction_model(
        ARCH_A3_HLG_PRIMARY, scaler_artifact=physical, dropout=0.05
    )
    names, groups, _ = local_particle_residual_field_layout()
    r0 = build_local_residual_field_reconstructor(
        LocalResidualFieldReconstructorConfig(
            variant="C0", particle_dim=RAW_TOKEN_DIM, field_dim=50,
            d_model=160, num_heads=5, num_layers=4, context_layers=1,
            dropout=0.05, attention_dropout=0.05,
            field_names=tuple(names),
            field_groups={key: tuple(value) for key, value in groups.items()},
        )
    )
    t10 = build_step3_consumer_model("T10_clean", model_size=model_size)
    reference = build_representative_architecture_resource_reference(
        r0_model=r0,
        t10_model=t10,
        a3_model=a3,
        particle_width=int(parent.max_constits), valid_particles=int(parent.max_constits),
        source_manifest_sha256=source_sha,
    )
    reference_artifact = reference.to_artifact()
    representative_reference_path = (
        output / "representative_architecture_resource_reference.json"
    )
    write_immutable_json(representative_reference_path, reference_artifact)

    fixed, fixed_evidence = build_clean_start_step8_fixed_storage(child, registry3)
    write_immutable_json(output / "measured_fixed_storage.json", fixed.to_artifact())
    write_immutable_json(output / "fixed_storage_provenance.json", fixed_evidence)
    budget = int(args.budget_gib) * 1024**3
    registry8, measurement = measure_step8_registry_states(
        registry3,
        physical45_scaler_artifact=physical,
        all50_scaler_artifact=all50,
        absolute_scaler_artifact=absolute,
        source_manifest_sha256=source_sha,
        deployed_reference=reference,
        fixed_storage=fixed,
        selected_budget_bytes=budget,
    )
    registry_path = output / "campaign_registry_step8.json"
    measurement_path = output / "step8_measurement.json"
    write_immutable_json(registry_path, registry8)
    write_immutable_json(measurement_path, measurement)
    write_immutable_json(output / "step3_measurement.json", step3_measurement)

    fixed_artifact = fixed.to_artifact()
    formula_sizes = {
        "r0": fixed.r0_weights_bytes,
        "consumer": fixed.final_deployable_bundle_bytes,
        "metadata": fixed.recipes_bindings_reports_bytes,
    }
    formula_parents = {
        role: {
            "path": f"formula://{role}", "size_bytes": int(size),
            "sha256": canonical_sha256({
                "role": role, "size_bytes": int(size),
                "fixed_storage_sha256": fixed_artifact["content_hash"],
            }),
            "basis": "clean_start_formula",
        }
        for role, size in formula_sizes.items()
    }
    reservations = build_campaign_reservations(
        registry8, execution_spec=execution,
        production_readiness=measurement["production_readiness"],
        fixed_parent_artifacts=formula_parents,
        final_deployable_bundle_bytes=fixed.final_deployable_bundle_bytes,
        representative_reference_sha256=reference_artifact["content_hash"],
    )
    reservations_path = output / "campaign_reservations.json"
    write_immutable_json(reservations_path, reservations)
    graph = build_prediction_anchored_tigris_graph(
        registry8, reservations=reservations, execution_spec=execution,
        artifact_root=str(Path(args.artifact_root).resolve()), pack_size=4,
    )
    graph_path = output / "prediction_anchored_tigris_graph.json"
    write_immutable_json(graph_path, graph)
    result = {
        "ok": True,
        "registry": str(registry_path), "reservations": str(reservations_path),
        "execution_spec": str(execution_path), "graph": str(graph_path),
        "artifact_root": str(Path(args.artifact_root).resolve()),
        "production_submission_allowed": True,
        "representative_measurements_only": True,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
