#!/usr/bin/env python3
"""Refresh a campaign preflight while preserving an existing execution lineage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

from jetclass_fresh.jetclass_data import RAW_TOKEN_DIM  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.local_particle_residual_field import (  # noqa: E402
    ARCH_A3_HLG_PRIMARY,
    STEP3_RUN_IDS,
    LocalResidualFieldReconstructorConfig,
    build_local_residual_field_reconstructor,
    build_representative_architecture_resource_reference,
    build_campaign_registry,
    build_clean_start_step8_fixed_storage,
    build_prediction_anchored_tigris_graph,
    build_step3_consumer_model,
    build_step7_hlg_correction_model,
    initialize_step3_root_from_reference,
    measure_step8_registry_states,
    record_step3_registry_measurements,
)
from teacher_logit_reco.local_particle_residual_field.bridge_campaign_policy import (  # noqa: E402
    build_campaign_reservations,
)
from teacher_logit_reco.local_particle_residual_field.bridge_contracts import (  # noqa: E402
    canonical_sha256,
    load_hashed_json,
    write_immutable_json,
)
from teacher_logit_reco.local_particle_residual_field.bridge_execution import (  # noqa: E402
    validate_prediction_anchored_execution_spec,
)
from teacher_logit_reco.local_particle_residual_field.targets import (  # noqa: E402
    local_particle_residual_field_layout,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-preflight",
        required=True,
        help="Existing preflight whose execution spec and representative assets are retained",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="New empty directory for the refreshed immutable preflight",
    )
    parser.add_argument(
        "--artifact-root",
        required=True,
        help="Existing campaign artifact root containing the reusable B0-B3 outputs",
    )
    parser.add_argument("--budget-gib", type=int, choices=(5, 6), default=5)
    return parser


def _read(root: Path, name: str) -> dict[str, object]:
    return load_hashed_json(root / name)


def _baseline_model_size(path: str | Path) -> str:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("model_config", "config"):
        value = payload.get(key) if isinstance(payload, dict) else None
        if isinstance(value, dict) and value.get("model_size") in {
            "tiny",
            "base",
            "large",
        }:
            return str(value["model_size"])
    raise ValueError("baseline checkpoint does not declare a supported model_size")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source = Path(args.source_preflight).resolve()
    output = Path(args.output_dir).resolve()
    artifact_root = Path(args.artifact_root).resolve()
    if source == output:
        raise ValueError("refreshed preflight must use a new directory")
    if not source.is_dir() or source.is_symlink():
        raise FileNotFoundError(f"missing/unsafe source preflight: {source}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refreshed preflight output is not empty: {output}")
    if not artifact_root.is_dir() or artifact_root.is_symlink():
        raise FileNotFoundError(f"missing/unsafe campaign artifact root: {artifact_root}")
    output.mkdir(parents=True, exist_ok=True)

    execution = _read(source, "prediction_anchored_execution_spec.json")
    validate_prediction_anchored_execution_spec(execution, verify_file_hashes=True)
    child = load_hashed_json(execution["child_manifest"]["path"])
    if child["content_hash"] != execution["child_manifest"]["content_hash"]:
        raise ValueError("execution-bound child split manifest changed")

    physical = _read(source, "representative_scaler_physical45.json")
    all50 = _read(source, "representative_scaler_all50.json")
    absolute = _read(source, "representative_absolute_scaler.json")
    source_reference_artifact = _read(
        source, "representative_architecture_resource_reference.json"
    )
    source_sha256 = str(execution["parent_manifest"]["sha256"])
    if source_reference_artifact.get("source_manifest_sha256") != source_sha256:
        raise ValueError("representative reference belongs to another execution")

    registry0 = build_campaign_registry(alternate_teacher_valid=True)
    model_size = _baseline_model_size(execution["baseline_checkpoint"]["path"])
    representative = {}
    for run_id in STEP3_RUN_IDS:
        model = build_step3_consumer_model(run_id, model_size=model_size)
        if run_id == "A0_C250":
            initialize_step3_root_from_reference(
                model,
                execution["baseline_checkpoint"]["path"],
                run_id=run_id,
                map_location="cpu",
            )
        representative[run_id] = {"model_state_dict": model.state_dict()}
    registry3, step3_measurement = record_step3_registry_measurements(
        registry0, representative
    )
    del representative

    a3 = build_step7_hlg_correction_model(
        ARCH_A3_HLG_PRIMARY, scaler_artifact=physical, dropout=0.05
    )
    names, groups, _ = local_particle_residual_field_layout()
    r0 = build_local_residual_field_reconstructor(
        LocalResidualFieldReconstructorConfig(
            variant="C0",
            particle_dim=RAW_TOKEN_DIM,
            field_dim=50,
            d_model=160,
            num_heads=5,
            num_layers=4,
            context_layers=1,
            dropout=0.05,
            attention_dropout=0.05,
            field_names=tuple(names),
            field_groups={key: tuple(value) for key, value in groups.items()},
        )
    )
    t10 = build_step3_consumer_model("T10_robust", model_size=model_size)
    reference = build_representative_architecture_resource_reference(
        r0_model=r0,
        t10_model=t10,
        a3_model=a3,
        particle_width=int(execution["max_constits"]),
        valid_particles=int(execution["max_constits"]),
        source_manifest_sha256=source_sha256,
    )
    reference_artifact = reference.to_artifact()

    fixed, fixed_evidence = build_clean_start_step8_fixed_storage(child, registry3)
    registry8, measurement = measure_step8_registry_states(
        registry3,
        physical45_scaler_artifact=physical,
        all50_scaler_artifact=all50,
        absolute_scaler_artifact=absolute,
        source_manifest_sha256=source_sha256,
        deployed_reference=reference,
        fixed_storage=fixed,
        selected_budget_bytes=int(args.budget_gib) * 1024**3,
    )
    fixed_artifact = fixed.to_artifact()
    formula_sizes = {
        "r0": fixed.r0_weights_bytes,
        "consumer": fixed.final_deployable_bundle_bytes,
        "metadata": fixed.recipes_bindings_reports_bytes,
    }
    formula_parents = {
        role: {
            "path": f"formula://{role}",
            "size_bytes": int(size),
            "sha256": canonical_sha256(
                {
                    "role": role,
                    "size_bytes": int(size),
                    "fixed_storage_sha256": fixed_artifact["content_hash"],
                }
            ),
            "basis": "clean_start_formula",
        }
        for role, size in formula_sizes.items()
    }
    reservations = build_campaign_reservations(
        registry8,
        execution_spec=execution,
        production_readiness=measurement["production_readiness"],
        fixed_parent_artifacts=formula_parents,
        final_deployable_bundle_bytes=fixed.final_deployable_bundle_bytes,
        representative_reference_sha256=reference_artifact["content_hash"],
    )
    graph = build_prediction_anchored_tigris_graph(
        registry8,
        reservations=reservations,
        execution_spec=execution,
        artifact_root=str(artifact_root),
        pack_size=4,
    )

    artifacts = {
        "prediction_anchored_execution_spec.json": execution,
        "campaign_registry_step1.json": registry0,
        "campaign_registry_step8.json": registry8,
        "step3_measurement.json": step3_measurement,
        "representative_scaler_physical45.json": physical,
        "representative_scaler_all50.json": all50,
        "representative_absolute_scaler.json": absolute,
        "representative_architecture_resource_reference.json": reference_artifact,
        "measured_fixed_storage.json": fixed_artifact,
        "fixed_storage_provenance.json": fixed_evidence,
        "step8_measurement.json": measurement,
        "campaign_reservations.json": reservations,
        "prediction_anchored_tigris_graph.json": graph,
    }
    for name, payload in artifacts.items():
        write_immutable_json(output / name, payload)

    result = {
        "ok": True,
        "mode": "preserve_existing_execution_lineage",
        "source_preflight": str(source),
        "output_dir": str(output),
        "artifact_root": str(artifact_root),
        "execution_spec_sha256": execution["content_hash"],
        "registry": str(output / "campaign_registry_step8.json"),
        "reservations": str(output / "campaign_reservations.json"),
        "execution_spec": str(
            output / "prediction_anchored_execution_spec.json"
        ),
        "graph": str(output / "prediction_anchored_tigris_graph.json"),
        "configuration_count": registry8["configuration_count"],
        "reconstruction_breadth_count": registry8[
            "reconstruction_breadth_count"
        ],
        "post_teacher_configuration_count": registry8[
            "post_teacher_configuration_count"
        ],
        "existing_b0_b3_outputs_modified": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
