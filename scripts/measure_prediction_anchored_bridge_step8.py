#!/usr/bin/env python3
"""Plan or publish Step 8 semantic evidence and final measured preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.jetclass_data import load_split_manifest  # noqa: E402
from teacher_logit_reco.local_particle_residual_field import (  # noqa: E402
    A3_INTERACTION_RUN_IDS,
    ALL50_RUN_IDS,
    NEGATIVE_CONTROL_RUN_IDS,
    NORMAL_PILOT_BUDGET_BYTES,
    PAIRED3_HARD_CEILING_BYTES,
    STEP8_SPECIAL_CANONICAL_RUN_IDS,
    BundleResourceReference,
    PredictionAnchoredAll50HLG,
    Step8FixedStorage,
    measure_step8_registry_states,
    resolve_step8_run_recipe,
    resource_reference_from_artifact,
)
from teacher_logit_reco.local_particle_residual_field.bridge_campaign import (  # noqa: E402
    validate_campaign_registry,
)
from teacher_logit_reco.local_particle_residual_field.bridge_contracts import (  # noqa: E402
    sha256_file,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "measure"), default="plan")
    parser.add_argument("--physical45-scaler", required=True)
    parser.add_argument("--all50-scaler", required=True)
    parser.add_argument("--absolute-scaler", required=True)
    parser.add_argument("--deployed-resource-reference", required=True)
    parser.add_argument("--registry", default="")
    parser.add_argument("--fixed-storage", default="", help="measured fixed-storage JSON; required for measure")
    parser.add_argument("--parent-manifest", default="")
    parser.add_argument("--particle-width", type=int, default=0, help="dry-run only")
    parser.add_argument("--budget-mode", choices=("5gib", "6gib"), default="5gib")
    parser.add_argument("--output", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _read(path: str, label: str) -> dict[str, object]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(f"missing/unsafe {label}: {source}")
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _reference(value: dict[str, object]) -> BundleResourceReference:
    return resource_reference_from_artifact(value)


def _fixed(value: dict[str, object]) -> Step8FixedStorage:
    validate_content_hash(value, expected_contract="prediction_anchored_step8_fixed_storage_v1")
    return Step8FixedStorage(
        child_split_manifest_bytes=int(value["child_split_manifest_bytes"]),
        r0_weights_bytes=int(value["r0_weights_bytes"]),
        target_logit_namespace_bytes={str(k): int(v) for k, v in value["target_logit_namespace_bytes"].items()},
        recipes_bindings_reports_bytes=int(value["recipes_bindings_reports_bytes"]),
        final_deployable_bundle_bytes=int(value["final_deployable_bundle_bytes"]),
        measurement_basis=str(value.get("measurement_basis", "filesystem_measured")),
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    physical = _read(args.physical45_scaler, "physical45 scaler")
    all50 = _read(args.all50_scaler, "all50 scaler")
    absolute = _read(args.absolute_scaler, "absolute scaler")
    reference = _reference(_read(args.deployed_resource_reference, "deployed resource reference"))
    validate_content_hash(physical)
    validate_content_hash(all50)
    validate_content_hash(absolute, expected_contract="prediction_anchored_absolute_output_scaler_v1")
    if physical.get("channel_policy") != "physical45" or all50.get("channel_policy") != "all50":
        raise ValueError("Step 8 scaler channel policies are not physical45/all50")
    if args.parent_manifest:
        path = Path(args.parent_manifest)
        parent = load_split_manifest(path)
        source_hash = sha256_file(path)
        particle_width = int(parent.max_constits)
        if source_hash != reference.source_manifest_sha256 or particle_width != reference.particle_width:
            raise ValueError("source manifest disagrees with deployed reference")
    else:
        if not args.dry_run or args.particle_width <= 0:
            raise ValueError("production requires --parent-manifest; explicit width is dry-run only")
        source_hash = reference.source_manifest_sha256
        particle_width = int(args.particle_width)
        if particle_width != reference.particle_width:
            raise ValueError("dry-run particle width disagrees with deployed reference")
    if absolute.get("source_manifest_sha256") != source_hash:
        raise ValueError("absolute scaler belongs to another source manifest")
    b1 = PredictionAnchoredAll50HLG(
        ALL50_RUN_IDS[0],
        physical45_scaler_artifact=physical,
        all50_scaler_artifact=all50,
    )
    plan = with_content_hash(
        {
            "contract": "prediction_anchored_step8_operator_plan_v1",
            "a3_interaction_run_ids": list(A3_INTERACTION_RUN_IDS),
            "canonical_special_run_ids": list(STEP8_SPECIAL_CANONICAL_RUN_IDS),
            "all50_run_ids": list(ALL50_RUN_IDS),
            "negative_control_run_ids": list(NEGATIVE_CONTROL_RUN_IDS),
            "recipes": {
                run_id: resolve_step8_run_recipe(run_id).to_artifact()
                for run_id in (*A3_INTERACTION_RUN_IDS, *ALL50_RUN_IDS, "D10_TALT_A3", *NEGATIVE_CONTROL_RUN_IDS)
            },
            "b1_model_config": b1.config_artifact(),
            "source_manifest_sha256": source_hash,
            "particle_width": particle_width,
            "budget_mode": args.budget_mode,
            "production_measurement_required": True,
            "dense_field_cache_persisted": False,
        }
    )
    if args.mode == "plan":
        if not args.dry_run:
            if not args.output:
                raise ValueError("plan mode requires --output unless --dry-run")
            write_immutable_json(args.output, plan)
        print(json.dumps({"dry_run": args.dry_run, "plan": plan}, indent=2, sort_keys=True))
        return 0
    if not args.registry or not args.fixed_storage:
        raise ValueError("measure mode requires --registry and --fixed-storage")
    registry = _read(args.registry, "campaign registry")
    validate_campaign_registry(registry)
    fixed = _fixed(_read(args.fixed_storage, "fixed storage"))
    budget = NORMAL_PILOT_BUDGET_BYTES if args.budget_mode == "5gib" else PAIRED3_HARD_CEILING_BYTES
    updated, measurement = measure_step8_registry_states(
        registry,
        physical45_scaler_artifact=physical,
        all50_scaler_artifact=all50,
        absolute_scaler_artifact=absolute,
        source_manifest_sha256=source_hash,
        deployed_reference=reference,
        fixed_storage=fixed,
        selected_budget_bytes=budget,
    )
    output: dict[str, object] = {"dry_run": args.dry_run, "measurement": measurement}
    if not args.dry_run:
        if not args.output_dir:
            raise ValueError("measure mode requires --output-dir unless --dry-run")
        root = Path(args.output_dir)
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"Step 8 output directory is not empty: {root}")
        root.mkdir(parents=True, exist_ok=True)
        write_immutable_json(root / "campaign_registry_step8.json", updated)
        write_immutable_json(root / "step8_measurement.json", measurement)
        write_immutable_json(root / "step8_operator_plan.json", plan)
        output["persistent_artifacts"] = [
            "campaign_registry_step8.json", "step8_measurement.json", "step8_operator_plan.json"
        ]
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
