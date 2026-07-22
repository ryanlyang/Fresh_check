#!/usr/bin/env python3
"""Plan or publish Step 7 HLG/direct architecture measurements."""

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
    ARCH_A3_HLG_PRIMARY,
    ARCH_A5_HLG_ABSOLUTE,
    ARCH_A5S_HLG_SCRATCH,
    DIRECT_HLT,
    DIRECT_R0REP,
    STEP7_HIERARCHY_ARCHITECTURE_IDS,
    DeployedBundleResourceReference,
    build_capacity_matched_direct_hlg,
    build_step7_hlg_correction_model,
    measure_step7_registry_states,
    measure_step7_resources,
)
from teacher_logit_reco.local_particle_residual_field.bridge_campaign import (  # noqa: E402
    validate_campaign_registry,
)
from teacher_logit_reco.local_particle_residual_field.bridge_contracts import (  # noqa: E402
    canonical_sha256,
    sha256_file,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "measure"), default="plan")
    parser.add_argument("--scaler", required=True, help="immutable physical45 scaler JSON")
    parser.add_argument("--absolute-scaler", required=True, help="q0.001/q0.999 absolute diagnostic scaler JSON")
    parser.add_argument("--deployed-resource-reference", required=True, help="measured canonical R0+A3+T10 resource JSON")
    parser.add_argument("--registry", default="", help="campaign registry JSON for measure mode")
    parser.add_argument("--parent-manifest", default="", help="source split manifest; required outside dry-run")
    parser.add_argument("--particle-width", type=int, default=0, help="dry-run-only explicit particle width")
    parser.add_argument("--output", default="", help="immutable Step 7 plan JSON")
    parser.add_argument("--output-dir", default="", help="empty measurement output directory")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _read_json(path: str, *, label: str) -> dict[str, object]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(f"missing/unsafe {label}: {source}")
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _reference(value: dict[str, object]) -> DeployedBundleResourceReference:
    validate_content_hash(value, expected_contract="prediction_anchored_deployed_resource_reference_v1")
    names = (
        "particle_width", "valid_particles", "r0_parameters", "r0_forward_flops",
        "a3_parameters", "a3_forward_flops", "t10_parameters", "t10_forward_flops",
        "r0_checkpoint_sha256", "a3_config_sha256", "t10_checkpoint_sha256",
        "source_manifest_sha256",
    )
    return DeployedBundleResourceReference(**{name: value[name] for name in names})


def _plan(
    scaler: dict[str, object],
    absolute: dict[str, object],
    reference: DeployedBundleResourceReference,
) -> dict[str, object]:
    profiles = {}
    configs = {}
    for architecture_id in STEP7_HIERARCHY_ARCHITECTURE_IDS:
        model = build_step7_hlg_correction_model(
            architecture_id,
            scaler_artifact=scaler,
            absolute_scaler_artifact=(
                absolute if architecture_id in {ARCH_A5_HLG_ABSOLUTE, ARCH_A5S_HLG_SCRATCH} else None
            ),
        )
        configs[architecture_id] = model.config_artifact()
        profiles[architecture_id] = measure_step7_resources(
            model,
            particle_width=reference.particle_width,
            valid_particles=reference.valid_particles,
        ).to_artifact()
    canonical_config_hash = configs[ARCH_A3_HLG_PRIMARY]["content_hash"]
    canonical_profile = profiles[ARCH_A3_HLG_PRIMARY]
    if reference.a3_config_sha256 != canonical_config_hash:
        raise ValueError("deployed reference is bound to a different canonical A3 config")
    if (
        reference.a3_parameters != canonical_profile["total_parameters"]
        or reference.a3_forward_flops != canonical_profile["forward_flops"]
    ):
        raise ValueError("deployed reference A3 resource values disagree with the executed profile")
    direct_matches = {}
    for run_id in (DIRECT_HLT, DIRECT_R0REP):
        model, profile, match = build_capacity_matched_direct_hlg(
            run_id,
            scaler_artifact=scaler if run_id == DIRECT_R0REP else None,
            reference=reference,
        )
        configs[run_id] = model.config_artifact()
        profiles[run_id] = profile.to_artifact()
        direct_matches[run_id] = match
    return with_content_hash(
        {
            "contract": "prediction_anchored_step7_operator_plan_v1",
            "hierarchy_architecture_ids": list(STEP7_HIERARCHY_ARCHITECTURE_IDS),
            "direct_control_ids": [DIRECT_HLT, DIRECT_R0REP],
            "model_configs": configs,
            "resource_profiles": profiles,
            "canonical_a3_profile": profiles[ARCH_A3_HLG_PRIMARY],
            "deployed_resource_reference": reference.to_artifact(),
            "direct_capacity_matches": direct_matches,
            "all_direct_capacity_tolerances_passed": all(value["passed"] for value in direct_matches.values()),
            "dense_field_cache_persisted": False,
            "oracle_inference_input_present": False,
        }
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    scaler = _read_json(args.scaler, label="physical45 scaler")
    absolute = _read_json(args.absolute_scaler, label="absolute scaler")
    reference = _reference(_read_json(args.deployed_resource_reference, label="deployed resource reference"))
    validate_content_hash(scaler)
    validate_content_hash(absolute, expected_contract="prediction_anchored_absolute_output_scaler_v1")
    if args.parent_manifest:
        parent_path = Path(args.parent_manifest)
        if parent_path.is_symlink() or not parent_path.is_file():
            raise FileNotFoundError(f"missing/unsafe parent manifest: {parent_path}")
        parent = load_split_manifest(parent_path)
        particle_width = int(parent.max_constits)
        source_hash = sha256_file(parent_path)
        if particle_width != reference.particle_width or source_hash != reference.source_manifest_sha256:
            raise ValueError("deployed resource reference disagrees with the source manifest")
    else:
        if not args.dry_run or int(args.particle_width) <= 0:
            raise ValueError("production requires --parent-manifest; --particle-width is dry-run-only")
        particle_width = int(args.particle_width)
        if particle_width != reference.particle_width:
            raise ValueError("dry-run particle width disagrees with deployed resource reference")
        source_hash = reference.source_manifest_sha256
    if absolute.get("source_manifest_sha256") != source_hash:
        raise ValueError("absolute scaler belongs to a different source manifest")
    if args.mode == "plan":
        plan = _plan(scaler, absolute, reference)
        plan = with_content_hash(
            {
                **{key: value for key, value in plan.items() if key != "content_hash"},
                "particle_width_source": "source_manifest" if args.parent_manifest else "debug_explicit_dry_run",
                "source_manifest_sha256": source_hash,
            }
        )
        if not args.dry_run:
            if not args.output:
                raise ValueError("plan mode requires --output unless --dry-run is used")
            write_immutable_json(args.output, plan)
        print(json.dumps({"dry_run": bool(args.dry_run), "plan": plan}, indent=2, sort_keys=True))
        return 0
    if not args.registry:
        raise ValueError("measure mode requires --registry")
    registry = _read_json(args.registry, label="campaign registry")
    validate_campaign_registry(registry)
    updated, measurement = measure_step7_registry_states(
        registry,
        scaler_artifact=scaler,
        absolute_scaler_artifact=absolute,
        source_manifest_sha256=source_hash,
        deployed_reference=reference,
    )
    output: dict[str, object] = {
        "dry_run": bool(args.dry_run),
        "measurement": measurement,
        "updated_registry_sha256": updated["content_hash"],
    }
    if not args.dry_run:
        if not args.output_dir:
            raise ValueError("measure mode requires --output-dir unless --dry-run is used")
        root = Path(args.output_dir)
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"Step 7 measurement directory is not empty: {root}")
        root.mkdir(parents=True, exist_ok=True)
        write_immutable_json(root / "step7_measurement.json", measurement)
        write_immutable_json(root / "campaign_registry_step7.json", updated)
        output["persistent_artifacts"] = ["campaign_registry_step7.json", "step7_measurement.json"]
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
