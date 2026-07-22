#!/usr/bin/env python3
"""Plan or publish Step 6 local/capacity architecture measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.local_particle_residual_field import (  # noqa: E402
    ARCH_A0M_CAPACITY_PARTICLE,
    STEP6_ARCHITECTURE_IDS,
    build_step6_correction_model,
    canonical_a3_resource_reference,
    measure_correction_resources,
    measure_step6_registry_states,
    particle_capacity_match,
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
from jetclass_fresh.jetclass_data import load_split_manifest  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "measure"), default="plan")
    parser.add_argument("--scaler", required=True, help="immutable physical45 scaler JSON")
    parser.add_argument("--registry", default="", help="campaign registry JSON for measure mode")
    parser.add_argument("--parent-manifest", default="", help="source split manifest; required for production")
    parser.add_argument(
        "--particle-width",
        type=int,
        default=0,
        help="debug-only explicit width, permitted only with --dry-run",
    )
    parser.add_argument("--output", default="", help="immutable Step 6 plan JSON")
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


def _plan(scaler: dict[str, object], particle_width: int) -> dict[str, object]:
    profiles = {}
    configs = {}
    raw_profiles = {}
    for architecture_id in STEP6_ARCHITECTURE_IDS:
        model = build_step6_correction_model(
            architecture_id,
            scaler_artifact=scaler,
        )
        profile = measure_correction_resources(
            model,
            particle_width=int(particle_width),
        )
        raw_profiles[architecture_id] = profile
        profiles[architecture_id] = profile.to_artifact()
        configs[architecture_id] = model.config_artifact()
    reference = canonical_a3_resource_reference(
        scaler_artifact=scaler,
        particle_width=int(particle_width),
    )
    capacity = particle_capacity_match(
        raw_profiles[ARCH_A0M_CAPACITY_PARTICLE], reference
    )
    if not capacity["passed"]:
        raise RuntimeError("Step 6 A0M capacity match is outside the locked tolerances")
    return with_content_hash(
        {
            "contract": "prediction_anchored_step6_operator_plan_v1",
            "particle_width": int(particle_width),
            "valid_mask_profile": "batch1_all_manifest_particles_valid",
            "architecture_ids": list(STEP6_ARCHITECTURE_IDS),
            "model_configs": configs,
            "resource_profiles": profiles,
            "canonical_a3_reference": reference.to_artifact(),
            "particle_capacity_match": capacity,
            "full_hlg_instantiated": False,
            "full_hlg_registry_measurement_written": False,
        }
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.parent_manifest:
        parent_path = Path(args.parent_manifest)
        if parent_path.is_symlink() or not parent_path.is_file():
            raise FileNotFoundError(f"missing/unsafe parent manifest: {parent_path}")
        parent = load_split_manifest(parent_path)
        particle_width = int(parent.max_constits)
        source_manifest_sha256 = sha256_file(parent_path)
    else:
        if not args.dry_run or int(args.particle_width) <= 0:
            raise ValueError(
                "production requires --parent-manifest; --particle-width is dry-run-only"
            )
        particle_width = int(args.particle_width)
        source_manifest_sha256 = canonical_sha256(
            {"debug_explicit_particle_width": particle_width}
        )
    scaler = _read_json(args.scaler, label="physical45 scaler")
    validate_content_hash(scaler)
    if args.mode == "plan":
        plan = _plan(scaler, particle_width)
        plan = with_content_hash(
            {
                **{key: value for key, value in plan.items() if key != "content_hash"},
                "particle_width_source": (
                    "source_manifest" if args.parent_manifest else "debug_explicit_dry_run"
                ),
                "source_manifest_sha256": (
                    source_manifest_sha256 if args.parent_manifest else None
                ),
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
    updated, measurement = measure_step6_registry_states(
        registry,
        scaler_artifact=scaler,
        particle_width=particle_width,
        source_manifest_sha256=source_manifest_sha256,
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
            raise FileExistsError(f"Step 6 measurement directory is not empty: {root}")
        root.mkdir(parents=True, exist_ok=True)
        write_immutable_json(root / "step6_measurement.json", measurement)
        write_immutable_json(root / "campaign_registry_step6.json", updated)
        output["persistent_artifacts"] = [
            "campaign_registry_step6.json",
            "step6_measurement.json",
        ]
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
