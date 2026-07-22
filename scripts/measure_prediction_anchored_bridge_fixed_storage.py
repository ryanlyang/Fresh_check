#!/usr/bin/env python3
"""Measure Step 8 fixed persistent bytes from actual immutable artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.local_particle_residual_field import (  # noqa: E402
    measure_step8_fixed_storage,
)
from teacher_logit_reco.local_particle_residual_field.bridge_contracts import (  # noqa: E402
    write_immutable_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child-split-manifest", required=True)
    parser.add_argument("--r0-weights", required=True)
    parser.add_argument(
        "--target-namespace", action="append", default=[], metavar="NAME=PATH",
        help="repeat for primary, all50, N3, and conditional alternate",
    )
    parser.add_argument(
        "--metadata-path", action="append", default=[],
        help="repeat for bounded recipe/binding/report files or directories",
    )
    parser.add_argument("--final-deployable-bundle", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _namespaces(values: list[str]) -> dict[str, str]:
    output = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError("--target-namespace must use NAME=PATH")
        name, path = raw.split("=", 1)
        if not name or not path or name in output:
            raise ValueError("target namespace name/path is empty or duplicated")
        output[name] = path
    return output


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    storage, measurement = measure_step8_fixed_storage(
        child_split_manifest_path=args.child_split_manifest,
        r0_weights_path=args.r0_weights,
        target_logit_namespace_paths=_namespaces(args.target_namespace),
        recipes_bindings_reports_paths=args.metadata_path,
        final_deployable_bundle_path=args.final_deployable_bundle,
    )
    output: dict[str, object] = {
        "dry_run": bool(args.dry_run),
        "fixed_storage": storage.to_artifact(),
        "measurement": measurement,
    }
    if not args.dry_run:
        if not args.output_dir:
            raise ValueError("production measurement requires --output-dir")
        root = Path(args.output_dir)
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"fixed-storage output directory is not empty: {root}")
        root.mkdir(parents=True, exist_ok=True)
        write_immutable_json(root / "measured_fixed_storage.json", storage.to_artifact())
        write_immutable_json(root / "measured_fixed_storage_provenance.json", measurement)
        output["persistent_artifacts"] = [
            "measured_fixed_storage.json", "measured_fixed_storage_provenance.json"
        ]
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
