#!/usr/bin/env python3
"""Fresh-process, HLT-only reload audit for a particle-view bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from teacher_logit_reco.local_particle_residual_field.particle_view.contracts import (  # noqa: E402
    sha256_file,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.local_particle_residual_field.particle_view.deployment import (  # noqa: E402
    PARTICLE_VIEW_BUNDLE_INPUT_NAMES,
    PARTICLE_VIEW_FRESH_RELOAD_AUDIT_CONTRACT,
    PARTICLE_VIEW_RELOAD_FIXTURE_CONTRACT,
    load_exported_particle_view_bundle,
    validate_particle_view_bundle_export,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-manifest", type=Path, required=True)
    parser.add_argument("--fixture-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    bundle_validation = validate_particle_view_bundle_export(
        args.bundle_manifest
    )
    bundle_payload = json.loads(args.bundle_manifest.read_text(encoding="utf-8"))
    fixture = json.loads(args.fixture_manifest.read_text(encoding="utf-8"))
    validate_content_hash(
        fixture, expected_contract=PARTICLE_VIEW_RELOAD_FIXTURE_CONTRACT
    )
    if fixture["bundle_export_sha256"] != bundle_validation["content_hash"]:
        raise ValueError("reload fixture belongs to a different bundle export")
    if fixture["contains_labels"] or fixture["contains_offline_inputs"]:
        raise ValueError("reload fixture is not HLT-only")
    data_path = args.fixture_manifest.parent / fixture["data_file"]
    if sha256_file(data_path) != fixture["data_file_sha256"]:
        raise ValueError("reload fixture file hash mismatch")
    with np.load(data_path, allow_pickle=False) as data:
        inputs = tuple(
            torch.from_numpy(np.ascontiguousarray(data[name]))
            for name in PARTICLE_VIEW_BUNDLE_INPUT_NAMES
        )
        reference = torch.from_numpy(
            np.ascontiguousarray(data["reference_logits"])
        ).float()
    module = load_exported_particle_view_bundle(args.bundle_manifest)
    with torch.no_grad():
        observed = module(*inputs).detach().cpu().float()
    maximum = float((observed - reference).abs().max().item())
    tolerance = float(bundle_payload["reload_tolerance"])
    if maximum > tolerance:
        raise RuntimeError("fresh-process bundle logits differ from reference")
    audit = with_content_hash(
        {
            "contract": PARTICLE_VIEW_FRESH_RELOAD_AUDIT_CONTRACT,
            "bundle_export_sha256": bundle_validation["content_hash"],
            "fixture_sha256": fixture["content_hash"],
            "archive_sha256": bundle_payload["archive_sha256"],
            "required_inputs": list(PARTICLE_VIEW_BUNDLE_INPUT_NAMES),
            "maximum_absolute_difference": maximum,
            "tolerance": tolerance,
            "only_hlt_inputs_visible": True,
            "oracle_imports_required": False,
            "passed": True,
        }
    )
    write_immutable_json(args.output, audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
