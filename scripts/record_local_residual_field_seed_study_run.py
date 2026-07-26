#!/usr/bin/env python3
"""Validate a completed P7b replicate and write its seed-study completion marker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.local_particle_residual_field.fusion_campaign import (  # noqa: E402
    stable_fusion_json_hash,
)
from teacher_logit_reco.local_particle_residual_field.seed_study import (  # noqa: E402
    SEED_STUDY_RUN_COMPLETION_CONTRACT,
    load_json_object,
    require_seed_study_manifest,
    save_json,
    sha256_file,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-manifest", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest_path = Path(args.study_manifest)
    manifest = require_seed_study_manifest(load_json_object(manifest_path))
    output_dir = Path(args.output_dir).resolve()
    expected = Path(manifest["run_dirs"][str(args.seed)]["P7b"]).resolve()
    if output_dir != expected:
        raise ValueError(f"P7b output directory differs from frozen manifest: {output_dir} != {expected}")
    report_path = output_dir / "run_report.json"
    source_path = output_dir / "source_metadata.json"
    checkpoint_path = output_dir / "best_model_val.pt"
    report = load_json_object(report_path)
    source = load_json_object(source_path)
    if report.get("run_id") != "P7b" or report.get("deployable") is not True:
        raise ValueError("completed run is not a deployable P7b")
    if report.get("runtime_inputs") != "HLT_only":
        raise ValueError("P7b completion is not HLT-only at runtime")
    if report.get("final_test") not in (None, {}):
        raise ValueError("seed study forbids final-test evaluation")
    config = source.get("config")
    if not isinstance(config, dict) or int(config.get("seed", -1)) != args.seed:
        raise ValueError("P7b source metadata seed mismatch")
    reference_source = load_json_object(manifest["paths"]["p7b_reference_source_metadata"])
    reference_config = reference_source.get("config")
    if not isinstance(reference_config, dict):
        raise ValueError("P7b reference source metadata is missing its config")
    recipe_differences = sorted(
        key
        for key in set(reference_config) | set(config)
        if reference_config.get(key) != config.get(key)
    )
    expected_recipe_differences = [
        "oracle_teacher_checkpoint",
        "oracle_teacher_logits_dir",
        "output_dir",
        "seed",
        "student_warm_start_checkpoint",
    ]
    if recipe_differences != expected_recipe_differences:
        raise ValueError(
            "P7b seed-study replicate does not match the frozen active recipe; "
            f"expected differences {expected_recipe_differences}, observed {recipe_differences}"
        )
    reference_report = load_json_object(
        Path(manifest["p7b_reference_dir"]) / "run_report.json"
    )
    checkpoint_identity_pairs = {
        "oracle_teacher_checkpoint": (
            reference_report.get("oracle_teacher_checkpoint_hash"),
            report.get("oracle_teacher_checkpoint_hash"),
        ),
        "student_warm_start_checkpoint": (
            (reference_report.get("student_initialization") or {}).get(
                "student_init_checkpoint_hash"
            ),
            (report.get("student_initialization") or {}).get(
                "student_init_checkpoint_hash"
            ),
        ),
    }
    for field_name, (reference_hash, candidate_hash) in checkpoint_identity_pairs.items():
        if not reference_hash or candidate_hash != reference_hash:
            raise ValueError(
                f"P7b {field_name} path changed without matching checkpoint identity: "
                f"reference={reference_hash!r}, candidate={candidate_hash!r}"
            )
    if bool(config.get("oracle_logit_only_fallback")):
        raise ValueError("P7b seed study must use the online frozen-oracle forward")
    if dict(report.get("oracle_teacher_logits_paths") or {}):
        raise ValueError("P7b seed study unexpectedly resolved cached oracle logits")
    if report.get("selected_consumer_id") != manifest.get("selected_consumer_id"):
        raise ValueError("P7b selected consumer differs from the frozen study manifest")
    completion = {
        "contract": SEED_STUDY_RUN_COMPLETION_CONTRACT,
        "ok": True,
        "seed_study_manifest_hash": manifest["artifact_hash"],
        "seed_study_recipe": "P7b",
        "seed": int(args.seed),
        "runtime_inputs": "HLT_only",
        "deployable": True,
        "final_test_evaluated": False,
        "oracle_execution_mode": "frozen_checkpoint_online",
        "recipe_difference_paths": recipe_differences,
        "path_alias_checkpoint_identities": {
            field_name: candidate_hash
            for field_name, (_, candidate_hash) in checkpoint_identity_pairs.items()
        },
        "manifest_path": str(manifest_path.resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "run_report_path": str(report_path.resolve()),
        "run_report_sha256": sha256_file(report_path),
    }
    completion["artifact_hash"] = stable_fusion_json_hash(completion)
    save_json(output_dir / "seed_study_completion.json", completion)
    print(json.dumps(completion, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
