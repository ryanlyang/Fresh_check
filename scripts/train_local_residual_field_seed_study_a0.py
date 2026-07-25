#!/usr/bin/env python3
"""Train one audited from-scratch A0 replicate for the matched seed study."""

from __future__ import annotations

import argparse
from dataclasses import asdict
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
    build_a0_seed_study_config,
    load_json_object,
    require_seed_study_manifest,
    save_json,
    sha256_file,
)
from teacher_logit_reco.local_particle_residual_field.tagger_train import (  # noqa: E402
    train_local_residual_field_tagger,
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
    expected_dir = Path(manifest["run_dirs"][str(args.seed)]["A0"]).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir != expected_dir:
        raise ValueError(f"A0 output directory differs from frozen manifest: {output_dir} != {expected_dir}")
    source_metadata = load_json_object(manifest["paths"]["a0_source_metadata"])
    config, audit = build_a0_seed_study_config(
        source_metadata,
        seed=args.seed,
        output_dir=output_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    save_json(output_dir / "seed_study_recipe_audit.json", audit)
    save_json(output_dir / "seed_study_train_config.json", {"config": asdict(config)})
    report = dict(train_local_residual_field_tagger(config))
    identity = {
        "seed_study_manifest_hash": manifest["artifact_hash"],
        "seed_study_recipe": "A0",
        "seed": int(args.seed),
        "runtime_inputs": "HLT_only",
        "deployable": True,
        "final_test_evaluated": False,
    }
    report.update(identity)
    save_json(output_dir / "run_report.json", report)
    source_path = output_dir / "source_metadata.json"
    source = load_json_object(source_path)
    source.update(identity)
    save_json(source_path, source)
    completion = {
        "contract": SEED_STUDY_RUN_COMPLETION_CONTRACT,
        "ok": True,
        **identity,
        "manifest_path": str(manifest_path.resolve()),
        "checkpoint_path": str((output_dir / "best_model_val.pt").resolve()),
        "checkpoint_sha256": sha256_file(output_dir / "best_model_val.pt"),
        "run_report_path": str((output_dir / "run_report.json").resolve()),
        "run_report_sha256": sha256_file(output_dir / "run_report.json"),
    }
    completion["artifact_hash"] = stable_fusion_json_hash(completion)
    save_json(output_dir / "seed_study_completion.json", completion)
    print(json.dumps(completion, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
