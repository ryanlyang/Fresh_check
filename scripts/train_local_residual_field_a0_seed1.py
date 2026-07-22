#!/usr/bin/env python3
"""Train A0_seed1 only from a validated immutable seed-recipe artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.hlt_baseline import save_json  # noqa: E402
from teacher_logit_reco.local_particle_residual_field.fusion_seed_control import (  # noqa: E402
    load_a0_seed1_recipe, sha256_file,
)
from teacher_logit_reco.local_particle_residual_field.fusion_campaign import stable_fusion_json_hash  # noqa: E402
from teacher_logit_reco.local_particle_residual_field.fusion_sources import (  # noqa: E402
    require_fusion_source_artifact_audit,
)
from teacher_logit_reco.local_particle_residual_field.tagger_train import (  # noqa: E402
    train_local_residual_field_tagger,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe-json", required=True)
    parser.add_argument("--audit-json", required=True)
    parser.add_argument("--source-artifact-audit", required=True)
    return parser.parse_args(argv)


def _load_json(path: str | Path) -> Mapping[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    recipe_path = Path(args.recipe_json)
    audit_path = Path(args.audit_json)
    source_audit_path = Path(args.source_artifact_audit)
    source_audit = require_fusion_source_artifact_audit(source_audit_path)
    recipe, config = load_a0_seed1_recipe(_load_json(recipe_path), _load_json(audit_path))
    if Path(config.output_dir).resolve() != recipe_path.parent.resolve():
        raise ValueError("recipe output_dir must be the directory containing the immutable recipe")

    report = dict(train_local_residual_field_tagger(config))
    identity = {
        "run_id": recipe.run_id,
        "display_alias": recipe.display_alias,
        "canonical_config_id": recipe.canonical_config_id,
        "seed_recipe_contract": recipe.contract,
        "seed_recipe_config_hash": recipe.config_hash,
        "seed_recipe_audit_hash": recipe.audit_hash,
        "runtime_inputs": "HLT_only",
        "uses_true_fields": False,
        "uses_offline_particles": False,
        "uses_teacher_logits_at_runtime": False,
        "deployable": True,
    }
    report.update(identity)
    output_dir = Path(config.output_dir)
    save_json(output_dir / "run_report.json", report)
    source_metadata_path = output_dir / "source_metadata.json"
    if source_metadata_path.is_file():
        source_metadata = dict(_load_json(source_metadata_path))
        source_metadata.update(identity)
        save_json(source_metadata_path, source_metadata)
    save_json(
        output_dir / "seed_control_recipe_binding.json",
        {**identity, "recipe_json": str(recipe_path), "audit_json": str(audit_path)},
    )
    completion = {
        "ok": True,
        "contract": "local_residual_field_a0_seed1_completion_v1",
        **identity,
        "checkpoint_path": str((output_dir / "best_model_val.pt").resolve()),
        "checkpoint_sha256": sha256_file(output_dir / "best_model_val.pt"),
        "run_report_path": str((output_dir / "run_report.json").resolve()),
        "run_report_sha256": sha256_file(output_dir / "run_report.json"),
        "recipe_path": str(recipe_path.resolve()), "recipe_sha256": sha256_file(recipe_path),
        "audit_path": str(audit_path.resolve()), "audit_sha256": sha256_file(audit_path),
        "source_artifact_audit_path": str(source_audit_path.resolve()),
        "source_artifact_audit_hash": source_audit["audit_hash"],
    }
    completion["artifact_hash"] = stable_fusion_json_hash(completion)
    save_json(output_dir / "seed_control_completion.json", completion)
    print(json.dumps({"ok": True, **identity, "checkpoint": report.get("checkpoint"),
                      "completion": str(output_dir / "seed_control_completion.json")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
