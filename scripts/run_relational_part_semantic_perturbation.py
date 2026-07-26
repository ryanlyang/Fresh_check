#!/usr/bin/env python3
"""Run complete seed-101 semantic diagnostics for confirmation models."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relational_part import (  # noqa: E402
    CONFIRMATION_SUMMARY_CONTRACT,
    SEMANTIC_RUN_DIAGNOSTICS_CONTRACT,
    build_semantic_diagnostics_bundle,
    load_hashed_json,
    semantic_diagnostic_rows,
    validate_semantic_run_diagnostics_artifact,
    validate_campaign_source,
    write_immutable_json,
)
from scripts.evaluate_relational_part_semantic_controls import (  # noqa: E402
    main as semantic_main,
)


def _model_contract(root: Path, run_id: str) -> Path:
    candidates = [
        root / "registry" / "model_contracts" / f"{run_id}.json",
        root / "registry" / "confirmation_model_contracts" / f"{run_id}.json",
    ]
    matches = [path for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"semantic model {run_id} has {len(matches)} model contracts"
        )
    return matches[0]


def _load_reusable_artifact(
    output: Path,
    *,
    row: dict,
    confirmation_summary: dict,
):
    if not output.is_file():
        return None
    artifact = load_hashed_json(
        output,
        expected_contract=SEMANTIC_RUN_DIAGNOSTICS_CONTRACT,
    )
    validate_semantic_run_diagnostics_artifact(
        artifact,
        expected_row=row,
        confirmation_summary_sha256=confirmation_summary["content_hash"],
        nominal_winner_run_id=str(
            confirmation_summary["nominal_relational_winner_id"]
        ),
    )
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    root = args.campaign_root
    campaign = load_hashed_json(root / "campaign_spec.json")
    validate_campaign_source(campaign, repo_root=REPO_ROOT)
    summary = load_hashed_json(
        root / "selection" / "confirmation_summary.json",
        expected_contract=CONFIRMATION_SUMMARY_CONTRACT,
    )
    artifacts = []
    semantic_root = root / "selection" / "semantic_controls"
    for row in semantic_diagnostic_rows(summary):
        run_id = row["run_id"]
        families = list(row["families"])
        contract = _model_contract(root, run_id)
        run_dir = root / "runs" / run_id / "seed_101"
        output = semantic_root / "per_run" / f"{run_id}.json"
        if not args.dry_run:
            artifact = _load_reusable_artifact(
                output,
                row=row,
                confirmation_summary=summary,
            )
            if artifact is not None:
                artifacts.append(artifact)
                print(f"reused authenticated semantic artifact: {run_id}")
                continue
        command = [
            "perturb",
            "--confirmation-summary",
            str(root / "selection" / "confirmation_summary.json"),
            "--campaign-spec",
            str(root / "campaign_spec.json"),
            "--screening-registry",
            str(root / "registry" / "screening_registry.json"),
            "--normalization",
            str(root / "inputs" / "relation_normalization.json"),
            "--region-normalization",
            str(root / "inputs" / "region_normalization.json"),
            "--cache-dir",
            str(root / "inputs" / "hlt_cache"),
            "--tree-root",
            str(root / "inputs" / "relation_tree_cache"),
            "--run-id",
            run_id,
            "--seed",
            "101",
            "--families",
            *families,
            "--model-contract",
            str(contract),
            "--checkpoint",
            str(run_dir / "best_model_val.pt"),
            "--checkpoint-registration",
            str(run_dir / "checkpoint_registration.json"),
            "--device",
            args.device,
            "--output",
            str(output),
        ]
        if args.dry_run:
            command.append("--dry-run")
        result = semantic_main(command)
        if result:
            return int(result)
        if not args.dry_run:
            artifacts.append(
                load_hashed_json(
                    output,
                    expected_contract=SEMANTIC_RUN_DIAGNOSTICS_CONTRACT,
                )
            )
    if args.dry_run:
        return 0
    bundle = build_semantic_diagnostics_bundle(
        confirmation_summary=summary,
        run_artifacts=artifacts,
    )
    write_immutable_json(
        semantic_root / "perturbation_metrics.json",
        bundle,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
