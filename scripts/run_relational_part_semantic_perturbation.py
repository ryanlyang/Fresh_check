#!/usr/bin/env python3
"""Resolve and run all three validation-only relation perturbations."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relational_part import load_hashed_json  # noqa: E402
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
            f"semantic winner {run_id} has {len(matches)} model contracts"
        )
    return matches[0]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    root = args.campaign_root
    unary = load_hashed_json(
        root
        / "selection"
        / "semantic_controls"
        / "unary_control_registry.json"
    )
    run_id = unary["nominal_winner_run_id"]
    families = list(unary["unary_source_relation_set"])
    contract = _model_contract(root, run_id)
    run_dir = root / "runs" / run_id / "seed_101"
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
        str(
            root
            / "selection"
            / "semantic_controls"
            / "perturbation_metrics.json"
        ),
    ]
    if args.dry_run:
        command.append("--dry-run")
    return semantic_main(command)


if __name__ == "__main__":
    raise SystemExit(main())
