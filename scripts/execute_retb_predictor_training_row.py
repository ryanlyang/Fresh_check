#!/usr/bin/env python3
"""Prepare aligned datasets and train one authenticated predictor row."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.prepare_retb_predictor_dataset import main as prepare_main  # noqa: E402
from scripts.train_retb_predictor import main as train_main  # noqa: E402
from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    load_hashed_json,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument(
        "--training-role",
        choices=("model_train", "scale_train"),
        default="model_train",
    )
    for split in ("model-train", "val-stop", "val-design"):
        parser.add_argument(
            f"--{split}-target-cache", required=True, type=Path
        )
        parser.add_argument(
            f"--{split}-evidence", required=True, type=Path
        )
    parser.add_argument("--target-normalizer", required=True, type=Path)
    parser.add_argument("--target-checkpoint", required=True, type=Path)
    parser.add_argument("--fusion-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    campaign = load_and_validate_campaign_source(
        args.campaign_root, repo_root=REPO_ROOT
    )
    run = load_hashed_json(args.run)
    if run.get("source") != campaign.get("source"):
        raise ValueError("predictor row run source differs")
    prepared = args.output_dir / "prepared"
    split_paths = {}
    for cli_name, attribute in (
        ("model-train", "model_train"),
        ("val-stop", "val_stop"),
        ("val-design", "val_design"),
    ):
        logical_role = (
            args.training_role
            if attribute == "model_train"
            else attribute
        )
        output = prepared / f"{logical_role}.npz"
        metadata = prepared / f"{logical_role}.json"
        prepare_main(
            [
                "--campaign-root",
                str(args.campaign_root),
                "--target-cache-manifest",
                str(getattr(args, f"{attribute}_target_cache")),
                "--evidence-npz",
                str(getattr(args, f"{attribute}_evidence")),
                "--target-checkpoint",
                str(args.target_checkpoint),
                "--fusion-checkpoint",
                str(args.fusion_checkpoint),
                "--expert-id",
                str(run["expert_id"]),
                "--pipeline-seed",
                str(run["pipeline_seed"]),
                "--output",
                str(output),
                "--metadata-output",
                str(metadata),
            ]
        )
        split_paths[cli_name] = output
    training = args.output_dir / "training"
    inference = args.output_dir / "val_design"
    train_main(
        [
            "--campaign-root",
            str(args.campaign_root),
            "--run",
            str(args.run),
            "--model-train",
            str(split_paths["model-train"]),
            "--val-stop",
            str(split_paths["val-stop"]),
            "--val-design",
            str(split_paths["val-design"]),
            "--target-normalizer",
            str(args.target_normalizer),
            "--target-checkpoint",
            str(args.target_checkpoint),
            "--fusion-checkpoint",
            str(args.fusion_checkpoint),
            "--output-dir",
            str(training),
            "--val-design-output",
            str(inference),
            "--device",
            args.device,
            "--training-role",
            args.training_role,
        ]
    )
    print(
        json.dumps(
            {
                "run_id": run["run_id"],
                "training_output": str(training),
                "val_design_output": str(inference),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
