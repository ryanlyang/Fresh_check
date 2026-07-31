#!/usr/bin/env python3
"""Select the globally frozen active token-refiner variant from complete rows."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    bind_source,
    load_hashed_json,
    with_content_hash,
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.provenance import (  # noqa: E402
    source_snapshot,
)
from teacher_logit_reco.relation_expert_token_bridge.workflow import (  # noqa: E402
    load_and_validate_campaign_source,
)


CONTRACT = "retb_selected_token_refiner_v1"
ELIGIBLE = ("TR1_NATIVE_BASE", "TR2_ALL_NATIVE")
SEEDS = (101, 202, 303)


def _metric(
    root: Path, carried_shape_role: str, variant: str, seed: int
) -> dict[str, Any]:
    return load_hashed_json(
        root
        / "runs"
        / "final_consumers"
        / (
            f"RETB_{carried_shape_role}_TR_REFINE_{variant}_ND0_NONE_"
            f"TOKEN_PREDICTED_S{seed}"
        )
        / "val_design"
        / "metrics.json"
    )


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise FloatingPointError(f"{name} is nonfinite")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--carried-shape-role", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    root = args.campaign_root.resolve()
    campaign = load_and_validate_campaign_source(root, repo_root=REPO_ROOT)
    ranking = []
    for variant in ELIGIBLE:
        rows = []
        for seed in SEEDS:
            metric = _metric(root, args.carried_shape_role, variant, seed)
            if (
                metric.get("source") != campaign.get("source")
                or metric["model_variant"] != variant
                or int(metric["pipeline_seed"]) != seed
                or metric["split"] != "val_design"
                or metric.get("stack_val_consumed") is not False
                or metric.get("final_test_consumed") is not False
            ):
                raise ValueError("token-refiner metric lineage differs")
            rmse = sum(
                _finite(value, "token_RMSE_after")
                for value in metric["token_RMSE_after"].values()
            ) / len(metric["token_RMSE_after"])
            rows.append(
                {
                    "pipeline_seed": seed,
                    "accuracy": _finite(
                        metric["metrics"]["accuracy"], "accuracy"
                    ),
                    "mean_token_RMSE_after": rmse,
                    "metric_sha256": metric["content_hash"],
                }
            )
        ranking.append(
            {
                "variant": variant,
                "mean_accuracy": sum(row["accuracy"] for row in rows)
                / len(rows),
                "mean_token_RMSE_after": sum(
                    row["mean_token_RMSE_after"] for row in rows
                )
                / len(rows),
                "seed_rows": rows,
            }
        )
    ranking.sort(
        key=lambda row: (
            -row["mean_accuracy"],
            row["mean_token_RMSE_after"],
            row["variant"],
        )
    )
    selected_variant = ranking[0]["variant"]
    selected_by_seed = {}
    for seed in SEEDS:
        output = (
            root
            / "runs"
            / "final_consumers"
            / (
                f"RETB_{args.carried_shape_role}_TR_REFINE_"
                f"{selected_variant}_ND0_NONE_"
                f"TOKEN_PREDICTED_S{seed}"
            )
        )
        registration = load_hashed_json(output / "registration.json")
        checkpoint = output / "best_model_val.pt"
        selected_by_seed[str(seed)] = {
            "registration_sha256": registration["content_hash"],
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": registration["checkpoint_sha256"],
        }
    artifact = bind_source(
        with_content_hash(
            {
                "contract": CONTRACT,
                "schema_version": 1,
                "carried_shape_role": args.carried_shape_role,
                "eligible_variants": list(ELIGIBLE),
                "ineligible_controls": [
                    "TR0_NONE",
                    "TR3_ZERO_NATIVE_SHAPE",
                ],
                "pipeline_seeds": list(SEEDS),
                "ranking_rule": [
                    "higher_mean_val_design_accuracy",
                    "lower_mean_val_design_token_RMSE_after",
                    "lexicographically_smaller_variant",
                ],
                "ranking": ranking,
                "selected_variant": selected_variant,
                "selected_by_seed": selected_by_seed,
                "scientific_underperformance_blocks_selection": False,
                "stack_val_consumed": False,
                "final_test_consumed": False,
            }
        ),
        source_snapshot=source_snapshot(REPO_ROOT),
    )
    write_immutable_json(args.output, artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
