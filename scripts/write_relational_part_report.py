#!/usr/bin/env python3
"""Recompute paired statistics and publish the sealed JSON/Markdown report."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relational_part import (  # noqa: E402
    CONFIRMATION_SUMMARY_CONTRACT,
    FINAL_EVALUATION_CONTRACT,
    LOCKED_FINALISTS_CONTRACT,
    build_relational_part_report,
    load_final_predictions,
    load_hashed_json,
    paired_prediction_statistics_many,
    render_relational_part_markdown,
    write_immutable_json,
)


def _write_immutable_text(path: Path, value: str) -> None:
    if path.exists():
        raise FileExistsError(f"report already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locked-finalists", type=Path, required=True)
    parser.add_argument("--confirmation-summary", type=Path, required=True)
    parser.add_argument(
        "--final-evaluation", type=Path, action="append", default=[]
    )
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=917_301)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    lock = load_hashed_json(
        args.locked_finalists, expected_contract=LOCKED_FINALISTS_CONTRACT
    )
    confirmation = load_hashed_json(
        args.confirmation_summary,
        expected_contract=CONFIRMATION_SUMMARY_CONTRACT,
    )
    evaluations = [
        load_hashed_json(path, expected_contract=FINAL_EVALUATION_CONTRACT)
        for path in args.final_evaluation
    ]
    expected = len(lock["evaluation_rows"]) * 3
    resolved = {
        "locked_finalists_sha256": lock["content_hash"],
        "confirmation_summary_sha256": confirmation["content_hash"],
        "expected_final_evaluation_count": expected,
        "supplied_final_evaluation_count": len(evaluations),
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
        "dry_run": bool(args.dry_run),
    }
    print(json.dumps(resolved, indent=2, sort_keys=True))
    if args.dry_run:
        return 0
    if len(evaluations) != expected:
        raise ValueError("report requires every locked run at all three seeds")
    by_key = {
        (str(value["run_id"]), int(value["seed"])): value
        for value in evaluations
    }
    if len(by_key) != len(evaluations):
        raise ValueError("duplicate final evaluation run/seed")
    prediction_cache = {}
    for path, evaluation in zip(args.final_evaluation, evaluations):
        prediction_path = path.parent / evaluation["prediction_file"]
        prediction = load_final_predictions(prediction_path)
        if prediction["file_sha256"] != evaluation["prediction_file_sha256"]:
            raise ValueError("final metric and prediction hashes disagree")
        prediction_cache[(evaluation["run_id"], int(evaluation["seed"]))] = (
            prediction
        )
    paired = {
        str(row["run_id"]): {}
        for row in lock["evaluation_rows"]
        if str(row["run_id"]) != str(lock["baseline_id"])
    }
    baseline_id = str(lock["baseline_id"])
    for seed in (101, 202, 303):
        candidates = {
            run_id: prediction_cache[(run_id, seed)]
            for run_id in paired
        }
        seed_statistics = paired_prediction_statistics_many(
            candidates,
            prediction_cache[(baseline_id, seed)],
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
        )
        for run_id, statistic in seed_statistics.items():
            paired[run_id][str(seed)] = statistic
    report = build_relational_part_report(
        locked_finalists=lock,
        confirmation_summary=confirmation,
        final_evaluations=evaluations,
        paired_statistics=paired,
    )
    write_immutable_json(args.json_output, report)
    _write_immutable_text(
        args.markdown_output, render_relational_part_markdown(report)
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
