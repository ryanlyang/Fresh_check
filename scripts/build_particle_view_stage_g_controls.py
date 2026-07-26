#!/usr/bin/env python3
"""Resolve A0-long and selected-resource controls from the fairness ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from teacher_logit_reco.local_particle_residual_field.particle_view import (  # noqa: E402
    DirectControlCandidate,
    build_stage_g_control_plan,
    write_immutable_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fairness-ledger", required=True)
    parser.add_argument("--candidate-registry", required=True)
    parser.add_argument("--a0-seed101-sha256", required=True)
    parser.add_argument("--a0-seed202-sha256", required=True)
    parser.add_argument("--a0-seed303-sha256", required=True)
    parser.add_argument("--a0-config-sha256", required=True)
    parser.add_argument("--output", required=True)
    return parser


def _load(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    raw_candidates = _load(args.candidate_registry)
    rows = raw_candidates.get("candidates", raw_candidates)
    candidates = [DirectControlCandidate(**row) for row in rows]
    plan = build_stage_g_control_plan(
        fairness_ledger=_load(args.fairness_ledger),
        candidates=candidates,
        a0_checkpoint_by_seed={
            101: args.a0_seed101_sha256,
            202: args.a0_seed202_sha256,
            303: args.a0_seed303_sha256,
        },
        a0_config_sha256=args.a0_config_sha256,
    )
    write_immutable_json(args.output, plan)
    print(
        f"jobs={plan['job_count']} warnings={len(plan['quality_warnings'])} "
        f"content_hash={plan['content_hash']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
