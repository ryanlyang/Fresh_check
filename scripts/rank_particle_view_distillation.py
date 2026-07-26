#!/usr/bin/env python3
"""Rank completed Step-7 rows strictly on model_val_select."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from teacher_logit_reco.local_particle_residual_field.particle_view import (  # noqa: E402
    rank_model_val_select_configurations,
    write_immutable_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", action="append", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    rows = [
        json.loads(Path(path).read_text(encoding="utf-8"))
        for path in args.metrics
    ]
    ranking = rank_model_val_select_configurations(rows)
    write_immutable_json(args.output, ranking)
    print(
        f"winner={ranking['winner']['configuration_id']} "
        f"content_hash={ranking['content_hash']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

