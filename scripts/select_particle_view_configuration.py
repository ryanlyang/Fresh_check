#!/usr/bin/env python3
"""Select the three immutable Step-8 winner families."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from teacher_logit_reco.local_particle_residual_field.particle_view import (  # noqa: E402
    select_particle_view_winner_families,
    write_immutable_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replica-metric", action="append", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    rows = [
        json.loads(Path(path).read_text(encoding="utf-8"))
        for path in args.replica_metric
    ]
    selection = select_particle_view_winner_families(rows)
    write_immutable_json(args.output, selection)
    print(
        "privileged="
        f"{selection['selected_privileged_scientific_model']['configuration_id']} "
        "deployable="
        f"{selection['selected_pre_stage_g_hlt_deployable_model']['configuration_id']} "
        f"content_hash={selection['content_hash']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
