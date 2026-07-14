#!/usr/bin/env python3
"""Create a provenance-explicit alias of an existing prediction block."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.constrained_coarse_to_fine import (  # noqa: E402
    DEFAULT_PREDICTION_SPLITS,
    cache_prediction_alias,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--alias-name", required=True)
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_PREDICTION_SPLITS))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report = cache_prediction_alias(
        args.prediction_dir,
        source_name=args.source_name,
        alias_name=args.alias_name,
        splits=tuple(args.splits),
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
