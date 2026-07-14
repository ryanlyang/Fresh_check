#!/usr/bin/env python3
"""Validate an HLT warm start against the exact campaign encoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.constrained_coarse_to_fine import (  # noqa: E402
    A0_HLT_BASELINE,
    build_end_to_end_tagger,
    load_hlt_stream_warm_start,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model, _ = build_end_to_end_tagger(A0_HLT_BASELINE, (), device="cpu")
    result = load_hlt_stream_warm_start(model.tagger, checkpoint)
    print(json.dumps({"ok": True, **result}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
