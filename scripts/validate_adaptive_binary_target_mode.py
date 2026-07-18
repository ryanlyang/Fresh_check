#!/usr/bin/env python3
"""Validate and print an immutable ABPH target-mode selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.adaptive_binary_pseudooffline.target_mode import (  # noqa: E402
    ABPH_TARGET_MODES,
    load_target_mode_selection,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--expected-mode", choices=ABPH_TARGET_MODES)
    parser.add_argument("--print-mode", action="store_true")
    args = parser.parse_args()
    result = load_target_mode_selection(
        args.selection,
        campaign_root=args.campaign_root,
        expected_mode=args.expected_mode,
    )
    print(result["selected_mode"] if args.print_mode else json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
