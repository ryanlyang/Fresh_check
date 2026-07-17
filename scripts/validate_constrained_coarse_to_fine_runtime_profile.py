#!/usr/bin/env python3
"""Validate a C2F candidate/approved runtime profile and optionally resolve one run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.constrained_coarse_to_fine.runtime_profiles import (  # noqa: E402
    execution_shell_exports,
    resolve_execution,
    validate_runtime_profile,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--expected-status", choices=("accelerated_candidate_v1", "accelerated_approved_v1"), required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--emit-shell", action="store_true")
    parser.add_argument("--no-current-environment-check", action="store_true")
    args = parser.parse_args()
    validated = validate_runtime_profile(
        args.profile,
        expected_status=args.expected_status,
        require_current_environment=not args.no_current_environment_check,
    )
    if args.emit_shell:
        if args.run_id is None:
            raise SystemExit("--emit-shell requires --run-id")
        print(execution_shell_exports(validated["profile"], args.run_id))
    else:
        output = {key: value for key, value in validated.items() if key != "profile"}
        if args.run_id is not None:
            output["execution"] = resolve_execution(validated["profile"], args.run_id)
        print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
