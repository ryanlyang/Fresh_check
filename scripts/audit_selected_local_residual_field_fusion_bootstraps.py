#!/usr/bin/env python3
"""Verify locked paired bootstrap artifacts after final evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.local_particle_residual_field.fusion_bootstrap_audit import audit_selected_fusion_bootstraps  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-fusion", required=True)
    report = audit_selected_fusion_bootstraps(parser.parse_args(argv).selected_fusion)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
