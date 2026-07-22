#!/usr/bin/env python3
"""Revalidate a frozen LPRF fusion source audit before dependent work."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.local_particle_residual_field import require_fusion_source_artifact_audit  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True)
    args = parser.parse_args(argv)
    report = require_fusion_source_artifact_audit(args.audit)
    print(json.dumps({"ok": True, "audit": str(Path(args.audit).resolve()), "audit_hash": report["audit_hash"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
