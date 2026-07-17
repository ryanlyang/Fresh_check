#!/usr/bin/env python3
"""Write one fail-closed C2F 10/30-epoch reconstructor certification report."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.constrained_coarse_to_fine.runtime_certification import write_reconstructor_certification


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True, choices=("C5-B3", "C6"))
    parser.add_argument("--mode", required=True, choices=("ten_epoch_certification", "fp32_reference_promotion"))
    parser.add_argument("--candidate-profile", required=True)
    parser.add_argument("--candidate-run-dir", required=True)
    parser.add_argument("--fp32-reference-run-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = write_reconstructor_certification(
        path=args.path, mode=args.mode, candidate_profile_path=args.candidate_profile,
        candidate_run_dir=args.candidate_run_dir, fp32_reference_run_dir=args.fp32_reference_run_dir,
        output_path=args.output,
    )
    print(f"ok={str(report['ok']).lower()} output={args.output}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
