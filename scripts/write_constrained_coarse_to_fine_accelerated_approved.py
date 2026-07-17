#!/usr/bin/env python3
"""Write one immutable C2F accelerated_approved_v1 promotion closure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.constrained_coarse_to_fine.runtime_profiles import write_approved_profile  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-profile", required=True)
    parser.add_argument("--c5-ten-epoch-certification", required=True)
    parser.add_argument("--c6-ten-epoch-certification", required=True)
    parser.add_argument("--c5-fp32-reference", required=True)
    parser.add_argument("--c6-fp32-reference", required=True)
    parser.add_argument("--c5-tagger-sanity", required=True)
    parser.add_argument("--c6-tagger-sanity", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--hlt-cache-dir", required=True)
    parser.add_argument("--offline-cache-dir", required=True)
    parser.add_argument("--target-cache-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    approved = write_approved_profile(
        candidate_profile_path=args.candidate_profile,
        c5_ten_epoch_certification=args.c5_ten_epoch_certification,
        c6_ten_epoch_certification=args.c6_ten_epoch_certification,
        c5_fp32_reference=args.c5_fp32_reference,
        c6_fp32_reference=args.c6_fp32_reference,
        c5_tagger_sanity=args.c5_tagger_sanity,
        c6_tagger_sanity=args.c6_tagger_sanity,
        manifest_path=args.manifest,
        hlt_cache_dir=args.hlt_cache_dir,
        offline_cache_dir=args.offline_cache_dir,
        target_cache_dir=args.target_cache_dir,
        output_path=args.output,
    )
    print(json.dumps({"ok": True, "output": args.output, "approved_profile_hash": approved["approved_profile_hash"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
