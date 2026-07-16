"""Audit ABPH split, HLT, and offline input contracts before training."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.adaptive_binary_pseudooffline import (  # noqa: E402
    AdaptiveBinaryExperimentLayout,
    AdaptiveBinaryInputAuditConfig,
    audit_input_contract,
    write_input_audit_report,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    defaults = AdaptiveBinaryExperimentLayout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(defaults.manifest_path))
    parser.add_argument("--hlt-cache-dir", default=str(defaults.hlt_cache_dir))
    parser.add_argument("--offline-cache-dir", default=str(defaults.offline_cache_dir))
    parser.add_argument("--campaign-mode", choices=("pilot", "highdata"), default="highdata")
    parser.add_argument("--hlt-only", action="store_true", help="Audit baseline inputs without offline caches")
    parser.add_argument("--output", default=str(defaults.audit_dir / "step1_input_audit.json"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = audit_input_contract(
        AdaptiveBinaryInputAuditConfig(
            manifest_path=args.manifest,
            hlt_cache_dir=args.hlt_cache_dir,
            offline_cache_dir=None if args.hlt_only else args.offline_cache_dir,
            campaign_mode=args.campaign_mode,
            require_offline=not bool(args.hlt_only),
        )
    )
    path = write_input_audit_report(report, Path(args.output))
    print(f"wrote {path}")
    print(f"ok={bool(report.get('ok'))}")
    return 0 if bool(report.get("ok")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
