#!/usr/bin/env python3
"""Render deployable HLT-only pseudo-particle caches from one selected C-tier model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.constrained_coarse_to_fine import (  # noqa: E402
    PSEUDO_PARTICLE_DEFAULT_SPLITS,
    PseudoParticleCacheConfig,
    audit_pseudo_particle_cache,
    cache_pseudo_particle_views,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-cache-dir", required=True)
    parser.add_argument("--manifest", dest="manifest_path", required=True)
    parser.add_argument("--hlt-cache-dir", required=True)
    parser.add_argument("--reconstructor-checkpoint", required=True)
    parser.add_argument("--splits", nargs="+", default=list(PSEUDO_PARTICLE_DEFAULT_SPLITS))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--shard-size", type=int, default=1024)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--num-views", type=int, default=None)
    parser.add_argument("--min-particle-pt", type=float, default=0.0)
    parser.add_argument("--dust-reliability", type=float, default=0.10)
    parser.add_argument("--max-jets-per-split", type=int, default=None)
    parser.add_argument("--seed", type=int, default=26061)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-verify-hlt-hash", action="store_true")
    parser.add_argument("--non-strict-checkpoint", action="store_true")
    parser.add_argument(
        "--allow-nonselected-checkpoint",
        action="store_true",
        help="Diagnostic only: allow a checkpoint not marked best_model_val.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--confirm-final-test", action="store_true")
    parser.add_argument("--no-audit", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = vars(build_parser().parse_args(argv))
    audit_enabled = not args.pop("no_audit")
    args["amp"] = not args.pop("no_amp")
    args["verify_hlt_hash"] = not args.pop("no_verify_hlt_hash")
    args["strict_checkpoint"] = not args.pop("non_strict_checkpoint")
    args["require_model_val_checkpoint"] = not args.pop("allow_nonselected_checkpoint")
    args["skip_existing"] = not args.pop("no_skip_existing")
    config = PseudoParticleCacheConfig(**args)
    report = cache_pseudo_particle_views(config)
    audits = {}
    if audit_enabled:
        for split in config.splits:
            audits[split] = audit_pseudo_particle_cache(
                config.output_cache_dir,
                variant=report["variant"],
                split=split,
                manifest_path=config.manifest_path,
                hlt_cache_dir=config.hlt_cache_dir,
                checkpoint_path=config.reconstructor_checkpoint,
                verify_hash=config.verify_hlt_hash,
            )
    result = {
        **report,
        "audits": audits,
        "ok": bool(report.get("ok")) and all(row.get("ok") for row in audits.values()),
    }
    result_path = Path(config.output_cache_dir) / report["variant"] / "cache_and_audit_report.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result["audit_report_path"] = str(result_path)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
