#!/usr/bin/env python3
"""Validate an exact bound teacher-logit cache and its live-side identity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.local_particle_residual_field.bridge_contracts import (  # noqa: E402
    load_hashed_json,
)
from teacher_logit_reco.local_particle_residual_field.bridge_logits import (  # noqa: E402
    build_live_teacher_config,
    load_teacher_logit_cache,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", required=True)
    parser.add_argument("--namespace-dir", required=True)
    parser.add_argument("--stack-train-distill-sha256", required=True)
    parser.add_argument("--selected-consumer", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    binding = load_hashed_json(args.binding)
    selected = load_hashed_json(args.selected_consumer) if args.selected_consumer else None
    live = build_live_teacher_config(binding, primary_selection=selected)
    manifest, arrays = load_teacher_logit_cache(
        args.namespace_dir,
        binding=binding,
        live_teacher_config=live,
        stack_train_distill_manifest_sha256=args.stack_train_distill_sha256,
        primary_selection=selected,
    )
    output = {
        "ok": True,
        "manifest_sha256": manifest["content_hash"],
        "binding_sha256": binding["content_hash"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "live_checkpoint_sha256": manifest["live_checkpoint_sha256"],
        "cache_namespace": manifest["cache_namespace"],
        "event_count": int(arrays["logits"].shape[0]),
        "class_count": int(arrays["logits"].shape[1]),
        "same_checkpoint_target_and_live": manifest["same_checkpoint_target_and_live"],
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

