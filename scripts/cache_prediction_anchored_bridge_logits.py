#!/usr/bin/env python3
"""Numerically generate one binding-locked compact B5 teacher-logit cache."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.local_particle_residual_field.bridge_contracts import (  # noqa: E402
    load_hashed_json,
)
from teacher_logit_reco.local_particle_residual_field.bridge_evaluation import (  # noqa: E402
    N3_F0_TEACHER_NAMESPACE,
)
from teacher_logit_reco.local_particle_residual_field.bridge_logits import (  # noqa: E402
    ALLOWED_NAMESPACES,
    build_live_teacher_config,
)
from teacher_logit_reco.local_particle_residual_field.bridge_teacher_execution import (  # noqa: E402
    cache_teacher_logits_from_execution_spec,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-spec", required=True)
    parser.add_argument("--binding", required=True)
    parser.add_argument("--namespace", choices=ALLOWED_NAMESPACES, required=True)
    parser.add_argument("--selected-consumer", default="")
    parser.add_argument("--r0-checkpoint", required=True)
    parser.add_argument("--r0-registration", required=True)
    parser.add_argument("--bridge-recipe", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--ram-root", required=True)
    parser.add_argument(
        "--allocation-id", default=os.environ.get("SLURM_JOB_ID", "local_teacher_cache")
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--shard-size", type=int, default=8192)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--test-capacity-bytes", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--allow-unverified-test-root", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    binding = load_hashed_json(args.binding)
    selected = load_hashed_json(args.selected_consumer) if args.selected_consumer else None
    if binding.get("binding_kind") == "primary" and selected is None:
        raise PermissionError("primary/N3 caching requires --selected-consumer")
    if binding.get("binding_kind") != "primary" and selected is not None:
        raise ValueError("non-primary cache must not receive the primary selection")
    if args.namespace == N3_F0_TEACHER_NAMESPACE and binding.get("binding_kind") != "primary":
        raise ValueError("N3 must use the selected primary teacher")
    live = build_live_teacher_config(binding, primary_selection=selected)
    if args.dry_run:
        result = {
            "ok": True,
            "dry_run": True,
            "namespace": args.namespace,
            "binding_sha256": binding["content_hash"],
            "checkpoint_sha256": binding["checkpoint_sha256"],
            "live_teacher_config_sha256": live["content_hash"],
            "same_checkpoint_target_and_live_required": True,
            "source_child": "stack_train_distill",
            "persistent_arrays": ["logits", "labels", "event_identity_hashes"],
            "persistent_dense_fields_written": False,
        }
    else:
        result = cache_teacher_logits_from_execution_spec(
            args.execution_spec,
            binding_path=args.binding,
            namespace=args.namespace,
            selected_consumer_path=args.selected_consumer or None,
            r0_checkpoint_path=args.r0_checkpoint,
            r0_registration_path=args.r0_registration,
            bridge_recipe_path=args.bridge_recipe,
            output_root=args.output_root,
            ram_root=args.ram_root,
            allocation_id=args.allocation_id,
            device=args.device,
            batch_size=int(args.batch_size),
            shard_size=int(args.shard_size),
            capacity_bytes=(int(args.test_capacity_bytes) or None),
            allow_unverified_test_root=bool(args.allow_unverified_test_root),
        )
        result["dry_run"] = False
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
