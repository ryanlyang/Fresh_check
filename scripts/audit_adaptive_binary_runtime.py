#!/usr/bin/env python3
"""Plan or execute the fixed Step-1 ABPH runtime reference benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


REFERENCE_CONTRACT = "adaptive_binary_pseudooffline_runtime_reference_v1"
REFERENCE_VARIANTS = (
    "B1_semantic_query_root",
    "D1_kt32_mh4_particles",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--execute", action="store_true")
    return parser


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _benchmark_command(
    *,
    python: str,
    campaign_root: Path,
    output_dir: Path,
    variant: str,
    device: str,
    batch_size: int,
) -> list[str]:
    return [
        str(python),
        "scripts/train_adaptive_binary_pseudooffline_variant.py",
        "--variant",
        variant,
        "--campaign-root",
        str(campaign_root),
        "--output-dir",
        str(output_dir),
        "--device",
        str(device),
        "--batch-size",
        str(int(batch_size)),
        "--maximum-updates",
        "20",
        "--runtime-reference-benchmark",
    ]


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(args.campaign_root).resolve()
    audit_root = root / "audits" / "runtime_reference"
    entries: list[dict[str, Any]] = []
    for variant in REFERENCE_VARIANTS:
        output_dir = audit_root / variant
        command = _benchmark_command(
            python=args.python,
            campaign_root=root,
            output_dir=output_dir,
            variant=variant,
            device=args.device,
            batch_size=args.batch_size,
        )
        environment = {
            "ABPH_RUNTIME_PROFILE_ENABLED": "1",
            "ABPH_RUNTIME_PROFILE_WARMUP_UPDATES": "5",
            "ABPH_RUNTIME_PROFILE_INTERVAL": "1",
        }
        # D1 inherits its root/hierarchy state. One renderer update moves the
        # reference window into its distribution-heavy production stage.
        if variant == "D1_kt32_mh4_particles":
            environment["ABPH_RENDERER_UPDATES"] = "1"
        entries.append(
            {
                "variant": variant,
                "updates": 20,
                "seed_index": 1,
                "output_dir": str(output_dir),
                "command": command,
                "environment": environment,
            }
        )

    plan: dict[str, Any] = {
        "contract": REFERENCE_CONTRACT,
        "campaign_root": str(root),
        "fixed_train_source_seed": 24731,
        "fixed_batch_order": True,
        "complete_model_val_rollout_required": True,
        "entries": entries,
    }
    plan["plan_hash"] = _canonical_hash(plan)
    _atomic_json(audit_root / "benchmark_plan.json", plan)

    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    for entry in entries:
        environment = os.environ.copy()
        environment.update(entry["environment"])
        completed = subprocess.run(
            entry["command"],
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            check=False,
        )
        if completed.returncode != 0:
            raise SystemExit(
                f"runtime benchmark {entry['variant']} failed with "
                f"exit code {completed.returncode}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
