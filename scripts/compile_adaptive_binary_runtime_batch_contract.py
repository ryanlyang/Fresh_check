#!/usr/bin/env python3
"""Compile full-step candidate measurements into one production batch contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.adaptive_binary_pseudooffline import (  # noqa: E402
    AdaptiveBinaryTargetBatchSource,
    FullStepBatchMeasurement,
    calibrate_runtime_batch_contract,
    canonical_hash,
    reconstructor_runtime_provenance,
    resolve_variant_config,
    write_runtime_batch_contract,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--measurement-dir", required=True)
    parser.add_argument("--requested-world-size", type=int, required=True)
    parser.add_argument("--output", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(args.campaign_root)
    resolved = resolve_variant_config(args.variant)
    grouping = str(resolved["model"]["hierarchy"].get("grouping", "exclusive_kt"))
    source = AdaptiveBinaryTargetBatchSource(
        hlt_cache_dir=root / "inputs" / "hlt_cache",
        target_cache_dir=root / "targets",
        split="model_train",
        grouping=grouping,
        batch_size=1,
        shuffle_shards=False,
        seed=24731,
        maximum_batches=1,
    )
    provenance = reconstructor_runtime_provenance(
        variant_name=args.variant,
        target_metadata=source.metadata,
        hlt_metadata=source.hlt_view.metadata,
    )
    measurement_dir = Path(args.measurement_dir)

    def probe(family: str, local_batch_size: int, accumulation_steps: int):
        path = measurement_dir / f"{family}_b{local_batch_size}.json"
        if not path.is_file():
            raise FileNotFoundError(f"candidate measurement is missing: {path}")
        measurement = FullStepBatchMeasurement.from_dict(
            json.loads(path.read_text(encoding="utf-8"))
        )
        if measurement.accumulation_steps != accumulation_steps:
            raise ValueError(f"candidate measurement accumulation mismatch: {path}")
        return measurement

    contract = calibrate_runtime_batch_contract(
        variant_name=args.variant,
        resolved_variant_config_hash=resolved["resolved_config_hash"],
        runtime_provenance_hash=canonical_hash(provenance),
        requested_world_size=int(args.requested_world_size),
        probe=probe,
    )
    output = Path(args.output) if args.output else (
        root / "runtime_batch_contracts" / args.variant / "runtime_batch_contract.json"
    )
    write_runtime_batch_contract(output, contract)
    print(json.dumps({"ok": True, "output": str(output), **contract.to_dict()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
