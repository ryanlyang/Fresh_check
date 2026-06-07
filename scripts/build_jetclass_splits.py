#!/usr/bin/env python3
"""Build the Step 2 JetClass five-way split manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh import (  # noqa: E402
    DEFAULT_DATA_DIR,
    DEFAULT_SPLIT_SEEDS,
    DEFAULT_SPLIT_TOTALS,
    audit_split_manifest,
    build_split_manifest_from_records,
    discover_file_records,
    save_split_manifest,
    split_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="JetClass ROOT data directory")
    parser.add_argument(
        "--out",
        default="checkpoints/jetclass_fresh_splits/split_manifest.json.gz",
        help="Output manifest path (.json or .json.gz)",
    )
    parser.add_argument("--pattern", default="*.root", help="ROOT filename glob searched recursively")
    parser.add_argument("--tree-name", default="tree", help="ROOT tree name")
    parser.add_argument("--max-constits", type=int, default=128)
    parser.add_argument("--model-train", type=int, default=DEFAULT_SPLIT_TOTALS["model_train"])
    parser.add_argument("--model-val", type=int, default=DEFAULT_SPLIT_TOTALS["model_val"])
    parser.add_argument("--stack-train", type=int, default=DEFAULT_SPLIT_TOTALS["stack_train"])
    parser.add_argument("--stack-val", type=int, default=DEFAULT_SPLIT_TOTALS["stack_val"])
    parser.add_argument("--final-test", type=int, default=DEFAULT_SPLIT_TOTALS["final_test"])
    parser.add_argument("--pretty", action="store_true", help="Pretty-print the JSON manifest")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    split_sizes = {
        "model_train": args.model_train,
        "model_val": args.model_val,
        "stack_train": args.stack_train,
        "stack_val": args.stack_val,
        "final_test": args.final_test,
    }

    records = discover_file_records(
        args.data_dir,
        pattern=args.pattern,
        tree_name=args.tree_name,
        require_all_classes=True,
    )
    manifest = build_split_manifest_from_records(
        records,
        data_dir=args.data_dir,
        split_sizes=split_sizes,
        split_seeds=DEFAULT_SPLIT_SEEDS,
        max_constits=args.max_constits,
    )
    audit = audit_split_manifest(manifest)
    if not audit["ok"]:
        print(json.dumps(audit, indent=2, sort_keys=True), file=sys.stderr)
        return 2

    save_split_manifest(manifest, args.out, pretty=args.pretty)
    report = {
        "manifest_path": str(Path(args.out)),
        "summary": split_summary(manifest),
        "audit": audit,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
