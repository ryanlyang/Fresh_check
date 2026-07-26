#!/usr/bin/env python3
"""Verify source ROOT capacity for every class before Slurm submission."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.jetclass_data import (  # noqa: E402
    LABEL_NAMES,
    discover_file_records,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", nargs="+", type=Path, required=True)
    parser.add_argument("--tree-name", default="tree")
    parser.add_argument("--pattern", default="*.root")
    parser.add_argument("--required-per-class", type=int, required=True)
    args = parser.parse_args(argv)
    records = discover_file_records(
        args.data_dir,
        pattern=args.pattern,
        tree_name=args.tree_name,
        require_all_classes=True,
        skip_unreadable=False,
    )
    counts = Counter()
    files = Counter()
    for record in records:
        counts[int(record.label)] += int(record.num_entries)
        files[int(record.label)] += 1
    required = int(args.required_per_class)
    report = {
        "data_roots": [str(path.resolve()) for path in args.data_dir],
        "tree_name": args.tree_name,
        "required_events_per_class": required,
        "classes": {
            LABEL_NAMES[label]: {
                "file_count": int(files[label]),
                "available_events": int(counts[label]),
                "enough": int(counts[label]) >= required,
            }
            for label in range(len(LABEL_NAMES))
        },
    }
    report["ok"] = all(row["enough"] for row in report["classes"].values())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
