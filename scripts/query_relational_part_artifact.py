#!/usr/bin/env python3
"""Print one dotted field from a JSON or gzipped JSON artifact."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("field")
    parser.add_argument("--length", action="store_true")
    args = parser.parse_args()
    opener = gzip.open if args.path.suffix == ".gz" else open
    with opener(args.path, "rt", encoding="utf-8") as stream:
        value: Any = json.load(stream)
    for part in args.field.split("."):
        if isinstance(value, list):
            value = value[int(part)]
        else:
            value = value[part]
    if args.length:
        value = len(value)
    if isinstance(value, bool):
        print("true" if value else "false")
    elif value is None:
        print("null")
    elif isinstance(value, (dict, list)):
        print(json.dumps(value, separators=(",", ":"), sort_keys=True))
    else:
        print(value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
