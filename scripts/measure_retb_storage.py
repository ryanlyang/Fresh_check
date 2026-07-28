#!/usr/bin/env python3
"""Create a source-evidence-bound RETB storage measurement artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teacher_logit_reco.relation_expert_token_bridge.contracts import (  # noqa: E402
    write_immutable_json,
)
from teacher_logit_reco.relation_expert_token_bridge.storage import (  # noqa: E402
    build_storage_measurements,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--evidence",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Repeat for each representative source-evidence file.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    measurements = json.loads(args.measurements_json.read_text(encoding="utf-8"))
    if not isinstance(measurements, dict):
        raise ValueError("measurements JSON must be an object")
    source_evidence: dict[str, dict[str, object]] = {}
    for raw in args.evidence:
        name, separator, path_text = raw.partition("=")
        if not separator or not name:
            raise ValueError("--evidence must use NAME=PATH")
        path = Path(path_text)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"evidence file is absent or unsafe: {path}")
        source_evidence[name] = {
            "path": str(path.resolve()),
            "sha256": _sha256_file(path),
            "bytes": int(path.stat().st_size),
            "purpose": name,
        }
    artifact = build_storage_measurements(
        measurements=measurements,
        source_evidence=source_evidence,
        measurement_profile="production_source_evidence",
    )
    result: dict[str, object] = {
        "content_hash": artifact["content_hash"],
        "evidence_hashes": artifact["evidence_hashes"],
        "dry_run": bool(args.dry_run),
    }
    if not args.dry_run:
        result["publication"] = write_immutable_json(args.output, artifact)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
