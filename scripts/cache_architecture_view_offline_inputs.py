#!/usr/bin/env python3
"""Cache offline JetClass views for AV10 offline-transfer runs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.jetclass_data import SPLIT_ORDER, load_offline_view, load_split_manifest  # noqa: E402
from jetclass_fresh.hlt_baseline import save_json  # noqa: E402
from teacher_logit_reco.architecture_view_part import save_cached_offline_view  # noqa: E402
from teacher_logit_reco.set_matching.train import source_metadata  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--splits", nargs="+", default=list(SPLIT_ORDER), choices=list(SPLIT_ORDER))
    parser.add_argument(
        "--data-dir",
        nargs="*",
        default=None,
        help="Optional override data directory/directories for the manifest ROOT files.",
    )
    parser.add_argument("--tree-name", default="tree")
    parser.add_argument("--max-constits", type=int, default=None)
    parser.add_argument("--read-chunk-size", type=int, default=50_000)
    parser.add_argument("--verify-label-branches", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _coerce_data_dir(values: list[str] | None):
    if values is None:
        return None
    if len(values) == 0:
        return None
    if len(values) == 1:
        return values[0]
    return list(values)


def main() -> int:
    args = parse_args()
    manifest = load_split_manifest(args.manifest_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = _coerce_data_dir(args.data_dir)
    split_reports = {}
    for split in args.splits:
        view = load_offline_view(
            manifest,
            split,
            data_dir=data_dir,
            tree_name=args.tree_name,
            max_constits=args.max_constits,
            verify_label_branches=bool(args.verify_label_branches),
            read_chunk_size=int(args.read_chunk_size),
        )
        metadata = save_cached_offline_view(view, output_dir, overwrite=bool(args.overwrite))
        split_reports[split] = {
            "n_jets": int(view.tokens.shape[0]),
            "offline_content_hash": metadata.get("offline_content_hash"),
            "jet_identity_hash": metadata.get("jet_identity_hash"),
            "array_path": metadata.get("array_path"),
            "metadata_path": metadata.get("metadata_path"),
        }

    report = {
        "task": "architecture_view_10class_offline_input_cache",
        "contract": "architecture_view_10class_offline_transfer_cache_v1",
        "manifest_path": str(args.manifest_path),
        "output_dir": str(output_dir),
        "splits": split_reports,
        "data_dir_override": data_dir,
        "source": source_metadata(),
    }
    save_json(output_dir / "offline_cache_report.json", report)
    print("architecture_view_offline_cache_complete:")
    print(f"  output_dir: {output_dir}")
    for split, split_report in split_reports.items():
        print(f"  {split}: n_jets={split_report['n_jets']} hash={split_report['offline_content_hash']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
