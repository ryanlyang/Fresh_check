#!/usr/bin/env python3
"""Build a manifest containing only selected labels from an existing manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.jetclass_data import (  # noqa: E402
    FileRecord,
    JetIdentity,
    SPLIT_ORDER,
    SplitManifest,
    audit_split_manifest,
    load_split_manifest,
    manifest_hash,
    save_split_manifest,
    split_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True, help="Existing split manifest to filter")
    parser.add_argument("--output-manifest", required=True, help="Filtered manifest path to write")
    parser.add_argument(
        "--label-names",
        nargs="+",
        required=True,
        help="Class names to keep, e.g. QCD Hbb. Labels must already be zero-based/contiguous.",
    )
    parser.add_argument("--output-report", default=None, help="Optional JSON report path")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print the output manifest JSON")
    return parser.parse_args()


def _label_ids_for_names(class_names: Sequence[str], label_names: Sequence[str]) -> list[int]:
    by_name = {name: idx for idx, name in enumerate(class_names)}
    missing = [name for name in label_names if name not in by_name]
    if missing:
        raise ValueError(f"Labels are not present in source manifest class_names: {missing}")
    ids = [int(by_name[name]) for name in label_names]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate label names are not allowed: {label_names}")
    expected = list(range(len(ids)))
    if ids != expected:
        raise ValueError(
            "This filtered-manifest builder preserves source label ids, so selected labels must "
            f"already be zero-based/contiguous and in source-label order. Got ids={ids}, expected {expected}."
        )
    return ids


def build_filtered_manifest(
    source_path: str | Path,
    *,
    label_names: Sequence[str],
) -> tuple[SplitManifest, dict]:
    source = load_split_manifest(source_path)
    source_hash = manifest_hash(source)
    label_ids = _label_ids_for_names(source.class_names, label_names)
    keep = set(label_ids)

    filtered_splits: dict[str, list[JetIdentity]] = {}
    before_counts: dict[str, int] = {}
    after_counts: dict[str, int] = {}
    before_class_counts: dict[str, dict[str, int]] = {}
    after_class_counts: dict[str, dict[str, int]] = {}

    for split in SPLIT_ORDER:
        identities = list(source.splits.get(split, []))
        before_counts[split] = len(identities)
        before_by_class = {name: 0 for name in source.class_names}
        after_by_class = {name: 0 for name in label_names}
        filtered: list[JetIdentity] = []
        for identity in identities:
            source_label = int(identity.label)
            if 0 <= source_label < len(source.class_names):
                before_by_class[source.class_names[source_label]] += 1
            if source_label in keep:
                filtered.append(
                    JetIdentity(
                        file=identity.file,
                        entry=int(identity.entry),
                        label=source_label,
                    )
                )
                after_by_class[source.class_names[source_label]] += 1
        filtered_splits[split] = filtered
        after_counts[split] = len(filtered)
        before_class_counts[split] = before_by_class
        after_class_counts[split] = after_by_class

    filtered_records = [
        FileRecord(path=record.path, label=int(record.label), num_entries=int(record.num_entries))
        for record in source.file_records
        if int(record.label) in keep
    ]
    filtered_prefix_map = {
        prefix: int(label)
        for prefix, label in source.file_prefix_to_label.items()
        if int(label) in keep
    }

    metadata = dict(source.metadata)
    metadata.update(
        {
            "label_filter_source_manifest": str(source_path),
            "label_filter_source_manifest_hash": source_hash,
            "label_filter_names": list(label_names),
            "label_filter_ids": label_ids,
            "label_filter_before_counts": before_counts,
            "label_filter_after_counts": after_counts,
            "label_filter_note": (
                "This manifest preserves source label ids and keeps only selected one-to-one jet identities."
            ),
        }
    )

    filtered_manifest = SplitManifest(
        data_dir=source.data_dir,
        max_constits=int(source.max_constits),
        class_names=list(label_names),
        file_prefix_to_label=filtered_prefix_map,
        split_sizes={split: int(after_counts[split]) for split in SPLIT_ORDER},
        split_seeds=dict(source.split_seeds),
        file_records=filtered_records,
        splits=filtered_splits,
        metadata=metadata,
    )
    audit = audit_split_manifest(filtered_manifest)
    if not audit["ok"]:
        raise ValueError(f"Filtered manifest failed audit: {audit}")

    report = {
        "ok": True,
        "source_manifest": str(source_path),
        "source_manifest_hash": source_hash,
        "filtered_manifest_hash": manifest_hash(filtered_manifest),
        "label_names": list(label_names),
        "label_ids": label_ids,
        "before_counts": before_counts,
        "after_counts": after_counts,
        "before_class_counts": before_class_counts,
        "after_class_counts": after_class_counts,
        "summary": split_summary(filtered_manifest),
        "audit": audit,
    }
    return filtered_manifest, report


def main() -> int:
    args = parse_args()
    manifest, report = build_filtered_manifest(args.source_manifest, label_names=args.label_names)
    save_split_manifest(manifest, args.output_manifest, pretty=args.pretty)
    if args.output_report:
        report_path = Path(args.output_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
