#!/usr/bin/env python3
"""Build fresh balanced splits after selecting a subset of JetClass labels.

Unlike ``build_label_filtered_split_manifest.py``, this script does not filter an
existing all-class split. It starts from the ROOT files, keeps only the selected
source classes, remaps them to compact labels 0..N-1, and then applies the split size caps. That means ``--model-train 500000 --label-names QCD Tbqq`` produces
500k binary training jets, balanced across QCD and Tbqq, assuming enough files
are available. In other words, the cap is applied after selecting QCD/Tbqq
(or whatever labels are requested), not before.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jetclass_fresh.jetclass_data import (  # noqa: E402
    DEFAULT_DATA_DIR,
    DEFAULT_SPLIT_SEEDS,
    FILE_PREFIX_TO_LABEL,
    LABEL_NAMES,
    MAX_CONSTITUENTS,
    SPLIT_ORDER,
    FileRecord,
    JetIdentity,
    SplitManifest,
    audit_split_manifest,
    discover_file_records,
    manifest_hash,
    save_split_manifest,
    split_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="JetClass ROOT data directory")
    parser.add_argument("--output-manifest", required=True, help="Output manifest path (.json or .json.gz)")
    parser.add_argument("--label-names", nargs="+", required=True, help="Source JetClass labels to keep")
    parser.add_argument("--output-report", default=None, help="Optional JSON report path")
    parser.add_argument("--pattern", default="*.root", help="ROOT filename glob searched recursively")
    parser.add_argument("--tree-name", default="tree", help="ROOT tree name")
    parser.add_argument("--max-constits", type=int, default=MAX_CONSTITUENTS)
    parser.add_argument("--model-train", type=int, required=True)
    parser.add_argument("--model-val", type=int, required=True)
    parser.add_argument("--stack-train", type=int, required=True)
    parser.add_argument("--stack-val", type=int, required=True)
    parser.add_argument("--final-test", type=int, required=True)
    parser.add_argument("--base-seed", type=int, default=52)
    parser.add_argument("--pretty", action="store_true", help="Pretty-print the output manifest JSON")
    return parser.parse_args()


def _label_ids_for_names(label_names: Sequence[str]) -> list[int]:
    by_name = {name: idx for idx, name in enumerate(LABEL_NAMES)}
    missing = [name for name in label_names if name not in by_name]
    if missing:
        raise ValueError(f"Unknown JetClass labels: {missing}; valid labels are {LABEL_NAMES}")
    ids = [int(by_name[name]) for name in label_names]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate label names are not allowed: {list(label_names)}")
    return ids


def _require_split_sizes(split_sizes: Mapping[str, int], n_classes: int) -> dict[str, int]:
    missing = [split for split in SPLIT_ORDER if split not in split_sizes]
    extra = [split for split in split_sizes if split not in SPLIT_ORDER]
    if missing or extra:
        raise ValueError(f"split_sizes must contain exactly {SPLIT_ORDER}; missing={missing}, extra={extra}")
    clean: dict[str, int] = {}
    for split in SPLIT_ORDER:
        total = int(split_sizes[split])
        if total <= 0:
            raise ValueError(f"{split} size must be positive, got {total}")
        if total % n_classes != 0:
            raise ValueError(
                f"{split} size {total} is not divisible by selected class count {n_classes}"
            )
        clean[split] = total
    return clean


def build_label_filtered_fresh_manifest(
    *,
    data_dir: str | Path,
    label_names: Sequence[str],
    split_sizes: Mapping[str, int],
    pattern: str = "*.root",
    tree_name: str = "tree",
    max_constits: int = MAX_CONSTITUENTS,
    base_seed: int = 52,
) -> tuple[SplitManifest, dict]:
    source_label_ids = _label_ids_for_names(label_names)
    n_classes = len(source_label_ids)
    clean_split_sizes = _require_split_sizes(split_sizes, n_classes)
    source_to_filtered = {int(source): int(index) for index, source in enumerate(source_label_ids)}
    filtered_to_source = {int(index): int(source) for index, source in enumerate(source_label_ids)}

    all_records = discover_file_records(
        data_dir,
        pattern=pattern,
        tree_name=tree_name,
        require_all_classes=False,
    )
    before_counts_by_source = {name: 0 for name in LABEL_NAMES}
    for record in all_records:
        if 0 <= int(record.label) < len(LABEL_NAMES):
            before_counts_by_source[LABEL_NAMES[int(record.label)]] += int(record.num_entries)

    selected_source_ids = set(source_label_ids)
    selected_records: list[FileRecord] = []
    for record in all_records:
        source_label = int(record.label)
        if source_label not in selected_source_ids:
            continue
        selected_records.append(
            FileRecord(
                path=record.path,
                label=source_to_filtered[source_label],
                num_entries=int(record.num_entries),
            )
        )

    records_by_filtered_label: dict[int, list[FileRecord]] = defaultdict(list)
    for record in sorted(selected_records, key=lambda item: (item.label, item.path)):
        if record.num_entries <= 0:
            raise ValueError(f"File has no entries: {record}")
        records_by_filtered_label[int(record.label)].append(record)

    missing = [label_names[i] for i in range(n_classes) if i not in records_by_filtered_label]
    if missing:
        raise ValueError(f"No file records found for selected labels: {missing}")

    per_class_counts = {split: int(clean_split_sizes[split]) // n_classes for split in SPLIT_ORDER}
    requested_per_class_total = int(sum(per_class_counts.values()))
    available_counts_by_filtered = {
        label_names[filtered_label]: int(sum(record.num_entries for record in records))
        for filtered_label, records in sorted(records_by_filtered_label.items())
    }
    short = {
        name: {"available": int(available), "requested": requested_per_class_total}
        for name, available in available_counts_by_filtered.items()
        if int(available) < requested_per_class_total
    }
    if short:
        raise ValueError(
            "Not enough jets for requested label-filtered split sizes. "
            f"Each selected class needs {requested_per_class_total}; short_classes={short}"
        )

    split_seeds = dict(DEFAULT_SPLIT_SEEDS)
    splits: dict[str, list[JetIdentity]] = {split: [] for split in SPLIT_ORDER}
    selected_entry_counts: dict[str, dict[str, int]] = {
        split: {name: 0 for name in label_names} for split in SPLIT_ORDER
    }

    for filtered_label in range(n_classes):
        records = records_by_filtered_label[filtered_label]
        counts = np.array([record.num_entries for record in records], dtype=np.int64)
        cumulative = np.cumsum(counts)
        total_available = int(cumulative[-1])
        remaining = np.arange(total_available, dtype=np.int64)
        for split in SPLIT_ORDER:
            count = int(per_class_counts[split])
            source_label = filtered_to_source[filtered_label]
            rng = np.random.RandomState(int(split_seeds[split]) + source_label * 100_003)
            chosen_positions = rng.choice(len(remaining), size=count, replace=False)
            chosen_global = remaining[chosen_positions]
            remaining = np.delete(remaining, chosen_positions)

            file_indices = np.searchsorted(cumulative, chosen_global, side="right")
            file_starts = np.concatenate(([0], cumulative[:-1]))
            for global_index, file_index in zip(chosen_global, file_indices):
                record = records[int(file_index)]
                entry = int(global_index - file_starts[int(file_index)])
                splits[split].append(
                    JetIdentity(file=record.path, entry=entry, label=filtered_label)
                )
            selected_entry_counts[split][label_names[filtered_label]] = count

    filtered_prefix_map = {
        prefix: source_to_filtered[int(label)]
        for prefix, label in FILE_PREFIX_TO_LABEL.items()
        if int(label) in selected_source_ids
    }
    metadata = {
        "base_seed": int(base_seed),
        "sampling": "label_filtered_balanced_classwise_sequential_without_replacement",
        "split_size_semantics": "requested split sizes are applied after label filtering/remapping",
        "source_label_names": list(label_names),
        "source_label_ids": [int(x) for x in source_label_ids],
        "source_to_filtered_label": {
            str(source): int(filtered) for source, filtered in source_to_filtered.items()
        },
        "filtered_to_source_label": {
            str(filtered): int(source) for filtered, source in filtered_to_source.items()
        },
        "selected_class_names": list(label_names),
        "jet_identity": "relative_source_file_path_plus_entry_index",
        "file_level_separation_claimed": False,
        "notes": (
            "Compact labels are 0..N-1 in --label-names order. This is a fresh binary/multiclass "
            "split, not a filtered view of an all-class manifest."
        ),
    }
    manifest = SplitManifest(
        data_dir=str(data_dir),
        max_constits=int(max_constits),
        class_names=list(label_names),
        file_prefix_to_label=filtered_prefix_map,
        split_sizes={split: int(clean_split_sizes[split]) for split in SPLIT_ORDER},
        split_seeds={split: int(split_seeds[split]) for split in SPLIT_ORDER},
        file_records=selected_records,
        splits=splits,
        metadata=metadata,
    )
    audit = audit_split_manifest(manifest)
    if not audit["ok"]:
        raise ValueError(f"Generated invalid label-filtered manifest: {audit}")

    report = {
        "ok": True,
        "mode": "fresh_label_filtered_splits",
        "data_dir": str(data_dir),
        "label_names": list(label_names),
        "source_label_ids": [int(x) for x in source_label_ids],
        "source_to_filtered_label": source_to_filtered,
        "filtered_to_source_label": filtered_to_source,
        "split_size_semantics": "after_label_filtering",
        "requested_split_sizes": {split: int(clean_split_sizes[split]) for split in SPLIT_ORDER},
        "requested_per_class_by_split": dict(per_class_counts),
        "requested_per_class_total": requested_per_class_total,
        "available_counts_all_source_labels": before_counts_by_source,
        "available_counts_selected_labels": available_counts_by_filtered,
        "selected_file_records": len(selected_records),
        "selected_entry_counts_by_split": selected_entry_counts,
        "summary": split_summary(manifest),
        "audit": audit,
        "manifest_hash": manifest_hash(manifest),
    }
    return manifest, report


def main() -> int:
    args = parse_args()
    split_sizes = {
        "model_train": args.model_train,
        "model_val": args.model_val,
        "stack_train": args.stack_train,
        "stack_val": args.stack_val,
        "final_test": args.final_test,
    }
    manifest, report = build_label_filtered_fresh_manifest(
        data_dir=args.data_dir,
        label_names=args.label_names,
        split_sizes=split_sizes,
        pattern=args.pattern,
        tree_name=args.tree_name,
        max_constits=args.max_constits,
        base_seed=args.base_seed,
    )
    save_split_manifest(manifest, args.output_manifest, pretty=args.pretty)
    if args.output_report:
        report_path = Path(args.output_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
