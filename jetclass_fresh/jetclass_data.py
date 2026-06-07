"""JetClass loading and five-way split management for the fresh replication.

This module deliberately stops at Step 2 of the protocol: it discovers ROOT
files, creates deterministic non-overlapping jet partitions, saves split
metadata, and loads offline raw-token views. HLT generation and model training
belong to later steps.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

import numpy as np


DEFAULT_DATA_DIR = "/home/ryreu/atlas/PracticeTagging/data/jetclass_part0"
MAX_CONSTITUENTS = 128
RAW_TOKEN_DIM = 14

LABEL_NAMES = [
    "QCD",
    "Hbb",
    "Hcc",
    "Hgg",
    "H4q",
    "Hqql",
    "Zqq",
    "Wqq",
    "Tbqq",
    "Tbl",
]

FILE_PREFIX_TO_LABEL = {
    "ZJetsToNuNu": 0,
    "HToBB": 1,
    "HToCC": 2,
    "HToGG": 3,
    "HToWW4Q": 4,
    "HToWW2Q1L": 5,
    "ZToQQ": 6,
    "WToQQ": 7,
    "TTBar": 8,
    "TTBarLep": 9,
}

LABEL_PREFIXES = [
    ("ZJetsToNuNu", 0),
    ("HToBB", 1),
    ("HToCC", 2),
    ("HToGG", 3),
    ("HToWW4Q", 4),
    ("HToWW2Q1L", 5),
    ("ZToQQ", 6),
    ("WToQQ", 7),
    ("TTBar", 8),
    ("TTBarLep", 9),
]

SPLIT_ORDER = [
    "model_train",
    "model_val",
    "stack_train",
    "stack_val",
    "final_test",
]

DEFAULT_SPLIT_TOTALS = {
    "model_train": 500_000,
    "model_val": 150_000,
    "stack_train": 250_000,
    "stack_val": 50_000,
    "final_test": 500_000,
}

DEFAULT_SPLIT_SEEDS = {
    "model_train": 153,
    "model_val": 254,
    "stack_train": 356,
    "stack_val": 457,
    "final_test": 558,
}

PARTICLE_READ_BRANCHES = [
    "part_px",
    "part_py",
    "part_pz",
    "part_energy",
    "part_charge",
    "part_isChargedHadron",
    "part_isNeutralHadron",
    "part_isPhoton",
    "part_isElectron",
    "part_isMuon",
    "part_d0val",
    "part_d0err",
    "part_dzval",
    "part_dzerr",
]

LABEL_BRANCHES = [
    "label_QCD",
    "label_Hbb",
    "label_Hcc",
    "label_Hgg",
    "label_H4q",
    "label_Hqql",
    "label_Zqq",
    "label_Wqq",
    "label_Tbqq",
    "label_Tbl",
]

PID_TOKEN_BRANCHES = [
    "part_isChargedHadron",
    "part_isNeutralHadron",
    "part_isPhoton",
    "part_isElectron",
    "part_isMuon",
]

TRACK_TOKEN_BRANCHES = [
    "part_d0val",
    "part_d0err",
    "part_dzval",
    "part_dzerr",
]


@dataclass(frozen=True)
class FileRecord:
    """A ROOT file and the class/entry count inferred for it."""

    path: str
    label: int
    num_entries: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "label": int(self.label),
            "label_name": LABEL_NAMES[int(self.label)],
            "num_entries": int(self.num_entries),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FileRecord":
        return cls(
            path=str(data["path"]),
            label=int(data["label"]),
            num_entries=int(data["num_entries"]),
        )


@dataclass(frozen=True)
class JetIdentity:
    """Stable jet identity used for split and leakage audits."""

    file: str
    entry: int
    label: int

    def key(self) -> str:
        return f"{self.file}#{int(self.entry)}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file": self.file,
            "entry": int(self.entry),
            "label": int(self.label),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "JetIdentity":
        return cls(
            file=str(data["file"]),
            entry=int(data["entry"]),
            label=int(data["label"]),
        )


@dataclass
class JetView:
    """Offline raw-token view returned by the Step 2 loader."""

    tokens: np.ndarray
    mask: np.ndarray
    labels: np.ndarray
    jet_ids: List[JetIdentity]
    split: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.tokens.ndim != 3 or self.tokens.shape[-1] != RAW_TOKEN_DIM:
            raise ValueError(
                f"tokens must have shape [N, max_constits, {RAW_TOKEN_DIM}], "
                f"got {self.tokens.shape}"
            )
        if self.mask.shape != self.tokens.shape[:2]:
            raise ValueError(f"mask shape {self.mask.shape} does not match tokens {self.tokens.shape[:2]}")
        if len(self.labels) != self.tokens.shape[0]:
            raise ValueError("labels length does not match number of jets")
        if len(self.jet_ids) != self.tokens.shape[0]:
            raise ValueError("jet_ids length does not match number of jets")


@dataclass
class SplitManifest:
    """Deterministic five-way split manifest."""

    data_dir: str
    max_constits: int
    class_names: List[str]
    file_prefix_to_label: Dict[str, int]
    split_sizes: Dict[str, int]
    split_seeds: Dict[str, int]
    file_records: List[FileRecord]
    splits: Dict[str, List[JetIdentity]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "data_dir": self.data_dir,
            "max_constits": int(self.max_constits),
            "class_names": list(self.class_names),
            "file_prefix_to_label": {k: int(v) for k, v in self.file_prefix_to_label.items()},
            "split_order": list(SPLIT_ORDER),
            "split_sizes": {k: int(v) for k, v in self.split_sizes.items()},
            "split_seeds": {k: int(v) for k, v in self.split_seeds.items()},
            "file_records": [record.to_dict() for record in self.file_records],
            "splits": {
                split: [identity.to_dict() for identity in self.splits.get(split, [])]
                for split in SPLIT_ORDER
            },
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SplitManifest":
        return cls(
            data_dir=str(data["data_dir"]),
            max_constits=int(data["max_constits"]),
            class_names=[str(x) for x in data["class_names"]],
            file_prefix_to_label={str(k): int(v) for k, v in data["file_prefix_to_label"].items()},
            split_sizes={str(k): int(v) for k, v in data["split_sizes"].items()},
            split_seeds={str(k): int(v) for k, v in data["split_seeds"].items()},
            file_records=[FileRecord.from_dict(x) for x in data["file_records"]],
            splits={
                str(split): [JetIdentity.from_dict(x) for x in rows]
                for split, rows in data["splits"].items()
            },
            metadata=dict(data.get("metadata", {})),
        )


def label_from_filename(path: str | Path) -> int:
    """Map a JetClass ROOT filename to the fixed 10-class label id."""

    name = Path(path).name
    for prefix, label in sorted(LABEL_PREFIXES, key=lambda item: len(item[0]), reverse=True):
        if name.startswith(prefix):
            return label
    raise ValueError(f"Could not map JetClass filename to a label: {name}")


def _relative_path(path: Path, data_dir: Path) -> str:
    try:
        return path.relative_to(data_dir).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve_data_file(data_dir: str | Path, file_name: str) -> Path:
    path = Path(file_name)
    if path.is_absolute():
        return path
    return Path(data_dir) / file_name


def _require_split_names(mapping: Mapping[str, Any], name: str) -> None:
    missing = [split for split in SPLIT_ORDER if split not in mapping]
    extra = [split for split in mapping if split not in SPLIT_ORDER]
    if missing or extra:
        raise ValueError(f"{name} must contain exactly {SPLIT_ORDER}; missing={missing}, extra={extra}")


def discover_file_records(
    data_dir: str | Path,
    *,
    pattern: str = "*.root",
    tree_name: str = "tree",
    require_all_classes: bool = True,
    ignore_unknown: bool = True,
) -> List[FileRecord]:
    """Find JetClass ROOT files and count entries without loading jet arrays."""

    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"JetClass data_dir does not exist: {root}")

    try:
        import uproot
    except ImportError as exc:
        raise ImportError("discover_file_records requires uproot on the research compute environment") from exc

    records: List[FileRecord] = []
    unknown_files: List[str] = []
    for path in sorted(root.rglob(pattern)):
        try:
            label = label_from_filename(path)
        except ValueError:
            if ignore_unknown:
                unknown_files.append(path.as_posix())
                continue
            raise

        with uproot.open(path) as handle:
            tree = handle[tree_name]
            records.append(
                FileRecord(
                    path=_relative_path(path, root),
                    label=label,
                    num_entries=int(tree.num_entries),
                )
            )

    if not records:
        raise FileNotFoundError(f"No JetClass ROOT files matching {pattern!r} found under {root}")

    labels_found = {record.label for record in records}
    if require_all_classes and labels_found != set(range(len(LABEL_NAMES))):
        missing = sorted(set(range(len(LABEL_NAMES))) - labels_found)
        missing_names = [LABEL_NAMES[i] for i in missing]
        raise ValueError(f"Missing JetClass classes in {root}: {missing_names}")

    return records


def build_split_manifest_from_records(
    file_records: Sequence[FileRecord],
    *,
    data_dir: str = DEFAULT_DATA_DIR,
    split_sizes: Mapping[str, int] | None = None,
    split_seeds: Mapping[str, int] | None = None,
    max_constits: int = MAX_CONSTITUENTS,
    base_seed: int = 52,
) -> SplitManifest:
    """Create deterministic balanced five-way jet-level partitions.

    Sampling is class-wise and sequential over the protocol split order. Each
    split uses its own seed and removes selected jets from the per-class pool,
    which guarantees that a stable `(file, entry)` identity appears in at most
    one split.
    """

    split_sizes = dict(DEFAULT_SPLIT_TOTALS if split_sizes is None else split_sizes)
    split_seeds = dict(DEFAULT_SPLIT_SEEDS if split_seeds is None else split_seeds)
    _require_split_names(split_sizes, "split_sizes")
    _require_split_names(split_seeds, "split_seeds")

    n_classes = len(LABEL_NAMES)
    per_class_counts: Dict[str, int] = {}
    for split in SPLIT_ORDER:
        total = int(split_sizes[split])
        if total % n_classes != 0:
            raise ValueError(f"{split} size {total} is not divisible by {n_classes}")
        per_class_counts[split] = total // n_classes

    records_by_label: Dict[int, List[FileRecord]] = defaultdict(list)
    for record in sorted(file_records, key=lambda item: (item.label, item.path)):
        if record.label < 0 or record.label >= n_classes:
            raise ValueError(f"Invalid label id in file record: {record}")
        if record.num_entries <= 0:
            raise ValueError(f"File has no entries: {record}")
        records_by_label[record.label].append(record)

    missing_labels = sorted(set(range(n_classes)) - set(records_by_label))
    if missing_labels:
        raise ValueError(f"No file records for classes: {[LABEL_NAMES[i] for i in missing_labels]}")

    splits: Dict[str, List[JetIdentity]] = {split: [] for split in SPLIT_ORDER}
    for label in range(n_classes):
        records = records_by_label[label]
        counts = np.array([record.num_entries for record in records], dtype=np.int64)
        cumulative = np.cumsum(counts)
        total_available = int(cumulative[-1])
        total_requested = int(sum(per_class_counts.values()))
        if total_requested > total_available:
            raise ValueError(
                f"Class {LABEL_NAMES[label]} has {total_available} jets, "
                f"but {total_requested} are requested"
            )

        remaining = np.arange(total_available, dtype=np.int64)
        for split in SPLIT_ORDER:
            count = int(per_class_counts[split])
            rng = np.random.RandomState(int(split_seeds[split]) + label * 100_003)
            chosen_positions = rng.choice(len(remaining), size=count, replace=False)
            chosen_global = remaining[chosen_positions]
            remaining = np.delete(remaining, chosen_positions)

            file_indices = np.searchsorted(cumulative, chosen_global, side="right")
            file_starts = np.concatenate(([0], cumulative[:-1]))
            for global_index, file_index in zip(chosen_global, file_indices):
                record = records[int(file_index)]
                entry = int(global_index - file_starts[int(file_index)])
                splits[split].append(JetIdentity(file=record.path, entry=entry, label=label))

    metadata = {
        "base_seed": int(base_seed),
        "sampling": "balanced_classwise_sequential_without_replacement",
        "jet_identity": "relative_source_file_path_plus_entry_index",
        "file_level_separation_claimed": False,
        "notes": "No HLT arrays or model inputs are generated in Step 2.",
    }

    manifest = SplitManifest(
        data_dir=str(data_dir),
        max_constits=int(max_constits),
        class_names=list(LABEL_NAMES),
        file_prefix_to_label=dict(FILE_PREFIX_TO_LABEL),
        split_sizes={split: int(split_sizes[split]) for split in SPLIT_ORDER},
        split_seeds={split: int(split_seeds[split]) for split in SPLIT_ORDER},
        file_records=list(file_records),
        splits=splits,
        metadata=metadata,
    )
    audit = audit_split_manifest(manifest)
    if not audit["ok"]:
        raise ValueError(f"Generated invalid split manifest: {audit}")
    return manifest


def _open_text(path: str | Path, mode: str):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def save_split_manifest(manifest: SplitManifest, path: str | Path, *, pretty: bool = False) -> None:
    """Save a split manifest as JSON or JSON gzip based on the suffix."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest.to_dict()
    indent = 2 if pretty else None
    with _open_text(path, "wt") as handle:
        json.dump(payload, handle, indent=indent, sort_keys=True, separators=None if pretty else (",", ":"))
        handle.write("\n")


def load_split_manifest(path: str | Path) -> SplitManifest:
    """Load a JSON or JSON-gzip split manifest."""

    with _open_text(path, "rt") as handle:
        return SplitManifest.from_dict(json.load(handle))


def manifest_hash(manifest: SplitManifest) -> str:
    """Stable content hash for a manifest."""

    payload = json.dumps(manifest.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def split_summary(manifest: SplitManifest) -> Dict[str, Any]:
    """Compact counts useful for logs and protocol reports."""

    class_counts: Dict[str, Dict[str, int]] = {}
    file_counts: Dict[str, int] = {}
    for split in SPLIT_ORDER:
        by_class = {name: 0 for name in manifest.class_names}
        files = set()
        for identity in manifest.splits.get(split, []):
            by_class[manifest.class_names[identity.label]] += 1
            files.add(identity.file)
        class_counts[split] = by_class
        file_counts[split] = len(files)

    files_by_label = {name: 0 for name in manifest.class_names}
    entries_by_label = {name: 0 for name in manifest.class_names}
    for record in manifest.file_records:
        name = manifest.class_names[record.label]
        files_by_label[name] += 1
        entries_by_label[name] += int(record.num_entries)

    return {
        "manifest_hash": manifest_hash(manifest),
        "split_counts": {split: len(manifest.splits.get(split, [])) for split in SPLIT_ORDER},
        "class_counts": class_counts,
        "files_used_by_split": file_counts,
        "files_by_class": files_by_label,
        "entries_by_class": entries_by_label,
    }


def audit_split_manifest(manifest: SplitManifest) -> Dict[str, Any]:
    """Verify split sizes, labels, and stable jet identity non-overlap."""

    duplicates_within: Dict[str, int] = {}
    identity_owner: Dict[str, str] = {}
    cross_split_overlaps: List[Dict[str, str]] = []
    split_counts: Dict[str, int] = {}
    class_counts: Dict[str, Dict[str, int]] = {}

    for split in SPLIT_ORDER:
        identities = manifest.splits.get(split, [])
        split_counts[split] = len(identities)
        seen_in_split = set()
        duplicates = 0
        by_class = {name: 0 for name in manifest.class_names}

        for identity in identities:
            key = identity.key()
            if key in seen_in_split:
                duplicates += 1
            seen_in_split.add(key)
            if key in identity_owner and identity_owner[key] != split:
                cross_split_overlaps.append(
                    {
                        "identity": key,
                        "first_split": identity_owner[key],
                        "second_split": split,
                    }
                )
            else:
                identity_owner[key] = split

            if identity.label < 0 or identity.label >= len(manifest.class_names):
                by_class.setdefault("INVALID", 0)
                by_class["INVALID"] += 1
            else:
                by_class[manifest.class_names[identity.label]] += 1

        duplicates_within[split] = duplicates
        class_counts[split] = by_class

    files_by_split = {
        split: {identity.file for identity in manifest.splits.get(split, [])}
        for split in SPLIT_ORDER
    }
    file_overlap_counts: Dict[str, int] = {}
    for idx, split_a in enumerate(SPLIT_ORDER):
        for split_b in SPLIT_ORDER[idx + 1 :]:
            key = f"{split_a}__{split_b}"
            file_overlap_counts[key] = len(files_by_split[split_a] & files_by_split[split_b])

    expected_counts_ok = all(split_counts.get(split) == manifest.split_sizes.get(split) for split in SPLIT_ORDER)
    duplicate_count = int(sum(duplicates_within.values()))
    ok = expected_counts_ok and duplicate_count == 0 and len(cross_split_overlaps) == 0

    return {
        "ok": bool(ok),
        "expected_counts_ok": bool(expected_counts_ok),
        "split_counts": split_counts,
        "class_counts": class_counts,
        "duplicates_within_split": duplicates_within,
        "duplicate_within_split_count": duplicate_count,
        "cross_split_overlap_count": len(cross_split_overlaps),
        "cross_split_overlap_examples": cross_split_overlaps[:10],
        "file_overlap_counts": file_overlap_counts,
        "file_level_separation_claimed": bool(manifest.metadata.get("file_level_separation_claimed", False)),
    }


def _import_awkward():
    try:
        import awkward as ak
    except ImportError as exc:
        raise ImportError("load_offline_view requires awkward on the research compute environment") from exc
    return ak


def _pad_jagged(array: Any, max_constits: int, *, value: float = 0.0, dtype: str = "float32") -> np.ndarray:
    ak = _import_awkward()
    padded = ak.fill_none(ak.pad_none(array, max_constits, clip=True), value)
    return ak.to_numpy(ak.values_astype(padded, dtype))


def _mask_from_jagged(array: Any, max_constits: int) -> np.ndarray:
    ak = _import_awkward()
    lengths = np.asarray(ak.to_numpy(ak.num(array, axis=1)), dtype=np.int64)
    clipped = np.minimum(lengths, int(max_constits))
    return np.arange(max_constits, dtype=np.int64)[None, :] < clipped[:, None]


def _tokens_from_arrays(arrays: Mapping[str, Any], max_constits: int) -> tuple[np.ndarray, np.ndarray]:
    px = _pad_jagged(arrays["part_px"], max_constits)
    py = _pad_jagged(arrays["part_py"], max_constits)
    pz = _pad_jagged(arrays["part_pz"], max_constits)
    energy = _pad_jagged(arrays["part_energy"], max_constits)
    mask = _mask_from_jagged(arrays["part_energy"], max_constits)

    pt = np.hypot(px, py).astype(np.float32)
    phi = np.arctan2(py, px).astype(np.float32)
    eta = np.zeros_like(pt, dtype=np.float32)
    valid_pt = mask & (pt > 0)
    eta[valid_pt] = np.arcsinh((pz[valid_pt] / np.maximum(pt[valid_pt], 1e-8))).astype(np.float32)
    eta = np.nan_to_num(eta, nan=0.0, posinf=0.0, neginf=0.0)

    tokens = np.zeros((pt.shape[0], int(max_constits), RAW_TOKEN_DIM), dtype=np.float32)
    tokens[:, :, 0] = pt
    tokens[:, :, 1] = eta
    tokens[:, :, 2] = phi
    tokens[:, :, 3] = energy
    tokens[:, :, 4] = _pad_jagged(arrays["part_charge"], max_constits)

    for offset, branch in enumerate(PID_TOKEN_BRANCHES, start=5):
        tokens[:, :, offset] = _pad_jagged(arrays[branch], max_constits)
    for offset, branch in enumerate(TRACK_TOKEN_BRANCHES, start=10):
        tokens[:, :, offset] = _pad_jagged(arrays[branch], max_constits)

    tokens *= mask[:, :, None]
    return tokens.astype(np.float32), mask.astype(bool)


def _verify_label_chunk(arrays: Mapping[str, Any], local_indices: np.ndarray, expected_label: int, source: Path) -> None:
    labels = []
    for branch in LABEL_BRANCHES:
        labels.append(np.asarray(arrays[branch][local_indices], dtype=np.int64))
    label_matrix = np.stack(labels, axis=1)
    expected = np.zeros((label_matrix.shape[0], len(LABEL_BRANCHES)), dtype=np.int64)
    expected[:, int(expected_label)] = 1
    if not np.array_equal(label_matrix, expected):
        bad = int(np.nonzero(np.any(label_matrix != expected, axis=1))[0][0])
        raise ValueError(
            f"Label branches in {source} do not match filename label "
            f"{LABEL_NAMES[int(expected_label)]}; first bad selected row {bad}"
        )


def load_offline_view(
    manifest: SplitManifest,
    split: str,
    *,
    data_dir: str | Path | None = None,
    tree_name: str = "tree",
    max_constits: int | None = None,
    verify_label_branches: bool = False,
    read_chunk_size: int = 50_000,
) -> JetView:
    """Load offline raw-token tensors for one manifest split.

    Returned token tensors have shape `[N, 128, 14]` by default and follow the
    fixed HLT generator's raw-token convention. Labels and jet IDs preserve the
    manifest order exactly.
    """

    if split not in SPLIT_ORDER:
        raise ValueError(f"Unknown split {split!r}; expected one of {SPLIT_ORDER}")

    try:
        import uproot
    except ImportError as exc:
        raise ImportError("load_offline_view requires uproot on the research compute environment") from exc

    data_dir = manifest.data_dir if data_dir is None else str(data_dir)
    max_constits = manifest.max_constits if max_constits is None else int(max_constits)
    identities = manifest.splits[split]
    n_jets = len(identities)
    tokens = np.zeros((n_jets, max_constits, RAW_TOKEN_DIM), dtype=np.float32)
    mask = np.zeros((n_jets, max_constits), dtype=bool)
    labels = np.zeros((n_jets,), dtype=np.int64)

    grouped: Dict[str, List[tuple[int, int, int]]] = defaultdict(list)
    for output_index, identity in enumerate(identities):
        grouped[identity.file].append((output_index, int(identity.entry), int(identity.label)))
        labels[output_index] = int(identity.label)

    read_branches = list(PARTICLE_READ_BRANCHES)
    if verify_label_branches:
        read_branches.extend(LABEL_BRANCHES)

    for file_name, rows in sorted(grouped.items()):
        source = _resolve_data_file(data_dir, file_name)
        rows = sorted(rows, key=lambda item: item[1])
        entries = np.array([entry for _, entry, _ in rows], dtype=np.int64)
        output_indices = np.array([index for index, _, _ in rows], dtype=np.int64)
        expected_labels = np.array([label for _, _, label in rows], dtype=np.int64)
        if len(set(expected_labels.tolist())) != 1:
            raise ValueError(f"Selected entries for {source} contain multiple labels")
        expected_label = int(expected_labels[0])

        with uproot.open(source) as handle:
            tree = handle[tree_name]
            missing = [branch for branch in read_branches if branch not in tree.keys()]
            if missing:
                raise KeyError(f"{source} is missing required branches: {missing}")

            chunk_ids = entries // int(read_chunk_size)
            for chunk_id in sorted(set(chunk_ids.tolist())):
                take = np.nonzero(chunk_ids == chunk_id)[0]
                chunk_entries = entries[take]
                chunk_outputs = output_indices[take]
                entry_start = int(chunk_id * read_chunk_size)
                entry_stop = min(int(entry_start + read_chunk_size), int(tree.num_entries))
                arrays = tree.arrays(read_branches, entry_start=entry_start, entry_stop=entry_stop, library="ak")
                local_indices = chunk_entries - entry_start

                selected_arrays = {branch: arrays[branch][local_indices] for branch in PARTICLE_READ_BRANCHES}
                chunk_tokens, chunk_mask = _tokens_from_arrays(selected_arrays, max_constits)
                tokens[chunk_outputs] = chunk_tokens
                mask[chunk_outputs] = chunk_mask

                if verify_label_branches:
                    _verify_label_chunk(arrays, local_indices, expected_label, source)

    return JetView(
        tokens=tokens,
        mask=mask,
        labels=labels,
        jet_ids=list(identities),
        split=split,
        metadata={
            "data_dir": str(data_dir),
            "tree_name": tree_name,
            "max_constits": int(max_constits),
            "source_manifest_hash": manifest_hash(manifest),
            "view": "offline",
        },
    )
