"""Official JetClass ROOT streaming for the full-scale ParT comparison.

Preprocessing is delegated to pinned Weaver and the unmodified upstream YAML.
No reconstructed HLT/offline-token cache is involved. Keep the returned training
and validation iterators alive between reporting epochs: Weaver visits successive
1% ranges in every ROOT file, so rebuilding an iterator would restart that scan.
An exact restart can recreate a loader and replay the consumed batches (without
model work); this intentionally trades startup I/O for an unchanged data stream.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import stat

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


WEAVER_VERSION = "0.4.17"
DATA_CONFIG = Path(__file__).resolve().parents[1] / "particle_transformer/data/JetClass/JetClass_full.yaml"
CLASSES = ("HToBB", "HToCC", "HToGG", "HToWW2Q1L", "HToWW4Q", "TTBar",
           "TTBarLep", "WToQQ", "ZToQQ", "ZJetsToNuNu")
SPLITS = {"train": ("train_100M", range(100)),
          "val": ("val_5M", range(120, 125)),
          "test": ("test_20M", range(100, 120))}
EVENTS_PER_FILE = 100_000
_READ_EVENT_COUNTS = {}


def _strict_read_files(filelist, branches, load_range=None, **kwargs):
    """Fail if Weaver silently skips any unreadable ROOT file in a fetch.

    This wraps only I/O, before official selection/transforms. Weaver's reader
    catches per-file exceptions, so a nonempty concatenation is not sufficient
    evidence that every requested event was read. Reproduce its exact truncation
    and minimum-one-entry range rule from the frozen per-file header counts.
    """
    from weaver.utils.data.fileio import _read_files
    expected = 0
    for path in filelist:
        if path not in _READ_EVENT_COUNTS:
            raise ValueError(f"Unregistered ROOT file requested by stream: {path}")
        count = _READ_EVENT_COUNTS[path]
        if load_range is None:
            expected += count
        else:
            start = int(load_range[0] * count)
            stop = max(start + 1, int(load_range[1] * count))
            expected += max(0, min(stop, count) - start)
    table = _read_files(filelist, branches, load_range=load_range, **kwargs)
    if len(table) != expected:
        raise RuntimeError(f"Incomplete ROOT fetch: expected {expected} events, read {len(table)}; "
                           "a corrupt/unreadable file may have been skipped. Training stopped.")
    return table


def _install_strict_reader(file_entries):
    import weaver.utils.dataset as weaver_dataset
    for path, count in file_entries.items():
        if path in _READ_EVENT_COUNTS and _READ_EVENT_COUNTS[path] != count:
            raise ValueError(f"Conflicting frozen ROOT entry count: {path}")
        _READ_EVENT_COUNTS[path] = count
    weaver_dataset._read_files = _strict_read_files


def _file_entry_counts(inventory, split, file_dict):
    selected = set(sum(file_dict.values(), []))
    counts = {}
    for row in inventory["splits"][split]["files"]:
        path = row["path"]
        if path not in selected:
            continue
        count = row.get("num_entries")
        if count is None:
            # Lightweight inventories and small diagnostic fixtures may omit
            # counts. Freeze their ROOT headers now, never guess 100k entries.
            import uproot
            with uproot.open(path) as source:
                count = int(source["tree"].num_entries)
        if type(count) is not int or count <= 0:
            raise ValueError(f"Invalid ROOT entry count for {path}: {count}")
        counts[path] = count
    return counts


def _inspect_split(directory, split, check_entries):
    if not directory.is_dir() or directory.is_symlink():
        raise FileNotFoundError(f"Missing or linked JetClass split directory: {directory}")
    indices = SPLITS[split][1]
    expected = {f"{label}_{index:03d}.root" for label in CLASSES for index in indices}
    observed = {path.name for path in directory.glob("*.root")}
    if observed != expected:
        missing, extra = sorted(expected - observed), sorted(observed - expected)
        raise ValueError(f"{split} inventory differs: missing={missing[:10]} "
                         f"({len(missing)} total), unexpected={extra[:10]} ({len(extra)} total)")
    records = []
    for name in sorted(expected):
        path = directory / name
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size <= 0:
            raise ValueError(f"Expected a nonempty regular ROOT file, not a link: {path}")
        row = {"name": name, "path": str(path), "size_bytes": metadata.st_size,
               "mtime_ns": metadata.st_mtime_ns}
        if check_entries:
            import uproot
            with uproot.open(path) as source:
                if "tree" not in source or getattr(source["tree"], "classname", "") != "TTree":
                    raise ValueError(f"Expected official ROOT TTree 'tree': {path}")
                row["num_entries"] = int(source["tree"].num_entries)
            if row["num_entries"] != EVENTS_PER_FILE:
                raise ValueError(f"Wrong event count in {path}: {row['num_entries']} != {EVENTS_PER_FILE}")
        records.append(row)
    return {
        "directory": str(directory), "files": records, "file_count": len(records),
        "size_bytes": sum(row["size_bytes"] for row in records),
        "expected_event_count": len(records) * EVENTS_PER_FILE,
        "event_count": sum(row["num_entries"] for row in records) if check_entries else None,
    }


def _hash_inventory(inventory):
    inventory["inventory_sha256"] = hashlib.sha256(
        json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return inventory


def inspect_dataset(data_root, include_test=False, check_entries=False, splits=None):
    """Check exact official filenames and record cheap file metadata.

    ``data_root`` is the directory containing ``Pythia``. By default ROOT payloads
    are not read and ``event_count`` is unknown; ``expected_event_count`` records
    the dataset specification, not an observation. ``check_entries=True`` opens
    every ROOT header and enforces 100,000 events per file without loading events.
    """
    root = Path(data_root).expanduser().resolve(strict=True)
    inventory = {"schema_version": 1, "data_root": str(root), "splits": {}}
    requested = tuple(splits) if splits is not None else (("train", "val", "test") if include_test else ("train", "val"))
    if not requested or len(set(requested)) != len(requested) or any(split not in SPLITS for split in requested):
        raise ValueError("splits must select unique train/val/test names")
    for split in requested:
        directory = root / "Pythia" / SPLITS[split][0]
        inventory["splits"][split] = _inspect_split(directory, split, check_entries)
    return _hash_inventory(inventory)


def inspect_test_dataset(test_directory, check_entries=True):
    """Inspect a standalone official test directory without train/val present.

    This accepts the extracted test directory itself, including an explicitly
    supplied nonstandard location, but still requires exactly the official 200
    filenames (indices 100--119) and, by default, their 100k-event ROOT headers.
    """
    directory = Path(test_directory).expanduser().absolute()
    if directory.is_symlink():
        raise ValueError("Test directory must not be a symbolic link")
    directory = directory.resolve(strict=True)
    inventory = {"schema_version": 1, "data_root": str(directory.parent.parent),
                 "splits": {"test": _inspect_split(directory, "test", check_entries)}}
    return _hash_inventory(inventory)


def rank_file_dict(split, inventory, rank=0, world_size=1):
    """Disjoint global-rank file shards, before Weaver shards among workers.

    Training is grouped by class just as in the official launcher. Validation
    and inference use a single file group, which also supports single-file test
    inventories. Never use LOCAL_RANK here: it repeats across distributed nodes.
    """
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    if type(world_size) is not int or world_size <= 0 or type(rank) is not int or not 0 <= rank < world_size:
        raise ValueError("rank/world_size must describe a valid global rank")
    records = inventory["splits"][split]["files"]
    if not records:
        raise ValueError(f"No {split} files in inventory")
    files = []
    seen = set()
    for record in sorted(records, key=lambda row: row["name"]):
        path = Path(record["path"])
        if path.name != record["name"] or str(path.resolve()) in seen:
            raise ValueError("Duplicate file or inconsistent path/name in inventory")
        metadata = path.lstat()
        if (not stat.S_ISREG(metadata.st_mode) or metadata.st_size != record["size_bytes"]
                or metadata.st_mtime_ns != record["mtime_ns"]):
            raise ValueError(f"ROOT file changed since inventory: {path}")
        seen.add(str(path.resolve()))
        files.append(str(path))
    if split == "train":
        groups = {label: [path for path in files if Path(path).name.startswith(label + "_")]
                  for label in CLASSES}
        if sum(map(len, groups.values())) != len(files) or any(not paths for paths in groups.values()):
            raise ValueError("Training inventory must contain all ten recognized classes")
    else:
        groups = {split: files}
    result = {name: paths[rank::world_size] for name, paths in groups.items()}
    if any(not paths for paths in result.values()):
        raise ValueError("Too few files for the requested global-rank sharding")
    return result


def _stream_seed(seed, split, epoch, rank):
    if type(seed) is not int or type(epoch) is not int or epoch < 0:
        raise ValueError("seed and nonnegative epoch must be integers")
    payload = f"part-fullscale-data-v1:{seed}:{split}:{epoch}:{rank}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & ((1 << 63) - 1)


def _seed_worker(worker_id):
    # Weaver also sets NumPy from worker_info.seed. Seed Python explicitly for
    # any future user collate/transform code, without touching the model process.
    del worker_id
    worker_seed = torch.initial_seed() & 0xFFFFFFFF
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class _IsolatedNumpyIterator:
    """Keep the workers=0 diagnostic stream independent of model-side NumPy."""

    def __init__(self, dataset, seed):
        previous = np.random.get_state()
        try:
            np.random.seed(seed & 0xFFFFFFFF)
            self.iterator = iter(dataset)
            self.state = np.random.get_state()
        finally:
            np.random.set_state(previous)

    def __iter__(self):
        return self

    def __next__(self):
        previous = np.random.get_state()
        try:
            np.random.set_state(self.state)
            return next(self.iterator)
        finally:
            self.state = np.random.get_state()
            np.random.set_state(previous)


class _SeededDataset(IterableDataset):
    def __init__(self, weaver_dataset, dataset_options, file_entries, seed, infinite):
        self._weaver_dataset = weaver_dataset
        self.dataset_options = dataset_options
        self.file_entries = file_entries
        self.seed = seed
        self.infinite = infinite
        self._local_iterator = None

    @property
    def weaver_dataset(self):
        if self._weaver_dataset is None:
            from weaver.utils.dataset import SimpleIterDataset
            self._weaver_dataset = SimpleIterDataset(**self.dataset_options)
        return self._weaver_dataset

    def __getstate__(self):
        # Weaver 0.4.17 DataConfig is not spawn-pickleable (__getattr__ recurses
        # while options is unset). Reconstruct its small config in spawned
        # workers; do not serialize prefetched events or an iterator cursor.
        state = dict(self.__dict__)
        state["_weaver_dataset"] = None
        state["_local_iterator"] = None
        return state

    def __iter__(self):
        _install_strict_reader(self.file_entries)
        if get_worker_info() is not None:
            return iter(self.weaver_dataset)
        if not self.infinite or self._local_iterator is None:
            self._local_iterator = _IsolatedNumpyIterator(self.weaver_dataset, self.seed)
        return self._local_iterator


def _validated_config(path):
    config = Path(path) if path is not None else DATA_CONFIG
    content = config.read_bytes()
    if content != DATA_CONFIG.read_bytes():
        raise ValueError("Full-scale loading requires the unchanged official JetClass_full.yaml")
    # Weaver silently prefers an adjacent generated YAML, even with a manual
    # source config. Do not allow that mechanism to alter the frozen preprocessing.
    autogen = Path(str(config).replace(".yaml", f".{hashlib.md5(content).hexdigest()}.auto.yaml"))
    if autogen.exists() or ".auto.yaml" in str(config):
        raise ValueError("Generated Weaver auto YAML could override the official preprocessing")
    return config


def build_loader(split, inventory, batch_size, rank=0, world_size=1, workers=2,
                 seed=101, epoch=0, data_config_path=None):
    """Return ``(DataLoader, Weaver DataConfig)`` with official preprocessing.

    Train/validation preserve the reference launcher's ``for_training=True``,
    shuffling, 1% fetches, infinite streams and drop-last behavior. Validation is
    rank-zero/world-size-one; the caller consumes 1.28M events per checkpoint and
    preserves that iterator across checkpoints, rather than repeatedly selecting
    the same prefix. Test is finite, file-at-a-time, unshuffled and keeps its last
    partial batch. It requires all observer branches in the official YAML.

    ``epoch`` is a stream identifier, not a request to reset each reporting epoch.
    Production keeps it at zero. For resume, recreate the same stream and replay
    previously consumed batches, retaining worker count, rank and batch size.
    Asynchronous Weaver reads are disabled to avoid shared NumPy RNG races;
    DataLoader workers still prefetch independently. No large disk cache is made.
    """
    if type(batch_size) is not int or batch_size <= 0 or type(workers) is not int or workers < 0:
        raise ValueError("batch_size must be positive and workers nonnegative")
    if split == "val" and (rank != 0 or world_size != 1):
        raise ValueError("Checkpoint validation runs on global rank zero with world_size=1")
    if importlib.metadata.version("weaver-core") != WEAVER_VERSION:
        raise RuntimeError(f"This data contract requires weaver-core=={WEAVER_VERSION}")
    from weaver.utils.dataset import SimpleIterDataset
    config_path = _validated_config(data_config_path)
    file_dict = rank_file_dict(split, inventory, rank, world_size)
    if workers and any(len(paths) < workers for paths in file_dict.values()):
        raise ValueError("Each file group needs at least one file per DataLoader worker")
    infinite = split in ("train", "val")
    stream_seed = _stream_seed(seed, split, epoch, rank)
    file_entries = _file_entry_counts(inventory, split, file_dict)
    dataset_options = dict(
        file_dict=file_dict, data_config_file=str(config_path), for_training=infinite,
        load_range_and_fraction=((0, 1), 1),
        fetch_by_files=not infinite, fetch_step=0.01 if infinite else 1,
        file_fraction=1, remake_weights=False, async_load=False,
        infinity_mode=infinite, in_memory=False, name=f"fullscale_{split}_rank{rank}",
    )
    dataset = SimpleIterDataset(**dataset_options)
    if dataset.config._missing_standardization_info or dataset.config.weight_name is not None:
        raise ValueError("Official JetClass preprocessing must be manual and unweighted")
    generator = torch.Generator().manual_seed(stream_seed)
    loader = DataLoader(
        _SeededDataset(dataset, dataset_options, file_entries, stream_seed, infinite), batch_size=batch_size,
        drop_last=infinite, pin_memory=True, num_workers=workers,
        persistent_workers=bool(workers and infinite),
        multiprocessing_context="spawn" if workers else None,
        worker_init_fn=_seed_worker, generator=generator,
    )
    return loader, dataset.config


def replay_batches(iterator, count):
    """Advance a newly reconstructed stream to its saved consumed-batch cursor."""
    if type(count) is not int or count < 0:
        raise ValueError("Replay count must be a nonnegative integer")
    for _ in range(count):
        next(iterator)
    return iterator


def shutdown_loader(loader):
    """Release persistent worker processes when a run or benchmark is finished."""
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None:
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if shutdown is not None:
            shutdown()
        loader._iterator = None
