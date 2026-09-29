"""Shared mmap tokens and compact, lazily reconstructed REGION trees.

Preserves the original token bytes, identity order, collator and epoch sampler.
Unlike the old loader, it does not instantiate/validate a million Python tree
dictionaries before training. Each compressed field is decompressed only once.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

from jetclass_fresh.jetclass_data import JetIdentity
from .contracts import load_hashed_json, sha256_file
from .region_tree import TREE_SCHEMA_CONTRACT

SPLITS = {"model_train": 1_000_000, "model_val": 125_000, "stack_val": 125_000}
SHARD_SIZE = 10_000
NODE_FIELDS = ("parent", "left", "right", "depth", "vectors", "pt", "mass",
               "multiplicity", "merge_delta_r", "merge_kt", "merge_z", "merge_mass")


def tree_tasks():
    return [{"split": split, "shard": start // SHARD_SIZE, "start": start,
             "stop": min(start + SHARD_SIZE, count)}
            for split, count in SPLITS.items() for start in range(0, count, SHARD_SIZE)]


def view_arrays(view):
    files, lookup, indices = [], {}, []
    for identity in view.jet_ids:
        if identity.file not in lookup:
            lookup[identity.file] = len(files)
            files.append(identity.file)
        indices.append(lookup[identity.file])
    return files, {
        "tokens": np.ascontiguousarray(view.tokens, dtype=np.float32),
        "mask": np.ascontiguousarray(view.mask, dtype=bool),
        "labels": np.ascontiguousarray(view.labels, dtype=np.int64),
        "jet_file_indices": np.asarray(indices, dtype=np.int32),
        "jet_entries": np.asarray([j.entry for j in view.jet_ids], dtype=np.int64),
    }


def open_arrays(directory, receipt, *, verify=True):
    arrays = {}
    for name, expected in receipt["files"].items():
        if Path(name).name != name or not name.endswith(".npy"):
            raise ValueError("Unsafe cache filename")
        path = directory / name
        if verify and sha256_file(path) != expected:
            raise ValueError(f"Cache file changed: {path}")
        arrays[Path(name).stem] = np.load(path, mmap_mode="r", allow_pickle=False)
    return arrays


def identity_at(arrays, files, index):
    return JetIdentity(file=files[int(arrays["jet_file_indices"][index])],
                       entry=int(arrays["jet_entries"][index]), label=int(arrays["labels"][index]))


def tree_at(arrays, index):
    start, stop = map(int, arrays["node_offsets"][index:index + 2])
    return {"contract": TREE_SCHEMA_CONTRACT,
            "n_particles": int(arrays["leaf_to_node"].shape[1]),
            "n_valid": int(arrays["n_valid"][index]), "n_nodes": stop - start,
            "root": int(arrays["root"][index]), "leaf_to_node": arrays["leaf_to_node"][index],
            "assignments": {str(k): arrays[f"assignment_K{k}"][index] for k in (2, 4, 8)},
            "actual_cluster_counts": {str(k): int(arrays[f"actual_count_K{k}"][index]) for k in (2, 4, 8)},
            **{name: arrays[name][start:stop] for name in NODE_FIELDS}}


class AblationDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, *, uses_region, binding):
        self.split = split
        directory = Path(root) / "inputs/cache" / split
        self.receipt = load_hashed_json(directory / "receipt.json")
        if self.receipt["content_hash"] != binding["splits"][split]["cache"]:
            raise ValueError("Cache receipt differs from bound training input")
        self.arrays = open_arrays(directory, self.receipt)
        self.trees = []
        if uses_region:
            for task in (t for t in tree_tasks() if t["split"] == split):
                path = Path(root) / "inputs/trees" / split / f"shard_{task['shard']:03d}.npz"
                receipt = load_hashed_json(path.with_suffix(".json"))
                if (receipt["content_hash"] != binding["splits"][split]["trees"][task["shard"]]
                        or receipt["npz_sha256"] != sha256_file(path)):
                    raise ValueError("Tree shard differs from training binding")
                with np.load(path, allow_pickle=False) as packed:
                    arrays = {name: packed[name] for name in packed.files}
                expected = [identity_at(self.arrays, self.receipt["jet_files"], i).key()
                            for i in range(task["start"], task["stop"])]
                if arrays.pop("identity").tolist() != expected:
                    raise ValueError("Tree identities differ from token rows")
                self.trees.append(arrays)

    def __len__(self):
        return len(self.arrays["labels"])

    def __getitem__(self, index):
        return {"tokens": self.arrays["tokens"][index], "mask": self.arrays["mask"][index],
                "label": self.arrays["labels"][index], "source_view": "offline",
                "identity": identity_at(self.arrays, self.receipt["jet_files"], index).key(),
                "region_tree": tree_at(self.trees[index // SHARD_SIZE], index % SHARD_SIZE)
                if self.trees else None}
